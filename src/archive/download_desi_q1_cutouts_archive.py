#!/usr/bin/env python3
"""
Download Euclid Q1 MER VIS tiles and render PNG cutouts for PHZ failure cases.

Tiles are fetched from the IRSA public archive (no account needed):
  https://irsa.ipac.caltech.edu/ibe/data/euclid/q1/MER/{tile_index}/VIS/{file_name}

Tile discovery uses the ESA Euclid TAP service via astroquery.

Reads:
  output/desi/phz_failures_star.csv
  output/desi/phz_failures_qso.csv
  output/desi/phz_failures_unclassified.csv
  output/desi/phz_failures_glob_cl.csv
  output/desi/phz_correct_galaxy.csv

Writes:
  output/euclid_cutouts/phz_{group}/png/{basename}.png
  output/euclid_cutouts/phz_{group}/contact_sheet_phz_{group}_px.png
  output/euclid_cutouts/phz_{group}/manifest.csv

Usage:
  python src/download_q1_phz_cutouts.py [--limit 100] [--overwrite]
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR    = PROJECT_ROOT / "output" / "desi_q1"
OUTPUT_DIR   = PROJECT_ROOT / "output" / "euclid_cutouts" / "q1_cutouts" / "desi_q1" 

IRSA_BASE    = "https://irsa.ipac.caltech.edu/ibe/data/euclid/q1/MER"
VIS_PIXEL_SCALE = 0.1   # arcsec/pixel

# Tiles are downloaded one at a time, all cutouts for a tile are made, then
# the tile is deleted. Peak disk use = ~1.4 GB (one tile at a time).
# Set to True to keep tiles after processing (faster reruns, but uses ~1.4 GB/tile).
KEEP_TILES = False

CSV_GROUPS = {
    "disagree_desi_gal_q1_star":         INPUT_DIR / "disagree_desi_gal_q1_star.csv",
    "disagree_desi_gal_q1_qso":          INPUT_DIR / "disagree_desi_gal_q1_qso.csv",
    "disagree_desi_gal_q1_unclassified": INPUT_DIR / "disagree_desi_gal_q1_unclassified.csv",
    "disagree_desi_gal_q1_cl":           INPUT_DIR / "disagree_desi_gal_q1_cl.csv",
    "agree_both_galaxy":                  INPUT_DIR / "agree_both_galaxy.csv",
}


# =============================================================================
# Tile discovery and assignment
# =============================================================================

def discover_tiles(ra_min: float, ra_max: float,
                   dec_min: float, dec_max: float) -> list[dict]:
    """Query ESA Euclid TAP for all VIS MER tiles overlapping the given bbox.

    Returns a list of dicts with keys: tile_index, file_name, corners
    where corners is a (4,2) array of (ra, dec) corner coordinates.
    """
    from astroquery.esa.euclid import Euclid

    # Add a small margin so edge sources aren't missed
    margin = 0.3
    polygon = (f"{ra_min - margin},{dec_min - margin}, "
               f"{ra_max + margin},{dec_min - margin}, "
               f"{ra_max + margin},{dec_max + margin}, "
               f"{ra_min - margin},{dec_max + margin}")

    print(f"  Querying TAP for tiles in bbox "
          f"RA [{ra_min:.1f}–{ra_max:.1f}] Dec [{dec_min:.1f}–{dec_max:.1f}] ...")
    job = Euclid.launch_job(f"""
        SELECT file_name, tile_index, fov
        FROM q1.mosaic_product
        WHERE instrument_name='VIS'
        AND INTERSECTS(POLYGON('ICRS', {polygon}), fov)=1
    """)
    t = job.get_results()
    print(f"  -> {len(t)} tiles found")

    tiles = []
    for row in t:
        vals = [float(x) for x in str(row['fov']).strip('()').split(',')]
        # corners: [(ra0,dec0), (ra1,dec1), (ra2,dec2), (ra3,dec3)]
        corners = np.array(vals).reshape(4, 2)
        tiles.append({
            'tile_index': int(row['tile_index']),
            'file_name':  str(row['file_name']).strip(),
            'corners':    corners,
            'ra_min':     corners[:, 0].min(),
            'ra_max':     corners[:, 0].max(),
            'dec_min':    corners[:, 1].min(),
            'dec_max':    corners[:, 1].max(),
        })
    return tiles


def point_in_polygon(ra: float, dec: float, corners: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test for a 4-corner tile footprint."""
    n = len(corners)
    inside = False
    x, y = ra, dec
    j = n - 1
    for i in range(n):
        xi, yi = corners[i]
        xj, yj = corners[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def assign_tiles(df: pd.DataFrame, tiles: list[dict]) -> pd.Series:
    """For each source in df, return the tile_index it falls in (or -1)."""
    result = []
    for _, row in df.iterrows():
        ra, dec = float(row['ra']), float(row['dec'])
        assigned = -1
        for tile in tiles:
            # Fast bounding-box pre-filter then polygon test
            if (tile['ra_min'] <= ra <= tile['ra_max'] and
                    tile['dec_min'] <= dec <= tile['dec_max']):
                if point_in_polygon(ra, dec, tile['corners']):
                    assigned = tile['tile_index']
                    break
        result.append(assigned)
    return pd.Series(result, index=df.index)


# =============================================================================
# FITS tile download
# =============================================================================

class HttpRangeFile:
    """Seekable file-like object backed by HTTP range requests.

    Allows astropy to open a remote FITS file and read only the bytes it needs
    (header + cutout pixel rows), avoiding downloading the full ~1.4 GB tile.
    Each Cutout2D of a 5" stamp downloads ~3-5 MB instead of 1.4 GB.
    """
    def __init__(self, url: str):
        self.url = url
        self._pos = 0
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "euclid-q1-cutout-script"})
        with urllib.request.urlopen(req, timeout=30) as r:
            self._size = int(r.headers["Content-Length"])

    def read(self, n: int = -1) -> bytes:
        if n == 0:
            return b""
        end = (self._size - 1) if n < 0 else min(self._pos + n - 1, self._size - 1)
        if self._pos > end:
            return b""
        headers = {"Range": f"bytes={self._pos}-{end}",
                   "User-Agent": "euclid-q1-cutout-script"}
        req = urllib.request.Request(self.url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        self._pos += len(data)
        return data

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        elif whence == 2:
            self._pos = self._size + pos
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readable(self) -> bool: return True
    def seekable(self) -> bool: return True
    def writable(self) -> bool: return False
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


def tile_url(tile_index: int, file_name: str) -> str:
    return f"{IRSA_BASE}/{tile_index}/VIS/{file_name}"


def download_tile(tile_index: int, file_name: str) -> Path | None:
    """Download a FITS tile from IRSA to local cache. Returns local path or None."""
    fits_path = FITS_CACHE / f"TILE{tile_index}.fits"
    FITS_CACHE.mkdir(parents=True, exist_ok=True)

    if fits_path.exists():
        size_gb = fits_path.stat().st_size / 1e9
        print(f"  [cache] TILE{tile_index}  ({size_gb:.2f} GB)")
        return fits_path

    url = tile_url(tile_index, file_name)
    print(f"  Downloading TILE{tile_index}  ({url.split('/')[-1]})")
    tmp_path = fits_path.with_suffix(".fits.tmp")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "euclid-q1-cutout-script"})
        downloaded = 0
        with urllib.request.urlopen(req, timeout=600) as r:
            total = int(r.headers.get("Content-Length", 0))
            with open(tmp_path, "wb") as f:
                while True:
                    block = r.read(4 << 20)
                    if not block:
                        break
                    f.write(block)
                    downloaded += len(block)
                    if total:
                        pct = 100 * downloaded / total
                        print(f"    {downloaded/1e9:.2f} / {total/1e9:.2f} GB  ({pct:.1f}%)",
                              end="\r", flush=True)
        print()
        tmp_path.rename(fits_path)
        print(f"  Saved: {fits_path}  ({fits_path.stat().st_size / 1e9:.2f} GB)")
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        print(f"  ERROR downloading TILE{tile_index}: {e}")
        return None
    return fits_path


# =============================================================================
# Cutout rendering  (same as ERO script)
# =============================================================================

def _cutout_data_remote(url: str, ra: float, dec: float,
                        size_arcsec: float) -> tuple[np.ndarray, object]:
    """Fetch only the cutout pixel rows from a remote FITS via range requests.

    Returns (image_array, wcs_cutout) using a single HTTP range request for
    pixel data (~3-5 MB) instead of the full tile (~1.4 GB).
    """
    from astropy.io.fits import Header
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS
    import astropy.units as u

    fobj = HttpRangeFile(url)

    # Read header blocks until END keyword found (each block = 2880 bytes)
    header_bytes = b""
    while True:
        block = fobj.read(2880)
        if not block:
            break
        header_bytes += block
        if b"END" + b" " * 77 in block:
            break
    data_start = fobj.tell()  # byte offset where image data begins

    header = Header.fromstring(header_bytes.decode("ascii", errors="replace"))
    naxis1    = int(header["NAXIS1"])
    naxis2    = int(header["NAXIS2"])
    bitpix    = int(header["BITPIX"])
    bscale    = float(header.get("BSCALE", 1.0))
    bzero     = float(header.get("BZERO", 0.0))
    bytes_pp  = abs(bitpix) // 8

    wcs   = WCS(header)
    coord = SkyCoord(ra=ra, dec=dec, unit="deg")
    xc, yc = wcs.world_to_pixel(coord)
    xc, yc = int(round(float(xc))), int(round(float(yc)))

    n_pix = max(10, int(round(size_arcsec / VIS_PIXEL_SCALE)))
    half  = n_pix // 2
    x1, x2 = max(0, xc - half), min(naxis1, xc + half)
    y1, y2 = max(0, yc - half), min(naxis2, yc + half)

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Cutout out of bounds: pixel ({xc},{yc}) in {naxis1}×{naxis2}")

    # ONE range request for rows y1..y2 (full row width, then slice columns)
    row_bytes = naxis1 * bytes_pp
    fobj.seek(data_start + y1 * row_bytes)
    raw = fobj.read((y2 - y1) * row_bytes)

    dtype_map = {16: ">i2", -32: ">f4", -64: ">f8", 8: ">u1", 32: ">i4"}
    dtype = np.dtype(dtype_map.get(bitpix, ">f4"))
    rows  = np.frombuffer(raw, dtype=dtype).reshape(y2 - y1, naxis1)
    image = rows[:, x1:x2].astype(np.float64)
    if bscale != 1.0 or bzero != 0.0:
        image = image * bscale + bzero

    # Build a trimmed WCS for the cutout so callers have the right coords
    wcs_cut = wcs.slice((slice(y1, y2), slice(x1, x2)))
    return image.astype(np.float32), wcs_cut


def make_cutout_image(fits_path: Path | str, ra: float, dec: float,
                      size_arcsec: float, title: str, dpi: int = 75,
                      c_star: float | None = None) -> np.ndarray:
    """Render a cutout stamp to a numpy array (no file saved)."""
    import io
    from astropy.io import fits as afits
    from astropy.coordinates import SkyCoord
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    import astropy.units as u

    if isinstance(fits_path, str):
        image, _ = _cutout_data_remote(fits_path, ra, dec, size_arcsec)
    else:
        with afits.open(fits_path, memmap=True) as hdul:
            hdu = next((h for h in hdul
                        if hasattr(h, "data") and isinstance(h.data, np.ndarray)
                        and h.data.ndim == 2), None)
            if hdu is None:
                raise RuntimeError(f"No 2D image HDU in {fits_path.name}")
            wcs    = WCS(hdu.header)
            coord  = SkyCoord(ra=ra, dec=dec, unit="deg")
            size   = u.Quantity([size_arcsec, size_arcsec], unit=u.arcsec)
            cutout = Cutout2D(hdu.data, coord, size, wcs=wcs)
            image  = cutout.data.astype(float)

    image  = np.asarray(image, dtype=float)
    finite = np.isfinite(image)
    if not finite.any():
        raise RuntimeError("No finite pixels in cutout")

    try:
        from astropy.visualization import ZScaleInterval
        vmin, vmax = ZScaleInterval(contrast=0.25).get_limits(image[finite])
    except Exception:
        vmin, vmax = np.nanpercentile(image[finite], [0.5, 99.5])
    if not (np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin):
        vmin, vmax = np.nanmin(image[finite]), np.nanmax(image[finite])

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
              interpolation="nearest")
    ax.set_title(title, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    n = image.shape[0]
    if c_star is not None and np.isfinite(c_star):
        crosshair_color = plt.cm.coolwarm(np.clip((c_star + 0.5) / 1.5, 0, 1))
    else:
        crosshair_color = "white"
    ax.axhline(n / 2, color=crosshair_color, lw=1.5)
    ax.axvline(n / 2, color=crosshair_color, lw=1.5)
    ax.plot(n / 2, n / 2, "o", color=crosshair_color,
            markersize=7, markeredgecolor="black", markeredgewidth=0.5)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def make_contact_sheet(cell_images: list, out_stem: Path,
                       ncols: int = 5, max_rows: int = 6) -> None:
    if not cell_images:
        return
    per_page = ncols * max_rows
    pages = [cell_images[i:i + per_page] for i in range(0, len(cell_images), per_page)]
    for pi, page_imgs in enumerate(pages, start=1):
        out_path = (Path(str(out_stem) + ".png") if len(pages) == 1
                    else Path(str(out_stem) + f"_p{pi}.png"))
        nrows = (len(page_imgs) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                                 squeeze=False)
        for i, ax in enumerate(axes.flat):
            if i < len(page_imgs):
                ax.imshow(page_imgs[i])
            ax.axis("off")
        plt.suptitle(f"Page {pi}/{len(pages)}  ({len(page_imgs)} sources)",
                     fontsize=10, y=1.002)
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Contact sheet: {out_path}")


# =============================================================================
# Per-group processing
# =============================================================================

def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def source_basename(row: pd.Series) -> str:
    ra  = f"{float(row['ra']):.6f}"
    dec = f"{float(row['dec']):+.6f}".replace("+", "p").replace("-", "m")
    return f"ra{ra}_dec{dec}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--limit",       type=int,   default=300)
    p.add_argument("--size-arcsec", type=float, default=5.0)
    p.add_argument("--overwrite",   action="store_true")
    p.add_argument("--dpi",         type=int,   default=75)
    return p.parse_args()


def render_cutouts_from_tile(fits_path: Path | str,
                             sources: list[dict],
                             args: argparse.Namespace) -> None:
    """Render all cutouts for sources on a tile; stores image arrays in src['image']."""
    for src in sources:
        try:
            c_star_val = src.get("corrected_bp_rp_excess", None)
            c_star_val = float(c_star_val) if pd.notna(c_star_val) else None
            src['image'] = make_cutout_image(fits_path,
                                             src['ra'], src['dec'],
                                             args.size_arcsec, src['title'],
                                             dpi=args.dpi, c_star=c_star_val)
            src['status'] = 'ok'
        except Exception as exc:
            src['status'] = 'error'
            src['message'] = str(exc)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Euclid Q1 MER VIS cutouts — PHZ failure cases")
    print(f"  Cutout size : {args.size_arcsec}\"  |  limit : {args.limit} per group")
    print(f"  Keep tiles  : {KEEP_TILES}  (peak disk ~1.4 GB/tile)")
    print("=" * 65)

    # ── Step 1: load all source positions ────────────────────────────────
    print("\n[Step 1] Loading source positions ...")
    all_dfs = {}
    for name, csv_path in CSV_GROUPS.items():
        if csv_path.exists():
            df = pd.read_csv(csv_path).head(args.limit)
            if "agree_" in name:
                df = df.sort_values("p_phz_gal_prob", ascending=False)
            else:
                df = df.sort_values("p_phz_gal_prob", ascending=True)
            df = df.reset_index(drop=True)
            all_dfs[name] = df
            print(f"  {name}: {len(df):,} sources")
        else:
            print(f"  {name}: NOT FOUND — {csv_path}")

    if not all_dfs:
        print("ERROR: no input CSVs found. Run analyze_phz_desi.py first.")
        return 1

    all_ra  = np.concatenate([df['ra'].values  for df in all_dfs.values()])
    all_dec = np.concatenate([df['dec'].values for df in all_dfs.values()])
    print(f"  RA range:  {all_ra.min():.2f} – {all_ra.max():.2f}")
    print(f"  Dec range: {all_dec.min():.2f} – {all_dec.max():.2f}")

    # ── Step 2: discover tiles ────────────────────────────────────────────
    print("\n[Step 2] Discovering Q1 VIS tiles ...")
    tiles = discover_tiles(all_ra.min(), all_ra.max(), all_dec.min(), all_dec.max())
    tile_by_index = {t['tile_index']: t for t in tiles}

    # ── Step 3: assign sources to tiles, prepare output dirs ─────────────
    print("\n[Step 3] Assigning sources to tiles ...")

    # sources_by_tile[tile_index] = list of source dicts
    sources_by_tile: dict[int, list[dict]] = {}
    group_records: dict[str, list[dict]] = {name: [] for name in all_dfs}

    for name, df in all_dfs.items():
        group_dir = OUTPUT_DIR / name
        group_dir.mkdir(parents=True, exist_ok=True)

        assigned = assign_tiles(df, tiles)
        n_missing = (assigned == -1).sum()
        needed = set(assigned[assigned != -1].unique())
        print(f"  {name}: {len(needed)} unique tiles"
              + (f", {n_missing} unmatched" if n_missing else ""))

        for idx, row in df.iterrows():
            ti     = int(assigned[idx])
            c_star = row.get("corrected_bp_rp_excess", None)
            c_star = float(c_star) if pd.notna(c_star) else None
            title = (f"RA={float(row['ra']):.4f}  Dec={float(row['dec']):+.4f}\n"
                     f"PHZ={row.get('phz_label','?')}  "
                     f"P(gal)={row['p_phz_gal_prob']:.3f}  "
                     f"z={row['d_desi_z']:.3f}  "
                     f"[{args.size_arcsec:.1f}\"]")
            src = {**row.to_dict(), 'group': name, 'tile_index': ti,
                   'ra': float(row['ra']), 'dec': float(row['dec']),
                   'title': title, 'corrected_bp_rp_excess': c_star,
                   'status': 'skipped', 'message': 'no tile'}
            group_records[name].append(src)
            if ti != -1:
                sources_by_tile.setdefault(ti, []).append(src)

    needed_tiles = sorted(sources_by_tile.keys())
    print(f"\n  Total unique tiles to process: {len(needed_tiles)}")

    # ── Step 4 & 5: for each tile — fetch cutouts ────────────────────────
    print(f"\n[Steps 4+5] Processing {len(needed_tiles)} tiles ...")
    if not KEEP_TILES:
        print("  Mode: HTTP range requests (no full tile download, ~3-5 MB per cutout)")
    else:
        print("  Mode: download full tiles (~1.4 GB each, faster for many cutouts/tile)")

    for t_idx, tile_index in enumerate(needed_tiles, 1):
        tile_info = tile_by_index[tile_index]
        srcs = sources_by_tile[tile_index]

        print(f"\n  [{t_idx}/{len(needed_tiles)}] TILE{tile_index}  ({len(srcs)} cutout(s))")

        if KEEP_TILES:
            fits_path = download_tile(tile_index, tile_info['file_name'])
            if fits_path is None:
                for s in srcs:
                    s['status'] = 'error'; s['message'] = 'download failed'
                continue
        else:
            fits_path = tile_url(tile_index, tile_info['file_name'])

        render_cutouts_from_tile(fits_path, srcs, args)

        for s in srcs:
            print(f"    {s['group']}  ra{s['ra']:.4f}  {s['status']}"
                  + (f"  — {s['message']}" if s.get('message') else ""))

    # ── Step 6: write manifests and contact sheets ────────────────────────
    print("\n[Step 6] Writing manifests and contact sheets ...")
    for name in all_dfs:
        group_dir = OUTPUT_DIR / name
        records = [{**{k: v for k, v in s.items()
                       if k not in ('image', 'title', 'corrected_bp_rp_excess')},
                    'status': s['status'],
                    'message': s.get('message', '')}
                   for s in group_records[name]]
        pd.DataFrame(records).to_csv(group_dir / "manifest.csv", index=False)
        cell_images = [s['image'] for s in group_records[name] if 'image' in s]
        make_contact_sheet(cell_images, group_dir / name)
        n_ok = sum(1 for r in records if r['status'] == 'ok')
        print(f"  {name}: {n_ok}/{len(records)} ok")

    print(f"\nAll outputs saved under: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
