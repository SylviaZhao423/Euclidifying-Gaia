#!/usr/bin/env python3
"""
Download Euclid ERO VIS cutouts for ERO×Q1 agreement/disagreement groups,
stratified by Q1 UMAP cluster.

Cutouts use log stretch with crosshair coloured by Gaia corrected BP/RP
excess (c_star, coolwarm colourmap: blue=star, red=galaxy).

Each ERO source is projected into Q1 UMAP space via the fitted reducer,
then assigned to the nearest Q1 cluster.  Per-cluster contact sheets and
manifests are saved under {group}/cluster{k}/.

Reads:
  output/q1_to_ero/disagree_ero_gal_q1_star.csv
  output/q1_to_ero/disagree_ero_star_q1_gal.csv
  output/q1_to_ero/agree_both_galaxy.csv
  output/q1_to_ero/agree_both_star.csv
  output/umap/arrays/ero_indiv.npz              (ERO Gaia features + source IDs)
  output/umap/arrays/q1_indiv.npz               (Q1 UMAP embedding + normalisation)
  output/umap/arrays/q1_indiv_reducer.joblib    (fitted UMAP reducer)
  output/umap/arrays/q1_cluster_labels.npy      (Q1 cluster labels)
  output/euclid_ero_fits_cache/*.fits

Writes:
  output/euclid_cutouts/ero_cutouts/q1_to_ero/{group}/manifest.csv
  output/euclid_cutouts/ero_cutouts/q1_to_ero/{group}/cluster{k}/contact_sheet.png
  output/euclid_cutouts/ero_cutouts/q1_to_ero/{group}/cluster{k}/manifest.csv
  output/euclid_cutouts/ero_cutouts/q1_to_ero/{group}/cluster{k}/cutouts/cutout_NNNN.png

Usage:
  python src/download_e_q1_to_ero_cutouts.py                # color composites, 30 per cell
  python src/download_e_q1_to_ero_cutouts.py --per-cell 20  # fewer sources per cell
  python src/download_e_q1_to_ero_cutouts.py --no-color     # grayscale VIS only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np
import pandas as pd

PROJECT_ROOT   = Path(__file__).resolve().parents[1]
INPUT_DIR      = PROJECT_ROOT / "output" / "q1_to_ero"
OUTPUT_DIR     = PROJECT_ROOT / "output" / "euclid_cutouts" / "ero_cutouts" / "q1_to_ero"
FITS_CACHE     = PROJECT_ROOT / "output" / "euclid_ero_fits_cache"
ARRAYS_DIR     = PROJECT_ROOT / "output" / "umap" / "arrays"
Q1_INDIV_NPZ   = ARRAYS_DIR / "q1_indiv.npz"
Q1_LABELS_NPY  = ARRAYS_DIR / "q1_cluster_labels.npy"

CDN_BASE = "https://cdn.euclid.esac.esa.int/Stack"

VIS_PIXEL_SCALE  = 0.1
NISP_PIXEL_SCALE = 0.3
SIZE_SCALE = 8.0
SIZE_MIN   = 5.0
SIZE_MAX   = 15.0

CSV_GROUPS = {
    "disagree_ero_gal_q1_star": INPUT_DIR / "disagree_ero_gal_q1_star.csv",
    "disagree_ero_star_q1_gal": INPUT_DIR / "disagree_ero_star_q1_gal.csv",
    "agree_both_galaxy":        INPUT_DIR / "agree_both_galaxy.csv",
    "agree_both_star":          INPUT_DIR / "agree_both_star.csv",
}

FIELD_TO_FILENAME = {
    "Horsehead":   "Euclid-VIS-ERO-Horsehead-Flattened.DR3.fits",
    "Messier 78":  "Euclid-VIS-ERO-Messier78-Flattened.DR3.fits",
    "Taurus":      "Euclid-VIS-ERO-Taurus-Flattened.DR3.fits",
    "Barnard 30":  "Euclid-VIS-ERO-Barnard30-Flattened.DR3.fits",
    "NGC 6254":    "Euclid-VIS-ERO-NGC6254-Flattened.DR3.fits",
    "NGC 6397":    "Euclid-VIS-ERO-NGC6397-Flattened.DR3.fits",
    "NGC 6822":    "Euclid-VIS-ERO-NGC6822-Flattened.DR3.fits",
    "Holmberg II": "Euclid-VIS-ERO-HolmbergII-Flattened.DR3.fits",
    "IC 10":       "Euclid-VIS-ERO-IC10-Flattened.DR3.fits",
    "IC 342":      "Euclid-VIS-ERO-IC342-Flattened.DR3.fits",
    "NGC 2403":    "Euclid-VIS-ERO-NGC2403-Flattened.DR3.fits",
    "NGC 6744":    "Euclid-VIS-ERO-NGC6744-Flattened.DR3.fits",
    "Fornax":      "Euclid-VIS-ERO-Fornax-Flattened.DR3.fits",
    "Dorado":      "Euclid-VIS-ERO-Dorado-Flattened.DR3.fits",
    "Perseus":     "Euclid-VIS-ERO-Perseus-Flattened.DR3.fits",
    "Abell 2390":  "Euclid-VIS-ERO-Abell2390-Flattened.DR3.fits",
    "Abell 2764":  "Euclid-VIS-ERO-Abell2764-Flattened.DR3.fits",
}

FIELD_TO_TARGET = {
    "Horsehead":   "ERO-Horsehead",
    "Messier 78":  "ERO-Messier78",
    "Taurus":      "ERO-Taurus",
    "Barnard 30":  "ERO-Barnard30",
    "NGC 6254":    "ERO-NGC6254",
    "NGC 6397":    "ERO-NGC6397",
    "NGC 6822":    "ERO-NGC6822",
    "Holmberg II": "ERO-HolmbergII",
    "IC 10":       "ERO-IC10",
    "IC 342":      "ERO-IC342",
    "NGC 2403":    "ERO-NGC2403",
    "NGC 6744":    "ERO-NGC6744",
    "Fornax":      "ERO-Fornax",
    "Dorado":      "ERO-Dorado",
    "Perseus":     "ERO-Perseus",
    "Abell 2390":  "ERO-Abell2390",
    "Abell 2764":  "ERO-Abell2764",
}


# ── FITS cache ────────────────────────────────────────────────────────────────

def get_cached_fits(field: str) -> Path | None:
    filename = FIELD_TO_FILENAME.get(field)
    if filename is None:
        print(f"  WARNING: unknown field '{field}'")
        return None
    path = FITS_CACHE / filename
    if not path.exists():
        print(f"  WARNING: FITS not in cache: {path.name}")
        print(f"  Run: python src/save_ero_fits.py --fields \"{field}\"")
        return None
    print(f"  [cache] {path.name}  ({path.stat().st_size / 1e9:.1f} GB)")
    return path


# ── NISP helpers ─────────────────────────────────────────────────────────────

def _find_tar_entry(tar_url: str, pattern: str) -> tuple[str, int, int]:
    import re as _re, urllib.request
    def _fetch_header(byte_offset):
        hdrs = {"Range": f"bytes={byte_offset}-{byte_offset+511}",
                "User-Agent": "euclid-fits-downloader"}
        req = urllib.request.Request(tar_url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    offset = 0
    long_name = None
    while True:
        block = _fetch_header(offset)
        if len(block) < 512 or block == b"\x00" * 512:
            break
        raw_name = block[0:100].rstrip(b"\x00").decode("utf-8", errors="replace")
        typeflag = block[156:157]
        size_oct = block[124:136].rstrip(b"\x00 ").decode("ascii", errors="replace")
        size     = int(size_oct, 8) if size_oct else 0
        padded   = ((size + 511) // 512) * 512
        if typeflag == b"L":
            name_hdrs = {"Range": f"bytes={offset+512}-{offset+512+size-1}",
                         "User-Agent": "euclid-fits-downloader"}
            with urllib.request.urlopen(
                    urllib.request.Request(tar_url, headers=name_hdrs), timeout=60) as r:
                long_name = r.read().rstrip(b"\x00").decode("utf-8", errors="replace")
            offset += 512 + padded
            continue
        name = long_name if long_name else raw_name
        long_name = None
        if typeflag not in (b"5", b"L", b"K") and size > 0:
            if _re.search(pattern, name, _re.IGNORECASE):
                return name, offset + 512, size
        offset += 512 + padded
    raise RuntimeError(f"No entry matching '{pattern}' in tar: {tar_url}")


def _stream_download(url, dest, start, size, chunk=4 << 20):
    import urllib.request
    headers = {"Range": f"bytes={start}-{start+size-1}",
               "User-Agent": "euclid-fits-downloader"}
    downloaded = 0
    with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=300) as r, \
         open(dest, "wb") as f:
        while True:
            block = r.read(chunk)
            if not block:
                break
            f.write(block)
            downloaded += len(block)
            print(f"    {downloaded/1e6:,.0f}/{size/1e6:,.0f} MB"
                  f"  ({100*downloaded/size:.1f}%)", end="\r", flush=True)
    print()
    if downloaded != size:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Incomplete download: {downloaded}/{size} bytes")


def get_nisp_fits(field: str, band: str) -> Path | None:
    import gzip, shutil
    target = FIELD_TO_TARGET.get(field)
    if target is None:
        return None
    fits_path = FITS_CACHE / f"Euclid-NISP-{band}-{target}-Flattened.DR3.fits"
    if fits_path.exists():
        return fits_path
    tar_url = f"{CDN_BASE}/Euclid-NISP-Stack-{target}.DR3.tar"
    gz_path = FITS_CACHE / f"Euclid-NISP-{band}-{target}-Flattened.DR3.fits.gz"
    pattern = rf"NISP-{band}-.*Flattened"
    print(f"  Scanning NISP tar for band {band} ({field})...")
    try:
        name, data_offset, data_size = _find_tar_entry(tar_url, pattern)
        print(f"    Found: {name}  ({data_size/1e9:.2f} GB)  offset={data_offset:,}")
        _stream_download(tar_url, gz_path, data_offset, data_size)
        print(f"  Decompressing {gz_path.name}...")
        with gzip.open(gz_path, "rb") as fi, open(fits_path, "wb") as fo:
            shutil.copyfileobj(fi, fo, length=32 << 20)
        gz_path.unlink()
        return fits_path
    except Exception as exc:
        print(f"  WARNING: NISP-{band} download failed for {field}: {exc}")
        gz_path.unlink(missing_ok=True)
        return None


def _extract_fits_cutout(fits_path: Path, ra: float, dec: float,
                         size_arcsec: float, max_nan_frac: float = 0.30) -> np.ndarray:
    from astropy.io import fits as afits
    from astropy.coordinates import SkyCoord
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    import astropy.units as u
    with afits.open(fits_path, memmap=True) as hdul:
        hdu = next((h for h in hdul
                    if hasattr(h, "data") and isinstance(h.data, np.ndarray)
                    and h.data.ndim == 2), None)
        if hdu is None:
            raise RuntimeError(f"No 2D HDU in {fits_path.name}")
        cutout = Cutout2D(hdu.data, SkyCoord(ra=ra, dec=dec, unit="deg"),
                          u.Quantity([size_arcsec, size_arcsec], unit=u.arcsec),
                          wcs=WCS(hdu.header), mode="partial", fill_value=np.nan)
    data = cutout.data.astype(float)
    nan_frac = np.isnan(data).mean()
    if nan_frac > max_nan_frac:
        raise RuntimeError(
            f"Source near tile edge: {nan_frac:.0%} NaN (>{max_nan_frac:.0%} threshold)")
    return data


def _lrgb_composite(vis, nisp_j, nisp_h, colour_alpha=0.25):
    from scipy.ndimage import gaussian_filter
    from astropy.visualization import LogStretch, ImageNormalize, ManualInterval
    vis_f  = np.where(np.isfinite(vis), vis, 0.0)
    p01    = float(np.nanpercentile(vis_f, 1.0))
    p995   = float(np.nanpercentile(vis_f, 99.5))
    vis_s  = np.clip(vis_f - p01 + 1.0, 1e-6, None)
    norm   = ImageNormalize(vis_s, interval=ManualInterval(1.0, max(p995-p01+1.0, 10.0)),
                            stretch=LogStretch())
    lum    = norm(vis_s).filled(0).astype(float)
    gray   = np.stack([lum, lum, lum], axis=-1)
    def _lin(img):
        img = np.where(np.isfinite(img), img, 0.0)
        lo, hi = float(np.nanpercentile(img, 1)), float(np.nanpercentile(img, 99))
        return np.clip((img - lo) / (hi - lo + 1e-30), 0, 1)
    r_raw = gaussian_filter(_lin(nisp_h), sigma=9.0)
    g_raw = gaussian_filter(_lin(nisp_j), sigma=9.0)
    b_raw = gaussian_filter(_lin(vis_f),  sigma=9.0)
    colour = np.stack([r_raw, g_raw, b_raw], axis=-1).clip(0, 1)
    colour_lum = (0.299*colour[..., 0] + 0.587*colour[..., 1]
                  + 0.114*colour[..., 2]) + 1e-10
    colour_matched = (colour * (lum / colour_lum)[..., None]).clip(0, 1)
    return ((1.0-colour_alpha)*gray + colour_alpha*colour_matched).clip(0,1).astype(np.float32)


def _add_scale_bar(ax, n_pix, size_arcsec, pixel_scale, color="white"):
    bar_as = 20.0
    for c in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
        if c / size_arcsec >= 0.15:
            bar_as = c
            break
    bar_px = bar_as / pixel_scale
    x0, y0 = n_pix * 0.06, n_pix * 0.08
    ax.plot([x0, x0 + bar_px], [y0, y0], color=color, lw=2.5, solid_capstyle="butt")
    txt = ax.text(x0 + bar_px / 2, y0 + n_pix * 0.04, f'{bar_as:.4g}"',
                  color=color, fontsize=9, ha="center", va="bottom")
    txt.set_path_effects([
        patheffects.Stroke(linewidth=2.5, foreground="black"),
        patheffects.Normal(),
    ])


# ── Cutout sizing ─────────────────────────────────────────────────────────────

def dynamic_cutout_size(row: pd.Series, args: argparse.Namespace) -> float:
    if args.size_arcsec is not None:
        return args.size_arcsec
    radius_px = None
    for col in ("flux_radius", "v_flux_radius", "e_v_flux_radius"):
        if col in row.index and pd.notna(row[col]):
            try:
                radius_px = float(row[col])
                break
            except (ValueError, TypeError):
                pass
    if radius_px is not None and np.isfinite(radius_px) and radius_px > 0:
        size = radius_px * VIS_PIXEL_SCALE * args.size_scale
    else:
        size = args.size_max
    return float(np.clip(size, args.size_min, args.size_max))


# ── Cutout rendering ──────────────────────────────────────────────────────────

def make_cutout_image(fits_path: Path, ra: float, dec: float,
                      size_arcsec: float, title: str,
                      dpi: int = 75,
                      c_star: float | None = None,
                      fits_path_nisp_j: Path | None = None,
                      fits_path_nisp_h: Path | None = None) -> np.ndarray:
    import io
    from astropy.visualization import LogStretch, ImageNormalize, ManualInterval

    vis = _extract_fits_cutout(fits_path, ra, dec, size_arcsec)

    # ── Faint-source SNR check — fall back to grayscale if SNR < 10 ──────────
    vis_f   = np.where(np.isfinite(vis), vis, 0.0)
    n_pix   = vis_f.shape[0]
    r       = max(1, n_pix // 8)
    cx_i = cy_i = n_pix // 2
    central  = vis_f[cy_i-r:cy_i+r, cx_i-r:cx_i+r]
    vis_bg   = float(np.nanpercentile(vis_f, 10))
    vis_rms  = float(np.nanstd(vis_f[vis_f < np.nanpercentile(vis_f, 50)]) or 1.0)
    vis_peak = float(np.nanmax(central))
    bright_enough = (vis_peak - vis_bg) / vis_rms >= 10.0

    use_color = (bright_enough and fits_path_nisp_j is not None
                 and fits_path_nisp_h is not None)

    if use_color:
        try:
            from scipy.ndimage import zoom as _zoom
            nisp_j = _extract_fits_cutout(fits_path_nisp_j, ra, dec, size_arcsec,
                                          max_nan_frac=0.50)
            nisp_h = _extract_fits_cutout(fits_path_nisp_h, ra, dec, size_arcsec,
                                          max_nan_frac=0.50)
            # Upsample NISP (0.3"/px) to VIS resolution (0.1"/px) before blending
            scale = NISP_PIXEL_SCALE / VIS_PIXEL_SCALE  # 3.0
            def _match(arr, tgt):
                th, tw = tgt.shape
                a = arr[:th, :tw]
                if a.shape[0] < th or a.shape[1] < tw:
                    pad = np.full(tgt.shape, np.nanmedian(a))
                    pad[:a.shape[0], :a.shape[1]] = a
                    a = pad
                return a
            nisp_j = _match(_zoom(nisp_j, scale, order=3), vis)
            nisp_h = _match(_zoom(nisp_h, scale, order=3), vis)
            rgb = _lrgb_composite(vis, nisp_j, nisp_h)
        except Exception as e:
            use_color = False

    # ── Log stretch for grayscale ─────────────────────────────────────────────
    p01   = float(np.nanpercentile(vis_f, 1.0))
    p995  = float(np.nanpercentile(vis_f, 99.5))
    shift = -p01 + 1.0
    img_s = np.clip(vis_f + shift, 1e-6, None)
    norm  = ImageNormalize(img_s,
                           interval=ManualInterval(1.0, max(p995+shift, 10.0)),
                           stretch=LogStretch())

    # ── Crosshair colour: P(gal) via coolwarm_r (red=star, blue=galaxy) ──────
    if c_star is not None and np.isfinite(c_star):
        crosshair_color = plt.cm.coolwarm_r(float(np.clip(c_star, 0, 1)))
    else:
        crosshair_color = "white"

    fig, ax = plt.subplots(figsize=(4, 4))
    if use_color:
        ax.imshow(rgb, origin="lower", interpolation="nearest")
    else:
        ax.imshow(img_s, origin="lower", cmap="gray", norm=norm,
                  interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    n   = img_s.shape[0]
    gap = n * 0.10
    cx  = cy = n / 2
    kw  = dict(color=crosshair_color, lw=1.0, alpha=0.8, solid_capstyle="butt")
    ax.plot([0,        cx - gap], [cy, cy], **kw)
    ax.plot([cx + gap, n - 1],   [cy, cy], **kw)
    ax.plot([cx, cx],  [0,        cy - gap], **kw)
    ax.plot([cx, cx],  [cy + gap, n - 1],   **kw)
    _add_scale_bar(ax, n, size_arcsec, VIS_PIXEL_SCALE, crosshair_color)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


# ── Contact sheet ─────────────────────────────────────────────────────────────

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

        # Fixed-height header so layout is consistent regardless of row count
        fig_h_in  = 5.0 * nrows
        header_in = 1.10
        top_frac  = 1.0 - header_in / fig_h_in
        _ih       = header_in / fig_h_in

        fig.subplots_adjust(top=top_frac, bottom=0.01, left=0.01, right=0.99,
                            hspace=0.05, wspace=0.05)

        legend_w = 0.28
        legend_h = 0.15 / fig_h_in
        legend_x = 0.07
        legend_y = top_frac + 0.20 / fig_h_in
        sup_y    = legend_y + legend_h * 0.5

        plt.suptitle(f"Page {pi} / {len(pages)}  ({len(page_imgs)} sources)",
                     fontsize=10, y=sup_y)

        cbar_ax = fig.add_axes([legend_x, legend_y, legend_w, legend_h])
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap="coolwarm_r",
                                  norm=plt.Normalize(vmin=0, vmax=1)),
            cax=cbar_ax, orientation="horizontal",
        )
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["star", "0.5", "galaxy"])
        cbar.ax.tick_params(labelsize=7, pad=1)
        cbar_ax.set_title("XGBoost P(gal) — crosshair colour",
                           fontsize=7, pad=3, loc="left")

        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Contact sheet: {out_path}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def source_basename(row: pd.Series) -> str:
    field = sanitize(row.get("field", "unknown"))
    ra    = f"{float(row['ra']):.6f}"
    dec   = f"{float(row['dec']):+.6f}".replace("+", "p").replace("-", "m")
    return f"{field}_ra{ra}_dec{dec}"


# ── Cluster assignment ────────────────────────────────────────────────────────

def build_cluster_map() -> dict[str, int]:
    """Build {source_id_str → cluster_int} by projecting ERO sources into Q1 UMAP space.

    Loads ERO Gaia features from ero_indiv.npz, projects via the Q1 reducer,
    then assigns each source to the nearest Q1 cluster (nearest-neighbour in
    UMAP space among points with valid cluster labels ≥ 0).
    """
    import joblib
    from sklearn.neighbors import NearestNeighbors

    ero_npz_path = ARRAYS_DIR / "ero_indiv.npz"
    reducer_path = ARRAYS_DIR / "q1_indiv_reducer.joblib"
    for p in [ero_npz_path, Q1_INDIV_NPZ, Q1_LABELS_NPY, reducer_path]:
        if not p.exists():
            print(f"  WARNING: {p.name} not found — cluster info unavailable.")
            return {}

    ero_data  = np.load(ero_npz_path, allow_pickle=True)
    q1_data   = np.load(Q1_INDIV_NPZ, allow_pickle=True)
    q1_emb    = q1_data["embedding"]
    q1_mean   = q1_data["mean"]
    q1_std    = q1_data["std"]
    q1_labels = np.load(Q1_LABELS_NPY)
    reducer   = joblib.load(reducer_path)

    valid_mask = q1_labels >= 0
    if valid_mask.sum() == 0:
        print("  WARNING: No valid Q1 cluster points — cluster info unavailable.")
        return {}

    # Project ERO features into Q1 UMAP space
    X_ero = ero_data["X_raw"].astype(float)
    std_safe = np.where(q1_std > 0, q1_std, 1.0)
    X_scaled = (X_ero - q1_mean) / std_safe
    X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0.0)
    ero_emb = reducer.transform(X_scaled)

    # Assign to nearest valid Q1 cluster
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(q1_emb[valid_mask])
    _, idx = nn.kneighbors(ero_emb)
    idx = idx.ravel()
    clusters = q1_labels[np.where(valid_mask)[0][idx]].astype(int)

    result = dict(zip(ero_data["source_ids"].astype(str), clusters))
    unique_clusters = sorted(set(clusters))
    print(f"  Cluster assignments: {len(result):,} sources → clusters {unique_clusters}")
    return result


# ── Per-group processing ──────────────────────────────────────────────────────

def process_group(name: str, csv_path: Path, args: argparse.Namespace,
                  cluster_map: dict[str, int] | None = None) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    if name in ("disagree_ero_gal_q1_star", "agree_both_star"):
        df = df.sort_values(["xgb_q1_prob", "class_star"], ascending=[True, True])
    else:
        df = df.sort_values(["xgb_q1_prob", "class_star"], ascending=[False, False])
    df = df.reset_index(drop=True)

    group_dir = OUTPUT_DIR / name
    group_dir.mkdir(parents=True, exist_ok=True)

    # Assign cluster to each source
    if cluster_map:
        df["cluster"] = df["g_source_id"].astype(str).map(cluster_map).fillna(-1).astype(int)
    else:
        df["cluster"] = -1

    # Load FITS for all needed fields upfront
    needed_fields = df["field"].dropna().unique().tolist()
    print(f"\n[{name}]  {len(df):,} total sources, fields: {needed_fields}")
    fits_cache: dict[str, Path | None] = {f: get_cached_fits(f) for f in needed_fields}

    # Pre-load NISP paths (only download if --no-color is not set)
    nisp_j_map: dict[str, Path | None] = {}
    nisp_h_map: dict[str, Path | None] = {}
    if not args.no_color:
        for f in needed_fields:
            nisp_j_map[f] = get_nisp_fits(f, "J")
            nisp_h_map[f] = get_nisp_fits(f, "H")

    all_records: list[dict] = []

    cluster_ids = sorted(df["cluster"].unique())

    for k in cluster_ids:
        cl_label = f"cluster{k}" if k >= 0 else "noise"
        cl_df    = df[df["cluster"] == k].copy().reset_index(drop=True)

        # Per-cluster limit (take highest-confidence sources first — already sorted)
        if args.per_cell is not None and len(cl_df) > args.per_cell:
            cl_df = cl_df.head(args.per_cell)

        cl_dir       = group_dir / cl_label
        cutouts_subdir = cl_dir / "cutouts"
        cl_dir.mkdir(parents=True, exist_ok=True)
        cutouts_subdir.mkdir(exist_ok=True)

        cl_records: list[dict] = []
        cl_cell_imgs: list     = []
        local_idx = 0
        n_cl = len(cl_df)
        print(f"  [{cl_label}]  {n_cl} sources")

        for i, row in cl_df.iterrows():
            field  = row.get("field", "")
            base   = source_basename(row)
            status, message, cutout_file = "ok", "", ""

            fits_path = fits_cache.get(field)
            if fits_path is None:
                status, message = "skipped", f"no FITS for field '{field}'"
            elif not fits_path.exists():
                status, message = "error", "FITS file missing from cache"
            else:
                try:
                    size_arcsec = dynamic_cutout_size(row, args)
                    # Use xgb_q1_prob (already P(galaxy) [0,1]) for crosshair
                    pgal = row.get("xgb_q1_prob", None)
                    c_star_val = float(pgal) if pd.notna(pgal) else None
                    sid = str(row.get('g_source_id', ''))
                    title = (f"{name}  {cl_label}\n"
                             f"RA={float(row['ra']):.4f}  Dec={float(row['dec']):+.4f}\n"
                             f"G={row['phot_g_mean_mag']:.2f}  id={sid}")
                    img = make_cutout_image(
                        fits_path, float(row["ra"]), float(row["dec"]),
                        size_arcsec, title, dpi=args.dpi,
                        c_star=c_star_val,
                        fits_path_nisp_j=nisp_j_map.get(field) if not args.no_color else None,
                        fits_path_nisp_h=nisp_h_map.get(field) if not args.no_color else None,
                    )
                    # ── Save individual cutout PNG ────────────────────────────
                    indiv_path = cutouts_subdir / f"cutout_{local_idx:04d}.png"
                    plt.imsave(str(indiv_path), img)
                    cutout_file = f"cutouts/cutout_{local_idx:04d}.png"
                    cl_cell_imgs.append(img)
                    local_idx += 1
                except Exception as exc:
                    status, message = "error", str(exc)

            print(f"    {i+1:>4}/{n_cl}  {base}  {status}"
                  + (f"  — {message}" if message else ""))

            rec = {**row.to_dict(),
                   "source_id":   row.get("g_source_id", ""),   # alias for load_cutout_cache
                   "status":      status,
                   "message":     message,
                   "cutout_file": cutout_file}
            cl_records.append(rec)
            all_records.append(rec)

        # ── Per-cluster contact sheet + manifest ──────────────────────────────
        make_contact_sheet(cl_cell_imgs, cl_dir / "contact_sheet")
        pd.DataFrame(cl_records).to_csv(cl_dir / "manifest.csv", index=False)
        print(f"    → {len(cl_cell_imgs)} cutouts, contact_sheet.png saved")

    # ── Overall group manifest (all clusters combined) ────────────────────────
    pd.DataFrame(all_records).to_csv(group_dir / "manifest.csv", index=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Euclid ERO VIS cutouts for ERO×Q1 sources — LRGB composite + XGBoost crosshair.")
    p.add_argument("--size-arcsec",  type=float, default=None,
                   help="Fixed cutout size in arcsec (overrides dynamic sizing).")
    p.add_argument("--size-min",     type=float, default=SIZE_MIN)
    p.add_argument("--size-max",     type=float, default=SIZE_MAX)
    p.add_argument("--size-scale",   type=float, default=SIZE_SCALE)
    p.add_argument("--per-cell",     type=int,   default=30,
                   help="Max sources per (group × cluster) cell (default: 30).")
    p.add_argument("--dpi",          type=int,   default=75)
    p.add_argument("--no-color",     action="store_true",
                   help="Skip NISP color composites; produce grayscale VIS cutouts only.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Euclid ERO VIS cutouts — LRGB composite + XGBoost P(gal) crosshair")
    print(f"  Per cell : {args.per_cell}  |  Color: {not args.no_color}")
    print("=" * 65)

    print("\n[cluster] Building ERO → Q1 cluster map...")
    cluster_map = build_cluster_map()
    if not cluster_map:
        print("  (No cluster info — contact sheets will not be split by cluster)")

    for name, csv_path in CSV_GROUPS.items():
        process_group(name, csv_path, args, cluster_map=cluster_map)

    print(f"\nAll outputs saved under: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
