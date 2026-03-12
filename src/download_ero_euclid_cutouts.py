#!/usr/bin/env python3
"""
Download Euclid ERO VIS FITS tiles and render PNG cutouts for ERO disagreement sets.

The ERO science FITS tiles are publicly available at:
  https://cdn.euclid.esac.esa.int/Stack/Euclid-VIS-Stack-ERO-{TARGET}.DR3.tar

Each tar contains a ~2.8 GB gzip-compressed VIS FITS image. This script:
  1. Reads the disagreement CSVs to find which ERO fields are needed.
  2. Uses HTTP range requests to extract just the VIS Flattened FITS.gz from
     each tar (skipping the weight map), saving to a local cache directory.
  3. Opens each FITS.gz with astropy and makes small cutouts with Cutout2D.
  4. Saves a PNG preview for each source and a contact sheet per group.

Reads:
  output/q1_to_ero/ero_disagree_euclid_galaxy_q1_low_pgal.csv
  output/q1_to_ero/ero_disagree_euclid_star_q1_high_pgal.csv

Writes:
  output/euclid_ero_cutouts/{group}/png/{basename}.png
  output/euclid_ero_cutouts/{group}/contact_sheet.png
  output/euclid_ero_cutouts/{group}/manifest.csv

Usage:
  python src/download_ero_euclid_cutouts.py
  python src/download_ero_euclid_cutouts.py --size-arcsec 30 --limit 20
  python src/download_ero_euclid_cutouts.py --delete-fits  # delete FITS.gz after cutouts
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR    = PROJECT_ROOT / "output" / "q1_to_ero"
OUTPUT_DIR   = PROJECT_ROOT / "output" / "euclid_ero_cutouts"
FITS_CACHE   = PROJECT_ROOT / "output" / "euclid_ero_fits_cache"

CSV_GROUPS = {
    "euclid_galaxy_q1_low_pgal": INPUT_DIR / "ero_disagree_euclid_galaxy_q1_low_pgal.csv",
    "euclid_star_q1_high_pgal":  INPUT_DIR / "ero_disagree_euclid_star_q1_high_pgal.csv",
}

CDN_BASE = "https://cdn.euclid.esac.esa.int/Stack"

# Map from ERO_FIELDS keys → ERO target name in the CDN URL
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Euclid ERO VIS cutout downloader.")
    p.add_argument("--size-arcsec", type=float, default=25.0,
                   help="Cutout half-size in arcsec (default: 25).")
    p.add_argument("--limit", type=int, default=None,
                   help="Max sources per CSV (default: all).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-render PNGs even if they already exist.")
    p.add_argument("--delete-fits", action="store_true",
                   help="Delete downloaded FITS.gz files after processing (default: keep).")
    p.add_argument("--dpi", type=int, default=75,
                   help="PNG output resolution in DPI (default: 75). Lower = faster.")
    return p.parse_args()


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def http_range(url: str, start: int, end: int, chunk: int = 1 << 20) -> bytes:
    """Download bytes [start, end] from url using HTTP range requests."""
    headers = {
        "Range": f"bytes={start}-{end}",
        "User-Agent": "euclid-ero-cutout-script",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def stream_download(url: str, dest: Path, start: int, size: int,
                    chunk: int = 4 << 20) -> None:
    """Stream-download bytes [start, start+size-1] from url into dest."""
    headers = {
        "Range": f"bytes={start}-{start + size - 1}",
        "User-Agent": "euclid-ero-cutout-script",
    }
    req = urllib.request.Request(url, headers=headers)
    downloaded = 0
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        while True:
            block = r.read(chunk)
            if not block:
                break
            f.write(block)
            downloaded += len(block)
            pct = 100 * downloaded / size
            mb = downloaded / 1e6
            total_mb = size / 1e6
            print(f"    {mb:,.0f} / {total_mb:,.0f} MB  ({pct:.1f}%)", end="\r", flush=True)
    print()
    if downloaded != size:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Incomplete download: got {downloaded:,} bytes, expected {size:,} bytes. "
            f"Deleted partial file — re-run to retry."
        )


# ── Tar header parsing ────────────────────────────────────────────────────────

def read_tar_first_entry(tar_url: str) -> tuple[str, int, int]:
    """
    Read tar headers to find the first regular file entry (skipping directory entries).
    Returns (filename, data_offset_in_tar, data_size_bytes).
    """
    # Fetch enough bytes to cover a directory entry + the following file entry
    header_bytes = http_range(tar_url, 0, 2047)
    offset = 0
    while offset + 512 <= len(header_bytes):
        block = header_bytes[offset:offset + 512]
        name = block[0:100].rstrip(b"\x00").decode("utf-8", errors="replace")
        size_oct = block[124:136].rstrip(b"\x00 ").decode("ascii", errors="replace")
        size = int(size_oct, 8) if size_oct else 0
        typeflag = block[156:157]
        # typeflag b'5' = directory, b'0' or b'\x00' = regular file
        if typeflag not in (b'5',) and size > 0:
            data_offset = offset + 512
            return name, data_offset, size
        # Skip to next header: current header (512) + padded data blocks
        padded = ((size + 511) // 512) * 512
        offset += 512 + padded
    raise RuntimeError(f"Could not find a regular file entry in tar: {tar_url}")


# ── FITS cache management ─────────────────────────────────────────────────────

def ensure_fits(field: str) -> Path | None:
    """
    Download the VIS Flattened FITS.gz for the given ERO field if not cached.
    Returns the local path, or None if the field is not in FIELD_TO_TARGET.
    """
    target = FIELD_TO_TARGET.get(field)
    if target is None:
        print(f"  WARNING: No CDN target mapping for field '{field}'")
        return None

    tar_url    = f"{CDN_BASE}/Euclid-VIS-Stack-{target}.DR3.tar"
    gz_path    = FITS_CACHE / f"Euclid-VIS-{target}-Flattened.DR3.fits.gz"
    fits_path  = FITS_CACHE / f"Euclid-VIS-{target}-Flattened.DR3.fits"

    FITS_CACHE.mkdir(parents=True, exist_ok=True)

    # If decompressed .fits already exists, always use it (--overwrite only re-renders PNGs)
    if fits_path.exists():
        size_gb = fits_path.stat().st_size / 1e9
        print(f"  [cache] {fits_path.name}  ({size_gb:.1f} GB, memory-mappable)")
        return fits_path

    # Fetch tar header to know the expected .gz size (needed for partial-download check)
    print(f"  Fetching tar header: {tar_url}")
    fname, data_offset, data_size = read_tar_first_entry(tar_url)
    print(f"    → {fname}  ({data_size / 1e9:.2f} GB compressed)")
    if "Flattened" not in fname:
        print(f"  WARNING: unexpected first entry in tar: {fname}")

    # Download .fits.gz if missing or incomplete (partial download)
    gz_complete = gz_path.exists() and gz_path.stat().st_size == data_size
    if not gz_complete:
        if gz_path.exists() and not gz_complete:
            print(f"  Partial download detected ({gz_path.stat().st_size / 1e6:.0f} / "
                  f"{data_size / 1e6:.0f} MB) — re-downloading")
        print(f"  Downloading FITS.gz → {gz_path.name}")
        stream_download(tar_url, gz_path, data_offset, data_size)
        print(f"  Saved: {gz_path}")
    else:
        print(f"  [cache] {gz_path.name}  ({gz_path.stat().st_size / 1e9:.2f} GB, complete)")

    # Decompress .fits.gz → .fits (one-time, enables memory-mapped random access)
    import gzip, shutil
    print(f"  Decompressing → {fits_path.name}  (one-time, ~5–10 min) ...")
    with gzip.open(gz_path, "rb") as f_in, open(fits_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=32 << 20)  # 32 MB chunks
    print(f"  Decompressed: {fits_path}  ({fits_path.stat().st_size / 1e9:.1f} GB)")

    # Remove the .gz now that we have the plain .fits (saves ~2.9 GB per field)
    gz_path.unlink()
    print(f"  Removed .gz to save space: {gz_path.name}")

    return fits_path


# ── Cutout rendering ──────────────────────────────────────────────────────────

def make_cutout_png(fits_path: Path, ra: float, dec: float,
                    size_arcsec: float, png_path: Path, title: str,
                    dpi: int = 75) -> None:
    """Open a (possibly gzip-compressed) FITS file, make a Cutout2D, and save PNG."""
    from astropy.io import fits as afits
    from astropy.coordinates import SkyCoord
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    import astropy.units as u

    with afits.open(fits_path, memmap=True) as hdul:
        # Find the primary image HDU
        hdu = None
        for h in hdul:
            if hasattr(h, "data") and isinstance(h.data, np.ndarray) and h.data.ndim == 2:
                hdu = h
                break
        if hdu is None:
            raise RuntimeError(f"No 2D image HDU found in {fits_path.name}")

        wcs    = WCS(hdu.header)
        coord  = SkyCoord(ra=ra, dec=dec, unit="deg")
        size   = u.Quantity([size_arcsec, size_arcsec], unit=u.arcsec)
        cutout = Cutout2D(hdu.data, coord, size, wcs=wcs)

    image = cutout.data.astype(float)
    finite = np.isfinite(image)
    if not finite.any():
        raise RuntimeError("No finite pixels in cutout")

    vmin, vmax = np.nanpercentile(image[finite], [0.5, 99.5])
    if not (np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin):
        vmin, vmax = np.nanmin(image[finite]), np.nanmax(image[finite])

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
              interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    n = image.shape[0]
    ax.axhline(n / 2, color="red", lw=0.5, alpha=0.5)
    ax.axvline(n / 2, color="red", lw=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── Contact sheet ─────────────────────────────────────────────────────────────

def make_contact_sheet(png_dir: Path, out_stem: Path,
                       ncols: int = 5, max_rows: int = 6) -> None:
    """Save paginated contact sheets.  One page holds ncols×max_rows images.
    If only one page is needed the file is <out_stem>.png, otherwise
    <out_stem>_p1.png, <out_stem>_p2.png, …
    """
    pngs = sorted(png_dir.glob("*.png"))
    if not pngs:
        return
    per_page = ncols * max_rows
    pages = [pngs[i:i + per_page] for i in range(0, len(pngs), per_page)]
    for pi, page_pngs in enumerate(pages, start=1):
        out_path = (Path(str(out_stem) + ".png") if len(pages) == 1
                    else Path(str(out_stem) + f"_p{pi}.png"))
        nrows = (len(page_pngs) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows),
                                 squeeze=False)
        for i, ax in enumerate(axes.flat):
            if i < len(page_pngs):
                ax.imshow(plt.imread(str(page_pngs[i])))
            ax.axis("off")
        plt.suptitle(f"Page {pi} / {len(pages)}  ({len(page_pngs)} sources)",
                     fontsize=10, y=1.002)
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Contact sheet: {out_path}")


# ── Per-group processing ──────────────────────────────────────────────────────

def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def source_basename(row: pd.Series) -> str:
    field = sanitize(row.get("field", "unknown"))
    ra    = f"{float(row['ra']):.6f}"
    dec   = f"{float(row['dec']):+.6f}".replace("+", "p").replace("-", "m")
    return f"{field}_ra{ra}_dec{dec}"


def process_group(name: str, csv_path: Path, args: argparse.Namespace) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    if name == "euclid_galaxy_q1_low_pgal":
        df = df.sort_values(["xgb_q1_prob", "class_star"], ascending=[True, True])
    else:
        df = df.sort_values(["xgb_q1_prob", "class_star"], ascending=[False, False])
    df = df.reset_index(drop=True)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    group_dir = OUTPUT_DIR / name
    png_dir   = group_dir / "png"
    group_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    # Find which fields are needed and download their FITS once
    needed_fields = [f for f in df["field"].dropna().unique() if f in FIELD_TO_TARGET]
    unknown = set(df["field"].dropna().unique()) - set(FIELD_TO_TARGET.keys())
    if unknown:
        print(f"  WARNING: unknown fields (no CDN mapping): {unknown}")

    fits_cache: dict[str, Path | None] = {}
    print(f"\n[{name}]  {len(df):,} sources across fields: {needed_fields}")
    for field in needed_fields:
        print(f"\n  Field: {field}")
        fits_cache[field] = ensure_fits(field)

    # Make cutouts
    records = []
    n = len(df)
    for idx, row in df.iterrows():
        field    = row.get("field", "")
        base     = source_basename(row)
        png_path = png_dir / f"{base}.png"
        status, message = "ok", ""

        fits_path = fits_cache.get(field)
        if fits_path is None:
            status, message = "skipped", f"no FITS for field '{field}'"
        elif not fits_path.exists():
            status, message = "error", "FITS file missing from cache"
        elif args.overwrite or not png_path.exists():
            try:
                title = (f"{field}  VIS  "
                         f"RA={float(row['ra']):.4f}  Dec={float(row['dec']):+.4f}\n"
                         f"cs={row['class_star']:.3f}  "
                         f"P(gal|Q1)={row['xgb_q1_prob']:.3f}  "
                         f"G={row['phot_g_mean_mag']:.2f}")
                make_cutout_png(fits_path, float(row["ra"]), float(row["dec"]),
                                args.size_arcsec, png_path, title, dpi=args.dpi)
            except Exception as exc:
                status, message = "error", str(exc)

        print(f"  {idx+1:>4}/{n}  {base}  {status}"
              + (f"  — {message}" if message else ""))
        records.append({**row.to_dict(), "png_path": str(png_path) if png_path.exists() else "",
                        "status": status, "message": message})

    pd.DataFrame(records).to_csv(group_dir / "manifest.csv", index=False)
    make_contact_sheet(png_dir, group_dir / f"contact_sheet_{name}")

    if args.delete_fits:
        for field, fp in fits_cache.items():
            if fp is not None and fp.exists():
                fp.unlink()
                print(f"  Deleted cache: {fp.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Euclid ERO VIS cutout downloader")
    print("=" * 70)
    print(f"  Cutout size : {args.size_arcsec}\"")
    print(f"  Delete FITS : {args.delete_fits}")

    for name, csv_path in CSV_GROUPS.items():
        process_group(name, csv_path, args)

    print(f"\nAll outputs saved under: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
