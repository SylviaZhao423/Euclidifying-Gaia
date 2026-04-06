#!/usr/bin/env python3
"""
Render PNG cutouts from cached Euclid ERO VIS FITS for disagreement/agreement sets.

Assumes all ERO FITS files are already cached in output/euclid_ero_fits_cache/.
Run src/download_ero_fits.py first if any are missing.

Source lists (from output/q1_to_ero/):
  disagree_ero_gal_q1_star  — ERO classifies as galaxy, Q1 classifies as star
  disagree_ero_star_q1_gal  — ERO classifies as star,   Q1 classifies as galaxy
  agree_both_galaxy         — both classifiers agree: galaxy
  agree_both_star           — both classifiers agree: star

Reads:
  output/q1_to_ero/disagree_ero_gal_q1_star.csv
  output/q1_to_ero/disagree_ero_star_q1_gal.csv
  output/q1_to_ero/agree_both_galaxy.csv
  output/q1_to_ero/agree_both_star.csv

Writes:
  output/euclid_cutouts/ero_cutouts/q1_to_ero/{group}/contact_sheet_{group}.png
  output/euclid_cutouts/ero_cutouts/q1_to_ero/{group}/manifest.csv

Usage:
  python src/download_q1_to_ero_cutouts.py
  python src/download_q1_to_ero_cutouts.py --size-arcsec 30 --limit 20
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR    = PROJECT_ROOT / "output" / "q1_to_ero"
OUTPUT_DIR   = PROJECT_ROOT / "output" / "euclid_cutouts" / "ero_cutouts" / "q1_to_ero"
FITS_CACHE   = PROJECT_ROOT / "output" / "euclid_ero_fits_cache"

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

VIS_PIXEL_SCALE = 0.1   # arcsec / pixel (Euclid VIS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Euclid ERO VIS cutout renderer.")
    p.add_argument("--size-arcsec", type=float, default=None,
                   help="Fixed cutout size in arcsec. Overrides dynamic sizing.")
    p.add_argument("--size-min", type=float, default=3.0)
    p.add_argument("--size-max", type=float, default=10.0)
    p.add_argument("--size-scale", type=float, default=8.0,
                   help="Multiplier on flux_radius for dynamic sizing (default: 8).")
    p.add_argument("--limit", type=int, default=300,
                   help="Max sources per CSV (default: 300).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-render PNGs even if they already exist.")
    p.add_argument("--dpi", type=int, default=75)
    return p.parse_args()


# ── Cache lookup ──────────────────────────────────────────────────────────────

def get_cached_fits(field: str) -> Path | None:
    """Return the cached FITS path for an ERO field, or None if not found."""
    filename = FIELD_TO_FILENAME.get(field)
    if filename is None:
        print(f"  WARNING: unknown field '{field}'")
        return None
    path = FITS_CACHE / filename
    if not path.exists():
        print(f"  WARNING: FITS not in cache: {path.name}")
        print(f"  Run: python src/download_ero_fits.py --fields \"{field}\"")
        return None
    print(f"  [cache] {path.name}  ({path.stat().st_size / 1e9:.1f} GB)")
    return path


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
                      dpi: int = 75, c_star: float | None = None) -> np.ndarray:
    import io
    from astropy.io import fits as afits
    from astropy.coordinates import SkyCoord
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    import astropy.units as u

    with afits.open(fits_path, memmap=True) as hdul:
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
    ax.set_title(title, fontsize=8)
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
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


# ── Contact sheet ─────────────────────────────────────────────────────────────

def save_cmap_legend(output_dir: Path) -> None:
    """Save a small coolwarm colorbar legend explaining the crosshair colour encoding."""
    import numpy as np
    fig, ax = plt.subplots(figsize=(4, 0.6))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.55, bottom=0.0)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap="coolwarm",
                              norm=plt.Normalize(vmin=0, vmax=1)),
        cax=ax, orientation="horizontal",
    )
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["point-like (star)", "intermediate", "extended (galaxy)"])
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("Crosshair colour = Gaia corrected BP/RP excess (c_star)",
                 fontsize=8, pad=4)
    out = output_dir / "crosshair_cmap_legend.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved cmap legend: {out}")


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
        plt.suptitle(f"Page {pi} / {len(pages)}  ({len(page_imgs)} sources)",
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
    if name in ("disagree_ero_gal_q1_star", "agree_both_star"):
        df = df.sort_values(["xgb_q1_prob", "class_star"], ascending=[True, True])
    else:
        df = df.sort_values(["xgb_q1_prob", "class_star"], ascending=[False, False])
    df = df.reset_index(drop=True)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    group_dir = OUTPUT_DIR / name
    group_dir.mkdir(parents=True, exist_ok=True)

    needed_fields = df["field"].dropna().unique().tolist()
    print(f"\n[{name}]  {len(df):,} sources across fields: {needed_fields}")

    fits_cache: dict[str, Path | None] = {}
    for field in needed_fields:
        fits_cache[field] = get_cached_fits(field)

    records = []
    cell_images = []
    n = len(df)
    for idx, row in df.iterrows():
        field  = row.get("field", "")
        base   = source_basename(row)
        status, message = "ok", ""

        fits_path = fits_cache.get(field)
        if fits_path is None:
            status, message = "skipped", f"no FITS for field '{field}'"
        elif not fits_path.exists():
            status, message = "error", "FITS file missing from cache"
        else:
            try:
                size_arcsec = dynamic_cutout_size(row, args)
                title = (f"{field}  VIS  "
                         f"RA={float(row['ra']):.4f}  Dec={float(row['dec']):+.4f}\n"
                         f"cs={row['class_star']:.3f}  "
                         f"P(gal|Q1)={row['xgb_q1_prob']:.3f}  "
                         f"G={row['phot_g_mean_mag']:.2f}  "
                         f"[{size_arcsec:.1f}\"]")
                c_star_val = row.get("c_star", None)
                c_star_val = float(c_star_val) if pd.notna(c_star_val) else None
                img = make_cutout_image(fits_path, float(row["ra"]), float(row["dec"]),
                                        size_arcsec, title, dpi=args.dpi,
                                        c_star=c_star_val)
                cell_images.append(img)
            except Exception as exc:
                status, message = "error", str(exc)

        print(f"  {idx+1:>4}/{n}  {base}  {status}"
              + (f"  — {message}" if message else ""))
        records.append({**row.to_dict(), "status": status, "message": message})

    pd.DataFrame(records).to_csv(group_dir / "manifest.csv", index=False)
    make_contact_sheet(cell_images, group_dir / name)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Euclid ERO VIS cutout renderer")
    print("=" * 70)
    if args.size_arcsec is not None:
        print(f"  Cutout size : {args.size_arcsec}\" (fixed)")
    else:
        print(f"  Cutout size : dynamic  [{args.size_min}\"–{args.size_max}\"]  "
              f"scale={args.size_scale}× flux_radius")

    save_cmap_legend(OUTPUT_DIR)

    for name, csv_path in CSV_GROUPS.items():
        process_group(name, csv_path, args)

    print(f"\nAll outputs saved under: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
