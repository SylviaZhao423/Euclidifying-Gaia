#!/usr/bin/env python3
"""
Render PNG cutouts for the ERO DESI crossmatch groups.

Reads the CSVs produced by analyze_ero_desi.py, assigns each source to its
nearest ERO field, then cuts VIS stamps using the cached FITS tiles.

Reads:
  output/desi_ero/disagree_desi_gal_ero_star.csv
  output/desi_ero/agree_both_galaxy.csv

Writes:
  output/euclid_cutouts/ero_cutouts/desi_ero/{group}/contact_sheet_{group}.png
  output/euclid_cutouts/ero_cutouts/desi_ero/{group}/manifest.csv

Usage:
  python src/download_ero_desi_cutouts.py [--limit 300] [--overwrite]
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
INPUT_DIR    = PROJECT_ROOT / "output" / "desi_ero"
OUTPUT_DIR   = PROJECT_ROOT / "output" / "euclid_cutouts" / "ero_cutouts" / "desi_ero"

sys.path.insert(0, str(Path(__file__).parent))
from download_q1_to_ero_cutouts import (
    FITS_CACHE, FIELD_TO_FILENAME, VIS_PIXEL_SCALE,
    get_cached_fits, make_cutout_image, make_contact_sheet, save_cmap_legend,
)

CSV_GROUPS = {
    "disagree_desi_gal_ero_star": INPUT_DIR / "disagree_desi_gal_ero_star.csv",
    "agree_both_galaxy":          INPUT_DIR / "agree_both_galaxy.csv",
}

# ERO field centres (same as q1_to_ero.py / download_ero_euclid_cutouts.py)
ERO_FIELDS = {
    "Horsehead":   (85.245,   -2.458),
    "Messier 78":  (86.755,    0.072),
    "Taurus":      (68.000,   28.000),
    "Barnard 30":  (83.830,   12.030),
    "NGC 6254":    (254.288, -4.101),
    "NGC 6397":    (265.175, -53.674),
    "NGC 6822":    (296.235,  -14.803),
    "Holmberg II": (124.768,  70.717),
    "IC 10":       (5.087,    59.304),
    "IC 342":      (56.702,   68.096),
    "NGC 2403":    (114.214,  65.601),
    "NGC 6744":    (287.444, -63.857),
    "Fornax":      (54.000,  -35.500),
    "Dorado":      (64.014,  -55.780),
    "Perseus":     (49.950,   41.512),
    "Abell 2390":  (328.403,  17.695),
    "Abell 2764":  (5.080,   -60.616),
}

_FIELD_NAMES = list(ERO_FIELDS.keys())
_FIELD_RA    = np.radians([v[0] for v in ERO_FIELDS.values()])
_FIELD_DEC   = np.radians([v[1] for v in ERO_FIELDS.values()])


def nearest_field(ra_deg: float, dec_deg: float) -> str:
    ra  = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    a = (np.sin((_FIELD_DEC - dec) / 2) ** 2
         + np.cos(dec) * np.cos(_FIELD_DEC) * np.sin((_FIELD_RA - ra) / 2) ** 2)
    return _FIELD_NAMES[int(np.argmin(a))]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--size-arcsec", type=float, default=None)
    p.add_argument("--size-min",    type=float, default=3.0)
    p.add_argument("--size-max",    type=float, default=10.0)
    p.add_argument("--size-scale",  type=float, default=8.0)
    p.add_argument("--limit",       type=int,   default=300)
    p.add_argument("--overwrite",   action="store_true")
    p.add_argument("--dpi",         type=int,   default=75)
    return p.parse_args()


def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def source_basename(row: pd.Series) -> str:
    field = sanitize(row.get("field", "unknown"))
    ra    = f"{float(row['ra']):.6f}"
    dec   = f"{float(row['dec']):+.6f}".replace("+", "p").replace("-", "m")
    return f"{field}_ra{ra}_dec{dec}"


def dynamic_cutout_size(row: pd.Series, args: argparse.Namespace) -> float:
    if args.size_arcsec is not None:
        return args.size_arcsec
    for col in ("e_v_flux_radius", "flux_radius"):
        if col in row.index and pd.notna(row[col]):
            try:
                r = float(row[col])
                if np.isfinite(r) and r > 0:
                    return float(np.clip(r * VIS_PIXEL_SCALE * args.size_scale,
                                         args.size_min, args.size_max))
            except (ValueError, TypeError):
                pass
    return args.size_max


def process_group(name: str, csv_path: Path, args: argparse.Namespace) -> None:
    if not csv_path.exists():
        print(f"  Skipping {name} — CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Assign nearest ERO field from coordinates
    df["field"] = [nearest_field(row["ra"], row["dec"]) for _, row in df.iterrows()]

    # Sort: failures → most star-like first; correct → most galaxy-like first
    if name == "disagree_desi_ero_star":
        df = df.sort_values("e_v_class_star", ascending=False)
    else:
        df = df.sort_values("e_v_class_star", ascending=True)
    df = df.head(args.limit).reset_index(drop=True)

    group_dir = OUTPUT_DIR / name
    group_dir.mkdir(parents=True, exist_ok=True)

    # Download/cache needed FITS tiles
    needed_fields = [f for f in df["field"].unique() if f in FIELD_TO_FILENAME]
    unknown = set(df["field"].unique()) - set(FIELD_TO_FILENAME.keys())
    if unknown:
        print(f"  WARNING: unknown fields: {unknown}")

    fits_cache: dict[str, Path | None] = {}
    print(f"\n[{name}]  {len(df):,} sources across fields: {needed_fields}")
    for field in needed_fields:
        print(f"\n  Field: {field}")
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
                c_star_val = row.get("corrected_bp_rp_excess", None)
                c_star_val = float(c_star_val) if pd.notna(c_star_val) else None
                title = (f"{field}  VIS  "
                         f"RA={float(row['ra']):.4f}  Dec={float(row['dec']):+.4f}\n"
                         f"cs={row['e_v_class_star']:.3f}  "
                         f"z={row['d_desi_z']:.3f}  "
                         f"Δχ²={row['d_desi_deltachi2']:.0f}  "
                         f"[{size_arcsec:.1f}\"]")
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


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Euclid ERO DESI cutouts — class_star vs DESI truth")
    print(f"  Limit: {args.limit} per group")
    print("=" * 65)

    save_cmap_legend(OUTPUT_DIR)

    for name, csv_path in CSV_GROUPS.items():
        process_group(name, csv_path, args)

    print(f"\nAll outputs saved under: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
