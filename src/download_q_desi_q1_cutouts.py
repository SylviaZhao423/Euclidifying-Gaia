#!/usr/bin/env python3
"""
Render PNG cutouts for the DESI-Q1 crossmatch groups using Euclid Q1 VIS tiles.

Reads the CSVs produced by analyze_desi_q1.py, assigns each source to its
nearest Q1 field, then downloads tiles on demand via save_q1_utils.py
and renders cutouts.

Reads:
  output/desi_q1/disagree_desi_gal_q1_star.csv
  output/desi_q1/disagree_desi_gal_q1_qso.csv
  output/desi_q1/disagree_desi_gal_q1_unclassified.csv
  output/desi_q1/disagree_desi_gal_q1_cl.csv
  output/desi_q1/agree_both_galaxy.csv

Writes:
  output/euclid_cutouts/q1_cutouts/desi_q1/{group}/contact_sheet_{group}.png
  output/euclid_cutouts/q1_cutouts/desi_q1/{group}/manifest.csv

Usage:
  python src/download_desi_q1_cutouts.py [--limit 300] [--size-arcsec 6]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR    = PROJECT_ROOT / "output" / "desi_q1"
OUTPUT_DIR   = PROJECT_ROOT / "output" / "euclid_cutouts" / "q1_cutouts" / "desi_q1"

sys.path.insert(0, str(Path(__file__).parent))
from save_q1_utils import make_q1_cutouts, _discover_tiles

CSV_GROUPS = {
    "disagree_desi_gal_q1_star":         INPUT_DIR / "disagree_desi_gal_q1_star.csv",
    "disagree_desi_gal_q1_qso":          INPUT_DIR / "disagree_desi_gal_q1_qso.csv",
    "disagree_desi_gal_q1_unclassified": INPUT_DIR / "disagree_desi_gal_q1_unclassified.csv",
    "disagree_desi_gal_q1_cl":           INPUT_DIR / "disagree_desi_gal_q1_cl.csv",
    "agree_both_galaxy":                 INPUT_DIR / "agree_both_galaxy.csv",
}

Q1_FIELDS = {
    "EDF-North":  (269.733, +66.018),
    "EDF-South":  ( 61.241, -48.423),
    "EDF-Fornax": ( 52.932, -28.088),
    "LDN-1641":   ( 85.750,  -8.367),
}

_FIELD_NAMES = list(Q1_FIELDS.keys())
_FIELD_RA    = np.radians([v[0] for v in Q1_FIELDS.values()])
_FIELD_DEC   = np.radians([v[1] for v in Q1_FIELDS.values()])


def nearest_q1_field(ra_deg: float, dec_deg: float) -> str:
    ra  = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    a = (np.sin((_FIELD_DEC - dec) / 2) ** 2
         + np.cos(dec) * np.cos(_FIELD_DEC) * np.sin((_FIELD_RA - ra) / 2) ** 2)
    return _FIELD_NAMES[int(np.argmin(a))]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Euclid Q1 DESI cutouts — PHZ classification vs DESI truth."
    )
    p.add_argument("--size-arcsec", type=float, default=6.0,
                   help="Cutout size in arcsec (default: 6).")
    p.add_argument("--limit",  type=int, default=300,
                   help="Max sources per group (default: 300).")
    p.add_argument("--dpi",    type=int, default=75)
    return p.parse_args()


def _title(row, size):
    phz  = row.get("phz_label", "?")
    pgal = row["p_phz_gal_prob"] if pd.notna(row.get("p_phz_gal_prob")) else float("nan")
    z    = row["d_desi_z"]       if pd.notna(row.get("d_desi_z"))       else float("nan")
    return (f"RA={float(row['ra']):.4f}  Dec={float(row['dec']):+.4f}\n"
            f"PHZ={phz}  P(gal)={pgal:.3f}  z={z:.3f}  [{size:.1f}\"]")


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Euclid Q1 DESI cutouts — PHZ classification vs DESI truth")
    print(f"  Limit      : {args.limit} per group")
    print(f"  Cutout size: {args.size_arcsec}\"")
    print("=" * 65)

    # ── Step 1: load and sort all groups ──────────────────────────────────────
    print("\n[Step 1] Loading source positions ...")
    all_dfs: dict[str, pd.DataFrame] = {}
    for name, csv_path in CSV_GROUPS.items():
        if not csv_path.exists():
            print(f"  Skipping {name} — CSV not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df["field"] = [nearest_q1_field(row["ra"], row["dec"]) for _, row in df.iterrows()]
        ascending = (name != "agree_both_galaxy")
        df = df.sort_values("p_phz_gal_prob", ascending=ascending).head(args.limit).reset_index(drop=True)
        all_dfs[name] = df
        print(f"  {name}: {len(df):,} sources")

    if not all_dfs:
        print("ERROR: no input CSVs found.")
        return 1

    # ── Step 2: one combined tile discovery for all sources ────────────────────
    print("\n[Step 2] Discovering tiles (one ESA TAP query for all groups) ...")
    all_ra  = np.concatenate([df["ra"].values  for df in all_dfs.values()])
    all_dec = np.concatenate([df["dec"].values for df in all_dfs.values()])
    tiles = _discover_tiles(all_ra.min(), all_ra.max(),
                            all_dec.min(), all_dec.max(),
                            field_name="EDF-North")
    if tiles:
        print(f"  {len(tiles)} tile(s) found — reusing for all groups.")
    else:
        print("  ESA TAP failed; each group will fall back to per-source IRSA TAP.")
        tiles = None

    # ── Step 3: render cutouts per group ──────────────────────────────────────
    print("\n[Step 3] Rendering cutouts ...")
    for name, df in all_dfs.items():
        print(f"\n[{name}]  {len(df):,} sources")
        make_q1_cutouts(
            df,
            output_dir=OUTPUT_DIR / name,
            field_col="field",
            size_arcsec=args.size_arcsec,
            title_fn=_title,
            tiles=tiles,
            contact_sheet_stem=name,
            dpi=args.dpi,
        )

    print(f"\nAll outputs saved under: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
