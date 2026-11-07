#!/usr/bin/env python3
"""Fetch Euclid photometry for deblend siblings identified by extract_qso_deblend_pairs.py.

Reads output/qso_color_vs_z/arrays/deblend_pairs.csv, pulls the 16 aperture
fluxes (VIS/Y/J/H × 1/2/3/4 FWHM) from euclid_q1.mer_final_cat for every
unique sibling_e_id, converts to AB mags, and computes the same 12 within-
aperture colours used by the photoz model.

Output:
  output/qso_color_vs_z/arrays/deblend_sibling_photometry.csv

Usage:
    PGUSER=siwei_zhao_2025 python src/extract_qso_deblend_sibling_photometry.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from builder_qso_gaia_crossmatch import WSDB_HOST, WSDB_DB   # noqa: E402
from apply_photz_catwise_euclid import _BAND_DB               # noqa: E402
from train_xgboost_qso_photz import add_colours               # noqa: E402
from analyze_qso_color_vs_specz import _flux_to_mag, ZPC_VIS  # noqa: E402

PROJECT_DIR  = Path(__file__).resolve().parent.parent
PAIRS_CSV    = PROJECT_DIR / "output" / "qso_color_vs_z" / "arrays" / "deblend_pairs.csv"
OUT_CSV      = PROJECT_DIR / "output" / "qso_color_vs_z" / "arrays" / "deblend_sibling_photometry.csv"
MER_TABLE    = "euclid_q1.mer_final_cat"
CHUNK        = 500


def fetch_sibling_photometry(sibling_ids: list[int], user: str) -> pd.DataFrame:
    import psycopg2
    fcols = [f"flux_{s}_{a}fwhm_aper"    for s in _BAND_DB.values() for a in (1, 2, 3, 4)]
    ecols = [f"fluxerr_{s}_{a}fwhm_aper" for s in _BAND_DB.values() for a in (1, 2, 3, 4)]
    sel   = ", ".join(["object_id", "right_ascension", "declination",
                       "deblended_flag"] + fcols + ecols)

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                            connect_timeout=30, keepalives=1,
                            keepalives_idle=30, keepalives_interval=10,
                            keepalives_count=5)
    conn.cursor().execute("SET statement_timeout = 120000")
    parts = []
    for i in range(0, len(sibling_ids), CHUNK):
        chunk = sibling_ids[i:i + CHUNK]
        cur = conn.cursor()
        cur.execute(f"SELECT {sel} FROM {MER_TABLE} WHERE object_id = ANY(%s)", (chunk,))
        rows = cur.fetchall()
        colnames = ["sibling_e_id", "sibling_e_ra", "sibling_e_dec",
                    "sibling_deblend_flag"] + fcols + ecols
        parts.append(pd.DataFrame(rows, columns=colnames))
        if (i // CHUNK + 1) % 5 == 0 or i + CHUNK >= len(sibling_ids):
            print(f"  {min(i+CHUNK, len(sibling_ids)):,}/{len(sibling_ids):,} fetched", flush=True)
    conn.close()
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main() -> int:
    user = os.environ.get("PGUSER", "siwei_zhao_2025")

    pairs = pd.read_csv(PAIRS_CSV)
    sibling_ids = pairs["sibling_e_id"].dropna().astype("int64").unique().tolist()
    print(f"Fetching photometry for {len(sibling_ids):,} unique sibling Euclid IDs...")

    df = fetch_sibling_photometry(sibling_ids, user)
    if df.empty:
        print("No photometry returned.")
        return 1
    print(f"  {len(df):,} rows returned from mer_final_cat")

    # Convert fluxes → AB mags
    for band, s in _BAND_DB.items():
        zpc = ZPC_VIS if s == "vis" else 1.0
        for a in (1, 2, 3, 4):
            fc = f"flux_{s}_{a}fwhm_aper"
            ec = f"fluxerr_{s}_{a}fwhm_aper"
            df[f"{band}_{a}"] = _flux_to_mag(df[fc], df[ec], zpc)

    # 12 within-aperture colours (same features as photoz model)
    df = add_colours(df)

    # Merge back pair context (primary info, separation)
    df = pairs.merge(df, on="sibling_e_id", how="left")

    n_phot = df["IE_2"].notna().sum()
    print(f"  {n_phot:,}/{len(df):,} siblings have valid 2FWHM IE photometry")
    print(f"  sibling_in_crossmatch=True: {df['sibling_in_crossmatch'].sum()} "
          f"(expected 0 — all siblings are raw-catalog only)")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  -> {OUT_CSV}  ({len(df):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
