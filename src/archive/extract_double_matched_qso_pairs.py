#!/usr/bin/env python3
"""
Extract the spectroscopic QSO PAIRS that collapse onto a single Euclid+Gaia source.

From the crossmatch_qso_{desi_dr1,sdss_dr16}_euclid_gaia join tables (euclid_source='Q1'),
find every Euclid source (e_id) claimed by MORE THAN ONE spectroscopic QSO — i.e. two (or
more) catalogued QSOs whose closest Euclid match is the SAME object. These are sub-arcsec
pairs that Euclid (and, as it turns out, Gaia) sees as one object while the spectroscopic
survey has two entries: exactly the tight-companion regime the dual-QSO finder targets.

For each such group this writes one row per PAIR with both members' ids / positions / spec-z,
the angular separation between the two QSO positions, and the shared Euclid + Gaia ids (so you
can tell whether they also share one Gaia source_id — for the current sample they all do).

Reads (WSDB; VPN; ~/.pgpass):
  {PGUSER}.crossmatch_qso_desi_dr1_euclid_gaia
  {PGUSER}.crossmatch_qso_sdss_dr16_euclid_gaia
Writes:
  output/qso_pairs/double_matched_qso_pairs.csv

Usage:
  PGUSER=siwei_zhao_2025 python src/extract_double_matched_qso_pairs.py
"""
from __future__ import annotations

import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from builder_qso_gaia_crossmatch import WSDB_HOST, WSDB_DB  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_CSV = PROJECT_DIR / "output" / "qso_pairs" / "double_matched_qso_pairs.csv"
TABLES = ["crossmatch_qso_desi_dr1_euclid_gaia", "crossmatch_qso_sdss_dr16_euclid_gaia"]


def _sep_arcsec(ra1, dec1, ra2, dec2):
    """Angular separation (arcsec) between two sky positions (deg)."""
    r = np.pi / 180.0
    cosd = (np.sin(dec1 * r) * np.sin(dec2 * r)
            + np.cos(dec1 * r) * np.cos(dec2 * r) * np.cos((ra1 - ra2) * r))
    return np.degrees(np.arccos(np.clip(cosd, -1.0, 1.0))) * 3600.0


def fetch(user: str) -> pd.DataFrame:
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30)
    frames = []
    for t in TABLES:
        frames.append(pd.read_sql_query(
            f"""SELECT qso_catalog, qso_id::text AS qso_id, qso_ra, qso_dec, qso_z,
                       e_id, e_ra, e_dec, euclid_sep_arcsec,
                       g_source_id, g_ra, g_dec, gaia_sep_arcsec, g_phot_g_mean_mag
                FROM {user}.{t}
                WHERE euclid_source = 'Q1' AND e_id IS NOT NULL""", conn))
    conn.close()
    return pd.concat(frames, ignore_index=True)


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per QSO pair sharing an Euclid e_id (member 'a' = brighter / lower G)."""
    rows = []
    for e_id, g in df.groupby("e_id"):
        if len(g) < 2:
            continue
        g = g.sort_values("g_phot_g_mean_mag")           # 'a' = brighter component
        for ra, rb in combinations(range(len(g)), 2):
            a, b = g.iloc[ra], g.iloc[rb]
            rows.append({
                "e_id": int(e_id),
                "shares_one_gaia": bool(a["g_source_id"] == b["g_source_id"]),
                "g_source_id_a": int(a["g_source_id"]),
                "g_source_id_b": int(b["g_source_id"]),
                "qso_catalog": a["qso_catalog"],
                "qso_id_a": a["qso_id"], "qso_id_b": b["qso_id"],
                "z_a": float(a["qso_z"]), "z_b": float(b["qso_z"]),
                "dz": float(a["qso_z"]) - float(b["qso_z"]),
                "ra_a": float(a["qso_ra"]), "dec_a": float(a["qso_dec"]),
                "ra_b": float(b["qso_ra"]), "dec_b": float(b["qso_dec"]),
                "pair_sep_arcsec": float(_sep_arcsec(a["qso_ra"], a["qso_dec"],
                                                     b["qso_ra"], b["qso_dec"])),
                "G_a": float(a["g_phot_g_mean_mag"]), "G_b": float(b["g_phot_g_mean_mag"]),
                "e_ra": float(a["e_ra"]), "e_dec": float(a["e_dec"]),
                "g_ra": float(a["g_ra"]), "g_dec": float(a["g_dec"]),
                "euclid_sep_a": float(a["euclid_sep_arcsec"]),
                "euclid_sep_b": float(b["euclid_sep_arcsec"]),
            })
    out = pd.DataFrame(rows).sort_values("pair_sep_arcsec").reset_index(drop=True)
    return out


def main() -> int:
    user = os.getenv("PGUSER", "siwei_zhao_2025")
    df = fetch(user)
    pairs = build_pairs(df)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUT_CSV, index=False)
    print(f"  {len(pairs)} double-matched QSO pairs "
          f"({int(pairs['shares_one_gaia'].sum())} also share one Gaia source)")
    if len(pairs):
        print(f"  pair separation: min {pairs['pair_sep_arcsec'].min():.2f}\", "
              f"median {pairs['pair_sep_arcsec'].median():.2f}\", "
              f"max {pairs['pair_sep_arcsec'].max():.2f}\"")
        print(f"  |dz|: median {pairs['dz'].abs().median():.3f}, "
              f"max {pairs['dz'].abs().max():.3f}")
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(pairs[["qso_catalog", "qso_id_a", "qso_id_b", "z_a", "z_b", "dz",
                         "pair_sep_arcsec", "G_a", "G_b", "shares_one_gaia"]].to_string(index=False))
    print(f"\n  -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
