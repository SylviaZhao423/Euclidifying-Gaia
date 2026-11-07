#!/usr/bin/env python3
"""Find QSO pairs among Euclid-deblended sources in crossmatch_qso_catwise_euclid_gaia.

For each deblend-flagged Q1 source (e_deblended_flag > 0), find its sibling
Euclid sources in the raw euclid_q1.mer_final_cat within a search radius.
Check whether any sibling is ALSO independently matched in our crossmatch
(i.e., matched to a different WISE QSO candidate).  Those are Euclid-resolved
QSO pairs.

Output columns:
  primary_*   — the deblend-flagged source from our crossmatch
  sibling_*   — the sibling Euclid source from the raw catalog
  pair_sep_arcsec — angular separation between primary and sibling Euclid positions
  sibling_in_crossmatch — True if sibling e_id appears in our crossmatch
  sibling_wise_id / sibling_g_source_id / sibling_cluster / sibling_model_pgal
      — populated when sibling_in_crossmatch is True

Usage:
    PGUSER=siwei_zhao_2025 python src/extract_qso_deblend_pairs.py
    ... --radius 3.0     # search radius in arcsec (default 3.0)
    ... --all            # include siblings NOT in crossmatch too
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from builder_qso_gaia_crossmatch import WSDB_HOST, WSDB_DB  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR     = PROJECT_DIR / "output" / "qso_color_vs_z" / "arrays"
OUT_CSV     = OUT_DIR / "deblend_pairs.csv"


def run(user: str, radius_arcsec: float, include_all: bool) -> pd.DataFrame:
    import psycopg2
    t = f"{user}.crossmatch_qso_catwise_euclid_gaia"

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                            connect_timeout=30, keepalives=1,
                            keepalives_idle=30, keepalives_interval=10,
                            keepalives_count=5)
    conn.cursor().execute("SET statement_timeout = 900000")

    radius_deg = radius_arcsec / 3600.0

    print(f"Querying deblend pairs (radius={radius_arcsec}\")...")
    sql = f"""
    WITH deblend AS (
        -- all deblend-flagged Q1 sources in our crossmatch
        SELECT
            g_source_id        AS primary_gaia_id,
            qso_id             AS primary_wise_id,
            qso_ra             AS primary_wise_ra,
            qso_dec            AS primary_wise_dec,
            e_id               AS primary_e_id,
            e_ra               AS primary_e_ra,
            e_dec              AS primary_e_dec,
            euclid_sep_arcsec  AS primary_euclid_sep,
            e_deblended_flag   AS primary_deblend_flag
        FROM {t}
        WHERE e_deblended_flag > 0
          AND euclid_source = 'Q1'
    ),
    siblings AS (
        -- all Euclid Q1 sources within radius of each deblend primary
        SELECT
            d.primary_gaia_id,
            d.primary_wise_id,
            d.primary_wise_ra,
            d.primary_wise_dec,
            d.primary_e_id,
            d.primary_e_ra,
            d.primary_e_dec,
            d.primary_euclid_sep,
            m.object_id        AS sibling_e_id,
            m.right_ascension  AS sibling_e_ra,
            m.declination      AS sibling_e_dec,
            q3c_dist(d.primary_e_ra, d.primary_e_dec,
                     m.right_ascension, m.declination) * 3600.0
                               AS pair_sep_arcsec
        FROM deblend d
        JOIN euclid_q1.mer_final_cat m
          ON q3c_radial_query(m.right_ascension, m.declination,
                              d.primary_e_ra, d.primary_e_dec, %(r)s)
         AND m.object_id != d.primary_e_id
    ),
    matched AS (
        -- siblings that also appear in our crossmatch
        SELECT
            s.*,
            c.g_source_id      AS sibling_gaia_id,
            c.qso_id           AS sibling_wise_id,
            c.qso_ra           AS sibling_wise_ra,
            c.qso_dec          AS sibling_wise_dec,
            c.euclid_sep_arcsec AS sibling_euclid_sep
        FROM siblings s
        JOIN {t} c
          ON c.e_id = s.sibling_e_id
         AND c.euclid_source = 'Q1'
    )
    SELECT
        m.primary_gaia_id,
        m.primary_wise_id,
        m.primary_wise_ra,
        m.primary_wise_dec,
        m.primary_e_id,
        m.primary_e_ra,
        m.primary_e_dec,
        m.primary_euclid_sep,
        m.sibling_e_id,
        m.sibling_e_ra,
        m.sibling_e_dec,
        m.pair_sep_arcsec,
        m.sibling_gaia_id,
        m.sibling_wise_id,
        m.sibling_wise_ra,
        m.sibling_wise_dec,
        m.sibling_euclid_sep
    FROM matched m
    ORDER BY m.pair_sep_arcsec
    """
    df_pairs = pd.read_sql_query(sql, conn, params={"r": radius_deg})
    print(f"  {len(df_pairs):,} sibling pairs where both are in our crossmatch")

    if include_all:
        # Fetch all deblend primaries, then loop per-source (q3c index only works with fixed center)
        cur2 = conn.cursor()
        cur2.execute(f"""
            SELECT e_id, e_ra, e_dec, g_source_id, qso_id
            FROM {t}
            WHERE e_deblended_flag > 0 AND euclid_source = 'Q1'
        """)
        primaries = cur2.fetchall()
        crossmatch_sibling_ids = set(df_pairs["sibling_e_id"].tolist()) if not df_pairs.empty else set()
        rows_all = []
        for i, (e_id, e_ra, e_dec, gaia_id, wise_id) in enumerate(primaries):
            cur2.execute("""
                SELECT object_id, right_ascension, declination,
                       q3c_dist(%s, %s, right_ascension, declination) * 3600.0
                FROM euclid_q1.mer_final_cat
                WHERE q3c_radial_query(right_ascension, declination, %s, %s, %s)
                  AND object_id != %s
            """, (e_ra, e_dec, e_ra, e_dec, radius_deg, int(e_id)))
            for sib_id, sib_ra, sib_dec, sep in cur2.fetchall():
                rows_all.append({
                    "primary_gaia_id":  gaia_id,
                    "primary_wise_id":  wise_id,
                    "primary_e_id":     e_id,
                    "primary_e_ra":     e_ra,
                    "primary_e_dec":    e_dec,
                    "sibling_e_id":     sib_id,
                    "sibling_e_ra":     sib_ra,
                    "sibling_e_dec":    sib_dec,
                    "pair_sep_arcsec":  sep,
                    "sibling_in_crossmatch": sib_id in crossmatch_sibling_ids,
                })
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(primaries)} primaries processed, {len(rows_all)} siblings so far",
                      flush=True)
        df_all = pd.DataFrame(rows_all).sort_values("pair_sep_arcsec").reset_index(drop=True)
        print(f"  {len(df_all):,} total siblings (crossmatch + non-crossmatch)")
        conn.close()
        return df_all

    conn.close()
    df_pairs["sibling_in_crossmatch"] = True
    return df_pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radius", type=float, default=3.0,
                    help="sibling search radius in arcsec (default 3.0)")
    ap.add_argument("--all", dest="include_all", action="store_true",
                    help="also report siblings not in crossmatch")
    args = ap.parse_args()

    user = os.environ.get("PGUSER", "siwei_zhao_2025")
    df = run(user, args.radius, args.include_all)

    if df.empty:
        print("No pairs found.")
        return 0

    print(f"\n{'='*60}")
    print(f"QSO deblend pairs summary")
    print(f"  Total pair rows        : {len(df):,}")
    print(f"  Unique primaries       : {df['primary_gaia_id'].nunique():,}")
    if "sibling_in_crossmatch" in df.columns:
        n_matched = df["sibling_in_crossmatch"].sum()
        print(f"  Siblings in crossmatch : {n_matched:,}  ({100*n_matched/len(df):.1f}%)")
    print(f"  Median separation      : {df['pair_sep_arcsec'].median():.2f}\"")
    print(f"  Min separation         : {df['pair_sep_arcsec'].min():.2f}\"")
    print(f"{'='*60}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  -> {OUT_CSV}  ({len(df):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
