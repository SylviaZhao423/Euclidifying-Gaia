#!/usr/bin/env python3
"""
Create a pre-filtered low-latitude Gaia DR3 table in WSDB.

Selects sources from the complement of the high-latitude region:
  - |b| <= 10  OR  |ecl_lat| <= 45  (galactic plane or ecliptic zone)
  - phot_g_mean_mag >= 17
  - phot_g_mean_mag IS NOT NULL

This is the complement of filter_gaia_source_hilat.py, corresponding to
the blue-shaded excluded zones in the scanning law plot.

Also pre-computes derived columns used by the UMAP/XGBoost pipeline
(bp_g, g_rp, corrected_bp_rp_excess, pm_significance, position_error)
so downstream queries are simple SELECTs with no expensive expressions.

Output table:
  siwei_zhao_2025.gaia_source_lolat

Usage:
  PGUSER=siwei_zhao_2025 python src/filter_gaia_source_lolat.py
  PGUSER=siwei_zhao_2025 python src/filter_gaia_source_lolat.py --count-only
"""

from __future__ import annotations

import argparse
import os
import sys


WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
MAG_CUT   = 17


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create pre-filtered low-latitude Gaia DR3 table in WSDB.")
    parser.add_argument("--count-only", action="store_true",
                        help="Just count matching rows, don't create the table.")
    args = parser.parse_args()

    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set.")
        return 1

    import psycopg2
    table_name = f"{username}.gaia_source_lolat"

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()

    # Complement of hilat: NOT (|b| > 10 AND |ecl_lat| > 45)
    # = |b| <= 10 OR |ecl_lat| <= 45
    where = f"""
        WHERE (ABS(b) <= 10 OR ABS(ecl_lat) <= 45)
          AND phot_g_mean_mag >= {MAG_CUT}
          AND phot_g_mean_mag IS NOT NULL
    """

    if args.count_only:
        print("Counting matching rows in gaia_dr3.gaia_source...")
        cursor.execute(f"SELECT COUNT(*) FROM gaia_dr3.gaia_source {where}")
        count = cursor.fetchone()[0]
        print(f"  -> {count:,} sources match low-lat + G >= {MAG_CUT}")
        cursor.close()
        conn.close()
        return 0

    print(f"Creating {table_name} from gaia_dr3.gaia_source...")
    print(f"  Filters: (|b| <= 10 OR |ecl_lat| <= 45), G >= {MAG_CUT}")
    print("  This may take several minutes...")

    query = f"""
        DROP TABLE IF EXISTS {table_name};
        CREATE TABLE {table_name} AS
        SELECT
            source_id,
            random_index,
            ra, dec,
            b, ecl_lat,
            phot_g_mean_mag,
            parallax_over_error,
            astrometric_gof_al,
            astrometric_chi2_al,
            astrometric_excess_noise_sig,
            ipd_gof_harmonic_amplitude,
            ruwe,
            phot_g_mean_flux_over_error,
            phot_bp_mean_flux_over_error,
            phot_rp_mean_flux_over_error,
            phot_bp_mean_mag - phot_g_mean_mag AS bp_g,
            phot_g_mean_mag - phot_rp_mean_mag AS g_rp,
            phot_bp_rp_excess_factor - (
                CASE
                    WHEN bp_rp < 0.5 THEN 1.154360 + 0.033772*bp_rp + 0.032277*bp_rp*bp_rp
                    WHEN bp_rp < 4.0 THEN 1.162004 + 0.011464*bp_rp + 0.049255*bp_rp*bp_rp
                                           - 0.005879*bp_rp*bp_rp*bp_rp
                    ELSE 1.057572 + 0.140537*bp_rp
                END
            ) AS corrected_bp_rp_excess,
            SQRT(
                (pmra*pmra*pmdec_error*pmdec_error
                 - 2.0*pmra_pmdec_corr*pmra_error*pmdec_error*pmra*pmdec
                 + pmdec*pmdec*pmra_error*pmra_error)
                / NULLIF(pmra_error*pmra_error*pmdec_error*pmdec_error
                         *(1.0 - pmra_pmdec_corr*pmra_pmdec_corr), 0)
            ) AS pm_significance,
            SQRT(ra_error*ra_error*dec_error*dec_error
                 *(1.0 - ra_dec_corr*ra_dec_corr)) AS position_error
        FROM gaia_dr3.gaia_source
        {where}
    """

    cursor.execute(query)
    conn.commit()

    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"  -> {count:,} rows written to {table_name}")

    # Create index on random_index for fast random sampling
    print("  Creating index on random_index...")
    cursor.execute(
        f"CREATE INDEX ON {table_name} (random_index)")
    conn.commit()

    cursor.close()
    conn.close()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
