#!/usr/bin/env python3
"""
Query 50k random Gaia DR3 sources for model application.

Saves the 14 XGBoost features to npz for use by analysis scripts.
Derived columns (corrected_bp_rp_excess, pm_significance, position_error)
are computed in SQL.

Outputs (into output/xgboost_q1/):
  gaia_random_50k.npz     — 50k random Gaia (G >= 17)
  gaia_hilat_50k.npz      — 50k high-latitude Gaia (|b| > 25)

Usage:
    PGUSER=siwei_zhao_2025 python src/query_gaia_random.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR  = PROJECT_DIR / "output" / "gaia_50k"

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'
MAG_CUT   = 17
N_SAMPLE  = 50_000

RAW_SELECT = """
    parallax_over_error, astrometric_gof_al, astrometric_chi2_al,
    astrometric_excess_noise_sig, ipd_gof_harmonic_amplitude, ruwe,
    phot_g_mean_flux_over_error, phot_bp_mean_flux_over_error,
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
"""


def query_and_save(username, sample='random'):
    """Query Gaia sources and save to npz.
    sample: 'random', 'hilat', or 'lolat'
    """
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    feat_names = list(SELECTED_GAIA_PARAMETERS)

    if sample == 'hilat':
        hilat_table = f"{username}.gaia_source_hilat"
        ri_threshold = int(N_SAMPLE * 15)
        query = f"""
            SELECT
                source_id, ra, dec, phot_g_mean_mag,
                parallax_over_error, astrometric_gof_al, astrometric_chi2_al,
                astrometric_excess_noise_sig, ipd_gof_harmonic_amplitude, ruwe,
                phot_g_mean_flux_over_error, phot_bp_mean_flux_over_error,
                phot_rp_mean_flux_over_error,
                bp_g, g_rp, corrected_bp_rp_excess,
                pm_significance, position_error
            FROM {hilat_table}
            WHERE random_index < {ri_threshold}
            ORDER BY random_index
            LIMIT {N_SAMPLE}
        """
        label = "high-lat"
        out_name = "gaia_hilat_50k.npz"
    elif sample == 'lolat':
        lolat_table = f"{username}.gaia_source_lolat"
        ri_threshold = int(N_SAMPLE * 15)
        query = f"""
            SELECT
                source_id, ra, dec, phot_g_mean_mag,
                parallax_over_error, astrometric_gof_al, astrometric_chi2_al,
                astrometric_excess_noise_sig, ipd_gof_harmonic_amplitude, ruwe,
                phot_g_mean_flux_over_error, phot_bp_mean_flux_over_error,
                phot_rp_mean_flux_over_error,
                bp_g, g_rp, corrected_bp_rp_excess,
                pm_significance, position_error
            FROM {lolat_table}
            WHERE random_index < {ri_threshold}
            ORDER BY random_index
            LIMIT {N_SAMPLE}
        """
        label = "low-lat"
        out_name = "gaia_lolat_50k.npz"
    else:
        ri_threshold = int(N_SAMPLE * 3)
        query = f"""
            SELECT
                source_id, ra, dec, phot_g_mean_mag,
                {RAW_SELECT}
            FROM gaia_dr3.gaia_source
            WHERE random_index < {ri_threshold}
              AND phot_g_mean_mag >= {MAG_CUT}
              AND phot_g_mean_mag IS NOT NULL
            ORDER BY random_index
            LIMIT {N_SAMPLE}
        """
        label = "random"
        out_name = "gaia_random_50k.npz"

    print(f"  Querying {N_SAMPLE:,} {label} Gaia DR3 sources (G >= {MAG_CUT})...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    for c in feat_names:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    print(f"  -> {len(df):,} sources returned")

    out_path = OUTPUT_DIR / out_name
    np.savez(out_path,
             source_ids=df["source_id"].values.astype(str),
             ra=df["ra"].values.astype(float),
             dec=df["dec"].values.astype(float),
             phot_g_mean_mag=df["phot_g_mean_mag"].values.astype(float),
             X=df[feat_names].values.astype(float))
    print(f"  -> {out_path}")
    return len(df)


def main():
    username = os.environ.get("PGUSER")
    if not username:
        print("ERROR: PGUSER not set.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1] Random Gaia sample...")
    n1 = query_and_save(username, sample='random')

    print(f"\n[2] High-latitude Gaia sample...")
    n2 = query_and_save(username, sample='hilat')

    print(f"\n[3] Low-latitude Gaia sample...")
    n3 = query_and_save(username, sample='lolat')

    print(f"\nDone: {n1:,} random + {n2:,} high-lat + {n3:,} low-lat saved to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
