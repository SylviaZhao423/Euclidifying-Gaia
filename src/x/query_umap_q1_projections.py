#!/usr/bin/env python3
"""
Query and project external sources into Q1 UMAP space, and cache as CSV.

This is the single source of truth for all projected coordinates.
Downstream analyze_* scripts load these CSVs instead of re-projecting.

Projections:
  A. Q1 features     — query Euclid features for Q1 sources (no projection)
  B. ERO features     — query ERO VIS morphological features (no projection)
  C. Gaia random      — query random Gaia DR3, project + run XGBoost
  D. Gaia high-lat    — same as C but |b|>10 and |ecl_lat|>45

Output CSVs:
  output/umap/arrays/q1_features.csv
  output/umap/arrays/ero_features.csv
  output/umap/arrays/xgb_gaia_projected.csv
  output/umap/arrays/xgb_gaia_hilat_projected.csv  (unless --no-hilat)

Usage:
  PGUSER=siwei_zhao_2025 python src/query_umap_q1_projections.py
  PGUSER=siwei_zhao_2025 python src/query_umap_q1_projections.py --no-hilat
  PGUSER=siwei_zhao_2025 python src/query_umap_q1_projections.py --only q1
  PGUSER=siwei_zhao_2025 python src/query_umap_q1_projections.py --only ero
  PGUSER=siwei_zhao_2025 python src/query_umap_q1_projections.py --only gaia
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
MODEL_PATH  = PROJECT_DIR / "output" / "xgboost_q1" / "xgb_classifier_q1.json"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
MAG_CUT   = 17

# Q1 Euclid features to query
Q1_DB_COLS = ["ellipticity", "flux_vis_psf", "fluxerr_vis_psf"]

# ERO VIS features to query
ERO_FEATURES = [
    "e_v_class_star", "e_v_flux_radius",
    "e_v_ellipticity", "e_v_elongation",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_connection(username: str):
    import psycopg2
    return psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)


def _load_q1():
    """Load Q1 embedding, mean, std, and reducer."""
    q1 = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    reducer = joblib.load(ARRAYS_DIR / "q1_indiv_reducer.joblib")
    std_safe = np.where(q1["std"] > 0, q1["std"], 1.0)
    return q1, reducer, q1["mean"], std_safe


def _project(X_raw: np.ndarray, q1_mean, std_safe, reducer) -> np.ndarray:
    """Standardise with Q1 stats and project via reducer.transform()."""
    X_scaled = (X_raw - q1_mean) / std_safe
    X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0.0)
    return reducer.transform(X_scaled)


# ── A. Q1 features ──────────────────────────────────────────────────────────

def query_q1_features(username: str) -> pd.DataFrame:
    """Query Euclid features for Q1 sources."""
    conn = _get_connection(username)
    select = ",\n            ".join([f"m_{c} AS {c}" for c in Q1_DB_COLS])
    query = f"""
        SELECT
            g_source_id          AS source_id,
            g_phot_g_mean_mag    AS phot_g_mean_mag,
            p_phz_classification AS phz,
            {select}
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep
        WHERE g_phot_g_mean_mag >= {MAG_CUT}
          AND g_source_id IS NOT NULL
        ORDER BY g_source_id
    """
    print("  Querying crossmatch_q1_mer_phz_vis_gaia_sep for Euclid features...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Compute flux significance
    df["fluxsig_vis_psf"] = (
        df["flux_vis_psf"] / df["fluxerr_vis_psf"].replace(0, np.nan)
    )

    df["source_id"] = df["source_id"].astype(str)
    print(f"  -> {len(df):,} sources")
    return df


# ── B. ERO features ─────────────────────────────────────────────────────────

def query_ero_features(username: str) -> pd.DataFrame:
    """Query ERO VIS morphological features."""
    conn = _get_connection(username)
    feats = ", ".join(ERO_FEATURES)
    query = f"""
        SELECT
            g_source_id       AS source_id,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            e_v_class_star    AS ero_class,
            {feats}
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE g_phot_g_mean_mag >= {MAG_CUT}
          AND g_source_id IS NOT NULL
        ORDER BY g_source_id
    """
    print("  Querying crossmatch_ero_gaia_0_3 for VIS features...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    for col in ERO_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[~np.isfinite(df[col]), col] = np.nan

    df["source_id"] = df["source_id"].astype(str)
    print(f"  -> {len(df):,} sources")
    return df


# ── C/D. Gaia random ────────────────────────────────────────────────────────

def query_gaia_subset(username: str, n_sample: int,
                      extra_where: str = "") -> pd.DataFrame:
    """Query a random Gaia DR3 subset with the UMAP/XGBoost feature columns."""
    import psycopg2

    raw_select = """
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

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)

    if extra_where:
        # Use pre-filtered hilat table (derived columns already computed).
        # The table has ~140M of ~1.8B Gaia rows (~1 in 13), so we need a
        # wider random_index window to collect enough sources.
        hilat_table = f"{username}.gaia_source_hilat"
        ri_threshold = int(n_sample * 15)
        query = f"""
            SELECT
                source_id,
                ra, dec,
                b, ecl_lat,
                phot_g_mean_mag,
                parallax_over_error, astrometric_gof_al, astrometric_chi2_al,
                astrometric_excess_noise_sig, ipd_gof_harmonic_amplitude, ruwe,
                phot_g_mean_flux_over_error, phot_bp_mean_flux_over_error,
                phot_rp_mean_flux_over_error,
                bp_g, g_rp, corrected_bp_rp_excess,
                pm_significance, position_error
            FROM {hilat_table}
            WHERE random_index < {ri_threshold}
            ORDER BY random_index
            LIMIT {n_sample}
        """
    else:
        ri_threshold = int(n_sample * 3)
        where_clause = (f"WHERE random_index < {ri_threshold}"
                        f" AND phot_g_mean_mag >= {MAG_CUT}"
                        f" AND phot_g_mean_mag IS NOT NULL")
        query = f"""
            SELECT
                source_id,
                ra, dec,
                b, ecl_lat,
                phot_g_mean_mag,
                {raw_select}
            FROM gaia_dr3.gaia_source
            {where_clause}
            ORDER BY random_index
            LIMIT {n_sample}
        """
    label = "high-lat " if extra_where else ""
    print(f"  Querying ~{n_sample:,} {label}Gaia DR3 sources (G >= {MAG_CUT})...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["source_id"] = df["source_id"].astype(str)
    print(f"  -> {len(df):,} sources")
    return df


def project_and_predict_gaia(df: pd.DataFrame, q1_mean, std_safe, reducer):
    """Project Gaia sources into Q1 UMAP space and run XGBoost."""
    import xgboost as xgb

    X = df[SELECTED_GAIA_PARAMETERS].values.astype(float)

    print("  Projecting into Q1 UMAP space...")
    emb = _project(X, q1_mean, std_safe, reducer)
    df = df.copy()
    df["umap1"] = emb[:, 0]
    df["umap2"] = emb[:, 1]
    print(f"  Projected {len(emb):,} sources")

    print("  Applying XGBoost classifier...")
    bst = xgb.Booster()
    bst.load_model(str(MODEL_PATH))
    dmat = xgb.DMatrix(X, feature_names=SELECTED_GAIA_PARAMETERS)
    df["gal_prob"] = bst.predict(dmat)
    n_gal = (df["gal_prob"] >= 0.5).sum()
    n_star = (df["gal_prob"] < 0.5).sum()
    print(f"  Predicted galaxies (P>=0.5): {n_gal:,}  |  stars: {n_star:,}")

    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Query and project external sources into Q1 UMAP space.")
    parser.add_argument("--only", choices=["q1", "ero", "gaia"],
                        help="Run only one projection (default: all).")
    parser.add_argument("--no-hilat", action="store_true",
                        help="Skip high-latitude Gaia subset.")
    parser.add_argument("--n-sample", type=int, default=50_000,
                        help="Number of random Gaia sources (default: 50000).")
    args = parser.parse_args()

    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set.")
        return 1

    do_all = args.only is None
    print("=" * 60)

    # ── A. Q1 features ───────────────────────────────────────────────────────
    if do_all or args.only == "q1":
        csv_path = ARRAYS_DIR / "q1_features.csv"
        print("[A] Q1 Euclid features...")
        df_q1 = query_q1_features(username)
        save_cols = (["source_id", "phot_g_mean_mag", "phz"]
                     + Q1_DB_COLS + ["fluxsig_vis_psf"])
        df_q1[save_cols].to_csv(csv_path, index=False)
        print(f"  Saved -> {csv_path}\n")

    # ── B. ERO features ──────────────────────────────────────────────────────
    if do_all or args.only == "ero":
        csv_path = ARRAYS_DIR / "ero_features.csv"
        print("[B] ERO VIS features...")
        df_ero = query_ero_features(username)
        df_ero.to_csv(csv_path, index=False)
        print(f"  Saved -> {csv_path}\n")

    # ── C. Gaia random ───────────────────────────────────────────────────────
    if do_all or args.only == "gaia":
        _, reducer, q1_mean, std_safe = _load_q1()

        csv_path = ARRAYS_DIR / "xgb_gaia_projected.csv"
        print("[C] Gaia random subset...")
        df_gaia = query_gaia_subset(username, args.n_sample)
        df_gaia = project_and_predict_gaia(df_gaia, q1_mean, std_safe, reducer)
        df_gaia.to_csv(csv_path, index=False)
        print(f"  Saved -> {csv_path}\n")

        # ── D. Gaia high-lat ─────────────────────────────────────────────────
        if not args.no_hilat:
            csv_path = ARRAYS_DIR / "xgb_gaia_hilat_projected.csv"
            print("[D] Gaia high-lat subset...")
            df_hilat = query_gaia_subset(
                username, args.n_sample,
                extra_where="ABS(b) > 10 AND ABS(ecl_lat) > 45")
            df_hilat = project_and_predict_gaia(
                df_hilat, q1_mean, std_safe, reducer)
            df_hilat.to_csv(csv_path, index=False)
            print(f"  Saved -> {csv_path}\n")

    print("=" * 60)
    print("Done. Downstream analyze_* scripts can now use --csv to skip DB queries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
