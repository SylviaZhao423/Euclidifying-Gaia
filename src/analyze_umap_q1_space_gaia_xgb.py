#!/usr/bin/env python3
"""
Apply the Q1 XGBoost classifier to a random Gaia DR3 subset, project into
Q1 individual UMAP space, and overlay the predictions on the cluster map.

Pipeline:
  1. Load Q1 UMAP embedding, mean/std, reducer, cluster labels
  2. Query a random Gaia DR3 subset with the same Gaia features used for UMAP
  3. Standardise with Q1 mean/std → project via reducer.transform()
  4. Apply Q1 XGBoost → star probability for each source
  5. Plot 4-panel figure:
       Col 0: Q1 cluster map (reference background)
       Col 1: All Gaia — coloured by XGBoost star probability
       Col 2: Predicted stars  (star_prob ≥ 0.5)
       Col 3: Predicted galaxies (star_prob < 0.5)

Reads:
  output/umap/arrays/q1_indiv.npz
  output/umap/arrays/q1_indiv_reducer.joblib
  output/umap/arrays/q1_cluster_labels.npy
  output/xgboost_q1/xgb_classifier_q1.json
  output/umap/arrays/xgb_gaia_projected.csv  (cached, optional)

Writes:
  output/umap/arrays/xgb_gaia_projected.csv
  output/umap/q1_individual/q1_space_gaia_xgb.png

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_umap_gaia_xgb.py
  python src/analyze_umap_gaia_xgb.py --csv          # re-plot from CSV
  python src/analyze_umap_gaia_xgb.py --n-sample 100000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "q1_individual"
MODEL_PATH  = PROJECT_DIR / "output" / "xgboost_q1" / "xgb_classifier_q1.json"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
MAG_CUT   = 17

CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#F298B6",
    "mediumpurple", "#17becf", "#d62728", "#8c564b",
    "#7f7f7f", "#bcbd22",
]

STAR_THRESH = 0.5   # XGBoost star probability threshold


# ── DB query ──────────────────────────────────────────────────────────────────

def query_gaia_subset(username: str, n_sample: int) -> pd.DataFrame:
    """Query a random Gaia DR3 subset with the UMAP/XGBoost feature columns.

    bp_g, g_rp, corrected_bp_rp_excess, pm_significance, position_error are
    derived — not raw gaia_source columns — so computed inline in SQL.
    """
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

    conn  = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = f"""
        SELECT
            source_id,
            ra, dec,
            phot_g_mean_mag,
            {raw_select}
        FROM gaia_dr3.gaia_source TABLESAMPLE SYSTEM(1)
        WHERE phot_g_mean_mag >= {MAG_CUT}
          AND phot_g_mean_mag IS NOT NULL
        LIMIT {n_sample}
    """
    print(f"  Querying ~{n_sample:,} random Gaia DR3 sources (G ≥ {MAG_CUT})...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  -> {len(df):,} sources")
    return df


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(q1_emb, q1_labels, gaia_emb, star_prob, output_dir, n_gaia):
    cluster_ids = sorted(set(q1_labels[q1_labels >= 0]))

    # Shared axis limits
    x_all = np.concatenate([q1_emb[:, 0], gaia_emb[:, 0]])
    y_all = np.concatenate([q1_emb[:, 1], gaia_emb[:, 1]])
    xlim  = (x_all.min() - 1, x_all.max() + 1)
    ylim  = (y_all.min() - 1, y_all.max() + 1)

    rng     = np.random.RandomState(0)
    MAX_PTS = 20_000

    def _sample(emb, mask):
        idx = np.where(mask)[0]
        if len(idx) > MAX_PTS:
            idx = rng.choice(idx, MAX_PTS, replace=False)
        return emb[idx]

    def _style(ax, title):
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("UMAP 1", fontsize=8)
        ax.set_ylabel("UMAP 2", fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

    fig, axes = plt.subplots(1, 4, figsize=(28, 5.5))
    fig.subplots_adjust(wspace=0.3, top=0.88)

    # ── Panel 0: Q1 source density (2D histogram) ────────────────────────────
    ax = axes[0]
    n_bins  = 250
    x_edges = np.linspace(xlim[0], xlim[1], n_bins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], n_bins + 1)
    H, _, _ = np.histogram2d(q1_emb[:, 0], q1_emb[:, 1],
                              bins=[x_edges, y_edges])
    H = np.where(H > 0, H, np.nan)
    from matplotlib.colors import LogNorm
    sc0 = ax.pcolormesh(x_edges, y_edges, H.T,
                        cmap="Blues", norm=LogNorm(vmin=1, vmax=np.nanmax(H)),
                        rasterized=True)
    fig.colorbar(sc0, ax=ax, pad=0.02, fraction=0.046,
                 label="Count (log)").ax.tick_params(labelsize=7)
    _style(ax, f"Q1 source density  (n={len(q1_emb):,})")

    # ── Panel 1: All Gaia coloured by star probability ────────────────────────
    ax = axes[1]
    # Q1 grey background
    pts = _sample(q1_emb, np.ones(len(q1_emb), dtype=bool))
    ax.scatter(pts[:, 0], pts[:, 1], c="lightgrey", s=2, alpha=0.15,
               linewidths=0, rasterized=True)
    # Gaia coloured by star prob
    idx = rng.choice(len(gaia_emb), min(MAX_PTS, len(gaia_emb)), replace=False)
    sc  = ax.scatter(
        gaia_emb[idx, 0], gaia_emb[idx, 1],
        c=star_prob[idx], cmap="RdYlBu",
        vmin=0, vmax=1, s=5, alpha=0.7,
        linewidths=0, rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("XGBoost star prob", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    _style(ax, f"Gaia random subset — all  (n={n_gaia:,})")

    # ── Panel 2: Predicted stars ──────────────────────────────────────────────
    ax = axes[2]
    pts = _sample(q1_emb, np.ones(len(q1_emb), dtype=bool))
    ax.scatter(pts[:, 0], pts[:, 1], c="lightgrey", s=2, alpha=0.15,
               linewidths=0, rasterized=True)
    m_star = star_prob >= STAR_THRESH
    pts_s  = _sample(gaia_emb, m_star)
    n_star = m_star.sum()
    ax.scatter(pts_s[:, 0], pts_s[:, 1], c="#EF9A9A", s=5, alpha=0.6,
               linewidths=0, rasterized=True)
    _style(ax, f"Predicted stars  (n={n_star:,},  prob ≥ {STAR_THRESH})")

    # ── Panel 3: Predicted galaxies ───────────────────────────────────────────
    ax = axes[3]
    pts = _sample(q1_emb, np.ones(len(q1_emb), dtype=bool))
    ax.scatter(pts[:, 0], pts[:, 1], c="lightgrey", s=2, alpha=0.15,
               linewidths=0, rasterized=True)
    m_gal  = star_prob < STAR_THRESH
    pts_g  = _sample(gaia_emb, m_gal)
    n_gal  = m_gal.sum()
    ax.scatter(pts_g[:, 0], pts_g[:, 1], c="#90CAF9", s=5, alpha=0.6,
               linewidths=0, rasterized=True)
    _style(ax, f"Predicted galaxies  (n={n_gal:,},  prob < {STAR_THRESH})")

    fig.suptitle(
        f"Random Gaia DR3 subset projected into Q1 UMAP space — Q1 XGBoost predictions\n"
        f"(G ≥ {MAG_CUT}  |  Q1 mean/std normalisation  |  Q1 reducer.transform())",
        fontsize=11, fontweight="bold", y=1.01,
    )

    out = output_dir / "q1_space_gaia_xgb.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="store_true",
                        help="Skip DB query; re-plot from saved CSV.")
    parser.add_argument("--n-sample", type=int, default=50_000,
                        help="Number of random Gaia sources to query (default: 50000).")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load Q1 UMAP ───────────────────────────────────────────────────────
    print("[1] Loading Q1 UMAP embedding, reducer, and cluster labels...")
    q1      = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    q1_emb  = q1["embedding"]
    q1_mean = q1["mean"]
    q1_std  = q1["std"]
    reducer = joblib.load(ARRAYS_DIR / "q1_indiv_reducer.joblib")

    labels_path = ARRAYS_DIR / "q1_cluster_labels.npy"
    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found. Run compute_q1_clusters.py first.")
        return 1
    q1_labels = np.load(labels_path)
    print(f"  Q1: {len(q1_emb):,} sources")

    # ── 2. Load XGBoost model ─────────────────────────────────────────────────
    print("\n[2] Loading XGBoost model...")
    import xgboost as xgb
    bst = xgb.Booster()
    bst.load_model(str(MODEL_PATH))
    print(f"  Loaded: {MODEL_PATH}")

    # ── 3. Load or query Gaia subset ──────────────────────────────────────────
    csv_path = ARRAYS_DIR / "xgb_gaia_projected.csv"

    if args.csv:
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found. Run without --csv first.")
            return 1
        print(f"\n[3] Loading Gaia subset from {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
        username = os.getenv("PGUSER")
        if not username:
            print("ERROR: PGUSER not set.")
            return 1
        print("\n[3] Querying random Gaia DR3 subset...")
        df = query_gaia_subset(username, args.n_sample)
        df["source_id"] = df["source_id"].astype(str)

        # ── 4. Project into Q1 UMAP space ─────────────────────────────────────
        print("\n[4] Projecting into Q1 UMAP space...")
        X        = df[SELECTED_GAIA_PARAMETERS].values.astype(float)
        std_safe = np.where(q1_std > 0, q1_std, 1.0)
        X_scaled = (X - q1_mean) / std_safe
        X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0.0)
        gaia_emb = reducer.transform(X_scaled)
        print(f"  Projected {len(gaia_emb):,} sources")

        # ── 5. Apply XGBoost ──────────────────────────────────────────────────
        print("\n[5] Applying XGBoost classifier...")
        dmat      = xgb.DMatrix(X_scaled, feature_names=SELECTED_GAIA_PARAMETERS)
        raw_pred  = bst.predict(dmat)
        # XGBClassifier saved via get_booster() returns raw probabilities for
        # the positive class (GALAXY=1). Star prob = 1 - galaxy_prob.
        star_prob = 1.0 - raw_pred
        print(f"  Predicted stars (prob ≥ {STAR_THRESH}): {(star_prob >= STAR_THRESH).sum():,}")
        print(f"  Predicted galaxies:                      {(star_prob <  STAR_THRESH).sum():,}")

        df["umap1"]     = gaia_emb[:, 0]
        df["umap2"]     = gaia_emb[:, 1]
        df["star_prob"] = star_prob
        df.to_csv(csv_path, index=False)
        print(f"  Saved → {csv_path}")

    # Load from CSV if --csv
    if args.csv:
        gaia_emb  = df[["umap1", "umap2"]].values
        star_prob = df["star_prob"].values

    # ── 6. Plot ───────────────────────────────────────────────────────────────
    print("\n[6] Plotting...")
    plot(q1_emb, q1_labels, gaia_emb, star_prob, OUTPUT_DIR, n_gaia=len(df))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
