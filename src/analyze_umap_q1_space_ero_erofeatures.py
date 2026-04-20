#!/usr/bin/env python3
"""
Project ERO sources into Q1 individual UMAP space and colour by ERO VIS features.

ERO sources are standardised with Q1 mean/std, then projected via
Q1 reducer.transform(). Each panel shows ERO points coloured by a feature
value, with the full Q1 embedding as a grey background.

Fixed 3-column × 2-row layout:
  6-column × 4-row layout (one row per band, one col per feature type):
         class_star | flux_radius | ellipticity | elongation
  VIS  :  e_v_*    |    e_v_*    |    e_v_*    |    e_v_*    |    e_v_*   |    e_v_*
  H    :  e_h_*    |    e_h_*    |    e_h_*    |    e_h_*    |    e_h_*   |    e_h_*
  J    :  e_j_*    |    e_j_*    |    e_j_*    |    e_j_*    |    e_j_*   |    e_j_*
  Y    :  e_y_*    |    e_y_*    |    e_y_*    |    e_y_*    |    e_y_*   |    e_y_*

Upper panel = ERO stars · Lower panel = ERO galaxies

Reads:
  output/umap/arrays/q1_indiv.npz
  output/umap/arrays/q1_indiv_reducer.joblib
  output/umap/arrays/ero_indiv_rebalanced.npz
  output/umap/arrays/ero_features.csv  (cached, optional)

Writes:
  output/umap/arrays/ero_features.csv
  output/umap/q1_individual/q1_space_ero_features.png

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_umap_q1_ero_features.py
  python src/analyze_umap_q1_ero_features.py --csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "q1_individual"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
MAG_CUT   = 17

ORDERED_FEATURES = [
    "e_v_class_star", "e_v_flux_radius",
    "e_v_ellipticity", "e_v_elongation",
]
NCOLS = 4


# ── DB query ──────────────────────────────────────────────────────────────────

def query_ero_features(username: str) -> pd.DataFrame:
    """Query ERO VIS/H/J/Y morphological features from crossmatch table."""
    import psycopg2

    conn  = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    feats = ", ".join(ORDERED_FEATURES)
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

    # Replace IEEE NaN floats with proper NaN
    for col in ORDERED_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[~np.isfinite(df[col]), col] = np.nan

    print(f"  -> {len(df):,} sources")
    return df


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_ero_features(
    q1_emb:    np.ndarray,   # (N_q1, 2)
    ero_emb:   np.ndarray,   # (N_ero, 2) — projected into Q1 space
    ero_y:     np.ndarray,   # 0=STAR, 1=GALAXY
    feat_vals: dict[str, np.ndarray],
    output_dir: Path,
    filename:   str = "q1_space_ero_erofeatures.png",
    xlim=None, ylim=None,
):
    features      = [f for f in ORDERED_FEATURES if f in feat_vals
                     and np.isfinite(feat_vals[f]).any()]
    n_features    = len(features)
    ncols         = NCOLS
    nfeature_rows = int(np.ceil(n_features / ncols))

    per_panel_height = 1.2
    panel_h   = 1.0
    inner_gap = 0.04
    outer_gap = 0.32

    height_ratios = []
    for i in range(nfeature_rows):
        height_ratios.extend([panel_h, inner_gap, panel_h])
        if i < nfeature_rows - 1:
            height_ratios.append(outer_gap)

    panel_fraction = panel_h / sum(height_ratios)
    fig_width  = ncols * 3.8
    fig_height = per_panel_height / panel_fraction

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = GridSpec(len(height_ratios), ncols,
                   figure=fig, height_ratios=height_ratios,
                   hspace=0, wspace=0.35)

    def _gs_top(feat_row): return feat_row * 4
    def _gs_bot(feat_row): return feat_row * 4 + 2

    m_star = ero_y == 0
    m_gal  = ero_y == 1

    n_ero_star = m_star.sum()
    n_ero_gal  = m_gal.sum()

    for fi, feat_name in enumerate(features):
        feat_row = fi // ncols
        col      = fi % ncols
        ax_top = fig.add_subplot(gs[_gs_top(feat_row), col])
        ax_bot = fig.add_subplot(gs[_gs_bot(feat_row), col])

        vals   = feat_vals[feat_name].astype(float)
        finite = vals[np.isfinite(vals)]
        lo, hi = np.nanpercentile(finite, 2), np.nanpercentile(finite, 98)
        if lo == hi:
            lo, hi = finite.min(), finite.max()
        vals_clipped = np.clip(vals, lo, hi)

        for ax, mask, label, n in [
                (ax_top, m_star, "ERO Stars",    n_ero_star),
                (ax_bot, m_gal,  "ERO Galaxies", n_ero_gal)]:
            # Q1 grey background
            ax.scatter(
                q1_emb[:, 0], q1_emb[:, 1],
                c="lightgrey", s=2, alpha=0.15, marker="o",
                rasterized=True, linewidths=0, zorder=0,
            )
            # ERO points coloured by feature
            valid = mask & np.isfinite(vals)
            sc = ax.scatter(
                ero_emb[valid, 0], ero_emb[valid, 1],
                c=vals_clipped[valid], cmap="RdYlBu_r",
                vmin=lo, vmax=hi,
                s=6, alpha=0.7, marker="o", rasterized=True,
                linewidths=0, zorder=1,
            )
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(alpha=0.25)
            if xlim is not None: ax.set_xlim(xlim)
            if ylim is not None: ax.set_ylim(ylim)
            ax.set_ylabel(f"{label}\n(n={n:,})", fontsize=7)

        fig.colorbar(sc, ax=[ax_top, ax_bot],
                     pad=0.02, fraction=0.046).ax.tick_params(labelsize=7)
        ax_top.set_title(feat_name, fontsize=8, fontweight="bold", pad=4)

    # hide empty panels
    for fi in range(n_features, nfeature_rows * ncols):
        feat_row = fi // ncols
        col      = fi % ncols
        fig.add_subplot(gs[_gs_top(feat_row), col]).set_visible(False)
        fig.add_subplot(gs[_gs_bot(feat_row), col]).set_visible(False)

    fig.suptitle(
        f"ERO sources in Q1 UMAP space colored by ERO features  "
        f"(G ≥ {MAG_CUT}, {n_features} features)  |  Upper = ERO stars · Lower = ERO galaxies",
        fontsize=11, fontweight="bold", y=1.08,
    )
    out = output_dir / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot ERO features coloured in Q1 UMAP space"
    )
    parser.add_argument("--csv", action="store_true",
                        help="Skip DB query; re-plot from saved CSV.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load Q1 embedding + reducer ───────────────────────────────────────
    print("[1] Loading Q1 UMAP embedding and reducer...")
    q1      = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    q1_emb  = q1["embedding"]
    q1_mean = q1["mean"]
    q1_std  = q1["std"]
    reducer = joblib.load(ARRAYS_DIR / "q1_indiv_reducer.joblib")
    print(f"  Q1: {len(q1_emb):,} sources")

    # ── 2. Load ERO Gaia features for projection ──────────────────────────────
    print("\n[2] Loading ERO arrays for projection...")
    ero_npz  = np.load(ARRAYS_DIR / "ero_indiv_rebalanced.npz", allow_pickle=True)
    ero_X    = ero_npz["X_raw"].astype(float)
    ero_y    = ero_npz["y"]
    ero_sids = ero_npz["source_ids"].astype(str)
    print(f"  ERO: {len(ero_y):,} sources  "
          f"(STAR={(ero_y==0).sum():,}  GAL={(ero_y==1).sum():,})")

    # ── 3. Project ERO into Q1 UMAP space ────────────────────────────────────
    print("\n[3] Projecting ERO into Q1 UMAP space...")
    std_safe = np.where(q1_std > 0, q1_std, 1.0)
    X_scaled = (ero_X - q1_mean) / std_safe
    X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0.0)
    ero_emb  = reducer.transform(X_scaled)
    print(f"  Done. ERO embedding shape: {ero_emb.shape}")

    # ── 4. Load or query ERO VIS features ────────────────────────────────────
    csv_path = ARRAYS_DIR / "ero_features.csv"

    if args.csv:
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found. Run without --csv first.")
            return 1
        print(f"\n[4] Loading ERO VIS features from {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
        username = os.getenv("PGUSER")
        if not username:
            print("ERROR: PGUSER not set.")
            return 1
        print("\n[4] Querying ERO VIS features from DB...")
        df = query_ero_features(username)
        df["source_id"] = df["source_id"].astype(str)
        df.to_csv(csv_path, index=False)
        print(f"  Saved → {csv_path}")

    # ── 5. Match ERO source IDs to projected embedding ────────────────────────
    print("\n[5] Matching ERO features to projected sources...")
    df["source_id"] = df["source_id"].astype(str)
    sid_to_idx = {sid: i for i, sid in enumerate(df["source_id"])}
    ero_idx    = np.array([sid_to_idx.get(sid, -1) for sid in ero_sids])
    n_matched  = (ero_idx >= 0).sum()
    print(f"  Matched {n_matched:,} / {len(ero_sids):,} projected ERO sources")

    feat_vals: dict[str, np.ndarray] = {}
    for col in ORDERED_FEATURES:
        if col not in df.columns:
            continue
        arr   = np.full(len(ero_emb), np.nan)
        valid = ero_idx >= 0
        arr[valid] = pd.to_numeric(
            df[col].values[ero_idx[valid]], errors="coerce"
        )
        n_finite = np.isfinite(arr).sum()
        if n_finite > 0:
            feat_vals[col] = arr
            print(f"    {col:<30} {n_finite:>8,} finite values")
        else:
            print(f"    {col:<30}  (all NaN — skipped)")

    # ── 6. Plot ───────────────────────────────────────────────────────────────
    x_all = np.concatenate([q1_emb[:, 0], ero_emb[:, 0]])
    y_all = np.concatenate([q1_emb[:, 1], ero_emb[:, 1]])
    xlim  = (x_all.min() - 1.0, x_all.max() + 1.0)
    ylim  = (y_all.min() - 1.0, y_all.max() + 1.0)

    print("\n[6] Plotting ERO features in Q1 UMAP space...")
    plot_ero_features(
        q1_emb, ero_emb, ero_y, feat_vals, OUTPUT_DIR,
        xlim=xlim, ylim=ylim,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
