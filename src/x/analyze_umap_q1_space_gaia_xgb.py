#!/usr/bin/env python3
"""
Plot random Gaia DR3 sources projected into Q1 UMAP space, coloured by
XGBoost P(gal) predictions.

Reads pre-projected CSVs produced by query_umap_q1_projections.py — no DB
query needed.

Pipeline:
  1. Load Q1 UMAP embedding + cluster labels
  2. Load projected Gaia sources from CSV (features, UMAP coords, P(gal))
  3. Plot 4-panel figure:
       Col 0: Gaia projected density (2D histogram)
       Col 1: All Gaia — coloured by XGBoost P(gal)
       Col 2: Predicted stars  (P(gal) < 0.5)
       Col 3: Predicted galaxies (P(gal) ≥ 0.5)

Reads:
  output/umap/arrays/q1_indiv.npz
  output/umap/arrays/q1_cluster_labels.npy
  output/umap/arrays/xgb_gaia_projected.csv       (from query_umap_q1_projections.py)
  output/umap/arrays/xgb_gaia_hilat_projected.csv  (with --hilat)

Writes:
  output/umap/q1_individual/q1_space_gaia_xgb.png

Usage:
  python src/analyze_umap_q1_space_gaia_xgb.py
  python src/analyze_umap_q1_space_gaia_xgb.py --hilat
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "q1_individual"

MAG_CUT   = 17

CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#F298B6",
    "mediumpurple", "#17becf", "#d62728", "#8c564b",
    "#7f7f7f", "#bcbd22",
]

GAL_THRESH = 0.5   # XGBoost galaxy probability threshold


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(q1_emb, q1_labels, gaia_emb, gal_prob, output_dir, n_gaia,
         suffix="", title_extra=""):
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

    # ── Panel 0: Gaia projected source density (2D histogram) ───────────────
    ax = axes[0]
    n_bins  = 250
    x_edges = np.linspace(xlim[0], xlim[1], n_bins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], n_bins + 1)
    H, _, _ = np.histogram2d(gaia_emb[:, 0], gaia_emb[:, 1],
                              bins=[x_edges, y_edges])
    H = np.where(H > 0, H, np.nan)
    from matplotlib.colors import LogNorm
    sc0 = ax.pcolormesh(x_edges, y_edges, H.T,
                        cmap="Blues", norm=LogNorm(vmin=1, vmax=np.nanmax(H)),
                        rasterized=True)
    fig.colorbar(sc0, ax=ax, pad=0.02, fraction=0.046,
                 label="Count (log)").ax.tick_params(labelsize=7)
    _style(ax, f"Gaia projected density  (n={n_gaia:,})")

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
        c=gal_prob[idx], cmap="RdYlBu_r",
        vmin=0, vmax=1, s=5, alpha=0.7,
        linewidths=0, rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("XGBoost P(gal)", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    _style(ax, f"Gaia random subset — all  (n={n_gaia:,})")

    # ── Panel 2: Predicted stars ──────────────────────────────────────────────
    ax = axes[2]
    pts = _sample(q1_emb, np.ones(len(q1_emb), dtype=bool))
    ax.scatter(pts[:, 0], pts[:, 1], c="lightgrey", s=2, alpha=0.15,
               linewidths=0, rasterized=True)
    m_star = gal_prob < GAL_THRESH
    pts_s  = _sample(gaia_emb, m_star)
    n_star = m_star.sum()
    ax.scatter(pts_s[:, 0], pts_s[:, 1], c="#EF9A9A", s=5, alpha=0.6,
               linewidths=0, rasterized=True)
    _style(ax, f"Predicted stars  (n={n_star:,},  P(gal) < {GAL_THRESH})")

    # ── Panel 3: Predicted galaxies ───────────────────────────────────────────
    ax = axes[3]
    pts = _sample(q1_emb, np.ones(len(q1_emb), dtype=bool))
    ax.scatter(pts[:, 0], pts[:, 1], c="lightgrey", s=2, alpha=0.15,
               linewidths=0, rasterized=True)
    m_gal  = gal_prob >= GAL_THRESH
    pts_g  = _sample(gaia_emb, m_gal)
    n_gal  = m_gal.sum()
    ax.scatter(pts_g[:, 0], pts_g[:, 1], c="#90CAF9", s=5, alpha=0.6,
               linewidths=0, rasterized=True)
    _style(ax, f"Predicted galaxies  (n={n_gal:,},  P(gal) ≥ {GAL_THRESH})")

    fig.suptitle(
        f"Random Gaia DR3 subset projected into Q1 UMAP space — Q1 XGBoost predictions\n"
        f"(G ≥ {MAG_CUT}  |  Q1 mean/std normalisation  |  Q1 reducer.transform()"
        f"{title_extra})",
        fontsize=11, fontweight="bold", y=1.01,
    )

    out = output_dir / f"q1_space_gaia_xgb{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot projected Gaia sources in Q1 UMAP space coloured by XGBoost P(gal)"
    )
    parser.add_argument("--no-hilat", action="store_true",
                        help="Skip high-latitude subset.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load Q1 UMAP ───────────────────────────────────────────────────────
    print("[1] Loading Q1 UMAP embedding and cluster labels...")
    q1      = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    q1_emb  = q1["embedding"]

    labels_path = ARRAYS_DIR / "q1_cluster_labels.npy"
    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found. Run analyze_umap_q1_clusters.py first.")
        return 1
    q1_labels = np.load(labels_path)
    print(f"  Q1: {len(q1_emb):,} sources")

    # ── 2. Variants to run ───────────────────────────────────────────────────
    variants = [
        ("xgb_gaia_projected.csv", "", ""),
    ]
    if not args.no_hilat:
        variants.append(
            ("xgb_gaia_hilat_projected.csv", "_hilat", "  |  |b|>10, |ecl_lat|>45"),
        )

    for csv_name, suffix, title_extra in variants:
        csv_path = ARRAYS_DIR / csv_name
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found.")
            print("  Run first:  python src/query_umap_q1_projections.py")
            return 1

        print(f"\n[2] Loading projected Gaia sources from {csv_name}...")
        df = pd.read_csv(csv_path)
        gaia_emb = df[["umap1", "umap2"]].values
        gal_prob = df["gal_prob"].values
        print(f"  {len(df):,} sources")
        print(f"  Predicted galaxies (P(gal) ≥ {GAL_THRESH}): {(gal_prob >= GAL_THRESH).sum():,}")
        print(f"  Predicted stars    (P(gal) < {GAL_THRESH}): {(gal_prob <  GAL_THRESH).sum():,}")

        print(f"\n[3] Plotting{suffix}...")
        plot(q1_emb, q1_labels, gaia_emb, gal_prob, OUTPUT_DIR, n_gaia=len(df),
             suffix=suffix, title_extra=title_extra)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
