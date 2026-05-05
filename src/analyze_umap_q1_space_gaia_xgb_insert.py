#!/usr/bin/env python3
"""
Report version of Q1 UMAP Gaia projection — first 2 panels only.

Produces a 2-panel figure:
  Col 0: Gaia projected density (2D histogram)
  Col 1: All Gaia — coloured by XGBoost P(gal)

Reads:
  output/umap/arrays/q1_indiv.npz
  output/umap/arrays/q1_cluster_labels.npy
  output/umap/arrays/xgb_gaia_projected.csv

Writes:
  output/umap/q1_individual/insert/q1_space_gaia_xgb.png

Usage:
  python src/analyze_umap_q1_space_gaia_xgb_report.py
  python src/analyze_umap_q1_space_gaia_xgb_report.py --hilat
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR    = PROJECT_DIR / "output" / "umap" / "arrays"
PROJECTED_DIR = ARRAYS_DIR / "projected"
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "q1_individual" / "insert"

MAG_CUT   = 17
GAL_THRESH = 0.5


def plot(q1_emb, gaia_emb, gal_prob, output_dir, n_gaia, suffix="",
         density_vmax=None, label="Gaia random subset"):
    x_all = np.concatenate([q1_emb[:, 0], gaia_emb[:, 0]])
    y_all = np.concatenate([q1_emb[:, 1], gaia_emb[:, 1]])
    xlim  = (x_all.min() - 1, x_all.max() + 1)
    ylim  = (y_all.min() - 1, y_all.max() + 1)

    rng     = np.random.RandomState(0)
    MAX_PTS = 20_000

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.28)

    def _style(ax, title):
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("UMAP 1", fontsize=12, fontweight="bold")
        ax.set_ylabel("UMAP 2", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.2)

    # Panel 0: Gaia projected density
    ax = axes[0]
    n_bins  = 250
    x_edges = np.linspace(xlim[0], xlim[1], n_bins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], n_bins + 1)
    H, _, _ = np.histogram2d(gaia_emb[:, 0], gaia_emb[:, 1],
                              bins=[x_edges, y_edges])
    H = np.where(H > 0, H, np.nan)
    vmax = density_vmax if density_vmax is not None else np.nanmax(H)
    sc0 = ax.pcolormesh(x_edges, y_edges, H.T,
                        cmap="Blues", norm=LogNorm(vmin=1, vmax=vmax),
                        rasterized=True)
    fig.colorbar(sc0, ax=ax, pad=0.02, fraction=0.046,
                 label="Count (log)").ax.tick_params(labelsize=11)
    _style(ax, f"Projected density  (n={n_gaia:,})")

    # Panel 1: All Gaia coloured by P(gal)
    ax = axes[1]
    # Q1 grey background
    q1_idx = rng.choice(len(q1_emb), min(MAX_PTS, len(q1_emb)), replace=False)
    ax.scatter(q1_emb[q1_idx, 0], q1_emb[q1_idx, 1], c="lightgrey", s=2,
               alpha=0.15, linewidths=0, rasterized=True)
    # Gaia coloured by P(gal)
    idx = rng.choice(len(gaia_emb), min(MAX_PTS, len(gaia_emb)), replace=False)
    sc = ax.scatter(
        gaia_emb[idx, 0], gaia_emb[idx, 1],
        c=gal_prob[idx], cmap="RdYlBu_r",
        vmin=0, vmax=1, s=5, alpha=0.7,
        linewidths=0, rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("XGBoost P(gal)", fontsize=11)
    cb.ax.tick_params(labelsize=11)
    _style(ax, f"XGBoost P(gal)  (n={n_gaia:,})")

    fig.suptitle(label, fontsize=14, fontweight="bold", y=1.02)

    out = output_dir / f"q1_space_gaia_xgb{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")


def _compute_density_max(gaia_emb, q1_emb, n_bins=250):
    """Compute max histogram bin count for density colorbar scaling."""
    x_all = np.concatenate([q1_emb[:, 0], gaia_emb[:, 0]])
    y_all = np.concatenate([q1_emb[:, 1], gaia_emb[:, 1]])
    xlim = (x_all.min() - 1, x_all.max() + 1)
    ylim = (y_all.min() - 1, y_all.max() + 1)
    x_edges = np.linspace(xlim[0], xlim[1], n_bins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], n_bins + 1)
    H, _, _ = np.histogram2d(gaia_emb[:, 0], gaia_emb[:, 1],
                              bins=[x_edges, y_edges])
    return np.max(H)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["random", "hilat"],
                        help="Only produce one variant (default: both).")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1] Loading Q1 UMAP embedding...")
    q1     = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    q1_emb = q1["embedding"]
    print(f"  Q1: {len(q1_emb):,} sources")

    variants = [
        ("gaia_random_projected.csv", "", "Gaia random subset"),
        ("gaia_hilat_projected.csv", "_hilat", "Gaia high-latitude subset"),
    ]
    if args.only == "random":
        variants = [variants[0]]
    elif args.only == "hilat":
        variants = [variants[1]]

    # Load all datasets to find shared density vmax
    datasets = []
    for csv_name, suffix, label in variants:
        csv_path = PROJECTED_DIR / csv_name
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found.")
            print("  Run first:  python src/query_umap_q1_projections.py")
            return 1
        print(f"[2] Loading {csv_name}...")
        df = pd.read_csv(csv_path)
        gaia_emb = df[["umap1", "umap2"]].values
        gal_prob = df["gal_prob"].values
        print(f"  {len(df):,} sources")
        datasets.append((gaia_emb, gal_prob, len(df), suffix, label))

    # Shared density vmax across all variants
    shared_vmax = max(_compute_density_max(d[0], q1_emb) for d in datasets)
    print(f"  Shared density vmax: {shared_vmax:.0f}")

    for gaia_emb, gal_prob, n_gaia, suffix, label in datasets:
        print(f"\n[3] Plotting{suffix}...")
        plot(q1_emb, gaia_emb, gal_prob, OUTPUT_DIR, n_gaia=n_gaia,
             suffix=suffix, density_vmax=shared_vmax, label=label)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
