#!/usr/bin/env python3
"""
Report version of Q1 UMAP Gaia projection — coloured by ERO-mix 8.63 model P(gal).

Mirrors analyze_umap_q1_space_gaia_xgb_insert.py but computes P(gal) on-the-fly
from the 14 Gaia features in the projected CSV using the ERO-mix 8.63 ensemble
(10 seed models trained on Q1 + ERO sources at sg=8.63).

Produces a 2-panel figure per variant + a 2x2 stacked figure:
  Col 0: Gaia projected density (2D histogram)
  Col 1: All Gaia — coloured by ERO-mix XGBoost P(gal)

Reads:
  output/umap/arrays/q1_indiv.npz
  output/umap/arrays/q1_cluster_labels.npy
  output/umap/arrays/projected/gaia_random_projected.csv
  output/umap/arrays/projected/gaia_hilat_projected.csv
  output/xgboost_mix_q1_ero/ratio_sg8.63_all_ero/seed_models/*.json

Writes:
  output/umap/q1_individual/insert/q1_space_gaia_xgb_ero.png
  output/umap/q1_individual/insert/q1_space_gaia_xgb_ero_hilat.png
  output/umap/q1_individual/insert/q1_space_gaia_xgb_ero_stacked.png

Usage:
  python src/analyze_umap_q1_space_gaia_xgb_ero_insert.py
  python src/analyze_umap_q1_space_gaia_xgb_ero_insert.py --only random
  python src/analyze_umap_q1_space_gaia_xgb_ero_insert.py --only hilat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR    = PROJECT_DIR / "output" / "umap" / "arrays"
PROJECTED_DIR = ARRAYS_DIR / "projected"
OUTPUT_DIR    = PROJECT_DIR / "output" / "umap" / "q1_individual" / "insert"

ERO_MIX_SEED_DIR = (PROJECT_DIR / "output" / "xgboost_mix_q1_ero"
                    / "ratio_sg8.63_all_ero" / "seed_models")
SEEDS = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]

MAG_CUT   = 17
GAL_THRESH = 0.5

FEAT_NAMES = list(SELECTED_GAIA_PARAMETERS)


def ensemble_predict(X: np.ndarray) -> np.ndarray:
    """Predict P(gal) using ERO-mix 8.63 10-seed ensemble mean."""
    probas = []
    for seed in SEEDS:
        mp = ERO_MIX_SEED_DIR / f"xgb_classifier_q1_seed{seed}.json"
        if not mp.exists():
            print(f"  WARNING: {mp.name} not found, skipping")
            continue
        m = xgb.XGBClassifier()
        m.load_model(str(mp))
        probas.append(m.predict_proba(X)[:, 1])
    if not probas:
        raise FileNotFoundError(f"No ERO-mix seed models found in {ERO_MIX_SEED_DIR}")
    return np.mean(probas, axis=0)


def _add_cluster_labels(ax, centroids):
    """Add subtle cluster reference labels (C0–C4) at centroid positions."""
    import matplotlib.patheffects as pe
    y_offsets = {0: -3, 1: -3, 2: -3}
    outside = {0, 1, 2}
    for k, (cx, cy) in centroids.items():
        cy_adj = cy + y_offsets.get(k, 0)
        if k in outside:
            ax.text(cx, cy_adj, f"C{k}", fontsize=13, fontweight="bold",
                    color="dimgray", alpha=1, ha="center", va="center", zorder=5)
        else:
            ax.text(cx, cy_adj, f"C{k}", fontsize=13, fontweight="bold",
                    color="dimgray", alpha=1, ha="center", va="center", zorder=5,
                    path_effects=[pe.withStroke(linewidth=2, foreground="white", alpha=0.3)])


def plot(q1_emb, gaia_emb, gal_prob, output_dir, n_gaia, suffix="",
         density_vmax=None, label="Gaia random subset",
         cluster_centroids=None):
    x_all = np.concatenate([q1_emb[:, 0], gaia_emb[:, 0]])
    y_all = np.concatenate([q1_emb[:, 1], gaia_emb[:, 1]])
    xlim  = (x_all.min() - 1, x_all.max() + 1)
    ylim  = (y_all.min() - 1, y_all.max() + 1)

    rng     = np.random.RandomState(0)
    MAX_PTS = 20_000

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.32)

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
    if cluster_centroids is not None:
        _add_cluster_labels(ax, cluster_centroids)

    # Panel 1: All Gaia coloured by ERO-mix P(gal)
    ax = axes[1]
    q1_idx = rng.choice(len(q1_emb), min(MAX_PTS, len(q1_emb)), replace=False)
    ax.scatter(q1_emb[q1_idx, 0], q1_emb[q1_idx, 1], c="lightgrey", s=2,
               alpha=0.15, linewidths=0, rasterized=True)
    idx = rng.choice(len(gaia_emb), min(MAX_PTS, len(gaia_emb)), replace=False)
    sc = ax.scatter(
        gaia_emb[idx, 0], gaia_emb[idx, 1],
        c=gal_prob[idx], cmap="RdYlBu_r",
        vmin=0, vmax=1, s=5, alpha=0.7,
        linewidths=0, rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("ERO-mix XGBoost P(gal)", fontsize=11)
    cb.ax.tick_params(labelsize=11)
    _style(ax, f"ERO-mix XGBoost P(gal)  (n={n_gaia:,})")

    out = output_dir / f"q1_space_gaia_xgb_ero{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")


def plot_stacked(q1_emb, datasets, output_dir, density_vmax,
                 cluster_centroids=None):
    """2×2 stacked figure: rows = random/hilat, cols = density / ERO-mix P(gal)."""
    all_x = [q1_emb[:, 0]]
    all_y = [q1_emb[:, 1]]
    for gaia_emb, _, _, _, _ in datasets:
        all_x.append(gaia_emb[:, 0])
        all_y.append(gaia_emb[:, 1])
    x_cat = np.concatenate(all_x)
    y_cat = np.concatenate(all_y)
    xlim = (x_cat.min() - 1, x_cat.max() + 1)
    ylim = (y_cat.min() - 1, y_cat.max() + 1)

    rng = np.random.RandomState(0)
    MAX_PTS = 20_000
    n_bins = 250
    x_edges = np.linspace(xlim[0], xlim[1], n_bins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], n_bins + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 5.8))
    fig.subplots_adjust(wspace=0.22, hspace=0.1)

    density_norm = LogNorm(vmin=1, vmax=density_vmax)
    last_density_im = None
    last_pgal_sc = None

    for row, (gaia_emb, gal_prob, n_gaia, suffix, label) in enumerate(datasets):
        ax = axes[row, 0]
        H, _, _ = np.histogram2d(gaia_emb[:, 0], gaia_emb[:, 1],
                                  bins=[x_edges, y_edges])
        H = np.where(H > 0, H, np.nan)
        last_density_im = ax.pcolormesh(x_edges, y_edges, H.T,
                            cmap="Blues", norm=density_norm,
                            rasterized=True)
        if cluster_centroids is not None:
            _add_cluster_labels(ax, cluster_centroids)

        ax = axes[row, 1]
        q1_idx = rng.choice(len(q1_emb), min(MAX_PTS, len(q1_emb)), replace=False)
        ax.scatter(q1_emb[q1_idx, 0], q1_emb[q1_idx, 1], c="lightgrey", s=2,
                   alpha=0.15, linewidths=0, rasterized=True)
        idx = rng.choice(len(gaia_emb), min(MAX_PTS, len(gaia_emb)), replace=False)
        last_pgal_sc = ax.scatter(
            gaia_emb[idx, 0], gaia_emb[idx, 1],
            c=gal_prob[idx], cmap="RdYlBu_r",
            vmin=0, vmax=1, s=5, alpha=0.7,
            linewidths=0, rasterized=True,
        )

        for col in range(2):
            a = axes[row, col]
            a.set_xlim(*xlim)
            a.set_ylim(*ylim)
            a.tick_params(labelsize=10)
            a.grid(alpha=0.2)
            if row == 1:
                a.set_xlabel("UMAP 1", fontsize=11)
            else:
                a.set_xticklabels([])
            if col == 0:
                a.set_ylabel("")
                a.text(-0.18, 0.5, label,
                       transform=a.transAxes, rotation=90,
                       fontsize=12, fontweight='bold',
                       ha='center', va='center')
                a.text(-0.1, 0.5, "UMAP 2",
                       transform=a.transAxes, rotation=90,
                       fontsize=10,
                       ha='center', va='center')
            else:
                a.set_ylabel("")

    cb0 = fig.colorbar(last_density_im, ax=axes[:, 0].tolist(),
                       pad=0.02, fraction=0.046, shrink=1.0, aspect=30)
    cb0.set_label("Count (log)", fontsize=11)
    cb0.ax.tick_params(labelsize=10)

    cb1 = fig.colorbar(last_pgal_sc, ax=axes[:, 1].tolist(),
                       pad=0.02, fraction=0.046, shrink=1.0, aspect=30)
    cb1.set_label("ERO-mix XGBoost P(gal)", fontsize=11)
    cb1.ax.tick_params(labelsize=10)

    axes[0, 0].set_title("Projected density", fontsize=13, fontweight="bold")
    axes[0, 1].set_title("ERO-mix XGBoost P(gal)", fontsize=13, fontweight="bold")

    out = output_dir / "q1_space_gaia_xgb_ero_stacked.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")


def plot_stacked_models(q1_emb, gaia_emb, gal_prob_q1, gal_prob_ero,
                        output_dir, density_vmax, cluster_centroids=None):
    """2×2 stacked figure for the random sample: rows = Q1 model / ERO-mix model,
    cols = density / P(gal)."""
    x_cat = np.concatenate([q1_emb[:, 0], gaia_emb[:, 0]])
    y_cat = np.concatenate([q1_emb[:, 1], gaia_emb[:, 1]])
    xlim = (x_cat.min() - 1, x_cat.max() + 1)
    ylim = (y_cat.min() - 1, y_cat.max() + 1)

    rng = np.random.RandomState(0)
    MAX_PTS = 20_000
    n_bins = 250
    x_edges = np.linspace(xlim[0], xlim[1], n_bins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], n_bins + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 5.8))
    fig.subplots_adjust(wspace=0.22, hspace=0.1)

    density_norm = LogNorm(vmin=1, vmax=density_vmax)
    last_density_im = None
    last_pgal_sc = None

    rows = [
        ("Q1-only model",   gal_prob_q1),
        ("ERO-mix 8.63",    gal_prob_ero),
    ]

    for row, (label, gal_prob) in enumerate(rows):
        ax = axes[row, 0]
        H, _, _ = np.histogram2d(gaia_emb[:, 0], gaia_emb[:, 1],
                                  bins=[x_edges, y_edges])
        H = np.where(H > 0, H, np.nan)
        last_density_im = ax.pcolormesh(x_edges, y_edges, H.T,
                            cmap="Blues", norm=density_norm,
                            rasterized=True)
        if cluster_centroids is not None:
            _add_cluster_labels(ax, cluster_centroids)

        ax = axes[row, 1]
        q1_idx = rng.choice(len(q1_emb), min(MAX_PTS, len(q1_emb)), replace=False)
        ax.scatter(q1_emb[q1_idx, 0], q1_emb[q1_idx, 1], c="lightgrey", s=2,
                   alpha=0.15, linewidths=0, rasterized=True)
        idx = rng.choice(len(gaia_emb), min(MAX_PTS, len(gaia_emb)), replace=False)
        last_pgal_sc = ax.scatter(
            gaia_emb[idx, 0], gaia_emb[idx, 1],
            c=gal_prob[idx], cmap="RdYlBu_r",
            vmin=0, vmax=1, s=5, alpha=0.7,
            linewidths=0, rasterized=True,
        )

        for col in range(2):
            a = axes[row, col]
            a.set_xlim(*xlim)
            a.set_ylim(*ylim)
            a.tick_params(labelsize=10)
            a.grid(alpha=0.2)
            if row == 1:
                a.set_xlabel("UMAP 1", fontsize=11)
            else:
                a.set_xticklabels([])
            if col == 0:
                a.set_ylabel("")
                a.text(-0.18, 0.5, label,
                       transform=a.transAxes, rotation=90,
                       fontsize=12, fontweight='bold',
                       ha='center', va='center')
                a.text(-0.1, 0.5, "UMAP 2",
                       transform=a.transAxes, rotation=90,
                       fontsize=10,
                       ha='center', va='center')
            else:
                a.set_ylabel("")

    cb0 = fig.colorbar(last_density_im, ax=axes[:, 0].tolist(),
                       pad=0.02, fraction=0.046, shrink=1.0, aspect=30)
    cb0.set_label("Count (log)", fontsize=11)
    cb0.ax.tick_params(labelsize=10)

    cb1 = fig.colorbar(last_pgal_sc, ax=axes[:, 1].tolist(),
                       pad=0.02, fraction=0.046, shrink=1.0, aspect=30)
    cb1.set_label("XGBoost P(gal)", fontsize=11)
    cb1.ax.tick_params(labelsize=10)

    axes[0, 0].set_title("Projected density", fontsize=13, fontweight="bold")
    axes[0, 1].set_title("XGBoost P(gal)", fontsize=13, fontweight="bold")

    out = output_dir / "q1_space_gaia_xgb_q1_vs_ero_random_stacked.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")


def _compute_density_max(gaia_emb, q1_emb, n_bins=250):
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

    cluster_centroids = None
    clabel_path = ARRAYS_DIR / "q1_cluster_labels.npy"
    if clabel_path.exists():
        labels = np.load(clabel_path)
        cluster_centroids = {}
        for k in sorted(set(labels[labels >= 0])):
            mask_k = labels == k
            cluster_centroids[k] = (
                q1_emb[mask_k, 0].mean(),
                q1_emb[mask_k, 1].mean(),
            )
        print(f"  Loaded cluster centroids: {list(cluster_centroids.keys())}")

    seed_count = sum(1 for s in SEEDS
                     if (ERO_MIX_SEED_DIR / f"xgb_classifier_q1_seed{s}.json").exists())
    print(f"[2] ERO-mix 8.63 ensemble: {seed_count}/{len(SEEDS)} seed models")

    variants = [
        ("gaia_random_projected.csv", "", "Gaia random subset"),
        ("gaia_hilat_projected.csv", "_hilat", "Gaia high-latitude subset"),
    ]
    if args.only == "random":
        variants = [variants[0]]
    elif args.only == "hilat":
        variants = [variants[1]]

    datasets = []
    random_q1_pgal = None
    random_emb = None
    for csv_name, suffix, label in variants:
        csv_path = PROJECTED_DIR / csv_name
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found.")
            print("  Run first:  python src/query_umap_q1_projections.py")
            return 1
        print(f"\n[3] Loading {csv_name}...")
        df = pd.read_csv(csv_path)
        gaia_emb = df[["umap1", "umap2"]].values
        X = df[FEAT_NAMES].astype(float).values
        print(f"  {len(df):,} sources — predicting with ERO-mix 8.63 ensemble...")
        gal_prob = ensemble_predict(X)
        print(f"  ERO-mix P(gal)>=0.5: {100*(gal_prob>=0.5).mean():.1f}%  "
              f"P(gal)>=0.9: {100*(gal_prob>=0.9).mean():.1f}%")
        if suffix == "":
            random_q1_pgal = df["gal_prob"].values
            random_emb = gaia_emb
            print(f"  Q1-only  P(gal)>=0.5: {100*(random_q1_pgal>=0.5).mean():.1f}%  "
                  f"P(gal)>=0.9: {100*(random_q1_pgal>=0.9).mean():.1f}%")
        datasets.append((gaia_emb, gal_prob, len(df), suffix, label))

    shared_vmax = max(_compute_density_max(d[0], q1_emb) for d in datasets)
    print(f"\n  Shared density vmax: {shared_vmax:.0f}")

    for gaia_emb, gal_prob, n_gaia, suffix, label in datasets:
        print(f"\n[4] Plotting{suffix}...")
        plot(q1_emb, gaia_emb, gal_prob, OUTPUT_DIR, n_gaia=n_gaia,
             suffix=suffix, density_vmax=shared_vmax, label=label,
             cluster_centroids=cluster_centroids)

    if len(datasets) == 2:
        print("\n[5] Plotting stacked 2×2 (random/hilat × density/P(gal))...")
        plot_stacked(q1_emb, datasets, OUTPUT_DIR, density_vmax=shared_vmax,
                     cluster_centroids=cluster_centroids)

    if random_q1_pgal is not None:
        random_ero_pgal = next(d[1] for d in datasets if d[3] == "")
        print("\n[6] Plotting Q1-only vs ERO-mix stacked (random sample)...")
        plot_stacked_models(q1_emb, random_emb, random_q1_pgal, random_ero_pgal,
                            OUTPUT_DIR, density_vmax=shared_vmax,
                            cluster_centroids=cluster_centroids)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
