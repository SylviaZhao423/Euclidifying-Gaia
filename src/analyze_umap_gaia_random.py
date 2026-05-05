#!/usr/bin/env python3
"""
Build a UMAP embedding from the 50k random Gaia sample (no Euclid labels).

Tests whether the galaxy cluster structure emerges from Gaia parameters
alone, without any training bias from the Euclid footprint.

After fitting UMAP + HDBSCAN, the Q1 sources are projected INTO this
Gaia-based UMAP to see where labelled stars/galaxies fall.

Reads:
  output/umap/arrays/gaia_random.npz       (50k random Gaia, 14 features)
  output/umap/arrays/q1_indiv.npz          (Q1 sources for projection)
  output/xgboost_q1_op_f1/xgb_classifier_q1.json  (for P(gal) overlay)

Writes:
  output/umap/gaia_random_umap/
    gaia_random_umap.png           — UMAP coloured by density + P(gal) + clusters
    gaia_random_umap_q1_proj.png   — Q1 sources projected into Gaia UMAP
    gaia_random_reducer.joblib     — fitted UMAP reducer
    gaia_random_embedding.npz      — embedding + cluster labels

Usage:
    python src/analyze_umap_gaia_random.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "gaia_random_umap"
MODEL_PATH  = PROJECT_DIR / "output" / "xgboost_q1_op_f1" / "xgb_classifier_q1.json"

UMAP_PARAMS = dict(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)


def main() -> int:
    import umap
    import hdbscan

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feat_names = list(SELECTED_GAIA_PARAMETERS)

    # ── 1. Load Gaia random sources ──────────────────────────────────────────
    print("[1] Loading 50k random Gaia sources...")
    g = np.load(ARRAYS_DIR / "gaia_random.npz", allow_pickle=True)
    X_gaia = g["X_features"].astype(float)
    g_mag  = g["phot_g_mean_mag"].astype(float)
    print(f"  {len(X_gaia):,} sources, {len(feat_names)} features")

    # ── 2. Standardise ───────────────────────────────────────────────────────
    print("\n[2] Standardising features...")
    mean_ = np.nanmean(X_gaia, axis=0)
    std_  = np.nanstd(X_gaia, axis=0)
    std_safe = np.where(std_ > 0, std_, 1.0)
    X_scaled = (X_gaia - mean_) / std_safe
    X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0.0)

    # ── 3. Fit UMAP ─────────────────────────────────────────────────────────
    print(f"\n[3] Fitting UMAP (n_neighbors={UMAP_PARAMS['n_neighbors']}, "
          f"min_dist={UMAP_PARAMS['min_dist']})...")
    reducer = umap.UMAP(**UMAP_PARAMS)
    embedding = reducer.fit_transform(X_scaled)
    print(f"  Embedding shape: {embedding.shape}")

    # Save reducer + embedding
    joblib.dump(reducer, OUTPUT_DIR / "gaia_random_reducer.joblib")

    # ── 4. HDBSCAN clustering ────────────────────────────────────────────────
    print("\n[4] Running HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=10)
    labels = clusterer.fit_predict(embedding)
    n_clusters = len(set(labels[labels >= 0]))
    print(f"  {n_clusters} clusters found, {(labels == -1).sum():,} noise points")

    for cid in sorted(set(labels[labels >= 0])):
        mask = labels == cid
        print(f"    C{cid}: {mask.sum():,} sources")

    # ── 5. Predict P(gal) with XGBoost ───────────────────────────────────────
    print("\n[5] Predicting P(gal) on Gaia sources...")
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    dmat = xgb.DMatrix(X_gaia, feature_names=feat_names)
    pgal = model.get_booster().predict(dmat)
    print(f"  P(gal) range: [{pgal.min():.4f}, {pgal.max():.4f}]")
    print(f"  P(gal) > 0.5: {(pgal >= 0.5).sum():,} ({100*(pgal >= 0.5).mean():.1f}%)")

    # ── 6. Save embedding ────────────────────────────────────────────────────
    np.savez(OUTPUT_DIR / "gaia_random_embedding.npz",
             embedding=embedding, labels=labels, pgal=pgal,
             g_mag=g_mag, mean=mean_, std=std_)
    print(f"\n  -> gaia_random_embedding.npz")

    # ── 7. Plot: Gaia UMAP (4 panels) ───────────────────────────────────────
    print("\n[6] Plotting Gaia UMAP...")
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
    fig.subplots_adjust(wspace=0.3)

    rng = np.random.RandomState(0)
    idx = rng.permutation(len(embedding))

    # [0] Density
    ax = axes[0]
    n_bins = 180
    x_edges = np.linspace(embedding[:, 0].min() - 1, embedding[:, 0].max() + 1, n_bins + 1)
    y_edges = np.linspace(embedding[:, 1].min() - 1, embedding[:, 1].max() + 1, n_bins + 1)
    H, _, _ = np.histogram2d(embedding[:, 0], embedding[:, 1], bins=[x_edges, y_edges])
    H = np.where(H > 0, H, np.nan)
    ax.pcolormesh(x_edges, y_edges, H.T, cmap="Blues",
                  norm=LogNorm(vmin=1, vmax=H[np.isfinite(H)].max()),
                  rasterized=True)
    ax.set_title(f"Density (n={len(embedding):,})", fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.2)

    # [1] P(gal)
    ax = axes[1]
    sc = ax.scatter(embedding[idx, 0], embedding[idx, 1],
                    c=pgal[idx], cmap="RdYlBu_r", vmin=0, vmax=1,
                    s=2, alpha=0.5, linewidths=0, rasterized=True)
    fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046, label="P(gal)")
    ax.set_title("XGBoost P(gal)", fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.2)

    # [2] Clusters
    ax = axes[2]
    cluster_colors = plt.cm.Set2(np.linspace(0, 1, max(n_clusters, 1)))
    for cid in sorted(set(labels[labels >= 0])):
        mask = labels == cid
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cluster_colors[cid % len(cluster_colors)]],
                   s=2, alpha=0.5, linewidths=0, rasterized=True,
                   label=f"C{cid} ({mask.sum():,})")
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1],
                   c="lightgrey", s=1, alpha=0.2, linewidths=0, rasterized=True,
                   label=f"Noise ({noise_mask.sum():,})")
    ax.legend(fontsize=7, markerscale=3, loc="best")
    ax.set_title(f"HDBSCAN ({n_clusters} clusters)", fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.2)

    # [3] G magnitude
    ax = axes[3]
    sc = ax.scatter(embedding[idx, 0], embedding[idx, 1],
                    c=g_mag[idx], cmap="viridis_r", vmin=17, vmax=22,
                    s=2, alpha=0.5, linewidths=0, rasterized=True)
    fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046, label="G mag")
    ax.set_title("Gaia G magnitude", fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.2)

    fig.suptitle("UMAP of 50k random Gaia sources (no Euclid labels)\n"
                 "Built from 14 Gaia DR3 parameters, G ≥ 17",
                 fontsize=13, fontweight="bold", y=0.98)

    out = OUTPUT_DIR / "gaia_random_umap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")

    # ── 8. Project Q1 sources into Gaia UMAP ────────────────────────────────
    print("\n[7] Projecting Q1 sources into Gaia UMAP...")
    q1 = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    X_q1 = q1["X_raw"]
    y_q1 = q1["y"]
    q1_scaled = (X_q1 - mean_) / std_safe
    q1_scaled = np.where(np.isfinite(q1_scaled), q1_scaled, 0.0)
    q1_proj = reducer.transform(q1_scaled)
    print(f"  Projected {len(q1_proj):,} Q1 sources")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.subplots_adjust(wspace=0.3)

    # Shared limits
    all_x = np.concatenate([embedding[:, 0], q1_proj[:, 0]])
    all_y = np.concatenate([embedding[:, 1], q1_proj[:, 1]])
    xlim = (all_x.min() - 1, all_x.max() + 1)
    ylim = (all_y.min() - 1, all_y.max() + 1)

    # [0] Gaia background + Q1 stars
    ax = axes[0]
    ax.scatter(embedding[:, 0], embedding[:, 1], c="lightgrey", s=1,
               alpha=0.2, linewidths=0, rasterized=True, label=f"Gaia ({len(embedding):,})")
    mask_star = y_q1 == 0
    ax.scatter(q1_proj[mask_star, 0], q1_proj[mask_star, 1],
               c="steelblue", s=3, alpha=0.4, linewidths=0, rasterized=True,
               label=f"Q1 Stars ({mask_star.sum():,})")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title("Q1 Stars projected", fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(alpha=0.2)

    # [1] Gaia background + Q1 galaxies
    ax = axes[1]
    ax.scatter(embedding[:, 0], embedding[:, 1], c="lightgrey", s=1,
               alpha=0.2, linewidths=0, rasterized=True, label=f"Gaia ({len(embedding):,})")
    mask_gal = y_q1 == 1
    ax.scatter(q1_proj[mask_gal, 0], q1_proj[mask_gal, 1],
               c="coral", s=3, alpha=0.4, linewidths=0, rasterized=True,
               label=f"Q1 Galaxies ({mask_gal.sum():,})")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title("Q1 Galaxies projected", fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(alpha=0.2)

    # [2] Q1 coloured by PHZ label
    ax = axes[2]
    ax.scatter(embedding[:, 0], embedding[:, 1], c="lightgrey", s=1,
               alpha=0.2, linewidths=0, rasterized=True)
    ax.scatter(q1_proj[mask_star, 0], q1_proj[mask_star, 1],
               c="steelblue", s=3, alpha=0.3, linewidths=0, rasterized=True,
               label=f"Star ({mask_star.sum():,})")
    ax.scatter(q1_proj[mask_gal, 0], q1_proj[mask_gal, 1],
               c="coral", s=3, alpha=0.3, linewidths=0, rasterized=True,
               label=f"Galaxy ({mask_gal.sum():,})")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title("Q1 Star/Galaxy overlay", fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(alpha=0.2)

    fig.suptitle("Q1 labelled sources projected into Gaia-only UMAP\n"
                 "(UMAP fitted on 50k random Gaia, no Euclid labels used)",
                 fontsize=13, fontweight="bold", y=0.98)

    out = OUTPUT_DIR / "gaia_random_umap_q1_proj.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
