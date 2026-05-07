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
    python src/analyze_umap_gaia_random.py                # full run (fit UMAP + plot)
    python src/analyze_umap_gaia_random.py --seed 123     # different UMAP seed
    python src/analyze_umap_gaia_random.py --plot-only    # replot from saved embedding
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


def _plot_gaia_umap(embedding, pgal, g_mag, labels, output_dir):
    """3-panel plot: density, P(gal), G magnitude."""
    print("\n  Plotting Gaia UMAP...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.3)

    rng = np.random.RandomState(0)
    idx = rng.permutation(len(embedding))

    # [0] Density
    ax = axes[0]
    n_bins = 300
    x_edges = np.linspace(embedding[:, 0].min() - 1, embedding[:, 0].max() + 1, n_bins + 1)
    y_edges = np.linspace(embedding[:, 1].min() - 1, embedding[:, 1].max() + 1, n_bins + 1)
    H, _, _ = np.histogram2d(embedding[:, 0], embedding[:, 1], bins=[x_edges, y_edges])
    H = np.where(H > 0, H, np.nan)
    sc0 = ax.pcolormesh(x_edges, y_edges, H.T, cmap="Blues",
                        norm=LogNorm(vmin=1, vmax=H[np.isfinite(H)].max()),
                        rasterized=True)
    fig.colorbar(sc0, ax=ax, pad=0.02, fraction=0.046, label="Count (log)")
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

    # [2] XGBoost P(gal) — same as panel 1 but with RdYlBu_r like the Q1 UMAP plots
    ax = axes[2]
    # Colour predicted stars vs galaxies
    mask_star = pgal < 0.5
    mask_gal  = pgal >= 0.5
    ax.scatter(embedding[mask_star, 0], embedding[mask_star, 1],
               c="steelblue", s=2, alpha=0.3, linewidths=0, rasterized=True,
               label=f"Pred. Star ({mask_star.sum():,})")
    ax.scatter(embedding[mask_gal, 0], embedding[mask_gal, 1],
               c="coral", s=2, alpha=0.5, linewidths=0, rasterized=True,
               label=f"Pred. Galaxy ({mask_gal.sum():,})")
    ax.set_title("XGBoost star/galaxy classification", fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(alpha=0.2)

    fig.suptitle("UMAP of 50k random Gaia sources (no Euclid labels)\n"
                 "Built from 14 Gaia DR3 parameters, G \u2265 17",
                 fontsize=13, fontweight="bold", y=1.02)

    out = output_dir / "gaia_random_umap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


def _plot_q1_projection(gaia_emb, q1_proj, y_q1, output_dir, phz_gal_prob=None):
    """3-panel: Q1 stars, Q1 galaxies, overlay — projected into Gaia UMAP."""
    print("\n  Plotting Q1 projection...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.3)

    all_x = np.concatenate([gaia_emb[:, 0], q1_proj[:, 0]])
    all_y = np.concatenate([gaia_emb[:, 1], q1_proj[:, 1]])
    xlim = (all_x.min() - 1, all_x.max() + 1)
    ylim = (all_y.min() - 1, all_y.max() + 1)

    mask_star = y_q1 == 0
    mask_gal  = y_q1 == 1

    # [0] Q1 stars
    ax = axes[0]
    ax.scatter(gaia_emb[:, 0], gaia_emb[:, 1], c="lightgrey", s=1,
               alpha=0.2, linewidths=0, rasterized=True, label=f"Gaia ({len(gaia_emb):,})")
    ax.scatter(q1_proj[mask_star, 0], q1_proj[mask_star, 1],
               c="coral", s=3, alpha=0.4, linewidths=0, rasterized=True,
               label=f"PHZ Stars ({mask_star.sum():,})")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title("Euclid PHZ Stars projected", fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(alpha=0.2)

    # [1] Q1 galaxies
    ax = axes[1]
    ax.scatter(gaia_emb[:, 0], gaia_emb[:, 1], c="lightgrey", s=1,
               alpha=0.2, linewidths=0, rasterized=True, label=f"Gaia ({len(gaia_emb):,})")
    ax.scatter(q1_proj[mask_gal, 0], q1_proj[mask_gal, 1],
               c="steelblue", s=3, alpha=0.4, linewidths=0, rasterized=True,
               label=f"PHZ Galaxies ({mask_gal.sum():,})")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title("Euclid PHZ Galaxies projected", fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(alpha=0.2)

    # [2] PHZ P(galaxy) colouring
    ax = axes[2]
    ax.scatter(gaia_emb[:, 0], gaia_emb[:, 1], c="lightgrey", s=1,
               alpha=0.2, linewidths=0, rasterized=True)
    if phz_gal_prob is not None:
        rng = np.random.RandomState(0)
        idx = rng.permutation(len(q1_proj))
        sc = ax.scatter(q1_proj[idx, 0], q1_proj[idx, 1],
                   c=phz_gal_prob[idx], cmap="RdYlBu_r", vmin=0, vmax=1,
                   s=3, alpha=0.5, linewidths=0, rasterized=True)
        fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046, label="PHZ P(galaxy)")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title("Euclid PHZ P(galaxy)", fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.2)

    fig.suptitle("Euclid PHZ-labelled sources projected into Gaia-only UMAP\n"
                 "(UMAP fitted on 50k random Gaia, no Euclid labels used in fitting)",
                 fontsize=13, fontweight="bold", y=1.02)

    out = output_dir / "gaia_random_umap_q1_proj.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Replot from saved embedding (no UMAP refit)")
    parser.add_argument("--seed", type=int, default=42,
                        help="UMAP random seed (default: 42)")
    args = parser.parse_args()

    global UMAP_PARAMS
    UMAP_PARAMS = dict(n_neighbors=15, min_dist=0.1, n_components=2,
                       random_state=args.seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feat_names = list(SELECTED_GAIA_PARAMETERS)

    if args.plot_only:
        print("[plot-only] Loading saved embedding...")
        saved = np.load(OUTPUT_DIR / "gaia_random_embedding.npz", allow_pickle=True)
        _plot_gaia_umap(saved["embedding"], saved["pgal"], saved["g_mag"],
                        saved["labels"], OUTPUT_DIR)

        # Also replot Q1 projection
        reducer = joblib.load(OUTPUT_DIR / "gaia_random_reducer.joblib")
        q1 = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
        mean_ = saved["mean"]; std_ = saved["std"]
        std_safe = np.where(std_ > 0, std_, 1.0)
        q1_scaled = (q1["X_raw"] - mean_) / std_safe
        q1_scaled = np.where(np.isfinite(q1_scaled), q1_scaled, 0.0)
        q1_proj = reducer.transform(q1_scaled)
        _plot_q1_projection(saved["embedding"], q1_proj, q1["y"], OUTPUT_DIR,
                            phz_gal_prob=q1["phz_gal_prob"])
        return 0

    import umap
    import hdbscan

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
    pgal = model.predict_proba(X_gaia)[:, 1]
    print(f"  P(gal) range: [{pgal.min():.4f}, {pgal.max():.4f}]")
    print(f"  P(gal) > 0.5: {(pgal >= 0.5).sum():,} ({100*(pgal >= 0.5).mean():.1f}%)")

    # ── 6. Save embedding ────────────────────────────────────────────────────
    np.savez(OUTPUT_DIR / "gaia_random_embedding.npz",
             embedding=embedding, labels=labels, pgal=pgal,
             g_mag=g_mag, mean=mean_, std=std_)
    print(f"\n  -> gaia_random_embedding.npz")

    _plot_gaia_umap(embedding, pgal, g_mag, labels, OUTPUT_DIR)

    # ── 8. Project Q1 sources into Gaia UMAP ────────────────────────────────
    print("\n[7] Projecting Q1 sources into Gaia UMAP...")
    q1 = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    q1_scaled = (q1["X_raw"] - mean_) / std_safe
    q1_scaled = np.where(np.isfinite(q1_scaled), q1_scaled, 0.0)
    q1_proj = reducer.transform(q1_scaled)
    print(f"  Projected {len(q1_proj):,} Q1 sources")

    _plot_q1_projection(embedding, q1_proj, q1["y"], OUTPUT_DIR,
                        phz_gal_prob=q1["phz_gal_prob"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
