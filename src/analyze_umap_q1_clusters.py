#!/usr/bin/env python3
"""
Plot Q1 UMAP clusters from saved labels.

Loads the pre-computed cluster labels (q1_cluster_labels.npy, produced by
query_umap_q1_50k.py) and the Q1 embedding (q1_indiv.npz), then generates
diagnostic plots: cluster labels + PHZ P(galaxy) combined figure.

Output
------
output/umap/q1_individual/insert/q1_cluster_labels.png
output/umap/q1_individual/insert/q1_cluster_phz_combined.png

Usage
-----
  python src/analyze_umap_q1_clusters.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "q1_individual"

# Cluster colours — synced with compile_umap_3d.py
CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#F298B6", "mediumpurple",
    "#17becf", "#d62728", "#8c564b", "#7f7f7f", "#bcbd22",
]
NOISE_COLOR = "#cccccc"


# ── Plot ──────────────────────────────────────────────────────────────────

def plot_clusters(emb: np.ndarray, labels: np.ndarray, y: np.ndarray,
                  out_dir: Path, phz_gal_prob: np.ndarray | None = None) -> None:
    """Save cluster labels plot and PHZ ground truth + density plot."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cluster_ids = sorted(set(labels[labels >= 0]))
    rng = np.random.RandomState(0)
    order = rng.permutation(len(emb))

    x_pad = 1
    xlim = (emb[:, 0].min() - x_pad, emb[:, 0].max() + x_pad)
    ylim = (emb[:, 1].min() - x_pad, emb[:, 1].max() + x_pad)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Cluster labels plot ──────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    noise_mask = labels == -1
    if noise_mask.any():
        ax1.scatter(emb[noise_mask, 0], emb[noise_mask, 1],
                    c=NOISE_COLOR, s=1, alpha=0.6, linewidths=0, rasterized=True)
    handles1 = []
    for k in cluster_ids:
        mask = labels[order] == k
        color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        n_k = (labels == k).sum()
        n_gal = (y[labels == k] == 1).sum()
        pct_gal = 100 * n_gal / n_k if n_k else 0
        ax1.scatter(emb[order[mask], 0], emb[order[mask], 1],
                    c=color, s=1, alpha=0.6, linewidths=0, rasterized=True)
        handles1.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                   markersize=7,
                   label=f"C{k}: {n_k:,}  ({pct_gal:.0f}% gal)"))
        cx = emb[labels == k, 0].mean()
        cy = emb[labels == k, 1].mean()
        ax1.annotate(f"C{k}", (cx, cy), fontsize=15, fontweight="bold",
                     ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color,
                               alpha=0.85, lw=1.5))
    if noise_mask.any():
        handles1.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=NOISE_COLOR,
                   markersize=7, label=f"noise: {noise_mask.sum():,}"))
    ax1.legend(handles=handles1, fontsize=12, loc="lower left", framealpha=0.8)
    ax1.set_xlabel("UMAP 1", fontsize=14, fontweight="bold")
    ax1.set_ylabel("UMAP 2", fontsize=14, fontweight="bold")
    # ax1.set_title(f"Q1 UMAP clusters  (HDBSCAN, {len(cluster_ids)} clusters)",
    #               fontsize=14, fontweight="bold")
    ax1.tick_params(labelsize=12)
    ax1.set_xlim(*xlim); ax1.set_ylim(*ylim)
    ax1.grid(alpha=0.2)
    fig1.tight_layout()
    p1 = out_dir / "q1_cluster_labels.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved → {p1}")

    # ── Combined: cluster labels + PHZ P(galaxy) ────────────────────────
    from matplotlib.lines import Line2D as L2D

    fig_c, (ax_cl, ax_pg) = plt.subplots(1, 2, figsize=(12, 5))
    fig_c.subplots_adjust(wspace=0.26)

    # Left panel: cluster labels (same as fig1)
    noise_mask_c = labels == -1
    if noise_mask_c.any():
        ax_cl.scatter(emb[noise_mask_c, 0], emb[noise_mask_c, 1],
                      c=NOISE_COLOR, s=0.6, alpha=0.6, linewidths=0, rasterized=True)
    handles_c = []
    for k in cluster_ids:
        mask = labels[order] == k
        color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        n_k = (labels == k).sum()
        n_gal = (y[labels == k] == 1).sum()
        pct_gal = 100 * n_gal / n_k if n_k else 0
        ax_cl.scatter(emb[order[mask], 0], emb[order[mask], 1],
                      c=color, s=0.6, alpha=0.6, linewidths=0, rasterized=True)
        handles_c.append(
            L2D([0], [0], marker='o', color='w', markerfacecolor=color,
                markersize=7,
                label=f"C{k}: {n_k:,}  ({pct_gal:.0f}% gal)"))
        cx = emb[labels == k, 0].mean()
        cy = emb[labels == k, 1].mean()
        ax_cl.annotate(f"C{k}", (cx, cy), fontsize=13, fontweight="bold",
                       ha="center", va="center",
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color,
                                 alpha=0.85, lw=1.5))
    if noise_mask_c.any():
        handles_c.append(
            L2D([0], [0], marker='o', color='w', markerfacecolor=NOISE_COLOR,
                markersize=7, label=f"noise: {noise_mask_c.sum():,}"))
    ax_cl.legend(handles=handles_c, fontsize=10, loc="lower left", framealpha=0.8)
    ax_cl.set_xlabel("UMAP 1", fontsize=13, fontweight="bold")
    ax_cl.set_ylabel("UMAP 2", fontsize=13, fontweight="bold")
    ax_cl.set_title("HDBSCAN clusters", fontsize=13, fontweight="bold")
    ax_cl.tick_params(labelsize=12)
    ax_cl.set_xlim(*xlim); ax_cl.set_ylim(*ylim)
    ax_cl.grid(alpha=0.2)

    # Right panel: PHZ P(galaxy) scatter
    pgal = phz_gal_prob if phz_gal_prob is not None else y.astype(float)
    rng2 = np.random.RandomState(42)
    idx_pg = rng2.permutation(len(emb))
    sc_pg = ax_pg.scatter(emb[idx_pg, 0], emb[idx_pg, 1],
                          c=pgal[idx_pg], cmap="RdYlBu_r",
                          vmin=0, vmax=1, s=0.6, alpha=0.6,
                          linewidths=0, rasterized=True)
    cb_pg = fig_c.colorbar(sc_pg, ax=ax_pg, pad=0.06, fraction=0.046)
    cb_pg.set_label("PHZ P(galaxy)", fontsize=12, fontweight="bold")
    cb_pg.ax.tick_params(labelsize=11)
    ax_pg.set_xlabel("UMAP 1", fontsize=13, fontweight="bold")
    ax_pg.set_ylabel("UMAP 2", fontsize=13, fontweight="bold")
    ax_pg.set_title("PHZ P(galaxy)", fontsize=13, fontweight="bold")
    ax_pg.tick_params(labelsize=12)
    ax_pg.set_xlim(*xlim); ax_pg.set_ylim(*ylim)
    ax_pg.grid(alpha=0.2)

    p_c = out_dir / "q1_cluster_phz_combined.png"
    fig_c.savefig(p_c, dpi=150, bbox_inches="tight")
    plt.close(fig_c)
    print(f"  Saved → {p_c}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _print_composition(labels, y, emb):
    cluster_ids = sorted(set(labels[labels >= 0]))
    print(f"  {'Cluster':>8}  {'N':>7}  {'Stars':>7}  {'Galaxies':>8}  {'Gal%':>6}  {'Centroid'}")
    print(f"  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*6}  {'─'*16}")
    for k in [-1] + cluster_ids:
        mask = labels == k
        n = mask.sum()
        if n == 0:
            continue
        n_gal = (y[mask] == 1).sum()
        n_star = (y[mask] == 0).sum()
        pct = 100 * n_gal / n
        cx, cy = emb[mask, 0].mean(), emb[mask, 1].mean()
        tag = "noise" if k == -1 else f"C{k}"
        print(f"  {tag:>8}  {n:>7,}  {n_star:>7,}  {n_gal:>8,}  {pct:>5.1f}%  ({cx:.1f}, {cy:.1f})")


def main() -> int:
    print("=" * 60)
    print("[1] Loading Q1 UMAP embedding + cluster labels...")

    npz_path = ARRAYS_DIR / "q1_indiv.npz"
    labels_path = ARRAYS_DIR / "q1_cluster_labels.npy"

    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found. Run query_umap_q1_50k.py first.")
        return 1
    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found. Run query_umap_q1_50k.py first.")
        return 1

    npz = np.load(npz_path, allow_pickle=True)
    q1_emb = npz["embedding"]
    y = npz["y"]  # 0=STAR, 1=GALAXY
    phz_gal_prob = npz["phz_gal_prob"] if "phz_gal_prob" in npz else None
    labels = np.load(labels_path)
    print(f"  {q1_emb.shape[0]:,} points, {q1_emb.shape[1]}D")

    print("\n  Cluster composition:")
    _print_composition(labels, y, q1_emb)

    print("\n[2] Plotting...")
    insert_dir = OUTPUT_DIR / "insert"
    insert_dir.mkdir(parents=True, exist_ok=True)
    plot_clusters(q1_emb, labels, y, insert_dir, phz_gal_prob=phz_gal_prob)

    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
