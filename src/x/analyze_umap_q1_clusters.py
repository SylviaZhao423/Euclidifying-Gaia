#!/usr/bin/env python3
"""
Cluster the Q1 UMAP embedding and save labels to disk.

This is the single source of truth for Q1 cluster assignments.
All downstream scripts (analyze_umap_q1_qso_3d.py,
download_q_q1_qso_cutouts.py, download_q_q1_rej_cutouts.py,
analyze_umap_q1_rej_3d.py, etc.) simply load the saved labels
instead of re-clustering independently.

Output
------
output/umap/arrays/q1_cluster_labels.npy
    Integer array of length N (one label per Q1 source).
    -1 = noise/unassigned.  0,1,2,… = cluster indices.

Usage
-----
  python src/analyze_umap_q1_clusters.py
  python src/analyze_umap_q1_clusters.py --min-cluster-size 100
  python src/analyze_umap_q1_clusters.py --no-split
  python src/analyze_umap_q1_clusters.py --split 3:3.0 --split 1:1.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "q1_individual"

# Post-processing: merge small clusters into a target, then split large ones.
# Applied after HDBSCAN unless --no-postprocess is given.
# Merge format: "SRC>DST" — relabel cluster SRC as DST
# Split format: "ID:Y"   — split cluster ID at UMAP2 threshold Y (lower → new id)
DEFAULT_MERGES = ["4>3"]          # merge C4 into C3 (central bits)
DEFAULT_SPLITS = ["2:5.0"]        # split big C2 at UMAP2=5.0 (waist between lobes)
DEFAULT_SWAPS  = ["2:3"]          # swap C2↔C3 so central cluster becomes C2

# Cluster colours — synced with compile_umap_3d.py
CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#F298B6", "mediumpurple",
    "#17becf", "#d62728", "#8c564b", "#7f7f7f", "#bcbd22",
]
NOISE_COLOR = "#cccccc"
CLASS_COLORS = {"STAR": "#E53935", "GALAXY": "#1976D2"}


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_q1_umap(embedding: np.ndarray,
                    n_expected: int = 5,
                    min_cluster_size: int = 150) -> np.ndarray:
    """HDBSCAN clustering; falls back to KMeans if hdbscan is unavailable."""
    try:
        import hdbscan
        print(f"  Clustering with HDBSCAN (min_cluster_size={min_cluster_size})...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=max(10, min_cluster_size // 10),
            cluster_selection_epsilon=0.3,
        )
        labels = clusterer.fit_predict(embedding)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"  -> {n_clusters} clusters found  ({(labels == -1).sum():,} noise points)")
    except ImportError:
        print(f"  hdbscan not available — falling back to KMeans(k={n_expected})...")
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=n_expected, random_state=42,
                        n_init=10).fit_predict(embedding)
        print(f"  -> {n_expected} clusters (KMeans)")
    return labels


def apply_merges(labels: np.ndarray,
                 merges: list[tuple[int, int]]) -> np.ndarray:
    """Merge clusters: relabel src → dst."""
    labels = labels.copy()
    for src, dst in merges:
        n = (labels == src).sum()
        if n == 0:
            print(f"  WARNING: --merge {src}>{dst} — C{src} has no points")
            continue
        labels[labels == src] = dst
        print(f"  Merged C{src} ({n:,} pts) → C{dst}")
    return labels


def renumber(labels: np.ndarray) -> np.ndarray:
    """Renumber clusters to 0,1,2,… (preserving -1 for noise)."""
    labels = labels.copy()
    old_ids = sorted(set(labels[labels >= 0]))
    mapping = {old: new for new, old in enumerate(old_ids)}
    for old, new in mapping.items():
        if old != new:
            labels[labels == old] = -(new + 100)  # temp negative
    for old, new in mapping.items():
        if old != new:
            labels[labels == -(new + 100)] = new
    return labels


def apply_splits(labels: np.ndarray,
                 embedding: np.ndarray,
                 splits: list[tuple[int, float]]) -> np.ndarray:
    """
    Split a cluster at a UMAP-2 threshold.

    Each entry is (cluster_id, umap2_threshold): points in that cluster
    with UMAP2 < threshold are relabelled as a new cluster id
    (max existing id + 1, incrementing for each split).
    """
    labels = labels.copy()
    next_id = int(labels.max()) + 1
    for cluster_id, y_thresh in splits:
        mask = (labels == cluster_id) & (embedding[:, 1] < y_thresh)
        if not mask.any():
            print(f"  WARNING: --split {cluster_id}:{y_thresh} matched no points")
            continue
        labels[mask] = next_id
        n_upper = (labels == cluster_id).sum()
        n_lower = mask.sum()
        print(f"  Split C{cluster_id} at UMAP2 < {y_thresh}: "
              f"{n_upper:,} upper (C{cluster_id}) + {n_lower:,} lower (C{next_id})")
        next_id += 1
    return labels


def apply_swaps(labels: np.ndarray,
                swaps: list[tuple[int, int]]) -> np.ndarray:
    """Swap cluster label pairs: A↔B."""
    labels = labels.copy()
    for a, b in swaps:
        mask_a = labels == a
        mask_b = labels == b
        if not mask_a.any() and not mask_b.any():
            print(f"  WARNING: --swap {a}:{b} — neither cluster exists")
            continue
        labels[mask_a] = b
        labels[mask_b] = a
        print(f"  Swapped C{a} ({mask_a.sum():,} pts) ↔ C{b} ({mask_b.sum():,} pts)")
    return labels


# ── Plot ──────────────────────────────────────────────────────────────────

def plot_clusters(emb: np.ndarray, labels: np.ndarray, y: np.ndarray,
                  out_dir: Path) -> None:
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
    fig1, ax1 = plt.subplots(figsize=(9, 7))
    noise_mask = labels == -1
    if noise_mask.any():
        ax1.scatter(emb[noise_mask, 0], emb[noise_mask, 1],
                    c=NOISE_COLOR, s=1, alpha=0.4, linewidths=0, rasterized=True)
    handles1 = []
    for k in cluster_ids:
        mask = labels[order] == k
        color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        n_k = (labels == k).sum()
        n_gal = (y[labels == k] == 1).sum()
        pct_gal = 100 * n_gal / n_k if n_k else 0
        ax1.scatter(emb[order[mask], 0], emb[order[mask], 1],
                    c=color, s=1, alpha=0.4, linewidths=0, rasterized=True)
        handles1.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                   markersize=7,
                   label=f"C{k}: {n_k:,}  ({pct_gal:.0f}% gal)"))
        cx = emb[labels == k, 0].mean()
        cy = emb[labels == k, 1].mean()
        ax1.annotate(f"C{k}", (cx, cy), fontsize=12, fontweight="bold",
                     ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color,
                               alpha=0.85, lw=1.5))
    if noise_mask.any():
        handles1.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=NOISE_COLOR,
                   markersize=7, label=f"noise: {noise_mask.sum():,}"))
    ax1.legend(handles=handles1, fontsize=12, loc="lower left", framealpha=0.8)
    ax1.set_xlabel("UMAP 1", fontsize=10)
    ax1.set_ylabel("UMAP 2", fontsize=10)
    ax1.set_title(f"Q1 UMAP clusters  (HDBSCAN, {len(cluster_ids)} clusters)",
                  fontsize=12, fontweight="bold")
    ax1.set_xlim(*xlim); ax1.set_ylim(*ylim)
    ax1.grid(alpha=0.2)
    fig1.tight_layout()
    p1 = out_dir / "q1_cluster_labels.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved → {p1}")

    # Panel 2 standalone: PHZ ground truth scatter + stacked 2D histograms
    from matplotlib.gridspec import GridSpec as GS
    from matplotlib.colors import LogNorm

    fig2 = plt.figure(figsize=(19, 8))
    gs2 = GS(2, 2, figure=fig2, height_ratios=[1, 1],
             hspace=0.25, wspace=0.2)

    # Left: scatter spanning both rows
    ax_sc = fig2.add_subplot(gs2[0:2, 0])
    for cls_val, cls_name in [(0, "STAR"), (1, "GALAXY")]:
        mask = y[order] == cls_val
        ax_sc.scatter(emb[order[mask], 0], emb[order[mask], 1],
                      c=CLASS_COLORS[cls_name], s=1, alpha=0.5,
                      linewidths=0, rasterized=True, label=cls_name)
    ax_sc.legend(fontsize=14, loc="lower left", framealpha=0.8, markerscale=4)
    ax_sc.set_xlabel("UMAP 1", fontsize=10)
    ax_sc.set_ylabel("UMAP 2", fontsize=10)
    ax_sc.set_title("Q1 PHZ ground truth  (Star / Galaxy)",
                    fontsize=12, fontweight="bold")
    ax_sc.set_xlim(*xlim); ax_sc.set_ylim(*ylim)
    ax_sc.grid(alpha=0.2)

    # Right column: stacked 2D histograms (top=STAR, bottom=GALAXY)
    n_bins_hist = 250
    x_edges_h = np.linspace(xlim[0], xlim[1], n_bins_hist + 1)
    y_edges_h = np.linspace(ylim[0], ylim[1], n_bins_hist + 1)

    cls_cmaps = {"STAR": "OrRd", "GALAXY": "Blues"}
    for row, cls_val, cls_name in [(0, 0, "STAR"), (1, 1, "GALAXY")]:
        ax_h = fig2.add_subplot(gs2[row, 1])
        cls_mask = y == cls_val
        H, _, _ = np.histogram2d(emb[cls_mask, 0], emb[cls_mask, 1],
                                  bins=[x_edges_h, y_edges_h])
        H = H.T
        H_plot = np.where(H > 0, H, np.nan)
        vmin = np.nanmin(H_plot)
        vmax = np.nanmax(H_plot)
        im = ax_h.pcolormesh(x_edges_h, y_edges_h, H_plot,
                              cmap=cls_cmaps[cls_name], norm=LogNorm(vmin=vmin, vmax=vmax),
                              rasterized=True)
        fig2.colorbar(im, ax=ax_h, shrink=0.85, label="Count (log)")
        ax_h.set_xlabel("UMAP 1", fontsize=10)
        ax_h.set_ylabel("UMAP 2", fontsize=10)
        n_cls = cls_mask.sum()
        ax_h.set_title(f"{cls_name} density  (N={n_cls:,})",
                       fontsize=12, fontweight="bold")
        ax_h.set_xlim(*xlim); ax_h.set_ylim(*ylim)

    p2 = out_dir / "q1_cluster_phz.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved → {p2}")


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
    parser = argparse.ArgumentParser(
        description="Cluster Q1 UMAP embedding and save labels.")
    parser.add_argument("--n-clusters",       type=int, default=5,
                        help="Expected clusters for KMeans fallback (default: 5)")
    parser.add_argument("--min-cluster-size", type=int, default=150,
                        help="HDBSCAN min_cluster_size (default: 150)")
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Disable default merges and splits")
    parser.add_argument("--merge", metavar="SRC>DST", action="append", default=[],
                        help="Extra merge: relabel cluster SRC as DST. Repeatable.")
    parser.add_argument("--split", metavar="ID:Y", action="append", default=[],
                        help="Extra split: cluster ID at UMAP2 threshold Y "
                             "(lower part → new id). Repeatable.")
    parser.add_argument("--swap", metavar="A:B", action="append", default=[],
                        help="Swap cluster labels A↔B. Repeatable.")
    args = parser.parse_args()

    print("=" * 60)
    print("[1] Loading Q1 UMAP embedding...")
    npz = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    q1_emb = npz["embedding"]
    y = npz["y"]  # 0=STAR, 1=GALAXY
    print(f"  {q1_emb.shape[0]:,} points, {q1_emb.shape[1]}D")

    print("\n[2] Clustering...")
    labels = cluster_q1_umap(q1_emb,
                              n_expected=args.n_clusters,
                              min_cluster_size=args.min_cluster_size)
    print("\n  Raw HDBSCAN clusters:")
    _print_composition(labels, y, q1_emb)

    # ── Post-processing: merge then split ────────────────────────────────
    if args.no_postprocess:
        raw_merges, raw_splits, raw_swaps = args.merge, args.split, args.swap
    else:
        raw_merges = DEFAULT_MERGES + args.merge
        raw_splits = DEFAULT_SPLITS + args.split
        raw_swaps  = DEFAULT_SWAPS  + args.swap

    # Parse merges
    merges: list[tuple[int, int]] = []
    for m in raw_merges:
        try:
            src, dst = m.split(">")
            merges.append((int(src), int(dst)))
        except ValueError:
            print(f"ERROR: --merge must be SRC>DST (e.g. 3>2), got: {m!r}")
            return 1

    if merges:
        print(f"\n[3a] Merging {len(merges)} cluster(s)...")
        labels = apply_merges(labels, merges)

    # Parse splits
    splits: list[tuple[int, float]] = []
    for s in raw_splits:
        try:
            cid, yval = s.split(":")
            splits.append((int(cid), float(yval)))
        except ValueError:
            print(f"ERROR: --split must be ID:Y (e.g. 2:5.0), got: {s!r}")
            return 1

    if splits:
        print(f"\n[3b] Splitting {len(splits)} cluster(s)...")
        labels = apply_splits(labels, q1_emb, splits)

    # Renumber to consecutive 0,1,2,...
    labels = renumber(labels)

    # Parse swaps
    swaps: list[tuple[int, int]] = []
    for s in raw_swaps:
        try:
            a, b = s.split(":")
            swaps.append((int(a), int(b)))
        except ValueError:
            print(f"ERROR: --swap must be A:B (e.g. 2:3), got: {s!r}")
            return 1

    if swaps:
        print(f"\n[3c] Swapping {len(swaps)} cluster pair(s)...")
        labels = apply_swaps(labels, swaps)

    print("\n  Final cluster composition:")
    _print_composition(labels, y, q1_emb)

    out = ARRAYS_DIR / "q1_cluster_labels.npy"
    np.save(out, labels)
    print(f"\n  Saved → {out}")

    print("\n[4] Plotting...")
    insert_dir = OUTPUT_DIR / "insert"
    insert_dir.mkdir(parents=True, exist_ok=True)
    plot_clusters(q1_emb, labels, y, insert_dir)

    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
