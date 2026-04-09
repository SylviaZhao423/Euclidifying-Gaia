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
  python src/compute_q1_clusters.py
  python src/compute_q1_clusters.py --min-cluster-size 100
  python src/compute_q1_clusters.py --no-split
  python src/compute_q1_clusters.py --split 3:3.0 --split 1:1.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"

DEFAULT_SPLITS = ["3:3.0"]   # applied unless --no-split is given


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cluster Q1 UMAP embedding and save labels.")
    parser.add_argument("--n-clusters",       type=int, default=5,
                        help="Expected clusters for KMeans fallback (default: 5)")
    parser.add_argument("--min-cluster-size", type=int, default=150,
                        help="HDBSCAN min_cluster_size (default: 150)")
    parser.add_argument("--no-split", action="store_true",
                        help=f"Disable the default split(s): {DEFAULT_SPLITS}")
    parser.add_argument("--split", metavar="ID:Y", action="append", default=[],
                        help="Extra split: cluster ID at UMAP2 threshold Y "
                             "(lower part → new id). Repeatable.")
    args = parser.parse_args()

    print("=" * 60)
    print("[1] Loading Q1 UMAP embedding...")
    npz = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    q1_emb = npz["embedding"]
    print(f"  {q1_emb.shape[0]:,} points, {q1_emb.shape[1]}D")

    print("\n[2] Clustering...")
    labels = cluster_q1_umap(q1_emb,
                              n_expected=args.n_clusters,
                              min_cluster_size=args.min_cluster_size)

    raw_splits = [] if args.no_split else DEFAULT_SPLITS + args.split
    splits: list[tuple[int, float]] = []
    for s in raw_splits:
        try:
            cid, y = s.split(":")
            splits.append((int(cid), float(y)))
        except ValueError:
            print(f"ERROR: --split must be ID:Y (e.g. 3:3.0), got: {s!r}")
            return 1

    if splits:
        print(f"\n[3] Applying {len(splits)} split(s)...")
        labels = apply_splits(labels, q1_emb, splits)

    cluster_ids = sorted(set(labels[labels >= 0]))
    print("\n  Final cluster sizes:")
    for k in cluster_ids:
        print(f"    C{k}: {(labels == k).sum():,}")
    print(f"    noise: {(labels == -1).sum():,}")

    out = ARRAYS_DIR / "q1_cluster_labels.npy"
    np.save(out, labels)
    print(f"\n  Saved → {out}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
