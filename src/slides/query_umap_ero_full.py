#!/usr/bin/env python3
"""
UMAP + HDBSCAN on all ERO sources (no subsampling).

Mirrors query_umap_q1_full.py but for the ERO pool. Queries
siwei_zhao_2025.crossmatch_ero_gaia_0_3 with:
  - e_v_class_star IS NOT NULL
  - g_phot_g_mean_mag >= 17

ERO has no PHZ classification, so the binary label is derived from
SExtractor CLASS_STAR with the same threshold used elsewhere in this
project (>= 0.65 → STAR, < 0.65 → GALAXY).

Outputs (under output/umap/ero_full/):
  ero_full.npz                 — embedding, labels, features, metadata
  ero_full_reducer.joblib      — fitted UMAP reducer
  ero_full_clusters.png        — cluster + CLASS_STAR ground truth plot

Usage:
    PGUSER=siwei_zhao_2025 python src/slides/query_umap_ero_full.py
    PGUSER=siwei_zhao_2025 python src/slides/query_umap_ero_full.py --test
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "ero_full"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

CLASS_MAPPING = {"STAR": 0, "GALAXY": 1}
MAG_CUT = 17
CLASS_STAR_THRESH = 0.65

# Q1 training-set star:galaxy ratio (from filter Sankey: 176,647 / 62,221).
# Used by --rebalance to bring the ERO pool (~18.45:1) down to Q1's ratio.
Q1_STAR_GAL_RATIO = 176_647 / 62_221  # ≈ 2.84

# Post-processing defaults — empty until raw ERO clusters are inspected.
# Tune via CLI (--no-postprocess to skip; supply --merges/--splits/--swaps
# if you wire those up).
DEFAULT_MERGES = []
DEFAULT_SPLITS = []
DEFAULT_SWAPS  = []

CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#F298B6", "mediumpurple",
    "#17becf", "#d62728", "#8c564b", "#7f7f7f", "#bcbd22",
]
NOISE_COLOR = "#cccccc"
CLASS_COLORS = {"STAR": "#E53935", "GALAXY": "#1976D2"}


# ── Query ────────────────────────────────────────────────────────────────────

def query_ero_full(username, test_mode=False):
    """Query ERO sources with class_star NOT NULL and G>=17, no subsample."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)

    gaia_cols = ", ".join([f"g_{c} AS {c}" for c in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT
            g_source_id,
            {gaia_cols},
            g_phot_g_mean_mag,
            e_v_alpha_j2000 AS ra,
            e_v_delta_j2000 AS dec,
            e_v_class_star AS class_star,
            CASE
                WHEN e_v_class_star >= {CLASS_STAR_THRESH} THEN 'STAR'
                WHEN e_v_class_star <  {CLASS_STAR_THRESH} THEN 'GALAXY'
            END AS label
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_class_star IS NOT NULL
          AND g_phot_g_mean_mag >= {MAG_CUT}
        ORDER BY g_source_id
    """
    if test_mode:
        query += " LIMIT 10000"

    print("  Querying ERO (class_star IS NOT NULL, G>=17, no subsample)...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    df[list(SELECTED_GAIA_PARAMETERS)] = df[list(SELECTED_GAIA_PARAMETERS)].replace(
        "NOT_AVAILABLE", np.nan
    )
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_star = (df["label"] == "STAR").sum()
    n_gal = (df["label"] == "GALAXY").sum()
    print(f"  -> {len(df):,} sources  STAR: {n_star:,}  GALAXY: {n_gal:,}")
    return df


# ── Rebalance ────────────────────────────────────────────────────────────────

def rebalance_indices_to_q1_ratio(y: np.ndarray,
                                  star_gal_ratio: float,
                                  seed: int = 42) -> np.ndarray:
    """Stratified resample of indices to match a target star:galaxy ratio.

    Keeps ALL galaxies and undersamples stars to (n_gal * ratio). For the full
    pipeline there is no total cap — only the ratio constraint matters.
    """
    rng = np.random.RandomState(seed)
    idx_star = np.where(y == 0)[0]
    idx_gal  = np.where(y == 1)[0]
    n_gal  = len(idx_gal)
    n_star = min(int(round(n_gal * star_gal_ratio)), len(idx_star))
    sel_star = rng.choice(idx_star, n_star, replace=False)
    sel = np.concatenate([sel_star, idx_gal])
    rng.shuffle(sel)
    return sel


# ── UMAP ─────────────────────────────────────────────────────────────────────

def fit_umap(X):
    """Fill NaN with median, standardize, fit UMAP."""
    X = X.copy()
    for col in range(X.shape[1]):
        bad = np.isnan(X[:, col])
        if bad.any():
            X[bad, col] = np.nanmedian(X[:, col])
    mean_ = np.nanmean(X, axis=0)
    std_ = np.nanstd(X, axis=0)
    std_safe = np.where(std_ > 0, std_, 1.0)
    X_scaled = (X - mean_) / std_safe

    print(f"  Fitting UMAP on {X_scaled.shape[0]:,} × {X_scaled.shape[1]} ...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        metric="euclidean", random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    print(f"  Done.")
    return embedding, reducer, mean_, std_


# ── Clustering ───────────────────────────────────────────────────────────────

def cluster_hdbscan(embedding, min_cluster_size=150):
    import hdbscan
    print(f"  HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(10, min_cluster_size // 10),
        cluster_selection_epsilon=0.3,
    )
    labels = clusterer.fit_predict(embedding)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  -> {n_clusters} clusters, {(labels == -1).sum():,} noise")
    return labels


def postprocess(labels, embedding, merges, splits, swaps):
    """Merge, split, renumber, swap — same logic as Q1 full pipeline."""
    for m in merges:
        src, dst = (int(x) for x in m.split(">"))
        n = (labels == src).sum()
        if n:
            labels[labels == src] = dst
            print(f"  Merged C{src} ({n:,}) -> C{dst}")

    next_id = int(labels.max()) + 1
    for s in splits:
        cid, yval = s.split(":")
        cid, yval = int(cid), float(yval)
        mask = (labels == cid) & (embedding[:, 1] < yval)
        if mask.any():
            labels[mask] = next_id
            print(f"  Split C{cid} at UMAP2<{yval}: "
                  f"{(labels == cid).sum():,} upper + {mask.sum():,} lower (C{next_id})")
            next_id += 1

    old_ids = sorted(set(labels[labels >= 0]))
    mapping = {old: new for new, old in enumerate(old_ids)}
    for old, new in mapping.items():
        if old != new:
            labels[labels == old] = -(new + 100)
    for old, new in mapping.items():
        if old != new:
            labels[labels == -(new + 100)] = new

    for s in swaps:
        a, b = (int(x) for x in s.split(":"))
        mask_a, mask_b = labels == a, labels == b
        labels[mask_a] = b
        labels[mask_b] = a
        print(f"  Swapped C{a} ({mask_a.sum():,}) <-> C{b} ({mask_b.sum():,})")

    return labels


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_clusters(emb, labels, y, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cluster_ids = sorted(set(labels[labels >= 0]))
    rng = np.random.RandomState(0)
    order = rng.permutation(len(emb))

    pad = 1
    xlim = (emb[:, 0].min() - pad, emb[:, 0].max() + pad)
    ylim = (emb[:, 1].min() - pad, emb[:, 1].max() + pad)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.subplots_adjust(wspace=0.25, top=0.90)

    # Panel 0: clusters
    ax = axes[0]
    noise = labels == -1
    if noise.any():
        ax.scatter(emb[noise, 0], emb[noise, 1],
                   c=NOISE_COLOR, s=1, alpha=0.2, linewidths=0, rasterized=True)

    handles = []
    for k in cluster_ids:
        mask = labels[order] == k
        color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        n_k = (labels == k).sum()
        n_gal = (y[labels == k] == 1).sum()
        pct = 100 * n_gal / n_k if n_k else 0
        ax.scatter(emb[order[mask], 0], emb[order[mask], 1],
                   c=color, s=1, alpha=0.2, linewidths=0, rasterized=True)
        handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                   markersize=7, label=f"C{k}: {n_k:,}  ({pct:.0f}% gal)"))
        cx, cy = emb[labels == k, 0].mean(), emb[labels == k, 1].mean()
        ax.annotate(f"C{k}", (cx, cy), fontsize=11, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color,
                              alpha=0.85, lw=1.5))

    if noise.any():
        handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=NOISE_COLOR,
                   markersize=7, label=f"noise: {noise.sum():,}"))

    ax.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.8)
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_title(f"ERO UMAP clusters  (HDBSCAN, {len(cluster_ids)} clusters, "
                 f"N={len(emb):,})", fontsize=12, fontweight="bold")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(alpha=0.2)

    # Panel 1: CLASS_STAR-derived ground truth
    ax = axes[1]
    for cls_val, cls_name in [(0, "STAR"), (1, "GALAXY")]:
        mask = y[order] == cls_val
        ax.scatter(emb[order[mask], 0], emb[order[mask], 1],
                   c=CLASS_COLORS[cls_name], s=1, alpha=0.2,
                   linewidths=0, rasterized=True, label=cls_name)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.8, markerscale=2)
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_title(f"ERO CLASS_STAR ground truth  "
                 f"(>={CLASS_STAR_THRESH} STAR / <{CLASS_STAR_THRESH} GALAXY)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(alpha=0.2)

    fig.suptitle("ERO UMAP (class_star NOT NULL, G>=17, no subsample)",
                 fontsize=15, fontweight="bold", y=0.99)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full ERO UMAP + clustering")
    parser.add_argument("--test", action="store_true", help="10K subset for testing")
    parser.add_argument("--min-cluster-size", type=int, default=150)
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--replot", action="store_true",
                        help="Skip query/UMAP/HDBSCAN — reload saved npz, "
                             "re-run post-processing and re-plot only.")
    parser.add_argument("--rebalance", action="store_true",
                        help="Rebalance ERO star:galaxy ratio to match the Q1 "
                             f"training set ({Q1_STAR_GAL_RATIO:.2f}:1) by "
                             "undersampling stars (keeps all galaxies). "
                             "Output filenames get a _rebal suffix.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_rebal" if args.rebalance else ""
    npz_path = OUTPUT_DIR / f"ero_full{suffix}.npz"
    reducer_path = OUTPUT_DIR / f"ero_full{suffix}_reducer.joblib"
    plot_path = OUTPUT_DIR / f"ero_full{suffix}_clusters.png"

    if args.replot:
        if not npz_path.exists():
            print(f"ERROR: {npz_path} not found. Run without --replot first.")
            return 1
        print("=" * 60)
        print("[replot] Loading saved embedding + raw HDBSCAN labels...")
        data = np.load(npz_path, allow_pickle=True)
        embedding = data["embedding"]
        y = data["y"]

        print("\n[replot] Re-clustering...")
        labels = cluster_hdbscan(embedding, min_cluster_size=args.min_cluster_size)

        cluster_ids = sorted(set(labels[labels >= 0]))
        print(f"\n  Raw clusters:")
        for k in [-1] + cluster_ids:
            mask = labels == k
            n = mask.sum()
            if n == 0:
                continue
            n_gal = (y[mask] == 1).sum()
            pct = 100 * n_gal / n
            tag = "noise" if k == -1 else f"C{k}"
            print(f"    {tag:>8}: {n:>7,}  ({pct:.0f}% gal)")

        if not args.no_postprocess:
            print("\n[replot] Post-processing...")
            labels = postprocess(labels, embedding,
                                 DEFAULT_MERGES, DEFAULT_SPLITS, DEFAULT_SWAPS)
            print(f"\n  Final clusters:")
            for k in sorted(set(labels[labels >= 0])):
                mask = labels == k
                n = mask.sum()
                n_gal = (y[mask] == 1).sum()
                pct = 100 * n_gal / n
                print(f"    C{k}: {n:>7,}  ({pct:.0f}% gal)")
            if (labels == -1).any():
                print(f"    noise: {(labels == -1).sum():,}")

        print("\n[replot] Saving updated labels...")
        np.savez(npz_path,
                 embedding=embedding, X_raw=data["X_raw"], y=y,
                 labels=labels,
                 source_ids=data["source_ids"],
                 phot_g_mean_mag=data["phot_g_mean_mag"],
                 ra=data["ra"], dec=data["dec"],
                 class_star=data["class_star"],
                 mean_=data["mean_"], std_=data["std_"])

        print("\n[replot] Plotting...")
        plot_clusters(embedding, labels, y, plot_path)
        print("\nDone.")
        return 0

    # ── Full pipeline ─────────────────────────────────────────────────────
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: set PGUSER"); return 1

    # 1. Query
    print("=" * 60)
    print("[1] Querying ERO sources (no subsample)...")
    df = query_ero_full(username, test_mode=args.test)

    X = df[list(SELECTED_GAIA_PARAMETERS)].values.astype(float)
    y = df["label"].map(CLASS_MAPPING).values

    # 1b. Optional rebalance
    if args.rebalance:
        print(f"\n[1b] Rebalancing ERO to Q1 ratio "
              f"({Q1_STAR_GAL_RATIO:.2f}:1, keep all galaxies)...")
        sel = rebalance_indices_to_q1_ratio(y, Q1_STAR_GAL_RATIO, seed=42)
        df = df.iloc[sel].reset_index(drop=True)
        X = X[sel]
        y = y[sel]
        n_star = (y == 0).sum(); n_gal = (y == 1).sum()
        print(f"  Rebalanced pool: {len(y):,}  STAR: {n_star:,}, "
              f"GALAXY: {n_gal:,}  (ratio {n_star/max(n_gal,1):.2f}:1)")

    # 2. UMAP
    print("\n[2] Fitting UMAP...")
    embedding, reducer, mean_, std_ = fit_umap(X)

    # 3. Cluster
    print("\n[3] Clustering...")
    labels = cluster_hdbscan(embedding, min_cluster_size=args.min_cluster_size)

    cluster_ids = sorted(set(labels[labels >= 0]))
    print(f"\n  Raw clusters:")
    for k in [-1] + cluster_ids:
        mask = labels == k
        n = mask.sum()
        if n == 0:
            continue
        n_gal = (y[mask] == 1).sum()
        pct = 100 * n_gal / n
        tag = "noise" if k == -1 else f"C{k}"
        print(f"    {tag:>8}: {n:>7,}  ({pct:.0f}% gal)")

    # 4. Post-process
    if not args.no_postprocess:
        print("\n[4] Post-processing (merge/split/swap)...")
        labels = postprocess(labels, embedding,
                             DEFAULT_MERGES, DEFAULT_SPLITS, DEFAULT_SWAPS)
        print(f"\n  Final clusters:")
        for k in sorted(set(labels[labels >= 0])):
            mask = labels == k
            n = mask.sum()
            n_gal = (y[mask] == 1).sum()
            pct = 100 * n_gal / n
            print(f"    C{k}: {n:>7,}  ({pct:.0f}% gal)")
        if (labels == -1).any():
            print(f"    noise: {(labels == -1).sum():,}")

    # 5. Save
    print("\n[5] Saving...")
    np.savez(npz_path,
             embedding=embedding,
             X_raw=X,
             y=y,
             labels=labels,
             source_ids=df["g_source_id"].values,
             phot_g_mean_mag=pd.to_numeric(df["g_phot_g_mean_mag"], errors="coerce").values,
             ra=pd.to_numeric(df["ra"], errors="coerce").values,
             dec=pd.to_numeric(df["dec"], errors="coerce").values,
             class_star=pd.to_numeric(df["class_star"], errors="coerce").values,
             mean_=mean_, std_=std_)
    joblib.dump(reducer, reducer_path)
    print(f"  -> {npz_path}")

    # 6. Plot
    print("\n[6] Plotting...")
    plot_clusters(embedding, labels, y, plot_path)

    print("\n" + "=" * 60)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
