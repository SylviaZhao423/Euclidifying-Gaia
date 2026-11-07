#!/usr/bin/env python3
"""
UMAP + HDBSCAN on all Q1 training sources (no subsampling).

Queries all sources from crossmatch_q1_mer_phz_vis_gaia_sep with
phz IN (1,2) and G >= 17 — same filters as query_umap_array.py but
without the 50K random subsample. Uses all ~240K sources.

Outputs (under output/umap/q1_full/):
  q1_full.npz                 — embedding, labels, features, metadata
  q1_full_reducer.joblib      — fitted UMAP reducer
  q1_full_clusters.png        — cluster + PHZ ground truth plot

Usage:
    PGUSER=siwei_zhao_2025 python src/query_umap_q1_full.py
    PGUSER=siwei_zhao_2025 python src/query_umap_q1_full.py --test  # 10K subset
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "q1_full"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

CLASS_MAPPING = {"STAR": 0, "GALAXY": 1}
MAG_CUT = 17

# Post-processing defaults for full Q1 (different from subsample due to
# different UMAP layout).  Raw HDBSCAN gives C0(306), C1(1381), C2(21372),
# C3(20251), C4(4538), C5(190998).
#   1. Merge C0→C5  (absorb tiny 306-pt into main star body)
#   2. Merge C4→C1  (bridge cluster: 4538+1381=5919)
#   3. Renumber → C0(old1=5919), C1(old2=21372), C2(old3=20251), C3(old5=191304)
#   4. Swap 0↔2   → C0=galaxy-rich(20251), C1=dense-middle(21372),
#                    C2=bridge(5919), C3=main-star(191304)
DEFAULT_MERGES = ["0>5", "4>1"]
DEFAULT_SPLITS = []
DEFAULT_SWAPS  = ["0:2"]

CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#F298B6", "mediumpurple",
    "#17becf", "#d62728", "#8c564b", "#7f7f7f", "#bcbd22",
]
NOISE_COLOR = "#cccccc"
CLASS_COLORS = {"STAR": "#E53935", "GALAXY": "#1976D2"}


# ── Query ────────────────────────────────────────────────────────────────────

def query_q1_full(username, test_mode=False):
    """Query phz IN (1,2), G>=17 sources — same as training, no subsample."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)

    gaia_cols = ", ".join([f"g_{c} AS {c}" for c in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT
            g_source_id,
            {gaia_cols},
            g_phot_g_mean_mag,
            m_right_ascension AS ra,
            m_declination AS dec,
            p_phz_gal_prob AS phz_gal_prob,
            CASE
                WHEN p_phz_classification = 1 THEN 'STAR'
                WHEN p_phz_classification = 2 THEN 'GALAXY'
            END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep
        WHERE p_phz_classification IN (1, 2)
          AND g_phot_g_mean_mag >= {MAG_CUT}
        ORDER BY g_source_id
    """
    if test_mode:
        query += " LIMIT 10000"

    print("  Querying Q1 (phz IN (1,2), G>=17, no subsample)...")
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
    """Merge, split, renumber, swap — same logic as analyze_umap_q1_clusters.py."""
    # Merge
    for m in merges:
        src, dst = (int(x) for x in m.split(">"))
        n = (labels == src).sum()
        if n:
            labels[labels == src] = dst
            print(f"  Merged C{src} ({n:,}) -> C{dst}")

    # Split
    next_id = int(labels.max()) + 1
    for s in splits:
        cid, yval = s.split(":")
        cid, yval = int(cid), float(yval)
        mask = (labels == cid) & (embedding[:, 1] < yval)
        if mask.any():
            labels[mask] = next_id
            print(f"  Split C{cid} at UMAP2<{yval}: {(labels == cid).sum():,} upper + {mask.sum():,} lower (C{next_id})")
            next_id += 1

    # Renumber
    old_ids = sorted(set(labels[labels >= 0]))
    mapping = {old: new for new, old in enumerate(old_ids)}
    for old, new in mapping.items():
        if old != new:
            labels[labels == old] = -(new + 100)
    for old, new in mapping.items():
        if old != new:
            labels[labels == -(new + 100)] = new

    # Swap
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
    ax.set_title(f"Q1 UMAP clusters  (HDBSCAN, {len(cluster_ids)} clusters, N={len(emb):,})",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(alpha=0.2)

    # Panel 1: PHZ ground truth
    ax = axes[1]
    for cls_val, cls_name in [(0, "STAR"), (1, "GALAXY")]:
        mask = y[order] == cls_val
        ax.scatter(emb[order[mask], 0], emb[order[mask], 1],
                   c=CLASS_COLORS[cls_name], s=1, alpha=0.2,
                   linewidths=0, rasterized=True, label=cls_name)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.8, markerscale=2)
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_title("Q1 PHZ ground truth  (Star / Galaxy)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(alpha=0.2)

    fig.suptitle("Q1 UMAP (phz 1/2, G>=17, no subsample)",
                 fontsize=15, fontweight="bold", y=0.99)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full Q1 UMAP + clustering")
    parser.add_argument("--test", action="store_true", help="10K subset for testing")
    parser.add_argument("--min-cluster-size", type=int, default=150)
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--replot", action="store_true",
                        help="Skip query/UMAP/HDBSCAN — reload saved npz, "
                             "re-run post-processing and re-plot only.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUTPUT_DIR / "q1_full.npz"

    if args.replot:
        # ── Replot mode: load saved data, redo post-process + plot ────────
        if not npz_path.exists():
            print(f"ERROR: {npz_path} not found. Run without --replot first.")
            return 1
        print("=" * 60)
        print("[replot] Loading saved embedding + raw HDBSCAN labels...")
        data = np.load(npz_path, allow_pickle=True)
        embedding = data["embedding"]
        y = data["y"]

        # Re-cluster from embedding (post-processing depends on raw labels)
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

        # Re-save with updated labels
        print("\n[replot] Saving updated labels...")
        np.savez(npz_path,
                 embedding=embedding, X_raw=data["X_raw"], y=y,
                 labels=labels,
                 source_ids=data["source_ids"],
                 phot_g_mean_mag=data["phot_g_mean_mag"],
                 ra=data["ra"], dec=data["dec"],
                 phz_gal_prob=data["phz_gal_prob"],
                 mean_=data["mean_"], std_=data["std_"])

        print("\n[replot] Plotting...")
        plot_clusters(embedding, labels, y, OUTPUT_DIR / "q1_full_clusters.png")
        print("\nDone.")
        return 0

    # ── Full pipeline ─────────────────────────────────────────────────────
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: set PGUSER"); return 1

    # 1. Query
    print("=" * 60)
    print("[1] Querying Q1 training sources (no subsample)...")
    df = query_q1_full(username, test_mode=args.test)

    X = df[list(SELECTED_GAIA_PARAMETERS)].values.astype(float)
    y = df["label"].map(CLASS_MAPPING).values

    # 2. UMAP
    print("\n[2] Fitting UMAP...")
    embedding, reducer, mean_, std_ = fit_umap(X)

    # 3. Cluster
    print("\n[3] Clustering...")
    labels = cluster_hdbscan(embedding, min_cluster_size=args.min_cluster_size)

    # Print raw composition
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
             phz_gal_prob=pd.to_numeric(df["phz_gal_prob"], errors="coerce").values,
             mean_=mean_, std_=std_)
    joblib.dump(reducer, OUTPUT_DIR / "q1_full_reducer.joblib")
    print(f"  -> {npz_path}")

    # 6. Plot
    print("\n[6] Plotting...")
    plot_clusters(embedding, labels, y, OUTPUT_DIR / "q1_full_clusters.png")

    print("\n" + "=" * 60)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
