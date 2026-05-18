#!/usr/bin/env python3
"""
ERO counterpart of query_umap_q1_50k.py.

Mirrors the Q1 50k pipeline but with ERO as the base dataset:
queries 50k random ERO sources (e_v_class_star NOT NULL, G>=17), fits UMAP
+ HDBSCAN, and saves the base-array trio used by downstream ERO-centric
analyses.

Auxiliary datasets that q1_50k.py already produces (q1_allphz, q1_desi,
ero_desi, gaia_random, gaia_hilat) are NOT re-queried here — they are
dataset-agnostic and live in output/umap/arrays/.

Outputs (under output/umap/ero_individual/arrays/):
    ero_indiv.npz                — 50k ERO sources: embedding, y (class_star
                                   threshold), source_ids, X_raw, mean, std,
                                   ra, dec, class_star, g_mag, model_pgal
                                   (Q1-trained XGB ensemble applied to ERO)
    ero_indiv_reducer.joblib     — fitted UMAP reducer (for projections)
    ero_cluster_labels.npy       — post-processed HDBSCAN cluster labels
                                   (noise=-1, clusters 0-N)

Usage:
    PGUSER=<user> python src/slides/query_umap_ero_50k.py
    PGUSER=<user> python src/slides/query_umap_ero_50k.py --test-mode
    PGUSER=<user> python src/slides/query_umap_ero_50k.py --no-postprocess
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import umap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[2]
ERO_DIR     = PROJECT_DIR / "output" / "umap" / "ero_individual"
ARRAYS_DIR  = ERO_DIR / "arrays"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
MAG_CUT   = 17
SAMPLE_SIZE = 50_000
CLASS_STAR_THRESH = 0.65

# Q1 training-set star:galaxy ratio (from filter Sankey: 176,647 / 62,221).
# Used by --rebalance to bring the ERO pool (~18.45:1) down to Q1's ratio.
Q1_STAR_GAL_RATIO = 176_647 / 62_221  # ≈ 2.84

# HDBSCAN post-processing defaults — empty until raw ERO clusters inspected.
# Use --no-postprocess explicitly if you want to skip.
DEFAULT_MERGES: list[str] = []
DEFAULT_SPLITS: list[str] = []
DEFAULT_SWAPS:  list[str] = []


# ── DB query ────────────────────────────────────────────────────────────────

def query_ero(username: str, test_mode: bool = False) -> tuple:
    """Query ERO sources with Gaia features.

    Returns (X, y, g_mag, source_ids, ra, dec, class_star).
    Label derived from CLASS_STAR threshold (>= 0.65 → STAR, else GALAXY).
    """
    import psycopg2

    gaia_cols = ", ".join(f"g_{col} AS {col}" for col in SELECTED_GAIA_PARAMETERS)
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
        ORDER BY g_random_index
    """
    if test_mode:
        query += " LIMIT 10000"

    print(f"  Querying ERO (class_star NOT NULL, G>={MAG_CUT})...")
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    df = pd.read_sql_query(query, conn)
    conn.close()

    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace("NOT_AVAILABLE", np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df[SELECTED_GAIA_PARAMETERS].values
    y = df["label"].map({"STAR": 0, "GALAXY": 1}).values
    g_mag = pd.to_numeric(df["g_phot_g_mean_mag"], errors="coerce").values
    source_ids = df["g_source_id"].astype(str).values
    ra  = pd.to_numeric(df["ra"],  errors="coerce").values
    dec = pd.to_numeric(df["dec"], errors="coerce").values
    class_star = pd.to_numeric(df["class_star"], errors="coerce").values

    n_star = (y == 0).sum()
    n_gal  = (y == 1).sum()
    print(f"  -> {len(X):,} sources — STAR: {n_star:,}, GALAXY: {n_gal:,}")

    return X, y, g_mag, source_ids, ra, dec, class_star


# ── Subsample + standardize ────────────────────────────────────────────────

def clean_for_umap(X: np.ndarray, sample_size: int,
                   rng: np.random.RandomState) -> tuple:
    """Subsample, fill NaN with column median, Z-score standardize.

    Returns (X_scaled, idx, mean_, std_).
    """
    idx = rng.choice(len(X), min(sample_size, len(X)), replace=False)
    X_sub = X[idx].copy()
    for col in range(X_sub.shape[1]):
        bad = np.isnan(X_sub[:, col])
        if bad.any():
            X_sub[bad, col] = np.nanmedian(X_sub[:, col])
    mean_ = np.nanmean(X_sub, axis=0)
    std_  = np.nanstd(X_sub,  axis=0)
    X_scaled = (X_sub - mean_) / np.where(std_ > 0, std_, 1.0)
    return X_scaled, idx, mean_, std_


def rebalance_indices_to_q1_ratio(y: np.ndarray,
                                  star_gal_ratio: float,
                                  target_total: int,
                                  seed: int = 42) -> np.ndarray:
    """Stratified resample of indices to match a target star:galaxy ratio.

    Keeps as many galaxies as fit, then undersamples stars to hit the ratio.
    """
    rng = np.random.RandomState(seed)
    idx_star = np.where(y == 0)[0]
    idx_gal  = np.where(y == 1)[0]
    n_gal_target = int(target_total / (1 + star_gal_ratio))
    n_gal        = min(len(idx_gal), n_gal_target)
    n_star       = min(int(round(n_gal * star_gal_ratio)), len(idx_star))
    sel_gal  = rng.choice(idx_gal,  n_gal,  replace=False)
    sel_star = rng.choice(idx_star, n_star, replace=False)
    sel = np.concatenate([sel_star, sel_gal])
    rng.shuffle(sel)
    return sel


# ── HDBSCAN clustering + post-processing ───────────────────────────────────

def cluster_hdbscan(embedding: np.ndarray,
                    min_cluster_size: int = 150) -> np.ndarray:
    """Run HDBSCAN on 2D embedding. Falls back to KMeans if unavailable."""
    try:
        import hdbscan
        print(f"  HDBSCAN (min_cluster_size={min_cluster_size})...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=max(10, min_cluster_size // 10),
            cluster_selection_epsilon=0.3,
        )
        labels = clusterer.fit_predict(embedding)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"  -> {n_clusters} clusters found  ({(labels == -1).sum():,} noise points)")
    except ImportError:
        from sklearn.cluster import KMeans
        print("  hdbscan not available — falling back to KMeans(k=5)...")
        labels = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(embedding)
    return labels


def postprocess_labels(labels: np.ndarray, embedding: np.ndarray,
                       merges: list[str], splits: list[str],
                       swaps: list[str]) -> np.ndarray:
    """Apply merges, splits, renumbering, and swaps to cluster labels."""
    labels = labels.copy()

    for m in merges:
        src, dst = m.split(">")
        src, dst = int(src), int(dst)
        n = (labels == src).sum()
        if n == 0:
            continue
        labels[labels == src] = dst
        print(f"  Merged C{src} ({n:,} pts) -> C{dst}")

    for s in splits:
        cid, yval = s.split(":")
        cid, yval = int(cid), float(yval)
        next_id = int(labels.max()) + 1
        mask = (labels == cid) & (embedding[:, 1] < yval)
        if not mask.any():
            continue
        labels[mask] = next_id
        print(f"  Split C{cid} at UMAP2 < {yval}: "
              f"{(labels == cid).sum():,} upper + {mask.sum():,} lower (C{next_id})")

    old_ids = sorted(set(labels[labels >= 0]))
    mapping = {old: new for new, old in enumerate(old_ids)}
    for old, new in mapping.items():
        if old != new:
            labels[labels == old] = -(new + 100)
    for old, new in mapping.items():
        if old != new:
            labels[labels == -(new + 100)] = new

    for s in swaps:
        a, b = s.split(":")
        a, b = int(a), int(b)
        mask_a = labels == a
        mask_b = labels == b
        labels[mask_a] = b
        labels[mask_b] = a
        print(f"  Swapped C{a} ({mask_a.sum():,} pts) <-> C{b} ({mask_b.sum():,} pts)")

    return labels


def print_composition(labels: np.ndarray, y: np.ndarray,
                      emb: np.ndarray) -> None:
    """Print cluster composition table."""
    cluster_ids = sorted(set(labels[labels >= 0]))
    print(f"  {'cluster':>8}  {'total':>7}  {'star':>7}  {'galaxy':>8}  {'gal%':>6}  {'centroid':>16}")
    print(f"  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*6}  {'─'*16}")
    for k in [-1] + cluster_ids:
        mask = labels == k
        n = mask.sum()
        if n == 0:
            continue
        n_gal  = (y[mask] == 1).sum()
        n_star = (y[mask] == 0).sum()
        pct = 100 * n_gal / n
        cx, cy = emb[mask, 0].mean(), emb[mask, 1].mean()
        tag = "noise" if k == -1 else f"C{k}"
        print(f"  {tag:>8}  {n:>7,}  {n_star:>7,}  {n_gal:>8,}  {pct:>5.1f}%  ({cx:.1f}, {cy:.1f})")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Query 50k ERO sources, fit UMAP + HDBSCAN, save arrays.")
    parser.add_argument("--test-mode", action="store_true",
                        help="Limit DB query to 10k sources")
    parser.add_argument("--min-cluster-size", type=int, default=150,
                        help="HDBSCAN min_cluster_size (default: 150)")
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Disable default merges/splits/swaps")
    parser.add_argument("--rebalance", action="store_true",
                        help="Rebalance ERO star:galaxy ratio to match the Q1 "
                             f"training set ({Q1_STAR_GAL_RATIO:.2f}:1) before "
                             "subsampling. Output filenames get a _rebal suffix.")
    args = parser.parse_args()

    suffix = "_rebal" if args.rebalance else ""

    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return 1

    ARRAYS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ERO UMAP Builder")
    print("=" * 60)
    print(f"  Features: {len(SELECTED_GAIA_PARAMETERS)}")
    print(f"  Magnitude cut: G >= {MAG_CUT}")
    print(f"  CLASS_STAR threshold: {CLASS_STAR_THRESH}")
    print(f"  Sample size: {SAMPLE_SIZE:,}")
    print(f"  Output: {ARRAYS_DIR}")

    # ── 1. Query ─────────────────────────────────────────────────────────
    print("\n[1] Querying ERO dataset...")
    X, y, g_mag, source_ids, ra, dec, class_star = query_ero(
        username, test_mode=args.test_mode)

    # ── 1b. Optional rebalance to Q1 star:galaxy ratio ──────────────────
    if args.rebalance:
        print(f"\n[1b] Rebalancing ERO to Q1 ratio "
              f"({Q1_STAR_GAL_RATIO:.2f}:1, target={SAMPLE_SIZE:,})...")
        sel = rebalance_indices_to_q1_ratio(
            y, Q1_STAR_GAL_RATIO, SAMPLE_SIZE, seed=42)
        X = X[sel]
        y = y[sel]
        g_mag = g_mag[sel]
        source_ids = source_ids[sel]
        ra = ra[sel]
        dec = dec[sel]
        class_star = class_star[sel]
        n_star = (y == 0).sum(); n_gal = (y == 1).sum()
        print(f"  Rebalanced pool: {len(y):,}  STAR: {n_star:,}, GALAXY: {n_gal:,}  "
              f"(ratio {n_star/max(n_gal,1):.2f}:1)")

    # ── 2. Subsample + standardize ───────────────────────────────────────
    print("\n[2] Subsampling and standardizing...")
    rng = np.random.RandomState(42)
    X_scaled, idx, mean_, std_ = clean_for_umap(X, SAMPLE_SIZE, rng)
    X_raw = X[idx].copy()
    y_sub = y[idx]
    g_mag_sub = g_mag[idx]
    source_ids_sub = source_ids[idx]
    ra_sub  = ra[idx]
    dec_sub = dec[idx]
    class_star_sub = class_star[idx]
    print(f"  {len(X_scaled):,} sources subsampled")

    # ── 3. Fit UMAP ─────────────────────────────────────────────────────
    print("\n[3] Fitting UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                        random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    print(f"  Embedding shape: {embedding.shape}")

    # ── 4. Compute Q1-model P(gal) via XGB seed ensemble (cross-domain) ─
    print("\n[4] Computing Q1-trained model P(gal) on ERO sources...")
    model_pgal = np.full(len(X_raw), np.nan)
    seed_dir = PROJECT_DIR / "output" / "xgboost_q1" / "seed_models"
    seed_paths = sorted(seed_dir.glob("xgb_classifier_q1_seed*.json"))
    if seed_paths:
        import xgboost as xgb
        X_inf = X_raw.astype(np.float32).copy()
        for c in range(X_inf.shape[1]):
            bad = ~np.isfinite(X_inf[:, c])
            if bad.any():
                med = np.nanmedian(X_inf[:, c])
                X_inf[bad, c] = med if np.isfinite(med) else 0.0
        all_probs = []
        for p in seed_paths:
            m = xgb.XGBClassifier()
            m.load_model(str(p))
            all_probs.append(m.predict_proba(X_inf)[:, 1])
        model_pgal = np.mean(all_probs, axis=0)
        print(f"  Q1 XGB ensemble ({len(seed_paths)} seeds) → P(gal) range "
              f"[{model_pgal.min():.3f}, {model_pgal.max():.3f}]")
    else:
        print(f"  WARNING: no seed models in {seed_dir} — model_pgal will be NaN")

    # ── 5. Save npz + reducer ────────────────────────────────────────────
    print("\n[5] Saving arrays...")
    npz_path = ARRAYS_DIR / f"ero_indiv{suffix}.npz"
    reducer_path = ARRAYS_DIR / f"ero_indiv{suffix}_reducer.joblib"
    np.savez(npz_path,
             embedding=embedding,
             y=y_sub,
             source_ids=np.array(source_ids_sub, dtype=str),
             X_raw=X_raw,
             mean=mean_,
             std=std_,
             ra=ra_sub,
             dec=dec_sub,
             class_star=class_star_sub,
             g_mag=g_mag_sub,
             model_pgal=model_pgal)
    joblib.dump(reducer, reducer_path)
    print(f"  -> {npz_path.name}  ({len(embedding):,} sources)")
    print(f"  -> {reducer_path.name}")

    # ── 6. HDBSCAN clustering ────────────────────────────────────────────
    print("\n[6] Clustering...")
    labels = cluster_hdbscan(embedding, min_cluster_size=args.min_cluster_size)

    print("\n  Raw HDBSCAN clusters:")
    print_composition(labels, y_sub, embedding)

    if args.no_postprocess:
        merges, splits, swaps = [], [], []
    else:
        merges, splits, swaps = DEFAULT_MERGES, DEFAULT_SPLITS, DEFAULT_SWAPS

    if merges or splits or swaps:
        print("\n  Post-processing...")
        labels = postprocess_labels(labels, embedding, merges, splits, swaps)
        print("\n  Final cluster composition:")
        print_composition(labels, y_sub, embedding)

    labels_path = ARRAYS_DIR / f"ero_cluster_labels{suffix}.npy"
    np.save(labels_path, labels)
    print(f"\n  -> {labels_path.name}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
