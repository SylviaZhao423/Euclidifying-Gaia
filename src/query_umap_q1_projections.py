#!/usr/bin/env python3
"""
Project ALL external sources into Q1 UMAP space and cache as CSV.

This is the single source of truth for all projected coordinates.
Downstream scripts (compile_umap_combined.py, analyze_*) load these CSVs
instead of re-projecting.

Projections:
  A. Q1 allphz       — all PHZ classes (0-7, -1) projected into Q1 UMAP space
  B. Q1×DESI         — Q1×DESI cross-match projected
  C. ERO regular      — ERO sources projected (ero_regular.npz)
  D. ERO rebalanced   — ERO rebalanced projected (ero_rebal.npz)
  E. ERO×DESI         — ERO×DESI cross-match projected (ero_desi.npz)
  F. Gaia random      — Gaia DR3 random subset projected (gaia_random.npz)
  G. Gaia high-lat    — Gaia DR3 high-lat subset projected (gaia_hilat.npz)

Output CSVs (under output/umap/arrays/projected/):
  q1_allphz_projected.csv
  q1_desi_projected.csv
  ero_regular_projected.csv
  ero_rebal_projected.csv
  ero_desi_projected.csv
  gaia_random_projected.csv
  gaia_hilat_projected.csv

Usage:
  python src/query_umap_q1_projections.py
  python src/query_umap_q1_projections.py --only gaia
  python src/query_umap_q1_projections.py --only allphz
  python src/query_umap_q1_projections.py --only ero
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR   = Path(__file__).resolve().parents[1]
ARRAYS_DIR    = PROJECT_DIR / "output" / "umap" / "arrays"
PROJECTED_DIR = ARRAYS_DIR / "projected"
MODEL_DIR     = PROJECT_DIR / "output" / "xgboost_q1"
MODEL_PATH    = MODEL_DIR / "xgb_classifier_q1.json"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load_q1():
    """Load Q1 embedding, labels, mean, std, and reducer."""
    q1 = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    reducer = joblib.load(ARRAYS_DIR / "q1_indiv_reducer.joblib")
    labels = np.load(ARRAYS_DIR / "q1_cluster_labels.npy")
    std_safe = np.where(q1["std"] > 0, q1["std"], 1.0)
    return q1, reducer, q1["mean"], std_safe, q1["embedding"], labels


def _project(X_raw: np.ndarray, q1_mean, std_safe, reducer) -> np.ndarray:
    """Standardise with Q1 stats and project via reducer.transform()."""
    X_scaled = (X_raw - q1_mean) / std_safe
    X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0.0)
    return reducer.transform(X_scaled)


def _assign_to_clusters(src_emb, q1_emb, q1_labels):
    """Assign each projected source to the nearest Q1 cluster via KNN."""
    from sklearn.neighbors import NearestNeighbors
    valid_mask = q1_labels >= 0
    if valid_mask.sum() == 0:
        return np.full(len(src_emb), -1, dtype=int)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(q1_emb[valid_mask])
    _, idx = nn.kneighbors(src_emb)
    idx = idx.ravel()
    return q1_labels[np.where(valid_mask)[0][idx]]


def _xgb_ensemble_pgal(X: np.ndarray) -> np.ndarray:
    """Run Q1 XGBoost seed ensemble → P(gal) array."""
    import xgboost as xgb
    seed_paths = sorted(MODEL_DIR.glob("xgb_classifier_q1_seed*.json"))
    if not seed_paths:
        # Fallback to single model
        if MODEL_PATH.exists():
            m = xgb.XGBClassifier()
            m.load_model(str(MODEL_PATH))
            X_clean = X.astype(np.float32).copy()
            for c in range(X_clean.shape[1]):
                bad = ~np.isfinite(X_clean[:, c])
                if bad.any():
                    X_clean[bad, c] = np.nanmedian(X_clean[:, c])
            return m.predict_proba(X_clean)[:, 1]
        return np.full(X.shape[0], np.nan)

    X_clean = X.astype(np.float32).copy()
    for c in range(X_clean.shape[1]):
        bad = ~np.isfinite(X_clean[:, c])
        if bad.any():
            med = np.nanmedian(X_clean[:, c])
            X_clean[bad, c] = med if np.isfinite(med) else 0.0
    all_probs = []
    for p in seed_paths:
        m = xgb.XGBClassifier()
        m.load_model(str(p))
        all_probs.append(m.predict_proba(X_clean)[:, 1])
    pgal = np.mean(all_probs, axis=0)
    print(f"  XGB ensemble ({len(seed_paths)} seeds) → P(gal) range "
          f"[{pgal.min():.3f}, {pgal.max():.3f}]  ({X.shape[0]:,} sources)")
    return pgal


# ── A. Q1 allphz ────────────────────────────────────────────────────────────

def project_allphz(q1_mean, std_safe, reducer, q1_emb, q1_labels) -> pd.DataFrame:
    """Load q1_allphz.npz, project into Q1 UMAP space, assign clusters, run XGB."""
    print("  Loading q1_allphz.npz...")
    data = np.load(ARRAYS_DIR / "q1_allphz.npz", allow_pickle=True)
    df = pd.DataFrame(data["X_features"], columns=SELECTED_GAIA_PARAMETERS)
    df["source_id"]       = data["source_ids"].astype(str)
    df["ra"]              = data["ra"].astype(float)
    df["dec"]             = data["dec"].astype(float)
    df["phot_g_mean_mag"] = data["phot_g_mean_mag"].astype(float)
    df["phz"]             = data["phz"].astype(int)
    df["phz_pgal"]        = data["phz_gal_prob"].astype(float)
    print(f"  {len(df):,} allphz sources")

    X = df[SELECTED_GAIA_PARAMETERS].values.astype(float)
    print("  Projecting into Q1 UMAP space...")
    emb = _project(X, q1_mean, std_safe, reducer)
    df["umap1"] = emb[:, 0]
    df["umap2"] = emb[:, 1]
    df["cluster"] = _assign_to_clusters(emb, q1_emb, q1_labels)

    print("  Running XGB P(gal)...")
    df["model_pgal"] = _xgb_ensemble_pgal(X)

    return df


# ── B. Q1×DESI ──────────────────────────────────────────────────────────────

def project_q1_desi(q1_mean, std_safe, reducer, q1_emb, q1_labels) -> pd.DataFrame:
    """Load q1_desi.npz, project, assign clusters, run XGB."""
    print("  Loading q1_desi.npz...")
    data = np.load(ARRAYS_DIR / "q1_desi.npz", allow_pickle=True)
    df = pd.DataFrame(data["X_features"], columns=SELECTED_GAIA_PARAMETERS)
    df["source_id"]       = data["source_ids"].astype(str)
    df["ra"]              = data["ra"].astype(float)
    df["dec"]             = data["dec"].astype(float)
    df["phot_g_mean_mag"] = data["phot_g_mean_mag"].astype(float)
    df["phz"]             = data["phz"].astype(int)
    df["phz_pgal"]        = data["phz_gal_prob"].astype(float)
    df["desi_z"]          = data["desi_z"].astype(float)
    df["spectype"]        = data["spectype"].astype(str)
    print(f"  {len(df):,} Q1×DESI sources")

    X = df[SELECTED_GAIA_PARAMETERS].values.astype(float)
    print("  Projecting into Q1 UMAP space...")
    emb = _project(X, q1_mean, std_safe, reducer)
    df["umap1"] = emb[:, 0]
    df["umap2"] = emb[:, 1]
    df["cluster"] = _assign_to_clusters(emb, q1_emb, q1_labels)

    print("  Running XGB P(gal)...")
    df["model_pgal"] = _xgb_ensemble_pgal(X)

    return df


# ── C/D. ERO regular / rebalanced ───────────────────────────────────────────

def project_ero(npz_name: str, q1_mean, std_safe, reducer,
                q1_emb, q1_labels) -> pd.DataFrame:
    """Load ERO npz, project into Q1 UMAP space, assign clusters, run XGB.

    npz_name: 'ero_regular.npz' or 'ero_rebal.npz'
    """
    npz_path = ARRAYS_DIR / npz_name
    print(f"  Loading {npz_name}...")
    npz = np.load(npz_path, allow_pickle=True)
    X_raw    = npz["X_raw"].astype(float)
    y        = npz["y"]
    sids     = npz["source_ids"].astype(str)
    gmag     = npz["g_mag"].astype(float) if "g_mag" in npz else None
    ra       = npz["ra"].astype(float) if "ra" in npz else None
    dec      = npz["dec"].astype(float) if "dec" in npz else None
    cs       = npz["class_star"].astype(float) if "class_star" in npz else None

    df = pd.DataFrame(X_raw, columns=SELECTED_GAIA_PARAMETERS)
    df["source_id"] = sids
    df["y"] = y
    if gmag is not None:
        df["phot_g_mean_mag"] = gmag
    if ra is not None:
        df["ra"] = ra
    if dec is not None:
        df["dec"] = dec
    if cs is not None:
        df["class_star"] = cs
    print(f"  {len(df):,} sources  (STAR={(y==0).sum():,}  GAL={(y==1).sum():,})")

    print("  Projecting into Q1 UMAP space...")
    emb = _project(X_raw, q1_mean, std_safe, reducer)
    df["umap1"] = emb[:, 0]
    df["umap2"] = emb[:, 1]
    df["cluster"] = _assign_to_clusters(emb, q1_emb, q1_labels)

    print("  Running XGB P(gal)...")
    df["model_pgal"] = _xgb_ensemble_pgal(X_raw)

    return df


# ── E. ERO×DESI ─────────────────────────────────────────────────────────────

def project_ero_desi(q1_mean, std_safe, reducer, q1_emb, q1_labels) -> pd.DataFrame:
    """Load ero_desi.npz, project, assign clusters, run XGB."""
    print("  Loading ero_desi.npz...")
    npz = np.load(ARRAYS_DIR / "ero_desi.npz", allow_pickle=True)
    X = npz["X_features"].astype(float)
    df = pd.DataFrame(X, columns=SELECTED_GAIA_PARAMETERS)
    df["source_id"]       = npz["source_ids"].astype(str)
    df["phot_g_mean_mag"] = npz["phot_g_mean_mag"].astype(float)
    df["ra"]              = npz["ra"].astype(float)
    df["dec"]             = npz["dec"].astype(float)
    df["spectype"]        = npz["spectype"].astype(str)
    df["desi_z"]          = npz["desi_z"].astype(float)
    df["class_star"]      = npz["class_star"].astype(float)
    df["cat_value"]       = npz["cat_value"].astype(int)
    print(f"  {len(df):,} ERO×DESI sources  "
          f"(cat=2: {(df['cat_value']==2).sum():,},  "
          f"cat=3: {(df['cat_value']==3).sum():,})")

    print("  Projecting into Q1 UMAP space...")
    emb = _project(X, q1_mean, std_safe, reducer)
    df["umap1"] = emb[:, 0]
    df["umap2"] = emb[:, 1]
    df["cluster"] = _assign_to_clusters(emb, q1_emb, q1_labels)

    print("  Running XGB P(gal)...")
    df["model_pgal"] = _xgb_ensemble_pgal(X)

    return df


# ── F/G. Gaia random / high-lat ─────────────────────────────────────────────

def project_gaia(npz_name: str, q1_mean, std_safe, reducer,
                 q1_emb, q1_labels) -> pd.DataFrame:
    """Load Gaia npz, project into Q1 UMAP space, assign clusters, run XGB."""
    npz_path = ARRAYS_DIR / npz_name
    print(f"  Loading {npz_name}...")
    npz = np.load(npz_path, allow_pickle=True)
    X = npz["X_features"].astype(float)
    df = pd.DataFrame(X, columns=SELECTED_GAIA_PARAMETERS)
    df["source_id"]       = npz["source_ids"].astype(str)
    df["ra"]              = npz["ra"].astype(float)
    df["dec"]             = npz["dec"].astype(float)
    df["phot_g_mean_mag"] = npz["phot_g_mean_mag"].astype(float)
    print(f"  {len(df):,} Gaia sources")

    print("  Projecting into Q1 UMAP space...")
    emb = _project(X, q1_mean, std_safe, reducer)
    df["umap1"] = emb[:, 0]
    df["umap2"] = emb[:, 1]
    df["cluster"] = _assign_to_clusters(emb, q1_emb, q1_labels)

    print("  Running XGB P(gal)...")
    df["gal_prob"] = _xgb_ensemble_pgal(X)

    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    global MODEL_DIR, MODEL_PATH, PROJECTED_DIR

    parser = argparse.ArgumentParser(
        description="Project all datasets into Q1 UMAP space and cache as CSV.")
    parser.add_argument("--only", choices=["allphz", "desi", "ero", "ero_desi", "gaia"],
                        help="Run only one projection group (default: all).")
    parser.add_argument(
        "--model", choices=["q1", "q1_op_f1"], default="q1",
        help="Which model to use: q1 (original) or q1_op_f1 (Optuna F1 tuned)",
    )
    args = parser.parse_args()

    if args.model == "q1_op_f1":
        MODEL_DIR     = PROJECT_DIR / "output" / "xgboost_q1_op_f1"
        MODEL_PATH    = MODEL_DIR / "xgb_classifier_q1.json"
        PROJECTED_DIR = ARRAYS_DIR / "projected_op_f1"

    PROJECTED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Q1 UMAP Projections")
    print("=" * 60)

    # Check prerequisite files
    for f in ["q1_indiv.npz", "q1_indiv_reducer.joblib", "q1_cluster_labels.npy"]:
        if not (ARRAYS_DIR / f).exists():
            print(f"ERROR: {ARRAYS_DIR / f} not found.")
            print("Run query_umap_q1_50k.py first.")
            return 1

    # Load Q1 base data
    print("\n  Loading Q1 base (embedding, reducer, labels)...")
    q1, reducer, q1_mean, std_safe, q1_emb, q1_labels = _load_q1()
    print(f"  Q1 base: {len(q1_emb):,} sources, "
          f"{len(set(q1_labels[q1_labels >= 0]))} clusters")

    do_all = args.only is None

    # ── A. Q1 allphz ─────────────────────────────────────────────────────────
    if do_all or args.only == "allphz":
        csv_path = PROJECTED_DIR / "q1_allphz_projected.csv"
        print(f"\n[A] Q1 allphz projection...")
        if not (ARRAYS_DIR / "q1_allphz.npz").exists():
            print("  q1_allphz.npz not found — skipping.")
        else:
            df = project_allphz(q1_mean, std_safe, reducer, q1_emb, q1_labels)
            df.to_csv(csv_path, index=False, float_format="%.17g")
            print(f"  Saved -> {csv_path}  ({len(df):,} sources)")

    # ── B. Q1×DESI ───────────────────────────────────────────────────────────
    if do_all or args.only == "desi":
        csv_path = PROJECTED_DIR / "q1_desi_projected.csv"
        print(f"\n[B] Q1×DESI projection...")
        if not (ARRAYS_DIR / "q1_desi.npz").exists():
            print("  q1_desi.npz not found — skipping.")
        else:
            df = project_q1_desi(q1_mean, std_safe, reducer, q1_emb, q1_labels)
            df.to_csv(csv_path, index=False, float_format="%.17g")
            print(f"  Saved -> {csv_path}  ({len(df):,} sources)")

    # ── C. ERO regular ───────────────────────────────────────────────────────
    if do_all or args.only == "ero":
        csv_path = PROJECTED_DIR / "ero_regular_projected.csv"
        print(f"\n[C] ERO regular projection...")
        if not (ARRAYS_DIR / "ero_regular.npz").exists():
            print("  ero_regular.npz not found — skipping.")
        else:
            df = project_ero("ero_regular.npz", q1_mean, std_safe, reducer,
                             q1_emb, q1_labels)
            df.to_csv(csv_path, index=False, float_format="%.17g")
            print(f"  Saved -> {csv_path}  ({len(df):,} sources)")

    # ── D. ERO rebalanced ────────────────────────────────────────────────────
    if do_all or args.only == "ero":
        csv_path = PROJECTED_DIR / "ero_rebal_projected.csv"
        print(f"\n[D] ERO rebalanced projection...")
        if not (ARRAYS_DIR / "ero_rebal.npz").exists():
            print("  ero_rebal.npz not found — skipping.")
        else:
            df = project_ero("ero_rebal.npz", q1_mean, std_safe,
                             reducer, q1_emb, q1_labels)
            df.to_csv(csv_path, index=False, float_format="%.17g")
            print(f"  Saved -> {csv_path}  ({len(df):,} sources)")

    # ── E. ERO×DESI ──────────────────────────────────────────────────────────
    if do_all or args.only == "ero_desi":
        csv_path = PROJECTED_DIR / "ero_desi_projected.csv"
        print(f"\n[E] ERO×DESI projection...")
        if not (ARRAYS_DIR / "ero_desi.npz").exists():
            print("  ero_desi.npz not found — skipping.")
        else:
            df = project_ero_desi(q1_mean, std_safe, reducer, q1_emb, q1_labels)
            df.to_csv(csv_path, index=False, float_format="%.17g")
            print(f"  Saved -> {csv_path}  ({len(df):,} sources)")

    # ── F. Gaia random ───────────────────────────────────────────────────────
    if do_all or args.only == "gaia":
        csv_path = PROJECTED_DIR / "gaia_random_projected.csv"
        print(f"\n[F] Gaia random projection...")
        if not (ARRAYS_DIR / "gaia_random.npz").exists():
            print("  gaia_random.npz not found — skipping.")
        else:
            df = project_gaia("gaia_random.npz", q1_mean, std_safe, reducer,
                              q1_emb, q1_labels)
            df.to_csv(csv_path, index=False, float_format="%.17g")
            print(f"  Saved -> {csv_path}  ({len(df):,} sources)")

        # ── G. Gaia high-lat ─────────────────────────────────────────────────
        csv_path = PROJECTED_DIR / "gaia_hilat_projected.csv"
        print(f"\n[G] Gaia high-lat projection...")
        if not (ARRAYS_DIR / "gaia_hilat.npz").exists():
            print("  gaia_hilat.npz not found — skipping.")
        else:
            df = project_gaia("gaia_hilat.npz", q1_mean, std_safe, reducer,
                              q1_emb, q1_labels)
            df.to_csv(csv_path, index=False, float_format="%.17g")
            print(f"  Saved -> {csv_path}  ({len(df):,} sources)")

    print("\n" + "=" * 60)
    print("Done. All projected CSVs saved to:")
    print(f"  {PROJECTED_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
