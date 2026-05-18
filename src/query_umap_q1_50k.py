#!/usr/bin/env python3
"""
Query Q1 sources, fit UMAP + HDBSCAN clustering, and query auxiliary datasets.

Step 1: Query 50k Q1 PHZ 1,2 (G>=17), fit UMAP, run HDBSCAN → base arrays
Step 2: Query auxiliary datasets (allphz, Q1×DESI, ERO, ERO×DESI) → npz files

Outputs (under output/umap/arrays/):
    q1_indiv.npz                — 50k Q1 PHZ 1,2: embedding, y, source_ids, X_raw, mean, std, ra, dec, phz_gal_prob, g_mag
    q1_indiv_reducer.joblib     — fitted UMAP reducer (for projecting new data)
    q1_cluster_labels.npy       — post-processed HDBSCAN cluster labels (noise=-1, clusters 0-4)
    q1_allphz.npz               — all Q1 PHZ classes (0-7, -1): X_features, source_ids, ra, dec, phot_g_mean_mag, phz, phz_gal_prob
    q1_desi.npz                 — Q1×DESI cross-match: same + desi_z, spectype
    ero_regular.npz             — 50k ERO random subsample: X_raw, y, source_ids, g_mag, ra, dec, class_star
    ero_rebal.npz               — ERO resampled at Q1 star:galaxy ratio
    ero_desi.npz                — ERO×DESI cross-match: X_features, source_ids, phot_g_mean_mag, ra, dec, spectype, desi_z, class_star, cat_value

Usage:
    PGUSER=<user> python src/query_umap_q1_50k.py
    PGUSER=<user> python src/query_umap_q1_50k.py --test-mode   # 10k limit for quick testing
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

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS, DERIVED_COLUMNS

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
MAG_CUT   = 17
SAMPLE_SIZE = 50_000

# HDBSCAN post-processing defaults (same as analyze_umap_q1_clusters.py)
DEFAULT_MERGES = ["4>3"]
DEFAULT_SPLITS = ["2:5.0"]
DEFAULT_SWAPS  = ["2:3"]


# ── DB query ────────────────────────────────────────────────────────────────

def query_q1(username: str, test_mode: bool = False) -> tuple:
    """Query Q1 PHZ 1,2 sources with Gaia features from WSDB.

    Returns (X, y, g_mag, source_ids, ra, dec, phz_gal_prob).
    """
    import psycopg2

    gaia_cols = ", ".join(f"g_{col} AS {col}" for col in SELECTED_GAIA_PARAMETERS)
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
            END as label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep
        WHERE p_phz_classification IN (1, 2)
            AND g_phot_g_mean_mag >= {MAG_CUT}
        ORDER BY g_random_index
    """
    if test_mode:
        query += " LIMIT 10000"

    print(f"  Querying Q1 (PHZ 1,2, G>={MAG_CUT})...")
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
    phz_gal_prob = pd.to_numeric(df["phz_gal_prob"], errors="coerce").values

    n_star = (y == 0).sum()
    n_gal  = (y == 1).sum()
    print(f"  -> {len(X):,} sources — STAR: {n_star:,}, GALAXY: {n_gal:,}")

    return X, y, g_mag, source_ids, ra, dec, phz_gal_prob


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

    # Merges
    for m in merges:
        src, dst = m.split(">")
        src, dst = int(src), int(dst)
        n = (labels == src).sum()
        if n == 0:
            continue
        labels[labels == src] = dst
        print(f"  Merged C{src} ({n:,} pts) -> C{dst}")

    # Splits
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

    # Renumber to consecutive 0,1,2,...
    old_ids = sorted(set(labels[labels >= 0]))
    mapping = {old: new for new, old in enumerate(old_ids)}
    for old, new in mapping.items():
        if old != new:
            labels[labels == old] = -(new + 100)
    for old, new in mapping.items():
        if old != new:
            labels[labels == -(new + 100)] = new

    # Swaps
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


# ── Auxiliary dataset queries ──────────────────────────────────────────────

def query_q1_allphz(username: str) -> pd.DataFrame:
    """Query ALL Q1 PHZ classes (0-7, -1) with Gaia features."""
    import psycopg2
    gaia_cols = ", ".join(f"g_{col} AS {col}" for col in SELECTED_GAIA_PARAMETERS)
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    df = pd.read_sql_query(f"""
        SELECT
            g_source_id AS source_id,
            m_right_ascension AS ra,
            m_declination AS dec,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            p_phz_classification AS phz,
            p_phz_gal_prob AS phz_gal_prob,
            {gaia_cols}
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep
        WHERE p_phz_classification IS NOT NULL
          AND g_phot_g_mean_mag >= {MAG_CUT}
          AND m_right_ascension IS NOT NULL
          AND m_declination IS NOT NULL
        ORDER BY g_source_id
    """, conn)
    conn.close()
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace("NOT_AVAILABLE", np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["phz"] = df["phz"].astype(int)
    print(f"  Q1 all-PHZ: {len(df):,} sources")
    for v in sorted(df["phz"].unique()):
        print(f"    phz={v:>3}: {(df['phz']==v).sum():>8,}")
    return df


def query_q1_allphz_desi(username: str) -> pd.DataFrame:
    """Query Q1 × DESI cross-match for ALL PHZ classes."""
    import psycopg2
    gaia_cols = ", ".join(
        f"NULL::float AS {col}" if col in DERIVED_COLUMNS else f"g_{col} AS {col}"
        for col in SELECTED_GAIA_PARAMETERS
    )
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    df = pd.read_sql_query(f"""
        SELECT
            g_source_id AS source_id,
            m_right_ascension AS ra,
            m_declination AS dec,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            p_phz_classification AS phz,
            p_phz_gal_prob AS phz_gal_prob,
            d_desi_z,
            d_desi_spectype AS spectype,
            {gaia_cols}
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep_desi
        WHERE p_phz_classification IS NOT NULL
          AND g_phot_g_mean_mag >= {MAG_CUT}
          AND m_right_ascension IS NOT NULL
          AND m_declination IS NOT NULL
        ORDER BY g_source_id
    """, conn)
    conn.close()
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace("NOT_AVAILABLE", np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["phz"] = df["phz"].astype(int)
    print(f"  Q1 all-PHZ × DESI: {len(df):,} sources")
    return df


def query_ero(username: str, test_mode: bool = False) -> tuple:
    """Query ERO full pool with Gaia features.

    Returns (X, y, g_mag, source_ids, ra, dec, class_star).
    """
    import psycopg2
    gaia_cols = ", ".join(f"g_{col} AS {col}" for col in SELECTED_GAIA_PARAMETERS)
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = f"""
        SELECT
            g_source_id,
            {gaia_cols},
            g_phot_g_mean_mag,
            e_v_alpha_j2000 AS ra,
            e_v_delta_j2000 AS dec,
            e_v_class_star AS class_star,
            CASE
                WHEN e_v_class_star >= 0.65 THEN 'STAR'
                WHEN e_v_class_star < 0.65 THEN 'GALAXY'
            END as label
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_class_star IS NOT NULL
            AND g_phot_g_mean_mag >= {MAG_CUT}
        ORDER BY g_random_index
    """
    if test_mode:
        query += " LIMIT 10000"
    print(f"  Querying ERO dataset...")
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
    print(f"  -> {len(X):,} ERO sources — STAR: {n_star:,}, GALAXY: {n_gal:,}")
    return X, y, g_mag, source_ids, ra, dec, class_star


def resample_ero_to_q1_ratio(X_ero, y_ero, source_ids_ero, g_mag_ero,
                              ra_ero, dec_ero, class_star_ero,
                              q1_star_ratio: float,
                              target_total: int = 50_000) -> tuple:
    """Stratified resample of ERO pool at the Q1 star:galaxy ratio."""
    rng = np.random.RandomState(42)
    idx_star = np.where(y_ero == 0)[0]
    idx_gal  = np.where(y_ero == 1)[0]
    n_gal_target  = int(target_total / (1 + q1_star_ratio))
    n_gal         = min(len(idx_gal), n_gal_target)
    n_star_target = min(int(round(n_gal * q1_star_ratio)), len(idx_star))
    idx_gal_s  = rng.choice(idx_gal,  n_gal,         replace=False)
    idx_star_s = rng.choice(idx_star, n_star_target,  replace=False)
    idx_bal    = np.concatenate([idx_star_s, idx_gal_s])
    rng.shuffle(idx_bal)
    return (X_ero[idx_bal], y_ero[idx_bal], source_ids_ero[idx_bal],
            g_mag_ero[idx_bal], ra_ero[idx_bal], dec_ero[idx_bal],
            class_star_ero[idx_bal])


def query_ero_desi(username: str) -> pd.DataFrame:
    """Query ERO×DESI cross-match with full Gaia features."""
    import psycopg2
    gaia_cols = ", ".join(f"g_{col} AS {col}" for col in SELECTED_GAIA_PARAMETERS)
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    df = pd.read_sql_query(f"""
        SELECT {gaia_cols},
               g_source_id AS source_id,
               g_phot_g_mean_mag AS phot_g_mean_mag,
               e_v_alpha_j2000 AS ra,
               e_v_delta_j2000 AS dec,
               d_desi_spectype AS spectype,
               d_desi_z,
               e_v_class_star,
               e_v_flux_radius,
               CASE WHEN e_v_class_star >= 0.65 THEN 2 ELSE 3 END AS cat_value
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3_desi
        WHERE g_phot_g_mean_mag >= {MAG_CUT}
          AND e_v_class_star IS NOT NULL
    """, conn)
    conn.close()
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"  ERO×DESI: {len(df):,} sources")
    return df


def query_gaia_subset(username: str, n_sample: int,
                      mode: str = "all") -> pd.DataFrame:
    """Query a random Gaia DR3 subset with the UMAP/XGBoost feature columns.

    Pulls from the pre-built {username}.gaia_source_hilat / _lolat tables,
    which already contain the derived columns (corrected_bp_rp_excess,
    pm_significance, position_error, bp_g, g_rp) — no SQL recomputation.

      mode="all"    → UNION ALL of gaia_source_hilat + gaia_source_lolat
                       (covers all of Gaia DR3, ~1.65B sources)
      mode="hilat"  → only gaia_source_hilat  (140M sources, high galactic |b|)
      mode="lolat"  → only gaia_source_lolat  (1.5B sources, low galactic |b|)
    """
    import psycopg2

    if mode not in ("all", "hilat", "lolat"):
        raise ValueError(f"mode must be all/hilat/lolat (got {mode!r})")

    SELECT_COLS = """
        source_id,
        random_index,
        ra, dec,
        b, ecl_lat,
        phot_g_mean_mag,
        parallax_over_error, astrometric_gof_al, astrometric_chi2_al,
        astrometric_excess_noise_sig, ipd_gof_harmonic_amplitude, ruwe,
        phot_g_mean_flux_over_error, phot_bp_mean_flux_over_error,
        phot_rp_mean_flux_over_error,
        bp_g, g_rp, corrected_bp_rp_excess,
        pm_significance, position_error
    """

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)

    if mode == "hilat":
        # High-latitude only: hilat is ~140M rows (~7.8% of full Gaia)
        # threshold scaling 15× ≈ ri ~ 750k for 50k sample (fast)
        ri_threshold = int(n_sample * 15)
        query = f"""
            SELECT {SELECT_COLS}
            FROM {username}.gaia_source_hilat
            WHERE random_index < {ri_threshold}
              AND phot_g_mean_mag >= {MAG_CUT}
            ORDER BY random_index
            LIMIT {n_sample}
        """
    elif mode == "lolat":
        # Low-latitude only: lolat is ~1.5B rows (~92% of full Gaia)
        # threshold scaling 3× is enough for a 50k sample
        ri_threshold = int(n_sample * 3)
        query = f"""
            SELECT {SELECT_COLS}
            FROM {username}.gaia_source_lolat
            WHERE random_index < {ri_threshold}
              AND phot_g_mean_mag >= {MAG_CUT}
            ORDER BY random_index
            LIMIT {n_sample}
        """
    else:  # mode == "all"
        # All-sky: UNION ALL of hilat + lolat covers full Gaia DR3.
        # random_index is preserved across both tables, so a global
        # threshold yields a representative all-sky random sample.
        ri_threshold = int(n_sample * 3)
        query = f"""
            SELECT * FROM (
                SELECT {SELECT_COLS}
                FROM {username}.gaia_source_hilat
                WHERE random_index < {ri_threshold}
                  AND phot_g_mean_mag >= {MAG_CUT}
                UNION ALL
                SELECT {SELECT_COLS}
                FROM {username}.gaia_source_lolat
                WHERE random_index < {ri_threshold}
                  AND phot_g_mean_mag >= {MAG_CUT}
            ) AS u
            ORDER BY random_index
            LIMIT {n_sample}
        """

    label = {"all": "all-sky ", "hilat": "high-lat ", "lolat": "low-lat "}[mode]
    print(f"  Querying ~{n_sample:,} {label}Gaia DR3 sources (G >= {MAG_CUT}) "
          f"from pre-built hi/lolat tables...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["source_id"] = df["source_id"].astype(str)
    print(f"  -> {len(df):,} sources")
    return df


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Query 50k Q1 sources, fit UMAP + HDBSCAN, save arrays.")
    parser.add_argument("--test-mode", action="store_true",
                        help="Limit DB query to 10k sources")
    parser.add_argument("--min-cluster-size", type=int, default=150,
                        help="HDBSCAN min_cluster_size (default: 150)")
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Disable default merges/splits/swaps")
    parser.add_argument("--n-sample", type=int, default=50_000,
                        help="Number of random Gaia sources (default: 50000)")
    parser.add_argument("--no-hilat", action="store_true",
                        help="Skip high-latitude Gaia subset")
    parser.add_argument("--no-lolat", action="store_true",
                        help="Skip low-latitude Gaia subset")
    args = parser.parse_args()

    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return 1

    ARRAYS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Q1 UMAP Builder")
    print("=" * 60)
    print(f"  Features: {len(SELECTED_GAIA_PARAMETERS)}")
    print(f"  Magnitude cut: G >= {MAG_CUT}")
    print(f"  Sample size: {SAMPLE_SIZE:,}")
    print(f"  Output: {ARRAYS_DIR}")

    # ── 1. Query ─────────────────────────────────────────────────────────
    print("\n[1] Querying Q1 dataset...")
    X, y, g_mag, source_ids, ra, dec, phz_gal_prob = query_q1(
        username, test_mode=args.test_mode)

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
    phz_gal_prob_sub = phz_gal_prob[idx]
    print(f"  {len(X_scaled):,} sources subsampled")

    # ── 3. Fit UMAP ─────────────────────────────────────────────────────
    print("\n[3] Fitting UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                        random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    print(f"  Embedding shape: {embedding.shape}")

    # ── 4. Compute Q1 model P(gal) via XGB seed ensemble ────────────────
    print("\n[4] Computing Q1 model P(gal) via XGB seed ensemble...")
    q1_model_pgal = np.full(len(X_raw), np.nan)
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
        q1_model_pgal = np.mean(all_probs, axis=0)
        print(f"  XGB ensemble ({len(seed_paths)} seeds) → P(gal) range "
              f"[{q1_model_pgal.min():.3f}, {q1_model_pgal.max():.3f}]")
    else:
        print(f"  WARNING: no seed models in {seed_dir} — model_pgal will be NaN")

    # ── 5. Save npz + reducer ────────────────────────────────────────────
    print("\n[5] Saving arrays...")
    np.savez(ARRAYS_DIR / "q1_indiv.npz",
             embedding=embedding,
             y=y_sub,
             source_ids=np.array(source_ids_sub, dtype=str),
             X_raw=X_raw,
             mean=mean_,
             std=std_,
             ra=ra_sub,
             dec=dec_sub,
             phz_gal_prob=phz_gal_prob_sub,
             g_mag=g_mag_sub,
             model_pgal=q1_model_pgal)
    joblib.dump(reducer, ARRAYS_DIR / "q1_indiv_reducer.joblib")
    print(f"  -> q1_indiv.npz  ({len(embedding):,} sources)")
    print(f"  -> q1_indiv_reducer.joblib")

    # ── 6. HDBSCAN clustering ────────────────────────────────────────────
    print("\n[6] Clustering...")
    labels = cluster_hdbscan(embedding, min_cluster_size=args.min_cluster_size)

    print("\n  Raw HDBSCAN clusters:")
    print_composition(labels, y_sub, embedding)

    # Post-processing
    if args.no_postprocess:
        merges, splits, swaps = [], [], []
    else:
        merges, splits, swaps = DEFAULT_MERGES, DEFAULT_SPLITS, DEFAULT_SWAPS

    if merges or splits or swaps:
        print("\n  Post-processing...")
        labels = postprocess_labels(labels, embedding, merges, splits, swaps)

    print("\n  Final cluster composition:")
    print_composition(labels, y_sub, embedding)

    np.save(ARRAYS_DIR / "q1_cluster_labels.npy", labels)
    print(f"\n  -> q1_cluster_labels.npy")

    # ── 7. Query Q1 all-PHZ ─────────────────────────────────────────────
    print("\n[7] Querying Q1 all-PHZ classes...")
    q1_allphz_df = query_q1_allphz(username)
    np.savez(ARRAYS_DIR / "q1_allphz.npz",
             source_ids=q1_allphz_df["source_id"].values.astype(str),
             ra=q1_allphz_df["ra"].values.astype(float),
             dec=q1_allphz_df["dec"].values.astype(float),
             phot_g_mean_mag=q1_allphz_df["phot_g_mean_mag"].values.astype(float),
             phz=q1_allphz_df["phz"].values.astype(int),
             phz_gal_prob=q1_allphz_df["phz_gal_prob"].values.astype(float),
             X_features=q1_allphz_df[SELECTED_GAIA_PARAMETERS].values.astype(float))
    print(f"  -> q1_allphz.npz  ({len(q1_allphz_df):,} sources)")

    # ── 8. Query Q1×DESI ────────────────────────────────────────────────
    print("\n[8] Querying Q1 × DESI cross-match...")
    q1_desi_df = query_q1_allphz_desi(username)
    np.savez(ARRAYS_DIR / "q1_desi.npz",
             source_ids=q1_desi_df["source_id"].values.astype(str),
             ra=q1_desi_df["ra"].values.astype(float),
             dec=q1_desi_df["dec"].values.astype(float),
             phot_g_mean_mag=q1_desi_df["phot_g_mean_mag"].values.astype(float),
             phz=q1_desi_df["phz"].values.astype(int),
             phz_gal_prob=q1_desi_df["phz_gal_prob"].values.astype(float),
             desi_z=q1_desi_df["d_desi_z"].values.astype(float),
             spectype=q1_desi_df["spectype"].values.astype(str),
             X_features=q1_desi_df[SELECTED_GAIA_PARAMETERS].values.astype(float))
    print(f"  -> q1_desi.npz  ({len(q1_desi_df):,} sources)")

    # ── 9. Query ERO full pool ──────────────────────────────────────────
    print("\n[9] Querying ERO dataset...")
    X_ero, y_ero, g_mag_ero, sids_ero, ra_ero, dec_ero, cs_ero = query_ero(
        username, test_mode=args.test_mode)

    # Subsample to 50k (random, ordered by g_random_index already)
    ero_rng = np.random.RandomState(42)
    ero_n = min(SAMPLE_SIZE, len(X_ero))
    ero_idx = ero_rng.choice(len(X_ero), ero_n, replace=False)

    np.savez(ARRAYS_DIR / "ero_regular.npz",
             X_raw=X_ero[ero_idx],
             y=y_ero[ero_idx],
             source_ids=sids_ero[ero_idx],
             g_mag=g_mag_ero[ero_idx],
             ra=ra_ero[ero_idx],
             dec=dec_ero[ero_idx],
             class_star=cs_ero[ero_idx])
    print(f"  -> ero_regular.npz  ({ero_n:,} sources)")

    # ── 10. ERO rebalanced at Q1 star:galaxy ratio ──────────────────────
    print("\n[10] Resampling ERO at Q1 star:galaxy ratio...")
    q1_star_ratio = (y_sub == 0).sum() / max((y_sub == 1).sum(), 1)
    print(f"  Q1 star:galaxy ratio = {q1_star_ratio:.2f}")
    (X_rebal, y_rebal, sids_rebal, gmag_rebal,
     ra_rebal, dec_rebal, cs_rebal) = resample_ero_to_q1_ratio(
        X_ero, y_ero, sids_ero, g_mag_ero, ra_ero, dec_ero, cs_ero,
        q1_star_ratio)
    np.savez(ARRAYS_DIR / "ero_rebal.npz",
             X_raw=X_rebal,
             y=y_rebal,
             source_ids=sids_rebal,
             g_mag=gmag_rebal,
             ra=ra_rebal,
             dec=dec_rebal,
             class_star=cs_rebal)
    n_star_r = (y_rebal == 0).sum()
    n_gal_r  = (y_rebal == 1).sum()
    print(f"  -> ero_rebal.npz  ({len(y_rebal):,} sources — "
          f"STAR: {n_star_r:,}, GALAXY: {n_gal_r:,})")

    # ── 11. Query ERO×DESI ──────────────────────────────────────────────
    print("\n[11] Querying ERO × DESI cross-match...")
    ero_desi_df = query_ero_desi(username)
    np.savez(ARRAYS_DIR / "ero_desi.npz",
             X_features=ero_desi_df[SELECTED_GAIA_PARAMETERS].values.astype(float),
             source_ids=ero_desi_df["source_id"].values.astype(str),
             phot_g_mean_mag=ero_desi_df["phot_g_mean_mag"].values.astype(float),
             ra=ero_desi_df["ra"].values.astype(float),
             dec=ero_desi_df["dec"].values.astype(float),
             spectype=ero_desi_df["spectype"].values.astype(str),
             desi_z=ero_desi_df["d_desi_z"].values.astype(float),
             class_star=ero_desi_df["e_v_class_star"].values.astype(float),
             flux_radius=ero_desi_df["e_v_flux_radius"].values.astype(float),
             cat_value=ero_desi_df["cat_value"].values)
    print(f"  -> ero_desi.npz  ({len(ero_desi_df):,} sources)")

    # ── 12. Query Gaia random (all-sky) ─────────────────────────────────
    print(f"\n[12] Querying Gaia DR3 random subset (all-sky)...")
    df_gaia = query_gaia_subset(username, args.n_sample, mode="all")
    np.savez(ARRAYS_DIR / "gaia_random.npz",
             source_ids=df_gaia["source_id"].values.astype(str),
             ra=df_gaia["ra"].values.astype(float),
             dec=df_gaia["dec"].values.astype(float),
             phot_g_mean_mag=df_gaia["phot_g_mean_mag"].values.astype(float),
             X_features=df_gaia[SELECTED_GAIA_PARAMETERS].values.astype(float))
    print(f"  -> gaia_random.npz  ({len(df_gaia):,} sources)")

    # ── 13. Query Gaia high-lat ─────────────────────────────────────────
    if not args.no_hilat:
        print(f"\n[13] Querying Gaia DR3 high-lat subset...")
        df_hilat = query_gaia_subset(username, args.n_sample, mode="hilat")
        np.savez(ARRAYS_DIR / "gaia_hilat.npz",
                 source_ids=df_hilat["source_id"].values.astype(str),
                 ra=df_hilat["ra"].values.astype(float),
                 dec=df_hilat["dec"].values.astype(float),
                 phot_g_mean_mag=df_hilat["phot_g_mean_mag"].values.astype(float),
                 X_features=df_hilat[SELECTED_GAIA_PARAMETERS].values.astype(float))
        print(f"  -> gaia_hilat.npz  ({len(df_hilat):,} sources)")

    # ── 14. Query Gaia low-lat ──────────────────────────────────────────
    if not args.no_lolat:
        print(f"\n[14] Querying Gaia DR3 low-lat subset...")
        df_lolat = query_gaia_subset(username, args.n_sample, mode="lolat")
        np.savez(ARRAYS_DIR / "gaia_lolat.npz",
                 source_ids=df_lolat["source_id"].values.astype(str),
                 ra=df_lolat["ra"].values.astype(float),
                 dec=df_lolat["dec"].values.astype(float),
                 phot_g_mean_mag=df_lolat["phot_g_mean_mag"].values.astype(float),
                 X_features=df_lolat[SELECTED_GAIA_PARAMETERS].values.astype(float))
        print(f"  -> gaia_lolat.npz  ({len(df_lolat):,} sources)")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
