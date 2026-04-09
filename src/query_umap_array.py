#!/usr/bin/env python3
"""
DB queries + ALL UMAP fitting for Q1 and ERO datasets. Saves all arrays to disk.
No plotting — run analyze_umap_plot.py to generate plots from saved arrays.

Outputs (under output/umap/arrays/, regardless of faint cut):
  combined.npz, combined_reducer.joblib
  combined_rebalanced.npz, combined_rebalanced_reducer.joblib
  q1_indiv.npz, q1_indiv_reducer.joblib
  ero_indiv.npz, ero_indiv_reducer.joblib
  ero_indiv_rebalanced.npz, ero_indiv_rebalanced_reducer.joblib
  ero_desi.npz
  q1_allphz.npz, q1_allphz_desi.npz

Usage:
    PGUSER=<user> python src/analyze_umap_query.py                  # G >= 17
    PGUSER=<user> python src/analyze_umap_query.py --faint-cut 21   # 17 <= G <= 21
    PGUSER=<user> python src/analyze_umap_query.py --faint-cut 22   # 17 <= G <= 22
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import umap
import joblib
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output' / 'umap'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

CLASS_MAPPING = {'STAR': 0, 'GALAXY': 1}

MAG_CUT       = 17
MAG_CUT_FAINT = 99   # overridden by --faint-cut N


def _mag_range_dir():
    if MAG_CUT_FAINT >= 90:
        return f'{MAG_CUT}\u2264G'
    return f'{MAG_CUT}\u2264G\u2264{MAG_CUT_FAINT}'


def _mag_range():
    if MAG_CUT_FAINT >= 90:
        return f'{MAG_CUT} \u2264 G'
    return f'{MAG_CUT} \u2264 G \u2264 {MAG_CUT_FAINT}'

Q1_N_STAR   = 181_173
Q1_N_GALAXY =  63_513
Q1_STAR_RATIO = Q1_N_STAR / Q1_N_GALAXY


def _resample_ero_to_q1_ratio(X_ero, y_ero, source_ids_ero, g_mag_ero,
                               target_total=50_000):
    """Stratified sample from the full ERO pool at the Q1 star:galaxy ratio.

    Takes as many galaxies as available (up to target_total/(1+ratio)), then
    samples stars to fill the remainder at Q1 ratio, keeping total~target_total.
    This replaces the old approach of downsampling only stars from a small subset.
    """
    rng = np.random.RandomState(42)
    idx_star = np.where(y_ero == 0)[0]
    idx_gal  = np.where(y_ero == 1)[0]

    n_gal_target  = int(target_total / (1 + Q1_STAR_RATIO))
    n_gal         = min(len(idx_gal), n_gal_target)
    n_star_target = min(int(round(n_gal * Q1_STAR_RATIO)), len(idx_star))

    idx_gal_s  = rng.choice(idx_gal,  n_gal,         replace=False)
    idx_star_s = rng.choice(idx_star, n_star_target,  replace=False)
    idx_bal    = np.concatenate([idx_star_s, idx_gal_s])
    rng.shuffle(idx_bal)
    g_mag_bal = g_mag_ero[idx_bal] if g_mag_ero is not None else None
    src_bal   = source_ids_ero[idx_bal] if source_ids_ero is not None else None
    return X_ero[idx_bal], y_ero[idx_bal], src_bal, g_mag_bal


def _lookup_gmag_by_source_ids(all_source_ids, all_g_mag, subset_source_ids):
    """Map source ids back to Gaia G for a sampled subset."""
    if all_source_ids is None or all_g_mag is None or subset_source_ids is None:
        return None
    g_map = {str(sid): g for sid, g in zip(all_source_ids, all_g_mag)}
    return np.array([g_map.get(str(sid), np.nan) for sid in subset_source_ids], dtype=float)


def _lookup_by_source_ids(all_source_ids, all_values, subset_source_ids):
    """Map any per-source array from full set to a subsampled set via source_id lookup."""
    if all_source_ids is None or all_values is None or subset_source_ids is None:
        return None
    lookup = {str(sid): v for sid, v in zip(all_source_ids, all_values)}
    return np.array([lookup.get(str(sid), np.nan) for sid in subset_source_ids], dtype=float)


def query_dataset(username, data_source, test_mode=False):
    """Query dataset. Returns X, y arrays."""
    import psycopg2

    conn = psycopg2.connect(
        host=WSDB_HOST, database=WSDB_DB, user=username
    )

    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])

    if data_source == 'q1':
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
            FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
            WHERE p_phz_classification IN (1, 2)
                AND g_phot_g_mean_mag >= {MAG_CUT}
                AND g_phot_g_mean_mag <= {MAG_CUT_FAINT}
            ORDER BY g_source_id
        """
    else:  # ero
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
                AND g_phot_g_mean_mag <= {MAG_CUT_FAINT}
            ORDER BY g_source_id
        """

    if test_mode:
        query += " LIMIT 10000"

    print(f"  Querying {data_source.upper()} dataset...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Clean features
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    source_ids = df['g_source_id'].astype(str).values
    X = df[SELECTED_GAIA_PARAMETERS].values
    y = df['label'].map(CLASS_MAPPING).values
    g_mag = pd.to_numeric(df['g_phot_g_mean_mag'], errors='coerce').values
    ra  = pd.to_numeric(df['ra'],  errors='coerce').values if 'ra'  in df.columns else None
    dec = pd.to_numeric(df['dec'], errors='coerce').values if 'dec' in df.columns else None

    # Extra column: phz_gal_prob (Q1) or class_star (ERO)
    extra = {}
    if 'phz_gal_prob' in df.columns:
        extra['phz_gal_prob'] = pd.to_numeric(df['phz_gal_prob'], errors='coerce').values
    if 'class_star' in df.columns:
        extra['class_star'] = pd.to_numeric(df['class_star'], errors='coerce').values

    n_star = (y == 0).sum()
    n_galaxy = (y == 1).sum()
    print(f"  -> {len(X):,} sources — STAR: {n_star:,}, GALAXY: {n_galaxy:,}")

    return X, y, g_mag, source_ids, ra, dec, extra


def _clean_for_umap(X, sample_size, rng):
    """Subsample, fill NaN with column median, and per-dataset standardize.
    Returns (X_scaled, idx, mean_, std_)."""
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


def _forced_sample(source_ids, forced_id_set, target_size, rng):
    """Return indices that always include forced IDs, then random fill to target_size."""
    forced_idx = np.where(np.isin(source_ids, list(forced_id_set)))[0]
    remaining   = np.setdiff1d(np.arange(len(source_ids)), forced_idx)
    n_random    = max(0, target_size - len(forced_idx))
    random_idx  = rng.choice(remaining, min(n_random, len(remaining)), replace=False)
    return np.concatenate([forced_idx, random_idx])


def _standardize(X, mean_=None, std_=None):
    """Z-score standardize X. If mean_/std_ not provided, compute from X."""
    if mean_ is None:
        mean_ = np.nanmean(X, axis=0)
    if std_ is None:
        std_ = np.nanstd(X, axis=0)
    std_safe = np.where(std_ > 0, std_, 1.0)
    return (X - mean_) / std_safe, mean_, std_


def _query_ero_desi_features(username):
    """Query ERO×DESI with full Gaia features for UMAP projection.

    Returns DataFrame with SELECTED_GAIA_PARAMETERS + spectype, desi_z, class_star, cat_value.
    cat_value: 2=ERO-Stars, 3=ERO-Galaxies.
    """
    import psycopg2
    conn = psycopg2.connect(host='wsdb.ast.cam.ac.uk', database='wsdb', user=username)
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])

    ero_df = pd.read_sql_query(f"""
        SELECT {gaia_cols},
               g_source_id AS source_id,
               g_phot_g_mean_mag AS phot_g_mean_mag,
               e_v_alpha_j2000 AS ra,
               e_v_delta_j2000 AS dec,
               d_desi_spectype AS spectype,
               d_desi_z,
               e_v_class_star,
               CASE WHEN e_v_class_star >= 0.65 THEN 2 ELSE 3 END AS cat_value
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3_desi
        WHERE d_desi_spectype IN ('GALAXY','STAR','QSO')
          AND d_desi_z IS NOT NULL
          AND g_phot_g_mean_mag >= {MAG_CUT}
          AND g_phot_g_mean_mag <= {MAG_CUT_FAINT}
          AND e_v_class_star IS NOT NULL
    """, conn)
    conn.close()

    for col in SELECTED_GAIA_PARAMETERS:
        ero_df[col] = pd.to_numeric(ero_df[col], errors='coerce')

    print(f"  ERO×DESI: {len(ero_df):,}")
    return ero_df


def _query_q1_allphz(username):
    """Query ALL Q1 PHZ classes (1-7, -1, 0) with Gaia features, RA/Dec, phz_gal_prob.

    Returns a DataFrame with columns: source_id, ra, dec, phot_g_mean_mag,
    phz, phz_gal_prob, and SELECTED_GAIA_PARAMETERS.
    """
    import psycopg2
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
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
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE p_phz_classification IS NOT NULL
          AND g_phot_g_mean_mag >= {MAG_CUT}
          AND m_right_ascension IS NOT NULL
          AND m_declination IS NOT NULL
        ORDER BY g_source_id
    """, conn)
    conn.close()
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['phz'] = df['phz'].astype(int)
    print(f"  Q1 all-PHZ: {len(df):,} sources")
    for v in sorted(df['phz'].unique()):
        print(f"    phz={v:>3}: {(df['phz']==v).sum():>8,}")
    return df


def _query_q1_allphz_desi(username):
    """Query Q1 × DESI cross-match for ALL PHZ classes.

    Returns a DataFrame with: source_id, ra, dec, phot_g_mean_mag,
    phz, phz_gal_prob, desi_z, spectype, and SELECTED_GAIA_PARAMETERS.
    """
    import psycopg2
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
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
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia_desi
        WHERE d_desi_z IS NOT NULL
          AND p_phz_classification IS NOT NULL
          AND g_phot_g_mean_mag >= {MAG_CUT}
          AND m_right_ascension IS NOT NULL
          AND m_declination IS NOT NULL
        ORDER BY g_source_id
    """, conn)
    conn.close()
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['phz'] = df['phz'].astype(int)
    print(f"  Q1 all-PHZ × DESI: {len(df):,} sources")
    return df


def fit_umap_comparison(X_q1, y_q1, X_ero, y_ero, sample_size=50000,
                        g_mag_q1=None, g_mag_ero=None,
                        source_ids_q1=None, source_ids_ero=None,
                        random_state=42, ero_random_state=None):
    """Fit Q1 individual, ERO individual, and combined UMAPs.

    Returns all embeddings, raw arrays, reducers, and normalisation stats
    needed to save npz arrays and project new data.
    """
    if ero_random_state is None:
        ero_random_state = random_state
    rng = np.random.RandomState(random_state)

    # --- Individual UMAP fits (per-dataset standardization, each fit independently) ---
    embeddings = []
    idx_q1_indiv = None
    X_q1_indiv_raw = None
    idx_ero_indiv = None
    X_ero_indiv_raw = None
    q1_indiv_mean = None
    q1_indiv_std  = None
    q1_indiv_reducer = None
    ero_indiv_mean = None
    ero_indiv_std  = None
    ero_indiv_reducer = None
    for X, y, name in [(X_q1, y_q1, 'Q1'), (X_ero, y_ero, 'ERO')]:
        X_clean, idx, mean_, std_ = _clean_for_umap(X, sample_size, rng)
        if name == 'Q1':
            idx_q1_indiv = idx
            X_q1_indiv_raw = X[idx].copy()
            q1_indiv_mean = mean_
            q1_indiv_std  = std_
        else:
            idx_ero_indiv = idx
            X_ero_indiv_raw = X[idx].copy()
            ero_indiv_mean = mean_
            ero_indiv_std  = std_
        y_sub = y[idx]
        seed  = ero_random_state if name == 'ERO' else random_state
        print(f"  Fitting UMAP for {name} ({len(X_clean):,} sources, seed={seed})...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=seed)
        embedding = reducer.fit_transform(X_clean)
        if name == 'Q1':
            q1_indiv_reducer = reducer
        else:
            ero_indiv_reducer = reducer
        embeddings.append((embedding, y_sub, name))

    # --- Combined UMAP fit: per-dataset standardization, shared rng ---
    # Cap Q1 at the ERO size so both datasets contribute equally
    half_sample = min(sample_size // 2, len(X_ero))
    X_q1_clean, idx_q1, q1_mean, q1_std = _clean_for_umap(X_q1, half_sample, rng)
    X_ero_clean, idx_ero, ero_mean, ero_std = _clean_for_umap(X_ero, half_sample, rng)
    X_q1_raw = X_q1[idx_q1].copy()
    X_ero_raw = X_ero[idx_ero].copy()
    y_q1_sub = y_q1[idx_q1]
    y_ero_sub = y_ero[idx_ero]
    n_q1 = len(X_q1_clean)

    X_combined_raw = np.vstack([X_q1_raw, X_ero_raw])
    X_combined = np.vstack([X_q1_clean, X_ero_clean])
    # Category labels: 0=Q1-STAR, 1=Q1-GALAXY, 2=ERO-STAR, 3=ERO-GALAXY
    cat_q1 = np.where(y_q1_sub == 0, 0, 1)
    cat_ero = np.where(y_ero_sub == 0, 2, 3)
    categories = np.concatenate([cat_q1, cat_ero])

    print(f"  Fitting UMAP for Combined ({len(X_combined):,} sources)...")
    reducer_comb = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding_comb = reducer_comb.fit_transform(X_combined)

    emb_q1_comb  = embedding_comb[:n_q1]   # noqa: F841
    emb_ero_comb = embedding_comb[n_q1:]    # noqa: F841

    # Combined G mag (subsampled with same indices)
    if g_mag_q1 is not None and g_mag_ero is not None:
        g_mag_combined = np.concatenate([g_mag_q1[idx_q1], g_mag_ero[idx_ero]])
    else:
        g_mag_combined = None

    source_ids_q1_sub    = source_ids_q1[idx_q1]        if source_ids_q1  is not None else None
    source_ids_ero_sub   = source_ids_ero[idx_ero]       if source_ids_ero is not None else None
    source_ids_ero_indiv = source_ids_ero[idx_ero_indiv] if source_ids_ero is not None else None
    source_ids_q1_indiv  = source_ids_q1[idx_q1_indiv]  if source_ids_q1  is not None else None

    embedding_ero_indiv = embeddings[1][0]
    y_ero_indiv         = embeddings[1][1]
    embedding_q1_indiv  = embeddings[0][0]
    y_q1_indiv          = embeddings[0][1]

    return embedding_comb, X_combined_raw, categories, g_mag_combined, reducer_comb, \
           q1_mean, q1_std, ero_mean, ero_std, source_ids_q1_sub, source_ids_ero_sub, \
           embedding_ero_indiv, y_ero_indiv, source_ids_ero_indiv, X_ero_indiv_raw, \
           embedding_q1_indiv, y_q1_indiv, source_ids_q1_indiv, X_q1_indiv_raw, \
           q1_indiv_reducer, q1_indiv_mean, q1_indiv_std, \
           ero_indiv_reducer, ero_indiv_mean, ero_indiv_std


def main():
    parser = argparse.ArgumentParser(
        description='UMAP fitting of Gaia feature space — queries + array generation only.'
    )
    parser.add_argument('--test-mode', action='store_true',
                        help='Use small sample for quick testing (10k sources)')
    parser.add_argument('--ero-umap-seed', type=int, default=42,
                        help='Random seed for the ERO individual UMAP fit only (default: 42)')
    parser.add_argument('--faint-cut', type=int, default=None, metavar='MAG',
                        help='Upper G-mag limit, e.g. 21, 22. Omit for no faint cut (G >= 17).')
    args = parser.parse_args()

    global MAG_CUT_FAINT
    if args.faint_cut is not None:
        MAG_CUT_FAINT = args.faint_cut

    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return 1

    ARRAYS_DIR = OUTPUT_DIR / 'arrays'
    ARRAYS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UMAP Fitting of Gaia Feature Space")
    print("=" * 60)
    print(f"  Features: {len(SELECTED_GAIA_PARAMETERS)}")
    print(f"  Magnitude cut: {_mag_range()}")
    print(f"  Arrays: {ARRAYS_DIR}")

    if args.test_mode:
        print(f"  TEST MODE: limited to 10k sources per dataset")

    print("\n[Step 1] Querying datasets...")
    X_q1, y_q1, g_mag_q1, source_ids_q1, ra_q1, dec_q1, extra_q1 = \
        query_dataset(username, 'q1', test_mode=args.test_mode)
    X_ero, y_ero, g_mag_ero, source_ids_ero, ra_ero, dec_ero, extra_ero = \
        query_dataset(username, 'ero', test_mode=args.test_mode)

    desi_ero_df = _query_ero_desi_features(username)
    q1_allphz_df     = _query_q1_allphz(username)
    q1_allphz_desi_df = _query_q1_allphz_desi(username)

    print("\n[Step 2] Fitting combined UMAP (per-dataset standardization, shared rng)...")
    embedding_comb, X_raw_comb, categories, g_mag_comb, reducer_comb, \
        q1_mean, q1_std, ero_mean, ero_std, source_ids_q1_sub, source_ids_ero_sub, \
        embedding_ero_indiv, y_ero_indiv, source_ids_ero_indiv, X_ero_indiv_raw, \
        embedding_q1_indiv, y_q1_indiv, source_ids_q1_indiv, X_q1_indiv_raw, \
        q1_indiv_reducer, q1_indiv_mean, q1_indiv_std, \
        ero_indiv_reducer, ero_indiv_mean, ero_indiv_std = \
        fit_umap_comparison(X_q1, y_q1, X_ero, y_ero,
                            g_mag_q1=g_mag_q1, g_mag_ero=g_mag_ero,
                            source_ids_q1=source_ids_q1, source_ids_ero=source_ids_ero,
                            ero_random_state=args.ero_umap_seed)

    # Map extra columns to subsampled source_ids
    _ero_indiv_ra   = _lookup_by_source_ids(source_ids_ero, ra_ero, source_ids_ero_indiv)
    _ero_indiv_dec  = _lookup_by_source_ids(source_ids_ero, dec_ero, source_ids_ero_indiv)
    _ero_indiv_cs   = _lookup_by_source_ids(source_ids_ero, extra_ero.get('class_star'), source_ids_ero_indiv)
    _ero_indiv_gmag = _lookup_by_source_ids(source_ids_ero, g_mag_ero, source_ids_ero_indiv)
    _q1_indiv_ra    = _lookup_by_source_ids(source_ids_q1, ra_q1, source_ids_q1_indiv)
    _q1_indiv_dec   = _lookup_by_source_ids(source_ids_q1, dec_q1, source_ids_q1_indiv)
    _q1_indiv_pgal  = _lookup_by_source_ids(source_ids_q1, extra_q1.get('phz_gal_prob'), source_ids_q1_indiv)
    _q1_indiv_gmag  = _lookup_by_source_ids(source_ids_q1, g_mag_q1, source_ids_q1_indiv)

    np.savez(ARRAYS_DIR / 'ero_indiv.npz',
             embedding=embedding_ero_indiv,
             y=y_ero_indiv,
             source_ids=np.array(source_ids_ero_indiv, dtype=str),
             X_raw=X_ero_indiv_raw,
             mean=ero_indiv_mean,
             std=ero_indiv_std,
             ra=_ero_indiv_ra,
             dec=_ero_indiv_dec,
             class_star=_ero_indiv_cs,
             g_mag=_ero_indiv_gmag,
             mag_cut_faint=MAG_CUT_FAINT)
    joblib.dump(ero_indiv_reducer, ARRAYS_DIR / 'ero_indiv_reducer.joblib')
    np.savez(ARRAYS_DIR / 'q1_indiv.npz',
             embedding=embedding_q1_indiv,
             y=y_q1_indiv,
             source_ids=np.array(source_ids_q1_indiv, dtype=str),
             X_raw=X_q1_indiv_raw,
             mean=q1_indiv_mean,
             std=q1_indiv_std,
             ra=_q1_indiv_ra,
             dec=_q1_indiv_dec,
             phz_gal_prob=_q1_indiv_pgal,
             g_mag=_q1_indiv_gmag,
             mag_cut_faint=MAG_CUT_FAINT)
    joblib.dump(q1_indiv_reducer, ARRAYS_DIR / 'q1_indiv_reducer.joblib')
    print(f"  Saved Q1 individual UMAP reducer -> {ARRAYS_DIR / 'q1_indiv_reducer.joblib'}")
    np.savez(ARRAYS_DIR / 'ero_desi.npz',
             X_features=desi_ero_df[SELECTED_GAIA_PARAMETERS].values.astype(float),
             source_ids=desi_ero_df['source_id'].values.astype(str),
             phot_g_mean_mag=desi_ero_df['phot_g_mean_mag'].values.astype(float),
             ra=desi_ero_df['ra'].values.astype(float),
             dec=desi_ero_df['dec'].values.astype(float),
             spectype=desi_ero_df['spectype'].values.astype(str),
             desi_z=desi_ero_df['d_desi_z'].values.astype(float),
             class_star=desi_ero_df['e_v_class_star'].values.astype(float),
             corrected_bp_rp_excess=desi_ero_df['corrected_bp_rp_excess'].values.astype(float),
             cat_value=desi_ero_df['cat_value'].values,
             mag_cut_faint=MAG_CUT_FAINT)
    print("  -> Individual UMAP arrays saved to output/umap/arrays/")

    # Save regular combined UMAP arrays
    np.savez(ARRAYS_DIR / 'combined.npz',
             embedding=embedding_comb,
             X_raw=X_raw_comb,
             categories=categories,
             g_mag=g_mag_comb if g_mag_comb is not None else np.array([]),
             source_ids_q1=np.array(source_ids_q1_sub, dtype=str),
             source_ids_ero=np.array(source_ids_ero_sub, dtype=str),
             q1_mean=q1_mean, q1_std=q1_std,
             ero_mean=ero_mean, ero_std=ero_std,
             mag_cut_faint=MAG_CUT_FAINT)
    joblib.dump(reducer_comb, ARRAYS_DIR / 'combined_reducer.joblib')
    print(f"  Saved combined UMAP arrays -> {ARRAYS_DIR / 'combined.npz'}")

    print("  -> ERO DESI arrays saved to ero_desi.npz")

    # Save all-PHZ arrays (all classification codes, not just star/galaxy)
    np.savez(ARRAYS_DIR / 'q1_allphz.npz',
             source_ids=q1_allphz_df['source_id'].values.astype(str),
             ra=q1_allphz_df['ra'].values.astype(float),
             dec=q1_allphz_df['dec'].values.astype(float),
             phot_g_mean_mag=q1_allphz_df['phot_g_mean_mag'].values.astype(float),
             phz=q1_allphz_df['phz'].values.astype(int),
             phz_gal_prob=q1_allphz_df['phz_gal_prob'].values.astype(float),
             X_features=q1_allphz_df[SELECTED_GAIA_PARAMETERS].values.astype(float),
             mag_cut_faint=MAG_CUT_FAINT)
    np.savez(ARRAYS_DIR / 'q1_allphz_desi.npz',
             source_ids=q1_allphz_desi_df['source_id'].values.astype(str),
             ra=q1_allphz_desi_df['ra'].values.astype(float),
             dec=q1_allphz_desi_df['dec'].values.astype(float),
             phot_g_mean_mag=q1_allphz_desi_df['phot_g_mean_mag'].values.astype(float),
             phz=q1_allphz_desi_df['phz'].values.astype(int),
             phz_gal_prob=q1_allphz_desi_df['phz_gal_prob'].values.astype(float),
             desi_z=q1_allphz_desi_df['d_desi_z'].values.astype(float),
             spectype=q1_allphz_desi_df['spectype'].values.astype(str),
             X_features=q1_allphz_desi_df[SELECTED_GAIA_PARAMETERS].values.astype(float),
             mag_cut_faint=MAG_CUT_FAINT)
    print("  -> All-PHZ arrays saved to q1_allphz.npz and q1_allphz_desi.npz")

    # Fit rebalanced combined UMAP
    print("\n[Step 3] Stratified ERO sample from full pool at Q1 star:galaxy ratio (~50k total)...")
    X_ero_bal, y_ero_bal, source_ids_ero_bal, _g_mag_ero_bal = \
        _resample_ero_to_q1_ratio(X_ero, y_ero, source_ids_ero, g_mag_ero, target_total=50_000)
    print(f"  Rebalanced ERO: {(y_ero_bal==0).sum():,} stars  {(y_ero_bal==1).sum():,} gal  "
          f"({(y_ero_bal==0).sum()/max((y_ero_bal==1).sum(),1):.2f}:1)")

    print("\n[Step 4] Fitting rebalanced combined UMAP (Q1 + rebalanced ERO)...")
    embedding_comb_rebal, X_raw_comb_rebal, categories_rebal, g_mag_comb_rebal, reducer_comb_rebal, \
        q1_mean_rebal, q1_std_rebal, ero_mean_rebal, ero_std_rebal, \
        source_ids_q1_sub_rebal, source_ids_ero_sub_rebal, \
        embedding_ero_indiv_rebal, y_ero_indiv_rebal, source_ids_ero_indiv_rebal, X_ero_indiv_raw_rebal, \
        _emb_q1_indiv_rebal, _y_q1_indiv_rebal, _src_q1_indiv_rebal, _X_q1_indiv_raw_rebal, \
        _q1_indiv_reducer_rebal, _q1_indiv_mean_rebal, _q1_indiv_std_rebal, \
        ero_indiv_reducer_rebal, ero_indiv_mean_rebal, ero_indiv_std_rebal = \
        fit_umap_comparison(X_q1, y_q1, X_ero_bal, y_ero_bal,
                            g_mag_q1=g_mag_q1, g_mag_ero=_g_mag_ero_bal,
                            source_ids_q1=source_ids_q1, source_ids_ero=source_ids_ero_bal)

    # Save rebalanced combined UMAP arrays
    np.savez(ARRAYS_DIR / 'combined_rebalanced.npz',
             embedding=embedding_comb_rebal,
             X_raw=X_raw_comb_rebal,
             categories=categories_rebal,
             g_mag=g_mag_comb_rebal if g_mag_comb_rebal is not None else np.array([]),
             source_ids_q1=np.array(source_ids_q1_sub_rebal, dtype=str),
             source_ids_ero=np.array(source_ids_ero_sub_rebal, dtype=str),
             q1_mean=q1_mean_rebal, q1_std=q1_std_rebal,
             ero_mean=ero_mean_rebal, ero_std=ero_std_rebal,
             mag_cut_faint=MAG_CUT_FAINT)
    joblib.dump(reducer_comb_rebal, ARRAYS_DIR / 'combined_rebalanced_reducer.joblib')
    print(f"  Saved rebalanced combined UMAP arrays -> {ARRAYS_DIR / 'combined_rebalanced.npz'}")

    # Save ERO individual rebalanced arrays
    _ero_rebal_ra   = _lookup_by_source_ids(source_ids_ero, ra_ero, source_ids_ero_indiv_rebal)
    _ero_rebal_dec  = _lookup_by_source_ids(source_ids_ero, dec_ero, source_ids_ero_indiv_rebal)
    _ero_rebal_cs   = _lookup_by_source_ids(source_ids_ero, extra_ero.get('class_star'), source_ids_ero_indiv_rebal)
    _ero_rebal_gmag = _lookup_by_source_ids(source_ids_ero, g_mag_ero, source_ids_ero_indiv_rebal)
    np.savez(ARRAYS_DIR / 'ero_indiv_rebalanced.npz',
             embedding=embedding_ero_indiv_rebal,
             y=y_ero_indiv_rebal,
             source_ids=np.array(source_ids_ero_indiv_rebal, dtype=str)
                        if source_ids_ero_indiv_rebal is not None else np.array([]),
             X_raw=X_ero_indiv_raw_rebal,
             mean=ero_indiv_mean_rebal,
             mag_cut_faint=MAG_CUT_FAINT,
             g_mag=_ero_rebal_gmag,
             std=ero_indiv_std_rebal,
             ra=_ero_rebal_ra,
             dec=_ero_rebal_dec,
             class_star=_ero_rebal_cs)
    joblib.dump(ero_indiv_reducer_rebal, ARRAYS_DIR / 'ero_indiv_rebalanced_reducer.joblib')
    print(f"  Saved ERO individual rebalanced arrays -> {ARRAYS_DIR / 'ero_indiv_rebalanced.npz'}")

    print("\n  -> All arrays saved. Run analyze_umap_plot.py to generate plots.")
    print("=" * 60)
    print("UMAP fitting complete!")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
