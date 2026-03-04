#!/usr/bin/env python3
"""
Standalone UMAP visualization of Gaia feature space for Q1 and ERO datasets.

3-panel plot: Q1 (2 colors), ERO (2 colors), Combined (4 colors).
Output: output/xgboost_combined/combined_umap.png

Usage:
    PGUSER=siwei_zhao_2025 python src/plot_umap_combined.py [--test-mode]
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output' / 'xgboost_combined'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

CLASS_MAPPING = {'STAR': 0, 'GALAXY': 1}

MAG_CUT = 18.5


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
                {gaia_cols},
                CASE
                    WHEN p_phz_classification = 1 THEN 'STAR'
                    WHEN p_phz_classification = 2 THEN 'GALAXY'
                END as label
            FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
            WHERE p_phz_classification IN (1, 2)
                AND g_phot_g_mean_mag >= {MAG_CUT}
            ORDER BY g_source_id
        """
    else:  # ero
        query = f"""
            SELECT
                {gaia_cols},
                CASE
                    WHEN e_v_class_star >= 0.65 THEN 'STAR'
                    WHEN e_v_class_star < 0.65 THEN 'GALAXY'
                END as label
            FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
            WHERE e_v_class_star IS NOT NULL
                AND g_phot_g_mean_mag >= {MAG_CUT}
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

    X = df[SELECTED_GAIA_PARAMETERS].values
    y = df['label'].map(CLASS_MAPPING).values

    n_star = (y == 0).sum()
    n_galaxy = (y == 1).sum()
    print(f"  -> {len(X):,} sources — STAR: {n_star:,}, GALAXY: {n_galaxy:,}")

    return X, y


def _clean_for_umap(X, sample_size, rng):
    """Subsample, fill NaN with median, standardize."""
    if len(X) > sample_size:
        idx = rng.choice(len(X), sample_size, replace=False)
    else:
        idx = np.arange(len(X))

    X_clean = X[idx].copy()
    for col in range(X_clean.shape[1]):
        col_data = X_clean[:, col]
        mask = np.isnan(col_data)
        if mask.any():
            col_data[mask] = np.nanmedian(col_data)
        std = np.nanstd(col_data)
        mean = np.nanmean(col_data)
        if std > 0:
            X_clean[:, col] = (col_data - mean) / std

    return X_clean, idx


def plot_umap_comparison(X_q1, y_q1, X_ero, y_ero, output_dir, sample_size=50000):
    """3-panel UMAP: Q1 (star/galaxy), ERO (star/galaxy), Combined (4 categories)."""
    rng = np.random.RandomState(42)

    # --- Panel 1 & 2: individual datasets ---
    embeddings = []
    for X, y, name in [(X_q1, y_q1, 'Q1'), (X_ero, y_ero, 'ERO')]:
        X_clean, idx = _clean_for_umap(X, sample_size, rng)
        y_sub = y[idx]

        print(f"  Fitting UMAP for {name} ({len(X_clean):,} sources)...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding = reducer.fit_transform(X_clean)
        embeddings.append((embedding, y_sub, name))

    # --- Panel 3: combined dataset with dataset origin tracked ---
    half_sample = sample_size // 2
    X_q1_clean, idx_q1 = _clean_for_umap(X_q1, half_sample, rng)
    X_ero_clean, idx_ero = _clean_for_umap(X_ero, half_sample, rng)
    y_q1_sub = y_q1[idx_q1]
    y_ero_sub = y_ero[idx_ero]

    X_combined = np.vstack([X_q1_clean, X_ero_clean])
    # Category labels: 0=Q1-STAR, 1=Q1-GALAXY, 2=ERO-STAR, 3=ERO-GALAXY
    cat_q1 = np.where(y_q1_sub == 0, 0, 1)
    cat_ero = np.where(y_ero_sub == 0, 2, 3)
    categories = np.concatenate([cat_q1, cat_ero])

    print(f"  Fitting UMAP for Combined ({len(X_combined):,} sources)...")
    reducer_comb = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding_comb = reducer_comb.fit_transform(X_combined)

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    star_galaxy_colors = {0: '#1976D2', 1: '#E53935'}
    label_names = {0: 'STAR', 1: 'GALAXY'}

    # Panels 1-2: individual datasets
    for ax, (embedding, y_sub, dataset_name) in zip(axes[:2], embeddings):
        for label_val in [1, 0]:  # galaxies first
            mask = y_sub == label_val
            count = mask.sum()
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=star_galaxy_colors[label_val],
                       label=f'{label_names[label_val]} (n={count:,})',
                       alpha=0.3, s=5)
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title(f'{dataset_name} Dataset', fontsize=13, fontweight='bold')
        ax.legend(markerscale=3, fontsize=10, loc='upper right')
        ax.grid(alpha=0.3)

    # Panel 3: combined with 4 categories
    ax = axes[2]
    cat_config = [
        (1, 'Q1 GALAXY', '#E53935', 'o'),   # red
        (3, 'ERO GALAXY', '#FF9800', 's'),   # orange
        (0, 'Q1 STAR', '#1976D2', 'o'),      # blue
        (2, 'ERO STAR', '#42A5F5', 's'),     # light blue
    ]
    for cat_val, cat_label, color, marker in cat_config:
        mask = categories == cat_val
        count = mask.sum()
        ax.scatter(embedding_comb[mask, 0], embedding_comb[mask, 1],
                   c=color, label=f'{cat_label} ({count:,})',
                   alpha=0.3, s=5, marker=marker)
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('Combined (Q1 + ERO)', fontsize=13, fontweight='bold')
    ax.legend(markerscale=3, fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)

    plt.suptitle('UMAP Visualization of Gaia Feature Space', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_umap.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='UMAP visualization of Gaia feature space'
    )
    parser.add_argument('--test-mode', action='store_true',
                        help='Use small sample for quick testing (10k sources)')
    args = parser.parse_args()

    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UMAP Visualization of Gaia Feature Space")
    print("=" * 60)
    print(f"  Features: {len(SELECTED_GAIA_PARAMETERS)}")
    print(f"  Magnitude cut: Gaia G >= {MAG_CUT}")
    print(f"  Output: {OUTPUT_DIR}")
    if args.test_mode:
        print(f"  TEST MODE: limited to 10k sources per dataset")

    print("\n[Step 1] Querying datasets...")
    X_q1, y_q1 = query_dataset(username, 'q1', test_mode=args.test_mode)
    X_ero, y_ero = query_dataset(username, 'ero', test_mode=args.test_mode)

    print("\n[Step 2] Generating UMAP comparison...")
    plot_umap_comparison(X_q1, y_q1, X_ero, y_ero, OUTPUT_DIR)

    print("\n  -> combined_umap.png saved")
    print("=" * 60)
    print("UMAP visualization complete!")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
