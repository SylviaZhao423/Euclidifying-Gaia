#!/usr/bin/env python3
"""
UMAP visualization of Gaia feature space for Q1 and ERO datasets.

Visualizes how stars and galaxies cluster in the high-dimensional
Gaia parameter space using UMAP dimensionality reduction.

Usage:
    python src/analyze_umap_visualization.py [--mag-cut 18.0] [--sample-size 50000]
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output' / 'umap_visualization'
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

# Use same features as classifier
sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS as FEATURES


def query_dataset(username, data_source, mag_cut=18.0, sample_size=None):
    """Query dataset with Gaia features and star/galaxy labels."""
    import psycopg2

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)

    columns = ', '.join([f'g.{p}' for p in FEATURES])

    if data_source == 'q1':
        query = f"""
            SELECT
                {columns},
                CASE
                    WHEN e.p_phz_classification = 1 THEN 'STAR'
                    WHEN e.p_phz_classification = 2 THEN 'GALAXY'
                END as label
            FROM gaia_dr3.gaia_source AS g
            INNER JOIN siwei_zhao_2025.crossmatch_q1_mer_phz_gaia AS e
                ON g.source_id = e.g_source_id
            WHERE e.p_phz_classification IN (1, 2)
                AND (-2.5 * LOG(g.phot_g_mean_flux) + 25.69) >= {mag_cut}
            ORDER BY g.random_index
        """
    else:
        query = f"""
            SELECT
                {columns},
                CASE
                    WHEN e.e_v_class_star >= 0.65 THEN 'STAR'
                    WHEN e.e_v_class_star < 0.65 THEN 'GALAXY'
                END as label
            FROM gaia_dr3.gaia_source AS g
            INNER JOIN siwei_zhao_2025.crossmatch_ero_gaia_0_3 AS e
                ON g.source_id = e.g_source_id
            WHERE e.e_v_class_star IS NOT NULL
                AND (-2.5 * LOG(g.phot_g_mean_flux) + 25.69) >= {mag_cut}
            ORDER BY g.random_index
        """

    if sample_size:
        query += f" LIMIT {sample_size}"

    print(f"  Querying {data_source.upper()} dataset...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Convert to numeric
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col].replace('NOT_AVAILABLE', np.nan), errors='coerce')

    print(f"  Retrieved {len(df):,} sources "
          f"(STAR: {(df['label']=='STAR').sum():,}, GALAXY: {(df['label']=='GALAXY').sum():,})")

    return df


def prepare_features(df):
    """Prepare feature matrix for UMAP, handling missing values."""
    X = df[FEATURES].copy()

    # Fill missing values with median
    for col in X.columns:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)

    # Standardize features
    X = (X - X.mean()) / X.std()

    return X.values


def plot_umap(embedding, labels, title, output_path):
    """Plot UMAP embedding colored by star/galaxy label."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create color mapping
    colors = {'STAR': '#1976D2', 'GALAXY': '#E53935'}

    # Plot galaxies first (so stars are on top if they overlap)
    for label in ['GALAXY', 'STAR']:
        mask = labels == label
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=colors[label], label=label, alpha=0.3, s=5)

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(markerscale=3, fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_combined_umap(q1_embedding, q1_labels, ero_embedding, ero_labels, output_path):
    """Plot Q1 and ERO UMAP embeddings side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    colors = {'STAR': '#1976D2', 'GALAXY': '#E53935'}

    for ax, embedding, labels, dataset in [
        (axes[0], q1_embedding, q1_labels, 'Q1'),
        (axes[1], ero_embedding, ero_labels, 'ERO')
    ]:
        for label in ['GALAXY', 'STAR']:
            mask = labels == label
            count = mask.sum()
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=colors[label], label=f'{label} (n={count:,})', alpha=0.3, s=5)

        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title(f'{dataset} Dataset', fontsize=13, fontweight='bold')
        ax.legend(markerscale=3, fontsize=10, loc='upper right')
        ax.grid(alpha=0.3)

    plt.suptitle('UMAP Visualization of Gaia Feature Space', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='UMAP visualization of Gaia features')
    parser.add_argument('--mag-cut', type=float, default=18.0,
                        help='Minimum Gaia G magnitude cut (default: 18.0)')
    parser.add_argument('--sample-size', type=int, default=50000,
                        help='Sample size per dataset for UMAP (default: 50000)')
    args = parser.parse_args()
    mag_cut = args.mag_cut
    sample_size = args.sample_size

    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    output_dir = OUTPUT_DIR / f'mag{mag_cut}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("UMAP Visualization: Q1 vs ERO")
    print(f"Magnitude cut: G >= {mag_cut}")
    print(f"Sample size: {sample_size:,} per dataset")
    print("=" * 70)

    # Query both datasets
    q1_df = query_dataset(username, 'q1', mag_cut, sample_size)
    ero_df = query_dataset(username, 'ero', mag_cut, sample_size)

    # Prepare features
    print("\nPreparing features...")
    q1_X = prepare_features(q1_df)
    ero_X = prepare_features(ero_df)

    # Fit UMAP separately for each dataset
    print("\nFitting UMAP for Q1...")
    reducer_q1 = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    q1_embedding = reducer_q1.fit_transform(q1_X)

    print("Fitting UMAP for ERO...")
    reducer_ero = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    ero_embedding = reducer_ero.fit_transform(ero_X)

    # Generate plots
    print("\nGenerating plots...")
    plot_umap(q1_embedding, q1_df['label'].values,
              f'Q1 Dataset (G >= {mag_cut})', output_dir / 'umap_q1.png')
    plot_umap(ero_embedding, ero_df['label'].values,
              f'ERO Dataset (G >= {mag_cut})', output_dir / 'umap_ero.png')
    plot_combined_umap(q1_embedding, q1_df['label'].values,
                       ero_embedding, ero_df['label'].values,
                       output_dir / 'umap_comparison.png')

    print(f"\nAll results saved to: {output_dir}")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
