#!/usr/bin/env python3
"""
Plot feature distributions for stars vs galaxies.
Helps visualize which Gaia parameters best discriminate between the two classes.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output' / 'feature_distributions'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'


def query_sample_data(username, data_source='q1', sample_size=10000):
    """Query sample of labeled data for visualization."""
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed!")
        return None

    try:
        conn = psycopg2.connect(
            host=WSDB_HOST,
            database=WSDB_DB,
            user=username
        )

        if data_source == 'q1':
            query = f"""
                SELECT
                    g.{', g.'.join(SELECTED_GAIA_PARAMETERS)},
                    CASE
                        WHEN e.p_phz_classification = 1 THEN 'STAR'
                        WHEN e.p_phz_classification = 2 THEN 'GALAXY'
                    END as label
                FROM gaia_dr3.gaia_source AS g
                INNER JOIN siwei_zhao_2025.crossmatch_q1_mer_phz_gaia AS e
                    ON g.source_id = e.g_source_id
                WHERE e.p_phz_classification IN (1, 2)
                    AND (-2.5 * LOG(g.phot_g_mean_flux) + 25.69) >= 18
                LIMIT {sample_size}
            """
        else:  # ero
            query = f"""
                SELECT
                    g.{', g.'.join(SELECTED_GAIA_PARAMETERS)},
                    CASE
                        WHEN e.e_v_class_star >= 0.65 THEN 'STAR'
                        WHEN e.e_v_class_star < 0.65 THEN 'GALAXY'
                    END as label
                FROM gaia_dr3.gaia_source AS g
                INNER JOIN siwei_zhao_2025.crossmatch_ero_gaia_0_3 AS e
                    ON g.source_id = e.g_source_id
                WHERE e.e_v_class_star IS NOT NULL
                    AND (-2.5 * LOG(g.phot_g_mean_flux) + 25.69) >= 18
                LIMIT {sample_size}
            """

        print(f"Querying {sample_size:,} sources from {data_source.upper()}...")
        df = pd.read_sql_query(query, conn)
        conn.close()

        # Clean data
        df_features = df[SELECTED_GAIA_PARAMETERS].copy()
        df_features = df_features.replace('NOT_AVAILABLE', np.nan)
        for col in SELECTED_GAIA_PARAMETERS:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')

        df_clean = pd.concat([df_features, df['label']], axis=1)

        print(f"✓ Retrieved {len(df):,} sources")
        print(f"  STAR: {(df['label']=='STAR').sum():,}")
        print(f"  GALAXY: {(df['label']=='GALAXY').sum():,}")

        return df_clean

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_feature_distribution(df, feature, data_source, output_dir):
    """Plot histogram comparing stars vs galaxies for a single feature."""
    # Remove NaN values
    df_clean = df[[feature, 'label']].dropna()

    if len(df_clean) == 0:
        print(f"  ⚠ Skipping {feature}: no valid data")
        return

    stars = df_clean[df_clean['label'] == 'STAR'][feature]
    galaxies = df_clean[df_clean['label'] == 'GALAXY'][feature]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot overlapping histograms
    bins = 50
    ax.hist(stars, bins=bins, alpha=0.6, label=f'STAR (n={len(stars):,})',
            color='steelblue', density=True)
    ax.hist(galaxies, bins=bins, alpha=0.6, label=f'GALAXY (n={len(galaxies):,})',
            color='coral', density=True)

    ax.set_xlabel(feature, fontsize=11, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax.set_title(f'{feature} Distribution: Stars vs Galaxies ({data_source.upper()})',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / f'{data_source}_{feature}_distribution.png', dpi=150)
    plt.close()


def plot_top_features_grid(df, top_features, data_source, output_dir):
    """Plot grid of top N features for comparison."""
    n_features = len(top_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for idx, feature in enumerate(top_features):
        ax = axes[idx]

        # Remove NaN
        df_clean = df[[feature, 'label']].dropna()
        if len(df_clean) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(feature)
            continue

        stars = df_clean[df_clean['label'] == 'STAR'][feature]
        galaxies = df_clean[df_clean['label'] == 'GALAXY'][feature]

        # Plot
        bins = 30
        ax.hist(stars, bins=bins, alpha=0.6, label='STAR', color='steelblue', density=True)
        ax.hist(galaxies, bins=bins, alpha=0.6, label='GALAXY', color='coral', density=True)

        ax.set_xlabel(feature, fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(feature, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')

    # Hide unused subplots
    for idx in range(len(top_features), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Top {len(top_features)} Feature Distributions ({data_source.upper()})',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / f'{data_source}_top_features_grid.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot feature distributions for stars vs galaxies')
    parser.add_argument('--data-source', type=str, default='q1', choices=['q1', 'ero'],
                        help='Data source to use')
    parser.add_argument('--sample-size', type=int, default=10000,
                        help='Number of sources to sample')
    parser.add_argument('--top-n', type=int, default=12,
                        help='Number of top features to plot in grid')
    args = parser.parse_args()

    # Get username
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    print("=" * 80)
    print(f"Feature Distribution Analysis: {args.data_source.upper()}")
    print("=" * 80)

    # Query data
    df = query_sample_data(username, args.data_source, args.sample_size)
    if df is None:
        return 1

    # Load feature importance to identify top features
    importance_file = PROJECT_DIR / 'output' / f'xgboost_{args.data_source}' / f'{args.data_source}_feature_importance.csv'
    if importance_file.exists():
        importance_df = pd.read_csv(importance_file)
        top_features = importance_df.head(args.top_n)['feature'].tolist()
        print(f"\nPlotting top {args.top_n} features by importance:")
        for i, feat in enumerate(top_features, 1):
            imp = importance_df[importance_df['feature'] == feat]['importance'].values[0]
            print(f"  {i:2d}. {feat:40s} ({imp:.4f})")
    else:
        print(f"\n⚠ No feature importance file found, using first {args.top_n} features")
        top_features = SELECTED_GAIA_PARAMETERS[:args.top_n]

    # Create grid plot of top features
    print(f"\nGenerating grid plot...")
    plot_top_features_grid(df, top_features, args.data_source, OUTPUT_DIR)
    print(f"✓ Saved: {OUTPUT_DIR}/{args.data_source}_top_features_grid.png")

    # Create individual plots for top features
    print(f"\nGenerating individual plots for top {args.top_n} features...")
    for feature in top_features:
        plot_feature_distribution(df, feature, args.data_source, OUTPUT_DIR)
    print(f"✓ Saved {args.top_n} individual plots to {OUTPUT_DIR}/")

    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
