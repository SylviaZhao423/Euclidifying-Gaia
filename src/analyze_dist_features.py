#!/usr/bin/env python3
"""
Population distribution comparison of the 28 XGBoost training features
across Q1 and ERO datasets, split by class label.

Produces 3 figures (all sources / galaxies only / stars only), each
showing density histograms for all 28 SELECTED_GAIA_PARAMETERS with
Q1 (pink) vs ERO (green), using the same magnitude cut and query
structure as train_xgboost_combined.py.

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_training_feature_distributions.py
"""

import math
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'output' / 'feature_distribution'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'
MAG_CUT   = 17   # same as train_xgboost_combined.py

Q1_COLOR  = '#BA68C8'   # pink  — matches existing population comparison plots
ERO_COLOR = '#66BB6A'   # green


# ---------------------------------------------------------------------------
# Data query
# ---------------------------------------------------------------------------

def query_dataset(username, data_source):
    """Query the 28 training features + class label for Q1 or ERO."""
    import psycopg2

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)

    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])

    if data_source == 'q1':
        query = f"""
            SELECT
                {gaia_cols},
                CASE
                    WHEN p_phz_classification = 1 THEN 'STAR'
                    WHEN p_phz_classification = 2 THEN 'GALAXY'
                END AS label
            FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
            WHERE p_phz_classification IN (1, 2)
              AND g_phot_g_mean_mag >= {MAG_CUT}
            ORDER BY g_source_id
        """
    else:
        query = f"""
            SELECT
                {gaia_cols},
                CASE
                    WHEN e_v_class_star >= 0.65 THEN 'STAR'
                    WHEN e_v_class_star <  0.65 THEN 'GALAXY'
                END AS label
            FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
            WHERE e_v_class_star IS NOT NULL
              AND g_phot_g_mean_mag >= {MAG_CUT}
            ORDER BY g_source_id
        """

    print(f"  Querying {data_source.upper()} dataset...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    n_star = (df['label'] == 'STAR').sum()
    n_gal  = (df['label'] == 'GALAXY').sum()
    print(f"  {data_source.upper()}: {len(df):,} sources  "
          f"(STAR: {n_star:,}  GALAXY: {n_gal:,})")
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_grid(q1_sub, ero_sub, title, output_path):
    """Shared helper: density histogram grid, Q1 vs ERO."""
    n_cols = 4
    n_rows = math.ceil(len(SELECTED_GAIA_PARAMETERS) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4 * n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(SELECTED_GAIA_PARAMETERS):
        ax = axes[idx]
        q1_data  = q1_sub[col].dropna()
        ero_data = ero_sub[col].dropna()

        if len(q1_data) == 0 and len(ero_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(col, fontsize=10, fontweight='bold')
            continue

        # Shared bin range from combined 1st–99th percentile
        all_data = pd.concat([q1_data, ero_data])
        lo = all_data.quantile(0.01)
        hi = all_data.quantile(0.99)
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        bins = np.linspace(lo, hi, 50)

        if len(q1_data) > 0:
            ax.hist(q1_data,  bins=bins, alpha=0.5, density=True,
                    color=Q1_COLOR,  label=f'Q1 (n={len(q1_data):,})')
        if len(ero_data) > 0:
            ax.hist(ero_data, bins=bins, alpha=0.5, density=True,
                    color=ERO_COLOR, label=f'ERO (n={len(ero_data):,})')

        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(col, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')

    # Hide any unused panels
    for idx in range(len(SELECTED_GAIA_PARAMETERS), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'{title}  ({len(SELECTED_GAIA_PARAMETERS)} features)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")



def plot_all_sources(q1_df, ero_df, output_dir):
    _plot_grid(
        q1_df, ero_df,
        title=f'Training Feature Distributions: Q1 vs ERO (All Sources, G ≥ {MAG_CUT})',
        output_path=output_dir / 'ero_vs_q1_all.png',
    )


def plot_by_class(q1_df, ero_df, output_dir):
    for class_label, fname_label, title_label in [
        ('GALAXY', 'galaxies', 'Galaxies only'),
        ('STAR',   'stars',    'Stars only'),
    ]:
        _plot_grid(
            q1_df[q1_df['label'] == class_label],
            ero_df[ero_df['label'] == class_label],
            title=f'Training Feature Distributions: Q1 vs ERO ({title_label}, G ≥ {MAG_CUT})',
            output_path=output_dir / f'ero_vs_q1_{fname_label}.png',
        )


# ---------------------------------------------------------------------------
# Star vs galaxy helpers
# ---------------------------------------------------------------------------

STAR_COLOR   = 'coral'
GALAXY_COLOR = 'steelblue'


def _plot_grid_sv(star_sub, gal_sub, title, output_path):
    """Density histogram grid: STAR vs GALAXY."""
    n_cols = 4
    n_rows = math.ceil(len(SELECTED_GAIA_PARAMETERS) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4 * n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(SELECTED_GAIA_PARAMETERS):
        ax = axes[idx]
        star_data = star_sub[col].dropna()
        gal_data  = gal_sub[col].dropna()

        if len(star_data) == 0 and len(gal_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(col, fontsize=10, fontweight='bold')
            continue

        all_data = pd.concat([star_data, gal_data])
        lo = all_data.quantile(0.01)
        hi = all_data.quantile(0.99)
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        bins = np.linspace(lo, hi, 50)

        if len(gal_data) > 0:
            ax.hist(gal_data,  bins=bins, alpha=0.5, density=True,
                    color=GALAXY_COLOR, label=f'GALAXY (n={len(gal_data):,})')
        if len(star_data) > 0:
            ax.hist(star_data, bins=bins, alpha=0.5, density=True,
                    color=STAR_COLOR,   label=f'STAR (n={len(star_data):,})')

        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(col, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')

    for idx in range(len(SELECTED_GAIA_PARAMETERS), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'{title}  ({len(SELECTED_GAIA_PARAMETERS)} features)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_star_vs_galaxy(q1_df, ero_df, output_dir):
    """Star vs galaxy feature distributions for combined, Q1-only, and ERO-only."""
    for ds_label, df in [
        ('Q1+ERO combined', pd.concat([q1_df, ero_df], ignore_index=True)),
        ('Q1',              q1_df),
        ('ERO',             ero_df),
    ]:
        fname = {
            'Q1+ERO combined': 'star_vs_galaxy_all',
            'Q1':              'star_vs_galaxy_q1',
            'ERO':             'star_vs_galaxy_ero',
        }[ds_label]

        _plot_grid_sv(
            star_sub=df[df['label'] == 'STAR'],
            gal_sub =df[df['label'] == 'GALAXY'],
            title   =f'Feature Distributions: STAR vs GALAXY — {ds_label} (G ≥ {MAG_CUT})',
            output_path=output_dir / f'{fname}.png',
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Training feature distributions: Q1 vs ERO  "
          f"(G ≥ {MAG_CUT},  {len(SELECTED_GAIA_PARAMETERS)} features)")
    print("=" * 70)

    print("\n[1] Querying data...")
    q1_df  = query_dataset(username, 'q1')
    ero_df = query_dataset(username, 'ero')

    print("\n[2] All sources...")
    plot_all_sources(q1_df, ero_df, OUTPUT_DIR)

    print("\n[3] Split by class (density + fraction)...")
    plot_by_class(q1_df, ero_df, OUTPUT_DIR)

    print("\n[4] Star vs galaxy (combined, Q1, ERO)...")
    plot_star_vs_galaxy(q1_df, ero_df, OUTPUT_DIR)

    print(f"\nAll saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
