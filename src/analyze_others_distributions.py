#!/usr/bin/env python3
"""
Compare source populations between Q1 and ERO datasets.

Plots normalised histograms of key Gaia properties to identify
underlying differences between the two samples that could affect
classifier performance beyond label quality and class imbalance.

Properties compared:
  - Magnitude: phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag
  - Colour: bp_rp, bp_g, g_rp
  - Sky position: ecliptic latitude (ecl_lat), galactic latitude (b), E(B-V)
  - Astrometric: pm, pm_significance, parallax, parallax_over_error
  - Separated by class (star vs galaxy) within each dataset

Usage:
    python src/analyze_population_comparison.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output' / 'population_comparison' / 'not_features'
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

# Properties to compare
# Contextual/positional parameters NOT in SELECTED_GAIA_PARAMETERS.
# Training features (bp_rp, bp_g, g_rp, corrected_bp_rp_excess,
# pm_significance, parallax_over_error, phot_g_mean_mag) are covered
# by analyze_training_feature_distributions.py instead.
PROPERTIES = [
    # ── Positional / sky (top row, 4th panel intentionally empty) ────────────
    'b', 'ecl_lat', 'ebv', None,
    # ── Magnitudes (redundant with training features) ─────────────────────────
    'phot_bp_mean_mag',
    'phot_rp_mean_mag',
    # ── Astrometry errors (excluded from training) ────────────────────────────
    'ra_error',
    'dec_error',
    # ── Astrometric quality (excluded from training) ──────────────────────────
    'astrometric_n_obs_ac',
    'visibility_periods_used',
    'astrometric_sigma5d_max',
    # ── Image parameters (excluded from training) ─────────────────────────────
    'ipd_frac_multi_peak',
    'ipd_frac_odd_win',
    'duplicated_source',
    # ── G-band photometry (excluded from training) ────────────────────────────
    'phot_g_n_obs',
    'phot_g_mean_flux',
    'phot_g_mean_flux_error',
    # ── BP-band photometry (excluded from training) ───────────────────────────
    'phot_bp_n_obs',
    'phot_bp_mean_flux_error',
    # ── RP-band photometry (excluded from training) ───────────────────────────
    'phot_rp_n_obs',
    'phot_rp_mean_flux_error',
    # ── BP/RP quality (excluded from training) ────────────────────────────────
    'phot_bp_n_blended_transits',
    'phot_rp_n_blended_transits',
    # ── Colour (excluded as redundant) ───────────────────────────────────────
    'bp_rp',
]
N_PROPERTIES = sum(1 for p in PROPERTIES if p is not None)  # excludes the None placeholder


def query_dataset(username, data_source, mag_cut=18.0):
    """Query dataset with key properties for population comparison.

    Queries directly from crossmatch tables (no JOIN with gaia_source).
    Gaia columns are stored with g_ prefix, aliased back to standard names.
    """
    import psycopg2

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)

    # Build column list - Gaia columns have g_ prefix in crossmatch tables
    # (filter out None placeholder used for layout spacing)
    real_props = [p for p in PROPERTIES if p is not None]
    columns = ', '.join([f'g_{p} AS {p}' for p in real_props])

    if data_source == 'q1':
        query = f"""
            SELECT
                {columns},
                CASE
                    WHEN p_phz_classification = 1 THEN 'STAR'
                    WHEN p_phz_classification = 2 THEN 'GALAXY'
                END as label
            FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
            WHERE p_phz_classification IN (1, 2)
            ORDER BY g_source_id
        """
    else:
        query = f"""
            SELECT
                {columns},
                CASE
                    WHEN e_v_class_star >= 0.65 THEN 'STAR'
                    WHEN e_v_class_star < 0.65 THEN 'GALAXY'
                END as label
            FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
            WHERE e_v_class_star IS NOT NULL
            ORDER BY g_source_id
        """

    print(f"  Querying {data_source.upper()} dataset...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Convert to numeric (explicit float64 handles boolean columns like duplicated_source)
    for col in real_props:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace('NOT_AVAILABLE', np.nan),
                                    errors='coerce').astype('float64')

    print(f"  Retrieved {len(df):,} sources "
          f"(STAR: {(df['label']=='STAR').sum():,}, GALAXY: {(df['label']=='GALAXY').sum():,})")

    return df


def plot_combined_comparison(q1_df, ero_df, output_dir):
    """Plot Q1 vs ERO distributions for all sources (not split by class)."""
    n_props = len(PROPERTIES)
    n_cols = 4
    n_rows = (n_props + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4 * n_rows))
    axes = axes.flatten()

    for idx, prop in enumerate(PROPERTIES):
        ax = axes[idx]
        if prop is None:
            ax.axis('off')
            continue
        q1_data = q1_df[prop].dropna()
        ero_data = ero_df[prop].dropna()

        if len(q1_data) == 0 or len(ero_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(prop)
            continue

        # Shared bin range
        lo = min(q1_data.quantile(0.01), ero_data.quantile(0.01))
        hi = max(q1_data.quantile(0.99), ero_data.quantile(0.99))
        bins = np.linspace(lo, hi, 50)

        ax.hist(q1_data, bins=bins, alpha=0.5, label=f'Q1 (n={len(q1_data):,})',
                    color='#BA68C8', density=True)
        ax.hist(ero_data, bins=bins, alpha=0.5, label=f'ERO (n={len(ero_data):,})',
                    color='#66BB6A', density=True)
        ax.set_xlabel(prop, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(prop, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')

    for idx in range(n_props, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Population Comparison: Q1 vs ERO (All Sources, {N_PROPERTIES} parameters)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'ero_vs_q1_all_sources.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'ero_vs_q1_all_sources.png'}")


def plot_by_class_comparison(q1_df, ero_df, output_dir):
    """Plot Q1 vs ERO distributions split by star and galaxy separately."""
    for class_name in ['STAR', 'GALAXY']:
        q1_class = q1_df[q1_df['label'] == class_name]
        ero_class = ero_df[ero_df['label'] == class_name]

        n_props = len(PROPERTIES)
        n_cols = 4
        n_rows = (n_props + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4 * n_rows))
        axes = axes.flatten()

        for idx, prop in enumerate(PROPERTIES):
            ax = axes[idx]
            if prop is None:
                ax.axis('off')
                continue
            q1_data = q1_class[prop].dropna()
            ero_data = ero_class[prop].dropna()

            if len(q1_data) == 0 or len(ero_data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(prop)
                continue

            lo = min(q1_data.quantile(0.01), ero_data.quantile(0.01))
            hi = max(q1_data.quantile(0.99), ero_data.quantile(0.99))
            bins = np.linspace(lo, hi, 50)

            ax.hist(q1_data, bins=bins, alpha=0.5, label=f'Q1 (n={len(q1_data):,})',
                    color='#BA68C8', density=True)
            ax.hist(ero_data, bins=bins, alpha=0.5, label=f'ERO (n={len(ero_data):,})',
                    color='#66BB6A', density=True)
            ax.set_xlabel(prop, fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(prop, fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, axis='y')

        for idx in range(n_props, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Population Comparison: Q1 vs ERO ({class_name}s only, {N_PROPERTIES} parameters)',
                     fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        fname = f'ero_vs_q1_{class_name.lower()}s.png'
        plt.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_dir / fname}")


def plot_star_vs_galaxy_all(q1_df, ero_df, output_dir):
    """Plot star vs galaxy distributions combining both datasets."""
    combined = pd.concat([q1_df, ero_df], ignore_index=True)
    stars = combined[combined['label'] == 'STAR']
    galaxies = combined[combined['label'] == 'GALAXY']

    n_props = len(PROPERTIES)
    n_cols = 4
    n_rows = (n_props + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4 * n_rows))
    axes = axes.flatten()

    for idx, prop in enumerate(PROPERTIES):
        ax = axes[idx]
        if prop is None:
            ax.axis('off')
            continue
        star_data = stars[prop].dropna().astype(float)
        gal_data = galaxies[prop].dropna().astype(float)

        if len(star_data) == 0 or len(gal_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(prop)
            continue

        lo = min(star_data.quantile(0.01), gal_data.quantile(0.01))
        hi = max(star_data.quantile(0.99), gal_data.quantile(0.99))
        bins = np.linspace(lo, hi, 50)

        ax.hist(gal_data, bins=bins, alpha=0.5,
                label=f'GALAXY (n={len(gal_data):,})', color='steelblue', density=True)
        ax.hist(star_data, bins=bins, alpha=0.5,
                label=f'STAR (n={len(star_data):,})', color='coral', density=True)
        ax.set_xlabel(prop, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(prop, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')

    for idx in range(n_props, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Star vs Galaxy Distributions (Q1 + ERO Combined, {N_PROPERTIES} parameters)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'star_vs_galaxy_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'star_vs_galaxy_all.png'}")


def plot_star_vs_galaxy_per_dataset(q1_df, ero_df, output_dir):
    """Plot star vs galaxy distributions within each dataset side by side."""
    for ds_name, df in [('Q1', q1_df), ('ERO', ero_df)]:
        stars = df[df['label'] == 'STAR']
        galaxies = df[df['label'] == 'GALAXY']

        n_props = len(PROPERTIES)
        n_cols = 4
        n_rows = (n_props + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4 * n_rows))
        axes = axes.flatten()

        for idx, prop in enumerate(PROPERTIES):
            ax = axes[idx]
            if prop is None:
                ax.axis('off')
                continue
            star_data = stars[prop].dropna().astype(float)
            gal_data = galaxies[prop].dropna().astype(float)

            if len(star_data) == 0 or len(gal_data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(prop)
                continue

            lo = min(star_data.quantile(0.01), gal_data.quantile(0.01))
            hi = max(star_data.quantile(0.99), gal_data.quantile(0.99))
            bins = np.linspace(lo, hi, 50)

            ax.hist(gal_data, bins=bins, alpha=0.5,
                    label=f'GALAXY (n={len(gal_data):,})', color='steelblue', density=True)
            ax.hist(star_data, bins=bins, alpha=0.5,
                    label=f'STAR (n={len(star_data):,})', color='coral', density=True)
            ax.set_xlabel(prop, fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(prop, fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, axis='y')

        for idx in range(n_props, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'{ds_name} Dataset: Star vs Galaxy Distributions ({N_PROPERTIES} parameters)',
                     fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        fname = f'star_vs_galaxy_{ds_name.lower()}.png'
        plt.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_dir / fname}")



def main():
    import argparse
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Population Comparison: Q1 vs ERO")
    print("=" * 70)

    # Query both datasets
    q1_df = query_dataset(username, 'q1')
    ero_df = query_dataset(username, 'ero')

    # Generate plots
    print("\nPlotting comparisons...")
    plot_combined_comparison(q1_df, ero_df, output_dir)
    plot_by_class_comparison(q1_df, ero_df, output_dir)
    plot_star_vs_galaxy_all(q1_df, ero_df, output_dir)
    plot_star_vs_galaxy_per_dataset(q1_df, ero_df, output_dir)

    print(f"\nAll results saved to: {output_dir}")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
