#!/usr/bin/env python3
"""
Compare Gaia observation count parameters between Q1 and ERO datasets.

Tests Diana's hypothesis that Gaia's coverage depends on ecliptic latitude,
which could affect data quality differently between Q1 (high ecl_lat) and
ERO (low ecl_lat) fields.

Parameters compared:
  - astrometric_n_obs_al, astrometric_n_obs_ac
  - astrometric_n_good_obs_al, astrometric_n_bad_obs_al
  - phot_g_n_obs, phot_bp_n_obs, phot_rp_n_obs
  - visibility_periods_used

Usage:
    python src/analyze_obs_count_comparison.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output' / 'obs_count_comparison'
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

# Observation count parameters to compare
PROPERTIES = [
    'astrometric_n_obs_al',
    'astrometric_n_obs_ac',
    'astrometric_n_good_obs_al',
    'astrometric_n_bad_obs_al',
    'phot_g_n_obs',
    'phot_bp_n_obs',
    'phot_rp_n_obs',
    'visibility_periods_used',
]

PROPERTY_LABELS = {
    'astrometric_n_obs_al': 'astrometric_n_obs_al',
    'astrometric_n_obs_ac': 'astrometric_n_obs_ac',
    'astrometric_n_good_obs_al': 'astrometric_n_good_obs_al',
    'astrometric_n_bad_obs_al': 'astrometric_n_bad_obs_al',
    'phot_g_n_obs': 'phot_g_n_obs',
    'phot_bp_n_obs': 'phot_bp_n_obs',
    'phot_rp_n_obs': 'phot_rp_n_obs',
    'visibility_periods_used': 'visibility_periods_used',
}


def query_dataset(username, data_source, mag_cut=18.0):
    """Query dataset with observation count parameters."""
    import psycopg2

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)

    columns = ', '.join([f'g.{p}' for p in PROPERTIES])

    if data_source == 'q1':
        query = f"""
            SELECT
                {columns},
                g.ecl_lat,
                CASE
                    WHEN e.p_phz_classification = 1 THEN 'STAR'
                    WHEN e.p_phz_classification = 2 THEN 'GALAXY'
                END as label
            FROM gaia_dr3.gaia_source AS g
            INNER JOIN siwei_zhao_2025.crossmatch_q1_mer_phz_gaia AS e
                ON g.source_id = e.g_source_id
            WHERE e.p_phz_classification IN (1, 2)
                AND (-2.5 * LOG(g.phot_g_mean_flux) + 25.69) >= {mag_cut}
            ORDER BY g.source_id
        """
    else:
        query = f"""
            SELECT
                {columns},
                g.ecl_lat,
                CASE
                    WHEN e.e_v_class_star >= 0.65 THEN 'STAR'
                    WHEN e.e_v_class_star < 0.65 THEN 'GALAXY'
                END as label
            FROM gaia_dr3.gaia_source AS g
            INNER JOIN siwei_zhao_2025.crossmatch_ero_gaia_0_3 AS e
                ON g.source_id = e.g_source_id
            WHERE e.e_v_class_star IS NOT NULL
                AND (-2.5 * LOG(g.phot_g_mean_flux) + 25.69) >= {mag_cut}
            ORDER BY g.source_id
        """

    print(f"  Querying {data_source.upper()} dataset...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Convert to numeric
    for col in PROPERTIES + ['ecl_lat']:
        df[col] = pd.to_numeric(df[col].replace('NOT_AVAILABLE', np.nan), errors='coerce')

    print(f"  Retrieved {len(df):,} sources "
          f"(STAR: {(df['label']=='STAR').sum():,}, GALAXY: {(df['label']=='GALAXY').sum():,})")
    print(f"  Mean ecl_lat: {df['ecl_lat'].mean():.1f}°")

    return df


def plot_combined_comparison(q1_df, ero_df, output_dir):
    """Plot Q1 vs ERO distributions for observation count parameters."""
    n_props = len(PROPERTIES)
    n_cols = 4
    n_rows = (n_props + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    for idx, prop in enumerate(PROPERTIES):
        ax = axes[idx]
        q1_data = q1_df[prop].dropna()
        ero_data = ero_df[prop].dropna()

        if len(q1_data) == 0 or len(ero_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(PROPERTY_LABELS.get(prop, prop))
            continue

        # Shared bin range
        lo = min(q1_data.quantile(0.01), ero_data.quantile(0.01))
        hi = max(q1_data.quantile(0.99), ero_data.quantile(0.99))
        bins = np.linspace(lo, hi, 50)

        ax.hist(q1_data, bins=bins, alpha=0.5,
                label=f'Q1 (n={len(q1_data):,}, med={q1_data.median():.0f})',
                color='#BA68C8', density=True)
        ax.hist(ero_data, bins=bins, alpha=0.5,
                label=f'ERO (n={len(ero_data):,}, med={ero_data.median():.0f})',
                color='#66BB6A', density=True)
        ax.set_xlabel(PROPERTY_LABELS.get(prop, prop), fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(PROPERTY_LABELS.get(prop, prop), fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')

    for idx in range(n_props, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Gaia Observation Counts: Q1 vs ERO',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'obs_count_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'obs_count_comparison.png'}")


def save_summary_stats(q1_df, ero_df, output_dir, mag_cut=18.0):
    """Save summary statistics comparing both datasets."""
    summary_path = output_dir / 'obs_count_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("OBSERVATION COUNT COMPARISON: Q1 vs ERO\n")
        f.write("=" * 70 + "\n")
        f.write(f"(Gaia G >= {mag_cut} mag filter applied)\n\n")

        f.write(f"Q1:  mean ecl_lat = {q1_df['ecl_lat'].mean():.1f}°, n = {len(q1_df):,}\n")
        f.write(f"ERO: mean ecl_lat = {ero_df['ecl_lat'].mean():.1f}°, n = {len(ero_df):,}\n\n")

        f.write(f"{'Parameter':<30} {'Q1 median':>12} {'ERO median':>12} {'Diff':>10} {'Diff %':>10}\n")
        f.write("-" * 76 + "\n")

        for prop in PROPERTIES:
            q1_med = q1_df[prop].median()
            ero_med = ero_df[prop].median()
            diff = q1_med - ero_med
            diff_pct = 100 * diff / ero_med if ero_med != 0 else 0
            f.write(f"{prop:<30} {q1_med:>12.1f} {ero_med:>12.1f} {diff:>+10.1f} {diff_pct:>+9.1f}%\n")

    print(f"  Saved: {summary_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Observation count comparison: Q1 vs ERO')
    parser.add_argument('--mag-cut', type=float, default=18.0,
                        help='Minimum Gaia G magnitude cut (default: 18.0)')
    args = parser.parse_args()
    mag_cut = args.mag_cut

    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    output_dir = OUTPUT_DIR / f'mag{mag_cut}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Observation Count Comparison: Q1 vs ERO")
    print(f"Magnitude cut: G >= {mag_cut}")
    print("=" * 70)

    # Query both datasets
    q1_df = query_dataset(username, 'q1', mag_cut)
    ero_df = query_dataset(username, 'ero', mag_cut)

    # Generate plots
    print("\nPlotting comparisons...")
    plot_combined_comparison(q1_df, ero_df, output_dir)
    save_summary_stats(q1_df, ero_df, output_dir, mag_cut)

    print(f"\nAll results saved to: {output_dir}")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
