#!/usr/bin/env python3
"""
Analyze Q1 Magnitude Calculation with Zero-Point Multiplicative Correction (ZPC).

This script:
1. Queries Q1-MER sources with VIS_DET = 1 (VIS detection flag)
2. Calculates magnitudes using ZPC-corrected fluxes
3. Analyzes magnitude distribution across IE, YE, JE, HE bands
4. Creates magnitude distribution plots

Formula: m_AB = -2.5 × log10(ZPC × FLUX_DETECTION_TOTAL) + 23.9

Zero-Point Multiplicative Corrections (ZPC):
- IE (I_E): ZPC = 1.05
- YE (Y_E): ZPC = 0.95
- JE (J_E): ZPC = 0.97
- HE (H_E): ZPC = 0.95

Reference: Table 5: Zero-point multiplicative correction (ZPC) applied to fluxes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import os
import math

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output'

# WSDB Configuration
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

# Zero-Point Multiplicative Corrections
ZPC_CORRECTIONS = {
    'IE': 1.05,   # I_E band
    'YE': 0.95,   # Y_E band
    'JE': 0.97,   # J_E band
    'HE': 0.95,   # H_E band
}

# Reference zero-point for magnitude calculation
ZP_REF = 23.9  # AB magnitude

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


def query_q1_with_vis_det(username):
    """
    Query Q1-MER sources with VIS_DET = 1 and calculate ZPC-corrected magnitudes.

    Only includes sources with:
    - VIS_DET = 1 (VIS detection flag set)
    - FLUX_DETECTION_TOTAL > 0 (valid flux measurement)

    Args:
        username: WSDB username

    Returns:
        DataFrame with columns: band, magnitude, flux, zpc_corrected_flux
    """
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed!")
        return None

    print(f"\n{'='*70}")
    print(f"Q1 Magnitude Analysis with ZPC (VIS_DET=1 sources only)")
    print(f"{'='*70}")

    # Connect to WSDB
    print(f"\nConnecting to WSDB...")
    try:
        conn = psycopg2.connect(
            host=WSDB_HOST,
            database=WSDB_DB,
            user=username
        )
    except Exception as e:
        print(f"ERROR connecting to WSDB: {e}")
        return None

    table_name = f"{username}.crossmatch_q1_mer_phz"

    print(f"\nQuerying {table_name}...")
    print(f"Filter: m_vis_det = 1")
    print(f"Bands: IE, YE, JE, HE (I_E, Y_E, J_E, H_E)")
    print(f"Formula: m_AB = -2.5 × log10(ZPC × FLUX) + {ZP_REF}\n")

    cursor = conn.cursor()

    # Single optimized query for all bands at once
    query = f"""
        SELECT
            m_flux_detection_total as flux,
            -2.5 * LOG(1.05 * m_flux_detection_total) / LN(10) + 23.9 as mag_ie,
            -2.5 * LOG(0.95 * m_flux_detection_total) / LN(10) + 23.9 as mag_ye,
            -2.5 * LOG(0.97 * m_flux_detection_total) / LN(10) + 23.9 as mag_je,
            -2.5 * LOG(0.95 * m_flux_detection_total) / LN(10) + 23.9 as mag_he
        FROM {table_name}
        WHERE m_vis_det = 1
          AND m_flux_detection_total > 0
          AND m_flux_detection_total IS NOT NULL
        LIMIT 100000
    """

    try:
        cursor.execute(query)
        results = cursor.fetchall()
        print(f"Query completed. Retrieved {len(results):,} sources.\n")

        # Parse results into structured format
        magnitudes = {band: [] for band in ZPC_CORRECTIONS.keys()}
        fluxes = []

        for row in results:
            if len(row) < 5:
                continue
            flux = row[0]
            mag_ie = row[1]
            mag_ye = row[2]
            mag_je = row[3]
            mag_he = row[4]

            if flux > 0:
                magnitudes['IE'].append(mag_ie)
                magnitudes['YE'].append(mag_ye)
                magnitudes['JE'].append(mag_je)
                magnitudes['HE'].append(mag_he)
                fluxes.append(flux)

        # Create results dataframe
        df = pd.DataFrame({
            'IE': magnitudes['IE'],
            'YE': magnitudes['YE'],
            'JE': magnitudes['JE'],
            'HE': magnitudes['HE'],
        })

        print(f"{'='*70}")
        print(f"Magnitude Statistics (VIS_DET=1 sources)")
        print(f"{'='*70}\n")

        for band in ['IE', 'YE', 'JE', 'HE']:
            mags = magnitudes[band]
            if mags:
                print(f"{band}-band (ZPC={ZPC_CORRECTIONS[band]}):")
                print(f"  Count: {len(mags):,} sources")
                print(f"  Mean:  {np.mean(mags):7.2f}")
                print(f"  Std:   {np.std(mags):7.2f}")
                print(f"  Min:   {np.min(mags):7.2f}")
                print(f"  Max:   {np.max(mags):7.2f}")
                print()

        cursor.close()
        conn.close()

        return df

    except Exception as e:
        print(f"ERROR executing query: {e}")
        import traceback
        traceback.print_exc()
        cursor.close()
        conn.close()
        return None


def create_magnitude_distributions(df):
    """Create magnitude distribution plots for all bands."""

    print(f"\n{'='*70}")
    print(f"Creating Magnitude Distribution Plots")
    print(f"{'='*70}\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Q1 Magnitude Distribution with ZPC Corrections (VIS_DET=1)',
                 fontsize=16, fontweight='bold')

    bands = ['IE', 'YE', 'JE', 'HE']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (band, ax, color) in enumerate(zip(bands, axes.flatten(), colors)):
        mags = df[band].dropna()

        ax.hist(mags, bins=50, color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Magnitude (AB)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Sources', fontsize=11, fontweight='bold')
        ax.set_title(f'{band}-band (ZPC={ZPC_CORRECTIONS[band]})',
                    fontsize=12, fontweight='bold')

        # Add statistics box
        mean_mag = np.mean(mags)
        std_mag = np.std(mags)
        count = len(mags)

        stats_text = f'N = {count:,}\nμ = {mean_mag:.2f}\nσ = {std_mag:.2f}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'q1_magnitude_with_zpc_distributions.png',
               dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {OUTPUT_DIR / 'q1_magnitude_with_zpc_distributions.png'}")
    plt.close()


def create_magnitude_comparison_plot(df):
    """Create comparison plot showing magnitude differences between ZPC corrections."""

    print(f"Creating ZPC Correction Comparison Plot...\n")

    fig, ax = plt.subplots(figsize=(14, 7))

    bands = ['IE', 'YE', 'JE', 'HE']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for band, color in zip(bands, colors):
        mags = df[band].dropna().values
        # Create magnitude bins for CDF
        sorted_mags = np.sort(mags)
        cumulative = np.arange(1, len(sorted_mags) + 1) / len(sorted_mags)

        ax.plot(sorted_mags, cumulative * 100, label=f'{band} (ZPC={ZPC_CORRECTIONS[band]})',
               linewidth=2.5, color=color, alpha=0.8)

    ax.set_xlabel('Magnitude (AB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Magnitude Distribution: ZPC Band Comparison (VIS_DET=1)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'q1_magnitude_with_zpc_cumulative.png',
               dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {OUTPUT_DIR / 'q1_magnitude_with_zpc_cumulative.png'}")
    plt.close()


def main():
    """Main execution."""
    print("="*70)
    print("Q1 Magnitude Calculation with Zero-Point Multiplicative Correction")
    print("="*70)

    # Get username from environment
    username = os.getenv('PGUSER')
    if not username:
        print("\nERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return

    print(f"\nLogged in as: {username}")
    print(f"WSDB Host: {WSDB_HOST}")
    print(f"Database: {WSDB_DB}")

    # Query data
    print("\n" + "="*70)
    print("Step 1: Query Q1 Magnitudes with ZPC")
    print("="*70)

    df = query_q1_with_vis_det(username)
    if df is None or df.empty:
        print("Failed to query data from WSDB")
        return

    # Create visualizations
    print("\n" + "="*70)
    print("Step 2: Create Visualizations")
    print("="*70)

    create_magnitude_distributions(df)
    create_magnitude_comparison_plot(df)

    # Summary
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)

    print("\nKey Information:")
    print(f"  ZPC Formula: m_AB = -2.5 × log10(ZPC × FLUX) + {ZP_REF}")
    print(f"\n  Zero-Point Multiplicative Corrections:")
    for band, zpc in ZPC_CORRECTIONS.items():
        print(f"    {band}: ZPC = {zpc}")

    print("\nInterpretation:")
    print("  • Different ZPC values calibrate fluxes for each band")
    print("  • Magnitudes account for band-specific efficiency variations")
    print("  • Filter: Only sources with VIS_DET=1 (successful VIS detection)")


if __name__ == '__main__':
    main()
