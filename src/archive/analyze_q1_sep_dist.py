#!/usr/bin/env python3
"""
Plot histogram: Number of Q1 sources vs Angular Separation
for Q1-Gaia crossmatched sources from WSDB.

Plot shows:
- X-axis: Angular separation between Q1 and Gaia positions (arcsec)
- Y-axis: Number of sources in each bin
- Two plots: Linear and Log scale

Queries directly from WSDB crossmatch_q1_mer_gaia table.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output'

# WSDB Configuration
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


def query_q1_gaia_angular_sep(username):
    """Query Q1-Gaia angular separation data from WSDB."""
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
        cursor = conn.cursor()

        # Query Q1-Gaia crossmatch table for angular separations
        query = """
            SELECT
                SQRT(POWER(m_right_ascension - g_ra, 2) + POWER(m_declination - g_dec, 2)) * 3600 as angular_sep_arcsec
            FROM siwei_zhao_2025.crossmatch_q1_mer_gaia
            WHERE m_right_ascension IS NOT NULL
              AND m_declination IS NOT NULL
              AND g_ra IS NOT NULL
              AND g_dec IS NOT NULL
        """

        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        # Convert to array
        angular_seps = np.array([r[0] for r in results])

        print(f"  ✓ Loaded {len(results):,} Q1-Gaia sources")
        print(f"    Angular separation range: {np.min(angular_seps):.6f} - {np.max(angular_seps):.3f} arcsec")
        print(f"    Median: {np.median(angular_seps):.4f} arcsec")
        print(f"    Mean: {np.mean(angular_seps):.4f} arcsec")

        return angular_seps

    except Exception as e:
        print(f"  ✗ Error querying Q1-Gaia data: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main execution."""
    print("=" * 80)
    print("Creating Histogram: Number of Q1 Sources vs Angular Separation (Q1-Gaia)")
    print("=" * 80)

    # Get username from environment
    username = os.getenv('PGUSER')
    if not username:
        print("\nERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return

    print(f"\nLogged in as: {username}")

    # Query Q1-Gaia data
    print("\nQuerying Q1-Gaia crossmatch data...")
    angular_seps = query_q1_gaia_angular_sep(username)
    if angular_seps is None:
        return

    # Create histogram with dual scales
    print("\nCreating histograms...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Get data and limits
    data = angular_seps
    max_sep = np.max(data)
    median_sep = np.median(data)
    mean_sep = np.mean(data)
    x_lim = (0, max_sep * 1.05)

    # Create legend label with statistics
    legend_label = (f'Q1-Gaia (n={len(data):,})\n'
                    f'Max: {max_sep:.4f}″, Median: {median_sep:.4f}″, Mean: {mean_sep:.4f}″')

    # Left plot: Linear scale (zoomed to 0-1 arcsec)
    ax1.hist(data, bins=800, color='steelblue', alpha=0.6, edgecolor='none')
    ax1.set_xlabel('Angular Separation (arcsec)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Count (linear scale)', fontsize=11, fontweight='bold')
    ax1.set_title('Q1-Gaia: Linear Scale (0-1 arcsec)',
                  fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend([legend_label], fontsize=10, loc='upper right')

    # Right plot: Log scale (full range)
    ax2.hist(data, bins=100, color='steelblue', alpha=0.6, edgecolor='none')
    ax2.set_xlabel('Angular Separation (arcsec)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Count (log scale)', fontsize=11, fontweight='bold')
    ax2.set_title('Q1-Gaia: Log Scale (full range)',
                  fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.set_xlim(x_lim)
    ax2.grid(True, alpha=0.3)
    ax2.legend([legend_label], fontsize=10, loc='upper right')

    # Add overall title
    fig.suptitle('Q1-Gaia Crossmatch - Angular Separation Distribution',
                 fontsize=13, fontweight='bold', y=1.00)

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path = OUTPUT_DIR / 'q1_angular_sep_dist.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {output_path}")

    plt.close()

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()
