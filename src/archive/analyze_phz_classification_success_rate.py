#!/usr/bin/env python3
"""
Analyze PHZ Classification Success Rate vs Magnitude.

This script:
1. Queries Q1-Gaia-PHZ crossmatch directly from WSDB by magnitude bins
2. Identifies CLASSIFIED vs UNCLASSIFIED/OTHER sources
3. Calculates classification success rate (% successfully classified) per magnitude bin
4. Identifies magnitude ranges where classification degrades
5. Creates visualization showing completeness as function of magnitude

PHZ_CLASSIFICATION Values:
- 1 (I1): STAR - Successfully classified
- 2 (I2): GALAXY - Successfully classified
- 4 (I4): QSO - Successfully classified
- 8 (I8): GLOBULAR CLUSTER - Successfully classified
- -1, 0, 3, 5, 6, 7: OTHER/Multi-class/Unclassified - FAILED

Classification Success Rate = (STAR + GALAXY + QSO + GLOBULAR) / (All sources)

Magnitude: MER detection flux (FLUX_DETECTION_TOTAL from IE detection mosaic, ZP=23.9 AB)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import os

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output'

# WSDB Configuration
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


def calculate_binomial_uncertainty(successes, trials):
    """
    Calculate binomial uncertainty on a fraction.

    This represents the statistical uncertainty due to the finite number of
    sources in the bin. It's the standard error of a binomial proportion.

    uncertainty = sqrt(p * (1 - p) / n)
    where p = fraction and n = number of trials
    """
    if trials == 0:
        return 0, 0, 0

    proportion = successes / trials

    # Binomial uncertainty (1-sigma)
    uncertainty = np.sqrt(proportion * (1 - proportion) / trials)

    lower = max(0, proportion - uncertainty)
    upper = min(1, proportion + uncertainty)

    return proportion, lower, upper


def query_phz_classification_success_rate(username):
    """
    Query PHZ classification success rate by magnitude bins.

    Compares number of successfully classified sources (I1, I2, I4, I8) vs
    total sources (including unclassified/OTHER) in each magnitude bin.

    Args:
        username: WSDB username

    Returns:
        tuple of (bin_centers, success_fraction, success_fraction_lower,
                  success_fraction_upper, classified_counts, total_counts)
    """
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed!")
        return None

    print(f"\n{'='*70}")
    print(f"PHZ Classification Success Rate Analysis")
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

    # Define magnitude bins (0.5 magnitude bins)
    mag_bins = np.arange(6, 28, 0.5)

    print(f"\nQuerying classification success rate from {table_name}...")
    print(f"Processing {len(mag_bins)-1} bins from mag {mag_bins[0]} to {mag_bins[-1]}")
    print(f"Magnitude column: MER detection flux (FLUX_DETECTION_TOTAL, ZP=23.9 AB)")
    print(f"Classification column: p_phz_classification (PHZ pipeline)\n")

    cursor = conn.cursor()

    # Single optimized query: count classified vs total for each magnitude bin
    # Magnitude computed from MER detection flux: mag = -2.5 * log10(flux) + 23.9
    query = f"""
        SELECT
            FLOOR((-2.5 * LOG(m_flux_detection_total) / LN(10) + 23.9 - {mag_bins[0]}) / 0.5) as bin_index,
            {mag_bins[0]} + (FLOOR((-2.5 * LOG(m_flux_detection_total) / LN(10) + 23.9 - {mag_bins[0]}) / 0.5) * 0.5) + 0.25 as bin_center,
            COUNT(*) as total_count,
            COUNT(CASE WHEN p_phz_classification IN (1, 2, 4, 8) THEN 1 END) as classified_count
        FROM {table_name}
        WHERE m_flux_detection_total > 0
          AND (-2.5 * LOG(m_flux_detection_total) / LN(10) + 23.9) >= {mag_bins[0]}
          AND (-2.5 * LOG(m_flux_detection_total) / LN(10) + 23.9) < {mag_bins[-1]}
          AND m_flux_detection_total IS NOT NULL
        GROUP BY bin_index, bin_center
        ORDER BY bin_index
    """

    try:
        cursor.execute(query)
        results = cursor.fetchall()
        print(f"Query completed. Processing {len(results)} magnitude bins...\n")

        # Parse results
        success_fraction = []
        success_fraction_lower = []
        success_fraction_upper = []
        classified_counts = []
        total_counts = []
        bin_centers = []

        print(f"{'Mag Range':<20} {'Total':<10} {'Classified':<12} {'Success Rate':<15}")
        print("-" * 70)

        for row in results:
            bin_idx, bin_center, total_count, classified_count = row

            if total_count > 0:
                # Calculate success rate with binomial uncertainties
                frac, frac_lower, frac_upper = calculate_binomial_uncertainty(
                    classified_count, total_count
                )

                success_fraction.append(frac)
                success_fraction_lower.append(frac_lower)
                success_fraction_upper.append(frac_upper)
                classified_counts.append(classified_count)
                total_counts.append(total_count)
                bin_centers.append(bin_center)

                mag_min = mag_bins[int(bin_idx)]
                mag_max = mag_bins[int(bin_idx) + 1]

                print(f"{mag_min:5.1f}-{mag_max:5.1f}         {total_count:9,d} {classified_count:11,d} {frac:13.1%}")

    except Exception as e:
        print(f"ERROR executing query: {e}")
        import traceback
        traceback.print_exc()
        cursor.close()
        conn.close()
        return None

    cursor.close()
    conn.close()

    # Convert to numpy arrays
    bin_centers_arr = np.array(bin_centers)
    success_frac_arr = np.array(success_fraction)
    success_frac_lower_arr = np.array(success_fraction_lower)
    success_frac_upper_arr = np.array(success_fraction_upper)
    classified_counts_arr = np.array(classified_counts)
    total_counts_arr = np.array(total_counts)

    return (bin_centers_arr, success_frac_arr, success_frac_lower_arr,
            success_frac_upper_arr, classified_counts_arr, total_counts_arr)


def identify_degradation_regions(bin_centers, success_frac, threshold=0.9):
    """
    Identify magnitude ranges where classification success rate degrades.

    Args:
        bin_centers: Array of magnitude bin centers
        success_frac: Array of success fractions
        threshold: Threshold below which classification is considered degraded (default 90%)

    Returns:
        List of (mag_start, mag_end, success_rate) tuples for degraded regions
    """
    degraded_regions = []
    in_degraded = False
    region_start = None

    for i, (mag, frac) in enumerate(zip(bin_centers, success_frac)):
        if frac < threshold and not in_degraded:
            # Start of degraded region
            in_degraded = True
            region_start = mag
        elif frac >= threshold and in_degraded:
            # End of degraded region
            in_degraded = False
            region_end = bin_centers[i-1] + 0.25  # End of previous bin
            avg_frac = np.mean(success_frac[max(0, i-3):i])
            degraded_regions.append((region_start, region_end, avg_frac))

    # Handle case where data ends while still in degraded region
    if in_degraded:
        region_end = bin_centers[-1] + 0.25
        avg_frac = np.mean(success_frac[-3:])
        degraded_regions.append((region_start, region_end, avg_frac))

    return degraded_regions


def create_visualization(bin_centers_arr, success_frac_arr, success_frac_lower_arr,
                        success_frac_upper_arr, classified_counts_arr, total_counts_arr):
    """Create visualization of PHZ classification success rate vs magnitude."""

    print(f"\nCreating visualization...")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Calculate error bars
    yerr_lower = success_frac_arr - success_frac_lower_arr
    yerr_upper = success_frac_upper_arr - success_frac_arr

    # Plot main success rate with error bars
    ax.errorbar(bin_centers_arr, success_frac_arr * 100,
                yerr=[yerr_lower * 100, yerr_upper * 100],
                fmt='o-', linewidth=2.5, markersize=6, capsize=5, capthick=1.5,
                color='steelblue', ecolor='steelblue', alpha=0.8,
                label='Classification Success Rate')

    # Add confidence bands (shaded region)
    ax.fill_between(bin_centers_arr, success_frac_lower_arr * 100, success_frac_upper_arr * 100,
                    color='steelblue', alpha=0.2)

    # Add reference lines
    ax.axhline(y=100, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect Classification (100%)')
    ax.axhline(y=90, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='Good Threshold (90%)')
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.3, label='Degradation Threshold (50%)')

    # Identify and highlight degradation regions
    degraded = identify_degradation_regions(bin_centers_arr, success_frac_arr, threshold=0.90)

    if degraded:
        for region_start, region_end, avg_frac in degraded:
            ax.axvspan(region_start, region_end, alpha=0.1, color='red')

        # Add text annotations for degraded regions
        for region_start, region_end, avg_frac in degraded:
            mid_mag = (region_start + region_end) / 2
            ax.text(mid_mag, 45, f'{avg_frac:.0%}', ha='center', va='top',
                   fontsize=9, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Add sample size annotation on secondary axis
    ax2 = ax.twinx()
    ax2.bar(bin_centers_arr, total_counts_arr / 1000, width=0.4,
            color='lightgray', alpha=0.3, label='Sample Size (thousands)')
    ax2.set_ylabel('Sample Size (thousands)', fontsize=11, fontweight='bold', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    # Formatting
    ax.set_xlabel('Magnitude (FLUX_DETECTION_TOTAL, ZP=23.9 AB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classification Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('PHZ Classification Success Rate vs Magnitude\n(Fraction of sources successfully classified)',
                 fontsize=14, fontweight='bold')

    ax.set_ylim(0, 105)
    ax.set_xlim(bin_centers_arr.min() - 0.5, bin_centers_arr.max() + 0.5)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))

    ax.grid(True, alpha=0.3, which='major')
    ax.grid(True, alpha=0.15, which='minor')

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10, ncol=2)

    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'q1_phz_classification_success_rate.png', dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {OUTPUT_DIR / 'q1_phz_classification_success_rate.png'}")
    plt.close()


def main():
    """Main execution."""
    print("="*70)
    print("PHZ Classification Success Rate Analysis")
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

    # Query data by magnitude bins
    print("\n" + "="*70)
    print("Step 1: Query Classification Success Rate by Magnitude")
    print("="*70)

    result = query_phz_classification_success_rate(username)
    if result is None:
        print("Failed to query data from WSDB")
        return

    (bin_centers_arr, success_frac_arr, success_frac_lower_arr,
     success_frac_upper_arr, classified_counts_arr, total_counts_arr) = result

    # Create visualization
    print("\n" + "="*70)
    print("Step 2: Create Visualization")
    print("="*70)

    create_visualization(bin_centers_arr, success_frac_arr, success_frac_lower_arr,
                        success_frac_upper_arr, classified_counts_arr, total_counts_arr)

    # Summary analysis
    print("\n" + "="*70)
    print("Analysis Summary")
    print("="*70)

    # Find critical magnitudes
    below_90 = bin_centers_arr[success_frac_arr < 0.90]
    below_80 = bin_centers_arr[success_frac_arr < 0.80]
    below_70 = bin_centers_arr[success_frac_arr < 0.70]

    print(f"\nClassification Completeness Milestones:")
    print(f"  Magnitudes with >90% success: {bin_centers_arr[success_frac_arr >= 0.90].min():.1f} - {bin_centers_arr[success_frac_arr >= 0.90].max():.1f}")

    if len(below_90) > 0:
        print(f"  Magnitudes with <90% success: {below_90.min():.1f} - {below_90.max():.1f}")
    if len(below_80) > 0:
        print(f"  Magnitudes with <80% success: {below_80.min():.1f} - {below_80.max():.1f}")
    if len(below_70) > 0:
        print(f"  Magnitudes with <70% success: {below_70.min():.1f} - {below_70.max():.1f}")

    # Overall statistics
    avg_success = np.mean(success_frac_arr)
    print(f"\nOverall Classification Success Rate: {avg_success:.1%}")
    print(f"Success Rate Range: {success_frac_arr.min():.1%} - {success_frac_arr.max():.1%}")

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print("\nInterpretation Guide:")
    print("✓ HIGH: >90% success rate indicates robust classification")
    print("✓ GOOD: 80-90% success rate is acceptable for bright/intermediate sources")
    print("⚠ CAUTION: 70-80% success rate indicates degradation starting")
    print("✗ LOW: <70% success rate indicates poor classification quality")
    print("\nKey Insight:")
    print("Look for magnitude ranges where the success rate drops sharply.")
    print("This indicates where the PHZ classifier struggles (e.g., faint sources).")


if __name__ == '__main__':
    main()
