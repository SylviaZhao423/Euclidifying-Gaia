#!/usr/bin/env python3
"""
PHZ Classification Distribution vs Magnitude (Full Dataset via WSDB Direct Query).

This script:
1. Queries Q1-Gaia-PHZ crossmatch directly from WSDB by magnitude bins
2. Analyzes STAR, GALAXY+GLOBULAR_CLUSTER, and QSO+OTHER classifications vs magnitude
3. Creates a stacked area chart showing the 3 combined categories summing to 100%
4. Creates 3 individual plots showing breakdown of each category
5. Uses PHZ_CLASSIFICATION from the PHZ pipeline (ESA DPDD)

PHZ_CLASSIFICATION Values (per ESA DPDD):
"Classification flag: I1 if the source is accepted as star, I2 if it is accepted as
a galaxy I4 if it is accepted as a QSO I8 if the source is accepted as a globular Cluster"

Dominant class encoding:
- 1 (I1): Source accepted as STAR
- 2 (I2): Source accepted as GALAXY
- 4 (I4): Source accepted as QSO
- 8 (I8): Source accepted as GLOBULAR CLUSTER
- -1, 0, 3, 5, 6, 7: Multi-class, outliers, or unclassified (OTHER)

Combined categories:
- STAR: classification == 1
- GALAXY_EXTENDED: classification == 2 OR classification == 8
- QSO_OTHER: classification == 4 OR classification in (-1, 0, 3, 5, 6, 7)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def query_phz_classification_by_magnitude(username):
    """
    Query Q1-Gaia-PHZ classification data by magnitude bins directly from WSDB.

    Args:
        username: WSDB username

    Returns:
        tuple of (bin_centers, star_fraction, galaxy_extended_fraction,
                  qso_other_fraction, counts)
    """
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed!")
        return None

    print(f"\n{'='*70}")
    print(f"PHZ Classification Distribution Analysis (Direct WSDB Query)")
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

    table_name = f"{username}.q1_gaia_phz_crossmatch"

    # Define magnitude bins (0.5 magnitude bins)
    mag_bins = np.arange(6, 28, 0.5)

    # Storage for all fractions and uncertainties
    star_fraction = []
    star_fraction_lower = []
    star_fraction_upper = []

    galaxy_fraction = []
    galaxy_fraction_lower = []
    galaxy_fraction_upper = []

    globular_fraction = []
    globular_fraction_lower = []
    globular_fraction_upper = []

    galaxy_extended_fraction = []
    galaxy_extended_fraction_lower = []
    galaxy_extended_fraction_upper = []

    qso_fraction = []
    qso_fraction_lower = []
    qso_fraction_upper = []

    other_fraction = []
    other_fraction_lower = []
    other_fraction_upper = []

    qso_other_fraction = []
    qso_other_fraction_lower = []
    qso_other_fraction_upper = []

    bin_centers = []
    counts = []

    print(f"\nQuerying magnitude bins from {table_name}...")
    print(f"(This processes {len(mag_bins)-1} bins from mag {mag_bins[0]} to {mag_bins[-1]})\n")

    cursor = conn.cursor()

    for i in range(len(mag_bins)-1):
        mag_min = mag_bins[i]
        mag_max = mag_bins[i+1]

        # Query for this magnitude bin
        query = f"""
            SELECT
                p_phz_classification,
                COUNT(*) as count
            FROM {table_name}
            WHERE m_mag_stargal_sep >= {mag_min}
              AND m_mag_stargal_sep < {mag_max}
              AND m_mag_stargal_sep IS NOT NULL
              AND p_phz_classification IS NOT NULL
            GROUP BY p_phz_classification
        """

        try:
            cursor.execute(query)
            results = cursor.fetchall()

            if not results:
                continue

            # Parse results into a dictionary
            class_counts = {row[0]: row[1] for row in results}
            total = sum(class_counts.values())

            if total > 0:
                # Extract counts for each classification
                stars = class_counts.get(1, 0)  # STAR
                galaxies = class_counts.get(2, 0)  # GALAXY
                qsos = class_counts.get(4, 0)  # QSO
                globulars = class_counts.get(8, 0)  # GLOBULAR_CLUSTER
                other = total - stars - galaxies - qsos - globulars

                # Calculate fractions with uncertainties
                frac_star, frac_star_lower, frac_star_upper = calculate_binomial_uncertainty(
                    stars, total
                )
                star_fraction.append(frac_star)
                star_fraction_lower.append(frac_star_lower)
                star_fraction_upper.append(frac_star_upper)

                frac_gal, frac_gal_lower, frac_gal_upper = calculate_binomial_uncertainty(
                    galaxies, total
                )
                galaxy_fraction.append(frac_gal)
                galaxy_fraction_lower.append(frac_gal_lower)
                galaxy_fraction_upper.append(frac_gal_upper)

                frac_glob, frac_glob_lower, frac_glob_upper = calculate_binomial_uncertainty(
                    globulars, total
                )
                globular_fraction.append(frac_glob)
                globular_fraction_lower.append(frac_glob_lower)
                globular_fraction_upper.append(frac_glob_upper)

                frac_gal_ext, frac_gal_ext_lower, frac_gal_ext_upper = calculate_binomial_uncertainty(
                    galaxies + globulars, total
                )
                galaxy_extended_fraction.append(frac_gal_ext)
                galaxy_extended_fraction_lower.append(frac_gal_ext_lower)
                galaxy_extended_fraction_upper.append(frac_gal_ext_upper)

                frac_qso, frac_qso_lower, frac_qso_upper = calculate_binomial_uncertainty(
                    qsos, total
                )
                qso_fraction.append(frac_qso)
                qso_fraction_lower.append(frac_qso_lower)
                qso_fraction_upper.append(frac_qso_upper)

                frac_other, frac_other_lower, frac_other_upper = calculate_binomial_uncertainty(
                    other, total
                )
                other_fraction.append(frac_other)
                other_fraction_lower.append(frac_other_lower)
                other_fraction_upper.append(frac_other_upper)

                frac_qso_other, frac_qso_other_lower, frac_qso_other_upper = calculate_binomial_uncertainty(
                    qsos + other, total
                )
                qso_other_fraction.append(frac_qso_other)
                qso_other_fraction_lower.append(frac_qso_other_lower)
                qso_other_fraction_upper.append(frac_qso_other_upper)

                bin_centers.append((mag_min + mag_max) / 2)
                counts.append(total)

                # Verify fractions sum to ~100%
                total_frac = frac_star + frac_gal_ext + frac_qso_other
                print(f"Magnitude {mag_min:5.1f}-{mag_max:5.1f}: "
                      f"{total:8,d} sources | "
                      f"STAR={frac_star:.1%} | "
                      f"GALAXY+GC={frac_gal_ext:.1%} (GAL={frac_gal:.1%} GC={frac_glob:.1%}) | "
                      f"QSO+OTHER={frac_qso_other:.1%} (QSO={frac_qso:.1%} OTH={frac_other:.1%}) | "
                      f"SUM={total_frac:.1%}")

        except Exception as e:
            print(f"ERROR querying magnitude bin {mag_min}-{mag_max}: {e}")
            continue

    cursor.close()
    conn.close()

    # Convert to numpy arrays
    bin_centers_arr = np.array(bin_centers)
    star_frac_arr = np.array(star_fraction)
    galaxy_extended_frac_arr = np.array(galaxy_extended_fraction)
    qso_other_frac_arr = np.array(qso_other_fraction)
    galaxy_frac_arr = np.array(galaxy_fraction)
    globular_frac_arr = np.array(globular_fraction)
    qso_frac_arr = np.array(qso_fraction)
    other_frac_arr = np.array(other_fraction)

    return (bin_centers_arr, star_frac_arr, galaxy_extended_frac_arr, qso_other_frac_arr,
            galaxy_frac_arr, globular_frac_arr, qso_frac_arr, other_frac_arr,
            star_fraction_lower, star_fraction_upper,
            galaxy_fraction_lower, galaxy_fraction_upper,
            globular_fraction_lower, globular_fraction_upper,
            qso_fraction_lower, qso_fraction_upper,
            other_fraction_lower, other_fraction_upper,
            qso_other_fraction_lower, qso_other_fraction_upper,
            counts)


def create_visualization(bin_centers_arr, star_frac_arr, galaxy_extended_frac_arr,
                        qso_other_frac_arr, galaxy_frac_arr, globular_frac_arr,
                        qso_frac_arr, other_frac_arr,
                        star_fraction_lower, star_fraction_upper,
                        galaxy_fraction_lower, galaxy_fraction_upper,
                        globular_fraction_lower, globular_fraction_upper,
                        qso_fraction_lower, qso_fraction_upper,
                        other_fraction_lower, other_fraction_upper,
                        qso_other_fraction_lower, qso_other_fraction_upper):
    """Create visualization of PHZ classification distribution."""

    print(f"\nCreating visualization...")

    # Create figure with better layout management
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle('PHZ Classification Distribution (fraction relative to total count per magnitude bin) vs Magnitude\n(Q1-Gaia Dataset)',
                 fontsize=15, fontweight='bold', y=1.02)

    # Use GridSpec for more control
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3, top=0.93)

    # ===== Top Row: Stacked Area Chart (spans all 3 columns) =====
    ax_stacked = fig.add_subplot(gs[0, :])

    # Calculate combined QSO+GC+OTHER for stacked chart
    qso_gc_other_combined = qso_frac_arr + globular_frac_arr + other_frac_arr

    ax_stacked.fill_between(bin_centers_arr, 0, star_frac_arr*100,
                            label='STAR (I1)', color='darkblue', alpha=0.6)
    ax_stacked.fill_between(bin_centers_arr, star_frac_arr*100,
                            (star_frac_arr + galaxy_frac_arr)*100,
                            label='GALAXY (I2)', color='darkgreen', alpha=0.6)
    ax_stacked.fill_between(bin_centers_arr,
                            (star_frac_arr + galaxy_frac_arr)*100,
                            (star_frac_arr + galaxy_frac_arr + qso_gc_other_combined)*100,
                            label='QSO+GLOBULAR_CLUSTER+OTHER', color='darkred', alpha=0.6)

    ax_stacked.axvline(x=15, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
    ax_stacked.axhline(y=100, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax_stacked.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.2)
    ax_stacked.set_xlabel('Magnitude (Q1 mag_stargal_sep)', fontsize=11, fontweight='bold')
    ax_stacked.set_ylabel('Fraction (%)', fontsize=11, fontweight='bold')
    ax_stacked.set_title('Combined Classification Distribution', fontsize=12, fontweight='bold')
    ax_stacked.set_ylim(0, 105)
    ax_stacked.grid(True, alpha=0.3)
    ax_stacked.legend(loc='upper left', fontsize=10, ncol=3)
    ax_stacked.tick_params(axis='x', rotation=45, labelsize=9)

    # ===== Bottom Row: 3 Individual Line Plots =====

    # STAR Classification
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.errorbar(bin_centers_arr, star_frac_arr*100,
                 yerr=[[(star_frac_arr[i] - star_fraction_lower[i])*100 for i in range(len(star_frac_arr))],
                       [(star_fraction_upper[i] - star_frac_arr[i])*100 for i in range(len(star_frac_arr))]],
                 fmt='o-', linewidth=1.5, markersize=3, capsize=4, capthick=1,
                 color='darkblue', ecolor='darkblue', alpha=0.7, label='STAR')
    ax1.axvline(x=15, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
    ax1.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Magnitude (Q1 mag_stargal_sep)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Fraction (%)', fontsize=10, fontweight='bold')
    ax1.set_title('STAR (I1)', fontsize=11, fontweight='bold')
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)
    ax1.tick_params(axis='x', rotation=45, labelsize=9)

    # GALAXY Classification
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.errorbar(bin_centers_arr, galaxy_frac_arr*100,
                 yerr=[[(galaxy_frac_arr[i] - galaxy_fraction_lower[i])*100 for i in range(len(galaxy_frac_arr))],
                       [(galaxy_fraction_upper[i] - galaxy_frac_arr[i])*100 for i in range(len(galaxy_frac_arr))]],
                 fmt='s-', linewidth=1.5, markersize=3, capsize=4, capthick=1,
                 color='darkgreen', ecolor='darkgreen', alpha=0.7, label='GALAXY (I2)')
    ax2.axvline(x=15, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Magnitude (Q1 mag_stargal_sep)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Fraction (%)', fontsize=10, fontweight='bold')
    ax2.set_title('GALAXY (I2)', fontsize=11, fontweight='bold')
    ax2.set_ylim(-5, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)
    ax2.tick_params(axis='x', rotation=45, labelsize=9)

    # QSO + GLOBULAR_CLUSTER + OTHER Classification
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.errorbar(bin_centers_arr, qso_frac_arr*100,
                 yerr=[[(qso_frac_arr[i] - qso_fraction_lower[i])*100 for i in range(len(qso_frac_arr))],
                       [(qso_fraction_upper[i] - qso_frac_arr[i])*100 for i in range(len(qso_frac_arr))]],
                 fmt='^-', linewidth=1.5, markersize=3, capsize=4, capthick=1,
                 color='darkorange', ecolor='darkorange', alpha=0.7, label='QSO (I4)')
    ax3.errorbar(bin_centers_arr, globular_frac_arr*100,
                 yerr=[[(globular_frac_arr[i] - globular_fraction_lower[i])*100 for i in range(len(globular_frac_arr))],
                       [(globular_fraction_upper[i] - globular_frac_arr[i])*100 for i in range(len(globular_frac_arr))]],
                 fmt='D-', linewidth=1.5, markersize=3, capsize=4, capthick=1,
                 color='violet', ecolor='violet', alpha=0.7, label='GLOBULAR_CLUSTER (I8)')
    ax3.errorbar(bin_centers_arr, other_frac_arr*100,
                 yerr=[[(other_frac_arr[i] - other_fraction_lower[i])*100 for i in range(len(other_frac_arr))],
                       [(other_fraction_upper[i] - other_frac_arr[i])*100 for i in range(len(other_frac_arr))]],
                 fmt='s-', linewidth=1.5, markersize=3, capsize=4, capthick=1,
                 color='red', ecolor='red', alpha=0.7, label='OTHER (Multi/Outlier)')
    ax3.axvline(x=15, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
    ax3.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Magnitude (Q1 mag_stargal_sep)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Fraction (%)', fontsize=10, fontweight='bold')
    ax3.set_title('QSO + GLOBULAR_CLUSTER + OTHER', fontsize=11, fontweight='bold')
    ax3.set_ylim(-5, 105)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=8)
    ax3.tick_params(axis='x', rotation=45, labelsize=9)

    plt.savefig(OUTPUT_DIR / 'q1_gaia_phz_classification_combined_wsdb.png', dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {OUTPUT_DIR / 'q1_gaia_phz_classification_combined_wsdb.png'}")
    plt.close()


def main():
    """Main execution."""
    print("="*70)
    print("PHZ Classification Distribution Analysis (Full Dataset via WSDB)")
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
    print("Step 1: Query Classification by Magnitude Bins")
    print("="*70)

    result = query_phz_classification_by_magnitude(username)
    if result is None:
        print("Failed to query data from WSDB")
        return

    (bin_centers_arr, star_frac_arr, galaxy_extended_frac_arr, qso_other_frac_arr,
     galaxy_frac_arr, globular_frac_arr, qso_frac_arr, other_frac_arr,
     star_fraction_lower, star_fraction_upper,
     galaxy_fraction_lower, galaxy_fraction_upper,
     globular_fraction_lower, globular_fraction_upper,
     qso_fraction_lower, qso_fraction_upper,
     other_fraction_lower, other_fraction_upper,
     qso_other_fraction_lower, qso_other_fraction_upper,
     counts) = result

    # Create visualization
    print("\n" + "="*70)
    print("Step 2: Create Visualization")
    print("="*70)

    create_visualization(bin_centers_arr, star_frac_arr, galaxy_extended_frac_arr,
                        qso_other_frac_arr, galaxy_frac_arr, globular_frac_arr,
                        qso_frac_arr, other_frac_arr,
                        star_fraction_lower, star_fraction_upper,
                        galaxy_fraction_lower, galaxy_fraction_upper,
                        globular_fraction_lower, globular_fraction_upper,
                        qso_fraction_lower, qso_fraction_upper,
                        other_fraction_lower, other_fraction_upper,
                        qso_other_fraction_lower, qso_other_fraction_upper)

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print("\nVisualization Guide:")
    print("✓ Stacked Area Chart (top): Shows 3 combined categories summing to 100%")
    print("✓ STAR Plot (bottom-left): Only STAR fraction")
    print("✓ GALAXY Plot (bottom-middle): GALAXY and GLOBULAR_CLUSTER breakdown")
    print("✓ QSO+OTHER Plot (bottom-right): QSO, OTHER, and combined line")
    print("\nInterpretation Guide:")
    print("✓ GOOD: STAR fraction high at bright mags, decreases at fainter mags")
    print("✓ GOOD: GALAXY+GC fraction low at bright mags, increases at fainter mags")
    print("✓ GOOD: QSO+OTHER fraction relatively constant and low")
    print("✓ GOOD: Fractions sum to 100% in each magnitude bin")
    print("✗ BAD: Combined fractions flat across all magnitudes (classifier not working)")


if __name__ == '__main__':
    main()
