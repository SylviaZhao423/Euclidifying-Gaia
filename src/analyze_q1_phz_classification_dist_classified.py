#!/usr/bin/env python3
"""
Analyze PHZ morphological classification distribution vs magnitude (CLASSIFIED ONLY).

This script:
1. Queries Q1-Gaia-PHZ crossmatch directly from WSDB by magnitude bins (SINGLE QUERY)
2. Restricts analysis to CLASSIFIED objects only (p_phz_classification IN 1, 2, 4, 8)
3. Excludes OTHER classifications (multi-class, outliers, unclassified)
4. Analyzes STAR, GALAXY, QSO, and GLOBULAR_CLUSTER classifications vs magnitude
5. Creates a stacked area chart showing 3 combined categories (STAR, GALAXY, QSO+GLOBULAR_CLUSTER) summing to 100%
6. Creates 3 individual plots showing breakdown: STAR alone, GALAXY alone, and QSO+GLOBULAR separately
7. Uses p_phz_classification from the PHZ pipeline (ESA DPDD)

PHZ_CLASSIFICATION Values (per ESA DPDD):
"Classification flag: I1 if the source is accepted as star, I2 if it is accepted as
a galaxy I4 if it is accepted as a QSO I8 if the source is accepted as a globular Cluster"

Dominant class encoding (CLASSIFIED ONLY):
- 1 (I1): Source accepted as STAR
- 2 (I2): Source accepted as GALAXY
- 4 (I4): Source accepted as QSO
- 8 (I8): Source accepted as GLOBULAR CLUSTER
(EXCLUDED: -1, 0, 3, 5, 6, 7 = OTHER/unclassified)

Visualization grouping:
- Top stacked chart shows 3 combined categories: STAR, GALAXY, QSO+GLOBULAR_CLUSTER
- Bottom-left plot: STAR (I1) fraction only
- Bottom-middle plot: GALAXY (I2) fraction only
- Bottom-right plot: QSO (I4) and GLOBULAR_CLUSTER (I8) shown separately

Magnitude: ZPC-corrected MER detection flux (FLUX_DETECTION_TOTAL from IE detection mosaic)
Filters for sources with VIS_DET = 1 (successful VIS band detection)

Zero-Point Multiplicative Corrections (ZPC):
- IE (I_E): ZPC = 1.05
- YE (Y_E): ZPC = 0.95
- JE (J_E): ZPC = 0.97
- HE (H_E): ZPC = 0.95

Using IE-band (I_E) with ZPC=1.05 for magnitude calculation
Formula: m_AB = -2.5 × log10(1.05 × FLUX_DETECTION_TOTAL) + 23.9
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

# Zero-Point Multiplicative Correction for IE-band
ZPC_IE = 1.05
ZP_REF = 23.9  # Reference zero-point

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


def query_phz_classification_by_magnitude_classified(username):
    """
    Query PHZ classification data by magnitude bins directly from WSDB (CLASSIFIED ONLY, with ZPC).

    Uses a single query with window functions to bin and classify all data at once,
    then processes results in Python. Much faster than looping over 45 queries.

    Only includes p_phz_classification IN (1, 2, 4, 8) - excludes OTHER.
    Filters for sources with VIS_DET = 1 (successful VIS band detection).
    Uses ZPC-corrected magnitudes.

    Args:
        username: WSDB username

    Returns:
        tuple of (bin_centers, star_fraction, galaxy_fraction,
                  qso_gc_fraction, qso_fraction, globular_fraction,
                  and their uncertainties, counts)
    """
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed!")
        return None

    print(f"\n{'='*70}")
    print(f"PHZ Classification Distribution Analysis (CLASSIFIED ONLY - WSDB Query)")
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

    # Define magnitude bins
    BIN_WIDTH = 0.5
    mag_bins = np.arange(6, 28, BIN_WIDTH)

    print(f"\nQuerying all magnitude bins from {table_name} (SINGLE QUERY)...")
    print(f"Processing {len(mag_bins)-1} bins from mag {mag_bins[0]} to {mag_bins[-1]}")
    print(f"Magnitude formula: m_AB = -2.5 × log10({ZPC_IE} × FLUX_DETECTION_TOTAL) + {ZP_REF}")
    print(f"Classification column: p_phz_classification (PHZ pipeline)")
    print(f"RESTRICTIONS:")
    print(f"  - Only classified objects (p_phz_classification IN 1, 2, 4, 8)")
    print(f"  - Only sources with VIS_DET = 1 (successful VIS band detection)\n")

    cursor = conn.cursor()

    # Single optimized query: bin all data at once with FLOOR
    # Magnitude computed from ZPC-corrected MER detection flux: mag = -2.5 * log10(ZPC * flux) + ZP_ref
    # ONLY includes p_phz_classification IN (1, 2, 4, 8)
    # ONLY includes m_vis_det = 1 (successful VIS band detection)
    query = f"""
        SELECT
            FLOOR((-2.5 * LOG10({ZPC_IE} * m_flux_detection_total) + {ZP_REF} - {mag_bins[0]}) / {BIN_WIDTH}) as bin_index,
            {mag_bins[0]} + (FLOOR((-2.5 * LOG10({ZPC_IE} * m_flux_detection_total) + {ZP_REF} - {mag_bins[0]}) / {BIN_WIDTH}) * {BIN_WIDTH}) + {BIN_WIDTH/2} as bin_center,
            CASE
                WHEN p_phz_classification = 1 THEN 'STAR'
                WHEN p_phz_classification = 2 THEN 'GALAXY'
                WHEN p_phz_classification = 4 THEN 'QSO'
                WHEN p_phz_classification = 8 THEN 'GLOBULAR'
            END as phz_class,
            COUNT(*) as count
        FROM {table_name}
        WHERE m_vis_det = 1
          AND m_flux_detection_total > 0
          AND (-2.5 * LOG10({ZPC_IE} * m_flux_detection_total) + {ZP_REF}) >= {mag_bins[0]}
          AND (-2.5 * LOG10({ZPC_IE} * m_flux_detection_total) + {ZP_REF}) < {mag_bins[-1]}
          AND m_flux_detection_total IS NOT NULL
          AND p_phz_classification IS NOT NULL
          AND p_phz_classification IN (1, 2, 4, 8)
        GROUP BY bin_index, bin_center, phz_class
        ORDER BY bin_index, phz_class
    """

    try:
        cursor.execute(query)
        results = cursor.fetchall()
        print(f"Query completed. Processing {len(results)} classification rows...\n")

        # Parse results into a dictionary keyed by (bin_index, phz_class)
        data_dict = {}
        for row in results:
            bin_idx, bin_center, phz_class, count = row
            key = (int(bin_idx), phz_class)
            data_dict[key] = count

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

        qso_fraction = []
        qso_fraction_lower = []
        qso_fraction_upper = []

        qso_gc_fraction = []
        qso_gc_fraction_lower = []
        qso_gc_fraction_upper = []

        bin_centers = []
        counts = []

        # Process bins
        for i in range(len(mag_bins) - 1):
            mag_min = mag_bins[i]
            mag_max = mag_bins[i + 1]
            bin_center = (mag_min + mag_max) / 2

            # Get counts for this bin from the dictionary
            star_count = data_dict.get((i, 'STAR'), 0)
            galaxy_count = data_dict.get((i, 'GALAXY'), 0)
            qso_count = data_dict.get((i, 'QSO'), 0)
            globular_count = data_dict.get((i, 'GLOBULAR'), 0)

            total = star_count + galaxy_count + qso_count + globular_count
            qso_gc_count = qso_count + globular_count

            if total > 0:
                # Calculate fractions with uncertainties for STAR
                frac_star, frac_star_lower, frac_star_upper = calculate_binomial_uncertainty(
                    star_count, total
                )
                star_fraction.append(frac_star)
                star_fraction_lower.append(frac_star_lower)
                star_fraction_upper.append(frac_star_upper)

                # Calculate fractions with uncertainties for GALAXY
                frac_gal, frac_gal_lower, frac_gal_upper = calculate_binomial_uncertainty(
                    galaxy_count, total
                )
                galaxy_fraction.append(frac_gal)
                galaxy_fraction_lower.append(frac_gal_lower)
                galaxy_fraction_upper.append(frac_gal_upper)

                # Calculate fractions with uncertainties for GLOBULAR
                frac_glob, frac_glob_lower, frac_glob_upper = calculate_binomial_uncertainty(
                    globular_count, total
                )
                globular_fraction.append(frac_glob)
                globular_fraction_lower.append(frac_glob_lower)
                globular_fraction_upper.append(frac_glob_upper)

                # Calculate fractions with uncertainties for QSO
                frac_qso, frac_qso_lower, frac_qso_upper = calculate_binomial_uncertainty(
                    qso_count, total
                )
                qso_fraction.append(frac_qso)
                qso_fraction_lower.append(frac_qso_lower)
                qso_fraction_upper.append(frac_qso_upper)

                # Calculate fractions with uncertainties for QSO_GC (QSO + GLOBULAR)
                frac_qso_gc, frac_qso_gc_lower, frac_qso_gc_upper = calculate_binomial_uncertainty(
                    qso_gc_count, total
                )
                qso_gc_fraction.append(frac_qso_gc)
                qso_gc_fraction_lower.append(frac_qso_gc_lower)
                qso_gc_fraction_upper.append(frac_qso_gc_upper)

                bin_centers.append(bin_center)
                counts.append(total)

                # Verify fractions sum to ~100%
                total_frac = frac_star + frac_gal + frac_qso_gc
                print(f"Magnitude {mag_min:5.1f}-{mag_max:5.1f}: "
                      f"{total:8,d} classified sources | "
                      f"STAR={frac_star:.1%} | "
                      f"GALAXY={frac_gal:.1%} | "
                      f"QSO+GC={frac_qso_gc:.1%} (QSO={frac_qso:.1%} GC={frac_glob:.1%}) | "
                      f"SUM={total_frac:.1%}")

    except Exception as e:
        print(f"ERROR executing optimized query: {e}")
        import traceback
        traceback.print_exc()
        cursor.close()
        conn.close()
        return None

    cursor.close()
    conn.close()

    # Convert to numpy arrays
    bin_centers_arr = np.array(bin_centers)
    star_frac_arr = np.array(star_fraction)
    galaxy_frac_arr = np.array(galaxy_fraction)
    globular_frac_arr = np.array(globular_fraction)
    qso_frac_arr = np.array(qso_fraction)
    qso_gc_frac_arr = np.array(qso_gc_fraction)

    return (bin_centers_arr, star_frac_arr, galaxy_frac_arr, qso_gc_frac_arr,
            qso_frac_arr, globular_frac_arr,
            star_fraction_lower, star_fraction_upper,
            galaxy_fraction_lower, galaxy_fraction_upper,
            globular_fraction_lower, globular_fraction_upper,
            qso_fraction_lower, qso_fraction_upper,
            qso_gc_fraction_lower, qso_gc_fraction_upper,
            counts)


def create_visualization(bin_centers_arr, star_frac_arr, galaxy_frac_arr, qso_gc_frac_arr,
                        qso_frac_arr, globular_frac_arr,
                        star_fraction_lower, star_fraction_upper,
                        galaxy_fraction_lower, galaxy_fraction_upper,
                        globular_fraction_lower, globular_fraction_upper,
                        qso_fraction_lower, qso_fraction_upper,
                        qso_gc_fraction_lower, qso_gc_fraction_upper):
    """Create visualization of PHZ classification distribution (classified only)."""

    print(f"\nCreating visualization...")

    # Create figure with better layout management
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle('PHZ Classification Distribution vs Magnitude (CLASSIFIED OBJECTS ONLY)',
                 fontsize=15, fontweight='bold', y=1)

    # Use GridSpec for more control
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3, top=0.93)

    # ===== Top Row: Stacked Area Chart (spans all 3 columns) =====
    ax_stacked = fig.add_subplot(gs[0, :])

    ax_stacked.fill_between(bin_centers_arr, 0, star_frac_arr*100,
                            label='STAR (I1)', color='darkblue', alpha=0.6)
    ax_stacked.fill_between(bin_centers_arr, star_frac_arr*100,
                            (star_frac_arr + galaxy_frac_arr)*100,
                            label='GALAXY (I2)', color='darkgreen', alpha=0.6)
    ax_stacked.fill_between(bin_centers_arr,
                            (star_frac_arr + galaxy_frac_arr)*100,
                            (star_frac_arr + galaxy_frac_arr + qso_gc_frac_arr)*100,
                            label='QSO+GLOBULAR_CLUSTER', color='darkred', alpha=0.6)

    ax_stacked.axhline(y=100, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax_stacked.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.2)
    ax_stacked.set_xlabel('Magnitude (FLUX_DETECTION_TOTAL, ZP=23.9 AB)', fontsize=11, fontweight='bold')
    ax_stacked.set_ylabel('Fraction (%)', fontsize=11, fontweight='bold')
    ax_stacked.set_title('Combined Classification Distribution (Classified Only, Summing to 100%)', fontsize=12, fontweight='bold')
    ax_stacked.set_ylim(0, 105)
    ax_stacked.set_xlim(bin_centers_arr.min() - 0.1, bin_centers_arr.max() + 0.1)
    ax_stacked.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax_stacked.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax_stacked.grid(True, alpha=0.3, which='both')
    ax_stacked.legend(loc='upper left', fontsize=10, ncol=3)
    ax_stacked.tick_params(axis='x', rotation=45, labelsize=9)

    # ===== Bottom Row: 3 Individual Line Plots =====

    # STAR Classification
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.errorbar(bin_centers_arr, star_frac_arr*100,
                 yerr=[[(star_frac_arr[i] - star_fraction_lower[i])*100 for i in range(len(star_frac_arr))],
                       [(star_fraction_upper[i] - star_frac_arr[i])*100 for i in range(len(star_frac_arr))]],
                 fmt='o-', linewidth=1.5, markersize=3, capsize=4, capthick=1,
                 color='darkblue', ecolor='darkblue', alpha=0.7, label='STAR (I1)')
    ax1.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Magnitude', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Fraction (%)', fontsize=10, fontweight='bold')
    ax1.set_title('STAR (I1)', fontsize=11, fontweight='bold')
    ax1.set_ylim(-5, 105)
    ax1.set_xlim(10, 28)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))
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
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Magnitude', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Fraction (%)', fontsize=10, fontweight='bold')
    ax2.set_title('GALAXY (I2)', fontsize=11, fontweight='bold')
    ax2.set_ylim(-5, 105)
    ax2.set_xlim(9, 28)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)
    ax2.tick_params(axis='x', rotation=45, labelsize=9)

    # QSO + GLOBULAR_CLUSTER Classification
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
    ax3.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Magnitude', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Fraction (%)', fontsize=10, fontweight='bold')
    ax3.set_title('QSO + GLOBULAR_CLUSTER', fontsize=11, fontweight='bold')
    ax3.set_ylim(-5, 105)
    ax3.set_xlim(9, 28)
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=8)
    ax3.tick_params(axis='x', rotation=45, labelsize=9)

    plt.savefig(OUTPUT_DIR / 'q1_phz_classification_dist_classified.png', dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {OUTPUT_DIR / 'q1_phz_classification_dist_classified.png'}")
    plt.close()


def main():
    """Main execution."""
    print("="*70)
    print("PHZ Classification Distribution Analysis (CLASSIFIED ONLY - via WSDB)")
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
    print("Step 1: Query Classification by Magnitude Bins (CLASSIFIED ONLY)")
    print("="*70)

    result = query_phz_classification_by_magnitude_classified(username)
    if result is None:
        print("Failed to query data from WSDB")
        return

    (bin_centers_arr, star_frac_arr, galaxy_frac_arr, qso_gc_frac_arr,
     qso_frac_arr, globular_frac_arr,
     star_fraction_lower, star_fraction_upper,
     galaxy_fraction_lower, galaxy_fraction_upper,
     globular_fraction_lower, globular_fraction_upper,
     qso_fraction_lower, qso_fraction_upper,
     qso_gc_fraction_lower, qso_gc_fraction_upper,
     counts) = result

    # Create visualization
    print("\n" + "="*70)
    print("Step 2: Create Visualization")
    print("="*70)

    create_visualization(bin_centers_arr, star_frac_arr, galaxy_frac_arr, qso_gc_frac_arr,
                        qso_frac_arr, globular_frac_arr,
                        star_fraction_lower, star_fraction_upper,
                        galaxy_fraction_lower, galaxy_fraction_upper,
                        globular_fraction_lower, globular_fraction_upper,
                        qso_fraction_lower, qso_fraction_upper,
                        qso_gc_fraction_lower, qso_gc_fraction_upper)

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print("\nVisualization Guide:")
    print("✓ Stacked Area Chart (top): Shows 3 combined categories summing to 100%")
    print("                            (STAR, GALAXY, QSO+GLOBULAR_CLUSTER)")
    print("✓ STAR Plot (bottom-left): STAR (I1) fraction only")
    print("✓ GALAXY Plot (bottom-middle): GALAXY (I2) fraction only")
    print("✓ Right Plot (bottom-right): QSO (I4) and GLOBULAR_CLUSTER (I8) shown separately")
    print("\nInterpretation Guide:")
    print("✓ GOOD: STAR fraction high at bright mags, decreases at fainter mags")
    print("✓ GOOD: GALAXY fraction low at bright mags, increases at fainter mags")
    print("✓ GOOD: QSO+GLOBULAR fraction relatively constant and low")
    print("✓ GOOD: Fractions sum to 100% in each magnitude bin (only classified sources)")
    print("✗ BAD: Combined fractions flat across all magnitudes (classifier not working)")


if __name__ == '__main__':
    main()
