#!/usr/bin/env python3
"""
Analyze ERO morphological classification distribution vs magnitude (Direct WSDB Query).

This script:
1. Queries ERO morphology classification directly from WSDB by magnitude bins
2. Analyzes two classification metrics using VIS band morphology indicators:
   - V_CLASS_STAR: POINT-LIKE vs EXTENDED (threshold >= 0.65)
   - V_SPREAD_MODEL: POINT-LIKE vs EXTENDED (threshold based on V_SPREADERR_MODEL)
3. Creates stacked area charts for both metrics
4. Creates individual line plots showing breakdown of each classification
5. Uses V_CLASS_STAR and V_SPREAD_MODEL from the SExtractor/Euclid VIS band morphology pipeline

V_CLASS_STAR Interpretation (per SExtractor/Euclid):
- 0.0 = Definitely extended
- 0.5 = Ambiguous
- 1.0 = Definitely point-like
- Classification: POINT-LIKE >= 0.65, EXTENDED < 0.65

V_SPREAD_MODEL Interpretation:
- Morphological parameter from SExtractor
- Classification: POINT-LIKE if spread <= sqrt(ε² + (κ × spreaderr)²)
                  EXTENDED otherwise
- Per-source threshold based on SExtractor formula
- Constants: ε ≈ 0.005 (instrumental bias), κ ≈ 4 (safety factor)

Magnitude: V_MAG_AUTO (Automatic aperture magnitude from VIS band)
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


def query_ero_classification_by_magnitude(username):
    """
    Query ERO morphology classification data by magnitude bins directly from WSDB (OPTIMIZED).

    Uses a single query with window functions to bin and classify all data at once,
    then processes results in Python. Much faster than looping over 35 queries.

    Queries two morphological classification metrics:
    1. CLASS_STAR: Based on v_class_star (threshold >= 0.65)
    2. SPREAD_MODEL: Based on v_spread_model with per-source threshold from SExtractor formula:
       threshold = sqrt(0.005^2 + (4 * v_spreaderr_model)^2)
       where 0.005 is instrumental bias and 4 is safety factor

    Args:
        username: WSDB username

    Returns:
        tuple of (bin_centers, class_star_point_frac, class_star_ext_frac,
                 spread_point_frac, spread_ext_frac, and their uncertainties, counts)
    """
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed!")
        return None

    print(f"\n{'='*70}")
    print(f"ERO Morphological Classification Distribution (Optimized WSDB Query)")
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

    table_name = 'euclid_ero.merge_cat'

    # Define magnitude bins (0.5 magnitude bins)
    mag_bins = np.arange(10, 28, 0.5)

    print(f"\nQuerying all magnitude bins from {table_name} (SINGLE QUERY)...")
    print(f"Processing {len(mag_bins)-1} bins from mag {mag_bins[0]} to {mag_bins[-1]}")
    print(f"Magnitude column: v_mag_auto (VIS band)")
    print(f"Classification metrics:")
    print(f"  1. v_class_star (VIS band) - threshold >= 0.65")
    print(f"  2. v_spread_model (VIS band) - threshold: sqrt(0.005² + (4*spreaderr)²)\n")

    cursor = conn.cursor()

    # Single optimized query: bin all data at once with WIDTH_BUCKET
    query = f"""
        SELECT
            FLOOR((v_mag_auto - {mag_bins[0]}) / 0.5) as bin_index,
            {mag_bins[0]} + (FLOOR((v_mag_auto - {mag_bins[0]}) / 0.5) * 0.5) +0.25 as bin_center,
            CASE
                WHEN v_class_star >= 0.65 THEN 'CS_POINT_LIKE'
                WHEN v_class_star < 0.65 THEN 'CS_EXTENDED'
            END as cs_classification,
            CASE
                WHEN v_spread_model <= SQRT(0.005*0.005 + (4.0*v_spreaderr_model)*(4.0*v_spreaderr_model))
                THEN 'SP_POINT_LIKE'
                ELSE 'SP_EXTENDED'
            END as sp_classification,
            COUNT(*) as count
        FROM {table_name}
        WHERE v_mag_auto >= {mag_bins[0]}
          AND v_mag_auto < {mag_bins[-1]}
          AND v_mag_auto IS NOT NULL
          AND v_class_star IS NOT NULL
          AND v_spread_model IS NOT NULL
          AND v_spreaderr_model IS NOT NULL
        GROUP BY bin_index, bin_center, cs_classification, sp_classification
        ORDER BY bin_index, cs_classification, sp_classification
    """

    try:
        cursor.execute(query)
        results = cursor.fetchall()
        print(f"Query completed. Processing {len(results)} classification rows...\n")

        # Parse results into a dictionary keyed by (bin_index, cs_class, sp_class)
        # This allows fast lookup
        data_dict = {}
        for row in results:
            bin_idx, bin_center, cs_class, sp_class, count = row
            key = (int(bin_idx), cs_class, sp_class)
            data_dict[key] = count

        # Storage for all fractions and uncertainties
        cs_point_like_fraction = []
        cs_point_like_fraction_lower = []
        cs_point_like_fraction_upper = []
        cs_extended_fraction = []
        cs_extended_fraction_lower = []
        cs_extended_fraction_upper = []

        sp_point_like_fraction = []
        sp_point_like_fraction_lower = []
        sp_point_like_fraction_upper = []
        sp_extended_fraction = []
        sp_extended_fraction_lower = []
        sp_extended_fraction_upper = []

        bin_centers = []
        counts = []

        # Process bins
        for i in range(len(mag_bins) - 1):
            mag_min = mag_bins[i]
            mag_max = mag_bins[i + 1]
            bin_center = (mag_min + mag_max) / 2

            # Get counts for this bin from the dictionary
            cs_pl_count = data_dict.get((i, 'CS_POINT_LIKE', 'SP_POINT_LIKE'), 0) + \
                         data_dict.get((i, 'CS_POINT_LIKE', 'SP_EXTENDED'), 0)
            cs_ext_count = data_dict.get((i, 'CS_EXTENDED', 'SP_POINT_LIKE'), 0) + \
                          data_dict.get((i, 'CS_EXTENDED', 'SP_EXTENDED'), 0)
            sp_pl_count = data_dict.get((i, 'CS_POINT_LIKE', 'SP_POINT_LIKE'), 0) + \
                         data_dict.get((i, 'CS_EXTENDED', 'SP_POINT_LIKE'), 0)
            sp_ext_count = data_dict.get((i, 'CS_POINT_LIKE', 'SP_EXTENDED'), 0) + \
                          data_dict.get((i, 'CS_EXTENDED', 'SP_EXTENDED'), 0)

            total = cs_pl_count + cs_ext_count

            if total > 0:
                # Calculate fractions with uncertainties
                frac_cs_pl, frac_cs_pl_lower, frac_cs_pl_upper = calculate_binomial_uncertainty(
                    cs_pl_count, total
                )
                cs_point_like_fraction.append(frac_cs_pl)
                cs_point_like_fraction_lower.append(frac_cs_pl_lower)
                cs_point_like_fraction_upper.append(frac_cs_pl_upper)

                frac_cs_ext, frac_cs_ext_lower, frac_cs_ext_upper = calculate_binomial_uncertainty(
                    cs_ext_count, total
                )
                cs_extended_fraction.append(frac_cs_ext)
                cs_extended_fraction_lower.append(frac_cs_ext_lower)
                cs_extended_fraction_upper.append(frac_cs_ext_upper)

                frac_sp_pl, frac_sp_pl_lower, frac_sp_pl_upper = calculate_binomial_uncertainty(
                    sp_pl_count, total
                )
                sp_point_like_fraction.append(frac_sp_pl)
                sp_point_like_fraction_lower.append(frac_sp_pl_lower)
                sp_point_like_fraction_upper.append(frac_sp_pl_upper)

                frac_sp_ext, frac_sp_ext_lower, frac_sp_ext_upper = calculate_binomial_uncertainty(
                    sp_ext_count, total
                )
                sp_extended_fraction.append(frac_sp_ext)
                sp_extended_fraction_lower.append(frac_sp_ext_lower)
                sp_extended_fraction_upper.append(frac_sp_ext_upper)

                bin_centers.append(bin_center)
                counts.append(total)

                # Print progress
                print(f"Magnitude {mag_min:5.1f}-{mag_max:5.1f}: "
                      f"{total:8,d} sources | "
                      f"CLASS_STAR: PL={frac_cs_pl:.1%} EXT={frac_cs_ext:.1%} | "
                      f"SPREAD_MODEL: PL={frac_sp_pl:.1%} EXT={frac_sp_ext:.1%}")

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
    cs_point_like_arr = np.array(cs_point_like_fraction)
    cs_extended_arr = np.array(cs_extended_fraction)
    sp_point_like_arr = np.array(sp_point_like_fraction)
    sp_extended_arr = np.array(sp_extended_fraction)

    return (bin_centers_arr,
            cs_point_like_arr, cs_extended_arr,
            sp_point_like_arr, sp_extended_arr,
            cs_point_like_fraction_lower, cs_point_like_fraction_upper,
            cs_extended_fraction_lower, cs_extended_fraction_upper,
            sp_point_like_fraction_lower, sp_point_like_fraction_upper,
            sp_extended_fraction_lower, sp_extended_fraction_upper,
            counts)


def create_visualization(bin_centers_arr,
                        cs_point_like_frac_arr, cs_extended_frac_arr,
                        sp_point_like_frac_arr, sp_extended_frac_arr,
                        cs_point_like_fraction_lower, cs_point_like_fraction_upper,
                        cs_extended_fraction_lower, cs_extended_fraction_upper,
                        sp_point_like_fraction_lower, sp_point_like_fraction_upper,
                        sp_extended_fraction_lower, sp_extended_fraction_upper):
    """Create 2x2 visualization of ERO morphology classification distributions."""

    print(f"\nCreating visualization...")

    # Create figure with better layout management
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('ERO Classification Distribution vs Magnitude',
                 fontsize=14, fontweight='bold', y=0.995)

    # Use GridSpec for more control
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3, top=0.94)

    # ===== Top Left: CLASS_STAR Stacked Area Chart =====
    ax_cs_stacked = fig.add_subplot(gs[0, 0])

    ax_cs_stacked.fill_between(bin_centers_arr, 0, cs_point_like_frac_arr*100,
                               label='POINT-LIKE (v_class_star >= 0.65)', color='darkblue', alpha=0.6)
    ax_cs_stacked.fill_between(bin_centers_arr, cs_point_like_frac_arr*100,
                               (cs_point_like_frac_arr + cs_extended_frac_arr)*100,
                               label='EXTENDED (v_class_star < 0.65)', color='darkgreen', alpha=0.6)

    ax_cs_stacked.axhline(y=100, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax_cs_stacked.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.2)
    ax_cs_stacked.set_xlabel('Magnitude (v_mag_auto)', fontsize=10, fontweight='bold')
    ax_cs_stacked.set_ylabel('Fraction (%)', fontsize=10, fontweight='bold')
    ax_cs_stacked.set_title('CLASS_STAR Distribution (v_class_star)', fontsize=12, fontweight='bold')
    ax_cs_stacked.set_ylim(0, 102)
    ax_cs_stacked.set_xlim(bin_centers_arr.min() - 0.4, bin_centers_arr.max() + 0.4)
    ax_cs_stacked.grid(True, alpha=0.3)
    ax_cs_stacked.legend(loc='best', fontsize=9)
    ax_cs_stacked.tick_params(axis='x', rotation=45, labelsize=9)

    # ===== Top Right: SPREAD_MODEL Stacked Area Chart =====
    ax_sp_stacked = fig.add_subplot(gs[0, 1])

    ax_sp_stacked.fill_between(bin_centers_arr, 0, sp_point_like_frac_arr*100,
                               label='POINT-LIKE (v_spread_model <= threshold)', color='darkblue', alpha=0.6)
    ax_sp_stacked.fill_between(bin_centers_arr, sp_point_like_frac_arr*100,
                               (sp_point_like_frac_arr + sp_extended_frac_arr)*100,
                               label='EXTENDED (v_spread_model > threshold)', color='darkgreen', alpha=0.6)

    ax_sp_stacked.axhline(y=100, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax_sp_stacked.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.2)
    ax_sp_stacked.set_xlabel('Magnitude (v_mag_auto)', fontsize=10, fontweight='bold')
    ax_sp_stacked.set_ylabel('Fraction (%)', fontsize=10, fontweight='bold')
    ax_sp_stacked.set_title('SPREAD_MODEL Distribution (v_spread_model)', fontsize=12, fontweight='bold')
    ax_sp_stacked.set_ylim(0, 102)
    ax_sp_stacked.set_xlim(bin_centers_arr.min() - 0.4, bin_centers_arr.max() + 0.4)
    ax_sp_stacked.grid(True, alpha=0.3)
    ax_sp_stacked.legend(loc='best', fontsize=9)
    ax_sp_stacked.tick_params(axis='x', rotation=45, labelsize=9)

    # ===== Bottom Left: CLASS_STAR Individual Line Plots =====
    ax_cs_pl = fig.add_subplot(gs[1, 0])
    ax_cs_pl.errorbar(bin_centers_arr, cs_point_like_frac_arr*100,
                      yerr=[[(cs_point_like_frac_arr[i] - cs_point_like_fraction_lower[i])*100 for i in range(len(cs_point_like_frac_arr))],
                            [(cs_point_like_fraction_upper[i] - cs_point_like_frac_arr[i])*100 for i in range(len(cs_point_like_frac_arr))]],
                      fmt='o-', linewidth=1.5, markersize=4, capsize=4, capthick=1,
                      color='darkblue', ecolor='darkblue', alpha=0.7, label='POINT-LIKE')
    ax_cs_pl.errorbar(bin_centers_arr, cs_extended_frac_arr*100,
                      yerr=[[(cs_extended_frac_arr[i] - cs_extended_fraction_lower[i])*100 for i in range(len(cs_extended_frac_arr))],
                            [(cs_extended_fraction_upper[i] - cs_extended_frac_arr[i])*100 for i in range(len(cs_extended_frac_arr))]],
                      fmt='s-', linewidth=1.5, markersize=4, capsize=4, capthick=1,
                      color='darkgreen', ecolor='darkgreen', alpha=0.7, label='EXTENDED')
    ax_cs_pl.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax_cs_pl.set_xlabel('Magnitude (v_mag_auto)', fontsize=10, fontweight='bold')
    ax_cs_pl.set_ylabel('Fraction (%)', fontsize=10, fontweight='bold')
    ax_cs_pl.set_title('CLASS_STAR Individual Classifications', fontsize=11, fontweight='bold')
    ax_cs_pl.set_ylim(-5, 105)
    ax_cs_pl.grid(True, alpha=0.3)
    ax_cs_pl.legend(loc='best', fontsize=9)
    ax_cs_pl.tick_params(axis='x', rotation=45, labelsize=9)

    # ===== Bottom Right: SPREAD_MODEL Individual Line Plots =====
    ax_sp_pl = fig.add_subplot(gs[1, 1])
    ax_sp_pl.errorbar(bin_centers_arr, sp_point_like_frac_arr*100,
                      yerr=[[(sp_point_like_frac_arr[i] - sp_point_like_fraction_lower[i])*100 for i in range(len(sp_point_like_frac_arr))],
                            [(sp_point_like_fraction_upper[i] - sp_point_like_frac_arr[i])*100 for i in range(len(sp_point_like_frac_arr))]],
                      fmt='o-', linewidth=1.5, markersize=4, capsize=4, capthick=1,
                      color='darkblue', ecolor='darkblue', alpha=0.7, label='POINT-LIKE')
    ax_sp_pl.errorbar(bin_centers_arr, sp_extended_frac_arr*100,
                      yerr=[[(sp_extended_frac_arr[i] - sp_extended_fraction_lower[i])*100 for i in range(len(sp_extended_frac_arr))],
                            [(sp_extended_fraction_upper[i] - sp_extended_frac_arr[i])*100 for i in range(len(sp_extended_frac_arr))]],
                      fmt='s-', linewidth=1.5, markersize=4, capsize=4, capthick=1,
                      color='darkgreen', ecolor='darkgreen', alpha=0.7, label='EXTENDED')
    ax_sp_pl.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax_sp_pl.set_xlabel('Magnitude (v_mag_auto)', fontsize=10, fontweight='bold')
    ax_sp_pl.set_ylabel('Fraction (%)', fontsize=10, fontweight='bold')
    ax_sp_pl.set_title('SPREAD_MODEL Individual Classifications', fontsize=11, fontweight='bold')
    ax_sp_pl.set_ylim(-5, 105)
    ax_sp_pl.grid(True, alpha=0.3)
    ax_sp_pl.legend(loc='best', fontsize=9)
    ax_sp_pl.tick_params(axis='x', rotation=45, labelsize=9)

    plt.savefig(OUTPUT_DIR / 'ero_classification_dist.png', dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {OUTPUT_DIR / 'ero_classification_dist.png'}")
    plt.close()


def main():
    """Main execution."""
    print("="*70)
    print("ERO Morphological Classification Analysis (Full Dataset via WSDB)")
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

    result = query_ero_classification_by_magnitude(username)
    if result is None:
        print("Failed to query data from WSDB")
        return

    (bin_centers_arr,
     cs_point_like_arr, cs_extended_arr,
     sp_point_like_arr, sp_extended_arr,
     cs_point_like_fraction_lower, cs_point_like_fraction_upper,
     cs_extended_fraction_lower, cs_extended_fraction_upper,
     sp_point_like_fraction_lower, sp_point_like_fraction_upper,
     sp_extended_fraction_lower, sp_extended_fraction_upper,
     counts) = result

    # Create visualization
    print("\n" + "="*70)
    print("Step 2: Create Visualization")
    print("="*70)

    create_visualization(bin_centers_arr,
                        cs_point_like_arr, cs_extended_arr,
                        sp_point_like_arr, sp_extended_arr,
                        cs_point_like_fraction_lower, cs_point_like_fraction_upper,
                        cs_extended_fraction_lower, cs_extended_fraction_upper,
                        sp_point_like_fraction_lower, sp_point_like_fraction_upper,
                        sp_extended_fraction_lower, sp_extended_fraction_upper)

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print("\nVisualization Guide:")
    print("✓ Top Left: CLASS_STAR Stacked Area - V_CLASS_STAR >= 0.65 vs < 0.65")
    print("✓ Top Right: SPREAD_MODEL Stacked Area - V_SPREAD_MODEL <= mean+3σ vs > mean+3σ")
    print("✓ Bottom Left: CLASS_STAR Individual Classifications with error bars")
    print("✓ Bottom Right: SPREAD_MODEL Individual Classifications with error bars")
    print("\nInterpretation Guide:")
    print("✓ GOOD: POINT-LIKE high at bright mags, decreases at fainter mags")
    print("✓ GOOD: EXTENDED low at bright mags, increases at fainter mags")
    print("✓ GOOD: Fractions sum to 100% in each magnitude bin")
    print("✗ BAD: Combined fractions flat across all magnitudes (classifier not working)")
    print("\nThreshold Details:")
    print("✓ CLASS_STAR: Simple threshold at v_class_star >= 0.65")
    print("✓ SPREAD_MODEL: Per-source threshold = sqrt(0.005² + (4 × v_spreaderr_model)²)")
    print("  (SExtractor formula: ε=0.005 instrumental bias, κ=4 safety factor)")


if __name__ == '__main__':
    main()
