#!/usr/bin/env python3
"""
Analyze PHZ Classification Probabilities vs Magnitude.

This script:
1. Loads Q1-PHZ cross-matched data with Gaia IDs
2. Analyzes PHZ STAR_PROB, GAL_PROB, and QSO_PROB distributions vs magnitude
3. Creates 1x3 diagnostic plots with binomial uncertainty estimates
4. Uses probability thresholds from PHZ MER PF documentation

PHZ Probability Thresholds (from configuration table):
- STAR_PROB > 0.58 (UNIONS/Northern Tiles) or 0.73 (DES/Southern Tiles)
- GAL_PROB > 0.22 (UNIONS) or 0.4 (DES)
- QSO_PROB > 0.66 (UNIONS) or 0.85 (DES)

Expected behavior:
- Bright sources (mag < ~15): High fraction of stars
- Faint sources (mag > ~20): Higher fraction of galaxies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'output'

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data_parquet(filename):
    """Load data from Parquet format."""
    filepath = DATA_DIR / filename
    df = pd.read_parquet(filepath)
    print(f"Loaded {len(df)} sources from {filepath}")
    return df


def calculate_binomial_uncertainty(successes, trials):
    """
    Calculate binomial uncertainty on a fraction.

    uncertainty = sqrt(p * (1 - p) / n)
    where p = fraction and n = number of trials
    """
    if trials == 0:
        return 0, 0, 0

    proportion = successes / trials
    uncertainty = np.sqrt(proportion * (1 - proportion) / trials)

    lower = max(0, proportion - uncertainty)
    upper = min(1, proportion + uncertainty)

    return proportion, lower, upper


def analyze_phz_probabilities_by_magnitude(df):
    """
    Analyze PHZ STAR_PROB, GAL_PROB, and QSO_PROB distributions vs magnitude.

    Plots both UNIONS and DES probability thresholds for comparison:
    - STAR_PROB > 0.58 (UNIONS) vs > 0.73 (DES)
    - GAL_PROB > 0.22 (UNIONS) vs > 0.4 (DES)
    - QSO_PROB > 0.66 (UNIONS) vs > 0.85 (DES)
    """
    print(f"\n{'='*70}")
    print(f"PHZ Classification Probabilities Analysis (UNIONS & DES Thresholds)")
    print(f"{'='*70}")

    # Define magnitude bins (0.5 magnitude bins)
    mag_bins = np.arange(6, 28, 0.5)

    # Storage for UNIONS STAR_PROB (0.58) fractions
    star_prob_fraction_58 = []
    star_prob_fraction_58_lower = []
    star_prob_fraction_58_upper = []

    # Storage for DES STAR_PROB (0.73) fractions
    star_prob_fraction_73 = []
    star_prob_fraction_73_lower = []
    star_prob_fraction_73_upper = []

    # Storage for UNIONS GAL_PROB (0.22) fractions
    gal_prob_fraction_22 = []
    gal_prob_fraction_22_lower = []
    gal_prob_fraction_22_upper = []

    # Storage for DES GAL_PROB (0.4) fractions
    gal_prob_fraction_40 = []
    gal_prob_fraction_40_lower = []
    gal_prob_fraction_40_upper = []

    # Storage for UNIONS QSO_PROB (0.66) fractions
    qso_prob_fraction_66 = []
    qso_prob_fraction_66_lower = []
    qso_prob_fraction_66_upper = []

    # Storage for DES QSO_PROB (0.85) fractions
    qso_prob_fraction_85 = []
    qso_prob_fraction_85_lower = []
    qso_prob_fraction_85_upper = []

    bin_centers = []
    counts = []

    for i in range(len(mag_bins)-1):
        mask = (df['mag_stargal_sep'] >= mag_bins[i]) & (df['mag_stargal_sep'] < mag_bins[i+1])
        bin_data = df[mask]

        if len(bin_data) > 0:
            # UNIONS STAR_PROB > 0.58 (classified as star)
            stars_58 = (bin_data['phz_star_prob'] > 0.58).sum()
            frac_star, frac_star_lower, frac_star_upper = calculate_binomial_uncertainty(
                stars_58, len(bin_data)
            )
            star_prob_fraction_58.append(frac_star)
            star_prob_fraction_58_lower.append(frac_star_lower)
            star_prob_fraction_58_upper.append(frac_star_upper)

            # DES STAR_PROB > 0.73 (classified as star)
            stars_73 = (bin_data['phz_star_prob'] > 0.73).sum()
            frac_star, frac_star_lower, frac_star_upper = calculate_binomial_uncertainty(
                stars_73, len(bin_data)
            )
            star_prob_fraction_73.append(frac_star)
            star_prob_fraction_73_lower.append(frac_star_lower)
            star_prob_fraction_73_upper.append(frac_star_upper)

            # UNIONS GAL_PROB > 0.22
            gals_22 = (bin_data['phz_gal_prob'] > 0.22).sum()
            frac_gal, frac_gal_lower, frac_gal_upper = calculate_binomial_uncertainty(
                gals_22, len(bin_data)
            )
            gal_prob_fraction_22.append(frac_gal)
            gal_prob_fraction_22_lower.append(frac_gal_lower)
            gal_prob_fraction_22_upper.append(frac_gal_upper)

            # DES GAL_PROB > 0.4
            gals_40 = (bin_data['phz_gal_prob'] > 0.4).sum()
            frac_gal, frac_gal_lower, frac_gal_upper = calculate_binomial_uncertainty(
                gals_40, len(bin_data)
            )
            gal_prob_fraction_40.append(frac_gal)
            gal_prob_fraction_40_lower.append(frac_gal_lower)
            gal_prob_fraction_40_upper.append(frac_gal_upper)

            # UNIONS QSO_PROB > 0.66
            qsos_66 = (bin_data['phz_qso_prob'] > 0.66).sum()
            frac_qso, frac_qso_lower, frac_qso_upper = calculate_binomial_uncertainty(
                qsos_66, len(bin_data)
            )
            qso_prob_fraction_66.append(frac_qso)
            qso_prob_fraction_66_lower.append(frac_qso_lower)
            qso_prob_fraction_66_upper.append(frac_qso_upper)

            # DES QSO_PROB > 0.85
            qsos_85 = (bin_data['phz_qso_prob'] > 0.85).sum()
            frac_qso, frac_qso_lower, frac_qso_upper = calculate_binomial_uncertainty(
                qsos_85, len(bin_data)
            )
            qso_prob_fraction_85.append(frac_qso)
            qso_prob_fraction_85_lower.append(frac_qso_lower)
            qso_prob_fraction_85_upper.append(frac_qso_upper)

            bin_centers.append((mag_bins[i] + mag_bins[i+1]) / 2)
            counts.append(len(bin_data))

            print(f"Magnitude {mag_bins[i]:5.1f}-{mag_bins[i+1]:5.1f}: "
                  f"{len(bin_data):6d} sources | "
                  f"STAR>0.58={frac_star:.1%} ±{(frac_star_upper-frac_star):.1%} | "
                  f"GAL>0.22={frac_gal:.1%} ±{(frac_gal_upper-frac_gal):.1%} | "
                  f"QSO>0.66={frac_qso:.1%} ±{(frac_qso_upper-frac_qso):.1%}")

    # Create figure with 1x3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle('PHZ Probabilities vs Magnitude',
                 fontsize=14, fontweight='bold')

    # ===== Plot 1: STAR_PROB Distribution =====
    # UNIONS threshold (0.58) - shows fraction classified as stars
    ax1.errorbar(bin_centers, [x*100 for x in star_prob_fraction_58],
                 yerr=[[x*100 for x in (np.array(star_prob_fraction_58) - np.array(star_prob_fraction_58_lower))],
                       [x*100 for x in (np.array(star_prob_fraction_58_upper) - np.array(star_prob_fraction_58))]],
                 fmt='o-', linewidth=2, markersize=4, capsize=4, capthick=1,
                 color='steelblue', ecolor='steelblue', alpha=0.7, label='UNIONS (> 0.58)')
    # DES threshold (0.73) - shows fraction classified as stars
    ax1.errorbar(bin_centers, [x*100 for x in star_prob_fraction_73],
                 yerr=[[x*100 for x in (np.array(star_prob_fraction_73) - np.array(star_prob_fraction_73_lower))],
                       [x*100 for x in (np.array(star_prob_fraction_73_upper) - np.array(star_prob_fraction_73))]],
                 fmt='s-', linewidth=2, markersize=4, capsize=4, capthick=1,
                 color='darkblue', ecolor='darkblue', alpha=0.7, label='DES (> 0.73)')
    ax1.axvline(x=15, color='orange', linestyle=':', linewidth=2, alpha=0.6, label='Bright/Faint divide (~mag 15)')
    ax1.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax1.set_xlabel('Magnitude (mag_stargal_sep)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fraction (%)', fontsize=12, fontweight='bold')
    ax1.set_title('PHZ_STAR_PROB Distribution', fontsize=13, fontweight='bold')
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)

    # ===== Plot 2: GAL_PROB Distribution =====
    # UNIONS threshold (0.22)
    ax2.errorbar(bin_centers, [x*100 for x in gal_prob_fraction_22],
                 yerr=[[x*100 for x in (np.array(gal_prob_fraction_22) - np.array(gal_prob_fraction_22_lower))],
                       [x*100 for x in (np.array(gal_prob_fraction_22_upper) - np.array(gal_prob_fraction_22))]],
                 fmt='o-', linewidth=2, markersize=4, capsize=4, capthick=1,
                 color='seagreen', ecolor='seagreen', alpha=0.7, label='UNIONS (> 0.22)')
    # DES threshold (0.4)
    ax2.errorbar(bin_centers, [x*100 for x in gal_prob_fraction_40],
                 yerr=[[x*100 for x in (np.array(gal_prob_fraction_40) - np.array(gal_prob_fraction_40_lower))],
                       [x*100 for x in (np.array(gal_prob_fraction_40_upper) - np.array(gal_prob_fraction_40))]],
                 fmt='s-', linewidth=2, markersize=4, capsize=4, capthick=1,
                 color='darkgreen', ecolor='darkgreen', alpha=0.7, label='DES (> 0.4)')
    ax2.axvline(x=15, color='orange', linestyle=':', linewidth=2, alpha=0.6, label='Bright/Faint divide (~mag 15)')
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax2.set_xlabel('Magnitude (mag_stargal_sep)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Fraction (%)', fontsize=12, fontweight='bold')
    ax2.set_title('PHZ_GAL_PROB Distribution', fontsize=13, fontweight='bold')
    ax2.set_ylim(-5, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)

    # ===== Plot 3: QSO_PROB Distribution =====
    # UNIONS threshold (0.66)
    ax3.errorbar(bin_centers, [x*100 for x in qso_prob_fraction_66],
                 yerr=[[x*100 for x in (np.array(qso_prob_fraction_66) - np.array(qso_prob_fraction_66_lower))],
                       [x*100 for x in (np.array(qso_prob_fraction_66_upper) - np.array(qso_prob_fraction_66))]],
                 fmt='o-', linewidth=2, markersize=4, capsize=4, capthick=1,
                 color='indianred', ecolor='indianred', alpha=0.7, label='UNIONS (> 0.66)')
    # DES threshold (0.85)
    ax3.errorbar(bin_centers, [x*100 for x in qso_prob_fraction_85],
                 yerr=[[x*100 for x in (np.array(qso_prob_fraction_85) - np.array(qso_prob_fraction_85_lower))],
                       [x*100 for x in (np.array(qso_prob_fraction_85_upper) - np.array(qso_prob_fraction_85))]],
                 fmt='s-', linewidth=2, markersize=4, capsize=4, capthick=1,
                 color='darkred', ecolor='darkred', alpha=0.7, label='DES (> 0.85)')
    ax3.axvline(x=15, color='orange', linestyle=':', linewidth=2, alpha=0.6, label='Bright/Faint divide (~mag 15)')
    ax3.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax3.set_xlabel('Magnitude (mag_stargal_sep)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Fraction (%)', fontsize=12, fontweight='bold')
    ax3.set_title('PHZ_QSO_PROB Distribution', fontsize=13, fontweight='bold')
    ax3.set_ylim(-5, 105)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=10)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'q1_phz_probabilities.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()

    return bin_centers, star_prob_fraction_58, gal_prob_fraction_22, qso_prob_fraction_66, counts


def print_summary_statistics(df):
    """Print summary statistics about PHZ classification probabilities."""
    print(f"\n{'='*70}")
    print("Summary Statistics")
    print(f"{'='*70}")

    print(f"\nPHZ_CLASSIFICATION Distribution:")
    class_counts = df['phz_classification'].value_counts()
    for class_val, count in class_counts.items():
        pct = 100 * count / len(df)
        print(f"  {class_val}: {count:,} ({pct:.2f}%)")

    print(f"\nPHZ_STAR_PROB Statistics:")
    print(f"  Mean: {df['phz_star_prob'].mean():.4f}")
    print(f"  Median: {df['phz_star_prob'].median():.4f}")
    print(f"  Min: {df['phz_star_prob'].min():.4f}")
    print(f"  Max: {df['phz_star_prob'].max():.4f}")

    print(f"\nPHZ_GAL_PROB Statistics:")
    print(f"  Mean: {df['phz_gal_prob'].mean():.4f}")
    print(f"  Median: {df['phz_gal_prob'].median():.4f}")
    print(f"  Min: {df['phz_gal_prob'].min():.4f}")
    print(f"  Max: {df['phz_gal_prob'].max():.4f}")

    print(f"\nPHZ_QSO_PROB Statistics:")
    print(f"  Mean: {df['phz_qso_prob'].mean():.4f}")
    print(f"  Median: {df['phz_qso_prob'].median():.4f}")
    print(f"  Min: {df['phz_qso_prob'].min():.4f}")
    print(f"  Max: {df['phz_qso_prob'].max():.4f}")

    print(f"\nMagnitude Range (mag_stargal_sep): {df['mag_stargal_sep'].min():.2f} - {df['mag_stargal_sep'].max():.2f}")


def main():
    """Main execution."""
    print("="*70)
    print("PHZ Probabilities Analysis")
    print("="*70)

    # Load Q1-PHZ cross-matched data
    df = load_data_parquet('q1_phz_matched_gaia_id.parquet')

    # Print summary statistics
    print_summary_statistics(df)

    # Analyze PHZ probabilities by magnitude
    analyze_phz_probabilities_by_magnitude(df)

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print("\nInterpretation Guide:")
    print("✓ GOOD: STAR fraction high at bright mags, decreasing at faint mags")
    print("✓ GOOD: GAL fraction low at bright mags, increasing at faint mags")
    print("✓ GOOD: QSO fraction relatively constant (rare at all magnitudes)")
    print("✗ BAD: STAR fraction flat or inverted (classifier not working)")
    print("✗ BAD: Sum of probabilities significantly > 100% (calibration issue)")
    print("\nThresholds used (UNIONS configuration - Northern Tiles):")
    print("  - STAR_PROB > 0.58: Star classification threshold")
    print("  - GAL_PROB > 0.22: Galaxy classification threshold")
    print("  - QSO_PROB > 0.66: QSO classification threshold")


if __name__ == '__main__':
    main()
