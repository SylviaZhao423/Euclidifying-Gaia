#!/usr/bin/env python3
"""
Compare PHZ Star Probability with Q1 Point-Like Probability vs Magnitude.

This script:
1. Loads Q1-PHZ cross-matched data with Gaia IDs
2. Loads Q1-Gaia cross-matched data for complete Q1 information
3. Analyzes both PHZ STAR_PROB and Q1 POINT_LIKE_PROB distributions vs magnitude
4. Creates 1x2 diagnostic plots for direct comparison
5. Uses binomial uncertainty estimates

This comparison allows validation of whether Q1 morphological classification
(POINT_LIKE_PROB) aligns with the official Euclid PHZ classification (STAR_PROB).
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


def analyze_phz_vs_q1_by_magnitude(df_phz, df_q1):
    """
    Analyze both PHZ STAR_PROB and Q1 POINT_LIKE_PROB distributions vs magnitude.

    Uses Q1's native mag_stargal_sep for binning.
    Matches datasets by object_id to ensure proper alignment.
    """
    print(f"\n{'='*70}")
    print(f"PHZ vs Q1 Morphology Comparison")
    print(f"{'='*70}")

    # Create a mapping from PHZ object_id to Q1 data for efficient lookup
    q1_by_object_id = df_q1.set_index('q1_object_id')

    # Define magnitude bins (0.5 magnitude bins)
    mag_bins = np.arange(6, 28, 0.5)

    # PHZ galaxy probability storage (GAL_PROB > 0.22, 0.4, and 0.99 thresholds)
    phz_gal_fraction_22 = []
    phz_gal_fraction_22_lower = []
    phz_gal_fraction_22_upper = []

    phz_gal_fraction_40 = []
    phz_gal_fraction_40_lower = []
    phz_gal_fraction_40_upper = []

    phz_gal_fraction_99 = []
    phz_gal_fraction_99_lower = []
    phz_gal_fraction_99_upper = []

    # Q1 point-like probability storage (PROB < 0.70, 0.96, and 0.99 thresholds)
    q1_extended_fraction_70 = []
    q1_extended_fraction_70_lower = []
    q1_extended_fraction_70_upper = []

    q1_extended_fraction_96 = []
    q1_extended_fraction_96_lower = []
    q1_extended_fraction_96_upper = []

    q1_extended_fraction_99 = []
    q1_extended_fraction_99_lower = []
    q1_extended_fraction_99_upper = []

    bin_centers = []
    counts = []

    for i in range(len(mag_bins)-1):
        mask = (df_phz['mag_stargal_sep'] >= mag_bins[i]) & (df_phz['mag_stargal_sep'] < mag_bins[i+1])
        bin_data_phz = df_phz[mask]

        # Get corresponding Q1 data for same sources (using object_id for matching)
        bin_data_q1 = q1_by_object_id.loc[bin_data_phz['object_id']]

        if len(bin_data_phz) > 0:
            # PHZ: Galaxy probability thresholds (0.22, 0.4, and 0.99) - count ABOVE threshold (galaxies)
            phz_galaxies_22 = (bin_data_phz['phz_gal_prob'] > 0.22).sum()
            frac_phz_22, frac_phz_22_lower, frac_phz_22_upper = calculate_binomial_uncertainty(
                phz_galaxies_22, len(bin_data_phz)
            )
            phz_gal_fraction_22.append(frac_phz_22)
            phz_gal_fraction_22_lower.append(frac_phz_22_lower)
            phz_gal_fraction_22_upper.append(frac_phz_22_upper)

            phz_galaxies_40 = (bin_data_phz['phz_gal_prob'] > 0.4).sum()
            frac_phz_40, frac_phz_40_lower, frac_phz_40_upper = calculate_binomial_uncertainty(
                phz_galaxies_40, len(bin_data_phz)
            )
            phz_gal_fraction_40.append(frac_phz_40)
            phz_gal_fraction_40_lower.append(frac_phz_40_lower)
            phz_gal_fraction_40_upper.append(frac_phz_40_upper)

            phz_galaxies_99 = (bin_data_phz['phz_gal_prob'] > 0.99).sum()
            frac_phz_99, frac_phz_99_lower, frac_phz_99_upper = calculate_binomial_uncertainty(
                phz_galaxies_99, len(bin_data_phz)
            )
            phz_gal_fraction_99.append(frac_phz_99)
            phz_gal_fraction_99_lower.append(frac_phz_99_lower)
            phz_gal_fraction_99_upper.append(frac_phz_99_upper)

            # Q1: Point-like probability thresholds (extended = 1 - point-like)
            q1_extended_70 = (bin_data_q1['q1_point_like_prob'] < 0.70).sum()
            frac_q1_70, frac_q1_70_lower, frac_q1_70_upper = calculate_binomial_uncertainty(
                q1_extended_70, len(bin_data_q1)
            )
            q1_extended_fraction_70.append(frac_q1_70)
            q1_extended_fraction_70_lower.append(frac_q1_70_lower)
            q1_extended_fraction_70_upper.append(frac_q1_70_upper)

            q1_extended_96 = (bin_data_q1['q1_point_like_prob'] < 0.96).sum()
            frac_q1_96, frac_q1_96_lower, frac_q1_96_upper = calculate_binomial_uncertainty(
                q1_extended_96, len(bin_data_q1)
            )
            q1_extended_fraction_96.append(frac_q1_96)
            q1_extended_fraction_96_lower.append(frac_q1_96_lower)
            q1_extended_fraction_96_upper.append(frac_q1_96_upper)

            q1_extended_99 = (bin_data_q1['q1_point_like_prob'] < 0.99).sum()
            frac_q1_99, frac_q1_99_lower, frac_q1_99_upper = calculate_binomial_uncertainty(
                q1_extended_99, len(bin_data_q1)
            )
            q1_extended_fraction_99.append(frac_q1_99)
            q1_extended_fraction_99_lower.append(frac_q1_99_lower)
            q1_extended_fraction_99_upper.append(frac_q1_99_upper)

            bin_centers.append((mag_bins[i] + mag_bins[i+1]) / 2)
            counts.append(len(bin_data_phz))

            print(f"Magnitude {mag_bins[i]:5.1f}-{mag_bins[i+1]:5.1f}: "
                  f"{len(bin_data_phz):6d} sources | "
                  f"PHZ_GAL>0.22={frac_phz_22:.1%} ±{(frac_phz_22_upper-frac_phz_22):.1%} | "
                  f"Q1_PROB<0.96={frac_q1_96:.1%} ±{(frac_q1_96_upper-frac_q1_96):.1%}")

    # Create figure with 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle('Q1 Classifier Sanity Check: PHZ vs Mer',
                 fontsize=14, fontweight='bold')

    # ===== Plot 1: PHZ_GAL_PROB Distribution =====
    ax1.errorbar(bin_centers, [x*100 for x in phz_gal_fraction_99],
                 yerr=[[x*100 for x in (np.array(phz_gal_fraction_99) - np.array(phz_gal_fraction_99_lower))],
                       [x*100 for x in (np.array(phz_gal_fraction_99_upper) - np.array(phz_gal_fraction_99))]],
                 fmt='^-', linewidth=2, markersize=7, capsize=5, capthick=1.5,
                 color='darkblue', ecolor='darkblue', alpha=0.7, label='GAL_PROB > 0.99')
    ax1.errorbar(bin_centers, [x*100 for x in phz_gal_fraction_40],
                 yerr=[[x*100 for x in (np.array(phz_gal_fraction_40) - np.array(phz_gal_fraction_40_lower))],
                       [x*100 for x in (np.array(phz_gal_fraction_40_upper) - np.array(phz_gal_fraction_40))]],
                 fmt='s-', linewidth=2, markersize=7, capsize=5, capthick=1.5,
                 color='steelblue', ecolor='steelblue', alpha=0.7, label='GAL_PROB > 0.4')
    ax1.errorbar(bin_centers, [x*100 for x in phz_gal_fraction_22],
                 yerr=[[x*100 for x in (np.array(phz_gal_fraction_22) - np.array(phz_gal_fraction_22_lower))],
                       [x*100 for x in (np.array(phz_gal_fraction_22_upper) - np.array(phz_gal_fraction_22))]],
                 fmt='o-', linewidth=2, markersize=7, capsize=5, capthick=1.5,
                 color='green', ecolor='green', alpha=0.7, label='GAL_PROB > 0.22')
    ax1.axvline(x=15, color='orange', linestyle=':', linewidth=2, alpha=0.6, label='Bright/Faint divide (~mag 15)')
    ax1.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax1.set_xlabel('Q1 mag_stargal_sep', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fraction Classified as Galaxies (%)', fontsize=12, fontweight='bold')
    ax1.set_title('PHZ_GAL_PROB (Galaxy Fraction)', fontsize=13, fontweight='bold')
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)

    # ===== Plot 2: Q1 POINT_LIKE_PROB Distribution =====
    ax2.errorbar(bin_centers, [x*100 for x in q1_extended_fraction_70],
                 yerr=[[x*100 for x in (np.array(q1_extended_fraction_70) - np.array(q1_extended_fraction_70_lower))],
                       [x*100 for x in (np.array(q1_extended_fraction_70_upper) - np.array(q1_extended_fraction_70))]],
                 fmt='s-', linewidth=2, markersize=7, capsize=5, capthick=1.5,
                 color='darkblue', ecolor='darkblue', alpha=0.7, label='POINT_LIKE_PROB < 0.70')
    ax2.errorbar(bin_centers, [x*100 for x in q1_extended_fraction_96],
                 yerr=[[x*100 for x in (np.array(q1_extended_fraction_96) - np.array(q1_extended_fraction_96_lower))],
                       [x*100 for x in (np.array(q1_extended_fraction_96_upper) - np.array(q1_extended_fraction_96))]],
                 fmt='o-', linewidth=2, markersize=7, capsize=5, capthick=1.5,
                 color='steelblue', ecolor='steelblue', alpha=0.7, label='POINT_LIKE_PROB < 0.96')
    ax2.errorbar(bin_centers, [x*100 for x in q1_extended_fraction_99],
                 yerr=[[x*100 for x in (np.array(q1_extended_fraction_99) - np.array(q1_extended_fraction_99_lower))],
                       [x*100 for x in (np.array(q1_extended_fraction_99_upper) - np.array(q1_extended_fraction_99))]],
                 fmt='^-', linewidth=2, markersize=7, capsize=5, capthick=1.5,
                 color='green', ecolor='green', alpha=0.7, label='POINT_LIKE_PROB < 0.99')
    ax2.axvline(x=15, color='orange', linestyle=':', linewidth=2, alpha=0.6, label='Bright/Faint divide (~mag 15)')
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax2.set_xlabel('Q1 mag_stargal_sep', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Fraction Classified as Extended (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Mer POINT_LIKE_PROB Distributions', fontsize=13, fontweight='bold')
    ax2.set_ylim(-5, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'q1_phz_vs_mer_sanity_check.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()

    return bin_centers, phz_gal_fraction_22, q1_extended_fraction_96, counts


def print_summary_statistics(df_phz, df_q1):
    """Print summary statistics about both classifications."""
    print(f"\n{'='*70}")
    print("Summary Statistics")
    print(f"{'='*70}")

    print(f"\nQ1 POINT_LIKE_FLAG Distribution:")
    flag_counts = df_q1['q1_point_like_flag'].value_counts().sort_index()
    for flag_val, count in flag_counts.items():
        pct = 100 * count / len(df_q1)
        if flag_val == 1:
            print(f"  Flag=1 (Point-Like): {count:,} ({pct:.2f}%)")
        else:
            print(f"  Flag={flag_val} (Non-Point-Like/Unclassified): {count:,} ({pct:.2f}%)")

    print(f"\nQ1 POINT_LIKE_PROB Statistics:")
    print(f"  Mean: {df_q1['q1_point_like_prob'].mean():.4f}")
    print(f"  Median: {df_q1['q1_point_like_prob'].median():.4f}")
    print(f"  Min: {df_q1['q1_point_like_prob'].min():.4f}")
    print(f"  Max: {df_q1['q1_point_like_prob'].max():.4f}")

    print(f"\nPHZ_STAR_PROB Statistics:")
    print(f"  Mean: {df_phz['phz_star_prob'].mean():.4f}")
    print(f"  Median: {df_phz['phz_star_prob'].median():.4f}")
    print(f"  Min: {df_phz['phz_star_prob'].min():.4f}")
    print(f"  Max: {df_phz['phz_star_prob'].max():.4f}")

    print(f"\nPHZ_CLASSIFICATION Distribution:")
    class_counts = df_phz['phz_classification'].value_counts()
    for class_val, count in class_counts.items():
        pct = 100 * count / len(df_phz)
        print(f"  {class_val}: {count:,} ({pct:.2f}%)")

    print(f"\nMagnitude Range (mag_stargal_sep): {df_phz['mag_stargal_sep'].min():.2f} - {df_phz['mag_stargal_sep'].max():.2f}")


def main():
    """Main execution."""
    print("="*70)
    print("Q1 Morphology Classification Sanity Check")
    print("="*70)

    # Load both datasets
    df_phz = load_data_parquet('q1_phz_matched_gaia_id.parquet')
    df_q1 = load_data_parquet('q1_gaia_subset.parquet')

    # Print summary statistics
    print_summary_statistics(df_phz, df_q1)

    # Analyze and plot
    analyze_phz_vs_q1_by_magnitude(df_phz, df_q1)

    print("\n" + "="*70)
    print("Sanity Check Complete!")
    print("="*70)
    print("\nInterpretation Guide:")
    print("✓ GOOD: Both plots show similar patterns (high fraction at bright mags, decreasing at faint)")
    print("✓ GOOD: PHZ and Q1 classifiers show agreement on star/point-like identification")
    print("✓ GOOD: Point-like/Star fraction decreases monotonically with magnitude")
    print("✗ BAD: Patterns differ significantly (may indicate misalignment)")
    print("✗ BAD: One classifier flat while other shows trend (one not working properly)")
    print("\nNote:")
    print("- Q1 uses morphology proxy (MU_MAX - MAG_AUTO) for classification")
    print("- PHZ uses full multi-wavelength photometry and spectral energy distribution")


if __name__ == '__main__':
    main()
