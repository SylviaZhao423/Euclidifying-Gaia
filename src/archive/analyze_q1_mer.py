#!/usr/bin/env python3
"""
Morphology classification sanity check for Q1 point-like vs extended classification.

This script:
1. Loads Q1-Gaia cross-matched data
2. Analyzes what fraction of sources are classified as point-like vs magnitude
3. Creates 1x2 diagnostic plots with binomial uncertainty estimates
4. Validates the Q1 morphology classifier using thresholds from MER PF documentation

Q1 Classification Method (from MER PF pipeline):
- Uses MU_MAX - MAG_AUTO as a proxy for spread function
- POINT_LIKE_PROB derived from 2D calibration array (MAG, MU_MAX - MAG_AUTO)
- POINT_LIKE_FLAG assigned if POINT_LIKE_PROB > 0.96 (high purity, lower completeness)
- Extended objects typically have POINT_LIKE_PROB <= 0.70

Expected behavior:
- Bright sources (mag < ~15): High fraction point-like (stars are brighter)
- Faint sources (mag > ~20): Higher fraction extended (galaxies dominate at faint magnitudes)
- Monotonic increase in extended fraction with magnitude
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


def analyze_q1_morphology_by_magnitude(df):
    """
    Analyze what fraction of sources are classified as point-like vs magnitude.

    This is a sanity check: we expect bright sources to be mostly point-like,
    and faint sources to be mostly extended.

    Uses thresholds based on Q1 MER PF documentation:
    - POINT_LIKE_FLAG (= 1): Flag classification for point-like sources
    - POINT_LIKE_PROB (< 0.96 and < 0.70): Two probability-based thresholds for non-point-like

    Uses Q1's native mag_stargal_sep (the magnitude used for POINT_LIKE_PROB calibration).
    Includes binomial uncertainty (1-sigma) to show statistical error due to
    finite number of sources in each magnitude bin.
    """
    print(f"\n{'='*70}")
    print(f"Q1 Morphological Classification Sanity Check")
    print(f"{'='*70}")

    # Use Q1's native classification magnitude (mag_stargal_sep)
    # This is the magnitude used to calibrate POINT_LIKE_PROB in the MER PF pipeline
    # Define magnitude bins (0.5 magnitude bins)
    mag_bins = np.arange(6, 28, 0.5)

    # For each bin, calculate fraction classified as point-like with binomial errors
    pointlike_fraction_flag = []
    pointlike_fraction_flag_lower = []
    pointlike_fraction_flag_upper = []

    extended_fraction_prob_96 = []
    extended_fraction_prob_96_lower = []
    extended_fraction_prob_96_upper = []

    extended_fraction_prob_70 = []
    extended_fraction_prob_70_lower = []
    extended_fraction_prob_70_upper = []

    bin_centers = []
    counts = []

    for i in range(len(mag_bins)-1):
        # Use Q1's native classification magnitude (mag_stargal_sep)
        mask = (df['q1_mag_stargal_sep'] >= mag_bins[i]) & (df['q1_mag_stargal_sep'] < mag_bins[i+1])
        bin_data = df[mask]

        if len(bin_data) > 0:
            # Using POINT_LIKE_FLAG (1 = point-like)
            pointlike_flag = (bin_data['q1_point_like_flag'] == 1).sum()
            frac_flag, frac_flag_lower, frac_flag_upper = calculate_binomial_uncertainty(
                pointlike_flag, len(bin_data)
            )
            pointlike_fraction_flag.append(frac_flag)
            pointlike_fraction_flag_lower.append(frac_flag_lower)
            pointlike_fraction_flag_upper.append(frac_flag_upper)

            # Using POINT_LIKE_PROB with 0.96 threshold (official threshold)
            extended_prob_96 = (bin_data['q1_point_like_prob'] < 0.96).sum()
            frac_prob_96, frac_prob_96_lower, frac_prob_96_upper = calculate_binomial_uncertainty(
                extended_prob_96, len(bin_data)
            )
            extended_fraction_prob_96.append(frac_prob_96)
            extended_fraction_prob_96_lower.append(frac_prob_96_lower)
            extended_fraction_prob_96_upper.append(frac_prob_96_upper)

            # Using 0.70 threshold for extended (conservative)
            extended_prob_70 = (bin_data['q1_point_like_prob'] < 0.70).sum()
            frac_prob_70, frac_prob_70_lower, frac_prob_70_upper = calculate_binomial_uncertainty(
                extended_prob_70, len(bin_data)
            )
            extended_fraction_prob_70.append(frac_prob_70)
            extended_fraction_prob_70_lower.append(frac_prob_70_lower)
            extended_fraction_prob_70_upper.append(frac_prob_70_upper)

            bin_centers.append((mag_bins[i] + mag_bins[i+1]) / 2)
            counts.append(len(bin_data))

            print(f"Magnitude {mag_bins[i]:5.1f}-{mag_bins[i+1]:5.1f}: "
                  f"{len(bin_data):6d} sources | "
                  f"FLAG=1={frac_flag:.1%} ±{(frac_flag_upper-frac_flag):.1%} | "
                  f"PROB<0.96={frac_prob_96:.1%} ±{(frac_prob_96_upper-frac_prob_96):.1%} | "
                  f"PROB<0.70={frac_prob_70:.1%} ±{(frac_prob_70_upper-frac_prob_70):.1%}")

    # Create figure with 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle('Q1 Classification Sanity Check',
                 fontsize=14, fontweight='bold')

    # ===== Plot 1: POINT_LIKE_FLAG=1 (point-like) classification fraction with errors =====
    ax1.errorbar(bin_centers, [x*100 for x in pointlike_fraction_flag],
                 yerr=[[x*100 for x in (np.array(pointlike_fraction_flag) - np.array(pointlike_fraction_flag_lower))],
                       [x*100 for x in (np.array(pointlike_fraction_flag_upper) - np.array(pointlike_fraction_flag))]],
                 fmt='o-', linewidth=2.5, markersize=7, capsize=5, capthick=1.5,
                 color='darkred', ecolor='darkred', alpha=0.7, label='1-sigma CI')
    ax1.axvline(x=15, color='orange', linestyle=':', linewidth=2, alpha=0.6, label='Bright/Faint divide (~mag 15)')
    ax1.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax1.set_xlabel('Q1 mag_stargal_sep', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fraction Classified as Point-Like (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Q1 POINT_LIKE_FLAG = 1 (Point-Like Objects)', fontsize=13, fontweight='bold')
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)

    # ===== Plot 2: POINT_LIKE_PROB thresholds with errors =====
    ax2.errorbar(bin_centers, [x*100 for x in extended_fraction_prob_96],
                 yerr=[[x*100 for x in (np.array(extended_fraction_prob_96) - np.array(extended_fraction_prob_96_lower))],
                       [x*100 for x in (np.array(extended_fraction_prob_96_upper) - np.array(extended_fraction_prob_96))]],
                 fmt='o-', linewidth=2.5, markersize=7, capsize=5, capthick=1.5,
                 color='steelblue', ecolor='steelblue', alpha=0.7, label='PROB < 0.96')
    ax2.errorbar(bin_centers, [x*100 for x in extended_fraction_prob_70],
                 yerr=[[x*100 for x in (np.array(extended_fraction_prob_70) - np.array(extended_fraction_prob_70_lower))],
                       [x*100 for x in (np.array(extended_fraction_prob_70_upper) - np.array(extended_fraction_prob_70))]],
                 fmt='s-', linewidth=2.5, markersize=7, capsize=5, capthick=1.5,
                 color='darkblue', ecolor='darkblue', alpha=0.7, label='PROB < 0.70')
    ax2.axvline(x=15, color='orange', linestyle=':', linewidth=2, alpha=0.6, label='Bright/Faint divide')
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax2.set_xlabel('Q1 mag_stargal_sep', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Fraction of Sources (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Q1 POINT_LIKE_PROB Distributions', fontsize=13, fontweight='bold')
    ax2.set_ylim(-5, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'q1_mer_sanity_check.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()

    return bin_centers, pointlike_fraction_flag, extended_fraction_prob_96, counts


def print_summary_statistics(df):
    """Print summary statistics about Q1 classification."""
    print(f"\n{'='*70}")
    print("Summary Statistics")
    print(f"{'='*70}")

    print(f"\nQ1 POINT_LIKE_FLAG Distribution:")
    flag_counts = df['q1_point_like_flag'].value_counts().sort_index()
    for flag_val, count in flag_counts.items():
        pct = 100 * count / len(df)
        if flag_val == 1:
            print(f"  Flag=1 (Point-Like): {count:,} ({pct:.2f}%)")
        else:
            print(f"  Flag={flag_val} (Non-Point-Like/Unclassified): {count:,} ({pct:.2f}%)")

    print(f"\nQ1 POINT_LIKE_PROB Statistics:")
    print(f"  Mean: {df['q1_point_like_prob'].mean():.4f}")
    print(f"  Median: {df['q1_point_like_prob'].median():.4f}")
    print(f"  Min: {df['q1_point_like_prob'].min():.4f}")
    print(f"  Max: {df['q1_point_like_prob'].max():.4f}")
    print(f"  Std Dev: {df['q1_point_like_prob'].std():.4f}")

    print(f"\nQ1 Classification by Probability Threshold:")
    pointlike_96 = (df['q1_point_like_prob'] > 0.96).sum()
    extended_70 = (df['q1_point_like_prob'] <= 0.70).sum()
    intermediate = ((df['q1_point_like_prob'] > 0.70) & (df['q1_point_like_prob'] <= 0.96)).sum()
    print(f"  PROB > 0.96 (Point-Like): {pointlike_96:,} ({100*pointlike_96/len(df):.2f}%)")
    print(f"  0.70 < PROB ≤ 0.96 (Intermediate): {intermediate:,} ({100*intermediate/len(df):.2f}%)")
    print(f"  PROB ≤ 0.70 (Extended): {extended_70:,} ({100*extended_70/len(df):.2f}%)")

    print(f"\nQ1 Magnitude Range (mag_stargal_sep): {df['q1_mag_stargal_sep'].min():.2f} - {df['q1_mag_stargal_sep'].max():.2f}")
    print(f"Q1 MU_MAX - MAG (spread proxy): {df['q1_mumax_minus_mag'].min():.4f} - {df['q1_mumax_minus_mag'].max():.4f}")

    print(f"\nCross-Match Quality:")
    sep = df['angular_separation_arcsec'].dropna()
    print(f"  Angular separation mean: {sep.mean():.4f} arcsec")
    print(f"  Angular separation median: {sep.median():.4f} arcsec")
    print(f"  Within 0.5 arcsec: {(sep <= 0.5).sum():,} ({100*(sep <= 0.5).sum()/len(sep):.2f}%)")


def main():
    """Main execution."""
    print("="*70)
    print("Q1 Morphology Classification Sanity Check")
    print("="*70)

    # Load Q1-Gaia cross-matched data
    df = load_data_parquet('q1_gaia_subset.parquet')

    # Print summary statistics
    print_summary_statistics(df)

    # Analyze morphology by magnitude
    analyze_q1_morphology_by_magnitude(df)

    print("\n" + "="*70)
    print("Sanity Check Complete!")
    print("="*70)
    print("\nInterpretation Guide:")
    print("✓ GOOD: Bright sources (mag<15) have high % point-like, decreases at fainter mags")
    print("✓ GOOD: Point-like fraction decreases monotonically with magnitude")
    print("✓ GOOD: Extended sources concentrated at fainter magnitudes")
    print("✗ BAD: Bright sources have high % extended (misclassification)")
    print("✗ BAD: Flat line across all magnitudes (classifier not working)")
    print("✗ BAD: Inverted trend (point-like more common at faint magnitudes)")
    print("\nThresholds used (from Q1 MER PF documentation):")
    print("  - POINT_LIKE_FLAG = 1: High purity point-source identification")
    print("  - POINT_LIKE_PROB < 0.96: Sources below official point-like threshold")
    print("  - POINT_LIKE_PROB < 0.70: Conservative extended source identification")


if __name__ == '__main__':
    main()
