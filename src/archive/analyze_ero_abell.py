#!/usr/bin/env python3
"""
Morphology classification sanity check for Abell 2390 and Abell 2764.

This script:
1. Loads parquet data from extraction scripts
2. Analyzes what fraction of sources are classified as extended vs magnitude
3. Validates that bright sources are mostly point-like and faint sources are mostly extended
4. Creates diagnostic plots to check for misclassification issues

Based on supervisor feedback: "The fainter (larger magnitude) the object the more of them
we expect, and we expect the proportion of extended objects to increase."
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

    # Binomial uncertainty
    uncertainty = np.sqrt(proportion * (1 - proportion) / trials)

    lower = max(0, proportion - uncertainty)
    upper = min(1, proportion + uncertainty)

    return proportion, lower, upper


def analyze_morphology_by_magnitude(df, cluster_name, output_prefix):
    """
    Analyze what fraction of sources are classified as extended vs magnitude.

    This is a sanity check: we expect bright sources to be mostly point-like,
    and faint sources to be mostly extended.

    Uses pre-calculated magnitude-dependent threshold from extraction scripts.
    Includes binomial uncertainty (1-sigma) to show statistical error due to
    finite number of sources in each magnitude bin.
    """
    print(f"\n{'='*60}")
    print(f"Morphological Classification Sanity Check - {cluster_name}")
    print(f"{'='*60}")

    # Define magnitude bins
    mag_bins = np.arange(10, 28, 0.5)  # 0.5 magnitude bins

    # For each bin, calculate fraction classified as extended with binomial errors
    extended_fraction_class_star = []
    extended_fraction_class_star_lower = []
    extended_fraction_class_star_upper = []

    extended_fraction_spread_error_weighted = []
    extended_fraction_spread_lower = []
    extended_fraction_spread_upper = []

    extended_count_class_star = []
    extended_count_spread_error_weighted = []
    bin_centers = []
    counts = []

    for i in range(len(mag_bins)-1):
        mask = (df['magnitude'] >= mag_bins[i]) & (df['magnitude'] < mag_bins[i+1])
        bin_data = df[mask]

        if len(bin_data) > 0:
            # Using class_star: extended if class_star < 0.65 (conservative threshold from ERO Abell 2390)
            extended_class = (bin_data['class_star'] < 0.65).sum()
            frac_class, frac_class_lower, frac_class_upper = calculate_binomial_uncertainty(
                extended_class, len(bin_data)
            )
            extended_fraction_class_star.append(frac_class)
            extended_fraction_class_star_lower.append(frac_class_lower)
            extended_fraction_class_star_upper.append(frac_class_upper)
            extended_count_class_star.append(extended_class)

            # Using spread_model with error-weighted threshold
            extended_spread_error_weighted = (bin_data['spread_model'] > bin_data['spread_model_threshold']).sum()
            frac_spread, frac_spread_lower, frac_spread_upper = calculate_binomial_uncertainty(
                extended_spread_error_weighted, len(bin_data)
            )
            extended_fraction_spread_error_weighted.append(frac_spread)
            extended_fraction_spread_lower.append(frac_spread_lower)
            extended_fraction_spread_upper.append(frac_spread_upper)
            extended_count_spread_error_weighted.append(extended_spread_error_weighted)

            bin_centers.append((mag_bins[i] + mag_bins[i+1]) / 2)
            counts.append(len(bin_data))

            print(f"Magnitude {mag_bins[i]:.1f}-{mag_bins[i+1]:.1f}: "
                  f"{len(bin_data):5d} sources | "
                  f"class_star<0.65: {frac_class:.1%} ±{(frac_class_upper-frac_class):.1%} | "
                  f"spread(error-weighted): {frac_spread:.1%} ±{(frac_spread_upper-frac_spread):.1%}")

    # Calculate fraction of total extended sources in each bin
    total_extended_class_star = sum(extended_count_class_star)
    total_extended_spread_error_weighted = sum(extended_count_spread_error_weighted)

    frac_of_total_class_star = [count / total_extended_class_star * 100 if total_extended_class_star > 0 else 0
                                 for count in extended_count_class_star]
    frac_of_total_spread_error_weighted = [count / total_extended_spread_error_weighted * 100 if total_extended_spread_error_weighted > 0 else 0
                                            for count in extended_count_spread_error_weighted]

    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # ===== TOP ROW: Fraction of sources in each bin classified as extended =====

    # Plot 1: Class star classification fraction with error bars
    ax1.errorbar(bin_centers, [x*100 for x in extended_fraction_class_star],
                 yerr=[[x*100 for x in (np.array(extended_fraction_class_star) - np.array(extended_fraction_class_star_lower))],
                       [x*100 for x in (np.array(extended_fraction_class_star_upper) - np.array(extended_fraction_class_star))]],
                 fmt='o-', linewidth=2, markersize=6, capsize=5, capthick=1.5,
                 color='steelblue', ecolor='steelblue', alpha=0.7)
    ax1.axvline(x=15, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.text(15.2, 95, 'Bright/Faint divide', fontsize=10, alpha=0.7)

    ax1.set_xlabel('Magnitude', fontsize=12)
    ax1.set_ylabel('Fraction classified as extended (%)', fontsize=12)
    ax1.set_title(f'{cluster_name} - Class Star (Threshold = 0.65)', fontsize=13, fontweight='bold')
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Spread model classification fraction with error bars
    ax2.errorbar(bin_centers, [x*100 for x in extended_fraction_spread_error_weighted],
                 yerr=[[x*100 for x in (np.array(extended_fraction_spread_error_weighted) - np.array(extended_fraction_spread_lower))],
                       [x*100 for x in (np.array(extended_fraction_spread_upper) - np.array(extended_fraction_spread_error_weighted))]],
                 fmt='o-', linewidth=2, markersize=6, capsize=5, capthick=1.5,
                 color='darkgreen', ecolor='darkgreen', alpha=0.7)
    ax2.axvline(x=15, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
    ax2.text(15.2, 95, 'Bright/Faint divide', fontsize=10, alpha=0.7)

    ax2.set_xlabel('Magnitude', fontsize=12)
    ax2.set_ylabel('Fraction classified as extended (%)', fontsize=12)
    ax2.set_title(f'{cluster_name} - Spread Model (Error-Weighted Threshold)', fontsize=13, fontweight='bold')
    ax2.set_ylim(-5, 105)
    ax2.grid(True, alpha=0.3)

    # ===== BOTTOM ROW: Fraction of total extended sources in each bin =====

    # Plot 3: Class star - distribution of extended sources
    ax3.plot(bin_centers, frac_of_total_class_star, 'o-', linewidth=2,
             markersize=6, color='steelblue')
    ax3.axvline(x=15, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
    ax3.text(15.2, max(frac_of_total_class_star)*0.9, 'Bright/Faint divide', fontsize=10, alpha=0.7)

    ax3.set_xlabel('Magnitude', fontsize=12)
    ax3.set_ylabel('Fraction of total extended sources (%)', fontsize=12)
    ax3.set_title(f'{cluster_name} - Class Star: Distribution of Extended Sources', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Spread model - distribution of extended sources
    ax4.plot(bin_centers, frac_of_total_spread_error_weighted, 'o-', linewidth=2,
             markersize=6, color='darkgreen')
    ax4.axvline(x=15, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
    ax4.text(15.2, max(frac_of_total_spread_error_weighted)*0.9, 'Bright/Faint divide', fontsize=10, alpha=0.7)

    ax4.set_xlabel('Magnitude', fontsize=12)
    ax4.set_ylabel('Fraction of total extended sources (%)', fontsize=12)
    ax4.set_title(f'{cluster_name} - Spread Model: Distribution of Extended Sources', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / f'{output_prefix}_sanity_check.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    # plt.show()

    return bin_centers, extended_fraction_class_star, extended_fraction_spread_error_weighted, counts

def main():
    """Main execution."""
    print("="*60)
    print("Morphology Classification Sanity Check")
    print("="*60)

    # Analyze Abell 2390
    print("\nAnalyzing Abell 2390...")
    parquet_file = DATA_DIR / 'abell2390_euclid_ero.parquet'
    if parquet_file.exists():
        df_2390 = load_data_parquet('abell2390_euclid_ero.parquet')
        bin_centers_2390, frac_class_2390, frac_spread_error_weighted_2390, counts_2390 = \
            analyze_morphology_by_magnitude(df_2390, 'Abell 2390', 'abell2390')
    else:
        print(f"File not found: {parquet_file}")
        print("Please run extract_and_analyze_abell2390_simplified.py first")

    # Analyze Abell 2764
    print("\nAnalyzing Abell 2764...")
    parquet_file = DATA_DIR / 'abell2764_euclid_ero.parquet'
    if parquet_file.exists():
        df_2764 = load_data_parquet('abell2764_euclid_ero.parquet')
        bin_centers_2764, frac_class_2764, frac_spread_error_weighted_2764, counts_2764 = \
            analyze_morphology_by_magnitude(df_2764, 'Abell 2764', 'abell2764')
    else:
        print(f"File not found: {parquet_file}")
        print("Please run extract_and_analyze_abell2764_simplified.py first")

    print("\n" + "="*60)
    print("Sanity check complete!")
    print("="*60)
    print("\nInterpretation guide:")
    print("✓ Good: Bright sources (mag<15) have low % extended, increases at fainter magnitudes")
    print("✗ Bad: Bright sources have high % extended (misclassification)")
    print("✗ Bad: Flat line across all magnitudes (classifier not working)")
    print("\nError bars show 1-sigma binomial uncertainty (statistical error from finite sample size):")
    print("✓ Real trend: Changes in fraction are larger than error bars")
    print("✗ Noise: Wiggles are within the error bars (just statistical fluctuation)")
    print("  Example: Small 'blips' at mag~12 might be from only a few sources in that bin")

if __name__ == '__main__':
    main()
