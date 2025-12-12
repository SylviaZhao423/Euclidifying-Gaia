#!/usr/bin/env python3
"""
Morphology classification sanity check for all Euclid ERO fields combined.

This script:
1. Loads combined parquet data from all ERO fields
2. Analyzes what fraction of sources are classified as extended vs magnitude
3. Creates 1x2 diagnostic plots with error estimates (binomial confidence intervals)
4. Validates that bright sources are mostly point-like and faint sources are mostly extended

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


def analyze_morphology_by_magnitude(df):
    """
    Analyze what fraction of sources are classified as extended vs magnitude.

    This is a sanity check: we expect bright sources to be mostly point-like,
    and faint sources to be mostly extended.

    Uses pre-calculated magnitude-dependent threshold from extraction scripts.
    Includes binomial uncertainty (1-sigma) to show statistical error due to
    finite number of sources in each magnitude bin. This helps distinguish real
    trends from statistical fluctuations caused by small sample sizes.
    """
    print(f"\n{'='*60}")
    print(f"Morphological Classification Sanity Check - All ERO Fields")
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

            # Using spread_model with error-weighted threshold
            extended_spread_error_weighted = (bin_data['spread_model'] > bin_data['spread_model_threshold']).sum()
            frac_spread, frac_spread_lower, frac_spread_upper = calculate_binomial_uncertainty(
                extended_spread_error_weighted, len(bin_data)
            )
            extended_fraction_spread_error_weighted.append(frac_spread)
            extended_fraction_spread_lower.append(frac_spread_lower)
            extended_fraction_spread_upper.append(frac_spread_upper)

            bin_centers.append((mag_bins[i] + mag_bins[i+1]) / 2)
            counts.append(len(bin_data))

            print(f"Magnitude {mag_bins[i]:.1f}-{mag_bins[i+1]:.1f}: "
                  f"{len(bin_data):5d} sources | "
                  f"class_star<0.65: {frac_class:.1%} ±{(frac_class_upper-frac_class):.1%} | "
                  f"spread(error-weighted): {frac_spread:.1%} ±{(frac_spread_upper-frac_spread):.1%}")

    # Create figure with 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ===== Plot 1: Class star classification fraction with errors =====

    ax1.errorbar(bin_centers, [x*100 for x in extended_fraction_class_star],
                 yerr=[[x*100 for x in (np.array(extended_fraction_class_star) - np.array(extended_fraction_class_star_lower))],
                       [x*100 for x in (np.array(extended_fraction_class_star_upper) - np.array(extended_fraction_class_star))]],
                 fmt='o-', linewidth=2, markersize=6, capsize=5, capthick=1.5,
                 color='steelblue', ecolor='steelblue', alpha=0.7, label='95% CI')
    ax1.axvline(x=15, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.text(15.2, 95, 'Bright/Faint divide', fontsize=10, alpha=0.7)

    ax1.set_xlabel('Magnitude', fontsize=12)
    ax1.set_ylabel('Fraction classified as extended (%)', fontsize=12)
    ax1.set_title('All ERO Fields - Class Star (Threshold = 0.65)', fontsize=13, fontweight='bold')
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)

    # ===== Plot 2: Spread model classification fraction with errors =====

    ax2.errorbar(bin_centers, [x*100 for x in extended_fraction_spread_error_weighted],
                 yerr=[[x*100 for x in (np.array(extended_fraction_spread_error_weighted) - np.array(extended_fraction_spread_lower))],
                       [x*100 for x in (np.array(extended_fraction_spread_upper) - np.array(extended_fraction_spread_error_weighted))]],
                 fmt='o-', linewidth=2, markersize=6, capsize=5, capthick=1.5,
                 color='darkgreen', ecolor='darkgreen', alpha=0.7, label='95% CI')
    ax2.axvline(x=15, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
    ax2.text(15.2, 95, 'Bright/Faint divide', fontsize=10, alpha=0.7)

    ax2.set_xlabel('Magnitude', fontsize=12)
    ax2.set_ylabel('Fraction classified as extended (%)', fontsize=12)
    ax2.set_title('All ERO Fields - Spread Model (Error-Weighted Threshold)', fontsize=13, fontweight='bold')
    ax2.set_ylim(-5, 105)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'all_ero_fields_sanity_check_with_errors.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()

    return bin_centers, extended_fraction_class_star, extended_fraction_spread_error_weighted, counts


def main():
    """Main execution."""
    print("="*60)
    print("Morphology Classification Sanity Check - All ERO Fields Combined")
    print("="*60)

    # Load combined data
    df_all = load_data_parquet('all_ero_fields.parquet')

    # Show field summary
    fields = df_all['field_name'].unique()
    print(f"\nAnalyzing {len(fields)} ERO fields combined:")
    for field in sorted(fields):
        count = len(df_all[df_all['field_name'] == field])
        print(f"  {field}: {count} sources")

    # Analyze all data together
    analyze_morphology_by_magnitude(df_all)

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
