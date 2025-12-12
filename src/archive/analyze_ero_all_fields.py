#!/usr/bin/env python3
"""
Morphology classification sanity check for all Euclid ERO fields combined.

This script:
1. Loads combined parquet data from all ERO fields
2. Analyzes what fraction of sources are classified as extended vs magnitude
3. Creates 2x2 diagnostic plots using all 17 ERO fields together
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


def analyze_morphology_by_magnitude(df):
    """
    Analyze what fraction of sources are classified as extended vs magnitude.

    This is a sanity check: we expect bright sources to be mostly point-like,
    and faint sources to be mostly extended.

    Uses pre-calculated magnitude-dependent threshold from extraction scripts.
    """
    print(f"\n{'='*60}")
    print(f"Morphological Classification Sanity Check - All ERO Fields")
    print(f"{'='*60}")

    # Define magnitude bins
    mag_bins = np.arange(10, 28, 0.5)  # 0.5 magnitude bins

    # For each bin, calculate fraction classified as extended
    extended_fraction_class_star = []
    extended_fraction_spread_error_weighted = []
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
            frac_class = extended_class / len(bin_data)
            extended_fraction_class_star.append(frac_class)
            extended_count_class_star.append(extended_class)

            # Using spread_model with error-weighted threshold
            extended_spread_error_weighted = (bin_data['spread_model'] > bin_data['spread_model_threshold']).sum()
            frac_spread_error_weighted = extended_spread_error_weighted / len(bin_data)
            extended_fraction_spread_error_weighted.append(frac_spread_error_weighted)
            extended_count_spread_error_weighted.append(extended_spread_error_weighted)

            bin_centers.append((mag_bins[i] + mag_bins[i+1]) / 2)
            counts.append(len(bin_data))

            print(f"Magnitude {mag_bins[i]:.1f}-{mag_bins[i+1]:.1f}: "
                  f"{len(bin_data):5d} sources | "
                  f"class_star<0.65: {frac_class:.1%} | "
                  f"spread(error-weighted): {frac_spread_error_weighted:.1%}")

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

    # Plot 1: Class star classification fraction
    ax1.plot(bin_centers, [x*100 for x in extended_fraction_class_star], 'o-', linewidth=2,
             markersize=6, color='steelblue')
    ax1.axvline(x=15, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.text(15.2, 95, 'Bright/Faint divide', fontsize=10, alpha=0.7)

    ax1.set_xlabel('Magnitude', fontsize=12)
    ax1.set_ylabel('Fraction classified as extended (%)', fontsize=12)
    ax1.set_title('All ERO Fields - Class Star (Threshold = 0.65)', fontsize=13, fontweight='bold')
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Spread model classification fraction
    ax2.plot(bin_centers, [x*100 for x in extended_fraction_spread_error_weighted], 'o-', linewidth=2,
             markersize=6, color='darkgreen')
    ax2.axvline(x=15, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
    ax2.text(15.2, 95, 'Bright/Faint divide', fontsize=10, alpha=0.7)

    ax2.set_xlabel('Magnitude', fontsize=12)
    ax2.set_ylabel('Fraction classified as extended (%)', fontsize=12)
    ax2.set_title('All ERO Fields - Spread Model (Error-Weighted Threshold)', fontsize=13, fontweight='bold')
    ax2.set_ylim(-5, 105)
    ax2.grid(True, alpha=0.3)

    # ===== BOTTOM ROW: Fraction of total extended sources in each bin =====

    # Plot 3: Class star - distribution of extended sources
    ax3.plot(bin_centers, frac_of_total_class_star, 'o-', linewidth=2,
             markersize=6, color='steelblue')
    ax3.axvline(x=15, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
    if max(frac_of_total_class_star) > 0:
        ax3.text(15.2, max(frac_of_total_class_star)*0.9, 'Bright/Faint divide', fontsize=10, alpha=0.7)

    ax3.set_xlabel('Magnitude', fontsize=12)
    ax3.set_ylabel('Fraction of total extended sources (%)', fontsize=12)
    ax3.set_title('All ERO Fields - Class Star: Distribution of Extended Sources', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Spread model - distribution of extended sources
    ax4.plot(bin_centers, frac_of_total_spread_error_weighted, 'o-', linewidth=2,
             markersize=6, color='darkgreen')
    ax4.axvline(x=15, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
    if max(frac_of_total_spread_error_weighted) > 0:
        ax4.text(15.2, max(frac_of_total_spread_error_weighted)*0.9, 'Bright/Faint divide', fontsize=10, alpha=0.7)

    ax4.set_xlabel('Magnitude', fontsize=12)
    ax4.set_ylabel('Fraction of total extended sources (%)', fontsize=12)
    ax4.set_title('All ERO Fields - Spread Model: Distribution of Extended Sources', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'all_ero_fields_sanity_check.png'
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


if __name__ == '__main__':
    main()
