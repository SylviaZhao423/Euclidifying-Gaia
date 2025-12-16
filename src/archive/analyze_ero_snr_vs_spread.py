#!/usr/bin/env python3
"""
Analyze relationship between SNR and spread_model classifier.

This script validates the spread_model implementation by comparing it with
Signal-to-Noise Ratio (SNR).

Key insight: Sources classified as extended (spread_model > threshold) should
show different SNR distributions than point-like sources, and SNR should
correlate with measurement reliability.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'output'

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data_parquet(filename):
    """Load data from Parquet format."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return None
    df = pd.read_parquet(filepath)
    print(f"Loaded {len(df)} sources from {filepath}")
    return df


def analyze_snr_vs_spread_model(df, cluster_name, output_prefix):
    """
    Analyze SNR vs spread_model relationship.

    Creates:
    1. 2D histogram of SNR vs spread_model for all sources
    2. 2D histogram of SNR vs spread_model for extended sources only
    3. SNR distribution plots comparing point-like vs extended sources
    4. Statistical summary tables
    """
    print(f"\n{'='*60}")
    print(f"SNR vs Spread Model Analysis - {cluster_name}")
    print(f"{'='*60}")

    # Check for required columns
    required_cols = ['snr', 'snr_log', 'spread_model', 'spread_model_threshold', 'class_star']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"WARNING: Missing columns {missing_cols}")
        print("Parquet file may need to be regenerated with SNR columns")
        return

    # Remove NaN values for SNR
    df_valid = df[df['snr'].notna() & df['snr_log'].notna()].copy()
    print(f"\nValid SNR measurements: {len(df_valid)} / {len(df)} sources ({100*len(df_valid)/len(df):.1f}%)")

    # For visualization: restrict to reasonable spread_model range [-0.1, 0.1]
    # This focuses on the physically meaningful range and removes capped/saturated values
    df_valid = df_valid[(df_valid['spread_model'] >= -0.1) & (df_valid['spread_model'] <= 0.1)].copy()
    print(f"After restricting spread_model to [-0.1, 0.1] for plotting: {len(df_valid)} sources ({100*len(df_valid)/len(df):.1f}% of original)")

    # Classify sources based on spread_model
    df_valid['extended_spread'] = df_valid['spread_model'] > df_valid['spread_model_threshold']
    df_valid['extended_class_star'] = df_valid['class_star'] < 0.65

    n_extended_spread = df_valid['extended_spread'].sum()
    n_extended_class = df_valid['extended_class_star'].sum()
    print(f"\nClassification counts (with valid SNR):")
    print(f"  Extended (spread_model): {n_extended_spread}")
    print(f"  Extended (class_star): {n_extended_class}")
    print(f"  Point-like (spread_model): {(~df_valid['extended_spread']).sum()}")
    print(f"  Point-like (class_star): {(~df_valid['extended_class_star']).sum()}")

    # Calculate SNR statistics by classification
    print(f"\nSNR statistics by spread_model classification:")
    for is_extended in [False, True]:
        mask = df_valid['extended_spread'] == is_extended
        snr_vals = df_valid.loc[mask, 'snr'].values
        classification = "Extended" if is_extended else "Point-like"
        if len(snr_vals) > 0:
            print(f"  {classification}:")
            print(f"    Count: {len(snr_vals)}")
            print(f"    SNR mean: {snr_vals.mean():.2f}")
            print(f"    SNR median: {np.median(snr_vals):.2f}")
            print(f"    SNR std: {snr_vals.std():.2f}")
            print(f"    SNR range: {snr_vals.min():.2f} - {snr_vals.max():.2f}")

    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

    # TOP ROW: 2D histograms (density visualization)

    # Plot 1: 2D histogram - SNR vs spread_model (all sources)
    h1 = ax1.hist2d(
        df_valid['spread_model'],
        df_valid['snr_log'],
        bins=[80, 80],
        cmap='viridis',
        norm=plt.matplotlib.colors.LogNorm()
    )
    ax1.axvline(x=0.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.set_xlabel('SPREAD_MODEL', fontsize=12)
    ax1.set_ylabel('log10(SNR)', fontsize=12)
    ax1.set_title(f'{cluster_name} - SNR vs Spread Model (All Sources)',
                  fontsize=13, fontweight='bold')
    ax1.set_xlim([-0.1, 0.1])
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(h1[3], ax=ax1)
    cbar1.set_label('Count (log scale)')

    # Plot 2: 2D histogram - SNR vs spread_model (extended sources only)
    df_extended = df_valid[df_valid['extended_spread']]
    if len(df_extended) > 0:
        h2 = ax2.hist2d(
            df_extended['spread_model'],
            df_extended['snr_log'],
            bins=[80, 80],
            cmap='viridis',
            norm=plt.matplotlib.colors.LogNorm()
        )
        ax2.axvline(x=0.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax2.set_xlabel('SPREAD_MODEL', fontsize=12)
        ax2.set_ylabel('log10(SNR)', fontsize=12)
        ax2.set_title(f'{cluster_name} - SNR vs Spread Model (Extended Only)',
                      fontsize=13, fontweight='bold')
        ax2.set_xlim([-0.1, 0.1])
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        cbar2 = plt.colorbar(h2[3], ax=ax2)
        cbar2.set_label('Count (log scale)')
    else:
        ax2.text(0.5, 0.5, 'No extended sources', ha='center', va='center',
                transform=ax2.transAxes, fontsize=14)

    # BOTTOM ROW: Scatter plots (individual sources with higher resolution)

    # Plot 3: Scatter plot - all sources with classification coloring
    point_like = df_valid[~df_valid['extended_spread']]

    # Use small dots and transparency for clear visualization
    # rasterized=True improves PDF rendering and reduces file size
    ax3.scatter(point_like['spread_model'], point_like['snr'],
               alpha=0.25, s=1.5, color='blue', label=f'Point-like (n={len(point_like)})',
               rasterized=True, marker='o')
    ax3.scatter(df_extended['spread_model'], df_extended['snr'],
               alpha=0.25, s=1.5, color='red', label=f'Extended (n={len(df_extended)})',
               rasterized=True, marker='o')
    ax3.axvline(x=0.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_xlabel('SPREAD_MODEL', fontsize=12)
    ax3.set_ylabel('SNR', fontsize=12)
    ax3.set_title(f'{cluster_name} - Scatter Plot (All Sources)',
                  fontsize=13, fontweight='bold')
    ax3.set_xlim([-0.1, 0.1])
    ax3.set_yscale('log')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Scatter plot - extended sources only
    if len(df_extended) > 0:
        ax4.scatter(df_extended['spread_model'], df_extended['snr'],
                   alpha=0.25, s=1.5, color='red', rasterized=True, marker='o')
        ax4.axvline(x=0.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax4.set_xlabel('SPREAD_MODEL', fontsize=12)
        ax4.set_ylabel('SNR', fontsize=12)
        ax4.set_title(f'{cluster_name} - Scatter Plot (Extended Only)',
                      fontsize=13, fontweight='bold')
        ax4.set_xlim([-0.1, 0.1])
        ax4.set_yscale('log')
        ax4.legend(loc='upper left', fontsize=10)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No extended sources', ha='center', va='center',
                transform=ax4.transAxes, fontsize=14)

    plt.tight_layout()
    output_path = OUTPUT_DIR / f'{output_prefix}_snr_vs_spread_model.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()

    # Additional validation: Check spread_model behavior at different SNR thresholds
    print(f"\n\nValidation: Spread Model behavior at different SNR thresholds")
    print(f"{'SNR threshold':<20} {'% Extended (spread)':<25} {'% Extended (class_star)':<25}")
    print(f"{'-'*70}")

    snr_thresholds = [3, 5, 10, 20, 50, 100]
    for snr_thresh in snr_thresholds:
        mask = df_valid['snr'] >= snr_thresh
        if mask.sum() > 0:
            pct_extended_spread = 100 * df_valid.loc[mask, 'extended_spread'].mean()
            pct_extended_class = 100 * df_valid.loc[mask, 'extended_class_star'].mean()
            print(f"SNR >= {snr_thresh:<17} {pct_extended_spread:>6.1f}%                 {pct_extended_class:>6.1f}%")

    return df_valid


def main():
    """Main execution."""
    print("="*60)
    print("SNR vs Spread Model Analysis")
    print("="*60)

    # Check for all_ero_fields data first
    print("\nLooking for all ERO fields combined data...")
    parquet_file = DATA_DIR / 'all_ero_fields.parquet'

    if parquet_file.exists():
        print("Found all_ero_fields.parquet - analyzing combined data")
        df_all = load_data_parquet('all_ero_fields.parquet')
        if df_all is not None:
            # Show field summary
            fields = df_all['field_name'].unique()
            print(f"\nAnalyzing {len(fields)} ERO fields combined:")
            for field in sorted(fields):
                count = len(df_all[df_all['field_name'] == field])
                print(f"  {field}: {count} sources")

            analyze_snr_vs_spread_model(df_all, 'All ERO Fields', 'all_ero_fields')
    else:
        print(f"File not found: {parquet_file}")
        print("Running individual cluster analysis instead...\n")

        # Analyze Abell 2390
        print("\nAnalyzing Abell 2390...")
        parquet_file = DATA_DIR / 'abell2390_euclid_ero.parquet'
        if parquet_file.exists():
            df_2390 = load_data_parquet('abell2390_euclid_ero.parquet')
            if df_2390 is not None:
                analyze_snr_vs_spread_model(df_2390, 'Abell 2390', 'abell2390')
        else:
            print(f"File not found: {parquet_file}")
            print("Please run extract_and_analyze_abell2390.py first (will regenerate with SNR)")

        # Analyze Abell 2764
        print("\n\nAnalyzing Abell 2764...")
        parquet_file = DATA_DIR / 'abell2764_euclid_ero.parquet'
        if parquet_file.exists():
            df_2764 = load_data_parquet('abell2764_euclid_ero.parquet')
            if df_2764 is not None:
                analyze_snr_vs_spread_model(df_2764, 'Abell 2764', 'abell2764')
        else:
            print(f"File not found: {parquet_file}")
            print("Please run extract_and_analyze_abell2764.py first (will regenerate with SNR)")

    print("\n" + "="*60)
    print("SNR analysis complete!")
    print("="*60)
    print("\nInterpretation guide (SExtractor Figure 10 reference):")
    print("✓ Good spread_model: Extended sources cluster at higher spread_model values")
    print("✓ Clear separation: Point-like sources near 0, extended sources at higher values")
    print("✓ Reliable implementation: Different SNR distributions between point-like and extended")
    print("✗ Problem: Extended sources distributed across all spread_model values (threshold issue)")
    print("✗ Problem: No SNR difference between point-like and extended sources (misclassification)")
    print("\nNote: At low SNR (<5), spread_model becomes unreliable (measurement noise dominates)")
    print("Reference: https://sextractor.readthedocs.io/en/latest/Model.html#spread-model-def")


if __name__ == '__main__':
    main()
