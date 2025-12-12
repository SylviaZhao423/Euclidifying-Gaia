#!/usr/bin/env python3
"""
Plot histogram: Number of sources vs Angular Separation
for Q1-Gaia crossmatched sources.

Plot shows:
- X-axis: Angular separation between Q1 and Gaia positions (arcsec)
- Y-axis: Number of sources in each bin
- Bar chart with histogram visualization

Uses q1_gaia_subset.parquet (352k sources).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'output'

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    """Main execution."""
    print("=" * 80)
    print("Creating Histogram: Number of Sources vs Angular Separation (Q1-Gaia)")
    print("=" * 80)

    # Load data
    print("\nLoading Q1-Gaia data...")
    data_file = DATA_DIR / 'q1_gaia_subset.parquet'
    df = pd.read_parquet(data_file)
    print(f"✓ Loaded {len(df):,} sources")

    # Remove NaN values
    df_clean = df[['angular_separation_arcsec']].dropna()
    print(f"✓ Using {len(df_clean):,} sources ({len(df) - len(df_clean):,} removed with NaN)")

    # Print statistics
    print("\nData Statistics:")
    print(f"  Angular Separation (arcsec):")
    print(f"    Min: {df_clean['angular_separation_arcsec'].min():.4f}")
    print(f"    Max: {df_clean['angular_separation_arcsec'].max():.4f}")
    print(f"    Median: {df_clean['angular_separation_arcsec'].median():.4f}")
    print(f"    Mean: {df_clean['angular_separation_arcsec'].mean():.4f}")

    # Create histogram with dual scales
    print("\nCreating histograms...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Get data and limits
    data = df_clean['angular_separation_arcsec']
    max_sep = data.max()
    median_sep = data.median()
    mean_sep = data.mean()
    x_lim = (0, max_sep * 1.05)

    # Create legend label with statistics
    legend_label = (f'Q1-Gaia (n={len(df_clean):,})\n'
                    f'Max: {max_sep:.4f}″, Median: {median_sep:.4f}″, Mean: {mean_sep:.4f}″')

    # Left plot: Linear scale
    ax1.hist(data, bins=50, color='steelblue', alpha=0.6, edgecolor='none')
    ax1.set_xlabel('Angular Separation (arcsec)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Count (linear scale)', fontsize=11, fontweight='bold')
    ax1.set_title('Q1-Gaia: Linear Scale',
                  fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend([legend_label], fontsize=10, loc='upper right')

    # Right plot: Log scale
    ax2.hist(data, bins=50, color='steelblue', alpha=0.6, edgecolor='none')
    ax2.set_xlabel('Angular Separation (arcsec)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Count (log scale)', fontsize=11, fontweight='bold')
    ax2.set_title('Q1-Gaia: Log Scale',
                  fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.set_xlim(x_lim)
    ax2.grid(True, alpha=0.3)
    ax2.legend([legend_label], fontsize=10, loc='upper right')

    # Add overall title
    fig.suptitle('Q1-Gaia Crossmatch - Angular Separation Distribution',
                 fontsize=13, fontweight='bold', y=1.00)

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path = OUTPUT_DIR / 'q1_angular_sep_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: {output_path}")

    plt.close()

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)

if __name__ == '__main__':
    main()
