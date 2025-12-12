#!/usr/bin/env python3
"""
Analyze distribution of Q1 extended_flag and extended_prob.

Compares sources with actual extended flag data vs NaN values:
1. extended_flag vs magnitude
2. extended_flag vs SNR
3. extended_prob distribution comparison

This helps understand what extended source data is available in Q1 catalog.
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


def analyze_extended_flag_distribution(df):
    """Analyze Q1 extended flag distributions."""
    print(f"\n{'='*60}")
    print(f"Q1 Extended Flag Distribution Analysis")
    print(f"{'='*60}")

    # Check for required columns
    required_cols = ['magnitude', 'extended_flag', 'snr', 'snr_log']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"WARNING: Missing columns {missing_cols}")
        return

    # Remove NaN values (extended_prob may be all NaN, so don't require it)
    df_valid = df[df[required_cols].notna().all(axis=1)].copy()
    print(f"\nValid measurements: {len(df_valid)} / {len(df)} sources ({100*len(df_valid)/len(df):.1f}%)")

    # Check if extended_prob has any valid data
    extended_prob_valid = df_valid['extended_prob'].notna().sum()
    print(f"Extended probability valid values: {extended_prob_valid} / {len(df_valid)} ({100*extended_prob_valid/len(df_valid):.1f}%)")

    if extended_prob_valid == 0:
        print("\nWARNING: extended_prob is all NaN in Q1 catalog!")
        print("Analyzing extended_flag only (no extended_prob data available).\n")

    # Flag value counts
    print(f"\nExtended flag distribution:")
    print(df_valid['extended_flag'].value_counts().sort_index())

    # Classify sentinel values
    df_valid['has_extended_flag'] = df_valid['extended_flag'] != 32767

    print(f"\nFlag data availability:")
    print(f"  Sources with extended_flag data: {df_valid['has_extended_flag'].sum()} ({100*df_valid['has_extended_flag'].sum()/len(df_valid) if len(df_valid) > 0 else 0:.2f}%)")

    df_with_flag = df_valid[df_valid['has_extended_flag']]
    df_no_flag = df_valid[~df_valid['has_extended_flag']]

    # Determine number of plots based on data availability
    if extended_prob_valid == 0:
        # Only 2 plots if extended_prob is all NaN
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        n_plots = 2
    else:
        # 3 plots if extended_prob has data
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        n_plots = 3

    # Plot 1: Extended flag vs magnitude (overlapping histograms with alpha blending)
    ax1 = axes[0]
    # Draw NaN first (larger dataset), then flag on top (smaller dataset)
    ax1.hist(df_no_flag['magnitude'], bins=30, alpha=0.6,
             label=f'NaN (N={len(df_no_flag)}, {len(df_no_flag)/len(df_valid)*100:.2f}%)',
             color='#ff7f0e')
    if len(df_with_flag) > 0:
        ax1.hist(df_with_flag['magnitude'], bins=30, alpha=0.8,
                 label=f'flag=1 (N={len(df_with_flag)}, {len(df_with_flag)/len(df_valid)*100:.2f}%)',
                 color='#d62728')
    ax1.set_xlabel('Magnitude', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('extended_flag vs Magnitude', fontsize=13, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')

    # Plot 2: Extended flag vs SNR (overlapping histograms)
    ax2 = axes[1]
    ax2.hist(np.log10(df_no_flag['snr'].values + 1), bins=30, alpha=0.6,
             label=f'NaN (N={len(df_no_flag)}, {len(df_no_flag)/len(df_valid)*100:.2f}%)',
             color='#ff7f0e')
    if len(df_with_flag) > 0:
        ax2.hist(np.log10(df_with_flag['snr'].values + 1), bins=30, alpha=0.8,
                 label=f'flag=1 (N={len(df_with_flag)}, {len(df_with_flag)/len(df_valid)*100:.2f}%)',
                 color='#d62728')
    ax2.set_xlabel('log10(SNR)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('extended_flag vs SNR', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    # Plot 3: Extended probability distribution (only if data exists)
    if extended_prob_valid > 0 and n_plots == 3:
        ax3 = axes[2]
        # Remove NaN values for this plot
        df_no_flag_prob = df_no_flag.dropna(subset=['extended_prob'])
        df_with_flag_prob = df_with_flag.dropna(subset=['extended_prob'])
        ax3.hist(df_no_flag_prob['extended_prob'], bins=50, alpha=0.6,
                 label=f'NaN (N={len(df_no_flag_prob)}, {len(df_no_flag_prob)/len(df_valid)*100:.2f}%)',
                 color='#ff7f0e')
        if len(df_with_flag_prob) > 0:
            ax3.hist(df_with_flag_prob['extended_prob'], bins=50, alpha=0.8,
                     label=f'flag=1 (N={len(df_with_flag_prob)}, {len(df_with_flag_prob)/len(df_valid)*100:.2f}%)',
                     color='#d62728')
        ax3.set_xlabel('extended_prob', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.set_title('extended_prob Distribution', fontsize=13, fontweight='bold')
        ax3.set_yscale('log')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'q1_extended_flags_histogram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()

    # Additional statistics
    print(f"\n\nStatistics for sources WITH extended_flag data:")
    print(f"  Magnitude range: {df_with_flag['magnitude'].min():.2f} - {df_with_flag['magnitude'].max():.2f}")
    print(f"  Magnitude mean: {df_with_flag['magnitude'].mean():.2f}")
    print(f"  SNR range: {df_with_flag['snr'].min():.2f} - {df_with_flag['snr'].max():.2f}")
    print(f"  SNR mean: {df_with_flag['snr'].mean():.2f}")
    print(f"  Extended probability mean: {df_with_flag['extended_prob'].mean():.3f}")

    print(f"\nStatistics for sources WITH NaN value (32767):")
    print(f"  Magnitude range: {df_no_flag['magnitude'].min():.2f} - {df_no_flag['magnitude'].max():.2f}")
    print(f"  Magnitude mean: {df_no_flag['magnitude'].mean():.2f}")
    print(f"  SNR range: {df_no_flag['snr'].min():.2f} - {df_no_flag['snr'].max():.2f}")
    print(f"  SNR mean: {df_no_flag['snr'].mean():.2f}")
    print(f"  Extended probability mean: {df_no_flag['extended_prob'].mean():.3f}")


def main():
    """Main execution."""
    print("="*60)
    print("Q1 Extended Flag Distribution Analysis")
    print("="*60)

    parquet_file = DATA_DIR / 'all_q1_fields.parquet'

    if parquet_file.exists():
        df = load_data_parquet('all_q1_fields.parquet')
        if df is not None:
            analyze_extended_flag_distribution(df)
    else:
        print(f"File not found: {parquet_file}")
        print("Please run extract_all_q1_fields.py first")

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == '__main__':
    main()
