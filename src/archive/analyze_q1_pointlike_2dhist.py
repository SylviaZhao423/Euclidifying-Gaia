#!/usr/bin/env python3
"""
Analyze 2D distributions of Q1 point_like_prob classifier.

Creates 2D histograms showing:
1. point_like_prob vs magnitude
2. point_like_prob vs SNR

Uses logarithmic color scaling to visualize both dense and sparse regions.
Helps understand how Q1's morphology classifier behaves across different
source properties.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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


def analyze_2d_distributions(df, cluster_name, output_prefix):
    """Create 2D histograms of Q1 morphology classifiers."""
    print(f"\n{'='*60}")
    print(f"Q1 2D Distribution Analysis - {cluster_name}")
    print(f"{'='*60}")

    # Check for required columns
    required_cols = ['magnitude', 'point_like_prob', 'snr', 'snr_log']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"WARNING: Missing columns {missing_cols}")
        return

    # Remove NaN values
    df_valid = df[df[required_cols].notna().all(axis=1)].copy()
    print(f"\nValid measurements: {len(df_valid)} / {len(df)} sources ({100*len(df_valid)/len(df):.1f}%)")

    # Create figure with 1x2 subplots for point_like_prob only
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: point_like_prob vs magnitude (2D histogram)
    ax1 = axes[0]
    h1 = ax1.hist2d(
        df_valid['magnitude'],
        df_valid['point_like_prob'],
        bins=[50, 50],
        cmap='viridis',
        norm=LogNorm()
    )
    ax1.set_xlabel('Magnitude', fontsize=12)
    ax1.set_ylabel('point_like_prob', fontsize=12)
    ax1.set_title(f'{cluster_name} - point_like_prob vs Magnitude', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1])
    cbar1 = plt.colorbar(h1[3], ax=ax1)
    cbar1.set_label('Count (log scale)')
    ax1.grid(True, alpha=0.3)

    # Plot 2: point_like_prob vs SNR (2D histogram)
    ax2 = axes[1]
    h2 = ax2.hist2d(
        df_valid['snr_log'],
        df_valid['point_like_prob'],
        bins=[50, 50],
        cmap='viridis',
        norm=LogNorm()
    )
    ax2.set_xlabel('log10(SNR)', fontsize=12)
    ax2.set_ylabel('point_like_prob', fontsize=12)
    ax2.set_title(f'{cluster_name} - point_like_prob vs SNR', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 1])
    cbar2 = plt.colorbar(h2[3], ax=ax2)
    cbar2.set_label('Count (log scale)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'q1_prob_2d_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()

    # Print statistics
    print(f"\nPoint-like Probability Statistics:")
    print(f"  Mean: {df_valid['point_like_prob'].mean():.4f}")
    print(f"  Median: {df_valid['point_like_prob'].median():.4f}")
    print(f"  Std: {df_valid['point_like_prob'].std():.4f}")
    print(f"  Range: {df_valid['point_like_prob'].min():.4f} - {df_valid['point_like_prob'].max():.4f}")

    # Correlation analysis
    print(f"\nCorrelation Analysis:")
    correlation_mag_prob = df_valid[['magnitude', 'point_like_prob']].corr().iloc[0, 1]
    correlation_snr_prob = df_valid[['snr_log', 'point_like_prob']].corr().iloc[0, 1]
    print(f"  Magnitude vs Point-like Probability: {correlation_mag_prob:.4f}")
    print(f"  log10(SNR) vs Point-like Probability: {correlation_snr_prob:.4f}")


def main():
    """Main execution."""
    print("="*60)
    print("Q1 2D Distribution Analysis")
    print("="*60)

    parquet_file = DATA_DIR / 'all_q1_fields.parquet'

    if parquet_file.exists():
        print("\nLooking for all Q1 fields combined data...")
        df = load_data_parquet('all_q1_fields.parquet')
        if df is not None:
            # Show field summary
            if 'field_name' in df.columns:
                fields = df['field_name'].unique()
                print(f"\nAnalyzing {len(fields)} Q1 fields combined:")
                for field in sorted(fields):
                    count = len(df[df['field_name'] == field])
                    print(f"  {field}: {count} sources")

            analyze_2d_distributions(df, 'All Q1 Fields', 'all_q1_fields')
    else:
        print(f"File not found: {parquet_file}")
        print("Please run extract_all_q1_fields.py first to generate all_q1_fields.parquet")

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    print("\nInterpretation guide:")
    print("✓ point_like_prob should vary smoothly with magnitude and SNR")
    print("✓ Brighter sources (low mag) tend to be more point-like (high prob)")
    print("✓ Higher SNR sources should have more reliable classification")
    print("✗ If point_like_prob shows no correlation with magnitude/SNR, classifier may not work well")


if __name__ == '__main__':
    main()
