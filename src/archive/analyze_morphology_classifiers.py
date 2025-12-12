#!/usr/bin/env python3
"""
Visualize and compare morphology classifiers from ERO, Q1, and Gaia.

Creates 2D histograms and scatter plots comparing:
- ERO class_star vs Gaia classprob_star
- ERO spread_model vs Gaia classprob_star
- Q1 point_like_prob vs Gaia classprob_star (based on MU_MAX - MAG_AUTO proxy)
- Q1 point_like_flag vs Gaia classprob_star

Q1 Classification Method (from MER PF pipeline):
- Uses MU_MAX - MAG_AUTO as a proxy for spread function
- POINT_LIKE_PROB derived from 2D calibration array (MAG, MU_MAX - MAG_AUTO)
- POINT_LIKE_FLAG assigned if POINT_LIKE_PROB > 0.96 (high purity, lower completeness)
- Extended objects typically have POINT_LIKE_PROB <= 0.7

Analyzes how well each classifier distinguishes point-like vs extended objects.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'output'

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_data():
    """Load ERO and Q1 morphology subsets."""
    print("Loading data...")

    ero_file = DATA_DIR / 'ero_gaia_subset.parquet'
    q1_file = DATA_DIR / 'q1_gaia_subset.parquet'

    ero_df = pd.read_parquet(ero_file)
    q1_df = pd.read_parquet(q1_file)

    print(f"  ERO: {len(ero_df):,} rows")
    print(f"  Q1: {len(q1_df):,} rows")

    return ero_df, q1_df


def plot_ero_classifiers(df):
    """Create 2D histograms for ERO classifiers vs Gaia."""
    print("\nCreating ERO classifier visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('ERO Morphology Classifiers vs Gaia classprob_star', fontsize=16, fontweight='bold')

    # Filter for valid data
    valid_class_star = df[['ero_class_star', 'gaia_classprob_star']].notna().all(axis=1)
    valid_spread = df[['ero_spread_model', 'gaia_classprob_star']].notna().all(axis=1)

    # Plot 1: ERO class_star vs Gaia classprob_star (2D histogram)
    ax = axes[0, 0]
    h = ax.hist2d(df.loc[valid_class_star, 'ero_class_star'],
                   df.loc[valid_class_star, 'gaia_classprob_star'],
                   bins=50, cmap='YlOrRd')
    ax.set_xlabel('ERO class_star', fontsize=11)
    ax.set_ylabel('Gaia classprob_star', fontsize=11)
    ax.set_title(f'class_star vs classprob_star (n={valid_class_star.sum():,})', fontsize=12)
    plt.colorbar(h[3], ax=ax, label='Count')

    # Plot 2: ERO class_star vs Gaia classprob_star (scatter with transparency)
    ax = axes[0, 1]
    sample = valid_class_star.copy()
    if sample.sum() > 10000:
        sample = np.random.choice(len(df), 10000, replace=False)
        sample_mask = np.zeros(len(df), dtype=bool)
        sample_mask[sample] = True
        sample = sample_mask & valid_class_star

    ax.scatter(df.loc[sample, 'ero_class_star'],
               df.loc[sample, 'gaia_classprob_star'],
               alpha=0.3, s=10, edgecolors='none')
    ax.set_xlabel('ERO class_star', fontsize=11)
    ax.set_ylabel('Gaia classprob_star', fontsize=11)
    ax.set_title(f'class_star vs classprob_star (scatter, n={sample.sum():,})', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Plot 3: ERO spread_model vs Gaia classprob_star (2D histogram)
    ax = axes[1, 0]
    # Filter for reasonable spread_model values
    valid_spread_clean = valid_spread & (df['ero_spread_model'] < 1.0)
    h = ax.hist2d(df.loc[valid_spread_clean, 'ero_spread_model'],
                   df.loc[valid_spread_clean, 'gaia_classprob_star'],
                   bins=50, cmap='YlOrRd')
    ax.set_xlabel('ERO spread_model', fontsize=11)
    ax.set_ylabel('Gaia classprob_star', fontsize=11)
    ax.set_title(f'spread_model vs classprob_star (n={valid_spread_clean.sum():,})', fontsize=12)
    plt.colorbar(h[3], ax=ax, label='Count')

    # Plot 4: ERO spread_model vs Gaia classprob_star (scatter)
    ax = axes[1, 1]
    sample = valid_spread_clean.copy()
    if sample.sum() > 10000:
        sample = np.random.choice(len(df), 10000, replace=False)
        sample_mask = np.zeros(len(df), dtype=bool)
        sample_mask[sample] = True
        sample = sample_mask & valid_spread_clean

    ax.scatter(df.loc[sample, 'ero_spread_model'],
               df.loc[sample, 'gaia_classprob_star'],
               alpha=0.3, s=10, edgecolors='none')
    ax.set_xlabel('ERO spread_model', fontsize=11)
    ax.set_ylabel('Gaia classprob_star', fontsize=11)
    ax.set_title(f'spread_model vs classprob_star (scatter, n={sample.sum():,})', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'ero_morphology_classifiers.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_q1_classifiers(df):
    """Create visualizations for Q1 classifiers vs Gaia."""
    print("\nCreating Q1 classifier visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Q1 Morphology Classifiers vs Gaia classprob_star\n(Q1 uses MU_MAX - MAG_AUTO proxy for S/G separation)',
                 fontsize=14, fontweight='bold')

    # Filter for valid data
    valid_pl_prob = df[['q1_point_like_prob', 'gaia_classprob_star']].notna().all(axis=1)
    valid_pl_flag = df[['q1_point_like_flag', 'gaia_classprob_star']].notna().all(axis=1)

    # Plot 1: Q1 point_like_prob vs Gaia classprob_star (2D histogram)
    ax = axes[0, 0]
    h = ax.hist2d(df.loc[valid_pl_prob, 'q1_point_like_prob'],
                   df.loc[valid_pl_prob, 'gaia_classprob_star'],
                   bins=50, cmap='YlGnBu')
    ax.axvline(x=0.96, color='red', linestyle='--', linewidth=2, label='POINT_LIKE_FLAG threshold (0.96)')
    ax.axvline(x=0.70, color='orange', linestyle='--', linewidth=2, label='Extended threshold (0.70)')
    ax.set_xlabel('Q1 point_like_prob', fontsize=11)
    ax.set_ylabel('Gaia classprob_star', fontsize=11)
    ax.set_title(f'point_like_prob vs classprob_star (n={valid_pl_prob.sum():,})', fontsize=12)
    plt.colorbar(h[3], ax=ax, label='Count')
    ax.legend(fontsize=9)

    # Plot 2: Q1 point_like_prob vs Gaia classprob_star (scatter)
    ax = axes[0, 1]
    sample = valid_pl_prob.copy()
    if sample.sum() > 10000:
        sample = np.random.choice(len(df), 10000, replace=False)
        sample_mask = np.zeros(len(df), dtype=bool)
        sample_mask[sample] = True
        sample = sample_mask & valid_pl_prob

    # Color by classification threshold
    colors = []
    for prob in df.loc[sample, 'q1_point_like_prob']:
        if prob > 0.96:
            colors.append('red')  # POINT_LIKE_FLAG
        elif prob > 0.70:
            colors.append('yellow')  # Intermediate
        else:
            colors.append('blue')  # Extended

    ax.scatter(df.loc[sample, 'q1_point_like_prob'],
               df.loc[sample, 'gaia_classprob_star'],
               c=colors, alpha=0.3, s=10, edgecolors='none')
    ax.axvline(x=0.96, color='red', linestyle='--', linewidth=2, label='POINT_LIKE_FLAG threshold')
    ax.axvline(x=0.70, color='orange', linestyle='--', linewidth=2, label='Extended threshold')
    ax.set_xlabel('Q1 point_like_prob', fontsize=11)
    ax.set_ylabel('Gaia classprob_star', fontsize=11)
    ax.set_title(f'point_like_prob vs classprob_star (scatter, n={sample.sum():,})', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Plot 3: Q1 point_like_flag vs Gaia classprob_star (box plot)
    ax = axes[1, 0]
    # Q1 point_like_flag: 1 = point-like, 32767 = non-point-like/unclassified
    flag_values = df.loc[valid_pl_flag, 'q1_point_like_flag'].unique()
    flag_values = sorted([v for v in flag_values if v is not None])

    data_to_plot = []
    labels = []
    for flag_val in flag_values:
        mask = (df['q1_point_like_flag'] == flag_val) & valid_pl_flag
        if mask.sum() > 0:
            data_to_plot.append(df.loc[mask, 'gaia_classprob_star'].values)
            if flag_val == 1:
                labels.append(f'Point-Like\n(n={mask.sum():,})')
            else:
                labels.append(f'Non-Point-Like\n(n={mask.sum():,})')

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors_box = ['lightcoral', 'lightblue']
    for patch, color in zip(bp['boxes'], colors_box[:len(bp['boxes'])]):
        patch.set_facecolor(color)
    ax.set_ylabel('Gaia classprob_star', fontsize=11)
    ax.set_title(f'point_like_flag vs classprob_star (Gaia comparison)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Q1 classification thresholds - probability distribution
    ax = axes[1, 1]
    pl_prob = df.loc[valid_pl_prob, 'q1_point_like_prob']
    ax.hist(pl_prob, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0.96, color='red', linestyle='--', linewidth=2, label='POINT_LIKE_FLAG (0.96)')
    ax.axvline(x=0.70, color='orange', linestyle='--', linewidth=2, label='Extended (0.70)')

    # Annotate regions
    ax.text(0.98, ax.get_ylim()[1]*0.9, 'Point-Like\nRegion', fontsize=10, ha='right',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    ax.text(0.70, ax.get_ylim()[1]*0.9, 'Extended\nRegion', fontsize=10, ha='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    ax.set_xlabel('Q1 point_like_prob', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Q1 point_like_prob Distribution & Thresholds', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'q1_morphology_classifiers.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_comparison(ero_df, q1_df):
    """Create comparison plots between ERO and Q1 classifiers."""
    print("\nCreating ERO vs Q1 comparison visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('ERO vs Q1: Morphology Classifier Comparison with Gaia', fontsize=16, fontweight='bold')

    # Plot 1: Distribution of Gaia classprob_star (ERO vs Q1)
    ax = axes[0, 0]
    ero_gaia = ero_df['gaia_classprob_star'].dropna()
    q1_gaia = q1_df['gaia_classprob_star'].dropna()

    ax.hist(ero_gaia, bins=50, alpha=0.6, label=f'ERO (n={len(ero_gaia):,})', color='coral')
    ax.hist(q1_gaia, bins=50, alpha=0.6, label=f'Q1 (n={len(q1_gaia):,})', color='steelblue')
    ax.set_xlabel('Gaia classprob_star', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Gaia classprob_star Distribution: ERO vs Q1', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Classifier correlation with Gaia (scatter)
    ax = axes[0, 1]
    ero_valid = ero_df[['ero_class_star', 'gaia_classprob_star']].notna().all(axis=1)
    q1_valid = q1_df[['q1_point_like_prob', 'gaia_classprob_star']].notna().all(axis=1)

    ero_sample = ero_valid.copy()
    if ero_sample.sum() > 5000:
        idx = np.random.choice(np.where(ero_valid)[0], 5000, replace=False)
        ero_sample = np.zeros(len(ero_df), dtype=bool)
        ero_sample[idx] = True

    q1_sample = q1_valid.copy()
    if q1_sample.sum() > 5000:
        idx = np.random.choice(np.where(q1_valid)[0], 5000, replace=False)
        q1_sample = np.zeros(len(q1_df), dtype=bool)
        q1_sample[idx] = True

    ax.scatter(ero_df.loc[ero_sample, 'ero_class_star'],
               ero_df.loc[ero_sample, 'gaia_classprob_star'],
               alpha=0.3, s=10, label='ERO class_star', color='coral', edgecolors='none')
    ax.scatter(q1_df.loc[q1_sample, 'q1_point_like_prob'],
               q1_df.loc[q1_sample, 'gaia_classprob_star'],
               alpha=0.3, s=10, label='Q1 point_like_prob', color='steelblue', edgecolors='none')
    ax.set_xlabel('Euclid Classifier', fontsize=11)
    ax.set_ylabel('Gaia classprob_star', fontsize=11)
    ax.set_title('Euclid Classifiers vs Gaia (sample)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: SNR comparison (log scale)
    ax = axes[1, 0]
    ero_snr = ero_df['ero_snr_log'].dropna()
    q1_snr = q1_df['gaia_snr_log'].dropna()

    ax.hist(ero_snr, bins=50, alpha=0.6, label=f'ERO SNR (n={len(ero_snr):,})', color='coral')
    ax.hist(q1_snr, bins=50, alpha=0.6, label=f'Q1/Gaia SNR (n={len(q1_snr):,})', color='steelblue')
    ax.set_xlabel('log10(SNR)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Signal-to-Noise Ratio Comparison', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Angular separation distribution
    ax = axes[1, 1]
    ero_sep = ero_df['angular_separation_arcsec'].dropna()
    q1_sep = q1_df['angular_separation_arcsec'].dropna()

    ax.hist(ero_sep, bins=50, alpha=0.6, label=f'ERO-Gaia (n={len(ero_sep):,})', color='coral')
    ax.hist(q1_sep, bins=50, alpha=0.6, label=f'Q1-Gaia (n={len(q1_sep):,})', color='steelblue')
    ax.set_xlabel('Angular Separation (arcsec)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Cross-Match Angular Separation', fontsize=12)
    ax.set_xlim(-0.5, 2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'ero_vs_q1_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def print_statistics(ero_df, q1_df):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("Summary Statistics & Classification Analysis")
    print("="*70)

    print(f"\nERO Dataset (n={len(ero_df):,}):")
    print(f"  class_star - Mean: {ero_df['ero_class_star'].mean():.4f}, Median: {ero_df['ero_class_star'].median():.4f}")
    print(f"              Range: {ero_df['ero_class_star'].min():.4f} - {ero_df['ero_class_star'].max():.4f}")
    print(f"  spread_model - Mean: {ero_df['ero_spread_model'].mean():.4f}, Median: {ero_df['ero_spread_model'].median():.4f}")
    print(f"  Gaia classprob_star - Mean: {ero_df['gaia_classprob_star'].mean():.4f}, Median: {ero_df['gaia_classprob_star'].median():.4f}")

    print(f"\nQ1 Dataset (n={len(q1_df):,}):")
    print(f"  point_like_prob - Mean: {q1_df['q1_point_like_prob'].mean():.4f}, Median: {q1_df['q1_point_like_prob'].median():.4f}")
    print(f"                    Range: {q1_df['q1_point_like_prob'].min():.4f} - {q1_df['q1_point_like_prob'].max():.4f}")

    # Q1 classification breakdown
    q1_pl_flag = q1_df['q1_point_like_flag'].value_counts()
    print(f"  point_like_flag counts: {dict(q1_pl_flag)}")

    # Count sources by Q1 thresholds
    pl_prob = q1_df['q1_point_like_prob'].notna()
    pl_count = (q1_df['q1_point_like_prob'] > 0.96).sum()
    ext_count = (q1_df['q1_point_like_prob'] <= 0.70).sum()
    intermediate = ((q1_df['q1_point_like_prob'] > 0.70) & (q1_df['q1_point_like_prob'] <= 0.96)).sum()

    print(f"\n  Q1 Classification by probability threshold:")
    print(f"    Point-like (PROB > 0.96): {pl_count:,} ({100*pl_count/len(q1_df):.1f}%)")
    print(f"    Intermediate (0.70 < PROB ≤ 0.96): {intermediate:,} ({100*intermediate/len(q1_df):.1f}%)")
    print(f"    Extended (PROB ≤ 0.70): {ext_count:,} ({100*ext_count/len(q1_df):.1f}%)")

    print(f"\n  Gaia classprob_star - Mean: {q1_df['gaia_classprob_star'].mean():.4f}, Median: {q1_df['gaia_classprob_star'].median():.4f}")

    print(f"\nCross-Match Quality:")
    print(f"  ERO-Gaia angular separation - Mean: {ero_df['angular_separation_arcsec'].mean():.4f} arcsec")
    print(f"                                 Median: {ero_df['angular_separation_arcsec'].median():.4f} arcsec")
    print(f"  Q1-Gaia angular separation - Mean: {q1_df['angular_separation_arcsec'].mean():.4f} arcsec")
    print(f"                                Median: {q1_df['angular_separation_arcsec'].median():.4f} arcsec")

    print(f"\nGaia Classification Agreement:")
    print(f"  ERO sources with Gaia classprob_star > 0.9: {(ero_df['gaia_classprob_star'] > 0.9).sum():,} ({100*(ero_df['gaia_classprob_star'] > 0.9).sum()/len(ero_df):.1f}%)")
    print(f"  Q1 sources with Gaia classprob_star > 0.9: {(q1_df['gaia_classprob_star'] > 0.9).sum():,} ({100*(q1_df['gaia_classprob_star'] > 0.9).sum()/len(q1_df):.1f}%)")


def main():
    """Main execution."""
    print("="*70)
    print("Visualize Morphology Classifiers: ERO vs Q1 vs Gaia")
    print("="*70)

    # Load data
    ero_df, q1_df = load_data()

    # Create visualizations
    plot_ero_classifiers(ero_df)
    plot_q1_classifiers(q1_df)
    plot_comparison(ero_df, q1_df)

    # Print statistics
    print_statistics(ero_df, q1_df)

    print("\n" + "="*70)
    print("Visualization complete!")
    print("="*70)
    print(f"\nOutput plots saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
