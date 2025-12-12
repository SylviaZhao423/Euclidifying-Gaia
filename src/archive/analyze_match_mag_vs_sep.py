#!/usr/bin/env python3
"""
Create 2D histograms: Magnitude Difference vs Angular Separation
for Q1-Gaia and ERO-Gaia crossmatched sources at different matching radii.

Plots include:
- Q1-Gaia Crossmatch (1.0 arcsec) - 1st row
- ERO-Gaia Crossmatch (0.05 arcsec) - 2nd row
- ERO-Gaia Crossmatch (0.08 arcsec) - 3rd row
- ERO-Gaia Crossmatch (0.1 arcsec) - 4th row
- ERO-Gaia Crossmatch (0.2 arcsec) - 5th row
- ERO-Gaia Crossmatch (1.0 arcsec) - 6th row

Each plot shows:
- X-axis: Angular separation between positions (arcsec)
- Y-axis: Magnitude difference
- Color: Number of sources in each bin (2D histogram, log scale)
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

def load_data(filepath, dataset_name):
    """
    Load data from parquet file and calculate magnitude difference.

    Args:
        filepath: path to parquet file
        dataset_name: name of dataset (ERO or Q1)

    Returns:
        DataFrame with mag_diff and angular_separation_arcsec columns
    """
    try:
        df = pd.read_parquet(filepath)
        print(f"  ✓ Loaded {len(df):,} {dataset_name} sources")

        # Calculate magnitude difference based on dataset
        if dataset_name.startswith('Q1'):
            df['mag_diff'] = df['q1_mag_stargal_sep'] - df['gaia_mag_g']
        else:  # ERO
            df['mag_diff'] = df['ero_mag_auto'] - df['gaia_mag_g']

        # Remove NaN values
        df_clean = df[['mag_diff', 'angular_separation_arcsec']].dropna()
        print(f"  ✓ Using {len(df_clean):,} sources ({len(df) - len(df_clean):,} removed with NaN)")

        return df_clean
    except Exception as e:
        print(f"  ✗ Error loading {filepath}: {e}")
        return None


def main():
    """Main execution."""
    print("=" * 80)
    print("Creating 2D Histograms: Magnitude Difference vs Angular Separation")
    print("=" * 80)

    # Find all available ERO-Gaia subsets
    print("\nScanning for available data files...")
    q1_file = DATA_DIR / 'q1_gaia_subset.parquet'
    ero_files = sorted(DATA_DIR.glob('ero_gaia_subset_*.parquet'))

    if not q1_file.exists():
        print(f"ERROR: Q1-Gaia file not found: {q1_file}")
        return

    if not ero_files:
        print(f"WARNING: No ERO-Gaia subset files found in {DATA_DIR}")
        return

    print(f"Found {len(ero_files)} ERO-Gaia subset files:")
    for f in ero_files:
        print(f"  - {f.name}")

    # Load Q1-Gaia data
    print("\nLoading Q1-Gaia data...")
    q1_clean = load_data(q1_file, 'Q1-Gaia')
    if q1_clean is None:
        return

    # Load all ERO-Gaia subset data
    print("\nLoading ERO-Gaia subset data...")
    ero_data = {}
    for f in ero_files:
        # Extract radius from filename (e.g., 'ero_gaia_subset_0_1.parquet' -> '0.1')
        radius_str = f.stem.replace('ero_gaia_subset_', '').replace('_', '.')
        ero_data[radius_str] = load_data(f, f'ERO-Gaia ({radius_str} arcsec)')

    # Filter out None values
    ero_data = {k: v for k, v in ero_data.items() if v is not None}

    if not ero_data:
        print("ERROR: Could not load any ERO-Gaia data")
        return

    # Sort radii for consistent ordering
    radii_order = ['0.05', '0.08', '0.1', '0.2', '1.0']
    sorted_radii = [r for r in radii_order if r in ero_data]

    # Calculate y-axis limits (magnitude difference range) for consistency
    print("\nCalculating y-axis limits for consistency...")
    all_mag_diffs = []
    all_mag_diffs.extend(q1_clean['mag_diff'])
    for radius in sorted_radii:
        all_mag_diffs.extend(ero_data[radius]['mag_diff'])

    y_min = np.percentile(all_mag_diffs, 0.5)  # 0.5th percentile
    y_max = np.percentile(all_mag_diffs, 99.5)  # 99.5th percentile
    print(f"  Y-axis limits: {y_min:.3f} to {y_max:.3f} mag")

    # Create grid of subplots: 1 for Q1, then one per ERO radius
    num_plots = 1 + len(sorted_radii)
    num_rows = num_plots

    print(f"\nCreating {num_plots} plots ({num_rows} rows x 1 column)...")
    fig, axes = plt.subplots(num_rows, 1, figsize=(10, 4 * num_rows))

    # Handle single plot case
    if num_rows == 1:
        axes = [axes]

    # Store histogram objects for colorbar
    h_objects = []

    # Plot 1: Q1-Gaia
    print("  Creating Q1-Gaia plot...")
    h = axes[0].hist2d(q1_clean['angular_separation_arcsec'],
                        q1_clean['mag_diff'],
                        bins=[50, 50],
                        cmap='viridis',
                        cmin=1,
                        norm=plt.matplotlib.colors.LogNorm())
    h_objects.append(h[3])

    axes[0].set_xlabel('Angular Separation (arcsec)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('(q1_mag_stargal_sep) - (gaia_mag_g)', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Q1-Gaia Crossmatch ({len(q1_clean):,} sources)',
                      fontsize=12, fontweight='bold', pad=10)
    axes[0].set_ylim(y_min, y_max)
    axes[0].grid(True, alpha=0.3, linestyle='-')

    # Plots 2+: ERO-Gaia at different radii
    for idx, radius in enumerate(sorted_radii, 1):
        ero_clean = ero_data[radius]
        print(f"  Creating ERO-Gaia ({radius} arcsec) plot...")

        h = axes[idx].hist2d(ero_clean['angular_separation_arcsec'],
                             ero_clean['mag_diff'],
                             bins=[50, 50],
                             cmap='viridis',
                             cmin=1,
                             norm=plt.matplotlib.colors.LogNorm())
        h_objects.append(h[3])

        axes[idx].set_xlabel('Angular Separation (arcsec)', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('(ero_mag_auto) - (gaia_mag_g)', fontsize=11, fontweight='bold')
        axes[idx].set_title(f'ERO-Gaia Crossmatch ({radius} arcsec, {len(ero_clean):,} sources)',
                           fontsize=12, fontweight='bold', pad=10)
        axes[idx].set_ylim(y_min, y_max)
        axes[idx].grid(True, alpha=0.3, linestyle='-')

    # Add shared colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.08, 0.015, 0.84])
    cbar = plt.colorbar(h_objects[-1], cax=cbar_ax)
    cbar.set_label('Count (log scale)', fontsize=11, fontweight='bold')

    # Overall title
    fig.suptitle('Magnitude Difference vs Angular Separation: Crossmatch Comparison',
                 fontsize=14, fontweight='bold', y=0.995)

    # Adjust layout
    plt.subplots_adjust(left=0.10, right=0.89, top=0.96, bottom=0.06, hspace=0.4)

    # Save figure
    output_path = OUTPUT_DIR / 'mag_diff_comparison_all_radii.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {output_path}")

    plt.close()

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)

if __name__ == '__main__':
    main()
