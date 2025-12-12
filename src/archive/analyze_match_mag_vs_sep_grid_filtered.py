#!/usr/bin/env python3
"""
Create a grid of 2D histograms comparing Q1-Gaia and ERO-Gaia at matching radii.

Layout:
- First column: Full Q1-Gaia (no angular separation limit) on top, empty on bottom
- Remaining columns: Q1-Gaia (filtered to radius) on top, ERO-Gaia (at radius) on bottom
- Columns: Full | 0.05 | 0.2 | 1.0 arcsec
- Rows: Q1-Gaia (top) vs ERO-Gaia (bottom)

All plots show:
- X-axis: Angular separation (arcsec)
- Y-axis: Magnitude difference
- Color: Number of sources (log scale)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import FuncFormatter, MultipleLocator

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
    print("Creating Grid: Q1-Gaia vs ERO-Gaia at Different Matching Radii (Filtered)")
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
        # Extract radius from filename
        radius_str = f.stem.replace('ero_gaia_subset_', '').replace('_', '.')
        ero_data[radius_str] = load_data(f, f'ERO-Gaia ({radius_str} arcsec)')

    # Filter out None values
    ero_data = {k: v for k, v in ero_data.items() if v is not None}

    if not ero_data:
        print("ERROR: Could not load any ERO-Gaia data")
        return

    # Sort radii for consistent ordering (only 0.05, 0.2, 1.0)
    radii_order = ['0.05', '0.2', '1.0']
    sorted_radii = [r for r in radii_order if r in ero_data]

    # Calculate global y-axis limits
    print("\nCalculating y-axis limits...")
    all_mag_diffs = []
    all_mag_diffs.extend(q1_clean['mag_diff'])
    for radius in sorted_radii:
        all_mag_diffs.extend(ero_data[radius]['mag_diff'])

    y_min = np.percentile(all_mag_diffs, 0.5)
    y_max = np.percentile(all_mag_diffs, 99.5)
    print(f"  Y-axis limits: {y_min:.3f} to {y_max:.3f} mag")

    # Create grid: 2 rows (Q1 top, ERO bottom) x (1 + len(radii)) columns
    # First column: Full Q1 (top) + empty (bottom)
    # Remaining columns: Filtered Q1 (top) + ERO at radius (bottom)
    num_cols = 1 + len(sorted_radii)
    num_rows = 2

    print(f"\nCreating grid: {num_rows} rows x {num_cols} columns")
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5.5 * num_cols, 6))

    # Store histogram objects for colorbar
    h_objects = []

    # Column 0: Full Q1-Gaia (no limit)
    print(f"\nColumn 1/{num_cols}: Full Q1-Gaia (no angular separation limit)")
    print(f"  Creating Q1-Gaia plot...")

    h = axes[0, 0].hist2d(q1_clean['angular_separation_arcsec'],
                           q1_clean['mag_diff'],
                           bins=[40, 40],
                           cmap='viridis',
                           cmin=1,
                           norm=plt.matplotlib.colors.LogNorm())
    h_objects.append(h[3])

    axes[0, 0].set_xlabel('Angular Sep. (arcsec)', fontsize=9, fontweight='bold')
    axes[0, 0].set_ylabel('Mag. Diff.', fontsize=9, fontweight='bold')
    axes[0, 0].set_title(f'Q1-Gaia (full, n={len(q1_clean):,})',
                         fontsize=10, fontweight='bold', pad=8)
    axes[0, 0].set_ylim(y_min, y_max)
    axes[0, 0].grid(True, alpha=0.3)

    # Bottom left is empty
    axes[1, 0].axis('off')

    # Remaining columns: Filtered Q1 (top) + ERO at radius (bottom)
    for col_idx, radius in enumerate(sorted_radii, 1):
        x_limit = float(radius)
        print(f"\nColumn {col_idx + 1}/{num_cols}: Angular separation range 0 - {radius} arcsec")

        # Top row: Q1-Gaia filtered
        print(f"  Creating Q1-Gaia plot...")
        q1_filtered = q1_clean[q1_clean['angular_separation_arcsec'] <= x_limit]

        h = axes[0, col_idx].hist2d(q1_filtered['angular_separation_arcsec'],
                                     q1_filtered['mag_diff'],
                                     bins=[40, 40],
                                     cmap='viridis',
                                     cmin=1,
                                     norm=plt.matplotlib.colors.LogNorm())
        h_objects.append(h[3])

        axes[0, col_idx].set_xlabel('Angular Sep. (arcsec)', fontsize=9, fontweight='bold')
        axes[0, col_idx].set_ylabel('Mag. Diff.', fontsize=9, fontweight='bold')
        axes[0, col_idx].set_title(f'Q1-Gaia (≤{radius}″, n={len(q1_filtered):,})',
                                   fontsize=10, fontweight='bold', pad=8)
        axes[0, col_idx].set_xlim(0, x_limit)
        axes[0, col_idx].set_ylim(y_min, y_max)
        axes[0, col_idx].grid(True, alpha=0.3)
        # Set 6 ticks evenly spaced from 0 to x_limit
        axes[0, col_idx].set_xticks(np.linspace(0, x_limit, 6))

        # Bottom row: ERO-Gaia at matching radius
        print(f"  Creating ERO-Gaia ({radius} arcsec) plot...")
        ero_clean = ero_data[radius]

        h = axes[1, col_idx].hist2d(ero_clean['angular_separation_arcsec'],
                                     ero_clean['mag_diff'],
                                     bins=[40, 40],
                                     cmap='viridis',
                                     cmin=1,
                                     norm=plt.matplotlib.colors.LogNorm())
        h_objects.append(h[3])

        axes[1, col_idx].set_xlabel('Angular Sep. (arcsec)', fontsize=9, fontweight='bold')
        axes[1, col_idx].set_ylabel('Mag. Diff.', fontsize=9, fontweight='bold')
        axes[1, col_idx].set_title(f'ERO-Gaia ({radius}″, n={len(ero_clean):,})',
                                  fontsize=10, fontweight='bold', pad=8)
        axes[1, col_idx].set_xlim(0, x_limit)
        axes[1, col_idx].set_ylim(y_min, y_max)
        axes[1, col_idx].grid(True, alpha=0.3)
        # Set 6 ticks evenly spaced from 0 to x_limit
        axes[1, col_idx].set_xticks(np.linspace(0, x_limit, 6))

    # Format x-axis labels to 2 decimal places for all axes
    formatter = FuncFormatter(lambda x, p: f'{x:.2f}')
    for i in range(num_rows):
        for j in range(num_cols):
            if axes[i, j].get_visible():  # Skip empty axes
                axes[i, j].xaxis.set_major_formatter(formatter)

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(h_objects[-1], cax=cbar_ax)
    cbar.set_label('Count (log scale)', fontsize=10, fontweight='bold')

    # Overall title
    fig.suptitle('Magnitude Difference (ero_mag_auto - gaia_mag_g) vs Angular Separation: Q1-Gaia vs ERO-Gaia Comparison',
                 fontsize=13, fontweight='bold', y=1.02)

    # Adjust layout
    plt.subplots_adjust(left=0.08, right=0.89, top=0.93, bottom=0.08, hspace=0.35, wspace=0.35)

    # Save figure
    output_path = OUTPUT_DIR / 'mag_diff_comparison_grid_filtered.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {output_path}")

    plt.close()

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()
