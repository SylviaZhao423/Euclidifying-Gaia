#!/usr/bin/env python3
"""
Create a grid of 2D histograms comparing Q1-Gaia and ERO-Gaia at matching radii.
Queries directly from WSDB crossmatch tables using hybrid magnitude approach.

Layout:
- First column: Full Q1-Gaia (no angular separation limit) on top, empty on bottom
- Remaining columns: Q1-Gaia (filtered to radius) on top, ERO-Gaia (at radius) on bottom
- Columns: Full | 0.05 | 0.1 | 0.3 arcsec
- Rows: Q1-Gaia (top) vs ERO-Gaia (bottom)

All plots show:
- X-axis: Angular separation (arcsec)
- Y-axis: Magnitude difference (source mag - Gaia mag)
- Color: Number of sources (log scale)

Magnitude difference definitions:
- Q1: Calibrated flux ratio approach
      mag_diff = -2.5*log10(m_flux_detection_total/g_phot_g_mean_flux) - median(flux_ratios)
      Normalizes to zero median flux ratio via calibration offset
- ERO: Direct magnitude difference
      mag_diff = e_v_mag_auto - g_phot_g_mean_mag
- Gaia Reference: g_phot_g_mean_mag (g-band mean magnitude from Gaia DR3)
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from matplotlib.ticker import FuncFormatter

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output'

# WSDB Configuration
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


def query_q1_gaia_data(username):
    """Query Q1-Gaia crossmatch data from WSDB using calibrated flux ratio magnitude.

    Q1 Magnitude Difference (calibrated flux ratio):
    - Source flux: m_flux_detection_total (MER Kron total flux)
    - Gaia flux: g_phot_g_mean_flux (Gaia DR3 g-band mean flux)
    - Magnitude difference: mag_diff = -2.5*log10(Q1_flux/Gaia_flux) - median(flux_ratios)

    This converts flux ratios to magnitude-like scale with calibration offset.
    """
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed!")
        return None

    try:
        conn = psycopg2.connect(
            host=WSDB_HOST,
            database=WSDB_DB,
            user=username
        )
        cursor = conn.cursor()

        # Query Q1-Gaia crossmatch table
        # Get raw fluxes for ratio calculation
        query = """
            SELECT
                m_flux_detection_total as q1_flux,
                g_phot_g_mean_flux as gaia_flux,
                SQRT(POWER(m_right_ascension - g_ra, 2) + POWER(m_declination - g_dec, 2)) * 3600 as angular_sep_arcsec
            FROM siwei_zhao_2025.crossmatch_q1_mer_gaia
            WHERE m_flux_detection_total > 0
              AND m_flux_detection_total IS NOT NULL
              AND m_flux_detection_total < 1e10
              AND g_phot_g_mean_flux > 0
              AND g_phot_g_mean_flux IS NOT NULL
              AND g_phot_g_mean_flux < 1e10
        """

        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        # Convert to lists
        q1_fluxes = []
        gaia_fluxes = []
        ang_seps = []

        for q1_flux, gaia_flux, ang_sep in results:
            q1_fluxes.append(q1_flux)
            gaia_fluxes.append(gaia_flux)
            ang_seps.append(ang_sep)

        # Calculate flux ratio in log scale (magnitude-like)
        flux_ratios = np.log10(np.array(q1_fluxes) / np.array(gaia_fluxes))

        # Calculate calibration offset from median
        offset = np.median(flux_ratios)
        mag_diffs = -2.5 * (flux_ratios - offset)

        print(f"  ✓ Loaded {len(results):,} Q1-Gaia sources")
        print(f"    Angular separation range: {np.min(ang_seps):.6f} - {np.max(ang_seps):.3f} arcsec")
        print(f"    Flux ratio (log10) range: {np.min(flux_ratios):.3f} to {np.max(flux_ratios):.3f}")
        print(f"    Calibration offset (median): {offset:.6f}")
        print(f"    Magnitude difference range: {np.min(mag_diffs):.3f} to {np.max(mag_diffs):.3f} mag")

        return {
            'mag_diff': mag_diffs,
            'angular_sep': np.array(ang_seps),
            'offset': offset
        }

    except Exception as e:
        print(f"  ✗ Error querying Q1-Gaia data: {e}")
        import traceback
        traceback.print_exc()
        return None


def query_ero_gaia_data(username, radius_str):
    """Query ERO-Gaia crossmatch data from WSDB at specified radius using magnitude difference.

    ERO Source Magnitude:
    - Source: e_v_mag_auto (automatic aperture magnitude, VIS band)

    Gaia Reference Magnitude:
    - Source: g_phot_g_mean_mag (Gaia DR3 g-band mean magnitude)

    Magnitude difference: mag_diff = ERO_mag - Gaia_mag
    """
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed!")
        return None

    try:
        conn = psycopg2.connect(
            host=WSDB_HOST,
            database=WSDB_DB,
            user=username
        )
        cursor = conn.cursor()

        # Determine table name from radius
        radius_suffix = radius_str.replace('.', '_')
        table_name = f"{username}.crossmatch_ero_gaia_{radius_suffix}"

        # Query ERO-Gaia crossmatch table
        # Direct magnitude difference: e_v_mag_auto - g_phot_g_mean_mag
        query = f"""
            SELECT
                e_v_mag_auto as ero_mag,
                g_phot_g_mean_mag as gaia_mag,
                SQRT(POWER(e_v_alpha_j2000 - g_ra, 2) + POWER(e_v_delta_j2000 - g_dec, 2)) * 3600 as angular_sep_arcsec
            FROM {table_name}
            WHERE e_v_mag_auto > 0
              AND e_v_mag_auto IS NOT NULL
              AND e_v_mag_auto < 50
              AND g_phot_g_mean_mag IS NOT NULL
              AND g_phot_g_mean_mag > 0
              AND g_phot_g_mean_mag < 30
        """

        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        # Convert to lists
        ero_mags = []
        gaia_mags = []
        ang_seps = []

        for ero_mag, gaia_mag, ang_sep in results:
            ero_mags.append(ero_mag)
            gaia_mags.append(gaia_mag)
            ang_seps.append(ang_sep)

        # Calculate magnitude difference
        mag_diffs = np.array(ero_mags) - np.array(gaia_mags)

        print(f"  ✓ Loaded {len(results):,} ERO-Gaia sources ({radius_str} arcsec)")
        print(f"    Angular separation range: {np.min(ang_seps):.6f} - {np.max(ang_seps):.3f} arcsec")
        print(f"    Magnitude difference range: {np.min(mag_diffs):.3f} to {np.max(mag_diffs):.3f} mag")
        print(f"    Median magnitude difference: {np.median(mag_diffs):.6f} mag")

        return {
            'mag_diff': mag_diffs,
            'angular_sep': np.array(ang_seps)
        }

    except Exception as e:
        print(f"  ✗ Error querying ERO-Gaia data ({radius_str}): {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main execution."""
    print("=" * 80)
    print("Creating Grid: Q1-Gaia vs ERO-Gaia at Different Matching Radii")
    print("(Hybrid Approach: Q1 uses Calibrated Flux Ratio, ERO uses Direct Magnitude)")
    print("Q1: Flux ratio calibration | ERO: e_v_mag_auto - g_phot_g_mean_mag")
    print("=" * 80)

    # Get username from environment
    username = os.getenv('PGUSER')
    if not username:
        print("\nERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return

    print(f"\nLogged in as: {username}")

    # Query Q1-Gaia data
    print("\nQuerying Q1-Gaia crossmatch data...")
    q1_data = query_q1_gaia_data(username)
    if q1_data is None:
        return

    # Query ERO-Gaia data at three radii
    print("\nQuerying ERO-Gaia crossmatch data...")
    radii = ['0.05', '0.1', '0.3']
    ero_data = {}

    for radius in radii:
        print(f"  Radius {radius} arcsec:")
        data = query_ero_gaia_data(username, radius)
        if data is not None:
            ero_data[radius] = data
        else:
            print(f"    WARNING: Could not load ERO-Gaia data for {radius} arcsec")

    if not ero_data:
        print("ERROR: Could not load any ERO-Gaia data")
        return

    # Calculate total valid Q1 sources for percentage calculations
    q1_total_valid = np.sum(np.isfinite(q1_data['mag_diff']))

    # Calculate ERO percentages relative to the 0.3 arcsec sample (largest)
    ero_0_3_total = len(ero_data['0.3']['mag_diff']) if '0.3' in ero_data else 1

    # Calculate global y-axis limits
    print("\nCalculating y-axis limits...")
    all_mag_diffs = list(q1_data['mag_diff'][np.isfinite(q1_data['mag_diff'])])
    for radius in radii:
        if radius in ero_data:
            valid_diffs = ero_data[radius]['mag_diff'][np.isfinite(ero_data[radius]['mag_diff'])]
            all_mag_diffs.extend(valid_diffs)

    if len(all_mag_diffs) > 0:
        y_min_raw = np.percentile(all_mag_diffs, 0.5)
        y_max_raw = np.percentile(all_mag_diffs, 99.5)
        # Make symmetric around 0
        y_lim = max(abs(y_min_raw), abs(y_max_raw))
        y_min = -y_lim
        y_max = y_lim
        print(f"  Y-axis limits (raw): {y_min_raw:.3f} to {y_max_raw:.3f} mag")
        print(f"  Y-axis limits (symmetric): {y_min:.3f} to {y_max:.3f} mag")
    else:
        print(f"  ERROR: No valid magnitude differences found!")
        return

    # Create grid: 2 rows (Q1 top, ERO bottom) x (1 + len(radii)) columns
    num_cols = 1 + len(radii)
    num_rows = 2

    print(f"\nCreating grid: {num_rows} rows x {num_cols} columns")
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5.5 * num_cols, 7))

    # Store histogram objects for colorbar
    h_objects = []

    # Column 0: Full Q1-Gaia (no limit)
    print(f"\nColumn 1/{num_cols}: Full Q1-Gaia (no angular separation limit)")

    # Filter out NaN values for plotting
    q1_valid = np.isfinite(q1_data['mag_diff'])
    q1_ang_valid = q1_data['angular_sep'][q1_valid]
    q1_mag_valid = q1_data['mag_diff'][q1_valid]

    h = axes[0, 0].hist2d(q1_ang_valid,
                          q1_mag_valid,
                          bins=[40, 40],
                          cmap='viridis',
                          cmin=1,
                          norm=plt.matplotlib.colors.LogNorm())
    h_objects.append(h[3])

    axes[0, 0].set_xlabel('Angular Sep. (arcsec)', fontsize=9, fontweight='bold')
    axes[0, 0].set_ylabel('Magnitude Diff. (mag)', fontsize=9, fontweight='bold')
    axes[0, 0].set_title(f'Q1-Gaia (full, n={len(q1_data["mag_diff"]):,} | 100.0%)',
                         fontsize=10, fontweight='bold', pad=8)
    axes[0, 0].set_ylim(y_min, y_max)
    axes[0, 0].grid(True, alpha=0.3)

    # Bottom left: show magnitude difference equations
    axes[1, 0].axis('off')
    # Get Q1 offset to display in equation
    q1_offset = q1_data.get('offset', 0)
    # Add text with magnitude equations
    equation_text = (
        f'Magnitude Difference Equations:\n\n'
        f'Q1-Gaia mag. diff. =\n'
        f'  -2.5*log(m_flux_detection_total/\n'
        f'           g_phot_g_mean_flux)\n'
        f'  - median(flux_ratio)\n'
        f'  = {q1_offset:.6f}\n\n'
        f'ERO-Gaia mag. diff. =\n'
        f'  e_v_mag_auto -\n'
        f'  g_phot_g_mean_mag'
    )
    axes[1, 0].text(-0.1, 0.5, equation_text, fontsize=9, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # Remaining columns: Filtered Q1 (top) + ERO at radius (bottom)
    for col_idx, radius in enumerate(radii, 1):
        x_limit = float(radius)
        print(f"\nColumn {col_idx + 1}/{num_cols}: Angular separation range 0 - {radius} arcsec")

        # Top row: Q1-Gaia filtered
        q1_mask = q1_data['angular_sep'] <= x_limit
        q1_filtered_mag_diff = q1_data['mag_diff'][q1_mask]
        q1_filtered_ang_sep = q1_data['angular_sep'][q1_mask]

        # Filter out NaN values
        q1_valid_mask = np.isfinite(q1_filtered_mag_diff)
        q1_filtered_mag_diff = q1_filtered_mag_diff[q1_valid_mask]
        q1_filtered_ang_sep = q1_filtered_ang_sep[q1_valid_mask]

        h = axes[0, col_idx].hist2d(q1_filtered_ang_sep,
                                     q1_filtered_mag_diff,
                                     bins=[40, 40],
                                     cmap='viridis',
                                     cmin=1,
                                     norm=plt.matplotlib.colors.LogNorm())
        h_objects.append(h[3])

        axes[0, col_idx].set_xlabel('Angular Sep. (arcsec)', fontsize=9, fontweight='bold')
        axes[0, col_idx].set_ylabel('Magnitude Diff. (mag)', fontsize=9, fontweight='bold')
        q1_pct_filtered = 100.0 * len(q1_filtered_mag_diff) / q1_total_valid if q1_total_valid > 0 else 0
        axes[0, col_idx].set_title(f'Q1-Gaia (≤{radius}″, n={len(q1_filtered_mag_diff):,} | {q1_pct_filtered:.1f}%)',
                                   fontsize=10, fontweight='bold', pad=8)
        axes[0, col_idx].set_xlim(0, x_limit)
        axes[0, col_idx].set_ylim(y_min, y_max)
        axes[0, col_idx].grid(True, alpha=0.3)
        # Set 6 ticks evenly spaced from 0 to x_limit
        axes[0, col_idx].set_xticks(np.linspace(0, x_limit, 6))

        # Bottom row: ERO-Gaia at matching radius
        if radius in ero_data:
            print(f"  Creating ERO-Gaia ({radius} arcsec) plot...")
            ero_clean = ero_data[radius]

            h = axes[1, col_idx].hist2d(ero_clean['angular_sep'],
                                         ero_clean['mag_diff'],
                                         bins=[40, 40],
                                         cmap='viridis',
                                         cmin=1,
                                         norm=plt.matplotlib.colors.LogNorm())
            h_objects.append(h[3])

            axes[1, col_idx].set_xlabel('Angular Sep. (arcsec)', fontsize=9, fontweight='bold')
            axes[1, col_idx].set_ylabel('Magnitude Diff. (mag)', fontsize=9, fontweight='bold')
            ero_pct_vs_0_3 = 100.0 * len(ero_clean['mag_diff']) / ero_0_3_total if ero_0_3_total > 0 else 0
            axes[1, col_idx].set_title(f'ERO-Gaia ({radius}″, n={len(ero_clean["mag_diff"]):,} | {ero_pct_vs_0_3:.1f}%)',
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
    fig.suptitle('Magnitude Difference vs Angular Separation: Q1-Gaia (Flux Ratio) vs ERO-Gaia (Direct Magnitude)',
                 fontsize=12, fontweight='bold', y=1.01)

    # Adjust layout
    plt.subplots_adjust(left=0.08, right=0.89, top=0.92, bottom=0.08, hspace=0.35, wspace=0.35)

    # Save figure
    output_path = OUTPUT_DIR / 'mag_vs_ang_sep.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {output_path}")

    plt.close()

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()
