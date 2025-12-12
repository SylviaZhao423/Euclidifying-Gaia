#!/usr/bin/env python3
"""
Create a grid of 2D histograms comparing Q1-Gaia and ERO-Gaia at matching radii.
Queries directly from WSDB crossmatch tables using PSF magnitudes for both datasets.

Layout:
- First column: Full Q1-Gaia (no angular separation limit) on top, empty on bottom
- Remaining columns: Q1-Gaia (filtered to radius) on top, ERO-Gaia (at radius) on bottom
- Columns: Full | 0.05 | 0.2 | 1.0 arcsec
- Rows: Q1-Gaia (top) vs ERO-Gaia (bottom)

All plots show:
- X-axis: Angular separation (arcsec)
- Y-axis: Magnitude difference (source PSF mag - Gaia g-band mag)
- Color: Number of sources (log scale)

Magnitude calculation:
- Q1: m_mag_psf = -2.5 * log10(m_flux_vis_psf) + 23.9 (VIS band PSF, calculated from flux)
- ERO: e_v_mag_psf = stored PSF magnitude (VIS band)
- Gaia: g_mag = g_phot_g_mean_mag (g-band mean magnitude)

Note: Q1 magnitude calculated from flux with ZP=23.9, ERO uses stored e_v_mag_psf
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

# Zero point for Q1 magnitude calculation: mag = -2.5 * log10(flux) + ZP
# (Not needed for ERO and Gaia as they use stored magnitudes)
ZP_Q1_VIS = 22   # Q1 VIS band PSF

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


def query_q1_gaia_data(username):
    """Query Q1-Gaia crossmatch data from WSDB."""
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
        # Calculate PSF magnitude from flux and angular separation to Gaia
        query = f"""
            SELECT
                (-2.5 * LOG(m_flux_vis_psf) / LN(10) + {ZP_Q1_VIS}) as q1_psf_mag,
                g_phot_g_mean_mag as gaia_mag,
                SQRT(POWER(m_right_ascension - g_ra, 2) + POWER(m_declination - g_dec, 2)) * 3600 as angular_sep_arcsec
            FROM siwei_zhao_2025.crossmatch_q1_mer_gaia
            WHERE m_flux_vis_psf > 0
              AND m_flux_vis_psf IS NOT NULL
              AND g_phot_g_mean_mag IS NOT NULL
        """

        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        # Convert to lists
        q1_mags = []
        gaia_mags = []
        ang_seps = []

        for q1_mag, gaia_mag, ang_sep in results:
            q1_mags.append(q1_mag)
            gaia_mags.append(gaia_mag)
            ang_seps.append(ang_sep)

        # Calculate magnitude difference
        mag_diffs = np.array(q1_mags) - np.array(gaia_mags)

        print(f"  ✓ Loaded {len(results):,} Q1-Gaia sources")
        print(f"    Angular separation range: {np.min(ang_seps):.6f} - {np.max(ang_seps):.3f} arcsec")
        print(f"    Mag diff range: {np.nanmin(mag_diffs):.3f} - {np.nanmax(mag_diffs):.3f} mag")

        return {
            'mag_diff': mag_diffs,
            'angular_sep': np.array(ang_seps)
        }

    except Exception as e:
        print(f"  ✗ Error querying Q1-Gaia data: {e}")
        import traceback
        traceback.print_exc()
        return None


def query_ero_gaia_data(username, radius_str):
    """Query ERO-Gaia crossmatch data from WSDB at specified radius."""
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
        # Use stored PSF magnitude (VIS band) and angular separation to Gaia
        query = f"""
            SELECT
                e_v_mag_psf as ero_psf_mag,
                g_phot_g_mean_mag as gaia_mag,
                SQRT(POWER(e_v_alpha_j2000 - g_ra, 2) + POWER(e_v_delta_j2000 - g_dec, 2)) * 3600 as angular_sep_arcsec
            FROM {table_name}
            WHERE e_v_mag_psf IS NOT NULL
              AND e_v_mag_psf < 50
              AND g_phot_g_mean_mag IS NOT NULL
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
        print(f"    Mag diff range: {np.nanmin(mag_diffs):.3f} - {np.nanmax(mag_diffs):.3f} mag")

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
    print("(PSF Magnitude Difference from WSDB Crossmatch Tables)")
    print("=" * 80)

    # Get username from environment
    username = os.getenv('PGUSER')
    if not username:
        print("\nERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return

    print(f"\nLogged in as: {username}")
    print(f"\nMagnitude sources:")
    print(f"  Q1: PSF mag = -2.5 * log10(m_flux_vis_psf) + {ZP_Q1_VIS} (calculated from flux)")
    print(f"  ERO: PSF mag = e_v_mag_psf (stored magnitude, VIS band)")
    print(f"  Gaia: g mag = g_phot_g_mean_mag (g-band)")

    # Query Q1-Gaia data
    print("\nQuerying Q1-Gaia crossmatch data...")
    q1_data = query_q1_gaia_data(username)
    if q1_data is None:
        return

    # Query ERO-Gaia data at three radii
    print("\nQuerying ERO-Gaia crossmatch data...")
    radii = ['0.05', '0.2', '1.0']
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

    # Calculate ERO percentages relative to the 1.0 arcsec sample (largest)
    ero_1_0_total = len(ero_data['1.0']['mag_diff']) if '1.0' in ero_data else 1

    # Calculate global y-axis limits
    print("\nCalculating y-axis limits...")
    all_mag_diffs = list(q1_data['mag_diff'][np.isfinite(q1_data['mag_diff'])])
    for radius in radii:
        if radius in ero_data:
            valid_diffs = ero_data[radius]['mag_diff'][np.isfinite(ero_data[radius]['mag_diff'])]
            all_mag_diffs.extend(valid_diffs)

    if len(all_mag_diffs) > 0:
        y_min = np.percentile(all_mag_diffs, 0.5)
        y_max = np.percentile(all_mag_diffs, 99.5)
        print(f"  Y-axis limits: {y_min:.3f} to {y_max:.3f} mag")
    else:
        print(f"  ERROR: No valid magnitude differences found!")
        return

    # Create grid: 2 rows (Q1 top, ERO bottom) x (1 + len(radii)) columns
    num_cols = 1 + len(radii)
    num_rows = 2

    print(f"\nCreating grid: {num_rows} rows x {num_cols} columns")
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5.5 * num_cols, 6))

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
    axes[0, 0].set_ylabel('Mag Diff (m_mag_VIS_PSF - g_phot_g_mean_mag)', fontsize=8, fontweight='bold')
    axes[0, 0].set_title(f'Q1-Gaia (full, n={len(q1_data["mag_diff"]):,} | 100.0%)',
                         fontsize=10, fontweight='bold', pad=8)
    axes[0, 0].set_ylim(y_min, y_max)
    axes[0, 0].grid(True, alpha=0.3)

    # Bottom left is empty
    axes[1, 0].axis('off')

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
        axes[0, col_idx].set_ylabel('Mag Diff (m_mag_VIS_PSF - g_phot_g_mean_mag)', fontsize=8, fontweight='bold')
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
            axes[1, col_idx].set_ylabel('Mag Diff (e_v_mag_psf - g_phot_g_mean_mag)', fontsize=8, fontweight='bold')
            ero_pct_vs_1_0 = 100.0 * len(ero_clean['mag_diff']) / ero_1_0_total if ero_1_0_total > 0 else 0
            axes[1, col_idx].set_title(f'ERO-Gaia ({radius}″, n={len(ero_clean["mag_diff"]):,} | {ero_pct_vs_1_0:.1f}%)',
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
    fig.suptitle('Magnitude Difference (PSF - Gaia g) vs Angular Separation: Q1-Gaia vs ERO-Gaia\n'
                 f'(WSDB Direct Query - Q1: flux-based mag (ZP={ZP_Q1_VIS}), ERO: stored e_v_mag_psf)',
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
