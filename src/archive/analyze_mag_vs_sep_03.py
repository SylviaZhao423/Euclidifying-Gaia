#!/usr/bin/env python3
"""
Create a grid of 2D histograms comparing Q1-Gaia (VIS detection only) and ERO-Gaia at matching radii.
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
- Q1 (VIS detection only): ZPC-corrected magnitude calculation
      m_Q1 = -2.5 × log10(ZPC × m_flux_detection_total) + 23.9
      mag_diff = m_Q1 - g_phot_g_mean_mag
      ZPC = 1.05 (IE-band correction)
      Filter: VIS_DET = 1 (VIS mosaic detection only, not NIR-only detection)
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

# ZPC Configuration (IE-band)
ZPC_IE = 1.05
ZP_REF = 23.9

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


def query_q1_gaia_data_vis(username):
    """Query Q1-Gaia crossmatch data from WSDB using ZPC-corrected magnitude calculation.

    Filters for VIS detection only (VIS_DET = 1).

    Q1 Magnitude Difference (ZPC-corrected, VIS detection):
    - Q1 magnitude: m_Q1 = -2.5 × log10(ZPC × m_flux_detection_total) + 23.9
    - Gaia magnitude: g_phot_g_mean_mag (Gaia DR3 g-band mean magnitude)
    - Magnitude difference: mag_diff = m_Q1 - g_phot_g_mean_mag
    - ZPC = 1.05 (IE-band correction)
    - Filter: VIS_DET = 1 (detected in VIS mosaic, excluding NIR-only detections)
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
        # Calculate ZPC-corrected magnitudes and magnitude differences directly in SQL
        query = f"""
            SELECT
                -2.5 * LOG10({ZPC_IE} * m_flux_detection_total) + {ZP_REF} as q1_mag,
                g_phot_g_mean_mag as gaia_mag,
                SQRT(POWER(m_right_ascension - g_ra, 2) + POWER(m_declination - g_dec, 2)) * 3600 as angular_sep_arcsec
            FROM siwei_zhao_2025.crossmatch_q1_mer_gaia
            WHERE m_flux_detection_total > 0
              AND m_flux_detection_total IS NOT NULL
              AND m_flux_detection_total < 1e10
              AND g_phot_g_mean_mag IS NOT NULL
              AND g_phot_g_mean_mag > 0
              AND g_phot_g_mean_mag < 30
              AND m_vis_det = 1
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

        # Calculate magnitude difference directly
        mag_diffs = np.array(q1_mags) - np.array(gaia_mags)

        print(f"  ✓ Loaded {len(results):,} Q1-Gaia sources (VIS detection only, m_vis_det = 1)")
        print(f"    Angular separation range: {np.min(ang_seps):.6f} - {np.max(ang_seps):.3f} arcsec")
        print(f"    Q1 magnitude range: {np.min(q1_mags):.3f} to {np.max(q1_mags):.3f} AB mag")
        print(f"    Gaia magnitude range: {np.min(gaia_mags):.3f} to {np.max(gaia_mags):.3f} mag")
        print(f"    Magnitude difference range: {np.min(mag_diffs):.3f} to {np.max(mag_diffs):.3f} mag")
        print(f"    Median magnitude difference: {np.median(mag_diffs):.6f} mag")

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
    print("Creating Grid: Q1-Gaia (VIS detection) vs ERO-Gaia at Different Matching Radii")
    print("(Hybrid Approach: Q1 uses ZPC-corrected magnitude (VIS only), ERO uses Direct Magnitude)")
    print(f"Q1 (VIS_DET=1): -2.5×log({ZPC_IE}×flux)+{ZP_REF} | ERO: e_v_mag_auto - g_phot_g_mean_mag")
    print("=" * 80)

    # Get username from environment
    username = os.getenv('PGUSER')
    if not username:
        print("\nERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return

    print(f"\nLogged in as: {username}")

    # Query Q1-Gaia data (VIS detection only)
    print("\nQuerying Q1-Gaia crossmatch data (VIS detection only)...")
    q1_data = query_q1_gaia_data_vis(username)
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
    print(f"\nColumn 1/{num_cols}: Full Q1-Gaia VIS (no angular separation limit)")

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
    axes[0, 0].set_ylabel('Mag. Diff.', fontsize=9, fontweight='bold')
    axes[0, 0].set_title(f'Q1-Gaia (full, n={len(q1_data["mag_diff"]):,} | 100.0%)',
                         fontsize=10, fontweight='bold', pad=8)
    axes[0, 0].set_ylim(-6, 6)
    axes[0, 0].grid(True, alpha=0.3)

    # Bottom left: show magnitude difference equations
    axes[1, 0].axis('off')
    # Add text with magnitude equations
    equation_text = (
        f'Magnitude Difference Equations:\n\n'
        f'  Q1-Gaia mag. diff.  (VIS_DET = 1 only) \n'
        f'= [-2.5log(ZPC * m_flux_detection_total + ZP]- g_phot_g_mean_mag\n'
        f'= [-2.5log({ZPC_IE} * m_flux_detection_total + {ZP_REF}] - g_phot_g_mean_mag\n\n'
        f'  ERO-Gaia mag. diff.\n'
        f'= e_v_mag_auto - g_phot_g_mean_mag'
    )
    axes[1, 0].text(-0.12, 0.5, equation_text, fontsize=8, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='deepskyblue', alpha=0.3))

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
        axes[0, col_idx].set_ylabel('Mag. Diff.', fontsize=9, fontweight='bold')
        q1_pct_filtered = 100.0 * len(q1_filtered_mag_diff) / q1_total_valid if q1_total_valid > 0 else 0
        axes[0, col_idx].set_title(f'Q1-Gaia ({radius}″, n={len(q1_filtered_mag_diff):,} | {q1_pct_filtered:.1f}%)',
                                   fontsize=10, fontweight='bold', pad=8)
        axes[0, col_idx].set_xlim(0, x_limit)
        axes[0, col_idx].set_ylim(-6, 6)
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
            axes[1, col_idx].set_ylabel('Mag. Diff.', fontsize=9, fontweight='bold')
            ero_pct_vs_0_3 = 100.0 * len(ero_clean['mag_diff']) / ero_0_3_total if ero_0_3_total > 0 else 0
            axes[1, col_idx].set_title(f'ERO-Gaia ({radius}″, n={len(ero_clean["mag_diff"]):,} | {ero_pct_vs_0_3:.1f}%)',
                                      fontsize=10, fontweight='bold', pad=8)
            axes[1, col_idx].set_xlim(0, x_limit)
            axes[1, col_idx].set_ylim(-6, 6)
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
    fig.suptitle('Magnitude Difference vs Angular Separation: Q1-Gaia vs ERO-Gaia',
                 fontsize=12, fontweight='bold', y=1.01)

    # Adjust layout
    plt.subplots_adjust(left=0.08, right=0.89, top=0.92, bottom=0.08, hspace=0.35, wspace=0.35)

    # Save figure
    output_path = OUTPUT_DIR / 'mag_vs_ang_sep_03.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {output_path}")

    plt.close()

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()
