#!/usr/bin/env python3
"""
Plot magnitude difference vs magnitude, colored by Gaia BP-RP color.
Compares Q1-Gaia and ERO-Gaia crossmatch data at 0.3 arcsec matching radius.

X-axis: Gaia magnitude (g_phot_g_mean_mag)
Y-axis: Magnitude difference (Source_mag - Gaia_mag)
Color: Gaia BP-RP color index (g_phot_bp_mean_mag - g_phot_rp_mean_mag)

Magnitude Difference Equations:
- Q1-Gaia: mag_diff = [-2.5×log10(ZPC × m_flux_detection_total) + 23.9] - g_phot_g_mean_mag
           (ZPC = 1.05 for IE-band, VIS_DET = 1 only)
- ERO-Gaia: mag_diff = e_v_mag_auto - g_phot_g_mean_mag

Filters:
- Angular separation <= 0.3 arcsec
- VIS_DET = 1 for Q1 (VIS mosaic detection only)
- Valid flux and magnitude values
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

# Matching radius
MATCHING_RADIUS = 0.3  # arcsec

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


def query_q1_gaia_data_at_radius(username, radius):
    """Query Q1-Gaia crossmatch data at specified radius with BP-RP color.

    Q1 Magnitude: m_Q1 = -2.5 × log10(ZPC × m_flux_detection_total) + 23.9
    Magnitude Difference: mag_diff = m_Q1 - g_phot_g_mean_mag
    ZPC = 1.05 (IE-band correction), VIS_DET = 1 only
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

        # Query Q1-Gaia crossmatch table with BP-RP color
        # Angular separation filter applied
        query = f"""
            SELECT
                -2.5 * LOG10({ZPC_IE} * m_flux_detection_total) + {ZP_REF} as q1_mag,
                g_phot_g_mean_mag as gaia_mag,
                g_phot_bp_mean_mag - g_phot_rp_mean_mag as bp_rp_color,
                SQRT(POWER(m_right_ascension - g_ra, 2) + POWER(m_declination - g_dec, 2)) * 3600 as angular_sep_arcsec
            FROM siwei_zhao_2025.crossmatch_q1_mer_gaia
            WHERE m_flux_detection_total > 0
              AND m_flux_detection_total IS NOT NULL
              AND m_flux_detection_total < 1e10
              AND g_phot_g_mean_mag IS NOT NULL
              AND g_phot_g_mean_mag > 0
              AND g_phot_g_mean_mag < 30
              AND g_phot_bp_mean_mag IS NOT NULL
              AND g_phot_rp_mean_mag IS NOT NULL
              AND m_vis_det = 1
              AND SQRT(POWER(m_right_ascension - g_ra, 2) + POWER(m_declination - g_dec, 2)) * 3600 <= {radius}
        """

        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        # Convert to lists
        q1_mags = []
        gaia_mags = []
        bp_rp_colors = []
        ang_seps = []

        for q1_mag, gaia_mag, bp_rp, ang_sep in results:
            q1_mags.append(q1_mag)
            gaia_mags.append(gaia_mag)
            bp_rp_colors.append(bp_rp)
            ang_seps.append(ang_sep)

        # Calculate magnitude difference
        mag_diffs = np.array(q1_mags) - np.array(gaia_mags)

        print(f"  ✓ Loaded {len(results):,} Q1-Gaia sources at {radius}″ (VIS detection only)")
        print(f"    Angular separation range: {np.min(ang_seps):.6f} - {np.max(ang_seps):.3f} arcsec")
        print(f"    Q1 magnitude range: {np.min(q1_mags):.3f} to {np.max(q1_mags):.3f} AB mag")
        print(f"    Gaia magnitude range: {np.min(gaia_mags):.3f} to {np.max(gaia_mags):.3f} mag")
        print(f"    Magnitude difference range: {np.min(mag_diffs):.3f} to {np.max(mag_diffs):.3f} mag")
        print(f"    BP-RP color range: {np.min(bp_rp_colors):.3f} to {np.max(bp_rp_colors):.3f}")
        print(f"    Median magnitude difference: {np.median(mag_diffs):.6f} mag")

        return {
            'mag_diff': mag_diffs,
            'gaia_mag': np.array(gaia_mags),
            'bp_rp_color': np.array(bp_rp_colors),
            'angular_sep': np.array(ang_seps)
        }

    except Exception as e:
        print(f"  ✗ Error querying Q1-Gaia data: {e}")
        import traceback
        traceback.print_exc()
        return None


def query_ero_gaia_data_at_radius(username, radius):
    """Query ERO-Gaia crossmatch data at specified radius with BP-RP color.

    ERO Magnitude: e_v_mag_auto (VIS band automatic aperture magnitude)
    Magnitude Difference: mag_diff = e_v_mag_auto - g_phot_g_mean_mag
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

        # Query ERO-Gaia crossmatch table with BP-RP color
        # Using crossmatch_ero_gaia_0_3 table for 0.3 arcsec radius
        radius_suffix = str(radius).replace('.', '_')
        table_name = f"{username}.crossmatch_ero_gaia_{radius_suffix}"

        query = f"""
            SELECT
                e_v_mag_auto as ero_mag,
                g_phot_g_mean_mag as gaia_mag,
                g_phot_bp_mean_mag - g_phot_rp_mean_mag as bp_rp_color,
                SQRT(POWER(e_v_alpha_j2000 - g_ra, 2) + POWER(e_v_delta_j2000 - g_dec, 2)) * 3600 as angular_sep_arcsec
            FROM {table_name}
            WHERE e_v_mag_auto > 0
              AND e_v_mag_auto IS NOT NULL
              AND e_v_mag_auto < 50
              AND g_phot_g_mean_mag IS NOT NULL
              AND g_phot_g_mean_mag > 0
              AND g_phot_g_mean_mag < 30
              AND g_phot_bp_mean_mag IS NOT NULL
              AND g_phot_rp_mean_mag IS NOT NULL
        """

        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        # Convert to lists
        ero_mags = []
        gaia_mags = []
        bp_rp_colors = []
        ang_seps = []

        for ero_mag, gaia_mag, bp_rp, ang_sep in results:
            ero_mags.append(ero_mag)
            gaia_mags.append(gaia_mag)
            bp_rp_colors.append(bp_rp)
            ang_seps.append(ang_sep)

        # Calculate magnitude difference
        mag_diffs = np.array(ero_mags) - np.array(gaia_mags)

        print(f"  ✓ Loaded {len(results):,} ERO-Gaia sources at {radius}″")
        print(f"    Angular separation range: {np.min(ang_seps):.6f} - {np.max(ang_seps):.3f} arcsec")
        print(f"    ERO magnitude range: {np.min(ero_mags):.3f} to {np.max(ero_mags):.3f} mag")
        print(f"    Gaia magnitude range: {np.min(gaia_mags):.3f} to {np.max(gaia_mags):.3f} mag")
        print(f"    Magnitude difference range: {np.min(mag_diffs):.3f} to {np.max(mag_diffs):.3f} mag")
        print(f"    BP-RP color range: {np.min(bp_rp_colors):.3f} to {np.max(bp_rp_colors):.3f}")
        print(f"    Median magnitude difference: {np.median(mag_diffs):.6f} mag")

        return {
            'mag_diff': mag_diffs,
            'gaia_mag': np.array(gaia_mags),
            'bp_rp_color': np.array(bp_rp_colors),
            'angular_sep': np.array(ang_seps)
        }

    except Exception as e:
        print(f"  ✗ Error querying ERO-Gaia data: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_dataset(ax, data, title, source_type, xlim=None, ylim=None):
    """Create scatter plot for a dataset.

    Parameters:
    -----------
    xlim : tuple, optional
        X-axis limits (xmin, xmax). If None, computed from data.
    ylim : tuple, optional
        Y-axis limits (ymin, ymax). If None, computed from data.
    """
    scatter = ax.scatter(
        data['gaia_mag'],
        data['mag_diff'],
        c=data['bp_rp_color'],
        cmap='RdYlBu_r',
        s=10,
        alpha=0.6,
        edgecolors='none',
        vmin=np.percentile(data['bp_rp_color'], 1),
        vmax=np.percentile(data['bp_rp_color'], 99)
    )

    ax.set_xlabel('Gaia G Magnitude', fontsize=11, fontweight='bold')
    ax.set_ylabel('Magnitude Difference', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Set axis limits
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        mag_diff_limits = np.percentile(np.abs(data['mag_diff']), 99)
        ax.set_ylim(-mag_diff_limits * 1, mag_diff_limits * 1)

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(data['gaia_mag'].min() - 0.5, data['gaia_mag'].max() + 1)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))

    return scatter


def plot_histogram(ax, data, title, bins=50, xlim=None, ylim=None):
    """Create 2D histogram for magnitude difference vs BP-RP color.

    Parameters:
    -----------
    bins : int, optional
        Number of bins for the 2D histogram (default: 50)
    xlim : tuple, optional
        X-axis limits (xmin, xmax). If None, computed from data.
    ylim : tuple, optional
        Y-axis limits (ymin, ymax). If None, computed from data.

    Returns:
    --------
    The histogram image object (mappable) for creating a colorbar
    """
    h = ax.hist2d(data['bp_rp_color'], data['mag_diff'],
                  bins=bins, cmap='viridis', norm=plt.matplotlib.colors.LogNorm())

    ax.set_xlabel('Gaia BP-RP Color Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Magnitude Difference', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Set axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    return h[3]  # Return the mappable for colorbar


def print_statistics(data, source_type):
    """Print statistics by magnitude bins."""
    print(f"\nStatistics for {source_type}:")
    print("=" * 90)
    mag_bins = np.arange(10, 28, 2)
    print(f"{'Mag Bin':<15} {'Count':<10} {'Mean Diff':<16} {'Std Dev':<16} {'Median BP-RP':<15}")
    print("-" * 90)

    for i in range(len(mag_bins) - 1):
        mask = (data['gaia_mag'] >= mag_bins[i]) & (data['gaia_mag'] < mag_bins[i+1])
        if np.sum(mask) > 0:
            count = np.sum(mask)
            mean_diff = np.mean(data['mag_diff'][mask])
            std_diff = np.std(data['mag_diff'][mask])
            median_color = np.median(data['bp_rp_color'][mask])
            print(f"{mag_bins[i]:.1f}-{mag_bins[i+1]:.1f}      {count:>8,d}  "
                  f"{mean_diff:>14.6f}  {std_diff:>14.6f}  {median_color:>13.3f}")


def main():
    """Main execution."""
    print("=" * 90)
    print("Magnitude Difference vs Magnitude, Colored by Gaia BP-RP")
    print(f"Q1-Gaia vs ERO-Gaia Crossmatch at {MATCHING_RADIUS}″ Radius")
    print("=" * 90)

    print("\nMagnitude Difference Equations:")
    print(f"  Q1-Gaia:  mag_diff = [-2.5×log10({ZPC_IE}×m_flux_detection_total)+{ZP_REF}] - g_phot_g_mean_mag")
    print(f"  ERO-Gaia: mag_diff = e_v_mag_auto - g_phot_g_mean_mag")
    print("=" * 90)

    # Get username from environment
    username = os.getenv('PGUSER')
    if not username:
        print("\nERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return

    print(f"\nLogged in as: {username}")

    # Query both datasets
    print(f"\nQuerying Q1-Gaia crossmatch data at {MATCHING_RADIUS}″...")
    q1_data = query_q1_gaia_data_at_radius(username, MATCHING_RADIUS)
    if q1_data is None:
        return

    print(f"\nQuerying ERO-Gaia crossmatch data at {MATCHING_RADIUS}″...")
    ero_data = query_ero_gaia_data_at_radius(username, MATCHING_RADIUS)
    if ero_data is None:
        print("WARNING: Could not load ERO-Gaia data")
        ero_data = None

    # Create 2x2 figure with scatter plots on top, 2D histograms on bottom
    # Use gridspec_kw to make rows have equal height and add spacing for colorbars
    print(f"\nCreating scatter plots and 2D histograms...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # Top row: scatter plots (mag diff vs mag, colored by BP-RP)
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]

    # Calculate unified axis limits for top row (using full dataset)
    mag_diff_min = min(q1_data['mag_diff'].min(), ero_data['mag_diff'].min())
    mag_diff_max = max(q1_data['mag_diff'].max(), ero_data['mag_diff'].max())
    ylim_top = (mag_diff_min, mag_diff_max)

    gaia_mag_min = min(q1_data['gaia_mag'].min(), ero_data['gaia_mag'].min())
    gaia_mag_max = max(q1_data['gaia_mag'].max(), ero_data['gaia_mag'].max())
    xlim_top = (gaia_mag_min - 0.5, gaia_mag_max + 1)

    scatter1 = plot_dataset(ax1, q1_data,
                            f'Q1-Gaia Mag. Diff. vs Gaia Mag.',
                            'Q1-Gaia', xlim=xlim_top, ylim=ylim_top)
    scatter2 = plot_dataset(ax2, ero_data,
                            f'ERO-Gaia Mag. Diff. vs Gaia Mag.',
                            'ERO-Gaia', xlim=xlim_top, ylim=ylim_top)

    # Bottom row: 2D histograms (mag diff vs BP-RP color)
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]

    # Calculate unified axis limits for bottom row (using full dataset)
    bp_rp_min = min(q1_data['bp_rp_color'].min(), ero_data['bp_rp_color'].min())
    bp_rp_max = max(q1_data['bp_rp_color'].max(), ero_data['bp_rp_color'].max())
    xlim_bottom = (bp_rp_min - 0.1, bp_rp_max + 0.1)
    ylim_bottom = (mag_diff_min - 0.1, 6)

    # Create 2D histograms using helper function
    h1 = plot_histogram(ax3, q1_data,
                        f'Q1-Gaia Mag. Diff. vs BP-RP',
                        bins=50, xlim=xlim_bottom, ylim=ylim_bottom)
    h2 = plot_histogram(ax4, ero_data,
                        f'ERO-Gaia Mag. Diff. vs BP-RP',
                        bins=50, xlim=xlim_bottom, ylim=ylim_bottom)

    # Add individual colorbars for each histogram in the bottom row
    cbar3 = plt.colorbar(h1, ax=ax3, label='Count')
    cbar4 = plt.colorbar(h2, ax=ax4, label='Count')

    # Add individual colorbars for each scatter plot in the top row
    cbar1 = plt.colorbar(scatter1, ax=ax1, label='BP-RP Color Index')
    cbar2 = plt.colorbar(scatter2, ax=ax2, label='BP-RP Color Index')

    fig.suptitle(
        f'Magnitude Difference Distribution at {MATCHING_RADIUS}″ Crossmatch Radius\n'
        f'Q1 = -2.5 × log({ZPC_IE} × flux) + {ZP_REF} - Gaia (n={len(q1_data["mag_diff"]):,}) | ERO = e_v_mag_auto - Gaia (n={len(ero_data["mag_diff"]):,})',
        fontsize=15, fontweight='bold')
    plt.subplots_adjust(top=0.90, bottom=0.08, wspace=0.08)

    # Save figure
    output_path = OUTPUT_DIR / f'mag_diff_vs_mag_bp_rp_q1_ero_{MATCHING_RADIUS:.1f}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {output_path}")
    plt.close()

    # Print statistics
    print_statistics(q1_data, 'Q1-Gaia')
    print_statistics(ero_data, 'ERO-Gaia')

    print("\n" + "=" * 90)
    print("Done!")
    print("=" * 90)


if __name__ == '__main__':
    main()
