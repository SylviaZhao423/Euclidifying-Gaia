#!/usr/bin/env python3
"""
Plot magnitude difference vs magnitude with color and morphology views.
Compares Q1-Gaia and ERO-Gaia crossmatch data at 0.3 arcsec matching radius.

Top row: Scatter plots colored by Gaia BP-RP color index
Bottom row: Scatter plots colored by morphological classification

X-axis: Gaia magnitude (g_phot_g_mean_mag)
Y-axis: Magnitude difference (Source_mag - Gaia_mag)

Magnitude Difference Equations:
- Q1-Gaia: mag_diff = [-2.5×log10(ZPC × m_flux_detection_total) + 23.9] - g_phot_g_mean_mag
           (ZPC = 1.05 for IE-band, VIS_DET = 1 only)
- ERO-Gaia: mag_diff = e_v_mag_auto - g_phot_g_mean_mag

Morphological Classification:
- Q1: phz_gal_prob (0=star-like, 1=galaxy-like)
- ERO: e_v_class_star (0=extended, 1=star-like)

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
    """Query Q1-Gaia crossmatch data at specified radius with BP-RP color and phz_gal_prob.

    Q1 Magnitude: m_Q1 = -2.5 × log10(ZPC × m_flux_detection_total) + 23.9
    Magnitude Difference: mag_diff = m_Q1 - g_phot_g_mean_mag
    ZPC = 1.05 (IE-band correction), VIS_DET = 1 only
    Classification: phz_gal_prob (galaxy probability, 0=star, 1=galaxy)
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

        # Query Q1-Gaia crossmatch table with BP-RP color and phz_gal_prob
        # Angular separation filter applied
        query = f"""
            SELECT
                -2.5 * LOG10({ZPC_IE} * m_flux_detection_total) + {ZP_REF} as q1_mag,
                g_phot_g_mean_mag as gaia_mag,
                g_phot_bp_mean_mag - g_phot_rp_mean_mag as bp_rp_color,
                COALESCE(p_phz_gal_prob, 0.5) as classification,
                SQRT(POWER(m_right_ascension - g_ra, 2) + POWER(m_declination - g_dec, 2)) * 3600 as angular_sep_arcsec
            FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
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
              AND p_phz_gal_prob IS NOT NULL
        """

        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        # Convert to lists
        q1_mags = []
        gaia_mags = []
        bp_rp_colors = []
        classifications = []
        ang_seps = []

        for q1_mag, gaia_mag, bp_rp, classification, ang_sep in results:
            q1_mags.append(q1_mag)
            gaia_mags.append(gaia_mag)
            bp_rp_colors.append(bp_rp)
            classifications.append(classification)
            ang_seps.append(ang_sep)

        # Calculate magnitude difference
        mag_diffs = np.array(q1_mags) - np.array(gaia_mags)
        classifications = np.array(classifications)

        print(f"  ✓ Loaded {len(results):,} Q1-Gaia sources at {radius}″ (VIS detection only)")
        print(f"    phz_gal_prob range: {np.min(classifications):.3f} to {np.max(classifications):.3f}")
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
            'classification': classifications,
            'angular_sep': np.array(ang_seps)
        }

    except Exception as e:
        print(f"  ✗ Error querying Q1-Gaia data: {e}")
        import traceback
        traceback.print_exc()
        return None


def query_ero_gaia_data_at_radius(username, radius):
    """Query ERO-Gaia crossmatch data at specified radius with BP-RP color and class_star.

    ERO Magnitude: e_v_mag_auto (VIS band automatic aperture magnitude)
    Magnitude Difference: mag_diff = e_v_mag_auto - g_phot_g_mean_mag
    Classification: e_v_class_star (0=extended, 1=star-like)
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

        # Query ERO-Gaia crossmatch table with BP-RP color and class_star
        # Using crossmatch_ero_gaia_0_3 table for 0.3 arcsec radius
        radius_suffix = str(radius).replace('.', '_')
        table_name = f"{username}.crossmatch_ero_gaia_{radius_suffix}"

        query = f"""
            SELECT
                e_v_mag_auto as ero_mag,
                g_phot_g_mean_mag as gaia_mag,
                g_phot_bp_mean_mag - g_phot_rp_mean_mag as bp_rp_color,
                COALESCE(e_v_class_star, 0.5) as classification,
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
              AND e_v_class_star IS NOT NULL
        """

        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        # Convert to lists
        ero_mags = []
        gaia_mags = []
        bp_rp_colors = []
        classifications = []
        ang_seps = []

        for ero_mag, gaia_mag, bp_rp, classification, ang_sep in results:
            ero_mags.append(ero_mag)
            gaia_mags.append(gaia_mag)
            bp_rp_colors.append(bp_rp)
            classifications.append(classification)
            ang_seps.append(ang_sep)

        # Calculate magnitude difference
        mag_diffs = np.array(ero_mags) - np.array(gaia_mags)
        classifications = np.array(classifications)

        print(f"  ✓ Loaded {len(results):,} ERO-Gaia sources at {radius}″")
        print(f"    class_star range: {np.min(classifications):.3f} to {np.max(classifications):.3f}")
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
            'classification': classifications,
            'angular_sep': np.array(ang_seps)
        }

    except Exception as e:
        print(f"  ✗ Error querying ERO-Gaia data: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_scatter_colored_by_bp_rp(ax, data, title, xlim=None, ylim=None):
    """Create scatter plot colored by BP-RP color.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : dict
        Data dictionary containing 'gaia_mag', 'mag_diff', 'bp_rp_color'
    title : str
        Plot title
    xlim : tuple, optional
        X-axis limits (xmin, xmax)
    ylim : tuple, optional
        Y-axis limits (ymin, ymax)

    Returns:
    --------
    scatter : matplotlib.collections.PathCollection
        Scatter plot object for colorbar
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


def plot_scatter_colored_by_classification(ax, data, title, source_type, xlim=None, ylim=None):
    """Create scatter plot colored by morphological classification.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : dict
        Data dictionary containing 'gaia_mag', 'mag_diff', 'classification'
    title : str
        Plot title
    source_type : str
        'Q1' or 'ERO' for colormap selection and labels
    xlim : tuple, optional
        X-axis limits (xmin, xmax)
    ylim : tuple, optional
        Y-axis limits (ymin, ymax)

    Returns:
    --------
    scatter : matplotlib.collections.PathCollection
        Scatter plot object for colorbar
    """
    # Choose colormap based on source type
    cmap = 'viridis_r' if source_type == 'Q1' else 'viridis'

    # Create scatter plot with continuous colormap
    scatter = ax.scatter(
        data['gaia_mag'],
        data['mag_diff'],
        c=data['classification'],
        s=15,
        alpha=0.6,
        edgecolors='none',
        cmap=cmap,
        vmin=0,
        vmax=1
    )

    ax.set_xlabel('Gaia G Magnitude', fontsize=11, fontweight='bold')
    ax.set_ylabel('Magnitude Difference', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Set axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))

    return scatter


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
    print("Magnitude Difference vs Magnitude: Color + Morphology Views")
    print(f"Q1-Gaia vs ERO-Gaia Crossmatch at {MATCHING_RADIUS}″ Radius")
    print("=" * 90)

    print("\nTop Row: Scatter plots colored by BP-RP color index")
    print("Bottom Row: Scatter plots colored by morphological classification")
    print("\nMagnitude Difference Equations:")
    print(f"  Q1-Gaia:  mag_diff = [-2.5×log10({ZPC_IE}×m_flux_detection_total)+{ZP_REF}] - g_phot_g_mean_mag")
    print(f"  ERO-Gaia: mag_diff = e_v_mag_auto - g_phot_g_mean_mag")
    print("\nMorphological Classification:")
    print("  Q1: phz_gal_prob (0=star-like, 1=galaxy-like)")
    print("  ERO: e_v_class_star (0=extended, 1=star-like)")
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
    print(f"\nCreating scatter plots and 2D histograms...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # Top row: scatter plots (mag diff vs mag, colored by BP-RP)
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]

    # Calculate unified axis limits for both rows
    mag_diff_min = min(q1_data['mag_diff'].min(), ero_data['mag_diff'].min())
    mag_diff_max = max(q1_data['mag_diff'].max(), ero_data['mag_diff'].max())
    ylim = (mag_diff_min, mag_diff_max)

    gaia_mag_min = min(q1_data['gaia_mag'].min(), ero_data['gaia_mag'].min())
    gaia_mag_max = max(q1_data['gaia_mag'].max(), ero_data['gaia_mag'].max())
    xlim = (gaia_mag_min - 0.5, gaia_mag_max + 1)

    scatter1 = plot_scatter_colored_by_bp_rp(ax1, q1_data,
                                             f'Q1-Gaia Mag. Diff. vs Gaia Mag. (Colored by BP-RP)',
                                             xlim=xlim, ylim=ylim)
    scatter2 = plot_scatter_colored_by_bp_rp(ax2, ero_data,
                                             f'ERO-Gaia Mag. Diff. vs Gaia Mag. (Colored by BP-RP)',
                                             xlim=xlim, ylim=ylim)

    # Bottom row: scatter plots colored by morphological classification
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]

    # Create morphological classification scatter plots
    scatter3 = plot_scatter_colored_by_classification(ax3, q1_data,
                                                      f'Q1-Gaia Mag. Diff. vs Gaia Mag. (Colored by phz_gal_prob)',
                                                      'Q1', xlim=xlim, ylim=ylim)
    scatter4 = plot_scatter_colored_by_classification(ax4, ero_data,
                                                      f'ERO-Gaia Mag. Diff. vs Gaia Mag. (Colored by class_star)',
                                                      'ERO', xlim=xlim, ylim=ylim)

    # Add individual colorbars for each scatter plot in the top row
    cbar1 = plt.colorbar(scatter1, ax=ax1, label='BP-RP Color Index')
    cbar2 = plt.colorbar(scatter2, ax=ax2, label='BP-RP Color Index')

    # Add individual colorbars for morphological classification in the bottom row
    cbar3 = plt.colorbar(scatter3, ax=ax3, label='phz_gal_prob')
    # cbar3.set_label('Galaxy Probability (phz_gal_prob)', fontsize=10, fontweight='bold')
    cbar3.ax.invert_yaxis()  # Flip for Q1 (so 1.0 purple at bottom, 0.0 yellow at top)
    cbar3.ax.text(2.6, 1.0, 'Point', transform=cbar3.ax.transAxes,
                  ha='left', va='center', fontsize=9, style='italic')
    cbar3.ax.text(2.6, 0.0, 'Extended', transform=cbar3.ax.transAxes,
                  ha='left', va='center', fontsize=9, style='italic')

    cbar4 = plt.colorbar(scatter4, ax=ax4, label='v_class_star')
    # cbar4.set_label('Class Star Value (e_v_class_star)', fontsize=10, fontweight='bold')
    cbar4.ax.text(2.6, 0.0, 'Extended', transform=cbar4.ax.transAxes,
                  ha='left', va='center', fontsize=9, style='italic')
    cbar4.ax.text(2.6, 1.0, 'Point', transform=cbar4.ax.transAxes,
                  ha='left', va='center', fontsize=9, style='italic')

    fig.suptitle(
        f'Magnitude Difference vs Gaia Magnitude at {MATCHING_RADIUS}″ Crossmatch Radius\n'
        f'Q1 = -2.5 × log({ZPC_IE} × flux) + {ZP_REF} - Gaia (n={len(q1_data["mag_diff"]):,}) | ERO = e_v_mag_auto - Gaia (n={len(ero_data["mag_diff"]):,})',
        fontsize=15, fontweight='bold')
    plt.subplots_adjust(top=0.90, bottom=0.08, wspace=0.08)

    # Save figure
    output_path = OUTPUT_DIR / f'mag_diff_vs_mag_{MATCHING_RADIUS:.1f}.png'
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
