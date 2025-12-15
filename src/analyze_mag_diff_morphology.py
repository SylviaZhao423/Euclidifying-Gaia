#!/usr/bin/env python3
"""
Plot Gaia G magnitude vs BP-RP color, colored by morphological classification with continuous colormap.
Compares Q1-Gaia and ERO-Gaia crossmatch data at 0.3 arcsec matching radius.

X-axis: Gaia BP-RP color index
Y-axis: Gaia G Magnitude (inverted, brighter at top)
Color: Continuous morphological classification values
  - Q1: phz_gal_prob (galaxy probability, 0=star-like, 1=galaxy-like)
  - ERO: e_v_class_star (class star value, 0=extended, 1=star-like)

Note: Magnitude difference is still calculated from source and Gaia magnitudes but not plotted.

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
from matplotlib.patches import Patch

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


def query_q1_gaia_data_with_classification(username, radius):
    """Query Q1-Gaia crossmatch data with phz_gal_prob.

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

        # Query Q1-Gaia crossmatch table with phz_gal_prob
        # Note: p_phz_gal_prob comes from the crossmatch table (p = PHZ)
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


def query_ero_gaia_data_with_classification(username, radius):
    """Query ERO-Gaia crossmatch data with class_star classification.

    ERO Magnitude: e_v_mag_auto (VIS band automatic aperture magnitude)
    Magnitude Difference: mag_diff = e_v_mag_auto - g_phot_g_mean_mag
    Classification: class_star (star-like vs extended)
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

        # Query ERO-Gaia crossmatch table with class_star
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


def plot_classification_scatter(ax, data, title, source_type, xlim=None, ylim=None):
    """Create scatter plot colored by morphological classification as continuous colormap.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : dict
        Data dictionary containing 'bp_rp_color', 'gaia_mag', 'classification'
    title : str
        Plot title
    source_type : str
        'Q1' or 'ERO' for colorbar labels
    xlim : tuple, optional
        X-axis limits
    ylim : tuple, optional
        Y-axis limits

    Returns:
    --------
    colorbar_obj : matplotlib.colorbar.Colorbar
        Colorbar object for display
    """
    # Choose colormap based on source type
    cmap = 'viridis_r' if source_type == 'Q1' else 'viridis'

    # Create scatter plot with continuous colormap
    scatter = ax.scatter(
        data['bp_rp_color'],
        data['gaia_mag'],
        c=data['classification'],
        s=15,
        alpha=0.6,
        edgecolors='none',
        cmap=cmap,
        vmin=0,
        vmax=1
    )

    ax.set_xlabel('Gaia BP-RP Color Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Gaia G Magnitude', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Invert Y-axis so brighter (smaller) magnitudes are at top

    # Set axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))

    # Add colorbar with appropriate label
    if source_type == 'Q1':
        cbar_label = 'Galaxy Probability (phz_gal_prob)'
    else:  # ERO
        cbar_label = 'Class Star Value (e_v_class_star)'

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(cbar_label, fontsize=10, fontweight='bold')

    # For Q1, flip the colorbar (so 1.0 purple at bottom, 0.0 yellow at top)
    if source_type == 'Q1':
        cbar.ax.invert_yaxis()
        # Add labels next to the tick values
        cbar.ax.text(2.6, 1.0, 'Point', transform=cbar.ax.transAxes,
                     ha='left', va='center', fontsize=9, style='italic')
        cbar.ax.text(2.6, 0.0, 'Extended', transform=cbar.ax.transAxes,
                     ha='left', va='center', fontsize=9, style='italic')
    else:  # ERO
        # Add labels next to the tick values
        cbar.ax.text(2.6, 0.0, 'Extended', transform=cbar.ax.transAxes,
                     ha='left', va='center', fontsize=9, style='italic')
        cbar.ax.text(2.6, 1.0, 'Point', transform=cbar.ax.transAxes,
                     ha='left', va='center', fontsize=9, style='italic')

    return cbar


def main():
    """Main execution."""
    print("=" * 90)
    print("Gaia G Magnitude vs BP-RP Color, Colored by Morphological Classification")
    print(f"Q1-Gaia vs ERO-Gaia Crossmatch at {MATCHING_RADIUS}″ Radius")
    print("=" * 90)

    print("\nNote: Magnitude difference is calculated but not plotted.")
    print("      Y-axis shows Gaia G magnitude (inverted scale, brighter at top)")
    print("=" * 90)

    # Get username from environment
    username = os.getenv('PGUSER')
    if not username:
        print("\nERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return

    print(f"\nLogged in as: {username}")

    # Query both datasets with classification
    print(f"\nQuerying Q1-Gaia crossmatch data at {MATCHING_RADIUS}″ with phz_gal_prob...")
    q1_data = query_q1_gaia_data_with_classification(username, MATCHING_RADIUS)
    if q1_data is None:
        return

    print(f"\nQuerying ERO-Gaia crossmatch data at {MATCHING_RADIUS}″ with class_star...")
    ero_data = query_ero_gaia_data_with_classification(username, MATCHING_RADIUS)
    if ero_data is None:
        print("WARNING: Could not load ERO-Gaia data")
        ero_data = None

    # Create 1x2 figure with classification-colored scatter plots
    print(f"\nCreating continuous colormap scatter plots...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax1 = axes[0]
    ax2 = axes[1]

    # Calculate unified axis limits
    gaia_mag_min = min(q1_data['gaia_mag'].min(), ero_data['gaia_mag'].min())
    gaia_mag_max = max(q1_data['gaia_mag'].max(), ero_data['gaia_mag'].max())
    ylim = (gaia_mag_max + 0.5, gaia_mag_min - 0.5)  # Inverted: fainter at bottom, brighter at top

    bp_rp_min = min(q1_data['bp_rp_color'].min(), ero_data['bp_rp_color'].min())
    bp_rp_max = max(q1_data['bp_rp_color'].max(), ero_data['bp_rp_color'].max())
    xlim = (bp_rp_min - 0.1, bp_rp_max + 0.1)

    # Create scatter plots with continuous colormaps
    plot_classification_scatter(ax1, q1_data,
                                f'Q1-Gaia vs BP-RP (n={len(q1_data["gaia_mag"]):,})',
                                'Q1', xlim=xlim, ylim=ylim)
    plot_classification_scatter(ax2, ero_data,
                                f'ERO-Gaia vs BP-RP (n={len(ero_data["gaia_mag"]):,})',
                                'ERO', xlim=xlim, ylim=ylim)

    # Add suptitle
    fig.suptitle(
        f'Gaia G Magnitude vs BP-RP Color, Colored by Morphological Classification at {MATCHING_RADIUS}″ Radius',
        fontsize=15, fontweight='bold')
    plt.subplots_adjust()

    # Save figure
    output_path = OUTPUT_DIR / f'mag_diff_morphology_q1_ero_{MATCHING_RADIUS:.1f}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {output_path}")
    plt.close()

    print("\n" + "=" * 90)
    print("Done!")
    print("=" * 90)


if __name__ == '__main__':
    main()
