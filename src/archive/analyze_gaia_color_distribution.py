#!/usr/bin/env python3
"""
Plot histogram of Gaia BP-RP color distribution.
Compares Q1-Gaia, ERO-Gaia crossmatch data, and all Gaia DR3 sources.

Shows the distribution of Gaia source colors to understand the population characteristics
and selection effects in the crossmatched samples.

BP-RP color index = g_phot_bp_mean_mag - g_phot_rp_mean_mag
  - BP-RP < 0: Very blue (hot stars, blue galaxies)
  - BP-RP ≈ 0-1: Blue to yellow (main sequence stars, star-forming galaxies)
  - BP-RP ≈ 1-2: Yellow to red (cool stars, red galaxies)
  - BP-RP > 2: Very red (M-dwarfs, very cool stars)

Filters:
- Angular separation <= 0.3 arcsec for crossmatched sources
- VIS_DET = 1 for Q1 (VIS mosaic detection only)
- Valid BP and RP magnitude values
- Magnitude cut: G < 22 for all Gaia sources (performance)
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

# Matching radius
MATCHING_RADIUS = 0.3  # arcsec

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


def query_q1_gaia_colors(username, radius):
    """Query Q1-Gaia crossmatch data for BP-RP colors.

    Returns BP-RP color index for all matched sources.
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

        # Query Q1-Gaia crossmatch table for BP-RP color
        query = f"""
            SELECT
                g_phot_bp_mean_mag - g_phot_rp_mean_mag as bp_rp_color,
                g_phot_g_mean_mag as gaia_mag
            FROM siwei_zhao_2025.crossmatch_q1_mer_gaia
            WHERE g_phot_bp_mean_mag IS NOT NULL
              AND g_phot_rp_mean_mag IS NOT NULL
              AND g_phot_g_mean_mag IS NOT NULL
              AND g_phot_g_mean_mag > 0
              AND g_phot_g_mean_mag < 30
              AND m_vis_det = 1
              AND SQRT(POWER(m_right_ascension - g_ra, 2) + POWER(m_declination - g_dec, 2)) * 3600 <= {radius}
        """

        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        # Extract colors and magnitudes
        bp_rp_colors = np.array([row[0] for row in results])
        gaia_mags = np.array([row[1] for row in results])

        print(f"  ✓ Loaded {len(bp_rp_colors):,} Q1-Gaia sources at {radius}″ (VIS detection only)")
        print(f"    BP-RP color range: {np.min(bp_rp_colors):.3f} to {np.max(bp_rp_colors):.3f}")
        print(f"    BP-RP color mean: {np.mean(bp_rp_colors):.3f}")
        print(f"    BP-RP color median: {np.median(bp_rp_colors):.3f}")
        print(f"    BP-RP color std dev: {np.std(bp_rp_colors):.3f}")
        print(f"    Gaia magnitude range: {np.min(gaia_mags):.3f} to {np.max(gaia_mags):.3f}")

        return {
            'bp_rp_color': bp_rp_colors,
            'gaia_mag': gaia_mags
        }

    except Exception as e:
        print(f"  ✗ Error querying Q1-Gaia data: {e}")
        import traceback
        traceback.print_exc()
        return None


def query_ero_gaia_colors(username, radius):
    """Query ERO-Gaia crossmatch data for BP-RP colors.

    Returns BP-RP color index for all matched sources.
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

        # Query ERO-Gaia crossmatch table for BP-RP color
        radius_suffix = str(radius).replace('.', '_')
        table_name = f"{username}.crossmatch_ero_gaia_{radius_suffix}"

        query = f"""
            SELECT
                g_phot_bp_mean_mag - g_phot_rp_mean_mag as bp_rp_color,
                g_phot_g_mean_mag as gaia_mag
            FROM {table_name}
            WHERE g_phot_bp_mean_mag IS NOT NULL
              AND g_phot_rp_mean_mag IS NOT NULL
              AND g_phot_g_mean_mag IS NOT NULL
              AND g_phot_g_mean_mag > 0
              AND g_phot_g_mean_mag < 30
        """

        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        # Extract colors and magnitudes
        bp_rp_colors = np.array([row[0] for row in results])
        gaia_mags = np.array([row[1] for row in results])

        print(f"  ✓ Loaded {len(bp_rp_colors):,} ERO-Gaia sources at {radius}″")
        print(f"    BP-RP color range: {np.min(bp_rp_colors):.3f} to {np.max(bp_rp_colors):.3f}")
        print(f"    BP-RP color mean: {np.mean(bp_rp_colors):.3f}")
        print(f"    BP-RP color median: {np.median(bp_rp_colors):.3f}")
        print(f"    BP-RP color std dev: {np.std(bp_rp_colors):.3f}")
        print(f"    Gaia magnitude range: {np.min(gaia_mags):.3f} to {np.max(gaia_mags):.3f}")

        return {
            'bp_rp_color': bp_rp_colors,
            'gaia_mag': gaia_mags
        }

    except Exception as e:
        print(f"  ✗ Error querying ERO-Gaia data: {e}")
        import traceback
        traceback.print_exc()
        return None


def query_all_gaia_colors(username, mag_limit=22):
    """Query all Gaia DR3 sources for BP-RP colors.

    Returns BP-RP color index for all Gaia sources (with magnitude cut for performance).

    Parameters:
    -----------
    username : str
        WSDB username
    mag_limit : float
        Magnitude limit for Gaia G-band (default: 22 mag)
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

        print(f"    Querying all Gaia DR3 sources (G < {mag_limit} mag)...")
        print(f"    This may take a few minutes for ~100M+ sources...")

        # Query all Gaia sources with BP-RP color and magnitude cut
        query = f"""
            SELECT
                phot_bp_mean_mag - phot_rp_mean_mag as bp_rp_color,
                phot_g_mean_mag as gaia_mag
            FROM gaia_dr3.gaia_source
            WHERE phot_bp_mean_mag IS NOT NULL
              AND phot_rp_mean_mag IS NOT NULL
              AND phot_g_mean_mag IS NOT NULL
              AND phot_g_mean_mag > 0
              AND phot_g_mean_mag < {mag_limit}
        """

        cursor.execute(query)
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        # Extract colors and magnitudes
        bp_rp_colors = np.array([row[0] for row in results])
        gaia_mags = np.array([row[1] for row in results])

        print(f"  ✓ Loaded {len(bp_rp_colors):,} Gaia DR3 sources (G < {mag_limit} mag)")
        print(f"    BP-RP color range: {np.min(bp_rp_colors):.3f} to {np.max(bp_rp_colors):.3f}")
        print(f"    BP-RP color mean: {np.mean(bp_rp_colors):.3f}")
        print(f"    BP-RP color median: {np.median(bp_rp_colors):.3f}")
        print(f"    BP-RP color std dev: {np.std(bp_rp_colors):.3f}")
        print(f"    Gaia magnitude range: {np.min(gaia_mags):.3f} to {np.max(gaia_mags):.3f}")

        return {
            'bp_rp_color': bp_rp_colors,
            'gaia_mag': gaia_mags
        }

    except Exception as e:
        print(f"  ✗ Error querying all Gaia data: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_color_histogram(ax, data, title, color='steelblue', bins=100):
    """Create histogram of BP-RP color distribution.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : dict
        Data dictionary containing 'bp_rp_color'
    title : str
        Plot title
    color : str
        Histogram color
    bins : int
        Number of bins for histogram
    """
    bp_rp = data['bp_rp_color']

    # Create histogram
    counts, bin_edges, patches = ax.hist(bp_rp, bins=bins, color=color,
                                         alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add mean and median lines
    mean_val = np.mean(bp_rp)
    median_val = np.median(bp_rp)

    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
               label=f'Mean = {mean_val:.3f}')
    ax.axvline(median_val, color='orange', linestyle='--', linewidth=2,
               label=f'Median = {median_val:.3f}')

    ax.set_xlabel('Gaia BP-RP Color Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=10)

    # Format y-axis with comma separators
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))

    # Add text annotations for color ranges
    y_max = ax.get_ylim()[1]
    ax.text(-0.5, y_max * 0.95, 'Very Blue\n(Hot stars)',
            ha='center', va='top', fontsize=8, style='italic', alpha=0.7)
    ax.text(0.5, y_max * 0.95, 'Blue-Yellow\n(Main Seq.)',
            ha='center', va='top', fontsize=8, style='italic', alpha=0.7)
    ax.text(1.5, y_max * 0.95, 'Yellow-Red\n(Cool stars)',
            ha='center', va='top', fontsize=8, style='italic', alpha=0.7)
    ax.text(2.5, y_max * 0.95, 'Very Red\n(M-dwarfs)',
            ha='center', va='top', fontsize=8, style='italic', alpha=0.7)


def plot_cumulative_distribution(ax, data, title, color='steelblue'):
    """Create cumulative distribution of BP-RP colors.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : dict
        Data dictionary containing 'bp_rp_color'
    title : str
        Plot title
    color : str
        Line color
    """
    bp_rp = data['bp_rp_color']

    # Sort the data
    sorted_bp_rp = np.sort(bp_rp)
    cumulative = np.arange(1, len(sorted_bp_rp) + 1) / len(sorted_bp_rp) * 100

    # Plot cumulative distribution
    ax.plot(sorted_bp_rp, cumulative, color=color, linewidth=2)

    # Add percentile lines
    percentiles = [25, 50, 75]
    percentile_vals = np.percentile(bp_rp, percentiles)

    for p, val in zip(percentiles, percentile_vals):
        ax.axvline(val, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.axhline(p, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.text(val, p + 2, f'{p}th: {val:.2f}', fontsize=8, ha='center')

    ax.set_xlabel('Gaia BP-RP Color Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)


def main():
    """Main execution."""
    print("=" * 90)
    print("Gaia BP-RP Color Distribution Analysis")
    print(f"Q1-Gaia vs ERO-Gaia vs Combined")
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
    q1_data = query_q1_gaia_colors(username, MATCHING_RADIUS)
    if q1_data is None:
        return

    print(f"\nQuerying ERO-Gaia crossmatch data at {MATCHING_RADIUS}″...")
    ero_data = query_ero_gaia_colors(username, MATCHING_RADIUS)
    if ero_data is None:
        print("WARNING: Could not load ERO-Gaia data")
        return

    # Combine Q1 and ERO data
    print(f"\nCombining Q1-Gaia and ERO-Gaia datasets...")
    combined_bp_rp = np.concatenate([q1_data['bp_rp_color'], ero_data['bp_rp_color']])
    combined_gaia_mag = np.concatenate([q1_data['gaia_mag'], ero_data['gaia_mag']])
    combined_data = {
        'bp_rp_color': combined_bp_rp,
        'gaia_mag': combined_gaia_mag
    }
    print(f"  ✓ Combined dataset has {len(combined_bp_rp):,} sources")
    print(f"    BP-RP color range: {np.min(combined_bp_rp):.3f} to {np.max(combined_bp_rp):.3f}")
    print(f"    BP-RP color mean: {np.mean(combined_bp_rp):.3f}")
    print(f"    BP-RP color median: {np.median(combined_bp_rp):.3f}")
    print(f"    BP-RP color std dev: {np.std(combined_bp_rp):.3f}")

    # Create 2x3 figure
    print(f"\nCreating color distribution plots...")
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))

    # Top row: Histograms
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[0, 2]

    plot_color_histogram(ax1, q1_data,
                         f'Q1-Gaia BP-RP Color Distribution (n={len(q1_data["bp_rp_color"]):,})',
                         color='steelblue', bins=100)
    plot_color_histogram(ax2, ero_data,
                         f'ERO-Gaia BP-RP Color Distribution (n={len(ero_data["bp_rp_color"]):,})',
                         color='coral', bins=100)
    plot_color_histogram(ax3, combined_data,
                         f'Combined Q1+ERO BP-RP Color Distribution (n={len(combined_data["bp_rp_color"]):,})',
                         color='purple', bins=100)

    # Bottom row: Cumulative distributions
    ax4 = axes[1, 0]
    ax5 = axes[1, 1]
    ax6 = axes[1, 2]

    plot_cumulative_distribution(ax4, q1_data,
                                 f'Q1-Gaia Cumulative BP-RP Distribution',
                                 color='steelblue')
    plot_cumulative_distribution(ax5, ero_data,
                                 f'ERO-Gaia Cumulative BP-RP Distribution',
                                 color='coral')
    plot_cumulative_distribution(ax6, combined_data,
                                 f'Combined Q1+ERO Cumulative BP-RP Distribution',
                                 color='purple')

    # Add suptitle
    fig.suptitle(
        f'Gaia BP-RP Color Distribution Comparison at {MATCHING_RADIUS}″ Crossmatch Radius\n'
        f'Q1-Gaia | ERO-Gaia | Combined Q1+ERO',
        fontsize=15, fontweight='bold')
    plt.subplots_adjust(top=0.92, bottom=0.08, wspace=0.25, hspace=0.3)

    # Save figure
    output_path = OUTPUT_DIR / f'gaia_color_distribution_q1_ero_combined_{MATCHING_RADIUS:.1f}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {output_path}")
    plt.close()

    # Print comparison statistics
    print("\n" + "=" * 90)
    print("Color Distribution Comparison:")
    print("=" * 90)
    print(f"{'Dataset':<20} {'Mean':<12} {'Median':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}")
    print("-" * 90)

    q1_bp_rp = q1_data['bp_rp_color']
    ero_bp_rp = ero_data['bp_rp_color']
    combined_bp_rp = combined_data['bp_rp_color']

    print(f"{'Q1-Gaia':<20} {np.mean(q1_bp_rp):<12.4f} {np.median(q1_bp_rp):<12.4f} "
          f"{np.std(q1_bp_rp):<12.4f} {np.min(q1_bp_rp):<12.4f} {np.max(q1_bp_rp):<12.4f}")
    print(f"{'ERO-Gaia':<20} {np.mean(ero_bp_rp):<12.4f} {np.median(ero_bp_rp):<12.4f} "
          f"{np.std(ero_bp_rp):<12.4f} {np.min(ero_bp_rp):<12.4f} {np.max(ero_bp_rp):<12.4f}")
    print(f"{'Combined Q1+ERO':<20} {np.mean(combined_bp_rp):<12.4f} {np.median(combined_bp_rp):<12.4f} "
          f"{np.std(combined_bp_rp):<12.4f} {np.min(combined_bp_rp):<12.4f} {np.max(combined_bp_rp):<12.4f}")

    print("\nPercentiles:")
    print(f"{'Dataset':<20} {'25th':<12} {'50th':<12} {'75th':<12} {'90th':<12} {'95th':<12}")
    print("-" * 90)
    q1_percentiles = np.percentile(q1_bp_rp, [25, 50, 75, 90, 95])
    ero_percentiles = np.percentile(ero_bp_rp, [25, 50, 75, 90, 95])
    combined_percentiles = np.percentile(combined_bp_rp, [25, 50, 75, 90, 95])

    print(f"{'Q1-Gaia':<20} {q1_percentiles[0]:<12.4f} {q1_percentiles[1]:<12.4f} "
          f"{q1_percentiles[2]:<12.4f} {q1_percentiles[3]:<12.4f} {q1_percentiles[4]:<12.4f}")
    print(f"{'ERO-Gaia':<20} {ero_percentiles[0]:<12.4f} {ero_percentiles[1]:<12.4f} "
          f"{ero_percentiles[2]:<12.4f} {ero_percentiles[3]:<12.4f} {ero_percentiles[4]:<12.4f}")
    print(f"{'Combined Q1+ERO':<20} {combined_percentiles[0]:<12.4f} {combined_percentiles[1]:<12.4f} "
          f"{combined_percentiles[2]:<12.4f} {combined_percentiles[3]:<12.4f} {combined_percentiles[4]:<12.4f}")

    print("\n" + "=" * 90)
    print("Done!")
    print("=" * 90)


if __name__ == '__main__':
    main()
