#!/usr/bin/env python3
"""
Analyze ERO magnitude distribution separated by CLASS_STAR classification.

Shows combined, star (point-like), and galaxy (extended) magnitude distributions
separately so the galaxy dip at 18-18.5 is not hidden by the ~19:1 star domination.

Classification: v_class_star >= 0.65 → STAR, < 0.65 → GALAXY
Magnitude: v_mag_auto (VIS band automatic aperture magnitude)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import os

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

OUTPUT_DIR.mkdir(exist_ok=True)


def query_mag_dist_by_class(username, gaia_matched_only=False, use_gaia_mag=False):
    """Query ERO magnitude distribution split by CLASS_STAR classification."""
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed!")
        return None

    print(f"\nConnecting to WSDB...")
    try:
        conn = psycopg2.connect(
            host=WSDB_HOST,
            database=WSDB_DB,
            user=username
        )
    except Exception as e:
        print(f"ERROR connecting to WSDB: {e}")
        return None

    BIN_WIDTH = 0.25
    mag_bins = np.arange(10, 28, BIN_WIDTH)

    if gaia_matched_only:
        table_name = 'siwei_zhao_2025.crossmatch_ero_gaia_0_3'
        class_col = 'e_v_class_star'
        extra_filter = 'AND g_phot_g_mean_mag IS NOT NULL'
        if use_gaia_mag:
            mag_col = 'g_phot_g_mean_mag'
            print(f"Querying GAIA-MATCHED distribution (Gaia G mag) from {table_name}...")
        else:
            mag_col = 'e_v_mag_auto'
            print(f"Querying GAIA-MATCHED distribution (Euclid VIS mag) from {table_name}...")
    else:
        table_name = 'euclid_ero.merge_cat'
        mag_col = 'v_mag_auto'
        class_col = 'v_class_star'
        extra_filter = ''
        print(f"Querying ALL ERO magnitude distribution from {table_name}...")

    print(f"Classification: {class_col} >= 0.65 → STAR, < 0.65 → GALAXY")
    print(f"Magnitude: {mag_col}\n")

    cursor = conn.cursor()

    query = f"""
        SELECT
            FLOOR(({mag_col} - {mag_bins[0]}) / {BIN_WIDTH}) as bin_index,
            {mag_bins[0]} + (FLOOR(({mag_col} - {mag_bins[0]}) / {BIN_WIDTH}) * {BIN_WIDTH}) + {BIN_WIDTH/2} as bin_center,
            CASE
                WHEN {class_col} >= 0.65 THEN 'STAR'
                WHEN {class_col} < 0.65 THEN 'GALAXY'
            END as classification,
            COUNT(*) as count
        FROM {table_name}
        WHERE {mag_col} >= {mag_bins[0]}
          AND {mag_col} < {mag_bins[-1]}
          AND {mag_col} IS NOT NULL
          AND {class_col} IS NOT NULL
          {extra_filter}
        GROUP BY bin_index, bin_center, classification
        ORDER BY bin_index, classification
    """

    try:
        cursor.execute(query)
        results = cursor.fetchall()
        print(f"Query completed. Processing {len(results)} rows...\n")

        data_dict = {}
        for row in results:
            bin_idx, bin_center, classification, count = row
            data_dict[(int(bin_idx), classification)] = count

        bin_centers = []
        star_counts = []
        galaxy_counts = []
        total_counts = []

        for i in range(len(mag_bins) - 1):
            mag_min = mag_bins[i]
            mag_max = mag_bins[i + 1]
            bin_center = (mag_min + mag_max) / 2

            n_star = data_dict.get((i, 'STAR'), 0)
            n_galaxy = data_dict.get((i, 'GALAXY'), 0)
            total = n_star + n_galaxy

            if total > 0:
                bin_centers.append(bin_center)
                star_counts.append(n_star)
                galaxy_counts.append(n_galaxy)
                total_counts.append(total)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        cursor.close()
        conn.close()
        return None

    cursor.close()
    conn.close()

    return (np.array(bin_centers),
            np.array(star_counts),
            np.array(galaxy_counts),
            np.array(total_counts))


def create_visualization(all_data, gaia_euclid_data, gaia_gaia_data):
    """Create 3x3-panel magnitude distribution: all ERO, Gaia-matched (Euclid mag), Gaia-matched (Gaia mag)."""

    print(f"\nCreating visualization...")

    bin_centers_all, star_counts_all, galaxy_counts_all, total_counts_all = all_data
    bin_centers_gaia_e, star_counts_gaia_e, galaxy_counts_gaia_e, total_counts_gaia_e = gaia_euclid_data
    bin_centers_gaia_g, star_counts_gaia_g, galaxy_counts_gaia_g, total_counts_gaia_g = gaia_gaia_data

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('ERO Magnitude Distribution by CLASS_STAR Classification',
                 fontsize=14, fontweight='bold', y=0.95)

    BIN_WIDTH = 0.25

    # Highlight region where class_star is unreliable (18.0-18.5)
    highlight_min, highlight_max = 17.8, 18.5

    # Gaia detection limit line
    gaia_limit = 21.0

    # Common y-axis limits (find max across all datasets)
    y_max = max(total_counts_all.max(), star_counts_all.max(), galaxy_counts_all.max(),
                total_counts_gaia_e.max(), star_counts_gaia_e.max(), galaxy_counts_gaia_e.max(),
                total_counts_gaia_g.max(), star_counts_gaia_g.max(), galaxy_counts_gaia_g.max())
    y_min = 1  # For log scale

    # Common x-axis limits
    x_min, x_max = 10, 28

    # Colors (swapped: coral for stars, steelblue for galaxies)
    star_color = 'coral'
    star_edge = 'darkred'
    galaxy_color = 'steelblue'
    galaxy_edge = 'darkblue'

    # --- Top Row: All ERO sources ---
    # Combined (all sources)
    ax = axes[0, 0]
    ax.axvspan(highlight_min, highlight_max, alpha=0.5, color='#FFCDD2')
    # ax.axvline(x=gaia_limit, color='seagreen', linestyle='--', linewidth=1.5, label=f'Gaia limit (~{gaia_limit})')
    ax.fill_between(bin_centers_all, total_counts_all, y_min, step='mid', color='gray', alpha=0.7)
    ax.set_xlabel('Magnitude (v_mag_auto)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title(f'All Sources (n={total_counts_all.sum():,})', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(y_min, y_max * 2)
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.legend(fontsize=9, loc='upper left')

    # Stars (point-like)
    ax = axes[0, 1]
    ax.axvspan(highlight_min, highlight_max, alpha=0.5, color='#FFCDD2')
    # ax.axvline(x=gaia_limit, color='seagreen', linestyle='--', linewidth=1.5)
    ax.fill_between(bin_centers_all, star_counts_all, y_min, step='mid', color=star_color, alpha=0.7)
    ax.set_xlabel('Magnitude (v_mag_auto)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title(f'STAR (v_class_star ≥ 0.65, n={star_counts_all.sum():,})', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(y_min, y_max * 2)
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45, labelsize=9)

    # Galaxies (extended)
    ax = axes[0, 2]
    ax.axvspan(highlight_min, highlight_max, alpha=0.5, color='#FFCDD2')
    # ax.axvline(x=gaia_limit, color='seagreen', linestyle='--', linewidth=1.5)
    ax.fill_between(bin_centers_all, galaxy_counts_all, y_min, step='mid', color=galaxy_color, alpha=0.7)
    ax.set_xlabel('Magnitude (v_mag_auto)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title(f'GALAXY (v_class_star < 0.65, n={galaxy_counts_all.sum():,})', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(y_min, y_max * 2)
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45, labelsize=9)

    # --- Middle Row: Gaia-matched sources (Euclid VIS magnitude) ---
    # Combined (all sources)
    ax = axes[1, 0]
    ax.axvspan(highlight_min, highlight_max, alpha=0.5, color='#FFCDD2')
    ax.axvline(x=gaia_limit, color='seagreen', linestyle='--', linewidth=1.5, label=f'Gaia limit (~{gaia_limit})')
    ax.fill_between(bin_centers_gaia_e, total_counts_gaia_e, y_min, step='mid', color='gray', alpha=0.7)
    ax.set_xlabel('Magnitude (v_mag_auto)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title(f'Gaia-Matched: All (n={total_counts_gaia_e.sum():,})', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(y_min, y_max * 2)
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.legend(fontsize=9, loc='upper left')

    # Stars (point-like)
    ax = axes[1, 1]
    ax.axvspan(highlight_min, highlight_max, alpha=0.5, color='#FFCDD2')
    ax.axvline(x=gaia_limit, color='seagreen', linestyle='--', linewidth=1.5, label=f'Gaia limit (~{gaia_limit})')
    ax.fill_between(bin_centers_gaia_e, star_counts_gaia_e, y_min, step='mid', color=star_color, alpha=0.7)
    ax.set_xlabel('Magnitude (v_mag_auto)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title(f'Gaia-Matched: STAR (n={star_counts_gaia_e.sum():,})', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(y_min, y_max * 2)
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45, labelsize=9)

    # Galaxies (extended)
    ax = axes[1, 2]
    ax.axvspan(highlight_min, highlight_max, alpha=0.5, color='#FFCDD2')
    ax.axvline(x=gaia_limit, color='seagreen', linestyle='--', linewidth=1.5, label=f'Gaia limit (~{gaia_limit})')
    ax.fill_between(bin_centers_gaia_e, galaxy_counts_gaia_e, y_min, step='mid', color=galaxy_color, alpha=0.7)
    ax.set_xlabel('Magnitude (v_mag_auto)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title(f'Gaia-Matched: GALAXY (n={galaxy_counts_gaia_e.sum():,})', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(y_min, y_max * 2)
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45, labelsize=9)

    # --- Bottom Row: Gaia-matched sources (Gaia G magnitude) ---
    # Training magnitude cut for Gaia G
    gaia_mag_cut = 18.5

    # Combined (all sources)
    ax = axes[2, 0]
    ax.fill_between(bin_centers_gaia_g, total_counts_gaia_g, y_min, step='mid', color='gray', alpha=0.7)
    ax.set_xlabel('Magnitude (phot_g_mean_mag)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title(f'Gaia-Matched: All (n={total_counts_gaia_g.sum():,})', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(y_min, y_max * 2)
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45, labelsize=9)

    # Stars (point-like)
    ax = axes[2, 1]
    ax.fill_between(bin_centers_gaia_g, star_counts_gaia_g, y_min, step='mid', color=star_color, alpha=0.7)
    ax.set_xlabel('Magnitude (phot_g_mean_mag)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title(f'Gaia-Matched: STAR (n={star_counts_gaia_g.sum():,})', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(y_min, y_max * 2)
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45, labelsize=9)

    # Galaxies (extended)
    ax = axes[2, 2]
    ax.axvline(x=gaia_mag_cut, color='red', linestyle='--', linewidth=1.5, label=f'Training cut (G~{gaia_mag_cut})')
    ax.fill_between(bin_centers_gaia_g, galaxy_counts_gaia_g, y_min, step='mid', color=galaxy_color, alpha=0.7)
    ax.set_xlabel('Magnitude (phot_g_mean_mag)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title(f'Gaia-Matched: GALAXY (n={galaxy_counts_gaia_g.sum():,})', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(y_min, y_max * 2)
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.legend(fontsize=9, loc='upper left')

    # Add row labels on the left
    fig.text(0.02, 0.82, 'All ERO\n(Euclid VIS)', fontsize=11, fontweight='bold', rotation=90, va='center', ha='center')
    fig.text(0.02, 0.50, 'Gaia-Matched\n(Euclid VIS)', fontsize=11, fontweight='bold', rotation=90, va='center', ha='center')
    fig.text(0.02, 0.18, 'Gaia-Matched\n(Gaia G)', fontsize=11, fontweight='bold', rotation=90, va='center', ha='center')

    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / 'ero_mag_dist_by_class.png', dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {OUTPUT_DIR / 'ero_mag_dist_by_class.png'}")
    plt.close()


def main():
    print("=" * 70)
    print("ERO Magnitude Distribution by CLASS_STAR Classification")
    print("=" * 70)

    username = os.getenv('PGUSER')
    if not username:
        print("\nERROR: PGUSER environment variable not set!")
        return

    print(f"\nLogged in as: {username}")

    # Query all ERO sources
    print("\n--- Querying ALL ERO sources ---")
    all_data = query_mag_dist_by_class(username, gaia_matched_only=False)
    if all_data is None:
        print("Failed to query all ERO data from WSDB")
        return

    # Query Gaia-matched sources (Euclid VIS magnitude)
    print("\n--- Querying GAIA-MATCHED sources (Euclid VIS mag) ---")
    gaia_euclid_data = query_mag_dist_by_class(username, gaia_matched_only=True, use_gaia_mag=False)
    if gaia_euclid_data is None:
        print("Failed to query Gaia-matched data (Euclid mag) from WSDB")
        return

    # Query Gaia-matched sources (Gaia G magnitude)
    print("\n--- Querying GAIA-MATCHED sources (Gaia G mag) ---")
    gaia_gaia_data = query_mag_dist_by_class(username, gaia_matched_only=True, use_gaia_mag=True)
    if gaia_gaia_data is None:
        print("Failed to query Gaia-matched data (Gaia mag) from WSDB")
        return

    create_visualization(all_data, gaia_euclid_data, gaia_gaia_data)

    print("\nDone!")


if __name__ == '__main__':
    main()
