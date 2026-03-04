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


def query_mag_dist_by_class(username, gaia_matched_only=False, use_gaia_mag=False,
                            gaia_mag_cut_filter=None, euclid_mag_cut_filter=None):
    """Query ERO magnitude distribution split by CLASS_STAR classification.

    Parameters
    ----------
    gaia_mag_cut_filter : float or None
        If set, only include sources with Gaia G mag >= this value.
        Only applies when gaia_matched_only=True and use_gaia_mag=True.
    euclid_mag_cut_filter : float or None
        If set, only include sources with Euclid VIS mag >= this value.
        Only applies when gaia_matched_only=True and use_gaia_mag=True.
    """
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
        extra_filter = 'AND g_phot_g_mean_mag IS NOT NULL AND e_v_class_star IS NOT NULL'
        if use_gaia_mag:
            mag_col = 'g_phot_g_mean_mag'
            if gaia_mag_cut_filter is not None:
                extra_filter += f' AND g_phot_g_mean_mag >= {gaia_mag_cut_filter}'
                print(f"Querying GAIA-MATCHED distribution (Gaia G mag, G >= {gaia_mag_cut_filter}) from {table_name}...")
            elif euclid_mag_cut_filter is not None:
                extra_filter += f' AND e_v_mag_auto >= {euclid_mag_cut_filter}'
                print(f"Querying GAIA-MATCHED distribution (Gaia G mag, VIS >= {euclid_mag_cut_filter}) from {table_name}...")
            else:
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


def _to_density(counts, bin_width):
    """Convert raw counts to density (integrates to 1)."""
    total = counts.sum()
    if total == 0:
        return counts * 0.0
    return counts / (total * bin_width)


def create_visualization(all_data, gaia_euclid_data, gaia_gaia_data, gaia_gaia_faint_data,
                         gaia_mag_cut_filter=18.5):
    """Create 4x5-panel magnitude distribution."""

    print(f"\nCreating visualization...")

    bin_centers_all, star_counts_all, galaxy_counts_all, total_counts_all = all_data
    bin_centers_gaia_e, star_counts_gaia_e, galaxy_counts_gaia_e, total_counts_gaia_e = gaia_euclid_data
    bin_centers_gaia_g, star_counts_gaia_g, galaxy_counts_gaia_g, total_counts_gaia_g = gaia_gaia_data
    bin_centers_faint, star_counts_faint, galaxy_counts_faint, total_counts_faint = gaia_gaia_faint_data

    BIN_WIDTH = 0.25
    MIN_COUNT = 10

    # Convert counts to density (for col 0)
    total_dens_all = _to_density(total_counts_all, BIN_WIDTH)
    total_dens_gaia_e = _to_density(total_counts_gaia_e, BIN_WIDTH)
    total_dens_gaia_g = _to_density(total_counts_gaia_g, BIN_WIDTH)
    total_dens_faint = _to_density(total_counts_faint, BIN_WIDTH)

    # Star and galaxy fractions
    star_frac_all = np.where(total_counts_all >= MIN_COUNT, star_counts_all / total_counts_all, np.nan)
    star_frac_gaia_e = np.where(total_counts_gaia_e >= MIN_COUNT, star_counts_gaia_e / total_counts_gaia_e, np.nan)
    star_frac_gaia_g = np.where(total_counts_gaia_g >= MIN_COUNT, star_counts_gaia_g / total_counts_gaia_g, np.nan)
    star_frac_faint = np.where(total_counts_faint >= MIN_COUNT, star_counts_faint / total_counts_faint, np.nan)
    gal_frac_all = np.where(total_counts_all >= MIN_COUNT, galaxy_counts_all / total_counts_all, np.nan)
    gal_frac_gaia_e = np.where(total_counts_gaia_e >= MIN_COUNT, galaxy_counts_gaia_e / total_counts_gaia_e, np.nan)
    gal_frac_gaia_g = np.where(total_counts_gaia_g >= MIN_COUNT, galaxy_counts_gaia_g / total_counts_gaia_g, np.nan)
    gal_frac_faint = np.where(total_counts_faint >= MIN_COUNT, galaxy_counts_faint / total_counts_faint, np.nan)

    fig, axes = plt.subplots(4, 5, figsize=(30, 18))
    fig.suptitle('ERO Magnitude Distribution by CLASS_STAR Classification',
                 fontsize=15, fontweight='bold', y=0.99)

    highlight_min, highlight_max = 17, 19
    gaia_mag_cut = 18.5
    x_min, x_max = 10, 28
    star_color = 'coral'
    galaxy_color = 'steelblue'
    vis_label = 'Magnitude (v_mag_auto)'
    gaia_label = 'Magnitude (phot_g_mean_mag)'

    # Counts and percentages
    n_total_all = int(total_counts_all.sum())
    n_star_all = int(star_counts_all.sum())
    n_gal_all = int(galaxy_counts_all.sum())

    n_total_ge = int(total_counts_gaia_e.sum())
    n_star_ge = int(star_counts_gaia_e.sum())
    n_gal_ge = int(galaxy_counts_gaia_e.sum())
    pct_total_ge = 100 * n_total_ge / n_total_all if n_total_all > 0 else 0
    pct_star_ge = 100 * n_star_ge / n_star_all if n_star_all > 0 else 0
    pct_gal_ge = 100 * n_gal_ge / n_gal_all if n_gal_all > 0 else 0

    n_total_gg = int(total_counts_gaia_g.sum())
    n_star_gg = int(star_counts_gaia_g.sum())
    n_gal_gg = int(galaxy_counts_gaia_g.sum())
    pct_total_gg = 100 * n_total_gg / n_total_all if n_total_all > 0 else 0
    pct_star_gg = 100 * n_star_gg / n_star_all if n_star_all > 0 else 0
    pct_gal_gg = 100 * n_gal_gg / n_gal_all if n_gal_all > 0 else 0

    n_total_f = int(total_counts_faint.sum())
    n_star_f = int(star_counts_faint.sum())
    n_gal_f = int(galaxy_counts_faint.sum())
    pct_total_f = 100 * n_total_f / n_total_gg if n_total_gg > 0 else 0
    pct_star_f = 100 * n_star_f / n_star_gg if n_star_gg > 0 else 0
    pct_gal_f = 100 * n_gal_f / n_gal_gg if n_gal_gg > 0 else 0

    # Row data: (bin_centers, total_dens, star_frac, star_counts, gal_frac, gal_counts,
    #            xlabel, title_total, title_star_count, title_star_frac, title_gal_count, title_gal_frac,
    #            vlines, highlight)
    rows = [
        (bin_centers_all, total_dens_all, star_frac_all, star_counts_all, gal_frac_all, galaxy_counts_all,
         vis_label,
         f'All ERO Sources (n={n_total_all:,})',
         f'Star Count (n={n_star_all:,})', f'Star Fraction (n={n_star_all:,})',
         f'Galaxy Count (n={n_gal_all:,})', f'Galaxy Fraction (n={n_gal_all:,})',
         [], False),
        (bin_centers_gaia_e, total_dens_gaia_e, star_frac_gaia_e, star_counts_gaia_e, gal_frac_gaia_e, galaxy_counts_gaia_e,
         vis_label,
         f'Gaia-Matched in Euclid VIS (n={n_total_ge:,}, {pct_total_ge:.1f}%)',
         f'Star Count (n={n_star_ge:,}, {pct_star_ge:.1f}%)', f'Star Fraction (n={n_star_ge:,}, {pct_star_ge:.1f}%)',
         f'Galaxy Count (n={n_gal_ge:,}, {pct_gal_ge:.1f}%)', f'Galaxy Fraction (n={n_gal_ge:,}, {pct_gal_ge:.1f}%)',
         [], True),
        (bin_centers_gaia_g, total_dens_gaia_g, star_frac_gaia_g, star_counts_gaia_g, gal_frac_gaia_g, galaxy_counts_gaia_g,
         gaia_label,
         f'Gaia-Matched in Gaia G (n={n_total_gg:,}, {pct_total_gg:.1f}%)',
         f'Star Count (n={n_star_gg:,}, {pct_star_gg:.1f}%)', f'Star Fraction (n={n_star_gg:,}, {pct_star_gg:.1f}%)',
         f'Galaxy Count (n={n_gal_gg:,}, {pct_gal_gg:.1f}%)', f'Galaxy Fraction (n={n_gal_gg:,}, {pct_gal_gg:.1f}%)',
         [gaia_mag_cut], False),
        (bin_centers_faint, total_dens_faint, star_frac_faint, star_counts_faint, gal_frac_faint, galaxy_counts_faint,
         gaia_label,
         f'Gaia G >= {gaia_mag_cut_filter} (n={n_total_f:,}, {pct_total_f:.1f}%)',
         f'Star Count (n={n_star_f:,}, {pct_star_f:.1f}%)', f'Star Fraction (n={n_star_f:,}, {pct_star_f:.1f}%)',
         f'Galaxy Count (n={n_gal_f:,}, {pct_gal_f:.1f}%)', f'Galaxy Fraction (n={n_gal_f:,}, {pct_gal_f:.1f}%)',
         [gaia_mag_cut], False),
    ]

    n_rows = len(rows)
    for row_idx, (bc, td, sf, sc, gf, gc, xlabel, t0, t1, t2, t3, t4, vlines, hl) in enumerate(rows):
        def _setup(ax, xlabel=xlabel, vlines=vlines, row_idx=row_idx):
            ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
            ax.set_xlim(x_min, x_max)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            for vl in vlines:
                ax.axvline(x=vl, color='#F27A71', linestyle='--', linewidth=1.5,
                           label=f'Training cut (G~{vl})')
            if vlines and row_idx == n_rows - 1:
                ax.legend(fontsize=8, loc='upper left')

        # Col 0: Total density
        ax = axes[row_idx, 0]
        ax.fill_between(bc, td, 0, step='mid', color='gray', alpha=0.7)
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title(t0, fontsize=10, fontweight='bold')
        ax.set_ylim(0, td.max() * 1.1)
        _setup(ax)

        # Col 1: Star count
        ax = axes[row_idx, 1]
        ax.fill_between(bc, sc, 0, step='mid', color=star_color, alpha=0.7)
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title(t1, fontsize=10, fontweight='bold')
        ax.set_ylim(0, sc.max() * 1.1)
        _setup(ax)

        # Col 2: Star fraction
        ax = axes[row_idx, 2]
        mask = np.isfinite(sf)
        ax.fill_between(bc[mask], sf[mask], 0, color=star_color, alpha=0.4)
        ax.plot(bc[mask], sf[mask], color=star_color, linewidth=1.5)
        ax.set_ylabel('Star Fraction', fontsize=11, fontweight='bold')
        ax.set_title(t2, fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1.05)
        _setup(ax)

        # Col 3: Galaxy count
        ax = axes[row_idx, 3]
        if hl:
            ax.axvspan(highlight_min, highlight_max, alpha=0.5, color='#FFCDD2')
        ax.fill_between(bc, gc, 0, step='mid', color=galaxy_color, alpha=0.7)
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title(t3, fontsize=10, fontweight='bold')
        ax.set_ylim(0, gc.max() * 1.1)
        _setup(ax)

        # Col 4: Galaxy fraction
        ax = axes[row_idx, 4]
        if hl:
            ax.axvspan(highlight_min, highlight_max, alpha=0.5, color='#FFCDD2')
        mask = np.isfinite(gf)
        ax.fill_between(bc[mask], gf[mask], 0, color=galaxy_color, alpha=0.4)
        ax.plot(bc[mask], gf[mask], color=galaxy_color, linewidth=1.5)
        ax.set_ylabel('Galaxy Fraction', fontsize=11, fontweight='bold')
        ax.set_title(t4, fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1.05)
        _setup(ax)

    plt.tight_layout()
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

    # Query Gaia-matched sources (Gaia G magnitude), only G >= 18.5
    gaia_mag_cut_filter = 18.5

    print(f"\n--- Querying GAIA-MATCHED sources (Gaia G mag, G >= {gaia_mag_cut_filter}) ---")
    gaia_gaia_faint_data = query_mag_dist_by_class(
        username, gaia_matched_only=True, use_gaia_mag=True, gaia_mag_cut_filter=gaia_mag_cut_filter
    )
    if gaia_gaia_faint_data is None:
        print("Failed to query Gaia-matched faint data from WSDB")
        return

    create_visualization(all_data, gaia_euclid_data, gaia_gaia_data, gaia_gaia_faint_data,
                         gaia_mag_cut_filter=gaia_mag_cut_filter)

    print("\nDone!")


if __name__ == '__main__':
    main()
