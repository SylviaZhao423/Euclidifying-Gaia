#!/usr/bin/env python3
"""
Analyze Q1 magnitude distribution separated by PHZ classification.

Produces two output files:
  q1_mag_dist_by_class.png           — all VIS-detected sources
  q1_mag_dist_by_class_classified.png — classified-only (p_phz_classification IN 1,2,4)

Classification: p_phz_classification = 1 → STAR, = 2 → GALAXY
Magnitude: -2.5 × log10(1.05 × flux_detection_total) + 23.9 (IE-band with ZPC)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import os

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output' / 'q1_mag_dist'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

ZPC_IE = 1.05
ZP_REF = 23.9

OUTPUT_DIR.mkdir(exist_ok=True)


def query_mag_dist_by_class(username, gaia_matched_only=False, use_gaia_mag=False,
                            gaia_mag_cut_filter=None, euclid_mag_cut_filter=None,
                            classified_only=False):
    """Query Q1 magnitude distribution split by PHZ classification.

    Parameters
    ----------
    classified_only : bool
        If True, restrict to p_phz_classification IN (1, 2, 4) only.
    gaia_mag_cut_filter : float or None
        If set, only include sources with Gaia G mag >= this value.
    euclid_mag_cut_filter : float or None
        If set, only include sources with Q1 IE mag >= this value.
    """
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed!")
        return None

    print(f"\nConnecting to WSDB...")
    try:
        conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    except Exception as e:
        print(f"ERROR connecting to WSDB: {e}")
        return None

    BIN_WIDTH = 0.25
    mag_bins = np.arange(10, 29.25, BIN_WIDTH)

    class_filter = 'AND p_phz_classification IN (1, 2, 4)' if classified_only else ''

    if gaia_matched_only:
        table_name = 'siwei_zhao_2025.crossmatch_q1_mer_phz_gaia'
        flux_col = 'm_flux_detection_total'
        extra_filter = (f'AND g_phot_g_mean_mag IS NOT NULL'
                        f' AND p_phz_classification IS NOT NULL'
                        f' AND m_vis_det = 1'
                        f' {class_filter}')
        if use_gaia_mag:
            mag_expr = 'g_phot_g_mean_mag'
            mag_label = 'Gaia G mag'
            if gaia_mag_cut_filter is not None:
                extra_filter += f' AND g_phot_g_mean_mag >= {gaia_mag_cut_filter}'
                print(f"Querying GAIA-MATCHED (Gaia G mag, G >= {gaia_mag_cut_filter})...")
            elif euclid_mag_cut_filter is not None:
                ie_expr = f'-2.5 * LOG10({ZPC_IE} * {flux_col}) + {ZP_REF}'
                extra_filter += (f' AND {flux_col} > 0 AND {flux_col} IS NOT NULL'
                                 f' AND ({ie_expr}) >= {euclid_mag_cut_filter}')
                print(f"Querying GAIA-MATCHED (Gaia G mag, IE >= {euclid_mag_cut_filter})...")
            else:
                print(f"Querying GAIA-MATCHED (Gaia G mag)...")
        else:
            mag_expr = f'-2.5 * LOG10({ZPC_IE} * {flux_col}) + {ZP_REF}'
            mag_label = 'Q1 IE mag'
            extra_filter += f' AND {flux_col} > 0 AND {flux_col} IS NOT NULL'
            print(f"Querying GAIA-MATCHED (Q1 IE mag)...")
    else:
        table_name = 'siwei_zhao_2025.crossmatch_q1_mer_phz'
        flux_col = 'm_flux_detection_total'
        mag_expr = f'-2.5 * LOG10({ZPC_IE} * {flux_col}) + {ZP_REF}'
        mag_label = 'Q1 IE mag'
        extra_filter = (f'AND {flux_col} > 0 AND {flux_col} IS NOT NULL'
                        f' AND m_vis_det = 1'
                        f' {class_filter}')
        label = 'classified only' if classified_only else 'all'
        print(f"Querying ALL Q1+PHZ ({label}) from {table_name}...")

    cursor = conn.cursor()
    query = f"""
        SELECT
            FLOOR(({mag_expr} - {mag_bins[0]}) / {BIN_WIDTH}) as bin_index,
            {mag_bins[0]} + (FLOOR(({mag_expr} - {mag_bins[0]}) / {BIN_WIDTH}) * {BIN_WIDTH}) + {BIN_WIDTH/2} as bin_center,
            CASE
                WHEN p_phz_classification = 1 THEN 'STAR'
                WHEN p_phz_classification = 2 THEN 'GALAXY'
                ELSE 'OTHER'
            END as classification,
            COUNT(*) as count
        FROM {table_name}
        WHERE ({mag_expr}) >= {mag_bins[0]}
          AND ({mag_expr}) < {mag_bins[-1]}
          AND {mag_expr} IS NOT NULL
          {extra_filter}
        GROUP BY bin_index, bin_center, classification
        ORDER BY bin_index, classification
    """

    try:
        cursor.execute(query)
        results = cursor.fetchall()
        print(f"  {len(results)} classification rows returned")

        data_dict = {}
        for row in results:
            bin_idx, bin_center, classification, count = row
            data_dict[(int(bin_idx), classification)] = count

        bin_centers, star_counts, galaxy_counts, total_counts = [], [], [], []
        for i in range(len(mag_bins) - 1):
            bin_center = (mag_bins[i] + mag_bins[i + 1]) / 2
            n_star   = data_dict.get((i, 'STAR'),   0)
            n_galaxy = data_dict.get((i, 'GALAXY'), 0)
            n_other  = data_dict.get((i, 'OTHER'),  0)
            total = n_star + n_galaxy + n_other
            if total > 0:
                bin_centers.append(bin_center)
                star_counts.append(n_star)
                galaxy_counts.append(n_galaxy)
                total_counts.append(total)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback; traceback.print_exc()
        cursor.close(); conn.close()
        return None

    cursor.close()
    conn.close()
    return (np.array(bin_centers), np.array(star_counts),
            np.array(galaxy_counts), np.array(total_counts))


def _to_density(counts, bin_width):
    total = counts.sum()
    return counts / (total * bin_width) if total > 0 else counts * 0.0


def create_visualization(all_data, gaia_euclid_data, gaia_gaia_data, gaia_gaia_faint_data,
                         gaia_mag_cut_filter=18.5, classified_only=False):
    """Create 4×5-panel magnitude distribution plot."""

    bin_centers_all,    star_counts_all,    galaxy_counts_all,    total_counts_all    = all_data
    bin_centers_gaia_e, star_counts_gaia_e, galaxy_counts_gaia_e, total_counts_gaia_e = gaia_euclid_data
    bin_centers_gaia_g, star_counts_gaia_g, galaxy_counts_gaia_g, total_counts_gaia_g = gaia_gaia_data
    bin_centers_faint,  star_counts_faint,  galaxy_counts_faint,  total_counts_faint  = gaia_gaia_faint_data

    BIN_WIDTH = 0.25
    MIN_COUNT = 10

    total_dens_all    = _to_density(total_counts_all,    BIN_WIDTH)
    total_dens_gaia_e = _to_density(total_counts_gaia_e, BIN_WIDTH)
    total_dens_gaia_g = _to_density(total_counts_gaia_g, BIN_WIDTH)
    total_dens_faint  = _to_density(total_counts_faint,  BIN_WIDTH)

    def frac(counts, total):
        return np.where(total >= MIN_COUNT, counts / total, np.nan)

    star_frac_all    = frac(star_counts_all,    total_counts_all)
    star_frac_gaia_e = frac(star_counts_gaia_e, total_counts_gaia_e)
    star_frac_gaia_g = frac(star_counts_gaia_g, total_counts_gaia_g)
    star_frac_faint  = frac(star_counts_faint,  total_counts_faint)
    gal_frac_all     = frac(galaxy_counts_all,    total_counts_all)
    gal_frac_gaia_e  = frac(galaxy_counts_gaia_e, total_counts_gaia_e)
    gal_frac_gaia_g  = frac(galaxy_counts_gaia_g, total_counts_gaia_g)
    gal_frac_faint   = frac(galaxy_counts_faint,  total_counts_faint)

    n_total_all = int(total_counts_all.sum())
    n_star_all  = int(star_counts_all.sum())
    n_gal_all   = int(galaxy_counts_all.sum())

    n_total_ge = int(total_counts_gaia_e.sum())
    n_star_ge  = int(star_counts_gaia_e.sum())
    n_gal_ge   = int(galaxy_counts_gaia_e.sum())
    pct_ge     = 100 * n_total_ge / n_total_all if n_total_all > 0 else 0
    pct_star_ge = 100 * n_star_ge / n_star_all  if n_star_all  > 0 else 0
    pct_gal_ge  = 100 * n_gal_ge  / n_gal_all   if n_gal_all   > 0 else 0

    n_total_gg = int(total_counts_gaia_g.sum())
    n_star_gg  = int(star_counts_gaia_g.sum())
    n_gal_gg   = int(galaxy_counts_gaia_g.sum())
    pct_gg     = 100 * n_total_gg / n_total_all if n_total_all > 0 else 0
    pct_star_gg = 100 * n_star_gg / n_star_all  if n_star_all  > 0 else 0
    pct_gal_gg  = 100 * n_gal_gg  / n_gal_all   if n_gal_all   > 0 else 0

    n_total_f  = int(total_counts_faint.sum())
    n_star_f   = int(star_counts_faint.sum())
    n_gal_f    = int(galaxy_counts_faint.sum())
    pct_f      = 100 * n_total_f / n_total_gg if n_total_gg > 0 else 0
    pct_star_f  = 100 * n_star_f  / n_star_gg  if n_star_gg  > 0 else 0
    pct_gal_f   = 100 * n_gal_f   / n_gal_gg   if n_gal_gg   > 0 else 0

    ie_label   = r'Magnitude ($-2.5\log_{10}(1.05 \times \mathrm{flux}) + 23.9$)'
    gaia_label = 'Magnitude (phot_g_mean_mag)'
    phz_limit  = 18.5
    star_color = 'coral'
    gal_color  = 'steelblue'
    suffix     = ' (Classified Only)' if classified_only else ''

    rows = [
        (bin_centers_all,    total_dens_all,    star_frac_all,    star_counts_all,    gal_frac_all,    galaxy_counts_all,
         ie_label,
         f'All Q1+PHZ Sources{suffix} (n={n_total_all:,})',
         f'Star Count (n={n_star_all:,})',              f'Star Fraction (n={n_star_all:,})',
         f'Galaxy Count (n={n_gal_all:,})',             f'Galaxy Fraction (n={n_gal_all:,})',
         []),
        (bin_centers_gaia_e, total_dens_gaia_e, star_frac_gaia_e, star_counts_gaia_e, gal_frac_gaia_e, galaxy_counts_gaia_e,
         ie_label,
         f'Gaia-Matched in Q1 IE{suffix} (n={n_total_ge:,}, {pct_ge:.1f}%)',
         f'Star Count (n={n_star_ge:,}, {pct_star_ge:.1f}%)',   f'Star Fraction (n={n_star_ge:,}, {pct_star_ge:.1f}%)',
         f'Galaxy Count (n={n_gal_ge:,}, {pct_gal_ge:.1f}%)',   f'Galaxy Fraction (n={n_gal_ge:,}, {pct_gal_ge:.1f}%)',
         [phz_limit]),
        (bin_centers_gaia_g, total_dens_gaia_g, star_frac_gaia_g, star_counts_gaia_g, gal_frac_gaia_g, galaxy_counts_gaia_g,
         gaia_label,
         f'Gaia-Matched in Gaia G{suffix} (n={n_total_gg:,}, {pct_gg:.1f}%)',
         f'Star Count (n={n_star_gg:,}, {pct_star_gg:.1f}%)',   f'Star Fraction (n={n_star_gg:,}, {pct_star_gg:.1f}%)',
         f'Galaxy Count (n={n_gal_gg:,}, {pct_gal_gg:.1f}%)',   f'Galaxy Fraction (n={n_gal_gg:,}, {pct_gal_gg:.1f}%)',
         [phz_limit]),
        (bin_centers_faint,  total_dens_faint,  star_frac_faint,  star_counts_faint,  gal_frac_faint,  galaxy_counts_faint,
         gaia_label,
         f'Gaia G >= {gaia_mag_cut_filter}{suffix} (n={n_total_f:,}, {pct_f:.1f}%)',
         f'Star Count (n={n_star_f:,}, {pct_star_f:.1f}%)',     f'Star Fraction (n={n_star_f:,}, {pct_star_f:.1f}%)',
         f'Galaxy Count (n={n_gal_f:,}, {pct_gal_f:.1f}%)',     f'Galaxy Fraction (n={n_gal_f:,}, {pct_gal_f:.1f}%)',
         [phz_limit]),
    ]

    fig, axes = plt.subplots(4, 5, figsize=(30, 18))
    title = f'Q1 Magnitude Distribution by PHZ Classification{suffix}'
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.99)

    n_rows = len(rows)
    for row_idx, (bc, td, sf, sc, gf, gc, xlabel, t0, t1, t2, t3, t4, vlines) in enumerate(rows):
        def _setup(ax, xlabel=xlabel, vlines=vlines, row_idx=row_idx):
            ax.set_xlabel(xlabel, fontsize=9, fontweight='bold')
            ax.set_xlim(10, 29)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            for vl in vlines:
                ax.axvline(x=vl, color='#F27A71', ls='--', lw=1.5,
                           label=f'Training cut (G~{vl})')
            if vlines and row_idx == n_rows - 1:
                ax.legend(fontsize=8, loc='upper left')

        ax = axes[row_idx, 0]
        ax.fill_between(bc, td, 0, step='mid', color='gray', alpha=0.7)
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title(t0, fontsize=10, fontweight='bold')
        ax.set_ylim(0, td.max() * 1.1)
        _setup(ax)

        ax = axes[row_idx, 1]
        ax.fill_between(bc, sc, 0, step='mid', color=star_color, alpha=0.7)
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title(t1, fontsize=10, fontweight='bold')
        ax.set_ylim(0, sc.max() * 1.1)
        _setup(ax)

        ax = axes[row_idx, 2]
        mask = np.isfinite(sf)
        ax.fill_between(bc[mask], sf[mask], 0, color=star_color, alpha=0.4)
        ax.plot(bc[mask], sf[mask], color=star_color, linewidth=1.5)
        ax.set_ylabel('Star Fraction', fontsize=11, fontweight='bold')
        ax.set_title(t2, fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1.05)
        _setup(ax)

        ax = axes[row_idx, 3]
        ax.fill_between(bc, gc, 0, step='mid', color=gal_color, alpha=0.7)
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title(t3, fontsize=10, fontweight='bold')
        ax.set_ylim(0, gc.max() * 1.1)
        _setup(ax)

        ax = axes[row_idx, 4]
        mask = np.isfinite(gf)
        ax.fill_between(bc[mask], gf[mask], 0, color=gal_color, alpha=0.4)
        ax.plot(bc[mask], gf[mask], color=gal_color, linewidth=1.5)
        ax.set_ylabel('Galaxy Fraction', fontsize=11, fontweight='bold')
        ax.set_title(t4, fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1.05)
        _setup(ax)

    plt.tight_layout()
    fname = 'q1_mag_dist_by_class_classified.png' if classified_only else 'q1_mag_dist_by_class.png'
    out = OUTPUT_DIR / fname
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


def _run(username, classified_only, gaia_mag_cut_filter=18.5):
    label = 'classified only' if classified_only else 'all'
    print(f"\n{'='*70}")
    print(f"Querying: {label}")
    print(f"{'='*70}")

    all_data = query_mag_dist_by_class(username, classified_only=classified_only)
    if all_data is None:
        return False

    gaia_euclid_data = query_mag_dist_by_class(
        username, gaia_matched_only=True, use_gaia_mag=False,
        classified_only=classified_only)
    if gaia_euclid_data is None:
        return False

    gaia_gaia_data = query_mag_dist_by_class(
        username, gaia_matched_only=True, use_gaia_mag=True,
        classified_only=classified_only)
    if gaia_gaia_data is None:
        return False

    gaia_gaia_faint_data = query_mag_dist_by_class(
        username, gaia_matched_only=True, use_gaia_mag=True,
        gaia_mag_cut_filter=gaia_mag_cut_filter,
        classified_only=classified_only)
    if gaia_gaia_faint_data is None:
        return False

    create_visualization(all_data, gaia_euclid_data, gaia_gaia_data, gaia_gaia_faint_data,
                         gaia_mag_cut_filter=gaia_mag_cut_filter,
                         classified_only=classified_only)
    return True


def main():
    print("=" * 70)
    print("Q1 Magnitude Distribution by PHZ Classification")
    print("=" * 70)

    username = os.getenv('PGUSER')
    if not username:
        print("\nERROR: PGUSER environment variable not set!")
        return

    _run(username, classified_only=False)
    _run(username, classified_only=True)

    print("\nDone!")


if __name__ == '__main__':
    main()
