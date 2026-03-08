#!/usr/bin/env python3
"""
ERO morphological classification distribution vs VIS magnitude — produces:
  ero_mag_dist2.png  — 2×6 grid:
    Top row:    source count histograms (gray=all, coral=extended/galaxy, steelblue=point-like/star)
    Bottom row: galaxy fraction vs magnitude with binomial error bars
    Col 1: CLASS_STAR (>= 0.65)
    Col 2: SPREAD_MODEL SExtractor (SM <= sqrt(ε² + (4×SME)²), ε=0.005)
    Col 3: SPREAD_MODEL simple fixed cut (SM > 0.003)
    Col 4: SPREAD_MODEL DES (SM + 3×SME > 0.003)
    Col 5: SPREAD_MODEL MODEST optimal (SM + 5/3×SME > 0.005)
    Col 6: Q1 PHZ (p_phz_classification: galaxy=2, star=1, unclassified=other)

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_ero_mag_dist2.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'output'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

MAG_MIN   = 10.0
MAG_MAX   = 29.0
BIN_WIDTH = 0.25

COLOR_ALL      = 'gray'
COLOR_EXTENDED = 'steelblue'
COLOR_POINT    = 'coral'
COLOR_UNCLASS  = 'forestgreen'

ZPC_IE = 1.05
ZP_REF = 23.9


# ─── Query ───────────────────────────────────────────────────────────────────

def query_data(username):
    """Query per-bin counts for all classifiers from euclid_ero.merge_cat."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()

    query = f"""
        SELECT
            FLOOR((v_mag_auto - {MAG_MIN}) / {BIN_WIDTH}) AS bin_idx,
            CASE WHEN v_class_star IS NULL OR v_class_star < 0.65 THEN 'CS_EXT' ELSE 'CS_POINT' END AS cs_class,
            CASE
                WHEN v_spread_model <= SQRT(0.005*0.005 + (4.0*v_spreaderr_model)*(4.0*v_spreaderr_model))
                THEN 'SP_POINT' ELSE 'SP_EXT'
            END AS sp_class,
            CASE
                WHEN v_spread_model + 3.0 * v_spreaderr_model <= 0.003
                THEN 'DES_POINT' ELSE 'DES_EXT'
            END AS des_class,
            CASE
                WHEN v_spread_model <= 0.003
                THEN 'SIMPLE_POINT' ELSE 'SIMPLE_EXT'
            END AS simple_class,
            CASE
                WHEN v_spread_model + (5.0/3.0) * v_spreaderr_model <= 0.005
                THEN 'MODEST_POINT' ELSE 'MODEST_EXT'
            END AS modest_class,
            COUNT(*) AS cnt
        FROM euclid_ero.merge_cat
        WHERE v_mag_auto        IS NOT NULL
          AND v_spread_model    IS NOT NULL
          AND v_spreaderr_model IS NOT NULL
        GROUP BY bin_idx, cs_class, sp_class, des_class, simple_class, modest_class
        ORDER BY bin_idx
    """

    print("Querying ERO classification data...")
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    print(f"  Retrieved {len(rows)} rows")

    n_bins = int((MAG_MAX - MAG_MIN) / BIN_WIDTH)
    bin_centers = MAG_MIN + (np.arange(n_bins) + 0.5) * BIN_WIDTH
    bin_edges   = MAG_MIN + np.arange(n_bins + 1) * BIN_WIDTH

    cs_point     = np.zeros(n_bins)
    cs_ext       = np.zeros(n_bins)
    sp_point     = np.zeros(n_bins)
    sp_ext       = np.zeros(n_bins)
    des_point    = np.zeros(n_bins)
    des_ext      = np.zeros(n_bins)
    simple_point = np.zeros(n_bins)
    simple_ext   = np.zeros(n_bins)
    modest_point = np.zeros(n_bins)
    modest_ext   = np.zeros(n_bins)

    for bin_idx, cs_class, sp_class, des_class, simple_class, modest_class, cnt in rows:
        i = int(bin_idx)
        if not (0 <= i < n_bins):
            continue
        if cs_class == 'CS_POINT':
            cs_point[i] += cnt
        else:
            cs_ext[i] += cnt
        if sp_class == 'SP_POINT':
            sp_point[i] += cnt
        else:
            sp_ext[i] += cnt
        if des_class == 'DES_POINT':
            des_point[i] += cnt
        else:
            des_ext[i] += cnt
        if simple_class == 'SIMPLE_POINT':
            simple_point[i] += cnt
        else:
            simple_ext[i] += cnt
        if modest_class == 'MODEST_POINT':
            modest_point[i] += cnt
        else:
            modest_ext[i] += cnt

    return (bin_centers, bin_edges,
            cs_point, cs_ext,
            sp_point, sp_ext,
            des_point, des_ext,
            simple_point, simple_ext,
            modest_point, modest_ext)


# ─── Q1 query ────────────────────────────────────────────────────────────────

def query_q1_data(username):
    """Query per-bin PHZ counts (GALAXY/STAR/UNCLASSIFIED) from crossmatch_q1_mer_phz."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()

    query = f"""
        SELECT
            FLOOR((-2.5 * LOG10({ZPC_IE} * m_flux_detection_total) + {ZP_REF} - {MAG_MIN})
                  / {BIN_WIDTH}) AS bin_idx,
            CASE
                WHEN p_phz_classification = 2 THEN 'GALAXY'
                WHEN p_phz_classification = 1 THEN 'STAR'
                ELSE 'UNCLASSIFIED'
            END AS phz_class,
            COUNT(*) AS cnt
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz
        WHERE m_flux_detection_total > 0
          AND m_flux_detection_total IS NOT NULL
          AND m_vis_det = 1
        GROUP BY bin_idx, phz_class
        ORDER BY bin_idx
    """

    print("Querying Q1 PHZ classification data...")
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    print(f"  Retrieved {len(rows)} rows")

    n_bins = int((MAG_MAX - MAG_MIN) / BIN_WIDTH)
    q1_galaxy = np.zeros(n_bins)
    q1_star   = np.zeros(n_bins)
    q1_unc    = np.zeros(n_bins)

    for bin_idx, phz_class, cnt in rows:
        try:
            i = int(bin_idx)
        except (ValueError, TypeError):
            continue
        if not (0 <= i < n_bins):
            continue
        if phz_class == 'GALAXY':
            q1_galaxy[i] += cnt
        elif phz_class == 'STAR':
            q1_star[i] += cnt
        else:
            q1_unc[i] += cnt

    return q1_galaxy, q1_star, q1_unc


# ─── Helpers ─────────────────────────────────────────────────────────────────

def binomial_errors(successes, total):
    """Return (fraction, lower_err, upper_err) with binomial 1-sigma errors."""
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(total > 0, successes / total, np.nan)
    err = np.sqrt(np.where(total > 0, p * (1 - p) / total, np.nan))
    return p, err


def count_panel(ax, bin_edges, all_counts, point_counts, ext_counts, xlabel, title):
    """Top-row panel: count histogram (gray fill + steelblue/coral steps)."""
    total = int(all_counts.sum())
    n_pt  = int(point_counts.sum())
    n_ext = int(ext_counts.sum())

    # Gray filled bars for all sources
    ax.bar(bin_edges[:-1], all_counts, width=BIN_WIDTH, align='edge',
           color=COLOR_ALL, alpha=0.3, label=f'All ({total:,})')
    # Step histograms for each class
    ax.stairs(point_counts, bin_edges, color=COLOR_POINT, linewidth=1.5,
              label=f'Point-like ({n_pt:,})')
    ax.stairs(ext_counts,   bin_edges, color=COLOR_EXTENDED, linewidth=1.5,
              label=f'Extended ({n_ext:,})')

    ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
    ax.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(MAG_MIN, MAG_MAX)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3, axis='y')


def q1_count_panel(ax, bin_edges, galaxy, star, unc, xlabel, title):
    """Top-row panel for Q1 PHZ: All (gray fill) + Galaxies + Stars + Unclassified steps."""
    total = galaxy + star + unc
    ax.bar(bin_edges[:-1], total, width=BIN_WIDTH, align='edge',
           color=COLOR_ALL, alpha=0.3, label=f'All ({int(total.sum()):,})')
    ax.stairs(galaxy, bin_edges, color=COLOR_EXTENDED,  linewidth=1.5,
              label=f'Galaxies ({int(galaxy.sum()):,})')
    ax.stairs(star,   bin_edges, color=COLOR_POINT,     linewidth=1.5,
              label=f'Stars ({int(star.sum()):,})')
    ax.stairs(unc,    bin_edges, color=COLOR_UNCLASS,   linewidth=1.5,
              label=f'Unclassified ({int(unc.sum()):,})')
    ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
    ax.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=18)
    ax.set_xlim(MAG_MIN, MAG_MAX)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3, axis='y')


def fraction_panel(ax, bin_centers, point_frac, point_err, ext_frac, ext_err, xlabel, title):
    """Bottom-row panel: fraction vs magnitude with error bars."""
    mask = ~np.isnan(point_frac)

    ax.errorbar(bin_centers[mask], point_frac[mask] * 100,
                yerr=point_err[mask] * 100,
                fmt='o-', color=COLOR_POINT, linewidth=1.5, markersize=3,
                capsize=3, capthick=1, alpha=0.8, label='Point-like')
    ax.errorbar(bin_centers[mask], ext_frac[mask] * 100,
                yerr=ext_err[mask] * 100,
                fmt='s-', color=COLOR_EXTENDED, linewidth=1.5, markersize=3,
                capsize=3, capthick=1, alpha=0.8, label='Extended')

    ax.axhline(y=50, color='gray', ls='--', linewidth=1, alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
    ax.set_ylabel('Fraction (%)', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(MAG_MIN, MAG_MAX)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ERO Classification Distribution vs VIS Magnitude")
    print("=" * 70)

    (bin_centers, bin_edges,
     cs_point, cs_ext,
     sp_point, sp_ext,
     des_point, des_ext,
     simple_point, simple_ext,
     modest_point, modest_ext) = query_data(username)

    q1_galaxy, q1_star, q1_unc = query_q1_data(username)
    q1_total = q1_galaxy + q1_star + q1_unc

    cs_total     = cs_point     + cs_ext
    sp_total     = sp_point     + sp_ext
    des_total    = des_point    + des_ext
    simple_total = simple_point + simple_ext
    modest_total = modest_point + modest_ext

    cs_pt_frac,     cs_pt_err     = binomial_errors(cs_point,     cs_total)
    cs_ex_frac,     cs_ex_err     = binomial_errors(cs_ext,       cs_total)
    sp_pt_frac,     sp_pt_err     = binomial_errors(sp_point,     sp_total)
    sp_ex_frac,     sp_ex_err     = binomial_errors(sp_ext,       sp_total)
    des_pt_frac,    des_pt_err    = binomial_errors(des_point,    des_total)
    des_ex_frac,    des_ex_err    = binomial_errors(des_ext,      des_total)
    simple_pt_frac, simple_pt_err = binomial_errors(simple_point, simple_total)
    simple_ex_frac, simple_ex_err = binomial_errors(simple_ext,   simple_total)
    modest_pt_frac, modest_pt_err = binomial_errors(modest_point, modest_total)
    modest_ex_frac, modest_ex_err = binomial_errors(modest_ext,   modest_total)

    q1_gal_frac,  q1_gal_err  = binomial_errors(q1_galaxy, q1_total)
    q1_star_frac, q1_star_err = binomial_errors(q1_star,   q1_total)

    print("\nGenerating plot...")

    fig = plt.figure(figsize=(48, 8))
    fig.suptitle('ERO Classification Distribution vs Euclid VIS Magnitude  |  Q1 PHZ',
                 fontsize=13, fontweight='bold', y=0.98)
    gs = GridSpec(2, 6, figure=fig, hspace=0.38, wspace=0.28)

    xlabel_ero = 'Euclid VIS (v_mag_auto)'
    xlabel_q1  = 'Q1 I_E'

    # Top row: count histograms  (order: CLASS_STAR, SExtractor, Simple, DES, MODEST, Q1 PHZ)
    count_panel(fig.add_subplot(gs[0, 0]),
                bin_edges, cs_total, cs_point, cs_ext,
                xlabel_ero, 'Source Count — CLASS_STAR')
    count_panel(fig.add_subplot(gs[0, 1]),
                bin_edges, sp_total, sp_point, sp_ext,
                xlabel_ero, 'Source Count — SPREAD_MODEL (SExtractor, ε=0.005, κ=4)')
    count_panel(fig.add_subplot(gs[0, 2]),
                bin_edges, simple_total, simple_point, simple_ext,
                xlabel_ero, 'Source Count — SPREAD_MODEL (simple, SM>0.003)')
    count_panel(fig.add_subplot(gs[0, 3]),
                bin_edges, des_total, des_point, des_ext,
                xlabel_ero, 'Source Count — SPREAD_MODEL (DES, SM+3×SME)')
    count_panel(fig.add_subplot(gs[0, 4]),
                bin_edges, modest_total, modest_point, modest_ext,
                xlabel_ero, 'Source Count — SPREAD_MODEL (MODEST, SM+5/3×SME)')
    q1_count_panel(fig.add_subplot(gs[0, 5]),
                   bin_edges, q1_galaxy, q1_star, q1_unc,
                   xlabel_q1, 'Source Count — Q1 PHZ (p_phz_classification)')

    # Bottom row: fraction plots (titles show the galaxy condition)
    fraction_panel(fig.add_subplot(gs[1, 0]),
                   bin_centers, cs_pt_frac, cs_pt_err, cs_ex_frac, cs_ex_err,
                   xlabel_ero, 'Galaxy fraction — CLASS_STAR < 0.65')
    fraction_panel(fig.add_subplot(gs[1, 1]),
                   bin_centers, sp_pt_frac, sp_pt_err, sp_ex_frac, sp_ex_err,
                   xlabel_ero, 'Galaxy fraction — SExtractor (SM > √(ε²+(κ×SME)²), ε=0.005, κ=4)')
    fraction_panel(fig.add_subplot(gs[1, 2]),
                   bin_centers, simple_pt_frac, simple_pt_err, simple_ex_frac, simple_ex_err,
                   xlabel_ero, 'Galaxy fraction — Simple fixed cut (SM > 0.003)')
    fraction_panel(fig.add_subplot(gs[1, 3]),
                   bin_centers, des_pt_frac, des_pt_err, des_ex_frac, des_ex_err,
                   xlabel_ero, 'Galaxy fraction — DES (SM + 3×SME > 0.003)')
    fraction_panel(fig.add_subplot(gs[1, 4]),
                   bin_centers, modest_pt_frac, modest_pt_err, modest_ex_frac, modest_ex_err,
                   xlabel_ero, 'Galaxy fraction — MODEST optimal (SM + 5/3×SME > 0.005)')
    fraction_panel(fig.add_subplot(gs[1, 5]),
                   bin_centers, q1_star_frac, q1_star_err, q1_gal_frac, q1_gal_err,
                   xlabel_q1, 'Galaxy fraction — Q1 PHZ (p_phz_classification = 2)')

    out = OUTPUT_DIR / 'ero_mag_dist_multi.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
