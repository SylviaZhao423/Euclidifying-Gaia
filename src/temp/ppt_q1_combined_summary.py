#!/usr/bin/env python3
"""
Q1 combined summary — 3 separate single-column figures for slides.

Each figure is one column of the original combined_summary.png from
analyze_q1_mag_dist_fields.py:
  q1_combined_summary_col1.png — All Q1 I_E (count + fraction)
  q1_combined_summary_col2.png — Gaia-matched, I_E (count + fraction)
  q1_combined_summary_col3.png — Gaia-matched, Gaia G (count + fraction)

All 3 PNGs are identical in size and subplot geometry; only the
distributions change between slides.

Usage:
    PGUSER=siwei_zhao_2025 python src/temp/ppt_q1_combined_summary.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'docs' / 'slides' / 'figs'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

ZPC_IE = 1.05
ZP_REF = 23.9

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Fixed plot constants (shared across all 3 figures) ───────────────────────

BINS = np.arange(10, 29.25, 0.25)
XLIM = (10, 29)

# Subplot geometry — explicit so bbox_inches='tight' never shifts the axes
FIG_W, FIG_H  = 7, 8
TOP_MARGIN     = 0.92
BOT_MARGIN     = 0.07
LEFT_MARGIN    = 0.14
RIGHT_MARGIN   = 0.96
HSPACE         = 0.32
HEIGHT_RATIOS  = [2.5, 1.5]


# ─── Data queries ─────────────────────────────────────────────────────────────

def query_gaia_matched(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = f"""
        SELECT
            -2.5 * LOG10({ZPC_IE} * m_flux_detection_total) + {ZP_REF} AS ie_mag,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            CASE
                WHEN p_phz_classification = 1 THEN 'STAR'
                WHEN p_phz_classification = 2 THEN 'GALAXY'
                ELSE 'UNCLASSIFIED'
            END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE m_flux_detection_total > 0
          AND m_flux_detection_total IS NOT NULL
          AND m_vis_det = 1
    """
    print("  Querying Gaia-matched Q1 data...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} Gaia-matched sources")
    return df


def query_all_q1(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = f"""
        SELECT
            -2.5 * LOG10({ZPC_IE} * m_flux_detection_total) + {ZP_REF} AS ie_mag,
            CASE
                WHEN p_phz_classification = 1 THEN 'STAR'
                WHEN p_phz_classification = 2 THEN 'GALAXY'
                ELSE 'UNCLASSIFIED'
            END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz
        WHERE m_flux_detection_total > 0
          AND m_flux_detection_total IS NOT NULL
          AND m_vis_det = 1
    """
    print("  Querying all Q1 data...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} Q1 sources")
    return df


# ─── Helper ───────────────────────────────────────────────────────────────────

def _peak_count(df, mag_col):
    """Return peak bin count for the 'All' histogram."""
    counts, _ = np.histogram(df[mag_col].dropna(), bins=BINS)
    return counts.max()


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_single_column(df, mag_col, xlabel, title_suffix, filename, count_ylim):
    """2-row single-column figure: count histogram (top) + class fraction (bottom).

    count_ylim is a fixed shared y-limit for the count panel so all 3
    slide figures are visually comparable.
    """
    mag_all  = df[mag_col].dropna()
    mag_star = df[df['label'] == 'STAR'][mag_col].dropna()
    mag_gal  = df[df['label'] == 'GALAXY'][mag_col].dropna()
    mag_unc  = df[df['label'] == 'UNCLASSIFIED'][mag_col].dropna()

    n_gal  = int(np.histogram(mag_gal,  bins=BINS)[0].sum())
    n_star = int(np.histogram(mag_star, bins=BINS)[0].sum())
    n_unc  = int(np.histogram(mag_unc,  bins=BINS)[0].sum())
    n_all  = n_star + n_gal + n_unc

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(FIG_W, FIG_H),
        gridspec_kw={'height_ratios': HEIGHT_RATIOS},
    )
    fig.subplots_adjust(
        left=LEFT_MARGIN, right=RIGHT_MARGIN,
        bottom=BOT_MARGIN, top=TOP_MARGIN,
        hspace=HSPACE,
    )

    # ── Top: count histogram ──────────────────────────────────────────────────
    ax_top.hist(mag_all,  bins=BINS, alpha=0.3, color='gray',
                label=f'All ({n_all:,})')
    ax_top.hist(mag_gal,  bins=BINS, alpha=0.7, color='steelblue',
                label=f'Galaxies ({n_gal:,})', histtype='step', linewidth=1.5)
    ax_top.hist(mag_star, bins=BINS, alpha=0.7, color='coral',
                label=f'Stars ({n_star:,})', histtype='step', linewidth=1.5)
    ax_top.hist(mag_unc,  bins=BINS, alpha=0.7, color='forestgreen',
                label=f'Unclassified ({n_unc:,})', histtype='step', linewidth=1.5)
    ax_top.set_ylabel('Count', fontsize=11)
    ax_top.set_xlim(XLIM)
    ax_top.set_ylim(0, count_ylim)
    ax_top.set_title(f'Sources Count ({title_suffix})', fontsize=12, fontweight='bold')
    ax_top.legend(fontsize=9, loc='upper left')
    ax_top.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax_top.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax_top.grid(alpha=0.3, axis='y')
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # ── Bottom: class fraction ────────────────────────────────────────────────
    bin_centers  = (BINS[:-1] + BINS[1:]) / 2
    counts_all,  _ = np.histogram(mag_all,  bins=BINS)
    counts_gal,  _ = np.histogram(mag_gal,  bins=BINS)
    counts_star, _ = np.histogram(mag_star, bins=BINS)
    counts_unc,  _ = np.histogram(mag_unc,  bins=BINS)
    with np.errstate(divide='ignore', invalid='ignore'):
        frac_gal  = np.where(counts_all > 0, counts_gal  / counts_all, np.nan)
        frac_star = np.where(counts_all > 0, counts_star / counts_all, np.nan)
        frac_unc  = np.where(counts_all > 0, counts_unc  / counts_all, np.nan)
    mask = counts_all >= 10
    ax_bot.plot(bin_centers[mask], frac_star[mask], color='coral',
                linewidth=1.5, alpha=0.3)
    ax_bot.plot(bin_centers[mask], frac_unc[mask],  color='forestgreen',
                linewidth=1.5, alpha=0.3)
    ax_bot.plot(bin_centers[mask], frac_gal[mask],  'o-', color='steelblue',
                markersize=3, linewidth=1.5)
    ax_bot.set_xlabel(xlabel, fontsize=11)
    ax_bot.set_ylabel('Class fraction', fontsize=11)
    ax_bot.set_xlim(XLIM)
    ax_bot.set_ylim(-0.02, 1.02)
    ax_bot.set_title(f'Class Fraction ({title_suffix})', fontsize=12, fontweight='bold')
    ax_bot.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
    ax_bot.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax_bot.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax_bot.grid(alpha=0.3)

    out = OUTPUT_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    print("=" * 60)
    print("Q1 Combined Summary — 3 separate slide figures")
    print("=" * 60)

    print("\n[1] Querying data...")
    df_gaia = query_gaia_matched(username)
    df_q1   = query_all_q1(username)

    # Pre-compute a shared y-limit for the count panel (10% headroom, round to 1000)
    peak = max(
        _peak_count(df_q1,   'ie_mag'),
        _peak_count(df_gaia, 'ie_mag'),
        _peak_count(df_gaia, 'phot_g_mean_mag'),
    )
    count_ylim = int(np.ceil(peak * 1.10 / 1000) * 1000)
    print(f"  Shared count y-limit: {count_ylim:,}")

    print("\n[2] Generating figures...")
    plot_single_column(
        df_q1,   'ie_mag',          'Euclid I_E',
        'All Q1, I_E',
        'q1_combined_summary_col1.png',
        count_ylim,
    )
    plot_single_column(
        df_gaia, 'ie_mag',          'Euclid I_E',
        'Gaia-matched, I_E',
        'q1_combined_summary_col2.png',
        count_ylim,
    )
    plot_single_column(
        df_gaia, 'phot_g_mean_mag', 'Gaia G',
        'Gaia-matched, Gaia G',
        'q1_combined_summary_col3.png',
        count_ylim,
    )

    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
