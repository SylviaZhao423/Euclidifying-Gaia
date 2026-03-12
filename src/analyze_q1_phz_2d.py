#!/usr/bin/env python3
"""
phz_star_prob and phz_gal_prob vs magnitude 2D histograms — produces:
  q1_prob_phz_star_2d.png  — 2x2 combined plot:
    Row 1: phz_star_prob vs I_E (all Q1+PHZ) | phz_star_prob vs Gaia G (Gaia-matched)
    Row 2: phz_gal_prob  vs I_E (all Q1+PHZ) | phz_gal_prob  vs Gaia G (Gaia-matched)

Extracted from analyze_q1_phz_neighbours.py.

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_q1_phz_star_2d.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'output'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

ZPC_IE = 1.05
ZP_REF = 23.9


# ─── Data queries ─────────────────────────────────────────────────────────────

def query_all_q1(username):
    """Query all Q1+PHZ sources (VIS-detected) with I_E, phz_star_prob, phz_gal_prob."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = f"""
        SELECT
            -2.5 * LOG10({ZPC_IE} * m_flux_detection_total) + {ZP_REF} AS ie_mag,
            p_phz_star_prob AS star_prob,
            p_phz_gal_prob  AS gal_prob
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz
        WHERE m_flux_detection_total > 0
          AND m_flux_detection_total IS NOT NULL
          AND m_vis_det = 1
          AND p_phz_star_prob IS NOT NULL
    """
    print("  Querying all Q1+PHZ sources (VIS-detected)...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} sources")
    return df


def query_gaia_matched(username):
    """Query Gaia-matched Q1+PHZ sources with I_E, Gaia G, phz_star_prob, phz_gal_prob."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = f"""
        SELECT
            -2.5 * LOG10({ZPC_IE} * m_flux_detection_total) + {ZP_REF} AS ie_mag,
            g_phot_g_mean_mag AS gaia_g,
            p_phz_star_prob AS star_prob,
            p_phz_gal_prob  AS gal_prob
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE m_flux_detection_total > 0
          AND m_flux_detection_total IS NOT NULL
          AND m_vis_det = 1
          AND g_phot_g_mean_mag IS NOT NULL
          AND p_phz_star_prob IS NOT NULL
    """
    print("  Querying Gaia-matched Q1+PHZ sources...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} sources")
    return df


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_phz_prob_combined(df_all, df_gaia, output_dir):
    """Combined 2x2 plot of phz_star_prob and phz_gal_prob vs magnitude 2D histograms."""
    ybins = np.arange(0, 1.01, 0.004)

    rows = [
        ('star_prob', 'phz_star_prob', 'prob=0.5', 0.5),
        ('gal_prob',  'phz_gal_prob',  'prob=0.5', 0.5),
    ]

    cols = [
        (df_all,  'ie_mag', np.arange(10, 28.05, 0.05), (10, 28), 'I_E',    f'all, n={len(df_all):,}'),
        (df_gaia, 'gaia_g', np.arange(10, 28.05, 0.05), (10, 28), 'Gaia G', f'Gaia-matched, n={len(df_gaia):,}'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(24, 12))

    for row_idx, (prob_col, ylabel, hline_label, hline_y) in enumerate(rows):
        for col_idx, (df, mag_col, xbins, xlim, xlabel, src_label) in enumerate(cols):
            ax = axes[row_idx, col_idx]
            df_v = df.dropna(subset=[mag_col, prob_col])
            xvals = df_v[mag_col].values
            yvals = df_v[prob_col].values

            h, xedges, yedges = np.histogram2d(xvals, yvals, bins=[xbins, ybins])
            h_masked = np.ma.masked_where(h == 0, h)
            im = ax.pcolormesh(xedges, yedges, h_masked.T,
                               cmap='GnBu', norm=LogNorm(vmin=1))
            ax.axhline(y=hline_y, color='red', ls='--', linewidth=1.5,
                       label=hline_label)
            cbar = fig.colorbar(im, ax=ax, pad=0.02)
            cbar.set_label('Count', fontsize=11, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
            ax.set_title(f'{ylabel} vs {xlabel} ({src_label})',
                         fontsize=12, fontweight='bold')
            ax.set_xlim(*xlim)
            ax.set_ylim(0, 1)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'q1_prob_phz_2d.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Q1 phz_star_prob / phz_gal_prob vs Magnitude 2D Histograms")
    print("=" * 70)

    print("\n[1] Querying all Q1+PHZ sources...")
    df_all = query_all_q1(username)

    print("\n[2] Querying Gaia-matched Q1+PHZ sources...")
    df_gaia = query_gaia_matched(username)

    print("\n[3] Generating phz_star_prob / phz_gal_prob vs magnitude 2D histograms...")
    plot_phz_prob_combined(df_all, df_gaia, OUTPUT_DIR)

    print(f"\nPlot saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
