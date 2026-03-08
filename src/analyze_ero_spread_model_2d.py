#!/usr/bin/env python3
"""
spread_model vs magnitude 2D histograms — produces:
  ero_spread_model_2d.png  — 3x2 combined plot:
    Row 1: spread_model 2D histograms
      (a) spread_model vs Euclid VIS mag (all ERO, from euclid_ero.merge_cat)
      (b) spread_model vs Gaia G (Gaia-matched, from crossmatch_ero_gaia_0_3)
    Row 2: 1D magnitude distributions classified by SM + 3×SME > 0
    Row 3: 1D magnitude distributions classified by SM + 5/3×SME > 0

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_ero_spread_model_2d.py
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

SM_MIN = -0.1
SM_MAX =  0.1
SM_STEP = 0.001   # 200 bins across the range


# ─── Data queries ────────────────────────────────────────────────────────────

def query_all_ero(username):
    """Query all ERO sources with VIS mag, spread_model, and spread_model_err."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = """
        SELECT
            v_mag_auto        AS vis_mag,
            v_spread_model    AS spread_model,
            v_spreaderr_model AS spread_model_err
        FROM euclid_ero.merge_cat
        WHERE v_mag_auto     IS NOT NULL
          AND v_spread_model IS NOT NULL
    """
    print("  Querying all ERO sources...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} sources")
    return df


def query_gaia_matched(username):
    """Query Gaia-matched ERO sources with VIS mag, Gaia G, spread_model, and spread_model_err."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = """
        SELECT
            e_v_mag_auto         AS vis_mag,
            g_phot_g_mean_mag    AS gaia_g,
            e_v_spread_model     AS spread_model,
            e_v_spreaderr_model  AS spread_model_err
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_mag_auto      IS NOT NULL
          AND g_phot_g_mean_mag IS NOT NULL
          AND e_v_spread_model  IS NOT NULL
    """
    print("  Querying Gaia-matched ERO sources...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} sources")
    return df


# ─── Plot ────────────────────────────────────────────────────────────────────

def plot_spread_model_combined(df_all, df_gaia, output_dir):
    """Combined 3x2 plot of 2D histograms: SM, SM+3×SME, SM+5/3×SME vs magnitude."""
    ybins = np.arange(SM_MIN, SM_MAX + SM_STEP, SM_STEP)

    # Precompute derived y-values (drop rows where spread_model_err is null)
    df_all_e  = df_all.dropna(subset=['spread_model', 'spread_model_err'])
    df_gaia_e = df_gaia.dropna(subset=['spread_model', 'spread_model_err'])

    rows = [
        # (df_vis,   y_vis,                         df_g,      y_g,                            ylabel,                                threshold)
        (df_all,    df_all['spread_model'].values,
         df_gaia,   df_gaia['spread_model'].values,
         'spread_model',                             0.0),
        (df_all_e,  df_all_e['spread_model'].values  + 3   * df_all_e['spread_model_err'].values,
         df_gaia_e, df_gaia_e['spread_model'].values + 3   * df_gaia_e['spread_model_err'].values,
         'spread_model + 3 * spreaderr_model',       0.003),
        (df_all_e,  df_all_e['spread_model'].values  + 5/3 * df_all_e['spread_model_err'].values,
         df_gaia_e, df_gaia_e['spread_model'].values + 5/3 * df_gaia_e['spread_model_err'].values,
         'spread_model + 5/3 * spreaderr_model',     0.005),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(24, 18))

    for row_idx, (df_v, yvals_vis, df_g, yvals_g, ylabel, threshold) in enumerate(rows):
        panels = [
            (df_v['vis_mag'].values, yvals_vis,
             np.arange(14, 28.05, 0.05), (14, 28),
             'Euclid VIS (v_mag_auto)',
             f'{ylabel} vs Euclid VIS (all ERO, n={len(df_v):,})',
             ylabel),
            (df_g['gaia_g'].values, yvals_g,
             np.arange(14, 22.05, 0.05), (14, 22),
             'Gaia G',
             f'{ylabel} vs Gaia G (Gaia-matched, n={len(df_g):,})',
             ylabel),
        ]
        for col_idx, (xvals, yvals, xbins, xlim, xlabel, title, yl) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            h, xedges, yedges = np.histogram2d(xvals, yvals, bins=[xbins, ybins])
            h_masked = np.ma.masked_where(h == 0, h)
            im = ax.pcolormesh(xedges, yedges, h_masked.T,
                               cmap='GnBu', norm=LogNorm(vmin=1))
            ax.axhline(y=threshold, color='red', ls='--', linewidth=1.5,
                       label=f'{yl} = {threshold}')
            cbar = fig.colorbar(im, ax=ax, pad=0.02)
            cbar.set_label('Count', fontsize=11, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
            ax.set_ylabel(yl, fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlim(*xlim)
            ax.set_ylim(SM_MIN, SM_MAX)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'ero_prob_spread_model_2d.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ERO spread_model vs Magnitude 2D Histograms")
    print("=" * 70)

    print("\n[1] Querying all ERO sources...")
    df_all = query_all_ero(username)

    print("\n[2] Querying Gaia-matched ERO sources...")
    df_gaia = query_gaia_matched(username)

    print("\n[3] Generating spread_model vs magnitude 2D histogram...")
    plot_spread_model_combined(df_all, df_gaia, OUTPUT_DIR)

    print(f"\nPlot saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
