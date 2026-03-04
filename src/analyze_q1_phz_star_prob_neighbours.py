#!/usr/bin/env python3
"""
phz_star_prob vs magnitude 2D histograms split by Gaia neighbour count
at two radii (cnt4 = 4", cnt0_25 = 0.25").

Produces:
  1. phz_star_prob_neighbours_2d.png — 2x4 grid (rows = radii, cols = count bins):
     Row 1 (cnt4):    =1 (isolated), =2 (one neighbour), 3–4 (moderate), >=5 (crowded)
     Row 2 (cnt0_25): =1 (isolated), =2 (one neighbour), 3–4 (moderate), >=5 (crowded)
  2. phz_star_prob_vs_cnt.png — phz_star_prob vs neighbour count, one panel per radius

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_phz_star_prob_neighbours.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

ZPC_IE = 1.05
ZP_REF = 23.9

RADII = [
    ('cnt4',  '4"'),
    ('cnt8',  '8"'),
    ('cnt16', '16"'),
    ('cnt32', '32"'),
]


# --- Data query ---------------------------------------------------------------

def query_data(username):
    """Query Q1 PHZ Gaia-matched sources joined with Gaia neighbour counts."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = f"""
        SELECT
            -2.5 * LOG10({ZPC_IE} * c.m_flux_detection_total) + {ZP_REF} AS ie_mag,
            c.g_phot_g_mean_mag AS gaia_g,
            c.p_phz_star_prob AS star_prob,
            n.cnt4, n.cnt8, n.cnt16, n.cnt32
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia c
        JOIN gaia_edr3_aux.gaia_source_neighbour_count n
            ON c.g_source_id = n.source_id
        WHERE c.m_flux_detection_total > 0
          AND c.m_flux_detection_total IS NOT NULL
          AND c.m_vis_det = 1
          AND c.g_phot_g_mean_mag IS NOT NULL
          AND c.p_phz_star_prob IS NOT NULL
    """
    print("  Querying Q1 PHZ + Gaia + neighbour counts...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} sources")
    return df


# --- Plots --------------------------------------------------------------------

def plot_star_prob_by_neighbours(df, output_dir):
    """3×4 grid: rows = radii (small→large), cols = count bins.
    Column headers on top row, horizontal row labels on left, N inside panel."""
    sub = df[df['gaia_g'].notna() & df['star_prob'].notna()]

    xbins = np.arange(14, 22.05, 0.05)
    ybins = np.arange(0, 1.01, 0.01)

    count_bins = [
        (lambda s, c: s[c] == 1,            '= 1 (isolated)'),
        (lambda s, c: s[c] == 2,            '= 2'),
        (lambda s, c: s[c].between(3, 4),   '= 3–4'),
        (lambda s, c: s[c] >= 5,            '≥ 5 (crowded)'),
    ]

    n_rows = len(RADII)
    n_cols = len(count_bins)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 6 * n_rows))

    for row, (cnt_col, radius_label) in enumerate(RADII):
        s = sub[sub[cnt_col].notna()]

        for col, (mask_fn, bin_label) in enumerate(count_bins):
            ax = axes[row, col]
            panel_df = s[mask_fn(s, cnt_col)]
            n = len(panel_df)

            h, xedges, yedges = np.histogram2d(
                panel_df['gaia_g'].values, panel_df['star_prob'].values,
                bins=[xbins, ybins])
            h_masked = np.ma.masked_where(h == 0, h)
            vmax = h_masked.max() if h_masked.count() > 0 else 10
            im = ax.pcolormesh(xedges, yedges, h_masked.T,
                               cmap='Blues', norm=LogNorm(vmin=1, vmax=max(2, vmax)))
            ax.axhline(y=0.5, color='red', ls='--', linewidth=1.5, label='prob=0.5')
            fig.colorbar(im, ax=ax, pad=0.02).set_label('Count', fontsize=10)

            if row == 0:
                ax.set_title(f'cnt {bin_label}', fontsize=12, fontweight='bold')
            else:
                ax.set_title('')
            if col == 0:
                ax.set_ylabel('phz_star_prob', fontsize=10)
                ax.annotate(f'r = {radius_label}', xy=(0, 0.5),
                            xycoords='axes fraction', xytext=(-0.17, 0.5),
                            textcoords='axes fraction', fontsize=12,
                            fontweight='bold', ha='center', va='center',
                            rotation=0)
            else:
                ax.set_ylabel('phz_star_prob', fontsize=10)
            ax.text(0.03, 0.97, f'N={n:,}', transform=ax.transAxes,
                    fontsize=9, va='top', ha='left')
            ax.set_xlabel('Gaia G' if row == n_rows - 1 else '', fontsize=10)
            ax.set_xlim(14, 22)
            ax.set_ylim(0, 1)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(alpha=0.3)

    plt.suptitle('phz_star_prob vs Gaia G by neighbour count',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.subplots_adjust(left=0.08)
    output_path = output_dir / 'phz_star_prob_neighbours_2d.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_star_prob_vs_cnt(df, output_dir):
    """1x5 row of phz_star_prob vs neighbour count at each radius."""
    base = df[df['star_prob'].notna()]
    ybins = np.arange(0, 1.01, 0.01)

    fig, axes = plt.subplots(1, len(RADII), figsize=(6 * len(RADII), 6))

    for ax, (cnt_col, radius_label) in zip(axes, RADII):
        sub = base[base[cnt_col].notna()]
        cnt_max = int(sub[cnt_col].quantile(0.99)) if len(sub) > 0 else 5
        cnt_max = max(cnt_max, 2)
        xbins = np.arange(0.5, cnt_max + 1.5, 1)

        h, xedges, yedges = np.histogram2d(
            sub[cnt_col].values, sub['star_prob'].values, bins=[xbins, ybins])
        h_masked = np.ma.masked_where(h == 0, h)
        vmax = h_masked.max() if h_masked.count() > 0 else 10
        im = ax.pcolormesh(xedges, yedges, h_masked.T,
                           cmap='Blues', norm=LogNorm(vmin=1, vmax=max(2, vmax)))
        ax.axhline(y=0.5, color='red', ls='--', linewidth=1.5, label='prob=0.5')
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Count', fontsize=10)
        ax.set_xlabel(f'{cnt_col} (neighbours within {radius_label})',
                      fontsize=10, fontweight='bold')
        ax.set_ylabel('phz_star_prob', fontsize=10)
        ax.set_title(f'r = {radius_label}  (N={len(sub):,})',
                     fontsize=12, fontweight='bold')
        ax.set_xlim(xbins[0], xbins[-1])
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'phz_star_prob_vs_cnt.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")




# --- Main ---------------------------------------------------------------------

def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("phz_star_prob vs Gaia G — split by neighbour count (multi-radius)")
    print("=" * 70)

    print("\n[1] Querying data...")
    df = query_data(username)

    print("\n[2] Generating phz_star_prob vs Gaia G by neighbour count...")
    plot_star_prob_by_neighbours(df, OUTPUT_DIR)

    print("\n[3] Generating phz_star_prob vs cnt (all radii)...")
    plot_star_prob_vs_cnt(df, OUTPUT_DIR)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
