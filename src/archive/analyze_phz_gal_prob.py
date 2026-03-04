#!/usr/bin/env python3
"""
phz_gal_prob vs magnitude 2D histograms — produces:
  1. phz_gal_prob_2d.png — 1x3 combined plot:
     (a) phz_gal_prob vs Q1 IE mag (all Q1+PHZ, VIS-detected)
     (b) phz_gal_prob vs Q1 IE mag (Gaia-matched)
     (c) phz_gal_prob vs Gaia G (Gaia-matched)

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_phz_gal_prob.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

# Zero-Point Multiplicative Correction for IE-band
ZPC_IE = 1.05
ZP_REF = 23.9


# ─── Data queries ────────────────────────────────────────────────────────────

def query_all_q1(username):
    """Query all Q1+PHZ sources (VIS-detected) with IE mag and phz_gal_prob."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = f"""
        SELECT
            -2.5 * LOG10({ZPC_IE} * m_flux_detection_total) + {ZP_REF} AS ie_mag,
            p_phz_gal_prob AS gal_prob
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz
        WHERE m_flux_detection_total > 0
          AND m_flux_detection_total IS NOT NULL
          AND m_vis_det = 1
          AND p_phz_gal_prob IS NOT NULL
    """
    print("  Querying all Q1+PHZ sources (VIS-detected)...")
    import pandas as pd
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} sources")
    return df


def query_gaia_matched(username):
    """Query Gaia-matched Q1+PHZ sources with IE mag, Gaia G, and phz_gal_prob."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = f"""
        SELECT
            -2.5 * LOG10({ZPC_IE} * m_flux_detection_total) + {ZP_REF} AS ie_mag,
            g_phot_g_mean_mag AS gaia_g,
            p_phz_gal_prob AS gal_prob
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE m_flux_detection_total > 0
          AND m_flux_detection_total IS NOT NULL
          AND m_vis_det = 1
          AND g_phot_g_mean_mag IS NOT NULL
          AND p_phz_gal_prob IS NOT NULL
    """
    print("  Querying Gaia-matched Q1+PHZ sources...")
    import pandas as pd
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} sources")
    return df


# ─── Plot ────────────────────────────────────────────────────────────────────

def plot_phz_gal_prob_combined(df_all, df_gaia, output_dir):
    """Combined 1x3 plot of phz_gal_prob vs magnitude 2D histograms."""
    ybins = np.arange(0, 1.01, 0.01)

    ie_mag_label = r'Magnitude ($-2.5\log_{10}(1.05 \times \mathrm{flux}) + 23.9$)'

    panels = [
        (df_all['ie_mag'].values, df_all['gal_prob'].values,
         np.arange(14, 28.05, 0.05), (14, 28),
         ie_mag_label,
         f'phz_gal_prob vs Q1 IE (all, n={len(df_all):,})'),
        (df_gaia['ie_mag'].values, df_gaia['gal_prob'].values,
         np.arange(14, 28.05, 0.05), (14, 28),
         ie_mag_label,
         f'phz_gal_prob vs Q1 IE (Gaia-matched, n={len(df_gaia):,})'),
        (df_gaia['gaia_g'].values, df_gaia['gal_prob'].values,
         np.arange(14, 28.05, 0.05), (14, 28),
         'Gaia G',
         f'phz_gal_prob vs Gaia G (Gaia-matched, n={len(df_gaia):,})'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(36, 6))

    for ax, (xvals, yvals, xbins, xlim, xlabel, title) in zip(axes, panels):
        h, xedges, yedges = np.histogram2d(xvals, yvals, bins=[xbins, ybins])
        h_masked = np.ma.masked_where(h == 0, h)
        im = ax.pcolormesh(xedges, yedges, h_masked.T,
                           cmap='GnBu', norm=LogNorm(vmin=1))
        ax.axhline(y=0.5, color='red', ls='--', linewidth=1.5, label='prob=0.5')
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Count', fontsize=11, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_ylabel('phz_gal_prob', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'phz_gal_prob_2d.png'
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
    print("phz_gal_prob vs Magnitude 2D Histograms")
    print("=" * 70)

    print("\n[1] Querying all Q1+PHZ sources...")
    df_all = query_all_q1(username)

    print("\n[2] Querying Gaia-matched Q1+PHZ sources...")
    df_gaia = query_gaia_matched(username)

    print("\n[3] Generating combined phz_gal_prob 2D histogram...")
    plot_phz_gal_prob_combined(df_all, df_gaia, OUTPUT_DIR)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
