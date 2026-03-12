#!/usr/bin/env python3
"""
class_star vs magnitude 2D histograms — produces:
  ero_prob_class_star_2d.png  — 1x3 combined plot:
    (a) class_star vs Euclid VIS mag (all ERO, from euclid_ero.merge_cat)
    (b) class_star vs Gaia G (Gaia-matched, from crossmatch_ero_gaia_0_3)
    (c) class_star vs Gaia G (Gaia-matched, excl. ERO VIS 17.5–18.5)

Extracted from analyze_ero_class_star_neighbours.py.

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_ero_class_star_2d.py
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

ERO_FIELDS = {
    'Horsehead':   (85.150,  -2.613),
    'Messier 78':  (86.690,  -0.015),
    'Taurus':      (64.983,  28.023),
    'Barnard 30':  (82.880,  12.316),
    'NGC 6254':    (254.303,  -4.100),
    'NGC 6397':    (265.174, -53.658),
    'NGC 6822':    (296.236, -14.788),
    'Holmberg II': (124.790,  70.706),
    'IC 10':       (  5.063,  59.288),
    'IC 342':      ( 56.730,  68.084),
    'NGC 2403':    (114.211,  65.586),
    'NGC 6744':    (287.430, -63.842),
    'Fornax':      ( 54.017, -35.267),
    'Dorado':      ( 64.014, -55.780),
    'Perseus':     ( 49.638,  41.651),
    'Abell 2390':  (328.397,  17.709),
    'Abell 2764':  (  5.713, -49.249),
}

_FIELD_NAMES = list(ERO_FIELDS.keys())
_FIELD_RA    = np.array([v[0] for v in ERO_FIELDS.values()])
_FIELD_DEC   = np.array([v[1] for v in ERO_FIELDS.values()])


def assign_fields(df):
    """Assign each source to its nearest ERO field centre."""
    ra  = np.radians(df['ra'].values)
    dec = np.radians(df['dec'].values)
    fra = np.radians(_FIELD_RA)
    fde = np.radians(_FIELD_DEC)
    a = (np.sin((fde - dec[:, None]) / 2) ** 2
         + np.cos(dec[:, None]) * np.cos(fde) * np.sin((fra - ra[:, None]) / 2) ** 2)
    nearest = np.argmin(a, axis=1)
    df['field'] = [_FIELD_NAMES[i] for i in nearest]
    return df


# ─── Data queries ─────────────────────────────────────────────────────────────

def query_gaia_matched(username):
    """Query Gaia-matched ERO sources for the class_star vs magnitude plot."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = """
        SELECT
            e_v_alpha_j2000   AS ra,
            e_v_delta_j2000   AS dec,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            e_v_mag_auto      AS v_mag_auto,
            e_v_class_star    AS class_star
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_class_star IS NOT NULL
        ORDER BY g_source_id
    """
    print("  Querying Gaia-matched ERO data...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} sources")
    return df


def query_all_ero(username):
    """Query all ERO sources from euclid_ero.merge_cat."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = """
        SELECT
            v_alpha_j2000 AS ra,
            v_delta_j2000 AS dec,
            v_mag_auto,
            v_class_star  AS class_star
        FROM euclid_ero.merge_cat
        WHERE v_class_star IS NOT NULL
    """
    print("  Querying all ERO sources...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} sources")
    return df


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_class_star_combined(df_gaia, df_ero, output_dir):
    """Combined 1x3 plot of class_star vs magnitude 2D histograms."""
    show_third_panel = False
    ybins = np.arange(0, 1.01, 0.004)

    sub_gaia      = df_gaia[(df_gaia['field'].notna()) & df_gaia['phot_g_mean_mag'].notna() & df_gaia['class_star'].notna()]
    sub_gaia_excl = sub_gaia[~sub_gaia['v_mag_auto'].between(17.5, 18.5)]
    sub_ero       = df_ero[(df_ero['field'].notna()) & df_ero['v_mag_auto'].notna() & df_ero['class_star'].notna()]

    panels = [
        (sub_ero['v_mag_auto'].values,           sub_ero['class_star'].values,
         np.arange(10, 28.05, 0.05), (10, 28),
         'Euclid VIS mag', 'class_star vs Euclid VIS (all ERO)'),
        (sub_gaia['phot_g_mean_mag'].values,      sub_gaia['class_star'].values,
         np.arange(10, 28.05, 0.05), (10, 28),
         'Gaia G', 'class_star vs Gaia G (Gaia-matched)'),
        (sub_gaia_excl['phot_g_mean_mag'].values, sub_gaia_excl['class_star'].values,
         np.arange(10, 28.05, 0.05), (10, 28),
         'Gaia G', 'class_star vs Gaia G (excl. ERO VIS 17.5–18.5)'),
    ]

    if not show_third_panel:
        panels = panels[:2]

    fig, axes = plt.subplots(1, len(panels), figsize=(10 * len(panels), 6))
    if len(panels) == 1:
        axes = [axes]
    for ax, (xvals, yvals, xbins, xlim, xlabel, title) in zip(axes, panels):
        h, xe, ye = np.histogram2d(xvals, yvals, bins=[xbins, ybins])
        hm = np.ma.masked_where(h == 0, h)
        im = ax.pcolormesh(xe, ye, hm.T, cmap='GnBu', norm=LogNorm(vmin=1))
        ax.axhline(y=0.65, color='red', ls='--', linewidth=1.5, label='cut=0.65')
        fig.colorbar(im, ax=ax, pad=0.02).set_label('Count', fontsize=11, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_ylabel('class_star', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = output_dir / 'ero_prob_class_star_2d.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ERO class_star vs Magnitude 2D Histograms")
    print("=" * 70)

    print("\n[1] Querying Gaia-matched ERO data...")
    df_gaia = query_gaia_matched(username)
    df_gaia = assign_fields(df_gaia)

    print("\n[2] Querying all ERO sources...")
    df_ero = query_all_ero(username)
    df_ero = assign_fields(df_ero)

    print("\n[3] Generating class_star vs magnitude 2D histogram...")
    plot_class_star_combined(df_gaia, df_ero, OUTPUT_DIR)

    print(f"\nPlot saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
