#!/usr/bin/env python3
"""
class_star vs magnitude and Gaia neighbour count — produces:
  1. ero_class_star_2d.png              — 1x3: class_star vs Euclid VIS / Gaia G / Gaia G excl. 17.5–18.5
  2. ero_class_star_neighbours_2d.png   — class_star vs Gaia G by neighbour count (4 radii × 4 bins)
  3. ero_class_star_vs_cnt.png          — class_star vs neighbour count at each radius

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_ero_class_star_neighbours.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR     = PROJECT_DIR / 'output'
NEIGHBOURS_DIR = PROJECT_DIR / 'output' / 'ero_neighbours'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

ERO_FIELDS = {
    'Horsehead':    (85.150, -2.613,  0.58),
    'Messier 78':   (86.690, -0.015,  0.60),
    'Taurus':       (64.983, +28.023, 0.61),
    'Barnard 30':   (82.880, +12.316, 0.60),
    'NGC 6254':     (254.303, -4.100, 0.60),
    'NGC 6397':     (265.174, -53.658, 0.61),
    'NGC 6822':     (296.236, -14.788, 0.60),
    'Holmberg II':  (124.790, +70.706, 0.60),
    'IC 10':        (5.063, +59.288, 0.62),
    'IC 342':       (56.730, +68.084, 0.59),
    'NGC 2403':     (114.211, +65.586, 0.60),
    'NGC 6744':     (287.430, -63.842, 0.60),
    'Fornax':       (54.017, -35.267, 0.57),
    'Dorado':       (64.014, -55.780, 0.60),
    'Perseus':      (49.638, +41.651, 0.70),
    'Abell 2390':   (328.397, +17.709, 0.75),
    'Abell 2764':   (5.713, -49.249, 0.75),
}

RADII = [
    ('cnt4',  '4"'),
    ('cnt8',  '8"'),
    ('cnt16', '16"'),
    ('cnt32', '32"'),
]


def assign_fields(df):
    """Assign each source to its nearest ERO field (within field radius)."""
    ra  = df['ra'].values
    dec = df['dec'].values
    n_sources = len(df)

    field_assignments = np.full(n_sources, None, dtype=object)
    min_distances     = np.full(n_sources, np.inf)

    for field_name, (field_ra, field_dec, area) in ERO_FIELDS.items():
        radius = np.sqrt(area / np.pi)
        ra1  = np.radians(ra);  dec1 = np.radians(dec)
        ra2  = np.radians(field_ra); dec2 = np.radians(field_dec)
        a    = np.sin((dec2 - dec1) / 2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2)**2
        dist = np.degrees(2 * np.arcsin(np.sqrt(a)))
        mask = (dist < radius) & (dist < min_distances)
        field_assignments[mask] = field_name
        min_distances[mask]     = dist[mask]

    df['field'] = field_assignments
    return df


# ─── Data queries ────────────────────────────────────────────────────────────

def query_ero_data(username):
    """Query ERO crossmatch (Gaia-matched) for the class_star vs magnitude plot."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = """
        SELECT
            e_v_alpha_j2000 AS ra,
            e_v_delta_j2000 AS dec,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            e_v_mag_auto AS v_mag_auto,
            e_v_class_star AS class_star
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_class_star IS NOT NULL
        ORDER BY g_source_id
    """
    print("  Querying ERO crossmatch data...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} sources")
    return df


def query_all_ero(username):
    """Query ALL ERO sources from euclid_ero.merge_cat."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = """
        SELECT
            v_alpha_j2000 AS ra,
            v_delta_j2000 AS dec,
            v_mag_auto,
            v_class_star AS class_star
        FROM euclid_ero.merge_cat
        WHERE v_class_star IS NOT NULL
    """
    print("  Querying all ERO sources...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} sources")
    return df


def query_neighbours(username):
    """Query ERO crossmatch joined with Gaia neighbour counts."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = """
        SELECT
            e_v_alpha_j2000    AS ra,
            e_v_delta_j2000    AS dec,
            g_phot_g_mean_mag  AS phot_g_mean_mag,
            e_v_class_star     AS class_star,
            n.cnt4, n.cnt8, n.cnt16, n.cnt32
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3 c
        JOIN gaia_edr3_aux.gaia_source_neighbour_count n
            ON c.g_source_id = n.source_id
        WHERE e_v_class_star IS NOT NULL
    """
    print("  Querying ERO crossmatch + neighbour counts...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} sources")
    return df


# ─── Plots ───────────────────────────────────────────────────────────────────

def plot_class_star_combined(df, df_ero, output_dir):
    """Combined 1x3 plot of class_star vs magnitude 2D histograms."""
    ybins = np.arange(0, 1.01, 0.01)

    sub_gaia      = df[(df['field'].notna()) & df['phot_g_mean_mag'].notna() & df['class_star'].notna()]
    sub_gaia_excl = sub_gaia[~sub_gaia['v_mag_auto'].between(17.5, 18.5)]
    sub_ero       = df_ero[(df_ero['field'].notna()) & df_ero['v_mag_auto'].notna() & df_ero['class_star'].notna()]

    panels = [
        (sub_ero['v_mag_auto'].values,          sub_ero['class_star'].values,
         np.arange(14, 26.05, 0.05), (14, 26),
         'Euclid VIS mag', 'class_star vs Euclid VIS (all ERO)'),
        (sub_gaia['phot_g_mean_mag'].values,     sub_gaia['class_star'].values,
         np.arange(14, 22.05, 0.05), (14, 22),
         'Gaia G', 'class_star vs Gaia G (Gaia-matched)'),
        (sub_gaia_excl['phot_g_mean_mag'].values, sub_gaia_excl['class_star'].values,
         np.arange(14, 22.05, 0.05), (14, 22),
         'Gaia G', 'class_star vs Gaia G (excl. ERO VIS 17.5–18.5)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(36, 6))
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


def plot_class_star_by_neighbours(df, output_dir):
    """4×4 grid: rows = radii, cols = neighbour count bins.
    Each panel: 2D histogram of class_star vs Gaia G (log counts).
    """
    sub = df[(df['field'].notna()) & df['phot_g_mean_mag'].notna() & df['class_star'].notna()]

    xbins = np.arange(14, 22.05, 0.05)
    ybins = np.arange(0, 1.01, 0.01)

    count_bins = [
        (lambda s, c: s[c] == 1,           '= 1 (isolated)'),
        (lambda s, c: s[c] == 2,           '= 2'),
        (lambda s, c: s[c].between(3, 4),  '= 3–4'),
        (lambda s, c: s[c] >= 5,           '≥ 5 (crowded)'),
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

            h, xe, ye = np.histogram2d(panel_df['phot_g_mean_mag'].values,
                                       panel_df['class_star'].values,
                                       bins=[xbins, ybins])
            hm = np.ma.masked_where(h == 0, h)
            vmax = hm.max() if hm.count() > 0 else 10
            im = ax.pcolormesh(xe, ye, hm.T, cmap='Blues',
                               norm=LogNorm(vmin=1, vmax=max(2, vmax)))
            fig.colorbar(im, ax=ax, pad=0.02).set_label('Count', fontsize=10)
            ax.axhline(y=0.65, color='red', ls='--', linewidth=1.5, label='cut=0.65')

            if row == 0:
                ax.set_title(f'cnt {bin_label}', fontsize=12, fontweight='bold')
            else:
                ax.set_title('')
            if col == 0:
                ax.set_ylabel('class_star', fontsize=10)
                ax.annotate(f'r = {radius_label}', xy=(0, 0.5),
                            xycoords='axes fraction', xytext=(-0.17, 0.5),
                            textcoords='axes fraction', fontsize=12,
                            fontweight='bold', ha='center', va='center', rotation=0)
            else:
                ax.set_ylabel('class_star', fontsize=10)
            ax.text(0.03, 0.97, f'N={n:,}', transform=ax.transAxes, fontsize=9, va='top', ha='left')
            ax.set_xlabel('Gaia G' if row == n_rows - 1 else '', fontsize=10)
            ax.set_xlim(14, 22)
            ax.set_ylim(0, 1)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(alpha=0.3)

    plt.suptitle('class_star vs Gaia G by neighbour count  (ERO)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.subplots_adjust(left=0.08)
    out = output_dir / 'ero_class_star_neighbours_2d.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_class_star_vs_cnt(df, output_dir):
    """1×4 row: class_star vs neighbour count 2D histogram at each radius."""
    base  = df[(df['field'].notna()) & df['class_star'].notna()]
    ybins = np.arange(0, 1.01, 0.01)

    fig, axes = plt.subplots(1, len(RADII), figsize=(6 * len(RADII), 6))
    for ax, (cnt_col, radius_label) in zip(axes, RADII):
        sub = base[base[cnt_col].notna()]
        cnt_max = int(sub[cnt_col].quantile(0.99)) if len(sub) > 0 else 5
        cnt_max = max(cnt_max, 2)
        xbins = np.arange(0.5, cnt_max + 1.5, 1)

        h, xe, ye = np.histogram2d(sub[cnt_col].values, sub['class_star'].values,
                                   bins=[xbins, ybins])
        hm = np.ma.masked_where(h == 0, h)
        vmax = hm.max() if hm.count() > 0 else 10
        im = ax.pcolormesh(xe, ye, hm.T, cmap='Blues',
                           norm=LogNorm(vmin=1, vmax=max(2, vmax)))
        ax.axhline(y=0.65, color='red', ls='--', linewidth=1.5, label='cut=0.65')
        fig.colorbar(im, ax=ax, pad=0.02).set_label('Count', fontsize=10)
        ax.set_xlabel(f'{cnt_col} (neighbours within {radius_label})',
                      fontsize=10, fontweight='bold')
        ax.set_ylabel('class_star', fontsize=10)
        ax.set_title(f'r = {radius_label}  (N={len(sub):,})', fontsize=12, fontweight='bold')
        ax.set_xlim(xbins[0], xbins[-1])
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = output_dir / 'ero_class_star_vs_cnt.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NEIGHBOURS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ERO class_star vs Magnitude and Neighbour Count")
    print("=" * 70)

    print("\n[1] Querying Gaia-matched ERO data...")
    df = query_ero_data(username)
    df = assign_fields(df)

    print("\n[2] Querying all ERO sources...")
    df_ero = query_all_ero(username)
    df_ero = assign_fields(df_ero)

    print("\n[3] Querying neighbour counts...")
    df_nb = query_neighbours(username)
    df_nb = assign_fields(df_nb)

    print("\n[4] Generating class_star vs magnitude 2D histogram...")
    plot_class_star_combined(df, df_ero, OUTPUT_DIR)

    print("\n[5] Generating class_star vs Gaia G by neighbour count...")
    plot_class_star_by_neighbours(df_nb, NEIGHBOURS_DIR)

    print("\n[6] Generating class_star vs neighbour count (all radii)...")
    plot_class_star_vs_cnt(df_nb, NEIGHBOURS_DIR)

    print(f"\nPlots saved to: {OUTPUT_DIR} and {NEIGHBOURS_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
