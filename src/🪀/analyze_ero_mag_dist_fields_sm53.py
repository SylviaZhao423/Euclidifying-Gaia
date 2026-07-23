#!/usr/bin/env python3
"""
Plot magnitude distributions and class fractions for each ERO field.
Classifier: SM + 5/3×SME ≥ 0.005  → GALAXY, else STAR.

Each field cell contains two sub-panels:
  top  — count histogram (stars coral, galaxies steelblue)
  bottom — class fraction (galaxy 'o-' steelblue, star coral alpha=0.3 no marker)

Produces (saved to output/ero_mag_dist_53sme/):
  1. gaia_fields_g_mag.png    — Gaia G,       Gaia-matched sources
  2. gaia_fields_e_mag.png    — Euclid VIS,   Gaia-matched sources
  3. ero_all_fields_e_mag.png — Euclid VIS,   all ERO sources
  4. combined_summary.png     — 2×3 summary across all fields combined

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_ero_mag_dist_fields_53sme.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'output' / 'ero_mag_dist_sm53'
WSDB_HOST   = 'wsdb.ast.cam.ac.uk'
WSDB_DB     = 'wsdb'

ERO_FIELDS = {
    'Horsehead':   (85.150,  -2.613, 0.58),
    'Messier 78':  (86.690,  -0.015, 0.60),
    'Taurus':      (64.983,  28.023, 0.61),
    'Barnard 30':  (82.880,  12.316, 0.60),
    'NGC 6254':    (254.303,  -4.100, 0.60),
    'NGC 6397':    (265.174, -53.658, 0.61),
    'NGC 6822':    (296.236, -14.788, 0.60),
    'Holmberg II': (124.790,  70.706, 0.60),
    'IC 10':       (  5.063,  59.288, 0.62),
    'IC 342':      ( 56.730,  68.084, 0.59),
    'NGC 2403':    (114.211,  65.586, 0.60),
    'NGC 6744':    (287.430, -63.842, 0.60),
    'Fornax':      ( 54.017, -35.267, 0.57),
    'Dorado':      ( 64.014, -55.780, 0.60),
    'Perseus':     ( 49.638,  41.651, 0.70),
    'Abell 2390':  (328.397,  17.709, 0.75),
    'Abell 2764':  (  5.713, -49.249, 0.75),
}

FIELD_CATEGORIES = {
    'Star-forming':      ['Horsehead', 'Messier 78', 'Taurus', 'Barnard 30'],
    'Globular Clusters': ['NGC 6254', 'NGC 6397'],
    'Nearby Galaxies':   ['NGC 6822', 'Holmberg II', 'IC 10', 'IC 342', 'NGC 2403', 'NGC 6744'],
    'Galaxy Groups':     ['Fornax', 'Dorado'],
    'Galaxy Clusters':   ['Perseus', 'Abell 2390', 'Abell 2764'],
}

CATEGORY_COLORS = {
    'Star-forming':      plt.cm.Reds,
    'Globular Clusters': plt.cm.Greens,
    'Nearby Galaxies':   plt.cm.Blues,
    'Galaxy Groups':     plt.cm.Purples,
    'Galaxy Clusters':   plt.cm.Oranges,
}

CLASSIFIER_LABEL = 'SM + 5/3×SME ≥ 0.005'


# ─── Shared helpers ───────────────────────────────────────────────────────────

def assign_fields(df):
    ra  = df['ra'].values
    dec = df['dec'].values
    n_sources = len(df)

    field_assignments = np.full(n_sources, None, dtype=object)
    min_distances     = np.full(n_sources, np.inf)

    for field_name, (field_ra, field_dec, _area) in ERO_FIELDS.items():
        ra1  = np.radians(ra);       dec1 = np.radians(dec)
        ra2  = np.radians(field_ra); dec2 = np.radians(field_dec)
        a    = (np.sin((dec2 - dec1) / 2)**2
                + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2)**2)
        dist = np.degrees(2 * np.arcsin(np.sqrt(a)))
        mask = dist < min_distances
        field_assignments[mask] = field_name
        min_distances[mask]     = dist[mask]

    df['field'] = field_assignments
    return df


def get_galactic_b(ra, dec):
    from astropy.coordinates import SkyCoord
    return SkyCoord(ra=ra, dec=dec, unit='deg').galactic.b.deg


# ─── Data queries ─────────────────────────────────────────────────────────────

def query_gaia_matched(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = """
        SELECT
            e_v_alpha_j2000  AS ra,
            e_v_delta_j2000  AS dec,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            e_v_mag_auto     AS v_mag_auto,
            CASE
                WHEN e_v_spread_model + (5.0/3) * e_v_spreaderr_model >= 0.005 THEN 'GALAXY'
                ELSE 'STAR'
            END AS label
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_spread_model    IS NOT NULL
          AND e_v_spreaderr_model IS NOT NULL
        ORDER BY g_source_id
    """
    print("Querying Gaia-matched ERO data (SM+5/3×SME classifier)...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Retrieved {len(df):,} Gaia-matched sources")
    return df


def query_all_ero(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = """
        SELECT
            v_alpha_j2000 AS ra,
            v_delta_j2000 AS dec,
            v_mag_auto,
            CASE
                WHEN v_spread_model + (5.0/3) * v_spreaderr_model >= 0.005 THEN 'GALAXY'
                ELSE 'STAR'
            END AS label
        FROM euclid_ero.merge_cat
        WHERE v_spread_model    IS NOT NULL
          AND v_spreaderr_model IS NOT NULL
    """
    print("Querying all ERO data (SM+5/3×SME classifier)...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Retrieved {len(df):,} ERO sources")
    return df


# ─── Shared plot helpers ──────────────────────────────────────────────────────

def _make_field_figure(n_fields, n_cols, cell_h=5):
    n_frows  = (n_fields + n_cols - 1) // n_cols
    fig      = plt.figure(figsize=(30, cell_h * n_frows))
    outer_gs = gridspec.GridSpec(n_frows, n_cols, figure=fig,
                                 hspace=0.25, wspace=0.2,
                                 top=0.94, bottom=0.03, left=0.06, right=0.98)
    return fig, outer_gs, n_frows


def _add_field_axes(fig, outer_gs, idx, n_cols):
    frow     = idx // n_cols
    fcol     = idx % n_cols
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[frow, fcol],
        hspace=0.08, height_ratios=[2.5, 1.5]
    )
    ax_top = fig.add_subplot(inner_gs[0])
    ax_bot = fig.add_subplot(inner_gs[1], sharex=ax_top)
    return ax_top, ax_bot


def _plot_fraction_panel(ax, bin_centers, fd_all, fd_gal, fd_star, bins, min_count=5):
    if len(fd_all) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes,
                fontsize=8)
        return
    counts_all,  _ = np.histogram(fd_all,  bins=bins)
    counts_gal,  _ = np.histogram(fd_gal,  bins=bins)
    counts_star, _ = np.histogram(fd_star, bins=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        frac_gal  = np.where(counts_all > 0, counts_gal  / counts_all, np.nan)
        frac_star = np.where(counts_all > 0, counts_star / counts_all, np.nan)
    mask = counts_all >= min_count
    ax.plot(bin_centers[mask], frac_star[mask], color='coral',
            linewidth=1.5, alpha=0.3)
    ax.plot(bin_centers[mask], frac_gal[mask],  'o-', color='steelblue',
            markersize=2, linewidth=1.5)


# ─── Per-field plots ──────────────────────────────────────────────────────────

def plot_gaia_g_mag(df, output_dir):
    df_assigned = df[df['field'].notna()].copy()
    print(f"\nGaia-matched sources assigned to fields: {len(df_assigned):,}")

    n_fields = len(ERO_FIELDS)
    n_cols   = 5
    fig, outer_gs, n_frows = _make_field_figure(n_fields, n_cols)

    bins        = np.arange(10, 29.25, 0.25)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax_top, ax_bot = _add_field_axes(fig, outer_gs, idx, n_cols)

        fd      = df_assigned[df_assigned['field'] == field_name]
        fd_star = fd[fd['label'] == 'STAR']['phot_g_mean_mag'].dropna()
        fd_gal  = fd[fd['label'] == 'GALAXY']['phot_g_mean_mag'].dropna()
        fd_all  = fd['phot_g_mean_mag'].dropna()
        gal_b   = get_galactic_b(*ERO_FIELDS[field_name][:2])

        n_gal  = int(np.histogram(fd_gal,  bins=bins)[0].sum())
        n_star = int(np.histogram(fd_star, bins=bins)[0].sum())
        n_all  = n_gal + n_star
        ax_top.hist(fd_all,  bins=bins, alpha=0.3, color='gray',
                    label=f'All ({n_all:,})')
        ax_top.hist(fd_gal,  bins=bins, alpha=0.7, color='steelblue',
                    label=f'Galaxies ({n_gal:,})', histtype='step', linewidth=1.5)
        ax_top.hist(fd_star, bins=bins, alpha=0.7, color='coral',
                    label=f'Stars ({n_star:,})', histtype='step', linewidth=1.5)
        ax_top.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
        ax_top.set_ylabel('Count', fontsize=8)
        ax_top.set_xlim(10, 29)
        ax_top.legend(fontsize=5, loc='upper left')
        ax_top.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax_top.grid(alpha=0.3, axis='y')
        plt.setp(ax_top.get_xticklabels(), visible=False)
        ax_top.tick_params(labelsize=7)

        _plot_fraction_panel(ax_bot, bin_centers, fd_all, fd_gal, fd_star, bins)
        ax_bot.set_xlabel('Gaia G', fontsize=8)
        ax_bot.set_ylabel('Fraction', fontsize=8)
        ax_bot.set_xlim(10, 29)
        ax_bot.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax_bot.set_ylim(-0.02, 1.02)
        ax_bot.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax_bot.grid(alpha=0.3)
        ax_bot.tick_params(labelsize=7)

    for idx in range(n_fields, n_frows * n_cols):
        fig.add_subplot(outer_gs[idx // n_cols, idx % n_cols]).axis('off')

    fig.suptitle(f'Gaia G Distribution & Class Fraction by Field (Gaia-matched, {CLASSIFIER_LABEL})',
                 fontsize=14, fontweight='bold', y=0.97)
    out = output_dir / 'gaia_fields_g_mag.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


def plot_gaia_e_mag(df, output_dir):
    df_assigned = df[df['field'].notna()].copy()

    n_fields = len(ERO_FIELDS)
    n_cols   = 5
    fig, outer_gs, n_frows = _make_field_figure(n_fields, n_cols)

    bins        = np.arange(10, 29.25, 0.25)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax_top, ax_bot = _add_field_axes(fig, outer_gs, idx, n_cols)

        fd      = df_assigned[df_assigned['field'] == field_name]
        fd_star = fd[fd['label'] == 'STAR']['v_mag_auto'].dropna()
        fd_gal  = fd[fd['label'] == 'GALAXY']['v_mag_auto'].dropna()
        fd_all  = fd['v_mag_auto'].dropna()
        gal_b   = get_galactic_b(*ERO_FIELDS[field_name][:2])

        n_gal  = int(np.histogram(fd_gal,  bins=bins)[0].sum())
        n_star = int(np.histogram(fd_star, bins=bins)[0].sum())
        n_all  = n_gal + n_star
        ax_top.hist(fd_all,  bins=bins, alpha=0.3, color='gray',
                    label=f'All ({n_all:,})')
        ax_top.hist(fd_gal,  bins=bins, alpha=0.7, color='steelblue',
                    label=f'Galaxies ({n_gal:,})', histtype='step', linewidth=1.5)
        ax_top.hist(fd_star, bins=bins, alpha=0.7, color='coral',
                    label=f'Stars ({n_star:,})', histtype='step', linewidth=1.5)
        ax_top.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
        ax_top.set_ylabel('Count', fontsize=8)
        ax_top.set_xlim(10, 29)
        ax_top.legend(fontsize=5, loc='upper left')
        ax_top.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax_top.grid(alpha=0.3, axis='y')
        plt.setp(ax_top.get_xticklabels(), visible=False)
        ax_top.tick_params(labelsize=7)

        _plot_fraction_panel(ax_bot, bin_centers, fd_all, fd_gal, fd_star, bins)
        ax_bot.set_xlabel('Euclid VIS (v_mag_auto)', fontsize=8)
        ax_bot.set_ylabel('Fraction', fontsize=8)
        ax_bot.set_xlim(10, 29)
        ax_bot.set_ylim(-0.02, 1.02)
        ax_bot.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax_bot.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax_bot.grid(alpha=0.3)
        ax_bot.tick_params(labelsize=7)

    for idx in range(n_fields, n_frows * n_cols):
        fig.add_subplot(outer_gs[idx // n_cols, idx % n_cols]).axis('off')

    fig.suptitle(f'Euclid VIS Distribution & Class Fraction by Field (Gaia-matched, {CLASSIFIER_LABEL})',
                 fontsize=14, fontweight='bold', y=0.97)
    out = output_dir / 'gaia_fields_e_mag.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


def plot_ero_all_e_mag(df, output_dir):
    df_assigned = df[df['field'].notna()].copy()
    print(f"\nAll ERO sources assigned to fields: {len(df_assigned):,}")
    field_counts = df_assigned['field'].value_counts()
    print("Sources per field:")
    for field in ERO_FIELDS.keys():
        print(f"  {field}: {field_counts.get(field, 0):,}")

    n_fields = len(ERO_FIELDS)
    n_cols   = 5
    fig, outer_gs, n_frows = _make_field_figure(n_fields, n_cols)

    bins        = np.arange(10, 29.25, 0.25)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax_top, ax_bot = _add_field_axes(fig, outer_gs, idx, n_cols)

        fd      = df_assigned[df_assigned['field'] == field_name]
        fd_star = fd[fd['label'] == 'STAR']['v_mag_auto'].dropna()
        fd_gal  = fd[fd['label'] == 'GALAXY']['v_mag_auto'].dropna()
        fd_all  = fd['v_mag_auto'].dropna()
        gal_b   = get_galactic_b(*ERO_FIELDS[field_name][:2])

        n_gal  = int(np.histogram(fd_gal,  bins=bins)[0].sum())
        n_star = int(np.histogram(fd_star, bins=bins)[0].sum())
        n_all  = n_gal + n_star
        ax_top.hist(fd_all,  bins=bins, alpha=0.3, color='gray',
                    label=f'All ({n_all:,})')
        ax_top.hist(fd_gal,  bins=bins, alpha=0.7, color='steelblue',
                    label=f'Galaxies ({n_gal:,})', histtype='step', linewidth=1.5)
        ax_top.hist(fd_star, bins=bins, alpha=0.7, color='coral',
                    label=f'Stars ({n_star:,})', histtype='step', linewidth=1.5)
        ax_top.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
        ax_top.set_ylabel('Count', fontsize=8)
        ax_top.set_xlim(10, 29)
        ax_top.legend(fontsize=5, loc='upper left')
        ax_top.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax_top.grid(alpha=0.3, axis='y')
        plt.setp(ax_top.get_xticklabels(), visible=False)
        ax_top.tick_params(labelsize=7)

        _plot_fraction_panel(ax_bot, bin_centers, fd_all, fd_gal, fd_star, bins,
                             min_count=10)
        ax_bot.set_xlabel('Euclid VIS (v_mag_auto)', fontsize=8)
        ax_bot.set_ylabel('Fraction', fontsize=8)
        ax_bot.set_xlim(10, 29)
        ax_bot.set_ylim(-0.02, 1.02)
        ax_bot.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax_bot.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax_bot.grid(alpha=0.3)
        ax_bot.tick_params(labelsize=7)

    for idx in range(n_fields, n_frows * n_cols):
        fig.add_subplot(outer_gs[idx // n_cols, idx % n_cols]).axis('off')

    fig.suptitle(f'Euclid VIS Distribution & Class Fraction by Field (All ERO, {CLASSIFIER_LABEL})',
                 fontsize=14, fontweight='bold', y=0.97)
    out = output_dir / 'ero_all_fields_e_mag.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ─── Combined summary plot ────────────────────────────────────────────────────

def plot_combined_summary(df_gaia, df_ero, output_dir):
    df_gaia_f = df_gaia[df_gaia['field'].notna()].copy()
    df_gaia_f = df_gaia_f.dropna(subset=['v_mag_auto', 'phot_g_mean_mag'])
    df_ero_f  = df_ero[df_ero['field'].notna()].copy()

    fig, axes = plt.subplots(2, 3, figsize=(21, 10))

    bins = np.arange(10, 29.25, 0.25)
    configs = [
        (df_ero_f,  'v_mag_auto',      bins, 'Euclid VIS (v_mag_auto)', (10, 29), 'All ERO, Euclid VIS'),
        (df_gaia_f, 'v_mag_auto',      bins, 'Euclid VIS (v_mag_auto)', (10, 29), 'Gaia-matched, Euclid VIS'),
        (df_gaia_f, 'phot_g_mean_mag', bins, 'Gaia G',                  (10, 29), 'Gaia-matched, Gaia G'),
    ]

    for col_idx, (df, mag_col, bins, xlabel, xlim, title_suffix) in enumerate(configs):
        mag_all  = df[mag_col].dropna()
        mag_star = df[df['label'] == 'STAR'][mag_col].dropna()
        mag_gal  = df[df['label'] == 'GALAXY'][mag_col].dropna()

        n_gal  = int(np.histogram(mag_gal,  bins=bins)[0].sum())
        n_star = int(np.histogram(mag_star, bins=bins)[0].sum())
        n_all  = n_gal + n_star

        ax = axes[0, col_idx]
        ax.hist(mag_all,  bins=bins, alpha=0.3, color='gray',
                label=f'All ({n_all:,})')
        ax.hist(mag_gal,  bins=bins, alpha=0.7, color='steelblue',
                label=f'Galaxies ({n_gal:,})', histtype='step', linewidth=1.5)
        ax.hist(mag_star, bins=bins, alpha=0.7, color='coral',
                label=f'Stars ({n_star:,})', histtype='step', linewidth=1.5)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_xlim(xlim)
        ax.set_title(f'Sources Count ({title_suffix})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
        ax.grid(alpha=0.3, axis='y')

        ax = axes[1, col_idx]
        bin_centers  = (bins[:-1] + bins[1:]) / 2
        counts_all,  _ = np.histogram(mag_all,  bins=bins)
        counts_gal,  _ = np.histogram(mag_gal,  bins=bins)
        counts_star, _ = np.histogram(mag_star, bins=bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            frac_gal  = np.where(counts_all > 0, counts_gal  / counts_all, np.nan)
            frac_star = np.where(counts_all > 0, counts_star / counts_all, np.nan)
        mask = counts_all >= 10
        ax.plot(bin_centers[mask], frac_star[mask], color='coral',
                linewidth=1.5, alpha=0.3)
        ax.plot(bin_centers[mask], frac_gal[mask],  'o-', color='steelblue',
                markersize=3, linewidth=1.5)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Class fraction', fontsize=10)
        ax.set_xlim(xlim)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'Class Fraction ({title_suffix})', fontsize=11, fontweight='bold')
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.grid(alpha=0.3)

    plt.suptitle(f'All ERO Fields Combined: Magnitude Distributions & Class Fraction ({CLASSIFIER_LABEL})',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = output_dir / 'combined_summary.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"ERO Fields: Magnitude Distributions ({CLASSIFIER_LABEL})")
    print("=" * 70)

    df_gaia = query_gaia_matched(username)
    print("\nAssigning Gaia-matched sources to fields...")
    df_gaia = assign_fields(df_gaia)

    print("\n--- Gaia-matched: Gaia G magnitude ---")
    plot_gaia_g_mag(df_gaia, OUTPUT_DIR)

    print("\n--- Gaia-matched: Euclid VIS magnitude ---")
    plot_gaia_e_mag(df_gaia, OUTPUT_DIR)

    df_ero = query_all_ero(username)
    print("\nAssigning all ERO sources to fields...")
    df_ero = assign_fields(df_ero)

    print("\n--- All ERO: Euclid VIS magnitude ---")
    plot_ero_all_e_mag(df_ero, OUTPUT_DIR)

    print("\n--- Combined summary (all fields) ---")
    plot_combined_summary(df_gaia, df_ero, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("Done!")
    print(f"All results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
