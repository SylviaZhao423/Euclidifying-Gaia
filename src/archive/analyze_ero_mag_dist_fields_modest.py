#!/usr/bin/env python3
"""
Plot magnitude distributions and class fractions for each ERO field.
Classifier: MODEST_CLASS-style 4-class scheme (Drlica-Wagner et al. 2018)
  using SM + (5/3)×SME in the Euclid VIS band (no WAVG equivalent available).

  Class 0 — SM + (5/3)×SME < −0.002              Unphysical PSF (likely stars)
  Class 1 — SM + (5/3)×SME >  0.005              High-confidence galaxies
  Class 2 — |SM + (5/3)×SME| < 0.002             High-confidence stars
  Class 3 — 0.002 ≤ SM + (5/3)×SME ≤ 0.005      Ambiguous

Note: the WAVG_SPREAD_MODEL veto in the original Class 1 definition is omitted
because no multi-epoch weighted-average spread model exists for ERO.

Each field cell contains two sub-panels:
  top    — count histogram per class (gray fill + 4 coloured steps)
  bottom — class fraction vs magnitude (4 coloured lines)

Produces (saved to output/ero_mag_dist_modest/):
  1. gaia_fields_g_mag.png    — Gaia G,       Gaia-matched sources
  2. gaia_fields_e_mag.png    — Euclid VIS,   Gaia-matched sources
  3. ero_all_fields_e_mag.png — Euclid VIS,   all ERO sources
  4. combined_summary.png     — 2×3 summary across all fields combined

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_ero_mag_dist_fields_modest.py
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
OUTPUT_DIR  = PROJECT_DIR / 'output' / 'ero_mag_dist_modest'
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

# MODEST_CLASS colours and labels
CLASS_COLORS = {
    0: 'slategray',   # Unphysical (likely stars)
    1: 'steelblue',   # High-confidence galaxies
    2: 'coral',       # High-confidence stars
    3: 'goldenrod',   # Ambiguous
}
CLASS_LABELS = {
    0: 'C0: Unphysical',
    1: 'C1: Galaxies',
    2: 'C2: Stars',
    3: 'C3: Ambiguous',
}

CLASSIFIER_LABEL = 'MODEST_CLASS (SM + 5/3×SME, no WAVG)'


# ─── Shared helpers ────────────────────────────────────────────────────────────

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


# ─── SQL expression helpers ────────────────────────────────────────────────────

def _modest_case(sm_col, sme_col):
    """Return SQL CASE expression assigning MODEST_CLASS 0–3."""
    x = f'({sm_col} + (5.0/3) * {sme_col})'
    return f"""
        CASE
            WHEN {x} < -0.002                        THEN 0
            WHEN {x} >  0.005                        THEN 1
            WHEN ABS({x}) < 0.002                    THEN 2
            ELSE                                          3
        END"""


# ─── Data queries ──────────────────────────────────────────────────────────────

def query_gaia_matched(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    modest = _modest_case('e_v_spread_model', 'e_v_spreaderr_model')
    query = f"""
        SELECT
            e_v_alpha_j2000   AS ra,
            e_v_delta_j2000   AS dec,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            e_v_mag_auto      AS v_mag_auto,
            {modest}          AS modest_class
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_spread_model    IS NOT NULL
          AND e_v_spreaderr_model IS NOT NULL
        ORDER BY g_source_id
    """
    print("Querying Gaia-matched ERO data (MODEST_CLASS)...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['modest_class'] = df['modest_class'].astype(int)
    print(f"Retrieved {len(df):,} Gaia-matched sources")
    _print_class_counts(df)
    return df


def query_all_ero(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    modest = _modest_case('v_spread_model', 'v_spreaderr_model')
    query = f"""
        SELECT
            v_alpha_j2000 AS ra,
            v_delta_j2000 AS dec,
            v_mag_auto,
            {modest}      AS modest_class
        FROM euclid_ero.merge_cat
        WHERE v_spread_model    IS NOT NULL
          AND v_spreaderr_model IS NOT NULL
    """
    print("Querying all ERO data (MODEST_CLASS)...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['modest_class'] = df['modest_class'].astype(int)
    print(f"Retrieved {len(df):,} ERO sources")
    _print_class_counts(df)
    return df


def _print_class_counts(df):
    for cls in [0, 1, 2, 3]:
        n = (df['modest_class'] == cls).sum()
        print(f"  {CLASS_LABELS[cls]}: {n:,}")


# ─── Shared plot helpers ───────────────────────────────────────────────────────

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


def _plot_count_panel(ax, fd, mag_col, bins, field_name, gal_b, xlabel=None):
    """Count histogram: gray fill (all) + 4 coloured class steps."""
    fd_all = fd[mag_col].dropna()
    n_all  = int(np.histogram(fd_all, bins=bins)[0].sum())
    ax.hist(fd_all, bins=bins, alpha=0.3, color='gray', label=f'All ({n_all:,})')

    for cls in [1, 2, 3, 0]:
        sub = fd[fd['modest_class'] == cls][mag_col].dropna()
        n   = int(np.histogram(sub, bins=bins)[0].sum())
        ax.hist(sub, bins=bins, histtype='step', linewidth=1.5,
                color=CLASS_COLORS[cls], label=f'{CLASS_LABELS[cls]} ({n:,})')

    ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Count', fontsize=8)
    ax.set_xlim(10, 29)
    ax.legend(fontsize=4.5, loc='upper left')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.grid(alpha=0.3, axis='y')
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.tick_params(labelsize=7)


def _plot_fraction_panel(ax, fd, mag_col, bins, min_count=5, xlabel=None):
    """Fraction panel: fraction of each MODEST_CLASS vs magnitude."""
    if len(fd) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=8)
        return

    bin_centers   = (bins[:-1] + bins[1:]) / 2
    counts_all, _ = np.histogram(fd[mag_col].dropna(), bins=bins)
    mask = counts_all >= min_count

    for cls in [1, 2, 3, 0]:
        sub = fd[fd['modest_class'] == cls][mag_col].dropna()
        counts_cls, _ = np.histogram(sub, bins=bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            frac = np.where(counts_all > 0, counts_cls / counts_all, np.nan)
        lw    = 1.5 if cls in (1, 2) else 1.0
        alpha = 1.0 if cls in (1, 2) else 0.6
        fmt   = 'o-' if cls == 1 else '-'
        ms    = 2    if cls == 1 else 0
        ax.plot(bin_centers[mask], frac[mask],
                fmt, color=CLASS_COLORS[cls], linewidth=lw,
                alpha=alpha, markersize=ms)


# ─── Per-field plots ───────────────────────────────────────────────────────────

def _plot_fields(df, mag_col, xlabel, suptitle, out_path, min_frac_count=5):
    df_assigned = df[df['field'].notna()].copy()
    print(f"\nSources assigned to fields: {len(df_assigned):,}")

    n_fields = len(ERO_FIELDS)
    n_cols   = 5
    bins     = np.arange(10, 29.25, 0.25)
    fig, outer_gs, n_frows = _make_field_figure(n_fields, n_cols)

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax_top, ax_bot = _add_field_axes(fig, outer_gs, idx, n_cols)
        fd     = df_assigned[df_assigned['field'] == field_name]
        gal_b  = get_galactic_b(*ERO_FIELDS[field_name][:2])

        _plot_count_panel(ax_top, fd, mag_col, bins, field_name, gal_b)
        _plot_fraction_panel(ax_bot, fd, mag_col, bins, min_count=min_frac_count)

        ax_bot.set_xlabel(xlabel, fontsize=8)
        ax_bot.set_ylabel('Fraction', fontsize=8)
        ax_bot.set_xlim(10, 29)
        ax_bot.set_ylim(-0.02, 1.02)
        ax_bot.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax_bot.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax_bot.grid(alpha=0.3)
        ax_bot.tick_params(labelsize=7)

    for idx in range(n_fields, n_frows * n_cols):
        fig.add_subplot(outer_gs[idx // n_cols, idx % n_cols]).axis('off')

    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.97)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_gaia_g_mag(df, output_dir):
    _plot_fields(
        df, 'phot_g_mean_mag', 'Gaia G',
        f'Gaia G Distribution & Class Fraction by Field (Gaia-matched, {CLASSIFIER_LABEL})',
        output_dir / 'gaia_fields_g_mag.png',
    )


def plot_gaia_e_mag(df, output_dir):
    _plot_fields(
        df, 'v_mag_auto', 'Euclid VIS (v_mag_auto)',
        f'Euclid VIS Distribution & Class Fraction by Field (Gaia-matched, {CLASSIFIER_LABEL})',
        output_dir / 'gaia_fields_e_mag.png',
    )


def plot_ero_all_e_mag(df, output_dir):
    df_assigned = df[df['field'].notna()].copy()
    print(f"\nAll ERO sources assigned to fields: {len(df_assigned):,}")
    field_counts = df_assigned['field'].value_counts()
    print("Sources per field:")
    for field in ERO_FIELDS.keys():
        print(f"  {field}: {field_counts.get(field, 0):,}")
    _plot_fields(
        df, 'v_mag_auto', 'Euclid VIS (v_mag_auto)',
        f'Euclid VIS Distribution & Class Fraction by Field (All ERO, {CLASSIFIER_LABEL})',
        output_dir / 'ero_all_fields_e_mag.png',
        min_frac_count=10,
    )


# ─── Combined summary plot ─────────────────────────────────────────────────────

def plot_combined_summary(df_gaia, df_ero, output_dir):
    df_gaia_f = df_gaia[df_gaia['field'].notna()].copy()
    df_ero_f  = df_ero[df_ero['field'].notna()].copy()

    fig, axes = plt.subplots(2, 3, figsize=(21, 10))
    bins        = np.arange(10, 29.25, 0.25)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    configs = [
        (df_ero_f,  'v_mag_auto',      'Euclid VIS (v_mag_auto)', 'All ERO, Euclid VIS'),
        (df_gaia_f, 'v_mag_auto',      'Euclid VIS (v_mag_auto)', 'Gaia-matched, Euclid VIS'),
        (df_gaia_f, 'phot_g_mean_mag', 'Gaia G',                  'Gaia-matched, Gaia G'),
    ]

    for col_idx, (df, mag_col, xlabel, title_suffix) in enumerate(configs):
        mag_all = df[mag_col].dropna()
        n_all   = int(np.histogram(mag_all, bins=bins)[0].sum())

        # ── count row ──
        ax = axes[0, col_idx]
        ax.hist(mag_all, bins=bins, alpha=0.3, color='gray', label=f'All ({n_all:,})')
        for cls in [1, 2, 3, 0]:
            sub = df[df['modest_class'] == cls][mag_col].dropna()
            n   = int(np.histogram(sub, bins=bins)[0].sum())
            ax.hist(sub, bins=bins, histtype='step', linewidth=1.5,
                    color=CLASS_COLORS[cls], label=f'{CLASS_LABELS[cls]} ({n:,})')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_xlim(10, 29)
        ax.set_title(f'Source Count ({title_suffix})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
        ax.grid(alpha=0.3, axis='y')

        # ── fraction row ──
        ax = axes[1, col_idx]
        counts_all, _ = np.histogram(mag_all, bins=bins)
        mask = counts_all >= 10
        for cls in [1, 2, 3, 0]:
            sub = df[df['modest_class'] == cls][mag_col].dropna()
            counts_cls, _ = np.histogram(sub, bins=bins)
            with np.errstate(divide='ignore', invalid='ignore'):
                frac = np.where(counts_all > 0, counts_cls / counts_all, np.nan)
            lw    = 1.5 if cls in (1, 2) else 1.0
            alpha = 1.0 if cls in (1, 2) else 0.6
            fmt   = 'o-' if cls == 1 else '-'
            ms    = 3    if cls == 1 else 0
            ax.plot(bin_centers[mask], frac[mask],
                    fmt, color=CLASS_COLORS[cls], linewidth=lw,
                    alpha=alpha, markersize=ms, label=CLASS_LABELS[cls])
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Class fraction', fontsize=10)
        ax.set_xlim(10, 29)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'Class Fraction ({title_suffix})', fontsize=11, fontweight='bold')
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.legend(fontsize=8, loc='center right')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.grid(alpha=0.3)

    plt.suptitle(
        f'All ERO Fields Combined: Magnitude Distributions & Class Fraction\n'
        f'({CLASSIFIER_LABEL})',
        fontsize=13, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    out = output_dir / 'combined_summary.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─── Main ──────────────────────────────────────────────────────────────────────

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
