#!/usr/bin/env python3
"""
Plot magnitude distributions and galaxy fractions for each Q1 field.

Combines Gaia-matched Q1 sources and all Q1 sources into a single script.
Uses field center coordinates to assign sources to their respective fields.

Produces (saved to output/q1_mag_dist/):
  1. gaia_fields_g_mag.png   — 2×3: Gaia G count (top) + galaxy fraction (bottom) per field (Gaia-matched)
  2. gaia_fields_e_mag.png   — 2×3: I_E count (top) + galaxy fraction (bottom) per field (Gaia-matched)
  3. q1_all_fields_e_mag.png — 2×3: I_E count (top) + galaxy fraction (bottom) per field (all Q1)
  4. combined_summary.png    — 2×3 summary across all fields combined

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_q1_mag_dist_fields.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'output' / 'q1_mag_dist'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

ZPC_IE = 1.05
ZP_REF = 23.9

# Q1 field definitions: name → (RA deg, Dec deg, area deg²)
Q1_FIELDS = {
    'EDF-North':  (269.733, +66.018, 22.9),
    'EDF-South':  ( 61.241, -48.423, 28.1),
    'EDF-Fornax': ( 52.932, -28.088, 12.1),
}

FIELD_CATEGORIES = {
    'Q1': ['EDF-North', 'EDF-South', 'EDF-Fornax'],
}

CATEGORY_COLORS = {
    'Q1': plt.cm.Blues,
}


# ─── Shared helpers ───────────────────────────────────────────────────────────

def assign_fields(df):
    """Assign each source to its nearest Q1 field (no distance cutoff).

    All sources in crossmatch_q1_mer_phz are legitimate Q1 observations;
    the circular-radius cutoff was an approximation that dropped real tile-edge
    sources.  The three EDF fields are widely separated on-sky so nearest-field
    assignment is unambiguous.
    """
    ra  = df['ra'].values
    dec = df['dec'].values
    n_sources = len(df)

    field_assignments = np.full(n_sources, None, dtype=object)
    min_distances     = np.full(n_sources, np.inf)

    for field_name, (field_ra, field_dec, _area) in Q1_FIELDS.items():
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


def get_field_color(field_name):
    """Get color for a field based on its category."""
    for category, fields in FIELD_CATEGORIES.items():
        if field_name in fields:
            cmap = CATEGORY_COLORS[category]
            idx  = fields.index(field_name)
            return cmap(0.4 + 0.15 * idx)
    return 'gray'


def get_galactic_b(ra, dec):
    """Return galactic latitude b in degrees."""
    from astropy.coordinates import SkyCoord
    c = SkyCoord(ra=ra, dec=dec, unit='deg')
    return c.galactic.b.deg


# ─── Data queries ─────────────────────────────────────────────────────────────

def query_gaia_matched(username):
    """Query Gaia-matched Q1+PHZ sources with I_E, Gaia G, RA/Dec, and label."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = f"""
        SELECT
            m_right_ascension AS ra,
            m_declination     AS dec,
            -2.5 * LOG10({ZPC_IE} * m_flux_detection_total) + {ZP_REF} AS ie_mag,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            CASE
                WHEN p_phz_classification = 1 THEN 'STAR'
                WHEN p_phz_classification = 2 THEN 'GALAXY'
                ELSE 'UNCLASSIFIED'
            END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia
        WHERE m_flux_detection_total > 0
          AND m_flux_detection_total IS NOT NULL
    """
    print("Querying Gaia-matched Q1 data...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Retrieved {len(df):,} Gaia-matched sources")
    return df


def query_all_q1(username):
    """Query all Q1+PHZ sources (VIS-detected) with I_E, RA/Dec, and label."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = f"""
        SELECT
            m_right_ascension AS ra,
            m_declination     AS dec,
            -2.5 * LOG10({ZPC_IE} * m_flux_detection_total) + {ZP_REF} AS ie_mag,
            CASE
                WHEN p_phz_classification = 1 THEN 'STAR'
                WHEN p_phz_classification = 2 THEN 'GALAXY'
                ELSE 'UNCLASSIFIED'
            END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis
        WHERE m_flux_detection_total > 0
          AND m_flux_detection_total IS NOT NULL
    """
    print("Querying all Q1 data...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Retrieved {len(df):,} Q1 sources")
    return df


# ─── Gaia-matched plots ───────────────────────────────────────────────────────

def plot_gaia_g_mag(df, output_dir):
    """2-row plot: Gaia G count (top) and galaxy fraction (bottom) per Q1 field."""
    df_assigned = df[df['field'].notna()].copy()
    print(f"\nGaia-matched sources assigned to fields: {len(df_assigned):,}")

    n_fields    = len(Q1_FIELDS)
    bins        = np.arange(10, 29.25, 0.25)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    fig, axes   = plt.subplots(2, n_fields, figsize=(7 * n_fields, 10))

    for idx, field_name in enumerate(Q1_FIELDS.keys()):
        fd      = df_assigned[df_assigned['field'] == field_name]
        fd_star = fd[fd['label'] == 'STAR']['phot_g_mean_mag'].dropna()
        fd_gal  = fd[fd['label'] == 'GALAXY']['phot_g_mean_mag'].dropna()
        fd_unc  = fd[fd['label'] == 'UNCLASSIFIED']['phot_g_mean_mag'].dropna()
        fd_all  = fd['phot_g_mean_mag'].dropna()
        gal_b   = get_galactic_b(*Q1_FIELDS[field_name][:2])
        color   = get_field_color(field_name)

        # Row 0: counts
        n_gal  = int(np.histogram(fd_gal,  bins=bins)[0].sum())
        n_star = int(np.histogram(fd_star, bins=bins)[0].sum())
        n_unc  = int(np.histogram(fd_unc,  bins=bins)[0].sum())
        n_all  = n_star + n_gal + n_unc
        ax = axes[0, idx]
        ax.hist(fd_all,  bins=bins, alpha=0.3, color='gray',
                label=f'All ({n_all:,})')
        ax.hist(fd_gal,  bins=bins, alpha=0.7, color='steelblue',
                label=f'Galaxies ({n_gal:,})', histtype='step', linewidth=1.5)
        ax.hist(fd_star, bins=bins, alpha=0.7, color='coral',
                label=f'Stars ({n_star:,})', histtype='step', linewidth=1.5)
        ax.hist(fd_unc,  bins=bins, alpha=0.7, color='forestgreen',
                label=f'Unclassified ({n_unc:,})', histtype='step', linewidth=1.5)
        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
        ax.set_xlabel('Gaia G', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(10, 29)
        ax.legend(fontsize=7, loc='upper left')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.grid(alpha=0.3, axis='y')

        # Row 1: galaxy fraction
        ax = axes[1, idx]
        if len(fd_all) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        else:
            counts_all,  _ = np.histogram(fd_all,  bins=bins)
            counts_gal,  _ = np.histogram(fd_gal,  bins=bins)
            counts_star, _ = np.histogram(fd_star, bins=bins)
            counts_unc,  _ = np.histogram(fd_unc,  bins=bins)
            with np.errstate(divide='ignore', invalid='ignore'):
                frac_gal  = np.where(counts_all > 0, counts_gal  / counts_all, np.nan)
                frac_star = np.where(counts_all > 0, counts_star / counts_all, np.nan)
                frac_unc  = np.where(counts_all > 0, counts_unc  / counts_all, np.nan)
            mask = counts_all >= 5
            ax.plot(bin_centers[mask], frac_star[mask], color='coral',
                    markersize=3, linewidth=1.5, alpha=0.3)
            ax.plot(bin_centers[mask], frac_unc[mask], color='forestgreen',
                    markersize=3, linewidth=1.5, alpha=0.3)
            ax.plot(bin_centers[mask], frac_gal[mask],  'o-', color='steelblue',
                    markersize=3, linewidth=1.5)
        ax.set_xlabel('Gaia G', fontsize=9)
        ax.set_ylabel('Class fraction', fontsize=9)
        ax.set_xlim(10, 29); ax.set_ylim(-0.02, 1.02)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.grid(alpha=0.3)

    plt.suptitle('Gaia G Distribution & Galaxy Fraction by Field (Gaia-matched, Q1)',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = output_dir / 'gaia_fields_g_mag.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def plot_gaia_e_mag(df, output_dir):
    """2-row plot: I_E count (top) and galaxy fraction (bottom) per Q1 field (Gaia-matched)."""
    df_assigned = df[df['field'].notna() & df['phot_g_mean_mag'].notna()].copy()

    n_fields    = len(Q1_FIELDS)
    bins        = np.arange(10, 29.25, 0.25)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    fig, axes   = plt.subplots(2, n_fields, figsize=(7 * n_fields, 10))

    for idx, field_name in enumerate(Q1_FIELDS.keys()):
        fd      = df_assigned[df_assigned['field'] == field_name]
        fd_star = fd[fd['label'] == 'STAR']['ie_mag'].dropna()
        fd_gal  = fd[fd['label'] == 'GALAXY']['ie_mag'].dropna()
        fd_unc  = fd[fd['label'] == 'UNCLASSIFIED']['ie_mag'].dropna()
        fd_all  = fd['ie_mag'].dropna()
        gal_b   = get_galactic_b(*Q1_FIELDS[field_name][:2])
        color   = get_field_color(field_name)

        # Row 0: counts
        n_gal  = int(np.histogram(fd_gal,  bins=bins)[0].sum())
        n_star = int(np.histogram(fd_star, bins=bins)[0].sum())
        n_unc  = int(np.histogram(fd_unc,  bins=bins)[0].sum())
        n_all  = n_star + n_gal + n_unc
        ax = axes[0, idx]
        ax.hist(fd_all,  bins=bins, alpha=0.3, color='gray',
                label=f'All ({n_all:,})')
        ax.hist(fd_gal,  bins=bins, alpha=0.7, color='steelblue',
                label=f'Galaxies ({n_gal:,})', histtype='step', linewidth=1.5)
        ax.hist(fd_star, bins=bins, alpha=0.7, color='coral',
                label=f'Stars ({n_star:,})', histtype='step', linewidth=1.5)
        ax.hist(fd_unc,  bins=bins, alpha=0.7, color='forestgreen',
                label=f'Unclassified ({n_unc:,})', histtype='step', linewidth=1.5)
        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
        ax.set_xlabel('I_E', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(10, 29)
        ax.legend(fontsize=7, loc='upper left')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.grid(alpha=0.3, axis='y')

        # Row 1: galaxy fraction
        ax = axes[1, idx]
        if len(fd_all) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        else:
            counts_all,  _ = np.histogram(fd_all,  bins=bins)
            counts_gal,  _ = np.histogram(fd_gal,  bins=bins)
            counts_star, _ = np.histogram(fd_star, bins=bins)
            counts_unc,  _ = np.histogram(fd_unc,  bins=bins)
            with np.errstate(divide='ignore', invalid='ignore'):
                frac_gal  = np.where(counts_all > 0, counts_gal  / counts_all, np.nan)
                frac_star = np.where(counts_all > 0, counts_star / counts_all, np.nan)
                frac_unc  = np.where(counts_all > 0, counts_unc  / counts_all, np.nan)
            mask = counts_all >= 5
            ax.plot(bin_centers[mask], frac_star[mask], color='coral',
                    markersize=3, linewidth=1.5, alpha=0.3)
            ax.plot(bin_centers[mask], frac_unc[mask], color='forestgreen',
                    markersize=3, linewidth=1.5, alpha=0.3)
            ax.plot(bin_centers[mask], frac_gal[mask],  'o-', color='steelblue',
                    markersize=3, linewidth=1.5)
        ax.set_xlabel('I_E', fontsize=9)
        ax.set_ylabel('Class fraction', fontsize=9)
        ax.set_xlim(10, 29); ax.set_ylim(-0.02, 1.02)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.grid(alpha=0.3)

    plt.suptitle('I_E Distribution & Galaxy Fraction by Field (Gaia-matched, Q1)',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = output_dir / 'gaia_fields_e_mag.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─── All Q1 plots ─────────────────────────────────────────────────────────────

def plot_q1_all_e_mag(df, output_dir):
    """2-row plot: I_E count (top) and galaxy fraction (bottom) per Q1 field (all Q1)."""
    df_assigned = df[df['field'].notna()].copy()
    print(f"\nAll Q1 sources assigned to fields: {len(df_assigned):,}")
    field_counts = df_assigned['field'].value_counts()
    print("Sources per field:")
    for field in Q1_FIELDS.keys():
        print(f"  {field}: {field_counts.get(field, 0):,}")

    n_fields    = len(Q1_FIELDS)
    bins        = np.arange(10, 29.25, 0.25)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    fig, axes   = plt.subplots(2, n_fields, figsize=(7 * n_fields, 10))

    for idx, field_name in enumerate(Q1_FIELDS.keys()):
        fd      = df_assigned[df_assigned['field'] == field_name]
        fd_star = fd[fd['label'] == 'STAR']['ie_mag'].dropna()
        fd_gal  = fd[fd['label'] == 'GALAXY']['ie_mag'].dropna()
        fd_unc  = fd[fd['label'] == 'UNCLASSIFIED']['ie_mag'].dropna()
        fd_all  = fd['ie_mag'].dropna()
        gal_b   = get_galactic_b(*Q1_FIELDS[field_name][:2])
        color   = get_field_color(field_name)

        # Row 0: counts
        n_gal  = int(np.histogram(fd_gal,  bins=bins)[0].sum())
        n_star = int(np.histogram(fd_star, bins=bins)[0].sum())
        n_unc  = int(np.histogram(fd_unc,  bins=bins)[0].sum())
        n_all  = n_star + n_gal + n_unc
        ax = axes[0, idx]
        ax.hist(fd_all,  bins=bins, alpha=0.3, color='gray',
                label=f'All ({n_all:,})')
        ax.hist(fd_gal,  bins=bins, alpha=0.7, color='steelblue',
                label=f'Galaxies ({n_gal:,})', histtype='step', linewidth=1.5)
        ax.hist(fd_star, bins=bins, alpha=0.7, color='coral',
                label=f'Stars ({n_star:,})', histtype='step', linewidth=1.5)
        ax.hist(fd_unc,  bins=bins, alpha=0.7, color='forestgreen',
                label=f'Unclassified ({n_unc:,})', histtype='step', linewidth=1.5)
        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
        ax.set_xlabel('I_E', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(10, 29)
        ax.legend(fontsize=7, loc='upper left')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.grid(alpha=0.3, axis='y')

        # Row 1: galaxy fraction
        ax = axes[1, idx]
        if len(fd_all) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        else:
            counts_all,  _ = np.histogram(fd_all,  bins=bins)
            counts_gal,  _ = np.histogram(fd_gal,  bins=bins)
            counts_star, _ = np.histogram(fd_star, bins=bins)
            counts_unc,  _ = np.histogram(fd_unc,  bins=bins)
            with np.errstate(divide='ignore', invalid='ignore'):
                frac_gal  = np.where(counts_all > 0, counts_gal  / counts_all, np.nan)
                frac_star = np.where(counts_all > 0, counts_star / counts_all, np.nan)
                frac_unc  = np.where(counts_all > 0, counts_unc  / counts_all, np.nan)
            mask = counts_all >= 10
            ax.plot(bin_centers[mask], frac_star[mask], color='coral',
                    markersize=3, linewidth=1.5, alpha=0.3)
            ax.plot(bin_centers[mask], frac_unc[mask], color='forestgreen',
                    markersize=3, linewidth=1.5, alpha=0.3)
            ax.plot(bin_centers[mask], frac_gal[mask],  'o-', color='steelblue',
                    markersize=3, linewidth=1.5)
        ax.set_xlabel('I_E', fontsize=9)
        ax.set_ylabel('Class fraction', fontsize=9)
        ax.set_xlim(10, 29); ax.set_ylim(-0.02, 1.02)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.grid(alpha=0.3)

    plt.suptitle('I_E Distribution & Galaxy Fraction by Field (All Q1)',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = output_dir / 'q1_all_fields_e_mag.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─── Combined summary plot ────────────────────────────────────────────────────

def plot_combined_summary(df_gaia, df_q1, output_dir):
    """2×3 summary: all Q1 fields combined into single histograms and fraction curves.

    Row 1: count histograms (All Q1 I_E, Gaia-matched I_E, Gaia-matched G)
    Row 2: galaxy fraction  (All Q1 I_E, Gaia-matched I_E, Gaia-matched G)
    """
    df_gaia_f = df_gaia[df_gaia['field'].notna()].copy()
    df_q1_f   = df_q1[df_q1['field'].notna()].copy()

    fig, axes = plt.subplots(2, 3, figsize=(21, 10))

    ie_bins = np.arange(10, 29.25, 0.25)
    configs = [
        (df_q1_f,   'ie_mag',          ie_bins,
         'I_E', (10, 29), 'All Q1, I_E'),
        (df_gaia_f, 'ie_mag',          ie_bins,
         'I_E', (10, 29), 'Gaia-matched, I_E'),
        (df_gaia_f, 'phot_g_mean_mag', np.arange(10, 29.25, 0.25),
         'Gaia G', (10, 29), 'Gaia-matched, Gaia G'),
    ]

    for col_idx, (df, mag_col, bins, xlabel, xlim, title_suffix) in enumerate(configs):
        mag_all  = df[mag_col].dropna()
        mag_star = df[df['label'] == 'STAR'][mag_col].dropna()
        mag_gal  = df[df['label'] == 'GALAXY'][mag_col].dropna()
        mag_unc  = df[df['label'] == 'UNCLASSIFIED'][mag_col].dropna()

        n_gal  = int(np.histogram(mag_gal,  bins=bins)[0].sum())
        n_star = int(np.histogram(mag_star, bins=bins)[0].sum())
        n_unc  = int(np.histogram(mag_unc,  bins=bins)[0].sum())
        n_all  = n_star + n_gal + n_unc

        # Row 1: count histogram
        ax = axes[0, col_idx]
        ax.hist(mag_all,  bins=bins, alpha=0.3, color='gray',
                label=f'All ({n_all:,})')
        ax.hist(mag_gal,  bins=bins, alpha=0.7, color='steelblue',
                label=f'Galaxies ({n_gal:,})', histtype='step', linewidth=1.5)
        ax.hist(mag_star, bins=bins, alpha=0.7, color='coral',
                label=f'Stars ({n_star:,})', histtype='step', linewidth=1.5)
        ax.hist(mag_unc,  bins=bins, alpha=0.7, color='forestgreen',
                label=f'Unclassified ({n_unc:,})', histtype='step', linewidth=1.5)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_xlim(xlim)
        ax.set_title(f'Sources Count ({title_suffix})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
        ax.grid(alpha=0.3, axis='y')

        # Row 2: class fractions
        ax = axes[1, col_idx]
        bin_centers = (bins[:-1] + bins[1:]) / 2
        counts_all,  _ = np.histogram(mag_all,  bins=bins)
        counts_gal,  _ = np.histogram(mag_gal,  bins=bins)
        counts_star, _ = np.histogram(mag_star, bins=bins)
        counts_unc,  _ = np.histogram(mag_unc,  bins=bins)

        with np.errstate(divide='ignore', invalid='ignore'):
            frac_gal  = np.where(counts_all > 0, counts_gal  / counts_all, np.nan)
            frac_star = np.where(counts_all > 0, counts_star / counts_all, np.nan)
            frac_unc  = np.where(counts_all > 0, counts_unc  / counts_all, np.nan)

        mask = counts_all >= 10
        ax.plot(bin_centers[mask], frac_star[mask], color='coral',
                markersize=3, linewidth=1.5, alpha=0.3)
        ax.plot(bin_centers[mask], frac_unc[mask], color='forestgreen',
                markersize=3, linewidth=1.5, alpha=0.3)
        ax.plot(bin_centers[mask], frac_gal[mask], 'o-', color='steelblue',
                markersize=3, linewidth=1.5)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Class fraction', fontsize=10)
        ax.set_xlim(xlim); ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'Class Fraction ({title_suffix})', fontsize=11, fontweight='bold')
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.grid(alpha=0.3)

    plt.suptitle('All Q1 Fields Combined: Magnitude Distributions & Class Fractions',
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
    print("Q1 Fields: Magnitude Distributions (Gaia-matched + All Q1)")
    print("=" * 70)

    df_gaia = query_gaia_matched(username)
    print("\nAssigning Gaia-matched sources to fields...")
    df_gaia = assign_fields(df_gaia)

    print("\n--- Gaia-matched: Gaia G magnitude ---")
    plot_gaia_g_mag(df_gaia, OUTPUT_DIR)

    print("\n--- Gaia-matched: I_E magnitude ---")
    plot_gaia_e_mag(df_gaia, OUTPUT_DIR)

    df_q1 = query_all_q1(username)
    print("\nAssigning all Q1 sources to fields...")
    df_q1 = assign_fields(df_q1)

    print("\n--- All Q1: I_E magnitude ---")
    plot_q1_all_e_mag(df_q1, OUTPUT_DIR)

    print("\n--- Combined summary (all fields) ---")
    plot_combined_summary(df_gaia, df_q1, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("Done!")
    print(f"All results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
