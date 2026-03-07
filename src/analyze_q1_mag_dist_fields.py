#!/usr/bin/env python3
"""
Plot magnitude distributions and galaxy fractions for each Q1 field.

Combines Gaia-matched Q1 sources and all Q1 sources into a single script.
Uses field center coordinates to assign sources to their respective fields.

Produces (saved to output/q1_mag_dist/):
  1. gaia_fields_g_mag_count.png    — Gaia G count per field (Gaia-matched)
  2. gaia_fields_g_mag_fraction.png — galaxy fraction vs Gaia G per field (Gaia-matched)
  3. gaia_fields_e_mag_count.png    — mAB count per field (Gaia-matched)
  4. gaia_fields_e_mag_fraction.png — galaxy fraction vs mAB per field (Gaia-matched)
  5. q1_all_fields_e_mag_count.png  — mAB count per field (all Q1)
  6. q1_all_fields_e_mag_fraction.png — galaxy fraction vs mAB per field (all Q1)
  7. combined_summary.png           — 2×3 summary across all fields

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_q1_mag_dist_fields.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    """Assign each source to its nearest Q1 field (within field radius)."""
    ra  = df['ra'].values
    dec = df['dec'].values
    n_sources = len(df)

    field_assignments = np.full(n_sources, None, dtype=object)
    min_distances     = np.full(n_sources, np.inf)

    for field_name, (field_ra, field_dec, area) in Q1_FIELDS.items():
        radius = np.sqrt(area / np.pi)
        ra1  = np.radians(ra);   dec1 = np.radians(dec)
        ra2  = np.radians(field_ra); dec2 = np.radians(field_dec)
        a    = np.sin((dec2 - dec1) / 2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2)**2
        dist = np.degrees(2 * np.arcsin(np.sqrt(a)))
        mask = (dist < radius) & (dist < min_distances)
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
    """Query Gaia-matched Q1+PHZ sources with mAB, Gaia G, RA/Dec, and label."""
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
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE m_flux_detection_total > 0
          AND m_flux_detection_total IS NOT NULL
          AND m_vis_det = 1
    """
    print("Querying Gaia-matched Q1 data...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Retrieved {len(df):,} Gaia-matched sources")
    return df


def query_all_q1(username):
    """Query all Q1+PHZ sources (VIS-detected) with mAB, RA/Dec, and label."""
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
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz
        WHERE m_flux_detection_total > 0
          AND m_flux_detection_total IS NOT NULL
          AND m_vis_det = 1
    """
    print("Querying all Q1 data...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Retrieved {len(df):,} Q1 sources")
    return df


# ─── Gaia-matched plots ───────────────────────────────────────────────────────

def plot_gaia_g_mag_count(df, output_dir):
    """Plot Gaia G magnitude distribution for each Q1 field (Gaia-matched)."""
    df_assigned = df[df['field'].notna()].copy()
    print(f"\nGaia-matched sources assigned to fields: {len(df_assigned):,}")

    n_fields = len(Q1_FIELDS)
    fig, axes = plt.subplots(1, n_fields, figsize=(7 * n_fields, 5))
    axes = axes.flatten()

    bins = np.arange(14, 22.5, 0.25)

    for idx, field_name in enumerate(Q1_FIELDS.keys()):
        ax = axes[idx]
        fd       = df_assigned[df_assigned['field'] == field_name]
        fd_star  = fd[fd['label'] == 'STAR']['phot_g_mean_mag'].dropna()
        fd_gal   = fd[fd['label'] == 'GALAXY']['phot_g_mean_mag'].dropna()
        fd_unc   = fd[fd['label'] == 'UNCLASSIFIED']['phot_g_mean_mag'].dropna()
        fd_all   = fd['phot_g_mean_mag'].dropna()
        gal_b    = get_galactic_b(*Q1_FIELDS[field_name][:2])

        ax.hist(fd_all,  bins=bins, alpha=0.3, color='gray',
                label=f'All ({len(fd_all):,})')
        ax.hist(fd_gal,  bins=bins, alpha=0.7, color='coral',
                label=f'Galaxies ({len(fd_gal):,})', histtype='step', linewidth=1.5)
        ax.hist(fd_star, bins=bins, alpha=0.7, color='steelblue',
                label=f'Stars ({len(fd_star):,})', histtype='step', linewidth=1.5)
        ax.hist(fd_unc,  bins=bins, alpha=0.7, color='forestgreen',
                label=f'Unclassified ({len(fd_unc):,})', histtype='step', linewidth=1.5)

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
        ax.set_xlabel('Gaia G', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(14, 22)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(alpha=0.3, axis='y')

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Sources Count vs Gaia G by Field (Gaia-matched, Q1)',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = output_dir / 'gaia_fields_g_mag_count.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def plot_gaia_g_mag_fraction(df, output_dir):
    """Plot galaxy fraction vs Gaia G for each Q1 field (Gaia-matched)."""
    df_assigned = df[df['field'].notna()].copy()

    n_fields = len(Q1_FIELDS)
    fig, axes = plt.subplots(1, n_fields, figsize=(7 * n_fields, 5))
    axes = axes.flatten()

    mag_bins    = np.arange(14, 22.5, 0.25)
    mag_centers = (mag_bins[:-1] + mag_bins[1:]) / 2

    for idx, field_name in enumerate(Q1_FIELDS.keys()):
        ax         = axes[idx]
        field_data = df_assigned[df_assigned['field'] == field_name].copy()
        gal_b      = get_galactic_b(*Q1_FIELDS[field_name][:2])
        color      = get_field_color(field_name)

        if len(field_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(field_name, fontsize=10, fontweight='bold')
            ax.set_xlim(14, 22); ax.set_ylim(-0.02, 1.02)
            continue

        gmag     = field_data['phot_g_mean_mag'].dropna()
        gmag_gal = field_data[field_data['label'] == 'GALAXY']['phot_g_mean_mag'].dropna()

        counts_all, _ = np.histogram(gmag,     bins=mag_bins)
        counts_gal, _ = np.histogram(gmag_gal, bins=mag_bins)

        with np.errstate(divide='ignore', invalid='ignore'):
            frac = np.where(counts_all > 0, counts_gal / counts_all, np.nan)

        mask = counts_all >= 5
        ax.plot(mag_centers[mask], frac[mask], 'o-', color=color, markersize=3, linewidth=1.5)

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
        ax.set_xlabel('Gaia G', fontsize=9)
        ax.set_ylabel('Galaxy fraction', fontsize=9)
        ax.set_xlim(14, 22); ax.set_ylim(-0.02, 1.02)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.grid(alpha=0.3)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Galaxy Fraction vs Gaia G by Field (Gaia-matched, Q1)',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = output_dir / 'gaia_fields_g_mag_fraction.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def plot_gaia_e_mag_count(df, output_dir):
    """Plot mAB distribution for each Q1 field (Gaia-matched)."""
    df_assigned = df[df['field'].notna() & df['phot_g_mean_mag'].notna()].copy()

    n_fields = len(Q1_FIELDS)
    fig, axes = plt.subplots(1, n_fields, figsize=(7 * n_fields, 5))
    axes = axes.flatten()

    bins = np.arange(14, 28.5, 0.25)

    for idx, field_name in enumerate(Q1_FIELDS.keys()):
        ax = axes[idx]
        fd      = df_assigned[df_assigned['field'] == field_name]
        fd_star = fd[fd['label'] == 'STAR']['ie_mag'].dropna()
        fd_gal  = fd[fd['label'] == 'GALAXY']['ie_mag'].dropna()
        fd_unc  = fd[fd['label'] == 'UNCLASSIFIED']['ie_mag'].dropna()
        fd_all  = fd['ie_mag'].dropna()
        gal_b   = get_galactic_b(*Q1_FIELDS[field_name][:2])

        ax.hist(fd_all,  bins=bins, alpha=0.3, color='gray',
                label=f'All ({len(fd_all):,})')
        ax.hist(fd_gal,  bins=bins, alpha=0.7, color='coral',
                label=f'Galaxies ({len(fd_gal):,})', histtype='step', linewidth=1.5)
        ax.hist(fd_star, bins=bins, alpha=0.7, color='steelblue',
                label=f'Stars ({len(fd_star):,})', histtype='step', linewidth=1.5)
        ax.hist(fd_unc,  bins=bins, alpha=0.7, color='forestgreen',
                label=f'Unclassified ({len(fd_unc):,})', histtype='step', linewidth=1.5)

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
        ax.set_xlabel('mAB', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(14, 28)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(alpha=0.3, axis='y')

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Sources Count vs mAB by Field (Gaia-matched, Q1)',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = output_dir / 'gaia_fields_e_mag_count.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def plot_gaia_e_mag_fraction(df, output_dir):
    """Plot galaxy fraction vs mAB for each Q1 field (Gaia-matched)."""
    df_assigned = df[df['field'].notna() & df['phot_g_mean_mag'].notna()].copy()

    n_fields = len(Q1_FIELDS)
    fig, axes = plt.subplots(1, n_fields, figsize=(7 * n_fields, 5))
    axes = axes.flatten()

    mag_bins    = np.arange(14, 28.5, 0.25)
    mag_centers = (mag_bins[:-1] + mag_bins[1:]) / 2

    for idx, field_name in enumerate(Q1_FIELDS.keys()):
        ax         = axes[idx]
        field_data = df_assigned[df_assigned['field'] == field_name].copy()
        gal_b      = get_galactic_b(*Q1_FIELDS[field_name][:2])
        color      = get_field_color(field_name)

        if len(field_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(field_name, fontsize=10, fontweight='bold')
            ax.set_xlim(14, 28); ax.set_ylim(-0.02, 1.02)
            continue

        vmag     = field_data['ie_mag'].dropna()
        vmag_gal = field_data[field_data['label'] == 'GALAXY']['ie_mag'].dropna()

        counts_all, _ = np.histogram(vmag,     bins=mag_bins)
        counts_gal, _ = np.histogram(vmag_gal, bins=mag_bins)

        with np.errstate(divide='ignore', invalid='ignore'):
            frac = np.where(counts_all > 0, counts_gal / counts_all, np.nan)

        mask = counts_all >= 5
        ax.plot(mag_centers[mask], frac[mask], 'o-', color=color, markersize=3, linewidth=1.5)

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
        ax.set_xlabel('mAB', fontsize=9)
        ax.set_ylabel('Galaxy fraction', fontsize=9)
        ax.set_xlim(14, 28); ax.set_ylim(-0.02, 1.02)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.grid(alpha=0.3)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Galaxy Fraction vs mAB by Field (Gaia-matched, Q1)',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = output_dir / 'gaia_fields_e_mag_fraction.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─── All Q1 plots ─────────────────────────────────────────────────────────────

def plot_q1_all_e_mag_count(df, output_dir):
    """Plot mAB distribution for each Q1 field (all Q1 sources)."""
    df_assigned = df[df['field'].notna()].copy()
    print(f"\nAll Q1 sources assigned to fields: {len(df_assigned):,}")
    print(f"Sources not in any field: {df['field'].isna().sum():,}")

    field_counts = df_assigned['field'].value_counts()
    print("\nSources per field:")
    for field in Q1_FIELDS.keys():
        print(f"  {field}: {field_counts.get(field, 0):,}")

    n_fields = len(Q1_FIELDS)
    fig, axes = plt.subplots(1, n_fields, figsize=(7 * n_fields, 5))
    axes = axes.flatten()

    bins = np.arange(14, 28.5, 0.25)

    for idx, field_name in enumerate(Q1_FIELDS.keys()):
        ax = axes[idx]
        fd      = df_assigned[df_assigned['field'] == field_name]
        fd_star = fd[fd['label'] == 'STAR']['ie_mag'].dropna()
        fd_gal  = fd[fd['label'] == 'GALAXY']['ie_mag'].dropna()
        fd_unc  = fd[fd['label'] == 'UNCLASSIFIED']['ie_mag'].dropna()
        fd_all  = fd['ie_mag'].dropna()
        gal_b   = get_galactic_b(*Q1_FIELDS[field_name][:2])

        ax.hist(fd_all,  bins=bins, alpha=0.3, color='gray',
                label=f'All ({len(fd_all):,})')
        ax.hist(fd_gal,  bins=bins, alpha=0.7, color='coral',
                label=f'Galaxies ({len(fd_gal):,})', histtype='step', linewidth=1.5)
        ax.hist(fd_star, bins=bins, alpha=0.7, color='steelblue',
                label=f'Stars ({len(fd_star):,})', histtype='step', linewidth=1.5)
        ax.hist(fd_unc,  bins=bins, alpha=0.7, color='forestgreen',
                label=f'Unclassified ({len(fd_unc):,})', histtype='step', linewidth=1.5)

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
        ax.set_xlabel('mAB', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(14, 28)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(alpha=0.3, axis='y')

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Sources Count vs mAB by Field (All Q1)',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = output_dir / 'q1_all_fields_e_mag_count.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def plot_q1_all_e_mag_fraction(df, output_dir):
    """Plot galaxy fraction vs mAB for each Q1 field (all Q1 sources)."""
    df_assigned = df[df['field'].notna()].copy()

    n_fields = len(Q1_FIELDS)
    fig, axes = plt.subplots(1, n_fields, figsize=(7 * n_fields, 5))
    axes = axes.flatten()

    bins        = np.arange(14, 28.5, 0.25)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for idx, field_name in enumerate(Q1_FIELDS.keys()):
        ax    = axes[idx]
        fd    = df_assigned[df_assigned['field'] == field_name]
        fd_gal = fd[fd['label'] == 'GALAXY']['ie_mag'].dropna()
        fd_all = fd['ie_mag'].dropna()
        gal_b  = get_galactic_b(*Q1_FIELDS[field_name][:2])
        color  = get_field_color(field_name)

        counts_all, _ = np.histogram(fd_all, bins=bins)
        counts_gal, _ = np.histogram(fd_gal, bins=bins)

        with np.errstate(divide='ignore', invalid='ignore'):
            frac = np.where(counts_all > 0, counts_gal / counts_all, np.nan)

        mask = counts_all >= 10
        ax.plot(bin_centers[mask], frac[mask], 'o-', color=color, markersize=3, linewidth=1.5)

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
        ax.set_xlabel('mAB', fontsize=9)
        ax.set_ylabel('Galaxy fraction', fontsize=9)
        ax.set_xlim(14, 28); ax.set_ylim(-0.02, 1.02)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.grid(alpha=0.3)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Galaxy Fraction vs mAB by Field (All Q1)',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = output_dir / 'q1_all_fields_e_mag_fraction.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ─── Combined summary plot ────────────────────────────────────────────────────

def plot_combined_summary(df_gaia, df_q1, output_dir):
    """2×3 summary: all Q1 fields combined into single histograms and fraction curves.

    Row 1: count histograms (All Q1 mAB, Gaia-matched mAB, Gaia-matched G)
    Row 2: galaxy fraction  (All Q1 mAB, Gaia-matched mAB, Gaia-matched G)
    """
    df_gaia_f = df_gaia[df_gaia['field'].notna()].copy()
    df_q1_f   = df_q1[df_q1['field'].notna()].copy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    configs = [
        (df_q1_f,   'ie_mag',        np.arange(14, 28.5, 0.25),
         'mAB', (14, 28), 'All Q1, mAB'),
        (df_gaia_f, 'ie_mag',        np.arange(14, 28.5, 0.25),
         'mAB', (14, 28), 'Gaia-matched, mAB'),
        (df_gaia_f, 'phot_g_mean_mag', np.arange(14, 22.5, 0.25),
         'Gaia G', (14, 22), 'Gaia-matched, Gaia G'),
    ]

    for col_idx, (df, mag_col, bins, xlabel, xlim, title_suffix) in enumerate(configs):
        mag_all  = df[mag_col].dropna()
        mag_star = df[df['label'] == 'STAR'][mag_col].dropna()
        mag_gal  = df[df['label'] == 'GALAXY'][mag_col].dropna()
        mag_unc  = df[df['label'] == 'UNCLASSIFIED'][mag_col].dropna()

        # Row 1: count histogram
        ax = axes[0, col_idx]
        ax.hist(mag_all,  bins=bins, alpha=0.3, color='gray',
                label=f'All ({len(mag_all):,})')
        ax.hist(mag_gal,  bins=bins, alpha=0.7, color='coral',
                label=f'Galaxies ({len(mag_gal):,})', histtype='step', linewidth=1.5)
        ax.hist(mag_star, bins=bins, alpha=0.7, color='steelblue',
                label=f'Stars ({len(mag_star):,})', histtype='step', linewidth=1.5)
        ax.hist(mag_unc,  bins=bins, alpha=0.7, color='forestgreen',
                label=f'Unclassified ({len(mag_unc):,})', histtype='step', linewidth=1.5)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_xlim(xlim)
        ax.set_title(f'Sources Count ({title_suffix})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(alpha=0.3, axis='y')

        # Row 2: galaxy fraction
        ax = axes[1, col_idx]
        bin_centers = (bins[:-1] + bins[1:]) / 2
        counts_all, _ = np.histogram(mag_all, bins=bins)
        counts_gal, _ = np.histogram(mag_gal, bins=bins)

        with np.errstate(divide='ignore', invalid='ignore'):
            frac = np.where(counts_all > 0, counts_gal / counts_all, np.nan)

        mask = counts_all >= 10
        ax.plot(bin_centers[mask], frac[mask], 'o-', color='steelblue',
                markersize=3, linewidth=1.5)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Galaxy fraction', fontsize=10)
        ax.set_xlim(xlim); ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'Galaxy Fraction ({title_suffix})', fontsize=11, fontweight='bold')
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.grid(alpha=0.3)

    plt.suptitle('All Q1 Fields Combined: Magnitude Distributions & Galaxy Fraction',
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
    plot_gaia_g_mag_count(df_gaia, OUTPUT_DIR)
    plot_gaia_g_mag_fraction(df_gaia, OUTPUT_DIR)

    print("\n--- Gaia-matched: mAB magnitude ---")
    plot_gaia_e_mag_count(df_gaia, OUTPUT_DIR)
    plot_gaia_e_mag_fraction(df_gaia, OUTPUT_DIR)

    df_q1 = query_all_q1(username)
    print("\nAssigning all Q1 sources to fields...")
    df_q1 = assign_fields(df_q1)

    print("\n--- All Q1: mAB magnitude ---")
    plot_q1_all_e_mag_count(df_q1, OUTPUT_DIR)
    plot_q1_all_e_mag_fraction(df_q1, OUTPUT_DIR)

    print("\n--- Combined summary (all fields) ---")
    plot_combined_summary(df_gaia, df_q1, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("Done!")
    print(f"All results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
