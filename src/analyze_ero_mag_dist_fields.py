#!/usr/bin/env python3
"""
Plot magnitude distributions and galaxy fractions for each ERO field.

Combines Gaia-matched ERO sources and all ERO sources into a single script.
Uses field center coordinates to assign sources to their respective fields.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output' / 'ero_mag_dist'
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

# ERO field definitions: name, RA (deg), Dec (deg), area (deg²)
ERO_FIELDS = {
    # Star-forming Regions
    'Horsehead': (85.150, -2.613, 0.58),
    'Messier 78': (86.690, -0.015, 0.60),
    'Taurus': (64.983, 28.023, 0.61),
    'Barnard 30': (82.880, 12.316, 0.60),
    # Globular Clusters
    'NGC 6254': (254.303, -4.100, 0.60),
    'NGC 6397': (265.174, -53.658, 0.61),
    # Nearby Galaxies
    'NGC 6822': (296.236, -14.788, 0.60),
    'Holmberg II': (124.790, 70.706, 0.60),
    'IC 10': (5.063, 59.288, 0.62),
    'IC 342': (56.730, 68.084, 0.59),
    'NGC 2403': (114.211, 65.586, 0.60),
    'NGC 6744': (287.430, -63.842, 0.60),
    # Galaxy Groups
    'Fornax': (54.017, -35.267, 0.57),
    'Dorado': (64.014, -55.780, 0.60),
    # Galaxy Clusters
    'Perseus': (49.638, 41.651, 0.70),
    'Abell 2390': (328.397, 17.709, 0.75),
    'Abell 2764': (5.713, -49.249, 0.75),
}

# Field categories for coloring
FIELD_CATEGORIES = {
    'Star-forming': ['Horsehead', 'Messier 78', 'Taurus', 'Barnard 30'],
    'Globular Clusters': ['NGC 6254', 'NGC 6397'],
    'Nearby Galaxies': ['NGC 6822', 'Holmberg II', 'IC 10', 'IC 342', 'NGC 2403', 'NGC 6744'],
    'Galaxy Groups': ['Fornax', 'Dorado'],
    'Galaxy Clusters': ['Perseus', 'Abell 2390', 'Abell 2764'],
}

# Colors for each category
CATEGORY_COLORS = {
    'Star-forming': plt.cm.Reds,
    'Globular Clusters': plt.cm.Greens,
    'Nearby Galaxies': plt.cm.Blues,
    'Galaxy Groups': plt.cm.Purples,
    'Galaxy Clusters': plt.cm.Oranges,
}


# ─── Shared helpers ──────────────────────────────────────────────────────────

def assign_fields(df):
    """Assign each source to its nearest field (vectorized for speed)."""
    ra = df['ra'].values
    dec = df['dec'].values
    n_sources = len(df)

    field_assignments = np.full(n_sources, None, dtype=object)
    min_distances = np.full(n_sources, np.inf)

    for field_name, (field_ra, field_dec, area) in ERO_FIELDS.items():
        radius = np.sqrt(area / np.pi)

        ra1 = np.radians(ra)
        dec1 = np.radians(dec)
        ra2 = np.radians(field_ra)
        dec2 = np.radians(field_dec)

        dra = ra2 - ra1
        ddec = dec2 - dec1
        a = np.sin(ddec/2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra/2)**2
        dist = np.degrees(2 * np.arcsin(np.sqrt(a)))

        mask = (dist < radius) & (dist < min_distances)
        field_assignments[mask] = field_name
        min_distances[mask] = dist[mask]

    df['field'] = field_assignments
    return df


def get_field_color(field_name):
    """Get color for a field based on its category."""
    for category, fields in FIELD_CATEGORIES.items():
        if field_name in fields:
            cmap = CATEGORY_COLORS[category]
            idx = fields.index(field_name)
            return cmap(0.4 + 0.15 * idx)
    return 'gray'


def get_galactic_b(ra, dec):
    """Return galactic latitude b in degrees."""
    from astropy.coordinates import SkyCoord
    c = SkyCoord(ra=ra, dec=dec, unit='deg')
    return c.galactic.b.deg


# ─── Data queries ────────────────────────────────────────────────────────────

def query_gaia_matched(username):
    """Query Gaia-matched ERO data from crossmatch table."""
    import psycopg2

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)

    query = """
        SELECT
            e_v_alpha_j2000 AS ra,
            e_v_delta_j2000 AS dec,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            e_v_mag_auto AS v_mag_auto,
            CASE
                WHEN e_v_class_star >= 0.65 THEN 'STAR'
                WHEN e_v_class_star < 0.65 THEN 'GALAXY'
            END as label
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_class_star IS NOT NULL
        ORDER BY g_source_id
    """

    print("Querying Gaia-matched ERO data...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"Retrieved {len(df):,} Gaia-matched sources")
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
            CASE
                WHEN v_class_star >= 0.65 THEN 'STAR'
                WHEN v_class_star < 0.65 THEN 'GALAXY'
            END as label
        FROM euclid_ero.merge_cat
        WHERE v_class_star IS NOT NULL
    """

    print("Querying all ERO data...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"Retrieved {len(df):,} ERO sources")
    return df


# ─── Gaia-matched plots ─────────────────────────────────────────────────────

def plot_gaia_g_mag_count(df, output_dir):
    """Plot Gaia G magnitude distribution for each ERO field (Gaia-matched)."""
    df_assigned = df[df['field'].notna()].copy()

    print(f"\nGaia-matched sources assigned to fields: {len(df_assigned):,}")

    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()

    bins = np.arange(14, 22.5, 0.25)

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        fd = df_assigned[df_assigned['field'] == field_name]
        fd_star = fd[fd['label'] == 'STAR']['phot_g_mean_mag'].dropna()
        fd_gal = fd[fd['label'] == 'GALAXY']['phot_g_mean_mag'].dropna()
        fd_all = fd['phot_g_mean_mag'].dropna()
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])

        ax.hist(fd_all, bins=bins, alpha=0.3, color='gray', label=f'All ({len(fd_all):,})')
        ax.hist(fd_gal, bins=bins, alpha=0.7, color='coral',
                label=f'Galaxies ({len(fd_gal):,})', histtype='step', linewidth=1.5)
        ax.hist(fd_star, bins=bins, alpha=0.7, color='steelblue',
                label=f'Stars ({len(fd_star):,})', histtype='step', linewidth=1.5)

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
        ax.set_xlabel('Gaia G', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(14, 22)
        ax.legend(fontsize=6, loc='upper left')
        ax.grid(alpha=0.3, axis='y')

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Sources Count vs Gaia G by Field (Gaia-matched)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = output_dir / 'gaia_fields_g_mag_count.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_gaia_g_mag_fraction(df, output_dir):
    """Plot galaxy fraction vs Gaia G magnitude for each ERO field (Gaia-matched)."""
    df_assigned = df[df['field'].notna()].copy()

    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()

    mag_bins = np.arange(14, 22.5, 0.25)
    mag_centers = (mag_bins[:-1] + mag_bins[1:]) / 2

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        field_data = df_assigned[df_assigned['field'] == field_name].copy()
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])
        color = get_field_color(field_name)

        if len(field_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(field_name, fontsize=10, fontweight='bold')
            ax.set_xlim(14, 22)
            ax.set_ylim(-0.02, 1.02)
            continue

        gmag = field_data['phot_g_mean_mag'].dropna()
        gmag_gal = field_data[field_data['label'] == 'GALAXY']['phot_g_mean_mag'].dropna()

        counts_all, _ = np.histogram(gmag, bins=mag_bins)
        counts_gal, _ = np.histogram(gmag_gal, bins=mag_bins)

        with np.errstate(divide='ignore', invalid='ignore'):
            frac = np.where(counts_all > 0, counts_gal / counts_all, np.nan)

        mask = counts_all >= 5
        ax.plot(mag_centers[mask], frac[mask], 'o-', color=color, markersize=3, linewidth=1.5)

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
        ax.set_xlabel('Gaia G', fontsize=9)
        ax.set_ylabel('Galaxy fraction', fontsize=9)
        ax.set_xlim(14, 22)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.grid(alpha=0.3)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Galaxy Fraction vs Gaia G by Field (Gaia-matched)',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = output_dir / 'gaia_fields_g_mag_fraction.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_gaia_e_mag_count(df, output_dir):
    """Plot Euclid VIS magnitude distribution for each ERO field (Gaia-matched)."""
    df_assigned = df[df['field'].notna() & df['phot_g_mean_mag'].notna()].copy()

    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()

    bins = np.arange(14, 26.5, 0.25)

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        fd = df_assigned[df_assigned['field'] == field_name]
        fd_star = fd[fd['label'] == 'STAR']['v_mag_auto'].dropna()
        fd_gal = fd[fd['label'] == 'GALAXY']['v_mag_auto'].dropna()
        fd_all = fd['v_mag_auto'].dropna()
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])

        ax.hist(fd_all, bins=bins, alpha=0.3, color='gray', label=f'All ({len(fd_all):,})')
        ax.hist(fd_gal, bins=bins, alpha=0.7, color='coral',
                label=f'Galaxies ({len(fd_gal):,})', histtype='step', linewidth=1.5)
        ax.hist(fd_star, bins=bins, alpha=0.7, color='steelblue',
                label=f'Stars ({len(fd_star):,})', histtype='step', linewidth=1.5)

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
        ax.set_xlabel('Euclid VIS (v_mag_auto)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(14, 26)
        ax.legend(fontsize=6, loc='upper left')
        ax.grid(alpha=0.3, axis='y')

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Sources Count vs Euclid Mag by Field (Gaia-matched)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = output_dir / 'gaia_fields_e_mag_count.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_gaia_e_mag_fraction(df, output_dir):
    """Plot galaxy fraction vs Euclid VIS magnitude for each ERO field (Gaia-matched)."""
    df_assigned = df[df['field'].notna() & df['phot_g_mean_mag'].notna()].copy()

    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()

    mag_bins = np.arange(14, 26.5, 0.25)
    mag_centers = (mag_bins[:-1] + mag_bins[1:]) / 2

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        field_data = df_assigned[df_assigned['field'] == field_name].copy()
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])
        color = get_field_color(field_name)

        if len(field_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(field_name, fontsize=10, fontweight='bold')
            ax.set_xlim(14, 26)
            ax.set_ylim(-0.02, 1.02)
            continue

        vmag = field_data['v_mag_auto'].dropna()
        vmag_gal = field_data[field_data['label'] == 'GALAXY']['v_mag_auto'].dropna()

        counts_all, _ = np.histogram(vmag, bins=mag_bins)
        counts_gal, _ = np.histogram(vmag_gal, bins=mag_bins)

        with np.errstate(divide='ignore', invalid='ignore'):
            frac = np.where(counts_all > 0, counts_gal / counts_all, np.nan)

        mask = counts_all >= 5
        ax.plot(mag_centers[mask], frac[mask], 'o-', color=color, markersize=3, linewidth=1.5)

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
        ax.set_xlabel('Euclid VIS mag', fontsize=9)
        ax.set_ylabel('Galaxy fraction', fontsize=9)
        ax.set_xlim(14, 26)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.grid(alpha=0.3)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Galaxy Fraction vs Euclid Mag by Field (Gaia-matched)',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = output_dir / 'gaia_fields_e_mag_fraction.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ─── All ERO plots ──────────────────────────────────────────────────────────

def plot_ero_all_e_mag_count(df, output_dir):
    """Plot Euclid VIS magnitude distribution for each ERO field (all ERO sources)."""
    from astropy.coordinates import SkyCoord
    df_assigned = df[df['field'].notna()].copy()

    print(f"\nAll ERO sources assigned to fields: {len(df_assigned):,}")
    print(f"Sources not in any field: {df['field'].isna().sum():,}")

    field_counts = df_assigned['field'].value_counts()
    print("\nSources per field:")
    for field in ERO_FIELDS.keys():
        count = field_counts.get(field, 0)
        print(f"  {field}: {count:,}")

    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()

    bins = np.arange(14, 28.5, 0.25)

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        fd = df_assigned[df_assigned['field'] == field_name]
        fd_star = fd[fd['label'] == 'STAR']['v_mag_auto'].dropna()
        fd_gal = fd[fd['label'] == 'GALAXY']['v_mag_auto'].dropna()
        fd_all = fd['v_mag_auto'].dropna()

        ra, dec = ERO_FIELDS[field_name][:2]
        c = SkyCoord(ra=ra, dec=dec, unit='deg')
        gal_b = c.galactic.b.deg

        ax.hist(fd_all, bins=bins, alpha=0.3, color='gray', label=f'All ({len(fd_all):,})')
        ax.hist(fd_gal, bins=bins, alpha=0.7, color='coral',
                label=f'Galaxies ({len(fd_gal):,})', histtype='step', linewidth=1.5)
        ax.hist(fd_star, bins=bins, alpha=0.7, color='steelblue',
                label=f'Stars ({len(fd_star):,})', histtype='step', linewidth=1.5)

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
        ax.set_xlabel('Euclid VIS (v_mag_auto)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(14, 28)
        ax.legend(fontsize=6, loc='upper left')
        ax.grid(alpha=0.3, axis='y')

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Sources Count vs Euclid Mag by Field (All ERO)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = output_dir / 'ero_all_fields_e_mag_count.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_ero_all_e_mag_fraction(df, output_dir):
    """Plot galaxy fraction vs Euclid magnitude for each ERO field (all ERO sources)."""
    from astropy.coordinates import SkyCoord
    df_assigned = df[df['field'].notna()].copy()

    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()

    bins = np.arange(14, 28, 0.25)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        fd = df_assigned[df_assigned['field'] == field_name]
        fd_gal = fd[fd['label'] == 'GALAXY']['v_mag_auto'].dropna()
        fd_all = fd['v_mag_auto'].dropna()

        ra, dec = ERO_FIELDS[field_name][:2]
        c = SkyCoord(ra=ra, dec=dec, unit='deg')
        gal_b = c.galactic.b.deg
        color = get_field_color(field_name)

        counts_all, _ = np.histogram(fd_all, bins=bins)
        counts_gal, _ = np.histogram(fd_gal, bins=bins)

        with np.errstate(divide='ignore', invalid='ignore'):
            frac = np.where(counts_all > 0, counts_gal / counts_all, np.nan)

        mask = counts_all >= 10
        ax.plot(bin_centers[mask], frac[mask], 'o-', color=color, markersize=3,
                linewidth=1.5)

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
        ax.set_xlabel('Euclid VIS (v_mag_auto)', fontsize=9)
        ax.set_ylabel('Galaxy fraction', fontsize=9)
        ax.set_xlim(14, 27)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.grid(alpha=0.3)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Galaxy Fraction vs Euclid Mag by Field (All ERO)',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = output_dir / 'ero_all_fields_e_mag_fraction.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ─── Combined summary plot ───────────────────────────────────────────────────

def plot_combined_summary(df_gaia, df_ero, output_dir):
    """2x3 summary: all 17 fields combined into single histograms and fraction curves.

    Row 1: count histograms (All ERO VIS, Gaia-matched VIS, Gaia-matched G)
    Row 2: galaxy fraction  (All ERO VIS, Gaia-matched VIS, Gaia-matched G)
    """
    df_gaia_f = df_gaia[df_gaia['field'].notna()].copy()
    df_ero_f = df_ero[df_ero['field'].notna()].copy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ── Column configs: (data, mag_col, bins, xlabel, xlim, title_suffix) ──
    configs = [
        (df_ero_f, 'v_mag_auto', np.arange(14, 28.5, 0.25),
         'Euclid VIS (v_mag_auto)', (14, 28), 'All ERO, Euclid VIS'),
        (df_gaia_f, 'v_mag_auto', np.arange(14, 26.5, 0.25),
         'Euclid VIS (v_mag_auto)', (14, 26), 'Gaia-matched, Euclid VIS'),
        (df_gaia_f, 'phot_g_mean_mag', np.arange(14, 22.5, 0.25),
         'Gaia G', (14, 22), 'Gaia-matched, Gaia G'),
    ]

    for col_idx, (df, mag_col, bins, xlabel, xlim, title_suffix) in enumerate(configs):
        mag_all = df[mag_col].dropna()
        mag_star = df[df['label'] == 'STAR'][mag_col].dropna()
        mag_gal = df[df['label'] == 'GALAXY'][mag_col].dropna()

        # Row 1: count histogram
        ax = axes[0, col_idx]
        ax.hist(mag_all, bins=bins, alpha=0.3, color='gray',
                label=f'All ({len(mag_all):,})')
        ax.hist(mag_gal, bins=bins, alpha=0.7, color='coral',
                label=f'Galaxies ({len(mag_gal):,})', histtype='step', linewidth=1.5)
        ax.hist(mag_star, bins=bins, alpha=0.7, color='steelblue',
                label=f'Stars ({len(mag_star):,})', histtype='step', linewidth=1.5)
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
        ax.set_xlim(xlim)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'Galaxy Fraction ({title_suffix})', fontsize=11, fontweight='bold')
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.grid(alpha=0.3)

    plt.suptitle('All ERO Fields Combined: Magnitude Distributions & Galaxy Fraction',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = output_dir / 'combined_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ERO Fields: Magnitude Distributions (Gaia-matched + All ERO)")
    print("=" * 70)

    # Query Gaia-matched data
    df_gaia = query_gaia_matched(username)
    print("\nAssigning Gaia-matched sources to fields...")
    df_gaia = assign_fields(df_gaia)

    # Gaia-matched plots
    print("\n--- Gaia-matched: Gaia G magnitude ---")
    plot_gaia_g_mag_count(df_gaia, output_dir)
    plot_gaia_g_mag_fraction(df_gaia, output_dir)

    print("\n--- Gaia-matched: Euclid VIS magnitude ---")
    plot_gaia_e_mag_count(df_gaia, output_dir)
    plot_gaia_e_mag_fraction(df_gaia, output_dir)

    # Query all ERO data
    df_ero = query_all_ero(username)
    print("\nAssigning all ERO sources to fields...")
    df_ero = assign_fields(df_ero)

    # All ERO plots
    print("\n--- All ERO: Euclid VIS magnitude ---")
    plot_ero_all_e_mag_count(df_ero, output_dir)
    plot_ero_all_e_mag_fraction(df_ero, output_dir)

    # Combined summary (all fields merged)
    print("\n--- Combined summary (all fields) ---")
    plot_combined_summary(df_gaia, df_ero, output_dir)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
