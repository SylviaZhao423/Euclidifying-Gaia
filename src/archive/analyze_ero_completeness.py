#!/usr/bin/env python3
"""
Investigate whether the bimodal galaxy magnitude distribution in ERO fields
is caused by varying Gaia completeness within and across the 17 fields.

Uses the GaiaUnlimited DR3 selection function (Cantat-Gaudin et al. 2023)
to compute completeness S(G) at each source's actual RA/Dec position
(not just the field center), capturing within-field M₁₀ variation.

Three analyses:
  1. Per-field galaxy G-mag histograms (stacked to show decomposition)
  2. GaiaUnlimited completeness S(G) per source (+ M₁₀ spatial maps)
  3. Completeness-corrected combined distribution (1/S(G) weighting)

Usage:
    python src/analyze_ero_completeness.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output' / 'ero_completeness'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

# ERO field definitions: name -> (RA, Dec, area_deg2)
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

FIELD_CATEGORIES = {
    'Star-forming': ['Horsehead', 'Messier 78', 'Taurus', 'Barnard 30'],
    'Globular Clusters': ['NGC 6254', 'NGC 6397'],
    'Nearby Galaxies': ['NGC 6822', 'Holmberg II', 'IC 10', 'IC 342', 'NGC 2403', 'NGC 6744'],
    'Galaxy Groups': ['Fornax', 'Dorado'],
    'Galaxy Clusters': ['Perseus', 'Abell 2390', 'Abell 2764'],
}

CATEGORY_COLORS = {
    'Star-forming': '#d62728',
    'Globular Clusters': '#2ca02c',
    'Nearby Galaxies': '#1f77b4',
    'Galaxy Groups': '#9467bd',
    'Galaxy Clusters': '#ff7f0e',
}


def get_field_color(field_name):
    for category, fields in FIELD_CATEGORIES.items():
        if field_name in fields:
            return CATEGORY_COLORS[category]
    return 'gray'


def get_field_category(field_name):
    for category, fields in FIELD_CATEGORIES.items():
        if field_name in fields:
            return category
    return 'Unknown'


def get_galactic_b(ra, dec):
    c = SkyCoord(ra=ra, dec=dec, unit='deg')
    return c.galactic.b.deg


# =============================================================================
# Data querying
# =============================================================================

def query_all_gaia_in_fields(username):
    """Query all Gaia DR3 sources within ERO field footprints via q3c cone search."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)

    frames = []
    for field_name, (field_ra, field_dec, area) in ERO_FIELDS.items():
        radius = np.sqrt(area / np.pi)
        query = f"""
            SELECT ra, dec, phot_g_mean_mag
            FROM gaia_dr3.gaia_source
            WHERE q3c_radial_query(ra, dec, {field_ra}, {field_dec}, {radius})
        """
        print(f"    {field_name} (r={radius:.3f}\u00b0)...", end='', flush=True)
        df_f = pd.read_sql_query(query, conn)
        df_f['field'] = field_name
        print(f" {len(df_f):,} sources")
        frames.append(df_f)

    conn.close()
    df = pd.concat(frames, ignore_index=True)
    print(f"  \u2713 {len(df):,} total Gaia sources in ERO fields")
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
    print("  Querying ALL ERO sources...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  ✓ {len(df):,} sources")
    return df


def query_ero_data(username):
    """Query ERO crossmatch data with coordinates, magnitudes, and classification."""
    import psycopg2

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)

    query = """
        SELECT
            e_v_alpha_j2000 AS ra,
            e_v_delta_j2000 AS dec,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            e_v_mag_auto AS v_mag_auto,
            e_v_class_star AS class_star,
            CASE
                WHEN e_v_class_star >= 0.65 THEN 'STAR'
                WHEN e_v_class_star < 0.65 THEN 'GALAXY'
            END as label
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_class_star IS NOT NULL
        ORDER BY g_source_id
    """

    print("  Querying ERO crossmatch data...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  ✓ {len(df):,} sources")

    return df


def assign_fields(df):
    """Assign each source to its nearest ERO field (within field radius)."""
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

        ddec = dec2 - dec1
        dra = ra2 - ra1
        a = np.sin(ddec/2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra/2)**2
        dist = np.degrees(2 * np.arcsin(np.sqrt(a)))

        mask = (dist < radius) & (dist < min_distances)
        field_assignments[mask] = field_name
        min_distances[mask] = dist[mask]

    df['field'] = field_assignments
    return df


# =============================================================================
# GaiaUnlimited completeness — per-source
# =============================================================================

def compute_per_source_completeness(df):
    """
    Compute S(G) at each source's actual RA/Dec using GaiaUnlimited.

    The selection function S(G, ℓ, b) from Cantat-Gaudin et al. (2023) uses
    a generalised sigmoid parametrised by M₁₀ (the median G magnitude of
    sources with astrometric_matched_transits ≤ 10 in the local HEALPix
    pixel):

        S(G | M₁₀) = 1 - 0.5 × [tanh((x(M₁₀) - G) / y(M₁₀)) + 1]^z(M₁₀)

    Uses mode='multi' which provides a pre-computed variable-resolution M₁₀
    map down to HEALPix level 10 (nside=1024, ~3.4 arcmin pixels), compared
    to the default mode='hpx7' which only uses level 7 (~27.5 arcmin).
    This captures within-field crowding gradients (e.g. globular cluster
    cores vs outskirts) at ~6× finer spatial resolution.

    Adds columns: 'completeness' (S(G) at source position)
    """
    from gaiaunlimited import DR3SelectionFunctionTCG

    print("  Loading GaiaUnlimited DR3 selection function (mode='multi', up to level 10)...")
    sf = DR3SelectionFunctionTCG(mode='multi')

    # Only compute for sources assigned to fields with valid G mag
    mask = df['field'].notna() & df['phot_g_mean_mag'].notna()
    idx = df.index[mask]

    ra_arr = df.loc[idx, 'ra'].values
    dec_arr = df.loc[idx, 'dec'].values
    gmag_arr = df.loc[idx, 'phot_g_mean_mag'].values.astype(float)

    n = len(idx)
    print(f"  Querying S(G) for {n:,} sources at their actual positions...")

    # Process in batches to manage memory
    batch_size = 50000
    completeness = np.full(n, np.nan)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        coords = SkyCoord(ra=ra_arr[start:end], dec=dec_arr[start:end], unit='deg')
        completeness[start:end] = sf.query(coords, gmag_arr[start:end])
        print(f"    Batch {start:,}-{end:,} done")

    # Write back to dataframe
    df['completeness'] = np.nan
    df.loc[idx, 'completeness'] = completeness

    # Summary statistics
    valid = df.loc[idx]
    print(f"  ✓ Completeness computed for {n:,} sources")
    print(f"    Median S(G): {np.nanmedian(completeness):.3f}")
    print(f"    Sources with S(G) < 0.5: {(completeness < 0.5).sum():,} "
          f"({100*(completeness < 0.5).sum()/n:.1f}%)")
    print(f"    Sources with S(G) < 0.1: {(completeness < 0.1).sum():,} "
          f"({100*(completeness < 0.1).sum()/n:.1f}%)")

    return sf


# =============================================================================
# Plotting
# =============================================================================

def plot_per_field_galaxy_histograms(df, output_dir):
    """Step 1: Per-field galaxy G-mag histograms + stacked version."""
    df_gal = df[(df['field'].notna()) & (df['label'] == 'GALAXY')].copy()
    bins = np.arange(14, 22.5, 0.25)

    # --- Individual fields ---
    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        field_data = df_gal[df_gal['field'] == field_name]['phot_g_mean_mag'].dropna()
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])
        color = get_field_color(field_name)

        if len(field_data) > 0:
            ax.hist(field_data, bins=bins, alpha=0.7, color=color)
            ax.set_title(f'{field_name} (n={len(field_data):,}, |b|={abs(gal_b):.1f}°)',
                         fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No galaxies', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10)

        ax.set_xlabel('Gaia G', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(14, 22)
        ax.grid(alpha=0.3, axis='y')

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Gaia-Matched ERO Galaxies: G-band Distribution by Field',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_field_galaxy_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ per_field_galaxy_histograms.png")

    # --- Stacked ---
    fig, ax = plt.subplots(figsize=(12, 6))
    field_order = sorted(ERO_FIELDS.keys(),
                         key=lambda f: abs(get_galactic_b(*ERO_FIELDS[f][:2])))

    field_data_list, field_labels, field_colors = [], [], []
    for field_name in field_order:
        data = df_gal[df_gal['field'] == field_name]['phot_g_mean_mag'].dropna()
        if len(data) > 0:
            field_data_list.append(data.values)
            gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])
            field_labels.append(f'{field_name} (|b|={abs(gal_b):.0f}°, n={len(data)})')
            field_colors.append(get_field_color(field_name))

    ax.hist(field_data_list, bins=bins, stacked=True, label=field_labels,
            color=field_colors, alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Gaia G magnitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('ERO Galaxies by Field (stacked, sorted by |b|)',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(14, 22)
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'stacked_galaxy_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ stacked_galaxy_histogram.png")


def plot_per_field_by_class(df, output_dir):
    """Per-field histograms showing stars, galaxies, and all sources + galaxy fraction."""
    df_field = df[df['field'].notna()].copy()
    bins = np.arange(14, 22.5, 0.25)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols

    # --- Per-field: stars, galaxies, all overlaid ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        fd = df_field[df_field['field'] == field_name]
        fd_star = fd[fd['label'] == 'STAR']['phot_g_mean_mag'].dropna()
        fd_gal = fd[fd['label'] == 'GALAXY']['phot_g_mean_mag'].dropna()
        fd_all = fd['phot_g_mean_mag'].dropna()
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])

        ax.hist(fd_all, bins=bins, alpha=0.3, color='gray', label=f'All ({len(fd_all):,})')
        ax.hist(fd_star, bins=bins, alpha=0.6, color='steelblue',
                label=f'Stars ({len(fd_star):,})', histtype='step', linewidth=1.5)
        ax.hist(fd_gal, bins=bins, alpha=0.7, color='coral',
                label=f'Galaxies ({len(fd_gal):,})', histtype='step', linewidth=1.5)

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
        ax.set_xlabel('Gaia G', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(14, 22)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(alpha=0.3, axis='y')

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Gaia-Matched ERO Sources by Class: G-band Distribution by Field',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_field_by_class.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ per_field_by_class.png")

    # --- Per-field: galaxy fraction per bin (raw only, before completeness) ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        fd = df_field[df_field['field'] == field_name]
        fd_gal = fd[fd['label'] == 'GALAXY']['phot_g_mean_mag'].dropna()
        fd_all = fd['phot_g_mean_mag'].dropna()
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])
        color = get_field_color(field_name)

        counts_all, _ = np.histogram(fd_all, bins=bins)
        counts_gal, _ = np.histogram(fd_gal, bins=bins)

        with np.errstate(divide='ignore', invalid='ignore'):
            frac = np.where(counts_all > 0, counts_gal / counts_all, np.nan)

        # Only plot bins with at least 5 sources for stability
        mask = counts_all >= 5
        ax.plot(bin_centers[mask], frac[mask], 'o-', color=color, markersize=3,
                linewidth=1.5, label='Raw')

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
        ax.set_xlabel('Gaia G', fontsize=9)
        ax.set_ylabel('Galaxy fraction', fontsize=9)
        ax.set_xlim(14, 22)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.grid(alpha=0.3)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Galaxy Fraction vs Gaia G by Field (raw)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_field_galaxy_fraction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ per_field_galaxy_fraction.png")

    # --- Combined: all fields stacked, 3 panels ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: Stars, galaxies, all overlaid (combined across all fields)
    ax = axes[0]
    all_star = df_field[df_field['label'] == 'STAR']['phot_g_mean_mag'].dropna()
    all_gal = df_field[df_field['label'] == 'GALAXY']['phot_g_mean_mag'].dropna()
    all_src = df_field['phot_g_mean_mag'].dropna()

    ax.hist(all_src, bins=bins, alpha=0.3, color='gray', label=f'All ({len(all_src):,})')
    ax.hist(all_star, bins=bins, alpha=0.7, color='steelblue',
            label=f'Stars ({len(all_star):,})', histtype='stepfilled')
    ax.hist(all_gal, bins=bins, alpha=0.7, color='coral',
            label=f'Galaxies ({len(all_gal):,})', histtype='stepfilled')
    ax.set_xlabel('Gaia G', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('(a) All ERO Fields: Stars vs Galaxies', fontsize=12, fontweight='bold')
    ax.set_xlim(14, 22)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # Panel 2: Galaxy fraction (all fields combined)
    ax = axes[1]
    counts_all, _ = np.histogram(all_src, bins=bins)
    counts_gal, _ = np.histogram(all_gal, bins=bins)
    counts_star, _ = np.histogram(all_star, bins=bins)

    with np.errstate(divide='ignore', invalid='ignore'):
        frac_all = np.where(counts_all > 0, counts_gal / counts_all, np.nan)

    mask = counts_all >= 10
    ax.plot(bin_centers[mask], frac_all[mask], 'o-', color='coral',
            markersize=4, linewidth=2, label='Galaxy fraction')
    ax.set_xlabel('Gaia G', fontsize=11, fontweight='bold')
    ax.set_ylabel('Galaxy fraction', fontsize=11, fontweight='bold')
    ax.set_title('(b) Galaxy Fraction vs G (all fields)', fontsize=12, fontweight='bold')
    ax.set_xlim(14, 22)
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Panel 3: Galaxy fraction per field (overlaid)
    ax = axes[2]
    field_order = sorted(ERO_FIELDS.keys(),
                         key=lambda f: abs(get_galactic_b(*ERO_FIELDS[f][:2])))
    for field_name in field_order:
        fd = df_field[df_field['field'] == field_name]
        fd_gal = fd[fd['label'] == 'GALAXY']['phot_g_mean_mag'].dropna()
        fd_all = fd['phot_g_mean_mag'].dropna()
        color = get_field_color(field_name)

        c_all, _ = np.histogram(fd_all, bins=bins)
        c_gal, _ = np.histogram(fd_gal, bins=bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            frac = np.where(c_all > 0, c_gal / c_all, np.nan)
        mask = c_all >= 5
        if mask.any():
            gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])
            ax.plot(bin_centers[mask], frac[mask], '-', color=color, alpha=0.7,
                    linewidth=1.2, label=f'{field_name} (|b|={abs(gal_b):.0f}°)')

    ax.set_xlabel('Gaia G', fontsize=11, fontweight='bold')
    ax.set_ylabel('Galaxy fraction', fontsize=11, fontweight='bold')
    ax.set_title('(c) Galaxy Fraction by Field', fontsize=12, fontweight='bold')
    ax.set_xlim(14, 22)
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
    ax.legend(fontsize=5.5, ncol=2, loc='upper left')
    ax.grid(alpha=0.3)

    plt.suptitle('ERO Source Classification: Stars, Galaxies & Galaxy Fraction',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / 'class_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ class_summary.png")


def plot_per_field_fraction_with_correction(df, output_dir):
    """Per-field galaxy fraction: raw vs completeness-corrected overlaid."""
    df_field = df[(df['field'].notna()) & df['phot_g_mean_mag'].notna()
                  & df['completeness'].notna()].copy()

    bins = np.arange(14, 22.5, 0.25)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    s_thresh = 0.05

    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        fd = df_field[df_field['field'] == field_name]
        fd_gal = fd[fd['label'] == 'GALAXY']
        fd_star = fd[fd['label'] == 'STAR']
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])
        color = get_field_color(field_name)

        # Raw fraction
        raw_gal, _ = np.histogram(fd_gal['phot_g_mean_mag'], bins=bins)
        raw_star, _ = np.histogram(fd_star['phot_g_mean_mag'], bins=bins)
        raw_all = raw_gal + raw_star
        with np.errstate(divide='ignore', invalid='ignore'):
            raw_frac = np.where(raw_all > 0, raw_gal / raw_all, np.nan)

        # Corrected fraction
        def _whist(sub):
            s = sub['completeness'].values
            g = sub['phot_g_mean_mag'].values
            v = s > s_thresh
            w = np.where(v, 1.0 / s, 0.0)
            c, _ = np.histogram(g[v], bins=bins, weights=w[v])
            return c

        corr_gal = _whist(fd_gal)
        corr_star = _whist(fd_star)
        corr_all = corr_gal + corr_star
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_frac = np.where(corr_all > 0, corr_gal / corr_all, np.nan)

        mask_raw = raw_all >= 5
        mask_corr = corr_all > 5
        ax.plot(bin_centers[mask_raw], raw_frac[mask_raw], 'o-', color=color,
                markersize=3, linewidth=1.5, alpha=0.8, label='Raw')
        ax.plot(bin_centers[mask_corr], corr_frac[mask_corr], 's--', color='gray',
                markersize=3, linewidth=1.5, alpha=0.7, label='Corrected')

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
        ax.set_xlabel('Gaia G', fontsize=9)
        ax.set_ylabel('Galaxy fraction', fontsize=9)
        ax.set_xlim(14, 22)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Galaxy Fraction vs Gaia G by Field: Raw vs Completeness-Corrected',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_field_galaxy_fraction_corrected.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ per_field_galaxy_fraction_corrected.png")


def plot_per_field_counts_corrected(df, output_dir):
    """Per-field source count histograms: raw vs completeness-corrected."""
    df_field = df[(df['field'].notna()) & df['phot_g_mean_mag'].notna()
                  & df['completeness'].notna()].copy()

    bins = np.arange(14, 22.5, 0.25)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    s_thresh = 0.05

    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        fd = df_field[df_field['field'] == field_name]
        fd_gal = fd[fd['label'] == 'GALAXY']
        fd_star = fd[fd['label'] == 'STAR']
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])

        # Raw counts
        raw_gal, _ = np.histogram(fd_gal['phot_g_mean_mag'], bins=bins)
        raw_star, _ = np.histogram(fd_star['phot_g_mean_mag'], bins=bins)
        raw_all = raw_gal + raw_star

        # Corrected counts (1/S weighting)
        def _whist(sub):
            s = sub['completeness'].values
            g = sub['phot_g_mean_mag'].values
            v = s > s_thresh
            w = np.where(v, 1.0 / s, 0.0)
            c, _ = np.histogram(g[v], bins=bins, weights=w[v])
            return c

        corr_gal = _whist(fd_gal)
        corr_star = _whist(fd_star)
        corr_all = corr_gal + corr_star

        raw_total = int(raw_all.sum())
        corr_total = int(corr_all.sum())

        ax.step(bin_centers, raw_star, where='mid', color='coral',
                linewidth=1.2, alpha=0.8, label=f'Stars raw ({int(raw_star.sum()):,})')
        ax.step(bin_centers, raw_gal, where='mid', color='steelblue',
                linewidth=1.2, alpha=0.8, label=f'Galaxies raw ({int(raw_gal.sum()):,})')
        ax.step(bin_centers, corr_star, where='mid', color='coral',
                linewidth=1.5, linestyle='--', label=f'Stars corr ({int(corr_star.sum()):,})')
        ax.step(bin_centers, corr_gal, where='mid', color='steelblue',
                linewidth=1.5, linestyle='--', label=f'Galaxies corr ({int(corr_gal.sum()):,})')

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)\n'
                     f'raw {raw_total:,} → corr {corr_total:,}',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Gaia G', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(14, 22)
        ax.legend(fontsize=6, loc='upper left')
        ax.grid(alpha=0.3, axis='y')

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Per-Field Source Counts: Raw vs Completeness-Corrected (1/S weighting)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_field_counts_corrected.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ per_field_counts_corrected.png")


def plot_completeness_distribution(df, output_dir):
    """
    Step 2: Show per-source completeness distribution and spatial variation.
    """
    df_gal = df[(df['field'].notna()) & (df['label'] == 'GALAXY')
                & df['completeness'].notna()].copy()

    # --- 2a: Histogram of S(G) values for all galaxies ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.hist(df_gal['completeness'], bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Completeness S(G)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of S(G) for Gaia-Matched ERO Galaxies',
                 fontsize=12, fontweight='bold')
    ax.axvline(x=0.5, color='red', ls='--', alpha=0.7, label='S=0.5')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    # --- 2b: S(G) vs G magnitude, color-coded by field ---
    ax = axes[1]
    for field_name in ERO_FIELDS:
        fd = df_gal[df_gal['field'] == field_name]
        if len(fd) > 0:
            ax.scatter(fd['phot_g_mean_mag'], fd['completeness'],
                       s=1, alpha=0.3, color=get_field_color(field_name),
                       label=field_name, rasterized=True)
    ax.set_xlabel('Gaia G magnitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Completeness S(G)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Source S(G) vs G Magnitude', fontsize=12, fontweight='bold')
    ax.set_xlim(14, 22)
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(y=0.5, color='gray', ls='--', alpha=0.5)
    ax.legend(fontsize=6, ncol=3, loc='lower left', markerscale=5)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'per_source_completeness.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ per_source_completeness.png")

    # --- 2c: Per-field completeness vs G, showing within-field spread ---
    # For select fields with most galaxies, show median + spread
    top_fields = df_gal.groupby('field').size().nlargest(9).index.tolist()
    n_cols = 3
    n_rows = (len(top_fields) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten()

    gmag_bins = np.arange(15, 22, 0.3)
    bin_centers = (gmag_bins[:-1] + gmag_bins[1:]) / 2

    for idx, field_name in enumerate(top_fields):
        ax = axes[idx]
        fd = df_gal[df_gal['field'] == field_name]
        color = get_field_color(field_name)

        medians, lo, hi = [], [], []
        for i in range(len(gmag_bins) - 1):
            bm = fd[(fd['phot_g_mean_mag'] >= gmag_bins[i]) &
                     (fd['phot_g_mean_mag'] < gmag_bins[i+1])]
            if len(bm) >= 5:
                s = bm['completeness'].values
                medians.append(np.median(s))
                lo.append(np.percentile(s, 16))
                hi.append(np.percentile(s, 84))
            else:
                medians.append(np.nan)
                lo.append(np.nan)
                hi.append(np.nan)

        medians, lo, hi = np.array(medians), np.array(lo), np.array(hi)
        valid = ~np.isnan(medians)

        ax.fill_between(bin_centers[valid], lo[valid], hi[valid],
                         alpha=0.3, color=color, label='16-84th %ile')
        ax.plot(bin_centers[valid], medians[valid], lw=2, color=color, label='Median')
        ax.set_xlabel('Gaia G', fontsize=10)
        ax.set_ylabel('S(G)', fontsize=10)
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])
        ax.set_title(f'{field_name} (n={len(fd):,}, |b|={abs(gal_b):.0f}°)',
                     fontsize=10, fontweight='bold')
        ax.set_xlim(15, 22)
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.5)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for idx in range(len(top_fields), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Within-Field Completeness Spread (per-source S(G))',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'within_field_completeness_spread.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ within_field_completeness_spread.png")


def plot_completeness_corrected(df, output_dir):
    """Step 3: Raw vs completeness-corrected galaxy magnitude distribution."""
    df_gal = df[(df['field'].notna()) & (df['label'] == 'GALAXY')
                & df['phot_g_mean_mag'].notna() & df['completeness'].notna()].copy()

    bins = np.arange(14, 22.5, 0.25)
    gmags = df_gal['phot_g_mean_mag'].values
    s_vals = df_gal['completeness'].values

    # Compute weights: 1/S(G) where S > 5%, else exclude
    weights = np.where(s_vals > 0.05, 1.0 / s_vals, np.nan)
    valid = ~np.isnan(weights)
    gmags_v = gmags[valid]
    weights_v = weights[valid]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.hist(gmags, bins=bins, color='steelblue', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Gaia G magnitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Raw Gaia-Matched ERO Galaxies', fontsize=13, fontweight='bold')
    ax.set_xlim(14, 22)
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    ax.hist(gmags_v, bins=bins, weights=weights_v, color='coral', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Gaia G magnitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Corrected count (1/S(G) weighted)', fontsize=12, fontweight='bold')
    ax.set_title('Completeness-Corrected ERO Galaxies', fontsize=13, fontweight='bold')
    ax.set_xlim(14, 22)
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Effect of Per-Source Gaia Completeness Correction',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / 'completeness_corrected_distribution.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ completeness_corrected_distribution.png")


def plot_corrected_fraction(df, output_dir):
    """Galaxy fraction per bin using raw and completeness-corrected counts,
    shown for both Gaia G and Euclid VIS (ERO) magnitudes."""
    df_field = df[(df['field'].notna()) & df['phot_g_mean_mag'].notna()
                  & df['completeness'].notna() & df['v_mag_auto'].notna()].copy()

    s_thresh = 0.05  # exclude sources with S(G) < 5%

    df_gal = df_field[df_field['label'] == 'GALAXY']
    df_star = df_field[df_field['label'] == 'STAR']

    # Magnitude configs: (column, bins, xlabel, xlim)
    mag_configs = [
        ('phot_g_mean_mag', np.arange(14, 22.5, 0.25), 'Gaia G', (14, 22)),
        ('v_mag_auto', np.arange(14, 26.5, 0.25), 'Euclid VIS (ERO)', (14, 26)),
    ]

    def weighted_hist(sub_df, col, bins):
        s = sub_df['completeness'].values
        m = sub_df[col].values
        valid = s > s_thresh
        w = np.where(valid, 1.0 / s, 0.0)
        counts, _ = np.histogram(m[valid], bins=bins, weights=w[valid])
        return counts

    fig, axes = plt.subplots(4, 2, figsize=(16, 22))

    for col_idx, (mag_col, bins, xlabel, xlim) in enumerate(mag_configs):
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Raw counts
        raw_gal, _ = np.histogram(df_gal[mag_col], bins=bins)
        raw_star, _ = np.histogram(df_star[mag_col], bins=bins)
        raw_all = raw_gal + raw_star
        with np.errstate(divide='ignore', invalid='ignore'):
            raw_frac = np.where(raw_all > 0, raw_gal / raw_all, np.nan)

        # Corrected counts
        corr_gal = weighted_hist(df_gal, mag_col, bins)
        corr_star = weighted_hist(df_star, mag_col, bins)
        corr_all = corr_gal + corr_star
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_frac = np.where(corr_all > 0, corr_gal / corr_all, np.nan)

        # Row 0: raw counts
        ax = axes[0, col_idx]
        ax.hist(df_gal[mag_col], bins=bins, alpha=0.7, color='coral',
                label=f'Galaxies ({len(df_gal):,})', histtype='stepfilled')
        ax.hist(df_star[mag_col], bins=bins, alpha=0.7, color='steelblue',
                label=f'Stars ({len(df_star):,})', histtype='stepfilled')
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title(f'Raw Counts ({xlabel})', fontsize=12, fontweight='bold')
        ax.set_xlim(xlim)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis='y')

        # Row 1: corrected counts
        ax = axes[1, col_idx]
        ax.bar(bin_centers, corr_star, width=bins[1]-bins[0], alpha=0.7,
               color='steelblue', label='Stars (corrected)')
        ax.bar(bin_centers, corr_gal, width=bins[1]-bins[0], alpha=0.7,
               color='coral', label='Galaxies (corrected)')
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_ylabel('Corrected count (1/S weighted)', fontsize=11, fontweight='bold')
        ax.set_title(f'Completeness-Corrected ({xlabel})', fontsize=12, fontweight='bold')
        ax.set_xlim(xlim)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis='y')

        # Row 2: raw vs corrected galaxy fraction
        ax = axes[2, col_idx]
        mask_raw = raw_all >= 10
        mask_corr = corr_all > 0
        ax.plot(bin_centers[mask_raw], raw_frac[mask_raw], 'o-', color='steelblue',
                markersize=3, linewidth=2, label='Raw')
        ax.plot(bin_centers[mask_corr], corr_frac[mask_corr], 's--', color='coral',
                markersize=3, linewidth=2, label='Completeness-corrected')
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_ylabel('Galaxy fraction', fontsize=11, fontweight='bold')
        ax.set_title(f'Galaxy Fraction: Raw vs Corrected ({xlabel})',
                     fontsize=12, fontweight='bold')
        ax.set_xlim(xlim)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Row 3: corrected galaxy fraction per field
        ax = axes[3, col_idx]
        field_order = sorted(ERO_FIELDS.keys(),
                             key=lambda f: abs(get_galactic_b(*ERO_FIELDS[f][:2])))
        for field_name in field_order:
            fd = df_field[df_field['field'] == field_name]
            fd_gal_f = fd[fd['label'] == 'GALAXY']
            fd_star_f = fd[fd['label'] == 'STAR']
            color = get_field_color(field_name)

            c_gal = weighted_hist(fd_gal_f, mag_col, bins)
            c_star = weighted_hist(fd_star_f, mag_col, bins)
            c_all = c_gal + c_star
            with np.errstate(divide='ignore', invalid='ignore'):
                frac = np.where(c_all > 0, c_gal / c_all, np.nan)
            mask = c_all > 5
            if mask.any():
                gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])
                ax.plot(bin_centers[mask], frac[mask], '-', color=color, alpha=0.7,
                        linewidth=1.2, label=f'{field_name} (|b|={abs(gal_b):.0f}°)')

        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_ylabel('Galaxy fraction (corrected)', fontsize=11, fontweight='bold')
        ax.set_title(f'Corrected Galaxy Fraction by Field ({xlabel})',
                     fontsize=12, fontweight='bold')
        ax.set_xlim(xlim)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.legend(fontsize=5.5, ncol=2, loc='upper left')
        ax.grid(alpha=0.3)

    plt.suptitle('Galaxy Fraction: Raw vs Completeness-Corrected (Gaia G & Euclid VIS)',
                 fontsize=14, fontweight='bold', y=1.005)
    plt.tight_layout()
    plt.savefig(output_dir / 'corrected_galaxy_fraction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ corrected_galaxy_fraction.png")


def plot_combined_summary(df, output_dir):
    """Combined 3-panel summary figure."""
    df_gal = df[(df['field'].notna()) & (df['label'] == 'GALAXY')
                & df['phot_g_mean_mag'].notna() & df['completeness'].notna()].copy()

    bins = np.arange(14, 22.5, 0.25)
    field_order = sorted(ERO_FIELDS.keys(),
                         key=lambda f: abs(get_galactic_b(*ERO_FIELDS[f][:2])))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- Panel 1: Stacked histogram ---
    ax = axes[0]
    field_data_list, field_colors, field_labels = [], [], []
    for field_name in field_order:
        data = df_gal[df_gal['field'] == field_name]['phot_g_mean_mag'].dropna()
        if len(data) > 0:
            field_data_list.append(data.values)
            gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])
            field_labels.append(f'{field_name} (|b|={abs(gal_b):.0f}°)')
            field_colors.append(get_field_color(field_name))

    ax.hist(field_data_list, bins=bins, stacked=True, label=field_labels,
            color=field_colors, alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Gaia G', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('(a) Galaxy G-mag by Field', fontsize=12, fontweight='bold')
    ax.set_xlim(14, 22)
    ax.legend(fontsize=5.5, ncol=2, loc='upper left')
    ax.grid(alpha=0.3, axis='y')

    # --- Panel 2: Per-source S(G) vs G scatter ---
    ax = axes[1]
    for field_name in ERO_FIELDS:
        fd = df_gal[df_gal['field'] == field_name]
        if len(fd) > 0:
            ax.scatter(fd['phot_g_mean_mag'], fd['completeness'],
                       s=1, alpha=0.3, color=get_field_color(field_name),
                       label=field_name, rasterized=True)
    ax.set_xlabel('Gaia G', fontsize=11, fontweight='bold')
    ax.set_ylabel('S(G) at source position', fontsize=11, fontweight='bold')
    ax.set_title('(b) Per-Source Completeness', fontsize=12, fontweight='bold')
    ax.set_xlim(14, 22)
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(y=0.5, color='gray', ls='--', alpha=0.5)
    ax.legend(fontsize=5, ncol=3, loc='lower left', markerscale=5)
    ax.grid(alpha=0.3)

    # --- Panel 3: Raw vs corrected density ---
    ax = axes[2]
    gmags = df_gal['phot_g_mean_mag'].values
    s_vals = df_gal['completeness'].values
    weights = np.where(s_vals > 0.05, 1.0 / s_vals, np.nan)
    valid = ~np.isnan(weights)

    ax.hist(gmags, bins=bins, density=True, alpha=0.5, color='steelblue',
            edgecolor='white', label='Raw')
    ax.hist(gmags[valid], bins=bins, weights=weights[valid], density=True,
            alpha=0.5, color='coral', edgecolor='white', label='Corrected (1/S)')
    ax.set_xlabel('Gaia G', fontsize=11, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax.set_title('(c) Raw vs Completeness-Corrected', fontsize=12, fontweight='bold')
    ax.set_xlim(14, 22)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Gaia Completeness & ERO Galaxy Bimodality (per-source S(G))',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / 'completeness_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ completeness_summary.png")


def plot_per_field_class_star_maps(df, output_dir):
    """Per-field spatial maps colored by class_star value."""
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    df_field = df[df['field'].notna() & df['class_star'].notna()].copy()

    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols

    # --- All sources colored by class_star ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    cmap = plt.cm.RdYlBu  # red=galaxy (low class_star), blue=star (high class_star)
    norm = Normalize(vmin=0, vmax=1)

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        fd = df_field[df_field['field'] == field_name].copy()
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])
        field_ra, field_dec, area = ERO_FIELDS[field_name]

        if len(fd) == 0:
            ax.text(0.5, 0.5, 'No sources', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{field_name}', fontsize=10)
            continue

        # Sort so galaxies (low class_star) are plotted on top
        fd = fd.sort_values('class_star', ascending=False)

        sc = ax.scatter(fd['ra'], fd['dec'], c=fd['class_star'], cmap=cmap,
                        norm=norm, s=0.3, alpha=0.5, rasterized=True)

        n_gal = (fd['label'] == 'GALAXY').sum()
        n_star = (fd['label'] == 'STAR').sum()
        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)\n'
                     f'{n_star:,} stars, {n_gal:,} galaxies',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('RA [deg]', fontsize=8)
        ax.set_ylabel('Dec [deg]', fontsize=8)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)
        ax.invert_xaxis()
        ax.grid(alpha=0.2)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('class_star (0=galaxy, 1=star)', fontsize=11)

    plt.suptitle('ERO Fields: Spatial Distribution Colored by class_star',
                 fontsize=14, fontweight='bold', y=1.00)
    fig.subplots_adjust(right=0.90)
    plt.tight_layout(rect=[0, 0, 0.91, 0.98])
    plt.savefig(output_dir / 'per_field_class_star_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ per_field_class_star_map.png")

    # --- Galaxies only, colored by class_star (to see spatial clustering) ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    norm_gal = Normalize(vmin=0, vmax=0.65)
    cmap_gal = plt.cm.viridis_r

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        fd = df_field[(df_field['field'] == field_name) & (df_field['label'] == 'GALAXY')].copy()
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])
        field_ra, field_dec, area = ERO_FIELDS[field_name]

        if len(fd) == 0:
            ax.text(0.5, 0.5, 'No galaxies', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{field_name}', fontsize=10)
            continue

        sc = ax.scatter(fd['ra'], fd['dec'], c=fd['class_star'], cmap=cmap_gal,
                        norm=norm_gal, s=1, alpha=0.6, rasterized=True)

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°, n={len(fd):,})',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('RA [deg]', fontsize=8)
        ax.set_ylabel('Dec [deg]', fontsize=8)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)
        ax.invert_xaxis()
        ax.grid(alpha=0.2)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = ScalarMappable(cmap=cmap_gal, norm=norm_gal)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('class_star (galaxies only, < 0.65)', fontsize=11)

    plt.suptitle('ERO Fields: Galaxy Spatial Distribution Colored by class_star',
                 fontsize=14, fontweight='bold', y=1.00)
    fig.subplots_adjust(right=0.90)
    plt.tight_layout(rect=[0, 0, 0.91, 0.98])
    plt.savefig(output_dir / 'per_field_class_star_galaxies.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ per_field_class_star_galaxies.png")

    # --- Galaxies split by bright/faint peak, to see spatial separation ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        fd = df_field[(df_field['field'] == field_name) & (df_field['label'] == 'GALAXY')].copy()
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])

        if len(fd) == 0:
            ax.text(0.5, 0.5, 'No galaxies', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{field_name}', fontsize=10)
            continue

        bright = fd[fd['phot_g_mean_mag'] < 19.0]
        faint = fd[fd['phot_g_mean_mag'] >= 19.0]

        ax.scatter(faint['ra'], faint['dec'], c='steelblue', s=0.5, alpha=0.3,
                   label=f'G≥19 ({len(faint):,})', rasterized=True)
        ax.scatter(bright['ra'], bright['dec'], c='coral', s=3, alpha=0.7,
                   label=f'G<19 ({len(bright):,})', rasterized=True)

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('RA [deg]', fontsize=8)
        ax.set_ylabel('Dec [deg]', fontsize=8)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)
        ax.invert_xaxis()
        ax.legend(fontsize=7, markerscale=3)
        ax.grid(alpha=0.2)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('ERO Galaxies: Bright (G<19, coral) vs Faint (G≥19, blue) Spatial Distribution',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_field_galaxies_bright_faint.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ per_field_galaxies_bright_faint.png")


def _get_m10_at_coords(sf, coords):
    """Extract M₁₀ values at given coordinates from the multi-resolution SF map."""
    import astropy_healpix as ah
    ra = coords.ra
    dec = coords.dec
    match_ipix = ah.lonlat_to_healpix(ra, dec, sf.max_nside, order="nested")
    i = sf.sorter[
        np.searchsorted(sf.index, match_ipix, side="right", sorter=sf.sorter) - 1
    ]
    return sf.m10map[i]["M10"]


def _draw_healpix_grid(ax, sf, ra_min, ra_max, dec_min, dec_max):
    """Draw actual multi-resolution HEALPix pixel boundaries from the SF map.

    Finds the real map pixels (levels 7–10) that cover the field and draws
    their boundaries, showing exactly where the correction is computed.
    """
    import astropy_healpix as ah
    import healpy as hp

    # Sample the field to find which multi-res map entries cover it
    n_sample = 60
    ra_s = np.linspace(ra_min, ra_max, n_sample)
    dec_s = np.linspace(dec_min, dec_max, n_sample)
    ra_m, dec_m = np.meshgrid(ra_s, dec_s)
    coords_s = SkyCoord(ra=ra_m.ravel(), dec=dec_m.ravel(), unit='deg')

    match_ipix = ah.lonlat_to_healpix(
        coords_s.ra, coords_s.dec, sf.max_nside, order="nested")
    idx_arr = sf.sorter[
        np.searchsorted(sf.index, match_ipix, side="right", sorter=sf.sorter) - 1
    ]
    unique_idx = np.unique(idx_arr)

    level, ipix = ah.uniq_to_level_ipix(sf.m10map[unique_idx]["UNIQ"])

    for lev, ip in zip(level, ipix):
        nside = ah.level_to_nside(lev)
        boundaries = hp.boundaries(nside, ip, step=16, nest=True)
        theta, phi = hp.vec2ang(boundaries.T)
        ra_b = np.degrees(phi)
        dec_b = 90.0 - np.degrees(theta)
        ra_b = np.append(ra_b, ra_b[0])
        dec_b = np.append(dec_b, dec_b[0])
        if (ra_b.max() < ra_min or ra_b.min() > ra_max or
                dec_b.max() < dec_min or dec_b.min() > dec_max):
            continue
        ax.plot(ra_b, dec_b, color='gray', linewidth=0.3, alpha=0.4)


def plot_class_star_with_sf_background(df, sf, output_dir):
    """Per-field class_star scatter with M₁₀ as background colormap."""
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    df_field = df[df['field'].notna() & df['class_star'].notna()].copy()

    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4.5 * n_rows))
    axes = axes.flatten()
    cmap_src = plt.cm.RdYlBu  # red=galaxy, blue=star
    norm_src = Normalize(vmin=0, vmax=1)
    # Custom grayscale: low M₁₀ (crowded) = light gray, high M₁₀ (complete) = white
    # Never reaches black so scatter points stay visible
    from matplotlib.colors import LinearSegmentedColormap
    cmap_bg = LinearSegmentedColormap.from_list(
        'gray_light', [(0.65, 0.65, 0.65), (1.0, 1.0, 1.0)])

    print(f"  Computing M₁₀ background grids for {n_fields} fields...")

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        fd = df_field[df_field['field'] == field_name].copy()
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])

        if len(fd) == 0:
            ax.text(0.5, 0.5, 'No sources', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{field_name}', fontsize=10)
            continue

        # Determine field extent from data
        ra_min, ra_max = fd['ra'].min(), fd['ra'].max()
        dec_min, dec_max = fd['dec'].min(), fd['dec'].max()
        ra_pad = (ra_max - ra_min) * 0.05
        dec_pad = (dec_max - dec_min) * 0.05
        ra_min -= ra_pad; ra_max += ra_pad
        dec_min -= dec_pad; dec_max += dec_pad

        # Create grid for background
        n_grid = 100
        ra_grid = np.linspace(ra_min, ra_max, n_grid)
        dec_grid = np.linspace(dec_min, dec_max, n_grid)
        ra_mesh, dec_mesh = np.meshgrid(ra_grid, dec_grid)

        # Get M₁₀ at each grid point
        coords_grid = SkyCoord(ra=ra_mesh.ravel(), dec=dec_mesh.ravel(), unit='deg')
        m10_vals = _get_m10_at_coords(sf, coords_grid)
        m10_grid = np.array(m10_vals, dtype=float).reshape(ra_mesh.shape)

        # Per-field normalization — use actual data range for maximum contrast
        m10_min_field = np.nanmin(m10_grid)
        m10_max_field = np.nanmax(m10_grid)
        # Tiny padding only to avoid vmin==vmax
        if m10_max_field - m10_min_field < 0.01:
            m10_min_field -= 0.005
            m10_max_field += 0.005
        norm_bg_field = Normalize(vmin=m10_min_field, vmax=m10_max_field)

        # Plot M₁₀ background
        im = ax.pcolormesh(ra_grid, dec_grid, m10_grid, cmap=cmap_bg,
                           norm=norm_bg_field, shading='auto', rasterized=True)
        # Draw actual multi-resolution HEALPix boundaries from the SF map
        _draw_healpix_grid(ax, sf, ra_min, ra_max, dec_min, dec_max)
        # Per-panel mini colorbar
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, aspect=20)
        cb.set_label('M₁₀', fontsize=7)
        cb.ax.tick_params(labelsize=6)

        # Sort so galaxies (low class_star) are plotted on top
        fd = fd.sort_values('class_star', ascending=False)

        ax.scatter(fd['ra'], fd['dec'], c=fd['class_star'], cmap=cmap_src,
                   norm=norm_src, s=0.5, alpha=0.7, edgecolors='none',
                   rasterized=True)

        n_gal = (fd['label'] == 'GALAXY').sum()
        n_star = (fd['label'] == 'STAR').sum()
        # Show M₁₀ range in title
        m10_min = np.nanmin(m10_grid)
        m10_max = np.nanmax(m10_grid)
        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)\n'
                     f'{n_star:,} stars, {n_gal:,} gal | '
                     f'M₁₀: {m10_min:.1f}–{m10_max:.1f}',
                     fontsize=8, fontweight='bold')
        ax.set_xlabel('RA [deg]', fontsize=8)
        ax.set_ylabel('Dec [deg]', fontsize=8)
        # Clip axes to data extent so HEALPix lines don't expand the plot
        ax.set_xlim(ra_max, ra_min)  # inverted for RA
        ax.set_ylim(dec_min, dec_max)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    # Shared class_star colorbar
    cbar_ax1 = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    sm1 = ScalarMappable(cmap=cmap_src, norm=norm_src)
    cbar1 = fig.colorbar(sm1, cax=cbar_ax1)
    cbar1.set_label('class_star (0=galaxy, 1=star)', fontsize=10)

    plt.suptitle('ERO Fields: class_star (dots) with M₁₀ Background (per-field scale)',
                 fontsize=14, fontweight='bold', y=1.00)
    fig.subplots_adjust(right=0.90)
    plt.tight_layout(rect=[0, 0, 0.90, 0.98])
    plt.savefig(output_dir / 'per_field_class_star_with_sf.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ per_field_class_star_with_sf.png")


def plot_class_star_with_sg_background(df, sf, output_dir, g_eval=18.0):
    """Per-field class_star scatter with S(G) completeness as background."""
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    G_EVAL = g_eval

    df_field = df[df['field'].notna() & df['class_star'].notna()].copy()

    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4.5 * n_rows))
    axes = axes.flatten()
    cmap_src = plt.cm.RdYlBu  # red=galaxy, blue=star
    norm_src = Normalize(vmin=0, vmax=1)
    # Background: red-yellow-white — low completeness = red, high = white
    from matplotlib.colors import LinearSegmentedColormap
    cmap_bg = LinearSegmentedColormap.from_list(
        'completeness', [(0.8, 0.2, 0.2), (1.0, 0.7, 0.3), (1.0, 1.0, 1.0)])

    print(f"  Computing S(G={G_EVAL}) background grids for {n_fields} fields...")

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        fd = df_field[df_field['field'] == field_name].copy()
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])

        if len(fd) == 0:
            ax.text(0.5, 0.5, 'No sources', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{field_name}', fontsize=10)
            continue

        # Determine field extent from data
        ra_min, ra_max = fd['ra'].min(), fd['ra'].max()
        dec_min, dec_max = fd['dec'].min(), fd['dec'].max()
        ra_pad = (ra_max - ra_min) * 0.05
        dec_pad = (dec_max - dec_min) * 0.05
        ra_min -= ra_pad; ra_max += ra_pad
        dec_min -= dec_pad; dec_max += dec_pad

        # Create grid for background
        n_grid = 100
        ra_grid = np.linspace(ra_min, ra_max, n_grid)
        dec_grid = np.linspace(dec_min, dec_max, n_grid)
        ra_mesh, dec_mesh = np.meshgrid(ra_grid, dec_grid)

        # Query S(G=18) at each grid point
        coords_grid = SkyCoord(ra=ra_mesh.ravel(), dec=dec_mesh.ravel(), unit='deg')
        gmag_grid = np.full(len(coords_grid), G_EVAL)
        sg_vals = sf.query(coords_grid, gmag_grid)
        sg_grid = np.array(sg_vals, dtype=float).reshape(ra_mesh.shape)

        # Per-field normalization spanning actual min–max
        sg_min_field = np.nanmin(sg_grid)
        sg_max_field = np.nanmax(sg_grid)
        if sg_max_field - sg_min_field < 0.01:
            sg_min_field -= 0.005
            sg_max_field += 0.005
        norm_bg_field = Normalize(vmin=sg_min_field, vmax=sg_max_field)

        # Plot S(G) background
        im = ax.pcolormesh(ra_grid, dec_grid, sg_grid, cmap=cmap_bg,
                           norm=norm_bg_field, shading='auto', rasterized=True)
        # Draw actual multi-resolution HEALPix boundaries from the SF map
        _draw_healpix_grid(ax, sf, ra_min, ra_max, dec_min, dec_max)
        # Per-panel mini colorbar
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, aspect=20)
        cb.set_label(f'S(G={G_EVAL})', fontsize=7)
        cb.ax.tick_params(labelsize=6)

        # Sort so galaxies (low class_star) are plotted on top
        fd = fd.sort_values('class_star', ascending=False)

        ax.scatter(fd['ra'], fd['dec'], c=fd['class_star'], cmap=cmap_src,
                   norm=norm_src, s=0.5, alpha=0.7, edgecolors='none',
                   rasterized=True)

        n_gal = (fd['label'] == 'GALAXY').sum()
        n_star = (fd['label'] == 'STAR').sum()
        sg_min = np.nanmin(sg_grid)
        sg_max = np.nanmax(sg_grid)
        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)\n'
                     f'{n_star:,} stars, {n_gal:,} gal | '
                     f'S(G={G_EVAL}): {sg_min:.2f}–{sg_max:.2f}',
                     fontsize=8, fontweight='bold')
        ax.set_xlabel('RA [deg]', fontsize=8)
        ax.set_ylabel('Dec [deg]', fontsize=8)
        ax.set_xlim(ra_max, ra_min)  # inverted for RA
        ax.set_ylim(dec_min, dec_max)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    # Shared class_star colorbar
    cbar_ax1 = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    sm1 = ScalarMappable(cmap=cmap_src, norm=norm_src)
    cbar1 = fig.colorbar(sm1, cax=cbar_ax1)
    cbar1.set_label('class_star (0=galaxy, 1=star)', fontsize=10)

    plt.suptitle(f'ERO Fields: class_star (dots) with S(G={G_EVAL}) Background (per-field scale)',
                 fontsize=14, fontweight='bold', y=1.00)
    fig.subplots_adjust(right=0.90)
    plt.tight_layout(rect=[0, 0, 0.90, 0.98])
    fname = f'per_field_class_star_with_sg{int(G_EVAL)}.png'
    plt.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {fname}")


def plot_option1_corrected_fraction(df_matched, df_ero_all, output_dir):
    """Option 1 galaxy fraction: use full ERO as denominator, estimate stars via 1/S(G).

    Per field, per Euclid VIS magnitude bin:
      corrected_star_count = sum(1/S(G_i)) for Gaia-matched sources
      corrected_galaxy_count = max(0, N_ero - corrected_star_count)
      corrected_galaxy_fraction = corrected_galaxy_count / N_ero
    """
    bins = np.arange(14, 26.75, 0.25)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    s_thresh = 0.05

    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()

    # Prepare matched sources with valid completeness and VIS mag
    df_m = df_matched[(df_matched['field'].notna())
                      & df_matched['v_mag_auto'].notna()
                      & df_matched['completeness'].notna()].copy()

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])
        color = get_field_color(field_name)

        # All ERO sources in this field
        ero_f = df_ero_all[df_ero_all['field'] == field_name]
        # Gaia-matched sources in this field
        match_f = df_m[df_m['field'] == field_name]

        # ERO counts per VIS bin
        n_ero, _ = np.histogram(ero_f['v_mag_auto'], bins=bins)

        # ERO class_star-based galaxy fraction (baseline)
        ero_gal = ero_f[ero_f['class_star'] < 0.65]
        n_ero_gal, _ = np.histogram(ero_gal['v_mag_auto'], bins=bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            ero_gal_frac = np.where(n_ero > 0, n_ero_gal / n_ero, np.nan)

        # Option 1: corrected star count from Gaia matches
        s_vals = match_f['completeness'].values
        vis_vals = match_f['v_mag_auto'].values
        valid = s_vals > s_thresh
        weights = np.where(valid, 1.0 / s_vals, 0.0)
        corrected_star, _ = np.histogram(vis_vals[valid], bins=bins, weights=weights[valid])

        corrected_galaxy = np.maximum(0, n_ero - corrected_star)
        with np.errstate(divide='ignore', invalid='ignore'):
            option1_frac = np.where(n_ero > 0, corrected_galaxy / n_ero, np.nan)

        # Plot
        mask_ero = n_ero >= 5
        mask_opt1 = (n_ero >= 5)

        ax.plot(bin_centers[mask_ero], ero_gal_frac[mask_ero], 'o-', color=color,
                markersize=2, linewidth=1.2, alpha=0.7, label='ERO class_star')
        ax.plot(bin_centers[mask_opt1], option1_frac[mask_opt1], 's-', color='black',
                markersize=2, linewidth=1.5, alpha=0.9, label='Option 1 (corrected)')

        n_ero_total = len(ero_f)
        n_match_total = len(match_f)
        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)\n'
                     f'ERO: {n_ero_total:,}, matched: {n_match_total:,}',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Euclid VIS mag', fontsize=9)
        ax.set_ylabel('Galaxy fraction', fontsize=9)
        ax.set_xlim(14, 26.5)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(alpha=0.3)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Option 1: Galaxy Fraction from Full ERO + Gaia Completeness Correction\n'
                 'corrected_stars = Σ(1/S(G)), galaxies = N_ERO − corrected_stars',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_field_galaxy_fraction_option1.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ per_field_galaxy_fraction_option1.png")


def plot_option2_gaia_rate_correction(df_matched, df_ero_all, df_gaia, output_dir):
    """Option 2: correct for ERO rejection of Gaia sources, plotted vs Gaia G.

    Assumption: Gaia sources not matched by ERO are predominantly stars.
    The Gaia match rate rate(G) = N_matched(G) / N_Gaia(G) per Gaia G bin
    captures ERO's detection efficiency. Dividing star counts by this rate
    estimates the true stellar population.

    Per field, per Gaia G bin:
      rate(G) = N_matched(G) / N_Gaia(G)
      corrected_star(G) = N_matched_star(G) / rate(G)
      fraction = N_matched_gal(G) / (N_matched_gal(G) + corrected_star(G))
    """
    bins_g = np.arange(14, 22.5, 0.25)
    bc_g = (bins_g[:-1] + bins_g[1:]) / 2
    rate_thresh = 0.05  # skip bins where rate < 5%

    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols

    # Prepare matched sources with valid G mag and field
    df_m = df_matched[(df_matched['field'].notna())
                      & df_matched['phot_g_mean_mag'].notna()].copy()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])
        color = get_field_color(field_name)

        # Gaia sources in this field
        gaia_f = df_gaia[df_gaia['field'] == field_name]
        # Matched sources in this field
        match_f = df_m[df_m['field'] == field_name]
        match_gal = match_f[match_f['label'] == 'GALAXY']
        match_star = match_f[match_f['label'] == 'STAR']

        # Counts per Gaia G bin
        n_gaia, _ = np.histogram(gaia_f['phot_g_mean_mag'].dropna(), bins=bins_g)
        n_match, _ = np.histogram(match_f['phot_g_mean_mag'], bins=bins_g)
        n_match_gal, _ = np.histogram(match_gal['phot_g_mean_mag'], bins=bins_g)
        n_match_star, _ = np.histogram(match_star['phot_g_mean_mag'], bins=bins_g)

        # Match rate per G bin
        with np.errstate(divide='ignore', invalid='ignore'):
            rate = np.where(n_gaia > 0, n_match / n_gaia, 0.0)

        # Corrected star count per G bin
        with np.errstate(divide='ignore', invalid='ignore'):
            corrected_star = np.where(rate > rate_thresh,
                                      n_match_star / rate, 0.0)

        # Option 2 galaxy fraction
        denom = n_match_gal + corrected_star
        with np.errstate(divide='ignore', invalid='ignore'):
            option2_frac = np.where(denom > 0, n_match_gal / denom, np.nan)

        # Raw matched galaxy fraction for comparison
        raw_denom = n_match_gal + n_match_star
        with np.errstate(divide='ignore', invalid='ignore'):
            raw_frac = np.where(raw_denom > 0, n_match_gal / raw_denom, np.nan)

        # Plot
        mask_raw = raw_denom >= 5
        mask_opt2 = denom >= 5

        ax.plot(bc_g[mask_raw], raw_frac[mask_raw], 's--', color=color,
                markersize=3, linewidth=1.2, alpha=0.7, label='Raw matched')
        ax.plot(bc_g[mask_opt2], option2_frac[mask_opt2], 'D-', color='black',
                markersize=3, linewidth=1.5, alpha=0.9,
                label='Corrected (Gaia rate)')

        n_gaia_tot = len(gaia_f)
        n_match_tot = len(match_f)
        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)\n'
                     f'Gaia: {n_gaia_tot:,}, matched: {n_match_tot:,}',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Gaia G', fontsize=9)
        ax.set_ylabel('Galaxy fraction', fontsize=9)
        ax.set_xlim(14, 22)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(alpha=0.3)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Option 2: Correct for ERO Rejection of Gaia Stars\n'
                 'corrected_stars(G) = N_star(G) / rate(G), '
                 'frac = N_gal / (N_gal + corrected_stars)',
                 fontsize=12, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_field_galaxy_fraction_option2.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ per_field_galaxy_fraction_option2.png")


def plot_class_star_vs_gmag(df, output_dir):
    """class_star distribution in narrow Gaia G bins to diagnose the G~18 bump."""
    df_field = df[(df['field'].notna()) & df['class_star'].notna()
                  & df['phot_g_mean_mag'].notna()].copy()

    # G magnitude bins to examine
    g_slices = [(14, 15), (15, 16), (16, 17), (17, 17.5), (17.5, 18),
                (18, 18.5), (18.5, 19), (19, 19.5), (19.5, 20),
                (20, 20.5), (20.5, 21), (21, 22)]
    cs_bins = np.linspace(0, 1, 51)

    # --- Panel 1: class_star histograms per G slice (all fields combined) ---
    n_cols = 4
    n_rows = (len(g_slices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()

    for idx, (g_lo, g_hi) in enumerate(g_slices):
        ax = axes[idx]
        sub = df_field[(df_field['phot_g_mean_mag'] >= g_lo) &
                       (df_field['phot_g_mean_mag'] < g_hi)]
        n_total = len(sub)
        n_gal = (sub['class_star'] < 0.65).sum()
        gal_frac = n_gal / n_total if n_total > 0 else 0

        if n_total > 0:
            ax.hist(sub['class_star'], bins=cs_bins, color='steelblue',
                    alpha=0.7, edgecolor='white')
        ax.axvline(x=0.65, color='red', ls='--', alpha=0.7, label='cut=0.65')
        ax.set_title(f'G = [{g_lo}, {g_hi})\n'
                     f'n={n_total:,}, gal frac={gal_frac:.1%}',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('class_star', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis='y')

    for idx in range(len(g_slices), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('class_star Distribution by Gaia G Bin (all fields)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'class_star_vs_gmag_histograms.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ class_star_vs_gmag_histograms.png")

    # --- Panel 2: per-field class_star distributions for G=[17,18] vs G=[18,19] ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    for ax, (g_lo, g_hi), title_tag in zip(
            axes, [(17, 18), (18, 19)], ['G=[17,18)', 'G=[18,19)']):
        sub = df_field[(df_field['phot_g_mean_mag'] >= g_lo) &
                       (df_field['phot_g_mean_mag'] < g_hi)]
        field_order = sorted(ERO_FIELDS.keys(),
                             key=lambda f: abs(get_galactic_b(*ERO_FIELDS[f][:2])))
        for field_name in field_order:
            fd = sub[sub['field'] == field_name]
            if len(fd) < 10:
                continue
            color = get_field_color(field_name)
            gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])
            ax.hist(fd['class_star'], bins=cs_bins, density=True,
                    histtype='step', linewidth=1.2, color=color, alpha=0.8,
                    label=f'{field_name} (|b|={abs(gal_b):.0f}°, n={len(fd):,})')

        ax.axvline(x=0.65, color='red', ls='--', alpha=0.7)
        ax.set_title(f'{title_tag}: class_star by Field (normalised)',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('class_star', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=6, ncol=2, loc='upper center')
        ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'class_star_vs_gmag_per_field.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ class_star_vs_gmag_per_field.png")

    # --- Panel 3: 2D histogram class_star vs G (all fields) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    g_vals = df_field['phot_g_mean_mag'].values
    cs_vals = df_field['class_star'].values
    h = ax.hist2d(g_vals, cs_vals, bins=[np.arange(14, 22.5, 0.25), cs_bins],
                  cmin=1, cmap='viridis', norm=plt.matplotlib.colors.LogNorm())
    plt.colorbar(h[3], ax=ax, label='Count')
    ax.axhline(y=0.65, color='red', ls='--', alpha=0.7, label='cut=0.65')
    ax.set_xlabel('Gaia G', fontsize=12, fontweight='bold')
    ax.set_ylabel('class_star', fontsize=12, fontweight='bold')
    ax.set_title('class_star vs Gaia G (all ERO fields, Gaia-matched)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / 'class_star_vs_gmag_2d.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ class_star_vs_gmag_2d.png")


# =============================================================================
# Main
# =============================================================================

def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ERO Completeness Analysis (per-source)")
    print("=" * 70)

    # Step 0: Query data + assign fields
    print("\n[Step 0] Querying data...")
    df = query_ero_data(username)
    df = assign_fields(df)
    n_assigned = df['field'].notna().sum()
    n_gal = ((df['field'].notna()) & (df['label'] == 'GALAXY')).sum()
    print(f"  Assigned to fields: {n_assigned:,} sources, {n_gal:,} galaxies")

    # Step 0b: Per-field class_star spatial maps
    print("\n[Step 0b] Per-field class_star spatial maps...")
    plot_per_field_class_star_maps(df, OUTPUT_DIR)

    # Step 1: Per-field galaxy histograms (before completeness — no change)
    print("\n[Step 1] Per-field galaxy magnitude histograms...")
    plot_per_field_galaxy_histograms(df, OUTPUT_DIR)

    # Step 1b: Stars, galaxies, all sources + galaxy fraction
    print("\n[Step 1b] Per-field by class + galaxy fraction...")
    plot_per_field_by_class(df, OUTPUT_DIR)

    # Step 2: Per-source completeness
    print("\n[Step 2] Computing per-source GaiaUnlimited completeness...")
    sf = compute_per_source_completeness(df)
    plot_completeness_distribution(df, OUTPUT_DIR)

    # Step 2b: class_star maps with SF background
    print("\n[Step 2b] class_star maps with SF background...")
    plot_class_star_with_sf_background(df, sf, OUTPUT_DIR)
    for g_val in [17, 18, 20, 21]:
        plot_class_star_with_sg_background(df, sf, OUTPUT_DIR, g_eval=g_val)

    # Step 2c: Per-field galaxy fraction with correction overlay
    print("\n[Step 2c] Per-field galaxy fraction (raw vs corrected)...")
    plot_per_field_fraction_with_correction(df, OUTPUT_DIR)

    print("\n[Step 2d] Per-field source counts (raw vs corrected)...")
    plot_per_field_counts_corrected(df, OUTPUT_DIR)

    # Step 2e: Option 1 — full ERO denominator with Gaia completeness correction
    print("\n[Step 2e] Querying full ERO catalogue for Option 1...")
    df_ero_all = query_all_ero(username)
    df_ero_all = assign_fields(df_ero_all)
    n_ero_assigned = df_ero_all['field'].notna().sum()
    print(f"  Assigned to fields: {n_ero_assigned:,} / {len(df_ero_all):,} ERO sources")

    print("\n[Step 2f] Option 1 galaxy fraction (full ERO + Gaia completeness)...")
    plot_option1_corrected_fraction(df, df_ero_all, OUTPUT_DIR)

    print("\n[Step 2g] Querying all Gaia sources in ERO fields for Option 2...")
    df_gaia = query_all_gaia_in_fields(username)

    print("\n[Step 2h] Option 2 galaxy fraction (Gaia match rate correction)...")
    plot_option2_gaia_rate_correction(df, df_ero_all, df_gaia, OUTPUT_DIR)

    # Step 2i: class_star distribution vs G magnitude
    print("\n[Step 2i] class_star distribution vs Gaia G...")
    plot_class_star_vs_gmag(df, OUTPUT_DIR)

    # Step 3: Completeness-corrected distribution
    print("\n[Step 3] Completeness-corrected distribution...")
    plot_completeness_corrected(df, OUTPUT_DIR)

    # Step 3b: Corrected galaxy fraction
    print("\n[Step 3b] Completeness-corrected galaxy fraction...")
    plot_corrected_fraction(df, OUTPUT_DIR)

    # Combined summary
    print("\n[Summary] Combined 3-panel figure...")
    plot_combined_summary(df, OUTPUT_DIR)

    # Per-field summary table
    print("\n[Table] Saving per-field completeness summary...")
    rows = []
    for field_name in ERO_FIELDS:
        ra, dec, area = ERO_FIELDS[field_name]
        gal_b = get_galactic_b(ra, dec)
        fd = df[(df['field'] == field_name) & (df['label'] == 'GALAXY')
                & df['completeness'].notna()]
        n_gal_field = len(fd)
        if n_gal_field > 0:
            s_median = fd['completeness'].median()
            s_lo = fd['completeness'].quantile(0.16)
            s_hi = fd['completeness'].quantile(0.84)
            frac_below_50 = (fd['completeness'] < 0.5).mean()
        else:
            s_median = s_lo = s_hi = frac_below_50 = np.nan

        rows.append({
            'field': field_name,
            'ra': ra, 'dec': dec,
            'galactic_b': gal_b,
            'abs_b': abs(gal_b),
            'category': get_field_category(field_name),
            'n_galaxies': n_gal_field,
            'S_median': s_median,
            'S_16pct': s_lo,
            'S_84pct': s_hi,
            'frac_S_below_0.5': frac_below_50,
        })

    table = pd.DataFrame(rows).sort_values('abs_b')
    table.to_csv(OUTPUT_DIR / 'field_completeness_table.csv', index=False)
    print(f"  ✓ field_completeness_table.csv")

    print("\n" + "=" * 90)
    print("FIELD COMPLETENESS SUMMARY (per-source, galaxies only)")
    print("=" * 90)
    print(f"{'Field':<15} {'|b|':>5} {'Category':<18} {'N_gal':>6} "
          f"{'S_med':>6} {'S_16%':>6} {'S_84%':>6} {'%<0.5':>6}")
    print("-" * 90)
    for _, r in table.iterrows():
        print(f"{r['field']:<15} {r['abs_b']:>5.1f} {r['category']:<18} {r['n_galaxies']:>6} "
              f"{r['S_median']:>6.3f} {r['S_16pct']:>6.3f} {r['S_84pct']:>6.3f} "
              f"{r['frac_S_below_0.5']:>5.1%}")

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
