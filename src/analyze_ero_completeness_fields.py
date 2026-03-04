#!/usr/bin/env python3
"""
ERO completeness analysis — produces:
  1. per_field_class_star_map.png
  2. per_field_class_star_map_ero_all.png
  3. per_field_class_star_with_sf.png
  4. per_field_class_star_with_sg{G}.png  (G=17,18,19,20)

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_ero_completeness_simple.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output' / 'ero_completeness'

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


def get_galactic_b(ra, dec):
    c = SkyCoord(ra=ra, dec=dec, unit='deg')
    return c.galactic.b.deg


# =============================================================================
# Data querying
# =============================================================================

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


def query_all_ero(username):
    """Query ALL ERO sources from euclid_ero.merge_cat."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = """
        SELECT
            v_alpha_j2000 AS ra,
            v_delta_j2000 AS dec,
            v_mag_auto,
            v_class_star AS class_star,
            CASE
                WHEN v_class_star >= 0.65 THEN 'STAR'
                WHEN v_class_star < 0.65 THEN 'GALAXY'
            END as label
        FROM euclid_ero.merge_cat
        WHERE v_class_star IS NOT NULL
    """
    print("  Querying ALL ERO sources...")
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
# Per-source completeness
# =============================================================================

def compute_per_source_completeness(df):
    """Compute S(G) at each source's actual RA/Dec using GaiaUnlimited."""
    from gaiaunlimited import DR3SelectionFunctionTCG

    print("  Loading GaiaUnlimited DR3 selection function (mode='multi')...")
    sf = DR3SelectionFunctionTCG(mode='multi')

    mask = df['field'].notna() & df['phot_g_mean_mag'].notna()
    idx = df.index[mask]

    ra_arr = df.loc[idx, 'ra'].values
    dec_arr = df.loc[idx, 'dec'].values
    gmag_arr = df.loc[idx, 'phot_g_mean_mag'].values.astype(float)

    n = len(idx)
    print(f"  Querying S(G) for {n:,} sources...")

    batch_size = 50000
    completeness = np.full(n, np.nan)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        coords = SkyCoord(ra=ra_arr[start:end], dec=dec_arr[start:end], unit='deg')
        completeness[start:end] = sf.query(coords, gmag_arr[start:end])
        print(f"    Batch {start:,}-{end:,} done")

    df['completeness'] = np.nan
    df.loc[idx, 'completeness'] = completeness

    print(f"  ✓ Completeness computed for {n:,} sources")
    print(f"    Median S(G): {np.nanmedian(completeness):.3f}")
    print(f"    Sources with S(G) < 0.5: {(completeness < 0.5).sum():,}")

    return sf


# =============================================================================
# Plot 1: per_field_by_class.png
# =============================================================================



# =============================================================================
# Plot 2: per_field_class_star_map.png
# =============================================================================

def plot_per_field_class_star_maps(df, output_dir):
    """Per-field spatial maps colored by class_star value (Gaia-matched)."""
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    df_field = df[df['field'].notna() & df['class_star'].notna()].copy()

    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    cmap = plt.cm.RdYlBu
    norm = Normalize(vmin=0, vmax=1)

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        fd = df_field[df_field['field'] == field_name].copy()
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])

        if len(fd) == 0:
            ax.text(0.5, 0.5, 'No sources', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{field_name}', fontsize=10)
            continue

        fd = fd.sort_values('class_star', ascending=False)
        ax.scatter(fd['ra'], fd['dec'], c=fd['class_star'], cmap=cmap,
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


# =============================================================================
# Plot 3: per_field_class_star_map_ero_all.png
# =============================================================================

def plot_per_field_all_ero(df_ero_all, output_dir):
    """Per-field spatial maps colored by class_star for ALL ERO sources."""
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    df_field = df_ero_all[df_ero_all['field'].notna() & df_ero_all['class_star'].notna()].copy()

    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    cmap = plt.cm.RdYlBu
    norm = Normalize(vmin=0, vmax=1)

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        fd = df_field[df_field['field'] == field_name].copy()
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])

        if len(fd) == 0:
            ax.text(0.5, 0.5, 'No sources', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{field_name}', fontsize=10)
            continue

        fd = fd.sort_values('class_star', ascending=False)
        n_gal = (fd['label'] == 'GALAXY').sum()
        n_star = (fd['label'] == 'STAR').sum()

        ax.scatter(fd['ra'], fd['dec'], c=fd['class_star'], cmap=cmap,
                   norm=norm, s=0.15, alpha=0.5, edgecolors='none',
                   rasterized=True)

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)\n'
                     f'{n_star:,} stars, {n_gal:,} galaxies  ({len(fd):,} total)',
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
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('class_star (0=galaxy, 1=star)', fontsize=11)

    plt.suptitle('ERO Fields: ALL ERO Sources Colored by class_star',
                 fontsize=14, fontweight='bold', y=1.00)
    fig.subplots_adjust(right=0.90)
    plt.tight_layout(rect=[0, 0, 0.91, 0.98])
    plt.savefig(output_dir / 'per_field_class_star_map_ero_all.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ per_field_class_star_map_ero_all.png")


# =============================================================================
# Plot 4: per_field_class_star_with_sf.png
# =============================================================================

def _get_m10_at_coords(sf, coords):
    """Extract M10 values at given coordinates from the multi-resolution SF map."""
    import astropy_healpix as ah
    ra = coords.ra
    dec = coords.dec
    match_ipix = ah.lonlat_to_healpix(ra, dec, sf.max_nside, order="nested")
    i = sf.sorter[
        np.searchsorted(sf.index, match_ipix, side="right", sorter=sf.sorter) - 1
    ]
    return sf.m10map[i]["M10"]


def _draw_healpix_grid(ax, sf, ra_min, ra_max, dec_min, dec_max):
    """Draw actual multi-resolution HEALPix pixel boundaries from the SF map."""
    import astropy_healpix as ah
    import healpy as hp

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
    """Per-field class_star scatter with M10 as background colormap."""
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable

    df_field = df[df['field'].notna() & df['class_star'].notna()].copy()

    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4.5 * n_rows))
    axes = axes.flatten()
    cmap_src = plt.cm.RdYlBu
    norm_src = Normalize(vmin=0, vmax=1)
    cmap_bg = LinearSegmentedColormap.from_list(
        'gray_light', [(0.65, 0.65, 0.65), (1.0, 1.0, 1.0)])

    print(f"  Computing M10 background grids for {n_fields} fields...")

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        fd = df_field[df_field['field'] == field_name].copy()
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])

        if len(fd) == 0:
            ax.text(0.5, 0.5, 'No sources', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{field_name}', fontsize=10)
            continue

        ra_min, ra_max = fd['ra'].min(), fd['ra'].max()
        dec_min, dec_max = fd['dec'].min(), fd['dec'].max()
        ra_pad = (ra_max - ra_min) * 0.05
        dec_pad = (dec_max - dec_min) * 0.05
        ra_min -= ra_pad; ra_max += ra_pad
        dec_min -= dec_pad; dec_max += dec_pad

        n_grid = 100
        ra_grid = np.linspace(ra_min, ra_max, n_grid)
        dec_grid = np.linspace(dec_min, dec_max, n_grid)
        ra_mesh, dec_mesh = np.meshgrid(ra_grid, dec_grid)

        coords_grid = SkyCoord(ra=ra_mesh.ravel(), dec=dec_mesh.ravel(), unit='deg')
        m10_vals = _get_m10_at_coords(sf, coords_grid)
        m10_grid = np.array(m10_vals, dtype=float).reshape(ra_mesh.shape)

        m10_min_field = np.nanmin(m10_grid)
        m10_max_field = np.nanmax(m10_grid)
        if m10_max_field - m10_min_field < 0.01:
            m10_min_field -= 0.005
            m10_max_field += 0.005
        norm_bg_field = Normalize(vmin=m10_min_field, vmax=m10_max_field)

        im = ax.pcolormesh(ra_grid, dec_grid, m10_grid, cmap=cmap_bg,
                           norm=norm_bg_field, shading='auto', rasterized=True)
        _draw_healpix_grid(ax, sf, ra_min, ra_max, dec_min, dec_max)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, aspect=20)
        cb.set_label('M\u2081\u2080', fontsize=7)
        cb.ax.tick_params(labelsize=6)

        fd = fd.sort_values('class_star', ascending=False)
        ax.scatter(fd['ra'], fd['dec'], c=fd['class_star'], cmap=cmap_src,
                   norm=norm_src, s=0.5, alpha=0.7, edgecolors='none',
                   rasterized=True)

        n_gal = (fd['label'] == 'GALAXY').sum()
        n_star = (fd['label'] == 'STAR').sum()
        m10_min = np.nanmin(m10_grid)
        m10_max = np.nanmax(m10_grid)
        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}\u00b0)\n'
                     f'{n_star:,} stars, {n_gal:,} gal | '
                     f'M\u2081\u2080: {m10_min:.1f}\u2013{m10_max:.1f}',
                     fontsize=8, fontweight='bold')
        ax.set_xlabel('RA [deg]', fontsize=8)
        ax.set_ylabel('Dec [deg]', fontsize=8)
        ax.set_xlim(ra_max, ra_min)
        ax.set_ylim(dec_min, dec_max)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    cbar_ax1 = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    sm1 = ScalarMappable(cmap=cmap_src, norm=norm_src)
    cbar1 = fig.colorbar(sm1, cax=cbar_ax1)
    cbar1.set_label('class_star (0=galaxy, 1=star)', fontsize=10)

    plt.suptitle('ERO Fields: class_star (dots) with M\u2081\u2080 Background (per-field scale)',
                 fontsize=14, fontweight='bold', y=1.00)
    fig.subplots_adjust(right=0.90)
    plt.tight_layout(rect=[0, 0, 0.90, 0.98])
    plt.savefig(output_dir / 'per_field_class_star_with_m10.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ per_field_class_star_with_sf.png")


# =============================================================================
# Plot 4b: per_field_class_star_with_sg{G}.png
# =============================================================================

def plot_class_star_with_sg_background(df, sf, output_dir, g_eval=18.0):
    """Per-field class_star scatter with S(G) completeness as background."""
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable

    G_EVAL = g_eval
    df_field = df[df['field'].notna() & df['class_star'].notna()].copy()

    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4.5 * n_rows))
    axes = axes.flatten()
    cmap_src = plt.cm.RdYlBu
    norm_src = Normalize(vmin=0, vmax=1)
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

        ra_min, ra_max = fd['ra'].min(), fd['ra'].max()
        dec_min, dec_max = fd['dec'].min(), fd['dec'].max()
        ra_pad = (ra_max - ra_min) * 0.05
        dec_pad = (dec_max - dec_min) * 0.05
        ra_min -= ra_pad; ra_max += ra_pad
        dec_min -= dec_pad; dec_max += dec_pad

        n_grid = 100
        ra_grid = np.linspace(ra_min, ra_max, n_grid)
        dec_grid = np.linspace(dec_min, dec_max, n_grid)
        ra_mesh, dec_mesh = np.meshgrid(ra_grid, dec_grid)

        coords_grid = SkyCoord(ra=ra_mesh.ravel(), dec=dec_mesh.ravel(), unit='deg')
        gmag_grid = np.full(len(coords_grid), G_EVAL)
        sg_vals = sf.query(coords_grid, gmag_grid)
        sg_grid = np.array(sg_vals, dtype=float).reshape(ra_mesh.shape)

        # Fixed global scale so background reflects true completeness
        norm_bg = Normalize(vmin=0.9990, vmax=1)

        im = ax.pcolormesh(ra_grid, dec_grid, sg_grid, cmap=cmap_bg,
                           norm=norm_bg, shading='auto', rasterized=True)
        _draw_healpix_grid(ax, sf, ra_min, ra_max, dec_min, dec_max)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, aspect=20)
        cb.set_label(f'S(G={int(G_EVAL)})', fontsize=7)
        cb.ax.tick_params(labelsize=6)

        fd = fd.sort_values('class_star', ascending=False)
        ax.scatter(fd['ra'], fd['dec'], c=fd['class_star'], cmap=cmap_src,
                   norm=norm_src, s=0.5, alpha=0.7, edgecolors='none',
                   rasterized=True)

        n_gal = (fd['label'] == 'GALAXY').sum()
        n_star = (fd['label'] == 'STAR').sum()
        sg_min = np.nanmin(sg_grid)
        sg_max = np.nanmax(sg_grid)
        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}\u00b0)\n'
                     f'{n_star:,} stars, {n_gal:,} gal | '
                     f'S(G={int(G_EVAL)}): {sg_min:.2f}\u2013{sg_max:.2f}',
                     fontsize=8, fontweight='bold')
        ax.set_xlabel('RA [deg]', fontsize=8)
        ax.set_ylabel('Dec [deg]', fontsize=8)
        ax.set_xlim(ra_max, ra_min)
        ax.set_ylim(dec_min, dec_max)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    cbar_ax1 = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    sm1 = ScalarMappable(cmap=cmap_src, norm=norm_src)
    cbar1 = fig.colorbar(sm1, cax=cbar_ax1)
    cbar1.set_label('class_star (0=galaxy, 1=star)', fontsize=10)

    plt.suptitle(f'ERO Fields: class_star (dots) with S(G={int(G_EVAL)}) Background (global 0-1 scale)',
                 fontsize=14, fontweight='bold', y=1.00)
    fig.subplots_adjust(right=0.90)
    plt.tight_layout(rect=[0, 0, 0.90, 0.98])
    fname = f'per_field_class_star_with_sg{int(G_EVAL)}.png'
    plt.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  \u2713 {fname}")



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
    print("ERO Completeness Analysis (simplified)")
    print("=" * 70)

    # Query Gaia-matched ERO data
    print("\n[1] Querying Gaia-matched ERO data...")
    df = query_ero_data(username)
    df = assign_fields(df)
    n_assigned = df['field'].notna().sum()
    print(f"  Assigned to fields: {n_assigned:,} sources")

    # Query all ERO sources
    print("\n[2] Querying all ERO sources...")
    df_ero_all = query_all_ero(username)
    df_ero_all = assign_fields(df_ero_all)
    n_ero_assigned = df_ero_all['field'].notna().sum()
    print(f"  Assigned to fields: {n_ero_assigned:,} / {len(df_ero_all):,}")

    # Plot 2: per_field_class_star_map.png
    print("\n[4] Per-field class_star spatial maps (Gaia-matched)...")
    plot_per_field_class_star_maps(df, OUTPUT_DIR)

    # Plot 3: per_field_class_star_map_ero_all.png
    print("\n[5] Per-field class_star spatial maps (all ERO)...")
    plot_per_field_all_ero(df_ero_all, OUTPUT_DIR)

    # Compute per-source completeness (needed for plots 4 & 5)
    print("\n[6] Computing per-source completeness...")
    sf = compute_per_source_completeness(df)

    # Plot 4: per_field_class_star_with_sf.png
    print("\n[7] class_star maps with M10 SF background...")
    plot_class_star_with_sf_background(df, sf, OUTPUT_DIR)

    # Plot 4b: class_star maps with S(G) background for G=16,17,18,19
    for g_val in [17, 18, 19, 20]:
        print(f"\n[7b] class_star maps with S(G={g_val}) background...")
        plot_class_star_with_sg_background(df, sf, OUTPUT_DIR, g_eval=g_val)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
