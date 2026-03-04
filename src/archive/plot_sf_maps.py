#!/usr/bin/env python3
"""
Plot per-field class_star maps with selection function backgrounds
using a GLOBAL color scale (same across all fields).

Generates:
  - per_field_class_star_with_sf.png     (M₁₀ background)
  - per_field_class_star_with_sg17.png   (S(G=17) background)
  - per_field_class_star_with_sg18.png   (S(G=18) background)
  - per_field_class_star_with_sg19.png   (S(G=19) background)

All saved to output/

Usage:
    python src/plot_sf_maps.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy_healpix as ah
import healpy as hp

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output'

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


def get_galactic_b(ra, dec):
    c = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    return c.galactic.b.deg


def _get_m10_at_coords(sf, coords):
    ra = coords.ra
    dec = coords.dec
    match_ipix = ah.lonlat_to_healpix(ra, dec, sf.max_nside, order="nested")
    i = sf.sorter[
        np.searchsorted(sf.index, match_ipix, side="right", sorter=sf.sorter) - 1
    ]
    return sf.m10map[i]["M10"]


def _draw_healpix_grid(ax, sf, ra_min, ra_max, dec_min, dec_max):
    """Draw actual multi-resolution HEALPix pixel boundaries from the SF map."""
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


def query_ero_data(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = """
        SELECT
            e_v_alpha_j2000 AS ra,
            e_v_delta_j2000 AS dec,
            g_phot_g_mean_mag AS phot_g_mean_mag,
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
    ra = df['ra'].values
    dec = df['dec'].values
    n_sources = len(df)
    field_assignments = np.full(n_sources, None, dtype=object)
    min_distances = np.full(n_sources, np.inf)

    for field_name, (field_ra, field_dec, area) in ERO_FIELDS.items():
        radius = np.sqrt(area / np.pi)
        ra1, dec1 = np.radians(ra), np.radians(dec)
        ra2, dec2 = np.radians(field_ra), np.radians(field_dec)
        dra, ddec = ra2 - ra1, dec2 - dec1
        a = np.sin(ddec/2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra/2)**2
        dist = np.degrees(2 * np.arcsin(np.sqrt(a)))
        mask = (dist < radius) & (dist < min_distances)
        field_assignments[mask] = field_name
        min_distances[mask] = dist[mask]

    df['field'] = field_assignments
    return df


def compute_field_grids(df, sf):
    """Pre-compute grid coordinates and M₁₀ values for each field (done once)."""
    df_field = df[df['field'].notna() & df['class_star'].notna()].copy()
    grids = {}

    for field_name in ERO_FIELDS:
        fd = df_field[df_field['field'] == field_name]
        if len(fd) == 0:
            grids[field_name] = None
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

        grids[field_name] = {
            'ra_grid': ra_grid, 'dec_grid': dec_grid,
            'ra_min': ra_min, 'ra_max': ra_max,
            'dec_min': dec_min, 'dec_max': dec_max,
            'coords_grid': coords_grid,
            'm10_grid': m10_grid,
            'fd': fd,
        }

    return grids


def plot_m10_map(df, sf, grids, output_dir):
    """Plot M₁₀ background with global color scale."""
    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4.5 * n_rows))
    axes = axes.flatten()
    cmap_src = plt.cm.RdYlBu
    norm_src = Normalize(vmin=0, vmax=1)
    cmap_bg = LinearSegmentedColormap.from_list(
        'gray_light', [(0.65, 0.65, 0.65), (1.0, 1.0, 1.0)])

    # Compute global M₁₀ range across all fields
    all_m10 = [g['m10_grid'] for g in grids.values() if g is not None]
    m10_global_min = min(np.nanmin(m) for m in all_m10)
    m10_global_max = max(np.nanmax(m) for m in all_m10)
    norm_bg = Normalize(vmin=m10_global_min, vmax=m10_global_max)

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        g = grids[field_name]
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])

        if g is None:
            ax.text(0.5, 0.5, 'No sources', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{field_name}', fontsize=10)
            continue

        fd = g['fd'].copy()
        im = ax.pcolormesh(g['ra_grid'], g['dec_grid'], g['m10_grid'],
                           cmap=cmap_bg, norm=norm_bg, shading='auto', rasterized=True)
        _draw_healpix_grid(ax, sf, g['ra_min'], g['ra_max'], g['dec_min'], g['dec_max'])

        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, aspect=20)
        cb.set_label('M₁₀', fontsize=7)
        cb.ax.tick_params(labelsize=6)

        fd = fd.sort_values('class_star', ascending=False)
        ax.scatter(fd['ra'], fd['dec'], c=fd['class_star'], cmap=cmap_src,
                   norm=norm_src, s=0.5, alpha=0.7, edgecolors='none', rasterized=True)

        n_gal = (fd['label'] == 'GALAXY').sum()
        n_star = (fd['label'] == 'STAR').sum()
        m10_min = np.nanmin(g['m10_grid'])
        m10_max = np.nanmax(g['m10_grid'])
        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)\n'
                     f'{n_star:,} stars, {n_gal:,} gal | '
                     f'M₁₀: {m10_min:.1f}–{m10_max:.1f}',
                     fontsize=8, fontweight='bold')
        ax.set_xlabel('RA [deg]', fontsize=8)
        ax.set_ylabel('Dec [deg]', fontsize=8)
        ax.set_xlim(g['ra_max'], g['ra_min'])
        ax.set_ylim(g['dec_min'], g['dec_max'])
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    cbar_ax1 = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    sm1 = ScalarMappable(cmap=cmap_src, norm=norm_src)
    cbar1 = fig.colorbar(sm1, cax=cbar_ax1)
    cbar1.set_label('class_star (0=galaxy, 1=star)', fontsize=10)

    plt.suptitle(f'ERO Fields: class_star (dots) with M₁₀ Background '
                 f'(global scale: {m10_global_min:.1f}–{m10_global_max:.1f})',
                 fontsize=14, fontweight='bold', y=1.00)
    fig.subplots_adjust(right=0.90)
    plt.tight_layout(rect=[0, 0, 0.90, 0.98])
    plt.savefig(output_dir / 'per_field_class_star_with_sf.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ per_field_class_star_with_sf.png")


def plot_sg_map(df, sf, grids, output_dir, g_eval):
    """Plot S(G) background with global 0–1 color scale."""
    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4.5 * n_rows))
    axes = axes.flatten()
    cmap_src = plt.cm.RdYlBu
    norm_src = Normalize(vmin=0, vmax=1)
    cmap_bg = LinearSegmentedColormap.from_list(
        'completeness', [(0.8, 0.2, 0.2), (1.0, 0.7, 0.3), (1.0, 1.0, 1.0)])
    norm_bg = Normalize(vmin=0.999, vmax=1)

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        g = grids[field_name]
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])

        if g is None:
            ax.text(0.5, 0.5, 'No sources', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{field_name}', fontsize=10)
            continue

        fd = g['fd'].copy()

        # Compute S(G) at grid points
        gmag_grid = np.full(len(g['coords_grid']), g_eval)
        sg_vals = sf.query(g['coords_grid'], gmag_grid)
        sg_grid = np.array(sg_vals, dtype=float).reshape(g['m10_grid'].shape)

        im = ax.pcolormesh(g['ra_grid'], g['dec_grid'], sg_grid,
                           cmap=cmap_bg, norm=norm_bg, shading='auto', rasterized=True)
        _draw_healpix_grid(ax, sf, g['ra_min'], g['ra_max'], g['dec_min'], g['dec_max'])

        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, aspect=20)
        cb.set_label(f'S(G={g_eval})', fontsize=7)
        cb.ax.tick_params(labelsize=6)

        fd = fd.sort_values('class_star', ascending=False)
        ax.scatter(fd['ra'], fd['dec'], c=fd['class_star'], cmap=cmap_src,
                   norm=norm_src, s=0.5, alpha=0.7, edgecolors='none', rasterized=True)

        n_gal = (fd['label'] == 'GALAXY').sum()
        n_star = (fd['label'] == 'STAR').sum()
        sg_min = np.nanmin(sg_grid)
        sg_max = np.nanmax(sg_grid)
        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)\n'
                     f'{n_star:,} stars, {n_gal:,} gal | '
                     f'S(G={g_eval}): {sg_min:.2f}–{sg_max:.2f}',
                     fontsize=8, fontweight='bold')
        ax.set_xlabel('RA [deg]', fontsize=8)
        ax.set_ylabel('Dec [deg]', fontsize=8)
        ax.set_xlim(g['ra_max'], g['ra_min'])
        ax.set_ylim(g['dec_min'], g['dec_max'])
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    cbar_ax1 = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    sm1 = ScalarMappable(cmap=cmap_src, norm=norm_src)
    cbar1 = fig.colorbar(sm1, cax=cbar_ax1)
    cbar1.set_label('class_star (0=galaxy, 1=star)', fontsize=10)

    plt.suptitle(f'ERO Fields: class_star (dots) with S(G={g_eval}) Background (global 0–1 scale)',
                 fontsize=14, fontweight='bold', y=1.00)
    fig.subplots_adjust(right=0.90)
    plt.tight_layout(rect=[0, 0, 0.90, 0.98])
    fname = f'per_field_class_star_with_sg{int(g_eval)}.png'
    plt.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {fname}")


def plot_gmag_map(df, sf, grids, output_dir):
    """Per-field scatter colored by Gaia G magnitude with HEALPix grid overlay."""
    df_field = df[df['field'].notna() & df['phot_g_mean_mag'].notna()].copy()

    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4.5 * n_rows))
    axes = axes.flatten()
    # Bright = yellow, faint = dark purple
    cmap_gmag = plt.cm.summer
    norm_gmag = Normalize(vmin=14, vmax=21)

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        g = grids[field_name]
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])

        if g is None:
            ax.text(0.5, 0.5, 'No sources', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{field_name}', fontsize=10)
            continue

        fd = df_field[df_field['field'] == field_name].copy()
        if len(fd) == 0:
            continue

        # Draw HEALPix grid first (underneath)
        _draw_healpix_grid(ax, sf, g['ra_min'], g['ra_max'], g['dec_min'], g['dec_max'])

        # Sort so faintest sources are plotted first, brightest on top
        fd = fd.sort_values('phot_g_mean_mag', ascending=False)

        sc = ax.scatter(fd['ra'], fd['dec'], c=fd['phot_g_mean_mag'], cmap=cmap_gmag,
                        norm=norm_gmag, s=0.5, alpha=0.7, edgecolors='none',
                        rasterized=True)

        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, aspect=20)
        cb.set_label('Gaia G', fontsize=7)
        cb.ax.tick_params(labelsize=6)

        n_src = len(fd)
        g_med = fd['phot_g_mean_mag'].median()
        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)\n'
                     f'{n_src:,} sources | median G={g_med:.1f}',
                     fontsize=8, fontweight='bold')
        ax.set_xlabel('RA [deg]', fontsize=8)
        ax.set_ylabel('Dec [deg]', fontsize=8)
        ax.set_xlim(g['ra_max'], g['ra_min'])
        ax.set_ylim(g['dec_min'], g['dec_max'])
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    sm = ScalarMappable(cmap=cmap_gmag, norm=norm_gmag)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Gaia G magnitude', fontsize=10)

    plt.suptitle('ERO Fields: Spatial Distribution Colored by Gaia G Magnitude',
                 fontsize=14, fontweight='bold', y=1.00)
    fig.subplots_adjust(right=0.90)
    plt.tight_layout(rect=[0, 0, 0.90, 0.98])
    plt.savefig(output_dir / 'per_field_gaia_gmag_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ per_field_gaia_gmag_map.png")


def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Selection Function Maps (global color scale)")
    print("=" * 70)

    # Query data
    print("\n[1] Querying data...")
    df = query_ero_data(username)
    df = assign_fields(df)
    n_assigned = df['field'].notna().sum()
    print(f"  Assigned to fields: {n_assigned:,} sources")

    # Load selection function
    print("\n[2] Loading selection function...")
    from gaiaunlimited import DR3SelectionFunctionTCG
    sf = DR3SelectionFunctionTCG(mode='multi')

    # Pre-compute grids (done once, reused for all plots)
    print("\n[3] Computing M₁₀ grids...")
    grids = compute_field_grids(df, sf)

    # Generate plots
    print("\n[4] Generating plots...")
    plot_m10_map(df, sf, grids, OUTPUT_DIR)
    plot_gmag_map(df, sf, grids, OUTPUT_DIR)
    for g_val in [16, 18, 20]:
        plot_sg_map(df, sf, grids, OUTPUT_DIR, g_eval=g_val)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
