#!/usr/bin/env python3
"""
Q1 completeness analysis — produces:
  1. q1_per_field_phz_map.png          — spatial maps coloured by PHZ label
  2. q1_per_field_phz_map_all.png      — spatial maps for all Q1 sources
  3. q1_per_field_phz_with_m10.png     — PHZ maps with M10 diagnostic background
  4. q1_per_field_phz_with_sg{G}.png   — PHZ maps with S(G) diagnostic background (G=17,18,19,20)

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_q1_completeness_fields.py
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
OUTPUT_DIR  = PROJECT_DIR / 'output' / 'q1_completeness'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

ZPC_IE = 1.05
ZP_REF = 23.9

Q1_FIELDS = {
    'EDF-North':  (269.733, +66.018, 22.9),
    'EDF-South':  ( 61.241, -48.423, 28.1),
    'EDF-Fornax': ( 52.932, -28.088, 12.1),
}

FIELD_COLOR = {
    'EDF-North':  '#1f77b4',
    'EDF-South':  '#ff7f0e',
    'EDF-Fornax': '#2ca02c',
}


def get_galactic_b(ra, dec):
    c = SkyCoord(ra=ra, dec=dec, unit='deg')
    return c.galactic.b.deg


# =============================================================================
# Data queries
# =============================================================================

def query_gaia_matched(username):
    """Query Gaia-matched Q1+PHZ sources with I_E, Gaia G, RA/Dec, and PHZ label."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = f"""
        SELECT
            m_right_ascension AS ra,
            m_declination     AS dec,
            -2.5 * LOG10({ZPC_IE} * m_flux_detection_total) + {ZP_REF} AS ie_mag,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            p_phz_star_prob   AS star_prob,
            CASE
                WHEN p_phz_classification = 1 THEN 'STAR'
                WHEN p_phz_classification = 2 THEN 'GALAXY'
                ELSE 'UNCLASSIFIED'
            END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia
        WHERE m_flux_detection_total > 0
          AND m_flux_detection_total IS NOT NULL
    """
    print("  Querying Gaia-matched Q1 data...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} Gaia-matched sources")
    return df


def query_all_q1(username):
    """Query all Q1+PHZ VIS-detected sources with I_E, RA/Dec, and PHZ label."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = f"""
        SELECT
            m_right_ascension AS ra,
            m_declination     AS dec,
            -2.5 * LOG10({ZPC_IE} * m_flux_detection_total) + {ZP_REF} AS ie_mag,
            p_phz_star_prob   AS star_prob,
            CASE
                WHEN p_phz_classification = 1 THEN 'STAR'
                WHEN p_phz_classification = 2 THEN 'GALAXY'
                ELSE 'UNCLASSIFIED'
            END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis
        WHERE m_flux_detection_total > 0
          AND m_flux_detection_total IS NOT NULL
    """
    print("  Querying all Q1 data...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} Q1 sources")
    return df


def assign_fields(df):
    """Assign each source to its nearest Q1 field (nearest-field, no cutoff)."""
    ra  = df['ra'].values
    dec = df['dec'].values
    field_assignments = np.full(len(df), None, dtype=object)
    min_distances     = np.full(len(df), np.inf)

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


# =============================================================================
# Per-source completeness
# =============================================================================

def compute_per_source_completeness(df):
    """Compute S(G) at each source's actual RA/Dec using GaiaUnlimited."""
    from gaiaunlimited import DR3SelectionFunctionTCG

    print("  Loading GaiaUnlimited DR3 selection function (mode='multi')...")
    sf = DR3SelectionFunctionTCG(mode='multi')

    mask = df['field'].notna() & df['phot_g_mean_mag'].notna()
    idx  = df.index[mask]

    ra_arr   = df.loc[idx, 'ra'].values
    dec_arr  = df.loc[idx, 'dec'].values
    gmag_arr = df.loc[idx, 'phot_g_mean_mag'].values.astype(float)

    n = len(idx)
    print(f"  Querying S(G) for {n:,} sources...")

    batch_size   = 50000
    completeness = np.full(n, np.nan)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        coords = SkyCoord(ra=ra_arr[start:end], dec=dec_arr[start:end], unit='deg')
        completeness[start:end] = sf.query(coords, gmag_arr[start:end])
        print(f"    Batch {start:,}–{end:,} done")

    df['completeness'] = np.nan
    df.loc[idx, 'completeness'] = completeness
    print(f"  ✓ Completeness computed  |  Median S(G): {np.nanmedian(completeness):.3f}")
    return sf


# =============================================================================
# Plot helpers
# =============================================================================

LABEL_COLORS = {'STAR': 'coral', 'GALAXY': 'steelblue', 'UNCLASSIFIED': 'gray'}


def _get_m10_at_coords(sf, coords):
    import astropy_healpix as ah
    ra  = coords.ra
    dec = coords.dec
    match_ipix = ah.lonlat_to_healpix(ra, dec, sf.max_nside, order="nested")
    i = sf.sorter[
        np.searchsorted(sf.index, match_ipix, side="right", sorter=sf.sorter) - 1
    ]
    return sf.m10map[i]["M10"]


def _draw_healpix_grid(ax, sf, ra_min, ra_max, dec_min, dec_max):
    import astropy_healpix as ah
    import healpy as hp

    n_sample = 60
    ra_s  = np.linspace(ra_min, ra_max, n_sample)
    dec_s = np.linspace(dec_min, dec_max, n_sample)
    ra_m, dec_m = np.meshgrid(ra_s, dec_s)
    coords_s = SkyCoord(ra=ra_m.ravel(), dec=dec_m.ravel(), unit='deg')

    match_ipix = ah.lonlat_to_healpix(
        coords_s.ra, coords_s.dec, sf.max_nside, order="nested")
    idx_arr    = sf.sorter[
        np.searchsorted(sf.index, match_ipix, side="right", sorter=sf.sorter) - 1
    ]
    unique_idx = np.unique(idx_arr)
    level, ipix = ah.uniq_to_level_ipix(sf.m10map[unique_idx]["UNIQ"])

    for lev, ip in zip(level, ipix):
        nside = ah.level_to_nside(lev)
        boundaries = hp.boundaries(nside, ip, step=16, nest=True)
        theta, phi = hp.vec2ang(boundaries.T)
        ra_b  = np.degrees(phi)
        dec_b = 90.0 - np.degrees(theta)
        ra_b  = np.append(ra_b,  ra_b[0])
        dec_b = np.append(dec_b, dec_b[0])
        if (ra_b.max() < ra_min or ra_b.min() > ra_max or
                dec_b.max() < dec_min or dec_b.min() > dec_max):
            continue
        ax.plot(ra_b, dec_b, color='gray', linewidth=0.3, alpha=0.4)


# =============================================================================
# Plot 1: spatial maps coloured by PHZ label (Gaia-matched)
# =============================================================================

def plot_per_field_phz_maps(df, output_dir):
    """1×3 spatial maps: dots coloured by PHZ label (STAR/GALAXY/UNCLASSIFIED)."""
    from matplotlib.lines import Line2D

    df_field = df[df['field'].notna()].copy()
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    for ax, field_name in zip(axes, Q1_FIELDS):
        fd = df_field[df_field['field'] == field_name].copy()
        gal_b = get_galactic_b(*Q1_FIELDS[field_name][:2])

        if len(fd) == 0:
            ax.text(0.5, 0.5, 'No sources', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(field_name, fontsize=10)
            continue

        # Plot UNCLASSIFIED first, then GALAXY, then STAR on top
        for lbl in ['UNCLASSIFIED', 'GALAXY', 'STAR']:
            sub = fd[fd['label'] == lbl]
            if len(sub):
                ax.scatter(sub['ra'], sub['dec'],
                           c=LABEL_COLORS[lbl], s=0.3, alpha=0.5,
                           label=lbl, rasterized=True)

        n_star = (fd['label'] == 'STAR').sum()
        n_gal  = (fd['label'] == 'GALAXY').sum()
        n_unc  = (fd['label'] == 'UNCLASSIFIED').sum()
        ax.set_title(f'{field_name}  (|b|={abs(gal_b):.1f}°)\n'
                     f'Stars {n_star:,}  Galaxies {n_gal:,}  Unclassified {n_unc:,}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('RA [deg]', fontsize=9)
        ax.set_ylabel('Dec [deg]', fontsize=9)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=8)
        ax.invert_xaxis()
        ax.grid(alpha=0.2)

    legend_handles = [Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=LABEL_COLORS[l], markersize=7, label=l)
                      for l in ['STAR', 'GALAXY', 'UNCLASSIFIED']]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('Q1 Fields: Spatial Distribution Coloured by PHZ Label (Gaia-matched)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = output_dir / 'q1_per_field_phz_map.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


# =============================================================================
# Plot 2: spatial maps for all Q1 sources
# =============================================================================

def plot_per_field_phz_maps_all(df_all, output_dir):
    """1×3 spatial maps for all Q1 sources (not just Gaia-matched)."""
    from matplotlib.lines import Line2D

    df_field = df_all[df_all['field'].notna()].copy()
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    for ax, field_name in zip(axes, Q1_FIELDS):
        fd = df_field[df_field['field'] == field_name].copy()
        gal_b = get_galactic_b(*Q1_FIELDS[field_name][:2])

        if len(fd) == 0:
            ax.text(0.5, 0.5, 'No sources', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(field_name, fontsize=10)
            continue

        for lbl in ['UNCLASSIFIED', 'GALAXY', 'STAR']:
            sub = fd[fd['label'] == lbl]
            if len(sub):
                ax.scatter(sub['ra'], sub['dec'],
                           c=LABEL_COLORS[lbl], s=0.15, alpha=0.4,
                           label=lbl, rasterized=True)

        n_star = (fd['label'] == 'STAR').sum()
        n_gal  = (fd['label'] == 'GALAXY').sum()
        n_unc  = (fd['label'] == 'UNCLASSIFIED').sum()
        ax.set_title(f'{field_name}  (|b|={abs(gal_b):.1f}°)\n'
                     f'Stars {n_star:,}  Galaxies {n_gal:,}  Unclassified {n_unc:,}  '
                     f'Total {len(fd):,}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('RA [deg]', fontsize=9)
        ax.set_ylabel('Dec [deg]', fontsize=9)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=8)
        ax.invert_xaxis()
        ax.grid(alpha=0.2)

    legend_handles = [Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=LABEL_COLORS[l], markersize=7, label=l)
                      for l in ['STAR', 'GALAXY', 'UNCLASSIFIED']]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('Q1 Fields: ALL Q1 Sources Coloured by PHZ Label',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = output_dir / 'q1_per_field_phz_map_all.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


# =============================================================================
# Plot 3: PHZ maps with M10 background
# =============================================================================

def plot_phz_with_m10_background(df, sf, output_dir):
    """1×3: PHZ label scatter with M10 as a diagnostic background only."""
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D

    df_field = df[df['field'].notna()].copy()
    cmap_bg  = LinearSegmentedColormap.from_list(
        'gray_light', [(0.65, 0.65, 0.65), (1.0, 1.0, 1.0)])

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    print(f"  Computing M10 background grids for {len(Q1_FIELDS)} fields...")

    for ax, field_name in zip(axes, Q1_FIELDS):
        fd = df_field[df_field['field'] == field_name].copy()
        gal_b = get_galactic_b(*Q1_FIELDS[field_name][:2])

        if len(fd) == 0:
            ax.text(0.5, 0.5, 'No sources', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(field_name, fontsize=10)
            continue

        ra_min, ra_max   = fd['ra'].min(),  fd['ra'].max()
        dec_min, dec_max = fd['dec'].min(), fd['dec'].max()
        ra_pad  = (ra_max  - ra_min)  * 0.03
        dec_pad = (dec_max - dec_min) * 0.03
        ra_min -= ra_pad;  ra_max  += ra_pad
        dec_min -= dec_pad; dec_max += dec_pad

        n_grid   = 100
        ra_grid  = np.linspace(ra_min,  ra_max,  n_grid)
        dec_grid = np.linspace(dec_min, dec_max, n_grid)
        ra_mesh, dec_mesh = np.meshgrid(ra_grid, dec_grid)
        coords_grid = SkyCoord(ra=ra_mesh.ravel(), dec=dec_mesh.ravel(), unit='deg')
        m10_vals = _get_m10_at_coords(sf, coords_grid)
        m10_grid = np.array(m10_vals, dtype=float).reshape(ra_mesh.shape)

        m10_lo, m10_hi = np.nanmin(m10_grid), np.nanmax(m10_grid)
        if m10_hi - m10_lo < 0.01:
            m10_lo -= 0.005; m10_hi += 0.005
        norm_bg = Normalize(vmin=m10_lo, vmax=m10_hi)

        im = ax.pcolormesh(ra_grid, dec_grid, m10_grid, cmap=cmap_bg,
                           norm=norm_bg, shading='auto', rasterized=True)
        _draw_healpix_grid(ax, sf, ra_min, ra_max, dec_min, dec_max)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, aspect=20)
        cb.set_label('M₁₀', fontsize=7)
        cb.ax.tick_params(labelsize=6)

        for lbl in ['UNCLASSIFIED', 'GALAXY', 'STAR']:
            sub = fd[fd['label'] == lbl]
            if len(sub):
                ax.scatter(sub['ra'], sub['dec'],
                           c=LABEL_COLORS[lbl], s=0.5, alpha=0.7,
                           rasterized=True)

        n_star = (fd['label'] == 'STAR').sum()
        n_gal  = (fd['label'] == 'GALAXY').sum()
        ax.set_title(f'{field_name}  (|b|={abs(gal_b):.1f}°)\n'
                     f'Stars {n_star:,}  Galaxies {n_gal:,}  |  '
                     f'M₁₀: {m10_lo:.1f}–{m10_hi:.1f}',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('RA [deg]', fontsize=9)
        ax.set_ylabel('Dec [deg]', fontsize=9)
        ax.set_xlim(ra_max, ra_min)
        ax.set_ylim(dec_min, dec_max)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=8)

    legend_handles = [Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=LABEL_COLORS[l], markersize=7, label=l)
                      for l in ['STAR', 'GALAXY', 'UNCLASSIFIED']]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('Q1 Fields: PHZ Label (dots) with M₁₀ Diagnostic Background (per-field scale)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = output_dir / 'q1_per_field_phz_with_m10.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


# =============================================================================
# Plot 4: PHZ maps with S(G) background
# =============================================================================

def plot_phz_with_sg_background(df, sf, output_dir, g_eval=18.0):
    """1×3: PHZ label scatter with S(G) as a diagnostic background only."""
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    from matplotlib.lines import Line2D

    G_EVAL   = g_eval
    df_field = df[df['field'].notna()].copy()
    cmap_bg  = LinearSegmentedColormap.from_list(
        'completeness', [(0.8, 0.2, 0.2), (1.0, 0.7, 0.3), (1.0, 1.0, 1.0)])
    norm_bg  = Normalize(vmin=0.9990, vmax=1)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    print(f"  Computing S(G={G_EVAL}) background grids for {len(Q1_FIELDS)} fields...")

    for ax, field_name in zip(axes, Q1_FIELDS):
        fd = df_field[df_field['field'] == field_name].copy()
        gal_b = get_galactic_b(*Q1_FIELDS[field_name][:2])

        if len(fd) == 0:
            ax.text(0.5, 0.5, 'No sources', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(field_name, fontsize=10)
            continue

        ra_min, ra_max   = fd['ra'].min(),  fd['ra'].max()
        dec_min, dec_max = fd['dec'].min(), fd['dec'].max()
        ra_pad  = (ra_max  - ra_min)  * 0.03
        dec_pad = (dec_max - dec_min) * 0.03
        ra_min -= ra_pad;  ra_max  += ra_pad
        dec_min -= dec_pad; dec_max += dec_pad

        n_grid   = 100
        ra_grid  = np.linspace(ra_min,  ra_max,  n_grid)
        dec_grid = np.linspace(dec_min, dec_max, n_grid)
        ra_mesh, dec_mesh = np.meshgrid(ra_grid, dec_grid)
        coords_grid = SkyCoord(ra=ra_mesh.ravel(), dec=dec_mesh.ravel(), unit='deg')
        gmag_grid   = np.full(len(coords_grid), G_EVAL)
        sg_vals  = sf.query(coords_grid, gmag_grid)
        sg_grid  = np.array(sg_vals, dtype=float).reshape(ra_mesh.shape)

        im = ax.pcolormesh(ra_grid, dec_grid, sg_grid, cmap=cmap_bg,
                           norm=norm_bg, shading='auto', rasterized=True)
        _draw_healpix_grid(ax, sf, ra_min, ra_max, dec_min, dec_max)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, aspect=20)
        cb.set_label(f'S(G={int(G_EVAL)})', fontsize=7)
        cb.ax.tick_params(labelsize=6)

        for lbl in ['UNCLASSIFIED', 'GALAXY', 'STAR']:
            sub = fd[fd['label'] == lbl]
            if len(sub):
                ax.scatter(sub['ra'], sub['dec'],
                           c=LABEL_COLORS[lbl], s=0.5, alpha=0.7,
                           rasterized=True)

        n_star = (fd['label'] == 'STAR').sum()
        n_gal  = (fd['label'] == 'GALAXY').sum()
        sg_min, sg_max = np.nanmin(sg_grid), np.nanmax(sg_grid)
        ax.set_title(f'{field_name}  (|b|={abs(gal_b):.1f}°)\n'
                     f'Stars {n_star:,}  Galaxies {n_gal:,}  |  '
                     f'S(G={int(G_EVAL)}): {sg_min:.4f}–{sg_max:.4f}',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('RA [deg]', fontsize=9)
        ax.set_ylabel('Dec [deg]', fontsize=9)
        ax.set_xlim(ra_max, ra_min)
        ax.set_ylim(dec_min, dec_max)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=8)

    legend_handles = [Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=LABEL_COLORS[l], markersize=7, label=l)
                      for l in ['STAR', 'GALAXY', 'UNCLASSIFIED']]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(f'Q1 Fields: PHZ Label (dots) with S(G={int(G_EVAL)}) Diagnostic Background',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fname = f'q1_per_field_phz_with_sg{int(G_EVAL)}.png'
    out   = output_dir / fname
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {fname}")


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
    print("Q1 Completeness Analysis")
    print("=" * 70)

    print("\n[1] Querying Gaia-matched Q1 data...")
    df = query_gaia_matched(username)
    df = assign_fields(df)
    print(f"  Assigned to fields: {df['field'].notna().sum():,} sources")

    print("\n[2] Querying all Q1 sources...")
    df_all = query_all_q1(username)
    df_all = assign_fields(df_all)
    print(f"  Assigned to fields: {df_all['field'].notna().sum():,} / {len(df_all):,}")

    print("\n[3] Per-field PHZ spatial maps (Gaia-matched)...")
    plot_per_field_phz_maps(df, OUTPUT_DIR)

    print("\n[4] Per-field PHZ spatial maps (all Q1)...")
    plot_per_field_phz_maps_all(df_all, OUTPUT_DIR)

    print("\n[5] Computing per-source Gaia completeness...")
    sf = compute_per_source_completeness(df)

    print("\n[6] PHZ maps with M10 background...")
    plot_phz_with_m10_background(df, sf, OUTPUT_DIR)

    for g_val in [17, 18, 19, 20]:
        print(f"\n[7] PHZ maps with S(G={g_val}) background...")
        plot_phz_with_sg_background(df, sf, OUTPUT_DIR, g_eval=g_val)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
