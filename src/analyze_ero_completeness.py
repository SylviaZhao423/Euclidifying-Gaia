#!/usr/bin/env python3
"""
Gaia DR3 completeness analysis — produces:
  1. completeness_multi_G_fields.png      — Mollweide maps at G=18,19,20,21,22
  2. completeness_summary.png             — 3-panel summary (S(G) scatter, galaxy fraction, density)
  3. selection_function_per_field.png      — median S(G) vs Gaia G per field
  4. galaxy_fraction_sg_corrected.png     — galaxy fraction per field, raw & 1/S(G) corrected

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_ero_selection.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import healpy as hp
from astropy.coordinates import SkyCoord

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output' / 'ero_completeness'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

G_VALUES = [18, 19, 20, 21, 22]

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

Q1_FIELDS = {
    'EDF-North':    (269.733, +66.018, 22.9),
    'EDF-South':    (61.241, -48.423, 28.1),
    'EDF-Fornax':   (52.932, -28.088, 12.1),
    'LDN 1641':     (85.75, -8.367, 0.5),
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


# ─── Data queries ────────────────────────────────────────────────────────────

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
    print(f"  Retrieved {len(df):,} sources")
    return df


# ─── Completeness map (Mollweide) ───────────────────────────────────────────

def build_completeness_maps(sf, nside=64):
    """Build full-sky completeness maps for all G_VALUES."""
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra_deg = np.degrees(phi)
    dec_deg = 90.0 - np.degrees(theta)

    batch = 10000
    coord_batches = []
    for i in range(0, npix, batch):
        j = min(i + batch, npix)
        coord_batches.append((i, j, SkyCoord(ra=ra_deg[i:j], dec=dec_deg[i:j], unit='deg')))

    maps = {}
    for g_val in G_VALUES:
        print(f"  Computing S(G={g_val}) for {npix} pixels...")
        completeness = np.zeros(npix)
        for i, j, coords in coord_batches:
            gmag = np.full(j - i, float(g_val))
            completeness[i:j] = sf.query(coords, gmag)
        maps[g_val] = completeness
        print(f"    Done (min={completeness.min():.3f}, median={np.median(completeness):.3f})")

    return maps


def draw_field_circle(ra_center, dec_center, area_deg2, color, n_pts=72):
    """Draw a circle on the current HEALPix Mollweide projection."""
    radius = np.sqrt(area_deg2 / np.pi)
    theta = np.linspace(0, 2 * np.pi, n_pts)
    cos_dec = max(np.cos(np.radians(dec_center)), 0.01)
    ra_pts = ra_center + radius * np.cos(theta) / cos_dec
    dec_pts = np.clip(dec_center + radius * np.sin(theta), -89.9, 89.9)
    hp.projplot(ra_pts, dec_pts, lonlat=True, coord='CG',
                color=color, linewidth=1.0, zorder=10)


def plot_completeness_map(maps, nside, output_dir):
    """Plot 1x5 row of Mollweide completeness maps with fields overlaid."""
    n = len(G_VALUES)

    fig = plt.figure(figsize=(6 * n, 5))

    for idx, g_val in enumerate(G_VALUES):
        hp.mollview(maps[g_val], coord='CG', title=f'G = {g_val}',
                    cmap='viridis', min=0, max=1,
                    sub=(1, n, idx + 1), fig=fig,
                    notext=True, cbar=False)

        for name, (ra, dec, area) in ERO_FIELDS.items():
            draw_field_circle(ra, dec, area, '#FF1493')
        for name, (ra, dec, area) in Q1_FIELDS.items():
            draw_field_circle(ra, dec, area, 'red')

    fig.subplots_adjust(bottom=0.12)
    cax = fig.add_axes([0.25, 0.05, 0.5, 0.025])
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Completeness S(G)', fontsize=11)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Q1 fields'),
        Line2D([0], [0], color='#EB759D', linewidth=2, label='ERO fields'),
    ]
    fig.legend(handles=legend_elements, loc='lower right', fontsize=9,
               framealpha=0.9, edgecolor='gray', bbox_to_anchor=(0.92, 0.02))

    fig.suptitle('Gaia DR3 Completeness by Magnitude with Euclid Q1 & ERO Fields',
                 fontsize=14, fontweight='bold', y=0.99)

    output_path = output_dir / 'completeness_multi_G_fields.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def print_completeness_table(maps, nside):
    """Print completeness at each field position for all G values."""
    header = f"{'Field':<15}" + ''.join(f"  G={g:>2}" for g in G_VALUES)
    print(f"\n{'='*80}")
    print("COMPLETENESS S(G) AT EUCLID FIELD POSITIONS")
    print(f"{'='*80}")
    print(header)
    print("-" * 80)

    for label, fields in [("Q1 Fields", Q1_FIELDS), ("ERO Fields", ERO_FIELDS)]:
        print(f"\n{label}:")
        for name, (ra, dec, area) in fields.items():
            theta = np.radians(90.0 - dec)
            phi = np.radians(ra)
            pix = hp.ang2pix(nside, theta, phi)
            vals = ''.join(f"  {maps[g][pix]:>5.3f}" for g in G_VALUES)
            print(f"  {name:<15}{vals}")


# ─── Completeness summary (3 panels) ────────────────────────────────────────

def compute_per_source_completeness(df, sf):
    """Compute S(G) at each source's actual RA/Dec."""
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

    print(f"  Completeness computed for {n:,} sources")
    print(f"    Median S(G): {np.nanmedian(completeness):.3f}")
    print(f"    Sources with S(G) < 0.5: {(completeness < 0.5).sum():,}")


def plot_completeness_summary(df, output_dir):
    """Combined 3-panel summary: S(G) scatter, galaxy fraction, density histogram."""
    df_gal = df[(df['field'].notna()) & (df['label'] == 'GALAXY')
                & df['phot_g_mean_mag'].notna() & df['completeness'].notna()].copy()

    bins = np.arange(14, 22.5, 0.25)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- Panel 1: Per-source S(G) vs G scatter ---
    ax = axes[0]
    for field_name in ERO_FIELDS:
        fd = df_gal[df_gal['field'] == field_name]
        if len(fd) > 0:
            ax.scatter(fd['phot_g_mean_mag'], fd['completeness'],
                       s=1, alpha=0.3, color=get_field_color(field_name),
                       label=field_name, rasterized=True)
    ax.set_xlabel('Gaia G', fontsize=11, fontweight='bold')
    ax.set_ylabel('S(G) at source position', fontsize=11, fontweight='bold')
    ax.set_title('Per-Source Completeness', fontsize=12, fontweight='bold')
    ax.set_xlim(14, 22)
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(y=0.5, color='gray', ls='--', alpha=0.5)
    ax.legend(fontsize=5, ncol=3, loc='lower left', markerscale=5)
    ax.grid(axis='y', alpha=0.3)

    # --- Panel 2: Raw vs corrected galaxy fraction ---
    ax = axes[1]
    df_field = df[(df['field'].notna()) & df['phot_g_mean_mag'].notna()
                  & df['completeness'].notna()].copy()
    df_star = df_field[df_field['label'] == 'STAR']
    bin_centers = (bins[:-1] + bins[1:]) / 2
    s_thresh = 0.05

    raw_gal, _ = np.histogram(df_gal['phot_g_mean_mag'], bins=bins)
    raw_star, _ = np.histogram(df_star['phot_g_mean_mag'], bins=bins)
    raw_all = raw_gal + raw_star
    with np.errstate(divide='ignore', invalid='ignore'):
        raw_frac = np.where(raw_all > 0, raw_gal / raw_all, np.nan)

    def _whist(sub):
        s = sub['completeness'].values
        g = sub['phot_g_mean_mag'].values
        v = s > s_thresh
        w = np.where(v, 1.0 / s, 0.0)
        c, _ = np.histogram(g[v], bins=bins, weights=w[v])
        return c

    corr_gal = _whist(df_gal)
    corr_star = _whist(df_star)
    corr_all = corr_gal + corr_star
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_frac = np.where(corr_all > 0, corr_gal / corr_all, np.nan)

    mask_raw = raw_all >= 10
    mask_corr = corr_all > 0
    ax.plot(bin_centers[mask_raw], raw_frac[mask_raw], 'o-', color='steelblue',
            markersize=3, linewidth=2, label='Raw')
    ax.plot(bin_centers[mask_corr], corr_frac[mask_corr], 's--', color='coral',
            markersize=3, linewidth=2, label='Completeness-corrected')
    ax.set_xlabel('Gaia G', fontsize=11, fontweight='bold')
    ax.set_ylabel('Galaxy fraction', fontsize=11, fontweight='bold')
    ax.set_title('Raw vs Completeness-Corrected Galaxy Fraction', fontsize=12, fontweight='bold')
    ax.set_xlim(14, 22)
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # --- Panel 3: Raw vs completeness-corrected galaxy density ---
    ax = axes[2]
    bins3 = np.arange(14, 22.5, 0.25)
    bc3 = (bins3[:-1] + bins3[1:]) / 2

    # Raw: same galaxy subset (with completeness), unweighted
    raw_gal3, _ = np.histogram(df_gal['phot_g_mean_mag'], bins=bins3)

    # Corrected: galaxies with valid completeness, weighted by 1/S(G)
    s_gal = df_gal['completeness'].values
    g_gal = df_gal['phot_g_mean_mag'].values
    v_gal = s_gal > 0.05
    w_gal = np.where(v_gal, 1.0 / s_gal, 0.0)
    corr_gal3, _ = np.histogram(g_gal[v_gal], bins=bins3, weights=w_gal[v_gal])

    # Normalize each to density (sum=1) so shape differences are visible
    raw_norm = raw_gal3 / raw_gal3.sum() if raw_gal3.sum() > 0 else raw_gal3
    corr_norm = corr_gal3 / corr_gal3.sum() if corr_gal3.sum() > 0 else corr_gal3

    ax.bar(bc3, raw_norm, width=bins3[1]-bins3[0], alpha=0.5,
           color='steelblue', edgecolor='white', linewidth=0.5, label='Raw')
    ax.bar(bc3, corr_norm, width=bins3[1]-bins3[0], alpha=0.5,
           color='coral', edgecolor='white', linewidth=0.5, label='Corrected (1/S)')
    ax.set_xlabel('Gaia G', fontsize=11, fontweight='bold')
    ax.set_ylabel('Normalized density', fontsize=11, fontweight='bold')
    ax.set_title('Raw vs Corrected Galaxy Density', fontsize=12, fontweight='bold')
    ax.set_xlim(14, 22)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Gaia Completeness & ERO Galaxy Bimodality (per-source S(G))',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    output_path = output_dir / 'completeness_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def get_galactic_b(ra, dec):
    c = SkyCoord(ra=ra, dec=dec, unit='deg')
    return c.galactic.b.deg


def plot_selection_function_per_field(df, output_dir):
    """Per-field median S(G) vs Gaia G magnitude."""
    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols
    bins = np.arange(14, 22.5, 0.25)
    bc = (bins[:-1] + bins[1:]) / 2

    sub = df[(df['field'].notna()) & df['phot_g_mean_mag'].notna()
             & df['completeness'].notna()].copy()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])
        color = get_field_color(field_name)

        fd = sub[sub['field'] == field_name]
        g_vals = fd['phot_g_mean_mag'].values
        s_vals = fd['completeness'].values

        median_s = np.full(len(bins) - 1, np.nan)
        p16 = np.full(len(bins) - 1, np.nan)
        p84 = np.full(len(bins) - 1, np.nan)
        for i in range(len(bins) - 1):
            in_bin = (g_vals >= bins[i]) & (g_vals < bins[i + 1])
            if in_bin.sum() >= 5:
                median_s[i] = np.median(s_vals[in_bin])
                p16[i] = np.percentile(s_vals[in_bin], 16)
                p84[i] = np.percentile(s_vals[in_bin], 84)

        mask = ~np.isnan(median_s)
        ax.plot(bc[mask], median_s[mask], 'o-', color=color, markersize=3, linewidth=1.5)
        ax.fill_between(bc[mask], p16[mask], p84[mask], alpha=0.2, color=color)

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)\nN={len(fd):,}',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Gaia G', fontsize=9)
        ax.set_ylabel('S(G)', fontsize=9)
        ax.set_xlim(14, 22)
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.grid(axis='y', alpha=0.3)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Gaia Selection Function S(G) vs Gaia G by Field (median ± 16–84%)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = output_dir / 'per_field_sf.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_galaxy_fraction_sg_per_field(df, output_dir):
    """Per-field galaxy fraction vs Gaia G, raw and corrected by 1/S(G)."""
    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols
    bins = np.arange(14, 22.5, 0.25)
    bc = (bins[:-1] + bins[1:]) / 2
    s_thresh = 0.05

    sub = df[(df['field'].notna()) & df['phot_g_mean_mag'].notna()
             & df['completeness'].notna() & df['label'].notna()].copy()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])

        fd = sub[sub['field'] == field_name]
        g_vals = fd['phot_g_mean_mag'].values
        s_vals = fd['completeness'].values
        labels = fd['label'].values
        is_gal = labels == 'GALAXY'

        # Raw galaxy fraction
        n_gal_raw, _ = np.histogram(g_vals[is_gal], bins=bins)
        n_all_raw, _ = np.histogram(g_vals, bins=bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            raw_frac = np.where(n_all_raw >= 10, n_gal_raw / n_all_raw, np.nan)

        # Corrected: weight by 1/S(G)
        valid = (s_vals > s_thresh) & np.isfinite(s_vals)
        weights = np.where(valid, 1.0 / s_vals, 0.0)

        corr_gal, _ = np.histogram(g_vals[valid & is_gal], bins=bins,
                                    weights=weights[valid & is_gal])
        corr_all, _ = np.histogram(g_vals[valid], bins=bins,
                                    weights=weights[valid])
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_frac = np.where(corr_all > 0, corr_gal / corr_all, np.nan)

        # Plot
        mask_raw = ~np.isnan(raw_frac)
        mask_corr = ~np.isnan(corr_frac)
        mask_both = mask_raw & mask_corr

        ax.plot(bc[mask_raw], raw_frac[mask_raw], 'o-', color='steelblue',
                markersize=3, linewidth=1.5, label='Raw')
        ax.plot(bc[mask_corr], corr_frac[mask_corr], 's-', color='coral',
                markersize=3, linewidth=1.5, label='Corrected (1/S)')
        if mask_both.any():
            ax.fill_between(bc[mask_both], raw_frac[mask_both], corr_frac[mask_both],
                            alpha=0.2, color='coral')

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Gaia G', fontsize=9)
        ax.set_ylabel('Galaxy fraction', fontsize=9)
        ax.set_xlim(14, 22)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(axis='y', alpha=0.3)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Galaxy Fraction vs Gaia G by Field (Raw & S(G)-Corrected)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = output_dir / 'per_field_galaxy_fraction_sf_corrected.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Gaia DR3 Completeness Analysis")
    print("=" * 70)

    # Load selection function once (shared by map and per-source)
    from gaiaunlimited import DR3SelectionFunctionTCG
    print("\n[1] Loading DR3SelectionFunctionTCG...")
    sf = DR3SelectionFunctionTCG(mode='multi')

    # Build Mollweide completeness maps
    nside = 64
    print("\n[2] Building full-sky completeness maps...")
    maps = build_completeness_maps(sf, nside=nside)
    print_completeness_table(maps, nside)

    print("\n[3] Generating Mollweide plot...")
    plot_completeness_map(maps, nside, OUTPUT_DIR)

    # Query ERO data and compute per-source completeness
    print("\n[4] Querying ERO crossmatch data...")
    df = query_ero_data(username)
    df = assign_fields(df)

    print("\n[5] Computing per-source completeness...")
    compute_per_source_completeness(df, sf)

    print("\n[6] Generating completeness summary...")
    plot_completeness_summary(df, OUTPUT_DIR)

    print("\n[7] Generating per-field selection function...")
    plot_selection_function_per_field(df, OUTPUT_DIR)

    print("\n[8] Generating per-field galaxy fraction (S(G)-corrected)...")
    plot_galaxy_fraction_sg_per_field(df, OUTPUT_DIR)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
