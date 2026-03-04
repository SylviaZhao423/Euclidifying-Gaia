#!/usr/bin/env python3
"""
Per-field map of the Gaia match rate at HEALPix level-10 resolution.

For each level-10 pixel in each ERO field, computes:
    match_rate = N_gaia_matched / N_ero_total

Saves to output/per_field_gaia_match_rate.png

Usage:
    python src/plot_gaia_match_rate.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

HEALPIX_LEVEL = 10
NSIDE = 2 ** HEALPIX_LEVEL

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


def assign_fields(df):
    ra = df['ra'].values
    dec = df['dec'].values
    n = len(df)
    assignments = np.full(n, None, dtype=object)
    min_dist = np.full(n, np.inf)

    for name, (fra, fdec, area) in ERO_FIELDS.items():
        radius = np.sqrt(area / np.pi)
        ra1, dec1 = np.radians(ra), np.radians(dec)
        ra2, dec2 = np.radians(fra), np.radians(fdec)
        dra, ddec = ra2 - ra1, dec2 - dec1
        a = np.sin(ddec/2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra/2)**2
        dist = np.degrees(2 * np.arcsin(np.sqrt(a)))
        mask = (dist < radius) & (dist < min_dist)
        assignments[mask] = name
        min_dist[mask] = dist[mask]

    df['field'] = assignments
    return df


def ra_dec_to_healpix(ra, dec, nside=NSIDE):
    """Convert RA/Dec (degrees) to HEALPix NESTED pixel indices."""
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    return hp.ang2pix(nside, theta, phi, nest=True)


def query_data(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)

    # All ERO sources
    print("  Querying ALL ERO sources...")
    df_ero = pd.read_sql_query("""
        SELECT v_alpha_j2000 AS ra, v_delta_j2000 AS dec, v_mag_auto
        FROM euclid_ero.merge_cat
        WHERE v_class_star IS NOT NULL
    """, conn)
    print(f"  ✓ {len(df_ero):,} ERO sources")

    # Gaia-matched ERO sources
    print("  Querying Gaia-matched ERO sources...")
    df_matched = pd.read_sql_query("""
        SELECT e_v_alpha_j2000 AS ra, e_v_delta_j2000 AS dec,
               g_phot_g_mean_mag AS phot_g_mean_mag
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_class_star IS NOT NULL
    """, conn)
    print(f"  ✓ {len(df_matched):,} Gaia-matched sources")

    conn.close()
    return df_ero, df_matched


def compute_match_rate_per_pixel(df_ero, df_matched):
    """Compute per-pixel match rate for sources assigned to fields."""
    df_ero = df_ero[df_ero['field'].notna()].copy()
    df_matched = df_matched[df_matched['field'].notna()].copy()

    df_ero['hpx'] = ra_dec_to_healpix(df_ero['ra'].values, df_ero['dec'].values)
    df_matched['hpx'] = ra_dec_to_healpix(df_matched['ra'].values, df_matched['dec'].values)

    # Count per (field, pixel)
    ero_counts = df_ero.groupby(['field', 'hpx']).size().reset_index(name='n_ero')
    ero_median_mag = df_ero.groupby(['field', 'hpx'])['v_mag_auto'].median().reset_index(name='median_mag')
    match_counts = df_matched.groupby(['field', 'hpx']).size().reset_index(name='n_matched')
    match_median_gmag = df_matched.groupby(['field', 'hpx'])['phot_g_mean_mag'].median().reset_index(name='median_gmag')

    merged = ero_counts.merge(match_counts, on=['field', 'hpx'], how='left')
    merged = merged.merge(ero_median_mag, on=['field', 'hpx'], how='left')
    merged = merged.merge(match_median_gmag, on=['field', 'hpx'], how='left')
    merged['n_matched'] = merged['n_matched'].fillna(0).astype(int)
    merged['match_rate'] = merged['n_matched'] / merged['n_ero']

    return merged


def draw_healpix_pixel(nside, ipix):
    """Draw boundary of a single HEALPix pixel."""
    boundaries = hp.boundaries(nside, ipix, step=16, nest=True)
    theta, phi = hp.vec2ang(boundaries.T)
    ra_b = np.degrees(phi)
    dec_b = 90.0 - np.degrees(theta)
    ra_b = np.append(ra_b, ra_b[0])
    dec_b = np.append(dec_b, dec_b[0])
    return ra_b, dec_b


def plot_match_rate(pixel_data, output_dir):
    """Per-field match rate map at HEALPix level 10."""
    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4.5 * n_rows))
    axes = axes.flatten()

    cmap = plt.cm.summer  # dark green=low, yellow=high
    norm = Normalize(vmin=0, vmax=1)

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])
        fd = pixel_data[pixel_data['field'] == field_name]

        if len(fd) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{field_name}', fontsize=10)
            continue

        # Draw each HEALPix pixel as a filled polygon
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection

        patches = []
        colors = []
        for _, row in fd.iterrows():
            ra_b, dec_b = draw_healpix_pixel(NSIDE, int(row['hpx']))
            verts = np.column_stack([ra_b, dec_b])
            patches.append(Polygon(verts, closed=True))
            colors.append(row['match_rate'])

        pc = PatchCollection(patches, cmap=cmap, norm=norm, edgecolors='gray',
                             linewidths=0.2, alpha=0.9)
        pc.set_array(np.array(colors))
        ax.add_collection(pc)

        # Set axis limits from pixel boundaries
        all_ra = np.concatenate([draw_healpix_pixel(NSIDE, int(h))[0] for h in fd['hpx']])
        all_dec = np.concatenate([draw_healpix_pixel(NSIDE, int(h))[1] for h in fd['hpx']])
        ra_pad = (all_ra.max() - all_ra.min()) * 0.02
        dec_pad = (all_dec.max() - all_dec.min()) * 0.02
        ax.set_xlim(all_ra.max() + ra_pad, all_ra.min() - ra_pad)  # inverted
        ax.set_ylim(all_dec.min() - dec_pad, all_dec.max() + dec_pad)

        cb = plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.04, aspect=20)
        cb.set_label('Match rate', fontsize=7)
        cb.ax.tick_params(labelsize=6)

        n_ero_total = fd['n_ero'].sum()
        n_matched_total = fd['n_matched'].sum()
        overall_rate = n_matched_total / n_ero_total if n_ero_total > 0 else 0
        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)\n'
                     f'{n_matched_total:,}/{n_ero_total:,} matched '
                     f'({overall_rate:.1%} overall)',
                     fontsize=8, fontweight='bold')
        ax.set_xlabel('RA [deg]', fontsize=8)
        ax.set_ylabel('Dec [deg]', fontsize=8)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Gaia match rate (N_matched / N_ERO)', fontsize=10)

    plt.suptitle(f'Gaia Match Rate per HEALPix Level {HEALPIX_LEVEL} Pixel (~3.4 arcmin)',
                 fontsize=14, fontweight='bold', y=1.00)
    fig.subplots_adjust(right=0.90)
    plt.tight_layout(rect=[0, 0, 0.90, 0.98])
    plt.savefig(output_dir / 'per_field_gaia_match_rate.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ per_field_gaia_match_rate.png")


def plot_median_mag(pixel_data, output_dir):
    """Per-field median Euclid VIS magnitude map at HEALPix level 10."""
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4.5 * n_rows))
    axes = axes.flatten()

    cmap = plt.cm.summer
    mag_min = pixel_data['median_mag'].min()
    mag_max = pixel_data['median_mag'].max()
    # Ensure vmin < 18 < vmax for TwoSlopeNorm
    vmin = min(mag_min, 18 - 1)
    vmax = max(mag_max, 18 + 1)
    norm = TwoSlopeNorm(vcenter=18, vmin=vmin, vmax=vmax)

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])
        fd = pixel_data[pixel_data['field'] == field_name]

        if len(fd) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{field_name}', fontsize=10)
            continue

        patches = []
        colors = []
        for _, row in fd.iterrows():
            ra_b, dec_b = draw_healpix_pixel(NSIDE, int(row['hpx']))
            verts = np.column_stack([ra_b, dec_b])
            patches.append(Polygon(verts, closed=True))
            colors.append(row['median_mag'])

        pc = PatchCollection(patches, cmap=cmap, norm=norm, edgecolors='gray',
                             linewidths=0.2, alpha=0.9)
        pc.set_array(np.array(colors))
        ax.add_collection(pc)

        # Set axis limits from pixel boundaries
        all_ra = np.concatenate([draw_healpix_pixel(NSIDE, int(h))[0] for h in fd['hpx']])
        all_dec = np.concatenate([draw_healpix_pixel(NSIDE, int(h))[1] for h in fd['hpx']])
        ra_pad = (all_ra.max() - all_ra.min()) * 0.02
        dec_pad = (all_dec.max() - all_dec.min()) * 0.02
        ax.set_xlim(all_ra.max() + ra_pad, all_ra.min() - ra_pad)  # inverted
        ax.set_ylim(all_dec.min() - dec_pad, all_dec.max() + dec_pad)

        cb = plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.04, aspect=20)
        cb.set_label('Median mag', fontsize=7)
        cb.ax.tick_params(labelsize=6)

        field_median = fd['median_mag'].median()
        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)\n'
                     f'median v_mag_auto = {field_median:.1f}',
                     fontsize=8, fontweight='bold')
        ax.set_xlabel('RA [deg]', fontsize=8)
        ax.set_ylabel('Dec [deg]', fontsize=8)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Median Euclid VIS magnitude (v_mag_auto)', fontsize=10)

    plt.suptitle(f'Median ERO Magnitude per HEALPix Level {HEALPIX_LEVEL} Pixel (~3.4 arcmin)',
                 fontsize=14, fontweight='bold', y=1.00)
    fig.subplots_adjust(right=0.90)
    plt.tight_layout(rect=[0, 0, 0.90, 0.98])
    plt.savefig(output_dir / 'per_field_median_mag.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ per_field_median_mag.png")


def plot_umap(pixel_data, output_dir):
    """UMAP embedding of HEALPix pixels, colored by match rate and median mag."""
    import umap

    # Compute galactic latitude for each pixel's field
    gal_b = pixel_data['field'].map(
        {name: abs(get_galactic_b(ra, dec))
         for name, (ra, dec, _) in ERO_FIELDS.items()}
    )

    # Feature matrix: log(n_ero), median_mag, |b|
    features = pd.DataFrame({
        'log_n_ero': np.log10(pixel_data['n_ero'].values),
        'median_mag': pixel_data['median_mag'].values,
        'abs_gal_b': gal_b.values,
    }).dropna()
    valid_idx = features.index
    features_arr = features.values

    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_arr)

    print("  Running UMAP...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    embedding = reducer.fit_transform(features_scaled)

    pd_valid = pixel_data.loc[valid_idx]

    # 2x2 figure: color by match_rate, median_mag, |b|, field
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: match rate
    ax = axes[0, 0]
    sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                    c=pd_valid['match_rate'].values, cmap='RdYlGn',
                    s=4, alpha=0.7, vmin=0, vmax=0.15)
    plt.colorbar(sc, ax=ax, label='Gaia match rate')
    ax.set_title('Colored by Gaia match rate', fontsize=11, fontweight='bold')

    # Panel 2: median magnitude
    ax = axes[0, 1]
    sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                    c=pd_valid['median_mag'].values, cmap='RdYlGn_r',
                    s=4, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='Median v_mag_auto')
    ax.set_title('Colored by median magnitude', fontsize=11, fontweight='bold')

    # Panel 3: |b|
    ax = axes[1, 0]
    sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                    c=gal_b.loc[valid_idx].values, cmap='viridis',
                    s=4, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='|b| [deg]')
    ax.set_title('Colored by galactic latitude |b|', fontsize=11, fontweight='bold')

    # Panel 4: field (categorical)
    ax = axes[1, 1]
    field_names = list(ERO_FIELDS.keys())
    field_codes = pd_valid['field'].map({n: i for i, n in enumerate(field_names)}).values
    sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                    c=field_codes, cmap='tab20', s=4, alpha=0.7,
                    vmin=0, vmax=len(field_names) - 1)
    cbar = plt.colorbar(sc, ax=ax, ticks=range(len(field_names)))
    cbar.ax.set_yticklabels(field_names, fontsize=6)
    cbar.set_label('Field')
    ax.set_title('Colored by field', fontsize=11, fontweight='bold')

    for ax in axes.flatten():
        ax.set_xlabel('UMAP 1', fontsize=9)
        ax.set_ylabel('UMAP 2', fontsize=9)
        ax.tick_params(labelsize=7)

    plt.suptitle('UMAP Embedding of HEALPix Level-10 Pixels\n'
                 'Features: log(N_ERO), median magnitude, |b|',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'umap_match_rate_vs_mag.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ umap_match_rate_vs_mag.png")


def compute_selection_function(pixel_data):
    """Compute Gaia DR3 selection function S(G) for each pixel at its median Gaia G mag."""
    from gaiaunlimited import DR3SelectionFunctionTCG

    print("  Loading Gaia DR3 selection function (mode='multi')...")
    sf = DR3SelectionFunctionTCG(mode='multi')

    # Only pixels that have Gaia-matched sources (and thus a median G mag)
    mask = pixel_data['median_gmag'].notna()
    hpx_valid = pixel_data.loc[mask, 'hpx'].values.astype(int)
    gmag_valid = pixel_data.loc[mask, 'median_gmag'].values

    # Get pixel center coordinates
    theta, phi = hp.pix2ang(NSIDE, hpx_valid, nest=True)
    ra_pix = np.degrees(phi)
    dec_pix = 90.0 - np.degrees(theta)

    coords = SkyCoord(ra=ra_pix * u.deg, dec=dec_pix * u.deg)

    print("  Evaluating S(G) at median Gaia G mag per pixel...")
    sg = np.full(len(pixel_data), np.nan)
    sg[mask.values] = sf.query(coords, gmag_valid)

    return sg


def plot_source_dist_rate_vs_gmag(df_matched, pixel_data, output_dir):
    """2D histogram of Gaia-matched sources: x = match rate of region, y = Gaia G mag."""
    # Assign each matched source its pixel and field
    df = df_matched[df_matched['field'].notna()].copy()
    df['hpx'] = ra_dec_to_healpix(df['ra'].values, df['dec'].values)

    # Look up the match rate of each source's (field, hpx) region
    rate_lookup = pixel_data.set_index(['field', 'hpx'])['match_rate']
    df['region_match_rate'] = df.set_index(['field', 'hpx']).index.map(rate_lookup)
    df = df.dropna(subset=['region_match_rate', 'phot_g_mean_mag'])

    rate = df['region_match_rate'].values
    gmag = df['phot_g_mean_mag'].values

    print(f"  {len(df):,} sources with valid match rate + G mag")

    fig, ax = plt.subplots(figsize=(8, 6))
    h = ax.hist2d(rate, gmag, bins=[100, 100], cmap='inferno',
                  range=[[0, min(0.3, np.percentile(rate, 99.9))],
                         [np.percentile(gmag, 0.5), np.percentile(gmag, 99.5)]],
                  norm=plt.matplotlib.colors.LogNorm())
    plt.colorbar(h[3], ax=ax, label='Number of sources')
    ax.set_xlabel('Gaia match rate of region (N_matched / N_ERO)', fontsize=11)
    ax.set_ylabel('Gaia G magnitude (phot_g_mean_mag)', fontsize=11)
    ax.set_title(f'Gaia-Matched Source Distribution ({len(df):,} sources)\n'
                 f'HEALPix Level-{HEALPIX_LEVEL} Regions',
                 fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'source_dist_rate_vs_gmag.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ source_dist_rate_vs_gmag.png")


def plot_source_dist_sf_vs_gmag(df_matched, output_dir):
    """2D histogram of Gaia-matched sources: x = S(G) at source position, y = Gaia G mag."""
    from gaiaunlimited import DR3SelectionFunctionTCG

    df = df_matched[df_matched['field'].notna() & df_matched['phot_g_mean_mag'].notna()].copy()
    gmag = df['phot_g_mean_mag'].values.astype(float)
    coords = SkyCoord(ra=df['ra'].values * u.deg, dec=df['dec'].values * u.deg)

    print(f"  Computing S(G) for {len(df):,} sources...")
    sf = DR3SelectionFunctionTCG(mode='multi')

    # Batch to avoid memory issues
    sg = np.zeros(len(df))
    batch = 50000
    for i in range(0, len(df), batch):
        j = min(i + batch, len(df))
        sg[i:j] = sf.query(coords[i:j], gmag[i:j])
        print(f"    Batch {i:,}-{j:,} done")

    fig, ax = plt.subplots(figsize=(8, 6))
    h = ax.hist2d(sg, gmag, bins=[100, 100], cmap='inferno',
                  norm=plt.matplotlib.colors.LogNorm(),
                  range=[[0, 1.05],
                         [np.percentile(gmag, 0.5), np.percentile(gmag, 99.5)]])
    plt.colorbar(h[3], ax=ax, label='Number of sources')
    ax.set_xlabel('Selection function S(G)', fontsize=11)
    ax.set_ylabel('Gaia G magnitude', fontsize=11)
    ax.set_title(f'Gaia-Matched Source Distribution ({len(df):,} sources)\n'
                 f'Selection Function (mode=multi) vs Gaia G',
                 fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'source_dist_sf_vs_gmag.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ source_dist_sf_vs_gmag.png")


def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Gaia Match Rate per HEALPix Pixel")
    print("=" * 70)

    print("\n[1] Querying data...")
    df_ero, df_matched = query_data(username)

    print("\n[2] Assigning fields...")
    df_ero = assign_fields(df_ero)
    df_matched = assign_fields(df_matched)
    print(f"  ERO in fields: {df_ero['field'].notna().sum():,}")
    print(f"  Matched in fields: {df_matched['field'].notna().sum():,}")

    print("\n[3] Computing per-pixel match rates...")
    pixel_data = compute_match_rate_per_pixel(df_ero, df_matched)
    print(f"  {len(pixel_data):,} pixels with data")
    print(f"  Match rate range: {pixel_data['match_rate'].min():.3f} – "
          f"{pixel_data['match_rate'].max():.3f}")
    print(f"  Median match rate: {pixel_data['match_rate'].median():.3f}")

    print(f"  Median magnitude range: {pixel_data['median_mag'].min():.1f} – "
          f"{pixel_data['median_mag'].max():.1f}")
    print(f"  Overall median magnitude: {pixel_data['median_mag'].median():.1f}")

    print("\n[4] Computing selection function S(G) per pixel...")
    pixel_data['sg'] = compute_selection_function(pixel_data)

    print("\n[5] Plotting...")
    plot_match_rate(pixel_data, OUTPUT_DIR)
    plot_median_mag(pixel_data, OUTPUT_DIR)
    plot_umap(pixel_data, OUTPUT_DIR)
    plot_source_dist_rate_vs_gmag(df_matched, pixel_data, OUTPUT_DIR)
    plot_source_dist_sf_vs_gmag(df_matched, OUTPUT_DIR)

    print(f"\nSaved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
