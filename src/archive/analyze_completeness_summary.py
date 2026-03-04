#!/usr/bin/env python3
"""
Completeness summary plot (3 panels):
  1. Per-source S(G) scatter
  2. Raw vs completeness-corrected galaxy fraction
  3. Raw vs completeness-corrected density histogram

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_completeness_summary.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output' / 'ero_completeness_new'

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
    ax.grid(alpha=0.3)

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
    ax.grid(alpha=0.3)

    # --- Panel 3: Raw vs completeness-corrected density histogram ---
    ax = axes[2]
    gmag_all = df_field['phot_g_mean_mag'].dropna()

    raw_counts, _ = np.histogram(gmag_all, bins=bins)
    raw_density = raw_counts / raw_counts.sum() if raw_counts.sum() > 0 else raw_counts

    s_vals = df_field['completeness'].values
    g_vals = df_field['phot_g_mean_mag'].values
    valid = s_vals > s_thresh
    weights = np.where(valid, 1.0 / s_vals, 0.0)
    corr_counts, _ = np.histogram(g_vals[valid], bins=bins, weights=weights[valid])
    corr_density = corr_counts / corr_counts.sum() if corr_counts.sum() > 0 else corr_counts

    ax.bar(bin_centers, raw_density, width=bins[1]-bins[0], alpha=0.5,
           color='steelblue', label='Raw')
    ax.bar(bin_centers, corr_density, width=bins[1]-bins[0], alpha=0.5,
           color='coral', label='Corrected (1/S)')
    ax.set_xlabel('Gaia G', fontsize=11, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax.set_title('Raw vs Completeness-Corrected', fontsize=12, fontweight='bold')
    ax.set_xlim(14, 22)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.suptitle('Gaia Completeness & ERO Galaxy Bimodality (per-source S(G))',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / 'completeness_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'completeness_summary.png'}")


def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Completeness Summary Plot")
    print("=" * 70)

    # Query data
    df = query_ero_data(username)
    df = assign_fields(df)

    # Compute per-source completeness
    print("\nComputing per-source completeness...")
    compute_per_source_completeness(df)

    # Generate plot
    print("\nGenerating completeness summary...")
    plot_completeness_summary(df, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
