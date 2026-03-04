#!/usr/bin/env python3
"""
Plot spatial distribution of ALL ERO sources (not just Gaia-matched)
per field, colored by class_star. Analogous to per_field_class_star_map.png
from analyze_ero_completeness.py but using the full ERO catalogue.

Usage:
    python src/plot_ero_all_sources.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from astropy.coordinates import SkyCoord
import astropy.units as u

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


def get_galactic_b(ra, dec):
    c = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    return c.galactic.b.deg


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
        dra = ra2 - ra1
        ddec = dec2 - dec1
        a = np.sin(ddec/2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra/2)**2
        dist = np.degrees(2 * np.arcsin(np.sqrt(a)))
        mask = (dist < radius) & (dist < min_distances)
        field_assignments[mask] = field_name
        min_distances[mask] = dist[mask]

    df['field'] = field_assignments
    return df


def plot_per_field_all_ero(df, output_dir):
    """Per-field spatial maps colored by class_star for ALL ERO sources."""
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

        # Sort so galaxies (low class_star) are plotted on top
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


def plot_per_field_galaxy_fraction_ero_all(df, output_dir):
    """Per-field galaxy fraction vs Euclid VIS magnitude for ALL ERO sources."""
    df_field = df[df['field'].notna()].copy()
    bins = np.arange(14, 28.5, 0.25)
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
        fd_star = fd[fd['label'] == 'STAR']['v_mag_auto'].dropna()
        fd_gal = fd[fd['label'] == 'GALAXY']['v_mag_auto'].dropna()
        fd_all = fd['v_mag_auto'].dropna()
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])

        ax.hist(fd_all, bins=bins, alpha=0.3, color='gray', label=f'All ({len(fd_all):,})')
        ax.hist(fd_star, bins=bins, alpha=0.6, color='steelblue',
                label=f'Stars ({len(fd_star):,})', histtype='step', linewidth=1.5)
        ax.hist(fd_gal, bins=bins, alpha=0.7, color='coral',
                label=f'Galaxies ({len(fd_gal):,})', histtype='step', linewidth=1.5)

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
        ax.set_xlabel('Euclid VIS (v_mag_auto)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(14, 28)
        ax.legend(fontsize=6, loc='upper left')
        ax.grid(alpha=0.3, axis='y')

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('ALL ERO Sources by Class: Euclid VIS Magnitude by Field',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_field_by_class_ero_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ per_field_by_class_ero_all.png")

    # --- Per-field: galaxy fraction per mag bin ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        fd = df_field[df_field['field'] == field_name]
        fd_gal = fd[fd['label'] == 'GALAXY']['v_mag_auto'].dropna()
        fd_all = fd['v_mag_auto'].dropna()
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])

        counts_all, _ = np.histogram(fd_all, bins=bins)
        counts_gal, _ = np.histogram(fd_gal, bins=bins)

        with np.errstate(divide='ignore', invalid='ignore'):
            frac = np.where(counts_all > 0, counts_gal / counts_all, np.nan)

        mask = counts_all >= 10
        ax.plot(bin_centers[mask], frac[mask], 'o-', color='coral', markersize=2,
                linewidth=1.5)

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)', fontsize=10, fontweight='bold')
        ax.set_xlabel('Euclid VIS (v_mag_auto)', fontsize=9)
        ax.set_ylabel('Galaxy fraction', fontsize=9)
        ax.set_xlim(14, 28)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.grid(alpha=0.3)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Galaxy Fraction vs Euclid VIS by Field (ALL ERO Sources)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_field_galaxy_fraction_ero_all.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ per_field_galaxy_fraction_ero_all.png")


def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ERO All Sources: Spatial Distribution")
    print("=" * 70)

    df = query_all_ero(username)
    df = assign_fields(df)
    n_assigned = df['field'].notna().sum()
    n_gal = ((df['field'].notna()) & (df['label'] == 'GALAXY')).sum()
    n_star = ((df['field'].notna()) & (df['label'] == 'STAR')).sum()
    print(f"  Assigned to fields: {n_assigned:,} ({n_star:,} stars, {n_gal:,} galaxies)")

    print("\n  Per-field counts:")
    for field_name in ERO_FIELDS:
        fd = df[df['field'] == field_name]
        print(f"    {field_name:<16}: {len(fd):>8,}")

    print("\n  Plotting spatial maps...")
    plot_per_field_all_ero(df, OUTPUT_DIR)

    print("\n  Plotting galaxy fraction...")
    plot_per_field_galaxy_fraction_ero_all(df, OUTPUT_DIR)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
