#!/usr/bin/env python3
"""
Overlay Q1 and ERO field positions on the Gaia scanning law map.

Uses gaiaunlimited to show the number of Gaia scans across the sky,
with Q1 and ERO Euclid fields highlighted using known field centers.

Usage:
    python src/analyze_scanning_law.py
"""

import os
import sys
from pathlib import Path
from tokenize import Whitespace
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from matplotlib.colors import LogNorm
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output'
MAX_PLOT_POINTS_PER_FIELD = 20000

# Euclid field centers (J2000) from Table 1
ERO_FIELDS = {
    # Star-forming Regions
    'Horsehead':    (85.150, -2.613,  0.58),
    'Messier 78':   (86.690, -0.015,  0.60),
    'Taurus':       (64.983, +28.023, 0.61),
    'Barnard 30':   (82.880, +12.316, 0.60),
    # Globular Clusters
    'NGC 6254':     (254.303, -4.100, 0.60),
    'NGC 6397':     (265.174, -53.658, 0.61),
    # Nearby Galaxies
    'NGC 6822':     (296.236, -14.788, 0.60),
    'Holmberg II':  (124.790, +70.706, 0.60),
    'IC 10':        (5.063, +59.288, 0.62),
    'IC 342':       (56.730, +68.084, 0.59),
    'NGC 2403':     (114.211, +65.586, 0.60),
    'NGC 6744':     (287.430, -63.842, 0.60),
    # Galaxy Groups
    'Fornax':       (54.017, -35.267, 0.57),
    'Dorado':       (64.014, -55.780, 0.60),
    # Galaxy Clusters
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

ALL_FIELDS = {
    **{f"Q1:{k}": v for k, v in Q1_FIELDS.items()},
    **{f"ERO:{k}": v for k, v in ERO_FIELDS.items()},
}

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'


def build_scanning_law_map(nside=64):
    """Build full-sky scanning law map using gaiaunlimited."""
    from gaiaunlimited.scanninglaw import GaiaScanningLaw
    from tqdm import tqdm

    print("  Loading Gaia DR3 scanning law...")
    sl = GaiaScanningLaw('dr3_nominal')

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra_deg = np.degrees(phi)
    dec_deg = 90.0 - np.degrees(theta)

    print(f"  Querying scan counts for {npix} HEALPix pixels (nside={nside})...")

    def get_totaln(ra, dec):
        return sum(sl.query(ra, dec, count_only=True))

    result = [get_totaln(ra, dec) for ra, dec in tqdm(zip(ra_deg, dec_deg), total=npix)]

    return np.array(result)


def query_gaia_in_all_fields(username):
    """Query Gaia sources directly inside each Q1 + ERO field footprint."""
    import psycopg2

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    frames = []
    print("  Querying Gaia DR3 directly in all Q1 + ERO field footprints...")
    for field_name, (field_ra, field_dec, area) in ALL_FIELDS.items():
        radius = np.sqrt(area / np.pi)
        query = f"""
            SELECT ra, dec
            FROM gaia_dr3.gaia_source
            WHERE q3c_radial_query(ra, dec, {field_ra}, {field_dec}, {radius})
        """
        print(f"    {field_name} (r={radius:.3f}°)...", end='', flush=True)
        df_f = pd.read_sql_query(query, conn)
        df_f['field'] = field_name
        print(f" {len(df_f):,} Gaia sources")
        frames.append(df_f)

    conn.close()
    return pd.concat(frames, ignore_index=True)


def plot_scanning_law_with_fields(scan_map, nside, output_path):
    """Plot Mollweide scanning law map with Q1 and ERO fields overlaid."""
    fig = plt.figure()

    # coord='CG': data in Celestial, display in Galactic (matches gaiaunlimited docs)
    hp.mollview(scan_map, coord='CG', title='',
                unit='Number of Gaia FoV transits', norm='log',
                cmap='viridis', hold=True, fig=fig)

    # healpy resets the figure size internally — restore it
    fig.set_size_inches(16, 6)

    # Stretch the map axes to fill the wide figure
    # fig.axes[0] = map, fig.axes[-1] = colorbar
    map_ax = fig.axes[0]
    map_ax.set_position([0.02, 0.12, 0.96, 0.82])  # [left, bottom, width, height]
    map_ax.set_aspect('auto')  # release fixed aspect so the oval stretches

    # Reposition colorbar to sit below the stretched map
    cbar_ax = fig.axes[-1]
    cbar_ax.set_position([0.25, 0.06, 0.50, 0.03])

    # Manual label offsets (in deg) to avoid overlapping: (d_ra, d_dec, ha)
    ERO_OFFSETS = {
        'Horsehead':    (0, -4, 'left'),
        'Messier 78':   (0, +4, 'center'),
        'Taurus':       (0, +4, 'center'),
        'Barnard 30':   (0, +4, 'center'),
        'NGC 6254':     (0, +4, 'center'),
        'NGC 6397':     (0, -5, 'center'),
        'NGC 6822':     (0, +4, 'center'),
        'Holmberg II':  (+5, 0, 'left'),
        'IC 10':        (0, -5, 'center'),
        'IC 342':       (0, -5, 'center'),
        'NGC 2403':     (-5, 0, 'right'),
        'NGC 6744':     (0, +4, 'center'),
        'Fornax':       (+3, 0, 'left'),
        'Dorado':       (+4, 0, 'left'),
        'Perseus':      (0, -5, 'center'),
        'Abell 2390':   (0, +4, 'center'),
        'Abell 2764':   (0, -4, 'center'),
    }

    Q1_OFFSETS = {
        'EDF-North':    (-8, 0, 'left'),
        'EDF-South':    (-8, 0, 'right'),
        'EDF-Fornax':   (-6, 0, 'left'),
        'LDN 1641':     (0, -8, 'center'),
    }

    # Draw true angular-size circles for each field
    # radius = sqrt(area / pi) in degrees
    def draw_field_circle(ra_center, dec_center, area_deg2, color, n_pts=72):
        """Draw a circle on the Mollweide projection with true angular size."""
        # area in deg² -> radius in degrees: r = sqrt(A / pi)
        radius = np.sqrt(area_deg2 / np.pi)
        theta = np.linspace(0, 2 * np.pi, n_pts)
        # Generate circle points in RA/Dec
        # Account for cos(dec) compression in RA
        cos_dec = np.cos(np.radians(dec_center))
        if cos_dec < 0.01:
            cos_dec = 0.01  # avoid division by zero near poles
        ra_pts = ra_center + radius * np.cos(theta) / cos_dec
        dec_pts = dec_center + radius * np.sin(theta)
        # Clip dec to valid range
        dec_pts = np.clip(dec_pts, -89.9, 89.9)
        hp.projplot(ra_pts, dec_pts, lonlat=True, coord='CG',
                    color=color, linewidth=1.5, zorder=10)
        # Fill with semi-transparent color
        hp.projscatter(ra_center, dec_center, lonlat=True, coord='CG',
                       color=color, s=3, marker='.', zorder=10)

    # Plot ERO fields
    for name, (ra, dec, area) in ERO_FIELDS.items():
        draw_field_circle(ra, dec, area, 'white')
        dra, ddec, ha = ERO_OFFSETS[name]
        hp.projtext(ra + dra, dec + ddec, name, lonlat=True, coord='CG',
                    color='white', fontsize=5, ha=ha, zorder=11)

    # Plot Q1 fields
    for name, (ra, dec, area) in Q1_FIELDS.items():
        draw_field_circle(ra, dec, area, 'red')
        dra, ddec, ha = Q1_OFFSETS[name]
        hp.projtext(ra + dra, dec + ddec, name, lonlat=True, coord='CG',
                    color='#FF6666', fontsize=6, fontweight='bold',
                    ha=ha, zorder=11)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2,
               label='Q1 fields'),
        Line2D([0], [0], color='white', linewidth=2,
               label='ERO fields'),
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=11,
               framealpha=0.9, edgecolor='white',
               labelcolor='white', facecolor='#333333')

    plt.title('Gaia DR3 Scanning Law with Euclid Q1 & ERO Fields',
              fontsize=14, fontweight='bold', y=1.01)

    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_scanning_law_field_cutouts(scan_map, nside, df_gaia, output_path):
    """Plot local RA/Dec cutouts of scanning law using Gaia source positions."""
    field_names = list(ALL_FIELDS.keys())
    n_cols = 4
    n_rows = int(np.ceil(len(field_names) / n_cols))
    rng = np.random.RandomState(0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.1 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    fig.subplots_adjust(left=0.05, right=0.90, top=0.96, bottom=0.05,
                        wspace=0.28, hspace=0.36)

    norm = LogNorm(vmin=14, vmax=165)
    last_sc = None

    for ax, field_name in zip(axes, field_names):
        ra0, dec0, area = ALL_FIELDS[field_name]
        radius = np.sqrt(area / np.pi)

        fd = df_gaia[df_gaia['field'] == field_name]
        if len(fd) > MAX_PLOT_POINTS_PER_FIELD:
            fd_plot = fd.sample(MAX_PLOT_POINTS_PER_FIELD, random_state=0)
        else:
            fd_plot = fd
        theta = np.radians(90.0 - fd_plot['dec'].to_numpy())
        phi = np.radians(np.mod(fd_plot['ra'].to_numpy(), 360.0))
        pix = hp.ang2pix(nside, theta, phi, nest=False)
        scans = scan_map[pix]

        last_sc = ax.scatter(
            fd_plot['ra'], fd_plot['dec'],
            c=scans,
            cmap="viridis", norm=norm,
            s=3 if field_name.startswith("Q1:") else 8,
            alpha=0.85, linewidths=0, rasterized=True
        )

        title_name = field_name.replace("Q1:", "").replace("ERO:", "")
        center_theta = np.radians(90.0 - dec0)
        center_phi = np.radians(np.mod(ra0, 360.0))
        center_pix = hp.ang2pix(nside, center_theta, center_phi, nest=False)
        center_val = scan_map[center_pix]
        ax.set_title(
            f"{title_name}\n"
            f"n={len(fd):,}, scans≈{center_val:.0f}",
            fontsize=10, fontweight="bold"
        )

        ax.set_xlim(ra0 + radius * 1.15, ra0 - radius * 1.15)
        ax.set_ylim(dec0 - radius * 1.15, dec0 + radius * 1.15)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("RA [deg]", fontsize=9)
        ax.set_ylabel("Dec [deg]", fontsize=9)

        prefix = field_name.split(":", 1)[0]
        spine_color = "red" if prefix == "Q1" else "white"
        spine_width = 2.0 if prefix == "Q1" else 1.4
        for spine in ax.spines.values():
            spine.set_color(spine_color)
            spine.set_linewidth(spine_width)

    for ax in axes[len(field_names):]:
        ax.axis("off")

    if last_sc is not None:
        cax = fig.add_axes([0.92, 0.17, 0.02, 0.68])
        cbar = fig.colorbar(last_sc, cax=cax)
        cbar.set_label("Number of Gaia FoV transits", fontsize=12)
        cbar.set_ticks([14, 20, 40, 80, 165])
        cbar.set_ticklabels(["14", "20", "40", "80", "165"])

    fig.suptitle(
        "Gaia scanning law in Euclid field cutouts",
        fontsize=16, fontweight="bold", y=0.995,
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def print_scan_statistics(scan_map, nside):
    """Print scan count statistics at each field position."""
    print("\n" + "=" * 60)
    print("SCAN COUNTS AT EUCLID FIELD POSITIONS")
    print("=" * 60)

    print(f"\n{'Field':<15} {'RA':>7} {'Dec':>7} {'Area':>6} {'Scans':>6}")
    print("-" * 50)

    print("\nQ1 Fields:")
    q1_scans = []
    for name, (ra, dec, area) in Q1_FIELDS.items():
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        pix = hp.ang2pix(nside, theta, phi)
        scans = scan_map[pix]
        q1_scans.append(scans)
        print(f"  {name:<15} {ra:>7.1f} {dec:>+7.1f} {area:>5.1f}° {scans:>5.0f}")

    print("\nERO Fields:")
    ero_scans = []
    for name, (ra, dec, area) in ERO_FIELDS.items():
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        pix = hp.ang2pix(nside, theta, phi)
        scans = scan_map[pix]
        ero_scans.append(scans)
        print(f"  {name:<15} {ra:>7.1f} {dec:>+7.1f} {area:>5.2f}° {scans:>5.0f}")

    q1_scans = np.array(q1_scans)
    ero_scans = np.array(ero_scans)
    print(f"\nQ1  median scans: {np.median(q1_scans):.0f} (range: {q1_scans.min():.0f}-{q1_scans.max():.0f})")
    print(f"ERO median scans: {np.median(ero_scans):.0f} (range: {ero_scans.min():.0f}-{ero_scans.max():.0f})")

    print(f"\nFull-sky scan map: min={scan_map.min():.0f}, max={scan_map.max():.0f} "
          f"(these are the colorbar limits)")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Gaia Scanning Law: Q1 vs ERO Field Coverage")
    print(f"Q1 fields: {len(Q1_FIELDS)}, ERO fields: {len(ERO_FIELDS)}")
    print("=" * 70)

    # Build scanning law map
    nside = 64
    scan_map = build_scanning_law_map(nside=nside)

    # Print statistics
    print_scan_statistics(scan_map, nside)

    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set.")
        print("Set it with: export PGUSER=your_username")
        return 1

    print("\nQuerying Gaia field cutout sources...")
    df_gaia = query_gaia_in_all_fields(username)

    # Plot
    print("\nGenerating plot...")
    plot_scanning_law_with_fields(scan_map, nside,
                                  OUTPUT_DIR / 'scanning_law_fields.png')
    plot_scanning_law_field_cutouts(
        scan_map, nside, df_gaia, OUTPUT_DIR / 'scanning_law_fields_cutouts.png'
    )

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
