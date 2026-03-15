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

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output'

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

    # Plot
    print("\nGenerating plot...")
    plot_scanning_law_with_fields(scan_map, nside,
                                  OUTPUT_DIR / 'scanning_law_fields.png')

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
