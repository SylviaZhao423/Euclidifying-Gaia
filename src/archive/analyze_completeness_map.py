#!/usr/bin/env python3
"""
Mollweide projection of Gaia DR3 completeness at multiple G magnitudes
with Euclid Q1 and ERO fields overlaid.

Uses gaiaunlimited DR3SelectionFunctionTCG (Cantat-Gaudin et al. 2023)
to compute S(G) at each HEALPix pixel for G = 15, 16, 17, 18, 19, 20, 21, 22.

Usage:
    python src/analyze_completeness_map.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from astropy.coordinates import SkyCoord

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output'

G_VALUES = [18, 19, 20, 21, 22]

# Euclid field centers (J2000): name -> (RA, Dec, area_deg2)
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


def build_completeness_maps(nside=64):
    """Build full-sky completeness maps for all G_VALUES using gaiaunlimited."""
    from gaiaunlimited import DR3SelectionFunctionTCG

    print("  Loading DR3SelectionFunctionTCG...")
    sf = DR3SelectionFunctionTCG(mode='multi')

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra_deg = np.degrees(phi)
    dec_deg = 90.0 - np.degrees(theta)

    # Pre-build SkyCoord objects in batches
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


def plot_all_panels(maps, nside, output_path):
    """Plot 1x5 row of Mollweide completeness maps with fields overlaid."""
    n = len(G_VALUES)
    n_cols = n
    n_rows = 1

    fig = plt.figure(figsize=(6 * n_cols, 5))

    for idx, g_val in enumerate(G_VALUES):
        hp.mollview(maps[g_val], coord='CG', title=f'G = {g_val}',
                    cmap='viridis', min=0, max=1,
                    sub=(n_rows, n_cols, idx + 1), fig=fig,
                    notext=True, cbar=False)

        # Overlay fields — hot pink visible against both yellow and dark purple
        for name, (ra, dec, area) in ERO_FIELDS.items():
            draw_field_circle(ra, dec, area, '#FF1493')
        for name, (ra, dec, area) in Q1_FIELDS.items():
            draw_field_circle(ra, dec, area, 'red')

    # Single shared colorbar at bottom
    fig.subplots_adjust(bottom=0.12)
    cax = fig.add_axes([0.25, 0.05, 0.5, 0.025])
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Completeness S(G)', fontsize=11)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Q1 fields'),
        Line2D([0], [0], color='#EB759D', linewidth=2, label='ERO fields'),
    ]
    fig.legend(handles=legend_elements, loc='lower right', fontsize=9,
               framealpha=0.9, edgecolor='gray', bbox_to_anchor=(0.92, 0.02))

    fig.suptitle('Gaia DR3 Completeness by Magnitude with Euclid Q1 & ERO Fields',
                 fontsize=14, fontweight='bold', y=0.99)

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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Gaia DR3 Completeness Maps at G = {', '.join(str(g) for g in G_VALUES)}")
    print("=" * 70)

    nside = 64
    maps = build_completeness_maps(nside=nside)

    print_completeness_table(maps, nside)

    print("\nGenerating plot...")
    plot_all_panels(maps, nside, OUTPUT_DIR / 'completeness_multi_G_fields.png')

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
