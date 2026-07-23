#!/usr/bin/env python3
"""
Paper-friendly insert version of the Gaia scanning law + Euclid fields plot.

Produces a clean Mollweide map with:
  - Gaia DR3 scanning law transit density
  - Q1 fields (red) and ERO fields (white)
  - Ecliptic |lat|=45° and galactic |lat|=10° boundaries
  - Larger fonts and markers for A4 readability

Usage:
    python src/analyze_dr3_scanning_law_zones_insert.py

Output:
    output/scanning_law_fields_zones_insert.png
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from matplotlib.lines import Line2D
from astropy.coordinates import SkyCoord
import astropy.units as u

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output'

# ── Euclid field definitions ──────────────────────────────────────────────
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
    'EDF-N':     (269.733, +66.018, 22.9),
    'EDF-S':     (61.241, -48.423, 28.1),
    'EDF-F':     (52.932, -28.088, 12.1),
    'LDN 1641':  (85.75, -8.367, 0.5),
}

# ── Label offsets (tuned for insert readability) ──────────────────────────
ERO_OFFSETS = {
    'Horsehead':    (0, +7, 'right'),
    'Messier 78':   (-8, -7, 'left'),
    'Taurus':       (0, +5, 'center'),
    'Barnard 30':   (0, +7, 'center'),
    'NGC 6254':     (0, +5, 'center'),
    'NGC 6397':     (0, -6, 'center'),
    'NGC 6822':     (0, +5, 'center'),
    'Holmberg II':  (+6, 0, 'left'),
    'IC 10':        (0, -6, 'center'),
    'IC 342':       (0, -6, 'center'),
    'NGC 2403':     (-6, 0, 'right'),
    'NGC 6744':     (0, +5, 'center'),
    'Fornax':       (+4, 0, 'left'),
    'Dorado':       (+5, 0, 'left'),
    'Perseus':      (0, -6, 'center'),
    'Abell 2390':   (0, +5, 'center'),
    'Abell 2764':   (0, -5, 'center'),
}

Q1_OFFSETS = {
    'EDF-N':     (-10, 0, 'left'),
    'EDF-S':     (-10, 0, 'right'),
    'EDF-F':     (-8, 0, 'left'),
    'LDN 1641':  (0, -13, 'center'),
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


def _ecliptic_boundary_to_equatorial(ecl_lat_deg, n_pts=360):
    ecl_lon = np.linspace(0, 360, n_pts, endpoint=False)
    sc = SkyCoord(
        lon=ecl_lon * u.deg,
        lat=np.full(n_pts, ecl_lat_deg) * u.deg,
        frame='barycentricmeanecliptic',
    )
    eq = sc.icrs
    return eq.ra.deg, eq.dec.deg


def _galactic_boundary_to_equatorial(gal_lat_deg, n_pts=360):
    gal_lon = np.linspace(0, 360, n_pts, endpoint=False)
    sc = SkyCoord(
        l=gal_lon * u.deg,
        b=np.full(n_pts, gal_lat_deg) * u.deg,
        frame='galactic',
    )
    eq = sc.icrs
    return eq.ra.deg, eq.dec.deg


def _plot_boundary(ra, dec, **kwargs):
    order = np.argsort(ra)
    ra_s, dec_s = ra[order], dec[order]
    dra = np.abs(np.diff(ra_s))
    splits = np.where(dra > 90)[0] + 1
    segments = np.split(np.arange(len(ra_s)), splits)
    for seg in segments:
        if len(seg) < 2:
            continue
        hp.projplot(ra_s[seg], dec_s[seg], lonlat=True, coord='CG', **kwargs)


def draw_field_circle(ra_center, dec_center, area_deg2, color, n_pts=72):
    radius = np.sqrt(area_deg2 / np.pi)
    theta = np.linspace(0, 2 * np.pi, n_pts)
    cos_dec = max(np.cos(np.radians(dec_center)), 0.01)
    ra_pts = ra_center + radius * np.cos(theta) / cos_dec
    dec_pts = np.clip(dec_center + radius * np.sin(theta), -89.9, 89.9)
    hp.projplot(ra_pts, dec_pts, lonlat=True, coord='CG',
                color=color, linewidth=2.0, zorder=10)
    hp.projscatter(ra_center, dec_center, lonlat=True, coord='CG',
                   color=color, s=6, marker='.', zorder=10)


def plot_scanning_law_with_zones(scan_map, nside, output_path):
    fig = plt.figure()

    hp.mollview(scan_map, coord='CG', title='',
                unit='', norm='log',
                cmap='viridis', hold=True, fig=fig)

    fig.set_size_inches(14, 5.6)
    map_ax = fig.axes[0]
    map_ax.set_position([0.02, 0.14, 0.96, 0.84])
    map_ax.set_aspect('auto')

    # Colorbar
    cbar_ax = fig.axes[-1]
    cbar_ax.set_position([0.25, 0.07, 0.50, 0.035])
    cbar_ax.tick_params(labelsize=12)
    cbar_ax.set_xlabel('Number of Gaia FoV transits', fontsize=13,
                       fontweight='bold', labelpad=4)

    # ── Shade excluded regions ─────────────────────────────────────────
    mask_nside = 128
    npix_mask = hp.nside2npix(mask_nside)
    theta_m, phi_m = hp.pix2ang(mask_nside, np.arange(npix_mask))
    ra_m = np.degrees(phi_m)
    dec_m = 90.0 - np.degrees(theta_m)

    sc_mask = SkyCoord(ra=ra_m * u.deg, dec=dec_m * u.deg, frame='icrs')
    ecl_lat = np.abs(sc_mask.barycentricmeanecliptic.lat.deg)
    gal_lat = np.abs(sc_mask.galactic.b.deg)

    excluded = ~((ecl_lat > 45) & (gal_lat > 10))
    hp.projscatter(ra_m[excluded], dec_m[excluded], lonlat=True, coord='CG',
                   color='deepskyblue', alpha=0.6, s=1, marker='s',
                   edgecolors='none', zorder=5)

    # ── Latitude boundary lines ────────────────────────────────────────
    for elat in [+45, -45]:
        ra, dec = _ecliptic_boundary_to_equatorial(elat, n_pts=720)
        _plot_boundary(ra, dec, color='orange', linewidth=1.8,
                       linestyle='--', zorder=8)

    for glat in [+10, -10]:
        ra, dec = _galactic_boundary_to_equatorial(glat, n_pts=720)
        _plot_boundary(ra, dec, color='cyan', linewidth=1.8,
                       linestyle='--', zorder=8)

    # ── ERO fields ─────────────────────────────────────────────────────
    for name, (ra, dec, area) in ERO_FIELDS.items():
        draw_field_circle(ra, dec, area, 'white')
        dra, ddec, ha = ERO_OFFSETS[name]
        hp.projtext(ra + dra, dec + ddec, name, lonlat=True, coord='CG',
                    color='white', fontsize=12, fontweight='bold',
                    ha=ha, zorder=11)

    # ── Q1 fields ──────────────────────────────────────────────────────
    for name, (ra, dec, area) in Q1_FIELDS.items():
        draw_field_circle(ra, dec, area, 'red')
        dra, ddec, ha = Q1_OFFSETS[name]
        hp.projtext(ra + dra, dec + ddec, name, lonlat=True, coord='CG',
                    color='#FF6666', fontsize=12, fontweight='bold',
                    ha=ha, zorder=11)

    # ── Legend ─────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2.5, label='Q1 fields'),
        Line2D([0], [0], color='white', linewidth=2.5, label='ERO fields'),
        Line2D([0], [0], color='orange', linewidth=1.8, linestyle='--',
               label='|ecl. lat| = 45\u00b0'),
        Line2D([0], [0], color='cyan', linewidth=1.8, linestyle='--',
               label='|gal. lat| = 10\u00b0'),
    ]
    plt.legend(handles=legend_elements, loc='lower left', fontsize=13,
               framealpha=0.9, edgecolor='white',
               labelcolor='white', facecolor='#333333',
               bbox_to_anchor=(0.0, -0.2))

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Gaia Scanning Law + Latitude Zones (insert version)")
    print("=" * 70)

    nside = 64
    scan_map = build_scanning_law_map(nside=nside)

    print("\nGenerating insert plot...")
    plot_scanning_law_with_zones(
        scan_map, nside,
        OUTPUT_DIR / 'scanning_law_fields_zones_insert.png',
    )

    print("Done.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
