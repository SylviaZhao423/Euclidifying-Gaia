#!/usr/bin/env python3
"""
ERO-Gaia 0.1-0.15" clump: 2D proper motion histogram — single panel.

Produces:
  ero_0_1_0_15_arcsec_clump_pm2d.png

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_ero_0_1_0_15_arcsec_clump_pm2d.py
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

PROJECT_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'docs' / 'slides' / 'figs'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

OUTPUT_DIR.mkdir(exist_ok=True)


def query_clump_pm(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT g_pmra, g_pmdec
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE SQRT(POWER(e_v_alpha_j2000 - g_ra, 2) + POWER(e_v_delta_j2000 - g_dec, 2)) * 3600 >= 0.10
          AND SQRT(POWER(e_v_alpha_j2000 - g_ra, 2) + POWER(e_v_delta_j2000 - g_dec, 2)) * 3600 <= 0.15
          AND g_pmra IS NOT NULL
          AND g_pmdec IS NOT NULL
    """)
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    pmra  = np.array([r[0] for r in results])
    pmdec = np.array([r[1] for r in results])
    print(f"  Loaded {len(pmra):,} sources in 0.1-0.15\" clump")
    return pmra, pmdec


def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER not set")
        return 1

    print("Querying ERO-Gaia clump (0.1-0.15\") proper motions...")
    pmra, pmdec = query_clump_pm(username)

    fig, ax = plt.subplots(figsize=(6, 7))

    h = ax.hist2d(pmra, pmdec, bins=60, cmap='Blues', norm=LogNorm(), cmin=1)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

    clump_pmra  =  3
    clump_pmdec = -18
    ax.add_patch(Circle((clump_pmra, clump_pmdec), 3.5,
                         color='red', fill=False, linewidth=1.5, linestyle='--', alpha=0.6))
    ax.annotate(f'Clump Center\n({clump_pmra:.1f}, {clump_pmdec:.1f})',
                xy=(clump_pmra, clump_pmdec),
                xytext=(clump_pmra + 8, clump_pmdec - 8),
                fontsize=8, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.6),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.6))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(h[3], cax=cax)
    cbar.set_label('Count', fontsize=9)

    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=12))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=10))
    ax.set_aspect('equal')
    ax.set_xlim(-36, 36)
    ax.set_ylim(-48, 24)
    ax.set_xlabel('PMRA (mas/yr)', fontsize=11, fontweight='bold')
    ax.set_ylabel('PMDEC (mas/yr)', fontsize=11, fontweight='bold')
    # ax.set_title('ERO-Gaia 0.1-0.15" Clump: 2D Proper Motion Distribution', fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()
    out = OUTPUT_DIR / 'ero_0_1_0_15_arcsec_clump_pm2d.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
