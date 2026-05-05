#!/usr/bin/env python3
"""
ERO-Gaia PM clump: matched pair astrometric offsets — single panel.

Produces:
  ero_gaia_pm_cluster_offsets.png

Usage:
    PGUSER=siwei_zhao_2025 python src/temp/ppt_analyze_pm_cluster_offsets.py
"""

import psycopg2
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

PROJECT_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'docs' / 'slides' / 'figs'

CLUMP_PMRA   =  3.2   # mas/yr
CLUMP_PMDEC  = -17.7  # mas/yr
PM_TOLERANCE =  1     # mas/yr


def query_offsets():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER not set")
        return None

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT
            e_v_alpha_j2000,
            e_v_delta_j2000,
            g_ra,
            g_dec
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE SQRT(POWER(e_v_alpha_j2000 - g_ra, 2) + POWER(e_v_delta_j2000 - g_dec, 2)) * 3600 >= 0.1
          AND SQRT(POWER(e_v_alpha_j2000 - g_ra, 2) + POWER(e_v_delta_j2000 - g_dec, 2)) * 3600 < 0.15
          AND g_pmra >= {CLUMP_PMRA - PM_TOLERANCE}
          AND g_pmra <= {CLUMP_PMRA + PM_TOLERANCE}
          AND g_pmdec >= {CLUMP_PMDEC - PM_TOLERANCE}
          AND g_pmdec <= {CLUMP_PMDEC + PM_TOLERANCE}
    """)
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    DEG_TO_MAS = 3_600_000
    residual_ra_mas  = np.array([(r[0] - r[2]) * DEG_TO_MAS for r in results])
    residual_dec_mas = np.array([(r[1] - r[3]) * DEG_TO_MAS for r in results])
    n = len(results)
    print(f"  Loaded {n:,} sources in PM clump")
    return residual_ra_mas, residual_dec_mas, n


def main():
    print("Querying PM clump astrometric offsets...")
    result = query_offsets()
    if result is None:
        return 1
    residual_ra_mas, residual_dec_mas, n_sources = result

    mean_ra_mas  = np.mean(residual_ra_mas)
    mean_dec_mas = np.mean(residual_dec_mas)

    fig, ax = plt.subplots(figsize=(7, 7))

    h = ax.hist2d(residual_ra_mas, residual_dec_mas,
                  bins=40, cmap='Blues', cmin=1, norm=LogNorm())

    ax.scatter([0], [0], marker='x', s=100, color='red', linewidths=2,
               label='Gaia reference (0, 0)', zorder=5)
    ax.scatter([mean_ra_mas], [mean_dec_mas], marker='x', s=100, color='limegreen', linewidths=2,
               label=f'Mean ERO positions offset: ({mean_ra_mas:.1f}, {mean_dec_mas:.1f}) mas',
               zorder=5)

    ax.add_patch(Circle((0, 0), 100, color='red', fill=False, linestyle='--', linewidth=2))
    ax.add_patch(Circle((0, 0), 150, color='red', fill=False, linestyle='--', linewidth=2))
    ax.add_artist(Line2D([0], [0], color='red', linestyle='--', linewidth=2,
                         label='Separation: 100-150 mas'))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(h[3], cax=cax)
    cbar.set_label('Count', fontsize=9)

    ax.set_xlabel('RA offset: ERO - Gaia (mas)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Dec offset: ERO - Gaia (mas)', fontsize=11, fontweight='bold')
    # ax.set_title(f'PM(3, -18) mas/yr ⊂ Sep(0.1, 0.15)" Clump Analysis:\n{n_sources:,} Matched Pairs Astrometric Offsets', fontsize=11, fontweight='bold', y=1)
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper left', framealpha=0.5, markerscale=0.8)

    plt.tight_layout()
    out = OUTPUT_DIR / 'ero_gaia_pm_cluster_offsets.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
