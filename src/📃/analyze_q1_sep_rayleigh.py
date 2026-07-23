#!/usr/bin/env python3
"""
Rayleigh PDF fit for Q1-Gaia angular separations — single panel.

Produces:
  output/q1_sep_rayleigh.png

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_q1_sep_rayleigh.py
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy.stats import rayleigh

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_DIR / 'output' / 'q1_sep_rayleigh.png'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'


def query_q1_gaia_separations(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT m_right_ascension, m_declination, g_ra, g_dec
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia
    """)
    rows = cursor.fetchall()
    q1_ra   = np.array([r[0] for r in rows], dtype=float)
    q1_dec  = np.array([r[1] for r in rows], dtype=float)
    g_ra    = np.array([r[2] for r in rows], dtype=float)
    g_dec   = np.array([r[3] for r in rows], dtype=float)
    cos_dec = np.cos(np.radians(g_dec))
    angular_seps = np.sqrt(((q1_ra - g_ra) * cos_dec)**2 + (q1_dec - g_dec)**2) * 3600
    cursor.close()
    conn.close()
    print(f"  Loaded {len(angular_seps):,} Q1-Gaia sources")
    return angular_seps


def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER not set")
        return 1

    print("Querying Q1-Gaia separations...")
    angular_seps = query_q1_gaia_separations(username)

    # Fit Rayleigh to core distribution (0–90.5th percentile)
    fit_limit = np.percentile(angular_seps, 90)
    params = rayleigh.fit(angular_seps[angular_seps <= fit_limit])
    loc, scale = params
    sigma = scale

    print(f"  Rayleigh σ = {sigma:.6f}\"")

    x_limit = np.percentile(angular_seps, 99.5)
    x_fit   = np.linspace(0, x_limit, 500)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.hist(angular_seps[angular_seps <= x_limit], bins=100, density=True,
            color='steelblue', alpha=0.6, edgecolor='none',
            label=f'Q1-Gaia Data (n={len(angular_seps):,})')
    ax.plot(x_fit, rayleigh.pdf(x_fit, loc, scale), 'r-', linewidth=2.5,
            label=f'Rayleigh fit (σ={sigma:.6f}″)')

    ax.set_xlabel('Angular Separation (arcsec)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=14, fontweight='bold', labelpad=15)
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=13, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_FILE}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
