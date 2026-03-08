#!/usr/bin/env python3
"""
Rayleigh PDF fit for Q1-Gaia angular separations — single panel.

Produces:
  q1_sep_dist_rayleigh_pdf.png  — PDF histogram with Rayleigh fit overlay

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_q1_sep_dist_rayleigh_pdf.py
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy.stats import rayleigh

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'output'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

OUTPUT_DIR.mkdir(exist_ok=True)


def query_q1_gaia_separations(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            SQRT(POWER(m_right_ascension - g_ra, 2) + POWER(m_declination - g_dec, 2)) * 3600
        FROM siwei_zhao_2025.crossmatch_q1_mer_gaia
        WHERE m_right_ascension IS NOT NULL
          AND m_declination IS NOT NULL
          AND g_ra IS NOT NULL
          AND g_dec IS NOT NULL
    """)
    angular_seps = np.array([r[0] for r in cursor.fetchall()])
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
    fit_limit = np.percentile(angular_seps, 90.5)
    params = rayleigh.fit(angular_seps[angular_seps <= fit_limit])
    loc, scale = params
    sigma = scale

    print(f"  Rayleigh σ = {sigma:.6f}\"")

    x_limit = np.percentile(angular_seps, 99.5)
    x_fit   = np.linspace(0, x_limit, 500)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(angular_seps[angular_seps <= x_limit], bins=100, density=True,
            color='steelblue', alpha=0.6, edgecolor='none',
            label=f'Q1-Gaia Data (n={len(angular_seps):,})')
    ax.plot(x_fit, rayleigh.pdf(x_fit, loc, scale), 'r-', linewidth=2.5,
            label=f'Rayleigh Fit (σ={sigma:.6f}″)')

    ax.set_xlabel('Angular Separation (arcsec)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax.set_title('Q1-Gaia Angular Separation: Rayleigh Distribution Fit', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / 'q1_sep_dist_rayleigh_pdf.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
