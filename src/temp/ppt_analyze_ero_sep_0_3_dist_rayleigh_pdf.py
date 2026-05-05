#!/usr/bin/env python3
"""
Rayleigh PDF fit for ERO-Gaia 0.3" angular separations — single panel.

Produces:
  ero_sep_0_3_dist_rayleigh_pdf.png

Usage:
    PGUSER=siwei_zhao_2025 python src/temp/ppt_analyze_ero_sep_0_3_dist_rayleigh_pdf.py
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy.stats import rayleigh

PROJECT_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'docs' / 'slides' / 'figs'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

OUTPUT_DIR.mkdir(exist_ok=True)


def query_ero_gaia_separations(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            SQRT(POWER(e_v_alpha_j2000 - g_ra, 2) + POWER(e_v_delta_j2000 - g_dec, 2)) * 3600
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_alpha_j2000 IS NOT NULL
          AND e_v_delta_j2000 IS NOT NULL
          AND g_ra IS NOT NULL
          AND g_dec IS NOT NULL
    """)
    angular_seps = np.array([r[0] for r in cursor.fetchall()])
    cursor.close()
    conn.close()
    print(f"  Loaded {len(angular_seps):,} ERO-Gaia sources (0.3\" radius)")
    return angular_seps


def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER not set")
        return 1

    print("Querying ERO-Gaia separations...")
    angular_seps = query_ero_gaia_separations(username)

    # Fit Rayleigh to core distribution (0–84th percentile)
    fit_limit = np.percentile(angular_seps, 84)
    params = rayleigh.fit(angular_seps[angular_seps <= fit_limit])
    loc, scale = params
    sigma = scale

    print(f"  Rayleigh σ = {sigma:.6f}\"")

    x_limit = np.percentile(angular_seps, 99.5)
    x_fit   = np.linspace(0, x_limit, 500)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.hist(angular_seps[angular_seps <= x_limit], bins=100, density=True,
            color='steelblue', alpha=0.6, edgecolor='none',
            label=f'ERO-Gaia Data (n={len(angular_seps):,})')
    ax.plot(x_fit, rayleigh.pdf(x_fit, loc, scale), 'r-', linewidth=2.5,
            label=f'Rayleigh Fit (σ={sigma:.6f}″)')

    # Bump transition annotation (dip between ~0.09–0.13 arcsec)
    hist_counts, hist_bins = np.histogram(angular_seps[angular_seps <= x_limit], bins=100)
    bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
    dip_mask = (bin_centers >= 0.09) & (bin_centers <= 0.13)
    if dip_mask.any():
        min_idx = np.argmin(hist_counts[dip_mask])
        transition_sep = bin_centers[dip_mask][min_idx]
        ax.axvline(x=transition_sep, color='limegreen', linestyle='--', linewidth=1.5, alpha=1)
        label_offset = x_limit * 0.03
        ax.text(transition_sep + label_offset, ax.get_ylim()[1] * 0.5,
                f'Bump transition\n≈{transition_sep:.4f}″',
                fontsize=9, ha='left',
                bbox=dict(boxstyle='round', facecolor='limegreen', alpha=0.4))

    ax.set_xlabel('Angular Separation (arcsec)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax.set_title('ERO-Gaia Angular Separation (0.3"): Rayleigh Distribution Fit', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / 'ero_sep_0_3_dist_rayleigh_pdf.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
