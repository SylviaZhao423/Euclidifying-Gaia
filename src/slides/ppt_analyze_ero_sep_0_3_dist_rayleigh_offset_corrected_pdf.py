#!/usr/bin/env python3
"""
ERO-Gaia 0.3" separation: PDF histogram — corrected vs uncorrected — single panel.

Produces:
  ero_sep_0_3_dist_rayleigh_offset_corrected_pdf.png

Usage:
    PGUSER=siwei_zhao_2025 python src/temp/ppt_analyze_ero_sep_0_3_dist_rayleigh_offset_corrected_pdf.py
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

NGC6397_RA    =  265.10   # degrees
NGC6397_DEC   =  -53.58   # degrees
NGC6397_PMRA  =    3.24   # mas/yr
NGC6397_PMDEC =  -17.66   # mas/yr

SKY_POS_TOLERANCE = 0.6   # degrees
PM_TOLERANCE      = 3     # mas/yr

OFFSET_RA_MAS  =   63.9   # mas
OFFSET_DEC_MAS = -113.5   # mas
MAS_TO_DEG     = 1.0 / 3_600_000
OFFSET_RA_DEG  = OFFSET_RA_MAS  * MAS_TO_DEG
OFFSET_DEC_DEG = OFFSET_DEC_MAS * MAS_TO_DEG


def query_data(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT e_v_alpha_j2000, e_v_delta_j2000, g_ra, g_dec
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_alpha_j2000 IS NOT NULL AND e_v_delta_j2000 IS NOT NULL
          AND g_ra IS NOT NULL AND g_dec IS NOT NULL
    """)
    rows_uncorr = cursor.fetchall()

    cursor.execute("""
        SELECT e_v_alpha_j2000, e_v_delta_j2000, g_ra, g_dec, g_pmra, g_pmdec
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_alpha_j2000 IS NOT NULL AND e_v_delta_j2000 IS NOT NULL
          AND g_ra IS NOT NULL AND g_dec IS NOT NULL
          AND g_pmra IS NOT NULL AND g_pmdec IS NOT NULL
    """)
    rows_corr = cursor.fetchall()
    cursor.close()
    conn.close()

    print(f"  Loaded {len(rows_uncorr):,} sources (uncorrected)")
    print(f"  Loaded {len(rows_corr):,} sources with PM (corrected)")
    return rows_uncorr, rows_corr


def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER not set")
        return 1

    print("Querying ERO-Gaia separation data...")
    rows_uncorr, rows_corr = query_data(username)

    # Uncorrected separations
    seps_uncorr = np.sqrt(
        (np.array([r[0] for r in rows_uncorr]) - np.array([r[2] for r in rows_uncorr]))**2 +
        (np.array([r[1] for r in rows_uncorr]) - np.array([r[3] for r in rows_uncorr]))**2
    ) * 3600

    # Identify NGC 6397 members for offset correction
    gaia_ras  = np.array([r[2] for r in rows_corr])
    gaia_decs = np.array([r[3] for r in rows_corr])
    pmras     = np.array([r[4] for r in rows_corr])
    pmdecs    = np.array([r[5] for r in rows_corr])

    sky_dist = np.sqrt((gaia_ras - NGC6397_RA)**2 + (gaia_decs - NGC6397_DEC)**2)
    pm_dist  = np.sqrt((pmras - NGC6397_PMRA)**2  + (pmdecs - NGC6397_PMDEC)**2)
    ngc6397_mask = (sky_dist <= SKY_POS_TOLERANCE) & (pm_dist <= PM_TOLERANCE)
    print(f"  NGC 6397 members: {np.sum(ngc6397_mask):,}")

    # Corrected separations
    dra  = np.array([r[0] for r in rows_corr]) - gaia_ras
    ddec = np.array([r[1] for r in rows_corr]) - gaia_decs
    dra[ngc6397_mask]  -= OFFSET_RA_DEG
    ddec[ngc6397_mask] -= OFFSET_DEC_DEG
    seps_corr = np.sqrt(dra**2 + ddec**2) * 3600

    # Rayleigh fits
    pct_limit_uncorr = np.percentile(seps_uncorr, 84)
    params_uncorr = rayleigh.fit(seps_uncorr[seps_uncorr <= pct_limit_uncorr])
    sigma_uncorr  = params_uncorr[1]

    pct_limit_corr = np.percentile(seps_corr, 91.5)
    params_corr = rayleigh.fit(seps_corr[seps_corr <= pct_limit_corr])
    sigma_corr  = params_corr[1]

    x_limit = max(np.percentile(seps_uncorr, 99.5), np.percentile(seps_corr, 99.5))
    x_fit   = np.linspace(0, x_limit, 500)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.hist(seps_uncorr[seps_uncorr <= x_limit], bins=100, density=True,
            color='#FFCDD2', alpha=0.5, edgecolor='none',
            label=f'Uncorrected (n={len(seps_uncorr):,})')
    ax.plot(x_fit, rayleigh.pdf(x_fit, *params_uncorr), color='#FFCDD2', linewidth=2.5,
            label=f'Uncorrected fit (σ={sigma_uncorr:.4f}″)')

    ax.hist(seps_corr[seps_corr <= x_limit], bins=100, density=True,
            color='steelblue', alpha=0.5, edgecolor='none',
            label=f'Corrected (n={len(seps_corr):,})')
    ax.plot(x_fit, rayleigh.pdf(x_fit, *params_corr), 'r-', linewidth=2.5,
            label=f'Corrected fit (σ={sigma_corr:.4f}″)')

    ax.set_xlabel('Angular Separation (arcsec)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    
    # ax.set_title('ERO-Gaia Angular Separation (0.3"):\nCorrected Rayleigh Distribution Fit (NGC 6397 offset)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / 'ero_sep_0_3_dist_rayleigh_offset_corrected_pdf.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
