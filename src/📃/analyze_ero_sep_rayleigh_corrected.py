#!/usr/bin/env python3
"""
ERO-Gaia 0.3" separation: PDF histogram — corrected vs uncorrected — single panel.

Produces:
  output/ero_sep_rayleigh_corrected.png

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_ero_sep_rayleigh_corrected.py
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy.stats import rayleigh

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_DIR / 'output' / 'ero_sep_rayleigh_corrected.png'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

NGC6397_RA    =  265.10   # degrees
NGC6397_DEC   =  -53.58   # degrees

SKY_POS_TOLERANCE = 0.5   # degrees (~1.85× tidal radius; wider to capture faint members without Gaia PM)
PM_TOLERANCE      = 2     # mas/yr  — consistent with analyze_ero_ngc6397.py diagnostic

OFFSET_RA_MAS  =   38.7   # mas angular offset — empirically measured in analyze_ero_ngc6397.py (|ΔPM| ≤ 2 mas/yr)
OFFSET_DEC_MAS = -114.7   # mas
MAS_TO_DEG     = 1.0 / 3_600_000


def query_data(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT e_v_alpha_j2000, e_v_delta_j2000, g_ra, g_dec
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
    """)
    rows_uncorr = cursor.fetchall()

    cursor.execute("""
        SELECT e_v_alpha_j2000, e_v_delta_j2000, g_ra, g_dec, g_pmra, g_pmdec
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
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

    # Uncorrected separations (with cos(dec) for RA component)
    ero_ra_u  = np.array([r[0] for r in rows_uncorr], dtype=float)
    ero_dec_u = np.array([r[1] for r in rows_uncorr], dtype=float)
    g_ra_u    = np.array([r[2] for r in rows_uncorr], dtype=float)
    g_dec_u   = np.array([r[3] for r in rows_uncorr], dtype=float)
    cos_dec_u = np.cos(np.radians(g_dec_u))
    seps_uncorr = np.sqrt(
        ((ero_ra_u - g_ra_u) * cos_dec_u)**2 +
        (ero_dec_u - g_dec_u)**2
    ) * 3600

    # Identify NGC 6397 members for offset correction
    gaia_ras  = np.array([r[2] for r in rows_corr], dtype=float)
    gaia_decs = np.array([r[3] for r in rows_corr], dtype=float)
    pmras     = np.array([r[4] for r in rows_corr], dtype=float)  # None → NaN
    pmdecs    = np.array([r[5] for r in rows_corr], dtype=float)  # None → NaN
    cos_dec   = np.cos(np.radians(gaia_decs))

    has_pm   = np.isfinite(pmras) & np.isfinite(pmdecs)

    # Step 1: Empirical cluster PM from bump sources (0.10–0.15" separation).
    # Bump sources are predominantly cluster members displaced by the systematic
    # offset — their median PM is a clean estimate uncontaminated by field stars.
    bump_mask     = (seps_uncorr >= 0.10) & (seps_uncorr <= 0.15) & has_pm
    cluster_pmra  = np.nanmedian(pmras[bump_mask])
    cluster_pmdec = np.nanmedian(pmdecs[bump_mask])
    print(f"  NGC 6397 empirical PM: ({cluster_pmra:.2f}, {cluster_pmdec:.2f}) mas/yr")

    # Step 2: Full cluster member selection using sky position + empirical PM
    sky_dist = np.sqrt(((gaia_ras - NGC6397_RA) * cos_dec)**2 +
                       (gaia_decs - NGC6397_DEC)**2)
    sky_mask     = sky_dist <= SKY_POS_TOLERANCE
    pm_dist      = np.sqrt((pmras - cluster_pmra)**2 + (pmdecs - cluster_pmdec)**2)
    ngc6397_mask = sky_mask & ((~has_pm) | (pm_dist <= PM_TOLERANCE))
    n_pm_sky   = (has_pm  & sky_mask & (pm_dist <= PM_TOLERANCE)).sum()
    n_nopm_sky = (~has_pm & sky_mask).sum()
    print(f"  NGC 6397 members (PM+sky):     {n_pm_sky:,}")
    print(f"  NGC 6397 members (sky-only):   {n_nopm_sky:,}")
    print(f"  NGC 6397 members (total):      {ngc6397_mask.sum():,}")

    # Corrected separations: work in angular space directly using the measured
    # angular offsets — no coordinate↔angular conversion needed.
    dra_ang = (np.array([r[0] for r in rows_corr], dtype=float) - gaia_ras) * cos_dec
    ddec    =  np.array([r[1] for r in rows_corr], dtype=float) - gaia_decs
    dra_ang[ngc6397_mask] -= OFFSET_RA_MAS  * MAS_TO_DEG
    ddec[ngc6397_mask]    -= OFFSET_DEC_MAS * MAS_TO_DEG
    seps_corr = np.sqrt(dra_ang**2 + ddec**2) * 3600

    # Rayleigh fits
    pct_limit_uncorr = np.percentile(seps_uncorr, 86)
    params_uncorr = rayleigh.fit(seps_uncorr[seps_uncorr <= pct_limit_uncorr])
    sigma_uncorr  = params_uncorr[1]

    pct_limit_corr = np.percentile(seps_corr, 92)
    params_corr = rayleigh.fit(seps_corr[seps_corr <= pct_limit_corr])
    sigma_corr  = params_corr[1]

    x_limit = 0.30  # match x-axis of analyze_ero_sep_rayleigh.py
    x_fit   = np.linspace(0, x_limit, 500)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4.5))

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

    ax.set_xlabel('Angular Separation (arcsec)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=14, fontweight='bold', labelpad=15)
    # ax.set_title('ERO-Gaia angular separation (NGC 6397 corrected)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 20)
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
