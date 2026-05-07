#!/usr/bin/env python3
"""
Combined NGC 6397 figure:
  Left  — 2D proper-motion histogram of 0.10–0.15" separation bump
  Right — ERO-Gaia separation PDF: corrected vs uncorrected

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_ero_ngc6397_combined.py

Output:
    output/ero_ngc6397_combined.png
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import rayleigh

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_DIR / "output" / "ero_ngc6397_combined.png"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

NGC6397_RA  = 265.10
NGC6397_DEC = -53.58

PM_TOLERANCE      = 2.0    # mas/yr
SKY_POS_TOLERANCE = 0.5    # degrees

OFFSET_RA_MAS  =   38.7
OFFSET_DEC_MAS = -114.7
MAS_TO_DEG     = 1.0 / 3_600_000


def query_data(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()

    # All sources with positions + PM
    print("Querying ERO-Gaia crossmatch data...")
    cursor.execute("""
        SELECT e_v_alpha_j2000, e_v_delta_j2000, g_ra, g_dec, g_pmra, g_pmdec
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    print(f"  Retrieved {len(rows):,} sources")
    return rows


def main():
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set")
        return 1

    rows = query_data(username)

    ero_ra  = np.array([r[0] for r in rows], dtype=float)
    ero_dec = np.array([r[1] for r in rows], dtype=float)
    g_ra    = np.array([r[2] for r in rows], dtype=float)
    g_dec   = np.array([r[3] for r in rows], dtype=float)
    pmra    = np.array([r[4] for r in rows], dtype=float)
    pmdec   = np.array([r[5] for r in rows], dtype=float)
    cos_dec = np.cos(np.radians(g_dec))
    has_pm  = np.isfinite(pmra) & np.isfinite(pmdec)

    # Uncorrected separations
    seps_uncorr = np.sqrt(
        ((ero_ra - g_ra) * cos_dec)**2 + (ero_dec - g_dec)**2
    ) * 3600

    # ── Left panel data: PM distribution of 0.10–0.15" bump ─────────────
    bump_mask = (seps_uncorr >= 0.10) & (seps_uncorr <= 0.15) & has_pm
    pmra_bump  = pmra[bump_mask]
    pmdec_bump = pmdec[bump_mask]
    cluster_pmra  = np.median(pmra_bump)
    cluster_pmdec = np.median(pmdec_bump)
    print(f"  Empirical cluster PM: ({cluster_pmra:.2f}, {cluster_pmdec:.2f}) mas/yr")
    print(f"  Bump sources with PM: {bump_mask.sum():,}")

    # ── Right panel data: corrected separations ──────────────────────────
    sky_dist = np.sqrt(((g_ra - NGC6397_RA) * cos_dec)**2 +
                       (g_dec - NGC6397_DEC)**2)
    sky_mask = sky_dist <= SKY_POS_TOLERANCE
    pm_dist  = np.sqrt((pmra - cluster_pmra)**2 + (pmdec - cluster_pmdec)**2)
    ngc6397_mask = sky_mask & ((~has_pm) | (pm_dist <= PM_TOLERANCE))
    print(f"  NGC 6397 members (total): {ngc6397_mask.sum():,}")

    dra_ang = (ero_ra - g_ra) * cos_dec
    ddec    = ero_dec - g_dec
    dra_ang[ngc6397_mask] -= OFFSET_RA_MAS  * MAS_TO_DEG
    ddec[ngc6397_mask]    -= OFFSET_DEC_MAS * MAS_TO_DEG
    seps_corr = np.sqrt(dra_ang**2 + ddec**2) * 3600

    # Rayleigh fits
    pct_uncorr = np.percentile(seps_uncorr, 86)
    params_uncorr = rayleigh.fit(seps_uncorr[seps_uncorr <= pct_uncorr])
    sigma_uncorr  = params_uncorr[1]

    pct_corr = np.percentile(seps_corr, 92)
    params_corr = rayleigh.fit(seps_corr[seps_corr <= pct_corr])
    sigma_corr  = params_corr[1]

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, (ax_pm, ax_sep) = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={"width_ratios": [1, 1.3]},
    )
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.13, top=0.92, wspace=0.22)

    # ── Left: 2D PM histogram ───────────────────────────────────────────
    h = ax_pm.hist2d(pmra_bump, pmdec_bump, bins=60,
                     cmap="Blues", norm=LogNorm(), cmin=1)
    ax_pm.axhline(0, color="red", ls="--", lw=1.5, alpha=0.6)
    ax_pm.axvline(0, color="red", ls="--", lw=1.5, alpha=0.6)
    ax_pm.add_patch(Circle((cluster_pmra, cluster_pmdec), 3.5,
                           color="red", fill=False, lw=2, ls="--", alpha=0.6))
    ax_pm.annotate(
        f"Clump centre\n({cluster_pmra:.2f}, {cluster_pmdec:.2f})",
        xy=(cluster_pmra, cluster_pmdec),
        xytext=(cluster_pmra + 7, cluster_pmdec - 7),
        fontsize=11, color="red", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5, alpha=0.8),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="red", alpha=0.8),
    )

    divider = make_axes_locatable(ax_pm)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cb = plt.colorbar(h[3], cax=cax)
    cb.set_label("Count", fontsize=12, fontweight="bold")
    cb.ax.tick_params(labelsize=11)

    ax_pm.set_xlim(-36, 36)
    ax_pm.set_ylim(-48, 24)
    ax_pm.set_aspect("equal")
    ax_pm.xaxis.set_major_locator(plt.MultipleLocator(12))
    ax_pm.yaxis.set_major_locator(plt.MultipleLocator(8))
    ax_pm.tick_params(labelsize=12)
    ax_pm.set_xlabel("PMRA (mas/yr)", fontsize=13, fontweight="bold")
    ax_pm.set_ylabel("PMDEC (mas/yr)", fontsize=13, fontweight="bold")

    # ── Right: separation PDF ────────────────────────────────────────────
    x_limit = 0.30
    x_fit = np.linspace(0, x_limit, 500)

    ax_sep.hist(seps_uncorr[seps_uncorr <= x_limit], bins=100, density=True,
                color='#FFCDD2', alpha=0.5, edgecolor='none',
                label=f'Uncorrected (n={len(seps_uncorr):,})')
    ax_sep.plot(x_fit, rayleigh.pdf(x_fit, *params_uncorr),
                color='#FFCDD2', lw=2.5,
                label=f'Uncorrected fit (\u03c3={sigma_uncorr:.4f}\u2033)')

    ax_sep.hist(seps_corr[seps_corr <= x_limit], bins=100, density=True,
                color='steelblue', alpha=0.5, edgecolor='none',
                label=f'Corrected (n={len(seps_corr):,})')
    ax_sep.plot(x_fit, rayleigh.pdf(x_fit, *params_corr), 'r-', lw=2.5,
                label=f'Corrected fit (\u03c3={sigma_corr:.4f}\u2033)')

    ax_sep.set_xlabel("Angular Separation (arcsec)", fontsize=13, fontweight="bold")
    ax_sep.set_ylabel("Probability Density", fontsize=13, fontweight="bold")
    ax_sep.set_ylim(0, 20)
    ax_sep.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax_sep.tick_params(labelsize=12)
    ax_sep.legend(fontsize=11, loc="upper right")
    ax_sep.grid(True, alpha=0.3)

    plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
