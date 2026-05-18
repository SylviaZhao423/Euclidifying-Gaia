#!/usr/bin/env python3
"""
ERO-Gaia angular separation: per-source PM correction split by Gaia
astrometric-solution availability. Mirrors
analyze_q1_sep_rayleigh_corrected_per_source.py but for the ERO sample.

Sources are split by Gaia DR3 g_astrometric_params_solved:
  - PM available (params=31 5-par + params=95 6-par) → per-source PM
    correction (Gaia 2016.0 → ERO observation epoch).
  - No PM (params=3 2-par solutions) → no correction possible, raw sep.
The uncorrected (all sources) distribution is overlaid as reference.

Reads (WSDB):
  siwei_zhao_2025.crossmatch_ero_gaia_0_3
    columns: e_v_alpha_j2000, e_v_delta_j2000,
             g_ra, g_dec, g_parallax, g_pmra, g_pmdec,
             g_astrometric_params_solved

Writes:
  output/ero_sep_rayleigh_corrected_per_source.png

Usage:
  PGUSER=siwei_zhao_2025 python src/slides/analyze_ero_sep_rayleigh_corrected_per_source.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rayleigh

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_DIR / "output" / "ero_sep_rayleigh_corrected_per_source.png"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

# Epochs for PM propagation
GAIA_REF_EPOCH = 2016.0
ERO_OBS_EPOCH  = 2023.5    # ERO program observed mid-2023 (single bulk epoch)
DELTA_T_YEARS  = ERO_OBS_EPOCH - GAIA_REF_EPOCH

MAS_TO_DEG = 1.0 / 3_600_000


def query_data(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cur = conn.cursor()
    cur.execute("""
        SELECT e_v_alpha_j2000, e_v_delta_j2000,
               g_ra, g_dec, g_parallax, g_pmra, g_pmdec,
               g_astrometric_params_solved AS params_solved
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
    """)
    rows = cur.fetchall()
    cur.close(); conn.close()
    print(f"  Loaded {len(rows):,} ERO-Gaia sources")
    return rows


def per_source_pm_correction(e_ra, e_dec, g_ra, g_dec, pmra, pmdec):
    """Per-source PM-only correction (Gaia 2016 → ERO obs epoch)."""
    cos_dec = np.cos(np.radians(g_dec))
    has_pm = np.isfinite(pmra) & np.isfinite(pmdec)
    dra_pm  = np.where(has_pm, pmra  * DELTA_T_YEARS * MAS_TO_DEG, 0.0)
    ddec_pm = np.where(has_pm, pmdec * DELTA_T_YEARS * MAS_TO_DEG, 0.0)
    g_ra_e  = g_ra  + dra_pm / cos_dec
    g_dec_e = g_dec + ddec_pm
    sep = np.sqrt(((e_ra - g_ra_e) * cos_dec) ** 2 +
                  (e_dec - g_dec_e) ** 2) * 3600
    return sep


def fit_rayleigh_core(seps, pct: float = 90.0):
    cut = np.percentile(seps, pct)
    params = rayleigh.fit(seps[seps <= cut])
    return params, cut


def main() -> int:
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set"); return 1

    print("Querying ERO-Gaia separation data...")
    rows = query_data(username)

    e_ra   = np.array([r[0] for r in rows], dtype=float)
    e_dec  = np.array([r[1] for r in rows], dtype=float)
    g_ra   = np.array([r[2] for r in rows], dtype=float)
    g_dec  = np.array([r[3] for r in rows], dtype=float)
    plx    = np.array([r[4] for r in rows], dtype=float)
    pmra   = np.array([r[5] for r in rows], dtype=float)
    pmdec  = np.array([r[6] for r in rows], dtype=float)
    nsol   = np.array([r[7] for r in rows], dtype=int)  # 3, 31, or 95

    has_pm  = np.isfinite(pmra) & np.isfinite(pmdec)
    has_plx = np.isfinite(plx)
    print(f"  with PM:       {has_pm.sum():,} ({100*has_pm.mean():.1f}%)")
    print(f"  with parallax: {has_plx.sum():,} ({100*has_plx.mean():.1f}%)")
    print(f"  parallax > 5 mas (nearby): {(has_plx & (plx > 5)).sum():,}")
    print(f"  2-par sources (no PM):    {(nsol == 3).sum():,}")
    print(f"  5-par sources:             {(nsol == 31).sum():,}")
    print(f"  6-par sources (5-par + pcol): {(nsol == 95).sum():,}")

    # Uncorrected sep
    cos_dec = np.cos(np.radians(g_dec))
    seps_uncorr = np.sqrt(((e_ra - g_ra) * cos_dec) ** 2 +
                          (e_dec - g_dec) ** 2) * 3600

    # PM-corrected sep (per-source PM)
    print(f"\nPropagating Gaia {GAIA_REF_EPOCH} → ERO {ERO_OBS_EPOCH} "
          f"(Δt = {DELTA_T_YEARS:.1f} yr)...")
    seps_corr = per_source_pm_correction(e_ra, e_dec, g_ra, g_dec, pmra, pmdec)

    # Split by Gaia PM availability
    seps_pm_avail = seps_corr[has_pm]
    seps_no_pm    = seps_uncorr[~has_pm]

    # Per-population fit percentile (No-PM has a heavier tail —
    # tighter cut focuses the fit on the visible core).
    PCT_U   = 90
    PCT_PM  = 90
    PCT_NPM = 60

    p_u,   cut_u   = fit_rayleigh_core(seps_uncorr,    pct=PCT_U)
    p_pm,  cut_pm  = fit_rayleigh_core(seps_pm_avail,  pct=PCT_PM)
    p_npm, cut_npm = fit_rayleigh_core(seps_no_pm,     pct=PCT_NPM)
    sigma_u   = p_u[1]
    sigma_pm  = p_pm[1]
    sigma_npm = p_npm[1]

    print()
    print(f"  σ uncorrected, all  (n={len(seps_uncorr):,})       "
          f"= {sigma_u:.6f}\"  (fit ≤ {PCT_U}th %ile = {cut_u:.4f}″)")
    print(f"  σ PM available, corrected (n={len(seps_pm_avail):,}) "
          f"= {sigma_pm:.6f}\"  (fit ≤ {PCT_PM}th %ile = {cut_pm:.4f}″)")
    print(f"  σ no PM, raw (2-par, n={len(seps_no_pm):,})         "
          f"= {sigma_npm:.6f}\"  (fit ≤ {PCT_NPM}th %ile = {cut_npm:.4f}″)")

    # No-PM σ vs percentile diagnostic
    print()
    print("  No-PM σ vs fit percentile (diagnostic):")
    for pct in (40, 50, 60, 70, 80, 90):
        p_, cut_ = fit_rayleigh_core(seps_no_pm, pct=pct)
        print(f"    pct={pct:>3}  cut={cut_:.4f}″  σ={p_[1]:.6f}\"")

    # ── Plot ──────────────────────────────────────────────────────────────
    x_limit = 0.6   # fixed scale shared with Q1 plot for direct comparison
    x_fit   = np.linspace(0, x_limit, 500)
    bin_edges = np.linspace(0, x_limit, 101)  # 100 shared bins

    fig, ax = plt.subplots(figsize=(10, 5.0))

    # Layered widest → narrowest so tallest peaks sit on top.
    ax.hist(seps_uncorr[seps_uncorr <= x_limit], bins=bin_edges, density=True,
            color="#FFCDD2", alpha=0.55, edgecolor="none", zorder=1,
            label=f"Uncorrected, all (n={len(seps_uncorr):,})")
    ax.plot(x_fit, rayleigh.pdf(x_fit, *p_u), color="#C62828", linewidth=2.0,
            zorder=4,
            label=f"Uncorrected fit (σ={sigma_u:.4f}″)")

    ax.hist(seps_no_pm[seps_no_pm <= x_limit], bins=bin_edges,
            density=True, color="#FFCC80", alpha=0.6, edgecolor="none",
            zorder=2,
            label=f"No PM, raw (n={len(seps_no_pm):,})")
    ax.plot(x_fit, rayleigh.pdf(x_fit, *p_npm),
            color="#E65100", linewidth=2.0, linestyle=":", zorder=5,
            label=f"No-PM fit (σ={sigma_npm:.4f}″)")

    ax.hist(seps_pm_avail[seps_pm_avail <= x_limit], bins=bin_edges,
            density=True, color="#FFCC80", alpha=0.65, edgecolor="none",
            zorder=3,
            label=f"PM available, corrected (n={len(seps_pm_avail):,})")
    ax.plot(x_fit, rayleigh.pdf(x_fit, *p_pm),
            color="#E65100", linewidth=2.0, linestyle="--", zorder=6,
            label=f"PM-available fit (σ={sigma_pm:.4f}″)")

    ax.set_xlabel("Angular Separation (arcsec)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=14, fontweight="bold",
                  labelpad=15)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=9, loc="upper right", ncol=1, framealpha=0.85)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
