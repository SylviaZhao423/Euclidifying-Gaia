#!/usr/bin/env python3
"""
Q1-Gaia angular separation: PM-corrected, split by PHZ classification.

Tests the reviewer's hypothesis that the heavy tail in the corrected-Q1
Rayleigh plot is driven by extended sources (galaxies) rather than stars.

Pipeline:
  1. Query Q1-Gaia matches with PHZ classification (1=STAR, 2=GALAXY).
  2. Propagate Gaia 2016.0 → Q1 epoch using pmra/pmdec where available.
  3. Compute corrected separations.
  4. Fit Rayleigh independently to STAR and GALAXY subsamples.
  5. Plot overlaid histograms with per-class fits.

Reads (WSDB):
  siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia
    columns: m_right_ascension, m_declination, g_ra, g_dec, g_pmra, g_pmdec,
             p_phz_classification

Writes:
  output/q1_sep_rayleigh_corrected_per_class.png

Usage:
  PGUSER=siwei_zhao_2025 python src/slides/analyze_q1_sep_rayleigh_corrected_per_class.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rayleigh

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_DIR / "output" / "q1_sep_rayleigh_corrected_per_class.png"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

GAIA_REF_EPOCH = 2016.0
Q1_OBS_EPOCH   = 2024.0
DELTA_T_YEARS  = Q1_OBS_EPOCH - GAIA_REF_EPOCH

MAS_TO_DEG = 1.0 / 3_600_000

PHZ_STAR   = 1
PHZ_GALAXY = 2


def query_data(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cur = conn.cursor()
    cur.execute("""
        SELECT m_right_ascension, m_declination,
               g_ra, g_dec, g_pmra, g_pmdec,
               p_phz_classification
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia
        WHERE p_phz_classification IN (1, 2)
    """)
    rows = cur.fetchall()
    cur.close(); conn.close()
    print(f"  Loaded {len(rows):,} Q1-Gaia sources (PHZ ∈ {{1,2}})")
    return rows


def compute_corrected_seps(q1_ra, q1_dec, g_ra, g_dec, pmra, pmdec):
    """Return corrected separation in arcsec for each source."""
    cos_dec = np.cos(np.radians(g_dec))
    has_pm = np.isfinite(pmra) & np.isfinite(pmdec)

    dra_pm  = np.where(has_pm, pmra  * DELTA_T_YEARS * MAS_TO_DEG, 0.0)
    ddec_pm = np.where(has_pm, pmdec * DELTA_T_YEARS * MAS_TO_DEG, 0.0)

    g_ra_q1  = g_ra  + dra_pm / cos_dec
    g_dec_q1 = g_dec + ddec_pm

    sep = np.sqrt(
        ((q1_ra - g_ra_q1) * cos_dec) ** 2 +
        (q1_dec - g_dec_q1) ** 2
    ) * 3600
    return sep, has_pm


def fit_rayleigh_core(seps, pct: float = 90.0):
    cut = np.percentile(seps, pct)
    params = rayleigh.fit(seps[seps <= cut])
    return params, cut


def main() -> int:
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set"); return 1

    print("Querying Q1-Gaia separation data with PHZ classification...")
    rows = query_data(username)

    q1_ra  = np.array([r[0] for r in rows], dtype=float)
    q1_dec = np.array([r[1] for r in rows], dtype=float)
    g_ra   = np.array([r[2] for r in rows], dtype=float)
    g_dec  = np.array([r[3] for r in rows], dtype=float)
    pmra   = np.array([r[4] for r in rows], dtype=float)
    pmdec  = np.array([r[5] for r in rows], dtype=float)
    phz    = np.array([r[6] for r in rows], dtype=int)

    seps_corr, has_pm = compute_corrected_seps(
        q1_ra, q1_dec, g_ra, g_dec, pmra, pmdec)

    star_mask = phz == PHZ_STAR
    gal_mask  = phz == PHZ_GALAXY

    n_star = int(star_mask.sum())
    n_gal  = int(gal_mask.sum())
    print(f"  STARs:   {n_star:,}  ({100*has_pm[star_mask].mean():.1f}% with PM)")
    print(f"  GALAXIEs: {n_gal:,}  ({100*has_pm[gal_mask].mean():.1f}% with PM)")

    # Per-class Rayleigh fits
    p_star, cut_star = fit_rayleigh_core(seps_corr[star_mask], pct=90)
    p_gal,  cut_gal  = fit_rayleigh_core(seps_corr[gal_mask],  pct=90)
    sigma_star = p_star[1]
    sigma_gal  = p_gal[1]

    print(f"  σ (stars)    = {sigma_star:.6f}\"")
    print(f"  σ (galaxies) = {sigma_gal:.6f}\"")
    print(f"  ratio gal/star = {sigma_gal/sigma_star:.2f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    x_limit = np.percentile(seps_corr, 99.5)
    x_fit   = np.linspace(0, x_limit, 500)
    bin_edges = np.linspace(0, x_limit, 101)  # 100 shared bins

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.hist(seps_corr[star_mask & (seps_corr <= x_limit)], bins=bin_edges,
            density=True, color="steelblue", alpha=0.5, edgecolor="none",
            label=f"Stars (n={n_star:,})")
    ax.plot(x_fit, rayleigh.pdf(x_fit, *p_star), color="navy",
            linewidth=2.2, label=f"Stars Rayleigh (σ={sigma_star:.4f}″)")

    ax.hist(seps_corr[gal_mask & (seps_corr <= x_limit)], bins=bin_edges,
            density=True, color="#E57373", alpha=0.5, edgecolor="none",
            label=f"Galaxies (n={n_gal:,})")
    ax.plot(x_fit, rayleigh.pdf(x_fit, *p_gal), color="#B71C1C",
            linewidth=2.2, label=f"Galaxies Rayleigh (σ={sigma_gal:.4f}″)")

    ax.set_xlabel("Angular Separation (arcsec)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=14, fontweight="bold",
                  labelpad=15)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
