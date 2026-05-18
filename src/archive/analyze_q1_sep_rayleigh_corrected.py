#!/usr/bin/env python3
"""
Q1-Gaia angular separation: PDF histogram — corrected vs uncorrected.

Mirrors analyze_ero_sep_rayleigh_corrected.py but for the Q1 cross-match.
The "correction" propagates Gaia DR3 catalogue positions (epoch 2016.0)
forward to the Q1 observation epoch using each source's pmra/pmdec, then
recomputes the small-angle separation against Q1's MER coordinates.

This addresses the reviewer's question about the heavy tail in the
uncorrected Q1 Rayleigh plot — much of that tail is high-PM stars whose
Gaia 2016 positions sit ~tens of mas off their Q1 (~2024) positions.

Reads (WSDB):
  siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia
    columns: m_right_ascension, m_declination, g_ra, g_dec, g_pmra, g_pmdec

Writes:
  output/q1_sep_rayleigh_corrected.png

Usage:
  PGUSER=siwei_zhao_2025 python src/slides/analyze_q1_sep_rayleigh_corrected.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rayleigh

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_DIR / "output" / "q1_sep_rayleigh_corrected.png"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

# Epoch baselines for PM propagation
GAIA_REF_EPOCH = 2016.0   # Gaia DR3 catalogue reference epoch
Q1_OBS_EPOCH   = 2024.0   # Euclid Q1 observation epoch (mean; per-tile not modelled)
DELTA_T_YEARS  = Q1_OBS_EPOCH - GAIA_REF_EPOCH  # = 8.0

MAS_TO_DEG = 1.0 / 3_600_000


def query_data(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT m_right_ascension, m_declination, g_ra, g_dec, g_pmra, g_pmdec
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia
    """)
    rows = cursor.fetchall()
    cursor.close(); conn.close()
    print(f"  Loaded {len(rows):,} Q1-Gaia sources")
    return rows


def main() -> int:
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set"); return 1

    print("Querying Q1-Gaia separation data...")
    rows = query_data(username)

    q1_ra  = np.array([r[0] for r in rows], dtype=float)
    q1_dec = np.array([r[1] for r in rows], dtype=float)
    g_ra   = np.array([r[2] for r in rows], dtype=float)
    g_dec  = np.array([r[3] for r in rows], dtype=float)
    pmra   = np.array([r[4] for r in rows], dtype=float)  # mas/yr (None → NaN)
    pmdec  = np.array([r[5] for r in rows], dtype=float)  # mas/yr (None → NaN)
    cos_dec = np.cos(np.radians(g_dec))

    # ── Uncorrected separations ───────────────────────────────────────────
    seps_uncorr = np.sqrt(
        ((q1_ra - g_ra) * cos_dec) ** 2 +
        (q1_dec - g_dec) ** 2
    ) * 3600

    # ── Corrected separations: propagate Gaia 2016 → Q1 obs epoch ─────────
    has_pm = np.isfinite(pmra) & np.isfinite(pmdec)
    n_with_pm = int(has_pm.sum())
    print(f"  Sources with Gaia 5-par astrometry (PM available): "
          f"{n_with_pm:,} / {len(rows):,} ({100*n_with_pm/len(rows):.1f}%)")

    # Δra in degrees: pmra is in mas/yr already includes cos(dec) factor
    # (Gaia convention: pmra* = pmra·cos(dec)). So sky-projected RA shift in
    # angular degrees is pmra·dt·MAS_TO_DEG, no additional cos(dec) needed.
    dra_pm  = np.where(has_pm, pmra  * DELTA_T_YEARS * MAS_TO_DEG, 0.0)
    ddec_pm = np.where(has_pm, pmdec * DELTA_T_YEARS * MAS_TO_DEG, 0.0)

    # Apply propagation: Gaia position at Q1 epoch
    g_ra_q1  = g_ra  + dra_pm / cos_dec  # ra increment in true RA
    g_dec_q1 = g_dec + ddec_pm

    seps_corr = np.sqrt(
        ((q1_ra - g_ra_q1) * cos_dec) ** 2 +
        (q1_dec - g_dec_q1) ** 2
    ) * 3600

    # ── Rayleigh fits (core only, percentile cut) ─────────────────────────
    pct_limit_uncorr = np.percentile(seps_uncorr, 90)
    params_uncorr = rayleigh.fit(seps_uncorr[seps_uncorr <= pct_limit_uncorr])
    sigma_uncorr = params_uncorr[1]

    pct_limit_corr = np.percentile(seps_corr, 90)
    params_corr = rayleigh.fit(seps_corr[seps_corr <= pct_limit_corr])
    sigma_corr = params_corr[1]

    print(f"  Rayleigh σ (uncorrected) = {sigma_uncorr:.6f}\"")
    print(f"  Rayleigh σ (corrected)   = {sigma_corr:.6f}\"")
    print(f"  σ ratio (corr/uncorr)    = {sigma_corr/sigma_uncorr:.3f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    x_limit = np.percentile(seps_uncorr, 99.5)
    x_fit   = np.linspace(0, x_limit, 500)
    bin_edges = np.linspace(0, x_limit, 101)  # 100 shared bins

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.hist(seps_uncorr[seps_uncorr <= x_limit], bins=bin_edges, density=True,
            color="#FFCDD2", alpha=0.5, edgecolor="none",
            label=f"Uncorrected (n={len(seps_uncorr):,})")
    ax.plot(x_fit, rayleigh.pdf(x_fit, *params_uncorr),
            color="#FFCDD2", linewidth=2.5,
            label=f"Uncorrected fit (σ={sigma_uncorr:.4f}″)")

    ax.hist(seps_corr[seps_corr <= x_limit], bins=bin_edges, density=True,
            color="steelblue", alpha=0.5, edgecolor="none",
            label=f"Corrected (n={len(seps_corr):,})")
    ax.plot(x_fit, rayleigh.pdf(x_fit, *params_corr), "r-", linewidth=2.5,
            label=f"Corrected fit (σ={sigma_corr:.4f}″)")

    ax.set_xlabel("Angular Separation (arcsec)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=14, fontweight="bold",
                  labelpad=15)
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=13, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
