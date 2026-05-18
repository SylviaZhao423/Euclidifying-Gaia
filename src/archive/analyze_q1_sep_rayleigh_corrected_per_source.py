#!/usr/bin/env python3
"""
Q1-Gaia angular separation: full 5-parameter per-source astrometric correction.

Uses astropy.SkyCoord.apply_space_motion with each source's complete Gaia
astrometric solution (ra, dec, parallax → distance, pmra, pmdec) — i.e.
per-source 5-parameter propagation rather than the simple per-source PM-only
formula in analyze_q1_sep_rayleigh_corrected.py.

Three variants are computed and overlaid for direct comparison:
  - uncorrected   : raw small-angle sep (Gaia 2016.0 vs Q1)
  - PM-only       : per-source pmra/pmdec, hand-rolled (matches earlier script)
  - 5-par astropy : per-source ra/dec/parallax/pmra/pmdec via astropy

Reads (WSDB):
  siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia
    columns: m_right_ascension, m_declination,
             g_ra, g_dec, g_parallax, g_pmra, g_pmdec, g_ref_epoch

Writes:
  output/q1_sep_rayleigh_corrected_per_source.png

Usage:
  PGUSER=siwei_zhao_2025 python src/slides/analyze_q1_sep_rayleigh_corrected_per_source.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rayleigh

import astropy.coordinates as ac
import astropy.units as u
from astropy.time import Time

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_DIR / "output" / "q1_sep_rayleigh_corrected_per_source.png"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

GAIA_REF_EPOCH = 2016.0
Q1_OBS_EPOCH   = 2024.0
DELTA_T_YEARS  = Q1_OBS_EPOCH - GAIA_REF_EPOCH

MAS_TO_DEG = 1.0 / 3_600_000


def query_data(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cur = conn.cursor()
    cur.execute("""
        SELECT m_right_ascension, m_declination,
               g_ra, g_dec, g_parallax, g_pmra, g_pmdec,
               COALESCE(p_phz_classification, -1) AS phz,
               g_astrometric_params_solved      AS params_solved
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia
    """)
    rows = cur.fetchall()
    cur.close(); conn.close()
    print(f"  Loaded {len(rows):,} Q1-Gaia sources")
    return rows


def per_source_pm_correction(q1_ra, q1_dec, g_ra, g_dec, pmra, pmdec):
    """Hand-rolled per-source PM-only correction (Gaia 2016 → Q1 2024)."""
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
    return sep


def per_source_5par_correction(q1_ra, q1_dec, g_ra, g_dec, plx, pmra, pmdec):
    """astropy 5-par per-source propagation (handles parallax + PM coherently).

    Sources with NaN/non-finite PM use raw Gaia 2016 position. Sources with
    non-positive parallax are propagated as PM-only (parallax set to a tiny
    positive value, distance ≈ 1/0.001 = 1 Mpc, so parallactic effect → 0).
    """
    pmra_safe  = np.where(np.isfinite(pmra),  pmra,  0.0)
    pmdec_safe = np.where(np.isfinite(pmdec), pmdec, 0.0)
    plx_safe   = np.where(np.isfinite(plx) & (plx > 0.001), plx, 0.001)

    c = ac.SkyCoord(
        ra=g_ra * u.deg,
        dec=g_dec * u.deg,
        distance=ac.Distance(parallax=plx_safe * u.mas, allow_negative=False),
        pm_ra_cosdec=pmra_safe * u.mas / u.yr,
        pm_dec=pmdec_safe * u.mas / u.yr,
        radial_velocity=0 * u.km / u.s,
        obstime=Time(GAIA_REF_EPOCH, format="decimalyear"),
        frame="icrs",
    )
    c_q1 = c.apply_space_motion(
        new_obstime=Time(Q1_OBS_EPOCH, format="decimalyear")
    )
    g_ra_q1  = c_q1.ra.deg
    g_dec_q1 = c_q1.dec.deg

    cos_dec = np.cos(np.radians(g_dec_q1))
    sep = np.sqrt(
        ((q1_ra - g_ra_q1) * cos_dec) ** 2 +
        (q1_dec - g_dec_q1) ** 2
    ) * 3600
    return sep


def fit_rayleigh_core(seps, pct: float = 90.0):
    cut = np.percentile(seps, pct)
    params = rayleigh.fit(seps[seps <= cut])
    return params, cut


def main() -> int:
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set"); return 1

    print("Querying Q1-Gaia separation data (5-par astrometry)...")
    rows = query_data(username)

    q1_ra  = np.array([r[0] for r in rows], dtype=float)
    q1_dec = np.array([r[1] for r in rows], dtype=float)
    g_ra   = np.array([r[2] for r in rows], dtype=float)
    g_dec  = np.array([r[3] for r in rows], dtype=float)
    plx    = np.array([r[4] for r in rows], dtype=float)
    pmra   = np.array([r[5] for r in rows], dtype=float)
    pmdec  = np.array([r[6] for r in rows], dtype=float)
    phz    = np.array([r[7] for r in rows], dtype=int)
    nsol   = np.array([r[8] for r in rows], dtype=int)  # 3, 31, or 95

    has_pm  = np.isfinite(pmra) & np.isfinite(pmdec)
    has_plx = np.isfinite(plx)
    print(f"  with PM:       {has_pm.sum():,} ({100*has_pm.mean():.1f}%)")
    print(f"  with parallax: {has_plx.sum():,} ({100*has_plx.mean():.1f}%)")
    print(f"  parallax > 5 mas (nearby): {(has_plx & (plx > 5)).sum():,}")
    print(f"  PHZ STAR (=1):    {(phz == 1).sum():,}")
    print(f"  PHZ GALAXY (=2):  {(phz == 2).sum():,}")

    # Uncorrected
    cos_dec = np.cos(np.radians(g_dec))
    seps_uncorr = np.sqrt(
        ((q1_ra - g_ra) * cos_dec) ** 2 +
        (q1_dec - g_dec) ** 2
    ) * 3600

    # PM-only per-source
    print("\n[1/2] PM-only per-source correction (hand-rolled)...")
    seps_pm = per_source_pm_correction(q1_ra, q1_dec, g_ra, g_dec, pmra, pmdec)

    # 5-par per-source via astropy
    print("[2/2] 5-par per-source correction (astropy SkyCoord)...")
    seps_5p = per_source_5par_correction(q1_ra, q1_dec, g_ra, g_dec,
                                          plx, pmra, pmdec)

    # PM-only and 5-par produce numerically identical separations (Δ = 0.000
    # mas at 95th %ile per-source) because both Gaia DR3 and MER positions
    # are barycentric ICRS, so parallax cancels. We therefore drop PM-only
    # from the plot and keep only the more general 5-par per-source curve.
    delta_pm5 = (seps_5p - seps_pm) * 1000  # mas
    print(f"  per-source |sep_5par - sep_PM| median = "
          f"{np.median(np.abs(delta_pm5)):.3f} mas, "
          f"95th %ile = {np.percentile(np.abs(delta_pm5), 95):.3f} mas "
          f"(→ PM-only dropped from plot, identical to 5-par)")

    # PM-only and 5-par produce numerically identical separations (parallax
    # cancels: both Gaia DR3 and MER are barycentric ICRS), so PM-only is
    # dropped from the plot. Sources split by Gaia PM availability:
    #   has_pm  → 5-par (params=31) + 6-par (params=95) → PM corrected
    #   ~has_pm → 2-par (params=3)  → no correction possible (raw sep)
    seps_pm_avail = seps_5p[has_pm]
    seps_no_pm    = seps_uncorr[~has_pm]

    # Per-population fit percentile (No-PM has a much heavier tail —
    # fitting to its 90th %ile pulls σ to the right of where the histogram
    # peaks. A tighter cut focuses the fit on the visible core.)
    PCT_U   = 90
    PCT_PM  = 90
    PCT_NPM = 92   # No-PM core is much narrower than its tail

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

    # Diagnostic: what does σ for No-PM look like at several percentiles?
    print()
    print("  No-PM σ vs fit percentile (diagnostic):")
    for pct in (40, 50, 60, 70, 80, 90):
        p_, cut_ = fit_rayleigh_core(seps_no_pm, pct=pct)
        print(f"    pct={pct:>3}  cut={cut_:.4f}″  σ={p_[1]:.6f}\"")

    # ── Plot ──────────────────────────────────────────────────────────────
    x_limit = 0.6   # fixed scale shared with ERO plot for direct comparison
    x_fit   = np.linspace(0, x_limit, 500)
    bin_edges = np.linspace(0, x_limit, 101)  # 100 shared bins

    fig, ax = plt.subplots(figsize=(10, 5.0))

    # Layered widest → narrowest so the tallest peaks sit on top.
    # Uncorrected (all) — widest, lowest peak — drawn first
    ax.hist(seps_uncorr[seps_uncorr <= x_limit], bins=bin_edges, density=True,
            color="#FFCDD2", alpha=0.55, edgecolor="none", zorder=1,
            label=f"Uncorrected, all (n={len(seps_uncorr):,})")
    ax.plot(x_fit, rayleigh.pdf(x_fit, *p_u), color="#C62828", linewidth=2.0,
            zorder=4,
            label=f"Uncorrected fit (σ={sigma_u:.4f}″)")

    # No PM (2-par) — medium width — middle layer
    ax.hist(seps_no_pm[seps_no_pm <= x_limit], bins=bin_edges,
            density=True, color="#FFCC80", alpha=0.6, edgecolor="none",
            zorder=2,
            label=f"No PM, raw (n={len(seps_no_pm):,})")
    ax.plot(x_fit, rayleigh.pdf(x_fit, *p_npm),
            color="#E65100", linewidth=2.0, linestyle=":", zorder=5,
            label=f"No-PM fit (σ={sigma_npm:.4f}″)")

    # PM available, corrected — narrowest, tallest peak — drawn last (on top)
    ax.hist(seps_pm_avail[seps_pm_avail <= x_limit], bins=bin_edges,
            density=True, color="#A5D6A7", alpha=0.65, edgecolor="none",
            zorder=3,
            label=f"PM available, corrected (n={len(seps_pm_avail):,})")
    ax.plot(x_fit, rayleigh.pdf(x_fit, *p_pm),
            color="#1B5E20", linewidth=2.0, linestyle="--", zorder=6,
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
