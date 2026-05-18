#!/usr/bin/env python3
"""
Combined NGC 6397 figure (corrected pipeline):
  Left  — 2D proper-motion histogram of 0.10–0.15" separation bump
  Right — ERO-Gaia separation PDF from the per-source PM + neighbour-PM
          proxy correction (mirrors analyze_ero_sep_rayleigh_corrected.py)

Parallel to analyze_ero_ngc6397_combined.py (which paired the same PM
diagnostic with the legacy NGC 6397 bulk-offset corrected curve). This
version uses the new corrected pipeline:
  - Per-source PM propagation for sources with Gaia 5/6-par solutions
    (Gaia 2016.0 → ERO 2023.5)
  - Neighbour-PM proxy (median of K nearest 5/6-par PMs within 30″)
    for 2-par sources, automatically capturing the NGC 6397 cluster
    bulk PM where present
  - Exact great-circle separation via astropy SkyCoord.separation

Reads (WSDB):
  siwei_zhao_2025.crossmatch_ero_gaia_0_3
    columns: e_v_alpha_j2000, e_v_delta_j2000,
             g_ra, g_dec, g_pmra, g_pmdec,
             g_astrometric_params_solved

Writes:
  output/ero_ngc6397_combined_corrected.png

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_ero_ngc6397_combined_corrected.py
"""

import os
import sys
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import cKDTree
from scipy.stats import rayleigh

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_DIR / "output" / "ero_ngc6397_combined_corrected.png"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

# Epochs for PM propagation
GAIA_REF_EPOCH = 2016.0
ERO_OBS_EPOCH  = 2023.5
DELTA_T_YEARS  = ERO_OBS_EPOCH - GAIA_REF_EPOCH

MAS_TO_DEG = 1.0 / 3_600_000

# Neighbour-PM proxy parameters
PROXY_RADIUS_ARCSEC  = 30.0
PROXY_MIN_NEIGHBOURS = 5


# ── Helpers (mirror analyze_ero_sep_rayleigh_corrected.py) ───────────────

def _sky_separation_arcsec(ra1, dec1, ra2, dec2):
    """Exact great-circle separation in arcsec via astropy (haversine)."""
    c1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg)
    c2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg)
    return c1.separation(c2).arcsec


def per_source_pm_correction(e_ra, e_dec, g_ra, g_dec, pmra, pmdec):
    cos_dec = np.cos(np.radians(g_dec))
    pmra_safe  = np.where(np.isfinite(pmra),  pmra,  0.0)
    pmdec_safe = np.where(np.isfinite(pmdec), pmdec, 0.0)
    dra_pm  = pmra_safe  * DELTA_T_YEARS * MAS_TO_DEG
    ddec_pm = pmdec_safe * DELTA_T_YEARS * MAS_TO_DEG
    g_ra_e  = g_ra  + dra_pm / cos_dec
    g_dec_e = g_dec + ddec_pm
    return _sky_separation_arcsec(e_ra, e_dec, g_ra_e, g_dec_e)


def _sky_to_xyz(ra_deg, dec_deg):
    ra  = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    cd  = np.cos(dec)
    return np.column_stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)])


def neighbour_pm_proxy(g_ra, g_dec, pmra, pmdec, has_pm,
                       radius_arcsec=PROXY_RADIUS_ARCSEC,
                       min_neighbours=PROXY_MIN_NEIGHBOURS):
    src_5p_xyz = _sky_to_xyz(g_ra[has_pm], g_dec[has_pm])
    src_np_xyz = _sky_to_xyz(g_ra[~has_pm], g_dec[~has_pm])
    chord_radius = 2.0 * np.sin(np.radians(radius_arcsec / 3600.0) / 2.0)

    print(f"  Building KDTree over {len(src_5p_xyz):,} 5/6-par sources...")
    tree = cKDTree(src_5p_xyz)
    print(f"  Querying neighbours within {radius_arcsec:.0f}″ for "
          f"{len(src_np_xyz):,} no-PM sources...")
    neighbour_lists = tree.query_ball_point(src_np_xyz, r=chord_radius,
                                             workers=-1)

    pmra_5p  = pmra[has_pm]
    pmdec_5p = pmdec[has_pm]
    n_np = len(src_np_xyz)
    proxy_pmra  = np.full(n_np, np.nan)
    proxy_pmdec = np.full(n_np, np.nan)
    for i, neighbours in enumerate(neighbour_lists):
        if len(neighbours) >= min_neighbours:
            proxy_pmra[i]  = np.median(pmra_5p[neighbours])
            proxy_pmdec[i] = np.median(pmdec_5p[neighbours])
    n_with_proxy = int(np.isfinite(proxy_pmra).sum())
    print(f"  Sources with valid proxy PM: {n_with_proxy:,} / {n_np:,} "
          f"({100*n_with_proxy/n_np:.1f}%)")
    return proxy_pmra, proxy_pmdec


def fit_rayleigh_core(seps, pct: float = 90.0):
    cut = np.percentile(seps, pct)
    return rayleigh.fit(seps[seps <= cut]), cut


# ── Query ──────────────────────────────────────────────────────────────

def query_data(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cur = conn.cursor()
    cur.execute("""
        SELECT e_v_alpha_j2000, e_v_delta_j2000,
               g_ra, g_dec, g_pmra, g_pmdec,
               g_astrometric_params_solved
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
    """)
    rows = cur.fetchall()
    cur.close(); conn.close()
    print(f"  Loaded {len(rows):,} ERO-Gaia sources")
    return rows


# ── Main ───────────────────────────────────────────────────────────────

def main():
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set"); return 1

    print("Querying ERO-Gaia data...")
    rows = query_data(username)

    e_ra   = np.array([r[0] for r in rows], dtype=float)
    e_dec  = np.array([r[1] for r in rows], dtype=float)
    g_ra   = np.array([r[2] for r in rows], dtype=float)
    g_dec  = np.array([r[3] for r in rows], dtype=float)
    pmra   = np.array([r[4] for r in rows], dtype=float)
    pmdec  = np.array([r[5] for r in rows], dtype=float)
    has_pm = np.isfinite(pmra) & np.isfinite(pmdec)

    # Uncorrected sep (used for both panels)
    seps_uncorr = _sky_separation_arcsec(e_ra, e_dec, g_ra, g_dec)

    # ── Left panel data: PM clump from the 0.10–0.15″ bump ──────────────
    bump_mask = (seps_uncorr >= 0.10) & (seps_uncorr <= 0.15) & has_pm
    pmra_bump, pmdec_bump = pmra[bump_mask], pmdec[bump_mask]
    cluster_pmra  = np.median(pmra_bump)
    cluster_pmdec = np.median(pmdec_bump)
    print(f"  Empirical NGC 6397 PM: ({cluster_pmra:.2f}, "
          f"{cluster_pmdec:.2f}) mas/yr   (from {bump_mask.sum():,} bump sources)")

    # ── Right panel data: per-source PM + neighbour proxy ───────────────
    print(f"\nPropagating Gaia {GAIA_REF_EPOCH} → ERO {ERO_OBS_EPOCH} "
          f"(Δt = {DELTA_T_YEARS:.1f} yr)...")
    seps_corr = per_source_pm_correction(
        e_ra, e_dec, g_ra, g_dec, pmra, pmdec)

    print("\nComputing neighbour-PM proxy for 2-par sources...")
    proxy_pmra, proxy_pmdec = neighbour_pm_proxy(
        g_ra, g_dec, pmra, pmdec, has_pm)

    pmra_with_proxy  = pmra.copy()
    pmdec_with_proxy = pmdec.copy()
    no_pm_idx = np.where(~has_pm)[0]
    pmra_with_proxy[no_pm_idx]  = proxy_pmra
    pmdec_with_proxy[no_pm_idx] = proxy_pmdec

    seps_proxy = per_source_pm_correction(
        e_ra, e_dec, g_ra, g_dec, pmra_with_proxy, pmdec_with_proxy)

    seps_pm_avail   = seps_corr[has_pm]
    seps_no_pm_prox = seps_proxy[~has_pm]

    # Per-population fit percentiles (match analyze_ero_sep_rayleigh_corrected.py)
    PCT_U, PCT_PM, PCT_PRX = 88, 96, 87
    p_u,    _ = fit_rayleigh_core(seps_uncorr,     pct=PCT_U)
    p_pm,   _ = fit_rayleigh_core(seps_pm_avail,   pct=PCT_PM)
    p_prx,  _ = fit_rayleigh_core(seps_no_pm_prox, pct=PCT_PRX)
    sigma_u, sigma_pm, sigma_prx = p_u[1], p_pm[1], p_prx[1]

    print()
    print(f"  σ uncorrected             = {sigma_u:.4f}″")
    print(f"  σ PM available, corrected = {sigma_pm:.4f}″")
    print(f"  σ no-PM, proxy-corrected  = {sigma_prx:.4f}″")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, (ax_pm, ax_sep) = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={"width_ratios": [1, 1.3]},
    )
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.13, top=0.92, wspace=0.22)

    # ── Left: 2D PM histogram of bump sources ───────────────────────────
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

    # ── Right: 3-population separation PDF (PM + proxy + uncorrected) ───
    x_limit = 0.4
    x_fit = np.linspace(0, x_limit, 500)
    bin_edges = np.linspace(0, x_limit, 101)

    # Layer order (bottom → top): PM-available → No-PM proxy → Uncorrected
    ax_sep.hist(seps_pm_avail[seps_pm_avail <= x_limit], bins=bin_edges,
                density=True, color="#A5D6A7", alpha=0.65, edgecolor="none",
                zorder=1,
                label=f"PM available, corrected (n={len(seps_pm_avail):,})")
    ax_sep.plot(x_fit, rayleigh.pdf(x_fit, *p_pm),
                color="#1B5E20", linewidth=2.0, linestyle="--", zorder=4,
                label=f"PM-available fit (σ={sigma_pm:.4f}″)")

    ax_sep.hist(seps_no_pm_prox[seps_no_pm_prox <= x_limit], bins=bin_edges,
                density=True, color="#FFCC80", alpha=0.7, edgecolor="none",
                zorder=2,
                label=f"No PM, proxy-corrected (n={len(seps_no_pm_prox):,})")
    ax_sep.plot(x_fit, rayleigh.pdf(x_fit, *p_prx),
                color="#E65100", linewidth=2.0, linestyle="-.", zorder=5,
                label=f"No-PM proxy fit (σ={sigma_prx:.4f}″)")

    ax_sep.hist(seps_uncorr[seps_uncorr <= x_limit], bins=bin_edges,
                density=True, color="#90CAF9", alpha=0.55, edgecolor="none",
                zorder=3,
                label=f"Uncorrected, all (n={len(seps_uncorr):,})")
    ax_sep.plot(x_fit, rayleigh.pdf(x_fit, *p_u), color="#0D47A1", linewidth=2.0,
                zorder=6,
                label=f"Uncorrected fit (σ={sigma_u:.4f}″)")
    ax_sep.axvline(0.3, color="dimgray", linewidth=1.2, linestyle="--", alpha=0.85)
    ax_sep.set_xlim(-0.01, x_limit)
    ax_sep.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax_sep.set_ylim(0, 28)
    ax_sep.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax_sep.set_xlabel("Angular Separation (arcsec)", fontsize=13,
                      fontweight="bold")
    ax_sep.set_ylabel("Probability Density", fontsize=13, fontweight="bold")
    ax_sep.tick_params(labelsize=12)
    ax_sep.legend(fontsize=13, loc="upper right", framealpha=0.85)
    ax_sep.grid(True, alpha=0.3)

    plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
