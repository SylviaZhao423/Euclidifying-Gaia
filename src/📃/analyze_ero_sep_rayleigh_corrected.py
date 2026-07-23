#!/usr/bin/env python3
"""
ERO-Gaia angular separation: per-source PM correction with a neighbour-PM
proxy applied to 2-par (no-PM) sources.

Extends analyze_ero_sep_rayleigh_corrected_per_source.py with a fourth curve:

  No-PM, proxy-corrected
    For every 2-par Gaia source (no pmra/pmdec), look up the K nearest
    5/6-par Gaia neighbours within RADIUS_ARCSEC on the sky and take the
    median (pmra, pmdec) as a proxy PM. Then propagate Gaia 2016 → ERO
    obs epoch with the proxy.

  In a cluster (NGC 6397 etc.) the local 5/6-par members all share the
  cluster bulk PM, so the median proxy IS the cluster PM → 2-par cluster
  members are properly propagated and the bump goes away.
  In the field, neighbouring 5/6-par stars have random Galactic PMs that
  cancel in the median → ~zero proxy → no spurious correction.

Reads (WSDB):
  siwei_zhao_2025.crossmatch_ero_gaia_0_3
    columns: e_v_alpha_j2000, e_v_delta_j2000,
             g_ra, g_dec, g_pmra, g_pmdec,
             g_astrometric_params_solved

Writes:
  output/ero_sep_rayleigh_corrected_per_source_proxy.png

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_ero_sep_rayleigh_corrected_per_source_proxy.py
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
from scipy.stats import rayleigh


def _sky_separation_arcsec(ra1, dec1, ra2, dec2):
    """Exact great-circle separation in arcsec via astropy (haversine)."""
    c1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg)
    c2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg)
    return c1.separation(c2).arcsec


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE          = PROJECT_DIR / "output" / "ero_sep_rayleigh_corrected.png"
OUTPUT_FILE_COMBINED = PROJECT_DIR / "output" / "ero_ngc6397_combined_corrected.png"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

# Epochs for PM propagation
GAIA_REF_EPOCH = 2016.0
ERO_OBS_EPOCH  = 2023.5
DELTA_T_YEARS  = ERO_OBS_EPOCH - GAIA_REF_EPOCH

MAS_TO_DEG = 1.0 / 3_600_000

# Neighbour-PM proxy parameters (tunable)
# (Proxy radius / min-neighbour parameters live in src/build_gaia_pm_proxy.py;
#  this script just reads the resulting siwei_zhao_2025.gaia_pm_proxy table.)


def query_data(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT src.e_v_alpha_j2000, src.e_v_delta_j2000,
               src.g_ra, src.g_dec, src.g_pmra, src.g_pmdec,
               src.g_astrometric_params_solved AS params_solved,
               proxy.proxy_pmra, proxy.proxy_pmdec, proxy.n_neighbours
        FROM {username}.crossmatch_ero_gaia_0_3 AS src
        LEFT JOIN {username}.gaia_pm_proxy AS proxy
            ON proxy.g_source_id = src.g_source_id
    """)
    rows = cur.fetchall()
    cur.close(); conn.close()
    print(f"  Loaded {len(rows):,} ERO-Gaia sources (LEFT JOIN gaia_pm_proxy)")
    return rows


def per_source_pm_correction(e_ra, e_dec, g_ra, g_dec, pmra, pmdec):
    """Per-source PM-only correction (Gaia 2016 → ERO obs epoch).

    Sources with NaN PM are propagated by zero (no correction).
    """
    cos_dec = np.cos(np.radians(g_dec))
    pmra_safe  = np.where(np.isfinite(pmra),  pmra,  0.0)
    pmdec_safe = np.where(np.isfinite(pmdec), pmdec, 0.0)
    dra_pm  = pmra_safe  * DELTA_T_YEARS * MAS_TO_DEG
    ddec_pm = pmdec_safe * DELTA_T_YEARS * MAS_TO_DEG
    g_ra_e  = g_ra  + dra_pm / cos_dec
    g_dec_e = g_dec + ddec_pm
    sep = _sky_separation_arcsec(e_ra, e_dec, g_ra_e, g_dec_e)
    return sep


# Proxy PM lookup is now done at SQL-JOIN time against the canonical
# {username}.gaia_pm_proxy table — see query_data() above. The previous
# in-script cKDTree-based neighbour_pm_proxy() function has been removed
# (use src/build_gaia_pm_proxy.py to (re)build the proxy table itself).


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

    e_ra        = np.array([r[0] for r in rows], dtype=float)
    e_dec       = np.array([r[1] for r in rows], dtype=float)
    g_ra        = np.array([r[2] for r in rows], dtype=float)
    g_dec       = np.array([r[3] for r in rows], dtype=float)
    pmra        = np.array([r[4] for r in rows], dtype=float)
    pmdec       = np.array([r[5] for r in rows], dtype=float)
    nsol        = np.array([r[6] for r in rows], dtype=int)
    proxy_pmra  = np.array([r[7] for r in rows], dtype=float)   # NaN where no proxy
    proxy_pmdec = np.array([r[8] for r in rows], dtype=float)   # NaN where no proxy
    n_neighbours = np.array([r[9] if r[9] is not None else 0 for r in rows],
                             dtype=int)

    has_pm    = np.isfinite(pmra)        & np.isfinite(pmdec)
    has_proxy = np.isfinite(proxy_pmra)  & np.isfinite(proxy_pmdec)
    print(f"  with PM (5/6-par):                     "
          f"{has_pm.sum():>9,} ({100*has_pm.mean():.1f}%)")
    print(f"  without PM (2-par), with proxy lookup: "
          f"{int((~has_pm & has_proxy).sum()):>9,} "
          f"({100*(~has_pm & has_proxy).mean():.1f}%)")
    print(f"  without PM (2-par), no proxy:          "
          f"{int((~has_pm & ~has_proxy).sum()):>9,} "
          f"({100*(~has_pm & ~has_proxy).mean():.1f}%)")

    # Uncorrected sep (exact great-circle via astropy)
    seps_uncorr = _sky_separation_arcsec(e_ra, e_dec, g_ra, g_dec)

    # PM-corrected sep (per-source PM where available, zero otherwise)
    print(f"\nPropagating Gaia {GAIA_REF_EPOCH} → ERO {ERO_OBS_EPOCH} "
          f"(Δt = {DELTA_T_YEARS:.1f} yr)...")
    seps_corr = per_source_pm_correction(
        e_ra, e_dec, g_ra, g_dec, pmra, pmdec)

    # Fold per-source PM AND DB-side proxy PMs into one PM vector:
    #   priority order = (a) source's own PM → (b) proxy PM → (c) zero
    pmra_with_proxy  = np.where(has_pm, pmra,  proxy_pmra)
    pmdec_with_proxy = np.where(has_pm, pmdec, proxy_pmdec)
    seps_proxy = per_source_pm_correction(
        e_ra, e_dec, g_ra, g_dec, pmra_with_proxy, pmdec_with_proxy)

    # Slices for plotting / fitting
    seps_pm_avail   = seps_corr[has_pm]                    # 5/6-par corrected
    seps_no_pm_raw  = seps_uncorr[~has_pm]                 # 2-par raw (diag only)
    seps_no_pm_prox = seps_proxy[~has_pm]                  # 2-par with proxy

    # Per-population fit percentile
    PCT_U   = 88
    PCT_PM  = 96
    PCT_PRX = 87

    p_u,    cut_u   = fit_rayleigh_core(seps_uncorr,     pct=PCT_U)
    p_pm,   cut_pm  = fit_rayleigh_core(seps_pm_avail,   pct=PCT_PM)
    p_prx,  cut_prx = fit_rayleigh_core(seps_no_pm_prox, pct=PCT_PRX)
    sigma_u   = p_u[1]
    sigma_pm  = p_pm[1]
    sigma_prx = p_prx[1]

    # Bump diagnostic: raw vs proxy-corrected, printed only (not plotted)
    in_bump_raw   = ((seps_no_pm_raw  >= 0.10) & (seps_no_pm_raw  <= 0.15)).sum()
    in_bump_proxy = ((seps_no_pm_prox >= 0.10) & (seps_no_pm_prox <= 0.15)).sum()

    print()
    print(f"  σ uncorrected, all     (n={len(seps_uncorr):,})       "
          f"= {sigma_u:.6f}\"  (≤{PCT_U}th %ile = {cut_u:.4f}″)")
    print(f"  σ PM available, corr   (n={len(seps_pm_avail):,}) "
          f"= {sigma_pm:.6f}\"  (≤{PCT_PM}th %ile = {cut_pm:.4f}″)")
    print(f"  σ no PM, proxy-corr    (n={len(seps_no_pm_prox):,})    "
          f"= {sigma_prx:.6f}\" (≤{PCT_PRX}th %ile = {cut_prx:.4f}″)")
    print()
    print(f"  no-PM sources in bump [0.10, 0.15″]:  raw={in_bump_raw:,}  "
          f"proxy-corrected={in_bump_proxy:,}  (Δ = {in_bump_raw-in_bump_proxy:+,})")

    # ── Plot ──────────────────────────────────────────────────────────────
    x_limit = 0.5
    x_fit   = np.linspace(0, x_limit, 500)
    bin_edges = np.linspace(0, x_limit, 101)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Layer order (bottom → top): PM-available → No-PM proxy → Uncorrected
    ax.hist(seps_pm_avail[seps_pm_avail <= x_limit], bins=bin_edges,
            density=True, color="#A5D6A7", alpha=0.65, edgecolor="none",
            zorder=1,
            label=f"PM available, corrected (n={len(seps_pm_avail):,})")
    ax.plot(x_fit, rayleigh.pdf(x_fit, *p_pm),
            color="#1B5E20", linewidth=2.0, linestyle="--", zorder=4,
            label=f"PM-available fit (σ={sigma_pm:.4f}″)")

    ax.hist(seps_no_pm_prox[seps_no_pm_prox <= x_limit], bins=bin_edges,
            density=True, color="#FFCC80", alpha=0.7, edgecolor="none",
            zorder=2,
            label=f"No PM, proxy-corrected (n={len(seps_no_pm_prox):,})")
    ax.plot(x_fit, rayleigh.pdf(x_fit, *p_prx),
            color="#E65100", linewidth=2.0, linestyle="-.", zorder=5,
            label=f"No-PM proxy fit (σ={sigma_prx:.4f}″)")

    ax.hist(seps_uncorr[seps_uncorr <= x_limit], bins=bin_edges, density=True,
            color="#90CAF9", alpha=0.55, edgecolor="none", zorder=3,
            label=f"Uncorrected, all (n={len(seps_uncorr):,})")
    ax.plot(x_fit, rayleigh.pdf(x_fit, *p_u), color="#0D47A1", linewidth=2.0,
            zorder=6,
            label=f"Uncorrected fit (σ={sigma_u:.4f}″)")

    ax.set_xlim(0, x_limit)
    ax.set_ylim(0, 30)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.set_xlabel("Angular Separation (arcsec)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=14, fontweight="bold",
                  labelpad=15)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=13, loc="upper right", ncol=1, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    # ax.set_title(f"ERO-Gaia sep — neighbour-PM proxy "
    #              f"(R={PROXY_RADIUS_ARCSEC:.0f}″, ≥{PROXY_MIN_NEIGHBOURS} 5/6-par neighbours)",
    #              fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {OUTPUT_FILE}")

    # ── Combined figure (NGC 6397 PM clump + corrected sep PDF) ───────────
    # Diagnostic of the cluster bulk PM that the proxy recovers, paired
    # with the corrected sep PDF on the right.
    bump_mask = ((seps_uncorr >= 0.10) & (seps_uncorr <= 0.15) & has_pm)
    pmra_bump  = pmra[bump_mask]
    pmdec_bump = pmdec[bump_mask]
    cluster_pmra  = float(np.median(pmra_bump))
    cluster_pmdec = float(np.median(pmdec_bump))
    print(f"\n  NGC 6397 empirical PM (from {bump_mask.sum():,} bump sources): "
          f"({cluster_pmra:.2f}, {cluster_pmdec:.2f}) mas/yr")

    fig2, (ax_pm, ax_sep) = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={"width_ratios": [1, 1.3]},
    )
    fig2.subplots_adjust(left=0.07, right=0.97, bottom=0.13, top=0.92, wspace=0.22)

    # Left: 2D PM histogram of bump sources
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
    cb  = plt.colorbar(h[3], cax=cax)
    cb.set_label("Count", fontsize=12, fontweight="bold")
    cb.ax.tick_params(labelsize=11)
    ax_pm.set_xlim(-36, 36); ax_pm.set_ylim(-48, 24); ax_pm.set_aspect("equal")
    ax_pm.xaxis.set_major_locator(plt.MultipleLocator(12))
    ax_pm.yaxis.set_major_locator(plt.MultipleLocator(8))
    ax_pm.tick_params(labelsize=12)
    ax_pm.set_xlabel("PMRA (mas/yr)", fontsize=13, fontweight="bold")
    ax_pm.set_ylabel("PMDEC (mas/yr)", fontsize=13, fontweight="bold")

    # Right: same 3-population sep PDF as the single-panel figure above
    x_limit_c = 0.4
    x_fit_c   = np.linspace(0, x_limit_c, 500)
    bin_edges_c = np.linspace(0, x_limit_c, 101)

    ax_sep.hist(seps_pm_avail[seps_pm_avail <= x_limit_c], bins=bin_edges_c,
                density=True, color="#A5D6A7", alpha=0.65, edgecolor="none",
                zorder=1,
                label=f"PM available, corrected (n={len(seps_pm_avail):,})")
    ax_sep.plot(x_fit_c, rayleigh.pdf(x_fit_c, *p_pm),
                color="#1B5E20", linewidth=2.0, linestyle="--", zorder=4,
                label=f"PM-available fit (σ={sigma_pm:.4f}″)")

    ax_sep.hist(seps_no_pm_prox[seps_no_pm_prox <= x_limit_c], bins=bin_edges_c,
                density=True, color="#FFCC80", alpha=0.7, edgecolor="none",
                zorder=2,
                label=f"No PM, proxy-corrected (n={len(seps_no_pm_prox):,})")
    ax_sep.plot(x_fit_c, rayleigh.pdf(x_fit_c, *p_prx),
                color="#E65100", linewidth=2.0, linestyle="-.", zorder=5,
                label=f"No-PM proxy fit (σ={sigma_prx:.4f}″)")

    ax_sep.hist(seps_uncorr[seps_uncorr <= x_limit_c], bins=bin_edges_c,
                density=True, color="#90CAF9", alpha=0.55, edgecolor="none",
                zorder=3,
                label=f"Uncorrected, all (n={len(seps_uncorr):,})")
    ax_sep.plot(x_fit_c, rayleigh.pdf(x_fit_c, *p_u),
                color="#0D47A1", linewidth=2.0, zorder=6,
                label=f"Uncorrected fit (σ={sigma_u:.4f}″)")
    ax_sep.axvline(0.3, color="dimgray", linewidth=1.2, linestyle="--", alpha=0.85)
    ax_sep.set_xlim(-0.01, x_limit_c)
    ax_sep.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax_sep.set_ylim(0, 28)
    ax_sep.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax_sep.set_xlabel("Angular Separation (arcsec)", fontsize=13, fontweight="bold")
    ax_sep.set_ylabel("Probability Density", fontsize=13, fontweight="bold")
    ax_sep.tick_params(labelsize=12)
    ax_sep.legend(fontsize=13, loc="upper right", framealpha=0.85)
    ax_sep.grid(True, alpha=0.3)

    plt.savefig(OUTPUT_FILE_COMBINED, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_FILE_COMBINED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
