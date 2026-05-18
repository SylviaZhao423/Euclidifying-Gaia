#!/usr/bin/env python3
"""
mag_diff_vs_sep with per-source PM + neighbour-PM proxy correction applied
to BOTH the ERO and Q1 panels.

VARIANT: the Q1 panel here drops the `p_phz_classification IN (1, 2)`
filter, so it includes ALL PHZ classes (stars, galaxies, QSOs, unknown,
unclassified, etc.). The ERO panel is unchanged.

Per-source PM (5/6-par sources) and neighbour-PM proxy (2-par sources)
are read directly from {username}.gaia_pm_proxy via LEFT JOIN — same
pattern as analyze_q1_sep_rayleigh_corrected.py / analyze_ero_sep_rayleigh_corrected.py.

  - Sources WITH Gaia 5/6-par solutions: own pmra/pmdec.
  - Sources WITHOUT PM (Gaia 2-par): proxy_pmra/proxy_pmdec from
    {username}.gaia_pm_proxy (median of ≥5 5/6-par neighbours within 30″).
  - Sources without either: no correction (raw separation).

Reads (WSDB):
  siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia    (Q1, all PHZ)
  {username}.crossmatch_ero_gaia_1_0                (ERO, 1.0″ matching)
  {username}.gaia_pm_proxy                          (LEFT JOIN)

Writes:
  output/mag_diff_vs_sep_corrected_q1allphz.png

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_mag_diff_vs_sep_corrected_q1allphz.py
"""

import os
import sys
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter


def _sky_separation_arcsec(ra1, dec1, ra2, dec2):
    """Exact great-circle separation in arcsec via astropy (haversine)."""
    c1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg)
    c2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg)
    return c1.separation(c2).arcsec


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_DIR / "output" / "mag_diff_vs_sep_corrected_q1allphz.png"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

ZP_AB        = 23.90
GAIA_AB_CORR = 0.11

# PM-correction epochs
GAIA_REF_EPOCH = 2016.0
Q1_OBS_EPOCH   = 2024.0
ERO_OBS_EPOCH  = 2023.5
MAS_TO_DEG     = 1.0 / 3_600_000

RADIUS_MARKS = [0.1, 0.3, 1.0]


# ── Correction helpers ────────────────────────────────────────────────────

def per_source_pm_correction(e_ra, e_dec, g_ra, g_dec, pmra, pmdec, dt_yr):
    """Per-source PM-only correction. NaN PM → no correction."""
    cos_dec = np.cos(np.radians(g_dec))
    pmra_safe  = np.where(np.isfinite(pmra),  pmra,  0.0)
    pmdec_safe = np.where(np.isfinite(pmdec), pmdec, 0.0)
    dra_pm  = pmra_safe  * dt_yr * MAS_TO_DEG
    ddec_pm = pmdec_safe * dt_yr * MAS_TO_DEG
    g_ra_e  = g_ra  + dra_pm / cos_dec
    g_dec_e = g_dec + ddec_pm
    sep = _sky_separation_arcsec(e_ra, e_dec, g_ra_e, g_dec_e)
    return sep


def fold_pm_with_proxy(pmra, pmdec, proxy_pmra, proxy_pmdec, label=""):
    """Combine own PM (5/6-par) with proxy PM (2-par via gaia_pm_proxy LEFT JOIN).

    Returns combined pmra/pmdec arrays — NaN where neither own nor proxy is available.
    """
    has_pm    = np.isfinite(pmra)    & np.isfinite(pmdec)
    has_proxy = np.isfinite(proxy_pmra) & np.isfinite(proxy_pmdec)
    n        = len(pmra)
    n_pm     = int(has_pm.sum())
    n_only_p = int((~has_pm & has_proxy).sum())
    n_none   = int((~has_pm & ~has_proxy).sum())
    print(f"  [{label}] with PM (5/6-par):                       {n_pm:,} "
          f"({100*n_pm/n:.1f}%)")
    print(f"  [{label}] without PM (2-par), with proxy lookup:   {n_only_p:,} "
          f"({100*n_only_p/n:.1f}%)")
    print(f"  [{label}] without PM (2-par), no proxy:            {n_none:,} "
          f"({100*n_none/n:.1f}%)")

    pmra_full  = np.where(has_pm, pmra,  proxy_pmra)
    pmdec_full = np.where(has_pm, pmdec, proxy_pmdec)
    return pmra_full, pmdec_full


# ── Queries ───────────────────────────────────────────────────────────────────

def query_q1(username):
    """Q1-Gaia: Δm + per-source PM + proxy-PM (LEFT JOIN gaia_pm_proxy) corrected sep."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT
            src.m_ie_mag - (src.g_phot_g_mean_mag + {GAIA_AB_CORR}) AS delta_m,
            src.m_right_ascension, src.m_declination,
            src.g_ra, src.g_dec, src.g_pmra, src.g_pmdec,
            proxy.proxy_pmra, proxy.proxy_pmdec
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia AS src
        LEFT JOIN {username}.gaia_pm_proxy AS proxy
            ON proxy.g_source_id = src.g_source_id
        WHERE src.m_ie_mag IS NOT NULL
          AND src.g_phot_g_mean_mag IS NOT NULL
          AND src.g_phot_g_mean_mag BETWEEN 0 AND 30
    """)
    rows = cur.fetchall()
    cur.close(); conn.close()

    dm         = np.array([r[0] for r in rows], dtype=float)
    q1_ra      = np.array([r[1] for r in rows], dtype=float)
    q1_dec     = np.array([r[2] for r in rows], dtype=float)
    g_ra       = np.array([r[3] for r in rows], dtype=float)
    g_dec      = np.array([r[4] for r in rows], dtype=float)
    pmra       = np.array([r[5] for r in rows], dtype=float)
    pmdec      = np.array([r[6] for r in rows], dtype=float)
    proxy_pmra = np.array([r[7] for r in rows], dtype=float)
    proxy_pmdec = np.array([r[8] for r in rows], dtype=float)

    sep_raw = _sky_separation_arcsec(q1_ra, q1_dec, g_ra, g_dec)

    pmra_full, pmdec_full = fold_pm_with_proxy(
        pmra, pmdec, proxy_pmra, proxy_pmdec, label="Q1")
    sep_corr = per_source_pm_correction(
        q1_ra, q1_dec, g_ra, g_dec, pmra_full, pmdec_full,
        dt_yr=Q1_OBS_EPOCH - GAIA_REF_EPOCH)

    print(f"  Q1: {len(dm):,} sources  |  median Δm = {np.nanmedian(dm):.3f} mag")
    print(f"  Q1 sep median: raw={np.nanmedian(sep_raw):.4f}\"  "
          f"corrected={np.nanmedian(sep_corr):.4f}\"")
    return dm, sep_corr, sep_raw


def query_ero(username):
    """ERO-Gaia: Δm + per-source PM + proxy-PM (LEFT JOIN gaia_pm_proxy) corrected sep."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cur = conn.cursor()
    table = f"{username}.crossmatch_ero_gaia_1_0"
    cur.execute(f"""
        SELECT
            src.e_v_mag_auto - (src.g_phot_g_mean_mag + {GAIA_AB_CORR}) AS delta_m,
            src.e_v_alpha_j2000, src.e_v_delta_j2000,
            src.g_ra, src.g_dec, src.g_pmra, src.g_pmdec,
            proxy.proxy_pmra, proxy.proxy_pmdec
        FROM {table} AS src
        LEFT JOIN {username}.gaia_pm_proxy AS proxy
            ON proxy.g_source_id = src.g_source_id
    """)
    rows = cur.fetchall()
    cur.close(); conn.close()

    dm          = np.array([r[0] for r in rows], dtype=float)
    ero_ra      = np.array([r[1] for r in rows], dtype=float)
    ero_dec     = np.array([r[2] for r in rows], dtype=float)
    g_ra        = np.array([r[3] for r in rows], dtype=float)
    g_dec       = np.array([r[4] for r in rows], dtype=float)
    pmra        = np.array([r[5] for r in rows], dtype=float)
    pmdec       = np.array([r[6] for r in rows], dtype=float)
    proxy_pmra  = np.array([r[7] for r in rows], dtype=float)
    proxy_pmdec = np.array([r[8] for r in rows], dtype=float)
    sep_raw = _sky_separation_arcsec(ero_ra, ero_dec, g_ra, g_dec)

    pmra_full, pmdec_full = fold_pm_with_proxy(
        pmra, pmdec, proxy_pmra, proxy_pmdec, label="ERO")
    sep_corr = per_source_pm_correction(
        ero_ra, ero_dec, g_ra, g_dec, pmra_full, pmdec_full,
        dt_yr=ERO_OBS_EPOCH - GAIA_REF_EPOCH)

    print(f"  ERO: {len(dm):,} sources  |  median Δm = {np.nanmedian(dm):.3f} mag")
    print(f"  ERO sep median: raw={np.nanmedian(sep_raw):.4f}\"  "
          f"corrected={np.nanmedian(sep_corr):.4f}\"")
    return dm, sep_corr, sep_raw


# ── Panel ─────────────────────────────────────────────────────────────────────

def _panel_density(ax, sep, dm, title, ylabel=True, sep_raw=None, xlabel=True,
                   panel_label=""):
    x_max = max(RADIUS_MARKS)
    mask = np.isfinite(dm) & np.isfinite(sep) & (sep <= x_max)
    _, _, _, img = ax.hist2d(sep[mask], dm[mask],
                             bins=[200, 80], range=[[0, x_max], [-6, 6]],
                             cmap="Blues", norm=LogNorm(), cmin=1, alpha=1.0)
    ax.axhline(0, color="white", lw=0.8, alpha=0.6, ls="--")

    # Three counts at the 0.3″ marker:
    #   raw          — sep_raw    ≤ 0.3 (matches _uncor canonical sample)
    #   corrected    — sep_corr   ≤ 0.3 (matches _pmcor canonical sample)
    #   intersection — BOTH       ≤ 0.3 (matches the canonical _sep INTERSECTION table)
    # The legend label reports the intersection count, which is the canonical
    # sample used by every downstream model/analysis.
    n_finite = np.isfinite(sep).sum()
    r = 0.3
    n_r_corr = int((np.isfinite(sep) & (sep <= r)).sum())
    pct_corr = 100 * n_r_corr / n_finite if n_finite else 0
    if sep_raw is not None:
        n_r_raw = int((np.isfinite(sep_raw) & (sep_raw <= r)).sum())
        n_r_int = int((np.isfinite(sep)     & (sep     <= r) &
                       np.isfinite(sep_raw) & (sep_raw <= r)).sum())
        pct_raw = 100 * n_r_raw / n_finite if n_finite else 0
        pct_int = 100 * n_r_int / n_finite if n_finite else 0
        print(f"  [{panel_label}] sources ≤ 0.3″ — "
              f"raw: {n_r_raw:,} ({pct_raw:.1f}%)  "
              f"corrected: {n_r_corr:,} ({pct_corr:.1f}%)  "
              f"intersection: {n_r_int:,} ({pct_int:.1f}%)")
    else:
        n_r_int = n_r_corr
        pct_int = pct_corr
    ax.axvline(r, color="black", linewidth=1.2, linestyle="--", alpha=0.85,
               label=f"{r:.1f}″  n={n_r_int:,} / {n_finite:,} ({pct_int:.2f}%)")

    ax.set_xlim(0, x_max); ax.set_ylim(-6, 6)
    ax.set_title(title, fontsize=13, fontweight="bold")
    if xlabel:
        ax.set_xlabel("Angular Separation (arcsec)", fontsize=12, fontweight="bold")
    else:
        plt.setp(ax.get_xticklabels(), visible=False)
    if ylabel:
        ax.set_ylabel(r"$\mathbf{\Delta m}$ (mag)", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.7)
    ax.grid(True, alpha=0.20)
    return img


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set"); return 1

    print("Querying Q1-Gaia (all PHZ classes)...")
    q1_dm, q1_sep, q1_sep_raw = query_q1(username)

    print("\nQuerying ERO-Gaia...")
    ero_dm, ero_sep, ero_sep_raw = query_ero(username)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6),
                              gridspec_kw={"hspace": 0.26})
    fig.subplots_adjust(left=0.10, right=0.90, top=0.91, bottom=0.10)

    img = _panel_density(
        axes[0], ero_sep, ero_dm,
        title=(f"ERO – Gaia  (n={len(ero_dm):,}),  "
               r"$\mathbf{\Delta m = v\_mag\_auto - G_{AB}}$  "),
        sep_raw=ero_sep_raw, xlabel=False, panel_label="ERO")

    img = _panel_density(
        axes[1], q1_sep, q1_dm,
        title=(f"Q1 – Gaia  (n={len(q1_dm):,}, all PHZ),  "
               r"$\mathbf{\Delta m = ie\_mag - G_{AB}}$  "),
        sep_raw=q1_sep_raw, panel_label="Q1")

    cbar_ax = fig.add_axes([0.92, 0.10, 0.018, 0.81])
    cb = fig.colorbar(img, cax=cbar_ax)
    cb.set_label("Count (log scale)", fontsize=11)
    cb.ax.tick_params(labelsize=10)

    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
