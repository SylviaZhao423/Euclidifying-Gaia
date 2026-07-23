#!/usr/bin/env python3
"""
QSO–Gaia angular-separation quality check (Rayleigh + chance-alignment).

Analogue of analyze_q1_sep_rayleigh_corrected.py, but for the QSO cross-match —
with two important differences that follow from QSOs being extragalactic:

  1. EPOCH.  The Q1 version propagates Gaia (2016.0) → Euclid Q1 (2024.0). SDSS
     positions are NOT at the Euclid epoch — SDSS imaging astrometry is ~2005.
     So if we PM-propagate at all, it is Gaia 2016 → SDSS_REF_EPOCH (~2005.5),
     never 2024.

  2. PROPER MOTION.  A genuine QSO counterpart has ~zero proper motion, so the
     PM-corrected and uncorrected separations should be essentially IDENTICAL.
     That is the diagnostic: unlike the Q1 STAR sample (where PM correction
     visibly tightens the tail of high-PM stars), for QSOs the two curves lying
     on top of each other confirms the matches are genuine zero-PM QSO
     identifications, not chance alignments with moving foreground stars.

What "good cross-matching" looks like here:
  - a tight Rayleigh core (σ ≈ SDSS⊕Gaia astrometric precision, ~0.1–0.2"),
  - corrected ≈ uncorrected (zero-PM confirmation),
  - a small chance-alignment tail (estimated from the flat r-density at large
    separation and reported as a contamination fraction within the 1" radius).

Requires WSDB (VPN up). Reads the crossmatch tables built by
crossmatch_sdss_dr16_qso_gaia.py / crossmatch_desi_dr1_qso_gaia.py, joined to
gaia_source for pmra/pmdec.

Writes:
  output/umap/qso/qso_sep_rayleigh.png

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_qso_sep_rayleigh.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rayleigh

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "qso"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
WSDB_USER = "siwei_zhao_2025"

GAIA_REF_EPOCH  = 2016.0    # Gaia DR3 catalogue reference epoch
SDSS_REF_EPOCH  = 2005.5    # representative SDSS imaging epoch (per-object not modelled)
MATCH_RADIUS    = 1.0       # arcsec (the cross-match radius)
MAS_TO_DEG      = 1.0 / 3_600_000

CATALOG_TABLES = {
    "SDSS_DR16Q": "crossmatch_sdss_dr16_qso_gaia",
    "DESI_DR1Q":  "crossmatch_desi_dr1_qso_gaia",
}


def query(user: str, table: str):
    """Return (qso_ra, qso_dec, g_ra, g_dec, pmra, pmdec) for one catalog, or None."""
    import psycopg2
    sql = f"""
        SELECT t.qso_ra, t.qso_dec, t.g_ra, t.g_dec, gg.pmra, gg.pmdec
        FROM {user}.{table} t
        JOIN gaia_dr3.gaia_source gg ON gg.source_id = t.g_source_id
    """
    try:
        conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                                connect_timeout=30, keepalives=1)
        cur = conn.cursor(); cur.execute(sql); rows = cur.fetchall()
        cur.close(); conn.close()
    except Exception as exc:
        print(f"  skipped {table} ({exc}).")
        return None
    if not rows:
        return None
    a = np.array(rows, dtype=float)  # None → nan
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]


def separations(qra, qdec, gra, gdec, pmra, pmdec, dt_years):
    """Small-angle separation (arcsec) between QSO position and Gaia position
    propagated by dt_years using pmra/pmdec (0 where PM missing)."""
    cos_dec = np.cos(np.radians(gdec))
    has_pm = np.isfinite(pmra) & np.isfinite(pmdec)
    dra  = np.where(has_pm, pmra  * dt_years * MAS_TO_DEG, 0.0)
    ddec = np.where(has_pm, pmdec * dt_years * MAS_TO_DEG, 0.0)
    g_ra_ep  = gra  + dra / cos_dec
    g_dec_ep = gdec + ddec
    return np.sqrt(((qra - g_ra_ep) * cos_dec) ** 2 + (qdec - g_dec_ep) ** 2) * 3600.0


def chance_fraction(sep, sigma, R=MATCH_RADIUS):
    """Estimate the chance-alignment contamination within R using the flat
    r-surface-density at large separation (where the Rayleigh core is gone).
    Chance matches have uniform sky density → dN/dr ∝ r; measure that density in
    the outer annulus [r1, R] and extrapolate a πr² count over [0, R]."""
    r1 = min(max(0.6, 4.0 * sigma), 0.9 * R)
    n_tail = int(np.sum((sep >= r1) & (sep <= R)))
    area_annulus = np.pi * (R ** 2 - r1 ** 2)
    lam = n_tail / area_annulus if area_annulus > 0 else 0.0   # per arcsec^2
    chance_total = lam * np.pi * R ** 2
    n_within = int(np.sum(sep <= R))
    return (chance_total / n_within if n_within else 0.0), r1, chance_total


def analyze_panel(ax, name, data):
    qra, qdec, gra, gdec, pmra, pmdec = data
    dt = SDSS_REF_EPOCH - GAIA_REF_EPOCH   # -10.5 (Gaia 2016 → SDSS ~2005.5)
    sep_uncorr = separations(qra, qdec, gra, gdec, pmra, pmdec, 0.0)
    sep_corr   = separations(qra, qdec, gra, gdec, pmra, pmdec, dt)

    core_u = sep_uncorr[sep_uncorr <= np.percentile(sep_uncorr, 90)]
    core_c = sep_corr[sep_corr <= np.percentile(sep_corr, 90)]
    sig_u = rayleigh.fit(core_u)[1]
    sig_c = rayleigh.fit(core_c)[1]
    n_pm = int((np.isfinite(pmra) & np.isfinite(pmdec)).sum())
    cfrac, r1, _ = chance_fraction(sep_uncorr, sig_u)

    print(f"  [{name}] N={len(qra):,}  σ_uncorr={sig_u:.4f}\"  σ_corr={sig_c:.4f}\"  "
          f"(Δ={100*(sig_c-sig_u)/sig_u:+.2f}%)  PM-avail={100*n_pm/len(qra):.0f}%  "
          f"chance≈{100*cfrac:.1f}% within {MATCH_RADIUS}\"")

    xmax = min(MATCH_RADIUS, np.percentile(sep_uncorr, 99.5))
    edges = np.linspace(0, xmax, 81)
    xf = np.linspace(1e-4, xmax, 400)
    ax.hist(sep_uncorr[sep_uncorr <= xmax], bins=edges, density=True, color="#FFCDD2",
            alpha=0.55, edgecolor="none", label=f"uncorrected (n={len(qra):,})")
    ax.plot(xf, rayleigh.pdf(xf, 0, sig_u), color="#e53935", lw=2,
            label=f"uncorr Rayleigh σ={sig_u:.4f}\"")
    ax.hist(sep_corr[sep_corr <= xmax], bins=edges, density=True, color="steelblue",
            alpha=0.45, edgecolor="none", label=f"PM-corr → {SDSS_REF_EPOCH:.0f} (n={len(qra):,})")
    ax.plot(xf, rayleigh.pdf(xf, 0, sig_c), color="navy", lw=2, ls="--",
            label=f"corr Rayleigh σ={sig_c:.4f}\"")
    # mark where the chance-alignment (flat r-density) tail is estimated from
    ax.axvline(r1, color="0.5", ls=":", lw=1, label=f"chance-tail start ({r1:.2f}\")")
    ax.set_title(f"{name}   chance≈{100*cfrac:.1f}% within {MATCH_RADIUS}\"",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("QSO–Gaia separation (arcsec)", fontsize=11)
    ax.set_ylabel("probability density", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)


def main() -> int:
    user = os.getenv("PGUSER", WSDB_USER)
    panels = []
    for name, table in CATALOG_TABLES.items():
        print(f"Reading {user}.{table} ...")
        d = query(user, table)
        if d is not None:
            panels.append((name, d))
    if not panels:
        print("No crossmatch tables readable (VPN? tables built?). Aborting.")
        return 1

    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 5), squeeze=False)
    for ax, (name, d) in zip(axes[0], panels):
        analyze_panel(ax, name, d)
    fig.suptitle("QSO–Gaia cross-match quality:  Rayleigh core + zero-PM check\n"
                 "corrected≈uncorrected ⇒ genuine zero-PM QSO counterparts "
                 "(epoch Gaia 2016 → SDSS ~2005.5, NOT Euclid 2024)",
                 fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.84)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "qso_sep_rayleigh.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
