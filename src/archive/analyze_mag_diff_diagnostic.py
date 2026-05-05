#!/usr/bin/env python3
"""
Diagnostic: identify what's special about the second-track sources in
the Q1-Gaia magnitude difference plot (Δm = ie_mag − G_AB).

Uses crossmatch_q1_mer_phz_gaia (3-way Q1 × PHZ × Gaia join) as the
final sample, with m_ie_mag taken directly from the stored column.

Pulls RA, Dec, ie_mag, G_AB, Δm, object_id for all matched sources,
then splits into "main track" (|Δm| < 1) vs "outlier track"
(Δm > 1 or Δm < -2) and plots:
  (a) Sky positions coloured by track membership
  (b) Δm vs ie_mag (magnitude-dependent systematics?)
  (c) Δm vs G_AB
  (d) Δm histogram comparing both tracks
  (e) Tile ID distribution (object_id // 1e9) by track

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_mag_diff_diagnostic.py

Output:
    output/mag_diff_diagnostic.png
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_DIR / "output" / "mag_diff_diagnostic.png"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

GAIA_AB_CORR = 0.11


def query(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()
    print("Querying diagnostic data (crossmatch_q1_mer_phz_gaia)...")
    cursor.execute(f"""
        SELECT
            m_right_ascension                                AS ra,
            m_declination                                    AS dec,
            m_ie_mag - (g_phot_g_mean_mag + {GAIA_AB_CORR}) AS delta_m,
            m_ie_mag                                         AS ie_mag,
            g_phot_g_mean_mag + {GAIA_AB_CORR}              AS g_ab
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE m_ie_mag IS NOT NULL
          AND g_phot_g_mean_mag IS NOT NULL
          AND g_phot_g_mean_mag BETWEEN 0 AND 30
          AND m_vis_det = 1
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    ra   = np.array([r[0] for r in rows], dtype=float)
    dec  = np.array([r[1] for r in rows], dtype=float)
    dm   = np.array([r[2] for r in rows], dtype=float)
    ie   = np.array([r[3] for r in rows], dtype=float)
    g_ab = np.array([r[4] for r in rows], dtype=float)
    print(f"  Retrieved {len(dm):,} sources")
    return ra, dec, dm, ie, g_ab


def main():
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set"); return 1

    ra, dec, dm, ie, g_ab = query(username)

    finite = np.isfinite(dm)
    ra, dec, dm, ie, g_ab = [a[finite] for a in (ra, dec, dm, ie, g_ab)]

    # Define tracks
    main_mask    = (dm >= -2) & (dm <= 1)
    outlier_mask = ~main_mask

    print(f"  Main track    (−2 ≤ Δm ≤ 1): {main_mask.sum():,} ({100*main_mask.mean():.1f}%)")
    print(f"  Outlier track (Δm < −2 or > 1): {outlier_mask.sum():,} ({100*outlier_mask.mean():.1f}%)")

    # Sky binning: 2-degree bins in RA/Dec to find regions with high outlier fraction
    ra_bins  = np.arange(np.floor(ra.min()),  np.ceil(ra.max())  + 2, 2)
    dec_bins = np.arange(np.floor(dec.min()), np.ceil(dec.max()) + 2, 2)

    h_all, _, _     = np.histogram2d(ra, dec, bins=[ra_bins, dec_bins])
    h_out, _, _     = np.histogram2d(ra[outlier_mask], dec[outlier_mask],
                                     bins=[ra_bins, dec_bins])
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_grid = np.where(h_all > 5, h_out / h_all, np.nan)

    # Report bins with >30% outlier fraction
    ra_centres  = (ra_bins[:-1]  + ra_bins[1:])  / 2
    dec_centres = (dec_bins[:-1] + dec_bins[1:]) / 2
    bad_mask    = frac_grid > 0.3
    n_bad       = int(bad_mask.sum())
    print(f"  2-deg sky bins with >30% outliers: {n_bad}")
    for i, j in zip(*np.where(bad_mask)):
        cnt = int(h_all[i, j])
        frac = frac_grid[i, j]
        print(f"    RA≈{ra_centres[i]:6.1f}  Dec≈{dec_centres[j]:6.1f}"
              f"  n={cnt:5,}  outlier_frac={frac:.2f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.08,
                        hspace=0.38, wspace=0.32)

    MAIN_C    = "steelblue"
    OUTLIER_C = "tomato"

    # (a) Sky positions
    ax = axes[0, 0]
    ax.scatter(ra[main_mask],    dec[main_mask],    s=0.5, c=MAIN_C,
               alpha=0.3, rasterized=True, label=f"Main (n={main_mask.sum():,})")
    ax.scatter(ra[outlier_mask], dec[outlier_mask], s=0.5, c=OUTLIER_C,
               alpha=0.5, rasterized=True, label=f"Outlier (n={outlier_mask.sum():,})")
    ax.set_xlabel("RA (deg)", fontsize=11)
    ax.set_ylabel("Dec (deg)", fontsize=11)
    ax.set_title("(a) Sky Positions by Track", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, markerscale=8, loc="upper right")
    ax.grid(alpha=0.2)

    # (b) Δm vs ie_mag
    ax = axes[0, 1]
    ax.scatter(ie[main_mask],    dm[main_mask],    s=0.5, c=MAIN_C,
               alpha=0.2, rasterized=True)
    ax.scatter(ie[outlier_mask], dm[outlier_mask], s=0.5, c=OUTLIER_C,
               alpha=0.5, rasterized=True)
    ax.axhline(0,  color="gray",   lw=0.8, ls="--", alpha=0.5)
    ax.axhline(1,  color="tomato", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(-2, color="tomato", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("ie_mag (AB)", fontsize=11)
    ax.set_ylabel("Δm = ie_mag − G_AB", fontsize=11)
    ax.set_title("(b) Δm vs ie_mag", fontsize=11, fontweight="bold")
    ax.set_ylim(-6, 6)
    ax.grid(alpha=0.2)

    # (c) Δm vs G_AB
    ax = axes[0, 2]
    ax.scatter(g_ab[main_mask],    dm[main_mask],    s=0.5, c=MAIN_C,
               alpha=0.2, rasterized=True)
    ax.scatter(g_ab[outlier_mask], dm[outlier_mask], s=0.5, c=OUTLIER_C,
               alpha=0.5, rasterized=True)
    ax.axhline(0,  color="gray",   lw=0.8, ls="--", alpha=0.5)
    ax.axhline(1,  color="tomato", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(-2, color="tomato", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("G_AB (mag)", fontsize=11)
    ax.set_ylabel("Δm = ie_mag − G_AB", fontsize=11)
    ax.set_title("(c) Δm vs G_AB", fontsize=11, fontweight="bold")
    ax.set_ylim(-6, 6)
    ax.grid(alpha=0.2)

    # (d) Δm histogram
    ax = axes[1, 0]
    bins = np.linspace(-6, 6, 120)
    ax.hist(dm[main_mask],    bins=bins, color=MAIN_C,    alpha=0.6,
            label=f"Main (n={main_mask.sum():,})", density=True)
    ax.hist(dm[outlier_mask], bins=bins, color=OUTLIER_C, alpha=0.6,
            label=f"Outlier (n={outlier_mask.sum():,})", density=True)
    ax.axvline(0,  color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Δm = ie_mag − G_AB (mag)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("(d) Δm Distribution", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # (e) RA distribution by track
    ax = axes[1, 1]
    ra_bins_hist = np.arange(0, 365, 5)
    ax.hist(ra[main_mask],    bins=ra_bins_hist, color=MAIN_C,    alpha=0.6,
            density=True, label="Main")
    ax.hist(ra[outlier_mask], bins=ra_bins_hist, color=OUTLIER_C, alpha=0.6,
            density=True, label="Outlier")
    ax.set_xlabel("RA (deg)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("(e) RA Distribution by Track", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # (f) Per-2deg-bin outlier fraction on sky
    ax = axes[1, 2]
    RA_c, DEC_c = np.meshgrid(ra_centres, dec_centres, indexing="ij")
    sc = ax.scatter(RA_c.ravel(), DEC_c.ravel(),
                    c=frac_grid.ravel(), cmap="RdYlGn_r",
                    vmin=0, vmax=1, s=8, alpha=0.8, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Outlier fraction (2° bins)")
    ax.set_xlabel("RA (deg)", fontsize=11)
    ax.set_ylabel("Dec (deg)", fontsize=11)
    ax.set_title("(f) Outlier Fraction per Sky Region (2° bins)",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.2)

    fig.suptitle(
        "Q1-Gaia Magnitude Difference Diagnostic  (crossmatch_q1_mer_phz_gaia)\n"
        "Main track: −2 ≤ Δm ≤ 1   |   Outlier track: Δm < −2 or > 1",
        fontsize=12, fontweight="bold",
    )

    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
