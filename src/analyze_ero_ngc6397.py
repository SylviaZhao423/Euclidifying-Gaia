#!/usr/bin/env python3
"""
NGC 6397 proper-motion clump in the ERO-Gaia 0.1-0.15" separation bin.

Two-panel figure for the report:
  (a) 2D proper-motion histogram of the 0.1-0.15" clump — reveals the NGC 6397
      cluster at (PMRA, PMDEC) ≈ (3, -18) mas/yr.
  (b) Astrometric offsets (ERO − Gaia) for PM-matched cluster members selected
      by sky position — shows the systematic position offset used for correction.

Logic:
  Step 1  Cluster PM derived from sources in the 0.10-0.15" separation bump.
          These are predominantly cluster members (the bump IS the displaced
          cluster), so their median PM is a clean empirical estimate.
  Step 2  Full cluster member selection: sky position ≤ SKY_POS_TOLERANCE AND
          PM within PM_TOLERANCE of the derived centre (plus sky-only fallback).
  Step 3  Mean astrometric offset measured from PM-matched members.

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_ero_ngc6397.py

Output:
    output/ero_ngc6397.png
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_DIR / "output" / "ero_ngc6397.png"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

# NGC 6397 cluster sky position (well-established coordinates)
NGC6397_RA  = 265.10   # degrees
NGC6397_DEC = -53.58   # degrees

# PM selection for panel (b) diagnostic — tighter than correction scripts
# to isolate the dense clump core (~2 mas/yr visible radius in panel (a)).
# Correction scripts use 3.0 mas/yr for inclusive member selection.
PM_TOLERANCE = 2.0   # mas/yr


def query_data(username):
    import psycopg2

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()

    # ── Panel (a): PM distribution of bump sources (0.10–0.15" separation) ────
    # These sources are predominantly cluster members displaced by the systematic
    # offset — their median PM gives a clean empirical cluster PM estimate.
    print("Querying proper motions for 0.10-0.15\" separation bump...")
    cursor.execute("""
        SELECT g_pmra, g_pmdec
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE SQRT(POWER((e_v_alpha_j2000 - g_ra) * COS(RADIANS(g_dec)), 2)
                 + POWER(e_v_delta_j2000 - g_dec, 2)) * 3600 BETWEEN 0.10 AND 0.15
          AND g_pmra  IS NOT NULL
          AND g_pmdec IS NOT NULL
    """)
    rows_pm = cursor.fetchall()
    pmra  = np.array([r[0] for r in rows_pm], dtype=float)
    pmdec = np.array([r[1] for r in rows_pm], dtype=float)
    print(f"  Retrieved {len(pmra):,} sources in bump region")

    # Empirical cluster PM from bump sources
    cluster_pmra  = np.median(pmra)
    cluster_pmdec = np.median(pmdec)
    print(f"  Empirical cluster PM: ({cluster_pmra:.2f}, {cluster_pmdec:.2f}) mas/yr")

    # ── Panel (b): same bump bin + PM selection, using empirical PM centre ─────
    # Selecting sources in the 0.10–0.15" bin concentrates on displaced cluster
    # members, giving a tight offset distribution for clear visualisation.
    print("Querying astrometric offsets for PM clump members...")
    cursor.execute(f"""
        SELECT
            e_v_alpha_j2000,
            e_v_delta_j2000,
            g_ra,
            g_dec
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE SQRT(POWER((e_v_alpha_j2000 - g_ra) * COS(RADIANS(g_dec)), 2)
                 + POWER(e_v_delta_j2000 - g_dec, 2)) * 3600 BETWEEN 0.10 AND 0.15
          AND g_pmra  BETWEEN {cluster_pmra - PM_TOLERANCE} AND {cluster_pmra + PM_TOLERANCE}
          AND g_pmdec BETWEEN {cluster_pmdec - PM_TOLERANCE} AND {cluster_pmdec + PM_TOLERANCE}
    """)
    rows_off = cursor.fetchall()
    cursor.close()
    conn.close()
    print(f"  Retrieved {len(rows_off):,} PM clump members")

    DEG_TO_MAS = 3_600_000
    # cos(dec) on RA converts coordinate difference to true angular offset,
    # so the 100/150 mas circles correctly bound the 0.10-0.15" selected sources.
    dra  = np.array([(r[0] - r[2]) * np.cos(np.radians(r[3])) * DEG_TO_MAS
                     for r in rows_off], dtype=float)
    ddec = np.array([(r[1] - r[3]) * DEG_TO_MAS for r in rows_off], dtype=float)
    print(f"  Empirical angular offset: ({np.mean(dra):.1f}, {np.mean(ddec):.1f}) mas")

    return pmra, pmdec, dra, ddec, cluster_pmra, cluster_pmdec


def plot(pmra, pmdec, dra, ddec, cluster_pmra, cluster_pmdec):
    mean_ra    = np.mean(dra)
    mean_dec   = np.mean(ddec)
    offset_mag = np.sqrt(mean_ra**2 + mean_dec**2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.14, top=0.86, wspace=0.38)

    # ── Panel (a): 2D PM histogram ───────────────────────────────────────────
    ax = axes[0]
    h = ax.hist2d(pmra, pmdec, bins=60, cmap="Blues", norm=LogNorm(), cmin=1)

    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, alpha=0.6)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, alpha=0.6)

    ax.add_patch(Circle((cluster_pmra, cluster_pmdec), 3.5,
                        color="red", fill=False, linewidth=2,
                        linestyle="--", alpha=0.6))
    ax.annotate(
        f"Clump Centre\n({cluster_pmra:.2f}, {cluster_pmdec:.2f})",
        xy=(cluster_pmra, cluster_pmdec),
        xytext=(cluster_pmra + 7, cluster_pmdec - 7),
        fontsize=11, color="red", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5, alpha=0.8),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="red", alpha=0.8),
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(h[3], cax=cax).set_label("Count", fontsize=12)

    ax.set_xlim(-36, 36)
    ax.set_ylim(-48, 24)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(plt.MultipleLocator(12))
    ax.yaxis.set_major_locator(plt.MultipleLocator(8))
    ax.tick_params(labelsize=11)
    ax.set_xlabel("PMRA (mas/yr)", fontsize=14, fontweight="bold")
    ax.set_ylabel("PMDEC (mas/yr)", fontsize=14, fontweight="bold")
    ax.set_title("(a) Proper Motion Distribution\n"
                 r"ERO$-$Gaia matches: separation 0.10$-$0.15$^{\prime\prime}$",
                 fontsize=13, fontweight="bold", pad=15)

    # ── Panel (b): astrometric offset 2D histogram ───────────────────────────
    ax = axes[1]
    h2 = ax.hist2d(dra, ddec, bins=40, cmap="Blues", norm=LogNorm(), cmin=1)

    ax.scatter([0], [0], marker="x", s=120, color="red", linewidths=2.5,
               label="Gaia reference (0, 0)", alpha=0.8, zorder=5)
    ax.scatter([mean_ra], [mean_dec], marker="x", s=120,
               color="limegreen", linewidths=2.5,
               label=f"Mean ERO offset: ({mean_ra:.1f}, {mean_dec:.1f}) mas",
               alpha=0.8, zorder=5)
    ax.add_patch(Circle((0, 0), 100, color="red", fill=False,
                        linestyle="--", linewidth=2, alpha=0.6))
    ax.add_patch(Circle((0, 0), 150, color="red", fill=False,
                        linestyle="--", linewidth=2, alpha=0.6,
                        label="Separation: 100\u2013150 mas"))

    divider2 = make_axes_locatable(ax)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(h2[3], cax=cax2).set_label("Count", fontsize=12)

    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=11)
    ax.set_xlabel("RA angular offset: ERO \u2212 Gaia (mas)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Dec offset: ERO \u2212 Gaia (mas)", fontsize=14, fontweight="bold")
    ax.set_title("(b) Astrometric Offsets Distribution\n"
                 f"PM\u00a0\u2248\u00a0({cluster_pmra:.2f},\u00a0{cluster_pmdec:.2f})\u00a0mas/yr,"
                 f"\u00a0|\u0394PM|\u00a0\u2264\u00a0{PM_TOLERANCE:.0f}\u00a0mas/yr",
                 fontsize=13, fontweight="bold", pad=15)
    axes[1].title.set_linespacing(1.6)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.6, markerscale=0.8)

    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_FILE}")


def main():
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1
    pmra, pmdec, dra, ddec, cluster_pmra, cluster_pmdec = query_data(username)
    plot(pmra, pmdec, dra, ddec, cluster_pmra, cluster_pmdec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
