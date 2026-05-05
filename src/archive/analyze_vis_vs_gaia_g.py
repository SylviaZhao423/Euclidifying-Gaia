#!/usr/bin/env python3
"""
2D histogram of Euclid VIS vs Gaia G magnitude for ERO and Q1 Gaia-matched sources.

Row 1 — source count (log scale):
  Left  — ERO × Gaia (crossmatch_ero_gaia_0_3), VIS = e_v_mag_auto
  Right — Q1  × Gaia (crossmatch_q1_mer_phz_gaia), VIS = m_ie_mag

Row 2 — mean classifier per bin:
  Left  — mean e_v_class_star  (0=extended → 1=star)
  Right — mean p_phz_gal_prob  (0=star     → 1=galaxy)

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_vis_vs_gaia_g.py

Output:
    output/vis_vs_gaia_g.png
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_DIR / "output" / "vis_vs_gaia_g.png"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

# Axis limits
G_LIM   = (10, 23)
VIS_LIM = (15, 27)

# Bin resolution
G_BINS   = np.arange(G_LIM[0],   G_LIM[1]   + 0.1, 0.1)
VIS_BINS = np.arange(VIS_LIM[0], VIS_LIM[1] + 0.1, 0.1)

MIN_COUNT = 5   # minimum sources per bin to show classifier value


def query_ero(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = """
        SELECT
            g_phot_g_mean_mag  AS g_mag,
            e_v_mag_auto       AS vis_mag,
            e_v_class_star     AS class_star
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE g_phot_g_mean_mag IS NOT NULL
          AND e_v_mag_auto      IS NOT NULL
          AND e_v_mag_auto > 0
          AND e_v_mag_auto < 50
    """
    print("Querying ERO-Gaia data...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} ERO sources")
    return df


def query_q1(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = """
        SELECT
            g_phot_g_mean_mag  AS g_mag,
            m_ie_mag           AS vis_mag,
            p_phz_gal_prob     AS gal_prob
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE m_vis_det = 1
          AND g_phot_g_mean_mag IS NOT NULL
          AND m_ie_mag          IS NOT NULL
          AND SQRT(POWER((m_right_ascension - g_ra) * COS(RADIANS(g_dec)), 2)
                + POWER(m_declination    - g_dec, 2)) * 3600 <= 0.3
    """
    print("Querying Q1-Gaia data...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} Q1 sources")
    return df


def _add_ref_lines(ax):
    lim = np.array([max(G_LIM[0], VIS_LIM[0]), min(G_LIM[1], VIS_LIM[1])])
    ax.plot(lim, lim, "w--", linewidth=1.2, alpha=0.7, label="VIS = $G$")
    ax.axvline(x=17, color="tomato", linewidth=1.5, linestyle="--", alpha=0.85, label="$G = 17$")


def _fmt_ax(ax, title, ylabel, xlabel=True):
    ax.set_xlim(G_LIM)
    ax.set_ylim(VIS_LIM)
    if xlabel:
        ax.set_xlabel("Gaia $G$ (mag, Vega)", fontsize=12, fontweight="bold")
    if ylabel:
        ax.set_ylabel("Euclid VIS (mag)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.6)
    ax.grid(True, alpha=0.15, color="white")


def _plot_count(ax, g_mag, vis_mag, title, ylabel=True, xlabel=True):
    h, xedges, yedges = np.histogram2d(g_mag, vis_mag, bins=[G_BINS, VIS_BINS])
    h = h.T
    h_masked = np.ma.masked_where(h == 0, h)
    norm = mcolors.LogNorm(vmin=1, vmax=h_masked.max())
    im = ax.pcolormesh(xedges, yedges, h_masked, norm=norm, cmap="viridis", rasterized=True)
    _add_ref_lines(ax)
    _fmt_ax(ax, title, ylabel, xlabel)
    return im


def _plot_mean(ax, g_mag, vis_mag, values, title, cmap, vmin, vmax, ylabel=True, xlabel=True):
    """2D bin mean of `values`, only shown where count ≥ MIN_COUNT."""
    # Count and weighted sum
    mask = np.isfinite(values)
    h_count, _, _ = np.histogram2d(g_mag[mask], vis_mag[mask], bins=[G_BINS, VIS_BINS])
    h_sum,   _, _ = np.histogram2d(g_mag[mask], vis_mag[mask], bins=[G_BINS, VIS_BINS],
                                   weights=values[mask])
    h_count = h_count.T
    h_sum   = h_sum.T

    with np.errstate(invalid="ignore", divide="ignore"):
        h_mean = np.where(h_count >= MIN_COUNT, h_sum / h_count, np.nan)

    h_masked = np.ma.masked_invalid(h_mean)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(xedges := G_BINS, yedges := VIS_BINS,
                       h_masked, norm=norm, cmap=cmap, rasterized=True)
    _add_ref_lines(ax)
    _fmt_ax(ax, title, ylabel, xlabel)
    return im


def _plot_mean2(ax, g_mag, vis_mag, values, title, cmap, vmin, vmax, ylabel=True, xlabel=True):
    """Walrus-free version."""
    mask = np.isfinite(values)
    h_count, xedges, yedges = np.histogram2d(g_mag[mask], vis_mag[mask], bins=[G_BINS, VIS_BINS])
    h_sum,   _,      _      = np.histogram2d(g_mag[mask], vis_mag[mask], bins=[G_BINS, VIS_BINS],
                                             weights=values[mask])
    h_count = h_count.T
    h_sum   = h_sum.T
    with np.errstate(invalid="ignore", divide="ignore"):
        h_mean = np.where(h_count >= MIN_COUNT, h_sum / h_count, np.nan)
    h_masked = np.ma.masked_invalid(h_mean)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(xedges, yedges, h_masked, norm=norm, cmap=cmap, rasterized=True)
    _add_ref_lines(ax)
    _fmt_ax(ax, title, ylabel, xlabel)
    return im


def main():
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set"); return 1

    df_ero = query_ero(username)
    df_q1  = query_q1(username)

    ero_g   = df_ero["g_mag"].values.astype(float)
    ero_vis = df_ero["vis_mag"].values.astype(float)
    ero_cs  = df_ero["class_star"].values.astype(float)

    q1_g    = df_q1["g_mag"].values.astype(float)
    q1_vis  = df_q1["vis_mag"].values.astype(float)
    q1_gp   = df_q1["gal_prob"].values.astype(float)

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 11),
        gridspec_kw={"hspace": 0.32, "wspace": 0.22},
    )
    fig.subplots_adjust(left=0.07, right=0.93, top=0.93, bottom=0.07)

    # ── Row 1: source counts ──────────────────────────────────────────────────
    im_ero = _plot_count(
        axes[0, 0], ero_g, ero_vis,
        title=f"ERO × Gaia  (n={len(df_ero):,})  |  count",
        ylabel=True, xlabel=False,
    )
    im_q1 = _plot_count(
        axes[0, 1], q1_g, q1_vis,
        title=f"Q1 × Gaia  (n={len(df_q1):,},  sep ≤ 0.3\")  |  count",
        ylabel=False, xlabel=False,
    )
    fig.colorbar(im_ero, ax=axes[0, 0], fraction=0.046, pad=0.03).set_label("Count (log)", fontsize=10)
    fig.colorbar(im_q1,  ax=axes[0, 1], fraction=0.046, pad=0.03).set_label("Count (log)", fontsize=10)

    # ── Row 2: mean classifier ────────────────────────────────────────────────
    im_cs = _plot_mean2(
        axes[1, 0], ero_g, ero_vis, ero_cs,
        title="ERO × Gaia  |  mean class_star per bin\n(0 = extended,  1 = star-like)",
        cmap="RdYlBu_r", vmin=0, vmax=1,
        ylabel=True, xlabel=True,
    )
    im_gp = _plot_mean2(
        axes[1, 1], q1_g, q1_vis, q1_gp,
        title="Q1 × Gaia  |  mean $p_{\\rm phz\\_gal\\_prob}$ per bin\n(0 = star,  1 = galaxy)",
        cmap="RdYlBu", vmin=0, vmax=1,
        ylabel=False, xlabel=True,
    )
    fig.colorbar(im_cs, ax=axes[1, 0], fraction=0.046, pad=0.03).set_label("Mean class_star", fontsize=10)
    fig.colorbar(im_gp, ax=axes[1, 1], fraction=0.046, pad=0.03).set_label("Mean $p_{\\rm gal\\_prob}$", fontsize=10)

    fig.suptitle("Euclid VIS vs Gaia $G$ — Gaia-matched Sources",
                 fontsize=14, fontweight="bold")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
