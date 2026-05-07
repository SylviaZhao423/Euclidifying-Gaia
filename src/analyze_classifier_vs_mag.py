#!/usr/bin/env python3
"""
Side-by-side 2D histograms of morphological classifiers vs Gaia G:
  Left  — ERO class_star vs Gaia G  (crossmatch_ero_gaia_0_3)
  Right — Q1  phz_gal_prob vs Gaia G (crossmatch_q1_mer_phz_vis_gaia_sep)

These are the Gaia-matched panels extracted from:
  ero_prob_class_star_2d.png  (panel 2)
  q1_prob_phz_2d.png          (bottom-right)

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_classifier_vs_gaia_g_2d.py

Output:
    output/classifier_vs_gaia_g_2d.png
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

CMAP_WPP = LinearSegmentedColormap.from_list(
    "white_pink_purple", ["#ffffff", "#f48fb1", "#6a1b9a"])


def truncate_cmap(cmap, minval=0.1, maxval=0.9, n=256):
    cmap_obj = cmap if hasattr(cmap, '__call__') else plt.get_cmap(cmap)
    return mcolors.LinearSegmentedColormap.from_list(
        f"trunc({cmap_obj.name},{minval:.2f},{maxval:.2f})",
        cmap_obj(np.linspace(minval, maxval, n)),
    )
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_DIR / "output" / "classifier_vs_mag.png"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

XBINS = np.arange(10, 23.05, 0.05)
YBINS = np.arange(0, 1.01,  0.004)
XLIM  = (10, 23)


def query_ero(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_class_star IS NOT NULL
    """)
    n_total = cur.fetchone()[0]
    query = """
        SELECT
            g_phot_g_mean_mag AS gaia_g,
            e_v_class_star    AS class_star
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_class_star    IS NOT NULL
          AND g_phot_g_mean_mag IS NOT NULL
    """
    print("Querying ERO-Gaia matched data...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} ERO sources (total with class_star: {n_total:,})")
    return df, n_total


def query_q1(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep
        WHERE p_phz_gal_prob IS NOT NULL
    """)
    n_total = cur.fetchone()[0]
    query = """
        SELECT
            g_phot_g_mean_mag AS gaia_g,
            p_phz_gal_prob    AS gal_prob
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep
        WHERE g_phot_g_mean_mag IS NOT NULL
          AND p_phz_gal_prob    IS NOT NULL
    """
    print("Querying Q1-Gaia matched data...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} Q1 sources (total with gal_prob: {n_total:,})")
    return df, n_total


def _plot_panel(ax, xvals, yvals, ylabel, title, hline_y, hline_label,
                show_ylabel=True):
    h, xedges, yedges = np.histogram2d(xvals, yvals, bins=[XBINS, YBINS])
    h_masked = np.ma.masked_where(h == 0, h)
    im = ax.pcolormesh(xedges, yedges, h_masked.T, cmap=truncate_cmap("PuBu", 0.3, 1.0), alpha=0.8,
                       norm=LogNorm(vmin=1), rasterized=True)
    ax.axvline(x=17, color="dimgray", ls="--", linewidth=1.2, alpha=0.6, label="$G = 17$")
    ax.axhline(y=hline_y, color="sandybrown", ls="--", linewidth=1.5, label=hline_label)
    ax.set_xlabel("Gaia G (mag)", fontsize=13, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=13, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(XLIM)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.tick_params(labelsize=12)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.7)
    ax.grid(alpha=0.25)
    return im


def main():
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set"); return 1

    df_ero, n_ero_total = query_ero(username)
    df_q1,  n_q1_total  = query_q1(username)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    fig.subplots_adjust(left=0.08, right=0.89, top=0.90, bottom=0.14, wspace=0.26)

    im1 = _plot_panel(
        axes[0],
        xvals=df_ero["gaia_g"].values,
        yvals=df_ero["class_star"].values,
        ylabel="class_star",
        title="ERO \u00d7 Gaia",
        hline_y=0.65, hline_label="threshold = 0.65",
    )
    # Box: G > 17, class_star > 0.9
    from matplotlib.patches import Rectangle
    rect0 = Rectangle(
        (17, 0.9), 23 - 17, 1.0 - 0.9,
        linewidth=2.5, edgecolor="orangered", facecolor="none",
        ls="--", clip_on=False, zorder=10,
    )
    axes[0].add_patch(rect0)

    im2 = _plot_panel(
        axes[1],
        xvals=df_q1["gaia_g"].values,
        yvals=df_q1["gal_prob"].values,
        ylabel="phz_gal_prob",
        title="Q1 \u00d7 Gaia",
        hline_y=0.5, hline_label="threshold = 0.5",
        show_ylabel=True,
    )
    # Box: G > 17, full y range
    rect1 = Rectangle(
        (17, 0), 23 - 17, 1.0,
        linewidth=2.5, edgecolor="orangered", facecolor="none",
        ls="--", clip_on=False, zorder=10,
    )
    axes[1].add_patch(rect1)
    # Single shared colorbar
    cbar_ax = fig.add_axes([0.91, 0.14, 0.014, 0.76])
    cb = fig.colorbar(im2, cax=cbar_ax)
    cb.set_label("Count", fontsize=12, fontweight="bold")
    cb.ax.tick_params(labelsize=11)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
