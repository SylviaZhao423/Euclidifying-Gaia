#!/usr/bin/env python3
"""
ERO source count + galaxy fraction vs Euclid VIS magnitude for all 4
SPREAD_MODEL classifiers (SExtractor, Simple, DES, MODEST) — 4-column figure.

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_ero_sm_dist.py

Output:
    output/ero_sm_dist.png
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE     = PROJECT_DIR / "output" / "ero_sm_dist.png"
OUTPUT_COMBINED = PROJECT_DIR / "output" / "ero_sm_dist_combined.png"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

MAG_MIN   = 10.0
MAG_MAX   = 29.0
BIN_WIDTH = 0.25

COLOR_ALL      = "gray"
COLOR_EXTENDED = "steelblue"
COLOR_POINT    = "coral"


def query_data(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()
    query = f"""
        SELECT
            FLOOR((v_mag_auto - {MAG_MIN}) / {BIN_WIDTH}) AS bin_idx,
            CASE
                WHEN v_spread_model <= SQRT(0.005*0.005 + (4.0*v_spreaderr_model)*(4.0*v_spreaderr_model))
                THEN 'SP_POINT' ELSE 'SP_EXT'
            END AS sp_class,
            CASE
                WHEN v_spread_model <= 0.003
                THEN 'SIMPLE_POINT' ELSE 'SIMPLE_EXT'
            END AS simple_class,
            CASE
                WHEN v_spread_model + 3.0 * v_spreaderr_model <= 0.003
                THEN 'DES_POINT' ELSE 'DES_EXT'
            END AS des_class,
            CASE
                WHEN v_spread_model + (5.0/3.0) * v_spreaderr_model <= 0.005
                THEN 'MODEST_POINT' ELSE 'MODEST_EXT'
            END AS modest_class,
            COUNT(*) AS cnt
        FROM euclid_ero.merge_cat
        GROUP BY bin_idx, sp_class, simple_class, des_class, modest_class
        ORDER BY bin_idx
    """
    print("Querying ERO spread model data...")
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    print(f"  Retrieved {len(rows)} rows")

    n_bins = int((MAG_MAX - MAG_MIN) / BIN_WIDTH)
    bin_centers = MAG_MIN + (np.arange(n_bins) + 0.5) * BIN_WIDTH
    bin_edges   = MAG_MIN + np.arange(n_bins + 1) * BIN_WIDTH

    sp_point     = np.zeros(n_bins)
    sp_ext       = np.zeros(n_bins)
    simple_point = np.zeros(n_bins)
    simple_ext   = np.zeros(n_bins)
    des_point    = np.zeros(n_bins)
    des_ext      = np.zeros(n_bins)
    modest_point = np.zeros(n_bins)
    modest_ext   = np.zeros(n_bins)

    # Full totals (all sources, not just in-range bins) for legend labels
    sp_point_tot = sp_ext_tot = 0
    simple_point_tot = simple_ext_tot = 0
    des_point_tot = des_ext_tot = 0
    modest_point_tot = modest_ext_tot = 0

    for bin_idx, sp_class, simple_class, des_class, modest_class, cnt in rows:
        # Accumulate full totals regardless of mag range
        if sp_class     == "SP_POINT":     sp_point_tot     += cnt
        else:                              sp_ext_tot        += cnt
        if simple_class == "SIMPLE_POINT": simple_point_tot += cnt
        else:                              simple_ext_tot    += cnt
        if des_class    == "DES_POINT":    des_point_tot    += cnt
        else:                              des_ext_tot       += cnt
        if modest_class == "MODEST_POINT": modest_point_tot += cnt
        else:                              modest_ext_tot    += cnt

        # Histogram bins (10–29 only)
        if bin_idx is None:
            continue
        i = int(bin_idx)
        if not (0 <= i < n_bins):
            continue
        if sp_class     == "SP_POINT":     sp_point[i]     += cnt
        else:                              sp_ext[i]        += cnt
        if simple_class == "SIMPLE_POINT": simple_point[i] += cnt
        else:                              simple_ext[i]    += cnt
        if des_class    == "DES_POINT":    des_point[i]    += cnt
        else:                              des_ext[i]       += cnt
        if modest_class == "MODEST_POINT": modest_point[i] += cnt
        else:                              modest_ext[i]    += cnt

    totals = (sp_point_tot, sp_ext_tot,
              simple_point_tot, simple_ext_tot,
              des_point_tot, des_ext_tot,
              modest_point_tot, modest_ext_tot)

    return (bin_centers, bin_edges,
            sp_point, sp_ext,
            simple_point, simple_ext,
            des_point, des_ext,
            modest_point, modest_ext,
            totals)



def count_panel(ax, bin_edges, point, ext, title, n_pt_total=None, n_ext_total=None):
    total = point + ext
    n_pt  = n_pt_total  if n_pt_total  is not None else int(point.sum())
    n_ext = n_ext_total if n_ext_total is not None else int(ext.sum())
    ax.bar(bin_edges[:-1], total, width=BIN_WIDTH, align="edge",
           color=COLOR_ALL, alpha=0.3, label=f"All ({n_pt + n_ext:,})")
    ax.stairs(ext,   bin_edges, color=COLOR_EXTENDED, linewidth=1.5, alpha=0.7,
              label=f"Extended ({n_ext:,})")
    ax.stairs(point, bin_edges, color=COLOR_POINT,    linewidth=1.5, alpha=0.7,
              label=f"Point-like ({n_pt:,})")
    ax.set_ylabel("Count", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim(MAG_MIN, MAG_MAX)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), visible=False)


def fraction_panel(ax, bin_centers, point, ext, title, xlabel):
    total = point + ext
    mask = total >= 10
    with np.errstate(divide="ignore", invalid="ignore"):
        pt_frac = np.where(total > 0, point / total, np.nan)
        ex_frac = np.where(total > 0, ext   / total, np.nan)
    ax.plot(bin_centers[mask], pt_frac[mask], color=COLOR_POINT,
            linewidth=1.5, alpha=0.3, label="Point-like")
    ax.plot(bin_centers[mask], ex_frac[mask], "o-", color=COLOR_EXTENDED,
            markersize=3, linewidth=1.5, label="Extended")
    ax.axhline(y=0.5, color="gray", ls="--", linewidth=1, alpha=0.4)
    ax.set_xlabel(xlabel, fontsize=10, fontweight="bold")
    ax.set_ylabel("Class fraction", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim(MAG_MIN, MAG_MAX)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    # ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)


def plot(bin_centers, bin_edges, sp_point, sp_ext,
         simple_point, simple_ext, des_point, des_ext,
         modest_point, modest_ext, totals):

    (sp_pt_tot, sp_ext_tot,
     simple_pt_tot, simple_ext_tot,
     des_pt_tot, des_ext_tot,
     modest_pt_tot, modest_ext_tot) = totals

    xlabel = "Euclid VIS (v_mag_auto)"

    classifiers = [
        (sp_point,     sp_ext,     sp_pt_tot,     sp_ext_tot,
         "Source Count — SPREAD_MODEL (SExtractor)",
         "Extended Fraction — SExtractor (SM > √(ε²+(κ×SME)²), ε=0.005, κ=4)"),
        (simple_point, simple_ext, simple_pt_tot, simple_ext_tot,
         "Source Count — SPREAD_MODEL (Fixed Cut)",
         "Extended Fraction — Fixed Cut (SM > 0.003)"),
        (des_point,    des_ext,    des_pt_tot,    des_ext_tot,
         "Source Count — SPREAD_MODEL (MODEST Classifier)",
         "Extended Fraction — MODEST Classifier (SM + 3×SME > 0.003)"),
        (modest_point, modest_ext, modest_pt_tot, modest_ext_tot,
         "Source Count — SPREAD_MODEL (Revised MODEST Classifier)",
         "Extended Fraction — Revised MODEST Classifier (SM + 5/3×SME > 0.005)"),
    ]

    fig, axes = plt.subplots(
        2, 4,
        figsize=(28, 8),
        gridspec_kw={"height_ratios": [2.5, 1.5]},
    )
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.08, top=0.93,
                        hspace=0.2, wspace=0.2)

    for col, (point, ext, n_pt_tot, n_ext_tot, top_title, bot_title) in enumerate(classifiers):
        count_panel(axes[0, col],    bin_edges,   point, ext, top_title, n_pt_tot, n_ext_tot)
        fraction_panel(axes[1, col], bin_centers, point, ext, bot_title, xlabel)

    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_FILE}")


def plot_combined(bin_centers, bin_edges, sp_point, sp_ext,
                  simple_point, simple_ext, des_point, des_ext,
                  modest_point, modest_ext, totals):
    """All 4 classifiers overlaid on a single 2-panel figure (like ero_cs_dist.png)."""

    (sp_pt_tot, sp_ext_tot,
     simple_pt_tot, simple_ext_tot,
     des_pt_tot, des_ext_tot,
     modest_pt_tot, modest_ext_tot) = totals

    classifiers = [
        (sp_point,     sp_ext,     sp_ext_tot,
         "SExtractor", "#1f77b4"),
        (simple_point, simple_ext, simple_ext_tot,
         "Fixed Cut", "#ff7f0e"),
        (des_point,    des_ext,    des_ext_tot,
         "MODEST", "#2ca02c"),
        (modest_point, modest_ext, modest_ext_tot,
         "Revised MODEST", "#d62728"),
    ]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [2.5, 1.5]},
    )
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.10, top=0.93,
                        hspace=0.25)

    # Top: count histogram — grey bars for total, coloured step lines for extended
    total_all = sp_point + sp_ext  # same total regardless of classifier
    n_total = int(total_all.sum())
    ax_top.bar(bin_edges[:-1], total_all, width=BIN_WIDTH, align="edge",
               color="gray", alpha=0.3, label=f"All ({n_total:,})")

    for point, ext, n_ext_tot, label, color in classifiers:
        ax_top.stairs(ext, bin_edges, color=color, linewidth=2.0, alpha=0.7,
                      label=f"{label} ({n_ext_tot:,})")

    ax_top.set_ylabel("Count", fontsize=13, fontweight="bold", labelpad=17)
    ax_top.set_xlim(MAG_MIN, MAG_MAX)
    ax_top.set_title("Extended source count — SPREAD_MODEL classifiers",
                     fontsize=14, fontweight="bold")
    ax_top.legend(fontsize=12, loc="upper left")
    ax_top.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax_top.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_top.tick_params(labelsize=12)
    ax_top.grid(alpha=0.3, axis="y")
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # Bottom: extended fraction overlay
    for point, ext, _, label, color in classifiers:
        total = point + ext
        mask = total >= 10
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(total > 0, ext / total, np.nan)
        ax_bot.plot(bin_centers[mask], frac[mask], "o-", color=color,
                    markersize=3, linewidth=2.0, alpha=0.7, label=label)

    ax_bot.axhline(y=0.5, color="gray", ls="--", linewidth=1, alpha=0.4)
    ax_bot.set_xlabel("Euclid VIS (v_mag_auto)", fontsize=13, fontweight="bold")
    ax_bot.set_ylabel("Extended fraction", fontsize=13, fontweight="bold", labelpad=30)
    ax_bot.set_xlim(MAG_MIN, MAG_MAX)
    ax_bot.set_ylim(-0.02, 1.02)
    ax_bot.set_title("Extended fraction — SPREAD_MODEL classifiers",
                     fontsize=14, fontweight="bold")
    ax_bot.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax_bot.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax_bot.tick_params(labelsize=12)
    # ax_bot.legend(fontsize=12, loc="best")
    ax_bot.grid(alpha=0.3)

    plt.savefig(OUTPUT_COMBINED, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_COMBINED}")


def main():
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1
    data = query_data(username)
    plot(*data)
    plot_combined(*data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
