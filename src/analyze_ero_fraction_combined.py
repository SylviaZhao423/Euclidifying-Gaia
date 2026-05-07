#!/usr/bin/env python3
"""
Combined ERO extended fraction plot:
  Top    — CLASS_STAR fraction vs Euclid VIS magnitude
  Bottom — SPREAD_MODEL fraction (4 classifiers) vs Euclid VIS magnitude

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_ero_fraction_combined.py

Output:
    output/ero_fraction_combined.png
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_DIR / "output" / "ero_fraction_combined.png"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

MAG_MIN   = 10.0
MAG_MAX   = 29.0
BIN_WIDTH = 0.25
BINS      = np.arange(MAG_MIN, MAG_MAX + BIN_WIDTH, BIN_WIDTH)
XLIM      = (MAG_MIN, MAG_MAX)


def query_cs(username):
    """Query CLASS_STAR data from ERO."""
    import psycopg2
    import pandas as pd
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = """
        SELECT
            v_mag_auto,
            CASE
                WHEN v_class_star >= 0.65 THEN 'STAR'
                WHEN v_class_star < 0.65  THEN 'GALAXY'
            END AS label
        FROM euclid_ero.merge_cat
    """
    print("Querying ERO CLASS_STAR data...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} sources")
    return df


def query_sm(username):
    """Query SPREAD_MODEL data from ERO."""
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
    print("Querying ERO SPREAD_MODEL data...")
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    print(f"  Retrieved {len(rows)} rows")

    n_bins = int((MAG_MAX - MAG_MIN) / BIN_WIDTH)
    bin_centers = MAG_MIN + (np.arange(n_bins) + 0.5) * BIN_WIDTH

    sp_point     = np.zeros(n_bins)
    sp_ext       = np.zeros(n_bins)
    simple_point = np.zeros(n_bins)
    simple_ext   = np.zeros(n_bins)
    des_point    = np.zeros(n_bins)
    des_ext      = np.zeros(n_bins)
    modest_point = np.zeros(n_bins)
    modest_ext   = np.zeros(n_bins)

    for bin_idx, sp_class, simple_class, des_class, modest_class, cnt in rows:
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

    return (bin_centers,
            sp_point, sp_ext,
            simple_point, simple_ext,
            des_point, des_ext,
            modest_point, modest_ext)


def plot(df_cs, sm_data):
    bin_centers_sm, sp_point, sp_ext, simple_point, simple_ext, \
        des_point, des_ext, modest_point, modest_ext = sm_data

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(10, 5.4),
        gridspec_kw={"height_ratios": [1, 1]},
    )
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.10, top=0.93,
                        hspace=0.25)

    # ── Top: CLASS_STAR fraction ─────────────────────────────────────────
    mag_all  = df_cs["v_mag_auto"].dropna()
    mag_star = df_cs[df_cs["label"] == "STAR"]["v_mag_auto"].dropna()
    mag_gal  = df_cs[df_cs["label"] == "GALAXY"]["v_mag_auto"].dropna()

    counts_all,  _ = np.histogram(mag_all,  bins=BINS)
    counts_gal,  _ = np.histogram(mag_gal,  bins=BINS)
    counts_star, _ = np.histogram(mag_star, bins=BINS)
    bin_centers_cs = (BINS[:-1] + BINS[1:]) / 2

    with np.errstate(divide="ignore", invalid="ignore"):
        frac_gal  = np.where(counts_all > 0, counts_gal  / counts_all, np.nan)
        frac_star = np.where(counts_all > 0, counts_star / counts_all, np.nan)
    mask = counts_all >= 10

    # ax_top.plot(bin_centers_cs[mask], frac_star[mask], color="coral",
                # linewidth=2.0, alpha=0.3)
    ax_top.plot(bin_centers_cs[mask], frac_gal[mask], "o-", color="steelblue",
                markersize=2, linewidth=1.8, alpha=0.7)
    ax_top.axhline(y=0.5, color="gray", ls="--", linewidth=1, alpha=0.4)
    ax_top.set_ylabel("Extended fraction", fontsize=13, fontweight="bold", labelpad=12)
    ax_top.set_title(r"Extended fraction — CLASS_STAR (threshold = 0.65)",
                     fontsize=14, fontweight="bold")
    ax_top.set_xlim(XLIM)
    ax_top.set_ylim(-0.02, 1.02)
    ax_top.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax_top.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax_top.tick_params(labelsize=12)
    ax_top.grid(alpha=0.3)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # ── Bottom: SPREAD_MODEL fraction (4 classifiers) ────────────────────
    classifiers = [
        (sp_point,     sp_ext,     "SExtractor",      "#1f77b4"),
        (simple_point, simple_ext, "Fixed Cut",        "#ff7f0e"),
        (des_point,    des_ext,    "MODEST",           "#2ca02c"),
        (modest_point, modest_ext, "Revised MODEST",   "hotpink"),
    ]

    for point, ext, label, color in classifiers:
        total = point + ext
        sm_mask = total >= 10
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(total > 0, ext / total, np.nan)
        ax_bot.plot(bin_centers_sm[sm_mask], frac[sm_mask], "o-", color=color,
                    markersize=2, linewidth=1.8, alpha=0.7, label=label)

    ax_bot.axhline(y=0.5, color="gray", ls="--", linewidth=1, alpha=0.4)
    ax_bot.set_xlabel("Euclid VIS (v_mag_auto)", fontsize=13, fontweight="bold")
    ax_bot.set_ylabel("Extended fraction", fontsize=13, fontweight="bold", labelpad=12)
    ax_bot.set_title("Extended fraction — SPREAD_MODEL classifiers",
                     fontsize=14, fontweight="bold")
    ax_bot.set_xlim(XLIM)
    ax_bot.set_ylim(-0.02, 1.02)
    ax_bot.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax_bot.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax_bot.tick_params(labelsize=12)
    ax_bot.legend(loc="lower left", bbox_to_anchor=(0.2, 0.02), prop={'size': 10, 'weight': 'bold'})
    ax_bot.grid(alpha=0.3)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_FILE}")


def main():
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1
    df_cs = query_cs(username)
    sm_data = query_sm(username)
    plot(df_cs, sm_data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
