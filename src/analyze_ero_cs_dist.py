#!/usr/bin/env python3
"""
ERO source count + class fraction vs Euclid VIS magnitude (all ERO sources).

Produces a single wide figure: count histogram (top) + class fraction (bottom).

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_ero_cs_dist.py

Output:
    output/ero_cs_dist.png
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR  = PROJECT_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "ero_cs_dist.png"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

BINS = np.arange(10, 29.25, 0.25)
XLIM = (10, 29)


def query_all_ero(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = """
        SELECT
            v_mag_auto,
            CASE
                WHEN v_class_star >= 0.65 THEN 'STAR'
                WHEN v_class_star < 0.65  THEN 'GALAXY'
            END AS label
        FROM euclid_ero.merge_cat
        WHERE v_class_star IS NOT NULL
    """
    print("Querying all ERO data...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} sources")
    return df


def plot(df):
    mag_all  = df["v_mag_auto"].dropna()
    mag_star = df[df["label"] == "STAR"]["v_mag_auto"].dropna()
    mag_gal  = df[df["label"] == "GALAXY"]["v_mag_auto"].dropna()

    n_gal  = int(np.histogram(mag_gal,  bins=BINS)[0].sum())
    n_star = int(np.histogram(mag_star, bins=BINS)[0].sum())
    n_all  = n_gal + n_star

    counts_all,  _ = np.histogram(mag_all,  bins=BINS)
    count_ylim = int(np.ceil(counts_all.max() * 1.10 / 1000) * 1000)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(12, 7),
        gridspec_kw={"height_ratios": [2.5, 1.5]},
    )
    fig.subplots_adjust(
        left=0.10, right=0.97,
        bottom=0.08, top=0.93,
        hspace=0.30,
    )

    # ── Top: count histogram ─────────────────────────────────────────────────
    ax_top.hist(mag_all,  bins=BINS, alpha=0.3, color="gray",
                label=f"All ({n_all:,})")
    ax_top.hist(mag_gal,  bins=BINS, alpha=0.7, color="steelblue",
                label=f"Galaxies ({n_gal:,})", histtype="step", linewidth=1.5)
    ax_top.hist(mag_star, bins=BINS, alpha=0.7, color="coral",
                label=f"Stars ({n_star:,})", histtype="step", linewidth=1.5)
    ax_top.set_ylabel("Count", fontsize=11)
    ax_top.set_xlim(XLIM)
    ax_top.set_ylim(0, count_ylim)
    ax_top.set_title("Sources Count (All ERO, Euclid VIS)", fontsize=12, fontweight="bold")
    ax_top.legend(fontsize=9, loc="upper left")
    ax_top.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax_top.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_top.grid(alpha=0.3, axis="y")
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # ── Bottom: class fraction ───────────────────────────────────────────────
    bin_centers  = (BINS[:-1] + BINS[1:]) / 2
    counts_gal,  _ = np.histogram(mag_gal,  bins=BINS)
    counts_star, _ = np.histogram(mag_star, bins=BINS)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_gal  = np.where(counts_all > 0, counts_gal  / counts_all, np.nan)
        frac_star = np.where(counts_all > 0, counts_star / counts_all, np.nan)
    mask = counts_all >= 10
    ax_bot.plot(bin_centers[mask], frac_star[mask], color="coral",
                linewidth=1.5, alpha=0.3)
    ax_bot.plot(bin_centers[mask], frac_gal[mask], "o-", color="steelblue",
                markersize=3, linewidth=1.5)
    ax_bot.set_xlabel("Euclid VIS (v_mag_auto)", fontsize=11)
    ax_bot.set_ylabel("Class fraction", fontsize=11)
    ax_bot.set_xlim(XLIM)
    ax_bot.set_ylim(-0.02, 1.02)
    ax_bot.set_title("Class Fraction (All ERO, Euclid VIS)", fontsize=12, fontweight="bold")
    ax_bot.axhline(y=0.5, color="gray", ls="--", alpha=0.4)
    ax_bot.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax_bot.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax_bot.grid(alpha=0.3)

    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_FILE}")


def main():
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1
    df = query_all_ero(username)
    plot(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
