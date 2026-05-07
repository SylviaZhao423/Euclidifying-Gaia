#!/usr/bin/env python3
"""
Q1 source count + class fraction vs Euclid I_E magnitude (all Q1 sources),
classified by PHZ: p_phz_classification (1=star, 2=galaxy, else=unclassified).

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_q1_phz_dist.py

Output:
    output/q1_phz_dist.png
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_DIR    = Path(__file__).resolve().parents[1]
OUTPUT_FILE    = PROJECT_DIR / "output" / "q1_phz_dist_class.png"
OUTPUT_ALLCLS  = PROJECT_DIR / "output" / "q1_phz_dist.png"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"


BINS = np.arange(10, 29.25, 0.25)
XLIM = (10, 29)


# PHZ classification value → (label, color)
PHZ_CLASSES = {
    -1: ("Reject",          "cadetblue"),
     0: ("Outlier",           "rosybrown"),
     1: ("Star",              "coral"),
     2: ("Galaxy",            "steelblue"),
     3: ("Star \u2229 Galaxy",     "mediumpurple"),
     4: ("QSO",               "goldenrod"),
     5: ("Star \u2229 QSO",        "salmon"),
     6: ("Galaxy \u2229 QSO",      "teal"),
     7: ("Star \u2229 Galaxy \u2229 QSO", "olivedrab"),
}


def query_all_q1(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = f"""
        SELECT
            m_ie_mag AS ie_mag,
            p_phz_classification AS phz
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis
        WHERE p_phz_classification IS NOT NULL
    """
    print("Querying all Q1 PHZ data...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} sources")
    return df


def plot(df):
    mag_all = df["ie_mag"].dropna()
    counts_all, _ = np.histogram(mag_all, bins=BINS)

    # Full (unclipped) counts for legend labels
    n_all_total = len(df)

    # Per-class histograms for all PHZ values
    class_counts = {}
    class_mags   = {}
    class_totals = {}
    for val, (label, color) in PHZ_CLASSES.items():
        mask_cls = df["phz"] == val
        class_totals[val] = int(mask_cls.sum())
        mags = df[mask_cls]["ie_mag"].dropna()
        class_mags[val]   = mags
        class_counts[val], _ = np.histogram(mags, bins=BINS)

    # Binary: classes 1+2 only
    mag_gal    = class_mags[2]
    mag_star   = class_mags[1]
    mag_binary = pd.concat([mag_gal, mag_star])
    counts_gal,    _ = np.histogram(mag_gal,    bins=BINS)
    counts_star,   _ = np.histogram(mag_star,   bins=BINS)
    counts_binary, _ = np.histogram(mag_binary, bins=BINS)
    n_binary = int(counts_binary.sum())

    count_ylim = int(np.ceil(counts_all.max() * 1.10 / 1000) * 1000)
    bin_centers = (BINS[:-1] + BINS[1:]) / 2

    # # ── 2×2 figure: q1_phz_dist_class.png (commented out) ────────────────────
    # fig, axes = plt.subplots(
    #     2, 2,
    #     figsize=(20, 6),
    #     gridspec_kw={"height_ratios": [2.5, 1.5]},
    # )
    # fig.subplots_adjust(
    #     left=0.07, right=0.98,
    #     bottom=0.08, top=0.93,
    #     hspace=0.25, wspace=0.25,
    # )
    #
    # def _fmt_count(ax):
    #     ax.set_xlim(XLIM)
    #     ax.set_ylim(0, count_ylim)
    #     ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    #     ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    #     ax.grid(alpha=0.3, axis="y")
    #     plt.setp(ax.get_xticklabels(), visible=False)
    #
    # def _fmt_frac(ax):
    #     ax.set_xlabel("Euclid VIS (ie_mag)", fontsize=11, fontweight="bold")
    #     ax.set_ylabel("Class fraction", fontsize=11)
    #     ax.set_xlim(XLIM)
    #     ax.set_ylim(-0.02, 1.02)
    #     ax.axhline(y=0.5, color="gray", ls="--", alpha=0.4)
    #     ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    #     ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    #     ax.grid(alpha=0.3)
    #
    # # ── Col 1: all PHZ classes ───────────────────────────────────────────────
    # ax = axes[0, 0]
    # ax.hist(mag_all, bins=BINS, alpha=0.3, color="gray", label=f"All ({n_all_total:,})")
    # for val, (label, color) in PHZ_CLASSES.items():
    #     ax.hist(class_mags[val], bins=BINS, alpha=0.7, color=color,
    #             label=f"{val}: {label} ({class_totals[val]:,})", histtype="step", linewidth=1.5)
    # ax.set_ylabel("Count", fontsize=11)
    # ax.set_title("Sources Count — Q1 PHZ (p_phz_classification)", fontsize=11, fontweight="bold")
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[1:] + handles[:1], labels[1:] + labels[:1], fontsize=8, loc="upper left")
    # _fmt_count(ax)
    #
    # ax = axes[1, 0]
    # mask = counts_all >= 1
    # for val, (label, color) in PHZ_CLASSES.items():
    #     with np.errstate(divide="ignore", invalid="ignore"):
    #         frac = np.where(counts_all > 0, class_counts[val] / counts_all, np.nan)
    #     fmt = "o-" if val == 2 else "-"
    #     ms  = 3     if val == 2 else 0
    #     alpha = 1.0 if val == 2 else 0.3
    #     ax.plot(bin_centers[mask], frac[mask], fmt, color=color,
    #             linewidth=1.5, markersize=ms, alpha=alpha, label=f"{val}: {label}")
    # ax.set_title("Class Fraction — Q1 PHZ (each class / all)", fontsize=11, fontweight="bold")
    # # ax.legend(fontsize=8, loc="upper left")
    # _fmt_frac(ax)
    #
    # # ── Col 2: classified only (1,2,3,4,5,6,7); 3,5,6,7 merged as Multi-class
    # mag_qso   = class_mags[4]
    # mag_multi = pd.concat([class_mags[v] for v in (3, 5, 6, 7)])
    # counts_qso,   _ = np.histogram(mag_qso,   bins=BINS)
    # counts_multi, _ = np.histogram(mag_multi, bins=BINS)
    #
    # mag_classified = pd.concat([class_mags[v] for v in (1, 2, 3, 4, 5, 6, 7)])
    # counts_classified, _ = np.histogram(mag_classified, bins=BINS)
    # n_classified_total = sum(class_totals[v] for v in (1, 2, 3, 4, 5, 6, 7))
    # n_gal_total   = class_totals[2]
    # n_star_total  = class_totals[1]
    # n_qso_total   = class_totals[4]
    # n_multi_total = sum(class_totals[v] for v in (3, 5, 6, 7))
    #
    # ax = axes[0, 1]
    # ax.hist(mag_classified, bins=BINS, alpha=0.3, color="gray",
    #         label=f"All classified ({n_classified_total:,})")
    # ax.hist(mag_gal,   bins=BINS, alpha=0.7, color=PHZ_CLASSES[2][1],
    #         label=f"2: Galaxy ({n_gal_total:,})",             histtype="step", linewidth=1.5)
    # ax.hist(mag_star,  bins=BINS, alpha=0.7, color=PHZ_CLASSES[1][1],
    #         label=f"1: Star ({n_star_total:,})",              histtype="step", linewidth=1.5)
    # ax.hist(mag_qso,   bins=BINS, alpha=0.7, color=PHZ_CLASSES[4][1],
    #         label=f"4: QSO ({n_qso_total:,})",                histtype="step", linewidth=1.5)
    # ax.hist(mag_multi, bins=BINS, alpha=0.7, color="darkslateblue",
    #         label=f"Multi-class 3,5,6,7 ({n_multi_total:,})", histtype="step", linewidth=1.5)
    # ax.set_ylabel("Count", fontsize=11)
    # ax.set_title("Sources Count — Q1 PHZ classified (classes 1-7, excl. -1 & 0)", fontsize=11, fontweight="bold")
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[1:] + handles[:1], labels[1:] + labels[:1], fontsize=9, loc="upper left")
    # _fmt_count(ax)
    #
    # ax = axes[1, 1]
    # mask2 = counts_classified >= 1
    # with np.errstate(divide="ignore", invalid="ignore"):
    #     frac_gal2   = np.where(counts_classified > 0, counts_gal   / counts_classified, np.nan)
    #     frac_star2  = np.where(counts_classified > 0, counts_star  / counts_classified, np.nan)
    #     frac_qso2   = np.where(counts_classified > 0, counts_qso   / counts_classified, np.nan)
    #     frac_multi2 = np.where(counts_classified > 0, counts_multi / counts_classified, np.nan)
    # ax.plot(bin_centers[mask2], frac_star2[mask2],  color=PHZ_CLASSES[1][1],  linewidth=1.5, alpha=0.3)
    # ax.plot(bin_centers[mask2], frac_qso2[mask2],   color=PHZ_CLASSES[4][1],  linewidth=1.5, alpha=0.3)
    # ax.plot(bin_centers[mask2], frac_multi2[mask2], color="darkslateblue",      linewidth=1.5, alpha=0.3)
    # ax.plot(bin_centers[mask2], frac_gal2[mask2],   "o-", color=PHZ_CLASSES[2][1], markersize=3, linewidth=1.5)
    # ax.set_title("Class Fraction — Q1 PHZ classified (each / classes 1-7)", fontsize=11, fontweight="bold")
    # _fmt_frac(ax)
    #
    # plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    # plt.close()
    # print(f"Saved: {OUTPUT_FILE}")

    # ── Single-column figure: all PHZ classes (ero_cs_dist layout) ───────────
    fig2, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(8, 5.4),
        gridspec_kw={"height_ratios": [2.5, 1.5]},
    )
    fig2.subplots_adjust(
        left=0.10, right=0.97,
        bottom=0.10, top=0.93,
        hspace=0.25,
    )

    ax_top.hist(mag_all, bins=BINS, alpha=0.3, color="gray", label=f"All ({n_all_total:,})")
    for val, (label, color) in PHZ_CLASSES.items():
        ax_top.hist(class_mags[val], bins=BINS, alpha=0.7, color=color,
                    label=f"{val}: {label} ({class_totals[val]:,})", histtype="step", linewidth=2.0)
    ax_top.set_ylabel("Count", fontsize=13, fontweight="bold")
    ax_top.set_title("Source count — Q1 PHZ (p_phz_classification)", fontsize=14, fontweight="bold")
    handles2, labels2 = ax_top.get_legend_handles_labels()
    ax_top.legend(handles2[1:] + handles2[:1], labels2[1:] + labels2[:1], fontsize=10, loc="upper left")
    ax_top.set_xlim(XLIM)
    ax_top.set_ylim(0, count_ylim)
    ax_top.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax_top.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_top.tick_params(labelsize=12)
    ax_top.grid(alpha=0.3, axis="y")
    plt.setp(ax_top.get_xticklabels(), visible=False)

    mask = counts_all >= 1
    for val, (label, color) in PHZ_CLASSES.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(counts_all > 0, class_counts[val] / counts_all, np.nan)
        fmt   = "o-" if val == 2 else "-"
        ms    = 3    if val == 2 else 0
        alpha = 1.0  if val == 2 else 0.3
        ax_bot.plot(bin_centers[mask], frac[mask], fmt, color=color,
                    linewidth=2.0, markersize=ms, alpha=alpha)
    ax_bot.set_xlabel("Euclid VIS (ie_mag)", fontsize=13, fontweight="bold")
    ax_bot.set_ylabel("Class fraction", fontsize=13, fontweight="bold", labelpad=30)
    ax_bot.set_title("Class fraction — Q1 PHZ (each class / all)", fontsize=14, fontweight="bold")
    ax_bot.set_xlim(XLIM)
    ax_bot.set_ylim(-0.02, 1.02)
    ax_bot.axhline(y=0.5, color="gray", ls="--", linewidth=1, alpha=0.4)
    ax_bot.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax_bot.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax_bot.tick_params(labelsize=12)
    ax_bot.grid(alpha=0.3)

    plt.savefig(OUTPUT_ALLCLS, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_ALLCLS}")


def main():
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1
    df = query_all_q1(username)
    plot(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
