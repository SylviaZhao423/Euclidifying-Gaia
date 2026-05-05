#!/usr/bin/env python3
"""
Gaia-matched source count + class fraction vs Gaia G magnitude.

Two columns side by side:
  Left  — ERO × Gaia (crossmatch_ero_gaia_0_3), classified by class_star ≥ 0.65
  Right — Q1  × Gaia (crossmatch_q1_mer_phz_vis_gaia_sep), classified by p_phz_classification

Both use Gaia G (g_phot_g_mean_mag) on the x-axis.

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_gaia_matched_class_dist.py

Output:
    output/gaia_matched_class_dist.png
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_DIR / "output" / "gaia_matched_class_dist.png"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

BINS = np.arange(10, 23.25, 0.25)
XLIM = (10, 23)

PHZ_CLASSES = {
    -1: ("Unclassified",          "darkcyan"),
     0: ("Outlier",               "rosybrown"),
     1: ("Star",                  "coral"),
     2: ("Galaxy",                "steelblue"),
     3: ("Star \u2229 Galaxy",    "darkslateblue"),
     4: ("QSO",                   "goldenrod"),
     5: ("Star \u2229 QSO",       "salmon"),
     6: ("Galaxy \u2229 QSO",     "teal"),
     7: ("Star \u2229 Gal \u2229 QSO", "olivedrab"),
}


# ── Queries ───────────────────────────────────────────────────────────────────

def query_ero(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = """
        SELECT
            g_phot_g_mean_mag  AS g_mag,
            CASE
                WHEN e_v_class_star >= 0.65 THEN 'STAR'
                ELSE 'GALAXY'
            END AS label
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_class_star IS NOT NULL
    """
    print("Querying ERO-Gaia matched data...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} sources")
    return df


def query_q1(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = """
        SELECT
            g_phot_g_mean_mag       AS g_mag,
            p_phz_classification    AS phz
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep
        WHERE g_phot_g_mean_mag IS NOT NULL
          AND p_phz_classification IN (-1, 0, 1, 2, 3, 4, 5, 6, 7)
    """
    print("Querying Q1-Gaia matched data...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} sources")
    return df


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _fmt_count(ax, count_ylim):
    ax.set_xlim(XLIM)
    ax.set_ylim(0, count_ylim)
    ax.set_ylabel("Count", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), visible=False)


def _fmt_frac(ax):
    ax.set_xlabel("Gaia G (mag)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Class fraction", fontsize=13, fontweight="bold")
    ax.set_xlim(XLIM)
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(y=0.5, color="gray", ls="--", alpha=0.4)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.3)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set"); return 1

    df_ero = query_ero(username)
    df_q1  = query_q1(username)

    bin_centers = (BINS[:-1] + BINS[1:]) / 2

    # ── ERO histograms ────────────────────────────────────────────────────────
    mag_ero_all  = df_ero["g_mag"].dropna()
    mag_ero_star = df_ero[df_ero["label"] == "STAR"]["g_mag"].dropna()
    mag_ero_gal  = df_ero[df_ero["label"] == "GALAXY"]["g_mag"].dropna()
    c_ero_all,  _ = np.histogram(mag_ero_all,  bins=BINS)
    c_ero_star, _ = np.histogram(mag_ero_star, bins=BINS)
    c_ero_gal,  _ = np.histogram(mag_ero_gal,  bins=BINS)
    n_ero_all  = len(df_ero)
    n_ero_star = int(c_ero_star.sum())
    n_ero_gal  = int(c_ero_gal.sum())

    # ── Q1 histograms ─────────────────────────────────────────────────────────
    mag_q1_all   = df_q1["g_mag"].dropna()
    mag_q1_star  = df_q1[df_q1["phz"] == 1]["g_mag"].dropna()
    mag_q1_gal   = df_q1[df_q1["phz"] == 2]["g_mag"].dropna()
    mag_q1_qso   = df_q1[df_q1["phz"] == 4]["g_mag"].dropna()
    mag_q1_multi = df_q1[df_q1["phz"].isin([3, 5, 6, 7])]["g_mag"].dropna()
    mag_q1_uncls = df_q1[df_q1["phz"].isin([-1, 0])]["g_mag"].dropna()
    c_q1_all,   _ = np.histogram(mag_q1_all,   bins=BINS)
    c_q1_star,  _ = np.histogram(mag_q1_star,  bins=BINS)
    c_q1_gal,   _ = np.histogram(mag_q1_gal,   bins=BINS)
    c_q1_qso,   _ = np.histogram(mag_q1_qso,   bins=BINS)
    c_q1_multi, _ = np.histogram(mag_q1_multi, bins=BINS)
    c_q1_uncls, _ = np.histogram(mag_q1_uncls, bins=BINS)
    n_q1_all   = len(df_q1)
    n_q1_star  = len(mag_q1_star)
    n_q1_gal   = len(mag_q1_gal)
    n_q1_qso   = len(mag_q1_qso)
    n_q1_multi = len(mag_q1_multi)
    n_q1_uncls = len(mag_q1_uncls)

    # ── Figure ────────────────────────────────────────────────────────────────
    count_ylim_ero = int(np.ceil(c_ero_all.max() * 1.12 / 1000) * 1000)
    count_ylim_q1  = int(np.ceil(c_q1_all.max()  * 1.12 / 1000) * 1000)

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 6),
        gridspec_kw={"height_ratios": [2.5, 1.5], "wspace": 0.25, "hspace": 0.22},
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.09)

    # ── ERO: count ───────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.hist(mag_ero_all,  bins=BINS, alpha=0.3, color="gray",
            label=f"All ({n_ero_all:,})")
    ax.hist(mag_ero_gal,  bins=BINS, alpha=0.7, color="steelblue",
            label=f"Extended ({n_ero_gal:,})", histtype="step", linewidth=2.0)
    ax.hist(mag_ero_star, bins=BINS, alpha=0.7, color="coral",
            label=f"Point-like ({n_ero_star:,})", histtype="step", linewidth=2.0)
    ax.set_title("ERO × Gaia",
                 fontsize=13, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[1:] + handles[:1], labels[1:] + labels[:1],
              fontsize=10, loc="upper left")
    ax.axvline(x=17, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
    _fmt_count(ax, count_ylim_ero)

    # ── ERO: fraction ─────────────────────────────────────────────────────────
    ax = axes[1, 0]
    mask = c_ero_all >= 10
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_ero_star = np.where(c_ero_all > 0, c_ero_star / c_ero_all, np.nan)
        frac_ero_gal  = np.where(c_ero_all > 0, c_ero_gal  / c_ero_all, np.nan)
    ax.plot(bin_centers[mask], frac_ero_star[mask], color="coral",
            linewidth=1.5, alpha=0.3)
    ax.plot(bin_centers[mask], frac_ero_gal[mask],  "o-", color="steelblue",
            markersize=3, linewidth=1.5)
    # title omitted — ylabel "Class Fraction" is sufficient
    ax.axvline(x=17, color="black", linestyle="--", linewidth=1.2, alpha=0.6,
               label="$G = 17$")
    ax.legend(fontsize=10, loc="upper left")
    _fmt_frac(ax)

    # ── Q1: count ─────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.hist(mag_q1_all,   bins=BINS, alpha=0.3, color="gray",
            label=f"All ({n_q1_all:,})")
    ax.hist(mag_q1_gal,   bins=BINS, alpha=0.7, color="steelblue",
            label=f"2: Galaxy ({n_q1_gal:,})", histtype="step", linewidth=2.0)
    ax.hist(mag_q1_star,  bins=BINS, alpha=0.7, color="coral",
            label=f"1: Star ({n_q1_star:,})", histtype="step", linewidth=2.0)
    ax.hist(mag_q1_qso,   bins=BINS, alpha=0.7, color="goldenrod",
            label=f"4: QSO ({n_q1_qso:,})", histtype="step", linewidth=2.0)
    ax.hist(mag_q1_multi, bins=BINS, alpha=0.7, color="darkslateblue",
            label=f"Multi-class 3,5,6,7 ({n_q1_multi:,})", histtype="step", linewidth=2.0)
    ax.hist(mag_q1_uncls, bins=BINS, alpha=0.7, color="darkcyan",
            label=f"Unclassified \u22121,0 ({n_q1_uncls:,})", histtype="step", linewidth=2.0)
    ax.set_title("Q1 × Gaia",
                 fontsize=13, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[1:] + handles[:1], labels[1:] + labels[:1],
              fontsize=10, loc="upper left")
    ax.axvline(x=17, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
    _fmt_count(ax, count_ylim_q1)
    ax.set_ylabel("")  # shared with left column

    # ── Q1: fraction ──────────────────────────────────────────────────────────
    ax = axes[1, 1]
    mask2 = c_q1_all >= 10
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_q1_gal   = np.where(c_q1_all > 0, c_q1_gal   / c_q1_all, np.nan)
        frac_q1_star  = np.where(c_q1_all > 0, c_q1_star  / c_q1_all, np.nan)
        frac_q1_qso   = np.where(c_q1_all > 0, c_q1_qso   / c_q1_all, np.nan)
        frac_q1_multi = np.where(c_q1_all > 0, c_q1_multi / c_q1_all, np.nan)
        frac_q1_uncls = np.where(c_q1_all > 0, c_q1_uncls / c_q1_all, np.nan)
    ax.plot(bin_centers[mask2], frac_q1_gal[mask2],   "o-", color="steelblue",
            markersize=3, linewidth=1.5)
    ax.plot(bin_centers[mask2], frac_q1_star[mask2],  color="coral",
            linewidth=1.5, alpha=0.3)
    ax.plot(bin_centers[mask2], frac_q1_qso[mask2],   color="goldenrod",
            linewidth=1.5, alpha=0.3)
    ax.plot(bin_centers[mask2], frac_q1_multi[mask2], color="darkslateblue",
            linewidth=1.5, alpha=0.3)
    ax.plot(bin_centers[mask2], frac_q1_uncls[mask2], color="darkcyan",
            linewidth=1.5, alpha=0.3)
    # title omitted — ylabel "Class Fraction" is sufficient
    ax.axvline(x=17, color="black", linestyle="--", linewidth=1.2, alpha=0.6,
               label="$G = 17$")
    ax.legend(fontsize=10, loc="upper left")
    _fmt_frac(ax)
    ax.set_ylabel("")  # shared with left column

    # suptitle removed — covered by LaTeX caption

    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
