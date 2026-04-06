#!/usr/bin/env python3
"""
2D histogram of DESI Δχ² vs redshift z + agree/disagree Δχ² histograms
for Q1 and ERO crossmatches.

Layout (2×2):
  Row 0: Q1 2D hist | ERO 2D hist
  Row 1: Q1 Δχ² agree/disagree | ERO Δχ² agree/disagree

Output:
  output/desi_z_dc2_2d.png

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_desi_z_dc2.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR  = PROJECT_DIR / "output"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"


def query_q1(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    sql = """
        SELECT d_desi_z, d_desi_zerr, d_desi_deltachi2, p_phz_classification
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_desi
        WHERE d_desi_z IS NOT NULL
          AND d_desi_deltachi2 IS NOT NULL
    """
    print("  Querying Q1-DESI crossmatch...")
    df = pd.read_sql_query(sql, conn)
    conn.close()
    print(f"  -> {len(df):,} sources")
    return df


def query_ero(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    sql = """
        SELECT d_desi_z, d_desi_zerr, d_desi_deltachi2, e_v_class_star
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3_desi
        WHERE d_desi_z IS NOT NULL
          AND d_desi_deltachi2 IS NOT NULL
    """
    print("  Querying ERO-DESI crossmatch...")
    df = pd.read_sql_query(sql, conn)
    conn.close()
    print(f"  -> {len(df):,} sources")
    return df


def plot(df_q1, df_ero, output_dir):
    df_q1["agree"] = df_q1["p_phz_classification"] == 2
    df_ero["agree"] = df_ero["e_v_class_star"] < 0.65

    datasets = [
        (df_q1, f"Q1 DESI  (n={len(df_q1):,})",  "Q1"),
        (df_ero, f"ERO DESI  (n={len(df_ero):,})", "ERO"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    # share x only between rows 0 and 1 (Δχ² axis); row 2 (zerr) uses its own scale
    for ci in range(2):
        axes[1][ci].sharex(axes[0][ci])
    fig.suptitle("DESI Δχ² vs redshift z", fontsize=13, fontweight="bold", y=0.93)
    fig.subplots_adjust(hspace=0.40, wspace=0.35)

    for ci, (df, title, label) in enumerate(datasets):
        sub = df[df["d_desi_deltachi2"] >= 1]
        dc2bins = np.logspace(np.log10(sub["d_desi_deltachi2"].min()),
                              np.log10(sub["d_desi_deltachi2"].max()), 80)
        zbins   = np.linspace(0, sub["d_desi_z"].max() * 1.02, 80)

        # ── Row 0: 2D histogram ───────────────────────────────────────────
        ax = axes[0][ci]
        h, xe, ye = np.histogram2d(sub["d_desi_deltachi2"].values,
                                   sub["d_desi_z"].values,
                                   bins=[dc2bins, zbins])
        hm = np.ma.masked_where(h == 0, h)
        im = ax.pcolormesh(xe, ye, hm.T, cmap="summer",
                           norm=LogNorm(vmin=1, vmax=hm.max()))
        cax = ax.inset_axes([1.02, 0, 0.03, 1])
        fig.colorbar(im, cax=cax).set_label("Count", fontsize=9, labelpad=8)
        ax.axvline(25, color="white", ls="--", lw=1, alpha=0.8, label="Δχ²=25")
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(left=0)
        ax.set_xlabel("DESI Δχ²", fontsize=11)
        ax.set_ylabel("DESI redshift z", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

        # ── Row 1: 1D Δχ² histogram, agree vs disagree ───────────────────
        ax2 = axes[1][ci]
        agree    = sub[sub["agree"]]["d_desi_deltachi2"]
        disagree = sub[~sub["agree"]]["d_desi_deltachi2"]
        ax2.hist(agree,    bins=dc2bins, color="steelblue", alpha=0.6,
                 label=f"{label} agree    (n={len(agree):,})")
        ax2.hist(disagree, bins=dc2bins, color="tomato",    alpha=0.6,
                 label=f"{label} disagree (n={len(disagree):,})")
        ax2.axvline(25, color="black", ls="--", lw=1, label="Δχ²=25")
        ax2.set_xlabel("DESI Δχ²", fontsize=11)
        ax2.set_ylabel("Count", fontsize=11)
        ax2.set_title("DESI Δχ² distribution", fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", alpha=0.3)

        # ── Row 2: 1D zerr histogram, agree vs disagree ──────────────────
        ax3 = axes[2][ci]
        sub_zerr = sub.dropna(subset=["d_desi_zerr"])
        agree_zerr    = sub_zerr[sub_zerr["agree"]]["d_desi_zerr"]
        disagree_zerr = sub_zerr[~sub_zerr["agree"]]["d_desi_zerr"]
        zerr_all = sub_zerr["d_desi_zerr"]
        zerr_bins = np.logspace(np.log10(max(zerr_all[zerr_all > 0].min(), 1e-6)),
                                np.log10(zerr_all.quantile(0.99)), 60)
        ax3.hist(agree_zerr[agree_zerr > 0],       bins=zerr_bins, color="steelblue", alpha=0.6,
                 label=f"{label} agree    (n={len(agree_zerr):,})")
        ax3.hist(disagree_zerr[disagree_zerr > 0], bins=zerr_bins, color="tomato",    alpha=0.6,
                 label=f"{label} disagree (n={len(disagree_zerr):,})")
        ax3.set_xscale("log")
        ax3.set_xlabel("DESI redshift error zerr (log scale)", fontsize=11)
        ax3.set_ylabel("Count", fontsize=11)
        ax3.set_title("DESI zerr distribution", fontsize=11)
        ax3.legend(fontsize=9)
        ax3.grid(axis="y", alpha=0.3)

    out = output_dir / "desi_z_dc2_2d.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def main():
    username = os.environ.get("PGUSER")
    if not username:
        print("ERROR: set PGUSER"); sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Querying data...")
    df_q1  = query_q1(username)
    df_ero = query_ero(username)

    print("\nPlotting...")
    plot(df_q1, df_ero, OUTPUT_DIR)


if __name__ == "__main__":
    main()
