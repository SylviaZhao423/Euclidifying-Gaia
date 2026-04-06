#!/usr/bin/env python3
"""
ERO class_star selection effects + Δmag overlay, probed with DESI spectroscopic galaxies.

Every source in crossmatch_ero_gaia_0_3_desi is a DESI-confirmed galaxy.
Focuses on cases where class_star >= 0.65 (ERO thinks it's a star) — what
makes these different from correctly-classified galaxies?

Outputs:
  output/desi_ero/desi_ero_bar.png
  output/desi_ero/ero_dmag_desi_overlay.png
  output/desi_ero/disagree_desi_gal_ero_star.csv
  output/desi_ero/agree_both_galaxy.csv

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_ero_desi.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR  = PROJECT_DIR / "output" / "desi_ero"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

CLASS_STAR_THRESH = 0.65

LABEL_COLORS = {"GALAXY": "steelblue", "STAR": "tomato"}


# ── Queries ───────────────────────────────────────────────────────────────────

def query(username):
    """Bar-plot query: DESI crossmatch only, classification columns."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    sql = """
        SELECT
            e_v_alpha_j2000          AS ra,
            e_v_delta_j2000          AS dec,
            e_v_class_star,
            e_v_mag_auto             AS vis_mag,
            e_v_spread_model,
            e_v_kron_radius,
            e_v_fwhm_image,
            g_corrected_bp_rp_excess AS corrected_bp_rp_excess,
            g_bp_g                   AS bp_g,
            g_g_rp                   AS g_rp,
            d_desi_z,
            d_desi_zerr,
            d_desi_deltachi2,
            d_desi_program,
            d_desi_spectype
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3_desi
        WHERE d_desi_z IS NOT NULL
    """
    print("  Querying crossmatch_ero_gaia_0_3_desi ...")
    df = pd.read_sql_query(sql, conn)
    conn.close()
    print(f"  -> {len(df):,} rows")
    return df


def query_ero_bg(username):
    """Full ERO catalogue for Δmag background density."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    sql = """
        SELECT
            e_v_mag_auto               AS v_mag_auto,
            e_v_class_star             AS class_star,
            g_corrected_bp_rp_excess   AS c_star,
            g_phot_g_mean_mag          AS phot_g_mean_mag
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_class_star IS NOT NULL
    """
    print("  Querying ERO crossmatch (background)...")
    df = pd.read_sql_query(sql, conn)
    conn.close()
    print(f"  -> {len(df):,} sources")
    return df


def query_desi_dmag(username):
    """DESI crossmatch with columns needed for the Δmag overlay plot."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    sql = """
        SELECT
            e_v_mag_auto               AS v_mag_auto,
            e_v_class_star             AS class_star,
            g_corrected_bp_rp_excess   AS c_star,
            g_phot_g_mean_mag          AS phot_g_mean_mag,
            d_desi_spectype,
            d_desi_z,
            d_desi_deltachi2
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3_desi
        WHERE d_desi_z IS NOT NULL
    """
    print("  Querying ERO-DESI crossmatch (Δmag overlay)...")
    df = pd.read_sql_query(sql, conn)
    conn.close()
    print(f"  -> {len(df):,} sources")
    return df


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_bar(df, output_dir):
    df["ero_label"]   = np.where(df["e_v_class_star"] >= CLASS_STAR_THRESH, "STAR", "GALAXY")
    df["ero_correct"] = df["ero_label"] == "GALAXY"

    n_total = len(df)
    counts  = df["ero_label"].value_counts()
    correct = df[df["ero_correct"]]
    wrong   = df[~df["ero_correct"]]

    print("\nERO class_star breakdown (all sources are DESI-confirmed galaxies):")
    for label, n in counts.items():
        print(f"  {label:10s}: {n:5,}  ({100*n/n_total:.1f}%)")

    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # fig.suptitle(
    #     f"ERO class_star classification of DESI-confirmed galaxies\n"
    #     f"n = {n_total:,} total  |  {len(wrong):,} misclassified as STAR "
    #     f"({100*len(wrong)/n_total:.1f}%)",
    #     fontsize=12, fontweight="bold",
    # )

    # # Panel 1: ERO class_star breakdown
    # labels_ordered = ["GALAXY", "STAR"]
    # bar_counts = [counts.get(l, 0) for l in labels_ordered]
    # bars = ax.bar(labels_ordered, bar_counts,
    #               color=[LABEL_COLORS[l] for l in labels_ordered],
    #               edgecolor="white", linewidth=0.5)
    # for bar, n in zip(bars, bar_counts):
    #     if n > 0:
    #         ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
    #                 f"{100*n/n_total:.1f}%  (n={n:,})", ha="center", va="bottom",
    #                 fontsize=10, fontweight="bold")
    # ax.set_ylabel("Count", fontsize=11)
    # ax.set_title("ERO class_star classification", fontsize=11)
    # ax.grid(axis="y", alpha=0.3)
    # ax.set_ylim(0, max(bar_counts) * 1.2)

    # plt.tight_layout()
    # out = output_dir / "desi_ero_bar.png"
    # plt.savefig(out, dpi=150, bbox_inches="tight")
    # plt.close()
    # print(f"  Saved: {out}")

    return df  # with ero_label column added


def make_dmag_plot(ero, desi, output_dir):
    """3×2 Δmag overlay: Row 0 = z, Row 1 = C*, Row 2 = Δχ²."""
    for df in (ero, desi):
        df['mag_diff'] = df['v_mag_auto'] - df['phot_g_mean_mag']

    xbins = np.arange(6,  23.1, 0.1)
    ybins = np.arange(-6.5, 7.6, 0.1)

    cs_splits = [
        (ero[ero['class_star'] <  0.65], desi[desi['class_star'] <  0.65], 'VIS galaxy (cs<0.65)'),
        (ero[ero['class_star'] >= 0.65], desi[desi['class_star'] >= 0.65], 'VIS star (cs≥0.65)'),
    ]

    fig, axes_arr = plt.subplots(3, 2, figsize=(15, 13))
    axes = [list(row) for row in axes_arr]
    fig.subplots_adjust(hspace=0.42, wspace=0.32)

    # Shared density colour scale
    vmaxes, histograms = [], []
    for ero_sub, _, _ in cs_splits:
        ero_clean = ero_sub.dropna(subset=['phot_g_mean_mag', 'mag_diff'])
        h, xe, ye = np.histogram2d(ero_clean['phot_g_mean_mag'].values,
                                   ero_clean['mag_diff'].values, bins=[xbins, ybins])
        hm = np.ma.masked_where(h == 0, h)
        if hm.count() > 0:
            vmaxes.append(hm.max())
        histograms.append((h, xe, ye))
    global_vmax = max(max(vmaxes, default=10), 2)

    for ci, ((ero_sub, desi_sub, title), (h, xe, ye)) in enumerate(zip(cs_splits, histograms)):
        ero_clean  = ero_sub.dropna(subset=['phot_g_mean_mag', 'mag_diff'])
        desi_clean = desi_sub.dropna(subset=['phot_g_mean_mag', 'mag_diff'])
        hm = np.ma.masked_where(h == 0, h)

        def _base(ax):
            ax.pcolormesh(xe, ye, hm.T, cmap='Greys',
                          norm=LogNorm(vmin=1, vmax=global_vmax), zorder=1)
            ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.6, zorder=2)
            ax.set_xlim(6, 23); ax.set_ylim(-6.5, 5.5)
            ax.set_xlabel('Gaia G', fontsize=11)
            if ci == 0:
                ax.set_ylabel('Δmag  (VIS − Gaia G)', fontsize=11)
            ax.grid(alpha=0.3)

        # Row 0: coloured by DESI redshift
        ax = axes[0][ci]
        _base(ax)
        desi_z = desi_clean.dropna(subset=['d_desi_z'])
        if len(desi_z):
            sc = ax.scatter(desi_z['phot_g_mean_mag'], desi_z['mag_diff'],
                            c=desi_z['d_desi_z'], cmap='plasma',
                            vmin=0, vmax=1.5, s=18, alpha=0.85,
                            linewidths=0.3, edgecolors='white', zorder=3,
                            label=f'DESI GALAXY (n={len(desi_z):,})')
            cax = ax.inset_axes([1.02, 0, 0.03, 1])
            fig.colorbar(sc, cax=cax).set_label('DESI redshift z', fontsize=8, labelpad=8)
        ax.set_title(f'{title}  —  coloured by DESI redshift\n'
                     f'n={len(ero_clean):,} ERO  |  {len(desi_clean):,} DESI',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left', markerscale=1.5)

        # Row 1: coloured by C*
        ax = axes[1][ci]
        _base(ax)
        desi_cs = desi_clean.dropna(subset=['c_star'])
        if len(desi_cs):
            sc = ax.scatter(desi_cs['phot_g_mean_mag'], desi_cs['mag_diff'],
                            c=desi_cs['c_star'], cmap='RdYlBu_r',
                            vmin=-1, vmax=1, s=18, alpha=0.85,
                            linewidths=0.3, edgecolors='white', zorder=3,
                            label=f'DESI GALAXY (n={len(desi_cs):,})')
            cax = ax.inset_axes([1.02, 0, 0.03, 1])
            fig.colorbar(sc, cax=cax).set_label('C* (corrected BP-RP excess)', fontsize=8, labelpad=8)
        ax.set_title(f'{title}  —  coloured by C*', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left', markerscale=1.5)

        # Row 2: coloured by Δχ²
        ax = axes[2][ci]
        _base(ax)
        desi_dc2 = desi_clean.dropna(subset=['d_desi_deltachi2'])
        if len(desi_dc2):
            dc2_vals = desi_dc2['d_desi_deltachi2'].clip(lower=1)
            vmax_dc2 = max(float(dc2_vals.quantile(0.99)), 2)
            sc = ax.scatter(desi_dc2['phot_g_mean_mag'], desi_dc2['mag_diff'],
                            c=dc2_vals, cmap='viridis',
                            norm=LogNorm(vmin=1, vmax=vmax_dc2),
                            s=18, alpha=0.85,
                            linewidths=0.3, edgecolors='white', zorder=3,
                            label=f'DESI GALAXY (n={len(desi_dc2):,})')
            cax = ax.inset_axes([1.02, 0, 0.03, 1])
            fig.colorbar(sc, cax=cax).set_label('DESI Δχ²', fontsize=8, labelpad=8)
        ax.set_title(f'{title}  —  coloured by DESI Δχ²', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left', markerscale=1.5)

    fig.suptitle(
        'ERO Δmag distributions with DESI spectroscopic labels overlaid',
        fontsize=13, fontweight='bold', y=0.95,
    )
    out = output_dir / 'ero_dmag_desi_overlay.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csvs(df: pd.DataFrame, output_dir: Path) -> None:
    """Export failure and correct CSVs with coordinates for cutout downloads."""
    export_cols = [
        "ra", "dec", "ero_label", "e_v_class_star",
        "vis_mag", "e_v_spread_model", "e_v_kron_radius", "e_v_fwhm_image",
        "corrected_bp_rp_excess", "bp_g", "g_rp",
        "d_desi_z", "d_desi_zerr", "d_desi_deltachi2", "d_desi_program",
    ]

    # Failures: ERO says STAR, DESI says GALAXY — sorted by class_star descending
    fail = df[df["ero_label"] == "STAR"].copy()
    fail = fail.sort_values("e_v_class_star", ascending=False)
    out = output_dir / "disagree_desi_gal_ero_star.csv"
    fail[[c for c in export_cols if c in fail.columns]].to_csv(out, index=False)
    print(f"  Saved: {out}  (n={len(fail):,})")

    # Correct: ERO says GALAXY, DESI says GALAXY — sorted by class_star ascending
    correct = df[df["ero_label"] == "GALAXY"].copy()
    correct = correct.sort_values("e_v_class_star", ascending=True)
    out = output_dir / "agree_both_galaxy.csv"
    correct[[c for c in export_cols if c in correct.columns]].to_csv(out, index=False)
    print(f"  Saved: {out}  (n={len(correct):,})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    username = os.environ.get("PGUSER")
    if not username:
        print("ERROR: set PGUSER"); sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Bar plot + CSVs ────────────────────────────────────────────────
    print("\n[1] Querying DESI crossmatch...")
    df = query(username)

    print("\n[2] Plotting bar chart and exporting CSVs...")
    df = plot_bar(df, OUTPUT_DIR)
    export_csvs(df, OUTPUT_DIR)

    # ── 2. Δmag overlay plot ──────────────────────────────────────────────
    print("\n[3] Querying data for Δmag overlay...")
    ero  = query_ero_bg(username)
    desi = query_desi_dmag(username)

    print("\n[4] Plotting Δmag overlay...")
    make_dmag_plot(ero, desi, OUTPUT_DIR)


if __name__ == "__main__":
    main()
