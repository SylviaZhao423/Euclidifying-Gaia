#!/usr/bin/env python3
"""
PHZ selection effects + Δmag overlay, probed with DESI spectroscopic galaxies.

Every source in crossmatch_q1_mer_phz_desi is a DESI-confirmed galaxy.
Focuses on the ~7% PHZ gets wrong — what makes them different?

Outputs:
  output/desi_q1/desi_q1_bar.png
  output/desi_q1/q1_dmag_desi_overlay.png
  output/desi_q1/disagree_desi_gal_q1_star.csv
  output/desi_q1/disagree_desi_gal_q1_qso.csv
  output/desi_q1/disagree_desi_gal_q1_unclassified.csv
  output/desi_q1/disagree_desi_gal_q1_cl.csv
  output/desi_q1/agree_both_galaxy.csv

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_q1_desi.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR  = PROJECT_DIR / "output" / "desi_q1"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

PHZ_COLORS = {"GALAXY": "steelblue", "STAR": "tomato", "QSO": "goldenrod",
              "GLOB_CL": "mediumseagreen", "UNCLASSIFIED": "silver"}

ZPC_IE = 1.05
ZP_REF = 23.9


# ── Queries ───────────────────────────────────────────────────────────────────

def query(username):
    """Bar-plot query: DESI crossmatch only, classification columns."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    sql = """
        SELECT
            m_right_ascension       AS ra,
            m_declination           AS dec,
            p_phz_classification,
            p_phz_gal_prob,
            p_phz_star_prob,
            p_phz_qso_prob,
            d_desi_z,
            d_desi_zerr,
            d_desi_deltachi2,
            d_desi_program,
            d_desi_spectype,
            m_flux_vis_2fwhm_aper,
            m_flux_detection_total,
            m_extended_prob,
            m_kron_radius
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_desi
        WHERE d_desi_z IS NOT NULL
    """
    print("  Querying crossmatch_q1_mer_phz_desi ...")
    df = pd.read_sql_query(sql, conn)
    conn.close()
    print(f"  -> {len(df):,} rows")
    return df


def query_q1_bg(username):
    """Full Q1×Gaia catalogue for Δmag background density."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    sql = f"""
        SELECT
            -2.5 * LOG10({ZPC_IE} * c.m_flux_detection_total) + {ZP_REF} AS ie_mag,
            c.p_phz_classification  AS phz_class,
            c.g_phot_g_mean_mag     AS gaia_g,
            c.g_corrected_bp_rp_excess AS c_star
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia c
        WHERE c.m_flux_detection_total > 0
          AND c.g_phot_g_mean_mag IS NOT NULL
    """
    print("  Querying Q1×Gaia crossmatch (background)...")
    df = pd.read_sql_query(sql, conn)
    conn.close()
    print(f"  -> {len(df):,} sources")
    return df


def query_desi_dmag(username):
    """DESI crossmatch joined with Gaia background to get g_phot_g_mean_mag."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    sql = f"""
        SELECT
            -2.5 * LOG10({ZPC_IE} * d.m_flux_detection_total) + {ZP_REF} AS ie_mag,
            d.p_phz_classification    AS phz_class,
            g.g_phot_g_mean_mag       AS gaia_g,
            g.g_corrected_bp_rp_excess AS c_star,
            d.d_desi_z,
            d.d_desi_deltachi2
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_desi d
        JOIN siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia g
          ON d.m_right_ascension = g.m_right_ascension
         AND d.m_declination     = g.m_declination
        WHERE d.m_flux_detection_total > 0
          AND d.d_desi_z IS NOT NULL
          AND g.g_phot_g_mean_mag IS NOT NULL
    """
    print("  Querying Q1-DESI crossmatch joined with Gaia (Δmag overlay)...")
    df = pd.read_sql_query(sql, conn)
    conn.close()
    print(f"  -> {len(df):,} sources")
    return df


# ── Plots ─────────────────────────────────────────────────────────────────────

def vis_mag(flux):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(flux > 0, -2.5 * np.log10(flux) + 31.4, np.nan)


def plot_bar(df, output_dir):
    df["phz_label"]   = df["p_phz_classification"].map(
        {1: "STAR", 2: "GALAXY", 3: "QSO", 4: "GLOB_CL"}
    ).fillna("UNCLASSIFIED")
    df["phz_correct"] = df["phz_label"] == "GALAXY"
    df["vis_mag"]     = vis_mag(df["m_flux_vis_2fwhm_aper"])

    n_total = len(df)
    counts  = df["phz_label"].value_counts()
    correct = df[df["phz_correct"]]
    wrong   = df[~df["phz_correct"]]

    print("\nPHZ classification breakdown (all sources are DESI-confirmed galaxies):")
    for label, n in counts.items():
        print(f"  {label:15s}: {n:6,}  ({100*n/n_total:.1f}%)")

    # fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    # fig.suptitle(
    #     f"PHZ classification of DESI-confirmed galaxies\n"
    #     f"n = {n_total:,} total  |  {len(wrong):,} misclassified ({100*len(wrong)/n_total:.1f}%)",
    #     fontsize=12, fontweight="bold",
    # )

    # # Panel 1: PHZ classification breakdown
    # labels_ordered = ["GALAXY", "STAR", "QSO", "GLOB_CL", "UNCLASSIFIED"]
    # bar_counts = [counts.get(l, 0) for l in labels_ordered]
    # bars = ax.bar(labels_ordered, bar_counts,
    #               color=[PHZ_COLORS[l] for l in labels_ordered],
    #               edgecolor="white", linewidth=0.5)
    # for bar, n in zip(bars, bar_counts):
    #     if n > 0:
    #         ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
    #                 f"{100*n/n_total:.1f}%  (n={n:,})", ha="center", va="bottom",
    #                 fontsize=9, fontweight="bold")
    # ax.set_ylabel("Count", fontsize=11)
    # ax.set_title("PHZ classification", fontsize=11)
    # ax.grid(axis="y", alpha=0.3)
    # ax.set_ylim(0, max(bar_counts) * 1.15)

    # plt.tight_layout()
    # out = output_dir / "desi_q1_bar.png"
    # plt.savefig(out, dpi=150, bbox_inches="tight")
    # plt.close()
    # print(f"  Saved: {out}")

    return df  # with phz_label column added


def make_dmag_plot(q1, desi, output_dir):
    """3×2 Δmag overlay: Row 0 = z, Row 1 = C*, Row 2 = Δχ²."""
    for df in (q1, desi):
        df['mag_diff'] = df['ie_mag'] - df['gaia_g']

    xbins = np.arange(6,  23.1, 0.1)
    ybins = np.arange(-6.5, 7.6, 0.1)

    phz_splits = [
        (q1[q1['phz_class'] == 2],   desi[desi['phz_class'] == 2],   'PHZ galaxy (phz=2)'),
        (q1[q1['phz_class'] != 2],   desi[desi['phz_class'] != 2],   'PHZ non-galaxy (phz≠2)'),
    ]

    fig, axes_arr = plt.subplots(3, 2, figsize=(15, 13))
    axes = [list(row) for row in axes_arr]
    fig.subplots_adjust(hspace=0.42, wspace=0.32)

    # Shared density colour scale
    vmaxes, histograms = [], []
    for q1_sub, _, _ in phz_splits:
        q1_clean = q1_sub.dropna(subset=['gaia_g', 'mag_diff'])
        h, xe, ye = np.histogram2d(q1_clean['gaia_g'].values,
                                   q1_clean['mag_diff'].values, bins=[xbins, ybins])
        hm = np.ma.masked_where(h == 0, h)
        if hm.count() > 0:
            vmaxes.append(hm.max())
        histograms.append((h, xe, ye))
    global_vmax = max(max(vmaxes, default=10), 2)

    for ci, ((q1_sub, desi_sub, title), (h, xe, ye)) in enumerate(zip(phz_splits, histograms)):
        q1_clean   = q1_sub.dropna(subset=['gaia_g', 'mag_diff'])
        desi_clean = desi_sub.dropna(subset=['gaia_g', 'mag_diff'])
        hm = np.ma.masked_where(h == 0, h)

        def _base(ax):
            ax.pcolormesh(xe, ye, hm.T, cmap='Greys',
                          norm=LogNorm(vmin=1, vmax=global_vmax), zorder=1)
            ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.6, zorder=2)
            ax.set_xlim(6, 23); ax.set_ylim(-6.5, 5.5)
            ax.set_xlabel('Gaia G', fontsize=11)
            if ci == 0:
                ax.set_ylabel('Δmag  (I_E − Gaia G)', fontsize=11)
            ax.grid(alpha=0.3)

        # Row 0: coloured by DESI redshift
        ax = axes[0][ci]
        _base(ax)
        desi_z = desi_clean.dropna(subset=['d_desi_z'])
        if len(desi_z):
            sc = ax.scatter(desi_z['gaia_g'], desi_z['mag_diff'],
                            c=desi_z['d_desi_z'], cmap='plasma',
                            vmin=0, vmax=1.5, s=18, alpha=0.85,
                            linewidths=0.3, edgecolors='white', zorder=3,
                            label=f'DESI GALAXY (n={len(desi_z):,})')
            cax = ax.inset_axes([1.02, 0, 0.03, 1])
            fig.colorbar(sc, cax=cax).set_label('DESI redshift z', fontsize=8, labelpad=8)
        ax.set_title(f'{title}  —  coloured by DESI redshift\n'
                     f'n={len(q1_clean):,} Q1  |  {len(desi_clean):,} DESI',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left', markerscale=1.5)

        # Row 1: coloured by C*
        ax = axes[1][ci]
        _base(ax)
        desi_cs = desi_clean.dropna(subset=['c_star'])
        if len(desi_cs):
            sc = ax.scatter(desi_cs['gaia_g'], desi_cs['mag_diff'],
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
            sc = ax.scatter(desi_dc2['gaia_g'], desi_dc2['mag_diff'],
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
        'Q1 Δmag distributions with DESI spectroscopic labels overlaid',
        fontsize=13, fontweight='bold', y=0.95,
    )
    out = output_dir / 'q1_dmag_desi_overlay.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ── CSV export ────────────────────────────────────────────────────────────────

def export_failure_csvs(df: pd.DataFrame, output_dir: Path) -> None:
    """Export one CSV per PHZ failure mode with coordinates for cutout downloads."""
    export_cols = [
        "ra", "dec", "phz_label", "p_phz_gal_prob", "p_phz_star_prob", "p_phz_qso_prob",
        "d_desi_z", "d_desi_zerr", "d_desi_deltachi2", "d_desi_program",
        "vis_mag", "m_flux_detection_total", "m_extended_prob", "m_kron_radius",
    ]
    label_to_file = {
        "STAR":         "disagree_desi_gal_q1_star.csv",
        "QSO":          "disagree_desi_gal_q1_qso.csv",
        "UNCLASSIFIED": "disagree_desi_gal_q1_unclassified.csv",
        "GLOB_CL":      "disagree_desi_gal_q1_cl.csv",
    }
    for label, fname in label_to_file.items():
        sub = df[df["phz_label"] == label].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values("p_phz_gal_prob", ascending=True)
        out = output_dir / fname
        sub[[c for c in export_cols if c in sub.columns]].to_csv(out, index=False)
        print(f"  Saved: {out}  (n={len(sub):,})")

    agree = df[df["phz_label"] == "GALAXY"].copy()
    agree = agree.sort_values("p_phz_gal_prob", ascending=False)
    out = output_dir / "agree_both_galaxy.csv"
    agree[[c for c in export_cols if c in agree.columns]].to_csv(out, index=False)
    print(f"  Saved: {out}  (n={len(agree):,})")


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
    export_failure_csvs(df, OUTPUT_DIR)

    # ── 2. Δmag overlay plot ──────────────────────────────────────────────
    print("\n[3] Querying data for Δmag overlay...")
    q1   = query_q1_bg(username)
    desi = query_desi_dmag(username)

    print("\n[4] Plotting Δmag overlay...")
    make_dmag_plot(q1, desi, OUTPUT_DIR)


if __name__ == "__main__":
    main()
