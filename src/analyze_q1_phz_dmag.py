#!/usr/bin/env python3
"""
mAB − Gaia G magnitude difference vs p_phz_star_prob and PHZ label — produces:
  1. q1_mag_diff_vs_star_prob.png  — Δmag vs p_phz_star_prob 2D hist in G magnitude slices
  2. q1_dmag_vs_gmag_by_phz.png   — Δmag vs Gaia G split by PHZ label (galaxy/star/unclassified)
  3. q1_dmag_xgb.png       — Q1 + Combined XGBoost probs split by Δmag sign (PHZ galaxy/star)

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_q1_phz_dmag.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR      = PROJECT_DIR / 'output'
MODEL_Q1_PATH   = PROJECT_DIR / 'output' / 'xgboost_combined' / 'xgb_classifier_q1.json'
MODEL_COMB_PATH = PROJECT_DIR / 'output' / 'xgboost_combined' / 'xgb_classifier_combined.json'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

ZPC_IE = 1.05
ZP_REF = 23.9
STAR_PROB_CUT = 0.5


# --- Data query ---------------------------------------------------------------

def query_data(username):
    """Query Gaia-matched Q1+PHZ sources with mAB, Gaia G, star_prob, phz_class, and Gaia features."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_feat_cols = ', '.join([f'c.g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT
            -2.5 * LOG10({ZPC_IE} * c.m_flux_detection_total) + {ZP_REF} AS ie_mag,
            c.g_phot_g_mean_mag      AS gaia_g,
            c.p_phz_star_prob        AS star_prob,
            c.p_phz_classification   AS phz_class,
            {gaia_feat_cols}
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia c
        WHERE c.m_flux_detection_total > 0
          AND c.m_flux_detection_total IS NOT NULL
          AND c.m_vis_det = 1
          AND c.g_phot_g_mean_mag IS NOT NULL
          AND c.p_phz_star_prob IS NOT NULL
    """
    print("  Querying Q1 PHZ + Gaia...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['mag_diff'] = df['ie_mag'] - df['gaia_g']
    print(f"  Retrieved {len(df):,} sources")
    return df


# --- Plots --------------------------------------------------------------------

def plot_mag_diff_vs_star_prob(df, output_dir):
    """1×5 row: Δmag vs p_phz_star_prob 2D histogram in G magnitude slices.
    Mirrors mag_diff_vs_class_star_combined.png from analyze_ero_class_star_neighbours.py.
    """
    base = df[df['mag_diff'].notna() & df['star_prob'].notna()
              & df['gaia_g'].notna()].copy()

    xbins = np.arange(0, 1.01, 0.01)
    ybins = np.arange(-2.5, 1.51, 0.05)

    sub_panels = [
        (base[base['gaia_g'] > 15.0],                    'All G > 15'),
        (base[base['gaia_g'].between(17.5, 18.5)],       '17.5 ≤ G < 18.5'),
        (base[base['gaia_g'].between(18.5, 19.5)],       '18.5 ≤ G < 19.5'),
        (base[base['gaia_g'].between(19.5, 21.0)],       '19.5 ≤ G < 21.0'),
        (base[base['gaia_g'] > 21.0],                    'G > 21'),
    ]

    histograms = []
    for panel_df, _ in sub_panels:
        h, xe, ye = np.histogram2d(panel_df['star_prob'].values,
                                   panel_df['mag_diff'].values,
                                   bins=[xbins, ybins])
        histograms.append((h, xe, ye))

    global_vmax = max(
        (np.ma.masked_where(h == 0, h).max()
         for h, _, _ in histograms
         if np.ma.masked_where(h == 0, h).count() > 0),
        default=10)
    global_vmax = max(2, global_vmax)

    fig, axes = plt.subplots(1, 5, figsize=(40, 7), sharey=True)
    for j, (ax, (panel_df, label), (h, xe, ye)) in enumerate(
            zip(axes, sub_panels, histograms)):
        n_gal  = (panel_df['star_prob'] <  STAR_PROB_CUT).sum()
        n_star = (panel_df['star_prob'] >= STAR_PROB_CUT).sum()
        hm = np.ma.masked_where(h == 0, h)
        im = ax.pcolormesh(xe, ye, hm.T, cmap='Blues',
                           norm=LogNorm(vmin=1, vmax=global_vmax))
        ax.axvline(x=STAR_PROB_CUT, color='red',  ls='--', linewidth=1.5,
                   label=f'star_prob = {STAR_PROB_CUT}')
        ax.axhline(y=0,             color='gray', ls='--', linewidth=1.0, alpha=0.7,
                   label='Δmag = 0')
        fig.colorbar(im, ax=ax, pad=0.02).set_label('Count', fontsize=10)
        ax.set_xlabel('p_phz_star_prob', fontsize=12, fontweight='bold')
        if j == 0:
            ax.set_ylabel('mAB − phot_g_mean_mag', fontsize=11, fontweight='bold')
        ax.set_title(
            f'{label}\nN={len(panel_df):,}  (gal<{STAR_PROB_CUT}: {n_gal:,},  '
            f'star≥{STAR_PROB_CUT}: {n_star:,})',
            fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(-2.5, 1.5)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)

    fig.suptitle('mAB − Gaia G magnitude difference vs p_phz_star_prob  (Q1)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = output_dir / 'q1_mag_diff_vs_star_prob.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_dmag_vs_gmag_by_phz(df, output_dir):
    """2D histogram of Gaia G vs Δmag split by PHZ label (3 panels).
    Uses p_phz_classification: phz=2 (galaxy), phz=1 (star), unclassified (other/NULL).
    Mirrors dmag_vs_gmag_by_class_star.png from analyze_ero_class_star_neighbours.py.
    """
    base = df[df['mag_diff'].notna() & df['gaia_g'].notna()].copy()

    xbins = np.arange(14, 22.1, 0.1)
    ybins = np.arange(-2.5, 1.51, 0.05)

    panels = [
        (base[base['phz_class'] == 2],          'PHZ galaxy (phz=2)'),
        (base[base['phz_class'] == 1],          'PHZ star   (phz=1)'),
        (base[~base['phz_class'].isin([1, 2])], 'Unclassified (phz≠1,2 or NULL)'),
    ]

    vmaxes, histograms = [], []
    for sub, _ in panels:
        h, xe, ye = np.histogram2d(sub['gaia_g'].values, sub['mag_diff'].values,
                                   bins=[xbins, ybins])
        hm = np.ma.masked_where(h == 0, h)
        if hm.count() > 0:
            vmaxes.append(hm.max())
        histograms.append((h, xe, ye))
    global_vmax = max(max(vmaxes, default=10), 2)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True)
    for ax, (sub, label), (h, xe, ye) in zip(axes, panels, histograms):
        hm = np.ma.masked_where(h == 0, h)
        im = ax.pcolormesh(xe, ye, hm.T, cmap='Blues',
                           norm=LogNorm(vmin=1, vmax=global_vmax))
        ax.axhline(y=0, color='gray', ls='--', linewidth=1.0, alpha=0.7, label='Δmag = 0')
        fig.colorbar(im, ax=ax, pad=0.02).set_label('Count', fontsize=10)
        ax.set_xlabel('Gaia G', fontsize=12, fontweight='bold')
        ax.set_ylabel('mAB − phot_g_mean_mag', fontsize=11)
        ax.set_title(f'{label}\nN={len(sub):,}', fontsize=12, fontweight='bold')
        ax.set_xlim(14, 22)
        ax.set_ylim(-2.5, 1.5)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)

    plt.suptitle('mAB − Gaia G vs Gaia G — split by PHZ label  (Q1)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = output_dir / 'q1_dmag_vs_gmag_by_phz.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_xgb_on_q1_by_phz(df, output_dir):
    """2×3 grid: rows = PHZ galaxy (phz=2) / PHZ star (phz=1),
    cols = Gaia G magnitude histogram / Q1 model prob / Combined model prob,
    each split by sign of Δmag.
    Mirrors ero_galaxy_dmag_xgb.png from analyze_ero_class_star_dmag.py.
    """
    import xgboost as xgb

    base = df[df['mag_diff'].notna() & df['gaia_g'].notna()].copy()

    for col in SELECTED_GAIA_PARAMETERS:
        base[col] = pd.to_numeric(base[col], errors='coerce')

    X = base[SELECTED_GAIA_PARAMETERS].values

    q1_model = xgb.XGBClassifier()
    q1_model.load_model(str(MODEL_Q1_PATH))
    base['xgb_q1_prob'] = q1_model.predict_proba(X)[:, 1]

    comb_model = xgb.XGBClassifier()
    comb_model.load_model(str(MODEL_COMB_PATH))
    base['xgb_comb_prob'] = comb_model.predict_proba(X)[:, 1]

    rows = [
        (base[base['phz_class'] == 2].copy(), 'PHZ galaxy (phz=2)'),
        (base[base['phz_class'] == 1].copy(), 'PHZ star   (phz=1)'),
    ]

    mag_bins  = np.arange(14, 22.1, 0.1)
    prob_bins = np.arange(0, 1.01, 0.02)

    fig, axes = plt.subplots(2, 3, figsize=(21, 10))

    for ri, (sub, row_label) in enumerate(rows):
        neg = sub[sub['mag_diff'] < 0]
        pos = sub[sub['mag_diff'] > 0]

        ax = axes[ri, 0]
        ax.hist(neg['gaia_g'], bins=mag_bins, alpha=0.6, color='steelblue',
                label=f'Δmag < 0 (mAB brighter)  N={len(neg):,}')
        ax.hist(pos['gaia_g'], bins=mag_bins, alpha=0.6, color='coral',
                label=f'Δmag > 0 (mAB fainter)   N={len(pos):,}')
        ax.set_xlabel('Gaia G', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'G magnitude — {row_label}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        if ri == 0:
            q1_vals   = neg['xgb_q1_prob'],       pos['xgb_q1_prob']
            comb_vals = neg['xgb_comb_prob'],      pos['xgb_comb_prob']
            lbl_q1    = 'Q1 model galaxy probability'
            lbl_comb  = 'Combined model galaxy probability'
        else:
            q1_vals   = 1 - neg['xgb_q1_prob'],   1 - pos['xgb_q1_prob']
            comb_vals = 1 - neg['xgb_comb_prob'],  1 - pos['xgb_comb_prob']
            lbl_q1    = 'Q1 model star probability (1 − galaxy prob)'
            lbl_comb  = 'Combined model star probability (1 − galaxy prob)'

        for col_idx, (vals, xlabel) in enumerate([(q1_vals, lbl_q1), (comb_vals, lbl_comb)], 1):
            ax = axes[ri, col_idx]
            ax.hist(vals[0], bins=prob_bins, alpha=0.6, color='steelblue',
                    label=f'Δmag < 0  N={len(neg):,}')
            ax.hist(vals[1], bins=prob_bins, alpha=0.6, color='coral',
                    label=f'Δmag > 0  N={len(pos):,}')
            ax.axvline(x=1, color='red', ls='--', linewidth=1.5, label='prob = 1')
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            model_name = 'Q1' if col_idx == 1 else 'Combined'
            ax.set_title(f'{model_name} model — {row_label}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

    plt.suptitle('Q1 sources split by Δmag sign — Q1 vs Combined XGBoost model (Gaia features only)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = output_dir / 'q1_dmag_xgb.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# --- Main ---------------------------------------------------------------------

def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Q1 mAB − Gaia G Δmag vs p_phz_star_prob, PHZ label, and XGBoost")
    print("=" * 70)

    print("\n[1] Querying data...")
    df = query_data(username)

    print("\n[2] Generating mAB−Gaia Δmag vs p_phz_star_prob in magnitude slices...")
    plot_mag_diff_vs_star_prob(df, OUTPUT_DIR)

    print("\n[3] Generating mAB−Gaia Δmag vs Gaia G split by PHZ label...")
    plot_dmag_vs_gmag_by_phz(df, OUTPUT_DIR)

    print("\n[4] Applying Q1 + Combined XGBoost models by PHZ label and Δmag sign...")
    plot_xgb_on_q1_by_phz(df, OUTPUT_DIR)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
