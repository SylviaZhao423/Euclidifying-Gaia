#!/usr/bin/env python3
"""
VIS − Gaia G magnitude difference vs class_star and XGBoost model — produces:
  1. ero_mag_diff_vs_class_star.png  — Δmag vs class_star 2D hist in G magnitude slices
  2. ero_dmag_vs_gmag_by_class_star.png — Δmag vs Gaia G split by class_star < / >= 0.65
  3. ero_dmag_xgb.png        — Q1 + Combined XGBoost probs split by Δmag sign

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_ero_class_star_dmag.py
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
WSDB_DB   = 'wsdb'

ERO_FIELDS = {
    'Horsehead':    (85.150, -2.613,  0.58),
    'Messier 78':   (86.690, -0.015,  0.60),
    'Taurus':       (64.983, +28.023, 0.61),
    'Barnard 30':   (82.880, +12.316, 0.60),
    'NGC 6254':     (254.303, -4.100, 0.60),
    'NGC 6397':     (265.174, -53.658, 0.61),
    'NGC 6822':     (296.236, -14.788, 0.60),
    'Holmberg II':  (124.790, +70.706, 0.60),
    'IC 10':        (5.063, +59.288, 0.62),
    'IC 342':       (56.730, +68.084, 0.59),
    'NGC 2403':     (114.211, +65.586, 0.60),
    'NGC 6744':     (287.430, -63.842, 0.60),
    'Fornax':       (54.017, -35.267, 0.57),
    'Dorado':       (64.014, -55.780, 0.60),
    'Perseus':      (49.638, +41.651, 0.70),
    'Abell 2390':   (328.397, +17.709, 0.75),
    'Abell 2764':   (5.713, -49.249, 0.75),
}


def assign_fields(df):
    """Assign each source to its nearest ERO field (within field radius)."""
    ra  = df['ra'].values
    dec = df['dec'].values
    n_sources = len(df)

    field_assignments = np.full(n_sources, None, dtype=object)
    min_distances     = np.full(n_sources, np.inf)

    for field_name, (field_ra, field_dec, area) in ERO_FIELDS.items():
        radius = np.sqrt(area / np.pi)
        ra1  = np.radians(ra);  dec1 = np.radians(dec)
        ra2  = np.radians(field_ra); dec2 = np.radians(field_dec)
        a    = np.sin((dec2 - dec1) / 2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2)**2
        dist = np.degrees(2 * np.arcsin(np.sqrt(a)))
        mask = (dist < radius) & (dist < min_distances)
        field_assignments[mask] = field_name
        min_distances[mask]     = dist[mask]

    df['field'] = field_assignments
    return df


# --- Data query ---------------------------------------------------------------

def query_data(username):
    """Query ERO crossmatch with Gaia features for Δmag and XGBoost plots."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_feat_cols = ', '.join([f'c.g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT
            e_v_alpha_j2000 AS ra,
            e_v_delta_j2000 AS dec,
            e_v_mag_auto    AS v_mag_auto,
            e_v_class_star  AS class_star,
            {gaia_feat_cols}
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3 c
        WHERE e_v_class_star IS NOT NULL
    """
    print("  Querying ERO crossmatch + Gaia features...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} sources")
    return df


# --- Plots --------------------------------------------------------------------

def plot_mag_diff_vs_class_star_combined(df, output_dir):
    """1×5 row: Δmag (v_mag_auto − Gaia G) vs class_star 2D histogram in G slices.
    Mirrors q1_mag_diff_vs_star_prob.png from analyze_q1_phz_dmag.py.
    """
    base = df[df['phot_g_mean_mag'].notna()
              & df['v_mag_auto'].notna()
              & df['class_star'].notna()].copy()
    base['mag_diff'] = base['v_mag_auto'] - base['phot_g_mean_mag']

    xbins = np.arange(0, 1.01, 0.01)
    ybins = np.arange(-2.5, 1.51, 0.05)

    sub_panels = [
        (base[base['phot_g_mean_mag'] > 15.0],                    'All G > 15'),
        (base[base['phot_g_mean_mag'].between(17.5, 18.5)],       '17.5 ≤ G < 18.5'),
        (base[base['phot_g_mean_mag'].between(18.5, 19.5)],       '18.5 ≤ G < 19.5'),
        (base[base['phot_g_mean_mag'].between(19.5, 21.0)],       '19.5 ≤ G < 21.0'),
        (base[base['phot_g_mean_mag'] > 21.0],                    'G > 21'),
    ]

    histograms = []
    for panel_df, _ in sub_panels:
        h, xe, ye = np.histogram2d(panel_df['class_star'].values,
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
        n_gal  = (panel_df['class_star'] <  0.65).sum()
        n_star = (panel_df['class_star'] >= 0.65).sum()
        hm = np.ma.masked_where(h == 0, h)
        im = ax.pcolormesh(xe, ye, hm.T, cmap='Blues',
                           norm=LogNorm(vmin=1, vmax=global_vmax))
        ax.axvline(x=0.65, color='red',  ls='--', linewidth=1.5, label='class_star = 0.65')
        ax.axhline(y=0,    color='gray', ls='--', linewidth=1.0, alpha=0.7, label='Δmag = 0')
        fig.colorbar(im, ax=ax, pad=0.02).set_label('Count', fontsize=10)
        ax.set_xlabel('class_star', fontsize=12, fontweight='bold')
        if j == 0:
            ax.set_ylabel('v_mag_auto − phot_g_mean_mag', fontsize=11, fontweight='bold')
        ax.set_title(f'{label}\nN={len(panel_df):,}  (gal<0.65: {n_gal:,},  star≥0.65: {n_star:,})',
                     fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(-2.5, 1.5)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)

    fig.suptitle('VIS − Gaia G magnitude difference vs class_star  (ERO)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = output_dir / 'ero_mag_diff_vs_class_star.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_dmag_vs_gmag_galaxy(df, output_dir):
    """2D histogram of Gaia G vs Δmag split by class_star < / >= 0.65.
    Mirrors q1_dmag_vs_gmag_by_phz.png from analyze_q1_phz_dmag.py.
    """
    base = df[df['phot_g_mean_mag'].notna()
              & df['v_mag_auto'].notna()
              & df['class_star'].notna()].copy()
    base['mag_diff'] = base['v_mag_auto'] - base['phot_g_mean_mag']

    xbins = np.arange(14, 22.1, 0.1)
    ybins = np.arange(-2.5, 1.51, 0.05)

    panels = [
        (base[base['class_star'] <  0.65], 'class_star < 0.65  (VIS galaxy)'),
        (base[base['class_star'] >= 0.65], 'class_star ≥ 0.65  (VIS star)'),
    ]

    vmaxes, histograms = [], []
    for sub, _ in panels:
        h, xe, ye = np.histogram2d(sub['phot_g_mean_mag'].values,
                                   sub['mag_diff'].values, bins=[xbins, ybins])
        hm = np.ma.masked_where(h == 0, h)
        if hm.count() > 0:
            vmaxes.append(hm.max())
        histograms.append((h, xe, ye))
    global_vmax = max(max(vmaxes, default=10), 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, (sub, label), (h, xe, ye) in zip(axes, panels, histograms):
        hm = np.ma.masked_where(h == 0, h)
        im = ax.pcolormesh(xe, ye, hm.T, cmap='Blues',
                           norm=LogNorm(vmin=1, vmax=global_vmax))
        ax.axhline(y=0, color='gray', ls='--', linewidth=1.0, alpha=0.7, label='Δmag = 0')
        fig.colorbar(im, ax=ax, pad=0.02).set_label('Count', fontsize=10)
        ax.set_xlabel('Gaia G', fontsize=12, fontweight='bold')
        ax.set_ylabel('v_mag_auto − phot_g_mean_mag', fontsize=11)
        ax.set_title(f'{label}\nN={len(sub):,}', fontsize=12, fontweight='bold')
        ax.set_xlim(14, 22)
        ax.set_ylim(-2.5, 1.5)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)

    plt.suptitle('Δmag vs Gaia G — split by class_star  (ERO)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = output_dir / 'ero_dmag_vs_gmag_by_class_star.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_xgb_q1_on_ero_galaxies(df, output_dir):
    """2×3 grid: rows = class_star < 0.65 / >= 0.65,
    cols = G magnitude histogram / Q1 model prob / Combined model prob,
    each split by sign of Δmag.
    """
    import xgboost as xgb

    base = df[df['phot_g_mean_mag'].notna()
              & df['v_mag_auto'].notna()
              & df['class_star'].notna()].copy()
    base['mag_diff'] = base['v_mag_auto'] - base['phot_g_mean_mag']

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
        (base[base['class_star'] <  0.65].copy(), 'class_star < 0.65  (VIS galaxy)'),
        (base[base['class_star'] >= 0.65].copy(), 'class_star ≥ 0.65  (VIS star)'),
    ]

    mag_bins  = np.arange(14, 22.1, 0.1)
    prob_bins = np.arange(0, 1.01, 0.02)

    fig, axes = plt.subplots(2, 3, figsize=(21, 10))

    for ri, (sub, row_label) in enumerate(rows):
        neg = sub[sub['mag_diff'] < 0]
        pos = sub[sub['mag_diff'] > 0]

        ax = axes[ri, 0]
        ax.hist(neg['phot_g_mean_mag'], bins=mag_bins, alpha=0.6, color='steelblue',
                label=f'Δmag < 0 (VIS brighter)  N={len(neg):,}')
        ax.hist(pos['phot_g_mean_mag'], bins=mag_bins, alpha=0.6, color='coral',
                label=f'Δmag > 0 (VIS fainter)   N={len(pos):,}')
        ax.set_xlabel('Gaia G', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'G magnitude — {row_label}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        if ri == 0:
            q1_vals   = neg['xgb_q1_prob'],        pos['xgb_q1_prob']
            comb_vals = neg['xgb_comb_prob'],       pos['xgb_comb_prob']
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

    plt.suptitle('ERO sources split by Δmag sign — Q1 vs Combined XGBoost model (Gaia features only)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = output_dir / 'ero_dmag_xgb.png'
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
    print("ERO VIS − Gaia G Δmag vs class_star")
    print("=" * 70)

    print("\n[1] Querying data...")
    df = query_data(username)
    df = assign_fields(df)

    print("\n[2] Generating VIS−Gaia Δmag vs class_star in magnitude slices...")
    plot_mag_diff_vs_class_star_combined(df, OUTPUT_DIR)

    print("\n[3] Generating Δmag vs Gaia G split by class_star...")
    plot_dmag_vs_gmag_galaxy(df, OUTPUT_DIR)

    print("\n[4] Applying Q1 + Combined XGBoost models by Δmag sign...")
    plot_xgb_q1_on_ero_galaxies(df, OUTPUT_DIR)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
