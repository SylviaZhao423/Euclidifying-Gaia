#!/usr/bin/env python3
"""
Plot purity gain landscape for selected mixing ratios (lower panels only).

Shows delta purity vs Q1-only baseline for selected ratios as 2D heatmaps
(threshold × galaxy fraction), with a recall line plot in the first panel.

Reads:
  WSDB (Q1 data, same query as training scripts)
  output/xgboost_mix_q1_ero/ratio_sg*/seed_models/xgb_classifier_q1_seed*.json

Writes:
  output/xgboost_mix_q1_ero/mix_purity_landscape_select.png

Usage:
    PGUSER=siwei_zhao_2025 python src/plot_purity_landscape_select.py
    PGUSER=siwei_zhao_2025 python src/plot_purity_landscape_select.py --dir xgboost_mix_q1_ero_g18.5_cs0.8
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'
MAG_CUT   = 17
CLASS_MAPPING = {'STAR': 0, 'GALAXY': 1}
SEEDS = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]

# Ratios to show (besides Q1-only baseline which is always included)
SELECT_RATIOS = [3.0, 5.0, 8.6]


def query_q1(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    feat_names = list(SELECTED_GAIA_PARAMETERS)
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in feat_names])
    query = f"""
        SELECT {gaia_cols},
               CASE WHEN p_phz_classification = 1 THEN 'STAR'
                    WHEN p_phz_classification = 2 THEN 'GALAXY' END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep
        WHERE p_phz_classification IN (1, 2)
          AND g_phot_g_mean_mag >= {MAG_CUT}
        ORDER BY g_random_index
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    for c in feat_names:
        df[c] = pd.to_numeric(df[c].replace('NOT_AVAILABLE', np.nan), errors='coerce')
    return df[feat_names].values, df['label'].map(CLASS_MAPPING).values


def _find_seed_dir(mix_dir, ratio):
    for d in mix_dir.glob("ratio_*"):
        if "sweep" in d.name:
            continue
        try:
            num_str = d.name.replace("ratio_sg", "").split("_")[0]
            r = float(num_str)
            if abs(r - ratio) < 0.1:
                sd = d / "seed_models"
                if sd.exists():
                    return sd
        except ValueError:
            continue
    return None


def _per_seed_predictions(seed_dir, X_q1, y_q1):
    seed_paths = sorted(seed_dir.glob("xgb_classifier_q1_seed*.json"))
    preds = []
    for sp in seed_paths:
        seed_val = int(sp.stem.replace("xgb_classifier_q1_seed", ""))
        _, X_ts, _, y_ts = train_test_split(
            X_q1, y_q1, test_size=0.15, random_state=seed_val, stratify=y_q1)
        m = xgb.XGBClassifier()
        m.load_model(str(sp))
        preds.append((y_ts, m.predict_proba(X_ts)[:, 1]))
    return preds


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Plot purity gain landscape for selected ratios")
    parser.add_argument("--dir", type=str, default="xgboost_mix_q1_ero",
                        help="Output directory name under output/")
    parser.add_argument("--ratios", type=float, nargs='+', default=SELECT_RATIOS,
                        help="Ratios to show (default: 4 6 8.6)")
    args = parser.parse_args()

    mix_dir = PROJECT_DIR / "output" / args.dir

    username = os.environ.get("PGUSER")
    if not username:
        print("ERROR: PGUSER not set.")
        return 1

    print("[1] Loading Q1 data from WSDB...")
    X_q1, y_q1 = query_q1(username)
    print(f"  {len(X_q1):,} sources")

    # Find all available ratios
    all_ratios = []
    for d in sorted(mix_dir.glob("ratio_*")):
        if "sweep" in d.name:
            continue
        try:
            num_str = d.name.replace("ratio_sg", "").split("_")[0]
            all_ratios.append(float(num_str))
        except ValueError:
            continue
    all_ratios = sorted(set(all_ratios))
    print(f"  Available ratios: {all_ratios}")

    # Q1-only baseline (first ratio) + selected ratios
    baseline_ratio = all_ratios[0]
    select_ratios = [baseline_ratio]
    for sr in args.ratios:
        best = min(all_ratios, key=lambda r: abs(r - sr))
        if best not in select_ratios:
            select_ratios.append(best)
    print(f"  Selected: {select_ratios}")

    # Compute TPR/FPR grid for selected ratios
    thresh_fine = np.linspace(0.5, 0.99, 50)
    pi_hm = np.logspace(-2.5, -0.3, 80)
    n_sel = len(select_ratios)

    ratio_labels = []
    tpr_grid = np.zeros((n_sel, len(thresh_fine)))
    fpr_grid = np.zeros((n_sel, len(thresh_fine)))

    print("\n[2] Computing per-seed TPR/FPR...")
    for ri, ratio in enumerate(select_ratios):
        seed_dir = _find_seed_dir(mix_dir, ratio)
        if seed_dir is None:
            print(f"  Ratio {ratio}: not found, skipping")
            continue

        label = f"{ratio:.0f}:1" if ratio == int(ratio) else f"{ratio:.1f}:1"
        ratio_labels.append(label)
        print(f"  {label}...")

        seed_preds = _per_seed_predictions(seed_dir, X_q1, y_q1)
        for ti, t in enumerate(thresh_fine):
            tp_s, fp_s, fn_s, tn_s = 0, 0, 0, 0
            for y_ts, p_ts in seed_preds:
                pred_gal = p_ts >= t
                tp_s += (pred_gal & (y_ts == 1)).sum()
                fp_s += (pred_gal & (y_ts == 0)).sum()
                fn_s += (~pred_gal & (y_ts == 1)).sum()
                tn_s += (~pred_gal & (y_ts == 0)).sum()
            n = len(seed_preds)
            tpr_grid[ri, ti] = (tp_s/n) / max(tp_s/n + fn_s/n, 1)
            fpr_grid[ri, ti] = (fp_s/n) / max(fp_s/n + tn_s/n, 1)

    # Compute purity maps and deltas
    purity_maps = {}
    for ri in range(n_sel):
        purity_2d = np.zeros((len(thresh_fine), len(pi_hm)))
        for ti in range(len(thresh_fine)):
            tpr = tpr_grid[ri, ti]
            fpr = fpr_grid[ri, ti]
            purity_2d[ti, :] = tpr * pi_hm / (tpr * pi_hm + fpr * (1 - pi_hm) + 1e-12)
        purity_maps[ri] = purity_2d

    deltas = {}
    for ri in range(1, n_sel):
        deltas[ri] = (purity_maps[ri] - purity_maps[0]) * 100
    all_delta_vals = np.concatenate([d.ravel() for d in deltas.values()])
    vabs = max(abs(all_delta_vals.min()), abs(all_delta_vals.max()))

    delta_cmap = LinearSegmentedColormap.from_list(
        'orange_white_purple',
        ['#EF7627', '#FF9A56', '#FFFFFF', '#D162A4', '#8B1A6B'])
    delta_norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

    # Plot: delta heatmaps only
    print("\n[3] Plotting...")
    n_panels = n_sel - 1  # deltas only (no recall panel)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.2)

    for col, ri in enumerate(range(1, n_sel)):
        ax = axes[col]
        im = ax.pcolormesh(pi_hm * 100, thresh_fine, deltas[ri],
                           cmap=delta_cmap, shading='auto', norm=delta_norm, alpha=0.8)
        cs = ax.contour(pi_hm * 100, thresh_fine, deltas[ri],
                        levels=[1, 2, 5, 10], colors='black',
                        linewidths=0.8, linestyles='--')
        ax.clabel(cs, fmt='+%d%%', fontsize=8)
        ax.axvspan(1, 3, color='white', alpha=0.1, zorder=2)
        ax.axvline(1, color='grey', lw=0.8, ls=':', alpha=0.5)
        ax.axvline(3, color='grey', lw=0.8, ls=':', alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlim(pi_hm.min() * 100, pi_hm.max() * 100)
        ax.set_xlabel("True galaxy fraction (%)", fontsize=11, fontweight='bold')
        if col == 0:
            ax.set_ylabel("P(gal) threshold", fontsize=11, fontweight='bold')
        ax.set_title(f"Purity gain: {ratio_labels[ri]} vs Q1-only",
                     fontsize=12, fontweight='bold')
        fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85, label="Purity gain (%)")

    out = PROJECT_DIR / "output" / "mix_purity_landscape_select.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
