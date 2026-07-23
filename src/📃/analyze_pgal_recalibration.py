#!/usr/bin/env python3
"""
Bayesian recalibration analysis for the Original model.

Shows how galaxy selection purity and contamination change with
the assumed true galaxy fraction. Per-seed evaluation (no leakage).

Reads:
  WSDB (Q1 data, same query as training scripts)
  output/xgboost_q1/seed_models/xgb_classifier_q1_seed*.json

Writes:
  output/pgal_recalibration_compare.png

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_pgal_recalibration.py
"""

import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'
MAG_CUT   = 17
CLASS_MAPPING = {'STAR': 0, 'GALAXY': 1}


def query_q1(username):
    """Query Q1 data — same query as training scripts."""
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

SEEDS = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]

MODELS = [
    ("Original",
     PROJECT_DIR / "output" / "xgboost_q1" / "seed_models",
     "-"),
    # ("Optuna F1",
    #  PROJECT_DIR / "output" / "xgboost_q1_op_f1" / "seed_models",
    #  "--"),
    # ("Optuna ROC",
    #  PROJECT_DIR / "output" / "xgboost_q1_op_roc" / "seed_models",
    #  "-."),
]

THRESHOLDS_MARK = [0.5, 0.9]  # thresholds to annotate with points


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--heatmap-model", choices=["original", "optuna"],
                        default="original",
                        help="Which model to use for the right-panel heatmap")
    args = parser.parse_args()

    username = os.environ.get("PGUSER")
    if not username:
        print("ERROR: PGUSER not set.")
        return 1

    feat_names = list(SELECTED_GAIA_PARAMETERS)

    print("[1] Loading Q1 data from WSDB...")
    X_raw, y = query_q1(username)
    print(f"  {len(X_raw):,} sources  STAR: {(y==0).sum():,}  GALAXY: {(y==1).sum():,}")

    pi_q1 = (y == 1).sum() / len(y)

    # Compute per-seed TPR/FPR at each threshold (no leakage)
    pi_range = np.logspace(-3, np.log10(0.5), 500)
    mark_points = [(0.01, "1%"), (0.03, "3%")]

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    # Import shared threshold colormap
    sys.path.insert(0, str(Path(__file__).parent))
    from analyze_mix_q1_ero import make_threshold_cmap

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    fig.subplots_adjust(wspace=0.18)

    thresh_range = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    cmap = make_threshold_cmap(n_intervals=5)  # pink|orange|blue|green|dark-green for 0.5→1.0
    norm = mcolors.Normalize(vmin=0.5, vmax=1.0)

    for model_label, seed_dir, ls in MODELS:
        print(f"\n[2] {model_label}: per-seed evaluation...")

        # Pre-compute per-seed predictions
        seed_preds = []
        for seed in SEEDS:
            model_path = seed_dir / f"xgb_classifier_q1_seed{seed}.json"
            if not model_path.exists():
                continue
            _, X_ts, _, y_ts = train_test_split(
                X_raw, y, test_size=0.15, random_state=seed, stratify=y)
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
            seed_preds.append((y_ts, model.predict_proba(X_ts)[:, 1]))

        def _avg_tpr_fpr(thresh):
            tp_s, fp_s, fn_s, tn_s = 0, 0, 0, 0
            for y_ts, p_ts in seed_preds:
                pred_gal = p_ts >= thresh
                tp_s += (pred_gal & (y_ts == 1)).sum()
                fp_s += (pred_gal & (y_ts == 0)).sum()
                fn_s += (~pred_gal & (y_ts == 1)).sum()
                tn_s += (~pred_gal & (y_ts == 0)).sum()
            n = len(seed_preds)
            tpr = (tp_s/n) / max(tp_s/n + fn_s/n, 1)
            fpr = (fp_s/n) / max(fp_s/n + tn_s/n, 1)
            return tpr, fpr

        for thresh in thresh_range:
            tpr, fpr = _avg_tpr_fpr(thresh)
            prec = tpr * pi_range / (tpr * pi_range + fpr * (1 - pi_range) + 1e-12)
            col = cmap(norm(thresh))
            axes[0].plot(pi_range * 100, prec * 100, color=col,
                         lw=1.2, ls=ls, alpha=0.85)

        # Mark specific thresholds with annotations
        for thresh_mark in THRESHOLDS_MARK:
            tpr, fpr = _avg_tpr_fpr(thresh_mark)
            col = cmap(norm(thresh_mark))

            if model_label == "Original":
                bbox_gap = dict(boxstyle='round,pad=0.12', facecolor='white',
                                edgecolor='none', alpha=0.7)
                # Label at 1%, 3%, and Q1 rate (~26%)
                all_marks = list(mark_points) + [(pi_q1, f"{pi_q1:.0%}")]
                for pi_mark, pi_label in all_marks:
                    p_mark = tpr * pi_mark / (tpr * pi_mark + fpr * (1 - pi_mark) + 1e-12)
                    # Dot marker on the curve
                    axes[0].scatter(pi_mark * 100, p_mark * 100, s=30, zorder=9,
                                    color=col, edgecolors='k', linewidths=0.5)
                    if pi_mark == pi_q1:
                        lbl = f"{p_mark:.0%} (t={thresh_mark})"
                        oxy = (14, -18)   # left and slightly down
                        ha = 'right'
                    elif pi_mark == mark_points[-1][0]:
                        lbl = f"{p_mark:.0%} (t={thresh_mark})"
                        oxy = (14, -18)   # left and slightly down
                        ha = 'right'
                    else:
                        lbl = f"{p_mark:.0%} (t={thresh_mark})"
                        oxy = (14, -18)   # left and slightly down
                        ha = 'right'
                    axes[0].annotate(lbl, xy=(pi_mark * 100, p_mark * 100),
                                     xytext=oxy, textcoords='offset points',
                                     fontsize=11, fontweight="bold", color=col,
                                     ha=ha, va='bottom', bbox=bbox_gap, zorder=10)

                print(f"  Threshold {thresh_mark}: TPR={tpr:.4f} FPR={fpr:.4f}")

    # Finalize [0] Purity
    ax = axes[0]
    ax.axhline(50, color="grey", lw=1, ls=":", alpha=0.5)
    ax.axvspan(1, 3, color="steelblue", alpha=0.12, zorder=0,
               label="Est. Gaia gal. fraction (1\u20133%)")
    ax.axvline(pi_q1 * 100, color="coral", lw=1.5, ls="--",
               label=f"Q1 rate ({pi_q1:.0%})")
    ax.set_xlabel("True galaxy fraction (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Galaxy selection purity (%)", fontsize=13, fontweight="bold")
    # ax.set_title("Purity of galaxy selection vs true galaxy fraction",
    #              fontsize=13, fontweight="bold", pad=10)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 50)  # extends past Q1 rate at ~26%
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # [1] 2D heatmap: purity as function of (galaxy fraction, threshold)
    hm_label, hm_seed_dir, _ = MODELS[0]
    print(f"\n[3] Building 2D purity heatmap ({hm_label}, per-seed)...")

    thresh_grid = np.linspace(0.3, 0.99, 70)
    pi_grid = np.logspace(-2.5, -0.3, 80)
    purity_2d = np.zeros((len(thresh_grid), len(pi_grid)))

    for i, t in enumerate(thresh_grid):
        tpr, fpr = _avg_tpr_fpr(t)
        for j, pi in enumerate(pi_grid):
            purity_2d[i, j] = tpr * pi / (tpr * pi + fpr * (1 - pi) + 1e-12)

    ax = axes[1]
    im = ax.pcolormesh(pi_grid * 100, thresh_grid, purity_2d * 100,
                        cmap='RdYlGn', shading='auto', vmin=0, vmax=100)
    # Contour lines at key purity levels
    cs = ax.contour(pi_grid * 100, thresh_grid, purity_2d * 100,
                     levels=[25, 50, 75, 90, 95], colors='black',
                     linewidths=0.8, linestyles='--')
    ax.clabel(cs, fmt='%d%%', fontsize=11)

    # Mark the 1-3% galaxy fraction band
    ax.axvspan(1, 3, color='white', alpha=0.15, zorder=2)
    ax.axvline(1, color='white', lw=0.8, ls=':', alpha=0.5)
    ax.axvline(3, color='white', lw=0.8, ls=':', alpha=0.5)
    # Q1 training ratio
    ax.axvline(pi_q1 * 100, color='coral', lw=1.5, ls='--', alpha=0.8, zorder=3)
    # ax.text(pi_q1 * 100 * 0.9, 0.90, f'Q1 ({pi_q1:.0%})', fontsize=10,
    #         color='coral', fontweight='bold')

    ax.set_xscale('log')
    ax.set_xlim(pi_grid.min() * 100, pi_grid.max() * 100)
    ax.set_xlabel("True galaxy fraction (%)", fontsize=13, fontweight="bold")
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("P(gal) threshold", fontsize=13, fontweight="bold", labelpad=10)
    # ax.set_title("Galaxy selection purity (%)", fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(labelsize=11)
    cb2 = fig.colorbar(im, ax=ax, pad=0.02)
    cb2.set_label("Purity (%)", fontsize=11, fontweight="bold")
    cb2.ax.tick_params(labelsize=10)

    # Colorbar for threshold (panels 0 and 1)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes[0], pad=0.02, shrink=0.8)
    cb.set_label("P(gal) threshold", fontsize=11, fontweight="bold")
    cb.ax.tick_params(labelsize=10)
    axes[0].tick_params(labelsize=11)

    # fig.suptitle("Bayesian re-calibration of XGBoost galaxy selection",
    #              fontsize=14, fontweight="bold", y=1.01)

    out = PROJECT_DIR / "output" / "pgal_recalibration.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
