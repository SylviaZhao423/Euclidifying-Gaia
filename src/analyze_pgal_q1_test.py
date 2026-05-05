#!/usr/bin/env python3
"""
Compare threshold analysis between Original and Optuna F1 models,
using the full 238k Q1 dataset with 10-seed ensemble mean.

3 panels:
  1. P(gal) distribution on Q1 test set (split by true label)
  2. Precision / Recall / F1 vs threshold
  3. Precision-Recall curve

Reads:
  WSDB (Q1 data, same query as training scripts)
  output/xgboost_q1/seed_models/*.json
  output/xgboost_q1_op_f1/seed_models/*.json

Writes:
  output/q1_threshold_analysis.png

Usage:
    PGUSER=siwei_zhao_2025 python src/plot_threshold_compare_full.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
CLASS_MAPPING = {"STAR": 0, "GALAXY": 1}
MAG_CUT   = 17
SEEDS     = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]

MODELS = [
    ("Original",
     PROJECT_DIR / "output" / "xgboost_q1" / "seed_models",
     "#2196F3", "#1565C0"),
    ("Optuna F1",
     PROJECT_DIR / "output" / "xgboost_q1_op_f1" / "seed_models",
     "#E53935", "#B71C1C"),
]


def query_q1(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    feat_names = list(SELECTED_GAIA_PARAMETERS)
    gaia_cols = ", ".join([f"g_{c} AS {c}" for c in feat_names])
    query = f"""
        SELECT g_phot_g_mean_mag AS phot_g_mean_mag, {gaia_cols},
               CASE WHEN p_phz_classification = 1 THEN 'STAR'
                    WHEN p_phz_classification = 2 THEN 'GALAXY' END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep
        WHERE p_phz_classification IN (1, 2)
          AND g_phot_g_mean_mag >= {MAG_CUT}
        ORDER BY g_source_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    for c in feat_names:
        df[c] = pd.to_numeric(df[c].replace("NOT_AVAILABLE", np.nan), errors="coerce")
    X = df[feat_names].values
    y = df["label"].map(CLASS_MAPPING).values
    return X, y


def ensemble_predict(seed_dir, X_test, feat_names):
    """Load all seed models and return mean P(gal)."""
    probas = []
    for seed in SEEDS:
        model_path = seed_dir / f"xgb_classifier_q1_seed{seed}.json"
        if not model_path.exists():
            print(f"    WARNING: {model_path.name} not found, skipping")
            continue
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        p = model.predict_proba(X_test)[:, 1]
        probas.append(p)
    return np.mean(probas, axis=0)


def main():
    username = os.environ.get("PGUSER")
    if not username:
        print("ERROR: PGUSER not set.")
        return 1

    feat_names = list(SELECTED_GAIA_PARAMETERS)

    # ── 1. Load Q1 data ──
    print("[1] Loading Q1 data from WSDB...")
    X, y = query_q1(username)
    print(f"  {len(X):,} sources  STAR: {(y==0).sum():,}  GALAXY: {(y==1).sum():,}")

    # Use seed-42 split (same as training)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    print(f"  Test set: {len(X_test):,}  STAR: {(y_test==0).sum():,}  GALAXY: {(y_test==1).sum():,}")

    # ── 2. Predict with each model ensemble ──
    model_preds = []
    for label, seed_dir, col_light, col_dark in MODELS:
        print(f"\n[2] Ensemble prediction: {label} ({len(SEEDS)} seeds)...")
        y_proba = ensemble_predict(seed_dir, X_test, feat_names)
        model_preds.append((label, y_proba, col_light, col_dark))
        print(f"  P(gal) range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")

    # ── 3. Plot ──
    print("\n[3] Plotting...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.35)

    thresholds = np.linspace(0, 1, 201)
    bins = np.linspace(0, 1, 101)

    for i, (label, y_proba, col_light, col_dark) in enumerate(model_preds):
        # [0] P(gal) distribution split by true label
        ax = axes[0]
        if i == 0:
            # Original: filled histograms, label just class name
            ax.hist(y_proba[y_test == 0], bins=bins, alpha=0.35, color='coral',
                    histtype='stepfilled', linewidth=0)
            ax.hist(y_proba[y_test == 0], bins=bins, color='coral',
                    histtype='step', linewidth=1.5,
                    label=f'STAR  (n={(y_test==0).sum():,})')
            ax.hist(y_proba[y_test == 1], bins=bins, alpha=0.35, color='steelblue',
                    histtype='stepfilled', linewidth=0)
            ax.hist(y_proba[y_test == 1], bins=bins, color='steelblue',
                    histtype='step', linewidth=1.5,
                    label=f'GALAXY  (n={(y_test==1).sum():,})')
        else:
            # Optuna F1: dashed outlines only, no fill, no extra labels
            ax.hist(y_proba[y_test == 0], bins=bins, color='coral',
                    histtype='step', linewidth=1.5, ls='--', label='_')
            ax.hist(y_proba[y_test == 1], bins=bins, color='steelblue',
                    histtype='step', linewidth=1.5, ls='--', label='_')

        # [1] Precision / Recall / F1 vs threshold
        ax = axes[1]
        prec_list, rec_list, f1_list = [], [], []
        for t in thresholds:
            pred = (y_proba >= t).astype(int)
            tp = ((pred == 1) & (y_test == 1)).sum()
            fp = ((pred == 1) & (y_test == 0)).sum()
            fn = ((pred == 0) & (y_test == 1)).sum()
            p = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2*p*r/(p+r) if (p + r) > 0 else 0.0
            prec_list.append(p)
            rec_list.append(r)
            f1_list.append(f1)

        best_idx = np.nanargmax(f1_list)
        best_t = thresholds[best_idx]

        ls = '-' if label == 'Original' else '--'
        # Only label metrics once (from first model)
        lbl_p = 'Precision' if label == 'Original' else '_'
        lbl_r = 'Recall'    if label == 'Original' else '_'
        lbl_f = 'F1'        if label == 'Original' else '_'
        ax.plot(thresholds, prec_list, color='steelblue', lw=2, ls=ls, label=lbl_p)
        ax.plot(thresholds, rec_list,  color='coral',     lw=2, ls=ls, label=lbl_r)
        ax.plot(thresholds, f1_list,   color='mediumseagreen', lw=2, ls=ls, label=lbl_f)

        # Annotate values at 0.5 and 0.9 (Original model only)
        if i == 0:
            prec_arr = np.array(prec_list)
            rec_arr = np.array(rec_list)
            f1_arr = np.array(f1_list)
            # Annotations at 0.5
            idx_05 = np.argmin(np.abs(thresholds - 0.5))
            ax.annotate(f'{prec_arr[idx_05]:.2f}', xy=(0.5, prec_arr[idx_05]),
                        xytext=(0.52, prec_arr[idx_05] + 0.04),
                        fontsize=7, color='steelblue', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='steelblue', lw=0.6))
            ax.annotate(f'{rec_arr[idx_05]:.2f}', xy=(0.5, rec_arr[idx_05]),
                        xytext=(0.52, rec_arr[idx_05] - 0.06),
                        fontsize=7, color='coral', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='coral', lw=0.6))
            ax.annotate(f'{f1_arr[idx_05]:.2f}', xy=(0.5, f1_arr[idx_05]),
                        xytext=(0.55, f1_arr[idx_05] + 0.04),
                        fontsize=7, color='mediumseagreen', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='mediumseagreen', lw=0.6))
            # Annotations at 0.9 — offset to lower left
            idx_09 = np.argmin(np.abs(thresholds - 0.9))
            ax.annotate(f'{prec_arr[idx_09]:.2f}', xy=(0.9, prec_arr[idx_09]),
                        xytext=(0.78, prec_arr[idx_09] - 0.06),
                        fontsize=7, color='steelblue', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='steelblue', lw=0.6))
            ax.annotate(f'{rec_arr[idx_09]:.2f}', xy=(0.9, rec_arr[idx_09]),
                        xytext=(0.78, rec_arr[idx_09] - 0.06),
                        fontsize=7, color='coral', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='coral', lw=0.6))
            ax.annotate(f'{f1_arr[idx_09]:.2f}', xy=(0.9, f1_arr[idx_09]),
                        xytext=(0.75, f1_arr[idx_09] - 0.06),
                        fontsize=7, color='mediumseagreen', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='mediumseagreen', lw=0.6))

        # # [2] Precision-Recall curve (commented out — enable with 3-panel layout)
        # ax = axes[2]
        # pr_prec, pr_rec, pr_thresh = precision_recall_curve(y_test, y_proba)
        # sc = ax.scatter(pr_rec[:-1], pr_prec[:-1], c=pr_thresh,
        #                 cmap='viridis', s=6, zorder=5, alpha=0.7)
        # idx_def = np.argmin(np.abs(thresholds - 0.5))
        # marker = 'o' if label == 'Original' else 's'
        # if i == 0:
        #     ax.scatter(rec_list[idx_def], prec_list[idx_def],
        #                s=80, color='#2196F3', marker='o', edgecolors='black', alpha=0.7,
        #                linewidths=1, zorder=6, label=f'Default (0.5)')
        #     ax.scatter(rec_list[best_idx], prec_list[best_idx],
        #                s=80, color='#E53935', marker='o', edgecolors='black', alpha=0.7,
        #                linewidths=1, zorder=6, label=f'Best F1 ({best_t:.2f})')

    # [0] Finalize
    axes[0].axvline(0.5, color='black', lw=1.5, ls='--', alpha=0.5)
    axes[0].axvline(0.9, color='green', lw=1.5, ls='--', alpha=0.5)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('P(gal)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=13, fontweight='bold')
    axes[0].set_title('P(gal) distribution — Q1 test set',
                       fontsize=13, fontweight='bold')
    from matplotlib.lines import Line2D as L2D
    h0, l0 = axes[0].get_legend_handles_labels()
    # extra0 = [
    #     L2D([0], [0], color='grey', lw=1.5, ls='-', label='— Original'),
    #     L2D([0], [0], color='grey', lw=1.5, ls='--', label='-- Optuna F1'),
    # ]
    axes[0].legend(handles=h0, fontsize=11, loc='upper left')
    axes[0].tick_params(labelsize=12)
    axes[0].grid(alpha=0.3)

    # [1] Finalize
    axes[1].axvline(0.5, color='black', lw=1.5, ls='--', alpha=0.5, label='Default (0.5)')
    axes[1].axvline(0.9, color='green', lw=1.5, ls='--', alpha=0.5, label='Threshold (0.9)')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel('P(gal) threshold', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Score', fontsize=13, fontweight='bold')
    axes[1].set_title('Metrics vs threshold — Q1 test set',
                       fontsize=13, fontweight='bold')
    # Add line style explanation
    from matplotlib.lines import Line2D
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles=handles, fontsize=11, ncol=1)
    axes[1].tick_params(labelsize=12)
    axes[1].grid(alpha=0.3)

    # # [2] Finalize (commented out — enable with 3-panel layout)
    # cb = fig.colorbar(sc, ax=axes[2], pad=0.02)
    # cb.set_label('P(gal) threshold', fontsize=9)
    # axes[2].set_xlim(0, 1)
    # axes[2].set_ylim(0, 1)
    # axes[2].set_xlabel('Recall', fontsize=11, fontweight='bold')
    # axes[2].set_ylabel('Precision', fontsize=11, fontweight='bold')
    # axes[2].set_title('Precision-Recall curve — Q1 test set',
    #                    fontsize=11, fontweight='bold')
    # axes[2].legend(fontsize=10)
    # axes[2].grid(alpha=0.3)

    fig.suptitle('Q1 threshold analysis  '
                 f'(solid = Original, dashed = Optuna F1)',
                 fontsize=14, fontweight='bold', y=1.0)

    out = PROJECT_DIR / "output" / "pgal_q1_test.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
