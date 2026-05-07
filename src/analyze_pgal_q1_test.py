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

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
MAG_CUT   = 17
CLASS_MAPPING = {"STAR": 0, "GALAXY": 1}
SEEDS     = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]

MODELS = [
    ("Original",
     PROJECT_DIR / "output" / "xgboost_q1" / "seed_models",
     "#2196F3", "#1565C0"),
]


def query_q1(username):
    """Query Q1 data — same query as training scripts."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    feat_names = list(SELECTED_GAIA_PARAMETERS)
    gaia_cols = ", ".join([f"g_{c} AS {c}" for c in feat_names])
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
        df[c] = pd.to_numeric(df[c].replace("NOT_AVAILABLE", np.nan), errors="coerce")
    return df[feat_names].values, df["label"].map(CLASS_MAPPING).values


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

    # ── 2. Per-seed predictions (each seed on its own held-out test set) ──
    thresholds = np.linspace(0, 1, 201)
    bins = np.linspace(0, 1, 101)

    for label, seed_dir, col_light, col_dark in MODELS:
        print(f"\n[2] Per-seed evaluation: {label} ({len(SEEDS)} seeds)...")

        # Accumulators for averaging across seeds
        hist_star_sum = np.zeros(len(bins) - 1)
        hist_gal_sum  = np.zeros(len(bins) - 1)
        all_prec = np.zeros((len(SEEDS), len(thresholds)))
        all_rec  = np.zeros((len(SEEDS), len(thresholds)))
        all_f1   = np.zeros((len(SEEDS), len(thresholds)))
        n_star_total, n_gal_total = 0, 0

        for si, seed in enumerate(SEEDS):
            # Same split as training: 85/15
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.15, random_state=seed, stratify=y)

            model_path = seed_dir / f"xgb_classifier_q1_seed{seed}.json"
            if not model_path.exists():
                print(f"    WARNING: {model_path.name} not found, skipping")
                continue
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
            y_proba = model.predict_proba(X_test)[:, 1]

            # Histogram counts
            hist_star_sum += np.histogram(y_proba[y_test == 0], bins=bins)[0]
            hist_gal_sum  += np.histogram(y_proba[y_test == 1], bins=bins)[0]
            n_star_total += (y_test == 0).sum()
            n_gal_total  += (y_test == 1).sum()

            # Metrics at each threshold
            for ti, t in enumerate(thresholds):
                pred = (y_proba >= t).astype(int)
                tp = ((pred == 1) & (y_test == 1)).sum()
                fp = ((pred == 1) & (y_test == 0)).sum()
                fn = ((pred == 0) & (y_test == 1)).sum()
                all_prec[si, ti] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
                all_rec[si, ti]  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_val = 2*all_prec[si,ti]*all_rec[si,ti]/(all_prec[si,ti]+all_rec[si,ti]) \
                    if (all_prec[si,ti] + all_rec[si,ti]) > 0 else 0.0
                all_f1[si, ti] = f1_val

        # Average histograms and metrics
        hist_star_mean = hist_star_sum / len(SEEDS)
        hist_gal_mean  = hist_gal_sum / len(SEEDS)
        prec_arr = np.nanmean(all_prec, axis=0)
        rec_arr  = np.nanmean(all_rec, axis=0)
        f1_arr   = np.nanmean(all_f1, axis=0)
        n_star_avg = n_star_total // len(SEEDS)
        n_gal_avg  = n_gal_total // len(SEEDS)

        # Verify against saved metrics at threshold 0.5
        idx_05 = np.argmin(np.abs(thresholds - 0.5))
        print(f"  Mean Precision@0.5: {prec_arr[idx_05]:.4f}  "
              f"Recall@0.5: {rec_arr[idx_05]:.4f}  F1@0.5: {f1_arr[idx_05]:.4f}")

    # ── 3. Plot ──
    print("\n[3] Plotting...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.subplots_adjust(hspace=0.2)

    # [0] Histogram — plot averaged bin counts as bar chart
    ax = axes[0]
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]
    ax.bar(bin_centres, hist_star_mean, width=bin_width, alpha=0.35, color='coral',
           edgecolor='none')
    ax.step(bins[:-1], hist_star_mean, where='post', color='coral', lw=1.5,
            label=f'STAR  (n={n_star_avg:,})')
    ax.bar(bin_centres, hist_gal_mean, width=bin_width, alpha=0.35, color='steelblue',
           edgecolor='none')
    ax.step(bins[:-1], hist_gal_mean, where='post', color='steelblue', lw=1.5,
            label=f'GALAXY  (n={n_gal_avg:,})')

    # [1] Metrics vs threshold — plot averaged curves
    ax = axes[1]
    ax.plot(thresholds, prec_arr, color='steelblue', lw=2, label='Precision')
    ax.plot(thresholds, rec_arr,  color='coral',     lw=2, label='Recall')
    ax.plot(thresholds, f1_arr,   color='mediumseagreen', lw=2, label='F1')

    # Inline labels at 0.5, 0.7, 0.9
    bbox = dict(boxstyle='round,pad=0.15', facecolor='white',
                edgecolor='none', alpha=1.0)
    for t_val in [0.5, 0.7, 0.9]:
        idx_t = np.argmin(np.abs(thresholds - t_val))
        for arr, color in [(prec_arr, 'steelblue'),
                           (rec_arr, 'coral'),
                           (f1_arr, 'mediumseagreen')]:
            ax.text(t_val, arr[idx_t], f'{arr[idx_t]:.3f}',
                    fontsize=10, fontweight='bold', color=color,
                    ha='center', va='center', bbox=bbox, zorder=10)

    # [0] Finalize
    # [0] Finalize — top panel (histogram), no x-label since shared
    axes[0].axvline(0.5, color='black', lw=1.5, ls='--', alpha=0.5)
    axes[0].axvline(0.7, color='orange', lw=1.5, ls='--', alpha=0.5)
    axes[0].axvline(0.9, color='green', lw=1.5, ls='--', alpha=0.5)
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Count per bin', fontsize=13, fontweight='bold')
    axes[0].text(0.98, 1.1, 'P(gal) distribution', transform=axes[0].transAxes,
                 fontsize=11, fontstyle='italic', fontweight='bold', color='#555555',
                 ha='right', va='top')
    from matplotlib.lines import Line2D as L2D
    h0, l0 = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=h0, fontsize=11, loc='upper left')
    axes[0].tick_params(labelsize=12)
    axes[0].tick_params(axis='x', labelbottom=False)
    axes[0].grid(alpha=0.3)

    # [1] Finalize — bottom panel (metrics)
    axes[1].axvline(0.5, color='black', lw=1.5, ls='--', alpha=0.5)
    axes[1].axvline(0.7, color='orange', lw=1.5, ls='--', alpha=0.5)
    axes[1].axvline(0.9, color='green', lw=1.5, ls='--', alpha=0.5)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_xlabel('P(gal) threshold', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Score', fontsize=13, fontweight='bold')
    axes[1].text(0.98, 1.1, 'Metrics vs classification threshold',
                 transform=axes[1].transAxes, fontsize=11, fontstyle='italic', fontweight='bold',
                 color='#555555', ha='right', va='top')
    axes[1].legend(fontsize=11, loc='lower left')
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

    # fig.suptitle('Q1 threshold analysis  '
    #              f'(ensemble mean, {len(SEEDS)} seeds)',
    #              fontsize=14, fontweight='bold', y=1.0)

    out = PROJECT_DIR / "output" / "pgal_q1_test.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
