#!/usr/bin/env python3
"""
Re-evaluate the existing Q1 ensemble (output/xgboost_q1/seed_models/)
at the F1-optimal threshold instead of the default 0.5.

No retraining. Just loads each seed model, computes y_proba on its
held-out test set, sweeps thresholds 0–1, and reports per-seed +
ensemble-mean metrics at:
  - default threshold 0.5
  - F1-optimal threshold (from per-seed mean F1)

Outputs:
  output/xgboost_q1/q1_optimal_threshold.txt
  output/xgboost_q1/q1_optimal_threshold.csv  (per-seed)

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_q1_optimal_threshold.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS
from train_xgboost_q1 import (
    CLASS_MAPPING, MAG_CUT, SEEDS, WSDB_DB, WSDB_HOST, query_q1, split_data,
)

PROJECT_DIR = Path(__file__).parent.parent
SEED_MODELS_DIR = PROJECT_DIR / 'output' / 'xgboost_q1' / 'seed_models'
OUTPUT_TXT = PROJECT_DIR / 'output' / 'xgboost_q1' / 'q1_optimal_threshold.txt'
OUTPUT_CSV = PROJECT_DIR / 'output' / 'xgboost_q1' / 'q1_optimal_threshold.csv'


def load_seed_model(seed):
    """Load saved booster (raw — predict_proba_booster handles inference)."""
    booster = xgb.Booster()
    booster.load_model(str(SEED_MODELS_DIR / f'xgb_classifier_q1_seed{seed}.json'))
    return booster


def predict_proba_booster(booster, X):
    dmat = xgb.DMatrix(X)
    return booster.predict(dmat)  # binary:logistic → P(class=1)


def metrics_at_threshold(y_true, y_proba, t):
    pred = (y_proba >= t).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, labels=[0, 1],
                                                   zero_division=0)
    return {
        'accuracy':         float((pred == y_true).mean()),
        'star_precision':   float(p[0]),
        'star_recall':      float(r[0]),
        'star_f1':          float(f1[0]),
        'galaxy_precision': float(p[1]),
        'galaxy_recall':    float(r[1]),
        'galaxy_f1':        float(f1[1]),
    }


def main():
    username = os.getenv('PGUSER')
    if not username:
        print('ERROR: PGUSER not set'); return 1

    print('=' * 70)
    print('Q1 ensemble — re-evaluate at F1-optimal threshold')
    print(f'  Seed models : {SEED_MODELS_DIR}')
    print('=' * 70)

    # Load Q1 data the same way as training
    print('\nLoading Q1 data...')
    X, y, _ = query_q1(username)

    thresholds = np.linspace(0.01, 0.99, 99)

    # Per-seed sweep
    all_y_test, all_y_proba = [], []
    per_seed_f1_curves = []

    for si, seed in enumerate(SEEDS):
        booster = load_seed_model(seed)
        _, _, X_te, _, _, y_te = split_data(X, y, seed)
        y_proba = predict_proba_booster(booster, X_te)
        all_y_test.append(y_te)
        all_y_proba.append(y_proba)

        f1_curve = []
        for t in thresholds:
            pred = (y_proba >= t).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                y_te, pred, labels=[0, 1], zero_division=0)
            f1_curve.append(f1[1])  # galaxy F1
        per_seed_f1_curves.append(f1_curve)

    f1_arr = np.array(per_seed_f1_curves)
    f1_mean = f1_arr.mean(axis=0)
    f1_std  = f1_arr.std(axis=0)
    best_idx = int(np.argmax(f1_mean))
    best_t = float(thresholds[best_idx])

    print(f'\n  F1-optimal threshold (per-seed mean Galaxy F1): t = {best_t:.3f}')
    print(f'  Mean Galaxy F1 at t={best_t:.3f}: {f1_mean[best_idx]:.4f} ± {f1_std[best_idx]:.4f}')

    # Compute full per-seed metrics at default 0.5 and at best_t
    rows_default, rows_best = [], []
    for si, seed in enumerate(SEEDS):
        m_def = metrics_at_threshold(all_y_test[si], all_y_proba[si], 0.5)
        m_best = metrics_at_threshold(all_y_test[si], all_y_proba[si], best_t)
        m_def['seed']  = seed; m_def['threshold']  = 0.5
        m_best['seed'] = seed; m_best['threshold'] = best_t
        m_def['roc_auc']  = float(roc_auc_score(all_y_test[si], all_y_proba[si]))
        m_best['roc_auc'] = m_def['roc_auc']
        rows_default.append(m_def)
        rows_best.append(m_best)

    df_def  = pd.DataFrame(rows_default)
    df_best = pd.DataFrame(rows_best)

    pd.concat([df_def, df_best]).to_csv(OUTPUT_CSV, index=False)
    print(f'  -> {OUTPUT_CSV}')

    # Summary table
    keys = [
        ('accuracy', 'Accuracy'),
        ('galaxy_precision', 'Galaxy Precision'),
        ('galaxy_recall', 'Galaxy Recall'),
        ('galaxy_f1', 'Galaxy F1'),
        ('star_f1', 'Star F1'),
        ('roc_auc', 'ROC AUC'),
    ]

    lines = []
    lines.append('Q1 ENSEMBLE — DEFAULT vs F1-OPTIMAL THRESHOLD')
    lines.append('=' * 75)
    lines.append(f'F1-optimal threshold (per-seed mean Galaxy F1): t = {best_t:.3f}')
    lines.append('')
    lines.append(f'  {"Metric":<20} {"@ t=0.5":>20} {"@ t={:.3f}".format(best_t):>20}      Δ')
    lines.append('  ' + '-' * 73)
    for key, label in keys:
        d_mu = df_def[key].mean(); d_sd = df_def[key].std()
        b_mu = df_best[key].mean(); b_sd = df_best[key].std()
        delta = b_mu - d_mu
        sign = '+' if delta >= 0 else ''
        lines.append(f'  {label:<20} {d_mu:.4f} ± {d_sd:.4f}    {b_mu:.4f} ± {b_sd:.4f}    {sign}{delta:+.4f}')

    txt = '\n'.join(lines)
    OUTPUT_TXT.write_text(txt + '\n')
    print('\n' + txt)
    print(f'\n  -> {OUTPUT_TXT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
