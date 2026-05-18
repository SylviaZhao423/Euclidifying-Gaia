#!/usr/bin/env python3
"""
Apples-to-apples test: train on PHZ-confidence-filtered Q1 sources,
evaluate on UNFILTERED Q1 test set (same per-seed splits as the
canonical train_xgboost_q1.py).

Design:
  1. Load full PHZ ∈ {1,2}, G ≥ 17 pool (244,556 sources). This is
     IDENTICAL to what train_xgboost_q1.py uses.
  2. For each seed, do the standard 70/15/15 split on the full pool
     (same random_state as canonical → identical test indices).
  3. Apply PHZ-confidence filter to TRAIN + VAL only (drops ambiguous
     labels from training/early-stopping).
  4. TEST set stays UNFILTERED — same n_test as canonical → directly
     comparable per-seed metrics.

This isolates the training-improvement effect from the test-composition
shift that confused the previous comparison.

Outputs to output/xgboost_q1_phz_conf_s{}_g{}_eval_unfilt/ :
  q1_summary.txt
  q1_seed_metrics.json
  seed_models/*.json
  q1_feature_importance.{png,csv}
  q1_threshold_analysis.png       (on the unfiltered test set)
  phz_conf_unfilt_comparison.txt   — direct A/B vs canonical (same test sets)

Usage:
    PGUSER=siwei_zhao_2025 python src/train_xgboost_q1_phz_confidence_eval_unfiltered.py
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS
from train_xgboost_q1 import (
    CLASS_MAPPING, MAG_CUT, SEEDS, WSDB_DB, WSDB_HOST,
    evaluate, plot_feature_importance, plot_threshold_analysis,
    split_data, train_model,
)

PROJECT_DIR = Path(__file__).parent.parent


def query_q1_full(username):
    """Same query as canonical train_xgboost_q1.py, plus the two PHZ-prob columns."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{c} AS {c}' for c in SELECTED_GAIA_PARAMETERS])
    df = pd.read_sql_query(f"""
        SELECT
            g_random_index AS random_index,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            {gaia_cols},
            p_phz_classification AS phz_cls,
            p_phz_star_prob,
            p_phz_gal_prob,
            CASE
                WHEN p_phz_classification = 1 THEN 'STAR'
                WHEN p_phz_classification = 2 THEN 'GALAXY'
            END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep
        WHERE p_phz_classification IN (1, 2)
          AND g_phot_g_mean_mag >= {MAG_CUT}
        ORDER BY g_random_index
    """, conn)
    conn.close()
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    X = df[SELECTED_GAIA_PARAMETERS].values
    y = df['label'].map(CLASS_MAPPING).values
    star_p = df['p_phz_star_prob'].values.astype(float)
    gal_p  = df['p_phz_gal_prob'].values.astype(float)
    print(f"  -> {len(X):,} sources  STAR: {(y==0).sum():,}  GALAXY: {(y==1).sum():,}")
    return X, y, star_p, gal_p


def split_with_filter(X, y, star_p, gal_p, seed, star_th, gal_th):
    """Standard 70/15/15 split on the full pool, then PHZ-confidence filter
    applied to TRAIN + VAL only. TEST stays unfiltered (same as canonical)."""
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(X, y, seed)

    # We need indices to map back to star_p/gal_p, so re-do the split with indices.
    from sklearn.model_selection import train_test_split
    idx_all = np.arange(len(X))
    idx_temp, idx_test = train_test_split(idx_all, test_size=0.15,
                                          random_state=seed, stratify=y)
    idx_train, idx_val = train_test_split(idx_temp, test_size=0.176,
                                          random_state=seed, stratify=y[idx_temp])

    def keep_mask(idx):
        s_p = star_p[idx]; g_p = gal_p[idx]; lbl = y[idx]
        return ((lbl == 0) & (s_p >= star_th)) | ((lbl == 1) & (g_p >= gal_th))

    m_tr = keep_mask(idx_train)
    m_v  = keep_mask(idx_val)
    X_tr_f = X[idx_train][m_tr]; y_tr_f = y[idx_train][m_tr]
    X_v_f  = X[idx_val][m_v];    y_v_f  = y[idx_val][m_v]
    X_te   = X[idx_test];        y_te   = y[idx_test]   # unfiltered

    return X_tr_f, X_v_f, X_te, y_tr_f, y_v_f, y_te, len(idx_train), len(idx_val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--star-prob', type=float, default=0.5)
    parser.add_argument('--gal-prob',  type=float, default=0.5)
    args = parser.parse_args()

    username = os.getenv('PGUSER')
    if not username:
        print('ERROR: PGUSER not set'); return 1

    out_dir = (PROJECT_DIR / 'output' /
               f'xgboost_q1_phz_conf_s{args.star_prob:g}_g{args.gal_prob:g}_eval_unfilt')
    seed_models_dir = out_dir / 'seed_models'
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_models_dir.mkdir(exist_ok=True)

    print('=' * 70)
    print('Q1 XGBoost — PHZ-conf filter on TRAIN+VAL, eval on UNFILTERED test')
    print(f'  Filter (train+val only): star_prob>={args.star_prob}  gal_prob>={args.gal_prob}')
    print(f'  Output : {out_dir}')
    print('=' * 70)

    print('\n[1] Loading full Q1 pool (PHZ ∈ {1,2}, G ≥ 17)...')
    X, y, star_p, gal_p = query_q1_full(username)

    all_metrics, all_importances, all_models, all_test_idx = [], [], [], []
    n_train_full_42, n_train_filt_42, n_val_full_42, n_val_filt_42, n_test_42 = (None,)*5

    for idx, seed in enumerate(SEEDS, start=1):
        X_tr, X_v, X_te, y_tr, y_v, y_te, n_train_pre, n_val_pre = \
            split_with_filter(X, y, star_p, gal_p, seed,
                              args.star_prob, args.gal_prob)
        ratio = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        print(f"  [seed {seed} {idx}/{len(SEEDS)}]  "
              f"train pre/post filter: {n_train_pre:,}/{len(X_tr):,}  "
              f"val pre/post: {n_val_pre:,}/{len(X_v):,}  "
              f"test (unfiltered): {len(X_te):,}  ratio={ratio:.1f}:1")

        model = train_model(X_tr, y_tr, X_v, y_v, verbose=False)
        m = evaluate(model, X_te, y_te)
        m_clean = {k: v for k, v in m.items() if k != 'report'}
        m_clean['seed'] = seed
        all_metrics.append(m_clean)
        all_importances.append(model.feature_importances_)
        all_models.append(model)

        if seed == 42:
            n_train_full_42, n_train_filt_42 = n_train_pre, len(X_tr)
            n_val_full_42, n_val_filt_42, n_test_42 = n_val_pre, len(X_v), len(X_te)

        model.get_booster().save_model(
            str(seed_models_dir / f'xgb_classifier_q1_seed{seed}.json'))

    pd.DataFrame(all_metrics).to_json(
        out_dir / 'q1_seed_metrics.json', orient='records', indent=2)
    all_models[0].get_booster().save_model(str(out_dir / 'xgb_classifier_q1.json'))

    metric_df = pd.DataFrame(all_metrics)
    means = metric_df.mean(numeric_only=True); stds = metric_df.std(numeric_only=True)

    print('\nGenerating plots...')
    metrics_summary = [
        ('galaxy_precision', 'Galaxy Precision', means['galaxy_precision'], stds['galaxy_precision']),
        ('galaxy_recall',    'Galaxy Recall',    means['galaxy_recall'],    stds['galaxy_recall']),
        ('galaxy_f1',        'Galaxy F1',        means['galaxy_f1'],        stds['galaxy_f1']),
    ]
    plot_feature_importance(np.array(all_importances), SELECTED_GAIA_PARAMETERS,
                            out_dir, metrics_summary=metrics_summary)
    # Threshold analysis on the UNFILTERED test set: pass full X, y so each
    # seed re-derives its same canonical test split internally.
    plot_threshold_analysis(X, y, all_models, SEEDS, out_dir)

    with open(out_dir / 'q1_summary.txt', 'w') as f:
        f.write('XGBOOST Q1 — PHZ-conf filter on TRAIN+VAL, eval on UNFILTERED test\n')
        f.write('=' * 70 + '\n')
        f.write(f'Train+val filter : (STAR & star_prob>={args.star_prob}) OR '
                f'(GALAXY & gal_prob>={args.gal_prob})\n')
        f.write(f'Test set         : UNFILTERED (same test indices as canonical)\n')
        f.write(f'Magnitude cut    : {MAG_CUT} <= G\n')
        f.write(f'Total pool       : {len(X):,}\n')
        f.write(f'Seed-42 splits   : '
                f'train {n_train_full_42:,} → {n_train_filt_42:,} '
                f'(kept {100*n_train_filt_42/n_train_full_42:.1f}%)  '
                f'val {n_val_full_42:,} → {n_val_filt_42:,} '
                f'(kept {100*n_val_filt_42/n_val_full_42:.1f}%)  '
                f'test {n_test_42:,} (unchanged)\n')
        f.write('\nTest-set metrics (UNFILTERED, identical to canonical test sets):\n')
        for key, label in [
            ('accuracy', 'Accuracy'),
            ('galaxy_precision', 'Galaxy Precision'),
            ('galaxy_recall', 'Galaxy Recall'),
            ('galaxy_f1', 'Galaxy F1'),
            ('star_f1', 'Star F1'),
            ('roc_auc', 'ROC AUC'),
        ]:
            f.write(f'  {label:<16}: {means[key]:.4f} ± {stds[key]:.4f}\n')

    # Comparison with canonical (which used same test sets)
    import re
    def _read(path, label):
        if not path.exists():
            return None
        out = {'label': label}
        for line in path.read_text().splitlines():
            for k, lbl in [('accuracy','Accuracy'),
                           ('galaxy_precision','Galaxy Precision'),
                           ('galaxy_recall','Galaxy Recall'),
                           ('galaxy_f1','Galaxy F1'),
                           ('star_f1','Star F1'),
                           ('roc_auc','ROC AUC')]:
                if lbl in line:
                    m = re.search(r':\s*([\d.]+)', line)
                    if m: out[k] = float(m.group(1))
        return out

    rows = [
        _read(PROJECT_DIR / 'output' / 'xgboost_q1' / 'archive' / '15_crossmatch' / 'q1_summary.txt',
              'old (uncor full)'),
        _read(PROJECT_DIR / 'output' / 'xgboost_q1' / 'q1_summary.txt',
              'new (corr full) [SAME test set]'),
        _read(PROJECT_DIR / 'output' /
              f'xgboost_q1_phz_conf_s{args.star_prob:g}_g{args.gal_prob:g}' / 'q1_summary.txt',
              'PHZ-conf, both filtered'),
        {'label': f'PHZ-conf filt train, UNFILT test',
         'accuracy': means['accuracy'], 'galaxy_precision': means['galaxy_precision'],
         'galaxy_recall': means['galaxy_recall'], 'galaxy_f1': means['galaxy_f1'],
         'star_f1': means['star_f1'], 'roc_auc': means['roc_auc']},
    ]
    cmp_path = out_dir / 'phz_conf_unfilt_comparison.txt'
    with open(cmp_path, 'w') as f:
        f.write('PHZ-CONF FILTER — train-time only — apples-to-apples vs canonical\n')
        f.write('=' * 88 + '\n')
        f.write(f"  {'Source pool':<38} {'Acc':>8} {'GalPrec':>8} {'GalRec':>8} {'GalF1':>8} {'StarF1':>8} {'AUC':>8}\n")
        f.write('  ' + '-' * 86 + '\n')
        for r in rows:
            if r is None: continue
            f.write(f"  {r['label']:<38} "
                    f"{r.get('accuracy', float('nan')):>8.4f} "
                    f"{r.get('galaxy_precision', float('nan')):>8.4f} "
                    f"{r.get('galaxy_recall', float('nan')):>8.4f} "
                    f"{r.get('galaxy_f1', float('nan')):>8.4f} "
                    f"{r.get('star_f1', float('nan')):>8.4f} "
                    f"{r.get('roc_auc', float('nan')):>8.4f}\n")

    print('\n' + cmp_path.read_text())
    print(f'  -> {cmp_path}')
    print('\n' + '=' * 70)
    print('Done.')
    print('=' * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
