#!/usr/bin/env python3
"""
Controlled test: train Q1 XGBoost ensemble on the INTERSECTION of the
old (uncorrected) and new (PM-corrected) Q1 source pools, then compare
metrics to both the old uncorrected model and the new full PM-corrected
model.

Logic:
  Old pool:    crossmatch_q1_mer_phz_vis_gaia_sep_uncor (PHZ {1,2}, G≥17) ≈ 238,868
  New pool:    crossmatch_q1_mer_phz_vis_gaia_sep       (PHZ {1,2}, G≥17) ≈ 244,556
  Intersection on g_source_id ≈ ~238k (most old sources survive PM correction)

If the intersection model matches the old model:
  → the +5,688 NEW sources brought in by PM correction are responsible
    for the slight degradation (cluster contamination / noisy labels)
If the intersection model is still worse than the old model:
  → something else (label flips, feature distribution shifts) is the cause

Features come from the PM-corrected table (so labels reflect the
PM-corrected closest-match assignment, not the uncorrected one).

Outputs:
    output/xgboost_q1_intersection/
      q1_summary.txt          — metrics
      q1_seed_metrics.json    — per-seed
      seed_models/*.json
      q1_feature_importance.png
      q1_threshold_analysis.png
      intersection_comparison.txt  — vs old + vs full-corr

Usage:
    PGUSER=siwei_zhao_2025 python src/train_xgboost_q1_intersection.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS
from train_xgboost_q1 import (
    CLASS_MAPPING, MAG_CUT, SEEDS, WSDB_DB, WSDB_HOST,
    evaluate, plot_feature_importance, plot_threshold_analysis,
    split_data, train_model,
)

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'output' / 'xgboost_q1_intersection'


def query_q1_intersection(username):
    """Q1 sources that exist in BOTH old (uncor) and new (PM-corr) sep tables.
    Features and labels come from the PM-corrected table."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'corr.g_{c} AS {c}' for c in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT
            corr.g_random_index AS random_index,
            corr.g_phot_g_mean_mag AS phot_g_mean_mag,
            {gaia_cols},
            CASE
                WHEN corr.p_phz_classification = 1 THEN 'STAR'
                WHEN corr.p_phz_classification = 2 THEN 'GALAXY'
            END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep AS corr
        INNER JOIN siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep_uncor AS uncor
            ON corr.g_source_id = uncor.g_source_id
        WHERE corr.p_phz_classification IN (1, 2)
          AND uncor.p_phz_classification IN (1, 2)
          AND corr.g_phot_g_mean_mag >= {MAG_CUT}
        ORDER BY corr.g_random_index
    """
    print("  Querying intersection (g_source_id in BOTH old + new pools)...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    X = df[SELECTED_GAIA_PARAMETERS].values
    y = df['label'].map(CLASS_MAPPING).values
    n_star, n_gal = (y == 0).sum(), (y == 1).sum()
    print(f"  -> {len(X):,} intersection sources  STAR: {n_star:,} ({100*n_star/len(X):.1f}%)  "
          f"GALAXY: {n_gal:,} ({100*n_gal/len(X):.1f}%)")
    return X, y


def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER not set"); return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_models_dir = OUTPUT_DIR / 'seed_models'
    seed_models_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("XGBoost Q1 — INTERSECTION test (old ∩ new source pools)")
    print(f"  Magnitude cut : {MAG_CUT} <= G")
    print(f"  Features      : {len(SELECTED_GAIA_PARAMETERS)}")
    print(f"  Output        : {OUTPUT_DIR}")
    print("=" * 70)

    X, y = query_q1_intersection(username)

    all_metrics, all_importances, all_models = [], [], []
    split_summary = None

    for idx, seed in enumerate(SEEDS, start=1):
        X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(X, y, seed)
        n_star, n_gal = (y_tr == 0).sum(), (y_tr == 1).sum()
        ratio = n_star / max(n_gal, 1)
        print(f"  [seed {seed} {idx}/{len(SEEDS)}]  "
              f"train={len(X_tr):,}  val={len(X_v):,}  test={len(X_te):,}  "
              f"ratio={ratio:.1f}:1")

        model = train_model(X_tr, y_tr, X_v, y_v, verbose=False)
        m = evaluate(model, X_te, y_te)
        m_clean = {k: v for k, v in m.items() if k != 'report'}
        m_clean['seed'] = seed
        all_metrics.append(m_clean)
        all_importances.append(model.feature_importances_)
        all_models.append(model)
        if seed == 42:
            split_summary = {
                'ratio': ratio, 'n_train': len(X_tr),
                'n_val': len(X_v), 'n_test': len(X_te),
            }
        model.get_booster().save_model(
            str(seed_models_dir / f'xgb_classifier_q1_seed{seed}.json'))

    pd.DataFrame(all_metrics).to_json(
        OUTPUT_DIR / 'q1_seed_metrics.json', orient='records', indent=2)
    all_models[0].get_booster().save_model(
        str(OUTPUT_DIR / 'xgb_classifier_q1.json'))

    metric_df = pd.DataFrame(all_metrics)
    means = metric_df.mean(numeric_only=True)
    stds  = metric_df.std(numeric_only=True)

    metrics_summary = [
        ('galaxy_precision', 'Galaxy Precision', means['galaxy_precision'], stds['galaxy_precision']),
        ('galaxy_recall',    'Galaxy Recall',    means['galaxy_recall'],    stds['galaxy_recall']),
        ('galaxy_f1',        'Galaxy F1',        means['galaxy_f1'],        stds['galaxy_f1']),
    ]
    print("\nGenerating plots...")
    plot_feature_importance(np.array(all_importances), SELECTED_GAIA_PARAMETERS,
                            OUTPUT_DIR, metrics_summary=metrics_summary)
    plot_threshold_analysis(X, y, all_models, SEEDS, OUTPUT_DIR)

    # Summary
    with open(OUTPUT_DIR / 'q1_summary.txt', 'w') as f:
        f.write("XGBOOST Q1 — INTERSECTION TEST\n")
        f.write("=" * 70 + "\n")
        f.write("Source pool   : g_source_id in BOTH crossmatch_q1_mer_phz_vis_gaia_sep_uncor\n")
        f.write("                AND crossmatch_q1_mer_phz_vis_gaia_sep, with PHZ {1,2} in both\n")
        f.write("Features      : from PM-corrected table (incl. derived columns)\n")
        f.write(f"Magnitude cut : {MAG_CUT} <= G\n")
        f.write(f"Total sources : {len(X):,}\n")
        if split_summary:
            f.write(f"Train ratio   : {split_summary['ratio']:.1f}:1  (STAR:GALAXY)\n")
            f.write(f"Train / Val / Test : {split_summary['n_train']:,} / "
                    f"{split_summary['n_val']:,} / {split_summary['n_test']:,}\n")
        f.write("\nTest-set metrics:\n")
        for key, label in [
            ('accuracy', 'Accuracy'),
            ('galaxy_precision', 'Galaxy Precision'),
            ('galaxy_recall', 'Galaxy Recall'),
            ('galaxy_f1', 'Galaxy F1'),
            ('star_f1', 'Star F1'),
            ('roc_auc', 'ROC AUC'),
        ]:
            f.write(f"  {label:<16}: {means[key]:.4f} ± {stds[key]:.4f}\n")

    # Comparison block
    cmp_path = OUTPUT_DIR / 'intersection_comparison.txt'
    rows = []

    def _read_summary(path, label):
        if not path.exists():
            return None
        text = path.read_text()
        out = {'label': label}
        import re
        for k, lbl in [('accuracy','Accuracy'),
                       ('galaxy_precision','Galaxy Precision'),
                       ('galaxy_recall','Galaxy Recall'),
                       ('galaxy_f1','Galaxy F1'),
                       ('star_f1','Star F1'),
                       ('roc_auc','ROC AUC')]:
            for line in text.splitlines():
                if lbl in line:
                    m = re.search(r':\s*([\d.]+)\s*', line)
                    if m:
                        out[k] = float(m.group(1))
                        break
        return out

    rows = [
        _read_summary(PROJECT_DIR / 'output' / 'xgboost_q1' / 'archive' /
                      '15_crossmatch' / 'q1_summary.txt', 'old (uncor full)'),
        _read_summary(PROJECT_DIR / 'output' / 'xgboost_q1' / 'q1_summary.txt',
                      'new (corr full)'),
        {'label': 'NEW (corr ∩ old)',
         'accuracy': means['accuracy'], 'galaxy_precision': means['galaxy_precision'],
         'galaxy_recall': means['galaxy_recall'], 'galaxy_f1': means['galaxy_f1'],
         'star_f1': means['star_f1'], 'roc_auc': means['roc_auc']},
    ]

    with open(cmp_path, 'w') as f:
        f.write("INTERSECTION TEST — comparison\n")
        f.write("=" * 75 + "\n")
        header = f"  {'Source pool':<22} {'Acc':>8} {'GalPrec':>8} {'GalRec':>8} {'GalF1':>8} {'StarF1':>8} {'AUC':>8}\n"
        f.write(header)
        f.write("  " + "-" * 73 + "\n")
        for r in rows:
            if r is None:
                continue
            f.write(f"  {r['label']:<22} "
                    f"{r.get('accuracy', float('nan')):>8.4f} "
                    f"{r.get('galaxy_precision', float('nan')):>8.4f} "
                    f"{r.get('galaxy_recall', float('nan')):>8.4f} "
                    f"{r.get('galaxy_f1', float('nan')):>8.4f} "
                    f"{r.get('star_f1', float('nan')):>8.4f} "
                    f"{r.get('roc_auc', float('nan')):>8.4f}\n")

    print("\n" + cmp_path.read_text())
    print(f"  -> {cmp_path}")
    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
