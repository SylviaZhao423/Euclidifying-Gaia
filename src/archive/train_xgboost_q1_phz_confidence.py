#!/usr/bin/env python3
"""
Q1 XGBoost ensemble with PHZ-confidence filtered training set.

Filter applied IN ADDITION to the standard PHZ ∈ {1,2}, G ≥ 17:
    (class = 1 STAR   AND p_phz_star_prob >= STAR_PROB_THRESH)
 OR (class = 2 GALAXY AND p_phz_gal_prob  >= GAL_PROB_THRESH)

Default thresholds: 0.5 / 0.5 (configurable via CLI).

Also produces histograms of p_phz_star_prob (class 1) and p_phz_gal_prob
(class 2) over the FULL pool (no extra filter), with the chosen
thresholds marked, so you can sanity-check where the cuts fall.

Outputs to output/xgboost_q1_phz_conf_s{star}_g{gal}/ :
  q1_phz_prob_histogram.png
  xgb_classifier_q1.json
  seed_models/*.json
  q1_seed_metrics.json
  q1_summary.txt
  q1_feature_importance.{png,csv}
  q1_threshold_analysis.png
  phz_conf_comparison.txt   — vs old uncor / new full corr / intersection

Usage:
    PGUSER=siwei_zhao_2025 python src/train_xgboost_q1_phz_confidence.py
    PGUSER=siwei_zhao_2025 python src/train_xgboost_q1_phz_confidence.py --star-prob 0.7 --gal-prob 0.7
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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


def query_q1_with_conf(username, star_th, gal_th):
    """Q1 PHZ {1,2}, G≥17, with star_prob/gal_prob confidence filter."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{c} AS {c}' for c in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT
            g_random_index AS random_index,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            {gaia_cols},
            p_phz_classification,
            p_phz_star_prob,
            p_phz_gal_prob,
            CASE
                WHEN p_phz_classification = 1 THEN 'STAR'
                WHEN p_phz_classification = 2 THEN 'GALAXY'
            END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep
        WHERE p_phz_classification IN (1, 2)
          AND g_phot_g_mean_mag >= {MAG_CUT}
          AND (
              (p_phz_classification = 1 AND p_phz_star_prob >= {star_th})
           OR (p_phz_classification = 2 AND p_phz_gal_prob  >= {gal_th})
          )
        ORDER BY g_random_index
    """
    print(f"  Querying Q1 (PHZ confidence filter: star≥{star_th}, gal≥{gal_th})...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    X = df[SELECTED_GAIA_PARAMETERS].values
    y = df['label'].map(CLASS_MAPPING).values
    n_star, n_gal = (y == 0).sum(), (y == 1).sum()
    print(f"  -> {len(X):,} sources  STAR: {n_star:,} ({100*n_star/len(X):.1f}%)  "
          f"GALAXY: {n_gal:,} ({100*n_gal/len(X):.1f}%)")
    return X, y


def query_q1_phz_probs_unfiltered(username):
    """Pull star_prob (class 1) and gal_prob (class 2) for the full PHZ ∈ {1,2}, G≥17 pool."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    df = pd.read_sql_query(f"""
        SELECT p_phz_classification AS cls,
               p_phz_star_prob AS star_prob,
               p_phz_gal_prob  AS gal_prob
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep
        WHERE p_phz_classification IN (1, 2)
          AND g_phot_g_mean_mag >= {MAG_CUT}
    """, conn)
    conn.close()
    return df


def plot_phz_prob_histograms(df, star_th, gal_th, output_path):
    """Two-panel hist: star_prob for class=1 (left), gal_prob for class=2 (right)."""
    star_probs = df.loc[df['cls'] == 1, 'star_prob'].values
    gal_probs  = df.loc[df['cls'] == 2, 'gal_prob'].values

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.linspace(0, 1, 51)

    n_star_kept = int((star_probs >= star_th).sum())
    n_star_drop = len(star_probs) - n_star_kept
    n_gal_kept  = int((gal_probs >= gal_th).sum())
    n_gal_drop  = len(gal_probs) - n_gal_kept

    axL.hist(star_probs, bins=bins, color='coral', edgecolor='white', alpha=0.85)
    axL.axvline(star_th, color='black', ls='--', lw=1.5,
                label=f'cut = {star_th:.2f}\n'
                      f'kept: {n_star_kept:,} ({100*n_star_kept/len(star_probs):.1f}%)\n'
                      f'drop: {n_star_drop:,} ({100*n_star_drop/len(star_probs):.1f}%)')
    axL.set_yscale('log')
    axL.set_xlabel('p_phz_star_prob', fontsize=12, fontweight='bold')
    axL.set_ylabel('Count (log)', fontsize=12, fontweight='bold')
    axL.set_title(f'Class = STAR (n={len(star_probs):,})', fontsize=12, fontweight='bold')
    axL.legend(fontsize=10, loc='upper left')
    axL.grid(alpha=0.3)

    axR.hist(gal_probs, bins=bins, color='steelblue', edgecolor='white', alpha=0.85)
    axR.axvline(gal_th, color='black', ls='--', lw=1.5,
                label=f'cut = {gal_th:.2f}\n'
                      f'kept: {n_gal_kept:,} ({100*n_gal_kept/len(gal_probs):.1f}%)\n'
                      f'drop: {n_gal_drop:,} ({100*n_gal_drop/len(gal_probs):.1f}%)')
    axR.set_yscale('log')
    axR.set_xlabel('p_phz_gal_prob', fontsize=12, fontweight='bold')
    axR.set_title(f'Class = GALAXY (n={len(gal_probs):,})', fontsize=12, fontweight='bold')
    axR.legend(fontsize=10, loc='upper left')
    axR.grid(alpha=0.3)

    fig.suptitle(
        f'Q1 PHZ probability distribution (PHZ class ∈ {{1,2}}, G ≥ {MAG_CUT})',
        fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {output_path}")
    return n_star_kept, n_star_drop, n_gal_kept, n_gal_drop


def main():
    parser = argparse.ArgumentParser(description='Q1 model w/ PHZ-confidence filter')
    parser.add_argument('--star-prob', type=float, default=0.5,
                        help='Min p_phz_star_prob for class=1 (default 0.5)')
    parser.add_argument('--gal-prob',  type=float, default=0.5,
                        help='Min p_phz_gal_prob for class=2 (default 0.5)')
    args = parser.parse_args()

    username = os.getenv('PGUSER')
    if not username:
        print('ERROR: PGUSER not set'); return 1

    out_dir = (PROJECT_DIR / 'output' /
               f'xgboost_q1_phz_conf_s{args.star_prob:g}_g{args.gal_prob:g}')
    seed_models_dir = out_dir / 'seed_models'
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_models_dir.mkdir(exist_ok=True)

    print('=' * 70)
    print(f'Q1 XGBoost — PHZ confidence filter')
    print(f'  star_prob >= {args.star_prob}   gal_prob >= {args.gal_prob}')
    print(f'  Output : {out_dir}')
    print('=' * 70)

    # Histograms first (full pool, unfiltered) to visualise the cut.
    print('\n[0] Building PHZ-prob histograms over the full PHZ ∈ {1,2}, G≥17 pool...')
    df_probs = query_q1_phz_probs_unfiltered(username)
    n_sk, n_sd, n_gk, n_gd = plot_phz_prob_histograms(
        df_probs, args.star_prob, args.gal_prob,
        out_dir / 'q1_phz_prob_histogram.png')
    print(f'  STAR  kept {n_sk:,} / dropped {n_sd:,}  '
          f'({100*n_sk/(n_sk+n_sd):.1f}% / {100*n_sd/(n_sk+n_sd):.1f}%)')
    print(f'  GAL   kept {n_gk:,} / dropped {n_gd:,}  '
          f'({100*n_gk/(n_gk+n_gd):.1f}% / {100*n_gd/(n_gk+n_gd):.1f}%)')

    print('\n[1] Querying filtered Q1 training pool...')
    X, y = query_q1_with_conf(username, args.star_prob, args.gal_prob)

    all_metrics, all_importances, all_models = [], [], []
    split_summary = None

    for idx, seed in enumerate(SEEDS, start=1):
        X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(X, y, seed)
        n_st, n_gl = (y_tr == 0).sum(), (y_tr == 1).sum()
        ratio = n_st / max(n_gl, 1)
        print(f"  [seed {seed} {idx}/{len(SEEDS)}]  "
              f"train={len(X_tr):,}  val={len(X_v):,}  test={len(X_te):,}  ratio={ratio:.1f}:1")
        model = train_model(X_tr, y_tr, X_v, y_v, verbose=False)
        m = evaluate(model, X_te, y_te)
        m_clean = {k: v for k, v in m.items() if k != 'report'}
        m_clean['seed'] = seed
        all_metrics.append(m_clean)
        all_importances.append(model.feature_importances_)
        all_models.append(model)
        if seed == 42:
            split_summary = {'ratio': ratio, 'n_train': len(X_tr),
                             'n_val': len(X_v), 'n_test': len(X_te)}
        model.get_booster().save_model(
            str(seed_models_dir / f'xgb_classifier_q1_seed{seed}.json'))

    pd.DataFrame(all_metrics).to_json(
        out_dir / 'q1_seed_metrics.json', orient='records', indent=2)
    all_models[0].get_booster().save_model(str(out_dir / 'xgb_classifier_q1.json'))

    metric_df = pd.DataFrame(all_metrics)
    means = metric_df.mean(numeric_only=True)
    stds  = metric_df.std(numeric_only=True)

    print('\nGenerating plots...')
    metrics_summary = [
        ('galaxy_precision', 'Galaxy Precision', means['galaxy_precision'], stds['galaxy_precision']),
        ('galaxy_recall',    'Galaxy Recall',    means['galaxy_recall'],    stds['galaxy_recall']),
        ('galaxy_f1',        'Galaxy F1',        means['galaxy_f1'],        stds['galaxy_f1']),
    ]
    plot_feature_importance(np.array(all_importances), SELECTED_GAIA_PARAMETERS,
                            out_dir, metrics_summary=metrics_summary)
    plot_threshold_analysis(X, y, all_models, SEEDS, out_dir)

    # Summary
    with open(out_dir / 'q1_summary.txt', 'w') as f:
        f.write('XGBOOST Q1 — PHZ-confidence filtered\n')
        f.write('=' * 70 + '\n')
        f.write(f'Filter        : (STAR & star_prob>={args.star_prob}) OR '
                f'(GALAXY & gal_prob>={args.gal_prob})\n')
        f.write(f'Magnitude cut : {MAG_CUT} <= G\n')
        f.write(f'Total sources : {len(X):,}\n')
        if split_summary:
            f.write(f'Train ratio   : {split_summary["ratio"]:.1f}:1  (STAR:GALAXY)\n')
            f.write(f'Train / Val / Test : {split_summary["n_train"]:,} / '
                    f'{split_summary["n_val"]:,} / {split_summary["n_test"]:,}\n')
        f.write('\nTest-set metrics:\n')
        for key, label in [
            ('accuracy', 'Accuracy'),
            ('galaxy_precision', 'Galaxy Precision'),
            ('galaxy_recall', 'Galaxy Recall'),
            ('galaxy_f1', 'Galaxy F1'),
            ('star_f1', 'Star F1'),
            ('roc_auc', 'ROC AUC'),
        ]:
            f.write(f'  {label:<16}: {means[key]:.4f} ± {stds[key]:.4f}\n')

    # Comparison
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
              'new (corr full)'),
        _read(PROJECT_DIR / 'output' / 'xgboost_q1_intersection' / 'q1_summary.txt',
              'intersection'),
        {'label': f'PHZ conf s>={args.star_prob} g>={args.gal_prob}',
         'accuracy': means['accuracy'], 'galaxy_precision': means['galaxy_precision'],
         'galaxy_recall': means['galaxy_recall'], 'galaxy_f1': means['galaxy_f1'],
         'star_f1': means['star_f1'], 'roc_auc': means['roc_auc']},
    ]
    cmp_path = out_dir / 'phz_conf_comparison.txt'
    with open(cmp_path, 'w') as f:
        f.write('PHZ-CONFIDENCE FILTER vs other Q1 ensembles\n')
        f.write('=' * 78 + '\n')
        f.write(f"  {'Source pool':<28} {'Acc':>8} {'GalPrec':>8} {'GalRec':>8} {'GalF1':>8} {'StarF1':>8} {'AUC':>8}\n")
        f.write('  ' + '-' * 76 + '\n')
        for r in rows:
            if r is None: continue
            f.write(f"  {r['label']:<28} "
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
