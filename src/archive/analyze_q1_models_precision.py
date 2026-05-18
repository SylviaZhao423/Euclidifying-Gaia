#!/usr/bin/env python3
"""
Q1 model PRECISION evaluation: how often does each model wrongly call
a star a galaxy (false positive on the GALAXY class)?

Truth source: PHZ "very confident" STAR sources, p_phz_star_prob >= 0.95.
This is a proxy — not real spectroscopy — but at this threshold PHZ STAR
classification is reliable to ~99%. Comparing models RELATIVELY against
the same proxy is the meaningful thing.

For each seed, restrict the proxy-truth STAR sample to sources held-OUT
of that seed's 15% test split (unbiased evaluation).

Models compared (same as DESI test):
  - old uncor full
  - new corr full
  - intersection
  - PHZ-conf filt train

Outputs:
  output/xgboost_q1_precision_eval/
    precision_evaluation.{txt, csv}

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_q1_models_precision.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS
from train_xgboost_q1 import (
    MAG_CUT, SEEDS, WSDB_DB, WSDB_HOST,
)

PROJECT_DIR = Path(__file__).parent.parent
OUT_DIR     = PROJECT_DIR / 'output' / 'xgboost_q1_precision_eval'

STAR_PROB_THRESH = 0.95  # PHZ-confident-star threshold (proxy truth)

MODELS = [
    ('old uncor full',
     PROJECT_DIR / 'output' / 'xgboost_q1' / 'archive' / '15_crossmatch' / 'seed_models'),
    ('new corr full',
     PROJECT_DIR / 'output' / 'xgboost_q1' / 'seed_models'),
    ('intersection',
     PROJECT_DIR / 'output' / 'xgboost_q1_intersection' / 'seed_models'),
    ('PHZ-conf filt train',
     PROJECT_DIR / 'output' / 'xgboost_q1_phz_conf_s0.5_g0.5_eval_unfilt' / 'seed_models'),
]


def query_proxy_truth_stars(username):
    """PHZ very-confident STARs (proxy 'true stars'). Returns features + g_source_id."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{c} AS {c}' for c in SELECTED_GAIA_PARAMETERS])
    df = pd.read_sql_query(f"""
        SELECT
            g_source_id,
            g_random_index,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            {gaia_cols},
            p_phz_star_prob
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep
        WHERE p_phz_classification = 1
          AND g_phot_g_mean_mag >= {MAG_CUT}
          AND p_phz_star_prob >= {STAR_PROB_THRESH}
        ORDER BY g_random_index
    """, conn)
    conn.close()
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def query_full_pool(username):
    """Same canonical pool as train_xgboost_q1.py (for reconstructing per-seed splits)."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    df = pd.read_sql_query(f"""
        SELECT g_source_id, g_random_index,
               CASE WHEN p_phz_classification = 1 THEN 0 ELSE 1 END AS y
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep
        WHERE p_phz_classification IN (1, 2)
          AND g_phot_g_mean_mag >= {MAG_CUT}
        ORDER BY g_random_index
    """, conn)
    conn.close()
    return df


def predict_proba_booster(booster, X):
    return booster.predict(xgb.DMatrix(X))


def main():
    username = os.getenv('PGUSER')
    if not username:
        print('ERROR: PGUSER not set'); return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 70)
    print(f'Q1 model PRECISION test: PHZ-confident stars (star_prob>={STAR_PROB_THRESH})')
    print('  False positive = model predicts GALAXY for a "true" star')
    print('=' * 70)

    print('\n[1] Querying proxy-truth STAR sample...')
    df_stars = query_proxy_truth_stars(username)
    print(f'  -> {len(df_stars):,} PHZ-very-confident stars (proxy truth)')

    X_stars = df_stars[SELECTED_GAIA_PARAMETERS].values
    star_ids = df_stars['g_source_id'].values

    print('\n[2] Reconstructing per-seed test splits...')
    df_full = query_full_pool(username)
    full_ids = df_full['g_source_id'].values
    full_y = df_full['y'].values
    full_id_to_idx = {sid: i for i, sid in enumerate(full_ids)}
    star_in_full = np.array([full_id_to_idx.get(sid, -1) for sid in star_ids])

    idx_all = np.arange(len(full_y))
    per_seed_test_mask = []
    for seed in SEEDS:
        _, idx_test = train_test_split(idx_all, test_size=0.15,
                                        random_state=seed, stratify=full_y)
        test_set = set(idx_test.tolist())
        mask = np.array([(i >= 0 and i in test_set) for i in star_in_full])
        per_seed_test_mask.append(mask)
    n_per_seed = [int(m.sum()) for m in per_seed_test_mask]
    print(f'  Per-seed proxy-stars ∩ test: avg {np.mean(n_per_seed):.0f}  '
          f'min {min(n_per_seed)}  max {max(n_per_seed)}')

    print('\n[3] Evaluating models...')
    rows = []
    for label, model_dir in MODELS:
        if not model_dir.exists():
            print(f'  SKIP {label}: dir not found'); continue
        per_seed_fp_rate = []
        for si, seed in enumerate(SEEDS):
            mp = model_dir / f'xgb_classifier_q1_seed{seed}.json'
            if not mp.exists():
                continue
            booster = xgb.Booster(); booster.load_model(str(mp))
            mask = per_seed_test_mask[si]
            if mask.sum() == 0:
                continue
            X_eval = X_stars[mask]
            proba  = predict_proba_booster(booster, X_eval)
            pred_galaxy = (proba >= 0.5).astype(int)
            fp_rate = float(pred_galaxy.mean())   # fraction of true stars called galaxy
            per_seed_fp_rate.append(fp_rate)

        if not per_seed_fp_rate:
            continue
        arr = np.array(per_seed_fp_rate)
        rows.append({
            'model': label,
            'n_seeds_used': len(per_seed_fp_rate),
            'avg_n_per_seed': float(np.mean(n_per_seed)),
            'fp_rate_mean': float(arr.mean()),
            'fp_rate_std':  float(arr.std()),
            'fp_rate_min':  float(arr.min()),
            'fp_rate_max':  float(arr.max()),
            'star_recall_mean': float(1 - arr.mean()),  # 1 - FP-on-star rate
        })

    df_out = pd.DataFrame(rows)
    csv_path = OUT_DIR / 'precision_evaluation.csv'
    df_out.to_csv(csv_path, index=False)

    txt = []
    txt.append('Q1 models — PRECISION test (false positives on PHZ-confident stars)')
    txt.append('=' * 95)
    txt.append(f'Truth set : PHZ STAR with star_prob >= {STAR_PROB_THRESH} '
               '(proxy — PHZ at this threshold is ~99% reliable)')
    txt.append('Per-seed  : DESI sources held-out from that seed\'s 15% test split')
    txt.append('Lower FP-rate = BETTER precision  (fewer real stars wrongly called galaxy)')
    txt.append('')
    txt.append(f'  {"model":<25} {"n/seed":>8}  {"FP rate (lower is better)":>32}  {"Star recall":>14}')
    txt.append('  ' + '-' * 90)
    for r in rows:
        fp = f"{r['fp_rate_mean']:.4f} ± {r['fp_rate_std']:.4f}"
        sr = f"{r['star_recall_mean']:.4f}"
        txt.append(f"  {r['model']:<25} {r['avg_n_per_seed']:>8.0f}  {fp:>32}  {sr:>14}")

    out = '\n'.join(txt) + '\n'
    (OUT_DIR / 'precision_evaluation.txt').write_text(out)
    print('\n' + out)
    print(f'  -> {csv_path}')
    print(f'  -> {OUT_DIR / "precision_evaluation.txt"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
