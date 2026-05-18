#!/usr/bin/env python3
"""
Evaluate the 4 trained Q1 ensembles against DESI spectroscopic ground
truth. DESI Q1 only contains GALAXY-targeted spectra (BGS/LRG/ELG),
so we measure the model's ability to recover TRUE galaxies, including
the 85 sources where PHZ said STAR but DESI confirms GALAXY.

For each seed, restrict the DESI-confirmed sample to sources held-OUT
of that seed's 15% test split — keeps the evaluation unbiased.
Aggregate across 10 seeds.

Models compared:
  - old uncor full      (output/xgboost_q1/archive/15_crossmatch/seed_models/)
  - new corr full       (output/xgboost_q1/seed_models/)
  - intersection        (output/xgboost_q1_intersection/seed_models/)
  - PHZ-conf filt train (output/xgboost_q1_phz_conf_s0.5_g0.5_eval_unfilt/seed_models/)

Outputs:
  output/xgboost_q1_desi_eval/
    desi_evaluation.txt
    desi_evaluation.csv

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_q1_models_vs_desi.py
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
    CLASS_MAPPING, MAG_CUT, SEEDS, WSDB_DB, WSDB_HOST,
)

PROJECT_DIR = Path(__file__).parent.parent
OUT_DIR     = PROJECT_DIR / 'output' / 'xgboost_q1_desi_eval'

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


def query_desi_ground_truth(username):
    """All DESI-spectro-confirmed Q1 sources (G ≥ 17, PHZ ∈ {1,2})."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{c} AS {c}' for c in SELECTED_GAIA_PARAMETERS])
    df = pd.read_sql_query(f"""
        SELECT
            g_source_id,
            g_random_index,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            {gaia_cols},
            p_phz_classification AS phz_cls,
            d_desi_spectype AS desi_truth
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep_desi
        WHERE g_phot_g_mean_mag >= {MAG_CUT}
          AND p_phz_classification IN (1, 2)
          AND d_desi_spectype IN ('STAR','GALAXY')
        ORDER BY g_random_index
    """, conn)
    conn.close()
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def query_full_pool_indexer(username):
    """Same query as canonical train_xgboost_q1.py — gives the same 244,556
    source indices used by the per-seed train/test split. We need this so
    each seed's test indices can be reconstructed and intersected with
    DESI-confirmed sources."""
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
    print('Q1 models vs DESI spectroscopic ground truth')
    print('=' * 70)

    print('\n[1] Querying DESI-confirmed Q1 sources...')
    df_desi = query_desi_ground_truth(username)
    n_total = len(df_desi)
    n_phz_star_desi_gal = int(((df_desi['phz_cls']==1) & (df_desi['desi_truth']=='GALAXY')).sum())
    n_phz_gal_desi_gal  = int(((df_desi['phz_cls']==2) & (df_desi['desi_truth']=='GALAXY')).sum())
    print(f'  -> {n_total:,} DESI-confirmed Q1 sources (G≥17, PHZ ∈ {{1,2}})')
    print(f'     PHZ STAR & DESI GALAXY (PHZ wrong): {n_phz_star_desi_gal:,}')
    print(f'     PHZ GAL  & DESI GALAXY (PHZ right): {n_phz_gal_desi_gal:,}')

    X_desi = df_desi[SELECTED_GAIA_PARAMETERS].values
    y_truth = (df_desi['desi_truth'] == 'GALAXY').astype(int).values  # 1 if galaxy
    desi_source_ids = df_desi['g_source_id'].values

    print('\n[2] Reconstructing per-seed test splits to find DESI ∩ test...')
    df_full = query_full_pool_indexer(username)
    full_source_ids = df_full['g_source_id'].values
    full_y = df_full['y'].values
    n_full = len(full_y)
    print(f'  Full canonical pool: {n_full:,} sources (matches train_xgboost_q1.py)')

    # Map DESI g_source_ids → indices in the canonical full pool
    full_id_to_idx = {sid: i for i, sid in enumerate(full_source_ids)}
    desi_in_full = np.array(
        [full_id_to_idx.get(sid, -1) for sid in desi_source_ids])
    in_pool_mask = desi_in_full >= 0
    n_in_pool = int(in_pool_mask.sum())
    print(f'  DESI sources in canonical pool: {n_in_pool}/{n_total}')

    # Per-seed: which DESI sources are in this seed's TEST split?
    idx_all = np.arange(n_full)
    per_seed_test_mask = []
    for seed in SEEDS:
        _, idx_test = train_test_split(idx_all, test_size=0.15,
                                        random_state=seed, stratify=full_y)
        test_set = set(idx_test.tolist())
        # mask: which DESI sources are in this seed's test split?
        mask = np.array([
            (i >= 0 and i in test_set) for i in desi_in_full])
        per_seed_test_mask.append(mask)
    n_per_seed = [int(m.sum()) for m in per_seed_test_mask]
    print(f'  Per-seed DESI ∩ test: avg {np.mean(n_per_seed):.0f}  '
          f'min {min(n_per_seed)}  max {max(n_per_seed)}')

    print('\n[3] Loading and evaluating models...')
    rows = []
    for label, model_dir in MODELS:
        if not model_dir.exists():
            print(f'  SKIP {label}: dir not found')
            continue
        # Determine which seed model files this dir actually has
        model_files = {seed: model_dir / f'xgb_classifier_q1_seed{seed}.json'
                       for seed in SEEDS}
        available = [s for s, p in model_files.items() if p.exists()]
        print(f'  [{label}] {len(available)} seed models available: {available[:3]}...')

        per_seed_acc, per_seed_n, per_seed_recall_phz_wrong = [], [], []
        for si, seed in enumerate(SEEDS):
            if seed not in available:
                continue
            booster = xgb.Booster()
            booster.load_model(str(model_files[seed]))
            mask = per_seed_test_mask[si]
            if mask.sum() == 0:
                continue
            X_eval = X_desi[mask]
            y_eval = y_truth[mask]   # all 1 (galaxy) for DESI Q1
            phz_eval = df_desi['phz_cls'].values[mask]

            proba = predict_proba_booster(booster, X_eval)
            pred  = (proba >= 0.5).astype(int)

            acc = float((pred == y_eval).mean())  # = galaxy recall (all truth=1)
            per_seed_acc.append(acc)
            per_seed_n.append(int(mask.sum()))

            # Recall on the PHZ-WRONG subset only (PHZ=STAR but DESI=GALAXY)
            wrong_mask = (phz_eval == 1)
            if wrong_mask.sum() > 0:
                rec_wrong = float((pred[wrong_mask] == 1).mean())
                per_seed_recall_phz_wrong.append((rec_wrong, int(wrong_mask.sum())))

        if not per_seed_acc:
            continue
        accs = np.array(per_seed_acc)
        rows.append({
            'model': label,
            'n_seeds_used': len(per_seed_acc),
            'avg_n_per_seed': float(np.mean(per_seed_n)),
            'galaxy_recall_mean': float(accs.mean()),
            'galaxy_recall_std':  float(accs.std()),
            'galaxy_recall_min':  float(accs.min()),
            'galaxy_recall_max':  float(accs.max()),
            'phz_wrong_subset_recall_mean':
                float(np.mean([r for r,_ in per_seed_recall_phz_wrong]))
                if per_seed_recall_phz_wrong else float('nan'),
            'phz_wrong_subset_n_avg':
                float(np.mean([n for _,n in per_seed_recall_phz_wrong]))
                if per_seed_recall_phz_wrong else 0.0,
        })

    df_out = pd.DataFrame(rows)
    csv_path = OUT_DIR / 'desi_evaluation.csv'
    df_out.to_csv(csv_path, index=False)

    txt = []
    txt.append('Q1 models vs DESI spectroscopic ground truth')
    txt.append('=' * 95)
    txt.append('Test pool : per-seed DESI ∩ test  (DESI sources held-out from that seed\'s training)')
    txt.append(f'Truth     : DESI spectype = GALAXY  (only galaxies in DESI Q1; n={n_total:,} total)')
    txt.append(f'Subgroup  : PHZ STAR but DESI says GALAXY = PHZ misclassifications ({n_phz_star_desi_gal} sources)')
    txt.append('')
    txt.append(f'  {"model":<25} {"n_per_seed":>10} {"GalRecall":>20}     {"PHZ-wrong recall":>20}')
    txt.append('  ' + '-' * 90)
    for r in rows:
        rec = f"{r['galaxy_recall_mean']:.4f} ± {r['galaxy_recall_std']:.4f}"
        rw  = (f"{r['phz_wrong_subset_recall_mean']:.4f} (n≈{r['phz_wrong_subset_n_avg']:.0f})"
               if r['phz_wrong_subset_n_avg'] > 0 else 'no PHZ-wrong in tests')
        txt.append(f"  {r['model']:<25} {r['avg_n_per_seed']:>10.0f}     {rec:<20}     {rw:>20}")

    out = '\n'.join(txt) + '\n'
    (OUT_DIR / 'desi_evaluation.txt').write_text(out)
    print('\n' + out)
    print(f'  -> {csv_path}')
    print(f'  -> {OUT_DIR / "desi_evaluation.txt"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
