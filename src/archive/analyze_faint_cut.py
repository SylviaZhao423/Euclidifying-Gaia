#!/usr/bin/env python3
"""
Faint-end magnitude cut experiment.

Tests whether capping the training sample at a faint magnitude limit
improves model performance, and whether the conventional G=18.5 bright
cut is necessary.

Design:
  - Bright end: G >= 15 (no bright cut)
  - Faint end: varies — G<=19.5, 20.0, 20.5, 21.0, 21.5, no cut (G<=21.75)
  - For each seed: fixed 80/20 stratified split on FULL dataset (G>=15)
  - Training filtered to G <= faint_cut; test set UNCHANGED
  - Metrics reported overall and by test magnitude bin:
      G < 20,  G in [20, 21],  G > 21

Both Q1 and ERO datasets.

Outputs:
  output/cut_sensitivity/faint_cut_q1.png
  output/cut_sensitivity/faint_cut_ero.png
  (+ JSON caches for each)

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_faint_cut.py
    PGUSER=siwei_zhao_2025 python src/analyze_faint_cut.py --retrain
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'output' / 'cut_sensitivity'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

SEEDS     = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]
TEST_SIZE = 0.2
VAL_FRAC  = 0.125

CLASS_MAPPING      = {'STAR': 0, 'GALAXY': 1}
CLASS_STAR_THRESH  = 0.65
MIN_GAL_IN_BIN     = 5

# Faint cuts to test; None = no faint cut (include all sources to G~21.75)
FAINT_CUTS   = [19.5, 20.0, 20.5, 21.0, 21.5, None]
FAINT_X      = [19.5, 20.0, 20.5, 21.0, 21.5, 21.8]   # numeric x for plotting
FAINT_LABELS = ['G≤19.5', 'G≤20.0', 'G≤20.5', 'G≤21.0', 'G≤21.5', 'no cut']

# Fixed test magnitude bins (label, lo, hi, colour)
TEST_BINS = [
    ('all',     15.0, 99.0, 'black'),
    ('G < 20',  15.0, 20.0, 'steelblue'),
    ('G 20–21', 20.0, 21.0, 'mediumseagreen'),
    ('G > 21',  21.0, 99.0, 'coral'),
]

XGB_PARAMS = {
    'objective':             'binary:logistic',
    'max_depth':             8,
    'learning_rate':         0.1,
    'n_estimators':          200,
    'subsample':             0.8,
    'colsample_bytree':      0.8,
    'tree_method':           'hist',
    'eval_metric':           'logloss',
    'early_stopping_rounds': 20,
}


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def query_q1(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT {gaia_cols},
               CASE WHEN p_phz_classification = 1 THEN 'STAR'
                    WHEN p_phz_classification = 2 THEN 'GALAXY' END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE p_phz_classification IN (1, 2)
          AND g_phot_g_mean_mag IS NOT NULL
          AND g_phot_g_mean_mag >= 15
        ORDER BY g_source_id
    """
    print("  Querying Q1 (G >= 15, STAR/GALAXY)...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"  Q1: {len(df):,}  STAR: {(df['label']=='STAR').sum():,}  "
          f"GALAXY: {(df['label']=='GALAXY').sum():,}")
    return df


def query_ero(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT {gaia_cols},
               CASE WHEN e_v_class_star >= {CLASS_STAR_THRESH} THEN 'STAR'
                    WHEN e_v_class_star <  {CLASS_STAR_THRESH} THEN 'GALAXY' END AS label
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_class_star IS NOT NULL
          AND g_phot_g_mean_mag IS NOT NULL
          AND g_phot_g_mean_mag >= 15
        ORDER BY g_source_id
    """
    print("  Querying ERO (G >= 15, class_star threshold)...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"  ERO: {len(df):,}  STAR: {(df['label']=='STAR').sum():,}  "
          f"GALAXY: {(df['label']=='GALAXY').sum():,}")
    return df


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(df, seeds):
    """Fixed 80/20 split per seed; vary faint-end training cut; fixed test set.

    Returns list of per-seed results, each a list over faint cuts:
    {
      'n_train_total': int, 'n_train_gal': int,
      '<bin_label>': {'precision': float, 'recall': float, 'f1': float,
                      'n_gal_test': int}
    }
    """
    X_all   = df[SELECTED_GAIA_PARAMETERS].values.astype(float)
    y_all   = df['label'].map(CLASS_MAPPING).values
    mag_all = df['phot_g_mean_mag'].values

    all_results = []

    for s_idx, seed in enumerate(seeds):
        print(f"  Seed {seed}  ({s_idx+1}/{len(seeds)})")

        # Fixed train/test split — same regardless of faint cut
        idx = np.arange(len(X_all))
        idx_train, idx_test = train_test_split(
            idx, test_size=TEST_SIZE, stratify=y_all, random_state=seed)

        y_test   = y_all[idx_test]
        mag_test = mag_all[idx_test]

        seed_results = []

        for cut in FAINT_CUTS:
            label = f'G<={cut}' if cut is not None else 'no cut'

            # Filter training set by faint cut
            if cut is not None:
                mask = mag_all[idx_train] <= cut
                idx_tr_cut = idx_train[mask]
            else:
                idx_tr_cut = idx_train

            n_train_total = len(idx_tr_cut)
            n_train_gal   = int((y_all[idx_tr_cut] == 1).sum())

            cut_result = {
                'n_train_total': n_train_total,
                'n_train_gal':   n_train_gal,
            }

            if n_train_total < 50 or n_train_gal < MIN_GAL_IN_BIN:
                for bn, *_ in TEST_BINS:
                    cut_result[bn] = {'precision': np.nan, 'recall': np.nan,
                                      'f1': np.nan, 'n_gal_test': 0}
                seed_results.append(cut_result)
                print(f"    {label}: skipped (too few galaxies in train)")
                continue

            idx_tr, idx_val = train_test_split(
                idx_tr_cut, test_size=VAL_FRAC,
                stratify=y_all[idx_tr_cut], random_state=seed)

            model = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
            model.fit(X_all[idx_tr], y_all[idx_tr],
                      eval_set=[(X_all[idx_val], y_all[idx_val])],
                      verbose=False)

            y_pred = model.predict(X_all[idx_test])

            # Metrics on overall test set and each magnitude bin
            for bn, blo, bhi, _ in TEST_BINS:
                bmask  = (mag_test >= blo) & (mag_test < bhi)
                yt     = y_test[bmask]
                yp     = y_pred[bmask]
                n_gal  = int((yt == 1).sum())
                if n_gal < MIN_GAL_IN_BIN or len(yt) < 10:
                    cut_result[bn] = {'precision': np.nan, 'recall': np.nan,
                                      'f1': np.nan, 'n_gal_test': n_gal}
                else:
                    p, r, f1, _ = precision_recall_fscore_support(
                        yt, yp, labels=[0, 1], zero_division=0)
                    cut_result[bn] = {
                        'precision':  float(p[1]),
                        'recall':     float(r[1]),
                        'f1':         float(f1[1]),
                        'n_gal_test': n_gal,
                    }

            seed_results.append(cut_result)

            # Quick print for 'all' bin
            m = cut_result.get('all', {})
            print(f"    {label:12s}  train={n_train_total:>7,}  gal={n_train_gal:>6,}"
                  f"  | all: P={m.get('precision', float('nan')):.3f}"
                  f"  R={m.get('recall', float('nan')):.3f}"
                  f"  F1={m.get('f1', float('nan')):.3f}")

        all_results.append(seed_results)

    return all_results


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _to_serial(obj):
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_serial(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serial(v) for v in obj]
    return obj


def _from_serial(obj):
    if obj is None:
        return np.nan
    if isinstance(obj, dict):
        return {k: _from_serial(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_serial(v) for v in obj]
    return obj


def save_cache(results, cache_file):
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(_to_serial(results), f)
    print(f"  Cache saved: {cache_file}")


def load_cache(cache_file):
    if not cache_file.exists():
        return None
    print(f"  Loading cache: {cache_file}")
    with open(cache_file) as f:
        return _from_serial(json.load(f))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_results(results, dataset_name, output_dir):
    n_seeds = len(results)
    n_cuts  = len(FAINT_CUTS)
    x       = np.array(FAINT_X)

    # Aggregate across seeds → (n_seeds, n_cuts) arrays per bin per metric
    agg = {bn: {m: np.full((n_seeds, n_cuts), np.nan)
                for m in ('precision', 'recall', 'f1')}
           for bn, *_ in TEST_BINS}
    n_train_gal = np.full((n_seeds, n_cuts), np.nan)
    n_train_tot = np.full((n_seeds, n_cuts), np.nan)

    for s, seed_res in enumerate(results):
        for c, cr in enumerate(seed_res):
            if not cr:
                continue
            n_train_gal[s, c] = cr.get('n_train_gal', np.nan)
            n_train_tot[s, c] = cr.get('n_train_total', np.nan)
            for bn, *_ in TEST_BINS:
                if bn in cr:
                    for m in ('precision', 'recall', 'f1'):
                        agg[bn][m][s, c] = cr[bn].get(m, np.nan)

    # ── layout: 2 rows ────────────────────────────────────────────────────
    # Row 1 (tall): P / F1 / R vs faint cut, lines by test bin
    # Row 2 (short): n_train_gal | n_train_total | galaxy fraction in training
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 9),
        gridspec_kw={'height_ratios': [3, 1.6], 'hspace': 0.38})

    metric_info = [
        ('precision', 'Galaxy Precision'),
        ('f1',        'Galaxy F1'),
        ('recall',    'Galaxy Recall'),
    ]

    for col, (metric, ylabel) in enumerate(metric_info):
        ax = axes[0, col]
        for bn, blo, bhi, color in TEST_BINS:
            mu  = np.nanmean(agg[bn][metric], axis=0)
            std = np.nanstd(agg[bn][metric],  axis=0)
            lw  = 2.5 if bn == 'all' else 1.8
            ls  = '-'  if bn == 'all' else '--'
            ax.plot(x, mu, color=color, linewidth=lw, linestyle=ls,
                    marker='o', markersize=6, label=bn, zorder=3)
            ax.fill_between(x, mu - std, mu + std,
                            alpha=0.12, color=color, zorder=2)

        ax.axvline(x=21.0, color='dimgray', ls=':', linewidth=1.3,
                   label='G = 21')
        ax.set_xlim(x[0] - 0.15, x[-1] + 0.15)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(FAINT_LABELS, fontsize=9, rotation=20, ha='right')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_xlabel('Faint training cut', fontsize=10, labelpad=5)
        ax.grid(alpha=0.3, axis='y')
        if col == 0:
            ax.legend(title='Test mag bin', fontsize=9, loc='lower right',
                      framealpha=0.9)

    # ── Row 2: training sample stats ──────────────────────────────────────
    stats = [
        (axes[1, 0], np.nanmean(n_train_gal, axis=0),
         np.nanstd(n_train_gal, axis=0),  'orchid',        'N galaxies in train'),
        (axes[1, 1], np.nanmean(n_train_tot, axis=0),
         np.nanstd(n_train_tot, axis=0),  'steelblue',     'N total in train'),
    ]
    for ax, mu, std, color, ylabel in stats:
        ax.plot(x, mu, color=color, linewidth=2, marker='o', markersize=6)
        ax.fill_between(x, mu - std, mu + std, alpha=0.2, color=color)
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(FAINT_LABELS, fontsize=9, rotation=20, ha='right')
        ax.set_xlabel('Faint training cut', fontsize=10, labelpad=5)
        ax.set_xlim(x[0] - 0.15, x[-1] + 0.15)
        ax.grid(alpha=0.3, axis='y')
        ax.axvline(x=21.0, color='dimgray', ls=':', linewidth=1.3)
        ax.set_ylim(0, None)

    # Galaxy fraction in training
    mu_gal  = np.nanmean(n_train_gal, axis=0)
    mu_tot  = np.nanmean(n_train_tot, axis=0)
    gal_frac = mu_gal / (mu_tot + 1e-9)
    ax = axes[1, 2]
    ax.plot(x, gal_frac, color='mediumseagreen', linewidth=2, marker='o', markersize=6)
    ax.set_ylabel('Galaxy fraction in train', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(FAINT_LABELS, fontsize=9, rotation=20, ha='right')
    ax.set_xlabel('Faint training cut', fontsize=10, labelpad=5)
    ax.set_xlim(x[0] - 0.15, x[-1] + 0.15)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3, axis='y')
    ax.axvline(x=21.0, color='dimgray', ls=':', linewidth=1.3)

    plt.suptitle(
        f'{dataset_name} model \u2014 effect of faint-end training cut  '
        f'(G\u226515 bright end, {n_seeds} seeds, mean \u00b1 1\u03c3)',
        fontsize=13, fontweight='bold', y=1.01)

    out = output_dir / f'faint_cut_{dataset_name.lower()}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    retrain  = '--retrain' in sys.argv
    username = os.getenv('PGUSER')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_models = len(FAINT_CUTS) * len(SEEDS)
    print("=" * 65)
    print("Faint-end cut experiment")
    print(f"  {len(FAINT_CUTS)} faint cuts × {len(SEEDS)} seeds = {n_models} models per dataset")
    print("=" * 65)

    for dataset, cache_name, query_fn in [
        ('Q1',  'faint_cut_q1.json',  query_q1),
        ('ERO', 'faint_cut_ero.json', query_ero),
    ]:
        cache_file = OUTPUT_DIR / cache_name
        print(f"\n{'='*20} {dataset} {'='*20}")

        if not retrain and cache_file.exists():
            results = load_cache(cache_file)
        else:
            if not username:
                print("ERROR: PGUSER not set"); return 1
            df = query_fn(username)
            print(f"\n  Running {n_models} models...")
            results = run_experiment(df, SEEDS)
            save_cache(results, cache_file)

        print(f"\n  Plotting {dataset}...")
        plot_results(results, dataset, OUTPUT_DIR)

    print(f"\nAll outputs in: {OUTPUT_DIR}")
    print("=" * 65)
    return 0


if __name__ == '__main__':
    sys.exit(main())
