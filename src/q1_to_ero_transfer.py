#!/usr/bin/env python3
"""
Q1 -> ERO transfer: train on full Q1 dataset, evaluate on full ERO dataset.

Asks: does a model trained on Q1 photometric-redshift labels generalise
to ERO X-ray morphology labels?

N_SEEDS random 80/20 splits of Q1 are used for training (early stopping on
the 20% Q1 holdout); ERO is always the full external test set.

Outputs (output/q1_to_ero/):
  q1_to_ero_metrics.png          — P/R/F1/AUC vs within-Q1 and within-ERO baselines
  q1_to_ero_prob_classstar.png   — XGB P(galaxy) vs ERO e_v_class_star scatter
  q1_to_ero_mag_stratified.png   — metrics binned by Gaia G magnitude
  q1_to_ero_importance.png       — Q1-trained feature importances

Usage:
    PGUSER=siwei_zhao_2025 python src/q1_to_ero_transfer.py
    PGUSER=siwei_zhao_2025 python src/q1_to_ero_transfer.py --retrain
"""

import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                              roc_curve)

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'output' / 'q1_to_ero'
CACHE_FILE  = OUTPUT_DIR / 'q1_to_ero_results.json'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

SEEDS     = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]
TEST_SIZE = 0.2   # fraction of Q1 used for early-stopping validation

CLASS_STAR_THRESH = 0.65

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

CLASS_MAPPING = {'STAR': 0, 'GALAXY': 1}

MAG_BINS  = [15, 17, 18, 19, 19.5, 20, 20.5, 21, 21.75]


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def query_q1(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{c} AS {c}' for c in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT
            {gaia_cols},
            CASE
                WHEN p_phz_classification = 1 THEN 'STAR'
                WHEN p_phz_classification = 2 THEN 'GALAXY'
            END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE p_phz_classification IN (1, 2)
          AND g_phot_g_mean_mag IS NOT NULL
          AND g_phot_g_mean_mag >= 15
        ORDER BY g_source_id
    """
    print("  Querying Q1 data (G >= 15)...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for c in SELECTED_GAIA_PARAMETERS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    n_star = (df['label'] == 'STAR').sum()
    n_gal  = (df['label'] == 'GALAXY').sum()
    print(f"  Q1: {len(df):,} total  STAR={n_star:,}  GALAXY={n_gal:,}  ratio={n_star/n_gal:.1f}:1")
    return df


def query_ero(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{c} AS {c}' for c in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT
            {gaia_cols},
            e_v_class_star,
            CASE
                WHEN e_v_class_star >= {CLASS_STAR_THRESH} THEN 'STAR'
                WHEN e_v_class_star <  {CLASS_STAR_THRESH} THEN 'GALAXY'
            END AS label
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_class_star IS NOT NULL
          AND g_phot_g_mean_mag IS NOT NULL
          AND g_phot_g_mean_mag >= 15
        ORDER BY g_source_id
    """
    print("  Querying ERO data (G >= 15)...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for c in SELECTED_GAIA_PARAMETERS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    n_star = (df['label'] == 'STAR').sum()
    n_gal  = (df['label'] == 'GALAXY').sum()
    print(f"  ERO: {len(df):,} total  STAR={n_star:,}  GALAXY={n_gal:,}  ratio={n_star/n_gal:.1f}:1")
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_transfer(df_q1, df_ero, seeds):
    """Train on Q1, evaluate on ERO, for each seed."""
    X_ero = df_ero[SELECTED_GAIA_PARAMETERS].values.astype(np.float32)
    y_ero = df_ero['label'].map(CLASS_MAPPING).values

    all_probs   = []   # shape [n_seeds, n_ero]
    all_imps    = []   # shape [n_seeds, n_features]
    all_metrics = []   # list of dicts

    X_q1 = df_q1[SELECTED_GAIA_PARAMETERS].values.astype(np.float32)
    y_q1 = df_q1['label'].map(CLASS_MAPPING).values

    for si, seed in enumerate(seeds):
        print(f"  Seed {seed} ({si+1}/{len(seeds)})...", end=' ', flush=True)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_q1, y_q1, test_size=TEST_SIZE, stratify=y_q1, random_state=seed)

        model = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

        probs = model.predict_proba(X_ero)[:, 1]   # P(galaxy)
        preds = (probs >= 0.5).astype(int)

        p, r, f1, _ = precision_recall_fscore_support(
            y_ero, preds, labels=[1], average=None, zero_division=0)
        auc = roc_auc_score(y_ero, probs)

        all_probs.append(probs.tolist())
        all_imps.append(model.feature_importances_.tolist())
        all_metrics.append({
            'precision': float(p[0]),
            'recall':    float(r[0]),
            'f1':        float(f1[0]),
            'auc':       float(auc),
        })
        print(f"P={p[0]:.3f}  R={r[0]:.3f}  F1={f1[0]:.3f}  AUC={auc:.3f}")

    return all_probs, all_imps, all_metrics


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _to_serialisable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def save_cache(path, all_probs, all_imps, all_metrics,
               ero_class_star, ero_labels, ero_mags):
    data = {
        'all_probs':      all_probs,
        'all_imps':       all_imps,
        'all_metrics':    all_metrics,
        'ero_class_star': [float(v) for v in ero_class_star],
        'ero_labels':     [int(v) for v in ero_labels],
        'ero_mags':       [float(v) for v in ero_mags],
        'features':       SELECTED_GAIA_PARAMETERS,
    }
    with open(path, 'w') as f:
        json.dump(data, f)
    print(f"  Cache saved: {path}")


def load_cache(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_metrics_summary(all_metrics, output_dir):
    """Bar chart of P/R/F1/AUC with mean ± std across seeds."""
    metrics_arr = {k: np.array([m[k] for m in all_metrics])
                   for k in ('precision', 'recall', 'f1', 'auc')}

    # Within-Q1 baseline from cutfaint cache (full dataset, G<=21.75)
    q1_cache_path = PROJECT_DIR / 'output' / 'cutfaint_sensitivity' / 'q1_cutfaint_results.json'

    q1_base = {}
    try:
        q1d = json.load(open(q1_cache_path))
        # all_metrics[seed][cut], last cut = index 27
        q1_f1s = [q1d['all_metrics'][si][27]['f1'] for si in range(10)
                  if q1d['all_metrics'][si][27]['f1'] is not None]
        q1_ps  = [q1d['all_metrics'][si][27]['precision'] for si in range(10)
                  if q1d['all_metrics'][si][27]['precision'] is not None]
        q1_rs  = [q1d['all_metrics'][si][27]['recall'] for si in range(10)
                  if q1d['all_metrics'][si][27]['recall'] is not None]
        q1_base = {'f1': np.mean(q1_f1s), 'precision': np.mean(q1_ps),
                   'recall': np.mean(q1_rs)}
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(8, 5))

    labels  = ['Galaxy\nPrecision', 'Galaxy\nRecall', 'Galaxy\nF1', 'ROC AUC']
    keys    = ['precision', 'recall', 'f1', 'auc']
    x       = np.arange(len(labels))
    width   = 0.25

    means = [metrics_arr[k].mean() for k in keys]
    stds  = [metrics_arr[k].std()  for k in keys]
    bars = ax.bar(x, means, width, yerr=stds, capsize=4,
                  color='steelblue', alpha=0.85, label='Q1 → ERO (transfer)')

    # Within-Q1 baseline lines
    if q1_base:
        for i, k in enumerate(keys[:3]):
            ax.plot([x[i] - width, x[i] + width],
                    [q1_base[k], q1_base[k]],
                    color='orange', lw=2, ls='--')
        ax.plot([], [], color='orange', lw=2, ls='--', label='Within-Q1 (full dataset)')


    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, mean + std + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Metric value', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')
    plt.suptitle(
        f'Q1 → ERO transfer  ({len(all_metrics)} seeds, mean ± 1σ)',
        fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout()
    out = output_dir / 'q1_to_ero_metrics.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_prob_vs_classstar(all_probs, ero_class_star, ero_labels, output_dir):
    """2D histogram of mean XGB P(galaxy) vs ERO e_v_class_star."""
    mean_probs = np.mean(all_probs, axis=0)
    class_star = np.array(ero_class_star)
    labels     = np.array(ero_labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: 2D hex density
    ax = axes[0]
    hb = ax.hexbin(class_star, mean_probs, gridsize=50, cmap='Blues',
                   mincnt=1, bins='log')
    plt.colorbar(hb, ax=ax, label='log10(count)')
    ax.axvline(CLASS_STAR_THRESH, color='tomato', lw=1.5, ls='--',
               label=f'ERO threshold ({CLASS_STAR_THRESH})')
    ax.axhline(0.5, color='steelblue', lw=1.5, ls='--',
               label='XGB threshold (0.5)')
    ax.set_xlabel('ERO e_v_class_star', fontsize=11, fontweight='bold')
    ax.set_ylabel('XGB P(galaxy)  [Q1-trained]', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.set_title('Density', fontsize=11)

    # Right: mean XGB probability in bins of class_star
    ax2 = axes[1]
    bins = np.linspace(0, 1, 21)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_means, bin_stds, bin_n = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (class_star >= lo) & (class_star < hi)
        if mask.sum() > 0:
            bin_means.append(mean_probs[mask].mean())
            bin_stds.append(mean_probs[mask].std())
            bin_n.append(mask.sum())
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
            bin_n.append(0)
    bin_means = np.array(bin_means)
    bin_stds  = np.array(bin_stds)

    ax2.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds,
                     alpha=0.25, color='steelblue')
    ax2.plot(bin_centers, bin_means, color='steelblue', lw=2)
    ax2.axvline(CLASS_STAR_THRESH, color='tomato', lw=1.5, ls='--',
                label=f'ERO threshold ({CLASS_STAR_THRESH})')
    ax2.axhline(0.5, color='steelblue', lw=1.5, ls='--',
                label='XGB threshold (0.5)')
    # Diagonal reference
    ax2.plot([0, 1], [1, 0], color='grey', lw=1, ls=':', label='Perfect anti-correlation')
    ax2.set_xlabel('ERO e_v_class_star', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Mean XGB P(galaxy)  [Q1-trained]', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=9)
    ax2.set_title('Mean XGB prob per class_star bin', fontsize=11)

    plt.suptitle(
        'Q1-trained XGB probability vs ERO morphological class_star',
        fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout()
    out = output_dir / 'q1_to_ero_prob_classstar.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_mag_stratified(all_probs, ero_labels, ero_mags, output_dir):
    """Precision, recall, F1 in bins of Gaia G magnitude for ERO sources."""
    mean_probs = np.mean(all_probs, axis=0)
    preds      = (mean_probs >= 0.5).astype(int)
    labels     = np.array(ero_labels)
    mags       = np.array(ero_mags)

    bins = MAG_BINS
    centers, ps, rs, f1s, ns, n_gals = [], [], [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (mags >= lo) & (mags < hi)
        if mask.sum() < 5:
            continue
        y_true = labels[mask]
        y_pred = preds[mask]
        if len(np.unique(y_true)) < 2:
            continue
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1], average=None, zero_division=0)
        centers.append(0.5 * (lo + hi))
        ps.append(p[0]);  rs.append(r[0]);  f1s.append(f1[0])
        ns.append(mask.sum())
        n_gals.append((y_true == 1).sum())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(centers, ps,  'o-', color='steelblue', lw=2, ms=5, label='Galaxy Precision')
    ax1.plot(centers, f1s, 's-', color='mediumseagreen', lw=2, ms=5,
             label='Galaxy F1')
    ax1.plot(centers, rs,  '^-', color='coral', lw=2, ms=5, label='Galaxy Recall')
    ax1.set_ylabel('Metric value', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=10)

    # Bottom: galaxy fraction per bin
    gal_frac = [ng / n if n > 0 else 0 for ng, n in zip(n_gals, ns)]
    ax2.bar(centers, gal_frac, width=np.diff(bins).mean() * 0.8,
            color='mediumseagreen', alpha=0.6)
    ax2.set_ylabel('Galaxy\nfraction', fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Gaia G magnitude  (ERO test sources)', fontsize=11, fontweight='bold')
    ax2.grid(alpha=0.3)

    # Annotate n per bin
    for c, n in zip(centers, ns):
        ax2.text(c, 0.02, str(n), ha='center', va='bottom', fontsize=7.5, color='darkgreen')

    plt.suptitle(
        'Q1 → ERO transfer — metrics vs Gaia magnitude  (mean XGB probability)',
        fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout()
    out = output_dir / 'q1_to_ero_mag_stratified.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_feature_importance(all_imps, output_dir):
    """Horizontal bar chart of mean feature importances."""
    mean_imps = np.mean(all_imps, axis=0)
    std_imps  = np.std(all_imps, axis=0)
    order     = np.argsort(mean_imps)

    fig, ax = plt.subplots(figsize=(8, 9))
    y = np.arange(len(SELECTED_GAIA_PARAMETERS))
    ax.barh(y, mean_imps[order], xerr=std_imps[order], capsize=3,
            color='steelblue', alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([SELECTED_GAIA_PARAMETERS[i] for i in order], fontsize=8.5)
    ax.set_xlabel('Mean feature importance  (across seeds)', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    plt.suptitle(
        f'Q1-trained model — feature importance  ({len(all_imps)} seeds)',
        fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout()
    out = output_dir / 'q1_to_ero_importance.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain', action='store_true')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    username = os.environ.get('PGUSER', 'siwei_zhao_2025')

    if CACHE_FILE.exists() and not args.retrain:
        print(f"Loading cached results from {CACHE_FILE}")
        data = load_cache(CACHE_FILE)
        all_probs      = data['all_probs']
        all_imps       = data['all_imps']
        all_metrics    = data['all_metrics']
        ero_class_star = data['ero_class_star']
        ero_labels     = data['ero_labels']
        ero_mags       = data['ero_mags']
    else:
        print("=== Querying data ===")
        df_q1  = query_q1(username)
        df_ero = query_ero(username)

        print("\n=== Training Q1 -> ERO transfer ===")
        all_probs, all_imps, all_metrics = run_transfer(df_q1, df_ero, SEEDS)

        ero_class_star = df_ero['e_v_class_star'].tolist()
        ero_labels     = df_ero['label'].map(CLASS_MAPPING).tolist()
        ero_mags       = df_ero['phot_g_mean_mag'].tolist()

        save_cache(CACHE_FILE, all_probs, all_imps, all_metrics,
                   ero_class_star, ero_labels, ero_mags)

    print("\n=== Summary (mean ± std across seeds) ===")
    for k in ('precision', 'recall', 'f1', 'auc'):
        vals = [m[k] for m in all_metrics]
        print(f"  Galaxy {k:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print("\n=== Plotting ===")
    print("[1] Metrics summary...")
    plot_metrics_summary(all_metrics, OUTPUT_DIR)
    print("[2] P(galaxy) vs class_star...")
    plot_prob_vs_classstar(all_probs, ero_class_star, ero_labels, OUTPUT_DIR)
    print("[3] Magnitude-stratified metrics...")
    plot_mag_stratified(all_probs, ero_labels, ero_mags, OUTPUT_DIR)
    print("[4] Feature importance...")
    plot_feature_importance(all_imps, OUTPUT_DIR)

    print(f"\nAll saved to: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == '__main__':
    main()
