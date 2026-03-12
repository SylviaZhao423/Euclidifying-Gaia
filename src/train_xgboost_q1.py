#!/usr/bin/env python3
"""
XGBoost Q1-only model training & evaluation.

Trains a 10-seed XGBoost ensemble (STAR / GALAXY) on Q1 PHZ-labelled
Gaia-matched sources with a G >= 17 bright-end magnitude cut.

Split: 70 % train / 15 % val (early stopping) / 15 % test.

Outputs saved to output/xgboost_q1/:
  xgb_classifier_q1.json        — baseline seed-42 model
  seed_models/*.json            — per-seed models
  q1_feature_importance.png     — horizontal bar chart
  q1_feature_importance.csv     — importances table
  q1_seed_metrics.json          — per-seed metrics summary
  q1_summary.txt                — ensemble metrics summary

Usage:
    PGUSER=siwei_zhao_2025 python src/train_xgboost_q1.py [--test-mode]
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'output' / 'xgboost_q1'
SEED_MODELS_DIR = OUTPUT_DIR / 'seed_models'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

CLASS_MAPPING = {'STAR': 0, 'GALAXY': 1}
CLASS_NAMES   = ['STAR', 'GALAXY']

MAG_CUT = 17   # Gaia G bright-end cut
SEEDS   = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]

XGB_PARAMS = {
    'objective':             'binary:logistic',
    'max_depth':             8,
    'learning_rate':         0.1,
    'n_estimators':          200,
    'subsample':             0.8,
    'colsample_bytree':      0.8,
    'random_state':          42,
    'tree_method':           'hist',
    'eval_metric':           'logloss',
    'early_stopping_rounds': 20,
}


# =============================================================================
# Data loading
# =============================================================================

def query_q1(username, test_mode=False):
    """Query Q1 PHZ sources (STAR/GALAXY) with Gaia G >= MAG_CUT."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT
            g_random_index AS random_index,
            {gaia_cols},
            CASE
                WHEN p_phz_classification = 1 THEN 'STAR'
                WHEN p_phz_classification = 2 THEN 'GALAXY'
            END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE p_phz_classification IN (1, 2)
          AND g_phot_g_mean_mag >= {MAG_CUT}
        ORDER BY g_source_id
    """
    if test_mode:
        query += " LIMIT 10000"

    print(f"  Querying Q1 dataset (G >= {MAG_CUT})...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    X  = df[SELECTED_GAIA_PARAMETERS].values
    y  = df['label'].map(CLASS_MAPPING).values

    n_star   = (y == 0).sum()
    n_galaxy = (y == 1).sum()
    print(f"  -> {len(X):,} sources  STAR: {n_star:,} ({100*n_star/len(X):.1f}%)  "
          f"GALAXY: {n_galaxy:,} ({100*n_galaxy/len(X):.1f}%)")

    return X, y


# =============================================================================
# Training & evaluation
# =============================================================================

def train_model(X_train, y_train, X_val, y_val, verbose=True):
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=verbose)
    return model


def split_data(X, y, seed):
    """Split data into train/val/test using the existing 70/15/15 pattern."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=seed, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(model, X_test, y_test):
    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    report       = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    roc_auc      = roc_auc_score(y_test, y_pred_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1])
    return {
        'accuracy':         (y_test == y_pred).sum() / len(y_test),
        'star_precision':   precision[0],
        'star_recall':      recall[0],
        'star_f1':          f1[0],
        'galaxy_precision': precision[1],
        'galaxy_recall':    recall[1],
        'galaxy_f1':        f1[1],
        'roc_auc':          roc_auc,
        'report':           report,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_threshold_analysis(y_test, y_proba_seeds, output_dir):
    """Precision / Recall / F1 vs threshold on Q1 test set.
    Uses mean probability across all seed models.
    2×2 grid:
      [0,0] P_gal distribution on test set (split by true label)
      [0,1] Precision / Recall / F1 vs threshold
      [1,0] ROC curve
      [1,1] Precision-Recall curve
    Output: q1_threshold_analysis.png
    """
    from sklearn.metrics import roc_curve, auc, precision_recall_curve

    y_proba = np.mean(y_proba_seeds, axis=0)   # ensemble mean
    thresholds = np.linspace(0, 1, 201)

    prec_list, rec_list, f1_list = [], [], []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        tp = ((pred == 1) & (y_test == 1)).sum()
        fp = ((pred == 1) & (y_test == 0)).sum()
        fn = ((pred == 0) & (y_test == 1)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec_list.append(p)
        rec_list.append(r)
        f1_list.append(2*p*r/(p+r) if (p + r) > 0 else 0.0)

    prec_arr = np.array(prec_list)
    rec_arr  = np.array(rec_list)
    f1_arr   = np.array(f1_list)
    best_idx = np.nanargmax(f1_arr)
    best_t   = thresholds[best_idx]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc_val = auc(fpr, tpr)
    pr_prec, pr_rec, pr_thresh = precision_recall_curve(y_test, y_proba)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.subplots_adjust(wspace=0.38)

    # [0] P_gal distribution split by true label
    ax = axes[0]
    bins = np.linspace(0, 1, 101)
    ax.hist(y_proba[y_test == 0], bins=bins, color='coral',     alpha=0.6, lw=0,
            label=f'STAR  (n={( y_test==0).sum():,})')
    ax.hist(y_proba[y_test == 1], bins=bins, color='steelblue', alpha=0.6, lw=0,
            label=f'GALAXY  (n={(y_test==1).sum():,})')
    ax.axvline(0.5,   color='black',        lw=1.5, ls='--', label='Default (0.5)')
    ax.axvline(best_t, color='mediumseagreen', lw=1.5, ls=':',  label=f'Best F1 ({best_t:.2f})')
    ax.set_yscale('log')
    ax.set_xlabel('P_gal', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('P_gal distribution — Q1 test set', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # [1] Precision / Recall / F1 vs threshold
    ax = axes[1]
    ax.plot(thresholds, prec_arr, color='steelblue',      lw=2, label='Precision')
    ax.plot(thresholds, rec_arr,  color='coral',          lw=2, label='Recall')
    ax.plot(thresholds, f1_arr,   color='mediumseagreen', lw=2, label='F1')
    ax.axvline(0.5,   color='black',          lw=1.5, ls='--', label='Default (0.5)')
    ax.axvline(best_t, color='mediumseagreen', lw=1.5, ls=':',  label=f'Best F1 ({best_t:.2f})')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('P_gal threshold', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Metrics vs threshold — Q1 test set', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='center left'); ax.grid(alpha=0.3)

    # [2] ROC curve
    ax = axes[2]
    ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC  (AUC = {roc_auc_val:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, ls='--', label='Random')
    # Mark operating point at default and best-F1 threshold
    idx_def  = np.argmin(np.abs(thresholds - 0.5))
    idx_best = best_idx
    fpr_c, tpr_c = roc_curve(y_test, y_proba)[:2]
    # interpolate to get FPR/TPR at our thresholds
    from scipy.interpolate import interp1d
    roc_thresh = roc_curve(y_test, y_proba)[2]
    ax.scatter(fpr[np.argmin(np.abs(roc_thresh - 0.5))],
               tpr[np.argmin(np.abs(roc_thresh - 0.5))],
               s=80, color='black',          zorder=6, label=f'Default (0.5)')
    ax.scatter(fpr[np.argmin(np.abs(roc_thresh - best_t))],
               tpr[np.argmin(np.abs(roc_thresh - best_t))],
               s=80, color='mediumseagreen', zorder=6, label=f'Best F1 ({best_t:.2f})')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC curve — Q1 test set (GALAXY = positive)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # [3] Precision-Recall curve
    ax = axes[3]
    sc = ax.scatter(pr_rec[:-1], pr_prec[:-1], c=pr_thresh,
                    cmap='viridis', s=8, zorder=5)
    fig.colorbar(sc, ax=ax, pad=0.02).set_label('P_gal threshold', fontsize=9)
    ax.scatter(rec_arr[idx_def],  prec_arr[idx_def],
               s=80, color='black',          zorder=6, label='Default (0.5)')
    ax.scatter(rec_arr[best_idx], prec_arr[best_idx],
               s=80, color='mediumseagreen', zorder=6, label=f'Best F1 ({best_t:.2f})')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall curve — Q1 test set', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle(f'Q1 model threshold analysis  (ensemble mean, {len(y_proba_seeds)} seeds)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = output_dir / 'q1_threshold_analysis.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {out}")


def plot_feature_importance(importances, feature_names, output_dir, metrics_summary=None):
    """Horizontal bar chart of feature importances with seed-to-seed CV."""
    imp = np.atleast_2d(importances)
    mu = imp.mean(axis=0)
    std = imp.std(axis=0)
    cv = std / (mu + 1e-9)

    order = np.argsort(mu)[::-1]
    feats = [feature_names[i] for i in order]
    mu_s = mu[order]
    std_s = std[order]
    cv_s = cv[order]

    n = len(feats)
    fig, ax = plt.subplots(figsize=(11, max(8, n * 0.38)))

    cv_abs_max = 0.5
    norm = plt.Normalize(vmin=0, vmax=cv_abs_max)
    colors = plt.cm.RdYlGn_r(norm(cv_s))

    bars = ax.barh(
        range(n), mu_s, xerr=std_s,
        color=colors, edgecolor='white', capsize=4,
        error_kw={'elinewidth': 1.2, 'ecolor': 'black'},
    )
    ax.set_yticks(range(n))
    ax.set_yticklabels([f'#{i+1}  {f}' for i, f in enumerate(feats)], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance  (mean ± 1σ across seeds)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 0.5)
    ax.set_xticks(np.arange(0, 0.51, 0.1))
    ax.grid(alpha=0.3, axis='x')

    for i, (bar, mean_val, std_val, cv_val) in enumerate(zip(bars, mu_s, std_s, cv_s)):
        label = f'{mean_val:.4f}  CV={cv_val:.2f}'
        ax.text(mean_val + std_val + 0.001, bar.get_y() + bar.get_height() / 2,
                label, va='center', fontsize=8)

    sm = plt.cm.ScalarMappable(
        cmap='RdYlGn_r',
        norm=norm
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label('CV = std / mean', fontsize=9)
    cbar.ax.text(0.4, 1.04, 'unstable', transform=cbar.ax.transAxes,
                 fontsize=8, va='bottom', ha='left', color='darkred')
    cbar.ax.text(0.4, -0.05, 'stable', transform=cbar.ax.transAxes,
                 fontsize=8, va='top', ha='left', color='darkgreen')

    if metrics_summary is not None:
        lines = ['Test-set metrics:']
        for _, label, mean_val, std_val in metrics_summary:
            lines.append(f'{label:<16}: {mean_val:.4f} ± {std_val:.4f}')
        ax.text(
            0.6, 0.05, '\n'.join(lines),
            transform=ax.transAxes,
            fontsize=9.5,
            family='monospace',
            va='bottom',
            ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#bbbbbb', alpha=0.95),
        )

    title = f'Q1 Model Feature Importance Stability  ({len(imp)} seeds, G >= {MAG_CUT})'
    plt.suptitle(title, fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()

    out = output_dir / 'q1_feature_importance.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {out}")

    csv_path = output_dir / 'q1_feature_importance.csv'
    out_df = pd.DataFrame({
        'feature': feats,
        'importance_mean': mu_s,
        'importance_std': std_s,
        'importance_cv': cv_s,
        'rank': range(1, n + 1),
    })
    out_df.to_csv(csv_path, index=False)
    print(f"  -> {csv_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='XGBoost Q1 model training')
    parser.add_argument('--test-mode', action='store_true',
                        help='Limit to 10 k sources for quick testing')
    args = parser.parse_args()

    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SEED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("XGBoost Q1 Model Training")
    print("=" * 70)
    print(f"  Features:       {len(SELECTED_GAIA_PARAMETERS)}")
    print(f"  Magnitude cut:  Gaia G >= {MAG_CUT}")
    print(f"  Output:         {OUTPUT_DIR}")
    print(f"  Training mode:  {len(SEEDS)}-seed ensemble")
    if args.test_mode:
        print("  TEST MODE: limited to 10 k sources")

    # ── Step 1: Query ─────────────────────────────────────────────────────────
    print("\n[1] Querying Q1 data...")
    X, y = query_q1(username, test_mode=args.test_mode)

    # ── Step 2-6: Train/evaluate/save ────────────────────────────────────────
    seeds = SEEDS
    all_metrics = []
    all_importances = []
    all_models = []
    all_y_test_proba = []
    y_test_ref = None
    split_summary = None

    for idx, seed in enumerate(seeds, start=1):
        print(f"\n[2] Splitting data (70/15/15, random_state={seed})...")
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed)

        n_star = (y_train == 0).sum()
        n_galaxy = (y_train == 1).sum()
        ratio = n_star / n_galaxy
        print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
        print(f"  Train STAR:GALAXY ratio  {ratio:.1f}:1")

        verbose = False
        print(f"\n[3] Training model for seed {seed} ({idx}/{len(seeds)})...")
        model = train_model(X_train, y_train, X_val, y_val, verbose=verbose)
        print(f"  Best iteration: {model.best_iteration}  "
              f"Best val logloss: {model.best_score:.4f}")

        print("\n[4] Evaluating on test set...")
        metrics = evaluate(model, X_test, y_test)
        print(f"  Accuracy:         {metrics['accuracy']:.4f}")
        print(f"  Galaxy Precision: {metrics['galaxy_precision']:.4f}")
        print(f"  Galaxy Recall:    {metrics['galaxy_recall']:.4f}")
        print(f"  Galaxy F1:        {metrics['galaxy_f1']:.4f}")
        print(f"  Star F1:          {metrics['star_f1']:.4f}")
        print(f"  ROC AUC:          {metrics['roc_auc']:.4f}")

        metrics_clean = {k: v for k, v in metrics.items() if k != 'report'}
        metrics_clean['seed'] = seed
        all_metrics.append(metrics_clean)
        all_importances.append(model.feature_importances_)
        all_models.append(model)
        all_y_test_proba.append(model.predict_proba(X_test)[:, 1])
        if y_test_ref is None:
            y_test_ref = y_test

        if seed == 42:
            split_summary = {
                'ratio': ratio,
                'n_train': len(X_train),
                'n_val': len(X_val),
                'n_test': len(X_test),
            }

        model_path = SEED_MODELS_DIR / f'xgb_classifier_q1_seed{seed}.json'
        model.get_booster().save_model(str(model_path))
        print(f"  -> {model_path}")

    # ── Step 5: Save baseline model / seed metrics ───────────────────────────
    print("\n[5] Saving model...")
    model_path = OUTPUT_DIR / 'xgb_classifier_q1.json'
    all_models[0].get_booster().save_model(str(model_path))
    print(f"  -> {model_path}")

    metrics_path = OUTPUT_DIR / 'q1_seed_metrics.json'
    pd.DataFrame(all_metrics).to_json(metrics_path, orient='records', indent=2)
    print(f"  -> {metrics_path}")

    # ── Step 6: Plots ─────────────────────────────────────────────────────────
    metric_df = pd.DataFrame(all_metrics)
    metric_means = metric_df.mean(numeric_only=True)
    metric_stds = metric_df.std(numeric_only=True)

    print("\n[6] Generating feature importance plot...")
    metrics_summary = [
        ('galaxy_precision', 'Galaxy Precision', metric_means['galaxy_precision'],
         metric_stds['galaxy_precision']),
        ('galaxy_recall', 'Galaxy Recall', metric_means['galaxy_recall'],
         metric_stds['galaxy_recall']),
        ('galaxy_f1', 'Galaxy F1', metric_means['galaxy_f1'],
         metric_stds['galaxy_f1']),
    ]
    plot_feature_importance(
        np.array(all_importances),
        SELECTED_GAIA_PARAMETERS,
        OUTPUT_DIR,
        metrics_summary=metrics_summary,
    )

    print("\n[6b] Generating threshold analysis plot...")
    plot_threshold_analysis(y_test_ref, np.array(all_y_test_proba), OUTPUT_DIR)

    # ── Step 7: Summary text ──────────────────────────────────────────────────
    summary_path = OUTPUT_DIR / 'q1_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("XGBOOST Q1 MODEL TRAINING SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Magnitude cut : Gaia G >= {MAG_CUT}\n")
        f.write(f"Features      : {len(SELECTED_GAIA_PARAMETERS)}\n")
        f.write(f"Training mode : {len(SEEDS)}-seed ensemble\n")
        if split_summary is not None:
            f.write(f"Train ratio   : {split_summary['ratio']:.1f}:1  (STAR:GALAXY)\n")
            f.write(
                f"Train / Val / Test : {split_summary['n_train']:,} / "
                f"{split_summary['n_val']:,} / {split_summary['n_test']:,}\n"
            )
        f.write("Baseline model: random_state=42\n\n")
        f.write("Parameters:\n")
        for i, p in enumerate(SELECTED_GAIA_PARAMETERS, 1):
            f.write(f"  {i:>2}. {p}\n")
        f.write("\nTest-set metrics:\n")
        for key, label in [
            ('accuracy', 'Accuracy'),
            ('galaxy_precision', 'Galaxy Precision'),
            ('galaxy_recall', 'Galaxy Recall'),
            ('galaxy_f1', 'Galaxy F1'),
            ('star_f1', 'Star F1'),
            ('roc_auc', 'ROC AUC'),
        ]:
            f.write(f"  {label:<16}: {metric_means[key]:.4f} ± {metric_stds[key]:.4f}\n")
    print(f"  -> {summary_path}")

    print("\n" + "=" * 70)
    print("Q1 model training complete!")
    print(f"All results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
