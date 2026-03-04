#!/usr/bin/env python3
"""
Line plot of Galaxy metrics across training ratios.

3×3 grid of subplots:
  Rows: Model dataset (Q1, Combo, ERO)
  Cols: Test dataset (Q1, Combo, ERO)

Within each subplot, x-axis = training ratio (@Q1 → @Combo → @ERO),
with lines for Galaxy Precision, F1, Recall.

Reads results from the training script's all_results dict, or can be
run standalone after train_xgboost_combined.py produces the summary.

Usage:
    PGUSER=siwei_zhao_2025 python src/plot_metrics_lines.py [--test-mode]
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output' / 'xgboost_combined'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

CLASS_MAPPING = {'STAR': 0, 'GALAXY': 1}
CLASS_NAMES = ['STAR', 'GALAXY']

MAG_CUT = 18.5

XGB_PARAMS = {
    'objective': 'binary:logistic',
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist',
    'eval_metric': 'logloss',
    'early_stopping_rounds': 20,
}


# =============================================================================
# Data loading (reused from train_xgboost_combined.py)
# =============================================================================

def query_dataset(username, data_source, test_mode=False):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])

    if data_source == 'q1':
        query = f"""
            SELECT g_random_index AS random_index, {gaia_cols},
                CASE WHEN p_phz_classification = 1 THEN 'STAR'
                     WHEN p_phz_classification = 2 THEN 'GALAXY' END as label
            FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
            WHERE p_phz_classification IN (1, 2) AND g_phot_g_mean_mag >= {MAG_CUT}
            ORDER BY g_source_id"""
    else:
        query = f"""
            SELECT g_random_index AS random_index, {gaia_cols},
                CASE WHEN e_v_class_star >= 0.65 THEN 'STAR'
                     WHEN e_v_class_star < 0.65 THEN 'GALAXY' END as label
            FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
            WHERE e_v_class_star IS NOT NULL AND g_phot_g_mean_mag >= {MAG_CUT}
            ORDER BY g_source_id"""

    if test_mode:
        query += " LIMIT 10000"

    print(f"  Querying {data_source.upper()} dataset...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    X = df[SELECTED_GAIA_PARAMETERS].values
    y = df['label'].map(CLASS_MAPPING).values
    random_index = df['random_index'].values

    n_star = (y == 0).sum()
    n_galaxy = (y == 1).sum()
    print(f"  -> {len(X):,} sources — STAR: {n_star:,}, GALAXY: {n_galaxy:,}")
    return X, y, random_index


def train_model(X_train, y_train, X_val, y_val, verbose=True):
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=verbose)
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1])
    return {
        'accuracy': (y_test == y_pred).sum() / len(y_test),
        'star_precision': precision[0], 'star_recall': recall[0], 'star_f1': f1[0],
        'galaxy_precision': precision[1], 'galaxy_recall': recall[1], 'galaxy_f1': f1[1],
        'roc_auc': roc_auc, 'report': report,
        'y_pred': y_pred, 'y_pred_proba': y_pred_proba,
    }


def downsample_training(X_train, y_train, ri_train, target_ratio, downsample_class):
    star_mask = (y_train == 0)
    galaxy_mask = (y_train == 1)
    if downsample_class == 'galaxy':
        n_galaxy_target = int(star_mask.sum() / target_ratio)
        galaxy_indices = np.where(galaxy_mask)[0]
        sorted_order = np.argsort(ri_train[galaxy_indices])
        keep_galaxy = galaxy_indices[sorted_order[:n_galaxy_target]]
        keep_mask = star_mask.copy()
        keep_mask[keep_galaxy] = True
    else:
        n_star_target = int(galaxy_mask.sum() * target_ratio)
        star_indices = np.where(star_mask)[0]
        sorted_order = np.argsort(ri_train[star_indices])
        keep_star = star_indices[sorted_order[:n_star_target]]
        keep_mask = galaxy_mask.copy()
        keep_mask[keep_star] = True

    X_ds, y_ds = X_train[keep_mask], y_train[keep_mask]
    n_s, n_g = (y_ds == 0).sum(), (y_ds == 1).sum()
    print(f"    Downsampled: STAR={n_s:,}, GALAXY={n_g:,}, Ratio={n_s/n_g:.1f}:1")
    return X_ds, y_ds


# =============================================================================
# Line plot
# =============================================================================

def plot_galaxy_metrics_lines(all_results, output_dir, q1_ratio, combo_ratio, ero_ratio):
    """3×3 grid of line plots showing metrics vs training ratio.

    Rows: Model dataset (Q1, Combo, ERO)
    Cols: Test dataset (Q1, Combo, ERO)
    X-axis per subplot: @Q1 ratio → @Combo ratio → @ERO ratio
    Lines: Galaxy Precision (blue), Galaxy F1 (green, bold), Galaxy Recall (orange)
    """
    model_datasets = ['Q1', 'Comb', 'ERO']
    test_datasets = ['Q1', 'Combo', 'ERO']
    ratio_suffixes = ['@Q1', '@Comb', '@ERO']
    x_ratios = [q1_ratio, combo_ratio, ero_ratio]
    x_labels = [f'@Q1\n({q1_ratio:.0f}:1)', f'@Combo\n({combo_ratio:.0f}:1)',
                f'@ERO\n({ero_ratio:.0f}:1)']

    row_labels = ['Model: Q1', 'Model: Combo', 'Model: ERO']
    col_labels = [f'Test: Q1 ({q1_ratio:.0f}:1)', f'Test: Combo ({combo_ratio:.0f}:1)',
                  f'Test: ERO ({ero_ratio:.0f}:1)']

    # Diagonal: (row_idx, col_idx) where model dataset matches test dataset
    diagonal = {(0, 0), (1, 1), (2, 2)}

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=True, sharey=True)

    for row_idx, model_ds in enumerate(model_datasets):
        for col_idx, test_ds in enumerate(test_datasets):
            ax = axes[row_idx, col_idx]

            prec_vals, f1_vals, rec_vals = [], [], []
            for ratio_suf in ratio_suffixes:
                label = f'{model_ds}{ratio_suf}→{test_ds}'
                r = all_results[label]
                prec_vals.append(r['galaxy_precision'])
                f1_vals.append(r['galaxy_f1'])
                rec_vals.append(r['galaxy_recall'])

            x = np.arange(3)

            ax.plot(x, prec_vals, 'o-', color='steelblue', linewidth=2,
                    markersize=6, label='Galaxy Precision')
            ax.plot(x, f1_vals, 's-', color='mediumseagreen', linewidth=2.5,
                    markersize=7, label='Galaxy F1', zorder=5)
            ax.plot(x, rec_vals, '^-', color='coral', linewidth=2,
                    markersize=6, label='Galaxy Recall')

            # Annotate values
            for i in range(3):
                ax.text(i, prec_vals[i] + 0.02, f'{prec_vals[i]:.2f}',
                        ha='center', va='bottom', fontsize=8, color='steelblue')
                ax.text(i, f1_vals[i] - 0.04, f'{f1_vals[i]:.2f}',
                        ha='center', va='top', fontsize=8, color='mediumseagreen',
                        fontweight='bold')
                ax.text(i, rec_vals[i] + 0.02, f'{rec_vals[i]:.2f}',
                        ha='center', va='bottom', fontsize=8, color='coral')

            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=9)
            ax.set_ylim(0, 1.1)
            ax.grid(alpha=0.3)

            # Highlight diagonal subplots
            if (row_idx, col_idx) in diagonal:
                for spine in ax.spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(2.5)

            # Column headers on top row
            if row_idx == 0:
                ax.set_title(col_labels[col_idx], fontsize=11, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                                       edgecolor='gray', alpha=0.8))

            # Row labels on left column
            if col_idx == 0:
                ax.text(-0.35, 0.5, row_labels[row_idx], ha='center', va='center',
                        fontsize=11, fontweight='bold', rotation=0,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                                  edgecolor='gray', alpha=0.8),
                        transform=ax.transAxes)

    # Single shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 2].legend(handles, labels, fontsize=9, loc='lower left',
                      bbox_to_anchor=(0.55, 1.05), framealpha=0.9)

    plt.suptitle('Galaxy Metrics vs Training Ratio: 27 Experiments',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.06, 0, 1, 0.95])
    plt.savefig(output_dir / 'combined_galaxy_metrics_lines.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> combined_galaxy_metrics_lines.png")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Line plot of galaxy metrics vs training ratio')
    parser.add_argument('--test-mode', action='store_true',
                        help='Use small sample for quick testing')
    args = parser.parse_args()

    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Galaxy Metrics Line Plot (27 experiments)")
    print("=" * 70)

    # Query data
    print("\n[Step 1] Querying datasets...")
    X_q1, y_q1, ri_q1 = query_dataset(username, 'q1', test_mode=args.test_mode)
    X_ero, y_ero, ri_ero = query_dataset(username, 'ero', test_mode=args.test_mode)

    # Split
    print("\n[Step 2] Splitting data...")
    X_q1_temp, X_q1_test, y_q1_temp, y_q1_test, ri_q1_temp, _ = train_test_split(
        X_q1, y_q1, ri_q1, test_size=0.15, random_state=42, stratify=y_q1)
    X_q1_train, X_q1_val, y_q1_train, y_q1_val, ri_q1_train, _ = train_test_split(
        X_q1_temp, y_q1_temp, ri_q1_temp, test_size=0.176, random_state=42, stratify=y_q1_temp)

    X_ero_temp, X_ero_test, y_ero_temp, y_ero_test, ri_ero_temp, _ = train_test_split(
        X_ero, y_ero, ri_ero, test_size=0.15, random_state=42, stratify=y_ero)
    X_ero_train, X_ero_val, y_ero_train, y_ero_val, ri_ero_train, _ = train_test_split(
        X_ero_temp, y_ero_temp, ri_ero_temp, test_size=0.176, random_state=42, stratify=y_ero_temp)

    X_comb_train = np.vstack([X_q1_train, X_ero_train])
    y_comb_train = np.concatenate([y_q1_train, y_ero_train])
    ri_comb_train = np.concatenate([ri_q1_train, ri_ero_train])
    X_comb_val = np.vstack([X_q1_val, X_ero_val])
    y_comb_val = np.concatenate([y_q1_val, y_ero_val])
    X_comb_test = np.vstack([X_q1_test, X_ero_test])
    y_comb_test = np.concatenate([y_q1_test, y_ero_test])

    q1_ratio = (y_q1_train == 0).sum() / (y_q1_train == 1).sum()
    ero_ratio = (y_ero_train == 0).sum() / (y_ero_train == 1).sum()
    combo_ratio = (y_comb_train == 0).sum() / (y_comb_train == 1).sum()

    print(f"  Q1 ratio: {q1_ratio:.1f}:1, Combo ratio: {combo_ratio:.1f}:1, ERO ratio: {ero_ratio:.1f}:1")

    # Train 9 models
    print("\n[Step 3] Training 9 model variants...")

    print("  Q1 baseline...")
    q1_model = train_model(X_q1_train, y_q1_train, X_q1_val, y_q1_val, verbose=False)
    print("  ERO baseline...")
    ero_model = train_model(X_ero_train, y_ero_train, X_ero_val, y_ero_val, verbose=False)
    print("  Combo baseline...")
    comb_model = train_model(X_comb_train, y_comb_train, X_comb_val, y_comb_val, verbose=False)

    print("  Q1@ERO ratio...")
    X_ds, y_ds = downsample_training(X_q1_train, y_q1_train, ri_q1_train, ero_ratio, 'galaxy')
    q1_at_ero = train_model(X_ds, y_ds, X_q1_val, y_q1_val, verbose=False)

    print("  Q1@Combo ratio...")
    X_ds, y_ds = downsample_training(X_q1_train, y_q1_train, ri_q1_train, combo_ratio, 'galaxy')
    q1_at_comb = train_model(X_ds, y_ds, X_q1_val, y_q1_val, verbose=False)

    print("  ERO@Q1 ratio...")
    X_ds, y_ds = downsample_training(X_ero_train, y_ero_train, ri_ero_train, q1_ratio, 'star')
    ero_at_q1 = train_model(X_ds, y_ds, X_ero_val, y_ero_val, verbose=False)

    print("  ERO@Combo ratio...")
    X_ds, y_ds = downsample_training(X_ero_train, y_ero_train, ri_ero_train, combo_ratio, 'star')
    ero_at_comb = train_model(X_ds, y_ds, X_ero_val, y_ero_val, verbose=False)

    print("  Comb@Q1 ratio...")
    X_ds, y_ds = downsample_training(X_comb_train, y_comb_train, ri_comb_train, q1_ratio, 'star')
    comb_at_q1 = train_model(X_ds, y_ds, X_comb_val, y_comb_val, verbose=False)

    print("  Comb@ERO ratio...")
    X_ds, y_ds = downsample_training(X_comb_train, y_comb_train, ri_comb_train, ero_ratio, 'galaxy')
    comb_at_ero = train_model(X_ds, y_ds, X_comb_val, y_comb_val, verbose=False)

    # Evaluate 27 experiments
    print("\n[Step 4] Evaluating 27 experiments...")
    models = [
        ('Q1@Q1', q1_model), ('Q1@Comb', q1_at_comb), ('Q1@ERO', q1_at_ero),
        ('ERO@Q1', ero_at_q1), ('ERO@Comb', ero_at_comb), ('ERO@ERO', ero_model),
        ('Comb@Q1', comb_at_q1), ('Comb@Comb', comb_model), ('Comb@ERO', comb_at_ero),
    ]
    test_sets = [('Q1', X_q1_test, y_q1_test), ('Combo', X_comb_test, y_comb_test),
                 ('ERO', X_ero_test, y_ero_test)]

    all_results = {}
    for prefix, model in models:
        for test_name, X_test, y_test in test_sets:
            label = f'{prefix}→{test_name}'
            r = evaluate(model, X_test, y_test)
            r['label'] = label
            all_results[label] = r
            print(f"  {label}")

    # Plot
    print("\n[Step 5] Generating line plot...")
    plot_galaxy_metrics_lines(all_results, OUTPUT_DIR, q1_ratio, combo_ratio, ero_ratio)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
