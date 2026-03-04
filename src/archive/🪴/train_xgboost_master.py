#!/usr/bin/env python3
"""
Master XGBoost training & evaluation script.

Combines all experiments from the three individual scripts into a single run:
  1. Q1 baseline → Q1 test
  2. Q1 baseline → ERO test (cross)
  3. ERO baseline → ERO test
  4. ERO baseline → Q1 test (cross)
  5. Q1 at ERO ratio → Q1 test (ratio-matched)
  6. ERO at Q1 ratio → ERO test (ratio-matched)

Queries data once, produces unified output in output/xgboost_master/.

Usage:
    python src/train_xgboost_master.py [--test-mode]
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
import umap
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output' / 'xgboost_master'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

CLASS_MAPPING = {'STAR': 0, 'GALAXY': 1}
CLASS_NAMES = ['STAR', 'GALAXY']

# Universal Gaia G magnitude cut applied to both datasets
MAG_CUT = 18.5

# XGBoost parameters (shared across all experiments)
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
# Data loading
# =============================================================================

def query_dataset(username, data_source, test_mode=False):
    """
    Query full dataset including random_index for later subsampling.

    Returns X, y, random_index arrays.
    """
    import psycopg2

    conn = psycopg2.connect(
        host=WSDB_HOST,
        database=WSDB_DB,
        user=username
    )

    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])

    if data_source == 'q1':
        query = f"""
            SELECT
                g_random_index AS random_index,
                {gaia_cols},
                CASE
                    WHEN p_phz_classification = 1 THEN 'STAR'
                    WHEN p_phz_classification = 2 THEN 'GALAXY'
                END as label
            FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
            WHERE p_phz_classification IN (1, 2)
                AND g_phot_g_mean_mag >= {MAG_CUT}
            ORDER BY g_source_id
        """
    else:  # ero
        query = f"""
            SELECT
                g_random_index AS random_index,
                {gaia_cols},
                CASE
                    WHEN e_v_class_star >= 0.65 THEN 'STAR'
                    WHEN e_v_class_star < 0.65 THEN 'GALAXY'
                END as label
            FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
            WHERE e_v_class_star IS NOT NULL
                AND g_phot_g_mean_mag >= {MAG_CUT}
            ORDER BY g_source_id
        """

    if test_mode:
        query += " LIMIT 10000"

    print(f"  Querying {data_source.upper()} dataset...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Clean features
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    X = df[SELECTED_GAIA_PARAMETERS].values
    y = df['label'].map(CLASS_MAPPING).values
    random_index = df['random_index'].values

    n_star = (y == 0).sum()
    n_galaxy = (y == 1).sum()
    print(f"  ✓ {len(X):,} sources — STAR: {n_star:,} ({100*n_star/len(X):.1f}%), "
          f"GALAXY: {n_galaxy:,} ({100*n_galaxy/len(X):.1f}%)")

    return X, y, random_index


# =============================================================================
# Training & evaluation
# =============================================================================

def train_model(X_train, y_train, X_val, y_val, verbose=True):
    """Train XGBoost binary classifier. Returns trained model."""
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=verbose
    )
    return model


def evaluate(model, X_test, y_test):
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1])

    return {
        'accuracy': (y_test == y_pred).sum() / len(y_test),
        'star_precision': precision[0],
        'star_recall': recall[0],
        'star_f1': f1[0],
        'galaxy_precision': precision[1],
        'galaxy_recall': recall[1],
        'galaxy_f1': f1[1],
        'roc_auc': roc_auc,
        'report': report,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
    }


def downsample_training(X_train, y_train, ri_train, target_ratio, downsample_class):
    """
    Downsample the training set using random_index for determinism.

    Parameters
    ----------
    target_ratio : float
        Target star:galaxy ratio.
    downsample_class : str
        'galaxy' to remove galaxies (increase ratio),
        'star' to remove stars (decrease ratio).
    """
    star_mask = (y_train == 0)
    galaxy_mask = (y_train == 1)
    n_star = star_mask.sum()
    n_galaxy = galaxy_mask.sum()

    if downsample_class == 'galaxy':
        n_galaxy_target = int(n_star / target_ratio)
        galaxy_indices = np.where(galaxy_mask)[0]
        galaxy_ri = ri_train[galaxy_indices]
        sorted_order = np.argsort(galaxy_ri)
        keep_galaxy = galaxy_indices[sorted_order[:n_galaxy_target]]

        keep_mask = star_mask.copy()
        keep_mask[keep_galaxy] = True
    else:  # downsample stars
        n_star_target = int(n_galaxy * target_ratio)
        star_indices = np.where(star_mask)[0]
        star_ri = ri_train[star_indices]
        sorted_order = np.argsort(star_ri)
        keep_star = star_indices[sorted_order[:n_star_target]]

        keep_mask = galaxy_mask.copy()
        keep_mask[keep_star] = True

    X_ds = X_train[keep_mask]
    y_ds = y_train[keep_mask]

    n_s = (y_ds == 0).sum()
    n_g = (y_ds == 1).sum()
    print(f"    Downsampled: STAR={n_s:,}, GALAXY={n_g:,}, Ratio={n_s/n_g:.1f}:1")

    return X_ds, y_ds


# =============================================================================
# Plotting
# =============================================================================

def plot_feature_importance_comparison(q1_model, ero_model, feature_names, output_dir):
    """Plot side-by-side feature importance comparison (all features) and save CSV."""
    q1_importance = q1_model.feature_importances_
    ero_importance = ero_model.feature_importances_

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'Q1': q1_importance,
        'ERO': ero_importance
    })

    n_features = len(feature_names)
    q1_sorted = importance_df.sort_values('Q1', ascending=False)
    ero_sorted = importance_df.sort_values('ERO', ascending=False)

    fig_height = max(12, n_features * 0.4)
    fig, axes = plt.subplots(1, 2, figsize=(18, fig_height))

    # Q1 subplot
    ax = axes[0]
    labels = [f"#{i+1} {f}" for i, f in enumerate(q1_sorted['feature'])]
    bars = ax.barh(range(n_features), q1_sorted['Q1'].values, color='lightsteelblue')
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=11, fontweight='bold')
    ax.set_title('Q1 Model Feature Importance', fontsize=12, fontweight='bold')
    ax.set_xlim(0, q1_sorted['Q1'].max() * 1.25)
    ax.grid(alpha=0.3, axis='x')
    for bar, val in zip(bars, q1_sorted['Q1'].values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                va='center', fontsize=7)

    # ERO subplot
    ax = axes[1]
    labels = [f"#{i+1} {f}" for i, f in enumerate(ero_sorted['feature'])]
    bars = ax.barh(range(n_features), ero_sorted['ERO'].values, color='lightsalmon')
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=11, fontweight='bold')
    ax.set_title('ERO Model Feature Importance', fontsize=12, fontweight='bold')
    ax.set_xlim(0, ero_sorted['ERO'].max() * 1.25)
    ax.grid(alpha=0.3, axis='x')
    for bar, val in zip(bars, ero_sorted['ERO'].values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                va='center', fontsize=7)

    plt.suptitle(f'Feature Importance Comparison: Q1 vs ERO Models (All {n_features} Features)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'master_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save CSV with full comparison
    importance_df['Q1_rank'] = importance_df['Q1'].rank(ascending=False).astype(int)
    importance_df['ERO_rank'] = importance_df['ERO'].rank(ascending=False).astype(int)
    importance_df['rank_diff'] = importance_df['ERO_rank'] - importance_df['Q1_rank']
    importance_df.to_csv(output_dir / 'master_feature_importance.csv', index=False)


def plot_galaxy_metrics(all_results, output_dir):
    """Plot Galaxy Precision & Recall bar chart for all evaluations."""
    labels = [r['label'] for r in all_results]
    precision = [r['galaxy_precision'] for r in all_results]
    recall = [r['galaxy_recall'] for r in all_results]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(labels))
    width = 0.3

    bars_p = ax.bar(x - width/2, precision, width, label='Galaxy Precision',
                    color='steelblue', edgecolor='white', linewidth=0.5)
    bars_r = ax.bar(x + width/2, recall, width, label='Galaxy Recall',
                    color='coral', edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars_p, precision):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars_r, recall):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=30, ha='right')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Galaxy Precision & Recall: All Experiments',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'master_galaxy_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_umap_comparison(X_q1, y_q1, X_ero, y_ero, output_dir, sample_size=50000):
    """Plot Q1 and ERO UMAP embeddings side by side, subsampled for speed."""
    rng = np.random.RandomState(42)

    datasets = []
    for X, y, name in [(X_q1, y_q1, 'Q1'), (X_ero, y_ero, 'ERO')]:
        if len(X) > sample_size:
            idx = rng.choice(len(X), sample_size, replace=False)
            X_sub, y_sub = X[idx], y[idx]
        else:
            X_sub, y_sub = X, y

        # Fill NaN with column median, then standardize
        X_clean = X_sub.copy()
        for col in range(X_clean.shape[1]):
            col_data = X_clean[:, col]
            mask = np.isnan(col_data)
            if mask.any():
                median_val = np.nanmedian(col_data)
                col_data[mask] = median_val
            std = np.nanstd(col_data)
            mean = np.nanmean(col_data)
            if std > 0:
                X_clean[:, col] = (col_data - mean) / std

        print(f"  Fitting UMAP for {name} ({len(X_sub):,} sources)...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding = reducer.fit_transform(X_clean)
        datasets.append((embedding, y_sub, name))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = {0: '#1976D2', 1: '#E53935'}
    label_names = {0: 'STAR', 1: 'GALAXY'}

    for ax, (embedding, y_sub, dataset_name) in zip(axes, datasets):
        for label_val in [1, 0]:  # galaxies first so stars on top
            mask = y_sub == label_val
            count = mask.sum()
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=colors[label_val], label=f'{label_names[label_val]} (n={count:,})',
                       alpha=0.3, s=5)
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title(f'{dataset_name} Dataset', fontsize=13, fontweight='bold')
        ax.legend(markerscale=3, fontsize=10, loc='upper right')
        ax.grid(alpha=0.3)

    plt.suptitle('UMAP Visualization of Gaia Feature Space', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'umap_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Master XGBoost training & evaluation script'
    )
    parser.add_argument('--test-mode', action='store_true',
                        help='Use small sample for quick testing (10k sources)')
    args = parser.parse_args()

    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Master XGBoost Training & Evaluation")
    print("=" * 80)
    print(f"  Features: {len(SELECTED_GAIA_PARAMETERS)}")
    print(f"  Magnitude cut: Gaia G >= {MAG_CUT} (both datasets)")
    print(f"  Output: {OUTPUT_DIR}")
    if args.test_mode:
        print(f"  TEST MODE: limited to 10k sources per dataset")

    # =========================================================================
    # Step 1: Query both datasets (once each)
    # =========================================================================
    print("\n[Step 1] Querying datasets...")
    X_q1, y_q1, ri_q1 = query_dataset(username, 'q1', test_mode=args.test_mode)
    X_ero, y_ero, ri_ero = query_dataset(username, 'ero', test_mode=args.test_mode)

    # =========================================================================
    # Step 2: Split train/val/test (same seeds as existing scripts)
    # =========================================================================
    print("\n[Step 2] Splitting data (70/15/15, random_state=42)...")

    # Q1 splits
    X_q1_temp, X_q1_test, y_q1_temp, y_q1_test, ri_q1_temp, _ = train_test_split(
        X_q1, y_q1, ri_q1, test_size=0.15, random_state=42, stratify=y_q1
    )
    X_q1_train, X_q1_val, y_q1_train, y_q1_val, ri_q1_train, _ = train_test_split(
        X_q1_temp, y_q1_temp, ri_q1_temp, test_size=0.176, random_state=42, stratify=y_q1_temp
    )

    n_q1_star_train = (y_q1_train == 0).sum()
    n_q1_galaxy_train = (y_q1_train == 1).sum()
    q1_train_ratio = n_q1_star_train / n_q1_galaxy_train

    print(f"  Q1  — train: {len(X_q1_train):,}, val: {len(X_q1_val):,}, test: {len(X_q1_test):,}"
          f"  (ratio {q1_train_ratio:.1f}:1)")

    # ERO splits
    X_ero_temp, X_ero_test, y_ero_temp, y_ero_test, ri_ero_temp, _ = train_test_split(
        X_ero, y_ero, ri_ero, test_size=0.15, random_state=42, stratify=y_ero
    )
    X_ero_train, X_ero_val, y_ero_train, y_ero_val, ri_ero_train, _ = train_test_split(
        X_ero_temp, y_ero_temp, ri_ero_temp, test_size=0.176, random_state=42, stratify=y_ero_temp
    )

    n_ero_star_train = (y_ero_train == 0).sum()
    n_ero_galaxy_train = (y_ero_train == 1).sum()
    ero_train_ratio = n_ero_star_train / n_ero_galaxy_train

    print(f"  ERO — train: {len(X_ero_train):,}, val: {len(X_ero_val):,}, test: {len(X_ero_test):,}"
          f"  (ratio {ero_train_ratio:.1f}:1)")

    # =========================================================================
    # Step 3: Train Q1 and ERO baseline models
    # =========================================================================
    print("\n[Step 3] Training baseline models...")
    print("\n--- Q1 baseline ---")
    q1_model = train_model(X_q1_train, y_q1_train, X_q1_val, y_q1_val, verbose=True)
    print(f"  Best iteration: {q1_model.best_iteration}, Best score: {q1_model.best_score:.4f}")

    print("\n--- ERO baseline ---")
    ero_model = train_model(X_ero_train, y_ero_train, X_ero_val, y_ero_val, verbose=True)
    print(f"  Best iteration: {ero_model.best_iteration}, Best score: {ero_model.best_score:.4f}")

    # =========================================================================
    # Step 4: Evaluate all 6 experiments
    # =========================================================================
    print("\n[Step 4] Evaluating all experiments...")
    all_results = []

    # 1. Q1 → Q1 (baseline)
    print("\n  [1/6] Q1 → Q1 (baseline)")
    r = evaluate(q1_model, X_q1_test, y_q1_test)
    r['label'] = 'Q1→Q1'
    r['description'] = 'Q1 baseline → Q1 test'
    all_results.append(r)

    # 2. Q1 → ERO (cross)
    print("  [2/6] Q1 → ERO (cross)")
    r = evaluate(q1_model, X_ero_test, y_ero_test)
    r['label'] = 'Q1→ERO'
    r['description'] = 'Q1 baseline → ERO test'
    all_results.append(r)

    # 3. ERO → ERO (baseline)
    print("  [3/6] ERO → ERO (baseline)")
    r = evaluate(ero_model, X_ero_test, y_ero_test)
    r['label'] = 'ERO→ERO'
    r['description'] = 'ERO baseline → ERO test'
    all_results.append(r)

    # 4. ERO → Q1 (cross)
    print("  [4/6] ERO → Q1 (cross)")
    r = evaluate(ero_model, X_q1_test, y_q1_test)
    r['label'] = 'ERO→Q1'
    r['description'] = 'ERO baseline → Q1 test'
    all_results.append(r)

    # =========================================================================
    # Step 5: Ratio-matched experiments
    # =========================================================================
    print("\n[Step 5] Training ratio-matched variants...")

    # 5. Q1 at ERO ratio → Q1 test
    print("\n  [5/6] Q1 downsampled to ERO ratio → Q1 test")
    X_q1_ds, y_q1_ds = downsample_training(
        X_q1_train, y_q1_train, ri_q1_train,
        target_ratio=ero_train_ratio, downsample_class='galaxy'
    )
    q1_ratio_model = train_model(X_q1_ds, y_q1_ds, X_q1_val, y_q1_val, verbose=False)
    r = evaluate(q1_ratio_model, X_q1_test, y_q1_test)
    ds_ratio = (y_q1_ds == 0).sum() / (y_q1_ds == 1).sum()
    r['label'] = f'Q1@ERO ratio→Q1'
    r['description'] = f'Q1 at ERO ratio ({ds_ratio:.0f}:1) → Q1 test'
    all_results.append(r)

    # 6. ERO at Q1 ratio → ERO test
    print("  [6/6] ERO downsampled to Q1 ratio → ERO test")
    X_ero_ds, y_ero_ds = downsample_training(
        X_ero_train, y_ero_train, ri_ero_train,
        target_ratio=q1_train_ratio, downsample_class='star'
    )
    ero_ratio_model = train_model(X_ero_ds, y_ero_ds, X_ero_val, y_ero_val, verbose=False)
    r = evaluate(ero_ratio_model, X_ero_test, y_ero_test)
    ds_ratio = (y_ero_ds == 0).sum() / (y_ero_ds == 1).sum()
    r['label'] = f'ERO@Q1 ratio→ERO'
    r['description'] = f'ERO at Q1 ratio ({ds_ratio:.0f}:1) → ERO test'
    all_results.append(r)

    # =========================================================================
    # Step 6: Print & save unified summary
    # =========================================================================
    print("\n" + "=" * 90)
    print("UNIFIED SUMMARY")
    print("=" * 90)
    header = (f"{'Experiment':<22} {'Accuracy':>9} {'Gal Prec':>9} {'Gal Rec':>8} "
              f"{'Gal F1':>7} {'Star Prec':>10} {'Star Rec':>9} {'ROC AUC':>8}")
    print(header)
    print("-" * 90)
    for r in all_results:
        print(f"{r['label']:<22} {r['accuracy']:>9.3f} {r['galaxy_precision']:>9.3f} "
              f"{r['galaxy_recall']:>8.3f} {r['galaxy_f1']:>7.3f} "
              f"{r['star_precision']:>10.3f} {r['star_recall']:>9.3f} {r['roc_auc']:>8.3f}")

    # Save parameter list
    param_list_path = OUTPUT_DIR / 'parameters_used.txt'
    with open(param_list_path, 'w') as f:
        f.write(f"PARAMETERS USED IN THIS RUN ({len(SELECTED_GAIA_PARAMETERS)} features)\n")
        f.write("=" * 60 + "\n\n")
        for i, param in enumerate(SELECTED_GAIA_PARAMETERS, 1):
            f.write(f"  {i:>2}. {param}\n")
    print(f"  ✓ Parameter list saved: {param_list_path}")

    # Save summary text file
    summary_path = OUTPUT_DIR / 'master_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("MASTER XGBOOST TRAINING & EVALUATION SUMMARY\n")
        f.write("=" * 90 + "\n")
        f.write(f"Magnitude cut: Gaia G >= {MAG_CUT} (both datasets)\n")
        f.write(f"Features: {len(SELECTED_GAIA_PARAMETERS)}\n")
        f.write(f"Q1 train ratio: {q1_train_ratio:.1f}:1 (STAR:GALAXY)\n")
        f.write(f"ERO train ratio: {ero_train_ratio:.1f}:1 (STAR:GALAXY)\n\n")
        f.write("Parameters:\n")
        for i, param in enumerate(SELECTED_GAIA_PARAMETERS, 1):
            f.write(f"  {i:>2}. {param}\n")
        f.write("\n")

        f.write(header + "\n")
        f.write("-" * 90 + "\n")
        for r in all_results:
            f.write(f"{r['label']:<22} {r['accuracy']:>9.3f} {r['galaxy_precision']:>9.3f} "
                    f"{r['galaxy_recall']:>8.3f} {r['galaxy_f1']:>7.3f} "
                    f"{r['star_precision']:>10.3f} {r['star_recall']:>9.3f} {r['roc_auc']:>8.3f}\n")

        f.write("\n")
        for r in all_results:
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"{r['description']}\n")
            f.write("=" * 70 + "\n")
            f.write(r['report'])
            f.write(f"ROC AUC: {r['roc_auc']:.4f}\n")

    print(f"\n  ✓ Summary saved: {summary_path}")

    # =========================================================================
    # Step 7: Generate all plots
    # =========================================================================
    print("\n[Step 7] Generating plots...")

    # Unified galaxy metrics bar chart
    plot_galaxy_metrics(all_results, OUTPUT_DIR)
    print("  ✓ master_galaxy_metrics.png")

    # Feature importance comparison (Q1 vs ERO baselines)
    plot_feature_importance_comparison(q1_model, ero_model, SELECTED_GAIA_PARAMETERS, OUTPUT_DIR)
    print("  ✓ master_feature_importance.png + .csv")

    # UMAP visualization
    print("\n  Generating UMAP comparison...")
    plot_umap_comparison(X_q1, y_q1, X_ero, y_ero, OUTPUT_DIR)
    print("  ✓ umap_comparison.png")

    # =========================================================================
    # Step 8: Save models
    # =========================================================================
    print("\n[Step 8] Saving models...")
    for name, model in [('q1', q1_model), ('ero', ero_model)]:
        model_path = OUTPUT_DIR / f'xgb_classifier_{name}.json'
        try:
            model.get_booster().save_model(str(model_path))
            print(f"  ✓ {model_path}")
        except Exception as e:
            print(f"  ⚠ Could not save {name} model: {e}")

    print("\n" + "=" * 80)
    print("Master training complete!")
    print(f"All results saved to: {OUTPUT_DIR}")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
