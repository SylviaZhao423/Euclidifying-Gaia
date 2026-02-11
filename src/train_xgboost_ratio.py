#!/usr/bin/env python3
"""
Train XGBoost classifiers with matched class ratios to isolate the effect of
class imbalance from truth table quality.

Approach:
  - Query each full dataset (Q1, ERO) once
  - Split a fixed test set (15%) from each - shared across experiments
  - Only downsample the TRAINING portion to change the ratio
  - This ensures baseline numbers match original training and changes
    are directly attributable to the ratio modification

Experiment:
  1. Q1 original (3:1 ratio) - baseline
  2. Q1 training downsampled to ERO ratio (~19:1) - same test set
  3. ERO original (19:1 ratio) - baseline
  4. ERO training downsampled to Q1 ratio (~3:1) - same test set

Uses Gaia's random_index for deterministic, reproducible subsampling.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_fscore_support
)
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output' / 'xgboost_ratio'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

CLASS_MAPPING = {'STAR': 0, 'GALAXY': 1}
CLASS_NAMES = ['STAR', 'GALAXY']


def query_full_dataset(username, data_source, mag_cut=18.0):
    """Query full dataset including random_index for later subsampling."""
    import psycopg2

    conn = psycopg2.connect(
        host=WSDB_HOST,
        database=WSDB_DB,
        user=username
    )

    # Build column list with g_ prefix, aliased back to standard names
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
                AND g_phot_g_mean_mag >= {mag_cut}
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
                AND g_phot_g_mean_mag >= {mag_cut}
            ORDER BY g_source_id
        """

    print(f"  Querying full {data_source.upper()} dataset...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Clean features
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    X = df[SELECTED_GAIA_PARAMETERS].values
    y = df['label'].map(CLASS_MAPPING).values
    random_index = df['random_index'].values

    n_star = (df['label'] == 'STAR').sum()
    n_galaxy = (df['label'] == 'GALAXY').sum()
    print(f"  STAR: {n_star:,}, GALAXY: {n_galaxy:,}, Ratio: {n_star/n_galaxy:.1f}:1")

    return X, y, random_index


def downsample_training(X_train, y_train, ri_train, target_ratio, downsample_class):
    """
    Downsample the training set using random_index for determinism.

    Parameters:
    -----------
    target_ratio : float
        Target star:galaxy ratio
    downsample_class : str
        'galaxy' to remove galaxies (increase ratio),
        'star' to remove stars (decrease ratio)
    """
    star_mask = (y_train == 0)
    galaxy_mask = (y_train == 1)
    n_star = star_mask.sum()
    n_galaxy = galaxy_mask.sum()

    if downsample_class == 'galaxy':
        # Keep all stars, subsample galaxies
        n_galaxy_target = int(n_star / target_ratio)
        # Use random_index to select deterministically: sort by random_index, take first n
        galaxy_indices = np.where(galaxy_mask)[0]
        galaxy_ri = ri_train[galaxy_indices]
        sorted_order = np.argsort(galaxy_ri)
        keep_galaxy = galaxy_indices[sorted_order[:n_galaxy_target]]

        keep_mask = star_mask.copy()
        keep_mask[keep_galaxy] = True

    else:  # downsample stars
        # Keep all galaxies, subsample stars
        n_star_target = int(n_galaxy * target_ratio)
        # Use random_index to select deterministically
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
    print(f"    Downsampled training: STAR={n_s:,}, GALAXY={n_g:,}, Ratio={n_s/n_g:.1f}:1")

    return X_ds, y_ds


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, experiment_name):
    """Train XGBoost on given splits and evaluate on fixed test set."""
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=8,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist',
        eval_metric='logloss',
        early_stopping_rounds=20,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Evaluate on fixed test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1])

    print(f"\n{'='*70}")
    print(f"  {experiment_name}")
    print(f"{'='*70}")
    print(report)
    print(f"  ROC AUC: {roc_auc:.4f}")

    results = {
        'experiment': experiment_name,
        'accuracy': (y_test == y_pred).sum() / len(y_test),
        'star_precision': precision[0],
        'star_recall': recall[0],
        'star_f1': f1[0],
        'galaxy_precision': precision[1],
        'galaxy_recall': recall[1],
        'galaxy_f1': f1[1],
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'report': report
    }

    return results


def plot_results(all_results, output_dir):
    """Plot comparison of all experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    labels = [r['experiment'] for r in all_results]
    colors = ['steelblue', 'lightsteelblue', 'coral', 'lightsalmon']

    metrics = [
        ('galaxy_recall', 'Galaxy Recall', axes[0]),
        ('galaxy_precision', 'Galaxy Precision', axes[1]),
    ]

    for metric_key, metric_name, ax in metrics:
        values = [r[metric_key] for r in all_results]
        bars = ax.bar(range(len(labels)), values, color=colors)
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.grid(alpha=0.3, axis='y')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.suptitle('Class Ratio Matching Experiment:\nIsolating Imbalance vs Truth Table Quality',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_dir / 'xgboost_ratio_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Comparison plot saved: {output_dir / 'xgboost_ratio_comparison.png'}")


def main():
    mag_cut = 18.0  # hardcoded magnitude cut

    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    print("=" * 70)
    print("Ratio-Matched Training Experiment")
    print(f"Magnitude cut: G >= {mag_cut}")
    print("=" * 70)
    print("\nApproach: fixed test set per dataset, only modify training ratios")
    print("Uses Gaia random_index for deterministic subsampling\n")

    all_results = []

    # ==================== LOAD BOTH DATASETS ====================
    print("[Step 1] Loading Q1 data...")
    X_q1, y_q1, ri_q1 = query_full_dataset(username, 'q1', mag_cut)

    X_q1_temp, X_q1_test, y_q1_temp, y_q1_test, ri_q1_temp, _ = train_test_split(
        X_q1, y_q1, ri_q1, test_size=0.15, random_state=42, stratify=y_q1
    )
    X_q1_train, X_q1_val, y_q1_train, y_q1_val, ri_q1_train, _ = train_test_split(
        X_q1_temp, y_q1_temp, ri_q1_temp, test_size=0.176, random_state=42, stratify=y_q1_temp
    )

    n_q1_star = (y_q1_train == 0).sum()
    n_q1_galaxy = (y_q1_train == 1).sum()
    q1_train_ratio = n_q1_star / n_q1_galaxy
    print(f"  Q1 train: STAR={n_q1_star:,}, GALAXY={n_q1_galaxy:,}, Ratio={q1_train_ratio:.1f}:1")
    print(f"  Q1 test:  {len(X_q1_test):,} (fixed for both Q1 experiments)")

    print("\n[Step 2] Loading ERO data...")
    X_ero, y_ero, ri_ero = query_full_dataset(username, 'ero', mag_cut)

    X_ero_temp, X_ero_test, y_ero_temp, y_ero_test, ri_ero_temp, _ = train_test_split(
        X_ero, y_ero, ri_ero, test_size=0.15, random_state=42, stratify=y_ero
    )
    X_ero_train, X_ero_val, y_ero_train, y_ero_val, ri_ero_train, _ = train_test_split(
        X_ero_temp, y_ero_temp, ri_ero_temp, test_size=0.176, random_state=42, stratify=y_ero_temp
    )

    n_ero_star = (y_ero_train == 0).sum()
    n_ero_galaxy = (y_ero_train == 1).sum()
    ero_train_ratio = n_ero_star / n_ero_galaxy
    print(f"  ERO train: STAR={n_ero_star:,}, GALAXY={n_ero_galaxy:,}, Ratio={ero_train_ratio:.1f}:1")
    print(f"  ERO test:  {len(X_ero_test):,} (fixed for both ERO experiments)")

    # ==================== Q1 EXPERIMENTS ====================
    # Experiment 1: Q1 original
    print("\n[Exp 1/4] Q1 original training ratio...")
    results = train_and_evaluate(
        X_q1_train, y_q1_train, X_q1_val, y_q1_val, X_q1_test, y_q1_test,
        f'Q1 original\n({q1_train_ratio:.0f}:1)'
    )
    results['n_star'] = n_q1_star
    results['n_galaxy'] = n_q1_galaxy
    results['ratio'] = q1_train_ratio
    all_results.append(results)

    # Experiment 2: Q1 training downsampled to ERO ratio
    print("\n[Exp 2/4] Q1 training downsampled to ERO ratio...")
    X_q1_ds, y_q1_ds = downsample_training(
        X_q1_train, y_q1_train, ri_q1_train, target_ratio=ero_train_ratio, downsample_class='galaxy'
    )
    n_ds_star = (y_q1_ds == 0).sum()
    n_ds_galaxy = (y_q1_ds == 1).sum()
    ds_ratio = n_ds_star / n_ds_galaxy
    results = train_and_evaluate(
        X_q1_ds, y_q1_ds, X_q1_val, y_q1_val, X_q1_test, y_q1_test,
        f'Q1 matched to\nERO ratio ({ds_ratio:.0f}:1)'
    )
    results['n_star'] = n_ds_star
    results['n_galaxy'] = n_ds_galaxy
    results['ratio'] = ds_ratio
    all_results.append(results)

    # ==================== ERO EXPERIMENTS ====================
    # Experiment 3: ERO original
    print("\n[Exp 3/4] ERO original training ratio...")
    results = train_and_evaluate(
        X_ero_train, y_ero_train, X_ero_val, y_ero_val, X_ero_test, y_ero_test,
        f'ERO original\n({ero_train_ratio:.0f}:1)'
    )
    results['n_star'] = n_ero_star
    results['n_galaxy'] = n_ero_galaxy
    results['ratio'] = ero_train_ratio
    all_results.append(results)

    # Experiment 4: ERO training downsampled to Q1 ratio
    print("\n[Exp 4/4] ERO training downsampled to Q1 ratio...")
    X_ero_ds, y_ero_ds = downsample_training(
        X_ero_train, y_ero_train, ri_ero_train, target_ratio=q1_train_ratio, downsample_class='star'
    )
    n_ds_star = (y_ero_ds == 0).sum()
    n_ds_galaxy = (y_ero_ds == 1).sum()
    ds_ratio = n_ds_star / n_ds_galaxy
    results = train_and_evaluate(
        X_ero_ds, y_ero_ds, X_ero_val, y_ero_val, X_ero_test, y_ero_test,
        f'ERO matched to\nQ1 ratio ({ds_ratio:.0f}:1)'
    )
    results['n_star'] = n_ds_star
    results['n_galaxy'] = n_ds_galaxy
    results['ratio'] = ds_ratio
    all_results.append(results)

    # ==================== SUMMARY ====================
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Experiment':<30} {'Train Stars':>12} {'Train Gal':>10} {'Ratio':>7} {'Gal Rec':>8} {'Gal Prec':>9} {'ROC AUC':>8}")
    print("-" * 90)
    for r in all_results:
        name = r['experiment'].replace('\n', ' ')
        print(f"{name:<30} {r['n_star']:>12,} {r['n_galaxy']:>10,} {r['ratio']:>6.1f}:1 "
              f"{r['galaxy_recall']:>8.3f} {r['galaxy_precision']:>9.3f} {r['roc_auc']:>8.3f}")

    # Plot
    plot_results(all_results, OUTPUT_DIR)

    # Save summary
    summary_path = OUTPUT_DIR / 'xgboost_ratio_summary.txt'
    with open(summary_path, 'w') as f:
        # Class imbalance analysis
        n_q1_total = len(X_q1)
        n_q1_star_all = (y_q1 == 0).sum()
        n_q1_gal_all = (y_q1 == 1).sum()
        n_ero_total = len(X_ero)
        n_ero_star_all = (y_ero == 0).sum()
        n_ero_gal_all = (y_ero == 1).sum()
        imbalance_factor = ero_train_ratio / q1_train_ratio

        f.write("CLASS IMBALANCE ANALYSIS: Q1 vs ERO\n")
        f.write("=" * 70 + "\n")
        f.write(f"(Gaia G >= {mag_cut} mag filter applied)\n\n")
        f.write("Q1 Dataset:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total sources:       {n_q1_total:>10,}\n")
        f.write(f"  STAR count:          {n_q1_star_all:>10,} ({100*n_q1_star_all/n_q1_total:.1f}%)\n")
        f.write(f"  GALAXY count:        {n_q1_gal_all:>10,} ({100*n_q1_gal_all/n_q1_total:.1f}%)\n")
        f.write(f"  Star:Galaxy ratio:   {q1_train_ratio:>10.1f}:1\n\n")
        f.write("ERO Dataset:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total sources:       {n_ero_total:>10,}\n")
        f.write(f"  STAR count:          {n_ero_star_all:>10,} ({100*n_ero_star_all/n_ero_total:.1f}%)\n")
        f.write(f"  GALAXY count:        {n_ero_gal_all:>10,} ({100*n_ero_gal_all/n_ero_total:.1f}%)\n")
        f.write(f"  Star:Galaxy ratio:   {ero_train_ratio:>10.1f}:1\n\n")
        f.write(f"ERO has {imbalance_factor:.1f}x more severe class imbalance than Q1.\n\n")

        # Ratio-matched experiment results
        f.write("=" * 90 + "\n")
        f.write("RATIO-MATCHED TRAINING EXPERIMENT\n")
        f.write("=" * 90 + "\n")
        f.write("Approach: fixed test set per dataset, only modify training ratios\n")
        f.write("Uses Gaia random_index for deterministic subsampling\n\n")
        f.write(f"{'Experiment':<30} {'Train Stars':>12} {'Train Gal':>10} {'Ratio':>7} {'Gal Rec':>8} {'Gal Prec':>9} {'ROC AUC':>8}\n")
        f.write("-" * 90 + "\n")
        for r in all_results:
            name = r['experiment'].replace('\n', ' ')
            f.write(f"{name:<30} {r['n_star']:>12,} {r['n_galaxy']:>10,} {r['ratio']:>6.1f}:1 "
                    f"{r['galaxy_recall']:>8.3f} {r['galaxy_precision']:>9.3f} {r['roc_auc']:>8.3f}\n")
        f.write("\n")
        for r in all_results:
            f.write(f"\n{'='*70}\n")
            f.write(f"{r['experiment'].replace(chr(10), ' ')}\n")
            f.write(f"{'='*70}\n")
            f.write(r['report'])
            f.write(f"\nROC AUC: {r['roc_auc']:.4f}\n")
    print(f"✓ Summary saved: {summary_path}")

    print("\n" + "=" * 70)
    print("Experiment complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
