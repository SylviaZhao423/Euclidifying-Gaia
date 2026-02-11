#!/usr/bin/env python3
"""
Cross-dataset validation for XGBoost star/galaxy classifiers.

Tests each model on both its own and the opposite dataset's test set:
- Q1 model → Q1 test data (baseline)
- Q1 model → ERO test data (cross)
- ERO model → ERO test data (baseline)
- ERO model → Q1 test data (cross)

Uses same train_test_split as train_xgboost_classifier.py for consistent baselines.

Usage:
    python src/train_xgboost_cross.py
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output' / 'xgboost_cross'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

CLASS_MAPPING = {'STAR': 0, 'GALAXY': 1}
CLASS_NAMES = ['STAR', 'GALAXY']


def query_full_dataset(username, data_source, mag_cut=18.0):
    """Query full dataset, then extract the same test split as train_xgboost_classifier.py."""
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed!")
        return None

    try:
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

        # Clean Gaia columns
        for col in SELECTED_GAIA_PARAMETERS:
            df[col] = pd.to_numeric(df[col].replace('NOT_AVAILABLE', np.nan), errors='coerce')

        X = df[SELECTED_GAIA_PARAMETERS].values
        y = df['label'].map(CLASS_MAPPING).values

        # Use same split as train_xgboost_classifier.py: 15% test, random_state=42
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )

        n_star = (y_test == 0).sum()
        n_galaxy = (y_test == 1).sum()
        print(f"  ✓ Test set: {len(X_test):,} sources")
        print(f"    STAR: {n_star:,} ({100*n_star/len(y_test):.1f}%)")
        print(f"    GALAXY: {n_galaxy:,} ({100*n_galaxy/len(y_test):.1f}%)")

        return X_test, y_test

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_cross_validation(model, X_test, y_test, model_name, test_data_name):
    """Evaluate model on cross-dataset and return metrics."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name} model → {test_data_name} test data")
    print(f"{'='*80}")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Classification report
    print("\nClassification Report:")
    print("="*70)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              STAR    GALAXY")
    print(f"Actual STAR   {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"       GALAXY {cm[1,0]:6d}  {cm[1,1]:6d}")

    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC: {roc_auc:.4f}")

    # Extract key metrics
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1])

    results = {
        'model': model_name,
        'test_data': test_data_name,
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


def plot_feature_importance_comparison(q1_model, ero_model, feature_names, output_dir):
    """Plot side-by-side feature importance comparison for both models."""
    # Get feature importances
    q1_importance = q1_model.feature_importances_
    ero_importance = ero_model.feature_importances_

    # Create DataFrame for easier manipulation
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'Q1': q1_importance,
        'ERO': ero_importance
    })

    # Plot top 20 features, each subplot ranked independently
    top_n = 20
    q1_sorted = importance_df.sort_values('Q1', ascending=False).head(top_n)
    ero_sorted = importance_df.sort_values('ERO', ascending=False).head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # Q1 model importance (ranked by Q1)
    ax = axes[0]
    colors = ['steelblue' if f == 'phot_bp_rp_excess_factor' else 'lightsteelblue'
              for f in q1_sorted['feature']]
    labels = [f"#{i+1} {f}" for i, f in enumerate(q1_sorted['feature'])]
    bars = ax.barh(range(top_n), q1_sorted['Q1'].values, color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=11, fontweight='bold')
    ax.set_title('Q1 Model Feature Importance', fontsize=12, fontweight='bold')
    ax.set_xlim(0, q1_sorted['Q1'].max() * 1.2)
    ax.grid(alpha=0.3, axis='x')
    for bar, val in zip(bars, q1_sorted['Q1'].values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=8)

    # ERO model importance (ranked by ERO)
    ax = axes[1]
    colors = ['coral' if f == 'phot_bp_rp_excess_factor' else 'lightsalmon'
              for f in ero_sorted['feature']]
    labels = [f"#{i+1} {f}" for i, f in enumerate(ero_sorted['feature'])]
    bars = ax.barh(range(top_n), ero_sorted['ERO'].values, color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=11, fontweight='bold')
    ax.set_title('ERO Model Feature Importance', fontsize=12, fontweight='bold')
    ax.set_xlim(0, ero_sorted['ERO'].max() * 1.2)
    ax.grid(alpha=0.3, axis='x')
    for bar, val in zip(bars, ero_sorted['ERO'].values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=8)

    plt.suptitle('Feature Importance Comparison: Q1 vs ERO Models',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Feature importance comparison saved: {output_dir / 'feature_importance_comparison.png'}")

    # Save CSV with full comparison
    importance_df['Q1_rank'] = importance_df['Q1'].rank(ascending=False).astype(int)
    importance_df['ERO_rank'] = importance_df['ERO'].rank(ascending=False).astype(int)
    importance_df['rank_diff'] = importance_df['ERO_rank'] - importance_df['Q1_rank']
    importance_df.to_csv(output_dir / 'feature_importance_comparison.csv', index=False)
    print(f"✓ Feature importance CSV saved: {output_dir / 'feature_importance_comparison.csv'}")

    return importance_df



def main():
    mag_cut = 18.0

    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    print("="*80)
    print("Cross-Dataset Validation")
    print(f"Magnitude cut: G >= {mag_cut}")
    print("="*80)

    # Load models
    print("\nLoading trained models...")
    q1_model_path = PROJECT_DIR / 'output' / 'xgboost_q1' / 'xgb_classifier_q1.json'
    ero_model_path = PROJECT_DIR / 'output' / 'xgboost_ero' / 'xgb_classifier_ero.json'

    q1_model = xgb.XGBClassifier()
    q1_model.load_model(str(q1_model_path))
    print(f"  ✓ Loaded Q1 model: {q1_model_path}")

    ero_model = xgb.XGBClassifier()
    ero_model.load_model(str(ero_model_path))
    print(f"  ✓ Loaded ERO model: {ero_model_path}")

    # Query full datasets and extract same test splits as train_xgboost_classifier.py
    print("\nQuerying datasets and extracting test splits...")
    print("  (Using same train_test_split as training script for consistent baselines)")
    q1_data = query_full_dataset(username, 'q1', mag_cut)
    ero_data = query_full_dataset(username, 'ero', mag_cut)

    if q1_data is None or ero_data is None:
        return 1

    X_q1, y_q1 = q1_data
    X_ero, y_ero = ero_data

    # Perform cross-validation
    results = []

    # 1. Q1 model on Q1 data (baseline - native performance)
    results.append(evaluate_cross_validation(q1_model, X_q1, y_q1, 'Q1', 'Q1'))

    # 2. Q1 model on ERO data (cross-validation)
    results.append(evaluate_cross_validation(q1_model, X_ero, y_ero, 'Q1', 'ERO'))

    # 3. ERO model on ERO data (baseline - native performance)
    results.append(evaluate_cross_validation(ero_model, X_ero, y_ero, 'ERO', 'ERO'))

    # 4. ERO model on Q1 data (cross-validation)
    results.append(evaluate_cross_validation(ero_model, X_q1, y_q1, 'ERO', 'Q1'))

    # Save detailed results
    print("\n" + "="*80)
    print("Summary Table")
    print("="*80)
    print(f"{'Model→Data':<15} {'Accuracy':>10} {'Galaxy Rec':>11} {'Galaxy Prec':>11} {'ROC AUC':>10}")
    print("-"*80)
    for r in results:
        label = f"{r['model']}→{r['test_data']}"
        print(f"{label:<15} {r['accuracy']:>10.3f} {r['galaxy_recall']:>11.3f} "
              f"{r['galaxy_precision']:>11.3f} {r['roc_auc']:>10.3f}")

    # Plot feature importance comparison
    print("\nGenerating feature importance comparison...")
    plot_feature_importance_comparison(q1_model, ero_model, SELECTED_GAIA_PARAMETERS, OUTPUT_DIR)

    # Save combined report
    report_path = OUTPUT_DIR / 'xgboost_cross_summary.txt'
    with open(report_path, 'w') as f:
        f.write("CROSS-DATASET VALIDATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Model→Data':<15} {'Accuracy':>10} {'Galaxy Rec':>11} {'Galaxy Prec':>11} {'ROC AUC':>10}\n")
        f.write("-" * 80 + "\n")
        for r in results:
            label = f"{r['model']}→{r['test_data']}"
            f.write(f"{label:<15} {r['accuracy']:>10.3f} {r['galaxy_recall']:>11.3f} "
                    f"{r['galaxy_precision']:>11.3f} {r['roc_auc']:>10.3f}\n")
        f.write("\n")
        for r in results:
            f.write("=" * 70 + "\n")
            f.write(f"{r['model']} model → {r['test_data']} test data\n")
            f.write("=" * 70 + "\n")
            f.write(r['report'] + "\n")
            f.write(f"ROC AUC: {r['roc_auc']:.4f}\n\n")
    print(f"  ✓ Saved: {report_path}")

    print("\n" + "="*80)
    print("Cross-validation complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
