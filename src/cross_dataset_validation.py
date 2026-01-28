#!/usr/bin/env python3
"""
Cross-dataset validation: Test each model on the opposite dataset's test set.
- Q1 model → ERO test data
- ERO model → Q1 test data

This tests whether models learned generalizable features or dataset-specific patterns.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output' / 'cross_validation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

CLASS_MAPPING = {'STAR': 0, 'GALAXY': 1}
CLASS_NAMES = ['STAR', 'GALAXY']


def query_test_data(username, data_source='q1', test_size=10000):
    """Query labeled test data from specified dataset."""
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

        if data_source == 'q1':
            query = f"""
                SELECT
                    g.{', g.'.join(SELECTED_GAIA_PARAMETERS)},
                    CASE
                        WHEN e.p_phz_classification = 1 THEN 'STAR'
                        WHEN e.p_phz_classification = 2 THEN 'GALAXY'
                    END as label
                FROM gaia_dr3.gaia_source AS g
                INNER JOIN siwei_zhao_2025.crossmatch_q1_mer_phz_gaia AS e
                    ON g.source_id = e.g_source_id
                WHERE e.p_phz_classification IN (1, 2)
                    AND (-2.5 * LOG(g.phot_g_mean_flux) + 25.69) >= 18
                LIMIT {test_size}
            """
        else:  # ero
            query = f"""
                SELECT
                    g.{', g.'.join(SELECTED_GAIA_PARAMETERS)},
                    CASE
                        WHEN e.e_v_class_star >= 0.65 THEN 'STAR'
                        WHEN e.e_v_class_star < 0.65 THEN 'GALAXY'
                    END as label
                FROM gaia_dr3.gaia_source AS g
                INNER JOIN siwei_zhao_2025.crossmatch_ero_gaia_0_3 AS e
                    ON g.source_id = e.g_source_id
                WHERE e.e_v_class_star IS NOT NULL
                    AND (-2.5 * LOG(g.phot_g_mean_flux) + 25.69) >= 18
                LIMIT {test_size}
            """

        print(f"  Querying {test_size:,} test samples from {data_source.upper()}...")
        df = pd.read_sql_query(query, conn)
        conn.close()

        # Clean data
        df_features = df[SELECTED_GAIA_PARAMETERS].copy()
        df_features = df_features.replace('NOT_AVAILABLE', np.nan)
        for col in SELECTED_GAIA_PARAMETERS:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')

        X = df_features.values
        y = df['label'].map(CLASS_MAPPING).values

        print(f"  ✓ Retrieved {len(df):,} sources")
        print(f"    STAR: {(df['label']=='STAR').sum():,} ({100*(df['label']=='STAR').sum()/len(df):.1f}%)")
        print(f"    GALAXY: {(df['label']=='GALAXY').sum():,} ({100*(df['label']=='GALAXY').sum()/len(df):.1f}%)")

        return X, y, df['label']

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


def plot_comparison(results_list, output_path):
    """Plot comparison of all cross-validation results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data for plotting
    labels = [f"{r['model']} → {r['test_data']}" for r in results_list]
    colors = ['steelblue', 'coral', 'lightgreen', 'gold']

    # 1. Galaxy Recall (most important metric)
    ax = axes[0, 0]
    galaxy_recalls = [r['galaxy_recall'] for r in results_list]
    bars = ax.bar(range(len(labels)), galaxy_recalls, color=colors[:len(labels)])
    ax.set_ylabel('Galaxy Recall', fontsize=11, fontweight='bold')
    ax.set_title('Galaxy Detection Rate', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, galaxy_recalls)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold')

    # 2. Overall Accuracy
    ax = axes[0, 1]
    accuracies = [r['accuracy'] for r in results_list]
    bars = ax.bar(range(len(labels)), accuracies, color=colors[:len(labels)])
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Overall Accuracy', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, accuracies)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold')

    # 3. Galaxy Precision
    ax = axes[1, 0]
    galaxy_precisions = [r['galaxy_precision'] for r in results_list]
    bars = ax.bar(range(len(labels)), galaxy_precisions, color=colors[:len(labels)])
    ax.set_ylabel('Galaxy Precision', fontsize=11, fontweight='bold')
    ax.set_title('Galaxy Classification Precision', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, galaxy_precisions)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold')

    # 4. ROC AUC
    ax = axes[1, 1]
    roc_aucs = [r['roc_auc'] for r in results_list]
    bars = ax.bar(range(len(labels)), roc_aucs, color=colors[:len(labels)])
    ax.set_ylabel('ROC AUC', fontsize=11, fontweight='bold')
    ax.set_title('ROC AUC Score', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, roc_aucs)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Cross-Dataset Validation: Model Generalization Analysis',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Comparison plot saved: {output_path}")


def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    print("="*80)
    print("Cross-Dataset Validation")
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

    # Query test data from both datasets
    print("\nQuerying test data...")
    q1_data = query_test_data(username, 'q1', test_size=30000)
    ero_data = query_test_data(username, 'ero', test_size=30000)

    if q1_data is None or ero_data is None:
        return 1

    X_q1, y_q1, _ = q1_data
    X_ero, y_ero, _ = ero_data

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

    # Plot comparison
    plot_comparison(results, OUTPUT_DIR / 'cross_validation_comparison.png')

    # Save detailed reports
    for r in results:
        report_path = OUTPUT_DIR / f"{r['model']}_on_{r['test_data']}_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"Model: {r['model']}\n")
            f.write(f"Test Data: {r['test_data']}\n")
            f.write(f"\n{r['report']}\n")
            f.write(f"\nROC AUC: {r['roc_auc']:.4f}\n")
        print(f"  ✓ Saved: {report_path}")

    print("\n" + "="*80)
    print("Cross-validation complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
