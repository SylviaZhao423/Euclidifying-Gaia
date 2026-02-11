#!/usr/bin/env python3
"""
Train XGBoost binary classifier for Gaia-Euclid source classification.

Classes: STAR (class 0) vs GALAXY (class 1)
Ground truth options:
    - Q1: Euclid PHZ classifications (classes 1 & 2, Gaia G >= 18, ~218k sources)
    - ERO: Euclid morphological class_star (>=0.65=STAR, <0.65=GALAXY, Gaia G >= 18, ~268k sources)
Features: 53 selected Gaia DR3 parameters (excluding data availability flags and features unavailable at G >= 18)

Usage:
    python src/train_xgboost_classifier.py [--test-mode] [--data-source {q1,ero}]

Examples:
    # Test on Q1 data
    python src/train_xgboost_classifier.py --test-mode --data-source q1

    # Full training on ERO data
    python src/train_xgboost_classifier.py --data-source ero
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS, GAIA_QUERY_COLUMNS

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_BASE_DIR = PROJECT_DIR / 'output'
OUTPUT_DIR = None  # Will be set in main() based on data source

# WSDB Configuration
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

# Class mapping - Binary classification
CLASS_MAPPING = {
    'STAR': 0,
    'GALAXY': 1
}
CLASS_NAMES = list(CLASS_MAPPING.keys())


def query_training_data(username, limit=None, test_mode=False, data_source='q1', mag_cut=18.0):
    """
    Query Gaia features + Euclid classifications from WSDB.

    Parameters:
    -----------
    username : str
        WSDB username
    limit : int, optional
        Limit number of rows (for testing)
    test_mode : bool
        Use smaller sample for quick testing
    data_source : str
        'q1' for Q1 PHZ classifications or 'ero' for ERO morphological classifications
    mag_cut : float
        Minimum Gaia G magnitude cut

    Returns:
    --------
    pd.DataFrame: Training data with features and labels
    """
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed!")
        print("Install with: pip install psycopg2-binary")
        return None

    try:
        conn = psycopg2.connect(
            host=WSDB_HOST,
            database=WSDB_DB,
            user=username
        )

        # Build column list with g_ prefix, aliased back to standard names
        gaia_cols = ', '.join([f'g_{col} AS {col}' for col in GAIA_QUERY_COLUMNS])

        if data_source == 'q1':
            # Query Q1 crossmatch table directly (no JOIN needed)
            # p_phz_classification: 1=STAR, 2=GALAXY
            # Magnitude cut at >= 18: PHZ unreliable at bright end (see Project Report page 9)
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
        elif data_source == 'ero':
            # Query ERO crossmatch table directly (no JOIN needed)
            # Use class_star (validated classifier): >=0.65=STAR, <0.65=GALAXY
            # Threshold consistent with all other analysis scripts in project
            # Magnitude cut at >= 18: consistent with Q1 for fair comparison
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
        else:
            raise ValueError(f"Unknown data_source: {data_source}. Use 'q1' or 'ero'")

        if test_mode:
            query += " LIMIT 10000"
        elif limit:
            query += f" LIMIT {limit}"

        print(f"Querying training data from WSDB...")
        print(f"  Data source: {data_source.upper()}")
        print(f"  Username: {username}")
        if test_mode:
            print(f"  Test mode: limiting to 10,000 sources")

        df = pd.read_sql_query(query, conn)
        conn.close()

        print(f"  ✓ Retrieved {len(df):,} sources")
        print(f"\n  Class distribution:")
        for class_name in CLASS_NAMES:
            count = (df['label'] == class_name).sum()
            percent = 100 * count / len(df)
            print(f"    {class_name:7s}: {count:7,d} ({percent:5.2f}%)")

        return df

    except Exception as e:
        print(f"  ✗ Error querying data: {e}")
        import traceback
        traceback.print_exc()
        return None


def prepare_data(df):
    """
    Prepare features and labels for training.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw data from query

    Returns:
    --------
    X, y, feature_names
    """
    # Replace 'NOT_AVAILABLE' strings with NaN (Gaia DR3 encoding)
    df = df.copy()
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)

    # Convert all columns to numeric
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Select features
    df_features = df[SELECTED_GAIA_PARAMETERS]

    # Separate features and labels
    X = df_features.values
    y = df['label'].map(CLASS_MAPPING).values

    # Check for missing values
    nan_counts = df_features.isna().sum()
    if nan_counts.sum() > 0:
        print(f"\nMissing value summary:")
        for feat, count in nan_counts[nan_counts > 0].items():
            percent = 100 * count / len(df)
            print(f"  {feat}: {count:,} ({percent:.2f}%)")
        print("\n  XGBoost will handle missing values internally")

    return X, y, SELECTED_GAIA_PARAMETERS


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train XGBoost multi-class classifier.

    Parameters:
    -----------
    X_train, y_train : training data
    X_val, y_val : validation data

    Returns:
    --------
    trained model
    """
    print("\nTraining XGBoost classifier...")

    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',  # Binary classification
        'max_depth': 8,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist',  # Faster training
        'eval_metric': 'logloss',
        'early_stopping_rounds': 20,  # Stop if no improvement for 20 rounds
    }

    model = xgb.XGBClassifier(**params)

    # Train with validation set
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    print(f"\n  ✓ Training completed")
    print(f"    Best iteration: {model.best_iteration}")
    print(f"    Best score: {model.best_score:.4f}")

    return model


def evaluate_model(model, X_test, y_test, feature_names, data_source, df=None):
    """
    Evaluate trained model and generate plots.
    """
    print("\nEvaluating model on test set...")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Classification report
    print("\nClassification Report:")
    print("=" * 70)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    print(report)

    # Save report
    with open(OUTPUT_DIR / f'{data_source}_classification_report.txt', 'w') as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, CLASS_NAMES, data_source)

    # Feature importance
    plot_feature_importance(model, feature_names, data_source)

    # ROC curves (one-vs-rest for each class)
    plot_roc_curves(y_test, y_pred_proba, CLASS_NAMES, data_source)

    # Feature distributions (star vs galaxy histograms)
    if df is not None:
        plot_feature_distributions(df, model, feature_names, data_source)

    print(f"\n  ✓ Evaluation complete")
    print(f"    Results saved to: {OUTPUT_DIR}")


def plot_confusion_matrix(cm, class_names, data_source):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{data_source}_confusion_matrix.png', dpi=150)
    plt.close()
    print(f"  ✓ Confusion matrix saved")


def plot_feature_importance(model, feature_names, data_source, top_n=None):
    """Plot and save feature importance for all features."""
    importance = model.feature_importances_

    # Get all features sorted by importance
    indices_all = np.argsort(importance)[::-1]

    # Use all features if top_n not specified
    if top_n is None:
        top_n = len(feature_names)

    indices = indices_all[:top_n]

    # Adjust figure size based on number of features
    fig_height = max(10, top_n * 0.35)
    plt.figure(figsize=(12, fig_height))
    plt.barh(range(top_n), importance[indices], color='steelblue')
    plt.yticks(range(top_n), [feature_names[i] for i in indices], fontsize=9)
    plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
    plt.title(f'Feature Importance Rankings (All {len(feature_names)} Features)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{data_source}_feature_importance.png', dpi=150)
    plt.close()
    print(f"  ✓ Feature importance plot saved (all {len(feature_names)} features)")

    # Save ALL importance values to CSV (not just top_n)
    importance_df = pd.DataFrame({
        'rank': range(1, len(feature_names) + 1),
        'feature': [feature_names[i] for i in indices_all],
        'importance': importance[indices_all]
    })
    importance_df.to_csv(OUTPUT_DIR / f'{data_source}_feature_importance.csv', index=False)
    print(f"  ✓ Feature importance CSV saved (all {len(feature_names)} features ranked)")


def plot_feature_distributions(df, model, feature_names, data_source, top_n=12):
    """Plot star vs galaxy distributions for top features by importance."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]

    # Clean features
    df_clean = df[list(top_features) + ['label']].copy()
    df_clean[top_features] = df_clean[top_features].replace('NOT_AVAILABLE', np.nan)
    for col in top_features:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Grid plot
    n_cols = 3
    n_rows = (top_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        feat_data = df_clean[[feature, 'label']].dropna()
        if len(feat_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(feature)
            continue

        stars = feat_data[feat_data['label'] == 'STAR'][feature].astype(float)
        galaxies = feat_data[feat_data['label'] == 'GALAXY'][feature].astype(float)

        ax.hist(stars, bins=30, alpha=0.6, label='STAR', color='steelblue', density=True)
        ax.hist(galaxies, bins=30, alpha=0.6, label='GALAXY', color='coral', density=True)
        ax.set_xlabel(feature, fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(feature, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')

    for idx in range(len(top_features), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Top {top_n} Feature Distributions: Stars vs Galaxies ({data_source.upper()})',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{data_source}_top_features_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Feature distribution grid saved (top {top_n} features)")


def plot_roc_curves(y_true, y_pred_proba, class_names, data_source):
    """Plot ROC curve for binary classification."""
    # For binary classification, use probabilities of positive class (GALAXY)
    y_pred_proba_pos = y_pred_proba[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba_pos)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, lw=2, color='steelblue',
             label=f'Binary Classifier (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curve: STAR vs GALAXY Classification', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{data_source}_roc_curve.png', dpi=150)
    plt.close()
    print(f"  ✓ ROC curve saved")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Train XGBoost binary classifier for Gaia-Euclid STAR/GALAXY classification'
    )
    parser.add_argument('--test-mode', action='store_true',
                        help='Use small sample for quick testing (10k sources)')
    parser.add_argument('--save-model', type=str, default='xgb_classifier.json',
                        help='Filename to save trained model')
    parser.add_argument('--data-source', type=str, default=None, choices=['q1', 'ero'],
                        help='Data source: q1, ero, or omit for both')
    args = parser.parse_args()

    # Get username
    username = os.getenv('PGUSER')
    if not username:
        print("\nERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return 1

    # Run for both data sources if not specified
    sources = [args.data_source] if args.data_source else ['q1', 'ero']

    mag_cut = 18.0  # hardcoded magnitude cut

    for data_source in sources:
        result = train_single(username, data_source, args.test_mode, args.save_model, mag_cut)
        if result != 0:
            return result

    return 0


def train_single(username, data_source, test_mode, save_model, mag_cut=18.0):
    """Train and evaluate a single model for the given data source."""
    global OUTPUT_DIR
    OUTPUT_DIR = OUTPUT_BASE_DIR / f'xgboost_{data_source}'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("XGBoost Binary Classifier Training: STAR vs GALAXY")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Features: {len(SELECTED_GAIA_PARAMETERS)}")
    print(f"  Classes: {', '.join(CLASS_NAMES)}")
    print(f"  Data source: {data_source.upper()}")
    print(f"  Magnitude cut: G >= {mag_cut}")
    if data_source == 'q1':
        print(f"    Ground truth: Euclid Q1 PHZ classifications")
    else:
        print(f"    Ground truth: Euclid ERO class_star (>=0.65=STAR, <0.65=GALAXY)")
    print(f"  Test mode: {test_mode}")
    print(f"  Output directory: {OUTPUT_DIR}")

    # Query data
    df = query_training_data(username, test_mode=test_mode, data_source=data_source, mag_cut=mag_cut)
    if df is None:
        return 1

    # Prepare data
    X, y, feature_names = prepare_data(df)

    # Split data: 70% train, 15% validation, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 ≈ 0.15/0.85
    )

    print(f"\nData split:")
    print(f"  Training:   {len(X_train):7,} samples ({100*len(X_train)/len(X):.1f}%)")
    print(f"  Validation: {len(X_val):7,} samples ({100*len(X_val)/len(X):.1f}%)")
    print(f"  Test:       {len(X_test):7,} samples ({100*len(X_test)/len(X):.1f}%)")

    # Train model
    model = train_xgboost(X_train, y_train, X_val, y_val)

    # Evaluate
    evaluate_model(model, X_test, y_test, feature_names, data_source, df=df)

    # Save model
    if save_model == 'xgb_classifier.json':
        model_filename = f'xgb_classifier_{data_source}.json'
    else:
        model_filename = save_model

    model_path = OUTPUT_DIR / model_filename
    try:
        model.get_booster().save_model(str(model_path))
        print(f"\n  ✓ Model saved to: {model_path}")
    except Exception as e:
        print(f"\n  ⚠ Could not save model: {e}")
        print(f"  Model training completed successfully, but skipping save.")

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
