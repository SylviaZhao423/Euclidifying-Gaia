#!/usr/bin/env python3
"""
Ablation study: compare derived feature variants.

Tests 4 combinations:
  A) pm_significance (chi2)     + position_error (simple)
  B) pm_significance (chi2)     + position_error_corr (with corr)
  C) pm_significance_ep (errprop) + position_error (simple)
  D) pm_significance_ep (errprop) + position_error_corr (with corr)

For each, trains on Q1 and ERO, evaluates on own test set + cross-dataset.
Reports Galaxy Precision & Recall for all 4x4=16 evaluations.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import GAIA_ORIGINAL_COLUMNS

PROJECT_DIR = Path(__file__).parent.parent
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

CLASS_MAPPING = {'STAR': 0, 'GALAXY': 1}
MAG_CUT = 18.0

# Base derived features (always included)
BASE_DERIVED = ['corrected_bp_rp_excess']

# Variants to test
VARIANTS = {
    'A: chi2 + simple':    ['pm_significance', 'position_error'],
    'B: chi2 + corr':      ['pm_significance', 'position_error_corr'],
    'C: errprop + simple':  ['pm_significance_ep', 'position_error'],
    'D: errprop + corr':    ['pm_significance_ep', 'position_error_corr'],
}


def query_data(username, data_source, features):
    """Query dataset with specific feature list."""
    import psycopg2

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in features])

    if data_source == 'q1':
        query = f"""
            SELECT {gaia_cols},
                   CASE WHEN p_phz_classification = 1 THEN 'STAR'
                        WHEN p_phz_classification = 2 THEN 'GALAXY' END as label
            FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
            WHERE p_phz_classification IN (1, 2)
                AND g_phot_g_mean_mag >= {MAG_CUT}
            ORDER BY g_source_id
        """
    else:
        query = f"""
            SELECT {gaia_cols},
                   CASE WHEN e_v_class_star >= 0.65 THEN 'STAR'
                        WHEN e_v_class_star < 0.65 THEN 'GALAXY' END as label
            FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
            WHERE e_v_class_star IS NOT NULL
                AND g_phot_g_mean_mag >= {MAG_CUT}
            ORDER BY g_source_id
        """

    df = pd.read_sql_query(query, conn)
    conn.close()

    for col in features:
        df[col] = pd.to_numeric(df[col].replace('NOT_AVAILABLE', np.nan), errors='coerce')

    X = df[features].values
    y = df['label'].map(CLASS_MAPPING).values
    return X, y


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train model and return galaxy metrics on test set."""
    model = xgb.XGBClassifier(
        objective='binary:logistic', max_depth=8, learning_rate=0.1,
        n_estimators=200, subsample=0.8, colsample_bytree=0.8,
        random_state=42, tree_method='hist', eval_metric='logloss',
        early_stopping_rounds=20
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1])
    auc = roc_auc_score(y_test, y_proba)

    return {
        'galaxy_precision': precision[1],
        'galaxy_recall': recall[1],
        'galaxy_f1': f1[1],
        'roc_auc': auc,
        'model': model
    }


def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER not set")
        return 1

    print("=" * 90)
    print("Ablation Study: Derived Feature Variants")
    print("=" * 90)

    all_results = []

    for variant_name, variant_features in VARIANTS.items():
        features = GAIA_ORIGINAL_COLUMNS + BASE_DERIVED + variant_features
        print(f"\n{'='*90}")
        print(f"Variant {variant_name} ({len(features)} features)")
        print(f"  Derived: {BASE_DERIVED + variant_features}")
        print(f"{'='*90}")

        # Query both datasets with this feature set
        print("  Querying Q1...")
        X_q1, y_q1 = query_data(username, 'q1', features)
        print(f"    {len(X_q1):,} sources")

        print("  Querying ERO...")
        X_ero, y_ero = query_data(username, 'ero', features)
        print(f"    {len(X_ero):,} sources")

        # Split both datasets (same random_state as main training script)
        X_q1_temp, X_q1_test, y_q1_temp, y_q1_test = train_test_split(
            X_q1, y_q1, test_size=0.15, random_state=42, stratify=y_q1)
        X_q1_train, X_q1_val, y_q1_train, y_q1_val = train_test_split(
            X_q1_temp, y_q1_temp, test_size=0.176, random_state=42, stratify=y_q1_temp)

        X_ero_temp, X_ero_test, y_ero_temp, y_ero_test = train_test_split(
            X_ero, y_ero, test_size=0.15, random_state=42, stratify=y_ero)
        X_ero_train, X_ero_val, y_ero_train, y_ero_val = train_test_split(
            X_ero_temp, y_ero_temp, test_size=0.176, random_state=42, stratify=y_ero_temp)

        # Train Q1 model
        print("  Training Q1 model...")
        q1_result = train_and_evaluate(X_q1_train, y_q1_train, X_q1_val, y_q1_val,
                                        X_q1_test, y_q1_test)
        q1_model = q1_result['model']

        # Train ERO model
        print("  Training ERO model...")
        ero_result = train_and_evaluate(X_ero_train, y_ero_train, X_ero_val, y_ero_val,
                                         X_ero_test, y_ero_test)
        ero_model = ero_result['model']

        # Cross-evaluate
        print("  Cross-evaluating...")

        # Q1 model on ERO test
        y_pred = q1_model.predict(X_ero_test)
        y_proba = q1_model.predict_proba(X_ero_test)[:, 1]
        p, r, f, _ = precision_recall_fscore_support(y_ero_test, y_pred, labels=[0, 1])
        q1_on_ero = {'galaxy_precision': p[1], 'galaxy_recall': r[1], 'galaxy_f1': f[1],
                      'roc_auc': roc_auc_score(y_ero_test, y_proba)}

        # ERO model on Q1 test
        y_pred = ero_model.predict(X_q1_test)
        y_proba = ero_model.predict_proba(X_q1_test)[:, 1]
        p, r, f, _ = precision_recall_fscore_support(y_q1_test, y_pred, labels=[0, 1])
        ero_on_q1 = {'galaxy_precision': p[1], 'galaxy_recall': r[1], 'galaxy_f1': f[1],
                      'roc_auc': roc_auc_score(y_q1_test, y_proba)}

        all_results.append({
            'variant': variant_name,
            'Q1→Q1_prec': q1_result['galaxy_precision'],
            'Q1→Q1_rec': q1_result['galaxy_recall'],
            'Q1→ERO_prec': q1_on_ero['galaxy_precision'],
            'Q1→ERO_rec': q1_on_ero['galaxy_recall'],
            'ERO→ERO_prec': ero_result['galaxy_precision'],
            'ERO→ERO_rec': ero_result['galaxy_recall'],
            'ERO→Q1_prec': ero_on_q1['galaxy_precision'],
            'ERO→Q1_rec': ero_on_q1['galaxy_recall'],
        })

    # Print comparison table
    print("\n\n" + "=" * 90)
    print("GALAXY PRECISION & RECALL COMPARISON")
    print("=" * 90)
    print(f"{'Variant':<22} {'Q1→Q1':>12} {'Q1→ERO':>12} {'ERO→ERO':>12} {'ERO→Q1':>12}")
    print(f"{'':22} {'Prec  Rec':>12} {'Prec  Rec':>12} {'Prec  Rec':>12} {'Prec  Rec':>12}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['variant']:<22} "
              f"{r['Q1→Q1_prec']:.3f} {r['Q1→Q1_rec']:.3f}  "
              f"{r['Q1→ERO_prec']:.3f} {r['Q1→ERO_rec']:.3f}  "
              f"{r['ERO→ERO_prec']:.3f} {r['ERO→ERO_rec']:.3f}  "
              f"{r['ERO→Q1_prec']:.3f} {r['ERO→Q1_rec']:.3f}")

    # Save to CSV
    output_dir = PROJECT_DIR / 'output' / 'ablation'
    output_dir.mkdir(parents=True, exist_ok=True)
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(output_dir / 'derived_feature_ablation.csv', index=False)
    print(f"\nResults saved to: {output_dir / 'derived_feature_ablation.csv'}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
