#!/usr/bin/env python3
"""
XGBoost Multi-class Classifier: Gaia Parameters → PHZ Classifications
=======================================================================

Trains a multi-class XGBoost classifier to predict Euclid PHZ classifications
using Gaia DR3 parameters as features.

Data Source: WSDB crossmatch_q1_mer_phz_gaia table
Features: ~50 Gaia parameters (astrometry, photometry, errors, etc.)
Target: PHZ classification (1=STAR, 2=GALAXY, 4=QSO, 8=GLOBULAR_CLUSTER)

The classifier learns to identify source types (STAR/GALAXY/QSO/GC) based on
their Gaia characteristics, using Euclid PHZ pipeline classifications as ground truth.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, accuracy_score, log_loss
)
import xgboost as xgb
import psycopg2
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

print("="*80)
print("XGBoost: Gaia Parameters → Euclid PHZ Classifications")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'
WSDB_USER = os.getenv('PGUSER')
WSDB_TABLE = 'siwei_zhao_2025.q1_mer_phz_gaia'  # Your crossmatch table

# Gaia features to use (comprehensive set)
GAIA_FEATURES = [
    # Astrometry
    'ra', 'dec', 'ra_error', 'dec_error',
    'parallax', 'parallax_error',
    'pm_ra_cosdec', 'pm_dec', 'pm_ra_cosdec_error', 'pm_dec_error',

    # Magnitudes & Colors
    'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',

    # Photometric properties
    'phot_g_mean_flux', 'phot_g_mean_flux_error',
    'phot_bp_mean_flux', 'phot_bp_mean_flux_error',
    'phot_rp_mean_flux', 'phot_rp_mean_flux_error',

    # Astrometric quality
    'astrometric_excess_noise', 'astrometric_excess_noise_sig',

    # Variability
    'phot_variable_flag', 'l', 'b',

    # Spectral parameters
    'teff_gspphot', 'teff_gspphot_lower', 'teff_gspphot_upper',
    'logg_gspphot', 'logg_gspphot_lower', 'logg_gspphot_upper',
    'mh_gspphot', 'mh_gspphot_lower', 'mh_gspphot_upper',
    'distance_gspphot', 'distance_gspphot_lower', 'distance_gspphot_upper',

    # Other quality metrics
    'visibility_periods_used', 'astrometric_sigma5d_max',
    'phot_bp_rp_excess_factor', 'ipd_gof_harmonic_amplitude',
]

# PHZ classification mapping
PHZ_CLASSES = {
    1: 'STAR',
    2: 'GALAXY',
    4: 'QSO',
    8: 'GLOBULAR_CLUSTER'
}

print(f"\nConfiguration:")
print(f"  WSDB Host: {WSDB_HOST}")
print(f"  Database: {WSDB_DB}")
print(f"  Table: {WSDB_TABLE}")
print(f"  User: {WSDB_USER}")
print(f"  Gaia Features: {len(GAIA_FEATURES)}")
print(f"  PHZ Classes: {list(PHZ_CLASSES.values())}")

# ============================================================================
# STEP 1: Query Data from WSDB
# ============================================================================
print("\nStep 1: Querying data from WSDB...")

try:
    conn = psycopg2.connect(
        host=WSDB_HOST,
        database=WSDB_DB,
        user=WSDB_USER
    )
    cursor = conn.cursor()

    # Build query
    feature_list = ', '.join(GAIA_FEATURES)
    query = f"""
    SELECT
        {feature_list},
        phz_classification
    FROM {WSDB_TABLE}
    WHERE phz_classification IN (1, 2, 4, 8)
    AND ra IS NOT NULL
    AND dec IS NOT NULL
    LIMIT 1000000
    """

    print(f"  Executing query (fetching up to 1M rows)...")
    df = pd.read_sql(query, conn)

    cursor.close()
    conn.close()

    print(f"  ✓ Loaded {len(df):,} rows")
    print(f"  Shape: {df.shape}")

except Exception as e:
    print(f"  ✗ Error connecting to WSDB: {e}")
    print(f"  Make sure PGUSER environment variable is set")
    exit(1)

# ============================================================================
# STEP 2: Data Cleaning and Exploration
# ============================================================================
print("\nStep 2: Data exploration and cleaning...")

# Target variable
print(f"\n  Target Distribution (phz_classification):")
target_counts = df['phz_classification'].value_counts().sort_index()
for class_id, count in target_counts.items():
    class_name = PHZ_CLASSES.get(class_id, 'UNKNOWN')
    pct = 100 * count / len(df)
    print(f"    {class_id} ({class_name:20s}): {count:8,d} ({pct:5.1f}%)")

# Missing values analysis
print(f"\n  Missing Values Analysis:")
missing = df[GAIA_FEATURES].isnull().sum()
missing_pct = 100 * missing / len(df)

features_with_missing = missing[missing > 0].sort_values(ascending=False)
if len(features_with_missing) > 0:
    print(f"    Features with missing values:")
    for feat, count in features_with_missing.head(10).items():
        pct = 100 * count / len(df)
        print(f"      {feat:35s}: {count:8,d} ({pct:5.1f}%)")
else:
    print(f"    No missing values!")

# Remove rows with any missing features
df_clean = df[GAIA_FEATURES + ['phz_classification']].dropna()
print(f"\n  After removing NaN: {len(df_clean):,} rows ({100*len(df_clean)/len(df):.1f}% retained)")

# ============================================================================
# STEP 3: Prepare Features and Target
# ============================================================================
print("\nStep 3: Feature preparation...")

X = df_clean[GAIA_FEATURES]
y = df_clean['phz_classification']

print(f"  Features shape: {X.shape}")
print(f"  Target shape: {y.shape}")
print(f"  Feature list:")
for i, feat in enumerate(GAIA_FEATURES, 1):
    print(f"    {i:2d}. {feat}")

# ============================================================================
# STEP 4: Train-Test Split
# ============================================================================
print("\nStep 4: Train-test split (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Training set: {len(X_train):,} samples")
print(f"    Distribution:")
for class_id in sorted(y_train.unique()):
    count = (y_train == class_id).sum()
    pct = 100 * count / len(y_train)
    class_name = PHZ_CLASSES.get(class_id, 'UNKNOWN')
    print(f"      {class_id} ({class_name:20s}): {count:8,d} ({pct:5.1f}%)")

print(f"  Test set: {len(X_test):,} samples")
print(f"    Distribution:")
for class_id in sorted(y_test.unique()):
    count = (y_test == class_id).sum()
    pct = 100 * count / len(y_test)
    class_name = PHZ_CLASSES.get(class_id, 'UNKNOWN')
    print(f"      {class_id} ({class_name:20s}): {count:8,d} ({pct:5.1f}%)")

# ============================================================================
# STEP 5: Feature Scaling
# ============================================================================
print("\nStep 5: Feature scaling (StandardScaler)...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  ✓ Scaler fitted on training data")
print(f"  Mean (first 5): {scaler.mean_[:5]}")
print(f"  Std (first 5):  {scaler.scale_[:5]}")

# ============================================================================
# STEP 6: Train XGBoost Multi-class Classifier
# ============================================================================
print("\nStep 6: Training XGBoost classifier...")

xgb_params = {
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'objective': 'multi:softmax',  # Multi-class classification
    'num_class': 4,  # 4 classes: STAR, GALAXY, QSO, GC
    'use_label_encoder': False,
    'eval_metric': 'mlogloss',
}

model = xgb.XGBClassifier(**xgb_params)
model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)

print(f"  ✓ Model trained successfully")
print(f"  Parameters: {xgb_params}")

# ============================================================================
# STEP 7: Model Evaluation
# ============================================================================
print("\nStep 7: Model evaluation...")

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
y_test_proba = model.predict_proba(X_test_scaled)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average='weighted')
test_logloss = log_loss(y_test, y_test_proba)

print(f"  Training Accuracy:   {train_acc:.4f}")
print(f"  Test Accuracy:       {test_acc:.4f}")
print(f"  Test F1-Score:       {test_f1:.4f}")
print(f"  Test Log Loss:       {test_logloss:.4f}")

print(f"\n  Classification Report (Test Set):")
target_names = [PHZ_CLASSES[cid] for cid in sorted(PHZ_CLASSES.keys())]
print(classification_report(y_test, y_test_pred, target_names=target_names))

print(f"\n  Confusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# ============================================================================
# STEP 8: Feature Importance
# ============================================================================
print("\nStep 8: Feature importance analysis...")

feature_importance = pd.DataFrame({
    'feature': GAIA_FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n  Top 20 Most Important Features:")
for idx, row in feature_importance.head(20).iterrows():
    print(f"    {row['feature']:35s}: {row['importance']:.6f}")

# ============================================================================
# STEP 9: Visualization
# ============================================================================
print("\nStep 9: Creating visualizations...")

fig = plt.figure(figsize=(20, 12))

# Plot 1: Confusion Matrix
ax1 = plt.subplot(2, 3, 1)
im = ax1.imshow(cm, cmap='Blues', aspect='auto')
ax1.set_xticks(range(len(PHZ_CLASSES)))
ax1.set_yticks(range(len(PHZ_CLASSES)))
ax1.set_xticklabels([PHZ_CLASSES[cid] for cid in sorted(PHZ_CLASSES.keys())])
ax1.set_yticklabels([PHZ_CLASSES[cid] for cid in sorted(PHZ_CLASSES.keys())])
ax1.set_xlabel('Predicted', fontsize=11)
ax1.set_ylabel('True', fontsize=11)
ax1.set_title(f'Confusion Matrix (Accuracy = {test_acc:.3f})', fontsize=12, fontweight='bold')

# Add text annotations
for i in range(len(PHZ_CLASSES)):
    for j in range(len(PHZ_CLASSES)):
        text = ax1.text(j, i, cm[i, j], ha="center", va="center", color="w", fontsize=10)

plt.colorbar(im, ax=ax1, label='Count')

# Plot 2: Feature Importance (Top 20)
ax2 = plt.subplot(2, 3, 2)
top_features = feature_importance.head(20)
ax2.barh(range(len(top_features)), top_features['importance'])
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(top_features['feature'], fontsize=9)
ax2.set_xlabel('Importance', fontsize=11)
ax2.set_title('Top 20 Feature Importance', fontsize=12, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Class Distribution
ax3 = plt.subplot(2, 3, 3)
class_counts = y_test.value_counts().sort_index()
colors = plt.cm.Set2(np.linspace(0, 1, len(class_counts)))
ax3.bar([PHZ_CLASSES[cid] for cid in class_counts.index], class_counts.values, color=colors)
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title('Test Set Class Distribution', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Prediction Accuracy by Class
ax4 = plt.subplot(2, 3, 4)
accuracies = []
class_names_list = []
for cid in sorted(PHZ_CLASSES.keys()):
    mask = y_test == cid
    if mask.sum() > 0:
        acc = accuracy_score(y_test[mask], y_test_pred[mask])
        accuracies.append(acc)
        class_names_list.append(PHZ_CLASSES[cid])

ax4.bar(class_names_list, accuracies, color=colors)
ax4.set_ylabel('Accuracy', fontsize=11)
ax4.set_ylim([0, 1])
ax4.set_title('Per-Class Accuracy', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# Plot 5: Prediction Probability Distribution
ax5 = plt.subplot(2, 3, 5)
max_proba = y_test_proba.max(axis=1)
ax5.hist(max_proba, bins=50, color='skyblue', edgecolor='black')
ax5.set_xlabel('Max Prediction Probability', fontsize=11)
ax5.set_ylabel('Frequency', fontsize=11)
ax5.set_title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
ax5.axvline(0.5, color='red', linestyle='--', label='50% threshold')
ax5.legend()
ax5.grid(alpha=0.3, axis='y')

# Plot 6: Summary Statistics
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
summary_text = f"""
XGBoost: Gaia → Euclid PHZ Classification

Dataset:
  Training samples:  {len(X_train):,}
  Test samples:      {len(X_test):,}
  Total features:    {len(GAIA_FEATURES)}
  Classes:           4 (STAR, GALAXY, QSO, GC)

Model Parameters:
  Max depth:         {xgb_params['max_depth']}
  Learning rate:     {xgb_params['learning_rate']}
  N estimators:      {xgb_params['n_estimators']}
  Subsample:         {xgb_params['subsample']}

Performance (Test Set):
  Accuracy:          {test_acc:.4f}
  F1-Score:          {test_f1:.4f}
  Log Loss:          {test_logloss:.4f}

Per-Class Accuracies:
""" + "\n".join([f"  {PHZ_CLASSES[cid]:20s}: {accuracies[i]:.4f}"
                 for i, cid in enumerate(sorted(PHZ_CLASSES.keys()))])

ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
    verticalalignment='center',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
output_path = '/Users/sylviazhao/Downloads/III/Euclidifying-Gaia/output/xgboost_gaia_phz_classification.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_path}")

# ============================================================================
# STEP 10: Save Model Artifacts
# ============================================================================
print("\nStep 10: Saving model artifacts...")

output_dir = '/Users/sylviazhao/Downloads/III/Euclidifying-Gaia/output'

# Save model
model_path = f'{output_dir}/xgboost_gaia_phz.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"  ✓ Saved model: {model_path}")

# Save scaler
scaler_path = f'{output_dir}/scaler_gaia_phz.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"  ✓ Saved scaler: {scaler_path}")

# Save feature importance
fi_path = f'{output_dir}/feature_importance_gaia_phz.csv'
feature_importance.to_csv(fi_path, index=False)
print(f"  ✓ Saved feature importance: {fi_path}")

# Save feature list
features_path = f'{output_dir}/features_gaia_phz.pkl'
with open(features_path, 'wb') as f:
    pickle.dump(GAIA_FEATURES, f)
print(f"  ✓ Saved feature list: {features_path}")

# Save class mapping
class_mapping_path = f'{output_dir}/phz_classes.pkl'
with open(class_mapping_path, 'wb') as f:
    pickle.dump(PHZ_CLASSES, f)
print(f"  ✓ Saved class mapping: {class_mapping_path}")

print("\n" + "="*80)
print("XGBoost Training Complete!")
print("="*80)
print(f"\nGenerated Files:")
print(f"  - output/xgboost_gaia_phz.pkl")
print(f"  - output/scaler_gaia_phz.pkl")
print(f"  - output/feature_importance_gaia_phz.csv")
print(f"  - output/features_gaia_phz.pkl")
print(f"  - output/phz_classes.pkl")
print(f"  - output/xgboost_gaia_phz_classification.png")
