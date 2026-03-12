#!/usr/bin/env python3
"""
XGBoost combined model: systematic 3×3×3 grid evaluation.

3 datasets (Q1, ERO, Combined) × 3 training ratios (native, @Q1, @ERO)
= 9 model variants, each evaluated on 3 test sets = 27 experiments.

Outputs saved to output/xgboost_combined/.

Usage:
    PGUSER=siwei_zhao_2025 python src/train_xgboost_combined.py [--test-mode]
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

# Universal Gaia G magnitude cut applied to both datasets
MAG_CUT = 17

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
    """Query dataset. Returns X, y, random_index arrays."""
    import psycopg2

    conn = psycopg2.connect(
        host=WSDB_HOST, database=WSDB_DB, user=username
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
    print(f"  -> {len(X):,} sources — STAR: {n_star:,} ({100*n_star/len(X):.1f}%), "
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
    """Downsample training set using random_index for determinism."""
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

def plot_feature_importance_comparison(q1_model, ero_model, combined_model,
                                       feature_names, output_dir):
    """Plot 3-panel feature importance comparison (Q1 / ERO / Combined) and save CSV."""
    q1_imp = q1_model.feature_importances_
    ero_imp = ero_model.feature_importances_
    comb_imp = combined_model.feature_importances_

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'Q1': q1_imp,
        'ERO': ero_imp,
        'Combined': comb_imp,
    })

    n_features = len(feature_names)
    fig_height = max(12, n_features * 0.4)
    fig, axes = plt.subplots(1, 3, figsize=(27, fig_height))

    configs = [
        ('Q1', 'Q1 Model', 'lightsteelblue'),
        ('ERO', 'ERO Model', 'lightsalmon'),
        ('Combined', 'Combined Model', 'lightgreen'),
    ]

    for ax, (col, title, color) in zip(axes, configs):
        sorted_df = importance_df.sort_values(col, ascending=False)
        labels = [f"#{i+1} {f}" for i, f in enumerate(sorted_df['feature'])]
        bars = ax.barh(range(n_features), sorted_df[col].values, color=color)
        ax.set_yticks(range(n_features))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance', fontsize=11, fontweight='bold')
        ax.set_title(f'{title} Feature Importance', fontsize=12, fontweight='bold')
        ax.set_xlim(0, sorted_df[col].max() * 1.25)
        ax.grid(alpha=0.3, axis='x')
        for bar, val in zip(bars, sorted_df[col].values):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                    va='center', fontsize=7)

    plt.suptitle(f'Feature Importance: Q1 vs ERO vs Combined ({n_features} Features)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save CSV
    for col in ['Q1', 'ERO', 'Combined']:
        importance_df[f'{col}_rank'] = importance_df[col].rank(ascending=False).astype(int)
    importance_df.to_csv(output_dir / 'combined_feature_importance.csv', index=False)


def plot_galaxy_metrics_by_test(results_by_row, highlight_positions, output_dir,
                               q1_ratio, combo_ratio, ero_ratio):
    """Plot Galaxy Precision, F1, Recall in a 3-row × 9-column grid.

    Layout: 3 test-set groups (Q1, Combo, ERO) × 3 training ratios each.
    Row labels on left (Model: Q1 / Combo / ERO), column headers on top.

    Parameters
    ----------
    results_by_row : list of (row_title, list of (col_index, result_dict))
        3 rows, each with 9 entries at positions 0..8.
    highlight_positions : set of (row_index, col_index)
        Diagonal positions to highlight with a green box.
    q1_ratio, combo_ratio, ero_ratio : float
        Training ratios for annotation labels.
    """
    n_rows = len(results_by_row)
    n_cols = 9

    # Row labels for left side
    row_model_labels = ['Model: Q1', 'Model: Combo', 'Model: ERO']  # matches dataset_prefixes order

    fig, axes = plt.subplots(n_rows, 1, figsize=(30, 5 * n_rows + 2),
                             gridspec_kw={'hspace': 0.55})
    if n_rows == 1:
        axes = [axes]

    width = 0.2

    # Column group headers
    group_labels = [
        f'Test ratio: Q1 ({q1_ratio:.0f}:1)',
        f'Test ratio: Combo ({combo_ratio:.0f}:1)',
        f'Test ratio: ERO ({ero_ratio:.0f}:1)',
    ]

    for row_idx, (ax, (row_title, entries)) in enumerate(zip(axes, results_by_row)):
        positions = [pos for pos, _ in entries]
        results = [r for _, r in entries]
        labels_at_pos = {pos: r['label'] for pos, r in entries}
        precision = [r['galaxy_precision'] for r in results]
        recall = [r['galaxy_recall'] for r in results]
        f1 = [r['galaxy_f1'] for r in results]

        x = np.array(positions, dtype=float)

        bar_alpha = 0.5
        bars_p = ax.bar(x - width, precision, width, label='Galaxy Precision',
                        color='steelblue', edgecolor='white', linewidth=0.5, alpha=bar_alpha)
        bars_f = ax.bar(x, f1, width, label='Galaxy F1',
                        color='mediumseagreen', edgecolor='white', linewidth=0.5, alpha=bar_alpha)
        bars_r = ax.bar(x + width, recall, width, label='Galaxy Recall',
                        color='coral', edgecolor='white', linewidth=0.5, alpha=bar_alpha)

        # Overlay trend lines connecting the 3 ratio variants within each test-set group
        entries_by_pos = {pos: r for pos, r in entries}
        for g_idx in range(3):
            group_cols = [g_idx * 3 + j for j in range(3)]
            gp = [entries_by_pos[c]['galaxy_precision'] for c in group_cols]
            gf = [entries_by_pos[c]['galaxy_f1'] for c in group_cols]
            gr = [entries_by_pos[c]['galaxy_recall'] for c in group_cols]
            gx = np.array(group_cols, dtype=float)

            ax.plot(gx - width, gp, 'o-', color='steelblue', linewidth=2.5,
                    markersize=5, zorder=5)
            ax.plot(gx, gf, 's-', color='mediumseagreen', linewidth=3,
                    markersize=6, zorder=6)
            ax.plot(gx + width, gr, '^-', color='coral', linewidth=2.5,
                    markersize=5, zorder=5)

        for bar, val in zip(bars_p, precision):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7, color='steelblue')
        for bar, val in zip(bars_f, f1):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold',
                    color='mediumseagreen')
        for bar, val in zip(bars_r, recall):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7, color='coral')

        # Highlight diagonal entries with a green box
        for pos, r in entries:
            if (row_idx, pos) in highlight_positions:
                ax.add_patch(plt.Rectangle(
                    (pos - 0.45, -0.02), 0.9, 1.12,
                    fill=False, edgecolor='red', linewidth=2.5,
                    linestyle='-', zorder=10
                ))

        # Vertical separator lines between test-set groups
        ax.axvline(x=2.5, color='gray', linewidth=1.0, linestyle='--', alpha=0.6)
        ax.axvline(x=5.5, color='gray', linewidth=1.0, linestyle='--', alpha=0.6)

        # Tick labels — horizontal, no rotation
        tick_labels = [labels_at_pos.get(i, '') for i in range(n_cols)]
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(tick_labels, fontsize=9, rotation=0, ha='center')
        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(0, 1.15)

        from matplotlib.transforms import blended_transform_factory
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for g_idx in range(3):
            center_x = g_idx * 3 + 1
            ax.text(center_x, -0.10,
                    '──────→ increasing training S/G ratio──────→',
                    ha='center', va='top', fontsize=11, fontstyle='italic', fontweight='bold',
                    transform=trans, clip_on=False)

        # Row label on the left — boxed, matching the column header style
        ax.set_ylabel('')  # clear default ylabel
        ax.text(-0.06, 0.5, row_model_labels[row_idx], ha='center', va='center',
                fontsize=11, fontweight='bold', rotation=0,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.8),
                transform=ax.transAxes)

        # No per-row title (group headers + suptitle are sufficient)
        ax.grid(alpha=0.3, axis='y')

        # Column group headers (only on first row)
        if row_idx == 0:
            for g_idx, g_label in enumerate(group_labels):
                center_x = g_idx * 3 + 1  # center of each group of 3
                ax.text(center_x, 1.22, g_label, ha='center', va='bottom',
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                                  edgecolor='gray', alpha=0.8),
                        transform=ax.transData)

        # Avg F1 annotation below each row
        f1_values = {pos: r['galaxy_f1'] for pos, r in entries}
        group_avgs = []
        dataset_name = row_title.split('-')[0].strip()
        test_labels = [f'→Q1({q1_ratio:.0f}:1)', f'→Combo({combo_ratio:.0f}:1)', f'→ERO({ero_ratio:.0f}:1)']
        for g_idx in range(3):
            group_f1s = [f1_values[g_idx * 3 + j] for j in range(3)
                         if (g_idx * 3 + j) in f1_values]
            if group_f1s:
                avg = np.mean(group_f1s)
                group_avgs.append(f'{dataset_name}{test_labels[g_idx]} avg F1={avg:.2f}')
            else:
                group_avgs.append('')

        annotation = '  |  '.join(group_avgs)
        ax.text(0.5, -0.25,
                "──────→ increasing test S/G ratio──────→\n" + annotation, ha='center', va='top',
                fontsize=13, fontstyle='italic', fontweight='bold', linespacing=2.5,
                transform=ax.transAxes)

    # Single shared legend in the blank space between suptitle and top-right data set header
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, fontsize=9, loc='lower left',
                   bbox_to_anchor=(0.92, 1.05), framealpha=0.9)

    plt.suptitle('Galaxy Metrics by Test Ratio: 27 Experiments (3 datasets × 3 ratios × 3 test sets)',
                 fontsize=15, fontweight='bold', y=0.94)
    plt.savefig(output_dir / 'combined_galaxy_metrics_by_test.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_galaxy_metrics_by_ratio(all_results, output_dir, q1_ratio, combo_ratio, ero_ratio):
    """Plot Galaxy Precision, F1, Recall in a 3-row × 9-column grid,
    grouped by TRAINING RATIO instead of test set.

    Layout:
      Top headers:  Training ratio: 3:1 (Q1) | 6:1 (Combo) | 19:1 (ERO)
      Row labels:   Model: Q1 | Model: Combo | Model: ERO
      Within each group of 3: →Q1, →Combo, →ERO (test sets)
      Lines connect across test sets within each ratio group.
    """
    model_datasets = ['Q1', 'Comb', 'ERO']
    ratio_suffixes = ['@Q1', '@Comb', '@ERO']
    test_datasets = ['Q1', 'Combo', 'ERO']

    row_labels = ['Model: Q1', 'Model: Combo', 'Model: ERO']
    group_labels = [
        f'Training ratio: Q1 ({q1_ratio:.0f}:1)',
        f'Training ratio: Combo ({combo_ratio:.0f}:1)',
        f'Training ratio: ERO ({ero_ratio:.0f}:1)',
    ]

    highlight_positions = {(0, 0), (1, 4), (2, 8)}
    n_rows = 3
    n_cols = 9
    width = 0.2

    fig, axes = plt.subplots(n_rows, 1, figsize=(30, 5 * n_rows + 2),
                             gridspec_kw={'hspace': 0.55})

    for row_idx, model_ds in enumerate(model_datasets):
        ax = axes[row_idx]

        # Column order: @Q1→Q1, @Q1→Combo, @Q1→ERO, @Combo→Q1, ...
        entries = []
        for ratio_idx, ratio_suf in enumerate(ratio_suffixes):
            for test_idx, test_ds in enumerate(test_datasets):
                col = ratio_idx * 3 + test_idx
                label = f'{model_ds}{ratio_suf}→{test_ds}'
                entries.append((col, all_results[label]))

        positions = [pos for pos, _ in entries]
        results = [r for _, r in entries]
        labels_at_pos = {pos: r['label'] for pos, r in entries}
        precision = [r['galaxy_precision'] for r in results]
        recall = [r['galaxy_recall'] for r in results]
        f1 = [r['galaxy_f1'] for r in results]

        x = np.array(positions, dtype=float)

        bar_alpha = 0.5
        bars_p = ax.bar(x - width, precision, width, label='Galaxy Precision',
                        color='steelblue', edgecolor='white', linewidth=0.5, alpha=bar_alpha)
        bars_f = ax.bar(x, f1, width, label='Galaxy F1',
                        color='mediumseagreen', edgecolor='white', linewidth=0.5, alpha=bar_alpha)
        bars_r = ax.bar(x + width, recall, width, label='Galaxy Recall',
                        color='coral', edgecolor='white', linewidth=0.5, alpha=bar_alpha)

        # Trend lines connecting test sets within each ratio group
        entries_by_pos = {pos: r for pos, r in entries}
        for g_idx in range(3):
            group_cols = [g_idx * 3 + j for j in range(3)]
            gp = [entries_by_pos[c]['galaxy_precision'] for c in group_cols]
            gf = [entries_by_pos[c]['galaxy_f1'] for c in group_cols]
            gr = [entries_by_pos[c]['galaxy_recall'] for c in group_cols]
            gx = np.array(group_cols, dtype=float)

            ax.plot(gx - width, gp, 'o-', color='steelblue', linewidth=2.5,
                    markersize=5, zorder=5)
            ax.plot(gx, gf, 's-', color='mediumseagreen', linewidth=3,
                    markersize=6, zorder=6)
            ax.plot(gx + width, gr, '^-', color='coral', linewidth=2.5,
                    markersize=5, zorder=5)

        for bar, val in zip(bars_p, precision):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7, color='steelblue')
        for bar, val in zip(bars_f, f1):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold',
                    color='mediumseagreen')
        for bar, val in zip(bars_r, recall):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7, color='coral')

        # Diagonal highlight
        for pos, r in entries:
            if (row_idx, pos) in highlight_positions:
                ax.add_patch(plt.Rectangle(
                    (pos - 0.45, -0.02), 0.9, 1.12,
                    fill=False, edgecolor='red', linewidth=2.5,
                    linestyle='-', zorder=10
                ))

        ax.axvline(x=2.5, color='gray', linewidth=1.0, linestyle='--', alpha=0.6)
        ax.axvline(x=5.5, color='gray', linewidth=1.0, linestyle='--', alpha=0.6)

        tick_labels = [labels_at_pos.get(i, '') for i in range(n_cols)]
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(tick_labels, fontsize=9, rotation=0, ha='center')
        ax.set_xlim(-0.5, n_cols - 0.5)

        from matplotlib.transforms import blended_transform_factory
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for g_idx in range(3):
            center_x = g_idx * 3 + 1
            ax.text(center_x, -0.10, '──────→ increasing test S/G ratio ──────→',
                    ha='center', va='top', fontsize=11, fontstyle='italic', fontweight='bold',
                    transform=trans, clip_on=False)
        ax.set_ylim(0, 1.15)

        ax.set_ylabel('')
        ax.text(-0.06, 0.5, row_labels[row_idx], ha='center', va='center',
                fontsize=11, fontweight='bold', rotation=0,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.8),
                transform=ax.transAxes)

        ax.grid(alpha=0.3, axis='y')

        if row_idx == 0:
            for g_idx, g_label in enumerate(group_labels):
                center_x = g_idx * 3 + 1
                ax.text(center_x, 1.22, g_label, ha='center', va='bottom',
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                                  edgecolor='gray', alpha=0.8),
                        transform=ax.transData)

        # Avg F1 annotation — averaged across test sets for each ratio
        f1_values = {pos: r['galaxy_f1'] for pos, r in entries}
        group_avgs = []
        ratio_names = [f'@Q1({q1_ratio:.0f}:1)', f'@Combo({combo_ratio:.0f}:1)',
                       f'@ERO({ero_ratio:.0f}:1)']
        for g_idx in range(3):
            group_f1s = [f1_values[g_idx * 3 + j] for j in range(3)]
            avg = np.mean(group_f1s)
            group_avgs.append(f'{model_ds}{ratio_names[g_idx]} avg F1={avg:.2f}')

        annotation = '  |  '.join(group_avgs)
        ax.text(0.5, -0.25,
                "──────→ increasing training S/G ratio ──────→\n" + annotation, ha='center', va='top',
                fontsize=13, fontstyle='italic', fontweight='bold', linespacing=2.5,
                transform=ax.transAxes)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, fontsize=9, loc='lower left',
                   bbox_to_anchor=(0.92, 1.05), framealpha=0.9)

    plt.suptitle('Galaxy Metrics by Training Ratio: 27 Experiments (3 datasets × 3 ratios × 3 test sets)',
                 fontsize=15, fontweight='bold', y=0.94)
    plt.savefig(output_dir / 'combined_galaxy_metrics_by_ratio.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='XGBoost combined model training & evaluation'
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
    print("XGBoost Combined Model Training & Evaluation (27 experiments)")
    print("=" * 80)
    print(f"  Features: {len(SELECTED_GAIA_PARAMETERS)}")
    print(f"  Magnitude cut: Gaia G >= {MAG_CUT} (both datasets)")
    print(f"  Output: {OUTPUT_DIR}")
    if args.test_mode:
        print(f"  TEST MODE: limited to 10k sources per dataset")

    # =========================================================================
    # Step 1: Query both datasets
    # =========================================================================
    print("\n[Step 1] Querying datasets...")
    X_q1, y_q1, ri_q1 = query_dataset(username, 'q1', test_mode=args.test_mode)
    X_ero, y_ero, ri_ero = query_dataset(username, 'ero', test_mode=args.test_mode)

    # =========================================================================
    # Step 2: Split train/val/test (70/15/15)
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

    # Combined splits (concatenate Q1 + ERO splits)
    X_comb_train = np.vstack([X_q1_train, X_ero_train])
    y_comb_train = np.concatenate([y_q1_train, y_ero_train])
    ri_comb_train = np.concatenate([ri_q1_train, ri_ero_train])
    X_comb_val = np.vstack([X_q1_val, X_ero_val])
    y_comb_val = np.concatenate([y_q1_val, y_ero_val])
    X_comb_test = np.vstack([X_q1_test, X_ero_test])
    y_comb_test = np.concatenate([y_q1_test, y_ero_test])

    n_comb_star = (y_comb_train == 0).sum()
    n_comb_galaxy = (y_comb_train == 1).sum()
    comb_train_ratio = n_comb_star / n_comb_galaxy

    print(f"  Combined — train: {len(X_comb_train):,}, val: {len(X_comb_val):,}, "
          f"test: {len(X_comb_test):,}  (ratio {comb_train_ratio:.1f}:1)")

    # =========================================================================
    # Step 3: Train all 9 model variants
    # =========================================================================
    print("\n[Step 3] Training 9 model variants...")

    # --- 3 baselines (native ratio) ---
    print("\n--- Q1 baseline (ratio {:.1f}:1) ---".format(q1_train_ratio))
    q1_model = train_model(X_q1_train, y_q1_train, X_q1_val, y_q1_val, verbose=True)
    print(f"  Best iteration: {q1_model.best_iteration}, Best score: {q1_model.best_score:.4f}")

    print("\n--- ERO baseline (ratio {:.1f}:1) ---".format(ero_train_ratio))
    ero_model = train_model(X_ero_train, y_ero_train, X_ero_val, y_ero_val, verbose=True)
    print(f"  Best iteration: {ero_model.best_iteration}, Best score: {ero_model.best_score:.4f}")

    print("\n--- Combined baseline (ratio {:.1f}:1) ---".format(comb_train_ratio))
    comb_model = train_model(X_comb_train, y_comb_train, X_comb_val, y_comb_val, verbose=True)
    print(f"  Best iteration: {comb_model.best_iteration}, Best score: {comb_model.best_score:.4f}")

    # --- Q1 @ ERO ratio (downsample galaxies) ---
    print("\n--- Q1 @ ERO ratio ({:.1f}:1) ---".format(ero_train_ratio))
    X_q1_ds_ero, y_q1_ds_ero = downsample_training(
        X_q1_train, y_q1_train, ri_q1_train,
        target_ratio=ero_train_ratio, downsample_class='galaxy'
    )
    q1_at_ero_model = train_model(X_q1_ds_ero, y_q1_ds_ero, X_q1_val, y_q1_val, verbose=False)

    # --- Q1 @ Combo ratio (downsample galaxies) ---
    print("\n--- Q1 @ Combo ratio ({:.1f}:1) ---".format(comb_train_ratio))
    X_q1_ds_comb, y_q1_ds_comb = downsample_training(
        X_q1_train, y_q1_train, ri_q1_train,
        target_ratio=comb_train_ratio, downsample_class='galaxy'
    )
    q1_at_comb_model = train_model(X_q1_ds_comb, y_q1_ds_comb, X_q1_val, y_q1_val, verbose=False)

    # --- ERO @ Q1 ratio (downsample stars) ---
    print("\n--- ERO @ Q1 ratio ({:.1f}:1) ---".format(q1_train_ratio))
    X_ero_ds_q1, y_ero_ds_q1 = downsample_training(
        X_ero_train, y_ero_train, ri_ero_train,
        target_ratio=q1_train_ratio, downsample_class='star'
    )
    ero_at_q1_model = train_model(X_ero_ds_q1, y_ero_ds_q1, X_ero_val, y_ero_val, verbose=False)

    # --- ERO @ Combo ratio (downsample stars) ---
    print("\n--- ERO @ Combo ratio ({:.1f}:1) ---".format(comb_train_ratio))
    X_ero_ds_comb, y_ero_ds_comb = downsample_training(
        X_ero_train, y_ero_train, ri_ero_train,
        target_ratio=comb_train_ratio, downsample_class='star'
    )
    ero_at_comb_model = train_model(X_ero_ds_comb, y_ero_ds_comb, X_ero_val, y_ero_val, verbose=False)

    # --- Combined @ Q1 ratio (downsample stars) ---
    print("\n--- Comb @ Q1 ratio ({:.1f}:1) ---".format(q1_train_ratio))
    X_comb_ds_q1, y_comb_ds_q1 = downsample_training(
        X_comb_train, y_comb_train, ri_comb_train,
        target_ratio=q1_train_ratio, downsample_class='star'
    )
    comb_at_q1_model = train_model(X_comb_ds_q1, y_comb_ds_q1, X_comb_val, y_comb_val, verbose=False)

    # --- Combined @ ERO ratio (downsample galaxies) ---
    print("\n--- Comb @ ERO ratio ({:.1f}:1) ---".format(ero_train_ratio))
    X_comb_ds_ero, y_comb_ds_ero = downsample_training(
        X_comb_train, y_comb_train, ri_comb_train,
        target_ratio=ero_train_ratio, downsample_class='galaxy'
    )
    comb_at_ero_model = train_model(X_comb_ds_ero, y_comb_ds_ero, X_comb_val, y_comb_val, verbose=False)

    # =========================================================================
    # Step 4: Systematic 3×3×3 evaluation (27 experiments)
    # =========================================================================
    print("\n[Step 4] Evaluating 27 experiments...")

    # Define the 9 models: (model_key, model_object, display_prefix)
    # Order: 3 datasets × 3 ratios = @Q1, @Combo, @ERO for each dataset
    models = [
        ('Q1@Q1',      q1_model,          'Q1@Q1'),
        ('Q1@Combo',   q1_at_comb_model,  'Q1@Comb'),
        ('Q1@ERO',     q1_at_ero_model,   'Q1@ERO'),
        ('ERO@Q1',     ero_at_q1_model,   'ERO@Q1'),
        ('ERO@Combo',  ero_at_comb_model,  'ERO@Comb'),
        ('ERO@ERO',    ero_model,          'ERO@ERO'),
        ('Comb@Q1',    comb_at_q1_model,   'Comb@Q1'),
        ('Comb@Combo', comb_model,         'Comb@Comb'),
        ('Comb@ERO',   comb_at_ero_model,  'Comb@ERO'),
    ]

    # 3 test sets
    test_sets = [
        ('Q1',   X_q1_test,   y_q1_test),
        ('Combo', X_comb_test, y_comb_test),
        ('ERO',  X_ero_test,  y_ero_test),
    ]

    all_results = {}
    exp_count = 0
    for model_key, model, display_prefix in models:
        for test_name, X_test, y_test in test_sets:
            exp_count += 1
            label = f'{display_prefix}→{test_name}'
            print(f"  [{exp_count}/27] {label}")
            r = evaluate(model, X_test, y_test)
            r['label'] = label
            r['description'] = f'{display_prefix} → {test_name} test'
            all_results[label] = r

    # =========================================================================
    # Step 5: Organize into 3-row × 9-column layout
    # =========================================================================
    # Rows: Q1-trained, ERO-trained, Comb-trained
    # Columns per row (9 total):
    #   Test Q1 group:    col 0=@Q1→Q1,  col 1=@Combo→Q1,  col 2=@ERO→Q1
    #   Test Combo group: col 3=@Q1→Combo, col 4=@Combo→Combo, col 5=@ERO→Combo
    #   Test ERO group:   col 6=@Q1→ERO, col 7=@Combo→ERO, col 8=@ERO→ERO

    dataset_prefixes = ['Q1', 'Comb', 'ERO']
    ratio_suffixes = ['@Q1', '@Comb', '@ERO']
    test_names = ['Q1', 'Combo', 'ERO']

    row_layout = []
    for ds in dataset_prefixes:
        row_title = f'{ds}-trained models'
        entries = []
        for test_idx, test_name in enumerate(test_names):
            for ratio_idx, ratio_suf in enumerate(ratio_suffixes):
                col = test_idx * 3 + ratio_idx
                label = f'{ds}{ratio_suf}→{test_name}'
                entries.append((col, all_results[label]))
        row_layout.append((row_title, entries))

    # Diagonal highlights: native ratio tested on matching test set
    # Q1@Q1→Q1 = row 0, col 0; Comb@Combo→Combo = row 2, col 4; ERO@ERO→ERO = row 1, col 8
    highlight_positions = {
        (0, 0),  # Q1@Q1→Q1
        (1, 4),  # Comb@Combo→Combo (Comb@Comb is the native-ratio combined)
        (2, 8),  # ERO@ERO→ERO
    }

    # =========================================================================
    # Step 6: Print summary
    # =========================================================================
    print("\n" + "=" * 90)
    print("UNIFIED SUMMARY (27 experiments)")
    print("=" * 90)
    header = (f"{'Experiment':<22} {'Accuracy':>9} {'Gal Prec':>9} {'Gal Rec':>8} "
              f"{'Gal F1':>7} {'Star Prec':>10} {'Star Rec':>9} {'ROC AUC':>8}")

    all_results_list = []
    for row_title, entries in row_layout:
        print(f"\n  {row_title}")
        print("  " + header)
        print("  " + "-" * 90)
        for _, r in entries:
            all_results_list.append(r)
            print(f"  {r['label']:<22} {r['accuracy']:>9.3f} {r['galaxy_precision']:>9.3f} "
                  f"{r['galaxy_recall']:>8.3f} {r['galaxy_f1']:>7.3f} "
                  f"{r['star_precision']:>10.3f} {r['star_recall']:>9.3f} {r['roc_auc']:>8.3f}")

    # Save parameter list
    param_list_path = OUTPUT_DIR / 'parameters_used.txt'
    with open(param_list_path, 'w') as f:
        f.write(f"PARAMETERS USED IN THIS RUN ({len(SELECTED_GAIA_PARAMETERS)} features)\n")
        f.write("=" * 60 + "\n\n")
        for i, param in enumerate(SELECTED_GAIA_PARAMETERS, 1):
            f.write(f"  {i:>2}. {param}\n")
    print(f"\n  -> Parameter list saved: {param_list_path}")

    # Save summary text file
    summary_path = OUTPUT_DIR / 'combined_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("XGBOOST COMBINED MODEL TRAINING & EVALUATION SUMMARY (27 experiments)\n")
        f.write("=" * 90 + "\n")
        f.write(f"Magnitude cut: Gaia G >= {MAG_CUT} (both datasets)\n")
        f.write(f"Features: {len(SELECTED_GAIA_PARAMETERS)}\n")
        f.write(f"Q1 train ratio: {q1_train_ratio:.1f}:1 (STAR:GALAXY)\n")
        f.write(f"ERO train ratio: {ero_train_ratio:.1f}:1 (STAR:GALAXY)\n")
        f.write(f"Combined train ratio: {comb_train_ratio:.1f}:1 (STAR:GALAXY)\n\n")
        f.write("Parameters:\n")
        for i, param in enumerate(SELECTED_GAIA_PARAMETERS, 1):
            f.write(f"  {i:>2}. {param}\n")
        f.write("\n")

        for row_title, entries in row_layout:
            f.write(f"\n{row_title}\n")
            f.write(header + "\n")
            f.write("-" * 90 + "\n")
            for _, r in entries:
                f.write(f"{r['label']:<22} {r['accuracy']:>9.3f} {r['galaxy_precision']:>9.3f} "
                        f"{r['galaxy_recall']:>8.3f} {r['galaxy_f1']:>7.3f} "
                        f"{r['star_precision']:>10.3f} {r['star_recall']:>9.3f} {r['roc_auc']:>8.3f}\n")

        f.write("\n")
        for r in all_results_list:
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"{r['description']}\n")
            f.write("=" * 70 + "\n")
            f.write(r['report'])
            f.write(f"ROC AUC: {r['roc_auc']:.4f}\n")

    print(f"  -> Summary saved: {summary_path}")

    # =========================================================================
    # Step 7: Generate plots
    # =========================================================================
    print("\n[Step 7] Generating plots...")

    # 3-row × 9-column galaxy metrics bar chart
    plot_galaxy_metrics_by_test(row_layout, highlight_positions, OUTPUT_DIR,
                        q1_train_ratio, comb_train_ratio, ero_train_ratio)
    print("  -> combined_galaxy_metrics_by_test.png")

    # Same 27 experiments, grouped by training ratio instead of test set
    plot_galaxy_metrics_by_ratio(all_results, OUTPUT_DIR,
                                 q1_train_ratio, comb_train_ratio, ero_train_ratio)
    print("  -> combined_galaxy_metrics_by_ratio.png")

    # 3-panel feature importance (Q1 / ERO / Combined baselines)
    plot_feature_importance_comparison(q1_model, ero_model, comb_model,
                                       SELECTED_GAIA_PARAMETERS, OUTPUT_DIR)
    print("  -> combined_feature_importance.png + .csv")

    # =========================================================================
    # Step 8: Save models
    # =========================================================================
    print("\n[Step 8] Saving models...")
    for name, model in [('q1', q1_model), ('ero', ero_model), ('combined', comb_model)]:
        model_path = OUTPUT_DIR / f'xgb_classifier_{name}.json'
        try:
            model.get_booster().save_model(str(model_path))
            print(f"  -> {model_path}")
        except Exception as e:
            print(f"  WARNING: Could not save {name} model: {e}")

    print("\n" + "=" * 80)
    print("Combined model training complete! (27 experiments)")
    print(f"All results saved to: {OUTPUT_DIR}")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
