#!/usr/bin/env python3
"""
Analyze how XGBoost classifier performance varies across ERO fields,
binned by absolute galactic latitude |b|.

Loads saved Q1 and ERO baseline models from the master script output,
queries ERO data with galactic coordinates, and evaluates per |b| bin.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output' / 'xgboost_master'
MODEL_DIR = OUTPUT_DIR

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

CLASS_MAPPING = {'STAR': 0, 'GALAXY': 1}
CLASS_NAMES = ['STAR', 'GALAXY']

# |b| bin edges
B_BINS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 70]


def query_ero_with_position(username):
    """Query ERO dataset with galactic latitude for field analysis."""
    import psycopg2

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)

    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])

    query = f"""
        SELECT
            {gaia_cols},
            g_b as galactic_b,
            g_l as galactic_l,
            g_phot_g_mean_mag as gmag,
            g_astrometric_matched_transits as amt,
            CASE
                WHEN e_v_class_star >= 0.65 THEN 'STAR'
                WHEN e_v_class_star < 0.65 THEN 'GALAXY'
            END as label
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_class_star IS NOT NULL
        ORDER BY g_source_id
    """

    print("Querying ERO dataset with positions...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Clean features
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['abs_b'] = df['galactic_b'].abs()
    df['y'] = df['label'].map(CLASS_MAPPING)

    n_star = (df['y'] == 0).sum()
    n_galaxy = (df['y'] == 1).sum()
    print(f"  {len(df):,} sources — STAR: {n_star:,}, GALAXY: {n_galaxy:,}")

    return df


def evaluate_per_bin(model, df, model_name):
    """Evaluate Booster model on each |b| bin. Returns list of result dicts."""
    results = []

    for i in range(len(B_BINS) - 1):
        b_lo, b_hi = B_BINS[i], B_BINS[i + 1]
        mask = (df['abs_b'] >= b_lo) & (df['abs_b'] < b_hi)
        df_bin = df[mask]

        if len(df_bin) < 50:
            continue

        X = df_bin[SELECTED_GAIA_PARAMETERS].values
        y = df_bin['y'].values

        n_star = (y == 0).sum()
        n_galaxy = (y == 1).sum()

        if n_galaxy < 5 or n_star < 5:
            continue

        dmat = xgb.DMatrix(X, feature_names=SELECTED_GAIA_PARAMETERS)
        y_proba = model.predict(dmat)  # Booster returns probabilities directly
        y_pred = (y_proba >= 0.5).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, labels=[0, 1], zero_division=0
        )

        try:
            roc = roc_auc_score(y, y_proba)
        except ValueError:
            roc = np.nan

        results.append({
            'b_lo': b_lo,
            'b_hi': b_hi,
            'b_label': f'{b_lo}-{b_hi}',
            'n_total': len(df_bin),
            'n_star': n_star,
            'n_galaxy': n_galaxy,
            'galaxy_frac': n_galaxy / len(df_bin),
            'star_galaxy_ratio': n_star / n_galaxy,
            'med_gmag': df_bin['gmag'].median(),
            'med_amt': df_bin['amt'].median(),
            'galaxy_precision': precision[1],
            'galaxy_recall': recall[1],
            'galaxy_f1': f1[1],
            'star_precision': precision[0],
            'star_recall': recall[0],
            'roc_auc': roc,
            'model': model_name,
        })

    return results


def print_results_table(results, model_name):
    """Print performance table for one model."""
    print(f"\n{'=' * 110}")
    print(f"  {model_name} model → ERO data, by |b| bin")
    print(f"{'=' * 110}")
    print(f"{'|b|':>6} {'N_total':>8} {'N_gal':>7} {'Gal%':>6} {'S:G':>6} "
          f"{'med_G':>6} {'med_amt':>8} "
          f"{'Gal_P':>7} {'Gal_R':>7} {'Gal_F1':>7} {'ROC_AUC':>8}")
    print('-' * 110)

    for r in results:
        print(f"{r['b_label']:>6} {r['n_total']:>8,} {r['n_galaxy']:>7,} "
              f"{r['galaxy_frac']:>5.1%} {r['star_galaxy_ratio']:>5.0f}:1 "
              f"{r['med_gmag']:>6.1f} {r['med_amt']:>8.0f} "
              f"{r['galaxy_precision']:>7.3f} {r['galaxy_recall']:>7.3f} "
              f"{r['galaxy_f1']:>7.3f} {r['roc_auc']:>8.3f}")


def create_plots(q1_results, ero_results, output_dir):
    """Create visualization of performance vs |b|."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Classifier Performance on ERO Data vs Galactic Latitude |b|',
                 fontsize=14, fontweight='bold')

    for results, model_name, color in [
        (q1_results, 'Q1', 'steelblue'),
        (ero_results, 'ERO', 'coral'),
    ]:
        b_centers = [(r['b_lo'] + r['b_hi']) / 2 for r in results]

        # Galaxy Precision
        ax = axes[0, 0]
        ax.plot(b_centers, [r['galaxy_precision'] for r in results],
                'o-', color=color, label=f'{model_name} model', linewidth=2, markersize=6)

        # Galaxy Recall
        ax = axes[0, 1]
        ax.plot(b_centers, [r['galaxy_recall'] for r in results],
                'o-', color=color, label=f'{model_name} model', linewidth=2, markersize=6)

        # ROC AUC
        ax = axes[1, 0]
        ax.plot(b_centers, [r['roc_auc'] for r in results],
                'o-', color=color, label=f'{model_name} model', linewidth=2, markersize=6)

        # Galaxy fraction (context)
        ax = axes[1, 1]
        ax.plot(b_centers, [r['galaxy_frac'] for r in results],
                'o-', color=color, label=f'{model_name} model', linewidth=2, markersize=6)

    # Format axes
    for ax in axes.flat:
        ax.set_xlabel('|b| (degrees)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    axes[0, 0].set_ylabel('Galaxy Precision', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Galaxy Precision vs |b|', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylim(0, 1.05)

    axes[0, 1].set_ylabel('Galaxy Recall', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Galaxy Recall vs |b|', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim(0, 1.05)

    axes[1, 0].set_ylabel('ROC AUC', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('ROC AUC vs |b|', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylim(0.4, 1.05)

    axes[1, 1].set_ylabel('Galaxy Fraction', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('ERO Galaxy Fraction vs |b| (context)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylim(0, 0.25)

    # Add secondary x-axis labels showing N sources per bin (from ERO results)
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlim(0, 72)

    plt.tight_layout()
    plot_path = output_dir / 'ero_field_performance.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved: {plot_path}")
    plt.close()

    # Also create a detailed bar chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Galaxy Precision & Recall on ERO Data by |b| Bin',
                 fontsize=14, fontweight='bold')

    b_labels = [r['b_label'] for r in ero_results]
    x = np.arange(len(b_labels))
    width = 0.35

    for idx, (results, model_name) in enumerate([(q1_results, 'Q1'), (ero_results, 'ERO')]):
        ax = axes[idx]
        prec = [r['galaxy_precision'] for r in results]
        rec = [r['galaxy_recall'] for r in results]

        bars_p = ax.bar(x - width/2, prec, width, label='Galaxy Precision',
                        color='steelblue', edgecolor='white')
        bars_r = ax.bar(x + width/2, rec, width, label='Galaxy Recall',
                        color='coral', edgecolor='white')

        for bar, val in zip(bars_p, prec):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        for bar, val in zip(bars_r, rec):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Add source counts as secondary labels
        ax2_labels = [f'n={r["n_total"]:,}\n({r["n_galaxy"]:,} gal)' for r in results]
        ax.set_xticks(x)
        ax.set_xticklabels([f'|b|={l}\n{c}' for l, c in zip(b_labels, ax2_labels)],
                           fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'{model_name} Model → ERO', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    bar_path = output_dir / 'ero_field_performance_bars.png'
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {bar_path}")
    plt.close()


def main():
    print("=" * 70)
    print("ERO Field Performance Analysis by Galactic Latitude")
    print("=" * 70)

    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    # Load saved models
    q1_model_path = MODEL_DIR / 'xgb_classifier_q1.json'
    ero_model_path = MODEL_DIR / 'xgb_classifier_ero.json'

    if not q1_model_path.exists() or not ero_model_path.exists():
        print(f"ERROR: Models not found in {MODEL_DIR}")
        print("Run train_xgboost_master.py first.")
        return 1

    print("\nLoading models...")
    q1_model = xgb.Booster()
    q1_model.load_model(str(q1_model_path))
    print("  Q1 model loaded")

    ero_model = xgb.Booster()
    ero_model.load_model(str(ero_model_path))
    print("  ERO model loaded")

    # Query ERO data
    df = query_ero_with_position(username)

    # Evaluate per bin
    print("\nEvaluating Q1 model on ERO by |b| bin...")
    q1_results = evaluate_per_bin(q1_model, df, 'Q1')
    print_results_table(q1_results, 'Q1')

    print("\nEvaluating ERO model on ERO by |b| bin...")
    ero_results = evaluate_per_bin(ero_model, df, 'ERO')
    print_results_table(ero_results, 'ERO')

    # Save CSV
    all_results = q1_results + ero_results
    results_df = pd.DataFrame(all_results)
    csv_path = OUTPUT_DIR / 'ero_field_performance.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved: {csv_path}")

    # Create plots
    print("\nGenerating plots...")
    create_plots(q1_results, ero_results, OUTPUT_DIR)

    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
