#!/usr/bin/env python3
"""
Sweep ERO mixing ratios and plot metrics vs training STAR:GALAXY ratio.

Trains a 10-seed XGBoost ensemble at each target ratio, evaluating on
pure Q1 test sets.  Produces a summary plot and CSV.

The Q1-only baseline (ratio ~2.9:1) is included automatically.

Reads:
  WSDB: crossmatch_q1_mer_phz_vis_gaia_sep, crossmatch_ero_gaia_0_3

Writes:
  output/xgboost_mix_q1_ero/ratio_sweep/ero_ratio_sweep.png
  output/xgboost_mix_q1_ero/ratio_sweep/ero_ratio_sweep.csv

Usage:
  PGUSER=siwei_zhao_2025 python src/train_sweep_mix_ratio.py
  PGUSER=siwei_zhao_2025 python src/train_sweep_mix_ratio.py --ratios 3 3.5 4 4.5 5
  PGUSER=siwei_zhao_2025 python src/train_sweep_mix_ratio.py --test-mode
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'output' / 'xgboost_mix_q1_ero' / 'ratio_sweep'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

CLASS_MAPPING = {'STAR': 0, 'GALAXY': 1}
MAG_CUT       = 17
MAG_CUT_FAINT = 99
SEEDS = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]
CLASS_STAR_THRESH = 0.9

XGB_PARAMS = {
    'objective':             'binary:logistic',
    'max_depth':             8,
    'learning_rate':         0.1,
    'n_estimators':          200,
    'subsample':             0.8,
    'colsample_bytree':      0.8,
    'random_state':          42,
    'tree_method':           'hist',
    'eval_metric':           'logloss',
    'early_stopping_rounds': 20,
}

DEFAULT_RATIOS = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]


# ── Data loading (same as train_xgboost_q1_ero.py) ──────────────────────────

def query_q1(username, test_mode=False):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{c} AS {c}' for c in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT g_phot_g_mean_mag AS phot_g_mean_mag, {gaia_cols},
               CASE WHEN p_phz_classification = 1 THEN 'STAR'
                    WHEN p_phz_classification = 2 THEN 'GALAXY' END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep
        WHERE p_phz_classification IN (1, 2)
          AND g_phot_g_mean_mag >= {MAG_CUT}
          AND g_phot_g_mean_mag <= {MAG_CUT_FAINT}
        ORDER BY g_source_id
    """
    if test_mode:
        query += " LIMIT 10000"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for c in SELECTED_GAIA_PARAMETERS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    X = df[SELECTED_GAIA_PARAMETERS].values
    y = df['label'].map(CLASS_MAPPING).values
    return X, y


def query_ero_stars(username, test_mode=False):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{c} AS {c}' for c in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT {gaia_cols}
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_class_star > {CLASS_STAR_THRESH}
          AND g_phot_g_mean_mag >= {MAG_CUT}
          AND g_phot_g_mean_mag <= {MAG_CUT_FAINT}
        ORDER BY g_source_id
    """
    if test_mode:
        query += " LIMIT 5000"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for c in SELECTED_GAIA_PARAMETERS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    X_ero = df[SELECTED_GAIA_PARAMETERS].values
    y_ero = np.zeros(len(X_ero), dtype=int)
    return X_ero, y_ero


# ── Training helpers ─────────────────────────────────────────────────────────

def train_and_eval_one_ratio(X_q1, y_q1, X_ero, y_ero, target_ratio, seeds,
                             return_model=False):
    """Train at a given ratio, return per-seed metrics dict list.
    If return_model=True, also return (model, X_test, y_test) from seed 42."""
    results = []
    seed42_model = None
    for seed in seeds:
        # Split Q1 70/15/15
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_q1, y_q1, test_size=0.15, random_state=seed, stratify=y_q1)
        X_train_q1, X_val, y_train_q1, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=seed, stratify=y_temp)

        rng = np.random.RandomState(seed)

        if target_ratio is None:
            # Q1 only — no ERO
            X_train = X_train_q1
            y_train = y_train_q1
            n_ero_used = 0
        else:
            n_q1_stars    = (y_train_q1 == 0).sum()
            n_q1_galaxies = (y_train_q1 == 1).sum()
            n_stars_needed = int(target_ratio * n_q1_galaxies)
            n_ero_to_add   = max(0, n_stars_needed - n_q1_stars)
            if n_ero_to_add < len(X_ero):
                ero_idx = rng.choice(len(X_ero), size=n_ero_to_add, replace=False)
                X_ero_use = X_ero[ero_idx]
                y_ero_use = y_ero[ero_idx]
            else:
                X_ero_use = X_ero
                y_ero_use = y_ero
            n_ero_used = len(X_ero_use)
            X_train = np.vstack([X_train_q1, X_ero_use])
            y_train = np.concatenate([y_train_q1, y_ero_use])
            shuffle_idx = rng.permutation(len(X_train))
            X_train = X_train[shuffle_idx]
            y_train = y_train[shuffle_idx]

        actual_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        # Train
        model = xgb.XGBClassifier(**{**XGB_PARAMS, 'random_state': seed})
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Evaluate on Q1 test set
        y_pred       = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=[0, 1], zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        results.append({
            'target_ratio':    target_ratio if target_ratio is not None else actual_ratio,
            'actual_ratio':    actual_ratio,
            'n_ero':           n_ero_used,
            'seed':            seed,
            'accuracy':        (y_test == y_pred).sum() / len(y_test),
            'galaxy_precision': prec[1],
            'galaxy_recall':   rec[1],
            'galaxy_f1':       f1[1],
            'star_f1':         f1[0],
            'roc_auc':         roc_auc,
        })
        if return_model and seed == 42:
            seed42_model = (model, X_test, y_test)
    if return_model:
        return results, seed42_model
    return results


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_sweep(df_summary, output_dir):
    """Plot metrics vs training ratio."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.3)

    x = df_summary['actual_ratio'].values

    # [0] ROC AUC
    ax = axes[0]
    ax.errorbar(x, df_summary['roc_auc_mean'], yerr=df_summary['roc_auc_std'],
                fmt='o-', color='steelblue', lw=2, markersize=8, capsize=4,
                label='ROC AUC')
    # Mark Q1-only point
    ax.scatter(x[0], df_summary['roc_auc_mean'].iloc[0],
               s=120, color='steelblue', edgecolors='k', linewidths=2, zorder=5)
    ax.set_xlabel('Training STAR:GALAXY ratio', fontsize=11)
    ax.set_ylabel('ROC AUC', fontsize=11)
    ax.set_title('Discriminative power (ROC AUC)', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    # Annotate values
    for i, row in df_summary.iterrows():
        label = 'Q1 only' if i == 0 else f'+{int(row["n_ero_mean"]):,} ERO'
        ax.annotate(f'{row["roc_auc_mean"]:.4f}\n{label}',
                    xy=(row['actual_ratio'], row['roc_auc_mean']),
                    xytext=(0, -20), textcoords='offset points',
                    fontsize=7, ha='center', color='steelblue')

    # [1] Galaxy Precision & Recall
    ax = axes[1]
    ax.errorbar(x, df_summary['galaxy_precision_mean'],
                yerr=df_summary['galaxy_precision_std'],
                fmt='s-', color='steelblue', lw=2, markersize=7, capsize=4,
                label='Galaxy Precision')
    ax.errorbar(x, df_summary['galaxy_recall_mean'],
                yerr=df_summary['galaxy_recall_std'],
                fmt='^-', color='coral', lw=2, markersize=7, capsize=4,
                label='Galaxy Recall')
    ax.errorbar(x, df_summary['galaxy_f1_mean'],
                yerr=df_summary['galaxy_f1_std'],
                fmt='D-', color='mediumseagreen', lw=2, markersize=6, capsize=4,
                label='Galaxy F1')
    # Mark Q1-only
    ax.scatter(x[0], df_summary['galaxy_precision_mean'].iloc[0],
               s=100, color='steelblue', edgecolors='k', linewidths=2, zorder=5)
    ax.scatter(x[0], df_summary['galaxy_recall_mean'].iloc[0],
               s=100, color='coral', edgecolors='k', linewidths=2, zorder=5)
    ax.set_xlabel('Training STAR:GALAXY ratio', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Galaxy Precision / Recall / F1', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.3, 1.0)

    # [2] Summary table as text
    ax = axes[2]
    ax.axis('off')
    header = f'{"Ratio":>7} {"ERO added":>10} {"AUC":>8} {"Prec":>8} {"Recall":>8} {"F1":>8}'
    lines = [header, '-' * 55]
    for _, row in df_summary.iterrows():
        tag = 'Q1 only' if row['n_ero_mean'] == 0 else f'+{int(row["n_ero_mean"]):,}'
        lines.append(
            f'{row["actual_ratio"]:>7.1f} {tag:>10} '
            f'{row["roc_auc_mean"]:>7.4f} '
            f'{row["galaxy_precision_mean"]:>7.4f} '
            f'{row["galaxy_recall_mean"]:>7.4f} '
            f'{row["galaxy_f1_mean"]:>7.4f}'
        )
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontsize=9, family='monospace', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#bbbbbb', alpha=0.95))
    ax.set_title('Summary (mean across 10 seeds)', fontsize=11, fontweight='bold')

    fig.suptitle(
        'XGBoost performance vs ERO star mixing ratio\n'
        'Q1 test set only (15% held out, pure Q1)  |  '
        f'{len(SELECTED_GAIA_PARAMETERS)} features, {len(SEEDS)} seeds',
        fontsize=13, fontweight='bold', y=1.02,
    )
    plt.tight_layout()
    out = output_dir / 'ero_ratio_sweep.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Sweep ERO mixing ratios')
    parser.add_argument('--test-mode', action='store_true',
                        help='Limit data for quick testing')
    parser.add_argument('--ratios', type=float, nargs='+', default=DEFAULT_RATIOS,
                        metavar='R', help='Target ratios to sweep (default: 3 3.5 4 ... 8)')
    parser.add_argument('--class-star', type=float, default=0.9,
                        help='ERO class_star threshold (default: 0.9)')
    parser.add_argument('--threshold-analysis', action='store_true',
                        help='Run P(gal) threshold analysis per ratio using gaia_random.npz')
    args = parser.parse_args()

    global CLASS_STAR_THRESH
    CLASS_STAR_THRESH = args.class_star

    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER not set!")
        return 1

    # Use separate output dir per class_star threshold
    if args.class_star != 0.9:
        output_dir = PROJECT_DIR / 'output' / 'xgboost_mix_q1_ero' / f'ratio_sweep_cs{args.class_star}'
    else:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("[1] Querying Q1 data...")
    X_q1, y_q1 = query_q1(username, test_mode=args.test_mode)
    n_q1_star   = (y_q1 == 0).sum()
    n_q1_galaxy = (y_q1 == 1).sum()
    q1_ratio = n_q1_star / n_q1_galaxy
    print(f"  Q1: {len(X_q1):,} sources  "
          f"STAR={n_q1_star:,}  GALAXY={n_q1_galaxy:,}  ratio={q1_ratio:.1f}:1")

    print("\n[2] Querying ERO stars...")
    X_ero, y_ero = query_ero_stars(username, test_mode=args.test_mode)
    print(f"  ERO: {len(X_ero):,} stars")

    # ── Sweep ────────────────────────────────────────────────────────────────
    all_results = []

    # Q1 only (baseline)
    print(f"\n[3] Training Q1-only baseline ({q1_ratio:.1f}:1)...")
    results = train_and_eval_one_ratio(X_q1, y_q1, X_ero, y_ero,
                                       target_ratio=None, seeds=SEEDS)
    all_results.extend(results)
    auc_mean = np.mean([r['roc_auc'] for r in results])
    print(f"  AUC={auc_mean:.4f}")

    # Each target ratio
    for ratio in sorted(args.ratios):
        if ratio <= q1_ratio:
            print(f"\n  Skipping ratio {ratio:.0f}:1 (below Q1 baseline {q1_ratio:.1f}:1)")
            continue
        print(f"\n[3] Training ratio {ratio:.0f}:1...")
        results = train_and_eval_one_ratio(X_q1, y_q1, X_ero, y_ero,
                                           target_ratio=ratio, seeds=SEEDS)
        all_results.extend(results)
        auc_mean = np.mean([r['roc_auc'] for r in results])
        prec_mean = np.mean([r['galaxy_precision'] for r in results])
        rec_mean = np.mean([r['galaxy_recall'] for r in results])
        print(f"  AUC={auc_mean:.4f}  Prec={prec_mean:.4f}  Recall={rec_mean:.4f}")

    # ── Aggregate ────────────────────────────────────────────────────────────
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(output_dir / 'ero_ratio_sweep_all.csv', index=False)

    # Group by target_ratio and compute mean/std
    grouped = df_all.groupby('target_ratio')
    summary_rows = []
    for ratio, grp in grouped:
        row = {'target_ratio': ratio}
        for col in ['actual_ratio', 'n_ero', 'accuracy', 'galaxy_precision',
                     'galaxy_recall', 'galaxy_f1', 'star_f1', 'roc_auc']:
            row[f'{col}_mean'] = grp[col].mean()
            row[f'{col}_std']  = grp[col].std()
        summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows).sort_values('actual_ratio_mean')
    df_summary = df_summary.reset_index(drop=True)
    # Use actual_ratio mean as x-axis
    df_summary['actual_ratio'] = df_summary['actual_ratio_mean']

    df_summary.to_csv(output_dir / 'ero_ratio_sweep.csv', index=False)
    print(f"\n  -> {output_dir / 'ero_ratio_sweep.csv'}")

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"\n{'Ratio':>7} {'ERO added':>10} {'AUC':>10} {'Precision':>10} "
          f"{'Recall':>10} {'F1':>10}")
    print("-" * 62)
    for _, row in df_summary.iterrows():
        tag = 'Q1 only' if row['n_ero_mean'] == 0 else f'+{int(row["n_ero_mean"]):,}'
        print(f'{row["actual_ratio"]:.1f}:1 {tag:>10} '
              f'{row["roc_auc_mean"]:>9.4f} '
              f'{row["galaxy_precision_mean"]:>9.4f} '
              f'{row["galaxy_recall_mean"]:>9.4f} '
              f'{row["galaxy_f1_mean"]:>9.4f}')

    # ── Plot ─────────────────────────────────────────────────────────────────
    print("\n[4] Plotting sweep...")
    plot_sweep(df_summary, output_dir)

    # ── Threshold analysis per ratio ────────────────────────────────────────
    if args.threshold_analysis:
        print("\n[5] Running threshold analysis per ratio...")

        # Load Gaia random sources
        gaia_npz = PROJECT_DIR / 'output' / 'umap' / 'arrays' / 'gaia_random.npz'
        if not gaia_npz.exists():
            print(f"  WARNING: {gaia_npz} not found, skipping threshold analysis")
        else:
            g = np.load(gaia_npz, allow_pickle=True)
            X_gaia = g['X_features'].astype(float)
            feat_names = list(SELECTED_GAIA_PARAMETERS)
            print(f"  Loaded {len(X_gaia):,} Gaia sources")

            # Retrain seed-42 at selected ratios and do threshold analysis
            analysis_ratios = [None] + [r for r in sorted(args.ratios) if r > q1_ratio]
            # Pick a subset: Q1-only, low, mid, high
            if len(analysis_ratios) > 5:
                indices = [0, 1, len(analysis_ratios)//2, -2, -1]
                analysis_ratios = [analysis_ratios[i] for i in indices]

            thresholds = [0.5, 0.7, 0.9]
            thresh_results = []
            all_pgal = []  # (label, p_gaia_array) for overlay plot

            for ratio in analysis_ratios:
                label = f"{ratio:.0f}:1" if ratio else "Q1 only"
                print(f"\n  Ratio {label}...")

                _, (model, X_test, y_test) = train_and_eval_one_ratio(
                    X_q1, y_q1, X_ero, y_ero,
                    target_ratio=ratio, seeds=[42], return_model=True)

                # P(gal) on Gaia
                dmat_gaia = xgb.DMatrix(X_gaia, feature_names=feat_names)
                p_gaia = model.get_booster().predict(dmat_gaia)

                # TPR/FPR on Q1 test
                dmat_test = xgb.DMatrix(X_test, feature_names=feat_names)
                p_test = model.get_booster().predict(dmat_test)

                pi_q1 = (y_test == 1).sum() / len(y_test)

                for t in thresholds:
                    pred_gal = p_test >= t
                    tp = (pred_gal & (y_test == 1)).sum()
                    fp = (pred_gal & (y_test == 0)).sum()
                    fn = (~pred_gal & (y_test == 1)).sum()
                    tn = (~pred_gal & (y_test == 0)).sum()
                    tpr = tp / (tp + fn)
                    fpr = fp / (fp + tn)
                    n_gaia_above = int((p_gaia >= t).sum())
                    pct_gaia = 100 * n_gaia_above / len(p_gaia)

                    # Bayesian recalibration at 1% and 3% galaxy fraction
                    purity_1 = tpr * 0.01 / (tpr * 0.01 + fpr * 0.99) if (tpr * 0.01 + fpr * 0.99) > 0 else 0
                    purity_3 = tpr * 0.03 / (tpr * 0.03 + fpr * 0.97) if (tpr * 0.03 + fpr * 0.97) > 0 else 0

                    thresh_results.append({
                        'ratio': ratio if ratio else q1_ratio,
                        'ratio_label': label,
                        'threshold': t,
                        'tpr': tpr, 'fpr': fpr,
                        'n_gaia': n_gaia_above,
                        'pct_gaia': pct_gaia,
                        'purity_1pct': purity_1,
                        'purity_3pct': purity_3,
                    })

                all_pgal.append((label, p_gaia))
                print(f"    P(gal)>0.5: {(p_gaia>=0.5).sum():,} ({100*(p_gaia>=0.5).mean():.1f}%)  "
                      f"P(gal)>0.9: {(p_gaia>=0.9).sum():,} ({100*(p_gaia>=0.9).mean():.1f}%)")

            # Save threshold results
            df_thresh = pd.DataFrame(thresh_results)
            df_thresh.to_csv(output_dir / 'threshold_analysis.csv', index=False)
            print(f"\n  -> {output_dir / 'threshold_analysis.csv'}")

            # ── Plot: P(gal) distribution overlay for all ratios ────────
            import matplotlib.cm as cm

            colors = cm.viridis(np.linspace(0, 0.9, len(all_pgal)))

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.subplots_adjust(wspace=0.25)

            bins = np.linspace(0, 1, 101)
            t_range = np.linspace(0, 1, 501)

            for (label, p_gaia), color in zip(all_pgal, colors):
                # [0] Histogram
                axes[0].hist(p_gaia, bins=bins, alpha=0.4, color=color,
                             label=label, histtype='stepfilled', linewidth=0)
                axes[0].hist(p_gaia, bins=bins, color=color,
                             histtype='step', linewidth=1.2)

                # [1] Galaxy count vs threshold
                counts = np.array([(p_gaia >= t).sum() for t in t_range])
                axes[1].plot(t_range, counts, color=color, lw=2, label=label)

            # Threshold markers
            for t, col, ls in [(0.5, 'red', '--'), (0.7, 'orange', '--'),
                                (0.9, 'green', '--')]:
                axes[0].axvline(t, color=col, lw=1.5, ls=ls, alpha=0.7)
                axes[1].axvline(t, color=col, lw=1.5, ls=ls, alpha=0.7)

            axes[0].set_xlabel("P(gal)", fontsize=11, fontweight='bold')
            axes[0].set_ylabel("Count", fontsize=11, fontweight='bold')
            axes[0].set_title("P(gal) distribution on 50k random Gaia",
                              fontsize=11, fontweight='bold')
            axes[0].legend(fontsize=8)
            axes[0].grid(alpha=0.3)

            axes[1].set_xlabel("P(gal) threshold", fontsize=11, fontweight='bold')
            axes[1].set_ylabel("N predicted galaxies", fontsize=11, fontweight='bold')
            axes[1].set_title("Predicted galaxy count vs threshold",
                              fontsize=11, fontweight='bold')
            axes[1].set_yscale("log")
            axes[1].axhline(len(X_gaia) * 0.01, color='grey', lw=1, ls=':',
                            label='1% of sample')
            axes[1].legend(fontsize=8)
            axes[1].grid(alpha=0.3)

            fig.suptitle("P(gal) on Gaia sources — effect of ERO mixing ratio",
                         fontsize=13, fontweight='bold', y=0.98)
            out_plot = output_dir / 'threshold_pgal_overlay.png'
            plt.savefig(out_plot, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  -> {out_plot}")

            # Print summary
            print(f"\n  {'Ratio':<10} {'Thresh':>6} {'TPR':>7} {'FPR':>7} "
                  f"{'N_Gaia':>8} {'%Gaia':>6} {'Pur@1%':>7} {'Pur@3%':>7}")
            print("  " + "-" * 70)
            for _, r in df_thresh.iterrows():
                print(f"  {r['ratio_label']:<10} {r['threshold']:>6.1f} "
                      f"{r['tpr']:>7.4f} {r['fpr']:>7.4f} "
                      f"{int(r['n_gaia']):>8,} {r['pct_gaia']:>5.1f}% "
                      f"{r['purity_1pct']:>6.1%} {r['purity_3pct']:>6.1%}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
