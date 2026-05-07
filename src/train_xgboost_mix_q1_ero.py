#!/usr/bin/env python3
"""
XGBoost Q1+ERO mixed model training & evaluation.

Augments Q1 training data with ERO sources classified as stars
(SExtractor class_star > 0.9) from crossmatch_ero_gaia_0_3.
The STAR:GALAXY ratio in training is controlled by --target-ratio.

Q1 labels:  PHZ classification (STAR / GALAXY)
ERO labels: class_star > 0.9 → STAR  (all added as y=0)

Split: 70% train / 15% val (early stopping) / 15% test.
ERO stars are added ONLY to the training split — val/test remain
pure Q1 so metrics are directly comparable to the Q1-only model.

Per-seed evaluation: each of the 10 seed models is evaluated on its
own held-out Q1 test set (no data leakage). Metrics and histograms
in q1_threshold_analysis.png are averaged across seeds.

Outputs saved to output/xgboost_mix_q1_ero/ratio_sgN/:
  xgb_classifier_q1.json        — baseline seed-42 model
  seed_models/*.json             — per-seed models
  q1_feature_importance.png      — horizontal bar chart
  q1_feature_importance.csv      — importances table
  q1_seed_metrics.json           — per-seed metrics
  q1_threshold_analysis.png      — P(gal) distribution + metrics vs threshold
  q1_summary.txt                 — ensemble metrics summary

Usage:
    # Sweep all ratios (3:1 to max, + all ERO) — default:
    PGUSER=siwei_zhao_2025 python src/train_xgboost_mix_q1_ero.py

    # Train at a single STAR:GALAXY ratio:
    PGUSER=siwei_zhao_2025 python src/train_xgboost_mix_q1_ero.py --target-ratio 5

    # Use all ERO stars without sweeping (max ratio ~8.6:1):
    PGUSER=siwei_zhao_2025 python src/train_xgboost_mix_q1_ero.py --no-sweep

    # Sweep with ERO stars restricted to G >= 18.5:
    PGUSER=siwei_zhao_2025 python src/train_xgboost_mix_q1_ero.py --ero-mag-cut 18.5

    # Sweep with lower class_star threshold:
    PGUSER=siwei_zhao_2025 python src/train_xgboost_mix_q1_ero.py --class-star 0.5

    # Sweep without class_star filter, G >= 18.5:
    PGUSER=siwei_zhao_2025 python src/train_xgboost_mix_q1_ero.py --class-star 0 --ero-mag-cut 18.5
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'output' / 'xgboost_mix_q1_ero'  # overridden in main()
SEED_MODELS_DIR = OUTPUT_DIR / 'seed_models'                 # overridden in main()

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

CLASS_MAPPING = {'STAR': 0, 'GALAXY': 1}
CLASS_NAMES   = ['STAR', 'GALAXY']

MAG_CUT       = 17   # Gaia G bright-end cut
MAG_CUT_FAINT = 99   # Gaia G faint-end cut (99 = no cut)
SEEDS   = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]

CLASS_STAR_THRESH = 0.9  # ERO SExtractor star threshold

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


# =============================================================================
# Data loading
# =============================================================================

def query_q1(username, test_mode=False):
    """Query Q1 PHZ sources (STAR/GALAXY) with Gaia G >= MAG_CUT."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT
            g_random_index AS random_index,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            {gaia_cols},
            CASE
                WHEN p_phz_classification = 1 THEN 'STAR'
                WHEN p_phz_classification = 2 THEN 'GALAXY'
            END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep
        WHERE p_phz_classification IN (1, 2)
          AND g_phot_g_mean_mag >= {MAG_CUT}
          AND g_phot_g_mean_mag <= {MAG_CUT_FAINT}
        ORDER BY g_random_index
    """
    if test_mode:
        query += " LIMIT 10000"

    print(f"  Querying Q1 dataset ({_mag_range()})...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    X   = df[SELECTED_GAIA_PARAMETERS].values
    y   = df['label'].map(CLASS_MAPPING).values
    mag = df['phot_g_mean_mag'].values.astype(float)

    n_star   = (y == 0).sum()
    n_galaxy = (y == 1).sum()
    print(f"  -> {len(X):,} Q1 sources  STAR: {n_star:,} ({100*n_star/len(X):.1f}%)  "
          f"GALAXY: {n_galaxy:,} ({100*n_galaxy/len(X):.1f}%)")

    return X, y, mag


def query_ero_stars(username, test_mode=False, class_star_thresh=0.9):
    """Query ERO sources with Gaia G >= MAG_CUT.

    class_star_thresh: filter to class_star > threshold. Set to 0 to disable.
    All returned sources are labelled STAR (y=0).
    """
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    if class_star_thresh > 0:
        cs_filter = f"AND e_v_class_star > {class_star_thresh}"
        cs_label = f"class_star > {class_star_thresh}"
    else:
        cs_filter = ""
        cs_label = "no class_star filter"
    query = f"""
        SELECT
            g_phot_g_mean_mag AS phot_g_mean_mag,
            e_v_class_star AS class_star,
            {gaia_cols}
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE g_phot_g_mean_mag >= {MAG_CUT}
          AND g_phot_g_mean_mag <= {MAG_CUT_FAINT}
          {cs_filter}
        ORDER BY g_random_index
    """
    if test_mode:
        query += " LIMIT 5000"

    print(f"  Querying ERO stars ({cs_label}, {_mag_range()})...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    X_ero   = df[SELECTED_GAIA_PARAMETERS].values
    y_ero   = np.zeros(len(X_ero), dtype=int)  # all STAR
    mag_ero = df['phot_g_mean_mag'].values.astype(float)

    # Report NaN rates
    n_total = len(X_ero)
    print(f"  -> {n_total:,} ERO stars (all labelled STAR)")
    for i, col in enumerate(SELECTED_GAIA_PARAMETERS):
        n_nan = np.isnan(X_ero[:, i].astype(float)).sum()
        if n_nan > 0:
            print(f"     {col:<35} {n_nan:>6,} NaN ({100*n_nan/n_total:.1f}%)")

    return X_ero, y_ero, mag_ero


def query_q1_extrap(username):
    """Full-feature query for sources OUTSIDE the training range.
    Used only for per-bin extrapolation evaluation."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    conditions = []
    if MAG_CUT > 15:
        conditions.append(f'g_phot_g_mean_mag < {MAG_CUT}')
    if MAG_CUT_FAINT < 90:
        conditions.append(f'g_phot_g_mean_mag > {MAG_CUT_FAINT}')
    if not conditions:
        return None
    where_extrap = ' OR '.join(conditions)
    query = f"""
        SELECT
            g_phot_g_mean_mag AS phot_g_mean_mag,
            {gaia_cols},
            CASE
                WHEN p_phz_classification = 1 THEN 'STAR'
                WHEN p_phz_classification = 2 THEN 'GALAXY'
            END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep
        WHERE p_phz_classification IN (1, 2)
          AND g_phot_g_mean_mag IS NOT NULL
          AND g_phot_g_mean_mag >= 15
          AND ({where_extrap})
        ORDER BY g_random_index
    """
    print(f"  Querying Q1 extrapolation sources (outside {_mag_range()})...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    X_e   = df[SELECTED_GAIA_PARAMETERS].values
    y_e   = df['label'].map(CLASS_MAPPING).values
    mag_e = df['phot_g_mean_mag'].values.astype(float)
    print(f"  -> {len(X_e):,} extrapolation sources")
    return X_e, y_e, mag_e


def query_q1_population(username):
    """All Gaia-matched Q1 sources (G >= 15), mag + label only — for fraction panels."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = """
        SELECT
            g_phot_g_mean_mag AS phot_g_mean_mag,
            CASE
                WHEN p_phz_classification = 1  THEN 'STAR'
                WHEN p_phz_classification = 2  THEN 'GALAXY'
                WHEN p_phz_classification IN (4, 8) THEN 'QSO_GC'
                ELSE 'UNCLASSIFIED'
            END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia
        WHERE g_phot_g_mean_mag IS NOT NULL
          AND g_phot_g_mean_mag >= 15
        ORDER BY g_random_index
    """
    print("  Querying Q1 population (G >= 15, all classes)...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['phot_g_mean_mag'] = pd.to_numeric(df['phot_g_mean_mag'], errors='coerce')
    return df


# =============================================================================
# Training & evaluation
# =============================================================================

def train_model(X_train, y_train, X_val, y_val, verbose=True):
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=verbose)
    return model


def split_data_with_ero(X_q1, y_q1, X_ero, y_ero, seed, target_ratio=None):
    """Split Q1 data into train/val/test (70/15/15), then add ERO stars to train only.

    Val and test remain pure Q1 so metrics are comparable to the Q1-only model.

    target_ratio: desired STAR:GALAXY ratio in training set. If None, use all ERO stars.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_q1, y_q1, test_size=0.15, random_state=seed, stratify=y_q1)
    X_train_q1, X_val, y_train_q1, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=seed, stratify=y_temp)

    # Subsample ERO stars if target_ratio is set
    rng = np.random.RandomState(seed)
    n_q1_stars    = (y_train_q1 == 0).sum()
    n_q1_galaxies = (y_train_q1 == 1).sum()

    if target_ratio is not None:
        n_stars_needed = int(target_ratio * n_q1_galaxies)
        n_ero_to_add   = max(0, n_stars_needed - n_q1_stars)
        if n_ero_to_add < len(X_ero):
            ero_idx = rng.choice(len(X_ero), size=n_ero_to_add, replace=False)
            X_ero_use = X_ero[ero_idx]
            y_ero_use = y_ero[ero_idx]
        else:
            X_ero_use = X_ero
            y_ero_use = y_ero
    else:
        X_ero_use = X_ero
        y_ero_use = y_ero

    # Augment training set with ERO stars
    X_train = np.vstack([X_train_q1, X_ero_use])
    y_train = np.concatenate([y_train_q1, y_ero_use])

    # Shuffle the combined training set
    shuffle_idx = rng.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test, len(X_train_q1), len(X_ero_use)


def evaluate(model, X_test, y_test):
    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    report       = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    roc_auc      = roc_auc_score(y_test, y_pred_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1])
    return {
        'accuracy':         (y_test == y_pred).sum() / len(y_test),
        'star_precision':   precision[0],
        'star_recall':      recall[0],
        'star_f1':          f1[0],
        'galaxy_precision': precision[1],
        'galaxy_recall':    recall[1],
        'galaxy_f1':        f1[1],
        'roc_auc':          roc_auc,
        'report':           report,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_threshold_analysis(X_q1, y_q1, all_models, seeds, output_dir):
    """Per-seed threshold analysis (no leakage).
    Each seed's model evaluated on its own Q1 held-out test set,
    histograms and metrics averaged across seeds.
    """
    from sklearn.metrics import roc_curve, auc, precision_recall_curve

    thresholds = np.linspace(0, 1, 201)
    bins = np.linspace(0, 1, 101)
    n_seeds = len(seeds)

    hist_star_sum = np.zeros(len(bins) - 1)
    hist_gal_sum  = np.zeros(len(bins) - 1)
    all_prec = np.zeros((n_seeds, len(thresholds)))
    all_rec  = np.zeros((n_seeds, len(thresholds)))
    all_f1   = np.zeros((n_seeds, len(thresholds)))
    n_star_total, n_gal_total = 0, 0
    y_test_42, y_proba_42 = None, None

    for si, seed in enumerate(seeds):
        _, X_test, _, y_test = train_test_split(
            X_q1, y_q1, test_size=0.15, random_state=seed, stratify=y_q1)
        y_proba = all_models[si].predict_proba(X_test)[:, 1]

        hist_star_sum += np.histogram(y_proba[y_test == 0], bins=bins)[0]
        hist_gal_sum  += np.histogram(y_proba[y_test == 1], bins=bins)[0]
        n_star_total += (y_test == 0).sum()
        n_gal_total  += (y_test == 1).sum()

        if seed == 42:
            y_test_42, y_proba_42 = y_test, y_proba

        for ti, t in enumerate(thresholds):
            pred = (y_proba >= t).astype(int)
            tp = ((pred == 1) & (y_test == 1)).sum()
            fp = ((pred == 1) & (y_test == 0)).sum()
            fn = ((pred == 0) & (y_test == 1)).sum()
            p = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            all_prec[si, ti] = p
            all_rec[si, ti]  = r
            all_f1[si, ti]   = 2*p*r/(p+r) if (p + r) > 0 else 0.0

    hist_star_mean = hist_star_sum / n_seeds
    hist_gal_mean  = hist_gal_sum / n_seeds
    prec_arr = np.nanmean(all_prec, axis=0)
    rec_arr  = np.nanmean(all_rec, axis=0)
    f1_arr   = np.nanmean(all_f1, axis=0)
    best_idx = np.nanargmax(f1_arr)
    best_t   = thresholds[best_idx]
    n_star_avg = n_star_total // n_seeds
    n_gal_avg  = n_gal_total // n_seeds

    if y_test_42 is None:
        y_test_42, y_proba_42 = y_test, y_proba
    fpr, tpr, _ = roc_curve(y_test_42, y_proba_42)
    roc_auc_val = auc(fpr, tpr)
    pr_prec, pr_rec, pr_thresh = precision_recall_curve(y_test_42, y_proba_42)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.subplots_adjust(wspace=0.38)

    # [0] P_gal distribution — averaged histogram
    ax = axes[0]
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]
    ax.bar(bin_centres, hist_star_mean, width=bin_width, alpha=0.6, color='coral', edgecolor='none',
           label=f'STAR  (n={n_star_avg:,})')
    ax.bar(bin_centres, hist_gal_mean, width=bin_width, alpha=0.6, color='steelblue', edgecolor='none',
           label=f'GALAXY  (n={n_gal_avg:,})')
    ax.axvline(0.5,   color='black',        lw=1.5, ls='--', label='Default (0.5)')
    ax.axvline(best_t, color='mediumseagreen', lw=1.5, ls=':',  label=f'Best F1 ({best_t:.2f})')
    ax.set_yscale('log')
    ax.set_xlabel('P_gal', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('P_gal distribution — Q1 test set', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # [1] Precision / Recall / F1 vs threshold — averaged
    ax = axes[1]
    ax.plot(thresholds, prec_arr, color='steelblue',      lw=2, label='Precision')
    ax.plot(thresholds, rec_arr,  color='coral',          lw=2, label='Recall')
    ax.plot(thresholds, f1_arr,   color='mediumseagreen', lw=2, label='F1')
    ax.axvline(0.5,   color='black',          lw=1.5, ls='--', label='Default (0.5)')
    ax.axvline(best_t, color='mediumseagreen', lw=1.5, ls=':',  label=f'Best F1 ({best_t:.2f})')
    bbox = dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=1.0)
    for t_val in [0.5, 0.7, 0.9]:
        idx_t = np.argmin(np.abs(thresholds - t_val))
        for arr, color in [(prec_arr, 'steelblue'), (rec_arr, 'coral'), (f1_arr, 'mediumseagreen')]:
            ax.text(t_val, arr[idx_t], f'{arr[idx_t]:.3f}', fontsize=9, fontweight='bold',
                    color=color, ha='center', va='center', bbox=bbox, zorder=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('P_gal threshold', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Metrics vs threshold — Q1 test set', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='center left'); ax.grid(alpha=0.3)

    # [2] ROC curve (seed-42)
    ax = axes[2]
    ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC  (AUC = {roc_auc_val:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, ls='--', label='Random')
    roc_thresh = roc_curve(y_test_42, y_proba_42)[2]
    ax.scatter(fpr[np.argmin(np.abs(roc_thresh - 0.5))],
               tpr[np.argmin(np.abs(roc_thresh - 0.5))],
               s=80, color='black',          zorder=6, label='Default (0.5)')
    ax.scatter(fpr[np.argmin(np.abs(roc_thresh - best_t))],
               tpr[np.argmin(np.abs(roc_thresh - best_t))],
               s=80, color='mediumseagreen', zorder=6, label=f'Best F1 ({best_t:.2f})')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC curve — Q1 test set (GALAXY = positive)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # [3] Precision-Recall curve (seed-42)
    ax = axes[3]
    idx_def = np.argmin(np.abs(thresholds - 0.5))
    sc = ax.scatter(pr_rec[:-1], pr_prec[:-1], c=pr_thresh,
                    cmap='viridis', s=8, zorder=5)
    fig.colorbar(sc, ax=ax, pad=0.02).set_label('P_gal threshold', fontsize=9)
    ax.scatter(rec_arr[idx_def], prec_arr[idx_def],
               s=80, color='black',          zorder=6, label='Default (0.5)')
    ax.scatter(rec_arr[best_idx], prec_arr[best_idx],
               s=80, color='mediumseagreen', zorder=6, label=f'Best F1 ({best_t:.2f})')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall curve — Q1 test set', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle(f'Q1+ERO model threshold analysis  (per-seed mean, {n_seeds} seeds)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = output_dir / 'q1_threshold_analysis.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {out}")


def plot_feature_importance(importances, feature_names, output_dir, metrics_summary=None):
    """Horizontal bar chart of feature importances with seed-to-seed CV."""
    imp = np.atleast_2d(importances)
    mu = imp.mean(axis=0)
    std = imp.std(axis=0)
    cv = std / (mu + 1e-9)

    order = np.argsort(mu)[::-1]
    feats = [feature_names[i] for i in order]
    mu_s = mu[order]
    std_s = std[order]
    cv_s = cv[order]

    n = len(feats)
    fig, ax = plt.subplots(figsize=(18, max(8, n * 0.38)))

    cv_abs_max = 0.5
    norm = plt.Normalize(vmin=0, vmax=cv_abs_max)
    colors = plt.cm.RdYlGn_r(norm(cv_s))

    bars = ax.barh(
        range(n), mu_s, xerr=std_s,
        color=colors, edgecolor='white', capsize=4,
        error_kw={'elinewidth': 1.2, 'ecolor': 'black'},
    )
    ax.set_yticks(range(n))
    ax.set_yticklabels([f'#{i+1}  {f}' for i, f in enumerate(feats)], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance  (mean \u00b1 1\u03c3 across seeds)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 0.5)
    ax.set_xticks(np.arange(0, 0.51, 0.1))
    ax.grid(alpha=0.3, axis='x')

    for i, (bar, mean_val, std_val, cv_val) in enumerate(zip(bars, mu_s, std_s, cv_s)):
        label = f'{mean_val:.4f}  CV={cv_val:.2f}'
        ax.text(mean_val + std_val + 0.001, bar.get_y() + bar.get_height() / 2,
                label, va='center', fontsize=8)

    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label('CV = std / mean', fontsize=9)
    cbar.ax.text(0.4, 1.04, 'unstable', transform=cbar.ax.transAxes,
                 fontsize=8, va='bottom', ha='left', color='darkred')
    cbar.ax.text(0.4, -0.05, 'stable', transform=cbar.ax.transAxes,
                 fontsize=8, va='top', ha='left', color='darkgreen')

    if metrics_summary is not None:
        lines = ['Test-set metrics (Q1 only):']
        for _, label, mean_val, std_val in metrics_summary:
            lines.append(f'{label:<16}: {mean_val:.4f} \u00b1 {std_val:.4f}')
        ax.text(
            0.6, 0.05, '\n'.join(lines),
            transform=ax.transAxes,
            fontsize=9.5,
            family='monospace',
            va='bottom',
            ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#bbbbbb', alpha=0.95),
        )

    title = f'Q1+ERO Model Feature Importance Stability  ({len(imp)} seeds, {_mag_range()})'
    plt.suptitle(title, fontsize=13, fontweight='bold', y=0.96)
    plt.tight_layout()

    out = output_dir / 'q1_feature_importance.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {out}")

    csv_path = output_dir / 'q1_feature_importance.csv'
    out_df = pd.DataFrame({
        'feature': feats,
        'importance_mean': mu_s,
        'importance_std': std_s,
        'importance_cv': cv_s,
        'rank': range(1, n + 1),
    })
    out_df.to_csv(csv_path, index=False)
    print(f"  -> {csv_path}")


# =============================================================================
# Per-bin performance plot
# =============================================================================

def _fmt_faint(m):
    return f'{m:.0f}' if m == int(m) else str(m)


def _mag_range():
    if MAG_CUT_FAINT >= 90:
        return f'{MAG_CUT} \u2264 G'
    return f'{MAG_CUT} \u2264 G \u2264 {_fmt_faint(MAG_CUT_FAINT)}'


def _mag_range_dir():
    if MAG_CUT_FAINT >= 90:
        return f'{MAG_CUT}\u2264G'
    return f'{MAG_CUT}\u2264G\u2264{_fmt_faint(MAG_CUT_FAINT)}'


_MAG_BINS = [round(e, 2) for e in np.arange(15.0, 22.01, 0.25)[:-1]]
_BIN_HALF = 0.125
_MIN_GAL  = 5


def plot_perbin_performance(all_models, all_test_idx, X_q1, y_q1, mag_q1,
                            df_pop, output_dir, extrap=None):
    """Evaluate each seed model on its own 15% test split (Q1 only), binned by G."""
    from matplotlib.lines import Line2D

    n_seeds = len(all_models)
    x       = np.array(_MAG_BINS)

    X_e, y_e, mag_e = extrap if extrap is not None else (
        np.empty((0, X_q1.shape[1])), np.empty(0, dtype=int), np.empty(0))
    all_prec, all_rec, all_f1, all_auc = [], [], [], []
    all_pred_gf, all_pred_sf = [], []
    for model, idx_test in zip(all_models, all_test_idx):
        X_t   = X_q1[idx_test]
        y_t   = y_q1[idx_test]
        mag_t = mag_q1[idx_test]
        prec_row, rec_row, f1_row, auc_row = [], [], [], []
        pred_gf_row, pred_sf_row = [], []
        for centre in _MAG_BINS:
            lo, hi = centre - _BIN_HALF, centre + _BIN_HALF
            in_range = (centre >= MAG_CUT) and (centre < MAG_CUT_FAINT)
            if in_range:
                mask = (mag_t >= lo) & (mag_t < hi)
                X_eval, y_eval = X_t[mask], y_t[mask]
            else:
                mask = (mag_e >= lo) & (mag_e < hi)
                X_eval, y_eval = X_e[mask], y_e[mask]
            n_gal  = int((y_eval == 1).sum())
            n_star = int((y_eval == 0).sum())
            if len(y_eval) < 10 or n_gal < _MIN_GAL or n_star < _MIN_GAL:
                prec_row.append(np.nan); rec_row.append(np.nan); f1_row.append(np.nan)
                auc_row.append(np.nan)
                pred_gf_row.append(np.nan); pred_sf_row.append(np.nan)
                continue
            y_pred       = model.predict(X_eval)
            y_pred_proba = model.predict_proba(X_eval)[:, 1]
            p, r, f1, _ = precision_recall_fscore_support(
                y_eval, y_pred, labels=[0, 1], zero_division=0)
            prec_row.append(p[1]); rec_row.append(r[1]); f1_row.append(f1[1])
            auc_row.append(roc_auc_score(y_eval, y_pred_proba))
            pred_gf_row.append((y_pred == 1).sum() / len(y_pred))
            pred_sf_row.append((y_pred == 0).sum() / len(y_pred))
        all_prec.append(prec_row); all_rec.append(rec_row); all_f1.append(f1_row)
        all_auc.append(auc_row)
        all_pred_gf.append(pred_gf_row); all_pred_sf.append(pred_sf_row)

    pred_gf = np.nanmean(np.array(all_pred_gf), axis=0)
    pred_sf = np.nanmean(np.array(all_pred_sf), axis=0)

    prec = np.array(all_prec); rec = np.array(all_rec); f1 = np.array(all_f1)
    auc  = np.array(all_auc)

    # per-bin classified fracs
    if X_e.shape[0] > 0:
        mag_cl = np.concatenate([mag_q1, mag_e])
        y_cl   = np.concatenate([y_q1,   y_e])
    else:
        mag_cl, y_cl = mag_q1, y_q1
    label_cl = np.where(y_cl == 0, 'STAR', 'GALAXY')
    gf_c, sf_c, ngal_c = [], [], []
    for centre in _MAG_BINS:
        lo, hi = centre - _BIN_HALF, centre + _BIN_HALF
        m = (mag_cl >= lo) & (mag_cl < hi)
        tot = m.sum()
        ng  = (label_cl[m] == 'GALAXY').sum()
        ns  = (label_cl[m] == 'STAR').sum()
        gf_c.append(ng / tot if tot > 0 else np.nan)
        sf_c.append(ns / tot if tot > 0 else np.nan)
        ngal_c.append(int(ng))
    gf_c = np.array(gf_c); sf_c = np.array(sf_c); ngal_c = np.array(ngal_c)

    # per-bin all-source fracs from population
    mag_p   = df_pop['phot_g_mean_mag'].values
    label_p = df_pop['label'].values
    gf_a, sf_a, of_a, ngal_a = [], [], [], []
    for centre in _MAG_BINS:
        lo, hi = centre - _BIN_HALF, centre + _BIN_HALF
        m   = (mag_p >= lo) & (mag_p < hi)
        tot = m.sum()
        ng  = (label_p[m] == 'GALAXY').sum()
        ns  = (label_p[m] == 'STAR').sum()
        no  = ((label_p[m] == 'QSO_GC') | (label_p[m] == 'UNCLASSIFIED')).sum()
        gf_a.append(ng / tot if tot > 0 else np.nan)
        sf_a.append(ns / tot if tot > 0 else np.nan)
        of_a.append(no / tot if tot > 0 else np.nan)
        ngal_a.append(int(ng))
    gf_a = np.array(gf_a); sf_a = np.array(sf_a)
    of_a = np.array(of_a); ngal_a = np.array(ngal_a)

    # --- plot ---
    def _shade(ax):
        ax.axvspan(x[0] - 0.15, MAG_CUT,       color='lightyellow', alpha=0.6, zorder=0)
        ax.axvspan(MAG_CUT_FAINT, x[-1] + 0.15, color='lightyellow', alpha=0.6, zorder=0)
        ax.axvline(x=MAG_CUT,       color='navy',    ls='--', lw=1.5, alpha=0.8)
        ax.axvline(x=18.5,          color='black',   ls='--', lw=1.5, alpha=0.8)
        ax.axvline(x=MAG_CUT_FAINT, color='darkred', ls='--', lw=1.5, alpha=0.8)
        ax.axhline(y=0.5, color='gray', ls=':', lw=1)
        ax.grid(alpha=0.3, axis='y')

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(14, 17),
        gridspec_kw={'height_ratios': [3, 1.2, 1.2]}, sharex=True)

    # top: metrics
    _shade(ax_top)
    metric_handles = []
    mu_dict = {}
    for label, arr, color, marker in [
        ('Galaxy Precision', prec, 'steelblue',     'o'),
        ('Galaxy F1',        f1,   'mediumseagreen', 's'),
        ('Galaxy Recall',    rec,  'coral',           '^'),
        ('ROC AUC',          auc,  'pink',    'D'),
    ]:
        mu  = np.nanmean(arr, axis=0)
        std = np.nanstd(arr,  axis=0)
        line, = ax_top.plot(x, mu, marker=marker, color=color, lw=2, markersize=8,
                            label=label, zorder=3)
        ax_top.fill_between(x, mu - std, mu + std, alpha=0.18, color=color, zorder=2)
        metric_handles.append(line)
        mu_dict[label] = mu

    offsets = {'Galaxy Precision': +0.053, 'Galaxy F1': +0.018, 'Galaxy Recall': -0.018, 'ROC AUC': -0.053}
    colors  = {'Galaxy Precision': 'steelblue', 'Galaxy F1': 'mediumseagreen', 'Galaxy Recall': 'coral', 'ROC AUC': 'pink'}
    short   = {'Galaxy Precision': 'P', 'Galaxy F1': 'F1', 'Galaxy Recall': 'R', 'ROC AUC': 'AUC'}
    for i in range(0, len(_MAG_BINS), 2):
        for lbl, mu in mu_dict.items():
            if np.isfinite(mu[i]):
                y_val = mu[i] + offsets[lbl]
                above = y_val < 0.92
                dy  =  0.018 if above else -0.018
                va  = 'bottom' if above else 'top'
                ax_top.text(x[i], y_val + dy, f'{short[lbl]}={mu[i]:.2f}',
                            ha='center', va=va, fontsize=6, color=colors[lbl],
                            zorder=6)
    ref_handles = [
        Line2D([0],[0], color='navy',  ls='--', lw=1.5, alpha=0.8,
               label=f'G = {MAG_CUT}  (train lo)'),
        Line2D([0],[0], color='black', ls='--', lw=1.5, alpha=0.8, label='G = 18.5'),
    ]
    if MAG_CUT_FAINT < 90:
        ref_handles.append(Line2D([0],[0], color='darkred', ls='--', lw=1.5, alpha=0.8,
                                  label=f'G = {MAG_CUT_FAINT}  (train hi)'))
    ax_top.legend(handles=metric_handles, fontsize=10, loc='lower left')
    ax_top.set_ylabel('Metric value', fontsize=12, fontweight='bold')
    ax_top.set_ylim(-0.05, 1.05)

    # mid: classified fracs
    _shade(ax_mid)
    ax_mid.bar(x, gf_c, width=0.22, color='orchid', alpha=0.25, zorder=1)
    ax_mid.plot(x, sf_c,    color='coral',   lw=2, zorder=4, label='Star (true)',   alpha=0.5)
    ax_mid.plot(x, gf_c,    color='#9B30B0', lw=2, zorder=4, label='Galaxy (true)')
    ax_mid.plot(x, pred_sf, color='coral',   lw=2, zorder=4, ls='--',
                label='Star (pred)', alpha=0.5)
    ax_mid.plot(x, pred_gf, color='#9B30B0', lw=2, zorder=4, ls='--',
                label='Galaxy (pred)')
    ax_mid.set_ylabel('Fraction', fontsize=11, fontweight='bold')
    ax_mid.set_ylim(0, 1.25)
    ax_mid.legend(fontsize=9, loc='upper left', ncol=4)
    ax_mid.set_title('PHZ-labelled fraction (solid) vs model-predicted fraction (dashed)  \u2014 STAR + GALAXY per bin',
                     fontsize=10, loc='left', pad=3)
    for i in range(0, len(_MAG_BINS), 2):
        if ngal_c[i] > 0:
            ax_mid.text(x[i], (gf_c[i] if np.isfinite(gf_c[i]) else 0) + 0.05,
                        f'{ngal_c[i]:,}', ha='center', fontsize=7, color='#333333', zorder=10)

    # bot: all sources
    _shade(ax_bot)
    ax_bot.bar(x, gf_a, width=0.22, color='orchid', alpha=0.25, zorder=1)
    ax_bot.plot(x, sf_a, color='coral',   lw=2, zorder=4, label='Star',         alpha=0.5)
    ax_bot.plot(x, gf_a, color='#9B30B0', lw=2, zorder=4, label='Galaxy')
    ax_bot.plot(x, of_a, color='#8B4513', lw=2, zorder=4, ls='--',
                label='Unclassified', alpha=0.5)
    ax_bot.set_ylabel('Fraction', fontsize=11, fontweight='bold')
    ax_bot.set_ylim(0, 1.25)
    ax_bot.legend(fontsize=9, loc='upper left', ncol=3)
    ax_bot.set_title('All Gaia-matched Q1 sources  (incl. unclassified, per bin)',
                     fontsize=10, loc='left', pad=3)
    for i in range(0, len(_MAG_BINS), 2):
        if ngal_a[i] > 0:
            ax_bot.text(x[i], (gf_a[i] if np.isfinite(gf_a[i]) else 0) + 0.05,
                        f'{ngal_a[i]:,}', ha='center', fontsize=7, color='#333333', zorder=10)

    tick_vals = [c for c in _MAG_BINS if c == round(c)]
    ax_bot.set_xlim(x[0] - 0.2, x[-1] + 0.2)
    ax_bot.set_xticks(tick_vals)
    ax_bot.set_xticklabels([f'G\u2248{c:.0f}' for c in tick_vals], fontsize=10)
    ax_bot.set_xlabel(
        f'Gaia G bin centre  (model trained on Q1+ERO, {_mag_range()}) ',
        fontsize=12, fontweight='bold', labelpad=13)

    plt.suptitle(
        f'Q1+ERO model ({_mag_range()}) \u2014 Galaxy metrics per magnitude bin  '
        f'(mean \u00b1 1\u03c3, {n_seeds} seeds)',
        fontsize=13, fontweight='bold', y=0.99)
    fig.subplots_adjust(top=0.90)
    plt.tight_layout()

    out = output_dir / 'q1_perbin_performance.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {out}")

    npz_out = output_dir / 'q1_perbin_data.npz'
    np.savez(npz_out,
             x=x,
             prec=prec, rec=rec, f1=f1, auc=auc,
             pred_gf=pred_gf, pred_sf=pred_sf,
             gf_c=gf_c, sf_c=sf_c, ngal_c=ngal_c,
             gf_a=gf_a, sf_a=sf_a, of_a=of_a, ngal_a=ngal_a,
             mag_cut=MAG_CUT, mag_cut_faint=MAG_CUT_FAINT)
    print(f"  -> {npz_out}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='XGBoost Q1+ERO model training')
    parser.add_argument('--test-mode', action='store_true',
                        help='Limit to 10 k Q1 + 5 k ERO sources for quick testing')
    parser.add_argument('--target-ratio', type=float, default=None, metavar='R',
                        help='Train at a single STAR:GALAXY ratio (e.g. 4, 5). '
                             'ERO stars are subsampled to achieve this. '
                             'Default: sweep all integer ratios + all ERO stars.')
    parser.add_argument('--no-sweep', action='store_true',
                        help='With no --target-ratio, use all ERO stars instead of sweeping.')
    parser.add_argument('--ero-mag-cut', type=float, default=None, metavar='MAG',
                        help='Additional bright-end magnitude cut on ERO stars only '
                             '(e.g. 18.5). Saves to xgboost_mix_q1_ero_g/ instead.')
    parser.add_argument('--class-star', type=float, default=0.9, metavar='THRESH',
                        help='ERO class_star threshold (default: 0.9). '
                             'Set to 0 to disable filtering (all ERO sources as stars).')
    args = parser.parse_args()

    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    if args.target_ratio is not None:
        ratio_desc = f'{args.target_ratio:.0f}:1 target'
    elif args.no_sweep:
        ratio_desc = 'all ERO stars'
    else:
        ratio_desc = 'sweep all ratios'

    print("=" * 70)
    print("XGBoost Q1+ERO Model Training")
    print("=" * 70)
    print(f"  Features:       {len(SELECTED_GAIA_PARAMETERS)}")
    print(f"  Magnitude cut:  {_mag_range()}")
    cs_desc = "no class_star filter" if args.class_star == 0 else f"class_star > {args.class_star}"
    print(f"  ERO criterion:  {cs_desc}")
    print(f"  ERO mixing:     {ratio_desc}")
    print(f"  Training mode:  {len(SEEDS)}-seed ensemble")
    if args.test_mode:
        print("  TEST MODE: limited sources")

    # ── Step 1: Query Q1 data ────────────────────────────────────────────────
    print("\n[1] Querying Q1 data...")
    X_q1, y_q1, mag_q1 = query_q1(username, test_mode=args.test_mode)

    # ── Step 1b: Query ERO stars ─────────────────────────────────────────────
    print("\n[1b] Querying ERO stars...")
    X_ero, y_ero, mag_ero = query_ero_stars(username, test_mode=args.test_mode,
                                              class_star_thresh=args.class_star)

    # Optional additional magnitude cut on ERO stars
    if args.ero_mag_cut is not None:
        mask = mag_ero >= args.ero_mag_cut
        print(f"  Filtering ERO to G >= {args.ero_mag_cut}: "
              f"{mask.sum():,} / {len(mask):,} sources kept")
        X_ero = X_ero[mask]
        y_ero = y_ero[mask]
        mag_ero = mag_ero[mask]

    n_q1_star   = (y_q1 == 0).sum()
    n_q1_galaxy = (y_q1 == 1).sum()
    n_ero       = len(X_ero)
    print(f"\n  Combined training pool:")
    print(f"    Q1 stars:     {n_q1_star:,}")
    print(f"    Q1 galaxies:  {n_q1_galaxy:,}")
    print(f"    ERO stars:    {n_ero:,}")
    print(f"    Total stars:  {n_q1_star + n_ero:,}")
    print(f"    STAR:GALAXY ratio (full pool): {(n_q1_star + n_ero) / n_q1_galaxy:.1f}:1")

    # ── Compute ratio list ─────────────────────────────────────────────────────
    if args.target_ratio is None and not args.no_sweep:
        n_train_gal = int(n_q1_galaxy * 0.85 * 0.824)
        n_train_star = int(n_q1_star * 0.85 * 0.824)
        q1_ratio = n_q1_star / n_q1_galaxy
        max_ratio = (n_train_star + n_ero) / max(n_train_gal, 1)
        min_ratio_int = int(np.ceil(q1_ratio))
        max_ratio_int = int(np.floor(max_ratio))
        # 0 = Q1 only (no ERO), integers = ERO-augmented, None = all ERO
        sweep_ratios = [0] + list(range(min_ratio_int, max_ratio_int + 1)) + [None]
        print(f"\n  Sweeping ratios: Q1 only, {min_ratio_int}:1 to {max_ratio_int}:1, "
              f"+ all ERO ({max_ratio:.1f}:1)")
        for sr in sweep_ratios:
            if sr == 0:
                label = "Q1 only (no ERO)"
                print(f"\n{'='*70}")
                print(f"  SWEEP: {label} — linking to output/xgboost_q1/")
                print(f"{'='*70}")
                _link_q1_baseline(args, n_q1_star, n_q1_galaxy)
            else:
                if sr is not None:
                    label = f"{sr}:1"
                else:
                    label = f"all ERO ({max_ratio:.1f}:1)"
                print(f"\n{'='*70}")
                print(f"  SWEEP: training ratio {label}")
                print(f"{'='*70}")
                _train_one_ratio(sr, args, username, X_q1, y_q1, mag_q1, X_ero, y_ero,
                                 n_q1_star, n_q1_galaxy, n_ero)
        return 0

    # Single ratio or --no-sweep
    if args.target_ratio == 0:
        _link_q1_baseline(args, n_q1_star, n_q1_galaxy)
    else:
        _train_one_ratio(args.target_ratio, args, username, X_q1, y_q1, mag_q1,
                         X_ero, y_ero, n_q1_star, n_q1_galaxy, n_ero)
    return 0


def _link_q1_baseline(args, n_q1_star, n_q1_galaxy):
    """Create a symlink to output/xgboost_q1/ for the Q1-only baseline.
    This ensures the mix sweep uses the exact same models as the Q1-only training,
    avoiding any inconsistency from retraining."""
    import shutil

    q1_ratio = n_q1_star / n_q1_galaxy
    # Output directory: base varies by ERO filtering
    base_dir = 'xgboost_mix_q1_ero'
    suffixes = []
    if args.ero_mag_cut:
        suffixes.append(f'g{args.ero_mag_cut:g}')
    if args.class_star != 0.9:
        suffixes.append(f'cs{args.class_star}' if args.class_star > 0 else 'nocs')
    if suffixes:
        base_dir += '_' + '_'.join(suffixes)
    link_dir = PROJECT_DIR / 'output' / base_dir / f'ratio_sg{q1_ratio:.1f}_q1_only'
    q1_dir = PROJECT_DIR / 'output' / 'xgboost_q1'

    link_dir.parent.mkdir(parents=True, exist_ok=True)

    if link_dir.is_symlink():
        link_dir.unlink()
    elif link_dir.exists():
        shutil.rmtree(link_dir)

    link_dir.symlink_to(q1_dir.resolve())
    print(f"  -> {link_dir} -> {q1_dir}")
    print(f"  Q1-only baseline: using existing models from {q1_dir}")


def _train_one_ratio(target_ratio, args, username, X_q1, y_q1, mag_q1,
                     X_ero, y_ero, n_q1_star, n_q1_galaxy, n_ero):
    """Train 10-seed ensemble at a single STAR:GALAXY ratio."""
    global OUTPUT_DIR, SEED_MODELS_DIR

    if target_ratio == 0:
        ratio_desc = 'Q1 only (no ERO)'
    elif target_ratio is not None:
        ratio_desc = f'{target_ratio:.0f}:1 target'
    else:
        ratio_desc = 'all ERO stars'

    if target_ratio == 0:
        q1_ratio = n_q1_star / n_q1_galaxy
        ratio_tag = f'ratio_sg{q1_ratio:.1f}_q1_only'
    elif target_ratio is not None:
        ratio_tag = f'ratio_sg{target_ratio:.0f}'
    else:
        actual_ratio = (0.7 * n_q1_star + n_ero) / (0.7 * n_q1_galaxy)
        ratio_tag = f'ratio_sg{actual_ratio:.2f}_all_ero'
    # Output directory: base varies by ERO filtering
    base_dir = 'xgboost_mix_q1_ero'
    suffixes = []
    if args.ero_mag_cut:
        suffixes.append(f'g{args.ero_mag_cut:g}')
    if args.class_star != 0.9:
        suffixes.append(f'cs{args.class_star}' if args.class_star > 0 else 'nocs')
    if suffixes:
        base_dir += '_' + '_'.join(suffixes)
    OUTPUT_DIR = PROJECT_DIR / 'output' / base_dir / ratio_tag
    SEED_MODELS_DIR = OUTPUT_DIR / 'seed_models'

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SEED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Output:         {OUTPUT_DIR}")

    # ── Step 2-4: Train/evaluate per seed ────────────────────────────────────
    seeds = SEEDS
    all_metrics = []
    all_importances = []
    all_models = []
    all_test_idx = []
    split_summary = None
    idx_all = np.arange(len(X_q1))

    for idx, seed in enumerate(seeds, start=1):
        print(f"\n[2] Splitting Q1 data (70/15/15), adding ERO to train (seed={seed})...")
        X_train, X_val, X_test, y_train, y_val, y_test, n_q1_train, n_ero_used = \
            split_data_with_ero(X_q1, y_q1, X_ero, y_ero, seed,
                                target_ratio=target_ratio)

        # Track Q1 test indices for per-bin evaluation
        _, idx_test, _, _ = train_test_split(
            idx_all, y_q1, test_size=0.15, random_state=seed, stratify=y_q1)
        all_test_idx.append(idx_test)

        n_star_train   = (y_train == 0).sum()
        n_galaxy_train = (y_train == 1).sum()
        ratio = n_star_train / n_galaxy_train
        print(f"  Train: {len(X_train):,} (Q1: {n_q1_train:,} + ERO: {n_ero_used:,})  "
              f"Val: {len(X_val):,}  Test: {len(X_test):,}")
        print(f"  Train STAR:GALAXY ratio  {ratio:.1f}:1")

        verbose = False
        print(f"\n[3] Training model for seed {seed} ({idx}/{len(seeds)})...")
        model = train_model(X_train, y_train, X_val, y_val, verbose=verbose)
        print(f"  Best iteration: {model.best_iteration}  "
              f"Best val logloss: {model.best_score:.4f}")

        print("\n[4] Evaluating on Q1 test set...")
        metrics = evaluate(model, X_test, y_test)
        print(f"  Accuracy:         {metrics['accuracy']:.4f}")
        print(f"  Galaxy Precision: {metrics['galaxy_precision']:.4f}")
        print(f"  Galaxy Recall:    {metrics['galaxy_recall']:.4f}")
        print(f"  Galaxy F1:        {metrics['galaxy_f1']:.4f}")
        print(f"  Star F1:          {metrics['star_f1']:.4f}")
        print(f"  ROC AUC:          {metrics['roc_auc']:.4f}")

        metrics_clean = {k: v for k, v in metrics.items() if k != 'report'}
        metrics_clean['seed'] = seed
        all_metrics.append(metrics_clean)
        all_importances.append(model.feature_importances_)
        all_models.append(model)

        if seed == 42:
            split_summary = {
                'ratio': ratio,
                'n_train': len(X_train),
                'n_train_q1': n_q1_train,
                'n_train_ero': n_ero_used,
                'n_val': len(X_val),
                'n_test': len(X_test),
            }

        model_path = SEED_MODELS_DIR / f'xgb_classifier_q1_seed{seed}.json'
        model.get_booster().save_model(str(model_path))
        print(f"  -> {model_path}")

    # ── Step 5: Save baseline model / seed metrics ───────────────────────────
    print("\n[5] Saving model...")
    model_path = OUTPUT_DIR / 'xgb_classifier_q1.json'
    all_models[0].get_booster().save_model(str(model_path))
    print(f"  -> {model_path}")

    metrics_path = OUTPUT_DIR / 'q1_seed_metrics.json'
    pd.DataFrame(all_metrics).to_json(metrics_path, orient='records', indent=2)
    print(f"  -> {metrics_path}")

    # ── Step 6: Plots ─────────────────────────────────────────────────────────
    metric_df = pd.DataFrame(all_metrics)
    metric_means = metric_df.mean(numeric_only=True)
    metric_stds = metric_df.std(numeric_only=True)

    print("\n[6] Generating feature importance plot...")
    metrics_summary = [
        ('galaxy_precision', 'Galaxy Precision', metric_means['galaxy_precision'],
         metric_stds['galaxy_precision']),
        ('galaxy_recall', 'Galaxy Recall', metric_means['galaxy_recall'],
         metric_stds['galaxy_recall']),
        ('galaxy_f1', 'Galaxy F1', metric_means['galaxy_f1'],
         metric_stds['galaxy_f1']),
    ]
    plot_feature_importance(
        np.array(all_importances),
        SELECTED_GAIA_PARAMETERS,
        OUTPUT_DIR,
        metrics_summary=metrics_summary,
    )

    print("\n[6b] Generating threshold analysis plot...")
    plot_threshold_analysis(X_q1, y_q1, all_models, seeds, OUTPUT_DIR)

    print("\n[6c] Generating per-bin performance plot...")
    extrap = query_q1_extrap(username)
    df_pop = query_q1_population(username)
    plot_perbin_performance(all_models, all_test_idx, X_q1, y_q1, mag_q1, df_pop, OUTPUT_DIR,
                            extrap=extrap)

    # ── Step 7: Summary text ──────────────────────────────────────────────────
    summary_path = OUTPUT_DIR / 'q1_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("XGBOOST Q1+ERO MODEL TRAINING SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Magnitude cut : {_mag_range()}\n")
        f.write(f"Features      : {len(SELECTED_GAIA_PARAMETERS)}\n")
        f.write(f"Training mode : {len(SEEDS)}-seed ensemble\n")
        f.write(f"ERO criterion : class_star > {CLASS_STAR_THRESH}\n")
        f.write(f"ERO mixing    : {ratio_desc}\n")
        if split_summary is not None:
            f.write(f"Train ratio   : {split_summary['ratio']:.1f}:1  (STAR:GALAXY)\n")
            f.write(
                f"Train (Q1)    : {split_summary['n_train_q1']:,}\n"
                f"Train (ERO)   : {split_summary['n_train_ero']:,}\n"
                f"Train (total) : {split_summary['n_train']:,}\n"
                f"Val (Q1 only) : {split_summary['n_val']:,}\n"
                f"Test (Q1 only): {split_summary['n_test']:,}\n"
            )
        f.write("Baseline model: random_state=42\n\n")
        f.write("NOTE: ERO stars added to training set only.\n")
        f.write("      Val/Test splits are pure Q1 for comparable evaluation.\n\n")
        f.write("Parameters:\n")
        for i, p in enumerate(SELECTED_GAIA_PARAMETERS, 1):
            f.write(f"  {i:>2}. {p}\n")
        f.write("\nTest-set metrics (Q1 only):\n")
        for key, label in [
            ('accuracy', 'Accuracy'),
            ('galaxy_precision', 'Galaxy Precision'),
            ('galaxy_recall', 'Galaxy Recall'),
            ('galaxy_f1', 'Galaxy F1'),
            ('star_f1', 'Star F1'),
            ('roc_auc', 'ROC AUC'),
        ]:
            f.write(f"  {label:<16}: {metric_means[key]:.4f} \u00b1 {metric_stds[key]:.4f}\n")
    print(f"  -> {summary_path}")

    # ── Step 8: Comparison with Q1-only model ─────────────────────────────────
    q1_only_summary = PROJECT_DIR / 'output' / 'xgboost_q1' / _mag_range_dir() / 'q1_summary.txt'
    if q1_only_summary.exists():
        print("\n" + "=" * 70)
        print("Comparison with Q1-only model:")
        print("=" * 70)
        import re
        q1_metrics = {}
        for line in q1_only_summary.read_text().splitlines():
            for key, label in [
                ('galaxy_precision', 'Galaxy Precision'),
                ('galaxy_recall',    'Galaxy Recall'),
                ('galaxy_f1',        'Galaxy F1'),
                ('star_f1',          'Star F1'),
                ('roc_auc',          'ROC AUC'),
                ('accuracy',         'Accuracy'),
            ]:
                if label in line:
                    m = re.search(r':\s*([\d.]+)\s*', line)
                    if m:
                        q1_metrics[key] = float(m.group(1))
        print(f"  {'Metric':<18} {'Q1-only':>10} {'Q1+ERO':>10} {'Delta':>10}")
        print("  " + "-" * 50)
        for key, label in [
            ('accuracy',         'Accuracy'),
            ('galaxy_precision', 'Gal Precision'),
            ('galaxy_recall',    'Gal Recall'),
            ('galaxy_f1',        'Gal F1'),
            ('star_f1',          'Star F1'),
            ('roc_auc',          'ROC AUC'),
        ]:
            q1_val  = q1_metrics.get(key, float('nan'))
            ero_val = metric_means[key]
            delta   = ero_val - q1_val
            sign    = '+' if delta >= 0 else ''
            print(f"  {label:<18} {q1_val:>10.4f} {ero_val:>10.4f} {sign}{delta:>9.4f}")

    print("\n" + "=" * 70)
    print(f"Q1+ERO model training complete! ({ratio_desc})")
    print(f"All results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    sys.exit(main())
