#!/usr/bin/env python3
"""
XGBoost Q1-only model training & evaluation — Optuna Galaxy F1 hyperparameters.

Same pipeline as train_xgboost_q1.py but using hyperparameters found by
Optuna when optimising for Galaxy F1 (d=11, n=757, lr=0.0237).

Outputs saved to output/xgboost_q1_op_f1/.

Usage:
    PGUSER=siwei_zhao_2025 python src/train_xgboost_q1_op_f1.py [--test-mode]
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
OUTPUT_DIR  = PROJECT_DIR / 'output' / 'xgboost_q1_op_f1'
SEED_MODELS_DIR = OUTPUT_DIR / 'seed_models'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

CLASS_MAPPING = {'STAR': 0, 'GALAXY': 1}
CLASS_NAMES   = ['STAR', 'GALAXY']

MAG_CUT       = 17   # Gaia G bright-end cut
MAG_CUT_FAINT = 99   # Gaia G faint-end cut (99 = no cut); overridden by --faint-cut
SEEDS   = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]

XGB_PARAMS = {
    'objective':             'binary:logistic',
    'max_depth':             11,                # Optuna Galaxy F1 tuned
    'learning_rate':         0.0237,            # Optuna Galaxy F1 tuned
    'n_estimators':          757,               # Optuna Galaxy F1 tuned
    'subsample':             0.574,             # Optuna Galaxy F1 tuned
    'colsample_bytree':      0.942,             # Optuna Galaxy F1 tuned
    'min_child_weight':      4,                 # Optuna Galaxy F1 tuned
    'gamma':                 0.280,             # Optuna Galaxy F1 tuned
    'reg_alpha':             8.459,             # Optuna Galaxy F1 tuned
    'reg_lambda':            2.84e-5,           # Optuna Galaxy F1 tuned
    'random_state':          42,
    'tree_method':           'hist',
    'eval_metric':           'logloss',
    'early_stopping_rounds': 20,
}


# =============================================================================
# Data loading
# =============================================================================

def query_q1(username, test_mode=False):
    """Query Q1 PHZ sources (STAR/GALAXY) with Gaia G >= MAG_CUT and G <= MAG_CUT_FAINT."""
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
    print(f"  -> {len(X):,} sources  STAR: {n_star:,} ({100*n_star/len(X):.1f}%)  "
          f"GALAXY: {n_galaxy:,} ({100*n_galaxy/len(X):.1f}%)")

    return X, y, mag


def query_q1_extrap(username):
    """Full-feature query for sources OUTSIDE the training range (G<MAG_CUT or G>MAG_CUT_FAINT).
    Used only for per-bin extrapolation evaluation — never seen during training."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    conditions = []
    if MAG_CUT > 15:
        conditions.append(f'g_phot_g_mean_mag < {MAG_CUT}')
    if MAG_CUT_FAINT < 90:
        conditions.append(f'g_phot_g_mean_mag > {MAG_CUT_FAINT}')
    if not conditions:
        return None  # no out-of-range region
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


def split_data(X, y, seed):
    """Split data into train/val/test using the existing 70/15/15 pattern."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=seed, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test


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

def plot_threshold_analysis(y_test, y_proba_seeds, output_dir):
    """Precision / Recall / F1 vs threshold on Q1 test set.
    Uses mean probability across all seed models.
    2×2 grid:
      [0,0] P_gal distribution on test set (split by true label)
      [0,1] Precision / Recall / F1 vs threshold
      [1,0] ROC curve
      [1,1] Precision-Recall curve
    Output: q1_threshold_analysis.png
    """
    from sklearn.metrics import roc_curve, auc, precision_recall_curve

    y_proba = np.mean(y_proba_seeds, axis=0)   # ensemble mean
    thresholds = np.linspace(0, 1, 201)

    prec_list, rec_list, f1_list = [], [], []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        tp = ((pred == 1) & (y_test == 1)).sum()
        fp = ((pred == 1) & (y_test == 0)).sum()
        fn = ((pred == 0) & (y_test == 1)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec_list.append(p)
        rec_list.append(r)
        f1_list.append(2*p*r/(p+r) if (p + r) > 0 else 0.0)

    prec_arr = np.array(prec_list)
    rec_arr  = np.array(rec_list)
    f1_arr   = np.array(f1_list)
    best_idx = np.nanargmax(f1_arr)
    best_t   = thresholds[best_idx]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc_val = auc(fpr, tpr)
    pr_prec, pr_rec, pr_thresh = precision_recall_curve(y_test, y_proba)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.subplots_adjust(wspace=0.38)

    # [0] P_gal distribution split by true label
    ax = axes[0]
    bins = np.linspace(0, 1, 101)
    ax.hist(y_proba[y_test == 0], bins=bins, color='coral',     alpha=0.6, lw=0,
            label=f'STAR  (n={( y_test==0).sum():,})')
    ax.hist(y_proba[y_test == 1], bins=bins, color='steelblue', alpha=0.6, lw=0,
            label=f'GALAXY  (n={(y_test==1).sum():,})')
    ax.axvline(0.5,   color='black',        lw=1.5, ls='--', label='Default (0.5)')
    ax.axvline(best_t, color='mediumseagreen', lw=1.5, ls=':',  label=f'Best F1 ({best_t:.2f})')
    ax.set_yscale('log')
    ax.set_xlabel('P_gal', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('P_gal distribution — Q1 test set', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # [1] Precision / Recall / F1 vs threshold
    ax = axes[1]
    ax.plot(thresholds, prec_arr, color='steelblue',      lw=2, label='Precision')
    ax.plot(thresholds, rec_arr,  color='coral',          lw=2, label='Recall')
    ax.plot(thresholds, f1_arr,   color='mediumseagreen', lw=2, label='F1')
    ax.axvline(0.5,   color='black',          lw=1.5, ls='--', label='Default (0.5)')
    ax.axvline(best_t, color='mediumseagreen', lw=1.5, ls=':',  label=f'Best F1 ({best_t:.2f})')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('P_gal threshold', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Metrics vs threshold — Q1 test set', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='center left'); ax.grid(alpha=0.3)

    # [2] ROC curve
    ax = axes[2]
    ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC  (AUC = {roc_auc_val:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, ls='--', label='Random')
    # Mark operating point at default and best-F1 threshold
    idx_def  = np.argmin(np.abs(thresholds - 0.5))
    idx_best = best_idx
    fpr_c, tpr_c = roc_curve(y_test, y_proba)[:2]
    # interpolate to get FPR/TPR at our thresholds
    from scipy.interpolate import interp1d
    roc_thresh = roc_curve(y_test, y_proba)[2]
    ax.scatter(fpr[np.argmin(np.abs(roc_thresh - 0.5))],
               tpr[np.argmin(np.abs(roc_thresh - 0.5))],
               s=80, color='black',          zorder=6, label=f'Default (0.5)')
    ax.scatter(fpr[np.argmin(np.abs(roc_thresh - best_t))],
               tpr[np.argmin(np.abs(roc_thresh - best_t))],
               s=80, color='mediumseagreen', zorder=6, label=f'Best F1 ({best_t:.2f})')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC curve — Q1 test set (GALAXY = positive)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # [3] Precision-Recall curve
    ax = axes[3]
    sc = ax.scatter(pr_rec[:-1], pr_prec[:-1], c=pr_thresh,
                    cmap='viridis', s=8, zorder=5)
    fig.colorbar(sc, ax=ax, pad=0.02).set_label('P_gal threshold', fontsize=9)
    ax.scatter(rec_arr[idx_def],  prec_arr[idx_def],
               s=80, color='black',          zorder=6, label='Default (0.5)')
    ax.scatter(rec_arr[best_idx], prec_arr[best_idx],
               s=80, color='mediumseagreen', zorder=6, label=f'Best F1 ({best_t:.2f})')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall curve — Q1 test set', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle(f'Q1 model threshold analysis  (ensemble mean, {len(y_proba_seeds)} seeds)',
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
    ax.set_xlabel('Feature Importance  (mean ± 1σ across seeds)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 0.5)
    ax.set_xticks(np.arange(0, 0.51, 0.1))
    ax.grid(alpha=0.3, axis='x')

    for i, (bar, mean_val, std_val, cv_val) in enumerate(zip(bars, mu_s, std_s, cv_s)):
        label = f'{mean_val:.4f}  CV={cv_val:.2f}'
        ax.text(mean_val + std_val + 0.001, bar.get_y() + bar.get_height() / 2,
                label, va='center', fontsize=8)

    sm = plt.cm.ScalarMappable(
        cmap='RdYlGn_r',
        norm=norm
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label('CV = std / mean', fontsize=9)
    cbar.ax.text(0.4, 1.04, 'unstable', transform=cbar.ax.transAxes,
                 fontsize=8, va='bottom', ha='left', color='darkred')
    cbar.ax.text(0.4, -0.05, 'stable', transform=cbar.ax.transAxes,
                 fontsize=8, va='top', ha='left', color='darkgreen')

    if metrics_summary is not None:
        lines = ['Test-set metrics:']
        for _, label, mean_val, std_val in metrics_summary:
            lines.append(f'{label:<16}: {mean_val:.4f} ± {std_val:.4f}')
        ax.text(
            0.6, 0.05, '\n'.join(lines),
            transform=ax.transAxes,
            fontsize=9.5,
            family='monospace',
            va='bottom',
            ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#bbbbbb', alpha=0.95),
        )

    title = f'Q1 Model Feature Importance Stability  ({len(imp)} seeds, {_mag_range()})'
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
    """Format faint-cut value: drop .0 for whole numbers, keep decimals otherwise."""
    return f'{m:.0f}' if m == int(m) else str(m)


def _mag_range():
    """Human-readable magnitude range string, omitting faint bound if no-faint-cut."""
    if MAG_CUT_FAINT >= 90:
        return f'{MAG_CUT} \u2264 G'
    return f'{MAG_CUT} \u2264 G \u2264 {_fmt_faint(MAG_CUT_FAINT)}'


def _mag_range_dir():
    """Compact version for use as a directory name (no spaces)."""
    if MAG_CUT_FAINT >= 90:
        return f'{MAG_CUT}\u2264G'
    return f'{MAG_CUT}\u2264G\u2264{_fmt_faint(MAG_CUT_FAINT)}'


_MAG_BINS = [round(e, 2) for e in np.arange(15.0, 22.01, 0.25)[:-1]]
_BIN_HALF = 0.125
_MIN_GAL  = 5


def plot_perbin_performance(all_models, all_test_idx, X, y, mag, df_pop, output_dir,
                            extrap=None):
    """Evaluate each seed model on its own 15% test split, binned by G magnitude.

    Produces a 3-panel figure matching q1_perbin_fixedtrain_performance.png:
      top  — Galaxy Precision / F1 / Recall per bin (mean ± 1σ across seeds)
      mid  — STAR / GALAXY fraction per bin (classified sources only, 17-21 window)
      bot  — STAR / GALAXY / Unclassified fraction per bin (full population G >= 15)
    """
    from matplotlib.lines import Line2D

    n_seeds = len(all_models)
    x       = np.array(_MAG_BINS)

    # --- collect per-seed, per-bin metrics ---
    # Within training window: use per-seed 15% held-out test set
    # Outside training window: use extrapolation dataset (never seen during training)
    X_e, y_e, mag_e = extrap if extrap is not None else (
        np.empty((0, X.shape[1])), np.empty(0, dtype=int), np.empty(0))
    all_prec, all_rec, all_f1, all_auc = [], [], [], []
    all_pred_gf, all_pred_sf = [], []          # model-predicted fractions per bin
    for model, idx_test in zip(all_models, all_test_idx):
        X_t   = X[idx_test]
        y_t   = y[idx_test]
        mag_t = mag[idx_test]
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

    # --- per-bin classified fracs (STAR/GALAXY labelled sources, full range) ---
    # Combine training data with extrapolation data for full G coverage
    if X_e.shape[0] > 0:
        mag_cl  = np.concatenate([mag, mag_e])
        y_cl    = np.concatenate([y,   y_e])
    else:
        mag_cl, y_cl = mag, y
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

    # --- per-bin all-source fracs (from population df_pop G >= 15) ---
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

    # Annotate every other bin with P / F1 / R / AUC values
    offsets = {'Galaxy Precision': +0.053, 'Galaxy F1': +0.018, 'Galaxy Recall': -0.018, 'ROC AUC': -0.053}
    colors  = {'Galaxy Precision': 'steelblue', 'Galaxy F1': 'mediumseagreen', 'Galaxy Recall': 'coral', 'ROC AUC': 'pink'}
    short   = {'Galaxy Precision': 'P', 'Galaxy F1': 'F1', 'Galaxy Recall': 'R', 'ROC AUC': 'AUC'}
    for i in range(0, len(_MAG_BINS), 2):
        for lbl, mu in mu_dict.items():
            if np.isfinite(mu[i]):
                y_val = mu[i] + offsets[lbl]
                above = y_val < 0.92   # flip below when near top to stay in bounds
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

    # mid: classified fracs (true labels + model predictions)
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
    ax_mid.set_title('PHZ-labelled fraction (solid) vs model-predicted fraction (dashed)  — STAR + GALAXY per bin',
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
    ax_bot.set_xticklabels([f'G≈{c:.0f}' for c in tick_vals], fontsize=10)
    ax_bot.set_xlabel(
        f'Gaia G bin centre  (model trained on {_mag_range()}) ',
        fontsize=12, fontweight='bold', labelpad=13)

    plt.suptitle(
        f'Q1 model ({_mag_range()}) — Galaxy metrics per magnitude bin  '
        f'(mean ± 1σ, {n_seeds} seeds)',
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
    parser = argparse.ArgumentParser(description='XGBoost Q1 model training')
    parser.add_argument('--test-mode', action='store_true',
                        help='Limit to 10 k sources for quick testing')
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--faint-cut', type=float, metavar='MAG',
                     help='Upper G-mag limit for training (e.g. 21, 21.5, 22). '
                          'Default: no faint cut.')
    grp.add_argument('--no-faint-cut', action='store_true',
                     help='Explicitly remove the faint-end cut (default behaviour).')
    args = parser.parse_args()

    global MAG_CUT_FAINT, OUTPUT_DIR, SEED_MODELS_DIR
    if args.faint_cut is not None:
        MAG_CUT_FAINT = args.faint_cut

    OUTPUT_DIR      = PROJECT_DIR / 'output' / 'xgboost_q1_op_f1'
    SEED_MODELS_DIR = OUTPUT_DIR / 'seed_models'

    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SEED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("XGBoost Q1 Model Training")
    print("=" * 70)
    print(f"  Features:       {len(SELECTED_GAIA_PARAMETERS)}")
    print(f"  Magnitude cut:  {_mag_range()}")
    print(f"  Output:         {OUTPUT_DIR}")
    print(f"  Training mode:  {len(SEEDS)}-seed ensemble")
    if args.test_mode:
        print("  TEST MODE: limited to 10 k sources")

    # ── Step 1: Query ─────────────────────────────────────────────────────────
    print("\n[1] Querying Q1 data...")
    X, y, mag = query_q1(username, test_mode=args.test_mode)

    # ── Step 2-6: Train/evaluate/save ────────────────────────────────────────
    seeds = SEEDS
    all_metrics = []
    all_importances = []
    all_models = []
    all_test_idx = []
    all_y_test_proba = []
    y_test_ref = None
    split_summary = None
    idx_all = np.arange(len(X))

    for idx, seed in enumerate(seeds, start=1):
        print(f"\n[2] Splitting data (70/15/15, random_state={seed})...")
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed)
        # Track test indices for per-bin evaluation (same split params as split_data)
        _, idx_test, _, _ = train_test_split(
            idx_all, y, test_size=0.15, random_state=seed, stratify=y)
        all_test_idx.append(idx_test)

        n_star = (y_train == 0).sum()
        n_galaxy = (y_train == 1).sum()
        ratio = n_star / n_galaxy
        print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
        print(f"  Train STAR:GALAXY ratio  {ratio:.1f}:1")

        verbose = False
        print(f"\n[3] Training model for seed {seed} ({idx}/{len(seeds)})...")
        model = train_model(X_train, y_train, X_val, y_val, verbose=verbose)
        print(f"  Best iteration: {model.best_iteration}  "
              f"Best val logloss: {model.best_score:.4f}")

        print("\n[4] Evaluating on test set...")
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
                'n_val': len(X_val),
                'n_test': len(X_test),
            }

        model_path = SEED_MODELS_DIR / f'xgb_classifier_q1_seed{seed}.json'
        model.get_booster().save_model(str(model_path))
        print(f"  -> {model_path}")

    # ── Ensemble prediction on fixed seed-42 test split ───────────────────────
    # Use one consistent test set for threshold analysis (not per-seed splits)
    _, _, X_test_ref, _, _, y_test_ref = split_data(X, y, seed=42)
    all_y_test_proba = []
    for m in all_models:
        all_y_test_proba.append(m.predict_proba(X_test_ref)[:, 1])

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
    plot_threshold_analysis(y_test_ref, np.array(all_y_test_proba), OUTPUT_DIR)

    print("\n[6c] Generating per-bin performance plot...")
    extrap = query_q1_extrap(username)
    df_pop = query_q1_population(username)
    plot_perbin_performance(all_models, all_test_idx, X, y, mag, df_pop, OUTPUT_DIR,
                            extrap=extrap)

    # ── Step 7: Summary text ──────────────────────────────────────────────────
    summary_path = OUTPUT_DIR / 'q1_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("XGBOOST Q1 MODEL TRAINING SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Magnitude cut : {_mag_range()}\n")
        f.write(f"Features      : {len(SELECTED_GAIA_PARAMETERS)}\n")
        f.write(f"Training mode : {len(SEEDS)}-seed ensemble\n")
        if split_summary is not None:
            f.write(f"Train ratio   : {split_summary['ratio']:.1f}:1  (STAR:GALAXY)\n")
            f.write(
                f"Train / Val / Test : {split_summary['n_train']:,} / "
                f"{split_summary['n_val']:,} / {split_summary['n_test']:,}\n"
            )
        f.write("Baseline model: random_state=42\n\n")
        f.write("Parameters:\n")
        for i, p in enumerate(SELECTED_GAIA_PARAMETERS, 1):
            f.write(f"  {i:>2}. {p}\n")
        f.write("\nTest-set metrics:\n")
        for key, label in [
            ('accuracy', 'Accuracy'),
            ('galaxy_precision', 'Galaxy Precision'),
            ('galaxy_recall', 'Galaxy Recall'),
            ('galaxy_f1', 'Galaxy F1'),
            ('star_f1', 'Star F1'),
            ('roc_auc', 'ROC AUC'),
        ]:
            f.write(f"  {label:<16}: {metric_means[key]:.4f} ± {metric_stds[key]:.4f}\n")
    print(f"  -> {summary_path}")

    print("\n" + "=" * 70)
    print("Q1 model training complete!")
    print(f"All results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
