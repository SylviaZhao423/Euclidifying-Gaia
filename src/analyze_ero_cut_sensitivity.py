#!/usr/bin/env python3
"""
ERO model cut-based magnitude sensitivity analysis.

For each Gaia G magnitude cut (G >= cut, 28 values from 15.0 to 21.75),
trains a fresh XGBoost model on ERO data and evaluates Galaxy Precision,
F1, and Recall on the held-out 20% test set from the same cut.

This shows how model performance changes as the training sample is
restricted to progressively fainter magnitude limits — each point on the
plot corresponds to a model trained exclusively on sources with G >= that
cut value.

Labels derived from e_v_class_star: >= 0.65 -> STAR, < 0.65 -> GALAXY.

Runs N_SEEDS random 80/20 stratified splits per cut (total: 28 x 10 = 280
model trainings). Feature importance is reported at the reference cut
G >= 18.5 (matching the combined training script).

Outputs:
  output/cut_sensitivity/ero_cut_performance.png
  output/cut_sensitivity/ero_cut_importance.png

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_ero_cut_sensitivity.py
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'output' / 'cut_sensitivity'
CACHE_FILE  = OUTPUT_DIR / 'ero_results.json'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

SEEDS     = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]
TEST_SIZE = 0.2    # 20% held-out test
VAL_FRAC  = 0.125  # fraction of training used for XGB early-stopping (≈10% of total)

CLASS_STAR_THRESH = 0.65   # e_v_class_star >= threshold -> STAR

# 0.25-mag cut values from G=15 to G=21.75 — 28 values
_edges   = np.arange(15.0, 22.01, 0.25)
MAG_CUTS = [round(e, 2) for e in _edges[:-1]]
REF_CUT  = 18.5   # reference cut used in combined training script

XGB_PARAMS = {
    'objective':             'binary:logistic',
    'max_depth':             8,
    'learning_rate':         0.1,
    'n_estimators':          200,
    'subsample':             0.8,
    'colsample_bytree':      0.8,
    'tree_method':           'hist',
    'eval_metric':           'logloss',
    'early_stopping_rounds': 20,
}

CLASS_MAPPING  = {'STAR': 0, 'GALAXY': 1}
MIN_GAL_IN_CUT = 5   # skip cut if fewer galaxies in test set


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query_ero(username):
    """Query ERO Gaia features for all G >= 15; per-cut filter applied in run_seed."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT
            {gaia_cols},
            CASE
                WHEN e_v_class_star >= {CLASS_STAR_THRESH} THEN 'STAR'
                WHEN e_v_class_star <  {CLASS_STAR_THRESH} THEN 'GALAXY'
            END AS label
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_class_star IS NOT NULL
          AND g_phot_g_mean_mag IS NOT NULL
          AND g_phot_g_mean_mag >= 15
        ORDER BY g_source_id
    """
    print("  Querying ERO training data (G >= 15, class_star IS NOT NULL)...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    n_star = (df['label'] == 'STAR').sum()
    n_gal  = (df['label'] == 'GALAXY').sum()
    print(f"  Retrieved {len(df):,}  STAR: {n_star:,}  GALAXY: {n_gal:,}")
    return df


# ---------------------------------------------------------------------------
# Cumulative population fractions
# ---------------------------------------------------------------------------

def compute_cumulative_fracs(df):
    """Cumulative (G >= cut) STAR/GALAXY fractions for each MAG_CUT."""
    mag   = df['phot_g_mean_mag'].values
    label = df['label'].values

    result = {'total': [], 'n_gal': [], 'star_frac': [], 'gal_frac': []}

    for cut in MAG_CUTS:
        mask  = mag >= cut
        tot   = mask.sum()
        n_star = (label[mask] == 'STAR').sum()
        n_gal  = (label[mask] == 'GALAXY').sum()
        result['total'].append(tot)
        result['n_gal'].append(n_gal)
        result['star_frac'].append(n_star / tot if tot > 0 else np.nan)
        result['gal_frac'].append(n_gal  / tot if tot > 0 else np.nan)

    for k in list(result.keys()):
        result[k] = np.array(result[k])

    return result


# ---------------------------------------------------------------------------
# Cut-based training + evaluation (one seed)
# ---------------------------------------------------------------------------

def run_seed(df, seed):
    """For each magnitude cut, train a fresh model and evaluate on its test set.

    Returns
    -------
    metrics_per_cut : list of dict  (precision, recall, f1 — galaxy class)
    all_imps        : list of 1-D arrays  feature importances for each valid cut
    n_per_cut       : list of dict  (n_train, n_test, n_gal_test per cut)
    """
    metrics_per_cut = []
    n_per_cut       = []
    all_imps        = []

    for cut in MAG_CUTS:
        df_cut = df[df['phot_g_mean_mag'] >= cut]
        X      = df_cut[SELECTED_GAIA_PARAMETERS].values.astype(float)
        y      = df_cut['label'].map(CLASS_MAPPING).values
        n_tot  = len(X)
        n_gal  = int((y == 1).sum())

        if n_tot < 50 or n_gal < MIN_GAL_IN_CUT:
            metrics_per_cut.append({'precision': np.nan, 'recall': np.nan, 'f1': np.nan})
            n_per_cut.append({'n_train': 0, 'n_test': 0, 'n_gal_test': 0})
            all_imps.append(None)
            continue

        idx = np.arange(n_tot)
        idx_train, idx_test = train_test_split(
            idx, test_size=TEST_SIZE, stratify=y, random_state=seed)
        idx_tr, idx_val = train_test_split(
            idx_train, test_size=VAL_FRAC, stratify=y[idx_train], random_state=seed)

        model = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
        model.fit(X[idx_tr], y[idx_tr],
                  eval_set=[(X[idx_val], y[idx_val])],
                  verbose=False)

        y_test     = y[idx_test]
        n_gal_test = int((y_test == 1).sum())
        n_per_cut.append({'n_train': len(idx_tr), 'n_test': len(idx_test),
                          'n_gal_test': n_gal_test})

        if n_gal_test < MIN_GAL_IN_CUT:
            metrics_per_cut.append({'precision': np.nan, 'recall': np.nan, 'f1': np.nan})
        else:
            p, r, f1, _ = precision_recall_fscore_support(
                y_test, model.predict(X[idx_test]),
                labels=[0, 1], zero_division=0)
            metrics_per_cut.append({'precision': p[1], 'recall': r[1], 'f1': f1[1]})

        all_imps.append(model.feature_importances_)

    return metrics_per_cut, all_imps, n_per_cut  # all_imps has len(MAG_CUTS) entries


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_cut_sensitivity(all_metrics, classified_fracs, output_dir):
    """Two-panel figure:
    Row 1: Galaxy Precision, F1, Recall vs training G cut (mean +/- 1 sigma).
    Row 2: Cumulative STAR/GALAXY fractions (class_star threshold).
    Yellow shading marks cuts < REF_CUT (18.5).
    """
    n_cuts  = len(MAG_CUTS)
    n_seeds = len(all_metrics)
    x       = np.array(MAG_CUTS)

    prec = np.array([[m['precision'] for m in s] for s in all_metrics])
    rec  = np.array([[m['recall']    for m in s] for s in all_metrics])
    f1   = np.array([[m['f1']        for m in s] for s in all_metrics])

    fig, (ax_top, ax_mid) = plt.subplots(
        2, 1, figsize=(14, 10),
        gridspec_kw={'height_ratios': [3, 1.2]}, sharex=True)

    def _shade(ax):
        ax.axvspan(x[0] - 0.15, REF_CUT, color='lightyellow', alpha=0.6, zorder=0)
        ax.axvline(x=REF_CUT, color='black', ls='--', linewidth=1.5, alpha=0.8)
        ax.axhline(y=0.5, color='gray', ls=':', linewidth=1)
        ax.grid(alpha=0.3, axis='y')

    # ── Row 1: metrics ───────────────────────────────────────────────────────
    _shade(ax_top)

    metric_handles = []
    for label, arr, color, marker in [
        ('Galaxy Precision', prec, 'steelblue',     'o'),
        ('Galaxy F1',        f1,   'mediumseagreen', 's'),
        ('Galaxy Recall',    rec,  'coral',          '^'),
    ]:
        mu  = np.nanmean(arr, axis=0)
        std = np.nanstd(arr,  axis=0)
        line, = ax_top.plot(x, mu, marker=marker, color=color, linewidth=2,
                            markersize=8, label=label, zorder=3)
        ax_top.fill_between(x, mu - std, mu + std, alpha=0.18, color=color, zorder=2)
        metric_handles.append(line)

    boundary_handle = Line2D([0], [0], color='black', ls='--', linewidth=1.5, alpha=0.8,
                             label=f'G = {REF_CUT} boundary')
    ax_top.legend(handles=metric_handles + [boundary_handle],
                  fontsize=10, loc='lower left', ncol=1)
    ax_top.set_ylabel('Metric value', fontsize=12, fontweight='bold')
    ax_top.set_ylim(-0.05, 1.05)

    # ── Row 2: classified-only fractions ─────────────────────────────────────
    _shade(ax_mid)
    gf_c = classified_fracs['gal_frac']
    sf_c = classified_fracs['star_frac']
    ax_mid.bar(x, gf_c, width=0.22, color='orchid', alpha=0.25, zorder=1)
    ax_mid.plot(x, sf_c, color='coral',   linewidth=2, zorder=4, label='Star',   alpha=0.5)
    ax_mid.plot(x, gf_c, color='#9B30B0', linewidth=2, zorder=4, label='Galaxy')
    ax_mid.set_ylabel('Fraction', fontsize=11, fontweight='bold')
    ax_mid.set_ylim(0, 1.25)
    ax_mid.legend(fontsize=9, loc='upper right', ncol=2)
    ax_mid.set_title(
        f'STAR + GALAXY fractions  (class_star threshold = {CLASS_STAR_THRESH}, cumulative G >= cut)',
        fontsize=10, loc='left', pad=3)

    n_gal_c = classified_fracs['n_gal']
    for i in range(0, n_cuts, 2):
        if n_gal_c[i] > 0:
            y_ann = (gf_c[i] if np.isfinite(gf_c[i]) else 0) + 0.05
            ax_mid.text(x[i], y_ann, f'{int(n_gal_c[i]):,}',
                        ha='center', fontsize=7, color='#333333', zorder=10)

    tick_vals = [c for c in MAG_CUTS if c == round(c)]
    ax_mid.set_xlim(x[0] - 0.2, x[-1] + 0.2)
    ax_mid.set_xticks(tick_vals)
    ax_mid.set_xticklabels([f'G>={c:.0f}' for c in tick_vals], fontsize=10)
    ax_mid.set_xlabel('Gaia G training cut  (model trained on G >= x)',
                      fontsize=13, fontweight='bold', labelpad=13)

    plt.suptitle(
        f'ERO model — Galaxy metrics vs training magnitude cut  '
        f'(mean \u00b1 1\u03c3, {n_seeds} random 80/20 splits)',
        fontsize=13, fontweight='bold', y=0.99)
    fig.subplots_adjust(top=0.90)
    plt.tight_layout()

    out = output_dir / 'ero_performance.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_feature_importance(all_importances, output_dir):
    """Horizontal bar chart: mean ± std feature importance averaged across all cuts × seeds."""
    imp   = np.array(all_importances)
    mu    = imp.mean(axis=0)
    std   = imp.std(axis=0)
    cv    = std / (mu + 1e-9)

    order = np.argsort(mu)[::-1]
    feats = [SELECTED_GAIA_PARAMETERS[i] for i in order]
    mu_s  = mu[order]
    std_s = std[order]
    cv_s  = cv[order]

    n_feat  = len(feats)
    cv_norm = cv_s / (cv_s.max() + 1e-9)
    colors  = plt.cm.RdYlGn_r(cv_norm)

    fig, ax = plt.subplots(figsize=(11, max(8, n_feat * 0.38)))
    ax.barh(range(n_feat), mu_s, xerr=std_s,
            color=colors, alpha=0.85, capsize=4,
            error_kw={'elinewidth': 1.3, 'ecolor': 'black'})
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels([f'#{i+1}  {f}' for i, f in enumerate(feats)], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Feature importance  (mean \u00b1 1\u03c3 across all cuts \u00d7 seeds)', fontsize=11,
                  fontweight='bold', labelpad=10)
    ax.grid(alpha=0.3, axis='x')
    ax.set_xlim(0, 0.45)

    for i, (m_v, s_v, c_v) in enumerate(zip(mu_s, std_s, cv_s)):
        ax.text(m_v + s_v + 0.0005, i, f'CV={c_v:.2f}',
                va='center', fontsize=7.5, color='#333333')

    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r',
                                norm=plt.Normalize(vmin=0, vmax=1.8))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label('CV = std / mean', fontsize=9)
    cbar.set_ticks(np.arange(0, 1.8, 0.2))
    cbar.ax.text(0.4, 1.04, 'unstable', transform=cbar.ax.transAxes,
                 fontsize=8, va='bottom', ha='left', color='darkred')
    cbar.ax.text(0.4, -0.05, 'stable', transform=cbar.ax.transAxes,
                 fontsize=8, va='top', ha='left', color='darkgreen')

    n_models = len(all_importances)
    plt.suptitle(
        f'ERO model feature importance  (mean across all cuts \u00d7 seeds, {n_models} models)',
        fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()

    out = output_dir / 'ero_importance.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_cv_vs_cut_lines(all_imps_by_seed, output_dir):
    """Line plot: CV (std/mean across seeds) per feature vs magnitude cut.
    Features sorted and coloured by mean importance rank.
    """
    n_seeds = len(all_imps_by_seed)
    n_cuts  = len(MAG_CUTS)
    n_feats = len(SELECTED_GAIA_PARAMETERS)
    x       = np.array(MAG_CUTS)

    imps = np.full((n_seeds, n_cuts, n_feats), np.nan)
    for s, seed_imps in enumerate(all_imps_by_seed):
        for c, imp in enumerate(seed_imps):
            if imp is not None:
                imps[s, c, :] = imp

    mu  = np.nanmean(imps, axis=0)   # (n_cuts, n_feats)
    std = np.nanstd(imps,  axis=0)
    cv  = std / (mu + 1e-9)          # (n_cuts, n_feats)

    # Sort features by overall mean importance (descending)
    order  = np.argsort(np.nanmean(mu, axis=0))[::-1]
    colors = plt.cm.tab20(np.linspace(0, 1, n_feats))

    fig, ax = plt.subplots(figsize=(14, 7))

    for rank, feat_idx in enumerate(order):
        feat_name = SELECTED_GAIA_PARAMETERS[feat_idx]
        cv_line   = cv[:, feat_idx]
        lw        = 2.0 if rank < 5 else 1.0
        alpha     = 1.0 if rank < 10 else 0.55
        ax.plot(x, cv_line, color=colors[rank], linewidth=lw, alpha=alpha,
                label=f'#{rank+1} {feat_name}')

    ax.axvline(x=REF_CUT, color='black', ls='--', linewidth=1.5, alpha=0.8,
               label=f'G = {REF_CUT} boundary')
    ax.axvspan(x[0] - 0.15, REF_CUT, color='lightyellow', alpha=0.4, zorder=0)

    tick_vals = [c for c in MAG_CUTS if c == round(c)]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([f'G>={c:.0f}' for c in tick_vals], fontsize=10)
    ax.set_xlim(x[0] - 0.2, 21.2)
    ax.set_xlabel('Gaia G training cut  (model trained on G >= x)',
                  fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('CV = std / mean  (across seeds)', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis='y')

    ax.legend(fontsize=7.5, loc='upper left', ncol=2, framealpha=0.9)

    plt.suptitle(
        f'ERO model \u2014 feature importance CV vs training cut  ({n_seeds} seeds)',
        fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout()

    out = output_dir / 'ero_cv_lines.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_cv_vs_cut_rows(all_imps_by_seed, output_dir):
    """Multi-row line plot: one subplot row per feature, CV vs magnitude cut.
    Features ordered top-to-bottom by mean importance (matching heatmap layout).
    Each row shows a line instead of colour blocks.
    """
    n_seeds = len(all_imps_by_seed)
    n_cuts  = len(MAG_CUTS)
    n_feats = len(SELECTED_GAIA_PARAMETERS)
    x       = np.array(MAG_CUTS)

    imps = np.full((n_seeds, n_cuts, n_feats), np.nan)
    for s, seed_imps in enumerate(all_imps_by_seed):
        for c, imp in enumerate(seed_imps):
            if imp is not None:
                imps[s, c, :] = imp

    mu  = np.nanmean(imps, axis=0)   # (n_cuts, n_feats)
    std = np.nanstd(imps,  axis=0)
    cv  = std / (mu + 1e-9)          # (n_cuts, n_feats)

    # Sort features by overall mean importance — same order as heatmap (#1 at top)
    order = np.argsort(np.nanmean(mu, axis=0))[::-1]

    row_h = 0.55
    fig_h = n_feats * row_h + 1.5
    fig, axes = plt.subplots(
        n_feats, 1,
        figsize=(14, fig_h),
        sharex=True,
        gridspec_kw={'hspace': 0.0}
    )

    for row, feat_idx in enumerate(order):
        ax        = axes[row]
        feat_name = SELECTED_GAIA_PARAMETERS[feat_idx]
        cv_line   = cv[:, feat_idx]

        ax.axvspan(x[0] - 0.15, REF_CUT, color='lightyellow', alpha=0.5, zorder=0)
        ax.axvline(x=REF_CUT, color='black', ls='--', linewidth=0.9, alpha=0.7)
        ax.plot(x, cv_line, color='steelblue', linewidth=1.2, zorder=3)
        ax.set_xlim(x[0] - 0.2, 21.2)
        ax.set_ylim(0, 1)
        ax.set_yticks([1])
        ax.set_yticklabels(['1'], fontsize=6)

        # Feature label on left as y-axis label
        ax.set_ylabel(f'#{row+1}  {feat_name}', rotation=0, ha='right', va='center',
                      fontsize=7.5)

        ax.grid(alpha=0.25, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if row < n_feats - 1:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='both', bottom=False)

    tick_vals = [c for c in MAG_CUTS if c == round(c)]
    axes[-1].set_xticks(tick_vals)
    axes[-1].set_xticklabels([f'G>={c:.0f}' for c in tick_vals], fontsize=9)
    axes[-1].set_xlabel('Gaia G training cut  (model trained on G >= x)',
                        fontsize=11, fontweight='bold', labelpad=8)

    plt.suptitle(
        f'ERO model \u2014 feature importance CV vs training cut  ({n_seeds} seeds)',
        fontsize=12, fontweight='bold', y=0.99)
    fig.subplots_adjust(left=0.28, top=0.97)

    out = output_dir / 'ero_cv_rows.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def _build_imps_array(all_imps_by_seed):
    """Build (n_seeds, n_cuts, n_feats) importance array from cache structure."""
    n_seeds = len(all_imps_by_seed)
    n_cuts  = len(MAG_CUTS)
    n_feats = len(SELECTED_GAIA_PARAMETERS)
    imps = np.full((n_seeds, n_cuts, n_feats), np.nan)
    for s, seed_imps in enumerate(all_imps_by_seed):
        for c, imp in enumerate(seed_imps):
            if imp is not None:
                imps[s, c, :] = imp
    return imps


def plot_imp_vs_cut_heatmap(all_imps_by_seed, output_dir):
    """Heatmap: mean feature importance (across seeds) per magnitude cut."""
    n_seeds = len(all_imps_by_seed)
    n_feats = len(SELECTED_GAIA_PARAMETERS)
    imps    = _build_imps_array(all_imps_by_seed)

    mu          = np.nanmean(imps, axis=0)
    order       = np.argsort(np.nanmean(mu, axis=0))[::-1]
    feat_labels = [f'#{i+1}  {SELECTED_GAIA_PARAMETERS[j]}' for i, j in enumerate(order)]
    imp_sorted  = mu[:, order].T

    vmax = np.nanpercentile(imp_sorted, 95)

    fig, ax = plt.subplots(figsize=(14, max(6, n_feats * 0.38)))
    im = ax.imshow(imp_sorted, aspect='auto', cmap='Blues',
                   vmin=0, vmax=0.2, origin='upper',
                   extent=[-0.5, len(MAG_CUTS) - 0.5, n_feats - 0.5, -0.5])

    tick_idx = [i for i, c in enumerate(MAG_CUTS) if c == round(c)]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f'G>={MAG_CUTS[i]:.0f}' for i in tick_idx], fontsize=9)
    ax.set_xlabel('Gaia G training cut  (model trained on G >= x)',
                  fontsize=11, fontweight='bold', labelpad=8)
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(feat_labels, fontsize=9)

    ref_idx = MAG_CUTS.index(REF_CUT)
    ax.axvline(x=ref_idx, color='black', ls='--', linewidth=1.5, alpha=0.8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.set_label('Mean feature importance  (across seeds)', fontsize=9)
    cbar.set_ticks(np.arange(0, 0.21, 0.05))

    plt.suptitle(
        f'ERO model \u2014 feature importance vs training cut  ({n_seeds} seeds)',
        fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout()

    out = output_dir / 'ero_imp_heatmap.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_imp_vs_cut_lines(all_imps_by_seed, output_dir):
    """Line plot: mean feature importance (± 1σ) per feature vs magnitude cut."""
    n_seeds = len(all_imps_by_seed)
    n_feats = len(SELECTED_GAIA_PARAMETERS)
    x       = np.array(MAG_CUTS)
    imps    = _build_imps_array(all_imps_by_seed)

    mu     = np.nanmean(imps, axis=0)
    std    = np.nanstd(imps,  axis=0)
    order  = np.argsort(np.nanmean(mu, axis=0))[::-1]
    colors = plt.cm.tab20(np.linspace(0, 1, n_feats))

    fig, ax = plt.subplots(figsize=(14, 7))

    for rank, feat_idx in enumerate(order):
        feat_name = SELECTED_GAIA_PARAMETERS[feat_idx]
        mu_line   = mu[:, feat_idx]
        std_line  = std[:, feat_idx]
        lw        = 2.0 if rank < 5 else 1.0
        alpha     = 1.0 if rank < 10 else 0.55
        ax.plot(x, mu_line, color=colors[rank], linewidth=lw, alpha=alpha,
                label=f'#{rank+1} {feat_name}')
        ax.fill_between(x, mu_line - std_line, mu_line + std_line,
                        alpha=0.08, color=colors[rank])

    ax.axvline(x=REF_CUT, color='black', ls='--', linewidth=1.5, alpha=0.8,
               label=f'G = {REF_CUT} boundary')
    ax.axvspan(x[0] - 0.15, REF_CUT, color='lightyellow', alpha=0.4, zorder=0)

    tick_vals = [c for c in MAG_CUTS if c == round(c)]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([f'G>={c:.0f}' for c in tick_vals], fontsize=10)
    ax.set_xlim(x[0] - 0.2, 21.2)
    ax.set_xlabel('Gaia G training cut  (model trained on G >= x)',
                  fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Feature importance  (mean \u00b1 1\u03c3 across seeds)',
                  fontsize=11, fontweight='bold')
    ax.set_ylim(0, 0.6)
    ax.grid(alpha=0.3, axis='y')
    ax.legend(fontsize=7.5, loc='upper left', ncol=2, framealpha=0.9)

    plt.suptitle(
        f'ERO model \u2014 feature importance vs training cut  ({n_seeds} seeds)',
        fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout()

    out = output_dir / 'ero_imp_lines.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_imp_vs_cut_rows(all_imps_by_seed, output_dir):
    """Multi-row line plot: one subplot row per feature, mean importance (± 1σ) vs magnitude cut."""
    n_seeds = len(all_imps_by_seed)
    n_feats = len(SELECTED_GAIA_PARAMETERS)
    x       = np.array(MAG_CUTS)
    imps    = _build_imps_array(all_imps_by_seed)

    mu    = np.nanmean(imps, axis=0)
    std   = np.nanstd(imps,  axis=0)
    order = np.argsort(np.nanmean(mu, axis=0))[::-1]

    row_h = 0.55
    fig_h = n_feats * row_h + 1.5
    fig, axes = plt.subplots(
        n_feats, 1,
        figsize=(14, fig_h),
        sharex=True,
        gridspec_kw={'hspace': 0.0}
    )

    for row, feat_idx in enumerate(order):
        ax        = axes[row]
        feat_name = SELECTED_GAIA_PARAMETERS[feat_idx]
        mu_line   = mu[:, feat_idx]
        std_line  = std[:, feat_idx]

        ax.axvspan(x[0] - 0.15, REF_CUT, color='lightyellow', alpha=0.5, zorder=0)
        ax.axvline(x=REF_CUT, color='black', ls='--', linewidth=0.9, alpha=0.7)
        ax.fill_between(x, mu_line - std_line, mu_line + std_line,
                        alpha=0.2, color='steelblue', zorder=2)
        ax.plot(x, mu_line, color='steelblue', linewidth=1.2, zorder=3)

        ax.set_xlim(x[0] - 0.2, x[-1] + 0.2)
        ax.set_ylim(0, None)

        ymax = np.nanmax(mu_line + std_line)
        if np.isfinite(ymax) and ymax > 0:
            ax.set_yticks([round(ymax, 2)])
            ax.set_yticklabels([f'{ymax:.2f}'], fontsize=6)
        else:
            ax.set_yticks([])

        ax.set_ylabel(f'#{row+1}  {feat_name}', rotation=0, ha='right', va='center',
                      fontsize=7.5)
        ax.grid(alpha=0.25, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if row < n_feats - 1:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='both', bottom=False)

    tick_vals = [c for c in MAG_CUTS if c == round(c)]
    axes[-1].set_xticks(tick_vals)
    axes[-1].set_xticklabels([f'G>={c:.0f}' for c in tick_vals], fontsize=9)
    axes[-1].set_xlabel('Gaia G training cut  (model trained on G >= x)',
                        fontsize=11, fontweight='bold', labelpad=8)

    plt.suptitle(
        f'ERO model \u2014 feature importance vs training cut  ({n_seeds} seeds, mean \u00b1 1\u03c3)',
        fontsize=12, fontweight='bold', y=0.99)
    fig.subplots_adjust(left=0.28, top=0.97)

    out = output_dir / 'ero_imp_rows.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_rank_vs_cut_rows(all_imps_by_seed, output_dir):
    """Multi-row line plot: one subplot row per feature, importance rank vs magnitude cut.
    Features ordered top-to-bottom by mean importance (matching heatmap layout).
    Each row shows mean rank ± 1σ across seeds. Rank 1 = most important.
    Y-axis is inverted so rank 1 appears at the top of each row.
    """
    n_seeds = len(all_imps_by_seed)
    n_cuts  = len(MAG_CUTS)
    n_feats = len(SELECTED_GAIA_PARAMETERS)
    x       = np.array(MAG_CUTS)

    imps = np.full((n_seeds, n_cuts, n_feats), np.nan)
    for s, seed_imps in enumerate(all_imps_by_seed):
        for c, imp in enumerate(seed_imps):
            if imp is not None:
                imps[s, c, :] = imp

    # Compute per-seed per-cut rank (1 = highest importance)
    ranks = np.full_like(imps, np.nan)
    for s in range(n_seeds):
        for c in range(n_cuts):
            row = imps[s, c, :]
            if not np.all(np.isnan(row)):
                valid = ~np.isnan(row)
                r = np.full(n_feats, np.nan)
                r[valid] = n_feats - np.argsort(np.argsort(row[valid]))  # 1 = highest
                ranks[s, c, :] = r

    mu_rank  = np.nanmean(ranks, axis=0)   # (n_cuts, n_feats)
    std_rank = np.nanstd(ranks,  axis=0)

    # Sort features by overall mean importance — same order as heatmap
    mu_imp = np.nanmean(imps, axis=0)
    order  = np.argsort(np.nanmean(mu_imp, axis=0))[::-1]

    row_h = 0.55
    fig_h = n_feats * row_h + 1.5
    fig, axes = plt.subplots(
        n_feats, 1,
        figsize=(14, fig_h),
        sharex=True,
        gridspec_kw={'hspace': 0.0}
    )

    for row, feat_idx in enumerate(order):
        ax          = axes[row]
        feat_name   = SELECTED_GAIA_PARAMETERS[feat_idx]
        rank_line   = mu_rank[:, feat_idx]
        rank_std    = std_rank[:, feat_idx]

        ax.axvspan(x[0] - 0.15, REF_CUT, color='lightyellow', alpha=0.5, zorder=0)
        ax.axvline(x=REF_CUT, color='black', ls='--', linewidth=0.9, alpha=0.7)
        ax.fill_between(x, rank_line - rank_std, rank_line + rank_std,
                        alpha=0.2, color='steelblue', zorder=2)
        ax.plot(x, rank_line, color='steelblue', linewidth=1.2, zorder=3)

        ax.set_xlim(x[0] - 0.2, x[-1] + 0.2)
        ax.set_ylim(0.5, n_feats + 0.5)   # rank 1 at top (visually inverted)
        ax.invert_yaxis()
        ax.set_yticks([])
        # Scale labels clipped inside axes — won't bleed into adjacent rows
        ax.text(0.01, 0.95, '1', transform=ax.transAxes, fontsize=5.5,
                va='top', ha='left', color='#666', clip_on=True)
        ax.text(0.01, 0.05, f'{n_feats}', transform=ax.transAxes, fontsize=5.5,
                va='bottom', ha='left', color='#666', clip_on=True)

        ax.set_ylabel(f'#{row+1}  {feat_name}', rotation=0, ha='right', va='center',
                      fontsize=7.5)

        ax.grid(alpha=0.25, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if row < n_feats - 1:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='both', bottom=False)

    tick_vals = [c for c in MAG_CUTS if c == round(c)]
    axes[-1].set_xticks(tick_vals)
    axes[-1].set_xticklabels([f'G>={c:.0f}' for c in tick_vals], fontsize=9)
    axes[-1].set_xlabel('Gaia G training cut  (model trained on G >= x)',
                        fontsize=11, fontweight='bold', labelpad=8)

    plt.suptitle(
        f'ERO model \u2014 feature importance rank vs training cut  ({n_seeds} seeds, mean \u00b1 1\u03c3)',
        fontsize=12, fontweight='bold', y=0.99)
    fig.subplots_adjust(left=0.28, top=0.97)

    out = output_dir / 'ero_rank_rows.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _to_serializable(obj):
    """Recursively convert numpy types and NaN to JSON-safe types."""
    if isinstance(obj, np.ndarray):
        return [_to_serializable(v) for v in obj.tolist()]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


def _from_serializable(obj):
    """Convert None back to NaN in nested structures."""
    if isinstance(obj, dict):
        return {k: _from_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_serializable(v) for v in obj]
    if obj is None:
        return np.nan
    return obj


def plot_cv_vs_cut(all_imps_by_seed, output_dir):
    """Heatmap: feature importance CV (std/mean across seeds) per magnitude cut.
    Features sorted by mean importance (highest first, matching bar chart).
    """
    n_seeds = len(all_imps_by_seed)
    n_cuts  = len(MAG_CUTS)
    n_feats = len(SELECTED_GAIA_PARAMETERS)

    # Build (n_seeds, n_cuts, n_feats) array; NaN for skipped cuts
    imps = np.full((n_seeds, n_cuts, n_feats), np.nan)
    for s, seed_imps in enumerate(all_imps_by_seed):
        for c, imp in enumerate(seed_imps):
            if imp is not None:
                imps[s, c, :] = imp

    mu  = np.nanmean(imps, axis=0)   # (n_cuts, n_feats)
    std = np.nanstd(imps,  axis=0)   # (n_cuts, n_feats)
    cv  = std / (mu + 1e-9)          # (n_cuts, n_feats)

    # Sort features by overall mean importance — same order as bar chart
    order       = np.argsort(np.nanmean(mu, axis=0))[::-1]
    feat_labels = [f'#{i+1}  {SELECTED_GAIA_PARAMETERS[j]}' for i, j in enumerate(order)]
    cv_sorted   = cv[:, order].T     # (n_feats, n_cuts) for imshow

    vmax = np.nanpercentile(cv_sorted, 95)

    fig, ax = plt.subplots(figsize=(14, max(6, n_feats * 0.38)))
    im = ax.imshow(cv_sorted, aspect='auto', cmap='Blues',
                   vmin=0, vmax=0.6, origin='upper',
                   extent=[-0.5, n_cuts - 0.5, n_feats - 0.5, -0.5])

    # X-axis: whole-magnitude ticks
    tick_idx = [i for i, c in enumerate(MAG_CUTS) if c == round(c)]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f'G>={MAG_CUTS[i]:.0f}' for i in tick_idx], fontsize=9)
    ax.set_xlabel('Gaia G training cut  (model trained on G >= x)',
                  fontsize=11, fontweight='bold', labelpad=8)

    # Y-axis: feature labels
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(feat_labels, fontsize=9)

    # Vertical line at REF_CUT
    ref_idx = MAG_CUTS.index(REF_CUT)
    ax.axvline(x=ref_idx, color='black', ls='--', linewidth=1.5, alpha=0.8,
               label=f'G = {REF_CUT} boundary')
    # ax.legend(fontsize=9, loc='lower right')

    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.set_label('CV = std / mean  (across seeds)', fontsize=9)

    plt.suptitle(
        f'ERO model \u2014 feature importance CV vs training cut  ({n_seeds} seeds)',
        fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout()

    out = output_dir / 'ero_cv_heatmap.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def save_cache(all_metrics, all_imps_by_seed, classified_fracs):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'all_metrics':      _to_serializable(all_metrics),
        'all_imps_by_seed': [[imp.tolist() if imp is not None else None for imp in seed_imps]
                              for seed_imps in all_imps_by_seed],
        'classified_fracs': _to_serializable(
            {k: v.tolist() for k, v in classified_fracs.items()}),
    }
    with open(CACHE_FILE, 'w') as f:
        json.dump(payload, f)
    print(f"  Cache saved: {CACHE_FILE}")


def load_cache():
    if not CACHE_FILE.exists():
        return None
    print(f"  Loading cached results from {CACHE_FILE}")
    with open(CACHE_FILE) as f:
        payload = json.load(f)
    all_metrics      = _from_serializable(payload['all_metrics'])
    all_imps_by_seed = [[np.array(imp) if imp is not None else None for imp in seed_imps]
                        for seed_imps in payload['all_imps_by_seed']]
    all_importances  = [imp for seed_imps in all_imps_by_seed
                        for imp in seed_imps if imp is not None]
    classified_fracs = {k: np.array(_from_serializable(v))
                        for k, v in payload['classified_fracs'].items()}
    return all_metrics, all_importances, all_imps_by_seed, classified_fracs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    retrain  = '--retrain' in sys.argv
    username = os.getenv('PGUSER')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"ERO model: cut-based magnitude sensitivity")
    print(f"  {len(MAG_CUTS)} cuts x {len(SEEDS)} seeds = {len(MAG_CUTS) * len(SEEDS)} models")
    print("=" * 65)

    if not retrain:
        cached = load_cache()
    else:
        cached = None
        print("  --retrain flag set: ignoring cache.")

    if cached is not None:
        all_metrics, all_importances, all_imps_by_seed, classified_fracs = cached
        print(f"  Loaded {len(all_metrics)} seeds from cache — skipping training.")
    else:
        if not username:
            print("ERROR: PGUSER not set")
            return 1

        print("\n[1] Querying ERO training data (G >= 15)...")
        df = query_ero(username)

        all_metrics      = []
        all_importances  = []
        all_imps_by_seed = []

        print(f"\n[2] Training {len(MAG_CUTS) * len(SEEDS)} models...")
        for i, seed in enumerate(SEEDS):
            print(f"\n  === Seed {seed}  ({i+1}/{len(SEEDS)}) ===")
            metrics, all_imps, n_per_cut = run_seed(df, seed)
            all_metrics.append(metrics)
            all_importances.extend([imp for imp in all_imps if imp is not None])
            all_imps_by_seed.append(all_imps)

            for cut, m, nc in zip(MAG_CUTS, metrics, n_per_cut):
                if np.isnan(m['f1']):
                    print(f"    G>={cut:5.2f}:  train={nc['n_train']:>6,}  "
                          f"test={nc['n_test']:>5,}  — skipped")
                else:
                    print(f"    G>={cut:5.2f}:  train={nc['n_train']:>6,}  "
                          f"test={nc['n_test']:>5,}  gal={nc['n_gal_test']:>4,}  "
                          f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

        print("\n[3] Computing cumulative population fractions...")
        classified_fracs = compute_cumulative_fracs(df)

        print("\n[3b] Saving results cache...")
        save_cache(all_metrics, all_imps_by_seed, classified_fracs)

    print("\n[4] Plotting cut sensitivity...")
    plot_cut_sensitivity(all_metrics, classified_fracs, OUTPUT_DIR)

    if all_importances:
        print(f"\n[5] Plotting feature importance ({len(all_importances)} models)...")
        plot_feature_importance(all_importances, OUTPUT_DIR)
    else:
        print(f"\n[5] No feature importances collected.")

    print(f"\n[6] Plotting CV heatmap vs training cut...")
    plot_cv_vs_cut(all_imps_by_seed, OUTPUT_DIR)

    print(f"\n[7] Plotting CV line plot vs training cut...")
    plot_cv_vs_cut_lines(all_imps_by_seed, OUTPUT_DIR)

    print(f"\n[8] Plotting CV multi-row line plot vs training cut...")
    plot_cv_vs_cut_rows(all_imps_by_seed, OUTPUT_DIR)

    print(f"\n[9] Plotting rank multi-row line plot vs training cut...")
    plot_rank_vs_cut_rows(all_imps_by_seed, OUTPUT_DIR)

    print(f"\n[10] Plotting importance heatmap vs training cut...")
    plot_imp_vs_cut_heatmap(all_imps_by_seed, OUTPUT_DIR)

    print(f"\n[11] Plotting importance line plot vs training cut...")
    plot_imp_vs_cut_lines(all_imps_by_seed, OUTPUT_DIR)

    print(f"\n[12] Plotting importance multi-row line plot vs training cut...")
    plot_imp_vs_cut_rows(all_imps_by_seed, OUTPUT_DIR)

    print(f"\nAll saved to: {OUTPUT_DIR}")
    print("=" * 65)
    return 0


if __name__ == '__main__':
    sys.exit(main())
