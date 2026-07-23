#!/usr/bin/env python3
"""
Q1 model per-magnitude-bin sensitivity analysis.

For each 0.25-mag bin centred on G = cut (28 bins from G=15.0 to G=21.75),
trains a fresh XGBoost model on sources within [cut - 0.125, cut + 0.125)
and evaluates Galaxy Precision, F1, and Recall on a held-out 20% test set
drawn from the same bin.

Unlike the cumulative scripts (analyze_q1_cutfaint/cutbright_sensitivity.py),
every bin is independent — only sources whose G falls within the narrow window
are used. This isolates how well the classifier works in each brightness regime
rather than how including/excluding a magnitude range affects overall performance.

Runs N_SEEDS random 80/20 stratified splits per bin (total: 28 × 10 = 280
model trainings).

Outputs (output/q1_perbin_sensitivity/):
  q1_perbin_performance.png
  q1_perbin_importance.png
  q1_perbin_cv_heatmap.png
  q1_perbin_cv_lines.png
  q1_perbin_cv_rows.png
  q1_perbin_rank_rows.png
  q1_perbin_imp_heatmap.png
  q1_perbin_imp_lines.png
  q1_perbin_imp_rows.png

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_q1_perbin_sensitivity.py
    PGUSER=siwei_zhao_2025 python src/analyze_q1_perbin_sensitivity.py --retrain
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
OUTPUT_DIR  = PROJECT_DIR / 'output' / 'q1_model_sensitivity' / 'perbin_sensitivity'
CACHE_FILE  = OUTPUT_DIR / 'q1_perbin_results.json'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

SEEDS     = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]
TEST_SIZE = 0.2
VAL_FRAC  = 0.125

# 0.25-mag steps from G=15.0 to G=21.75 — 28 bin centres
_edges   = np.arange(15.0, 22.01, 0.25)
MAG_BINS = [round(e, 2) for e in _edges[:-1]]   # bin centres
BIN_HALF = 0.125                                 # half-width ⇒ each bin spans ±0.125

# Reference lines on the performance plot
REF_LINES = [17.0, 18.5, 21.0]

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
MIN_GAL_IN_BIN = 5


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def query_q1(username):
    """Query full Q1 dataset (G >= 15); bin filter applied in run_seed."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT
            {gaia_cols},
            g_phot_g_mean_mag AS phot_g_mean_mag,
            CASE
                WHEN p_phz_classification = 1 THEN 'STAR'
                WHEN p_phz_classification = 2 THEN 'GALAXY'
            END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia
        WHERE p_phz_classification IN (1, 2)
          AND g_phot_g_mean_mag IS NOT NULL
          AND g_phot_g_mean_mag >= 15
        ORDER BY g_random_index
    """
    print("  Querying Q1 training data (G >= 15, STAR/GALAXY only)...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    n_star = (df['label'] == 'STAR').sum()
    n_gal  = (df['label'] == 'GALAXY').sum()
    print(f"  Retrieved {len(df):,}  STAR: {n_star:,}  GALAXY: {n_gal:,}")
    return df


def query_q1_population(username):
    """All Gaia-matched Q1 sources (G >= 15), magnitude + label only."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = """
        SELECT
            g_phot_g_mean_mag AS phot_g_mean_mag,
            CASE
                WHEN p_phz_classification = 1 THEN 'STAR'
                WHEN p_phz_classification = 2 THEN 'GALAXY'
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


# ---------------------------------------------------------------------------
# Per-bin population fractions
# ---------------------------------------------------------------------------

def compute_bin_fracs(df_train, df_pop):
    """Per-bin STAR/GALAXY fractions for each bin centre in MAG_BINS."""
    mag_t   = df_train['phot_g_mean_mag'].values
    label_t = df_train['label'].values
    mag_p   = df_pop['phot_g_mean_mag'].values
    label_p = df_pop['label'].values

    classified = {'total': [], 'n_gal': [], 'star_frac': [], 'gal_frac': []}
    all_src    = {'total': [], 'n_gal': [], 'star_frac': [], 'gal_frac': [], 'other_frac': []}

    for centre in MAG_BINS:
        lo, hi = centre - BIN_HALF, centre + BIN_HALF

        mask_t = (mag_t >= lo) & (mag_t < hi)
        tot_t  = mask_t.sum()
        n_star_t = (label_t[mask_t] == 'STAR').sum()
        n_gal_t  = (label_t[mask_t] == 'GALAXY').sum()
        classified['total'].append(tot_t)
        classified['n_gal'].append(n_gal_t)
        classified['star_frac'].append(n_star_t / tot_t if tot_t > 0 else np.nan)
        classified['gal_frac'].append(n_gal_t  / tot_t if tot_t > 0 else np.nan)

        mask_p = (mag_p >= lo) & (mag_p < hi)
        tot_p  = mask_p.sum()
        ns = (label_p[mask_p] == 'STAR').sum()
        ng = (label_p[mask_p] == 'GALAXY').sum()
        no = ((label_p[mask_p] == 'QSO_GC') | (label_p[mask_p] == 'UNCLASSIFIED')).sum()
        all_src['total'].append(tot_p)
        all_src['n_gal'].append(ng)
        all_src['star_frac'].append(ns / tot_p if tot_p > 0 else np.nan)
        all_src['gal_frac'].append(ng  / tot_p if tot_p > 0 else np.nan)
        all_src['other_frac'].append(no / tot_p if tot_p > 0 else np.nan)

    for d in (classified, all_src):
        for k in list(d.keys()):
            d[k] = np.array(d[k])

    return classified, all_src


# ---------------------------------------------------------------------------
# Training (one seed) — per-bin
# ---------------------------------------------------------------------------

def run_seed(df, seed):
    """For each bin, train on sources within [centre ± BIN_HALF) and evaluate."""
    df = df[df['label'].isin(['STAR', 'GALAXY'])].copy()

    metrics_per_bin = []
    n_per_bin       = []
    all_imps        = []

    for centre in MAG_BINS:
        lo, hi  = centre - BIN_HALF, centre + BIN_HALF
        df_bin  = df[(df['phot_g_mean_mag'] >= lo) & (df['phot_g_mean_mag'] < hi)]
        X       = df_bin[SELECTED_GAIA_PARAMETERS].values.astype(float)
        y       = df_bin['label'].map(CLASS_MAPPING).values
        n_tot   = len(X)
        n_gal   = int((y == 1).sum())
        n_star  = int((y == 0).sum())

        if n_tot < 50 or n_gal < MIN_GAL_IN_BIN or n_star < MIN_GAL_IN_BIN:
            metrics_per_bin.append({'precision': np.nan, 'recall': np.nan, 'f1': np.nan})
            n_per_bin.append({'n_train': 0, 'n_test': 0, 'n_gal_test': 0})
            all_imps.append(None)
            continue

        idx = np.arange(n_tot)
        idx_train, idx_test = train_test_split(
            idx, test_size=TEST_SIZE, stratify=y, random_state=seed)
        idx_tr, idx_val = train_test_split(
            idx_train, test_size=VAL_FRAC, stratify=y[idx_train], random_state=seed)

        if len(np.unique(y[idx_tr])) < 2:
            metrics_per_bin.append({'precision': np.nan, 'recall': np.nan, 'f1': np.nan})
            n_per_bin.append({'n_train': len(idx_tr), 'n_test': len(idx_test), 'n_gal_test': 0})
            all_imps.append(None)
            continue

        model = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
        model.fit(X[idx_tr], y[idx_tr],
                  eval_set=[(X[idx_val], y[idx_val])],
                  verbose=False)

        y_test     = y[idx_test]
        n_gal_test = int((y_test == 1).sum())
        n_per_bin.append({'n_train': len(idx_tr), 'n_test': len(idx_test),
                          'n_gal_test': n_gal_test})

        if n_gal_test < MIN_GAL_IN_BIN:
            metrics_per_bin.append({'precision': np.nan, 'recall': np.nan, 'f1': np.nan})
        else:
            p, r, f1, _ = precision_recall_fscore_support(
                y_test, model.predict(X[idx_test]),
                labels=[0, 1], zero_division=0)
            metrics_per_bin.append({'precision': p[1], 'recall': r[1], 'f1': f1[1]})

        all_imps.append(model.feature_importances_)

    return metrics_per_bin, all_imps, n_per_bin


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _shade_refs(ax, shade_edges=False):
    """Draw reference vertical lines and a y=0.5 guide.
    shade_edges=True fills left of G=17 and right of G=21 in lightyellow."""
    x = np.array(MAG_BINS)
    if shade_edges:
        ax.axvspan(x[0] - 0.15, 17.0, color='lightyellow', alpha=0.6, zorder=0)
        ax.axvspan(21.0, x[-1] + 0.15, color='lightyellow', alpha=0.6, zorder=0)
    ref_styles = {17.0: ('navy', '--'), 18.5: ('black', '--'), 21.0: ('darkred', '--')}
    for ref, (color, ls) in ref_styles.items():
        ax.axvline(x=ref, color=color, ls=ls, linewidth=1.5, alpha=0.8)
    ax.axhline(y=0.5, color='gray', ls=':', linewidth=1)
    ax.grid(alpha=0.3, axis='y')


def plot_bin_sensitivity(all_metrics, classified_fracs, all_fracs, output_dir):
    """Three-panel performance figure — per-bin variant."""
    n_bins  = len(MAG_BINS)
    n_seeds = len(all_metrics)
    x       = np.array(MAG_BINS)

    prec = np.array([[m['precision'] for m in s] for s in all_metrics])
    rec  = np.array([[m['recall']    for m in s] for s in all_metrics])
    f1   = np.array([[m['f1']        for m in s] for s in all_metrics])

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(14, 13),
        gridspec_kw={'height_ratios': [3, 1.2, 1.2]}, sharex=True)

    # --- top panel: metrics ---
    _shade_refs(ax_top, shade_edges=True)
    metric_handles = []
    for label, arr, color, marker in [
        ('Galaxy Precision', prec, 'steelblue',     'o'),
        ('Galaxy F1',        f1,   'mediumseagreen', 's'),
        ('Galaxy Recall',    rec,  'coral',           '^'),
    ]:
        mu  = np.nanmean(arr, axis=0)
        std = np.nanstd(arr,  axis=0)
        line, = ax_top.plot(x, mu, marker=marker, color=color, linewidth=2,
                            markersize=8, label=label, zorder=3)
        ax_top.fill_between(x, mu - std, mu + std, alpha=0.18, color=color, zorder=2)
        metric_handles.append(line)

    ref_handles = [
        Line2D([0], [0], color='navy',    ls='--', linewidth=1.5, alpha=0.8, label='G = 17'),
        Line2D([0], [0], color='black',   ls='--', linewidth=1.5, alpha=0.8, label='G = 18.5'),
        Line2D([0], [0], color='darkred', ls='--', linewidth=1.5, alpha=0.8, label='G = 21'),
    ]
    ax_top.legend(handles=metric_handles + ref_handles,
                  fontsize=10, loc='lower left', ncol=1)
    ax_top.set_ylabel('Metric value', fontsize=12, fontweight='bold')
    ax_top.set_ylim(-0.05, 1.05)

    # --- mid panel: classified fracs per bin ---
    _shade_refs(ax_mid, shade_edges=True)
    gf_c    = classified_fracs['gal_frac']
    sf_c    = classified_fracs['star_frac']
    n_gal_c = classified_fracs['n_gal']
    ax_mid.bar(x, gf_c, width=0.22, color='orchid', alpha=0.25, zorder=1)
    ax_mid.plot(x, sf_c, color='coral',   linewidth=2, zorder=4, label='Star',   alpha=0.5)
    ax_mid.plot(x, gf_c, color='#9B30B0', linewidth=2, zorder=4, label='Galaxy')
    ax_mid.set_ylabel('Fraction', fontsize=11, fontweight='bold')
    ax_mid.set_ylim(0, 1.25)
    ax_mid.legend(fontsize=9, loc='upper left', ncol=2)
    ax_mid.set_title('Classified only — STAR + GALAXY  (per bin)',
                     fontsize=10, loc='left', pad=3)
    for i in range(0, n_bins, 2):
        if n_gal_c[i] > 0:
            y_ann = (gf_c[i] if np.isfinite(gf_c[i]) else 0) + 0.05
            ax_mid.text(x[i], y_ann, f'{int(n_gal_c[i]):,}',
                        ha='center', fontsize=7, color='#333333', zorder=10)

    # --- bot panel: all sources per bin ---
    _shade_refs(ax_bot, shade_edges=True)
    gf_a    = all_fracs['gal_frac']
    sf_a    = all_fracs['star_frac']
    of_a    = all_fracs['other_frac']
    n_gal_a = all_fracs['n_gal']
    ax_bot.bar(x, gf_a, width=0.22, color='orchid', alpha=0.25, zorder=1)
    ax_bot.plot(x, sf_a, color='coral',   linewidth=2, zorder=4, label='Star',         alpha=0.5)
    ax_bot.plot(x, gf_a, color='#9B30B0', linewidth=2, zorder=4, label='Galaxy')
    ax_bot.plot(x, of_a, color='#8B4513', linewidth=2, zorder=4, linestyle='--',
                label='Unclassified', alpha=0.5)
    ax_bot.set_ylabel('Fraction', fontsize=11, fontweight='bold')
    ax_bot.set_ylim(0, 1.25)
    ax_bot.legend(fontsize=9, loc='upper left', ncol=3)
    ax_bot.set_title('All Gaia-matched Q1 sources  (incl. unclassified, per bin)',
                     fontsize=10, loc='left', pad=3)
    for i in range(0, n_bins, 2):
        if n_gal_a[i] > 0:
            y_ann = (gf_a[i] if np.isfinite(gf_a[i]) else 0) + 0.05
            ax_bot.text(x[i], y_ann, f'{int(n_gal_a[i]):,}',
                        ha='center', fontsize=7, color='#333333', zorder=10)

    tick_vals = [c for c in MAG_BINS if c == round(c)]
    ax_bot.set_xlim(x[0] - 0.2, x[-1] + 0.2)
    ax_bot.set_xticks(tick_vals)
    ax_bot.set_xticklabels([f'G≈{c:.0f}' for c in tick_vals], fontsize=10)
    ax_bot.set_xlabel(
        f'Gaia G bin centre  (model trained on sources within ±{BIN_HALF} mag)',
        fontsize=13, fontweight='bold', labelpad=13)

    plt.suptitle(
        f'Q1 model — Galaxy metrics vs magnitude bin  '
        f'(mean ± 1σ, {n_seeds} random 80/20 splits)',
        fontsize=13, fontweight='bold', y=0.99)
    fig.subplots_adjust(top=0.90)
    plt.tight_layout()

    out = output_dir / 'q1_perbin_performance.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_feature_importance(all_importances, output_dir):
    """Horizontal bar chart: mean ± std across all bins × seeds."""
    imp = np.array(all_importances)
    mu  = imp.mean(axis=0)
    std = imp.std(axis=0)
    cv  = std / (mu + 1e-9)

    order = np.argsort(mu)[::-1]
    feats = [SELECTED_GAIA_PARAMETERS[i] for i in order]
    mu_s  = mu[order];  std_s = std[order];  cv_s = cv[order]

    n_feat = len(feats)
    cv_abs_max = 2.0
    norm   = plt.Normalize(vmin=0, vmax=cv_abs_max)
    colors = plt.cm.RdYlGn_r(norm(cv_s))

    fig, ax = plt.subplots(figsize=(11, max(8, n_feat * 0.38)))
    ax.barh(range(n_feat), mu_s, xerr=std_s, color=colors, alpha=0.85, capsize=4,
            error_kw={'elinewidth': 1.3, 'ecolor': 'black'})
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels([f'#{i+1}  {f}' for i, f in enumerate(feats)], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.5)
    ax.set_xticks(np.arange(0, 0.51, 0.1))
    ax.set_xlabel('Feature importance  (mean ± 1σ across all bins × seeds)',
                  fontsize=11, fontweight='bold', labelpad=10)
    ax.grid(alpha=0.3, axis='x')

    for i, (m_v, s_v, c_v) in enumerate(zip(mu_s, std_s, cv_s)):
        ax.text(m_v + s_v + 0.0005, i, f'CV={c_v:.2f}',
                va='center', fontsize=7.5, color='#333333')

    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label('CV = std / mean', fontsize=9)
    cbar.set_ticks(np.arange(0, cv_abs_max + 0.1, 0.2))
    cbar.ax.text(0.4, 1.04, 'unstable', transform=cbar.ax.transAxes,
                 fontsize=8, va='bottom', ha='left', color='darkred')
    cbar.ax.text(0.4, -0.05, 'stable', transform=cbar.ax.transAxes,
                 fontsize=8, va='top', ha='left', color='darkgreen')

    plt.suptitle(
        f'Q1 per-bin model — feature importance  '
        f'(mean across all bins × seeds, {len(all_importances)} models)',
        fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()

    out = output_dir / 'q1_perbin_importance.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Importance array helpers
# ---------------------------------------------------------------------------

def _build_imps_array(all_imps_by_seed):
    n_seeds = len(all_imps_by_seed)
    n_bins  = len(MAG_BINS)
    n_feats = len(SELECTED_GAIA_PARAMETERS)
    imps = np.full((n_seeds, n_bins, n_feats), np.nan)
    for s, seed_imps in enumerate(all_imps_by_seed):
        for c, imp in enumerate(seed_imps):
            if imp is not None:
                imps[s, c, :] = imp
    return imps


def _ref_vlines(ax, bins_axis=True, shade_edges=False):
    """Draw three reference vlines; bins_axis=True means x is mag centre.
    shade_edges=True fills left of G=17 and right of G=21 in lightyellow."""
    x = np.array(MAG_BINS)
    if shade_edges:
        if bins_axis:
            ax.axvspan(x[0] - 0.15, 17.0, color='lightyellow', alpha=0.6, zorder=0)
            ax.axvspan(21.0, x[-1] + 0.15, color='lightyellow', alpha=0.6, zorder=0)
        else:
            idx_17 = int(np.argmin(np.abs(x - 17.0)))
            idx_21 = int(np.argmin(np.abs(x - 21.0)))
            ax.axvspan(-0.5, idx_17, color='lightyellow', alpha=0.6, zorder=0)
            ax.axvspan(idx_21, len(MAG_BINS) - 0.5, color='lightyellow', alpha=0.6, zorder=0)
    ref_styles = {17.0: ('navy', '--'), 18.5: ('black', '--'), 21.0: ('darkred', '--')}
    for ref, (color, ls) in ref_styles.items():
        if bins_axis:
            ax.axvline(x=ref, color=color, ls=ls, linewidth=1.5, alpha=0.8)
        else:
            idx = int(np.argmin(np.abs(x - ref)))
            ax.axvline(x=idx, color=color, ls=ls, linewidth=1.5, alpha=0.8)


def plot_cv_vs_bin_heatmap(all_imps_by_seed, output_dir):
    """Heatmap: feature importance CV (std/mean across seeds) per bin."""
    n_seeds = len(all_imps_by_seed)
    n_feats = len(SELECTED_GAIA_PARAMETERS)
    imps    = _build_imps_array(all_imps_by_seed)
    mu      = np.nanmean(imps, axis=0)
    std     = np.nanstd(imps,  axis=0)
    cv      = std / (mu + 1e-9)

    order       = np.argsort(np.nanmean(mu, axis=0))[::-1]
    feat_labels = [f'#{i+1}  {SELECTED_GAIA_PARAMETERS[j]}' for i, j in enumerate(order)]
    cv_sorted   = cv[:, order].T

    fig, ax = plt.subplots(figsize=(14, max(6, n_feats * 0.38)))
    im = ax.imshow(cv_sorted, aspect='auto', cmap='Blues',
                   vmin=0, vmax=0.6, origin='upper',
                   extent=[-0.5, len(MAG_BINS) - 0.5, n_feats - 0.5, -0.5])

    tick_idx = [i for i, c in enumerate(MAG_BINS) if c == round(c)]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f'G≈{MAG_BINS[i]:.0f}' for i in tick_idx], fontsize=9)
    ax.set_xlabel(f'Gaia G bin centre  (±{BIN_HALF} mag)',
                  fontsize=11, fontweight='bold', labelpad=8)
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(feat_labels, fontsize=9)
    _ref_vlines(ax, bins_axis=False, shade_edges=True)

    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.set_label('CV = std / mean  (across seeds)', fontsize=9)

    plt.suptitle(f'Q1 per-bin model — feature importance CV vs bin  ({n_seeds} seeds)',
                 fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout()
    out = output_dir / 'q1_perbin_cv_heatmap.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_cv_vs_bin_lines(all_imps_by_seed, output_dir):
    """Line plot: CV per feature vs bin centre."""
    n_seeds = len(all_imps_by_seed)
    n_feats = len(SELECTED_GAIA_PARAMETERS)
    x       = np.array(MAG_BINS)
    imps    = _build_imps_array(all_imps_by_seed)
    mu   = np.nanmean(imps, axis=0)
    std  = np.nanstd(imps,  axis=0)
    cv   = std / (mu + 1e-9)
    order  = np.argsort(np.nanmean(mu, axis=0))[::-1]
    colors = plt.cm.tab20(np.linspace(0, 1, n_feats))

    fig, ax = plt.subplots(figsize=(14, 7))
    for rank, feat_idx in enumerate(order):
        cv_line = cv[:, feat_idx]
        lw    = 2.0 if rank < 5 else 1.0
        alpha = 1.0 if rank < 10 else 0.55
        ax.plot(x, cv_line, color=colors[rank], linewidth=lw, alpha=alpha,
                label=f'#{rank+1} {SELECTED_GAIA_PARAMETERS[feat_idx]}')

    _ref_vlines(ax, shade_edges=True)
    tick_vals = [c for c in MAG_BINS if c == round(c)]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([f'G≈{c:.0f}' for c in tick_vals], fontsize=10)
    ax.set_xlim(x[0] - 0.2, x[-1] + 0.2)
    ax.set_xlabel(f'Gaia G bin centre  (±{BIN_HALF} mag)',
                  fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('CV = std / mean  (across seeds)', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis='y')
    ax.legend(fontsize=7.5, loc='upper right', ncol=2, framealpha=0.9)

    plt.suptitle(f'Q1 per-bin model — feature importance CV vs bin  ({n_seeds} seeds)',
                 fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout()
    out = output_dir / 'q1_perbin_cv_lines.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_cv_vs_bin_rows(all_imps_by_seed, output_dir):
    """Multi-row: one subplot per feature, CV vs bin centre."""
    n_seeds = len(all_imps_by_seed)
    n_feats = len(SELECTED_GAIA_PARAMETERS)
    x       = np.array(MAG_BINS)
    imps    = _build_imps_array(all_imps_by_seed)
    mu   = np.nanmean(imps, axis=0)
    std  = np.nanstd(imps,  axis=0)
    cv   = std / (mu + 1e-9)
    order = np.argsort(np.nanmean(mu, axis=0))[::-1]

    fig, axes = plt.subplots(n_feats, 1, figsize=(14, n_feats * 0.55 + 1.5),
                             sharex=True, gridspec_kw={'hspace': 0.0})
    for row, feat_idx in enumerate(order):
        ax = axes[row]
        _ref_vlines(ax, shade_edges=True)
        ax.plot(x, cv[:, feat_idx], color='steelblue', linewidth=1.2, zorder=3)
        ax.set_xlim(x[0] - 0.2, x[-1] + 0.2)
        ax.set_ylim(0, 1)
        ax.set_yticks([1])
        ax.set_yticklabels(['1'], fontsize=6)
        ax.set_ylabel(f'#{row+1}  {SELECTED_GAIA_PARAMETERS[feat_idx]}',
                      rotation=0, ha='right', va='center', fontsize=7.5)
        ax.grid(alpha=0.25, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if row < n_feats - 1:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='both', bottom=False)

    tick_vals = [c for c in MAG_BINS if c == round(c)]
    axes[-1].set_xticks(tick_vals)
    axes[-1].set_xticklabels([f'G≈{c:.0f}' for c in tick_vals], fontsize=9)
    axes[-1].set_xlabel(f'Gaia G bin centre  (±{BIN_HALF} mag)',
                        fontsize=11, fontweight='bold', labelpad=8)
    plt.suptitle(
        f'Q1 per-bin model — feature importance CV vs bin  ({n_seeds} seeds)',
        fontsize=12, fontweight='bold', y=0.99)
    fig.subplots_adjust(left=0.28, top=0.97)
    out = output_dir / 'q1_perbin_cv_rows.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_rank_vs_bin_rows(all_imps_by_seed, output_dir):
    """Multi-row: importance rank vs bin centre."""
    n_seeds = len(all_imps_by_seed)
    n_bins  = len(MAG_BINS)
    n_feats = len(SELECTED_GAIA_PARAMETERS)
    x       = np.array(MAG_BINS)
    imps    = _build_imps_array(all_imps_by_seed)

    ranks = np.full_like(imps, np.nan)
    for s in range(n_seeds):
        for c in range(n_bins):
            row = imps[s, c, :]
            if not np.all(np.isnan(row)):
                valid = ~np.isnan(row)
                r = np.full(n_feats, np.nan)
                r[valid] = n_feats - np.argsort(np.argsort(row[valid]))
                ranks[s, c, :] = r

    mu_rank  = np.nanmean(ranks, axis=0)
    std_rank = np.nanstd(ranks,  axis=0)
    mu_imp   = np.nanmean(imps,  axis=0)
    order    = np.argsort(np.nanmean(mu_imp, axis=0))[::-1]

    fig, axes = plt.subplots(n_feats, 1, figsize=(14, n_feats * 0.55 + 1.5),
                             sharex=True, gridspec_kw={'hspace': 0.0})
    for row, feat_idx in enumerate(order):
        ax = axes[row]
        rank_line = mu_rank[:, feat_idx]
        rank_std  = std_rank[:, feat_idx]

        _ref_vlines(ax, shade_edges=True)
        ax.fill_between(x, rank_line - rank_std, rank_line + rank_std,
                        alpha=0.2, color='steelblue', zorder=2)
        ax.plot(x, rank_line, color='steelblue', linewidth=1.2, zorder=3)
        ax.set_xlim(x[0] - 0.2, x[-1] + 0.2)
        ax.set_ylim(0.5, n_feats + 0.5)
        ax.invert_yaxis()
        ax.set_yticks([])
        ax.text(0.01, 0.95, '1', transform=ax.transAxes, fontsize=5.5,
                va='top', ha='left', color='#666', clip_on=True)
        ax.text(0.01, 0.05, f'{n_feats}', transform=ax.transAxes, fontsize=5.5,
                va='bottom', ha='left', color='#666', clip_on=True)
        ax.set_ylabel(f'#{row+1}  {SELECTED_GAIA_PARAMETERS[feat_idx]}',
                      rotation=0, ha='right', va='center', fontsize=7.5)
        ax.grid(alpha=0.25, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if row < n_feats - 1:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='both', bottom=False)

    tick_vals = [c for c in MAG_BINS if c == round(c)]
    axes[-1].set_xticks(tick_vals)
    axes[-1].set_xticklabels([f'G≈{c:.0f}' for c in tick_vals], fontsize=9)
    axes[-1].set_xlabel(f'Gaia G bin centre  (±{BIN_HALF} mag)',
                        fontsize=11, fontweight='bold', labelpad=8)
    plt.suptitle(
        f'Q1 per-bin model — feature importance rank vs bin  '
        f'({n_seeds} seeds, mean ± 1σ)',
        fontsize=12, fontweight='bold', y=0.99)
    fig.subplots_adjust(left=0.28, top=0.97)
    out = output_dir / 'q1_perbin_rank_rows.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_imp_vs_bin_heatmap(all_imps_by_seed, output_dir):
    """Heatmap: mean feature importance (across seeds) per bin."""
    n_seeds = len(all_imps_by_seed)
    n_feats = len(SELECTED_GAIA_PARAMETERS)
    imps    = _build_imps_array(all_imps_by_seed)
    mu      = np.nanmean(imps, axis=0)

    order       = np.argsort(np.nanmean(mu, axis=0))[::-1]
    feat_labels = [f'#{i+1}  {SELECTED_GAIA_PARAMETERS[j]}' for i, j in enumerate(order)]
    imp_sorted  = mu[:, order].T

    fig, ax = plt.subplots(figsize=(14, max(6, n_feats * 0.38)))
    im = ax.imshow(imp_sorted, aspect='auto', cmap='Blues',
                   vmin=0, vmax=0.2, origin='upper',
                   extent=[-0.5, len(MAG_BINS) - 0.5, n_feats - 0.5, -0.5])

    tick_idx = [i for i, c in enumerate(MAG_BINS) if c == round(c)]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f'G≈{MAG_BINS[i]:.0f}' for i in tick_idx], fontsize=9)
    ax.set_xlabel(f'Gaia G bin centre  (±{BIN_HALF} mag)',
                  fontsize=11, fontweight='bold', labelpad=8)
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(feat_labels, fontsize=9)
    _ref_vlines(ax, bins_axis=False, shade_edges=True)

    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.set_label('Mean feature importance  (across seeds)', fontsize=9)
    cbar.set_ticks(np.arange(0, 0.21, 0.05))

    plt.suptitle(
        f'Q1 per-bin model — feature importance vs bin  ({n_seeds} seeds)',
        fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout()
    out = output_dir / 'q1_perbin_imp_heatmap.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_imp_vs_bin_lines(all_imps_by_seed, output_dir):
    """Line plot: mean feature importance (± 1σ) per feature vs bin centre."""
    n_seeds = len(all_imps_by_seed)
    n_feats = len(SELECTED_GAIA_PARAMETERS)
    x       = np.array(MAG_BINS)
    imps    = _build_imps_array(all_imps_by_seed)
    mu   = np.nanmean(imps, axis=0)
    std  = np.nanstd(imps,  axis=0)
    order  = np.argsort(np.nanmean(mu, axis=0))[::-1]
    colors = plt.cm.tab20(np.linspace(0, 1, n_feats))

    fig, ax = plt.subplots(figsize=(14, 7))
    for rank, feat_idx in enumerate(order):
        mu_line  = mu[:, feat_idx]
        std_line = std[:, feat_idx]
        lw    = 2.0 if rank < 5 else 1.0
        alpha = 1.0 if rank < 10 else 0.55
        ax.plot(x, mu_line, color=colors[rank], linewidth=lw, alpha=alpha,
                label=f'#{rank+1} {SELECTED_GAIA_PARAMETERS[feat_idx]}')
        ax.fill_between(x, mu_line - std_line, mu_line + std_line,
                        alpha=0.08, color=colors[rank])

    _ref_vlines(ax, shade_edges=True)
    tick_vals = [c for c in MAG_BINS if c == round(c)]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([f'G≈{c:.0f}' for c in tick_vals], fontsize=10)
    ax.set_xlim(x[0] - 0.2, x[-1] + 0.2)
    ax.set_xlabel(f'Gaia G bin centre  (±{BIN_HALF} mag)',
                  fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Feature importance  (mean ± 1σ across seeds)',
                  fontsize=11, fontweight='bold')
    ax.set_ylim(0, 0.6)
    ax.grid(alpha=0.3, axis='y')
    ax.legend(fontsize=7.5, loc='upper left', ncol=2, framealpha=0.9)

    plt.suptitle(
        f'Q1 per-bin model — feature importance vs bin  ({n_seeds} seeds)',
        fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout()
    out = output_dir / 'q1_perbin_imp_lines.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_imp_vs_bin_rows(all_imps_by_seed, output_dir):
    """Multi-row: mean importance ± 1σ per feature vs bin centre."""
    n_seeds = len(all_imps_by_seed)
    n_feats = len(SELECTED_GAIA_PARAMETERS)
    x       = np.array(MAG_BINS)
    imps    = _build_imps_array(all_imps_by_seed)
    mu   = np.nanmean(imps, axis=0)
    std  = np.nanstd(imps,  axis=0)
    order = np.argsort(np.nanmean(mu, axis=0))[::-1]

    fig, axes = plt.subplots(n_feats, 1, figsize=(14, n_feats * 0.55 + 1.5),
                             sharex=True, gridspec_kw={'hspace': 0.0})
    for row, feat_idx in enumerate(order):
        ax = axes[row]
        mu_line  = mu[:, feat_idx]
        std_line = std[:, feat_idx]
        _ref_vlines(ax, shade_edges=True)
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
        ax.set_ylabel(f'#{row+1}  {SELECTED_GAIA_PARAMETERS[feat_idx]}',
                      rotation=0, ha='right', va='center', fontsize=7.5)
        ax.grid(alpha=0.25, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if row < n_feats - 1:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='both', bottom=False)

    tick_vals = [c for c in MAG_BINS if c == round(c)]
    axes[-1].set_xticks(tick_vals)
    axes[-1].set_xticklabels([f'G≈{c:.0f}' for c in tick_vals], fontsize=9)
    axes[-1].set_xlabel(f'Gaia G bin centre  (±{BIN_HALF} mag)',
                        fontsize=11, fontweight='bold', labelpad=8)
    plt.suptitle(
        f'Q1 per-bin model — feature importance vs bin  '
        f'({n_seeds} seeds, mean ± 1σ)',
        fontsize=12, fontweight='bold', y=0.99)
    fig.subplots_adjust(left=0.28, top=0.97)
    out = output_dir / 'q1_perbin_imp_rows.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _to_serializable(obj):
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
    if isinstance(obj, dict):
        return {k: _from_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_serializable(v) for v in obj]
    if obj is None:
        return np.nan
    return obj


def save_cache(all_metrics, all_imps_by_seed, classified_fracs, all_fracs):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'all_metrics':      _to_serializable(all_metrics),
        'all_imps_by_seed': [[imp.tolist() if imp is not None else None
                               for imp in seed_imps]
                              for seed_imps in all_imps_by_seed],
        'classified_fracs': _to_serializable({k: v.tolist() for k, v in classified_fracs.items()}),
        'all_fracs':        _to_serializable({k: v.tolist() for k, v in all_fracs.items()}),
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
    all_fracs        = {k: np.array(_from_serializable(v))
                        for k, v in payload['all_fracs'].items()}
    return all_metrics, all_importances, all_imps_by_seed, classified_fracs, all_fracs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    retrain  = '--retrain' in sys.argv
    username = os.getenv('PGUSER')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Q1 model: per-magnitude-bin sensitivity  (±0.125 mag bins)")
    print(f"  {len(MAG_BINS)} bins x {len(SEEDS)} seeds = {len(MAG_BINS)*len(SEEDS)} models")
    print(f"  Ref lines: G = {REF_LINES}")
    print(f"  Cache:  {CACHE_FILE}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 65)

    if not retrain:
        cached = load_cache()
    else:
        cached = None
        print("  --retrain flag set: ignoring cache.")

    if cached is not None:
        all_metrics, all_importances, all_imps_by_seed, classified_fracs, all_fracs = cached
        print(f"  Loaded {len(all_metrics)} seeds from cache — skipping training.")
    else:
        if not username:
            print("ERROR: PGUSER not set"); return 1

        print("\n[1] Querying Q1 training data...")
        df = query_q1(username)

        print("\n[1b] Querying Q1 population (for fraction panels)...")
        df_pop = query_q1_population(username)

        all_metrics      = []
        all_importances  = []
        all_imps_by_seed = []

        print(f"\n[2] Training {len(MAG_BINS) * len(SEEDS)} models (per ±0.125 mag bin)...")
        for i, seed in enumerate(SEEDS):
            print(f"\n  === Seed {seed}  ({i+1}/{len(SEEDS)}) ===")
            metrics, all_imps, n_per_bin = run_seed(df, seed)
            all_metrics.append(metrics)
            all_importances.extend([imp for imp in all_imps if imp is not None])
            all_imps_by_seed.append(all_imps)

            for centre, m, nb in zip(MAG_BINS, metrics, n_per_bin):
                if np.isnan(m['f1']):
                    print(f"    G≈{centre:5.2f}:  train={nb['n_train']:>5,}  — skipped")
                else:
                    print(f"    G≈{centre:5.2f}:  train={nb['n_train']:>5,}  "
                          f"test={nb['n_test']:>4,}  gal={nb['n_gal_test']:>3,}  "
                          f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

        print("\n[3] Computing per-bin population fractions...")
        classified_fracs, all_fracs = compute_bin_fracs(df, df_pop)

        print("\n[3b] Saving cache...")
        save_cache(all_metrics, all_imps_by_seed, classified_fracs, all_fracs)

    print("\n[4] Plotting performance...")
    plot_bin_sensitivity(all_metrics, classified_fracs, all_fracs, OUTPUT_DIR)

    if all_importances:
        print(f"\n[5] Plotting feature importance ({len(all_importances)} models)...")
        plot_feature_importance(all_importances, OUTPUT_DIR)

    print("\n[6] Plotting CV heatmap...")
    plot_cv_vs_bin_heatmap(all_imps_by_seed, OUTPUT_DIR)

    print("\n[7] Plotting CV lines...")
    plot_cv_vs_bin_lines(all_imps_by_seed, OUTPUT_DIR)

    print("\n[8] Plotting CV rows...")
    plot_cv_vs_bin_rows(all_imps_by_seed, OUTPUT_DIR)

    print("\n[9] Plotting rank rows...")
    plot_rank_vs_bin_rows(all_imps_by_seed, OUTPUT_DIR)

    print("\n[10] Plotting importance heatmap...")
    plot_imp_vs_bin_heatmap(all_imps_by_seed, OUTPUT_DIR)

    print("\n[11] Plotting importance lines...")
    plot_imp_vs_bin_lines(all_imps_by_seed, OUTPUT_DIR)

    print("\n[12] Plotting importance rows...")
    plot_imp_vs_bin_rows(all_imps_by_seed, OUTPUT_DIR)

    print(f"\nAll saved to: {OUTPUT_DIR}")
    print("=" * 65)
    return 0


if __name__ == '__main__':
    sys.exit(main())
