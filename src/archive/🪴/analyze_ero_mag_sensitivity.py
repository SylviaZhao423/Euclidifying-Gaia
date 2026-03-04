#!/usr/bin/env python3
"""
ERO model magnitude sensitivity analysis.

Trains the ERO XGBoost model N_SEEDS times with different random
stratified train/test splits (80/20) on ERO data (G ≥ 15, no faint cut),
then evaluates Galaxy Precision, F1, and Recall in Gaia G magnitude bins
from G=15 to G=22.

Labels derived from e_v_class_star: >= 0.65 → STAR, < 0.65 → GALAXY.

Also plots feature importance consistency across seeds to check whether
the model learns stable physical signal or noisy/arbitrary features.

Outputs:
  output/mag_sensitivity_ero/ero_mag_sensitivity.png
  output/mag_sensitivity_ero/ero_feature_importance.png

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_ero_magnitude_sensitivity.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'output' / 'mag_sensitivity'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

SEEDS     = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]
TEST_SIZE = 0.2     # 20% held-out test
VAL_FRAC  = 0.125   # fraction of training used as XGB validation (≈ 10% of total)

CLASS_STAR_THRESH = 0.65   # e_v_class_star >= threshold → STAR

# 0.25-mag bins from G=15 to G=22 (G=18.5 falls on a bin boundary)
_edges   = np.arange(15.0, 22.01, 0.25)
MAG_BINS = [(round(_edges[i], 2), round(_edges[i+1], 2))
            for i in range(len(_edges) - 1)]
BIN_LABELS = [f'{lo:.2f}' for lo, _ in MAG_BINS]

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
MIN_GAL_IN_BIN = 5   # skip bin if fewer than this many galaxies in test set


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query_ero(username):
    """Query ERO Gaia features for training (e_v_class_star IS NOT NULL)."""
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
    print("  Querying ERO training data (G ≥ 15, class_star IS NOT NULL)...")
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
# Population fractions
# ---------------------------------------------------------------------------

def _bin_counts(mag, label, classes):
    """Count sources per MAG_BIN per class label."""
    counts = {c: np.zeros(len(MAG_BINS), dtype=float) for c in classes}
    totals = np.zeros(len(MAG_BINS), dtype=float)
    for i, (lo, hi) in enumerate(MAG_BINS):
        mask = (mag >= lo) & (mag < hi)
        for c in classes:
            counts[c][i] = (label[mask] == c).sum()
        totals[i] = mask.sum()
    return counts, totals


def compute_classified_fracs(df_train):
    """Return per-bin STAR/GALAXY fractions from the training sample."""
    mag   = df_train['phot_g_mean_mag'].values
    label = df_train['label'].values
    c, tot = _bin_counts(mag, label, ['STAR', 'GALAXY'])
    return {
        'total':     tot,
        'star_frac': np.where(tot > 0, c['STAR']   / tot, np.nan),
        'gal_frac':  np.where(tot > 0, c['GALAXY'] / tot, np.nan),
    }


# ---------------------------------------------------------------------------
# Single-seed training + evaluation
# ---------------------------------------------------------------------------

def run_seed(df, seed):
    """Train one ERO model, evaluate per magnitude bin.

    Returns
    -------
    metrics_per_bin : list of dict  (precision, recall, f1 — galaxy class)
    importances     : 1-D array     (feature_importances_)
    n_per_bin       : list of dict  (total, galaxy counts in each test bin)
    """
    X   = df[SELECTED_GAIA_PARAMETERS].values.astype(float)
    y   = df['label'].map(CLASS_MAPPING).values
    mag = df['phot_g_mean_mag'].values

    idx = np.arange(len(X))

    idx_train, idx_test = train_test_split(
        idx, test_size=TEST_SIZE, stratify=y, random_state=seed)
    idx_tr, idx_val = train_test_split(
        idx_train, test_size=VAL_FRAC, stratify=y[idx_train], random_state=seed)

    model = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed)
    model.fit(X[idx_tr], y[idx_tr],
              eval_set=[(X[idx_val], y[idx_val])],
              verbose=False)

    X_test   = X[idx_test]
    y_test   = y[idx_test]
    mag_test = mag[idx_test]

    metrics_per_bin = []
    n_per_bin       = []
    for lo, hi in MAG_BINS:
        mask  = (mag_test >= lo) & (mag_test < hi)
        n_tot = int(mask.sum())
        n_gal = int((y_test[mask] == 1).sum())
        if n_tot < 10 or n_gal < MIN_GAL_IN_BIN:
            metrics_per_bin.append({'precision': np.nan, 'recall': np.nan, 'f1': np.nan})
        else:
            p, r, f1, _ = precision_recall_fscore_support(
                y_test[mask], model.predict(X_test[mask]),
                labels=[0, 1], zero_division=0)
            metrics_per_bin.append({'precision': p[1], 'recall': r[1], 'f1': f1[1]})
        n_per_bin.append({'total': n_tot, 'galaxy': n_gal})

    return metrics_per_bin, model.feature_importances_, n_per_bin


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_mag_sensitivity(all_metrics, all_n, classified_fracs, output_dir):
    """Two-panel figure:
    Row 1: Galaxy Precision, F1, Recall vs G magnitude bin (mean ± 1σ).
    Row 2: Population fractions — STAR + GALAXY (class_star threshold).
    Yellow shading marks G < 18.5 (ERO training cut in combined script).
    """
    n_bins  = len(MAG_BINS)
    n_seeds = len(all_metrics)

    prec = np.array([[m['precision'] for m in s] for s in all_metrics])
    rec  = np.array([[m['recall']    for m in s] for s in all_metrics])
    f1   = np.array([[m['f1']        for m in s] for s in all_metrics])

    x = np.arange(n_bins)
    g185_idx = (18.5 - MAG_BINS[0][0]) / (MAG_BINS[0][1] - MAG_BINS[0][0]) - 0.5

    fig, (ax_top, ax_mid) = plt.subplots(
        2, 1, figsize=(14, 10),
        gridspec_kw={'height_ratios': [3, 1.2]}, sharex=True)

    def _shade(ax):
        ax.axvspan(-0.5, g185_idx, color='lightyellow', alpha=0.6, zorder=0)
        ax.axvline(x=g185_idx, color='black', ls='--', linewidth=1.5, alpha=0.8)
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

    from matplotlib.lines import Line2D
    boundary_handle = Line2D([0], [0], color='black', ls='--', linewidth=1.5, alpha=0.8,
                             label='G = 18.5 boundary')
    ax_top.legend(handles=metric_handles + [boundary_handle],
                  fontsize=10, loc='lower left', ncol=1)
    ax_top.set_ylabel('Metric value', fontsize=12, fontweight='bold')
    ax_top.set_ylim(-0.05, 1.05)

    # ── Row 2: classified-only fractions ─────────────────────────────────────
    _shade(ax_mid)
    gf_c = classified_fracs['gal_frac']
    sf_c = classified_fracs['star_frac']
    ax_mid.bar(x, gf_c, color='orchid', alpha=0.25, zorder=1)
    ax_mid.plot(x, sf_c, color='coral',   linewidth=2, zorder=4, label='Star',   alpha=0.5)
    ax_mid.plot(x, gf_c, color='#9B30B0', linewidth=2, zorder=4, label='Galaxy')
    ax_mid.set_ylabel('Fraction', fontsize=11, fontweight='bold')
    ax_mid.set_ylim(0, 1.25)
    ax_mid.legend(fontsize=9, loc='upper right', ncol=2)
    ax_mid.set_title(
        f'STAR + GALAXY fractions  (class_star threshold = {CLASS_STAR_THRESH})',
        fontsize=10, loc='left', pad=3)

    n_tot_c = classified_fracs['total']
    for i in range(n_bins):
        if i % 2 == 0 and n_tot_c[i] > 0:
            y_ann = max(gf_c[i] if np.isfinite(gf_c[i]) else 0,
                        sf_c[i] if np.isfinite(sf_c[i]) else 0) + 0.05
            ax_mid.text(i, y_ann, f'{int(n_tot_c[i]):,}',
                        ha='center', fontsize=7, color='#333333', zorder=10)

    # X-axis ticks
    tick_idx    = [i for i, (lo, _) in enumerate(MAG_BINS) if lo == round(lo)]
    tick_labels = [f'G={lo:.0f}' for lo, _ in [MAG_BINS[i] for i in tick_idx]]
    last_edge = MAG_BINS[-1][1]
    if abs(last_edge - round(last_edge)) < 1e-9:
        tick_idx.append(n_bins - 0.5)
        tick_labels.append(f'G={last_edge:.0f}')

    ax_mid.set_xlim(-0.5, n_bins - 0.5)
    ax_mid.set_xticks(tick_idx)
    ax_mid.set_xticklabels(tick_labels, fontsize=10)
    ax_mid.set_xlabel('Gaia G magnitude', fontsize=13, fontweight='bold', labelpad=13)

    plt.suptitle(
        f'ERO model — Galaxy metrics vs Gaia G magnitude  '
        f'(mean ± 1σ, {n_seeds} random 80/20 splits)',
        fontsize=13, fontweight='bold', y=0.99)
    fig.subplots_adjust(top=0.90)
    plt.tight_layout()

    out = output_dir / 'ero_mag_performance.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_feature_importance(all_importances, output_dir):
    """Horizontal bar chart: mean ± std feature importance across seeds.
    Bars coloured by CV (std/mean): green = stable, red = unstable."""
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
    ax.set_xlabel('Feature importance  (mean ± 1σ across seeds)', fontsize=11,
                  fontweight='bold', labelpad=10)
    ax.grid(alpha=0.3, axis='x')
    ax.set_xlim(0, (mu_s + std_s).max() * 1.1)

    for i, (m_v, s_v, c_v) in enumerate(zip(mu_s, std_s, cv_s)):
        ax.text(m_v + s_v + 0.0005, i, f'CV={c_v:.2f}',
                va='center', fontsize=7.5, color='#333333')

    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r',
                                norm=plt.Normalize(vmin=0, vmax=cv_s.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label('CV = std / mean', fontsize=9)
    cbar.set_ticks(np.arange(0, cv_s.max(), 0.1))
    cbar.ax.text(0.4, 1.04, 'unstable', transform=cbar.ax.transAxes,
                 fontsize=8, va='bottom', ha='left', color='darkred')
    cbar.ax.text(0.4, -0.05, 'stable', transform=cbar.ax.transAxes,
                 fontsize=8, va='top', ha='left', color='darkgreen')

    n_seeds = len(all_importances)
    plt.suptitle(
        f'ERO model feature importance consistency  ({n_seeds} seeds)',
        fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout()

    out = output_dir / 'ero_mag_importance.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER not set")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ERO model: magnitude sensitivity + feature consistency")
    print("=" * 65)

    print("\n[1] Querying ERO training data...")
    df = query_ero(username)

    all_metrics     = []
    all_importances = []
    all_n           = []

    print(f"\n[2] Training {len(SEEDS)} models (seeds: {SEEDS})...")
    for i, seed in enumerate(SEEDS):
        print(f"\n  --- Seed {seed}  ({i+1}/{len(SEEDS)}) ---")
        metrics, importances, n_bins = run_seed(df, seed)
        all_metrics.append(metrics)
        all_importances.append(importances)
        all_n.append(n_bins)

        for (lo, hi), m, nb in zip(MAG_BINS, metrics, n_bins):
            if np.isnan(m['f1']):
                print(f"    G {lo:4.1f}–{hi:4.1f}:  N={nb['total']:>6,}  "
                      f"gal={nb['galaxy']:>5,}  — skipped (too few galaxies)")
            else:
                print(f"    G {lo:4.1f}–{hi:4.1f}:  N={nb['total']:>6,}  "
                      f"gal={nb['galaxy']:>5,}  "
                      f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

    print("\n[3] Computing population fractions...")
    classified_fracs = compute_classified_fracs(df)

    print("\n[4] Plotting magnitude sensitivity...")
    plot_mag_sensitivity(all_metrics, all_n, classified_fracs, OUTPUT_DIR)

    print("\n[5] Plotting feature importance consistency...")
    plot_feature_importance(all_importances, OUTPUT_DIR)

    print(f"\nAll saved to: {OUTPUT_DIR}")
    print("=" * 65)
    return 0


if __name__ == '__main__':
    sys.exit(main())
