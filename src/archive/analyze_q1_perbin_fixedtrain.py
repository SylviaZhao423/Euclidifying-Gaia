#!/usr/bin/env python3
"""
Q1 fixed-training-range per-bin evaluation.

Loads the canonical saved Q1 seed models from train_xgboost_q1.py
(trained on 17 <= G <= 21), then evaluates their Galaxy Precision /
Recall / F1 on every 0.25-mag bin from G=15 to G=21.75.

This reveals how the production model generalises to brightness regimes
outside its training range — complementing analyze_q1_perbin_sensitivity.py
which trains a fresh model per bin.

Procedure per seed:
  1. Reconstruct the canonical held-out test split inside 17 <= G <= 21
     using the same seed and split logic as train_xgboost_q1.py.
  2. Evaluate that saved model on:
       - held-out-only sources for bins inside 17 <= G <= 21
       - all sources for bins outside the training window
  3. Aggregate metrics and feature importances across the canonical seeds.

Outputs (output/q1_perbin_fixedtrain/  or  output/q1_perbin_fixedtrain_nohi/):
  q1_perbin_fixedtrain_performance.png
  q1_perbin_fixedtrain_importance.png

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_q1_perbin_fixedtrain.py
    PGUSER=siwei_zhao_2025 python src/analyze_q1_perbin_fixedtrain.py --retrain
    PGUSER=siwei_zhao_2025 python src/analyze_q1_perbin_fixedtrain.py --no-hi
    PGUSER=siwei_zhao_2025 python src/analyze_q1_perbin_fixedtrain.py --no-hi --retrain
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
_SUBDIR     = 'perbin_fixedtrain'       # overridden to 'perbin_fixedtrain_nohi' with --no-hi
OUTPUT_DIR  = PROJECT_DIR / 'output' / 'q1_model_sensitivity' / _SUBDIR
CACHE_FILE  = OUTPUT_DIR / 'q1_perbin_fixedtrain_results.json'
SEED_MODELS_DIR = PROJECT_DIR / 'output' / 'xgboost_q1' / 'seed_models'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

SEEDS     = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]
TEST_SIZE = 0.2
VAL_FRAC  = 0.125

TRAIN_LO = 17.0
# TRAIN_HI can be overridden to None at runtime via --no-hi (no faint-end cut).
TRAIN_HI: float | None = 21.0

# 0.25-mag steps from G=15.0 to G=21.75 — 28 bin centres
_edges   = np.arange(15.0, 22.01, 0.25)
MAG_BINS = [round(e, 2) for e in _edges[:-1]]
BIN_HALF = 0.125

REF_LINES_BASE = {17.0: 'navy', 18.5: 'black'}
REF_LINES_HI   = {21.0: 'darkred'}   # added only when TRAIN_HI is not None

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
    """Query full Q1 dataset (G >= 15); saved-model train window is 17<=G<=21."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT
            g_source_id AS source_id,
            {gaia_cols},
            g_phot_g_mean_mag AS phot_g_mean_mag,
            CASE
                WHEN p_phz_classification = 1 THEN 'STAR'
                WHEN p_phz_classification = 2 THEN 'GALAXY'
            END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE p_phz_classification IN (1, 2)
          AND g_phot_g_mean_mag IS NOT NULL
          AND g_phot_g_mean_mag >= 15
        ORDER BY g_source_id
    """
    print("  Querying Q1 data (G >= 15, STAR/GALAXY only)...")
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
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE g_phot_g_mean_mag IS NOT NULL
          AND g_phot_g_mean_mag >= 15
        ORDER BY g_source_id
    """
    print("  Querying Q1 population (G >= 15, all classes)...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['phot_g_mean_mag'] = pd.to_numeric(df['phot_g_mean_mag'], errors='coerce')
    return df


# ---------------------------------------------------------------------------
# Per-bin population fractions (same as perbin script)
# ---------------------------------------------------------------------------

def compute_bin_fracs(df_train, df_pop):
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
# Saved-model evaluation
# ---------------------------------------------------------------------------

def split_data(X, y, seed):
    """Same 70/15/15 split used in train_xgboost_q1.py."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=seed, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test


def reconstruct_window_test_ids(df, seed):
    """Return held-out source IDs for the canonical saved-model split."""
    hi_mask = (df['phot_g_mean_mag'] <= TRAIN_HI) if TRAIN_HI is not None else True
    df_win = df[(df['phot_g_mean_mag'] >= TRAIN_LO) & hi_mask].copy()
    y_win = df_win['label'].map(CLASS_MAPPING).values
    X_dummy = np.zeros((len(df_win), 1), dtype=float)
    _, _, X_test, _, _, _ = split_data(X_dummy, y_win, seed)
    n_test = len(X_test)

    idx = np.arange(len(df_win))
    _, idx_test, _, _ = train_test_split(
        idx, y_win, test_size=0.15, random_state=seed, stratify=y_win)
    assert len(idx_test) == n_test
    return set(df_win.iloc[idx_test]['source_id'].astype(str).tolist())


def evaluate_saved_seed(df, seed, model_path):
    """Evaluate one saved production seed model across all magnitude bins."""
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    X_all   = df[SELECTED_GAIA_PARAMETERS].values.astype(float)
    y_all   = df['label'].map(CLASS_MAPPING).values
    mag_all = df['phot_g_mean_mag'].values
    sid_all = df['source_id'].astype(str).values

    heldout_ids = reconstruct_window_test_ids(df, seed)
    in_train_window = (
        (mag_all >= TRAIN_LO) if TRAIN_HI is None
        else (mag_all >= TRAIN_LO) & (mag_all <= TRAIN_HI)
    )
    use_eval = (~in_train_window) | np.isin(sid_all, list(heldout_ids))

    n_inrange_test = int(np.sum(in_train_window & use_eval))
    n_outrange = int(np.sum(~in_train_window))
    print(f"    loaded {model_path.name}: {n_inrange_test:,} held-out in-range + {n_outrange:,} out-of-range eval sources", end="")

    metrics_per_bin = []
    for centre in MAG_BINS:
        lo, hi = centre - BIN_HALF, centre + BIN_HALF
        idx_bin = np.where((mag_all >= lo) & (mag_all < hi) & use_eval)[0]
        n_tot  = len(idx_bin)
        n_gal  = int((y_all[idx_bin] == 1).sum())
        n_star = int((y_all[idx_bin] == 0).sum())

        if n_tot < 10 or n_gal < MIN_GAL_IN_BIN or n_star < MIN_GAL_IN_BIN:
            metrics_per_bin.append({'precision': np.nan, 'recall': np.nan, 'f1': np.nan})
            continue

        p, r, f1, _ = precision_recall_fscore_support(
            y_all[idx_bin], model.predict(X_all[idx_bin]),
            labels=[0, 1], zero_division=0)
        metrics_per_bin.append({'precision': p[1], 'recall': r[1], 'f1': f1[1]})

    print()
    return metrics_per_bin, model.feature_importances_


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _shade(ax):
    x = np.array(MAG_BINS)
    ax.axvspan(x[0] - 0.15, TRAIN_LO, color='lightyellow', alpha=0.6, zorder=0)
    if TRAIN_HI is not None:
        ax.axvspan(TRAIN_HI, x[-1] + 0.15, color='lightyellow', alpha=0.6, zorder=0)
    ref_lines = {**REF_LINES_BASE, **(REF_LINES_HI if TRAIN_HI is not None else {})}
    for ref, color in ref_lines.items():
        ax.axvline(x=ref, color=color, ls='--', linewidth=1.5, alpha=0.8)
    ax.axhline(y=0.5, color='gray', ls=':', linewidth=1)
    ax.grid(alpha=0.3, axis='y')


def plot_performance(all_metrics, classified_fracs, all_fracs, output_dir):
    n_bins  = len(MAG_BINS)
    n_seeds = len(all_metrics)
    x       = np.array(MAG_BINS)

    prec = np.array([[m['precision'] for m in s] for s in all_metrics])
    rec  = np.array([[m['recall']    for m in s] for s in all_metrics])
    f1   = np.array([[m['f1']        for m in s] for s in all_metrics])

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(14, 13),
        gridspec_kw={'height_ratios': [3, 1.2, 1.2]}, sharex=True)

    # --- top panel ---
    _shade(ax_top)
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
        Line2D([0], [0], color='navy',  ls='--', lw=1.5, alpha=0.8, label='G = 17  (train lo)'),
        Line2D([0], [0], color='black', ls='--', lw=1.5, alpha=0.8, label='G = 18.5'),
    ]
    if TRAIN_HI is not None:
        ref_handles.append(
            Line2D([0], [0], color='darkred', ls='--', lw=1.5, alpha=0.8,
                   label=f'G = {TRAIN_HI}  (train hi)')
        )
    ax_top.legend(handles=metric_handles + ref_handles,
                  fontsize=10, loc='lower left', ncol=1)
    ax_top.set_ylabel('Metric value', fontsize=12, fontweight='bold')
    ax_top.set_ylim(-0.05, 1.05)

    # --- mid panel: classified fracs ---
    _shade(ax_mid)
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

    # --- bot panel: all sources ---
    _shade(ax_bot)
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
    _hi_str = f'≤ {TRAIN_HI:.0f}' if TRAIN_HI is not None else '(no faint cut)'
    ax_bot.set_xlabel(
        f'Gaia G bin centre  (model trained on {TRAIN_LO:.0f} ≤ G {_hi_str}, '
        f'evaluated on 20% held-out test set)',
        fontsize=12, fontweight='bold', labelpad=13)

    _range_str = f'{TRAIN_LO:.0f}≤G≤{TRAIN_HI:.0f}' if TRAIN_HI is not None else f'G≥{TRAIN_LO:.0f}'
    plt.suptitle(
        f'Q1 saved model ({_range_str}) — Galaxy metrics per magnitude bin  '
        f'(mean ± 1σ, {n_seeds} canonical seed models)',
        fontsize=13, fontweight='bold', y=0.99)
    fig.subplots_adjust(top=0.90)
    plt.tight_layout()

    out = output_dir / 'q1_perbin_fixedtrain_performance.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_feature_importance(all_importances, output_dir):
    imp = np.array(all_importances)
    mu  = imp.mean(axis=0)
    std = imp.std(axis=0)
    cv  = std / (mu + 1e-9)

    order = np.argsort(mu)[::-1]
    feats = [SELECTED_GAIA_PARAMETERS[i] for i in order]
    mu_s  = mu[order];  std_s = std[order];  cv_s = cv[order]

    n_feat = len(feats)
    norm   = plt.Normalize(vmin=0, vmax=2.0)
    colors = plt.cm.RdYlGn_r(norm(cv_s))

    fig, ax = plt.subplots(figsize=(11, max(8, n_feat * 0.38)))
    ax.barh(range(n_feat), mu_s, xerr=std_s, color=colors, alpha=0.85, capsize=4,
            error_kw={'elinewidth': 1.3, 'ecolor': 'black'})
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels([f'#{i+1}  {f}' for i, f in enumerate(feats)], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.5)
    ax.set_xticks(np.arange(0, 0.51, 0.1))
    ax.set_xlabel('Feature importance  (mean ± 1σ across seeds)',
                  fontsize=11, fontweight='bold', labelpad=10)
    ax.grid(alpha=0.3, axis='x')

    for i, (m_v, s_v, c_v) in enumerate(zip(mu_s, std_s, cv_s)):
        ax.text(m_v + s_v + 0.0005, i, f'CV={c_v:.2f}',
                va='center', fontsize=7.5, color='#333333')

    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label('CV = std / mean', fontsize=9)
    cbar.set_ticks(np.arange(0, 2.1, 0.2))
    cbar.ax.text(0.4, 1.04, 'unstable', transform=cbar.ax.transAxes,
                 fontsize=8, va='bottom', ha='left', color='darkred')
    cbar.ax.text(0.4, -0.05, 'stable', transform=cbar.ax.transAxes,
                 fontsize=8, va='top', ha='left', color='darkgreen')

    _range_str = f'{TRAIN_LO:.0f}≤G≤{TRAIN_HI:.0f}' if TRAIN_HI is not None else f'G≥{TRAIN_LO:.0f}'
    plt.suptitle(
        f'Q1 saved model ({_range_str}) — feature importance  '
        f'({len(all_importances)} canonical seed models)',
        fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()

    out = output_dir / 'q1_perbin_fixedtrain_importance.png'
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


def save_cache(all_metrics, all_importances, classified_fracs, all_fracs):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'all_metrics':     _to_serializable(all_metrics),
        'all_importances': [[float(v) for v in imp] for imp in all_importances],
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
    all_metrics     = _from_serializable(payload['all_metrics'])
    all_importances = [np.array(imp) for imp in payload['all_importances']]
    classified_fracs = {k: np.array(_from_serializable(v))
                        for k, v in payload['classified_fracs'].items()}
    all_fracs        = {k: np.array(_from_serializable(v))
                        for k, v in payload['all_fracs'].items()}
    return all_metrics, all_importances, classified_fracs, all_fracs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global TRAIN_HI, OUTPUT_DIR, CACHE_FILE

    retrain  = '--retrain' in sys.argv
    no_hi    = '--no-hi'   in sys.argv
    username = os.getenv('PGUSER')

    if no_hi:
        TRAIN_HI   = None
        OUTPUT_DIR = PROJECT_DIR / 'output' / 'q1_model_sensitivity' / 'perbin_fixedtrain_nohi'
        CACHE_FILE = OUTPUT_DIR / 'q1_perbin_fixedtrain_nohi_results.json'

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _range_str = f'{TRAIN_LO:.0f}≤G≤{TRAIN_HI:.0f}' if TRAIN_HI is not None else f'G≥{TRAIN_LO:.0f} (no faint cut)'
    print("=" * 65)
    print(f"Q1 fixed-train per-bin evaluation  "
          f"(saved model {_range_str}, evaluate per bin)")
    print(f"  {len(SEEDS)} seeds")
    print(f"  Cache:  {CACHE_FILE}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 65)

    if not retrain:
        cached = load_cache()
    else:
        cached = None
        print("  --retrain flag set: ignoring cache.")

    if cached is not None:
        all_metrics, all_importances, classified_fracs, all_fracs = cached
        print(f"  Loaded {len(all_metrics)} seeds from cache — skipping evaluation.")
    else:
        if not username:
            print("ERROR: PGUSER not set"); return 1

        print("\n[1] Querying Q1 training data...")
        df = query_q1(username)

        print("\n[1b] Querying Q1 population (for fraction panels)...")
        df_pop = query_q1_population(username)

        all_metrics     = []
        all_importances = []

        seed_model_paths = [SEED_MODELS_DIR / f'xgb_classifier_q1_seed{seed}.json' for seed in SEEDS]
        missing = [p.name for p in seed_model_paths if not p.exists()]
        if missing:
            print("ERROR: missing saved seed models:")
            for name in missing:
                print(f"  - {name}")
            print("Run train_xgboost_q1.py first.")
            return 1

        print(f"\n[2] Evaluating {len(SEEDS)} saved models ({_range_str})...")
        for i, (seed, model_path) in enumerate(zip(SEEDS, seed_model_paths)):
            print(f"\n  === Seed {seed}  ({i+1}/{len(SEEDS)}) ===")
            metrics, imp = evaluate_saved_seed(df, seed, model_path)
            all_metrics.append(metrics)
            if imp is not None:
                all_importances.append(imp)

            for centre, m in zip(MAG_BINS, metrics):
                if np.isnan(m['f1']):
                    print(f"    G≈{centre:5.2f}:  — skipped (too few)")
                else:
                    print(f"    G≈{centre:5.2f}:  "
                          f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

        print("\n[3] Computing per-bin population fractions...")
        classified_fracs, all_fracs = compute_bin_fracs(df, df_pop)

        print("\n[3b] Saving cache...")
        save_cache(all_metrics, all_importances, classified_fracs, all_fracs)

    print("\n[4] Plotting performance...")
    plot_performance(all_metrics, classified_fracs, all_fracs, OUTPUT_DIR)

    if all_importances:
        print(f"\n[5] Plotting feature importance ({len(all_importances)} seeds)...")
        plot_feature_importance(all_importances, OUTPUT_DIR)

    print(f"\nAll saved to: {OUTPUT_DIR}")
    print("=" * 65)
    return 0


if __name__ == '__main__':
    sys.exit(main())
