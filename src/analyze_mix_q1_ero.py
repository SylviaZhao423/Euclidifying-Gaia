#!/usr/bin/env python3
"""
Analyse XGBoost Q1+ERO model performance across mixing ratios.

Pure evaluation on Q1 test sets — no Gaia application needed.
Each seed model is evaluated on its own held-out Q1 test split (no leakage).

Produces:
  1. mix_ratio_sweep.png / .csv         — metrics at default threshold vs ratio
  2. mix_ratio_sweep_thresholds.png     — precision/recall/F1 vs ratio, colored by threshold
  3. mix_purity_comprehensive.png       — purity heatmaps (ratio × threshold) at different galaxy fractions
  4. mix_purity_landscape.png           — purity(threshold × galaxy fraction) per ratio + delta vs Q1-only

Reads:
  WSDB (Q1 data, same query as training scripts)
  output/xgboost_mix_q1_ero/ratio_sg*/q1_seed_metrics.json
  output/xgboost_mix_q1_ero/ratio_sg*/seed_models/xgb_classifier_q1_seed*.json

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_mix_q1_ero.py
    PGUSER=siwei_zhao_2025 python src/analyze_mix_q1_ero.py --dir xgboost_mix_q1_ero_g
    PGUSER=siwei_zhao_2025 python src/analyze_mix_q1_ero.py --dir xgboost_mix_q1_ero_cs0.5
    PGUSER=siwei_zhao_2025 python src/analyze_mix_q1_ero.py --dir xgboost_mix_q1_ero_g_nocs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np


import pandas as pd


def make_threshold_cmap(n_intervals=4):
    """Discrete threshold colormap: light→dark within each interval.

    4 intervals (0.5→0.9):  pink | orange | blue | green
    5 intervals (0.5→1.0):  pink | orange | blue | green | dark-green

    Each interval has 4 shades from light (low threshold) to dark (high).
    For 5 intervals, the green group is extended to 8 shades (4 green + 4 dark-green).
    """
    tab = cm.tab20c
    # tab20c groups: 0-3 blue, 4-7 orange, 8-11 green, 12-15 purple, 16-19 grey
    # We use: pink (custom), orange, blue, green
    pink_shades = ['#f6a4c0', '#f48db0', '#f06292', '#e91e8c']  # light→dark

    colors = []
    # Group 1: pinks (custom)
    colors.extend([mcolors.to_rgba(c) for c in pink_shades])
    # Group 2: oranges (light→dark)
    colors.extend([tab(i / 19) for i in reversed(range(4, 8))])
    # Group 3: blues (light→dark)
    colors.extend([tab(i / 19) for i in reversed(range(0, 4))])
    # Group 4: greens (light→dark)
    colors.extend([tab(i / 19) for i in reversed(range(8, 12))])

    if n_intervals == 5:
        # Group 5: darker greens for 0.9→1.0 (interpolate beyond tab20c greens)
        base_green = np.array(tab(8 / 19))  # darkest tab20c green
        dark_greens = []
        for f in [0.85, 0.70, 0.55, 0.40]:  # progressively darker
            dark_greens.append(tuple(base_green[:3] * f) + (1.0,))
        colors.extend(dark_greens)

    return mcolors.ListedColormap(colors, name=f'threshold_{n_intervals}grp')
import xgboost as xgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

import os

PROJECT_DIR = Path(__file__).resolve().parents[1]
MIX_DIR     = PROJECT_DIR / "output" / "xgboost_mix_q1_ero"

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'
MAG_CUT   = 17
CLASS_MAPPING = {'STAR': 0, 'GALAXY': 1}
SEEDS = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]


def query_q1(username):
    """Query Q1 data — same query as training scripts."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    feat_names = list(SELECTED_GAIA_PARAMETERS)
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in feat_names])
    query = f"""
        SELECT {gaia_cols},
               CASE WHEN p_phz_classification = 1 THEN 'STAR'
                    WHEN p_phz_classification = 2 THEN 'GALAXY' END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep
        WHERE p_phz_classification IN (1, 2)
          AND g_phot_g_mean_mag >= {MAG_CUT}
        ORDER BY g_random_index
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    for c in feat_names:
        df[c] = pd.to_numeric(df[c].replace('NOT_AVAILABLE', np.nan), errors='coerce')
    return df[feat_names].values, df['label'].map(CLASS_MAPPING).values


# ── Load saved metrics from all ratio dirs ──────────────────────────────────

def load_all_ratios():
    """Scan ratio_sg* dirs and load q1_seed_metrics.json from each."""
    rows = []
    ratio_dirs = sorted(MIX_DIR.glob("ratio_*"))

    for d in ratio_dirs:
        if "sweep" in d.name:
            continue

        metrics_path = d / "q1_seed_metrics.json"
        if not metrics_path.exists():
            continue

        # e.g. ratio_sg5, ratio_sg8.63_all_ero, ratio_sg2.8_q1_only
        tag = d.name
        try:
            num_str = tag.replace("ratio_sg", "").split("_")[0]
            ratio = float(num_str)
        except ValueError:
            continue

        with open(metrics_path) as f:
            seed_metrics = json.load(f)

        for sm in seed_metrics:
            sm["ratio"] = ratio
            sm["dir"] = str(d)
            rows.append(sm)

    if not rows:
        return None
    return pd.DataFrame(rows)


def _find_seed_dir(ratio):
    """Find seed_models directory for a given ratio."""
    for d in MIX_DIR.glob("ratio_*"):
        if "sweep" in d.name:
            continue
        try:
            num_str = d.name.replace("ratio_sg", "").split("_")[0]
            r = float(num_str)
            if abs(r - ratio) < 0.1:
                sd = d / "seed_models"
                if sd.exists():
                    return sd
        except ValueError:
            continue
    return None


def _per_seed_predictions(seed_dir, X_q1, y_q1):
    """Load seed models, predict on each seed's own test split. Returns list of (y_test, p_test)."""
    seed_paths = sorted(seed_dir.glob("xgb_classifier_q1_seed*.json"))
    preds = []
    for sp in seed_paths:
        seed_val = int(sp.stem.replace("xgb_classifier_q1_seed", ""))
        _, X_ts, _, y_ts = train_test_split(
            X_q1, y_q1, test_size=0.15, random_state=seed_val, stratify=y_q1)
        m = xgb.XGBClassifier()
        m.load_model(str(sp))
        preds.append((y_ts, m.predict_proba(X_ts)[:, 1]))
    return preds


def _compute_tpr_fpr_grid(ratios, X_q1, y_q1, thresh_fine):
    """Compute TPR/FPR grid: shape (n_ratios, n_thresholds)."""
    n_r = len(ratios)
    n_t = len(thresh_fine)
    tpr_grid = np.zeros((n_r, n_t))
    fpr_grid = np.zeros((n_r, n_t))
    ratio_vals = []
    ratio_labels = []

    for ri, ratio in enumerate(ratios):
        seed_dir = _find_seed_dir(ratio)
        if seed_dir is None:
            continue

        seed_preds = _per_seed_predictions(seed_dir, X_q1, y_q1)
        if not seed_preds:
            continue

        label = f"{ratio:.0f}:1" if ratio == int(ratio) else f"{ratio:.1f}:1"
        ratio_vals.append(ratio)
        ratio_labels.append(label)

        for ti, t in enumerate(thresh_fine):
            tp_s, fp_s, fn_s, tn_s = 0, 0, 0, 0
            for y_ts, p_ts in seed_preds:
                pred_gal = p_ts >= t
                tp_s += (pred_gal & (y_ts == 1)).sum()
                fp_s += (pred_gal & (y_ts == 0)).sum()
                fn_s += (~pred_gal & (y_ts == 1)).sum()
                tn_s += (~pred_gal & (y_ts == 0)).sum()
            n = len(seed_preds)
            tpr_grid[ri, ti] = (tp_s/n) / max(tp_s/n + fn_s/n, 1)
            fpr_grid[ri, ti] = (fp_s/n) / max(fp_s/n + tn_s/n, 1)

    return ratio_vals, ratio_labels, tpr_grid[:len(ratio_vals)], fpr_grid[:len(ratio_vals)]


# ── Sweep plot ──────────────────────────────────────────────────────────────

def plot_sweep(df_summary, output_dir):
    """Metrics at default threshold (0.5) vs ratio."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    x = df_summary["ratio"].values

    for col, label, color in [
        ("roc_auc_mean",           "ROC AUC",   "#333333"),
        ("galaxy_precision_mean",  "Precision",  "steelblue"),
        ("galaxy_recall_mean",     "Recall",     "coral"),
        ("galaxy_f1_mean",         "F1",         "mediumseagreen"),
    ]:
        std_col = col.replace("_mean", "_std")
        ax.errorbar(x, df_summary[col].values, yerr=df_summary[std_col].values,
                    fmt="o-", color=color, lw=2, capsize=4, label=label)

    q1_ratio = x[0]
    ax.axvline(q1_ratio, color="grey", lw=1, ls=":", alpha=0.5)
    ax.text(q1_ratio, ax.get_ylim()[0] + 0.005, "Q1\nonly", ha="center",
            fontsize=9, color="grey", fontweight="bold", va="bottom")

    ax.set_xlabel("Training STAR:GALAXY ratio", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=11)
    out = output_dir / "mix_ratio_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


def plot_sweep_thresholds(ratio_vals, ratio_labels, tpr_grid, fpr_grid,
                          thresh_fine, output_dir):
    """Precision/Recall/F1 vs ratio, colored by threshold."""
    n_gal_frac  = 0.26
    n_star_frac = 0.74

    fig, axes = plt.subplots(3, 1, figsize=(12, 5.5), sharex=True)
    fig.subplots_adjust(hspace=0.18)

    t_cmap = make_threshold_cmap(n_intervals=4)  # pink|orange|blue|green for 0.5→0.9
    t_norm = plt.Normalize(thresh_fine[0], thresh_fine[-1])
    x = np.array(ratio_vals)
    n_r = len(ratio_vals)
    n_t = len(thresh_fine)

    prec_grid = np.zeros((n_t, n_r))
    rec_grid  = np.zeros((n_t, n_r))
    f1_grid_m = np.zeros((n_t, n_r))

    for ti in range(n_t):
        for ri in range(n_r):
            tpr = tpr_grid[ri, ti]
            fpr = fpr_grid[ri, ti]
            prec = tpr * n_gal_frac / (tpr * n_gal_frac + fpr * n_star_frac + 1e-12)
            prec_grid[ti, ri] = prec
            rec_grid[ti, ri]  = tpr
            f1_grid_m[ti, ri] = 2 * prec * tpr / (prec + tpr + 1e-12)

    for ax, metric_name, grid in zip(axes,
                                      ['Precision', 'F1', 'Recall'],
                                      [prec_grid, f1_grid_m, rec_grid]):
        for ti in range(0, n_t, 2):
            t = thresh_fine[ti]
            ax.plot(x, grid[ti, :], color=t_cmap(t_norm(t)), lw=0.7, alpha=0.6)

        for t_mark in [0.5, 0.9]:
            ti_mark = np.argmin(np.abs(thresh_fine - t_mark))
            color = t_cmap(t_norm(t_mark))
            ax.plot(x, grid[ti_mark, :], color=color, lw=1, alpha=1.0, zorder=5)
            ax.text(x[-1] + 0.3, grid[ti_mark, -1], f't={t_mark}',
                    fontsize=11, fontweight='bold', color=color, va='center')
            # Dot markers + bare number labels at each ratio
            # Labels go above for the upper line, below for the lower line in each panel
            # Precision: t=0.9 up, t=0.5 down; F1/Recall: t=0.5 up, t=0.9 down
            if metric_name == 'Precision':
                label_above = (t_mark == 0.9)
            else:
                label_above = (t_mark == 0.5)
            y_off = 5 if label_above else -5
            va = 'bottom' if label_above else 'top'

            for ri in range(n_r):
                ax.scatter(x[ri], grid[ti_mark, ri], s=20, color=color,
                           edgecolors='k', linewidths=0.4, zorder=9)
                if ri == 0:
                    ax.annotate(f'{grid[ti_mark, ri]:.2f}',
                                xy=(x[ri], grid[ti_mark, ri]),
                                xytext=(-8, y_off), textcoords='offset points',
                                fontsize=11, fontweight='bold', color=color,
                                ha='right', va=va, zorder=10)
                else:
                    ax.annotate(f'{grid[ti_mark, ri]:.2f}',
                                xy=(x[ri], grid[ti_mark, ri]),
                                xytext=(0, y_off), textcoords='offset points',
                                fontsize=11, fontweight='bold', color=color,
                                ha='center', va=va, zorder=10)

        for ti in range(0, n_t - 1):
            t_mid = (thresh_fine[ti] + thresh_fine[ti + 1]) / 2
            ax.fill_between(x, grid[ti, :], grid[ti + 1, :],
                            color=t_cmap(t_norm(t_mid)), alpha=0.15)

        ax.axvline(x[0], color="grey", lw=1, ls=":", alpha=0.5)
        ax.set_xlim(x[0] - 0.8, x[-1] + 0.8)
        ax.set_ylabel(metric_name, fontsize=13, fontweight="bold", labelpad=10)
        # Set y-axis: Precision gets round bounds, others auto
        all_vals = grid.ravel()
        if metric_name == 'Precision':
            ax.set_ylim(0.81, 1.04)
            ax.set_yticks([0.9, 1.0])
        elif metric_name == 'F1':
            ax.set_ylim(0.45, 0.68)
            ax.set_yticks([0.5, 0.6])
        else:
            ax.set_ylim(0.30, 0.53)
            ax.set_yticks([0.4, 0.5])
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=11)

        idx = list(axes).index(ax)
        if idx < len(axes) - 1:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
        if idx > 0:
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='x', which='both', top=False)
        if idx == len(axes) - 1:
            ax.set_xlabel("Training STAR:GALAXY ratio", fontsize=13, fontweight="bold", labelpad=10)

    sm = plt.cm.ScalarMappable(cmap=t_cmap, norm=plt.Normalize(0.5, 0.9))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), pad=0.02, shrink=0.8)
    cbar.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9])
    cbar.set_label("P(gal) threshold", fontsize=12, fontweight='bold')

    # fig.suptitle("Metrics vs training ratio — colored by classification threshold",
    #              fontsize=14, fontweight="bold", y=1.02)
    out = output_dir / "mix_ratio_sweep_thresholds.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


def plot_purity_comprehensive(ratio_vals, ratio_labels, tpr_grid, fpr_grid,
                               thresh_fine, output_dir):
    """Purity heatmaps: ratio × threshold at different galaxy fractions."""
    n_ratios_hm = len(ratio_vals)
    gal_fractions = [0.005, 0.01, 0.03]
    frac_labels = ["0.5%", "1%", "3%"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.subplots_adjust(hspace=0.18)

    for idx, (ax, pi, flabel) in enumerate(zip(axes, gal_fractions, frac_labels)):
        purity = np.zeros((n_ratios_hm, len(thresh_fine)))
        for ri in range(n_ratios_hm):
            for ti in range(len(thresh_fine)):
                tpr = tpr_grid[ri, ti]
                fpr = fpr_grid[ri, ti]
                purity[ri, ti] = tpr * pi / (tpr * pi + fpr * (1 - pi) + 1e-12)

        im = ax.pcolormesh(thresh_fine, range(n_ratios_hm), purity * 100,
                           cmap='RdYlGn', shading='auto', vmin=0, vmax=100)
        cs = ax.contour(thresh_fine, np.arange(n_ratios_hm), purity * 100,
                        levels=[25, 50, 75, 90], colors='black',
                        linewidths=0.8, linestyles='--')
        ax.clabel(cs, fmt='%d%%', fontsize=11)

        bbox = dict(boxstyle='round,pad=0.15', facecolor='white',
                    edgecolor='none')
        for t_mark in [0.5, 0.7, 0.9]:
            ti_mark = np.argmin(np.abs(thresh_fine - t_mark))
            ax.axvline(t_mark, color='white', lw=0.5, ls=':', alpha=0.5)
            for ri in range(n_ratios_hm):
                val = purity[ri, ti_mark] * 100
                ax.text(t_mark, ri, f'{val:.0f}', ha='center', va='center',
                        fontsize=7, fontweight='bold', color='black')

        ax.set_yticks(range(n_ratios_hm))
        ax.set_yticklabels(ratio_labels, fontsize=10)
        ax.set_ylabel("Training ratio", fontsize=11, fontweight='bold')

        # Panel label
        ax.text(0.98, 1.11, f'{flabel} galaxy fraction', transform=ax.transAxes,
                fontsize=11, fontstyle='italic', fontweight='bold', color='#555555',
                ha='right', va='top')

        # Remove shared edges
        if idx < len(axes) - 1:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', labelbottom=False, bottom=False)
        if idx > 0:
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='x', top=False)
        if idx == len(axes) - 1:
            ax.set_xlabel("P(gal) threshold", fontsize=11, fontweight='bold')

    # Single shared colorbar
    fig.colorbar(im, ax=axes.tolist(), pad=0.02, shrink=0.8, label="Purity (%)")

    out = output_dir / "mix_purity_comprehensive.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {out}")


def plot_purity_landscape(ratio_vals, ratio_labels, tpr_grid, fpr_grid,
                           thresh_fine, output_dir):
    """Per-ratio purity heatmaps (threshold × galaxy fraction) + delta vs Q1-only."""
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
    delta_cmap = LinearSegmentedColormap.from_list(
        'orange_white_purple',
        ['#EF7627', '#FF9A56', '#FFFFFF', '#D162A4', '#8B1A6B'])

    pi_hm = np.logspace(-2.5, -0.3, 80)
    n_ratios_hm = len(ratio_vals)

    # Compute purity maps
    purity_maps = {}
    for ri in range(n_ratios_hm):
        purity_2d = np.zeros((len(thresh_fine), len(pi_hm)))
        for ti in range(len(thresh_fine)):
            tpr = tpr_grid[ri, ti]
            fpr = fpr_grid[ri, ti]
            purity_2d[ti, :] = tpr * pi_hm / (tpr * pi_hm + fpr * (1 - pi_hm) + 1e-12)
        purity_maps[ri] = purity_2d

    # Delta vs Q1-only
    deltas = {}
    for ri in range(1, n_ratios_hm):
        deltas[ri] = (purity_maps[ri] - purity_maps[0]) * 100
    all_delta_vals = np.concatenate([d.ravel() for d in deltas.values()])
    global_dmin = all_delta_vals.min()
    global_dmax = all_delta_vals.max()
    vabs = max(abs(global_dmin), abs(global_dmax))
    delta_norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

    fig, axes = plt.subplots(2, n_ratios_hm, figsize=(4.5 * n_ratios_hm, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.35)

    # Row 0: absolute purity
    for ri in range(n_ratios_hm):
        ax = axes[0, ri]
        im = ax.pcolormesh(pi_hm * 100, thresh_fine, purity_maps[ri] * 100,
                           cmap='RdYlGn', shading='auto', vmin=0, vmax=100)
        cs = ax.contour(pi_hm * 100, thresh_fine, purity_maps[ri] * 100,
                        levels=[25, 50, 75, 90], colors='black',
                        linewidths=0.8, linestyles='--')
        ax.clabel(cs, fmt='%d%%', fontsize=7)
        ax.axvspan(1, 3, color='white', alpha=0.15, zorder=2)
        ax.axvline(1, color='white', lw=0.8, ls=':', alpha=0.5)
        ax.axvline(3, color='white', lw=0.8, ls=':', alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlim(pi_hm.min() * 100, pi_hm.max() * 100)
        ax.set_xlabel("True galaxy fraction (%)", fontsize=9, fontweight='bold')
        if ri == 0:
            ax.set_ylabel("P(gal) threshold", fontsize=10, fontweight='bold')
        ax.set_title(f"Ratio {ratio_labels[ri]}", fontsize=11, fontweight='bold')
        fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85, label="Purity (%)")

    # Row 1 col 0: recall line plot
    ax = axes[1, 0]
    colors_recall = cm.viridis(np.linspace(0, 0.9, n_ratios_hm))
    for ri in range(n_ratios_hm):
        ax.plot(thresh_fine, tpr_grid[ri, :] * 100, color=colors_recall[ri],
                lw=1.5, label=ratio_labels[ri])
    ax.set_xlabel("P(gal) threshold", fontsize=9, fontweight='bold')
    ax.set_ylabel("Recall (%)", fontsize=10, fontweight='bold')
    ax.set_title("Recall vs threshold", fontsize=11, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(alpha=0.3)

    # Row 1 col 1+: delta heatmaps
    for ri in range(1, n_ratios_hm):
        ax = axes[1, ri]
        im = ax.pcolormesh(pi_hm * 100, thresh_fine, deltas[ri],
                           cmap=delta_cmap, shading='auto', norm=delta_norm)
        cs = ax.contour(pi_hm * 100, thresh_fine, deltas[ri],
                        levels=[1, 2, 5, 10], colors='black',
                        linewidths=0.8, linestyles='--')
        ax.clabel(cs, fmt='+%d%%', fontsize=7)
        ax.axvspan(1, 3, color='white', alpha=0.1, zorder=2)
        ax.axvline(1, color='grey', lw=0.8, ls=':', alpha=0.5)
        ax.axvline(3, color='grey', lw=0.8, ls=':', alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlim(pi_hm.min() * 100, pi_hm.max() * 100)
        ax.set_xlabel("True galaxy fraction (%)", fontsize=9, fontweight='bold')
        if ri == 1:
            ax.set_ylabel("P(gal) threshold", fontsize=10, fontweight='bold')
        ax.set_title("Purity gain vs Q1-only", fontsize=11, fontweight='bold')
        fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85, label="Purity gain (%)")

    fig.suptitle("Purity landscape — all ratios\n"
                 "(top: absolute purity, bottom: recall + purity gain over Q1-only)",
                 fontsize=14, fontweight='bold', y=1.0)
    out = output_dir / "mix_purity_landscape.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {out}")


def plot_recalibration(ratio_vals, ratio_labels, tpr_grid, fpr_grid,
                       thresh_fine, X_q1, y_q1, output_dir):
    """Bayesian recalibration: purity curves + 2D heatmap for best ratio.
    Format matches pgal_recalibration.png exactly."""
    import matplotlib.colors as mcolors

    pi_q1 = (y_q1 == 1).sum() / len(y_q1)
    pi_range = np.logspace(-3, np.log10(0.5), 500)
    mark_points = [(0.01, "1%"), (0.03, "3%")]
    thresholds_mark = [0.5, 0.9]
    recal_display_ratios = {ratio_vals[0], 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, ratio_vals[-1]}

    # Threshold colormap (same as pgal_recalibration.png)
    cmap = make_threshold_cmap(n_intervals=5)
    norm = mcolors.Normalize(vmin=0.5, vmax=1.0)

    # Compute per-ratio TPR/FPR at marked thresholds
    thresh_range = [0.5, 0.7, 0.9]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    fig.subplots_adjust(wspace=0.18)

    # [0] Purity curves — selected ratios, colored by threshold
    ax = axes[0]
    recal_idx = [i for i, rv in enumerate(ratio_vals)
                 if any(abs(rv - dr) < 0.1 for dr in recal_display_ratios)]

    # Line styles: solid for Q1-only (first), dashed for best (last), dotted for middle
    ratio_linestyles = {}
    for ri in recal_idx:
        if ri == recal_idx[0]:
            ratio_linestyles[ri] = '-'
        elif ri == recal_idx[-1]:
            ratio_linestyles[ri] = '--'
        else:
            ratio_linestyles[ri] = ':'

    for ri in recal_idx:
        ls = ratio_linestyles[ri]
        for t in thresh_range:
            ti = np.argmin(np.abs(thresh_fine - t))
            tpr_v = tpr_grid[ri, ti]
            fpr_v = fpr_grid[ri, ti]
            if fpr_v == 0:
                continue
            prec = tpr_v * pi_range / (tpr_v * pi_range + fpr_v * (1 - pi_range) + 1e-12)
            col = cmap(norm(t))
            lw = 1.5 if ri in [recal_idx[0], recal_idx[-1]] else 0.8
            ax.plot(pi_range * 100, prec * 100, color=col, lw=lw, ls=ls, alpha=0.85)

    # Annotate Q1-only (first) and best ratio (last) at mark points + Q1 rate
    bbox_gap = dict(boxstyle='round,pad=0.12', facecolor='white',
                    edgecolor='none', alpha=0.7)
    annotate_ratios = [recal_idx[0], recal_idx[-1]]  # Q1-only and best
    for ri in annotate_ratios:
        rlabel = ratio_labels[ri]
        is_baseline = (ri == recal_idx[0])
        marker = 'o' if is_baseline else 's'  # circle for Q1, square for best

        for thresh_mark in thresholds_mark:
            ti = np.argmin(np.abs(thresh_fine - thresh_mark))
            tpr_v = tpr_grid[ri, ti]
            fpr_v = fpr_grid[ri, ti]
            col = cmap(norm(thresh_mark))

            all_marks = list(mark_points) + [(pi_q1, f"{pi_q1:.0%}")]
            for pi_mark, pi_label in all_marks:
                p_mark = tpr_v * pi_mark / (tpr_v * pi_mark + fpr_v * (1 - pi_mark) + 1e-12)
                ax.scatter(pi_mark * 100, p_mark * 100, s=30, marker=marker,
                           zorder=9, color=col, edgecolors='k', linewidths=0.5)
                # Q1-only: italic, below point; best: bold, above point
                lbl = f"{p_mark:.0%}"
                # Per-point offsets: key = (is_baseline, thresh_mark, pi_mark)
                # All 12 labels adjustable independently
                offsets = {
                    # baseline (Q1-only), t=0.5
                    (True,  0.5, 0.01): (18, -3),     # baseline, t=0.5, @ 1%
                    (True,  0.5, 0.03): (18, -3),     # baseline, t=0.5, @ 3%
                    (True,  0.5, pi_q1): (18, -10),    # baseline, t=0.5, @ 26%
                    # baseline (Q1-only), t=0.9
                    (True,  0.9, 0.01): (18, -3),     # baseline, t=0.9, @ 1%
                    (True,  0.9, 0.03): (18, -3),     # baseline, t=0.9, @ 3%
                    (True,  0.9, pi_q1): (18, 10),    # baseline, t=0.9, @ 26%
                    # best ratio (8.6:1), t=0.5
                    (False, 0.5, 0.01): (-18, 3),     # best, t=0.5, @ 1%
                    (False, 0.5, 0.03): (-18, 3),     # best, t=0.5, @ 3%
                    (False, 0.5, pi_q1): (-17, -4),   # best, t=0.5, @ 26%
                    # best ratio (8.6:1), t=0.9
                    (False, 0.9, 0.01): (-18, 3),     # best, t=0.9, @ 1%
                    (False, 0.9, 0.03): (-18, 3),     # best, t=0.9, @ 3%
                    (False, 0.9, pi_q1): (-18, 10),   # best, t=0.9, @ 26%
                }
                oxy = offsets.get((is_baseline, thresh_mark, pi_mark), (0, 5))
                ha, va = 'center', 'center'
                ax.annotate(lbl, xy=(pi_mark * 100, p_mark * 100),
                            xytext=oxy, textcoords='offset points',
                            fontsize=11, fontweight="bold",
                            color=col, ha=ha, va=va, zorder=10)

    ax.axhline(50, color="grey", lw=1, ls=":", alpha=0.5)
    ax.axvspan(1, 3, color="steelblue", alpha=0.12, zorder=0)
    ax.axvline(pi_q1 * 100, color="coral", lw=1.5, ls="--")
    ax.set_xlabel("True galaxy fraction (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Galaxy selection purity (%)", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 50)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)

    # Custom legend for ratio models
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='darkgray', lw=1.5, ls='-', marker='o', markersize=5,
               markerfacecolor='darkgray', markeredgecolor='k', markeredgewidth=0.5,
               label=f'{ratio_labels[recal_idx[0]]} Q1-only'),
        Line2D([0], [0], color='darkgray', lw=0.8, ls=':', label='intermediate ratios'),
        Line2D([0], [0], color='darkgray', lw=1.5, ls='--', marker='s', markersize=5,
               markerfacecolor='darkgray', markeredgecolor='k', markeredgewidth=0.5,
               label=f'{ratio_labels[recal_idx[-1]]} all ERO'),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc='lower right')

    # Threshold colorbar for left panel
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.8)
    cb.set_label("P(gal) threshold", fontsize=11, fontweight="bold")
    cb.ax.tick_params(labelsize=10)

    # [1] 2D heatmap for best ratio
    best_ri_hm = len(ratio_vals) - 1
    thresh_hm = np.linspace(0.3, 0.99, 70)
    pi_grid = np.logspace(-2.5, -0.3, 80)
    purity_2d = np.zeros((len(thresh_hm), len(pi_grid)))

    seed_dir = _find_seed_dir(ratio_vals[best_ri_hm])
    if seed_dir:
        seed_preds = _per_seed_predictions(seed_dir, X_q1, y_q1)
        for i, t in enumerate(thresh_hm):
            tp_s, fp_s, fn_s, tn_s = 0, 0, 0, 0
            for y_ts, p_ts in seed_preds:
                pred_gal = p_ts >= t
                tp_s += (pred_gal & (y_ts == 1)).sum()
                fp_s += (pred_gal & (y_ts == 0)).sum()
                fn_s += (~pred_gal & (y_ts == 1)).sum()
                tn_s += (~pred_gal & (y_ts == 0)).sum()
            nn = len(seed_preds)
            tpr_v = (tp_s/nn) / max(tp_s/nn + fn_s/nn, 1)
            fpr_v = (fp_s/nn) / max(fp_s/nn + tn_s/nn, 1)
            for j, pi in enumerate(pi_grid):
                purity_2d[i, j] = tpr_v * pi / (tpr_v * pi + fpr_v * (1 - pi) + 1e-12)

    ax = axes[1]
    im = ax.pcolormesh(pi_grid * 100, thresh_hm, purity_2d * 100,
                       cmap='RdYlGn', shading='auto', vmin=0, vmax=100)
    cs = ax.contour(pi_grid * 100, thresh_hm, purity_2d * 100,
                    levels=[25, 50, 75, 90, 95], colors='black',
                    linewidths=0.8, linestyles='--')
    ax.clabel(cs, fmt='%d%%', fontsize=11)
    ax.axvspan(1, 3, color='white', alpha=0.15, zorder=2)
    ax.axvline(1, color='white', lw=0.8, ls=':', alpha=0.5)
    ax.axvline(3, color='white', lw=0.8, ls=':', alpha=0.5)
    ax.axvline(pi_q1 * 100, color='coral', lw=1.5, ls='--', alpha=0.8, zorder=3)
    ax.set_xscale('log')
    ax.set_xlim(pi_grid.min() * 100, pi_grid.max() * 100)
    ax.set_xlabel("True galaxy fraction (%)", fontsize=13, fontweight="bold")
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("P(gal) threshold", fontsize=13, fontweight="bold", labelpad=10)
    ax.tick_params(labelsize=11)
    cb2 = fig.colorbar(im, ax=ax, pad=0.02)
    cb2.set_label("Purity (%)", fontsize=11, fontweight="bold")
    cb2.ax.tick_params(labelsize=10)

    out = output_dir / "mix_threshold_recalibration.png"
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  -> {out}")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyse Q1+ERO model performance across mixing ratios (Q1 test set only)")
    parser.add_argument("--dir", type=str, default="xgboost_mix_q1_ero",
                        help="Output directory name under output/ (default: xgboost_mix_q1_ero)")
    args = parser.parse_args()

    global MIX_DIR
    MIX_DIR = PROJECT_DIR / "output" / args.dir

    # ── 1. Load saved per-seed metrics ──────────────────────────────────────
    print(f"[1] Scanning for trained ratio models in {MIX_DIR}...")
    df_all = load_all_ratios()
    if df_all is None:
        print("ERROR: No ratio models found. Run train_xgboost_mix_q1_ero.py first.")
        return 1

    ratios = sorted(df_all["ratio"].unique())
    print(f"  Found {len(ratios)} ratios: {ratios}")
    print(f"  Total seed entries: {len(df_all)}")

    # ── 2. Summary table ────────────────────────────────────────────────────
    summary_rows = []
    for ratio in ratios:
        sub = df_all[df_all["ratio"] == ratio]
        row = {"ratio": ratio}
        for col in ["accuracy", "galaxy_precision", "galaxy_recall",
                     "galaxy_f1", "star_f1", "roc_auc"]:
            row[f"{col}_mean"] = sub[col].mean()
            row[f"{col}_std"]  = sub[col].std()
        summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows)

    csv_path = MIX_DIR / "mix_ratio_sweep.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"  -> {csv_path}")

    print(f"\n  {'Ratio':>7} {'AUC':>10} {'Gal_P':>10} {'Gal_R':>10} "
          f"{'Gal_F1':>10} {'Star_F1':>10}")
    print("  " + "-" * 65)
    for _, r in df_summary.iterrows():
        print(f"  {r['ratio']:>6.1f}:1 {r['roc_auc_mean']:>9.4f} "
              f"{r['galaxy_precision_mean']:>9.4f} "
              f"{r['galaxy_recall_mean']:>9.4f} "
              f"{r['galaxy_f1_mean']:>9.4f} "
              f"{r['star_f1_mean']:>9.4f}")

    # ── 3. Basic sweep plot ─────────────────────────────────────────────────
    print(f"\n[2] Plotting sweep...")
    plot_sweep(df_summary, MIX_DIR)

    # ── 4. Per-seed threshold analysis ──────────────────────────────────────
    print(f"\n[3] Loading Q1 data from WSDB...")
    username = os.environ.get("PGUSER")
    if not username:
        print("  ERROR: PGUSER not set, skipping threshold analysis")
        return 0
    X_q1, y_q1 = query_q1(username)
    print(f"  Q1 data: {len(X_q1):,} sources  "
          f"STAR: {(y_q1==0).sum():,}  GALAXY: {(y_q1==1).sum():,}")

    thresh_fine = np.linspace(0.5, 0.9, 50)
    ratio_vals, ratio_labels, tpr_grid, fpr_grid = \
        _compute_tpr_fpr_grid(ratios, X_q1, y_q1, thresh_fine)
    print(f"  Computed grid for {len(ratio_vals)} ratios × {len(thresh_fine)} thresholds")

    # ── 5. Plots ────────────────────────────────────────────────────────────
    print(f"\n[4] Plotting threshold-colored sweep...")
    plot_sweep_thresholds(ratio_vals, ratio_labels, tpr_grid, fpr_grid,
                          thresh_fine, MIX_DIR)

    print(f"\n[5] Plotting comprehensive purity heatmaps...")
    plot_purity_comprehensive(ratio_vals, ratio_labels, tpr_grid, fpr_grid,
                               thresh_fine, MIX_DIR)

    print(f"\n[6] Plotting purity landscape...")
    plot_purity_landscape(ratio_vals, ratio_labels, tpr_grid, fpr_grid,
                           thresh_fine, MIX_DIR)

    print(f"\n[7] Plotting recalibration...")
    plot_recalibration(ratio_vals, ratio_labels, tpr_grid, fpr_grid,
                       thresh_fine, X_q1, y_q1, MIX_DIR)

    return 0


if __name__ == "__main__":
    sys.exit(main())
