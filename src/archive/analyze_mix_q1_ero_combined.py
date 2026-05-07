#!/usr/bin/env python3
"""
Analyse XGBoost Q1+ERO models across mixing ratios.

Reads saved models and metrics from output/xgboost_mix_q1_ero/ratio_sg*/
(produced by train_xgboost_mix_q1_ero.py --target-ratio N).

Produces:
  1. Sweep plot: metrics vs training ratio
  2. Threshold analysis: P(gal) distribution on Gaia at each ratio
  3. Bayesian recalibration: purity vs galaxy fraction per ratio

Reads:
  output/xgboost_mix_q1_ero/ratio_sg*/q1_seed_metrics.json
  output/xgboost_mix_q1_ero/ratio_sg*/seed_models/xgb_classifier_q1_seed*.json
  output/umap/arrays/gaia_random.npz
  output/umap/arrays/q1_indiv.npz

Writes:
  output/xgboost_mix_q1_ero/
    mix_ratio_sweep.png
    mix_ratio_sweep.csv
    mix_threshold_pgal_overlay.png
    mix_threshold_recalibration.png
    mix_threshold_analysis.csv

Usage:
    python src/analyze_mix_q1_ero.py
    python src/analyze_mix_q1_ero.py --no-threshold   # skip threshold/recalibration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
MIX_DIR     = PROJECT_DIR / "output" / "xgboost_mix_q1_ero"
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"

SEEDS = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]


# ── Load saved metrics from all ratio dirs ──────────────────────────────────

def load_all_ratios():
    """Scan ratio_sg* dirs and load q1_seed_metrics.json from each."""
    rows = []
    ratio_dirs = sorted(MIX_DIR.glob("ratio_*"))

    for d in ratio_dirs:
        # Skip sweep directories
        if "sweep" in d.name:
            continue

        metrics_path = d / "q1_seed_metrics.json"
        if not metrics_path.exists():
            continue

        # Extract ratio from dir name
        # e.g. ratio_sg5, ratio_sg8.63_all_ero, ratio_sg2.8_q1_only
        tag = d.name
        try:
            # Strip prefix and any suffix after the number
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


# ── Sweep plot ──────────────────────────────────────────────────────────────

def plot_sweep(df_summary, output_dir):
    """Basic sweep plot — metrics at default threshold (0.5) vs ratio."""
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
                          thresh_fine, df_summary, output_dir):
    """Sweep plot with threshold as color dimension.
    Each metric (Precision, Recall, F1) is drawn as a family of lines
    colored by P(gal) threshold. ROC AUC as a single black line.
    """
    # Reconstruct precision, recall, F1 from TPR/FPR
    # Q1 test set class balance: ~74% star, 26% galaxy
    n_star_frac = 0.74
    n_gal_frac  = 0.26

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    fig.subplots_adjust(hspace=0.05)

    t_cmap = cm.RdYlGn
    x = np.array(ratio_vals)

    # Compute metric grids: shape (n_thresholds, n_ratios)
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
                                      ['Precision', 'Recall', 'F1'],
                                      [prec_grid, rec_grid, f1_grid_m]):
        # pcolormesh: x=ratio, y=metric value → but we need (x=ratio, y=threshold, color=metric)
        # Actually: plot as colored scatter/mesh with x=ratio, y=metric, color=threshold
        # Use LineCollection for continuous color along each ratio column
        from matplotlib.collections import LineCollection

        # Draw as colored lines — one line per threshold, colored by threshold
        for ti in range(0, n_t, 2):  # every 2nd for density
            t = thresh_fine[ti]
            ax.plot(x, grid[ti, :], color=t_cmap(t), lw=1.0, alpha=0.6)

        # Highlight key thresholds
        bbox = dict(boxstyle='round,pad=0.15', facecolor='white',
                    edgecolor='none', alpha=0.9)
        for t_mark in [0.5, 0.7, 0.9]:
            ti_mark = np.argmin(np.abs(thresh_fine - t_mark))
            color = t_cmap(t_mark)
            ax.plot(x, grid[ti_mark, :], color=color, lw=3, alpha=1.0, zorder=5)
            ax.text(x[-1] + 0.15, grid[ti_mark, -1], f't={t_mark}',
                    fontsize=9, fontweight='bold', color=color, va='center', bbox=bbox)

        # Fill between consecutive thresholds with color gradient
        for ti in range(0, n_t - 1, 1):
            t_mid = (thresh_fine[ti] + thresh_fine[ti + 1]) / 2
            ax.fill_between(x, grid[ti, :], grid[ti + 1, :],
                            color=t_cmap(t_mid), alpha=0.15)

        ax.axvline(x[0], color="grey", lw=1, ls=":", alpha=0.5)
        ax.set_ylabel(metric_name, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=11)

        # Remove shared edges between stacked panels
        idx = list(axes).index(ax)
        if idx < len(axes) - 1:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', labelbottom=False)
        if idx > 0:
            ax.spines['top'].set_visible(False)
        if idx == len(axes) - 1:
            ax.set_xlabel("Training STAR:GALAXY ratio", fontsize=12, fontweight="bold")

    # Shared threshold colorbar
    sm = plt.cm.ScalarMappable(cmap=t_cmap, norm=plt.Normalize(0.3, 1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), pad=0.02, shrink=0.8)
    cbar.set_label("P(gal) threshold", fontsize=12, fontweight='bold')

    fig.suptitle("Metrics vs training ratio — colored by classification threshold",
                 fontsize=14, fontweight="bold", y=1.02)
    out = output_dir / "mix_ratio_sweep_thresholds.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyse Q1+ERO mixed models across ratios")
    parser.add_argument("--no-threshold", action="store_true",
                        help="Skip threshold and recalibration analysis")
    parser.add_argument("--dir", type=str, default="xgboost_mix_q1_ero",
                        help="Output directory name under output/ (default: xgboost_mix_q1_ero)")
    parser.add_argument("--hilat", action="store_true",
                        help="Use high-latitude Gaia sample instead of random")
    args = parser.parse_args()

    global MIX_DIR
    MIX_DIR = PROJECT_DIR / "output" / args.dir

    feat_names = list(SELECTED_GAIA_PARAMETERS)

    # ── 1. Load metrics from all trained ratios ──────────────────────────────
    print(f"[1] Scanning for trained ratio models in {MIX_DIR}...")
    df_all = load_all_ratios()
    if df_all is None:
        print("ERROR: No ratio models found in output/xgboost_mix_q1_ero/ratio_sg*/")
        print("Run train_xgboost_mix_q1_ero.py --target-ratio N first.")
        return 1

    ratios = sorted(df_all["ratio"].unique())
    print(f"  Found {len(ratios)} ratios: {ratios}")
    print(f"  Total seed entries: {len(df_all)}")

    # ── 2. Compute summary ───────────────────────────────────────────────────
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

    # Print table
    print(f"\n  {'Ratio':>7} {'AUC':>10} {'Gal_P':>10} {'Gal_R':>10} "
          f"{'Gal_F1':>10} {'Star_F1':>10}")
    print("  " + "-" * 65)
    for _, r in df_summary.iterrows():
        print(f"  {r['ratio']:>6.1f}:1 {r['roc_auc_mean']:>9.4f} "
              f"{r['galaxy_precision_mean']:>9.4f} "
              f"{r['galaxy_recall_mean']:>9.4f} "
              f"{r['galaxy_f1_mean']:>9.4f} "
              f"{r['star_f1_mean']:>9.4f}")

    # ── 3. Sweep plot ────────────────────────────────────────────────────────
    print(f"\n[2] Plotting sweep...")
    plot_sweep(df_summary, MIX_DIR)

    if args.no_threshold:
        return 0

    # ── 4. Threshold + recalibration analysis ────────────────────────────────
    print(f"\n[3] Threshold and recalibration analysis...")

    # Load Gaia sample + Q1 test data
    if args.hilat:
        gaia_npz = ARRAYS_DIR / "gaia_hilat.npz"
        sample_label = "high-lat"
    else:
        gaia_npz = ARRAYS_DIR / "gaia_random.npz"
        sample_label = "random"
    q1_npz = ARRAYS_DIR / "q1_indiv.npz"
    if not gaia_npz.exists() or not q1_npz.exists():
        print(f"  WARNING: {gaia_npz.name} or q1_indiv.npz not found, skipping")
        return 0

    # Save to subfolder if hilat
    if args.hilat:
        save_dir = MIX_DIR / "hilat"
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = MIX_DIR

    g = np.load(gaia_npz, allow_pickle=True)
    X_gaia = g["X_features"].astype(float)
    print(f"  Loaded {len(X_gaia):,} {sample_label} Gaia sources")

    q1 = np.load(q1_npz, allow_pickle=True)
    X_q1, y_q1 = q1["X_raw"], q1["y"]
    print(f"  Q1 data: {len(X_q1):,} sources (per-seed test splits used)")

    thresholds = [0.5, 0.7, 0.9]
    # Selected ratios for recalibration plot only (less clutter)
    recal_display_ratios = {3.0, 5.0, max(ratios)}  # 3:1, 5:1, all-ERO
    thresh_results = []
    all_pgal = []

    for ratio in ratios:
        seed_dir = None
        for d in MIX_DIR.glob("ratio_*"):
            if "sweep" in d.name:
                continue
            try:
                num_str = d.name.replace("ratio_sg", "").split("_")[0]
                r = float(num_str)
                if abs(r - ratio) < 0.1:
                    seed_dir = d / "seed_models"
                    break
            except ValueError:
                continue

        if seed_dir is None or not seed_dir.exists():
            print(f"  Ratio {ratio}: no seed_models dir, skipping")
            continue

        # Ensemble predict on Gaia + Q1 test
        label = f"{ratio:.0f}:1" if ratio == int(ratio) else f"{ratio:.1f}:1"
        print(f"\n  Ratio {label}...")
        seed_paths = sorted(seed_dir.glob("xgb_classifier_q1_seed*.json"))
        if not seed_paths:
            print(f"    No seed models found, skipping")
            continue

        # Load all seed models, predict on Gaia + per-seed Q1 test splits
        gaia_probas = []
        seed_preds = []  # list of (y_test, p_test) per seed
        for sp in seed_paths:
            seed_val = int(sp.stem.replace("xgb_classifier_q1_seed", ""))
            model = xgb.XGBClassifier()
            model.load_model(str(sp))
            gaia_probas.append(model.predict_proba(X_gaia)[:, 1])
            _, X_ts, _, y_ts = train_test_split(
                X_q1, y_q1, test_size=0.15, random_state=seed_val, stratify=y_q1)
            seed_preds.append((y_ts, model.predict_proba(X_ts)[:, 1]))
        p_gaia = np.mean(gaia_probas, axis=0)
        all_pgal.append((label, p_gaia))

        # Per-seed TPR/FPR on Q1 test set (no leakage)
        for t in thresholds:
            tp_sum, fp_sum, fn_sum, tn_sum = 0, 0, 0, 0
            for y_ts, p_ts in seed_preds:
                pred_gal = p_ts >= t
                tp_sum += (pred_gal & (y_ts == 1)).sum()
                fp_sum += (pred_gal & (y_ts == 0)).sum()
                fn_sum += (~pred_gal & (y_ts == 1)).sum()
                tn_sum += (~pred_gal & (y_ts == 0)).sum()
            n = len(seed_preds)
            tp = tp_sum / n
            fp = fp_sum / n
            fn = fn_sum / n
            tn = tn_sum / n
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            n_gaia_above = int((p_gaia >= t).sum())
            pct_gaia = 100 * n_gaia_above / len(p_gaia)
            purity_1 = tpr * 0.01 / (tpr * 0.01 + fpr * 0.99 + 1e-12)
            purity_3 = tpr * 0.03 / (tpr * 0.03 + fpr * 0.97 + 1e-12)

            thresh_results.append({
                "ratio": ratio, "ratio_label": label,
                "threshold": t, "tpr": tpr, "fpr": fpr,
                "n_gaia": n_gaia_above, "pct_gaia": pct_gaia,
                "purity_1pct": purity_1, "purity_3pct": purity_3,
            })

        print(f"    P(gal)>0.5: {(p_gaia>=0.5).sum():,} ({100*(p_gaia>=0.5).mean():.1f}%)  "
              f"P(gal)>0.9: {(p_gaia>=0.9).sum():,} ({100*(p_gaia>=0.9).mean():.1f}%)")

    df_thresh = pd.DataFrame(thresh_results)
    df_thresh.to_csv(save_dir / "mix_threshold_analysis.csv", index=False)
    print(f"\n  -> {MIX_DIR / 'mix_threshold_analysis.csv'}")

    # ── P(gal) overlay plot ──────────────────────────────────────────────────
    if all_pgal:
        colors = cm.viridis(np.linspace(0, 0.9, len(all_pgal)))

        fig, ax1 = plt.subplots(figsize=(10, 5.5))

        bins = np.linspace(0, 1, 101)
        t_range = np.linspace(0, 1, 501)

        # Left y-axis: histogram
        for (label, p_gaia), color in zip(all_pgal, colors):
            ax1.hist(p_gaia, bins=bins, alpha=0.3, color=color,
                     histtype='stepfilled', linewidth=0)
            ax1.hist(p_gaia, bins=bins, color=color,
                     histtype='step', linewidth=1.2, label=label)

        ax1.set_yscale("log")
        ax1.set_ylim(0.5, 2e4)
        ax1.set_xlabel("P(gal)", fontsize=13, fontweight='bold')
        ax1.set_ylabel("Source count", fontsize=13, fontweight='bold',
                        color='#2196F3')
        ax1.tick_params(axis='y', labelcolor='#2196F3', labelsize=11)
        ax1.tick_params(axis='x', labelsize=11)
        ax1.grid(alpha=0.3)

        # Right y-axis: fraction curves
        ax2 = ax1.twinx()
        for (label, p_gaia), color in zip(all_pgal, colors):
            counts = np.array([(p_gaia >= t).sum() for t in t_range])
            frac = counts / len(X_gaia) * 100
            ax2.plot(t_range, frac, color=color, lw=1.5)

        ax2.set_yscale("log")
        ax2.set_ylim(0.05, 100)
        ax2.set_ylabel("Fraction above threshold (%)", fontsize=13,
                        fontweight='bold', color='#E53935')
        ax2.set_yticks([0.1, 0.5, 1, 2, 5, 10, 20, 50, 100])
        ax2.set_yticklabels(["0.1", "0.5", "1", "2", "5", "10", "20", "50", "100"])
        ax2.tick_params(axis='y', labelcolor='#E53935', labelsize=11)

        # Threshold markers
        for t, col, ls in [(0.5, 'black', '--'), (0.9, 'green', '--')]:
            ax1.axvline(t, color=col, lw=1.5, ls=ls, alpha=0.5)

        ax1.legend(fontsize=10, loc='upper right')

        fig.suptitle("P(gal) on Gaia sources — effect of ERO mixing ratio",
                     fontsize=13, fontweight='bold', y=0.98)
        out = save_dir / "mix_threshold_pgal_overlay.png"
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  -> {out}")

    # ── Bayesian recalibration plot (2 panels: curves + 2D heatmap) ─────────
    if len(df_thresh) > 0:
        from matplotlib.colors import LogNorm as _LN
        pi_range = np.logspace(-4, -0.3, 500)
        ratio_labels = list(dict.fromkeys(df_thresh["ratio_label"]))
        n_ratios = len(ratio_labels)
        cmap_ratio = cm.viridis(np.linspace(0, 0.9, n_ratios))

        # Compute Q1-only baseline TPR/FPR (per-seed, no leakage)
        q1_only_dir = PROJECT_DIR / "output" / "xgboost_q1" / "seed_models"
        q1_baseline_curves = {}
        if q1_only_dir.exists():
            q1_seed_paths = sorted(q1_only_dir.glob("xgb_classifier_q1_seed*.json"))
            if q1_seed_paths:
                for t in thresholds:
                    tp_s, fp_s, fn_s, tn_s = 0, 0, 0, 0
                    for sp in q1_seed_paths:
                        seed_val = int(sp.stem.replace("xgb_classifier_q1_seed", ""))
                        _, X_ts, _, y_ts = train_test_split(
                            X_q1, y_q1, test_size=0.15, random_state=seed_val, stratify=y_q1)
                        m = xgb.XGBClassifier()
                        m.load_model(str(sp))
                        pred_gal = m.predict_proba(X_ts)[:, 1] >= t
                        tp_s += (pred_gal & (y_ts == 1)).sum()
                        fp_s += (pred_gal & (y_ts == 0)).sum()
                        fn_s += (~pred_gal & (y_ts == 1)).sum()
                        tn_s += (~pred_gal & (y_ts == 0)).sum()
                    n = len(q1_seed_paths)
                    q1_baseline_curves[t] = (tp_s/n / max(tp_s/n + fn_s/n, 1),
                                             fp_s/n / max(fp_s/n + tn_s/n, 1))
                print("  Loaded Q1-only baseline for comparison")

        fig, axes_r = plt.subplots(1, 2, figsize=(18, 7))
        fig.subplots_adjust(wspace=0.3)

        # [0] Purity curves — only selected ratios for clarity
        ax = axes_r[0]
        recal_labels = [rl for rl in ratio_labels
                        if any(abs(float(rl.split(":")[0]) - dr) < 0.1
                               for dr in recal_display_ratios)]
        recal_colors = cm.viridis(np.linspace(0, 0.9, len(recal_labels)))
        for i, rlabel in enumerate(recal_labels):
            sub = df_thresh[df_thresh["ratio_label"] == rlabel]
            color = recal_colors[i]
            for _, r in sub.iterrows():
                t = r["threshold"]
                tpr, fpr = r["tpr"], r["fpr"]
                if fpr == 0:
                    continue
                prec = tpr * pi_range / (tpr * pi_range + fpr * (1 - pi_range) + 1e-12)
                ls = '-' if t == 0.5 else ('--' if t == 0.7 else ':')
                lbl = f"{rlabel}, t={t}" if t == 0.9 else '_'
                ax.plot(pi_range * 100, prec * 100, color=color,
                        lw=2, ls=ls, alpha=0.8, label=lbl)

        # Plot Q1-only baseline (red)
        for t, (tpr, fpr) in q1_baseline_curves.items():
            if fpr == 0:
                continue
            prec = tpr * pi_range / (tpr * pi_range + fpr * (1 - pi_range) + 1e-12)
            ls = '-' if t == 0.5 else ('--' if t == 0.7 else ':')
            lbl = f"Q1-only, t={t}" if t == 0.9 else '_'
            ax.plot(pi_range * 100, prec * 100, color='red',
                    lw=2.5, ls=ls, alpha=0.9, label=lbl)

        # Annotate purity at 3% only (less clutter), for Q1 + each ratio at t=0.9
        # Collect all annotations first, then place with staggered offsets
        annot_items = []  # (pi_mark, purity, color, label)

        if 0.9 in q1_baseline_curves:
            tpr_q1, fpr_q1 = q1_baseline_curves[0.9]
            p3 = tpr_q1 * 0.03 / (tpr_q1 * 0.03 + fpr_q1 * 0.97 + 1e-12)
            annot_items.append((0.03, p3, 'red', f"Q1: {p3:.0%}"))

        for i, rlabel in enumerate(recal_labels):
            sub = df_thresh[(df_thresh["ratio_label"] == rlabel) &
                            (df_thresh["threshold"] == 0.9)]
            if len(sub) == 0:
                continue
            r = sub.iloc[0]
            tpr, fpr = r["tpr"], r["fpr"]
            color = recal_colors[i]
            p3 = tpr * 0.03 / (tpr * 0.03 + fpr * 0.97 + 1e-12)
            annot_items.append((0.03, p3, color, f"{rlabel}: {p3:.0%}"))

        # Sort by purity and stagger offsets vertically
        annot_items.sort(key=lambda x: x[1])
        y_offsets = [-18, -6, 6, 18, 30]  # stagger for up to 5 items
        for idx, (pi_mark, purity, color, lbl) in enumerate(annot_items):
            ax.scatter(pi_mark * 100, purity * 100, s=40, zorder=5,
                       color=color, edgecolors='k', linewidths=0.5)
            oy = y_offsets[idx] if idx < len(y_offsets) else 6
            ax.annotate(lbl, xy=(pi_mark * 100, purity * 100),
                        xytext=(-8, oy), textcoords="offset points",
                        fontsize=10, color=color, fontweight="bold",
                        ha="right")

        ax.axhline(50, color="grey", lw=1, ls=":", alpha=0.5)
        ax.axvspan(1, 3, color="steelblue", alpha=0.12, zorder=0,
                   label="Est. Gaia gal. fraction (1\u20133%)")
        ax.set_xlabel("True galaxy fraction (%)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Galaxy selection purity (%)", fontsize=11, fontweight="bold")
        ax.set_title("Purity vs galaxy fraction\n"
                     "(solid=0.5, dashed=0.7, dotted=0.9)",
                     fontsize=11, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xlim(0.01, 50); ax.set_ylim(0, 100)
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

        # [1] 2D heatmap: purity as f(galaxy fraction, threshold)
        # Use the highest-ratio model (most conservative)
        best_ratio_label = ratio_labels[-1]
        best_sub = df_thresh[df_thresh["ratio_label"] == best_ratio_label]
        # Build from all thresholds in the continuous range
        thresh_grid = np.linspace(0.3, 0.99, 70)
        pi_grid = np.logspace(-2.5, -0.3, 80)
        purity_2d = np.zeros((len(thresh_grid), len(pi_grid)))

        # Re-predict TPR/FPR at fine thresholds for the best ratio
        # Use the saved ensemble test predictions
        best_ratio = ratios[-1]
        best_seed_dir = None
        for d in MIX_DIR.glob("ratio_*"):
            if "sweep" in d.name:
                continue
            try:
                rv = float(d.name.replace("ratio_sg", "").split("_")[0])
                if abs(rv - best_ratio) < 0.01:
                    best_seed_dir = d / "seed_models"
                    break
            except ValueError:
                continue

        if best_seed_dir and best_seed_dir.exists():
            seed_paths = sorted(best_seed_dir.glob("xgb_classifier_q1_seed*.json"))
            # Pre-compute per-seed predictions (load each model once)
            seed_preds = []  # list of (y_test, p_test) per seed
            for sp in seed_paths:
                seed_val = int(sp.stem.replace("xgb_classifier_q1_seed", ""))
                _, X_ts, _, y_ts = train_test_split(
                    X_q1, y_q1, test_size=0.15, random_state=seed_val, stratify=y_q1)
                m = xgb.XGBClassifier()
                m.load_model(str(sp))
                seed_preds.append((y_ts, m.predict_proba(X_ts)[:, 1]))

            for i, t in enumerate(thresh_grid):
                tp_s, fp_s, fn_s, tn_s = 0, 0, 0, 0
                for y_ts, p_ts in seed_preds:
                    pred_gal = p_ts >= t
                    tp_s += (pred_gal & (y_ts == 1)).sum()
                    fp_s += (pred_gal & (y_ts == 0)).sum()
                    fn_s += (~pred_gal & (y_ts == 1)).sum()
                    tn_s += (~pred_gal & (y_ts == 0)).sum()
                n = len(seed_preds)
                tpr = (tp_s/n) / max(tp_s/n + fn_s/n, 1)
                fpr = (fp_s/n) / max(fp_s/n + tn_s/n, 1)
                for j, pi in enumerate(pi_grid):
                    purity_2d[i, j] = tpr * pi / (tpr * pi + fpr * (1 - pi) + 1e-12)

        ax = axes_r[1]
        im = ax.pcolormesh(pi_grid * 100, thresh_grid, purity_2d * 100,
                           cmap='RdYlGn', shading='auto', vmin=0, vmax=100)
        cs = ax.contour(pi_grid * 100, thresh_grid, purity_2d * 100,
                        levels=[25, 50, 75, 90], colors='black',
                        linewidths=0.8, linestyles='--')
        ax.clabel(cs, fmt='%d%%', fontsize=8)
        ax.axvspan(1, 3, color='white', alpha=0.15, zorder=2)
        ax.axvline(1, color='white', lw=0.8, ls=':', alpha=0.5)
        ax.axvline(3, color='white', lw=0.8, ls=':', alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlim(pi_grid.min() * 100, pi_grid.max() * 100)
        ax.set_xlabel("True galaxy fraction (%)", fontsize=11, fontweight="bold")
        ax.set_ylabel("P(gal) threshold", fontsize=11, fontweight="bold")
        ax.set_title(f"Galaxy selection purity (%) — ratio {best_ratio_label}",
                     fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, pad=0.02, label="Purity (%)")

        fig.suptitle("Bayesian recalibration — ERO mixing ratios",
                     fontsize=13, fontweight="bold", y=0.98)
        out = save_dir / "mix_threshold_recalibration.png"
        fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  -> {out}")

    # ── Purity vs ratio plot (at 1% and 3% galaxy fraction) ─────────────────
    if len(df_thresh) > 0 and q1_baseline_curves:
        fig_pv, ax_pv = plt.subplots(figsize=(8, 5))

        # Collect purity at 1% and 3% for each ratio at t=0.9
        ratio_labels_plot = []
        purity_1_vals = []
        purity_3_vals = []

        # Q1-only first
        if 0.9 in q1_baseline_curves:
            tpr_q1, fpr_q1 = q1_baseline_curves[0.9]
            p1 = tpr_q1 * 0.01 / (tpr_q1 * 0.01 + fpr_q1 * 0.99 + 1e-12)
            p3 = tpr_q1 * 0.03 / (tpr_q1 * 0.03 + fpr_q1 * 0.97 + 1e-12)
            ratio_labels_plot.append("Q1\nonly")
            purity_1_vals.append(p1 * 100)
            purity_3_vals.append(p3 * 100)

        for rlabel in ratio_labels:
            sub = df_thresh[(df_thresh["ratio_label"] == rlabel) &
                            (df_thresh["threshold"] == 0.9)]
            if len(sub) == 0:
                continue
            r = sub.iloc[0]
            tpr, fpr = r["tpr"], r["fpr"]
            p1 = tpr * 0.01 / (tpr * 0.01 + fpr * 0.99 + 1e-12)
            p3 = tpr * 0.03 / (tpr * 0.03 + fpr * 0.97 + 1e-12)
            ratio_labels_plot.append(rlabel.replace(":1", "\n:1"))
            purity_1_vals.append(p1 * 100)
            purity_3_vals.append(p3 * 100)

        x_pos = np.arange(len(ratio_labels_plot))
        width = 0.35

        bars1 = ax_pv.bar(x_pos - width/2, purity_1_vals, width,
                          color="#E53935", alpha=0.85, edgecolor="white",
                          label="at 1% galaxy fraction")
        bars3 = ax_pv.bar(x_pos + width/2, purity_3_vals, width,
                          color="#4CAF50", alpha=0.85, edgecolor="white",
                          label="at 3% galaxy fraction")

        # Annotate values
        for bar in list(bars1) + list(bars3):
            h = bar.get_height()
            ax_pv.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                       f"{h:.0f}%", ha="center", va="bottom",
                       fontsize=9, fontweight="bold")

        ax_pv.set_xticks(x_pos)
        ax_pv.set_xticklabels(ratio_labels_plot, fontsize=10, fontweight="bold")
        ax_pv.set_ylabel("Galaxy selection purity (%)", fontsize=12, fontweight="bold")
        ax_pv.set_xlabel("Training STAR:GALAXY ratio", fontsize=12, fontweight="bold")
        ax_pv.set_title("Recalibrated purity at threshold 0.9\n"
                        "for estimated Gaia galaxy fractions",
                        fontsize=12, fontweight="bold")
        ax_pv.set_ylim(0, 100)
        ax_pv.legend(fontsize=10)
        ax_pv.grid(alpha=0.3, axis="y")

        out_pv = save_dir / "mix_purity_vs_ratio.png"
        fig_pv.savefig(out_pv, dpi=150, bbox_inches="tight")
        plt.close(fig_pv)
        print(f"  -> {out_pv}")

    # ── Comprehensive heatmap: purity(threshold, ratio) at different fractions ─
    if len(df_thresh) > 0 and len(all_pgal) >= 2:
        print("\n  Computing comprehensive purity heatmap...")
        thresh_fine = np.linspace(0.3, 0.99, 50)
        ratio_vals = []
        ratio_labels_hm = []
        tpr_grid = np.zeros((len(all_pgal), len(thresh_fine)))
        fpr_grid = np.zeros((len(all_pgal), len(thresh_fine)))

        # Re-use seed_preds from each ratio (recompute since not stored)
        for ri, ratio in enumerate(ratios):
            seed_dir = None
            for d in MIX_DIR.glob("ratio_*"):
                if "sweep" in d.name:
                    continue
                try:
                    num_str = d.name.replace("ratio_sg", "").split("_")[0]
                    r = float(num_str)
                    if abs(r - ratio) < 0.1:
                        seed_dir = d / "seed_models"
                        break
                except ValueError:
                    continue
            if seed_dir is None or not seed_dir.exists():
                continue

            seed_paths = sorted(seed_dir.glob("xgb_classifier_q1_seed*.json"))
            if not seed_paths:
                continue

            # Per-seed predictions
            sp_list = []
            for sp in seed_paths:
                seed_val = int(sp.stem.replace("xgb_classifier_q1_seed", ""))
                _, X_ts, _, y_ts = train_test_split(
                    X_q1, y_q1, test_size=0.15, random_state=seed_val, stratify=y_q1)
                m = xgb.XGBClassifier()
                m.load_model(str(sp))
                sp_list.append((y_ts, m.predict_proba(X_ts)[:, 1]))

            label = f"{ratio:.0f}:1" if ratio == int(ratio) else f"{ratio:.1f}:1"
            ratio_vals.append(ratio)
            ratio_labels_hm.append(label)

            for ti, t in enumerate(thresh_fine):
                tp_s, fp_s, fn_s, tn_s = 0, 0, 0, 0
                for y_ts, p_ts in sp_list:
                    pred_gal = p_ts >= t
                    tp_s += (pred_gal & (y_ts == 1)).sum()
                    fp_s += (pred_gal & (y_ts == 0)).sum()
                    fn_s += (~pred_gal & (y_ts == 1)).sum()
                    tn_s += (~pred_gal & (y_ts == 0)).sum()
                n = len(sp_list)
                tpr_grid[ri, ti] = (tp_s/n) / max(tp_s/n + fn_s/n, 1)
                fpr_grid[ri, ti] = (fp_s/n) / max(fp_s/n + tn_s/n, 1)

        # Build purity heatmaps for different galaxy fractions
        gal_fractions = [0.005, 0.01, 0.03, 0.05]
        frac_labels = ["0.5%", "1%", "3%", "5%"]
        n_ratios_hm = len(ratio_vals)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.subplots_adjust(hspace=0.3, wspace=0.25)

        for ax, pi, flabel in zip(axes.flat, gal_fractions, frac_labels):
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
            ax.clabel(cs, fmt='%d%%', fontsize=8)

            # Annotate purity values at key thresholds
            bbox = dict(boxstyle='round,pad=0.15', facecolor='white',
                        edgecolor='none', alpha=0.8)
            for t_mark in [0.5, 0.7, 0.9]:
                ti_mark = np.argmin(np.abs(thresh_fine - t_mark))
                ax.axvline(t_mark, color='white', lw=0.5, ls=':', alpha=0.5)
                for ri in range(n_ratios_hm):
                    val = purity[ri, ti_mark] * 100
                    color = 'white' if val < 50 else 'black'
                    ax.text(t_mark, ri, f'{val:.0f}', ha='center', va='center',
                            fontsize=7, fontweight='bold', color=color, bbox=bbox)

            ax.set_yticks(range(n_ratios_hm))
            ax.set_yticklabels(ratio_labels_hm, fontsize=10)
            ax.set_xlabel("P(gal) threshold", fontsize=11, fontweight='bold')
            ax.set_ylabel("Training ratio", fontsize=11, fontweight='bold')
            ax.set_title(f"Purity at {flabel} true galaxy fraction",
                         fontsize=12, fontweight='bold')
            fig.colorbar(im, ax=ax, pad=0.02, label="Purity (%)")

        fig.suptitle("Galaxy selection purity — ratio × threshold × galaxy fraction",
                     fontsize=14, fontweight='bold', y=0.98)
        out_hm = save_dir / "mix_purity_comprehensive.png"
        fig.savefig(out_hm, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  -> {out_hm}")

        # ── Sweep plot colored by threshold ───────────────────────────────────
        plot_sweep_thresholds(ratio_vals, ratio_labels_hm, tpr_grid, fpr_grid,
                              thresh_fine, df_summary, save_dir)

        # ── 2D heatmaps: purity(galaxy_fraction, threshold) per ratio ─────────
        # Like the existing single-ratio heatmap but tiled for selected ratios,
        # with a delta row showing improvement over Q1-only baseline.
        pi_hm = np.logspace(-2.5, -0.3, 80)  # galaxy fractions
        # All ratios: top = absolute purity, bottom = delta vs Q1-only
        # First column in bottom row = recall vs threshold (line plot, no galaxy fraction axis)
        from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
        delta_cmap = LinearSegmentedColormap.from_list(
            'orange_white_purple',
            ['#EF7627', '#FF9A56', '#FFFFFF', '#D162A4', '#8B1A6B'])

        # Compute purity maps for all ratios
        purity_maps = {}
        for ri in range(n_ratios_hm):
            purity_2d = np.zeros((len(thresh_fine), len(pi_hm)))
            for ti in range(len(thresh_fine)):
                tpr = tpr_grid[ri, ti]
                fpr = fpr_grid[ri, ti]
                purity_2d[ti, :] = tpr * pi_hm / (tpr * pi_hm + fpr * (1 - pi_hm) + 1e-12)
            purity_maps[ri] = purity_2d

        # Compute deltas vs Q1-only (index 0) and find global range
        deltas = {}
        for ri in range(1, n_ratios_hm):
            deltas[ri] = (purity_maps[ri] - purity_maps[0]) * 100
        all_delta_vals = np.concatenate([d.ravel() for d in deltas.values()])
        global_dmin = all_delta_vals.min()
        global_dmax = all_delta_vals.max()
        delta_norm = TwoSlopeNorm(vmin=global_dmin, vcenter=0, vmax=global_dmax)

        # Layout: n_ratios columns, 2 rows
        # Bottom row col 0 = recall line plot, rest = delta heatmaps
        fig, axes = plt.subplots(2, n_ratios_hm, figsize=(4.5 * n_ratios_hm, 9))
        fig.subplots_adjust(hspace=0.3, wspace=0.35)

        # Row 0: absolute purity heatmaps
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
            ax.set_title(f"Ratio {ratio_labels_hm[ri]}", fontsize=11, fontweight='bold')
            fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85, label="Purity (%)")

        # Row 1, col 0: recall vs threshold (line plot for all ratios)
        ax = axes[1, 0]
        colors_recall = cm.viridis(np.linspace(0, 0.9, n_ratios_hm))
        for ri in range(n_ratios_hm):
            ax.plot(thresh_fine, tpr_grid[ri, :] * 100, color=colors_recall[ri],
                    lw=1.5, label=ratio_labels_hm[ri])
        ax.set_xlabel("P(gal) threshold", fontsize=9, fontweight='bold')
        ax.set_ylabel("Recall (%)", fontsize=10, fontweight='bold')
        ax.set_title("Recall vs threshold", fontsize=11, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.legend(fontsize=7, loc='upper right', ncol=2)
        ax.grid(alpha=0.3)

        # Row 1, col 1+: delta purity heatmaps
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
            ax.set_title(f"Purity gain vs Q1-only", fontsize=11, fontweight='bold')
            fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85, label="Purity gain (%)")

        fig.suptitle("Purity landscape — all ratios\n"
                     "(top: absolute purity, bottom: recall + purity gain over Q1-only)",
                     fontsize=14, fontweight='bold', y=1.0)
        out_land = save_dir / "mix_purity_landscape.png"
        fig.savefig(out_land, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  -> {out_land}")

    # Print threshold summary
    if len(df_thresh) > 0:
        print(f"\n  {'Ratio':<10} {'Thresh':>6} {'TPR':>7} {'FPR':>7} "
              f"{'N_Gaia':>8} {'%Gaia':>6} {'Pur@1%':>7} {'Pur@3%':>7}")
        print("  " + "-" * 70)
        for _, r in df_thresh.iterrows():
            print(f"  {r['ratio_label']:<10} {r['threshold']:>6.1f} "
                  f"{r['tpr']:>7.4f} {r['fpr']:>7.4f} "
                  f"{int(r['n_gaia']):>8,} {r['pct_gaia']:>5.1f}% "
                  f"{r['purity_1pct']:>6.1%} {r['purity_3pct']:>6.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
