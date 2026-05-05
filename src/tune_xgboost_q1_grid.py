#!/usr/bin/env python3
"""
Hyperparameter tuning for XGBoost Q1 star/galaxy classifier
on the FULL 238k dataset (queried from WSDB).

Uses scikit-learn GridSearchCV with 5-fold stratified CV, tracking
multiple metrics simultaneously:
  - ROC AUC, F1 (weighted), Average Precision
  - Per-class: Galaxy precision/recall/F1, Star precision/recall/F1

Outputs saved to output/xgboost_q1_tuning/:
  hyperparam_grid_results.csv     — full grid results (all metrics)
  hyperparam_heatmap_*.png        — heatmaps per metric
  hyperparam_metric_comparison.png — top configs across all metrics
  best_hyperparams.json           — best params (by ROC AUC)

Usage:
    PGUSER=siwei_zhao_2025 python src/tune_xgboost_q1_full.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR  = PROJECT_DIR / "output" / "xgboost_q1_tuning_grid"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
CLASS_MAPPING = {"STAR": 0, "GALAXY": 1}
MAG_CUT       = 17
MAG_CUT_FAINT = 99

# ── Parameter grid ──────────────────────────────────────────────
PARAM_GRID = {
    "max_depth":      [4, 6, 8, 10],
    "n_estimators":   [100, 200, 500],
    "learning_rate":  [0.01, 0.05, 0.1, 0.2],
}

FIXED_PARAMS = {
    "objective":        "binary:logistic",
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "tree_method":      "hist",
    "eval_metric":      "logloss",
    "random_state":     42,
}

# ── Multi-metric scoring ───────────────────────────────────────
# Galaxy = class 1 (positive), Star = class 0
SCORING = {
    "roc_auc":           "roc_auc",
    "average_precision": "average_precision",
    "f1_weighted":       "f1_weighted",
    # Galaxy (class 1) metrics
    "galaxy_precision":  make_scorer(precision_score, pos_label=1, zero_division=0),
    "galaxy_recall":     make_scorer(recall_score,    pos_label=1, zero_division=0),
    "galaxy_f1":         make_scorer(f1_score,        pos_label=1, zero_division=0),
    # Star (class 0) metrics
    "star_precision":    make_scorer(precision_score, pos_label=0, zero_division=0),
    "star_recall":       make_scorer(recall_score,    pos_label=0, zero_division=0),
    "star_f1":           make_scorer(f1_score,        pos_label=0, zero_division=0),
}

# Display names for plots
METRIC_LABELS = {
    "roc_auc":           "ROC AUC",
    "average_precision": "Average Precision",
    "f1_weighted":       "F1 (weighted)",
    "galaxy_precision":  "Galaxy Precision",
    "galaxy_recall":     "Galaxy Recall",
    "galaxy_f1":         "Galaxy F1",
    "star_precision":    "Star Precision",
    "star_recall":       "Star Recall",
    "star_f1":           "Star F1",
}


# ── Data loading ────────────────────────────────────────────────

def load_from_db(username: str):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    feat_names = list(SELECTED_GAIA_PARAMETERS)
    gaia_cols = ", ".join([f"g_{col} AS {col}" for col in feat_names])
    query = f"""
        SELECT
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
    print("  Querying WSDB...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    for col in feat_names:
        df[col] = pd.to_numeric(df[col].replace("NOT_AVAILABLE", np.nan),
                                errors="coerce")
    X = df[feat_names].values.astype(float)
    y = df["label"].map(CLASS_MAPPING).values
    return X, y, feat_names


# ── Plotting ────────────────────────────────────────────────────

def plot_heatmap(results_df, metric_key, output_dir):
    """Heatmap for one metric across the grid."""
    col = f"mean_test_{metric_key}"
    label = METRIC_LABELS[metric_key]

    lrs = sorted(results_df["param_learning_rate"].unique())
    depths = sorted(results_df["param_max_depth"].unique())
    n_ests = sorted(results_df["param_n_estimators"].unique())

    n_lr = len(lrs)
    fig, axes = plt.subplots(1, n_lr, figsize=(4.5 * n_lr, 4))
    if n_lr == 1:
        axes = [axes]

    vmin = results_df[col].min()
    vmax = results_df[col].max()

    for ax, lr in zip(axes, lrs):
        sub = results_df[results_df["param_learning_rate"] == lr]
        grid = np.zeros((len(depths), len(n_ests)))
        for i, d in enumerate(depths):
            for j, n in enumerate(n_ests):
                row = sub[(sub["param_max_depth"] == d) &
                          (sub["param_n_estimators"] == n)]
                if len(row) > 0:
                    grid[i, j] = row[col].values[0]

        im = ax.imshow(grid, cmap="RdYlGn", aspect="auto",
                       vmin=vmin, vmax=vmax, origin="lower")
        for i in range(len(depths)):
            for j in range(len(n_ests)):
                v = grid[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=7.5, fontweight="bold",
                        color="white" if v < (vmin + vmax) / 2 else "black")
        ax.set_xticks(range(len(n_ests)))
        ax.set_xticklabels(n_ests)
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels(depths)
        ax.set_xlabel("n_estimators", fontsize=10)
        ax.set_ylabel("max_depth", fontsize=10)
        ax.set_title(f"lr = {lr}", fontsize=10, fontweight="bold")

    fig.suptitle(f"{label} — 5-fold CV on 238k sources",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.colorbar(im, ax=axes, label=label, shrink=0.8)

    out = output_dir / f"hyperparam_heatmap_{metric_key}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


def plot_metric_comparison(results_df, output_dir):
    """Bar chart: for each metric, show the best config and the original config."""
    metrics = list(METRIC_LABELS.keys())
    original = {"max_depth": 8, "n_estimators": 200, "learning_rate": 0.1}

    orig_row = results_df[
        (results_df["param_max_depth"] == original["max_depth"]) &
        (results_df["param_n_estimators"] == original["n_estimators"]) &
        (results_df["param_learning_rate"] == original["learning_rate"])
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.35

    best_vals = []
    orig_vals = []
    best_labels = []

    for m in metrics:
        col = f"mean_test_{m}"
        best_idx = results_df[col].idxmax()
        best_row = results_df.loc[best_idx]
        best_vals.append(best_row[col])
        best_labels.append(
            f"d={int(best_row['param_max_depth'])}, "
            f"n={int(best_row['param_n_estimators'])}, "
            f"lr={best_row['param_learning_rate']}"
        )
        if len(orig_row) > 0:
            orig_vals.append(orig_row.iloc[0][col])
        else:
            orig_vals.append(0)

    bars1 = ax.bar(x - width/2, best_vals, width, label="Best config",
                   color="#4CAF50", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, orig_vals, width,
                   label="Original (d=8, n=200, lr=0.1)",
                   color="#2196F3", alpha=0.85, edgecolor="white")

    # Annotate best config params
    for i, (bar, lbl) in enumerate(zip(bars1, best_labels)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                lbl, ha="center", va="bottom", fontsize=6, rotation=45,
                color="#333333")

    # Annotate values
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.015,
                f"{bar.get_height():.3f}", ha="center", va="top",
                fontsize=7.5, fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics],
                       rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title("Best tuned config vs original — all metrics",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.3, axis="y")

    out = output_dir / "hyperparam_metric_comparison.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")


def plot_summary_grid(results_df, output_dir):
    """Heatmap: Original + best-per-metric configs across all metrics."""
    col_prefix = "mean_test_"
    metric_keys = list(METRIC_LABELS.keys())
    metric_labels = list(METRIC_LABELS.values())

    def label(row, name):
        return (f"{name}\n(d={int(row['param_max_depth'])}, "
                f"n={int(row['param_n_estimators'])}, "
                f"lr={row['param_learning_rate']})")

    def row_to_dict(row):
        return {k: row[f"{col_prefix}{k}"] for k in metric_keys}

    orig = results_df[
        (results_df["param_max_depth"] == 8) &
        (results_df["param_n_estimators"] == 200) &
        (results_df["param_learning_rate"] == 0.1)
    ].iloc[0]

    configs = [(label(orig, "Original"), row_to_dict(orig))]
    for mk, ml in METRIC_LABELS.items():
        col = f"{col_prefix}{mk}"
        best = results_df.loc[results_df[col].idxmax()]
        configs.append((label(best, f"Best {ml}"), row_to_dict(best)))

    config_labels = [c[0] for c in configs]
    n_configs = len(configs)
    n_metrics = len(metric_keys)

    values = np.zeros((n_configs, n_metrics))
    for i, (_, vals) in enumerate(configs):
        for j, key in enumerate(metric_keys):
            values[i, j] = vals.get(key, 0)

    col_min = values.min(axis=0)
    col_max = values.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    normed = (values - col_min) / col_range

    fig, ax = plt.subplots(figsize=(12, 0.8 * n_configs + 2))
    im = ax.imshow(normed, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    for j in range(n_metrics):
        best_i = np.argmax(values[:, j])
        for i in range(n_configs):
            val = values[i, j]
            is_best = (i == best_i)
            color = "black" if normed[i, j] > 0.4 else "white"
            text = f"{val:.4f}"
            if is_best:
                text += " \u2605"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    fontweight="bold" if is_best else "normal", color=color)

    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(metric_labels, fontsize=10, fontweight="bold",
                       rotation=30, ha="right")
    ax.set_yticks(range(n_configs))
    ax.set_yticklabels(config_labels, fontsize=9)

    ax.set_title("Grid search tuning summary \u2014 best config per metric\n"
                 "(\u2605 = best in column, colour normalised per column)",
                 fontsize=12, fontweight="bold", pad=15)

    for i in range(n_configs + 1):
        ax.axhline(i - 0.5, color="white", lw=1.5)
    for j in range(n_metrics + 1):
        ax.axvline(j - 0.5, color="white", lw=1.5)

    fig.colorbar(im, ax=ax, label="Relative score (per column)",
                 shrink=0.8, pad=0.02)

    out = output_dir / "hyperparam_summary_grid.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")


def plot_tradeoff(results_df, output_dir):
    """Galaxy & Star precision vs recall scatter on the same axes."""
    fig, ax = plt.subplots(figsize=(9, 7))

    # Shared F1 color scale across both classes
    all_f1 = pd.concat([results_df["mean_test_galaxy_f1"],
                        results_df["mean_test_star_f1"]])
    vmin, vmax = all_f1.min(), all_f1.max()

    # ── Galaxy points (circles) ──
    sc_gal = ax.scatter(
        results_df["mean_test_galaxy_recall"],
        results_df["mean_test_galaxy_precision"],
        c=results_df["mean_test_galaxy_f1"], cmap="RdYlBu",
        vmin=vmin, vmax=vmax,
        s=55, marker="o", edgecolors="grey", linewidths=0.5, zorder=3,
        label="Galaxy")

    # ── Star points (squares) ──
    ax.scatter(
        results_df["mean_test_star_recall"],
        results_df["mean_test_star_precision"],
        c=results_df["mean_test_star_f1"], cmap="RdYlBu",
        vmin=vmin, vmax=vmax,
        s=55, marker="s", edgecolors="grey", linewidths=0.5, zorder=3,
        label="Star")

    # ── Mark original config ──
    orig = results_df[
        (results_df["param_max_depth"] == 8) &
        (results_df["param_n_estimators"] == 200) &
        (results_df["param_learning_rate"] == 0.1)
    ]
    if len(orig) > 0:
        o = orig.iloc[0]
        ax.scatter(o["mean_test_galaxy_recall"], o["mean_test_galaxy_precision"],
                   marker="*", s=250, color="red", zorder=5,
                   label="Original — Galaxy")
        ax.scatter(o["mean_test_star_recall"], o["mean_test_star_precision"],
                   marker="*", s=250, color="navy", zorder=5,
                   label="Original — Star")

    ax.set_xlabel("Recall (completeness)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision (purity)", fontsize=11, fontweight="bold")
    ax.set_title("Precision–recall trade-off across hyperparameters",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    fig.colorbar(sc_gal, ax=ax, label="F1 score")

    out = output_dir / "hyperparam_precision_recall_tradeoff.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")


# ── Load cached results ─────────────────────────────────────────

def load_results() -> pd.DataFrame:
    """Load grid search results from CSV. Raises if not found."""
    csv_path = OUTPUT_DIR / "hyperparam_grid_results.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print("Run without --plot-only first to generate results.")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} configurations from {csv_path}")
    return df


# ── Print summary ───────────────────────────────────────────────

def print_summary(results_df: pd.DataFrame):
    """Print best-per-metric table and top-10."""
    orig = results_df[
        (results_df["param_max_depth"] == 8) &
        (results_df["param_n_estimators"] == 200) &
        (results_df["param_learning_rate"] == 0.1)
    ]

    print(f"\n  Best configuration per metric:")
    print(f"  {'Metric':<22} {'Best Score':>10} {'depth':>5} {'n_est':>5} "
          f"{'lr':>5}  {'Orig Score':>10}")
    print("  " + "-" * 75)

    for metric_key, label in METRIC_LABELS.items():
        col = f"mean_test_{metric_key}"
        if col not in results_df.columns:
            continue
        best_idx = results_df[col].idxmax()
        best_row = results_df.loc[best_idx]
        orig_val = orig.iloc[0][col] if len(orig) > 0 else float("nan")
        print(f"  {label:<22} {best_row[col]:>10.5f} "
              f"{int(best_row['param_max_depth']):>5} "
              f"{int(best_row['param_n_estimators']):>5} "
              f"{best_row['param_learning_rate']:>5.2f}  "
              f"{orig_val:>10.5f}")

    print(f"\n  Top 10 by ROC AUC (with galaxy & star metrics):")
    print(f"  {'Rank':>4} {'depth':>5} {'n_est':>5} {'lr':>5}  "
          f"{'AUC':>7} {'Gal_P':>7} {'Gal_R':>7} {'Gal_F1':>7} "
          f"{'Star_P':>7} {'Star_R':>7} {'Star_F1':>7}")
    print("  " + "-" * 85)
    for _, r in results_df.head(10).iterrows():
        print(f"  {int(r['rank_test_roc_auc']):>4} "
              f"{int(r['param_max_depth']):>5} "
              f"{int(r['param_n_estimators']):>5} "
              f"{r['param_learning_rate']:>5.2f}  "
              f"{r['mean_test_roc_auc']:>7.4f} "
              f"{r['mean_test_galaxy_precision']:>7.4f} "
              f"{r['mean_test_galaxy_recall']:>7.4f} "
              f"{r['mean_test_galaxy_f1']:>7.4f} "
              f"{r['mean_test_star_precision']:>7.4f} "
              f"{r['mean_test_star_recall']:>7.4f} "
              f"{r['mean_test_star_f1']:>7.4f}")

    if len(orig) > 0:
        o = orig.iloc[0]
        print(f"\n  Original (d=8, n=200, lr=0.1):")
        print(f"    AUC={o['mean_test_roc_auc']:.4f}  "
              f"Gal P={o['mean_test_galaxy_precision']:.4f}  "
              f"Gal R={o['mean_test_galaxy_recall']:.4f}  "
              f"Gal F1={o['mean_test_galaxy_f1']:.4f}  "
              f"Star P={o['mean_test_star_precision']:.4f}  "
              f"Star R={o['mean_test_star_recall']:.4f}  "
              f"Star F1={o['mean_test_star_f1']:.4f}")


# ── Generate all plots ──────────────────────────────────────────

def generate_plots(results_df: pd.DataFrame):
    """Generate all plots from results DataFrame."""
    print(f"\n  Plotting...")
    for m in ["roc_auc", "galaxy_f1", "galaxy_precision", "galaxy_recall",
              "star_f1", "f1_weighted"]:
        col = f"mean_test_{m}"
        if col in results_df.columns:
            out = plot_heatmap(results_df, m, OUTPUT_DIR)
            print(f"  -> {out}")

    plot_metric_comparison(results_df, OUTPUT_DIR)
    plot_tradeoff(results_df, OUTPUT_DIR)
    plot_summary_grid(results_df, OUTPUT_DIR)


# ── Main ────────────────────────────────────────────────────────

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for XGBoost Q1 classifier (full dataset)"
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip tuning, load cached results from CSV and regenerate plots"
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        # ── Plot-only mode: load from CSV ───────────────────────
        print("[1] Loading cached grid search results...")
        results_df = load_results()
        print_summary(results_df)
        generate_plots(results_df)
        return 0

    # ── Full tuning mode ────────────────────────────────────────

    # 1. Load data
    print("[1] Loading full Q1 data from WSDB...")
    username = os.environ.get("PGUSER")
    if not username:
        print("ERROR: PGUSER not set. Run with:")
        print("  PGUSER=siwei_zhao_2025 python src/tune_xgboost_q1_full.py")
        print("Or use --plot-only to regenerate plots from cached results.")
        return 1

    X, y, feat_names = load_from_db(username)

    n_star   = (y == 0).sum()
    n_galaxy = (y == 1).sum()
    print(f"  {len(X):,} sources  STAR: {n_star:,} ({100*n_star/len(X):.1f}%)  "
          f"GALAXY: {n_galaxy:,} ({100*n_galaxy/len(X):.1f}%)")
    print(f"  Features: {len(feat_names)}")

    # 2. Grid search
    n_combos = 1
    for v in PARAM_GRID.values():
        n_combos *= len(v)
    print(f"\n[2] Running GridSearchCV: {n_combos} combos × 5 folds = "
          f"{n_combos * 5} fits on {len(X):,} sources...")
    print(f"  Tracking {len(SCORING)} metrics: {', '.join(SCORING.keys())}")

    estimator = xgb.XGBClassifier(**FIXED_PARAMS, verbosity=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator, PARAM_GRID,
        cv=cv,
        scoring=SCORING,
        refit="roc_auc",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    grid.fit(X, y)

    # 3. Save results
    results_df = pd.DataFrame(grid.cv_results_)
    keep = [c for c in results_df.columns
            if c.startswith("param_") or c.startswith("mean_test_") or
            c.startswith("std_test_") or c.startswith("rank_test_") or
            c.startswith("mean_train_") or c == "mean_fit_time"]
    results_df = results_df[keep].sort_values("rank_test_roc_auc")

    csv_path = OUTPUT_DIR / "hyperparam_grid_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n[3] Results saved to {csv_path}")

    print_summary(results_df)

    # 4. Plots
    print(f"\n[4] Generating plots...")
    generate_plots(results_df)

    # 5. Save best params
    best = {**FIXED_PARAMS, **grid.best_params_}
    best["_primary_metric"] = "roc_auc"
    best["_cv_score"] = float(grid.best_score_)
    best["_n_sources"] = int(len(X))
    best_path = OUTPUT_DIR / "best_hyperparams.json"
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"  -> Saved: {best_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
