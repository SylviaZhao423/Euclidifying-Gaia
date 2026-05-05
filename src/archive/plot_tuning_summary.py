#!/usr/bin/env python3
"""
Summary heatmaps of hyperparameter tuning results.

Produces TWO heatmaps:
  1. Grid search summary  → hyperparam_summary_grid.png
  2. Optuna summary       → hyperparam_summary_optuna.png

Rows = configs (original + best per metric), Cols = all metrics.

Reads from: output/xgboost_q1_tuning/
  hyperparam_grid_results.csv
  optuna_results.csv
  optuna_best_params.json

Usage:
    python src/plot_tuning_summary.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR  = PROJECT_DIR / "output" / "xgboost_q1_tuning_grid"

# ── Metrics ──
METRICS = [
    ("ROC AUC",          "roc_auc"),
    ("Galaxy\nPrecision", "galaxy_precision"),
    ("Galaxy\nRecall",    "galaxy_recall"),
    ("Galaxy\nF1",        "galaxy_f1"),
    ("Star\nPrecision",   "star_precision"),
    ("Star\nRecall",      "star_recall"),
    ("Star\nF1",          "star_f1"),
]
METRIC_LABELS = [m[0] for m in METRICS]
METRIC_KEYS   = [m[1] for m in METRICS]


def plot_summary(configs, title, out_path):
    """
    configs: list of (label, {metric_key: value, ...}) tuples
    """
    config_labels = [c[0] for c in configs]
    n_configs = len(configs)
    n_metrics = len(METRICS)

    # Build value matrix
    values = np.zeros((n_configs, n_metrics))
    for i, (_, vals) in enumerate(configs):
        for j, key in enumerate(METRIC_KEYS):
            values[i, j] = vals.get(key, 0)

    # Per-column normalisation
    col_min = values.min(axis=0)
    col_max = values.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    normed = (values - col_min) / col_range

    # Plot
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
                text = f"{val:.4f} \u2605"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=9, fontweight="bold" if is_best else "normal",
                    color=color)

    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(METRIC_LABELS, fontsize=10, fontweight="bold")
    ax.set_yticks(range(n_configs))
    ax.set_yticklabels(config_labels, fontsize=9)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

    for i in range(n_configs + 1):
        ax.axhline(i - 0.5, color="white", lw=1.5)
    for j in range(n_metrics + 1):
        ax.axvline(j - 0.5, color="white", lw=1.5)

    fig.colorbar(im, ax=ax, label="Relative score (per column)",
                 shrink=0.8, pad=0.02)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def extract_grid_configs(df, col_prefix="mean_test_"):
    """Extract best-per-metric configs from grid search results."""
    def label(row, name):
        return (f"{name}\n(d={int(row['param_max_depth'])}, "
                f"n={int(row['param_n_estimators'])}, "
                f"lr={row['param_learning_rate']})")

    def row_to_dict(row):
        return {k: row[f"{col_prefix}{k}"] for k in METRIC_KEYS}

    orig = df[(df["param_max_depth"] == 8) &
              (df["param_n_estimators"] == 200) &
              (df["param_learning_rate"] == 0.1)].iloc[0]

    configs = [(label(orig, "Original"), row_to_dict(orig))]

    for metric_name, metric_key in METRICS:
        col = f"{col_prefix}{metric_key}"
        best = df.loc[df[col].idxmax()]
        clean_name = metric_name.replace("\n", " ")
        configs.append((label(best, f"Best {clean_name}"), row_to_dict(best)))

    return configs


# ═══════════════════════════════════════════════════════════════
#  1. Grid search summary
# ═══════════════════════════════════════════════════════════════
print("[1] Grid search summary...")
grid_df = pd.read_csv(OUTPUT_DIR / "hyperparam_grid_results.csv")
grid_configs = extract_grid_configs(grid_df)

plot_summary(
    grid_configs,
    "Grid search tuning summary — best config per metric\n"
    "(★ = best in column, colour normalised per column)",
    OUTPUT_DIR / "hyperparam_summary_grid.png",
)


# ═══════════════════════════════════════════════════════════════
#  2. Optuna summary
# ═══════════════════════════════════════════════════════════════
print("[2] Optuna summary...")

OPTUNA_DIR  = PROJECT_DIR / "output" / "xgboost_q1_tuning_optuna"
optuna_json = OPTUNA_DIR / "optuna_best_params.json"
optuna_csv  = OPTUNA_DIR / "optuna_results.csv"

if not optuna_json.exists() or not optuna_csv.exists():
    print("  Optuna results not found, skipping.")
else:
    with open(optuna_json) as f:
        optuna_best = json.load(f)

    # Optuna best metrics
    opt_metrics = optuna_best.get("_metrics", {})
    # Map to our keys
    opt_vals = {}
    key_map = {
        "ROC AUC": "roc_auc",
        "Galaxy Precision": "galaxy_precision",
        "Galaxy Recall": "galaxy_recall",
        "Galaxy F1": "galaxy_f1",
        "Star Precision": "star_precision",
        "Star Recall": "star_recall",
        "Star F1": "star_f1",
    }
    for long_name, short_key in key_map.items():
        if long_name in opt_metrics:
            opt_vals[short_key] = opt_metrics[long_name]

    opt_label = (f"Optuna Best (AUC)\n"
                 f"(d={optuna_best.get('max_depth','?')}, "
                 f"n={optuna_best.get('n_estimators','?')}, "
                 f"lr={optuna_best.get('learning_rate', 0):.4f})")

    # Build configs: Original, Grid best per metric, Optuna best
    # Start with grid configs, append Optuna
    optuna_configs = grid_configs.copy()
    if opt_vals:
        optuna_configs.append((opt_label, opt_vals))

    plot_summary(
        optuna_configs,
        "Tuning summary — Grid search + Optuna\n"
        "(★ = best in column, colour normalised per column)",
        OUTPUT_DIR / "hyperparam_summary_optuna.png",
    )

print("Done!")
