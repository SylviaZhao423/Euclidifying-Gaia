#!/usr/bin/env python3
"""
Heatmap of ALL 48 grid search configurations across all metrics.

Rows = all configs (sorted by ROC AUC), Cols = all metrics.
Cells coloured per column, annotated with values.

Reads from: output/xgboost_q1_tuning/hyperparam_grid_results.csv
Writes to:  output/xgboost_q1_tuning/hyperparam_grid_heatmap_all.png

Usage:
    python src/plot_tuning_grid_heatmap.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR  = PROJECT_DIR / "output" / "xgboost_q1_tuning_grid"

df = pd.read_csv(OUTPUT_DIR / "hyperparam_grid_results.csv")
df = df.sort_values("rank_test_roc_auc").reset_index(drop=True)

# ── Metrics ──
metrics = [
    ("ROC\nAUC",       "mean_test_roc_auc"),
    ("Galaxy\nPrec",   "mean_test_galaxy_precision"),
    ("Galaxy\nRecall", "mean_test_galaxy_recall"),
    ("Galaxy\nF1",     "mean_test_galaxy_f1"),
    ("Star\nPrec",     "mean_test_star_precision"),
    ("Star\nRecall",   "mean_test_star_recall"),
    ("Star\nF1",       "mean_test_star_f1"),
]
metric_labels = [m[0] for m in metrics]
metric_cols   = [m[1] for m in metrics]

# ── Row labels ──
row_labels = []
for _, r in df.iterrows():
    row_labels.append(
        f"d={int(r['param_max_depth']):>2}, n={int(r['param_n_estimators']):>3}, "
        f"lr={r['param_learning_rate']:.2f}")

# Mark original
orig_idx = df[(df["param_max_depth"] == 8) &
              (df["param_n_estimators"] == 200) &
              (df["param_learning_rate"] == 0.1)].index[0]

n_configs = len(df)
n_metrics = len(metrics)

# ── Build value matrix ──
values = np.zeros((n_configs, n_metrics))
for i in range(n_configs):
    for j, col in enumerate(metric_cols):
        values[i, j] = df.iloc[i][col]

# ── Per-column normalisation ──
col_min = values.min(axis=0)
col_max = values.max(axis=0)
col_range = col_max - col_min
col_range[col_range == 0] = 1
normed = (values - col_min) / col_range

# ── Plot ──
fig, ax = plt.subplots(figsize=(10, 18))

im = ax.imshow(normed, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

# Annotate cells
for j in range(n_metrics):
    best_i = np.argmax(values[:, j])
    for i in range(n_configs):
        val = values[i, j]
        is_best = (i == best_i)
        color = "black" if normed[i, j] > 0.4 else "white"
        text = f"{val:.3f}"
        if is_best:
            text += " ★"
        ax.text(j, i, text, ha="center", va="center",
                fontsize=6.5, fontweight="bold" if is_best else "normal",
                color=color)

# Find which row is best for each metric
best_rows = set()
for j in range(n_metrics):
    best_rows.add(np.argmax(values[:, j]))

# Row labels — highlight original + best-per-metric rows
ax.set_yticks(range(n_configs))

# Build labels with markers
marked_labels = []
for i, lbl in enumerate(row_labels):
    # Which metrics does this row win?
    wins = [metric_labels[j].replace("\n", " ") for j in range(n_metrics)
            if np.argmax(values[:, j]) == i]
    if wins:
        lbl += "  ◄ best " + ", ".join(wins)
    if i == orig_idx:
        lbl += "  ◄ ORIGINAL"
    marked_labels.append(lbl)

ylabels = ax.set_yticklabels(marked_labels, fontsize=7, family="monospace")
ylabels[orig_idx].set_color("red")
ylabels[orig_idx].set_fontweight("bold")
for i in best_rows:
    if i != orig_idx:
        ylabels[i].set_color("#2E7D32")
        ylabels[i].set_fontweight("bold")

ax.set_xticks(range(n_metrics))
ax.set_xticklabels(metric_labels, fontsize=9.5, fontweight="bold")

# Grid lines
for i in range(n_configs + 1):
    ax.axhline(i - 0.5, color="white", lw=0.5)
for j in range(n_metrics + 1):
    ax.axvline(j - 0.5, color="white", lw=1)

# Mark original row
ax.axhline(orig_idx - 0.5, color="red", lw=1.5, ls="-")
ax.axhline(orig_idx + 0.5, color="red", lw=1.5, ls="-")

# Mark best-per-metric rows
for i in best_rows:
    if i != orig_idx:
        ax.axhline(i - 0.5, color="#2E7D32", lw=1.2, ls="--")
        ax.axhline(i + 0.5, color="#2E7D32", lw=1.2, ls="--")

ax.set_title("All 48 grid search configurations — sorted by ROC AUC\n"
             "(★ = best in column, red row = original d=8, n=200, lr=0.1)",
             fontsize=12, fontweight="bold", pad=12)

fig.colorbar(im, ax=ax, label="Relative score (per column)",
             shrink=0.4, pad=0.02)

out = OUTPUT_DIR / "hyperparam_grid_heatmap_all.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
