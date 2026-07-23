#!/usr/bin/env python3
"""
Precision–recall tradeoff scatter: grid search cloud + Optuna best points.

Grid search 48 configs as background cloud, Optuna best-per-metric
highlighted on top, original marked. Colour = ROC AUC.

Reads from:
  output/xgboost_q1_tuning_grid/hyperparam_grid_results.csv
  output/xgboost_q1_tuning_optuna/optuna_best_params.json

Writes to:
  output/xgboost_q1_tuning_optuna/optuna_precision_recall_tradeoff.png

Usage:
    python src/plot_optuna_prec_recall.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

PROJECT_DIR = Path(__file__).resolve().parents[1]
GRID_DIR    = PROJECT_DIR / "output" / "xgboost_q1_tuning_grid"
OPTUNA_DIR  = PROJECT_DIR / "output" / "xgboost_q1_tuning_optuna"

# ── Shared colour scale ──
CMAP = "viridis"
VMIN, VMAX = 0.82, 0.86

# ── Load grid search results (48 configs) ──
grid = pd.read_csv(GRID_DIR / "hyperparam_grid_results.csv")

# ── Load Optuna best-per-metric ──
with open(OPTUNA_DIR / "optuna_best_params.json") as f:
    optuna_results = json.load(f)

optuna_configs = []
for name, data in optuna_results.items():
    m = data.get("all_metrics", {})
    if m:
        optuna_configs.append({
            "name": name.split("\n")[0],
            "roc_auc":  m.get("roc_auc", 0),
            "gal_prec": m.get("galaxy_precision", 0),
            "gal_rec":  m.get("galaxy_recall", 0),
            "gal_f1":   m.get("galaxy_f1", 0),
            "star_prec": m.get("star_precision", 0),
            "star_rec":  m.get("star_recall", 0),
            "star_f1":   m.get("star_f1", 0),
        })
opt_df = pd.DataFrame(optuna_configs)
opt_tuned = opt_df[~opt_df["name"].str.contains("Original")]
orig = opt_df[opt_df["name"].str.contains("Original")]

# ── Plot ──
fig, ax = plt.subplots(figsize=(8, 4))

# Grid search cloud — Galaxy (circles) + Star (squares)
grid_auc = grid["mean_test_roc_auc"]

sc = ax.scatter(grid["mean_test_galaxy_recall"],
                grid["mean_test_galaxy_precision"],
                c=grid_auc, cmap=CMAP, vmin=VMIN, vmax=VMAX,
                s=40, marker="o", alpha=0.7,
                linewidths=0.3, zorder=2)

ax.scatter(grid["mean_test_star_recall"],
           grid["mean_test_star_precision"],
           c=grid_auc, cmap=CMAP, vmin=VMIN, vmax=VMAX,
           s=40, marker="s", alpha=0.7,
           linewidths=0.3, zorder=2)

# Optuna best — Galaxy (circles) + Star (squares), larger
opt_auc = opt_tuned["roc_auc"]

ax.scatter(opt_tuned["gal_rec"], opt_tuned["gal_prec"],
           c=opt_auc, cmap=CMAP, vmin=VMIN, vmax=VMAX,
           s=100, marker="o", edgecolors="grey", linewidths=0.8, zorder=4)

ax.scatter(opt_tuned["star_rec"], opt_tuned["star_prec"],
           c=opt_auc, cmap=CMAP, vmin=VMIN, vmax=VMAX,
           s=100, marker="s", edgecolors="grey", linewidths=0.8, zorder=4)

# Labels — Galaxy side
gal_label_offsets = {
    "Best ROC AUC":          (-100, 2),
    "Best Galaxy F1":        (-100, -6),
    "Best Galaxy Precision": (10, -2),
    "Best Galaxy Recall":    (10, -2),
}
for _, row in opt_tuned.iterrows():
    short = row["name"].replace("Optuna ", "")
    oxy = gal_label_offsets.get(short, (10, -2))
    ax.annotate(short, (row["gal_rec"], row["gal_prec"]),
                fontsize=9, xytext=oxy, textcoords="offset points",
                color="#333333", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5))

# Labels — Star side, with connecting lines to Galaxy pair
star_label_offsets = {
    "Galaxy F1":        (-40, -35),
    "ROC AUC":          (-40, 50),
    "Galaxy Recall":    (-70, 30),
    "Galaxy Precision": (-70, -25),
}
for _, row in opt_tuned.iterrows():
    short = row["name"].replace("Best ", "")
    ax.plot([row["gal_rec"], row["star_rec"]],
            [row["gal_prec"], row["star_prec"]],
            ls=":", color="#AAAAAA", lw=0.8, zorder=1)
    oxy = star_label_offsets.get(short, (10, -2))
    ax.annotate(f"({short})", (row["star_rec"], row["star_prec"]),
                fontsize=7, xytext=oxy, textcoords="offset points",
                color="#666666", style="italic",
                arrowprops=dict(arrowstyle="-", color="#CCCCCC", lw=0.5))

# Original config
if len(orig) > 0:
    o = orig.iloc[0]
    ax.scatter(o["gal_rec"], o["gal_prec"], marker="*", s=180,
               color="limegreen", zorder=5)
    ax.scatter(o["star_rec"], o["star_prec"], marker="*", s=180,
               color="limegreen", zorder=5)

# Axes
ax.set_xlabel("Recall (completeness)", fontsize=13, fontweight="bold", labelpad=6)
ax.set_ylabel("Precision (purity)", fontsize=13, fontweight="bold", labelpad=10)
ax.grid(alpha=0.3)

# Legend
legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
           markersize=8, label="Galaxy"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#888888",
           markersize=8, label="Star"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="limegreen",
           markersize=15, label="Original config"),
]
ax.legend(handles=legend_handles, fontsize=11, loc="upper right")
ax.tick_params(labelsize=12)

# Colorbar
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("ROC AUC", fontsize=12, fontweight="bold", labelpad=10)
cb.ax.tick_params(labelsize=11)
cb.locator = ticker.MultipleLocator(0.01)
cb.update_ticks()

out = PROJECT_DIR / "output" / "optuna_precision_recall_tradeoff.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
