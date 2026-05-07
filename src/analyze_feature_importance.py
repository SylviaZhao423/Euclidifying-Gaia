#!/usr/bin/env python3
"""
Paper-quality feature importance plot — reproduces the exact plot from
train_xgboost_q1.py but reads from saved CSV.

Reads:
  output/xgboost_q1/q1_feature_importance.csv
  output/xgboost_q1/q1_seed_metrics.json

Writes:
  output/feature_importance.png

Usage:
    python src/plot_feature_importance.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]

# ── Load data ──
df = pd.read_csv(PROJECT_DIR / "output" / "xgboost_q1" / "q1_feature_importance.csv")

with open(PROJECT_DIR / "output" / "xgboost_q1" / "q1_seed_metrics.json") as f:
    seed_metrics = json.load(f)
metric_df = pd.DataFrame(seed_metrics)

# Already sorted by rank in CSV
feats = df["feature"].values
mu_s  = df["importance_mean"].values
std_s = df["importance_std"].values
cv_s  = df["importance_cv"].values
n = len(feats)

# ── Plot (same as train_xgboost_q1.py) ──
fig, ax = plt.subplots(figsize=(11, 5.2))

cv_abs_max = 0.3
norm = plt.Normalize(vmin=0, vmax=cv_abs_max)
colors = plt.cm.RdYlGn_r(norm(cv_s))

bars = ax.barh(
    range(n), mu_s, xerr=std_s, height=0.7,
    color=colors, edgecolor='white', capsize=4,
    error_kw={'elinewidth': 1.2, 'ecolor': 'black'},
)
ax.set_yticks(range(n))
ax.set_yticklabels([f'#{i+1}  {f}' for i, f in enumerate(feats)], fontsize=12)
import matplotlib.pyplot as _plt
_plt.setp(ax.get_yticklabels(), fontfamily='DejaVu Sans Mono', fontweight=900)
ax.invert_yaxis()
ax.set_ylim(n - 0.5, -0.5)
ax.set_xlabel('Feature importance  (mean \u00b1 1\u03c3 across seeds)', fontsize=13, fontweight='bold', labelpad=8)
ax.set_xlim(0, 0.5)
ax.set_xticks(np.arange(0, 0.51, 0.1))
ax.grid(alpha=0.3, axis='x')

for i, (bar, mean_val, std_val, cv_val) in enumerate(zip(bars, mu_s, std_s, cv_s)):
    label = f'{mean_val:.3f} \u00b1 {std_val:.3f}'
    ax.text(mean_val + std_val + 0.001, bar.get_y() + bar.get_height() / 2,
            label, va='center', fontsize=10)

sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
cbar.set_label('CV = std / mean', fontsize=9, fontweight='bold')
import matplotlib.ticker as ticker
cbar.locator = ticker.MultipleLocator(0.1)
cbar.update_ticks()
cbar.ax.text(0.4, 1.04, 'unstable', transform=cbar.ax.transAxes,
             fontsize=9, fontweight='bold', va='bottom', ha='left', color='darkred')
cbar.ax.text(0.4, -0.05, 'stable', transform=cbar.ax.transAxes,
             fontsize=9, fontweight='bold', va='top', ha='left', color='darkgreen')

# Test-set metrics box
metrics_summary = [
    ('galaxy_precision', 'Galaxy Precision',
     metric_df['galaxy_precision'].mean(), metric_df['galaxy_precision'].std()),
    ('galaxy_recall', 'Galaxy Recall',
     metric_df['galaxy_recall'].mean(), metric_df['galaxy_recall'].std()),
    ('galaxy_f1', 'Galaxy F1',
     metric_df['galaxy_f1'].mean(), metric_df['galaxy_f1'].std()),
    ('roc_auc', 'ROC AUC',
     metric_df['roc_auc'].mean(), metric_df['roc_auc'].std()),
]
lines = ['Test-set metrics:']
for _, label, mean_val, std_val in metrics_summary:
    lines.append(f'{label:<16}: {mean_val:.3f} \u00b1 {std_val:.3f}')
ax.text(
    0.53, 0.05, '\n'.join(lines),
    transform=ax.transAxes,
    fontsize=10,
    family='monospace',fontweight='bold',
    va='bottom',
    ha='left',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
              edgecolor='#bbbbbb', alpha=0.95),
)

ax.tick_params(axis='x', labelsize=12)

n_seeds = len(seed_metrics)
title = f'Q1 Model Feature Importance Stability  ({n_seeds} seeds, 17 \u2264 G)'
# plt.suptitle(title, fontsize=14, fontweight='bold', y=0.96)
plt.tight_layout()

out = PROJECT_DIR / "output" / "feature_importance.png"
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
