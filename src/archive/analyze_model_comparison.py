#!/usr/bin/env python3
"""
Paired t-test comparison between Original and Optuna F1 models.

Compares per-seed metrics across the 10-seed ensemble and tests
whether the differences are statistically significant.

Reads:
  output/xgboost_q1/q1_seed_metrics.json
  output/xgboost_q1_op_f1/q1_seed_metrics.json

Writes:
  output/model_comparison.csv
  output/model_comparison.txt

Usage:
    python src/analyze_model_comparison.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODELS = [
    ("Original",  PROJECT_DIR / "output" / "xgboost_q1" / "q1_seed_metrics.json"),
    ("Optuna F1", PROJECT_DIR / "output" / "xgboost_q1_op_f1" / "q1_seed_metrics.json"),
]

METRICS = [
    ("Galaxy Precision", "galaxy_precision"),
    ("Galaxy Recall",    "galaxy_recall"),
    ("Galaxy F1",        "galaxy_f1"),
    ("Star F1",          "star_f1"),
    ("ROC AUC",          "roc_auc"),
]

SEEDS = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]

# ── Load ──
print("[1] Loading seed metrics...")
model_data = {}
for label, path in MODELS:
    with open(path) as f:
        data = json.load(f)
    model_data[label] = data
    print(f"  {label}: {len(data)} seeds")

# ── Paired t-test ──
print("\n[2] Paired t-test results:")
print(f"  {'Metric':<20} {'Orig mean':>10} {'Optuna mean':>11} "
      f"{'Delta':>8} {'t-stat':>8} {'p-value':>8} {'Sig?':>6}")
print("  " + "-" * 75)

results = []
orig_data = model_data["Original"]
opt_data  = model_data["Optuna F1"]

for label, key in METRICS:
    o_vals = [s[key] for s in orig_data]
    n_vals = [s[key] for s in opt_data]

    t_stat, p_val = stats.ttest_rel(n_vals, o_vals)
    delta = np.mean(n_vals) - np.mean(o_vals)
    sig = "Yes" if p_val < 0.05 else "No"

    print(f"  {label:<20} {np.mean(o_vals):>10.5f} {np.mean(n_vals):>11.5f} "
          f"{delta:>+8.5f} {t_stat:>8.3f} {p_val:>8.4f} {sig:>6}")

    results.append({
        "Metric": label,
        "Original_mean": np.mean(o_vals),
        "Original_std": np.std(o_vals),
        "Optuna_F1_mean": np.mean(n_vals),
        "Optuna_F1_std": np.std(n_vals),
        "Delta": delta,
        "t_statistic": t_stat,
        "p_value": p_val,
        "Significant": sig,
    })

# ── Per-seed Galaxy F1 ──
print(f"\n[3] Per-seed Galaxy F1:")
print(f"  {'Seed':<8} {'Original':>10} {'Optuna F1':>10} {'Delta':>8} {'Winner':>8}")
print("  " + "-" * 50)

o_f1 = [s["galaxy_f1"] for s in orig_data]
n_f1 = [s["galaxy_f1"] for s in opt_data]
optuna_wins = 0

for seed, o, n in zip(SEEDS, o_f1, n_f1):
    d = n - o
    w = "Optuna" if d > 0 else "Orig"
    if d > 0:
        optuna_wins += 1
    print(f"  {seed:<8} {o:>10.5f} {n:>10.5f} {d:>+8.5f} {w:>8}")

print(f"\n  Optuna F1 wins {optuna_wins}/{len(SEEDS)} seeds")

# ── Save ──
df = pd.DataFrame(results)
csv_path = PROJECT_DIR / "output" / "model_comparison.csv"
df.to_csv(csv_path, index=False)
print(f"\n  -> {csv_path}")

# ── Plot ──
print("\n[4] Plotting...")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.subplots_adjust(wspace=0.35)

# [0] Per-seed Galaxy F1 comparison (paired dot plot)
ax = axes[0]
for i, (seed, o, n) in enumerate(zip(SEEDS, o_f1, n_f1)):
    ax.plot([0, 1], [o, n], color='#BBBBBB', lw=0.8, zorder=1)
    ax.scatter(0, o, s=50, color='#2196F3', zorder=3, edgecolors='white', linewidths=0.5)
    ax.scatter(1, n, s=50, color='#E53935', zorder=3, edgecolors='white', linewidths=0.5)

# Means
ax.scatter(0, np.mean(o_f1), s=120, color='#2196F3', marker='D', zorder=4,
           edgecolors='black', linewidths=1, label=f'Original mean = {np.mean(o_f1):.4f}')
ax.scatter(1, np.mean(n_f1), s=120, color='#E53935', marker='D', zorder=4,
           edgecolors='black', linewidths=1, label=f'Optuna F1 mean = {np.mean(n_f1):.4f}')
ax.plot([0, 1], [np.mean(o_f1), np.mean(n_f1)], color='black', lw=1.5, ls='--', zorder=2)

ax.set_xticks([0, 1])
ax.set_xticklabels(['Original', 'Optuna F1'], fontsize=11, fontweight='bold')
ax.set_ylabel('Galaxy F1', fontsize=11, fontweight='bold')
ax.set_title(f'Per-seed Galaxy F1\np = {results[2]["p_value"]:.4f}, '
             f'Optuna wins {optuna_wins}/{len(SEEDS)} seeds',
             fontsize=11, fontweight='bold')
ax.set_xlim(-0.3, 1.3)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# [1] Delta bar chart across all metrics
ax = axes[1]
metric_names = [r["Metric"] for r in results]
deltas = [r["Delta"] for r in results]
p_vals = [r["p_value"] for r in results]
colors = ['#4CAF50' if p < 0.05 else '#BBBBBB' for p in p_vals]

bars = ax.barh(range(len(metric_names)), deltas, color=colors, edgecolor='white', height=0.6)

for i, (bar, d, p) in enumerate(zip(bars, deltas, p_vals)):
    sig_str = f'p={p:.4f} *' if p < 0.05 else f'p={p:.2f}'
    x_pos = bar.get_width() + 0.0003 if d >= 0 else bar.get_width() - 0.0003
    ha = 'left' if d >= 0 else 'right'
    ax.text(x_pos, i, f' {d:+.4f}  ({sig_str})',
            va='center', ha=ha, fontsize=8, fontweight='bold')

ax.axvline(0, color='black', lw=0.8)
ax.set_yticks(range(len(metric_names)))
ax.set_yticklabels(metric_names, fontsize=10)
ax.set_xlabel('Optuna F1 - Original', fontsize=11, fontweight='bold')
ax.set_title('Metric differences\ngreen = significant at p < 0.05',
             fontsize=11, fontweight='bold')
ax.grid(alpha=0.3, axis='x')

fig.suptitle('Model comparison: Original vs Optuna F1 (10-seed ensemble)',
             fontsize=13, fontweight='bold', y=0.98)

out_plot = PROJECT_DIR / "output" / "model_comparison.png"
plt.savefig(out_plot, dpi=200, bbox_inches='tight')
plt.close()
print(f"  -> {out_plot}")

# Save text summary
txt_path = PROJECT_DIR / "output" / "model_comparison.txt"
with open(txt_path, "w") as f:
    f.write("MODEL COMPARISON: Original vs Optuna F1\n")
    f.write("=" * 60 + "\n\n")
    f.write("Paired t-test across 10-seed ensemble\n\n")
    f.write(f"{'Metric':<20} {'Orig':>10} {'Optuna':>10} {'Delta':>8} "
            f"{'p-value':>8} {'Sig':>5}\n")
    f.write("-" * 65 + "\n")
    for r in results:
        f.write(f"{r['Metric']:<20} {r['Original_mean']:>10.5f} "
                f"{r['Optuna_F1_mean']:>10.5f} {r['Delta']:>+8.5f} "
                f"{r['p_value']:>8.4f} {r['Significant']:>5}\n")
    f.write(f"\nOptuna F1 wins {optuna_wins}/{len(SEEDS)} seeds on Galaxy F1\n")
print(f"  -> {txt_path}")
