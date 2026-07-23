#!/usr/bin/env python3
"""
Compare P(gal) distributions on Gaia sources between Original and Optuna F1 models.

Overlays both models on the same histogram and galaxy-count plot.

Reads:
  output/umap/arrays/gaia_random.npz
  output/xgboost_q1/xgb_classifier_q1.json
  output/xgboost_q1_op_f1/xgb_classifier_q1.json

Writes:
  output/xgboost_q1_op_f1/gaia_pgal_test.png

Usage:
    python src/analyze_pgal_gaia_test.py
"""

from pathlib import Path
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
GAIA_DIR    = PROJECT_DIR / "output" / "gaia_50k"

parser = argparse.ArgumentParser()
parser.add_argument("--hilat", action="store_true",
                    help="Use high-latitude Gaia sample instead of random")
args = parser.parse_args()

SEED_DIR = PROJECT_DIR / "output" / "xgboost_q1" / "seed_models"
SEEDS = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]

# ── Load Gaia data ──
if args.hilat:
    sample_name = "high-lat"
    npz_file = "gaia_hilat_50k.npz"
else:
    sample_name = "random"
    npz_file = "gaia_random_50k.npz"

print(f"[1] Loading Gaia {sample_name} sources...")
g = np.load(GAIA_DIR / npz_file, allow_pickle=True)
X_gaia = g["X"].astype(float)
feat_names = list(SELECTED_GAIA_PARAMETERS)
print(f"  {len(X_gaia):,} sources")

# ── Predict P(gal) with ensemble mean ──
print(f"\n[2] Ensemble prediction ({len(SEEDS)} seeds)...")
probas = []
for seed in SEEDS:
    model_path = SEED_DIR / f"xgb_classifier_q1_seed{seed}.json"
    if not model_path.exists():
        print(f"  WARNING: {model_path.name} not found, skipping")
        continue
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    probas.append(model.predict_proba(X_gaia)[:, 1])
p = np.mean(probas, axis=0)
predictions = [("Original", p, "steelblue")]
print(f"  P(gal)>0.5: {(p>=0.5).sum():,} ({100*(p>=0.5).mean():.1f}%)")
print(f"  P(gal)>0.7: {(p>=0.7).sum():,} ({100*(p>=0.7).mean():.1f}%)")
print(f"  P(gal)>0.9: {(p>=0.9).sum():,} ({100*(p>=0.9).mean():.1f}%)")

# ── Plot: single panel with twin y-axis ──
print("\n[3] Plotting...")
fig, ax1 = plt.subplots(figsize=(10, 5.5))

bins = np.linspace(0, 1, 101)
t_range = np.linspace(0, 1, 501)

# Left y-axis: histogram
for label, p, color in predictions:
    ax1.hist(p, bins=bins, alpha=0.25, color=color,
             histtype='stepfilled', linewidth=0)
    ax1.hist(p, bins=bins, color=color,
             histtype='step', linewidth=1.5, label="P(gal) counts")

ax1.set_yscale("log")
ax1.set_ylim(0.5, 1e4)
ax1.set_xlabel("P(gal)", fontsize=14, fontweight="bold")
ax1.set_ylabel("Source count", fontsize=14, fontweight="bold", color="steelblue", labelpad=10)
ax1.tick_params(axis='y', labelcolor="steelblue", labelsize=12)
plt.setp(ax1.get_yticklabels(), fontweight="bold")
ax1.tick_params(axis='x', labelsize=12)
ax1.grid(alpha=0.3)

# Right y-axis: fraction curve
ax2 = ax1.twinx()
for label, p, color in predictions:
    counts = np.array([(p >= t).sum() for t in t_range])
    frac = counts / len(X_gaia) * 100
    ax2.plot(t_range, frac, color="#E53935", lw=1.5,
             label="Fraction above threshold (%)")

ax2.set_yscale("log")
ax2.set_ylim(0.05, 100)
ax2.set_ylabel("Fraction above threshold (%)", fontsize=14, fontweight="bold", color="#E53935", labelpad=10)
ax2.set_yticks([0.1, 0.5, 1, 2, 5, 10, 20, 50, 100])
ax2.set_yticklabels(["0.1", "0.5", "1", "2", "5", "10", "20", "50", "100"])
ax2.tick_params(axis='y', labelcolor="#E53935", labelsize=12)
plt.setp(ax2.get_yticklabels(), fontweight="bold")

# Threshold markers
for t, col, ls, lbl in [(0.5, 'black', '--', 'Default (0.5)'),
                         (0.9, 'green', '--', 'Threshold (0.9)')]:
    ax1.axvline(t, color=col, lw=1.5, ls=ls, alpha=0.5)

# Inline gap labels for percentages
bbox_gap = dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=1.0)
for label, p, color in predictions:
    for t in [0.5, 0.9]:
        pct = 100 * (p >= t).mean()
        ax2.text(t, pct, f"{pct:.1f}%", fontsize=14, fontweight="bold",
                 color="#E53935", ha='center', va='center', bbox=bbox_gap, zorder=10)

out = PROJECT_DIR / "output" / f"pgal_gaia.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> {out}")
