#!/usr/bin/env python3
"""
Combined P(gal) comparison: random vs high-lat Gaia samples overlaid,
both models overlaid. All on same axes.

Panel 1: P(gal) histogram — 4 traces (random×2 models + hilat×2 models)
Panel 2: Fraction vs threshold — 4 traces

solid = random, dashed = high-lat
blue = Original, red = Optuna F1

Reads:
  output/umap/arrays/gaia_random.npz
  output/umap/arrays/gaia_hilat.npz
  output/xgboost_q1/xgb_classifier_q1.json
  output/xgboost_q1_op_f1/xgb_classifier_q1.json

Writes:
  output/pgal_gaia_combined.png

Usage:
    python src/analyze_pgal_gaia_combined.py
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"

MODELS = [
    ("Original", PROJECT_DIR / "output" / "xgboost_q1" / "xgb_classifier_q1.json",
     "#2196F3"),
]

SAMPLES = [
    ("Random",   "gaia_random.npz", "#5C6BC0"),
    ("High-lat", "gaia_hilat.npz",  "#26A69A"),
]

feat_names = list(SELECTED_GAIA_PARAMETERS)
bins = np.linspace(0, 1, 101)
t_range = np.linspace(0, 1, 501)

# ── Load + predict ──
print("[1] Loading and predicting...")
traces = []  # (label, p, color, ls, n)

for sample_name, npz_file, sample_color in SAMPLES:
    g = np.load(ARRAYS_DIR / npz_file, allow_pickle=True)
    X = g["X_features"].astype(float)
    n = len(X)
    print(f"  {sample_name}: {n:,} sources")

    for model_label, model_path, _ in MODELS:
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        dmat = xgb.DMatrix(X, feature_names=feat_names)
        p = model.get_booster().predict(dmat)
        label = sample_name
        traces.append((label, p, sample_color, '-', n))
        print(f"    {model_label}: P>0.5={100*(p>=0.5).mean():.1f}%  "
              f"P>0.9={100*(p>=0.9).mean():.1f}%  "
              f"P>0.95={100*(p>=0.95).mean():.1f}%")

# ── Plot ──
print("\n[2] Plotting...")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.subplots_adjust(wspace=0.35)

for label, p, color, ls, n in traces:
    # [0] Histogram — outlines only
    axes[0].hist(p, bins=bins, color=color, histtype='step',
                 linewidth=1.5, ls=ls, label=label)

    # [1] Fraction vs threshold
    counts = np.array([(p >= t).sum() for t in t_range])
    frac = counts / n * 100
    axes[1].plot(t_range, frac, color=color, lw=2, ls=ls, label=label)

# Threshold markers
for t, col, mls, lbl in [(0.5, 'black', '--', 'Default (0.5)'),
                           (0.9, 'green', '--', 'Threshold (0.9)')]:
    axes[0].axvline(t, color=col, lw=1.5, ls=mls, alpha=0.5)
    axes[1].axvline(t, color=col, lw=1.5, ls=mls, alpha=0.5, label=lbl)

# Annotate percentages at thresholds
for label, p, color, ls, n in traces:
    for t in [0.5, 0.9]:
        pct = 100 * (p >= t).mean()
        is_random = ("Random" in label)
        dx = 0.04 if is_random else -0.12
        ymult = 1.5 if is_random else 0.4
        axes[1].annotate(
            f"{pct:.1f}%", xy=(t, pct), xytext=(t + dx, pct * ymult),
            fontsize=8, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

# Finalize [0]
axes[0].set_yscale("log")
axes[0].set_ylim(0.5, 2e4)
axes[0].set_xlabel("P(gal)", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Count", fontsize=13, fontweight="bold")
axes[0].set_title("P(gal) distribution — 50k Gaia sources (G \u2265 17)",
                   fontsize=13, fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].tick_params(labelsize=12)
axes[0].grid(alpha=0.3)

# Finalize [1]
axes[1].set_yscale("log")
axes[1].set_ylim(0.05, 100)
axes[1].set_xlabel("P(gal) threshold", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Fraction predicted as galaxy (%)",
                    fontsize=13, fontweight="bold")
axes[1].set_title("Predicted galaxy fraction vs threshold",
                   fontsize=13, fontweight="bold")
axes[1].legend(fontsize=10)
axes[1].tick_params(labelsize=12)
axes[1].grid(alpha=0.3)

fig.suptitle("P(gal) on Gaia sources — Random vs High-latitude",
             fontsize=14, fontweight="bold", y=0.98)

out = PROJECT_DIR / "output" / "pgal_gaia_combined.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"  -> {out}")
