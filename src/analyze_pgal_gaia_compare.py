#!/usr/bin/env python3
"""
Compare P(gal) distributions on Gaia sources between Original and Optuna F1 models.

Overlays both models on the same histogram and galaxy-count plot.

Reads:
  output/umap/arrays/gaia_random.npz
  output/xgboost_q1/xgb_classifier_q1.json
  output/xgboost_q1_op_f1/xgb_classifier_q1.json

Writes:
  output/xgboost_q1_op_f1/gaia_pgal_compare.png

Usage:
    python src/plot_pgal_compare_models.py
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
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"

parser = argparse.ArgumentParser()
parser.add_argument("--hilat", action="store_true",
                    help="Use high-latitude Gaia sample instead of random")
args = parser.parse_args()

MODELS = [
    ("Original",
     PROJECT_DIR / "output" / "xgboost_q1" / "xgb_classifier_q1.json",
     "#2196F3"),
    ("Optuna F1",
     PROJECT_DIR / "output" / "xgboost_q1_op_f1" / "xgb_classifier_q1.json",
     "#E53935"),
]

# ── Load Gaia data ──
if args.hilat:
    sample_name = "high-lat"
    npz_file = "gaia_hilat.npz"
else:
    sample_name = "random"
    npz_file = "gaia_random.npz"

print(f"[1] Loading Gaia {sample_name} sources...")
g = np.load(ARRAYS_DIR / npz_file, allow_pickle=True)
X_gaia = g["X_features"].astype(float)
feat_names = list(SELECTED_GAIA_PARAMETERS)
print(f"  {len(X_gaia):,} sources")

# ── Predict P(gal) with each model ──
predictions = []
for label, model_path, color in MODELS:
    print(f"\n[2] Predicting with {label}...")
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    dmat = xgb.DMatrix(X_gaia, feature_names=feat_names)
    p = model.get_booster().predict(dmat)
    predictions.append((label, p, color))
    print(f"  P(gal)>0.5: {(p>=0.5).sum():,} ({100*(p>=0.5).mean():.1f}%)")
    print(f"  P(gal)>0.7: {(p>=0.7).sum():,} ({100*(p>=0.7).mean():.1f}%)")
    print(f"  P(gal)>0.9: {(p>=0.9).sum():,} ({100*(p>=0.9).mean():.1f}%)")

# ── Plot ──
print("\n[3] Plotting...")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.subplots_adjust(wspace=0.30)

bins = np.linspace(0, 1, 101)
t_range = np.linspace(0, 1, 501)

for label, p, color in predictions:
    # [0] Histogram — step outline
    axes[0].hist(p, bins=bins, alpha=0.25, color=color,
                 histtype='stepfilled', linewidth=0)
    axes[0].hist(p, bins=bins, color=color,
                 histtype='step', linewidth=1.5, label=label)

    # [1] Galaxy fraction vs threshold
    counts = np.array([(p >= t).sum() for t in t_range])
    frac = counts / len(X_gaia) * 100  # percentage
    axes[1].plot(t_range, frac, color=color, lw=2.5, label=label)

# Threshold markers
for t, col, ls, lbl in [(0.5, 'black', '--', 'Default (0.5)'),
                         (0.9, 'green', '--', 'Threshold (0.9)')]:
    axes[0].axvline(t, color=col, lw=1.5, ls=ls, alpha=0.5)
    axes[1].axvline(t, color=col, lw=1.5, ls=ls, alpha=0.5, label=lbl)

# Annotate percentages at thresholds
annot_offsets = {
    ("#2196F3", 0.5):  (0.04, 1.5),    # Original → above
    ("#2196F3", 0.9): (0.03, 2.5),
    ("#E53935", 0.5):  (-0.12, 0.3),    # Optuna → below
    ("#E53935", 0.9): (-0.08, 0.2),
}
for label, p, color in predictions:
    for t in [0.5, 0.9]:
        pct = 100 * (p >= t).mean()
        dx, ymult = annot_offsets.get((color, t), (0.03, 1.5))
        axes[1].annotate(
            f"{pct:.1f}%", xy=(t, pct), xytext=(t + dx, pct * ymult),
            fontsize=8, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

axes[0].set_yscale("log")
axes[0].set_ylim(0.5, 2e4)
axes[0].set_xlabel("P(gal)", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Count", fontsize=13, fontweight="bold")
axes[0].set_title(f"P(gal) distribution — 50k {sample_name} Gaia (G \u2265 17)",
                   fontsize=13, fontweight="bold")
axes[0].legend(fontsize=11, loc='upper right')
axes[0].tick_params(labelsize=12)
axes[0].grid(alpha=0.3)

axes[1].set_xlabel("P(gal) threshold", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Fraction predicted as galaxy (%)",
                    fontsize=13, fontweight="bold")
axes[1].set_title("Predicted galaxy fraction vs threshold",
                   fontsize=13, fontweight="bold")
axes[1].set_yscale("log")
axes[1].set_ylim(0.05, 100)
axes[1].legend(fontsize=11, loc='lower left')
axes[1].tick_params(labelsize=12)
axes[1].grid(alpha=0.3)

suffix = "_hilat" if args.hilat else ""
fig.suptitle(f"P(gal) on Gaia {sample_name} sources — Original vs Optuna F1 model",
             fontsize=14, fontweight="bold", y=0.98)
out = PROJECT_DIR / "output" / f"pgal_gaia_compare{suffix}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> {out}")
