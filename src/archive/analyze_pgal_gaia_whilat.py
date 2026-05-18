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
  output/pgal_gaia_whilat.png

Usage:
    python src/analyze_pgal_gaia_whilat.py
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
GAIA_DIR    = PROJECT_DIR / "output" / "gaia_50k"

SEEDS = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]

MODELS = [
    ("Q1-only",  PROJECT_DIR / "output" / "xgboost_q1" / "seed_models",
     "#2196F3", "lightpink"),
    ("8.6:1 ERO", PROJECT_DIR / "output" / "xgboost_mix_q1_ero" / "ratio_sg8.63_all_ero" / "seed_models",
     "#2E7D32", "#A5D6A7"),
]

SAMPLES = [
    ("Random",   "gaia_random_50k.npz"),
    ("High-lat", "gaia_hilat_50k.npz"),
]

feat_names = list(SELECTED_GAIA_PARAMETERS)
bins = np.linspace(0, 1, 101)
t_range = np.linspace(0, 1, 501)

# ── Load + predict (ensemble mean) ──
print("[1] Loading and predicting (ensemble)...")
traces = []  # (label, p, color, ls, n)

for model_label, seed_dir, color_random, color_hilat in MODELS:
    # Load seed models
    seed_models = []
    for seed in SEEDS:
        mp = seed_dir / f"xgb_classifier_q1_seed{seed}.json"
        if not mp.exists():
            continue
        m = xgb.XGBClassifier()
        m.load_model(str(mp))
        seed_models.append(m)
    print(f"  {model_label}: loaded {len(seed_models)} seed models")

    for sample_name, npz_file in SAMPLES:
        g = np.load(GAIA_DIR / npz_file, allow_pickle=True)
        X = g["X"].astype(float)
        n = len(X)

        color = color_random if sample_name == "Random" else color_hilat
        ls = '-' if model_label == "Q1-only" else '--'

        probas = [m.predict_proba(X)[:, 1] for m in seed_models]
        p = np.mean(probas, axis=0)
        label = f"{sample_name} ({model_label})"
        traces.append((label, p, color, ls, n))
        print(f"    {label}: P>0.5={100*(p>=0.5).mean():.1f}%  "
              f"P>0.9={100*(p>=0.9).mean():.1f}%")

# ── Plot: single panel with twin y-axis ──
# Match format of analyze_pgal_gaia.py exactly
print("\n[2] Plotting...")
fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot hilat first (behind), then random on top
for label, p, color, ls, n in reversed(traces):
    ax1.hist(p, bins=bins, alpha=0.25, color=color,
             histtype='stepfilled', linewidth=0)
    ax1.hist(p, bins=bins, color=color, histtype='step',
             linewidth=1.5, label=f"{label} counts")

ax1.set_yscale("log")
ax1.set_ylim(0.5, 1e4)
ax1.set_xlabel("P(gal)", fontsize=14, fontweight="bold")
ax1.tick_params(axis='y', labelcolor="steelblue", labelsize=12)
plt.setp(ax1.get_yticklabels(), fontweight="bold")
ax1.tick_params(axis='x', labelsize=12)
ax1.grid(alpha=0.3)

# Right y-axis: fraction curves
ax2 = ax1.twinx()
for label, p, color, ls, n in traces:
    counts = np.array([(p >= t).sum() for t in t_range])
    frac = counts / n * 100
    ax2.plot(t_range, frac, color=color, lw=1.5, ls='--',
             label=f"{label} fraction (%)")

ax2.set_yscale("log")
ax2.set_ylim(0.05, 100)
ax2.tick_params(axis='y', labelcolor="#E53935", labelsize=12)
plt.setp(ax2.get_yticklabels(), fontweight="bold")

# Threshold markers
for t, col, ls, lbl in [(0.5, 'black', '--', 'Default (0.5)'),
                         (0.9, 'green', '--', 'Threshold (0.9)')]:
    ax1.axvline(t, color=col, lw=1.5, ls=ls, alpha=0.5)

# Inline gap labels on fraction curves
bbox_gap = dict(boxstyle='round,pad=0.15', facecolor='white',
                edgecolor='none', alpha=1.0)
for label, p, color, ls, n in traces:
    for t in [0.5, 0.9]:
        pct = 100 * (p >= t).mean()
        ax2.text(t, pct, f"{pct:.1f}%", fontsize=13, fontweight="bold",
                 color=color, ha='center', va='center', bbox=bbox_gap, zorder=10)

# Combined legend
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=12, loc='upper right')

out = PROJECT_DIR / "output" / "pgal_gaia_whilat.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> {out}")

# ── 2-panel plot: Random (left) and High-lat (right) ──
print("\n[3] Plotting 2-panel split...")
fig, (ax_r, ax_h) = plt.subplots(1, 2, figsize=(16, 5))
fig.subplots_adjust(wspace=0.25)

# Group traces by sample
sample_traces = {"Random": [], "High-lat": []}
for label, p, color, ls, n in traces:
    for sname in sample_traces:
        if sname in label:
            sample_traces[sname].append((label, p, color, ls, n))

for ax, sample_name in zip([ax_r, ax_h], ["Random", "High-lat"]):
    st = sample_traces[sample_name]

    # Left y-axis: histograms
    for label, p, color, ls, n in reversed(st):
        ax.hist(p, bins=bins, alpha=0.5, color=color,
                histtype='stepfilled', linewidth=0)
        ax.hist(p, bins=bins, color=color, histtype='step',
                linewidth=1.5, ls=ls, label=label)

    ax.set_yscale("log")
    ax.set_ylim(0.5, 1e4)
    ax.set_xlabel("P(gal)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Source count", fontsize=14, fontweight="bold",
                  color="steelblue", labelpad=10)
    ax.tick_params(axis='y', labelcolor="steelblue", labelsize=12)
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    ax.tick_params(axis='x', labelsize=12)
    ax.grid(alpha=0.3)
    ax.set_title(f"{sample_name} Gaia sources", fontsize=14, fontweight="bold")

    # Right y-axis: fraction curves
    ax2 = ax.twinx()
    for label, p, color, ls, n in st:
        counts = np.array([(p >= t).sum() for t in t_range])
        frac = counts / n * 100
        ax2.plot(t_range, frac, color=color, lw=1.5, ls=ls)

    ax2.set_yscale("log")
    ax2.set_ylim(0.05, 100)
    ax2.set_ylabel("Fraction above threshold (%)", fontsize=14,
                   fontweight="bold", color="#E53935", labelpad=10)
    ax2.set_yticks([0.1, 0.5, 1, 2, 5, 10, 20, 50, 100])
    ax2.set_yticklabels(["0.1", "0.5", "1", "2", "5", "10", "20", "50", "100"])
    ax2.tick_params(axis='y', labelcolor="#E53935", labelsize=12)
    plt.setp(ax2.get_yticklabels(), fontweight="bold")

    for t, col, mls in [(0.5, 'black', '--'), (0.9, 'green', '--')]:
        ax.axvline(t, color=col, lw=1.5, ls=mls, alpha=0.5)

    # Inline labels
    bbox_gap2 = dict(boxstyle='round,pad=0.15', facecolor='white',
                     edgecolor='none', alpha=1.0)
    for label, p, color, ls, n in st:
        for t in [0.5, 0.9]:
            pct = 100 * (p >= t).mean()
            ax2.text(t, pct, f"{pct:.1f}%", fontsize=11, fontweight="bold",
                     color=color, ha='center', va='center', bbox=bbox_gap2, zorder=10)

    ax.legend(fontsize=9, loc='upper right')

out2 = PROJECT_DIR / "output" / "pgal_gaia_whilat_split.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> {out2}")
