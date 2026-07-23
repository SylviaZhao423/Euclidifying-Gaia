#!/usr/bin/env python3
"""
P(gal) distribution split into 3 panels: Random / High-lat / Low-lat.

Applies Q1-only model (10-seed ensemble) to all three Gaia samples.

Reads:
  output/gaia_50k/gaia_random_50k.npz
  output/gaia_50k/gaia_hilat_50k.npz
  output/gaia_50k/gaia_lolat_50k.npz
  output/xgboost_q1/seed_models/*.json

Writes:
  output/gaia_lat_exploration/pgal_gaia_3way_split.png

Usage:
    python src/analyze_pgal_gaia_3way_split.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
GAIA_DIR = PROJECT_DIR / "output" / "gaia_50k"
SEED_DIR = PROJECT_DIR / "output" / "xgboost_q1" / "seed_models"
OUTPUT_DIR = PROJECT_DIR / "output"
FEAT_NAMES = list(SELECTED_GAIA_PARAMETERS)
SEEDS = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]

SAMPLES = [
    ("Random",   "gaia_random_50k.npz", "#2196F3"),
    ("High-lat", "gaia_hilat_50k.npz",  "lightpink"),
    ("Low-lat",  "gaia_lolat_50k.npz",  "#FF9800"),
]


def ensemble_predict(X):
    """Predict P(gal) using 10-seed ensemble mean."""
    probas = []
    for seed in SEEDS:
        mp = SEED_DIR / f"xgb_classifier_q1_seed{seed}.json"
        if not mp.exists():
            continue
        m = xgb.XGBClassifier()
        m.load_model(str(mp))
        probas.append(m.predict_proba(X)[:, 1])
    return np.mean(probas, axis=0)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1] Loading and predicting...")
    seed_count = sum(1 for s in SEEDS if (SEED_DIR / f"xgb_classifier_q1_seed{s}.json").exists())
    print(f"  Using {seed_count} seed models")

    data = {}
    for sample_name, npz_file, color in SAMPLES:
        g = np.load(GAIA_DIR / npz_file)
        X = g["X"].astype(float)
        p = ensemble_predict(X)
        n = len(X)
        data[sample_name] = (X, p, color, n)
        print(f"  {sample_name}: {n:,} sources  "
              f"P>0.5={100*(p>=0.5).mean():.1f}%  "
              f"P>0.9={100*(p>=0.9).mean():.1f}%")

    bins = np.linspace(0, 1, 101)
    t_range = np.linspace(0, 1, 501)
    bbox_gap = dict(boxstyle='round,pad=0.15', facecolor='white',
                    edgecolor='none', alpha=1.0)

    print("\n[2] Plotting P(gal) split (3 panels)...")
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 4.2))
    fig.subplots_adjust(wspace=0.11)

    for i, (ax, (sample_name, _, _)) in enumerate(zip(axes, SAMPLES)):
        X, p, color, n = data[sample_name]

        ax.hist(p, bins=bins, alpha=0.25, color=color,
                histtype='stepfilled', linewidth=0)
        ax.hist(p, bins=bins, color=color, histtype='step',
                linewidth=1.5, label=f"{sample_name} counts")
        ax.set_yscale("log")
        ax.set_ylim(0.5, 1e4)
        ax.set_xlabel("P(gal)", fontsize=14, fontweight="bold")
        ax.tick_params(axis='x', labelsize=12)
        ax.grid(alpha=0.3)
        ax.set_title(f"{sample_name} Gaia sources", fontsize=14, fontweight="bold")

        # Left y-axis: label only on leftmost panel
        if i == 0:
            ax.set_ylabel("Source count", fontsize=14, fontweight="bold",
                          color="steelblue", labelpad=10)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis='y', labelleft=False)
        ax.tick_params(axis='y', labelcolor="steelblue", labelsize=12)
        plt.setp(ax.get_yticklabels(), fontweight="bold")

        ax2 = ax.twinx()
        counts_arr = np.array([(p >= t).sum() for t in t_range])
        frac = counts_arr / n * 100
        ax2.plot(t_range, frac, color="#E53935", lw=1.5)
        ax2.set_yscale("log")
        ax2.set_ylim(0.05, 100)
        ax2.set_yticks([0.1, 0.5, 1, 2, 5, 10, 20, 50, 100])
        ax2.set_yticklabels(["0.1", "0.5", "1", "2", "5", "10", "20", "50", "100"])

        # Right y-axis: label only on rightmost panel
        if i == 2:
            ax2.set_ylabel("Fraction above threshold (%)", fontsize=14,
                           fontweight="bold", color="#E53935", labelpad=10)
            ax2.tick_params(axis='y', labelcolor="#E53935", labelsize=12)
            plt.setp(ax2.get_yticklabels(), fontweight="bold")
        else:
            ax2.set_ylabel("")
            ax2.tick_params(axis='y', labelright=False)

        for t, col, ls in [(0.5, 'black', '--'), (0.9, 'green', '--')]:
            ax.axvline(t, color=col, lw=1.5, ls=ls, alpha=0.5)

        for t in [0.5, 0.9]:
            pct = 100 * (p >= t).mean()
            ax2.text(t, pct, f"{pct:.1f}%", fontsize=14, fontweight="bold",
                     color="#E53935", ha='center', va='center', bbox=bbox_gap, zorder=10)

        ax.legend(loc='upper right', fontsize=13)

    out = OUTPUT_DIR / "pgal_gaia_3way_split.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
