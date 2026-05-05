#!/usr/bin/env python3
"""
Compare Bayesian recalibration between Original and Optuna F1 models.

Shows how galaxy selection purity and contamination change with
the assumed true galaxy fraction, for both models overlaid.

solid = Original, dashed = Optuna F1

Reads:
  output/umap/arrays/q1_indiv.npz
  output/xgboost_q1/xgb_classifier_q1.json
  output/xgboost_q1_op_f1/xgb_classifier_q1.json

Writes:
  output/pgal_recalibration_compare.png

Usage:
    python src/analyze_pgal_recalibration_compare.py
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"

SEEDS = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]

MODELS = [
    ("Original",
     PROJECT_DIR / "output" / "xgboost_q1" / "seed_models",
     "-"),
    ("Optuna F1",
     PROJECT_DIR / "output" / "xgboost_q1_op_f1" / "seed_models",
     "--"),
]

THRESHOLDS_MARK = [0.5, 0.9, 0.95]  # thresholds to annotate with points


def ensemble_predict(seed_dir, X, feat_names):
    probas = []
    for seed in SEEDS:
        model_path = seed_dir / f"xgb_classifier_q1_seed{seed}.json"
        if not model_path.exists():
            continue
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        probas.append(model.predict_proba(X)[:, 1])
    return np.mean(probas, axis=0)


def main():
    feat_names = list(SELECTED_GAIA_PARAMETERS)

    # Load Q1 data
    print("[1] Loading Q1 data...")
    q1 = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    X_raw, y = q1["X_raw"], q1["y"]

    # Seed-42 test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_raw, y, test_size=0.15, random_state=42, stratify=y)
    pi_q1 = (y_test == 1).sum() / len(y_test)
    print(f"  Test: {len(X_test):,}  Galaxy fraction: {pi_q1:.2%}")

    # Compute TPR/FPR for each model at each threshold
    pi_range = np.logspace(-4, -0.3, 500)
    mark_points = [(0.01, "1%"), (0.02, "2%"), (0.03, "3%")]

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    fig.subplots_adjust(wspace=0.18)

    # Continuous threshold range from 0.5 to 0.99
    thresh_range = np.linspace(0.5, 0.99, 50)
    cmap = cm.RdYlGn  # red (low threshold) → green (high threshold)
    norm = mcolors.Normalize(vmin=0.5, vmax=1.0)

    for model_label, seed_dir, ls in MODELS:
        print(f"\n[2] {model_label}: ensemble prediction...")
        y_proba = ensemble_predict(seed_dir, X_test, feat_names)

        for thresh in thresh_range:
            pred_gal = y_proba >= thresh
            tp = (pred_gal & (y_test == 1)).sum()
            fp = (pred_gal & (y_test == 0)).sum()
            fn = (~pred_gal & (y_test == 1)).sum()
            tn = (~pred_gal & (y_test == 0)).sum()
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)

            prec = tpr * pi_range / (tpr * pi_range + fpr * (1 - pi_range) + 1e-12)
            col = cmap(norm(thresh))

            axes[0].plot(pi_range * 100, prec * 100, color=col,
                         lw=1.2, ls=ls, alpha=0.7)

        # Mark specific thresholds with annotations
        for thresh_mark in THRESHOLDS_MARK:
            pred_gal = y_proba >= thresh_mark
            tp = (pred_gal & (y_test == 1)).sum()
            fp = (pred_gal & (y_test == 0)).sum()
            fn = (~pred_gal & (y_test == 1)).sum()
            tn = (~pred_gal & (y_test == 0)).sum()
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            col = cmap(norm(thresh_mark))

            if model_label == "Original":
                # Custom offsets per (threshold, pi) to avoid overlap
                annot_offsets = {
                    (0.5,  0.01): (8, -5),
                    (0.5,  0.02): (8, -5),
                    (0.5,  0.03): (8, -5),
                    (0.9,  0.01): (8, -5),
                    (0.9,  0.02): (8, -5),
                    (0.9,  0.03): (8, -5),
                    (0.95, 0.01): (-18, 5),
                    (0.95, 0.02): (-18, 5),
                    (0.95, 0.03): (-40, 8),
                }
                for pi_mark, _ in mark_points:
                    p_mark = tpr * pi_mark / (tpr * pi_mark + fpr * (1 - pi_mark) + 1e-12)
                    axes[0].scatter(pi_mark * 100, p_mark * 100, s=40, zorder=5,
                                    color=col, edgecolors="k", linewidths=0.5)
                    oxy = annot_offsets.get((thresh_mark, pi_mark), (8, 2))
                    # Include threshold in label for the 3% point
                    if pi_mark == 0.03:
                        lbl = f"{p_mark:.0%} (t={thresh_mark})"
                    else:
                        lbl = f"{p_mark:.0%}"
                    axes[0].annotate(
                        lbl, xy=(pi_mark * 100, p_mark * 100),
                        xytext=oxy, textcoords="offset points",
                        fontsize=7, color=col, fontweight="bold")

                print(f"  Threshold {thresh_mark}: TPR={tpr:.4f} FPR={fpr:.4f}")

    # Finalize [0] Purity
    ax = axes[0]
    ax.axhline(50, color="grey", lw=1, ls=":", alpha=0.5)
    ax.axvspan(1, 3, color="steelblue", alpha=0.12, zorder=0,
               label="Est. Gaia gal. fraction (1\u20133%)")
    ax.axvline(pi_q1 * 100, color="coral", lw=1.5, ls="--",
               label=f"Q1 rate ({pi_q1:.0%})")
    ax.set_xlabel("True galaxy fraction (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Galaxy selection purity (%)", fontsize=13, fontweight="bold")
    ax.set_title("Purity of galaxy selection vs true galaxy fraction",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 50)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # [2] 2D heatmap: purity as function of (galaxy fraction, threshold)
    # Use Original model
    print("\n[3] Building 2D purity heatmap...")
    _, orig_seed_dir, _ = MODELS[0]
    y_proba_orig = ensemble_predict(orig_seed_dir, X_test, feat_names)

    thresh_grid = np.linspace(0.3, 0.99, 70)
    pi_grid = np.logspace(-2.5, -0.3, 80)  # 0.3% to 50%
    purity_2d = np.zeros((len(thresh_grid), len(pi_grid)))

    for i, t in enumerate(thresh_grid):
        pred_gal = y_proba_orig >= t
        tp = (pred_gal & (y_test == 1)).sum()
        fp = (pred_gal & (y_test == 0)).sum()
        fn = (~pred_gal & (y_test == 1)).sum()
        tn = (~pred_gal & (y_test == 0)).sum()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        for j, pi in enumerate(pi_grid):
            purity_2d[i, j] = tpr * pi / (tpr * pi + fpr * (1 - pi) + 1e-12)

    ax = axes[1]
    im = ax.pcolormesh(pi_grid * 100, thresh_grid, purity_2d * 100,
                        cmap='RdYlGn', shading='auto', vmin=0, vmax=100)
    # Contour lines at key purity levels
    cs = ax.contour(pi_grid * 100, thresh_grid, purity_2d * 100,
                     levels=[25, 50, 75, 90, 95], colors='black',
                     linewidths=0.8, linestyles='--')
    ax.clabel(cs, fmt='%d%%', fontsize=8)

    # Mark the 1-3% galaxy fraction band
    ax.axvspan(1, 3, color='white', alpha=0.15, zorder=2)
    ax.axvline(1, color='white', lw=0.8, ls=':', alpha=0.5)
    ax.axvline(3, color='white', lw=0.8, ls=':', alpha=0.5)
    # Q1 training ratio
    ax.axvline(pi_q1 * 100, color='coral', lw=1.5, ls='--', alpha=0.8, zorder=3)
    ax.text(pi_q1 * 100 * 1, 0.82, f'Q1 \n ({pi_q1:.0%})', fontsize=8,
            color='coral', fontweight='bold')

    ax.set_xscale('log')
    ax.set_xlim(pi_grid.min() * 100, pi_grid.max() * 100)
    ax.set_xlabel("True galaxy fraction (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("P(gal) threshold", fontsize=13, fontweight="bold")
    ax.set_title("Galaxy selection purity (%)", fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(labelsize=11)
    cb2 = fig.colorbar(im, ax=ax, pad=0.02)
    cb2.set_label("Purity (%)", fontsize=11, fontweight="bold")
    cb2.ax.tick_params(labelsize=10)

    # Colorbar for threshold (panel 0)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes[0], pad=0.02, shrink=0.8)
    cb.set_label("P(gal) threshold", fontsize=11, fontweight="bold")
    cb.ax.tick_params(labelsize=10)
    axes[0].tick_params(labelsize=11)

    fig.suptitle("Bayesian re-calibration of XGBoost galaxy selection",
                 fontsize=14, fontweight="bold", y=1.01)

    out = PROJECT_DIR / "output" / "pgal_recalibration_compare.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
