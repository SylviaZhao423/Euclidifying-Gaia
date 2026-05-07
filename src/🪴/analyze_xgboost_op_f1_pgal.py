#!/usr/bin/env python3
"""
Investigate XGBoost P(gal) threshold on random Gaia sources
using the Optuna F1-tuned model.

Two plots:
  1. P(gal) distribution of the 50k random Gaia sample, with threshold
     markers and predicted galaxy counts.
  2. Bayesian re-calibration: how galaxy-selection purity degrades as the
     true galaxy fraction drops below the Q1 training rate (~26%).

Reads:
  output/umap/arrays/gaia_random.npz          (Gaia features)
  output/umap/arrays/q1_indiv.npz             (Q1 labels for TPR/FPR)
  output/xgboost_q1_op_f1/xgb_classifier_q1.json

Writes (into output/xgboost_q1_op_f1/):
  gaia_pgal_distribution.png
  gaia_base_rate_recalibration.png

Usage:
  python src/analyze_gaia_pgal_threshold_op_f1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
MODEL_DIR   = PROJECT_DIR / "output" / "xgboost_q1_op_f1"
MODEL_LABEL = "Q1-only (Optuna F1)"


# ── Plot 1: P(gal) distribution ─────────────────────────────────────────────

def plot_pgal_distribution(p: np.ndarray, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(wspace=0.2)

    thresholds_mark = [
        (0.5, "red",    f"Threshold 0.5\n(n\u22650.5 = {(p >= 0.5).sum():,}, {100*(p >= 0.5).mean():.1f}%)"),
        (0.7, "orange", f"Threshold 0.7\n(n\u22650.7 = {(p >= 0.7).sum():,}, {100*(p >= 0.7).mean():.1f}%)"),
        (0.9, "green",  f"Threshold 0.9\n(n\u22650.9 = {(p >= 0.9).sum():,}, {100*(p >= 0.9).mean():.1f}%)"),
    ]

    # [0] Full histogram
    ax = axes[0]
    bins = np.linspace(0, 1, 201)
    ax.hist(p, bins=bins, color="steelblue", alpha=0.8, edgecolor="none")
    for t, col, lab in thresholds_mark:
        ax.axvline(t, color=col, lw=2, ls="--", label=lab)
    ax.set_xlabel("P(gal)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"P(gal) distribution \u2014 {len(p)//1000}k random Gaia (G \u2265 17)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # [1] Predicted galaxy count vs threshold
    ax = axes[1]
    t_range = np.linspace(0, 1, 501)
    counts = np.array([(p >= t).sum() for t in t_range])
    ax.plot(t_range, counts, color="steelblue", lw=2)
    ax.axhline(len(p) * 0.01, color="grey", lw=1, ls=":",
               label="1% of sample")
    for t, col, _ in thresholds_mark:
        ax.axvline(t, color=col, lw=1.5, ls="--")
    bbox_gap = dict(boxstyle='round,pad=0.15', facecolor='white',
                    edgecolor='none', alpha=1.0)
    for t, col, _ in thresholds_mark:
        c = int((p >= t).sum())
        ax.text(t, c, f"{c:,}", fontsize=9, fontweight="bold",
                color=col, ha='center', va='center', bbox=bbox_gap, zorder=10)
    ax.set_xlabel("P(gal) threshold", fontsize=11)
    ax.set_ylabel("N predicted galaxies", fontsize=11)
    ax.set_title("Predicted galaxy count vs threshold",
                 fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"XGBoost P(gal) on random Gaia sources \u2014 {MODEL_LABEL} model",
        fontsize=13, fontweight="bold", y=0.98,
    )
    out = output_dir / "gaia_pgal_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")


# ── Plot 2: Bayesian re-calibration ─────────────────────────────────────────

def plot_base_rate_recalibration(
    threshold_curves: list[tuple[float, float, float, str]],
    pi_q1: float,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(wspace=0.2)

    pi_range = np.logspace(-4, -0.3, 500)
    mark_points = [(0.01, "1%"), (0.03, "3%"), (0.05, "5%")]

    # [0] Purity vs galaxy fraction
    ax = axes[0]
    for thresh, tpr, fpr, col in threshold_curves:
        prec = tpr * pi_range / (tpr * pi_range + fpr * (1 - pi_range))
        ax.plot(pi_range * 100, prec * 100, color=col, lw=2.5,
                label=f"Threshold {thresh} (TPR={tpr:.3f}, FPR={fpr:.4f})")
        bbox_gap = dict(boxstyle='round,pad=0.15', facecolor='white',
                        edgecolor='none', alpha=1.0)
        for pi_mark, pi_label in mark_points:
            p_mark = tpr * pi_mark / (tpr * pi_mark + fpr * (1 - pi_mark))
            ax.text(pi_mark * 100, p_mark * 100, f"{p_mark:.0%}",
                    fontsize=7, fontweight="bold", color=col,
                    ha='center', va='center', bbox=bbox_gap, zorder=10)
    ax.axhline(50, color="grey", lw=1, ls=":", alpha=0.5, label="50% purity")
    ax.axvspan(1, 3, color="steelblue", alpha=0.12, zorder=0,
               label="Est. Gaia gal. fraction (1\u20133%)")
    ax.axvline(pi_q1 * 100, color="coral", lw=1.5, ls="--",
               label=f"Q1 rate ({pi_q1:.0%})")
    ax.set_xlabel("True galaxy fraction (%)", fontsize=11)
    ax.set_ylabel("Galaxy selection purity (%)", fontsize=11)
    ax.set_title("Purity of galaxy selection vs true galaxy fraction",
                 fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(0.01, 50)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # [1] Contamination vs galaxy fraction
    ax = axes[1]
    for thresh, tpr, fpr, col in threshold_curves:
        prec = tpr * pi_range / (tpr * pi_range + fpr * (1 - pi_range))
        ax.plot(pi_range * 100, (1 - prec) * 100, color=col, lw=2.5,
                label=f"Threshold {thresh}")
        for pi_mark, pi_label in mark_points:
            p_mark = tpr * pi_mark / (tpr * pi_mark + fpr * (1 - pi_mark))
            contam = 1 - p_mark
            ax.text(pi_mark * 100, contam * 100, f"{contam:.0%}",
                    fontsize=7, fontweight="bold", color=col,
                    ha='center', va='center', bbox=bbox_gap, zorder=10)
    ax.axhline(50, color="grey", lw=1, ls=":", alpha=0.5)
    ax.axvspan(1, 3, color="steelblue", alpha=0.12, zorder=0,
               label="Est. Gaia gal. fraction (1\u20133%)")
    ax.axvline(pi_q1 * 100, color="coral", lw=1.5, ls="--",
               label=f"Q1 rate ({pi_q1:.0%})")
    ax.set_xlabel("True galaxy fraction (%)", fontsize=11)
    ax.set_ylabel("Contamination rate (%)", fontsize=11)
    ax.set_title("Star contamination in galaxy selection vs true galaxy fraction",
                 fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(0.01, 50)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Bayesian re-calibration of XGBoost galaxy selection \u2014 {MODEL_LABEL} model",
        fontsize=13, fontweight="bold", y=0.98,
    )
    out = output_dir / "gaia_base_rate_recalibration.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    output_dir = MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load Gaia features from gaia_random.npz ─────────────────────────
    npz_path = ARRAYS_DIR / "gaia_random.npz"
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found.")
        print("Run query_umap_q1_50k.py first.")
        return 1

    print(f"[1] Loading Gaia sources ({MODEL_LABEL})...")
    g = np.load(npz_path, allow_pickle=True)
    feat_names = list(SELECTED_GAIA_PARAMETERS)
    X_gaia = g["X_features"].astype(float)
    print(f"  {len(X_gaia):,} sources, {len(feat_names)} features")

    # ── 2. Load model and predict P(gal) ───────────────────────────────────
    model_path = MODEL_DIR / "xgb_classifier_q1.json"
    q1_path    = ARRAYS_DIR / "q1_indiv.npz"
    if not model_path.exists():
        print(f"ERROR: {model_path} not found.")
        print("Run train_xgboost_q1_op_F1.py first.")
        return 1
    if not q1_path.exists():
        print(f"ERROR: {q1_path} not found.")
        return 1

    print(f"\n[2] Predicting P(gal) on Gaia sample...")
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    dmat_gaia = xgb.DMatrix(X_gaia, feature_names=feat_names)
    p = model.get_booster().predict(dmat_gaia)

    print(f"  P(gal) range: [{p.min():.4f}, {p.max():.4f}]")
    print(f"  P(gal) > 0.5: {(p >= 0.5).sum():,}  ({100*(p >= 0.5).mean():.1f}%)")
    print(f"  P(gal) > 0.7: {(p >= 0.7).sum():,}  ({100*(p >= 0.7).mean():.1f}%)")
    print(f"  P(gal) > 0.9: {(p >= 0.9).sum():,}  ({100*(p >= 0.9).mean():.1f}%)")

    # ── 3. Compute TPR / FPR at multiple thresholds on Q1 data ─────────────
    print(f"\n[3] Computing TPR / FPR at multiple thresholds on Q1 data...")
    q1    = np.load(q1_path, allow_pickle=True)
    X_raw = q1["X_raw"]
    y     = q1["y"]

    dmat_q1 = xgb.DMatrix(X_raw, feature_names=feat_names)
    proba_q1 = model.get_booster().predict(dmat_q1)

    pi_q1 = (y == 1).sum() / len(y)

    thresholds_to_plot = [
        (0.5, "red"),
        (0.7, "orange"),
        (0.9, "green"),
    ]
    threshold_curves = []
    for thresh, col in thresholds_to_plot:
        pred_gal = proba_q1 >= thresh
        tp  = (pred_gal & (y == 1)).sum()
        fp  = (pred_gal & (y == 0)).sum()
        fn  = (~pred_gal & (y == 1)).sum()
        tn  = (~pred_gal & (y == 0)).sum()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        threshold_curves.append((thresh, tpr, fpr, col))
        print(f"  Threshold {thresh}:  TPR={tpr:.4f}  FPR={fpr:.4f}  Precision={prec:.4f}")

    # ── 4. Print threshold table ───────────────────────────────────────────
    TPR_05 = threshold_curves[0][1]
    FPR_05 = threshold_curves[0][2]
    print(f"\n[4] Precision under different true galaxy fractions (threshold=0.5):")
    print(f"  {'Gal fraction':>14} {'Star:Gal':>10} {'Purity':>8} {'Contamination':>14}")
    print("  " + "-" * 50)
    for pi in [0.26, 0.10, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001]:
        prec = TPR_05 * pi / (TPR_05 * pi + FPR_05 * (1 - pi))
        r = (1 - pi) / pi
        print(f"  {pi:>14.3f} {r:>8.0f}:1 {prec:>8.1%} {1-prec:>14.1%}")

    # ── 5. Plots ───────────────────────────────────────────────────────────
    print("\n[5] Plotting P(gal) distribution...")
    plot_pgal_distribution(p, output_dir)

    print("\n[6] Plotting Bayesian re-calibration...")
    plot_base_rate_recalibration(threshold_curves, pi_q1, output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
