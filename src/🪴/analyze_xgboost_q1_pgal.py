#!/usr/bin/env python3
"""
Investigate XGBoost P(gal) threshold on random Gaia sources
using the Original Q1-only model.

Two plots:
  1. P(gal) distribution of the 50k random Gaia sample, with threshold
     markers and predicted galaxy counts.
  2. Bayesian re-calibration: how galaxy-selection purity degrades as the
     true galaxy fraction drops below the Q1 training rate (~26%).

Reads:
  output/umap/arrays/gaia_random.npz          (Gaia features)
  WSDB (Q1 data for per-seed TPR/FPR evaluation)
  output/xgboost_q1/seed_models/*.json

Writes (into output/xgboost_q1/):
  gaia_pgal_distribution.png
  gaia_base_rate_recalibration.png

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_pgal_gaia.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR   = PROJECT_DIR / "output" / "xgboost_q1"
MODEL_LABEL = "Q1-only"

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'
MAG_CUT   = 17
CLASS_MAPPING = {'STAR': 0, 'GALAXY': 1}
SEEDS = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]


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
    bins = np.linspace(0, 1, 101)
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
                    edgecolor='none', alpha=0.7)
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
    bbox_gap = dict(boxstyle='round,pad=0.15', facecolor='white',
                    edgecolor='none', alpha=0.7)
    for thresh, tpr, fpr, col in threshold_curves:
        prec = tpr * pi_range / (tpr * pi_range + fpr * (1 - pi_range))
        ax.plot(pi_range * 100, prec * 100, color=col, lw=2.5,
                label=f"Threshold {thresh} (TPR={tpr:.3f}, FPR={fpr:.4f})")
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

    # ── 1. Load random Gaia sources ─────────────────────────────────────────
    gaia_path = PROJECT_DIR / "output" / "gaia_50k" / "gaia_random_50k.npz"
    if not gaia_path.exists():
        print(f"ERROR: {gaia_path} not found. Run query_gaia_random.py first.")
        return 1

    print(f"[1] Loading Gaia sources ({MODEL_LABEL})...")
    feat_names = list(SELECTED_GAIA_PARAMETERS)
    X_gaia = np.load(gaia_path)["X"].astype(float)
    print(f"  {len(X_gaia):,} sources, {len(feat_names)} features")

    # ── 2. Ensemble predict P(gal) ───────────────────────────────────────────
    seed_dir = MODEL_DIR / "seed_models"
    if not seed_dir.exists():
        print(f"ERROR: {seed_dir} not found. Run train_xgboost_q1.py first.")
        return 1

    print(f"\n[2] Ensemble prediction on Gaia sample...")
    probas = []
    for seed in SEEDS:
        mp = seed_dir / f"xgb_classifier_q1_seed{seed}.json"
        if not mp.exists():
            continue
        m = xgb.XGBClassifier()
        m.load_model(str(mp))
        probas.append(m.predict_proba(X_gaia)[:, 1])
    p = np.mean(probas, axis=0)
    print(f"  {len(probas)} seed models")

    print(f"  P(gal) range: [{p.min():.4f}, {p.max():.4f}]")
    print(f"  P(gal) > 0.5: {(p >= 0.5).sum():,}  ({100*(p >= 0.5).mean():.1f}%)")
    print(f"  P(gal) > 0.7: {(p >= 0.7).sum():,}  ({100*(p >= 0.7).mean():.1f}%)")
    print(f"  P(gal) > 0.9: {(p >= 0.9).sum():,}  ({100*(p >= 0.9).mean():.1f}%)")

    # ── 3. Compute TPR / FPR via per-seed evaluation on Q1 data ─────────────
    print(f"\n[3] Computing TPR / FPR at multiple thresholds on Q1 data...")
    username = os.environ.get("PGUSER")
    if not username:
        print("ERROR: PGUSER not set.")
        return 1

    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in feat_names])
    query = f"""
        SELECT {gaia_cols},
               CASE WHEN p_phz_classification = 1 THEN 'STAR'
                    WHEN p_phz_classification = 2 THEN 'GALAXY' END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep
        WHERE p_phz_classification IN (1, 2)
          AND g_phot_g_mean_mag >= {MAG_CUT}
        ORDER BY g_random_index
    """
    df_q1 = pd.read_sql_query(query, conn)
    conn.close()
    for c in feat_names:
        df_q1[c] = pd.to_numeric(df_q1[c].replace('NOT_AVAILABLE', np.nan), errors='coerce')
    X_q1 = df_q1[feat_names].values
    y_q1 = df_q1['label'].map(CLASS_MAPPING).values
    print(f"  Q1: {len(X_q1):,} sources  STAR: {(y_q1==0).sum():,}  GALAXY: {(y_q1==1).sum():,}")

    pi_q1 = (y_q1 == 1).sum() / len(y_q1)

    # Per-seed predictions
    seed_dir = MODEL_DIR / "seed_models"
    seed_preds = []
    for seed in SEEDS:
        mp = seed_dir / f"xgb_classifier_q1_seed{seed}.json"
        if not mp.exists():
            continue
        _, X_ts, _, y_ts = train_test_split(
            X_q1, y_q1, test_size=0.15, random_state=seed, stratify=y_q1)
        m = xgb.XGBClassifier()
        m.load_model(str(mp))
        seed_preds.append((y_ts, m.predict_proba(X_ts)[:, 1]))
    print(f"  Loaded {len(seed_preds)} seed models")

    thresholds_to_plot = [
        (0.5, "red"),
        (0.7, "orange"),
        (0.9, "green"),
    ]
    threshold_curves = []
    for thresh, col in thresholds_to_plot:
        tp_s, fp_s, fn_s, tn_s = 0, 0, 0, 0
        for y_ts, p_ts in seed_preds:
            pred_gal = p_ts >= thresh
            tp_s += (pred_gal & (y_ts == 1)).sum()
            fp_s += (pred_gal & (y_ts == 0)).sum()
            fn_s += (~pred_gal & (y_ts == 1)).sum()
            tn_s += (~pred_gal & (y_ts == 0)).sum()
        n = len(seed_preds)
        tpr = (tp_s/n) / max(tp_s/n + fn_s/n, 1)
        fpr = (fp_s/n) / max(fp_s/n + tn_s/n, 1)
        prec = (tp_s/n) / max(tp_s/n + fp_s/n, 1)
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
