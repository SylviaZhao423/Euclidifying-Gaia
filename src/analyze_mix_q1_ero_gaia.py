#!/usr/bin/env python3
"""
Apply Q1+ERO mixed models to Gaia sources and analyse predictions.

Requires Gaia sample (from gaia_random.npz or gaia_hilat.npz).
Each seed model is evaluated on its own held-out Q1 test split (no leakage).

Produces:
  1. mix_threshold_pgal_overlay.png     — P(gal) distribution on Gaia per ratio
  2. mix_threshold_analysis.csv         — threshold analysis table

Reads:
  WSDB (Q1 data, same query as training scripts)
  output/xgboost_mix_q1_ero/ratio_sg*/seed_models/xgb_classifier_q1_seed*.json
  output/gaia_50k/gaia_random_50k.npz (or gaia_hilat_50k.npz)

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_mix_q1_ero_gaia.py
    PGUSER=siwei_zhao_2025 python src/analyze_mix_q1_ero_gaia.py --hilat
    PGUSER=siwei_zhao_2025 python src/analyze_mix_q1_ero_gaia.py --dir xgboost_mix_q1_ero_g
    PGUSER=siwei_zhao_2025 python src/analyze_mix_q1_ero_gaia.py --dir xgboost_mix_q1_ero_cs0.5
    PGUSER=siwei_zhao_2025 python src/analyze_mix_q1_ero_gaia.py --dir xgboost_mix_q1_ero_g_nocs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

import os

PROJECT_DIR = Path(__file__).resolve().parents[1]
MIX_DIR     = PROJECT_DIR / "output" / "xgboost_mix_q1_ero"

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'
MAG_CUT   = 17
CLASS_MAPPING = {'STAR': 0, 'GALAXY': 1}
SEEDS = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]


def query_q1(username):
    """Query Q1 data — same query as training scripts."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    feat_names = list(SELECTED_GAIA_PARAMETERS)
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
    df = pd.read_sql_query(query, conn)
    conn.close()
    for c in feat_names:
        df[c] = pd.to_numeric(df[c].replace('NOT_AVAILABLE', np.nan), errors='coerce')
    return df[feat_names].values, df['label'].map(CLASS_MAPPING).values


def _find_seed_dir(ratio):
    """Find seed_models directory for a given ratio."""
    for d in MIX_DIR.glob("ratio_*"):
        if "sweep" in d.name:
            continue
        try:
            num_str = d.name.replace("ratio_sg", "").split("_")[0]
            r = float(num_str)
            if abs(r - ratio) < 0.1:
                sd = d / "seed_models"
                if sd.exists():
                    return sd
        except ValueError:
            continue
    return None


def _get_ratios():
    """Scan ratio directories and return sorted list of ratios."""
    ratios = []
    for d in sorted(MIX_DIR.glob("ratio_*")):
        if "sweep" in d.name:
            continue
        if not (d / "q1_seed_metrics.json").exists():
            continue
        try:
            num_str = d.name.replace("ratio_sg", "").split("_")[0]
            ratios.append(float(num_str))
        except ValueError:
            continue
    return sorted(set(ratios))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply Q1+ERO models to Gaia sources")
    parser.add_argument("--dir", type=str, default="xgboost_mix_q1_ero",
                        help="Output directory name under output/")
    parser.add_argument("--hilat", action="store_true",
                        help="Use high-latitude Gaia sample instead of random")
    args = parser.parse_args()

    global MIX_DIR
    MIX_DIR = PROJECT_DIR / "output" / args.dir

    username = os.environ.get("PGUSER")
    if not username:
        print("ERROR: PGUSER not set.")
        return 1

    # ── 1. Load data ────────────────────────────────────────────────────────
    if args.hilat:
        save_dir = MIX_DIR / "hilat"
        save_dir.mkdir(parents=True, exist_ok=True)
        sample_label = "high-lat"
    else:
        save_dir = MIX_DIR
        sample_label = "random"

    gaia_file = "gaia_hilat_50k.npz" if args.hilat else "gaia_random_50k.npz"
    gaia_path = PROJECT_DIR / "output" / "gaia_50k" / gaia_file
    if not gaia_path.exists():
        print(f"ERROR: {gaia_path} not found. Run query_gaia_random.py first.")
        return 1
    X_gaia = np.load(gaia_path)["X"].astype(float)
    print(f"[1] Loaded {len(X_gaia):,} {sample_label} Gaia sources")

    print(f"  Querying Q1 data from WSDB...")
    X_q1, y_q1 = query_q1(username)
    print(f"  Q1 data: {len(X_q1):,} sources  "
          f"STAR: {(y_q1==0).sum():,}  GALAXY: {(y_q1==1).sum():,}")

    ratios = _get_ratios()
    print(f"  Found {len(ratios)} ratios: {ratios}")

    # ── 2. Per-ratio predictions ────────────────────────────────────────────
    thresholds = [0.5, 0.7, 0.9]
    recal_display_ratios = {3.0, 5.0, max(ratios)}
    thresh_results = []
    all_pgal = []

    for ratio in ratios:
        seed_dir = _find_seed_dir(ratio)
        if seed_dir is None:
            print(f"  Ratio {ratio}: no seed_models dir, skipping")
            continue

        label = f"{ratio:.0f}:1" if ratio == int(ratio) else f"{ratio:.1f}:1"
        print(f"\n  Ratio {label}...")
        seed_paths = sorted(seed_dir.glob("xgb_classifier_q1_seed*.json"))
        if not seed_paths:
            print(f"    No seed models found, skipping")
            continue

        # Ensemble P(gal) on Gaia + per-seed Q1 test predictions
        gaia_probas = []
        seed_preds = []
        for sp in seed_paths:
            seed_val = int(sp.stem.replace("xgb_classifier_q1_seed", ""))
            model = xgb.XGBClassifier()
            model.load_model(str(sp))
            gaia_probas.append(model.predict_proba(X_gaia)[:, 1])
            _, X_ts, _, y_ts = train_test_split(
                X_q1, y_q1, test_size=0.15, random_state=seed_val, stratify=y_q1)
            seed_preds.append((y_ts, model.predict_proba(X_ts)[:, 1]))
        p_gaia = np.mean(gaia_probas, axis=0)
        all_pgal.append((label, p_gaia))

        # Per-seed TPR/FPR
        for t in thresholds:
            tp_sum, fp_sum, fn_sum, tn_sum = 0, 0, 0, 0
            for y_ts, p_ts in seed_preds:
                pred_gal = p_ts >= t
                tp_sum += (pred_gal & (y_ts == 1)).sum()
                fp_sum += (pred_gal & (y_ts == 0)).sum()
                fn_sum += (~pred_gal & (y_ts == 1)).sum()
                tn_sum += (~pred_gal & (y_ts == 0)).sum()
            n = len(seed_preds)
            tp = tp_sum / n
            fp = fp_sum / n
            fn = fn_sum / n
            tn = tn_sum / n
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            n_gaia_above = int((p_gaia >= t).sum())
            pct_gaia = 100 * n_gaia_above / len(p_gaia)
            purity_1 = tpr * 0.01 / (tpr * 0.01 + fpr * 0.99 + 1e-12)
            purity_3 = tpr * 0.03 / (tpr * 0.03 + fpr * 0.97 + 1e-12)

            thresh_results.append({
                "ratio": ratio, "ratio_label": label,
                "threshold": t, "tpr": tpr, "fpr": fpr,
                "n_gaia": n_gaia_above, "pct_gaia": pct_gaia,
                "purity_1pct": purity_1, "purity_3pct": purity_3,
            })

        print(f"    P(gal)>0.5: {(p_gaia>=0.5).sum():,} ({100*(p_gaia>=0.5).mean():.1f}%)  "
              f"P(gal)>0.9: {(p_gaia>=0.9).sum():,} ({100*(p_gaia>=0.9).mean():.1f}%)")

    df_thresh = pd.DataFrame(thresh_results)
    df_thresh.to_csv(save_dir / "mix_threshold_analysis.csv", index=False)
    print(f"\n  -> {save_dir / 'mix_threshold_analysis.csv'}")

    # ── 3. P(gal) overlay plot ──────────────────────────────────────────────
    if all_pgal:
        colors = cm.viridis(np.linspace(0, 0.9, len(all_pgal)))
        fig, ax1 = plt.subplots(figsize=(10, 5.5))

        bins = np.linspace(0, 1, 101)
        t_range = np.linspace(0, 1, 501)

        for (label, p_gaia), color in zip(all_pgal, colors):
            ax1.hist(p_gaia, bins=bins, alpha=0.3, color=color,
                     histtype='stepfilled', linewidth=0)
            ax1.hist(p_gaia, bins=bins, color=color,
                     histtype='step', linewidth=1.2, label=label)

        ax1.set_yscale("log")
        ax1.set_ylim(0.5, 2e4)
        ax1.set_xlabel("P(gal)", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Source count", fontsize=14, fontweight='bold', color='steelblue', labelpad=10)
        ax1.tick_params(axis='y', labelcolor='steelblue', labelsize=12)
        plt.setp(ax1.get_yticklabels(), fontweight='bold')
        ax1.tick_params(axis='x', labelsize=12)
        ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        for (label, p_gaia), color in zip(all_pgal, colors):
            counts = np.array([(p_gaia >= t).sum() for t in t_range])
            frac = counts / len(X_gaia) * 100
            ax2.plot(t_range, frac, color=color, lw=1.5)

        ax2.set_yscale("log")
        ax2.set_ylim(0.05, 100)
        ax2.set_ylabel("Fraction above threshold (%)", fontsize=14,
                        fontweight='bold', color='#E53935', labelpad=10)
        ax2.set_yticks([0.1, 0.5, 1, 2, 5, 10, 20, 50, 100])
        ax2.set_yticklabels(["0.1", "0.5", "1", "2", "5", "10", "20", "50", "100"])
        ax2.tick_params(axis='y', labelcolor='#E53935', labelsize=12)
        plt.setp(ax2.get_yticklabels(), fontweight='bold')

        for t, col, ls in [(0.5, 'black', '--'), (0.9, 'green', '--')]:
            ax1.axvline(t, color=col, lw=1.5, ls=ls, alpha=0.5)

        # Inline fraction labels for first (2.8:1) and last (8.6:1) ratio
        bbox_gap = dict(boxstyle='round,pad=0.15', facecolor='white',
                        edgecolor='none', alpha=1.0)
        for idx_label in [0, -1]:  # first and last
            label_l, p_gaia_l = all_pgal[idx_label]
            color_l = colors[idx_label]
            for t in [0.5, 0.9]:
                pct = 100 * (p_gaia_l >= t).mean()
                ax2.text(t, pct, f"{label_l}: {pct:.1f}%", fontsize=13, fontweight="bold",
                         color=color_l, ha='center', va='center', bbox=bbox_gap, zorder=10)

        ax1.legend(fontsize=10, loc='upper right', ncols=2)
        # fig.suptitle("P(gal) on Gaia sources — effect of ERO mixing ratio",
        #              fontsize=13, fontweight='bold', y=0.98)
        out = save_dir / "mix_threshold_pgal_overlay.png"
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  -> {out}")

    # ── Print summary ───────────────────────────────────────────────────────
    if len(df_thresh) > 0:
        print(f"\n  {'Ratio':<10} {'Thresh':>6} {'TPR':>7} {'FPR':>7} "
              f"{'N_Gaia':>8} {'%Gaia':>6} {'Pur@1%':>7} {'Pur@3%':>7}")
        print("  " + "-" * 70)
        for _, r in df_thresh.iterrows():
            print(f"  {r['ratio_label']:<10} {r['threshold']:>6.1f} "
                  f"{r['tpr']:>7.4f} {r['fpr']:>7.4f} "
                  f"{int(r['n_gaia']):>8,} {r['pct_gaia']:>5.1f}% "
                  f"{r['purity_1pct']:>6.1%} {r['purity_3pct']:>6.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
