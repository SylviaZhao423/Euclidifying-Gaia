#!/usr/bin/env python3
"""
Combined ERO mixing analysis plots.

Produces:
  1. mix_recalibration_combined.png   — purity curves + purity gain heatmap
  2. mix_ratio_sweep_thresholds.png   — precision/F1/recall vs ratio, colored by threshold
  3. mix_threshold_pgal_overlay.png   — P(gal) distribution on Gaia per ratio

Reads:
  WSDB (Q1 data)
  output/xgboost_mix_q1_ero/ratio_sg*/seed_models/*.json
  output/gaia_50k/gaia_random_50k.npz

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_recalibration_combined.py
    PGUSER=siwei_zhao_2025 python src/analyze_recalibration_combined.py --dir xgboost_mix_q1_ero_g18.5
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS
from analyze_mix_q1_ero import make_threshold_cmap

PROJECT_DIR = Path(__file__).resolve().parents[1]

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'
MAG_CUT   = 17
CLASS_MAPPING = {'STAR': 0, 'GALAXY': 1}
SEEDS = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]

SELECT_RATIOS = [8.6]


def query_q1(username):
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


def _find_seed_dir(mix_dir, ratio):
    for d in mix_dir.glob("ratio_*"):
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


def _per_seed_predictions(seed_dir, X_q1, y_q1):
    seed_paths = sorted(seed_dir.glob("xgb_classifier_q1_seed*.json"))
    preds = []
    for sp in seed_paths:
        seed_val = int(sp.stem.replace("xgb_classifier_q1_seed", ""))
        _, X_ts, _, y_ts = train_test_split(
            X_q1, y_q1, test_size=0.15, random_state=seed_val, stratify=y_q1)
        m = xgb.XGBClassifier()
        m.load_model(str(sp))
        preds.append((y_ts, m.predict_proba(X_ts)[:, 1]))
    return preds


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="xgboost_mix_q1_ero")
    parser.add_argument("--ratios", type=float, nargs='+', default=SELECT_RATIOS)
    args = parser.parse_args()

    mix_dir = PROJECT_DIR / "output" / args.dir

    username = os.environ.get("PGUSER")
    if not username:
        print("ERROR: PGUSER not set.")
        return 1

    print("[1] Loading Q1 data...")
    X_q1, y_q1 = query_q1(username)
    pi_q1 = (y_q1 == 1).sum() / len(y_q1)
    print(f"  {len(X_q1):,} sources, pi_q1={pi_q1:.2%}")

    # Find all ratios
    all_ratios = []
    for d in sorted(mix_dir.glob("ratio_*")):
        if "sweep" in d.name:
            continue
        try:
            num_str = d.name.replace("ratio_sg", "").split("_")[0]
            all_ratios.append(float(num_str))
        except ValueError:
            continue
    all_ratios = sorted(set(all_ratios))
    print(f"  Available ratios: {all_ratios}")

    # Delta heatmap ratios (baseline + selected)
    baseline_ratio = all_ratios[0]
    delta_ratios = [baseline_ratio]
    for sr in args.ratios:
        best = min(all_ratios, key=lambda r: abs(r - sr))
        if best not in delta_ratios:
            delta_ratios.append(best)
    n_delta_ratios = len(delta_ratios)
    delta_labels = [f"{r:.0f}:1" if r == int(r) else f"{r:.1f}:1" for r in delta_ratios]

    # All ratios for purity curves
    all_ratio_labels = [f"{r:.0f}:1" if r == int(r) else f"{r:.1f}:1" for r in all_ratios]
    n_all = len(all_ratios)
    print(f"  All ratios (curves): {all_ratios}")
    print(f"  Delta ratios (heatmaps): {delta_ratios}")

    # Compute TPR/FPR grid for ALL ratios
    thresh_fine = np.linspace(0.3, 0.99, 70)
    pi_hm = np.logspace(-2.5, -0.3, 80)
    pi_range = np.logspace(-3, np.log10(0.5), 500)

    print("\n[2] Computing per-seed TPR/FPR...")
    tpr_grid_all = np.zeros((n_all, len(thresh_fine)))
    fpr_grid_all = np.zeros((n_all, len(thresh_fine)))

    for ri, ratio in enumerate(all_ratios):
        seed_dir = _find_seed_dir(mix_dir, ratio)
        if seed_dir is None:
            continue
        print(f"  {all_ratio_labels[ri]}...")
        seed_preds = _per_seed_predictions(seed_dir, X_q1, y_q1)
        for ti, t in enumerate(thresh_fine):
            tp_s, fp_s, fn_s, tn_s = 0, 0, 0, 0
            for y_ts, p_ts in seed_preds:
                pred_gal = p_ts >= t
                tp_s += (pred_gal & (y_ts == 1)).sum()
                fp_s += (pred_gal & (y_ts == 0)).sum()
                fn_s += (~pred_gal & (y_ts == 1)).sum()
                tn_s += (~pred_gal & (y_ts == 0)).sum()
            n = len(seed_preds)
            tpr_grid_all[ri, ti] = (tp_s/n) / max(tp_s/n + fn_s/n, 1)
            fpr_grid_all[ri, ti] = (fp_s/n) / max(fp_s/n + tn_s/n, 1)

    # Map delta_ratios to indices in all_ratios
    delta_idx = [min(range(n_all), key=lambda i: abs(all_ratios[i] - r)) for r in delta_ratios]

    # Purity maps + deltas (using delta_idx into tpr_grid_all)
    purity_maps = {}
    for ci, ai in enumerate(delta_idx):
        purity_2d = np.zeros((len(thresh_fine), len(pi_hm)))
        for ti in range(len(thresh_fine)):
            tpr = tpr_grid_all[ai, ti]
            fpr = fpr_grid_all[ai, ti]
            purity_2d[ti, :] = tpr * pi_hm / (tpr * pi_hm + fpr * (1 - pi_hm) + 1e-12)
        purity_maps[ci] = purity_2d

    deltas = {}
    for ci in range(1, n_delta_ratios):
        deltas[ci] = (purity_maps[ci] - purity_maps[0]) * 100
    all_delta_vals = np.concatenate([d.ravel() for d in deltas.values()])
    vabs = max(abs(all_delta_vals.min()), abs(all_delta_vals.max()))

    delta_cmap = LinearSegmentedColormap.from_list(
        'orange_white_purple',
        ['#EF7627', '#FF9A56', '#FFFFFF', '#D162A4', '#8B1A6B'])
    delta_norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

    # ── Plot: 2 panels (matching pgal_recalibration.png layout) ──
    print("\n[3] Plotting...")
    n_delta = n_delta_ratios - 1
    fig, axes = plt.subplots(1, 1 + n_delta, figsize=(14, 4.8))
    fig.subplots_adjust(wspace=0.18)

    # ── Panel 0: Purity curves (all ratios, matches mix_threshold_recalibration) ──
    ax = axes[0]
    cmap_t = make_threshold_cmap(n_intervals=5)
    norm_t = mcolors.Normalize(vmin=0.5, vmax=1.0)
    thresh_range_curves = [0.5, 0.7, 0.9]

    # Line styles: solid for Q1-only (first), dashed for best (last), dotted for middle
    for ri in range(n_all):
        ls = '-' if ri == 0 else ('--' if ri == n_all - 1 else ':')
        lw = 1.5 if ri in [0, n_all - 1] else 0.8
        for t in thresh_range_curves:
            ti = np.argmin(np.abs(thresh_fine - t))
            tpr_v = tpr_grid_all[ri, ti]
            fpr_v = fpr_grid_all[ri, ti]
            if fpr_v == 0:
                continue
            prec = tpr_v * pi_range / (tpr_v * pi_range + fpr_v * (1 - pi_range) + 1e-12)
            col = cmap_t(norm_t(t))
            ax.plot(pi_range * 100, prec * 100, color=col, lw=lw, ls=ls, alpha=0.85)

    # Annotations for baseline (first) and best (last) of ALL ratios
    thresholds_mark = [0.5, 0.9]
    mark_points_ann = [(0.01, "1%"), (0.03, "3%")]
    for ri_idx in [0, n_all - 1]:
        is_baseline = (ri_idx == 0)
        marker = 'o' if is_baseline else 's'
        for thresh_mark in thresholds_mark:
            ti = np.argmin(np.abs(thresh_fine - thresh_mark))
            tpr_v = tpr_grid_all[ri_idx, ti]
            fpr_v = fpr_grid_all[ri_idx, ti]
            col = cmap_t(norm_t(thresh_mark))
            all_marks = list(mark_points_ann) + [(pi_q1, f"{pi_q1:.0%}")]
            for pi_mark, _ in all_marks:
                p_mark = tpr_v * pi_mark / (tpr_v * pi_mark + fpr_v * (1 - pi_mark) + 1e-12)
                ax.scatter(pi_mark * 100, p_mark * 100, s=30, marker=marker,
                           zorder=9, color=col, edgecolors='k', linewidths=0.5)
                lbl = f"{p_mark:.0%}"
                # Per-point offsets matching mix_threshold_recalibration
                offsets = {
                    (True,  0.5, 0.01): (18, -3),
                    (True,  0.5, 0.03): (18, -3),
                    (True,  0.5, pi_q1): (18, -10),
                    (True,  0.9, 0.01): (18, -3),
                    (True,  0.9, 0.03): (18, -3),
                    (True,  0.9, pi_q1): (18, 10),
                    (False, 0.5, 0.01): (-18, 3),
                    (False, 0.5, 0.03): (-18, 3),
                    (False, 0.5, pi_q1): (-17, -4),
                    (False, 0.9, 0.01): (-18, 3),
                    (False, 0.9, 0.03): (-18, 3),
                    (False, 0.9, pi_q1): (-18, 10),
                }
                oxy = offsets.get((is_baseline, thresh_mark, pi_mark), (0, 5))
                ax.annotate(lbl, xy=(pi_mark * 100, p_mark * 100),
                            xytext=oxy, textcoords='offset points',
                            fontsize=11, fontweight="bold", color=col,
                            ha='center', va='center', zorder=10)

    ax.axhline(50, color="grey", lw=1, ls=":", alpha=0.5)
    ax.axvspan(1, 3, color="steelblue", alpha=0.12, zorder=0)
    ax.axvline(pi_q1 * 100, color="coral", lw=1.5, ls="--")
    ax.set_xlabel("True galaxy fraction (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Galaxy selection purity (%)", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 50)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)

    legend_handles = [
        Line2D([0], [0], color='darkgray', lw=1.5, ls='-', marker='o', markersize=5,
               markerfacecolor='darkgray', markeredgecolor='k', markeredgewidth=0.5,
               label=f'{all_ratio_labels[0]} Q1-only'),
        Line2D([0], [0], color='darkgray', lw=0.8, ls=':', label='intermediate ratios'),
        Line2D([0], [0], color='darkgray', lw=1.5, ls='--', marker='s', markersize=5,
               markerfacecolor='darkgray', markeredgecolor='k', markeredgewidth=0.5,
               label=f'{all_ratio_labels[-1]} all ERO'),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc='lower right')

    sm = cm.ScalarMappable(cmap=cmap_t, norm=norm_t)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.8)
    cb.set_label("P(gal) threshold", fontsize=13, fontweight="bold")
    cb.ax.tick_params(labelsize=10)

    # ── Panels 1+: delta heatmaps (3:1 and 8.6:1 only) ──
    for col_idx, ci in enumerate(range(1, n_delta_ratios)):
        ax = axes[1 + col_idx]
        im = ax.pcolormesh(pi_hm * 100, thresh_fine, deltas[ci],
                           cmap=delta_cmap, shading='auto', norm=delta_norm, alpha=0.8)
        cs = ax.contour(pi_hm * 100, thresh_fine, deltas[ci],
                        levels=[1, 2, 5, 10, 15], colors='black',
                        linewidths=0.8, linestyles='--')
        # Label +1% manually, rest auto
        ax.clabel(cs, levels=[2, 5, 10, 15], fmt='+%d%%', fontsize=11, inline_spacing=8)
        ax.clabel(cs, levels=[1], fmt='+%d%%', fontsize=11,
                  manual=[(30, 0.55)])
        ax.axvspan(1, 3, color='white', alpha=0.1, zorder=2)
        ax.axvline(1, color='grey', lw=0.8, ls=':', alpha=0.5)
        ax.axvline(3, color='grey', lw=0.8, ls=':', alpha=0.5)
        ax.set_xscale('log')
        ax.set_ylim(0.3, 1.0)
        ax.set_xlim(pi_hm.min() * 100, pi_hm.max() * 100)
        ax.set_xlabel("True galaxy fraction (%)", fontsize=13, fontweight='bold')
        ax.set_ylabel("P(gal) threshold", fontsize=13, fontweight='bold')
        ax.set_title(f"Purity gain: {delta_labels[ci]} vs Q1-only",
                     fontsize=13, fontweight='bold')
        cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
        cb.set_label("Purity gain (%)", fontsize=13, fontweight='bold')

    out = PROJECT_DIR / "output" / "mix_recalibration_combined.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {out}")

    # ── mix_ratio_sweep_thresholds.png ────────────────────────────────────
    # Need thresh_fine in 0.5-0.9 range for this plot
    thresh_sweep = np.linspace(0.5, 0.9, 50)
    tpr_sweep = np.zeros((n_all, len(thresh_sweep)))
    fpr_sweep = np.zeros((n_all, len(thresh_sweep)))
    for ri in range(n_all):
        for ti, t in enumerate(thresh_sweep):
            ti_src = np.argmin(np.abs(thresh_fine - t))
            tpr_sweep[ri, ti] = tpr_grid_all[ri, ti_src]
            fpr_sweep[ri, ti] = fpr_grid_all[ri, ti_src]

    print("\n[4] Plotting sweep thresholds...")
    from analyze_mix_q1_ero import plot_sweep_thresholds
    plot_sweep_thresholds(all_ratios, all_ratio_labels, tpr_sweep, fpr_sweep,
                          thresh_sweep, PROJECT_DIR / "output")

    # ── mix_threshold_pgal_overlay.png ────────────────────────────────────
    print("\n[5] Plotting P(gal) overlay on Gaia...")
    GAIA_DIR = PROJECT_DIR / "output" / "gaia_50k"
    gaia_path = GAIA_DIR / "gaia_random_50k.npz"
    if gaia_path.exists():
        X_gaia = np.load(gaia_path)["X"].astype(float)
        print(f"  Loaded {len(X_gaia):,} Gaia sources")

        colors = cm.viridis(np.linspace(0, 0.9, n_all))
        all_pgal = []
        for ri, ratio in enumerate(all_ratios):
            seed_dir = _find_seed_dir(mix_dir, ratio)
            if seed_dir is None:
                continue
            seed_paths = sorted(seed_dir.glob("xgb_classifier_q1_seed*.json"))
            if not seed_paths:
                continue
            gaia_probas = []
            for sp in seed_paths:
                m = xgb.XGBClassifier()
                m.load_model(str(sp))
                gaia_probas.append(m.predict_proba(X_gaia)[:, 1])
            p_gaia = np.mean(gaia_probas, axis=0)
            label = all_ratio_labels[ri]
            all_pgal.append((label, p_gaia))
            print(f"    {label}: P>0.5={100*(p_gaia>=0.5).mean():.1f}%  "
                  f"P>0.9={100*(p_gaia>=0.9).mean():.1f}%")

        if all_pgal:
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
            ax1.set_ylabel("Source count", fontsize=14, fontweight='bold',
                           color='steelblue', labelpad=10)
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

            # Inline fraction labels for first and last ratio
            bbox_gap = dict(boxstyle='round,pad=0.15', facecolor='white',
                            edgecolor='none', alpha=1.0)
            for idx_label in [0, -1]:
                label_l, p_gaia_l = all_pgal[idx_label]
                color_l = colors[idx_label]
                for t in [0.5, 0.9]:
                    pct = 100 * (p_gaia_l >= t).mean()
                    ax2.text(t, pct, f"{label_l}: {pct:.1f}%", fontsize=13,
                             fontweight="bold", color=color_l, ha='center',
                             va='center', bbox=bbox_gap, zorder=10)

            ax1.legend(fontsize=10, loc='upper right', ncols=2)
            out = PROJECT_DIR / "output" / "mix_threshold_pgal_overlay.png"
            plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
            print(f"  -> {out}")
    else:
        print(f"  WARNING: {gaia_path} not found, skipping pgal overlay")

    return 0


if __name__ == "__main__":
    sys.exit(main())
