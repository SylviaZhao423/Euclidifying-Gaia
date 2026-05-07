#!/usr/bin/env python3
"""
Corner plot of Q1-Gaia training sample split by XGBoost P(gal).

Upper triangle: true STAR sources (PHZ label)
Lower triangle: true GALAXY sources (PHZ label)
Diagonal: both overlaid

Split at P(gal) = 0.9 to show correctly classified vs misclassified.

Features: corrected_bp_rp_excess, g_rp, pm_significance, G magnitude

Reads:
  WSDB (Q1 data)
  output/xgboost_q1/seed_models/*.json

Writes:
  output/q1_corner_star_galaxy.png

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_q1_corner.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
SEED_DIR = PROJECT_DIR / "output" / "xgboost_q1" / "seed_models"
OUTPUT_DIR = PROJECT_DIR / "output"
FEAT_NAMES = list(SELECTED_GAIA_PARAMETERS)
SEEDS = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'
MAG_CUT = 17
CLASS_MAPPING = {'STAR': 0, 'GALAXY': 1}


def query_q1(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in FEAT_NAMES])
    query = f"""
        SELECT g_phot_g_mean_mag AS phot_g_mean_mag, {gaia_cols},
               CASE WHEN p_phz_classification = 1 THEN 'STAR'
                    WHEN p_phz_classification = 2 THEN 'GALAXY' END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep
        WHERE p_phz_classification IN (1, 2)
          AND g_phot_g_mean_mag >= {MAG_CUT}
        ORDER BY g_random_index
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    for c in FEAT_NAMES:
        df[c] = pd.to_numeric(df[c].replace('NOT_AVAILABLE', np.nan), errors='coerce')
    X = df[FEAT_NAMES].values
    y = df['label'].map(CLASS_MAPPING).values
    mag = df['phot_g_mean_mag'].values.astype(float)
    return X, y, mag


def ensemble_predict(X):
    probas = []
    for seed in SEEDS:
        mp = SEED_DIR / f"xgb_classifier_q1_seed{seed}.json"
        if not mp.exists():
            continue
        m = xgb.XGBClassifier()
        m.load_model(str(mp))
        probas.append(m.predict_proba(X)[:, 1])
    return np.mean(probas, axis=0)


def make_corner(sample_arrays, sample_labels, sample_colors, ranges,
                corner_labels, title, out_path):
    """Generic corner plot with KDE contours."""
    n_corner = len(corner_labels)
    fig, axes = plt.subplots(n_corner, n_corner, figsize=(14, 14))
    fig.subplots_adjust(hspace=0.08, wspace=0.08)

    for row in range(n_corner):
        for col in range(n_corner):
            ax = axes[row, col]

            if col > row:
                ax.axis('off')
                continue

            if row == col:
                for vals, label, color in zip(sample_arrays, sample_labels, sample_colors):
                    v = vals[:, row]
                    ax.hist(v, bins=50, range=ranges[row], alpha=0.3,
                            color=color, density=True)
                    try:
                        kde = gaussian_kde(v)
                        xg = np.linspace(ranges[row][0], ranges[row][1], 200)
                        ax.plot(xg, kde(xg), color=color, lw=1.5, label=label)
                    except Exception:
                        pass
                if row == 0:
                    ax.legend(fontsize=7, loc='upper right')
            else:
                for vals, label, color in zip(sample_arrays, sample_labels, sample_colors):
                    vx = vals[:, col]
                    vy = vals[:, row]
                    ax.scatter(vx, vy, s=0.3, alpha=0.05, color=color, rasterized=True)
                    try:
                        xy = np.vstack([vx, vy])
                        kde = gaussian_kde(xy)
                        xg = np.linspace(ranges[col][0], ranges[col][1], 60)
                        yg = np.linspace(ranges[row][0], ranges[row][1], 60)
                        Xg, Yg = np.meshgrid(xg, yg)
                        Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
                        ax.contour(Xg, Yg, Z, levels=4, colors=[color],
                                   linewidths=1.0, alpha=0.8)
                    except Exception:
                        pass

            ax.set_xlim(ranges[col])
            if row != col:
                ax.set_ylim(ranges[row])

            if row == n_corner - 1:
                ax.set_xlabel(corner_labels[col], fontsize=11, fontweight='bold')
            else:
                ax.tick_params(axis='x', labelbottom=False)
            if col == 0 and row != 0:
                ax.set_ylabel(corner_labels[row], fontsize=11, fontweight='bold')
            elif col == 0 and row == 0:
                ax.set_ylabel("Density", fontsize=9)
            else:
                ax.tick_params(axis='y', labelleft=False)
            ax.tick_params(labelsize=9)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.92)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {out_path}")


def main():
    username = os.environ.get("PGUSER")
    if not username:
        print("ERROR: PGUSER not set.")
        return 1

    print("[1] Loading Q1 data from WSDB...")
    X, y, mag = query_q1(username)
    print(f"  {len(X):,} sources  STAR: {(y==0).sum():,}  GALAXY: {(y==1).sum():,}")

    print("\n[2] Ensemble prediction...")
    p = ensemble_predict(X)
    print(f"  P(gal) range: [{p.min():.4f}, {p.max():.4f}]")

    # Feature indices
    corner_feats = ['corrected_bp_rp_excess', 'g_rp', 'pm_significance']
    corner_idx = [FEAT_NAMES.index(f) for f in corner_feats]
    corner_labels = ['$C^*$', '$G-G_{RP}$', 'PM sig.', 'G mag']

    def build_array(mask):
        vals = np.column_stack([X[mask, fi] for fi in corner_idx] + [mag[mask]])
        good = np.all(np.isfinite(vals), axis=1)
        return vals[good]

    # ── Plot 1: True stars vs true galaxies, all sources ──
    print("\n[3] Corner: true STAR vs true GALAXY (all sources)...")
    arr_star = build_array(y == 0)
    arr_gal = build_array(y == 1)

    all_vals = np.vstack([arr_star, arr_gal])
    ranges = []
    for j in range(len(corner_labels)):
        lo, hi = np.nanpercentile(all_vals[:, j], [1, 99])
        pad = (hi - lo) * 0.1
        ranges.append((lo - pad, hi + pad))

    make_corner(
        [arr_star, arr_gal],
        [f'STAR (n={len(arr_star):,})', f'GALAXY (n={len(arr_gal):,})'],
        ['coral', 'steelblue'],
        ranges, corner_labels,
        "Q1 training sample — true STAR vs true GALAXY",
        OUTPUT_DIR / "q1_corner_star_vs_galaxy.png"
    )

    # ── Plot 2: Correctly classified vs misclassified at P=0.9 ──
    print("\n[4] Corner: classification at P(gal) = 0.9...")
    # True positive galaxies (TP): true galaxy, P>=0.9
    # False negative galaxies (FN): true galaxy, P<0.9
    # True negative stars (TN): true star, P<0.9
    # False positive stars (FP): true star, P>=0.9
    tp = build_array((y == 1) & (p >= 0.9))
    fn = build_array((y == 1) & (p < 0.9))
    tn = build_array((y == 0) & (p < 0.9))
    fp = build_array((y == 0) & (p >= 0.9))

    print(f"  TP (galaxy, P>=0.9): {len(tp):,}")
    print(f"  FN (galaxy, P<0.9):  {len(fn):,}")
    print(f"  TN (star, P<0.9):    {len(tn):,}")
    print(f"  FP (star, P>=0.9):   {len(fp):,}")

    # Corner: TP vs FN — which galaxies does the model miss?
    make_corner(
        [tp, fn],
        [f'Detected galaxies (P\u22650.9, n={len(tp):,})',
         f'Missed galaxies (P<0.9, n={len(fn):,})'],
        ['#2E7D32', '#E53935'],
        ranges, corner_labels,
        "Q1 galaxies — detected (P\u22650.9) vs missed (P<0.9)",
        OUTPUT_DIR / "q1_corner_galaxy_detected_vs_missed.png"
    )

    # Corner: TN vs FP — which stars get misclassified?
    if len(fp) > 50:
        make_corner(
            [tn, fp],
            [f'Correct stars (P<0.9, n={len(tn):,})',
             f'False galaxies (P\u22650.9, n={len(fp):,})'],
            ['steelblue', '#FF9800'],
            ranges, corner_labels,
            "Q1 stars — correctly rejected vs false galaxy candidates",
            OUTPUT_DIR / "q1_corner_star_correct_vs_false.png"
        )
    else:
        print(f"  Only {len(fp)} false positives at P>=0.9, skipping star corner")

    # ── Plot 3: Same at P=0.5 ──
    print("\n[5] Corner: classification at P(gal) = 0.5...")
    tp5 = build_array((y == 1) & (p >= 0.5))
    fn5 = build_array((y == 1) & (p < 0.5))
    fp5 = build_array((y == 0) & (p >= 0.5))

    print(f"  TP (galaxy, P>=0.5): {len(tp5):,}")
    print(f"  FN (galaxy, P<0.5):  {len(fn5):,}")
    print(f"  FP (star, P>=0.5):   {len(fp5):,}")

    make_corner(
        [tp5, fn5],
        [f'Detected galaxies (P\u22650.5, n={len(tp5):,})',
         f'Missed galaxies (P<0.5, n={len(fn5):,})'],
        ['#2E7D32', '#E53935'],
        ranges, corner_labels,
        "Q1 galaxies — detected (P\u22650.5) vs missed (P<0.5)",
        OUTPUT_DIR / "q1_corner_galaxy_detected_vs_missed_p05.png"
    )

    # ── Plot 4: Split corner — TP lower, FN upper, FP overlaid (P=0.5) ──
    print("\n[6] Split corner: TP / FN / FP at P(gal) = 0.5...")
    n_c = len(corner_labels)
    fig, axes = plt.subplots(n_c, n_c, figsize=(14, 14))
    fig.subplots_adjust(hspace=0.08, wspace=0.08)

    groups = [
        (tp5, 'Detected galaxies (TP)', '#2E7D32'),
        (fn5, 'Missed galaxies (FN)', '#E53935'),
        (fp5, 'Misclassified stars (FP)', '#FF9800'),
    ]

    for row in range(n_c):
        for col in range(n_c):
            ax = axes[row, col]

            if row == col:
                # Diagonal: all three overlaid
                for vals, label, color in groups:
                    v = vals[:, row]
                    ax.hist(v, bins=50, range=ranges[row], alpha=0.3,
                            color=color, density=True)
                    try:
                        kde = gaussian_kde(v)
                        xg = np.linspace(ranges[row][0], ranges[row][1], 200)
                        ax.plot(xg, kde(xg), color=color, lw=1.5, label=label)
                    except Exception:
                        pass
                if row == 0:
                    ax.legend(fontsize=6, loc='upper right')

            elif col < row:
                # Lower triangle: TP (detected galaxies) + FP (misclassified stars)
                for vals, label, color in [groups[0], groups[2]]:
                    vx = vals[:, col]
                    vy = vals[:, row]
                    ax.scatter(vx, vy, s=0.3, alpha=0.05, color=color, rasterized=True)
                    try:
                        xy = np.vstack([vx, vy])
                        kde = gaussian_kde(xy)
                        xg = np.linspace(ranges[col][0], ranges[col][1], 60)
                        yg = np.linspace(ranges[row][0], ranges[row][1], 60)
                        Xg, Yg = np.meshgrid(xg, yg)
                        Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
                        ax.contour(Xg, Yg, Z, levels=4, colors=[color],
                                   linewidths=1.0, alpha=0.8)
                    except Exception:
                        pass

            else:
                # Upper triangle: FN (missed galaxies) + FP (misclassified stars)
                for vals, label, color in [groups[1], groups[2]]:
                    vx = vals[:, col]
                    vy = vals[:, row]
                    ax.scatter(vx, vy, s=0.3, alpha=0.05, color=color, rasterized=True)
                    try:
                        xy = np.vstack([vx, vy])
                        kde = gaussian_kde(xy)
                        xg = np.linspace(ranges[col][0], ranges[col][1], 60)
                        yg = np.linspace(ranges[row][0], ranges[row][1], 60)
                        Xg, Yg = np.meshgrid(xg, yg)
                        Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
                        ax.contour(Xg, Yg, Z, levels=4, colors=[color],
                                   linewidths=1.0, alpha=0.8)
                    except Exception:
                        pass

            ax.set_xlim(ranges[col])
            if row != col:
                ax.set_ylim(ranges[row])

            if row == n_c - 1:
                ax.set_xlabel(corner_labels[col], fontsize=11, fontweight='bold')
            else:
                ax.tick_params(axis='x', labelbottom=False)
            if col == 0 and row != 0:
                ax.set_ylabel(corner_labels[row], fontsize=11, fontweight='bold')
            elif col == 0 and row == 0:
                ax.set_ylabel("Density", fontsize=9)
            else:
                ax.tick_params(axis='y', labelleft=False)
            ax.tick_params(labelsize=9)

    fig.suptitle("Q1 classification at P(gal) = 0.5\n"
                 "Lower: TP + FP | Upper: FN + FP | Diagonal: all three",
                 fontsize=13, fontweight='bold', y=0.93)
    out = OUTPUT_DIR / "q1_corner_split_p05.png"
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  -> {out}")

    # ── Plot 5: Same split at P=0.9 (FP as scatter only since n=22) ──
    print("\n[7] Split corner: TP / FN / FP at P(gal) = 0.9...")
    fig, axes = plt.subplots(n_c, n_c, figsize=(14, 14))
    fig.subplots_adjust(hspace=0.08, wspace=0.08)

    groups9 = [
        (tp, 'Detected galaxies (TP)', '#2E7D32'),
        (fn, 'Missed galaxies (FN)', '#E53935'),
        (fp, 'Misclassified stars (FP)', '#FF9800'),
    ]

    for row in range(n_c):
        for col in range(n_c):
            ax = axes[row, col]

            if row == col:
                for vals, label, color in groups9[:2]:  # TP and FN only for KDE
                    v = vals[:, row]
                    ax.hist(v, bins=50, range=ranges[row], alpha=0.3,
                            color=color, density=True)
                    try:
                        kde = gaussian_kde(v)
                        xg = np.linspace(ranges[row][0], ranges[row][1], 200)
                        ax.plot(xg, kde(xg), color=color, lw=1.5, label=label)
                    except Exception:
                        pass
                # FP as rug plot on diagonal
                if len(fp) > 0:
                    ax.scatter(fp[:, row], np.zeros(len(fp)), s=10, marker='|',
                               color='#FF9800', alpha=0.8, zorder=5,
                               label=f'FP stars (n={len(fp)})')
                if row == 0:
                    ax.legend(fontsize=6, loc='upper right')

            elif col < row:
                # Lower: TP
                vals, label, color = groups9[0]
                vx, vy = vals[:, col], vals[:, row]
                ax.scatter(vx, vy, s=0.3, alpha=0.05, color=color, rasterized=True)
                try:
                    xy = np.vstack([vx, vy])
                    kde = gaussian_kde(xy)
                    xg = np.linspace(ranges[col][0], ranges[col][1], 60)
                    yg = np.linspace(ranges[row][0], ranges[row][1], 60)
                    Xg, Yg = np.meshgrid(xg, yg)
                    Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
                    ax.contour(Xg, Yg, Z, levels=4, colors=[color],
                               linewidths=1.0, alpha=0.8)
                except Exception:
                    pass
                # FP as individual points
                if len(fp) > 0:
                    ax.scatter(fp[:, col], fp[:, row], s=25, marker='x',
                               color='#FF9800', zorder=5, alpha=0.9)

            else:
                # Upper: FN
                vals, label, color = groups9[1]
                vx, vy = vals[:, col], vals[:, row]
                ax.scatter(vx, vy, s=0.3, alpha=0.05, color=color, rasterized=True)
                try:
                    xy = np.vstack([vx, vy])
                    kde = gaussian_kde(xy)
                    xg = np.linspace(ranges[col][0], ranges[col][1], 60)
                    yg = np.linspace(ranges[row][0], ranges[row][1], 60)
                    Xg, Yg = np.meshgrid(xg, yg)
                    Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
                    ax.contour(Xg, Yg, Z, levels=4, colors=[color],
                               linewidths=1.0, alpha=0.8)
                except Exception:
                    pass
                if len(fp) > 0:
                    ax.scatter(fp[:, col], fp[:, row], s=25, marker='x',
                               color='#FF9800', zorder=5, alpha=0.9)

            ax.set_xlim(ranges[col])
            if row != col:
                ax.set_ylim(ranges[row])
            if row == n_c - 1:
                ax.set_xlabel(corner_labels[col], fontsize=11, fontweight='bold')
            else:
                ax.tick_params(axis='x', labelbottom=False)
            if col == 0 and row != 0:
                ax.set_ylabel(corner_labels[row], fontsize=11, fontweight='bold')
            elif col == 0 and row == 0:
                ax.set_ylabel("Density", fontsize=9)
            else:
                ax.tick_params(axis='y', labelleft=False)
            ax.tick_params(labelsize=9)

    fig.suptitle("Q1 classification at P(gal) = 0.9\n"
                 "Lower: TP (green) | Upper: FN (red) | \u00d7: FP stars (orange, n=22)",
                 fontsize=13, fontweight='bold', y=0.93)
    out = OUTPUT_DIR / "q1_corner_split_p09.png"
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  -> {out}")

    # ── Repeat with top 4 XGBoost importance features ──
    print("\n[8] Corner plots with top 4 XGBoost features...")
    top4_feats = ['corrected_bp_rp_excess', 'g_rp', 'astrometric_chi2_al', 'position_error']
    top4_idx = [FEAT_NAMES.index(f) for f in top4_feats]
    top4_labels = ['$C^*$', '$G-G_{RP}$', '$\\chi^2_{AL}$', 'Pos. error']

    def build_array_top4(mask):
        vals = np.column_stack([X[mask, fi] for fi in top4_idx])
        good = np.all(np.isfinite(vals), axis=1)
        return vals[good]

    # Ranges from all sources
    arr_all_t4 = build_array_top4(np.ones(len(X), dtype=bool))
    ranges_t4 = []
    for j in range(len(top4_labels)):
        lo, hi = np.nanpercentile(arr_all_t4[:, j], [1, 99])
        pad = (hi - lo) * 0.1
        ranges_t4.append((lo - pad, hi + pad))

    # Star vs Galaxy
    arr_star_t4 = build_array_top4(y == 0)
    arr_gal_t4 = build_array_top4(y == 1)
    make_corner(
        [arr_star_t4, arr_gal_t4],
        [f'STAR (n={len(arr_star_t4):,})', f'GALAXY (n={len(arr_gal_t4):,})'],
        ['coral', 'steelblue'],
        ranges_t4, top4_labels,
        "Q1 training sample — top 4 XGBoost features",
        OUTPUT_DIR / "q1_corner_top4_star_vs_galaxy.png"
    )

    # Split corner at P=0.5: TP / FN / FP
    print("\n[9] Split corner top4 at P=0.5...")
    tp5_t4 = build_array_top4((y == 1) & (p >= 0.5))
    fn5_t4 = build_array_top4((y == 1) & (p < 0.5))
    fp5_t4 = build_array_top4((y == 0) & (p >= 0.5))
    print(f"  TP: {len(tp5_t4):,}  FN: {len(fn5_t4):,}  FP: {len(fp5_t4):,}")

    n_c4 = len(top4_labels)
    fig, axes = plt.subplots(n_c4, n_c4, figsize=(14, 14))
    fig.subplots_adjust(hspace=0.08, wspace=0.08)

    groups_t4 = [
        (tp5_t4, 'Detected galaxies (TP)', '#2E7D32'),
        (fn5_t4, 'Missed galaxies (FN)', '#E53935'),
        (fp5_t4, 'Misclassified stars (FP)', '#FF9800'),
    ]

    for row in range(n_c4):
        for col in range(n_c4):
            ax = axes[row, col]

            if row == col:
                for vals, label, color in groups_t4:
                    v = vals[:, row]
                    ax.hist(v, bins=50, range=ranges_t4[row], alpha=0.3,
                            color=color, density=True)
                    try:
                        kde = gaussian_kde(v)
                        xg = np.linspace(ranges_t4[row][0], ranges_t4[row][1], 200)
                        ax.plot(xg, kde(xg), color=color, lw=1.5, label=label)
                    except Exception:
                        pass
                if row == 0:
                    ax.legend(fontsize=6, loc='upper right')

            elif col < row:
                # Lower: TP + FP
                for vals, label, color in [groups_t4[0], groups_t4[2]]:
                    vx, vy = vals[:, col], vals[:, row]
                    ax.scatter(vx, vy, s=0.3, alpha=0.05, color=color, rasterized=True)
                    try:
                        xy = np.vstack([vx, vy])
                        kde = gaussian_kde(xy)
                        xg = np.linspace(ranges_t4[col][0], ranges_t4[col][1], 60)
                        yg = np.linspace(ranges_t4[row][0], ranges_t4[row][1], 60)
                        Xg, Yg = np.meshgrid(xg, yg)
                        Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
                        ax.contour(Xg, Yg, Z, levels=4, colors=[color],
                                   linewidths=1.0, alpha=0.8)
                    except Exception:
                        pass
            else:
                # Upper: FN + FP
                for vals, label, color in [groups_t4[1], groups_t4[2]]:
                    vx, vy = vals[:, col], vals[:, row]
                    ax.scatter(vx, vy, s=0.3, alpha=0.05, color=color, rasterized=True)
                    try:
                        xy = np.vstack([vx, vy])
                        kde = gaussian_kde(xy)
                        xg = np.linspace(ranges_t4[col][0], ranges_t4[col][1], 60)
                        yg = np.linspace(ranges_t4[row][0], ranges_t4[row][1], 60)
                        Xg, Yg = np.meshgrid(xg, yg)
                        Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
                        ax.contour(Xg, Yg, Z, levels=4, colors=[color],
                                   linewidths=1.0, alpha=0.8)
                    except Exception:
                        pass

            ax.set_xlim(ranges_t4[col])
            if row != col:
                ax.set_ylim(ranges_t4[row])
            if row == n_c4 - 1:
                ax.set_xlabel(top4_labels[col], fontsize=11, fontweight='bold')
            else:
                ax.tick_params(axis='x', labelbottom=False)
            if col == 0 and row != 0:
                ax.set_ylabel(top4_labels[row], fontsize=11, fontweight='bold')
            elif col == 0 and row == 0:
                ax.set_ylabel("Density", fontsize=9)
            else:
                ax.tick_params(axis='y', labelleft=False)
            ax.tick_params(labelsize=9)

    fig.suptitle("Top 4 XGBoost features — Q1 classification at P(gal) = 0.5\n"
                 "Lower: TP + FP | Upper: FN + FP | Diagonal: all three",
                 fontsize=13, fontweight='bold', y=0.93)
    out = OUTPUT_DIR / "q1_corner_top4_split_p05.png"
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  -> {out}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
