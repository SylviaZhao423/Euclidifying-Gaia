#!/usr/bin/env python3
"""
Exploratory analysis: how high-latitude selection affects model predictions.

4 plots:
  1. P(gal) vs G magnitude — random vs high-lat, Q1-only vs 8.6:1
  2. Model agreement — P(gal) Q1-only vs 8.6:1, for random and high-lat
  3. Feature distributions — top 3 features, random vs high-lat, colored by P(gal)
  4. Cumulative galaxy count vs threshold

Reads:
  output/gaia_50k/gaia_random_50k.npz
  output/gaia_50k/gaia_hilat_50k.npz
  output/xgboost_q1/seed_models/*.json
  output/xgboost_mix_q1_ero/ratio_sg8.63_all_ero/seed_models/*.json

Usage:
    python src/analyze_gaia_hilat_exploration.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
GAIA_DIR = PROJECT_DIR / "output" / "gaia_50k"
SEEDS = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]
FEAT_NAMES = list(SELECTED_GAIA_PARAMETERS)

OUTPUT_DIR = PROJECT_DIR / "output"


def ensemble_predict(seed_dir, X):
    probas = []
    for seed in SEEDS:
        mp = seed_dir / f"xgb_classifier_q1_seed{seed}.json"
        if not mp.exists():
            continue
        m = xgb.XGBClassifier()
        m.load_model(str(mp))
        probas.append(m.predict_proba(X)[:, 1])
    return np.mean(probas, axis=0)


def main():
    # Load data
    print("[1] Loading Gaia samples...")
    g_rand = np.load(GAIA_DIR / "gaia_random_50k.npz")
    g_hilat = np.load(GAIA_DIR / "gaia_hilat_50k.npz")
    g_lolat = np.load(GAIA_DIR / "gaia_lolat_50k.npz")

    X_rand = g_rand["X"].astype(float)
    X_hilat = g_hilat["X"].astype(float)
    X_lolat = g_lolat["X"].astype(float)
    mag_rand = g_rand["phot_g_mean_mag"].astype(float)
    mag_hilat = g_hilat["phot_g_mean_mag"].astype(float)
    mag_lolat = g_lolat["phot_g_mean_mag"].astype(float)
    print(f"  Random: {len(X_rand):,}  High-lat: {len(X_hilat):,}  Low-lat: {len(X_lolat):,}")

    # Predict
    print("\n[2] Predicting P(gal)...")
    q1_dir = PROJECT_DIR / "output" / "xgboost_q1" / "seed_models"
    ero_dir = PROJECT_DIR / "output" / "xgboost_mix_q1_ero" / "ratio_sg8.63_all_ero" / "seed_models"

    p_rand_q1 = ensemble_predict(q1_dir, X_rand)
    p_rand_ero = ensemble_predict(ero_dir, X_rand)
    p_hilat_q1 = ensemble_predict(q1_dir, X_hilat)
    p_hilat_ero = ensemble_predict(ero_dir, X_hilat)
    p_lolat_q1 = ensemble_predict(q1_dir, X_lolat)
    p_lolat_ero = ensemble_predict(ero_dir, X_lolat)
    print("  Done.")

    # ── Plot 1: P(gal) vs G magnitude ──
    print("\n[3] Plot 1: P(gal) vs magnitude...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.subplots_adjust(hspace=0.25, wspace=0.25)

    for ax, p, mag, title in [
        (axes[0, 0], p_rand_q1, mag_rand, "Random — Q1-only"),
        (axes[0, 1], p_hilat_q1, mag_hilat, "High-lat — Q1-only"),
        (axes[0, 2], p_lolat_q1, mag_lolat, "Low-lat — Q1-only"),
        (axes[1, 0], p_rand_ero, mag_rand, "Random — 8.6:1 ERO"),
        (axes[1, 1], p_hilat_ero, mag_hilat, "High-lat — 8.6:1 ERO"),
        (axes[1, 2], p_lolat_ero, mag_lolat, "Low-lat — 8.6:1 ERO"),
    ]:
        ax.scatter(mag, p, s=0.3, alpha=0.3, c=p, cmap='RdYlGn', vmin=0, vmax=1)
        ax.axhline(0.5, color='black', lw=1, ls='--', alpha=0.5)
        ax.axhline(0.9, color='green', lw=1, ls='--', alpha=0.5)
        pct05 = 100 * (p >= 0.5).mean()
        pct09 = 100 * (p >= 0.9).mean()
        ax.text(0.97, 0.95, f"P>0.5: {pct05:.1f}%\nP>0.9: {pct09:.1f}%",
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.set_xlabel("Gaia G (mag)", fontsize=11, fontweight='bold')
        ax.set_ylabel("P(gal)", fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

    out = OUTPUT_DIR / "hilat_pgal_vs_mag.png"
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  -> {out}")

    # ── Plot 2: Model agreement ──
    print("\n[4] Plot 2: Model agreement...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.subplots_adjust(wspace=0.25)

    for ax, p_q1, p_ero, title in [
        (ax1, p_rand_q1, p_rand_ero, "Random"),
        (ax2, p_hilat_q1, p_hilat_ero, "High-lat"),
        (ax3, p_lolat_q1, p_lolat_ero, "Low-lat"),
    ]:
        ax.hexbin(p_q1, p_ero, gridsize=80, cmap='Blues', mincnt=1,
                  extent=[0, 1, 0, 1])
        ax.plot([0, 1], [0, 1], 'r--', lw=1, alpha=0.5, label='Agreement line')
        ax.axvline(0.5, color='grey', lw=0.8, ls=':', alpha=0.5)
        ax.axhline(0.5, color='grey', lw=0.8, ls=':', alpha=0.5)
        # Quadrant counts
        agree_star = ((p_q1 < 0.5) & (p_ero < 0.5)).sum()
        agree_gal = ((p_q1 >= 0.5) & (p_ero >= 0.5)).sum()
        q1_only_gal = ((p_q1 >= 0.5) & (p_ero < 0.5)).sum()
        ero_only_gal = ((p_q1 < 0.5) & (p_ero >= 0.5)).sum()
        n = len(p_q1)
        ax.text(0.25, 0.25, f"Both star\n{agree_star/n:.0%}", transform=ax.transAxes,
                fontsize=9, ha='center', va='center', fontweight='bold', color='steelblue')
        ax.text(0.75, 0.75, f"Both galaxy\n{agree_gal/n:.0%}", transform=ax.transAxes,
                fontsize=9, ha='center', va='center', fontweight='bold', color='green')
        ax.text(0.75, 0.25, f"Q1→gal, ERO→star\n{q1_only_gal/n:.0%}", transform=ax.transAxes,
                fontsize=8, ha='center', va='center', color='coral')
        ax.text(0.25, 0.75, f"Q1→star, ERO→gal\n{ero_only_gal/n:.0%}", transform=ax.transAxes,
                fontsize=8, ha='center', va='center', color='orange')
        ax.set_xlabel("P(gal) — Q1-only", fontsize=11, fontweight='bold')
        ax.set_ylabel("P(gal) — 8.6:1 ERO", fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    out = OUTPUT_DIR / "hilat_model_agreement.png"
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  -> {out}")

    # ── Plot 3: Feature distributions ──
    print("\n[5] Plot 3: Feature distributions...")
    top_feats = ['corrected_bp_rp_excess', 'g_rp', 'pm_significance']
    feat_idx = [FEAT_NAMES.index(f) for f in top_feats]

    fig, axes = plt.subplots(len(top_feats), 3, figsize=(20, 4 * len(top_feats)))
    fig.subplots_adjust(hspace=0.3, wspace=0.25)

    for row, (fi, fname) in enumerate(zip(feat_idx, top_feats)):
        for col, (X, p, sample) in enumerate([
            (X_rand, p_rand_q1, "Random"),
            (X_hilat, p_hilat_q1, "High-lat"),
            (X_lolat, p_lolat_q1, "Low-lat"),
        ]):
            ax = axes[row, col]
            vals = X[:, fi]
            valid = np.isfinite(vals)
            v = vals[valid]
            pv = p[valid]

            # Clip to 1-99 percentile for display
            lo, hi = np.nanpercentile(v, [1, 99])
            nbins = 80

            # Star-like (P<0.5) and galaxy-like (P>=0.5)
            mask_star = pv < 0.5
            mask_gal = pv >= 0.5
            ax.hist(v[mask_star], bins=nbins, range=(lo, hi), alpha=0.5,
                    color='coral', label=f'P<0.5 ({mask_star.sum():,})', density=True)
            ax.hist(v[mask_gal], bins=nbins, range=(lo, hi), alpha=0.5,
                    color='steelblue', label=f'P≥0.5 ({mask_gal.sum():,})', density=True)
            ax.set_xlabel(fname, fontsize=10, fontweight='bold')
            ax.set_ylabel("Density", fontsize=10, fontweight='bold')
            ax.set_title(f"{sample} — {fname}", fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    out = OUTPUT_DIR / "hilat_feature_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  -> {out}")

    # ── Plot 4: Cumulative galaxy count vs threshold ──
    print("\n[6] Plot 4: Cumulative galaxy count...")
    fig, ax = plt.subplots(figsize=(10, 5))

    t_range = np.linspace(0, 1, 501)
    # Extrapolate to full Gaia (estimated ~1.5 billion sources with G>=17)
    n_gaia_full = 1.5e9

    for p, sample, model, color, ls in [
        (p_rand_q1, "Random", "Q1-only", "#2196F3", '-'),
        (p_rand_ero, "Random", "8.6:1 ERO", "#2E7D32", '--'),
        (p_hilat_q1, "High-lat", "Q1-only", "lightpink", '-'),
        (p_hilat_ero, "High-lat", "8.6:1 ERO", "#A5D6A7", '--'),
        (p_lolat_q1, "Low-lat", "Q1-only", "#FF9800", '-'),
        (p_lolat_ero, "Low-lat", "8.6:1 ERO", "#E65100", '--'),
    ]:
        frac = np.array([(p >= t).mean() for t in t_range])
        count = frac * n_gaia_full
        ax.plot(t_range, count, color=color, ls=ls, lw=2,
                label=f"{sample}, {model}")

    ax.set_yscale('log')
    ax.set_xlabel("P(gal) threshold", fontsize=13, fontweight='bold')
    ax.set_ylabel("Estimated galaxy candidates\n(full Gaia, G≥17)", fontsize=12, fontweight='bold')
    ax.axvline(0.5, color='black', lw=1.5, ls='--', alpha=0.5)
    ax.axvline(0.9, color='green', lw=1.5, ls='--', alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=11)

    out = OUTPUT_DIR / "hilat_cumulative_galaxy_count.png"
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  -> {out}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
