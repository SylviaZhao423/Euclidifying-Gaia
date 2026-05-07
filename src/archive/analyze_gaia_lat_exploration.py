#!/usr/bin/env python3
"""
Three-way spatial generalisation analysis: random vs high-lat vs low-lat.

Applies Q1-only model (10-seed ensemble) to all three Gaia samples and
produces comparison plots.

Produces:
  1. pgal_gaia_3way.png               — P(gal) overlaid (single panel)
  2. pgal_gaia_3way_split.png          — P(gal) split (3 panels)
  3. gaia_3way_pgal_vs_mag.png         — P(gal) vs G magnitude (3 panels)
  4. gaia_3way_feature_distributions.png — top features split by P(gal)
  5. gaia_3way_cumulative_count.png    — extrapolated galaxy count vs threshold

Reads:
  output/gaia_50k/gaia_random_50k.npz
  output/gaia_50k/gaia_hilat_50k.npz
  output/gaia_50k/gaia_lolat_50k.npz
  output/xgboost_q1/seed_models/*.json

Usage:
    python src/analyze_gaia_lat_exploration.py
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
OUTPUT_DIR = PROJECT_DIR / "output" / "gaia_lat_exploration"
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

    # ── Load and predict ──
    print("[1] Loading and predicting...")
    seed_count = sum(1 for s in SEEDS if (SEED_DIR / f"xgb_classifier_q1_seed{s}.json").exists())
    print(f"  Using {seed_count} seed models")

    data = {}  # sample_name -> (X, mag, p, color, n)
    for sample_name, npz_file, color in SAMPLES:
        g = np.load(GAIA_DIR / npz_file)
        X = g["X"].astype(float)
        mag = g["phot_g_mean_mag"].astype(float)
        p = ensemble_predict(X)
        n = len(X)
        data[sample_name] = (X, mag, p, color, n)
        print(f"  {sample_name}: {n:,} sources  "
              f"P>0.5={100*(p>=0.5).mean():.1f}%  "
              f"P>0.9={100*(p>=0.9).mean():.1f}%")

    bins = np.linspace(0, 1, 101)
    t_range = np.linspace(0, 1, 501)
    bbox_gap = dict(boxstyle='round,pad=0.15', facecolor='white',
                    edgecolor='none', alpha=1.0)

    # ── Plot 1: P(gal) overlaid (single panel) ──
    print("\n[2] Plot 1: P(gal) overlaid...")
    fig, ax1 = plt.subplots(figsize=(10, 5))

    for sample_name, _, _ in reversed(SAMPLES):
        X, mag, p, color, n = data[sample_name]
        ax1.hist(p, bins=bins, alpha=0.2, color=color,
                 histtype='stepfilled', linewidth=0)
        ax1.hist(p, bins=bins, color=color, histtype='step',
                 linewidth=1.5, label=f"{sample_name} counts")

    ax1.set_yscale("log")
    ax1.set_ylim(0.5, 1e4)
    ax1.set_xlabel("P(gal)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Source count", fontsize=14, fontweight="bold",
                   color="steelblue", labelpad=10)
    ax1.tick_params(axis='y', labelcolor="steelblue", labelsize=12)
    plt.setp(ax1.get_yticklabels(), fontweight="bold")
    ax1.tick_params(axis='x', labelsize=12)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    for sample_name, _, _ in SAMPLES:
        X, mag, p, color, n = data[sample_name]
        counts = np.array([(p >= t).sum() for t in t_range])
        frac = counts / n * 100
        ax2.plot(t_range, frac, color="#E53935", lw=1.5, ls='--')

    ax2.set_yscale("log")
    ax2.set_ylim(0.05, 100)
    ax2.set_ylabel("Fraction above threshold (%)", fontsize=14,
                   fontweight="bold", color="#E53935", labelpad=10)
    ax2.set_yticks([0.1, 0.5, 1, 2, 5, 10, 20, 50, 100])
    ax2.set_yticklabels(["0.1", "0.5", "1", "2", "5", "10", "20", "50", "100"])
    ax2.tick_params(axis='y', labelcolor="#E53935", labelsize=12)
    plt.setp(ax2.get_yticklabels(), fontweight="bold")

    for t, col, ls in [(0.5, 'black', '--'), (0.9, 'green', '--')]:
        ax1.axvline(t, color=col, lw=1.5, ls=ls, alpha=0.5)

    for sample_name, _, _ in SAMPLES:
        _, _, p, color, _ = data[sample_name]
        for t in [0.5, 0.9]:
            pct = 100 * (p >= t).mean()
            ax2.text(t, pct, f"{pct:.1f}%", fontsize=11, fontweight="bold",
                     color="#E53935", ha='center', va='center', bbox=bbox_gap, zorder=10)

    ax1.legend(fontsize=10, loc='upper right')

    out = OUTPUT_DIR / "pgal_gaia_3way.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  -> {out}")

    # ── Plot 2: P(gal) split (3 panels) ──
    print("\n[3] Plot 2: P(gal) split...")
    fig, axes = plt.subplots(1, 3, figsize=(22, 5))
    fig.subplots_adjust(wspace=0.25)

    for ax, (sample_name, _, _) in zip(axes, SAMPLES):
        X, mag, p, color, n = data[sample_name]

        ax.hist(p, bins=bins, alpha=0.25, color=color,
                histtype='stepfilled', linewidth=0)
        ax.hist(p, bins=bins, color=color, histtype='step',
                linewidth=1.5, label=f"{sample_name} counts")
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

        ax2 = ax.twinx()
        counts_arr = np.array([(p >= t).sum() for t in t_range])
        frac = counts_arr / n * 100
        ax2.plot(t_range, frac, color="#E53935", lw=1.5)
        ax2.set_yscale("log")
        ax2.set_ylim(0.05, 100)
        ax2.set_ylabel("Fraction above threshold (%)", fontsize=14,
                       fontweight="bold", color="#E53935", labelpad=10)
        ax2.set_yticks([0.1, 0.5, 1, 2, 5, 10, 20, 50, 100])
        ax2.set_yticklabels(["0.1", "0.5", "1", "2", "5", "10", "20", "50", "100"])
        ax2.tick_params(axis='y', labelcolor="#E53935", labelsize=12)
        plt.setp(ax2.get_yticklabels(), fontweight="bold")

        for t, col, ls in [(0.5, 'black', '--'), (0.9, 'green', '--')]:
            ax.axvline(t, color=col, lw=1.5, ls=ls, alpha=0.5)

        for t in [0.5, 0.9]:
            pct = 100 * (p >= t).mean()
            ax2.text(t, pct, f"{pct:.1f}%", fontsize=13, fontweight="bold",
                     color="#E53935", ha='center', va='center', bbox=bbox_gap, zorder=10)

        ax.legend(fontsize=11, loc='upper right')

    out = OUTPUT_DIR / "pgal_gaia_3way_split.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  -> {out}")

    # ── Plot 3: P(gal) vs G magnitude ──
    print("\n[4] Plot 3: P(gal) vs magnitude...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.subplots_adjust(wspace=0.25)

    for ax, (sample_name, _, _) in zip(axes, SAMPLES):
        X, mag, p, color, n = data[sample_name]
        ax.scatter(mag, p, s=0.3, alpha=0.3, c=p, cmap='RdYlGn', vmin=0, vmax=1)
        ax.axhline(0.5, color='black', lw=1, ls='--', alpha=0.5)
        ax.axhline(0.9, color='green', lw=1, ls='--', alpha=0.5)
        pct05 = 100 * (p >= 0.5).mean()
        pct09 = 100 * (p >= 0.9).mean()
        ax.text(0.97, 0.95, f"P>0.5: {pct05:.1f}%\nP>0.9: {pct09:.1f}%",
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.set_xlabel("Gaia G (mag)", fontsize=12, fontweight='bold')
        ax.set_ylabel("P(gal)", fontsize=12, fontweight='bold')
        ax.set_title(sample_name, fontsize=13, fontweight='bold')
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

    out = OUTPUT_DIR / "gaia_3way_pgal_vs_mag.png"
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  -> {out}")

    # ── Plot 4: Feature distributions ──
    print("\n[5] Plot 4: Feature distributions...")
    top_feats = ['corrected_bp_rp_excess', 'g_rp', 'pm_significance']
    feat_idx = [FEAT_NAMES.index(f) for f in top_feats]

    fig, axes = plt.subplots(len(top_feats), 3, figsize=(20, 4 * len(top_feats)))
    fig.subplots_adjust(hspace=0.3, wspace=0.25)

    for row, (fi, fname) in enumerate(zip(feat_idx, top_feats)):
        for col, (sample_name, _, _) in enumerate(SAMPLES):
            X, mag, p, color, n = data[sample_name]
            ax = axes[row, col]
            vals = X[:, fi]
            valid = np.isfinite(vals)
            v = vals[valid]
            pv = p[valid]

            lo, hi = np.nanpercentile(v, [1, 99])
            nbins = 80

            mask_star = pv < 0.5
            mask_gal = pv >= 0.5
            ax.hist(v[mask_star], bins=nbins, range=(lo, hi), alpha=0.5,
                    color='coral', label=f'P<0.5 ({mask_star.sum():,})', density=True)
            ax.hist(v[mask_gal], bins=nbins, range=(lo, hi), alpha=0.5,
                    color='steelblue', label=f'P\u22650.5 ({mask_gal.sum():,})', density=True)
            ax.set_xlabel(fname, fontsize=10, fontweight='bold')
            ax.set_ylabel("Density", fontsize=10, fontweight='bold')
            ax.set_title(f"{sample_name} \u2014 {fname}", fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    out = OUTPUT_DIR / "gaia_3way_feature_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  -> {out}")

    # ── Plot 5: Cumulative galaxy count ──
    print("\n[6] Plot 5: Cumulative galaxy count...")
    fig, ax = plt.subplots(figsize=(10, 5))
    n_gaia_full = 1.5e9

    for sample_name, _, _ in SAMPLES:
        X, mag, p, color, n = data[sample_name]
        frac = np.array([(p >= t).mean() for t in t_range])
        count = frac * n_gaia_full
        ax.plot(t_range, count, color=color, lw=2, label=sample_name)

    ax.set_yscale('log')
    ax.set_xlabel("P(gal) threshold", fontsize=13, fontweight='bold')
    ax.set_ylabel("Estimated galaxy candidates\n(full Gaia, G\u226517)", fontsize=12, fontweight='bold')
    ax.axvline(0.5, color='black', lw=1.5, ls='--', alpha=0.5)
    ax.axvline(0.9, color='green', lw=1.5, ls='--', alpha=0.5)

    # Inline labels at 0.5 and 0.9
    for sample_name, _, _ in SAMPLES:
        _, _, p, color, _ = data[sample_name]
        for t in [0.5, 0.9]:
            frac_t = (p >= t).mean()
            count_t = frac_t * n_gaia_full
            ax.text(t, count_t, f"{count_t/1e6:.0f}M", fontsize=10, fontweight="bold",
                    color=color, ha='center', va='bottom', bbox=bbox_gap, zorder=10)

    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=11)

    out = OUTPUT_DIR / "gaia_3way_cumulative_count.png"
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  -> {out}")

    # ── Plot 6: Galaxy fraction vs magnitude ──
    print("\n[7] Plot 6: Galaxy fraction vs magnitude...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.subplots_adjust(wspace=0.25)

    mag_bins = np.arange(17, 23, 0.5)
    mag_centres = 0.5 * (mag_bins[:-1] + mag_bins[1:])

    for ax, thresh, thresh_label in zip(axes, [0.5, 0.7, 0.9],
                                         ["P(gal) \u2265 0.5", "P(gal) \u2265 0.7", "P(gal) \u2265 0.9"]):
        for sample_name, _, _ in SAMPLES:
            X, mag, p, color, n = data[sample_name]
            frac_per_bin = []
            for i in range(len(mag_bins) - 1):
                mask = (mag >= mag_bins[i]) & (mag < mag_bins[i + 1])
                if mask.sum() > 0:
                    frac_per_bin.append(100 * (p[mask] >= thresh).mean())
                else:
                    frac_per_bin.append(np.nan)
            ax.plot(mag_centres, frac_per_bin, 'o-', color=color, lw=2,
                    markersize=4, label=sample_name)

        ax.set_xlabel("Gaia G (mag)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Galaxy fraction (%)", fontsize=12, fontweight='bold')
        ax.set_title(thresh_label, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=11)

    out = OUTPUT_DIR / "gaia_3way_galaxy_frac_vs_mag.png"
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  -> {out}")

    # ── Corner plots of galaxy candidates ──
    from scipy.stats import gaussian_kde

    corner_feats = ['corrected_bp_rp_excess', 'g_rp', 'pm_significance']
    corner_idx = [FEAT_NAMES.index(f) for f in corner_feats]
    corner_labels = ['$C^*$', '$G-G_{RP}$', 'PM sig.', 'G mag']
    n_corner = len(corner_feats) + 1  # +1 for G magnitude

    for thresh, thresh_label in [(0.9, "P(gal) \u2265 0.9"), (0.5, "P(gal) \u2265 0.5")]:
        print(f"\n[8] Corner plot: {thresh_label}...")

        fig, axes = plt.subplots(n_corner, n_corner, figsize=(14, 14))
        fig.subplots_adjust(hspace=0.08, wspace=0.08)

        sample_arrays = {}
        for sample_name, _, _ in SAMPLES:
            X, mag, p, color, n = data[sample_name]
            mask = p >= thresh
            vals = np.column_stack([X[mask, fi] for fi in corner_idx] + [mag[mask]])
            good = np.all(np.isfinite(vals), axis=1)
            sample_arrays[sample_name] = vals[good]
            print(f"  {sample_name}: {good.sum():,} candidates ({thresh_label})")

        all_vals = np.vstack(list(sample_arrays.values()))
        ranges = []
        for j in range(n_corner):
            lo, hi = np.nanpercentile(all_vals[:, j], [1, 99])
            pad = (hi - lo) * 0.1
            ranges.append((lo - pad, hi + pad))

        for row in range(n_corner):
            for col in range(n_corner):
                ax = axes[row, col]

                if col > row:
                    ax.axis('off')
                    continue

                if row == col:
                    # Diagonal: 1D KDE + histogram
                    for sample_name, _, _ in SAMPLES:
                        _, _, _, color, _ = data[sample_name]
                        v = sample_arrays[sample_name][:, row]
                        ax.hist(v, bins=50, range=ranges[row], alpha=0.3,
                                color=color, density=True)
                        # KDE overlay
                        try:
                            kde = gaussian_kde(v)
                            xg = np.linspace(ranges[row][0], ranges[row][1], 200)
                            ax.plot(xg, kde(xg), color=color, lw=1.5,
                                    label=sample_name)
                        except Exception:
                            pass
                    if row == 0:
                        ax.legend(fontsize=8, loc='upper right')
                else:
                    # Off-diagonal: 2D KDE contours
                    for sample_name, _, _ in SAMPLES:
                        _, _, _, color, _ = data[sample_name]
                        vx = sample_arrays[sample_name][:, col]
                        vy = sample_arrays[sample_name][:, row]
                        # Scatter as background
                        ax.scatter(vx, vy, s=0.5, alpha=0.08, color=color,
                                   rasterized=True)
                        # KDE contours
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

        fig.suptitle(f"Galaxy candidates ({thresh_label}) \u2014 feature comparison by latitude",
                     fontsize=14, fontweight='bold', y=0.92)

        tag = str(thresh).replace('.', '')
        out = OUTPUT_DIR / f"gaia_3way_corner_p{tag}.png"
        fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  -> {out}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
