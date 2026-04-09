#!/usr/bin/env python3
"""
Plot Q1 XGBoost model confusion (TP/TN/FP/FN) on the ERO individual UMAP.

Loads the saved Q1 seed ensemble (output/xgboost_q1/seed_models/) and runs
predictions on the 50k sources embedded in ero_indiv.npz.  ERO truth labels
come from e_v_class_star (threshold 0.65).  Colours each source by outcome:

  TP  — model says GALAXY,  ERO truth = GALAXY   (correct)
  TN  — model says STAR,    ERO truth = STAR      (correct)
  FP  — model says GALAXY,  ERO truth = STAR      (false positive)
  FN  — model says STAR,    ERO truth = GALAXY    (false negative)

Reads:
  output/umap/arrays/ero_indiv.npz
  output/umap/arrays/ero_indiv_rebalanced.npz
  output/xgboost_q1/seed_models/xgb_classifier_q1_seed*.json

Writes (default: both files if the corresponding .npz exists):
  output/umap/ero_individual/ero_umap_confusion.png
  output/umap/ero_individual/ero_umap_confusion_rebalanced.png

Usage:
  python src/analyze_umap_ero_disagreement.py
  python src/analyze_umap_ero_disagreement.py --regular-only
  python src/analyze_umap_ero_disagreement.py --rebalanced-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR     = Path(__file__).resolve().parents[1]
ARRAYS_DIR      = PROJECT_DIR / "output" / "umap" / "arrays"
SEED_MODELS_DIR = PROJECT_DIR / "output" / "xgboost_q1" / "seed_models"
OUTPUT_DIR      = PROJECT_DIR / "output" / "umap" / "ero_individual"

MAG_CUT        = 17
PGAL_THRESHOLD = 0.50


# ── Colormap configuration ────────────────────────────────────────────────────
CMAP = {
    "density": "CET_L19",
    "pgal":    "RdYlBu_r",
}
CMAP_SLICE = {
    "density": (0.3, 1.0),
    "pgal":    (0.0, 1.0),
}

# ── Colours ───────────────────────────────────────────────────────────────────
COLOR = {
    "TP": "#1976D2",   # blue   — correct galaxy
    "TN": "#E53935",   # red    — correct star
    "FP": "lightpink",    # orange — false positive
    "FN": "deepskyblue",      # plum   — false negative
}


def get_plot_cmap(name: str, *, bad_alpha_zero: bool = False):
    normalized_name = name.replace("-", "_")
    try:
        cmap = plt.get_cmap(normalized_name).copy()
        if bad_alpha_zero:
            cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
        return cmap
    except ValueError:
        pass
    try:
        import colorcet as cc
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Install colorcet:  pip install colorcet"
        ) from e
    aliases = {
        "CET_L06": "CET_L6", "cet_l06": "CET_L6",
        "CET-L06": "CET_L6", "cet-l06": "CET_L6",
        "L06": "CET_L6",     "l06": "CET_L6",
    }
    key = aliases.get(normalized_name, normalized_name)
    if hasattr(cc, f"m_{key}"):
        cmap = getattr(cc, f"m_{key}").copy()
    elif key in cc.cm:
        cmap = cc.cm[key].copy()
    else:
        tail = key.replace("CET_", "", 1).lower() if key.startswith("CET_") else key.lower()
        if tail in cc.cm:
            cmap = cc.cm[tail].copy()
        else:
            raise KeyError(f"{name!r} not found in colorcet")
    if bad_alpha_zero:
        cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    return cmap


def slice_cmap(cmap, minval: float = 0.0, maxval: float = 1.0, n: int = 256):
    if minval <= 0.0 and maxval >= 1.0:
        return cmap.copy()
    return LinearSegmentedColormap.from_list(
        f"slice_{cmap.name}_{minval:.2f}_{maxval:.2f}",
        cmap(np.linspace(minval, maxval, n)),
    )


def load_ensemble_proba(X: np.ndarray) -> np.ndarray:
    seed_paths = sorted(SEED_MODELS_DIR.glob("xgb_classifier_q1_seed*.json"))
    if not seed_paths:
        raise FileNotFoundError(
            f"No seed models found in {SEED_MODELS_DIR}.\n"
            "Run train_xgboost_q1.py first."
        )
    all_probs = []
    for p in seed_paths:
        m = xgb.XGBClassifier()
        m.load_model(str(p))
        all_probs.append(m.predict_proba(X.astype(np.float32))[:, 1])
    proba = np.mean(all_probs, axis=0)
    print(f"  Ensemble of {len(seed_paths)} seed models — mean P(galaxy) computed.")
    return proba


def load_ero_class_star_from_npz(npz_data) -> np.ndarray:
    """Load e_v_class_star from ERO npz file (no DB needed)."""
    return npz_data["class_star"].astype(float)


def _run_pipeline(use_rebalanced: bool) -> bool:
    """Load one ERO UMAP npz, predict, plot, save. Returns True if figure written."""
    npz_name = "ero_indiv_rebalanced.npz" if use_rebalanced else "ero_indiv.npz"
    npz_path = ARRAYS_DIR / npz_name
    if not npz_path.exists():
        print(f"WARNING: {npz_path} not found — skip ({'rebalanced' if use_rebalanced else 'regular'}).")
        return False

    print(f"\n{'='*60}\nLoading ERO UMAP arrays ({npz_name})...")
    data       = np.load(npz_path, allow_pickle=True)
    mag_cut_faint = int(data['mag_cut_faint']) if 'mag_cut_faint' in data else 99
    mag_label = f'G \u2265 {MAG_CUT}' if mag_cut_faint >= 90 else f'{MAG_CUT} \u2264 G \u2264 {mag_cut_faint}'
    embedding  = data["embedding"]
    y_true     = data["y"]          # 0=STAR, 1=GALAXY  (ERO e_v_class_star truth)
    X_raw      = data["X_raw"]      # 16 Gaia features, un-standardised
    source_ids = data["source_ids"].astype(str)

    print(f"  {len(y_true):,} sources  (STAR={(y_true==0).sum():,}  "
          f"GALAXY={(y_true==1).sum():,})")

    # ── Impute NaN then predict with Q1 model ────────────────────────────────
    X_clean = X_raw.copy()
    for c in range(X_clean.shape[1]):
        bad = np.isnan(X_clean[:, c])
        if bad.any():
            X_clean[bad, c] = np.nanmedian(X_clean[:, c])

    print(f"\nRunning Q1 ensemble on {len(X_clean):,} ERO UMAP sources...")
    proba  = load_ensemble_proba(X_clean)
    y_pred = (proba >= PGAL_THRESHOLD).astype(int)

    # ── Outcome labels ────────────────────────────────────────────────────────
    outcome = np.empty(len(y_true), dtype="U2")
    outcome[(y_pred == 1) & (y_true == 1)] = "TP"
    outcome[(y_pred == 0) & (y_true == 0)] = "TN"
    outcome[(y_pred == 1) & (y_true == 0)] = "FP"
    outcome[(y_pred == 0) & (y_true == 1)] = "FN"

    counts = {k: (outcome == k).sum() for k in ("TP", "TN", "FP", "FN")}
    total  = len(y_true)
    print(f"\nOutcome breakdown (threshold={PGAL_THRESHOLD}):")
    for k, n in counts.items():
        print(f"  {k}: {n:>6,}  ({100*n/total:.1f}%)")
    accuracy = (counts["TP"] + counts["TN"]) / total
    print(f"  Accuracy: {100*accuracy:.2f}%")

    # ── Load ERO class_star from npz ─────────────────────────────────────────
    print("\nLoading e_v_class_star from npz...")
    class_star = load_ero_class_star_from_npz(data)
    ero_pgal   = 1.0 - class_star   # flip so high = galaxy
    n_missing  = np.isnan(ero_pgal).sum()
    print(f"  Retrieved {(~np.isnan(ero_pgal)).sum():,} values  ({n_missing:,} NaN)")

    # ── Plotting setup ────────────────────────────────────────────────────────
    rng = np.random.RandomState(0)
    x_lo = embedding[:, 0].min() - 1;  x_hi = embedding[:, 0].max() + 1
    y_lo = embedding[:, 1].min() - 1;  y_hi = embedding[:, 1].max() + 1

    n_bins  = 180
    x_edges = np.linspace(x_lo, x_hi, n_bins + 1)
    y_edges = np.linspace(y_lo, y_hi, n_bins + 1)

    from matplotlib.colors import LogNorm
    from matplotlib.gridspec import GridSpec

    cmap_dens     = slice_cmap(get_plot_cmap(CMAP["density"], bad_alpha_zero=True),
                               *CMAP_SLICE["density"])
    cmap_dens_ero = slice_cmap(get_plot_cmap(CMAP["density"], bad_alpha_zero=True), 0.3, 1.0)
    cmap_pgal     = slice_cmap(get_plot_cmap(CMAP["pgal"]), *CMAP_SLICE["pgal"])

    # ── Font size constants ───────────────────────────────────────────────────
    FS_TITLE = 9
    FS_LABEL = 8
    FS_TICK  = 7

    MAX_PTS = 15_000

    def _downsample(mask):
        idx = np.where(mask)[0]
        if len(idx) > MAX_PTS:
            idx = rng.choice(idx, MAX_PTS, replace=False)
        return idx

    # ── Shared axis-styling helpers ───────────────────────────────────────────
    def _style_sc(ax, title):
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("UMAP 1", fontsize=FS_LABEL)
        ax.set_ylabel("UMAP 2", fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.2)
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(fontsize=6, loc="upper right", markerscale=1.5)

    def _style_sum(ax, title):
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("UMAP 1", fontsize=FS_LABEL)
        ax.set_ylabel("UMAP 2", fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.2)

    def _style_dn(ax):
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(alpha=0.2)

    # ── Figure layout: 5 rows × 5 cols ───────────────────────────────────────
    scatter_h = 0.5
    density_h = 0.5
    outer_gap = 0.02
    height_ratios = [scatter_h, density_h, outer_gap, scatter_h, density_h]
    panel_frac = scatter_h / sum(height_ratios)
    fig_height = 3.0 / panel_frac
    fig_width  = 37.5   # 5 cols × 7.5

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = GridSpec(5, 5, figure=fig, height_ratios=height_ratios,
                   hspace=0.22, wspace=0.25, top=0.93)

    # New col 0: ERO label scatter + density
    ax_ero_star_sc = fig.add_subplot(gs[0, 0])   # ERO stars scatter
    ax_ero_star_d  = fig.add_subplot(gs[1, 0])   # ERO stars density
    ax_ero_gal_sc  = fig.add_subplot(gs[3, 0])   # ERO galaxies scatter
    ax_ero_gal_d   = fig.add_subplot(gs[4, 0])   # ERO galaxies density

    # Summary column axes (span scatter+density)
    ax_ero   = fig.add_subplot(gs[0:2, 1])   # ERO P(galaxy)
    ax_pgal  = fig.add_subplot(gs[0:2, 2])   # Q1 Model P(galaxy)
    ax_all_d = fig.add_subplot(gs[3:5, 2])   # All density (row group 1)
    ax_blank = fig.add_subplot(gs[3:5, 1])   # ERO source density (row group 1)

    # Outcome scatter axes
    ax_fp_sc = fig.add_subplot(gs[0, 3])
    ax_tn_sc = fig.add_subplot(gs[0, 4])
    ax_fn_sc = fig.add_subplot(gs[3, 3])
    ax_tp_sc = fig.add_subplot(gs[3, 4])

    # Outcome density axes
    ax_fp_d = fig.add_subplot(gs[1, 3])
    ax_tn_d = fig.add_subplot(gs[1, 4])
    ax_fn_d = fig.add_subplot(gs[4, 3])
    ax_tp_d = fig.add_subplot(gs[4, 4])

    # ── Outcome masks ─────────────────────────────────────────────────────────
    fp_mask = outcome == "FP"
    fn_mask = outcome == "FN"
    tp_mask = outcome == "TP"
    tn_mask = outcome == "TN"

    ero_star_mask = y_true == 0
    ero_gal_mask  = y_true == 1

    # ── Pre-compute density histograms (shared LogNorm scale) ─────────────────
    hist_masks = [np.ones(len(embedding), dtype=bool),
                  fp_mask, fn_mask, tp_mask, tn_mask,
                  ero_star_mask, ero_gal_mask]
    hist_keys  = ["ALL", "FP", "FN", "TP", "TN", "ERO_STAR", "ERO_GAL"]
    histograms = {}
    for key, mask in zip(hist_keys, hist_masks):
        H, _, _ = np.histogram2d(embedding[mask, 0], embedding[mask, 1],
                                  bins=[x_edges, y_edges])
        histograms[key] = np.where(H > 0, H, np.nan)

    all_vals    = [h for h in histograms.values() if not np.all(np.isnan(h))]
    shared_norm = LogNorm(
        vmin=np.nanmin([np.nanmin(h) for h in all_vals]),
        vmax=np.nanmax([np.nanmax(h) for h in all_vals]),
    )

    H_all = histograms["ALL"]
    ero_dens_norm = LogNorm(vmin=np.nanmin(H_all), vmax=np.nanmax(H_all))

    # ── Scatter panels ────────────────────────────────────────────────────────
    idx_all = _downsample(np.ones(len(embedding), dtype=bool))
    bg_idx  = idx_all

    # Col 0: ERO label panels (stars row group 0, galaxies row group 1)
    for ax_s, mask, color, label_str in [
        (ax_ero_star_sc, ero_star_mask, COLOR["TN"],
         f"ERO stars: class_star ≥ 0.65  (n={ero_star_mask.sum():,})"),
        (ax_ero_gal_sc,  ero_gal_mask,  COLOR["TP"],
         f"ERO galaxies: class_star < 0.65  (n={ero_gal_mask.sum():,})"),
    ]:
        ax_s.scatter(embedding[bg_idx, 0], embedding[bg_idx, 1],
                     c="lightgrey", s=2, alpha=0.12, linewidths=0, zorder=1,
                     label=f"ERO all (n={len(y_true):,})")
        idx = _downsample(mask)
        ax_s.scatter(embedding[idx, 0], embedding[idx, 1],
                     c=color, s=6, alpha=0.15, linewidths=0, zorder=3,
                     label=label_str)
        _style_sc(ax_s, label_str)

    for ax_d, key in [(ax_ero_star_d, "ERO_STAR"), (ax_ero_gal_d, "ERO_GAL")]:
        sc2 = ax_d.pcolormesh(x_edges, y_edges, histograms[key].T,
                              cmap=cmap_dens, norm=shared_norm, rasterized=True)
        _style_dn(ax_d)
        fig.colorbar(sc2, ax=ax_d, label="Count (log)", pad=0.02, fraction=0.046)

    # Col 1: ERO P(galaxy) = 1 - e_v_class_star
    sc_ero = ax_ero.scatter(embedding[idx_all, 0], embedding[idx_all, 1],
                             c=ero_pgal[idx_all], cmap=cmap_pgal,
                             vmin=0, vmax=1, s=3, alpha=0.6, linewidths=0,
                             rasterized=True)
    fig.colorbar(sc_ero, ax=ax_ero, label="ERO P(galaxy)", pad=0.02, fraction=0.046)
    _style_sum(ax_ero, f"ERO: ERO P(galaxy)  (n={len(y_true):,})")

    # Col 1: Q1 Model P(galaxy)
    sc = ax_pgal.scatter(embedding[idx_all, 0], embedding[idx_all, 1],
                         c=proba[idx_all], cmap=cmap_pgal,
                         vmin=0, vmax=1, s=3, alpha=0.6, linewidths=0,
                         rasterized=True)
    fig.colorbar(sc, ax=ax_pgal, label="P(galaxy)", pad=0.02, fraction=0.046)
    _style_sum(ax_pgal, f"ERO: Model P(galaxy)  (n={len(y_true):,})")

    # Human-readable labels (avoid TP/FP framing since ERO labels are uncertain)
    DISPLAY = {
        "FP": "Disagree — Q1: galaxy,  ERO: star",
        "TN": "Agree — Q1: star,  ERO: star",
        "FN": "Disagree — Q1: star,  ERO: galaxy",
        "TP": "Agree — Q1: galaxy,  ERO: galaxy",
    }

    # Outcome scatter panels: FP, TN (row 0) and FN, TP (row 1)
    for ax_s, mask, key in [
        (ax_fp_sc, fp_mask, "FP"),
        (ax_tn_sc, tn_mask, "TN"),
        (ax_fn_sc, fn_mask, "FN"),
        (ax_tp_sc, tp_mask, "TP"),
    ]:
        title = f"{DISPLAY[key]}  (n={counts[key]:,})"
        ax_s.scatter(embedding[bg_idx, 0], embedding[bg_idx, 1],
                     c="lightgrey", s=2, alpha=0.12, linewidths=0, zorder=1,
                     label=f"ERO all (n={len(y_true):,})")
        idx   = _downsample(mask)
        alpha = 0.15 if key in ("TN", "TP") else 0.6
        ax_s.scatter(embedding[idx, 0], embedding[idx, 1],
                     c=COLOR[key], s=6, alpha=alpha,
                     linewidths=0, zorder=3,
                     label=f"{DISPLAY[key]}  (n={counts[key]:,})")
        _style_sc(ax_s, title)

    # ── Density panels ────────────────────────────────────────────────────────

    # Col 0 row-group 1: ERO source density
    sc_blank = ax_blank.pcolormesh(x_edges, y_edges, H_all.T,
                                   cmap=cmap_dens_ero, norm=ero_dens_norm, rasterized=True)
    fig.colorbar(sc_blank, ax=ax_blank, label="Count (log)", pad=0.02, fraction=0.046)
    _style_sum(ax_blank, f"ERO: source density  (n={len(y_true):,})")

    # Col 1 row-group 1: All source density
    sc_all = ax_all_d.pcolormesh(x_edges, y_edges, H_all.T,
                                  cmap=cmap_dens_ero, norm=ero_dens_norm, rasterized=True)
    fig.colorbar(sc_all, ax=ax_all_d, label="Count (log)", pad=0.02, fraction=0.046)
    _style_sum(ax_all_d, f"ERO: source density  (n={len(y_true):,})")

    # Outcome density panels
    for ax_d, key in [
        (ax_fp_d, "FP"), (ax_tn_d, "TN"),
        (ax_fn_d, "FN"), (ax_tp_d, "TP"),
    ]:
        sc2 = ax_d.pcolormesh(x_edges, y_edges, histograms[key].T,
                              cmap=cmap_dens, norm=shared_norm, rasterized=True)
        _style_dn(ax_d)
        fig.colorbar(sc2, ax=ax_d, label="Count (log)", pad=0.02, fraction=0.046)

    label = "Rebalanced" if use_rebalanced else "Individual"
    fig.suptitle(
        f"ERO {label} UMAP — Q1 XGBoost Model vs ERO labels  "
        f"({mag_label},  {len(SELECTED_GAIA_PARAMETERS)} features,  "
        f"threshold={PGAL_THRESHOLD})\n"
        f"Q1 gal & ERO gal={counts['TP']:,}  "
        f"Q1 star & ERO star={counts['TN']:,}  "
        f"Q1 gal & ERO star={counts['FP']:,}  "
        f"Q1 star & ERO gal={counts['FN']:,}",
        fontsize=12, fontweight="bold", y=0.995
    )

    suffix = "_rebalanced" if use_rebalanced else ""
    out = OUTPUT_DIR / f"ero_agreement{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> Saved: {out}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--regular-only",
        action="store_true",
        help="Only produce ero_umap_confusion.png (ero_indiv.npz)",
    )
    grp.add_argument(
        "--rebalanced-only",
        action="store_true",
        help="Only produce ero_umap_confusion_rebalanced.png (ero_indiv_rebalanced.npz)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.regular_only:
        tasks = [False]
    elif args.rebalanced_only:
        tasks = [True]
    else:
        tasks = [False, True]

    ok_any = False
    for use_rebalanced in tasks:
        if _run_pipeline(use_rebalanced):
            ok_any = True

    if not ok_any:
        print("ERROR: No figure written — ensure analyze_umap.py produced the needed .npz files.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
