#!/usr/bin/env python3
"""
Plot Q1 XGBoost model confusion (TP/TN/FP/FN) on the Q1 individual UMAP.

Loads the saved Q1 seed ensemble (output/xgboost_q1/<mag_range>/seed_models/)
and runs predictions on the sources embedded in q1_indiv.npz.  Colours
each source by its outcome:

  TP  — model says GALAXY,  PHZ truth = GALAXY   (correct)
  TN  — model says STAR,    PHZ truth = STAR      (correct)
  FP  — model says GALAXY,  PHZ truth = STAR      (false positive)
  FN  — model says STAR,    PHZ truth = GALAXY    (false negative / missed galaxy)

Produces a 2-row × 4-col figure:
  Row 1: scatter coloured by TP/TN/FP/FN
           col 0 = all,  col 1 = FP only,  col 2 = FN only,
           col 3 = ensemble P(galaxy) continuous colour
  Row 2: 2D source-density histograms for the same panels

Reads:
  output/umap/<mag_range>/arrays/q1_indiv.npz
  output/xgboost_q1/<mag_range>/seed_models/xgb_classifier_q1_seed*.json

Writes:
  output/umap/<mag_range>/q1_individual/q1_confusion.png

Usage:
  python src/analyze_umap_q1_confusion.py                  # G >= 17
  python src/analyze_umap_q1_confusion.py --faint-cut 21   # 17 <= G <= 21
  python src/analyze_umap_q1_confusion.py --faint-cut 22   # 17 <= G <= 22
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

PROJECT_DIR    = Path(__file__).resolve().parents[1]

MAG_CUT        = 17
MAG_CUT_FAINT  = 99   # overridden by --faint-cut
PGAL_THRESHOLD = 0.50


def _mag_range_dir(faint: int) -> str:
    return f"{MAG_CUT}≤G" if faint >= 90 else f"{MAG_CUT}≤G≤{faint}"


# Paths — set after arg parsing
ARRAYS_DIR      = None
SEED_MODELS_DIR = None
OUTPUT_DIR      = None

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

# ── Colormap configuration ───────────────────────────────────────────────────
# Change these two strings only.
# Supports:
#   - Matplotlib names, e.g. "viridis", "RdYlBu_r"
#   - Colorcet names, e.g. "CET_L19", "CET_L6", "CET_R3"
CMAP = {
    "density": "CET_L19",
    # "density": "Blues",
    "pgal": "RdYlBu_r",
}

# Use only a slice of each colormap if desired.
# (0.0, 1.0) = full map, (0.6, 1.0) = upper 40% only.
CMAP_SLICE = {
    "density": (0.3, 1.0),
    "pgal": (0.0, 1.0),
}

# ── Colours ──────────────────────────────────────────────────────────────────
COLOR = {
    "TP": "#1976D2",   # blue   — correct galaxy
    "TN": "#E53935",   # red    — correct star
    "FP": "lightpink",     # false positive (pred GAL, true STAR)
    "FN": "deepskyblue",   # false negative (pred STAR, true GAL)
}


def get_plot_cmap(name: str, *, bad_alpha_zero: bool = False):
    """Return a copy of a matplotlib or colorcet colormap.

    If `bad_alpha_zero` is True, NaN/masked values are transparent.
    Colorcet aliases like CET_L06 -> CET_L6 are accepted.
    """
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
            "Install colorcet:  pip install colorcet  (or conda install colorcet)"
        ) from e

    aliases = {
        "CET_L06": "CET_L6",
        "cet_l06": "CET_L6",
        "CET-L06": "CET_L6",
        "cet-l06": "CET_L6",
        "L06": "CET_L6",
        "l06": "CET_L6",
    }
    key = aliases.get(normalized_name, normalized_name)

    if hasattr(cc, f"m_{key}"):
        cmap = getattr(cc, f"m_{key}").copy()
    elif key in cc.cm:
        cmap = cc.cm[key].copy()
    else:
        # Short keys like "l19" for CET_L19
        tail = key.replace("CET_", "", 1).lower() if key.startswith("CET_") else key.lower()
        if tail in cc.cm:
            cmap = cc.cm[tail].copy()
        else:
            similar = [k for k in cc.cm.keys() if "L6" in k or "L06" in k]
            raise KeyError(
                f"{name!r} → resolved {key!r} not in colorcet. "
                f"Try 'CET_L6' for L06. Example keys: {similar[:8]!r}…"
            )

    if bad_alpha_zero:
        cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    return cmap


def slice_cmap(cmap, minval: float = 0.0, maxval: float = 1.0, n: int = 256):
    """Return a truncated copy of a colormap."""
    if minval <= 0.0 and maxval >= 1.0:
        return cmap.copy()
    return LinearSegmentedColormap.from_list(
        f"slice_{cmap.name}_{minval:.2f}_{maxval:.2f}",
        cmap(np.linspace(minval, maxval, n)),
    )


def load_ensemble_proba(X: np.ndarray) -> np.ndarray:
    """Run all seed models and return mean P(galaxy)."""
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


def load_phz_gal_prob(source_ids: np.ndarray, username: str) -> np.ndarray:
    """Fetch p_phz_gal_prob from DB for the given source_ids. Returns array aligned to source_ids."""
    import psycopg2
    ids = [int(s) for s in source_ids]
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT g_source_id, p_phz_gal_prob
                FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
                WHERE g_source_id = ANY(%s)
                """,
                (ids,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    lookup = {str(r[0]): r[1] for r in rows}
    return np.array([lookup.get(str(s), np.nan) for s in source_ids], dtype=float)


def main() -> int:
    global MAG_CUT_FAINT, ARRAYS_DIR, SEED_MODELS_DIR, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Q1 UMAP confusion plot")
    parser.add_argument(
        "--faint-cut", type=int, default=None, metavar="MAG",
        help="Upper G-mag limit (e.g. 21, 22). Omit for no faint cut (G >= 17).",
    )
    args = parser.parse_args()

    if args.faint_cut is not None:
        MAG_CUT_FAINT = args.faint_cut

    mag_dir = _mag_range_dir(MAG_CUT_FAINT)
    ARRAYS_DIR      = PROJECT_DIR / "output" / "umap"       / "arrays"
    SEED_MODELS_DIR = PROJECT_DIR / "output" / "xgboost_q1" / mag_dir / "seed_models"
    OUTPUT_DIR      = PROJECT_DIR / "output" / "umap"       / "q1_individual"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Magnitude range : {mag_dir}")
    print(f"UMAP arrays     : {ARRAYS_DIR}")
    print(f"Seed models     : {SEED_MODELS_DIR}")

    # ── Load UMAP arrays ─────────────────────────────────────────────────────
    npz_path = ARRAYS_DIR / "q1_indiv.npz"
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found. Run analyze_umap.py first.")
        return 1

    import os
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER environment variable not set.")
        return 1

    print("Loading Q1 individual UMAP arrays...")
    data       = np.load(npz_path, allow_pickle=True)
    mag_label = f'G \u2265 {MAG_CUT}' if MAG_CUT_FAINT >= 90 else f'{MAG_CUT} \u2264 G \u2264 {MAG_CUT_FAINT}'
    umap_mag_cut_faint = int(data['mag_cut_faint']) if 'mag_cut_faint' in data else 99
    umap_dataset_label = f'G \u2265 {MAG_CUT}' if umap_mag_cut_faint >= 90 else f'{MAG_CUT} \u2264 G \u2264 {umap_mag_cut_faint}'
    embedding  = data["embedding"]
    y_true     = data["y"]          # 0=STAR, 1=GALAXY  (PHZ truth)
    X_raw      = data["X_raw"]      # 16 features, un-standardised
    source_ids = data["source_ids"].astype(str)

    print(f"  {len(y_true):,} sources  (STAR={( y_true==0).sum():,}  "
          f"GALAXY={(y_true==1).sum():,})")

    # ── Impute NaN then predict ───────────────────────────────────────────────
    X_clean = X_raw.copy()
    for c in range(X_clean.shape[1]):
        bad = np.isnan(X_clean[:, c])
        if bad.any():
            X_clean[bad, c] = np.nanmedian(X_clean[:, c])

    print(f"\nRunning Q1 ensemble on {len(X_clean):,} UMAP sources...")
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

    # ── Fetch PHZ galaxy probability ──────────────────────────────────────────
    print("\nFetching p_phz_gal_prob from DB...")
    phz_gal_prob = load_phz_gal_prob(source_ids, username)
    n_missing = np.isnan(phz_gal_prob).sum()
    print(f"  Retrieved {(~np.isnan(phz_gal_prob)).sum():,} values  ({n_missing:,} NaN)")

    # ── Plotting setup ────────────────────────────────────────────────────────
    rng = np.random.RandomState(0)
    x_lo = embedding[:, 0].min() - 1;  x_hi = embedding[:, 0].max() + 1
    y_lo = embedding[:, 1].min() - 1;  y_hi = embedding[:, 1].max() + 1

    n_bins  = 200
    x_edges = np.linspace(x_lo, x_hi, n_bins + 1)
    y_edges = np.linspace(y_lo, y_hi, n_bins + 1)

    from matplotlib.colors import LogNorm
    from matplotlib.gridspec import GridSpec

    cmap_dens    = slice_cmap(
        get_plot_cmap(CMAP["density"], bad_alpha_zero=True),
        *CMAP_SLICE["density"],
    )
    cmap_dens_q1 = slice_cmap(get_plot_cmap(CMAP["density"], bad_alpha_zero=True), 0.3, 1.0)
    cmap_pgal = slice_cmap(get_plot_cmap(CMAP["pgal"]), *CMAP_SLICE["pgal"])

    # ── Font size constants (change here to affect all panels) ────────────────
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
    # Col layout:
    #   0  per-row        →  PHZ stars (row0) / PHZ galaxies (row1)
    #   1  rows 0:2 span  →  PHZ P(galaxy)
    #   2  rows 0:2 span  →  Ensemble P(galaxy)
    #   2  rows 3:5 span  →  All source density
    #   1  rows 3:5 span  →  Q1 source density
    #   3  per-row        →  FP (row0) / FN (row1)
    #   4  per-row        →  TN (row0) / TP (row1)

    scatter_h = 0.5
    density_h = 0.5
    outer_gap = 0.02
    height_ratios = [scatter_h, density_h, outer_gap, scatter_h, density_h]
    panel_frac = scatter_h / sum(height_ratios)
    fig_height = 3.0 / panel_frac
    fig_width  = 38   # 5 cols × 7.5

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = GridSpec(5, 5, figure=fig, height_ratios=height_ratios,
                   hspace=0.22, wspace=0.25, top=0.93)

    # New col 0: PHZ class scatter + density
    ax_phz_star_sc = fig.add_subplot(gs[0, 0])   # PHZ stars scatter
    ax_phz_star_d  = fig.add_subplot(gs[1, 0])   # PHZ stars density
    ax_phz_gal_sc  = fig.add_subplot(gs[3, 0])   # PHZ galaxies scatter
    ax_phz_gal_d   = fig.add_subplot(gs[4, 0])   # PHZ galaxies density

    # Summary column axes (span scatter+density)
    ax_phz   = fig.add_subplot(gs[0:2, 1])   # PHZ P(galaxy)
    ax_pgal  = fig.add_subplot(gs[0:2, 2])   # Ensemble P(galaxy)
    ax_all_d = fig.add_subplot(gs[3:5, 2])   # All density (row group 1)
    ax_blank = fig.add_subplot(gs[3:5, 1])   # Q1 source density

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

    phz_star_mask = y_true == 0
    phz_gal_mask  = y_true == 1

    # ── Pre-compute density histograms (shared LogNorm scale) ─────────────────
    hist_masks  = [np.ones(len(embedding), dtype=bool),
                   fp_mask, fn_mask, tp_mask, tn_mask,
                   phz_star_mask, phz_gal_mask]
    hist_keys   = ["ALL", "FP", "FN", "TP", "TN", "PHZ_STAR", "PHZ_GAL"]
    histograms  = {}
    for key, mask in zip(hist_keys, hist_masks):
        H, _, _ = np.histogram2d(embedding[mask, 0], embedding[mask, 1],
                                  bins=[x_edges, y_edges])
        histograms[key] = np.where(H > 0, H, np.nan)

    all_vals   = [h for h in histograms.values() if not np.all(np.isnan(h))]
    shared_norm = LogNorm(
        vmin=np.nanmin([np.nanmin(h) for h in all_vals]),
        vmax=np.nanmax([np.nanmax(h) for h in all_vals]),
    )

    H_all = histograms["ALL"]
    q1_dens_norm = LogNorm(vmin=np.nanmin(H_all), vmax=np.nanmax(H_all))

    # ── Scatter panels ────────────────────────────────────────────────────────
    idx_all = _downsample(np.ones(len(embedding), dtype=bool))
    bg_idx  = idx_all

    # Col 0: PHZ stars / PHZ galaxies
    for ax_s, mask, color, label_str in [
        (ax_phz_star_sc, phz_star_mask, COLOR["TN"],
         f"PHZ stars (n={phz_star_mask.sum():,})"),
        (ax_phz_gal_sc,  phz_gal_mask,  COLOR["TP"],
         f"PHZ galaxies (n={phz_gal_mask.sum():,})"),
    ]:
        ax_s.scatter(embedding[bg_idx, 0], embedding[bg_idx, 1],
                     c="lightgrey", s=2, alpha=0.12, linewidths=0, zorder=1,
                     label=f"Q1 all (n={len(y_true):,})")
        idx = _downsample(mask)
        ax_s.scatter(embedding[idx, 0], embedding[idx, 1],
                     c=color, s=6, alpha=0.15, linewidths=0, zorder=3,
                     label=label_str)
        _style_sc(ax_s, label_str)

    for ax_d, key in [(ax_phz_star_d, "PHZ_STAR"), (ax_phz_gal_d, "PHZ_GAL")]:
        sc2 = ax_d.pcolormesh(x_edges, y_edges, histograms[key].T,
                              cmap=cmap_dens, norm=shared_norm, rasterized=True)
        _style_dn(ax_d)
        fig.colorbar(sc2, ax=ax_d, label="Count (log)", pad=0.02, fraction=0.046)

    # Col 1: PHZ P(galaxy)
    sc_phz = ax_phz.scatter(embedding[idx_all, 0], embedding[idx_all, 1],
                             c=phz_gal_prob[idx_all], cmap=cmap_pgal,
                             vmin=0, vmax=1, s=3, alpha=0.6, linewidths=0,
                             rasterized=True)
    fig.colorbar(sc_phz, ax=ax_phz, label="PHZ P(galaxy)", pad=0.02, fraction=0.046)
    _style_sum(ax_phz, f"Q1: PHZ P(galaxy)  (n={len(y_true):,})")

    # Col 2: Ensemble P(galaxy)
    sc = ax_pgal.scatter(embedding[idx_all, 0], embedding[idx_all, 1],
                         c=proba[idx_all], cmap=cmap_pgal,
                         vmin=0, vmax=1, s=3, alpha=0.6, linewidths=0,
                         rasterized=True)
    fig.colorbar(sc, ax=ax_pgal, label="P(galaxy)", pad=0.02, fraction=0.046)
    _style_sum(ax_pgal, f"Q1: Model P(galaxy)  (n={len(y_true):,})")

    # Outcome scatter panels: FP, TN (row 0) and FN, TP (row 1)
    for ax_s, mask, key, title in [
        (ax_fp_sc, fp_mask, "FP", f"FP — pred galaxy, true star  (n={counts['FP']:,})"),
        (ax_tn_sc, tn_mask, "TN", f"TN — correct star  (n={counts['TN']:,})"),
        (ax_fn_sc, fn_mask, "FN", f"FN — pred star, true galaxy  (n={counts['FN']:,})"),
        (ax_tp_sc, tp_mask, "TP", f"TP — correct galaxy  (n={counts['TP']:,})"),
    ]:
        ax_s.scatter(embedding[bg_idx, 0], embedding[bg_idx, 1],
                     c="lightgrey", s=2, alpha=0.12, linewidths=0, zorder=1,
                     label=f"Q1 all (n={len(y_true):,})")
        idx   = _downsample(mask)
        alpha = 0.15 if key in ("TN", "TP") else 0.6
        ax_s.scatter(embedding[idx, 0], embedding[idx, 1],
                     c=COLOR[key], s=6, alpha=alpha,
                     linewidths=0, zorder=3,
                     label=f"{key} (n={counts[key]:,})")
        _style_sc(ax_s, title)

    # ── Density panels ────────────────────────────────────────────────────────

    # Col 0 row-group 1: Q1 source density (same as col 1)
    sc_blank = ax_blank.pcolormesh(x_edges, y_edges, H_all.T,
                                   cmap=cmap_dens_q1, norm=q1_dens_norm, rasterized=True)
    fig.colorbar(sc_blank, ax=ax_blank, label="Count (log)", pad=0.02, fraction=0.046)
    _style_sum(ax_blank, f"Q1: source density  (n={len(y_true):,})")

    # Col 1 row-group 1: All source density
    sc_all = ax_all_d.pcolormesh(x_edges, y_edges, H_all.T,
                                  cmap=cmap_dens_q1, norm=q1_dens_norm, rasterized=True)
    fig.colorbar(sc_all, ax=ax_all_d, label="Count (log)", pad=0.02, fraction=0.046)
    _style_sum(ax_all_d, f"Q1: source density  (n={len(y_true):,})")

    # Outcome density panels
    for ax_d, key in [
        (ax_fp_d, "FP"), (ax_tn_d, "TN"),
        (ax_fn_d, "FN"), (ax_tp_d, "TP"),
    ]:
        sc2 = ax_d.pcolormesh(x_edges, y_edges, histograms[key].T,
                              cmap=cmap_dens, norm=shared_norm, rasterized=True)
        _style_dn(ax_d)
        fig.colorbar(sc2, ax=ax_d, label="Count (log)", pad=0.02, fraction=0.046)

    fig.suptitle(
        f"Q1 Individual UMAP — XGBoost Q1 Model Confusion\n"
        f"UMAP dataset: {umap_dataset_label}  |  model trained on: {mag_label}  |  "
        f"{len(SELECTED_GAIA_PARAMETERS)} features,  threshold={PGAL_THRESHOLD}  |  "
        f"precision={100*accuracy:.1f}%  "
        f"galaxy recall={100*counts['TP']/(counts['TP']+counts['FN']):.1f}%",
        fontsize=12, fontweight="bold", y=0.995
    )

    out = OUTPUT_DIR / "q1_confusion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
