#!/usr/bin/env python3
"""
Project ERO sources into the Q1 individual UMAP space and show where they sit.

Uses the fitted Q1 UMAP reducer (q1_indiv_reducer.joblib) and Q1 normalisation
stats (mean/std from q1_indiv.npz) to project ERO Gaia features into the same
2D space as Q1 sources.

Layout: 2 rows × 4 cols
  Col 0 (spans both rows): Q1 source density
  Cols 1–3:
    Row 0 (scatter): ERO all | ERO stars | ERO galaxies
    Row 1 (density): ERO all | ERO stars | ERO galaxies

Reads:
  output/umap/arrays/q1_indiv.npz
  output/umap/arrays/q1_indiv_reducer.joblib
  output/umap/arrays/ero_indiv.npz
  output/umap/arrays/ero_indiv_rebalanced.npz

Writes:
  output/umap/q1_individual/q1_space_ero.png
  output/umap/q1_individual/q1_space_ero_rebalanced.png

Usage:
  python src/analyze_umap_q1_ero_overlay.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.gridspec import GridSpec

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "q1_individual"

# ── Colormap ──────────────────────────────────────────────────────────────────
CMAP_DENSITY = "CET_L19"

# Q1 background colour (single, all sources)
Q1_BG = "lightgrey"

# ERO overlay colours (lighter shades)
ERO_ALL_C  = "#CE93D8"   # light purple — ERO all
ERO_STAR_C = "#EF9A9A"   # light red    — ERO stars
ERO_GAL_C  = "#90CAF9"   # light blue   — ERO galaxies


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
        raise ModuleNotFoundError("Install colorcet:  pip install colorcet") from e
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


def _run_pipeline(
    npz_name: str,
    out_name: str,
    q1_emb: np.ndarray,
    q1_y: np.ndarray,
    q1_mean: np.ndarray,
    q1_std: np.ndarray,
    reducer,
) -> bool:
    npz_path = ARRAYS_DIR / npz_name
    if not npz_path.exists():
        print(f"WARNING: {npz_path} not found — skipping.")
        return False

    print(f"\n{'='*60}")
    print(f"Loading ERO arrays ({npz_name})...")
    ero       = np.load(npz_path, allow_pickle=True)
    ero_X_raw = ero["X_raw"].astype(float)
    ero_y     = ero["y"]
    print(f"  ERO: {len(ero_y):,} sources  "
          f"(STAR={(ero_y==0).sum():,}  GALAXY={(ero_y==1).sum():,})")

    # ── Standardise with Q1 stats, impute NaN → 0 ────────────────────────────
    std_safe = np.where(q1_std > 0, q1_std, 1.0)
    X_scaled = (ero_X_raw - q1_mean) / std_safe
    X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0.0)

    print("Projecting into Q1 UMAP space (this may take ~1 min)...")
    ero_emb = reducer.transform(X_scaled)
    print(f"  Done.  ERO embedding shape: {ero_emb.shape}")

    # ── Axis limits ───────────────────────────────────────────────────────────
    rng   = np.random.RandomState(0)
    x_all = np.concatenate([q1_emb[:, 0], ero_emb[:, 0]])
    y_all = np.concatenate([q1_emb[:, 1], ero_emb[:, 1]])
    x_lo  = x_all.min() - 1;  x_hi = x_all.max() + 1
    y_lo  = y_all.min() - 1;  y_hi = y_all.max() + 1

    n_bins  = 180
    x_edges = np.linspace(x_lo, x_hi, n_bins + 1)
    y_edges = np.linspace(y_lo, y_hi, n_bins + 1)

    cmap_dens    = slice_cmap(get_plot_cmap(CMAP_DENSITY, bad_alpha_zero=True), 0.3, 1.0)
    cmap_dens_q1 = slice_cmap(get_plot_cmap(CMAP_DENSITY, bad_alpha_zero=True), 0.3, 1.0)

    FS_TITLE = 9;  FS_LABEL = 8;  FS_TICK = 7
    MAX_PTS  = 15_000

    def _downsample(arr, mask):
        idx = np.where(mask)[0]
        if len(idx) > MAX_PTS:
            idx = rng.choice(idx, MAX_PTS, replace=False)
        return arr[idx]

    def _style_sc(ax, title):
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("UMAP 1", fontsize=FS_LABEL)
        ax.set_ylabel("UMAP 2", fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.2)
        h, _ = ax.get_legend_handles_labels()
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

    # ── Histograms ────────────────────────────────────────────────────────────
    def _hist(emb, mask):
        H, _, _ = np.histogram2d(emb[mask, 0], emb[mask, 1],
                                  bins=[x_edges, y_edges])
        return np.where(H > 0, H, np.nan)

    H_ero_all = _hist(ero_emb, np.ones(len(ero_y), dtype=bool))
    H_ero_str = _hist(ero_emb, ero_y == 0)
    H_ero_gal = _hist(ero_emb, ero_y == 1)

    ero_all_norm = LogNorm(vmin=np.nanmin(H_ero_all), vmax=np.nanmax(H_ero_all))

    finite_ero  = [h for h in [H_ero_all, H_ero_str, H_ero_gal]
                   if not np.all(np.isnan(h))]
    shared_norm = LogNorm(
        vmin=np.nanmin([np.nanmin(h) for h in finite_ero]),
        vmax=np.nanmax([np.nanmax(h) for h in finite_ero]),
    )

    # ── Figure layout: 2 rows × 4 cols ───────────────────────────────────────
    scatter_h = 0.5;  density_h = 0.5
    height_ratios = [scatter_h, density_h]
    panel_frac = scatter_h / sum(height_ratios)
    fig = plt.figure(figsize=(30.0, 3.0 / panel_frac))
    gs  = GridSpec(2, 4, figure=fig, height_ratios=height_ratios,
                   hspace=0.22, wspace=0.25, top=0.88)

    ax_q1_d   = fig.add_subplot(gs[0:2, 0])   # Q1 density (full height)
    ax_all_sc = fig.add_subplot(gs[0, 1])
    ax_str_sc = fig.add_subplot(gs[0, 2])
    ax_gal_sc = fig.add_subplot(gs[0, 3])
    ax_all_d  = fig.add_subplot(gs[1, 1])
    ax_str_d  = fig.add_subplot(gs[1, 2])
    ax_gal_d  = fig.add_subplot(gs[1, 3])

    # Col 0: All ERO projected density
    sc_q1 = ax_q1_d.pcolormesh(x_edges, y_edges, H_ero_all.T,
                                cmap=cmap_dens_q1, norm=ero_all_norm, rasterized=True, alpha=0.7)
    fig.colorbar(sc_q1, ax=ax_q1_d, label="Count (log)", pad=0.02, fraction=0.046)
    _style_sum(ax_q1_d, f"All ERO projected density  (n={len(ero_y):,})")

    # Q1 background helper
    def _bg(ax):
        pts = _downsample(q1_emb, np.ones(len(q1_y), dtype=bool))
        ax.scatter(pts[:, 0], pts[:, 1], c=Q1_BG, s=2, alpha=0.3,
                   linewidths=0, label=f"Q1 all (n={len(q1_y):,})", zorder=1)

    n_ero_str = (ero_y == 0).sum()
    n_ero_gal = (ero_y == 1).sum()

    # Col 1: ERO all
    _bg(ax_all_sc)
    pts = _downsample(ero_emb, np.ones(len(ero_y), dtype=bool))
    ax_all_sc.scatter(pts[:, 0], pts[:, 1], c=ERO_ALL_C, s=6, alpha=0.6,
                      linewidths=0, label=f"ERO all (n={len(ero_y):,})", zorder=3)
    _style_sc(ax_all_sc, f"ERO: all  (n={len(ero_y):,})")

    # Col 2: ERO stars
    _bg(ax_str_sc)
    pts = _downsample(ero_emb, ero_y == 0)
    ax_str_sc.scatter(pts[:, 0], pts[:, 1], c=ERO_STAR_C, s=6, alpha=0.5,
                      linewidths=0, label=f"ERO STAR (n={n_ero_str:,})", zorder=3)
    _style_sc(ax_str_sc, f"ERO: stars  (n={n_ero_str:,})")

    # Col 3: ERO galaxies
    _bg(ax_gal_sc)
    pts = _downsample(ero_emb, ero_y == 1)
    ax_gal_sc.scatter(pts[:, 0], pts[:, 1], c=ERO_GAL_C, s=6, alpha=0.5,
                      linewidths=0, label=f"ERO GALAXY (n={n_ero_gal:,})", zorder=3)
    _style_sc(ax_gal_sc, f"ERO: galaxies  (n={n_ero_gal:,})")

    # Density panels
    for ax_d, H in [
        (ax_all_d, H_ero_all),
        (ax_str_d, H_ero_str),
        (ax_gal_d, H_ero_gal),
    ]:
        sc2 = ax_d.pcolormesh(x_edges, y_edges, H.T,
                              cmap=cmap_dens, norm=shared_norm, rasterized=True, alpha=0.7)
        _style_dn(ax_d)
        fig.colorbar(sc2, ax=ax_d, label="Count (log)", pad=0.02, fraction=0.046)

    label = "Rebalanced" if "rebalanced" in npz_name else "Individual"
    fig.suptitle(
        f"ERO {label} sources projected into Q1 individual UMAP space\n"
        "(Q1 mean/std normalisation  |  Q1 reducer.transform())",
        fontsize=12, fontweight="bold", y=0.995,
    )

    out = OUTPUT_DIR / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> Saved: {out}")
    return True


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Q1 embedding + normalisation stats (shared across both runs) ─────
    print("Loading Q1 UMAP arrays and reducer...")
    q1      = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    q1_emb  = q1["embedding"]
    q1_y    = q1["y"]
    q1_mean = q1["mean"]
    q1_std  = q1["std"]

    reducer = joblib.load(ARRAYS_DIR / "q1_indiv_reducer.joblib")
    print(f"  Q1: {len(q1_y):,} sources  "
          f"(STAR={(q1_y==0).sum():,}  GALAXY={(q1_y==1).sum():,})")

    tasks = [
        ("ero_regular.npz",  "q1_space_ero.png"),
        ("ero_rebal.npz",   "q1_space_ero_rebalanced.png"),
    ]

    ok_any = False
    for npz_name, out_name in tasks:
        if _run_pipeline(npz_name, out_name, q1_emb, q1_y, q1_mean, q1_std, reducer):
            ok_any = True

    if not ok_any:
        print("ERROR: No figure written.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
