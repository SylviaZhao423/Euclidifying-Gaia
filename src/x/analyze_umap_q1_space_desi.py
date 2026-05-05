#!/usr/bin/env python3
"""
Project DESI-confirmed galaxies (Q1 cross-match and ERO cross-match) into
the Q1 individual UMAP space.

Layout: 1 row × 3 cols
  Col 0: Q1 source density (background reference)
  Col 1: Q1  × DESI galaxies scatter overlay
  Col 2: ERO × DESI galaxies scatter overlay

Reads:
  output/umap/arrays/q1_indiv.npz
  output/umap/arrays/q1_indiv_reducer.joblib
  output/umap/arrays/q1_allphz_desi.npz
  output/umap/arrays/ero_desi.npz

Writes:
  output/umap/q1_individual/q1_space_desi.png

Usage:
  python src/analyze_umap_q1_desi_overlay.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.gridspec import GridSpec

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "q1_individual"

CMAP_DENSITY = "CET_L19"

Q1_BG       = "lightgrey"
Q1_DESI_C   = "#A5D6A7"   # light green — Q1 × DESI galaxies
ERO_DESI_C  = "#CE93D8"   # light purple — ERO × DESI galaxies


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


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Q1 embedding + normalisation stats ───────────────────────────────
    print("Loading Q1 UMAP arrays and reducer...")
    q1      = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    q1_emb  = q1["embedding"]
    q1_y    = q1["y"]
    q1_mean = q1["mean"]
    q1_std  = q1["std"]
    reducer = joblib.load(ARRAYS_DIR / "q1_indiv_reducer.joblib")
    print(f"  Q1: {len(q1_y):,} sources  "
          f"(STAR={(q1_y==0).sum():,}  GALAXY={(q1_y==1).sum():,})")

    std_safe = np.where(q1_std > 0, q1_std, 1.0)

    def _project(X_raw: np.ndarray, label: str) -> np.ndarray:
        X_scaled = (X_raw - q1_mean) / std_safe
        X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0.0)
        print(f"Projecting {label} into Q1 UMAP space...")
        emb = reducer.transform(X_scaled)
        print(f"  Done.  shape: {emb.shape}")
        return emb

    # ── Load DESI arrays ──────────────────────────────────────────────────────
    q1_desi_path  = ARRAYS_DIR / "q1_allphz_desi.npz"
    ero_desi_path = ARRAYS_DIR / "ero_desi.npz"
    for p in [q1_desi_path, ero_desi_path]:
        if not p.exists():
            print(f"ERROR: {p} not found.")
            return 1

    q1_desi_all = np.load(q1_desi_path, allow_pickle=True)
    q1_desi_mask = np.isin(q1_desi_all["phz"].astype(int), [1, 2])
    q1_desi_X   = q1_desi_all["X_features"].astype(float)[q1_desi_mask]
    ero_desi    = np.load(ero_desi_path, allow_pickle=True)
    ero_desi_X  = ero_desi["X_features"].astype(float)

    n_q1_desi  = len(q1_desi_X)
    n_ero_desi = len(ero_desi_X)
    print(f"  Q1  × DESI galaxies: {n_q1_desi:,}")
    print(f"  ERO × DESI galaxies: {n_ero_desi:,}")

    q1_desi_emb  = _project(q1_desi_X,  f"Q1×DESI ({n_q1_desi:,})")
    ero_desi_emb = _project(ero_desi_X, f"ERO×DESI ({n_ero_desi:,})")

    # ── Axis limits ───────────────────────────────────────────────────────────
    rng   = np.random.RandomState(0)
    x_all = np.concatenate([q1_emb[:, 0], q1_desi_emb[:, 0], ero_desi_emb[:, 0]])
    y_all = np.concatenate([q1_emb[:, 1], q1_desi_emb[:, 1], ero_desi_emb[:, 1]])
    x_lo  = x_all.min() - 1;  x_hi = x_all.max() + 1
    y_lo  = y_all.min() - 1;  y_hi = y_all.max() + 1

    n_bins  = 180
    x_edges = np.linspace(x_lo, x_hi, n_bins + 1)
    y_edges = np.linspace(y_lo, y_hi, n_bins + 1)

    cmap_dens = slice_cmap(get_plot_cmap(CMAP_DENSITY, bad_alpha_zero=True), 0.3, 1.0)

    FS_TITLE = 9;  FS_LABEL = 8;  FS_TICK = 7
    MAX_PTS  = 15_000

    def _downsample(arr, n=MAX_PTS):
        if len(arr) > n:
            idx = rng.choice(len(arr), n, replace=False)
            return arr[idx]
        return arr

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

    def _style_dn(ax):
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(alpha=0.2)

    # ── Histograms ────────────────────────────────────────────────────────────
    def _hist(emb):
        H, _, _ = np.histogram2d(emb[:, 0], emb[:, 1], bins=[x_edges, y_edges])
        return np.where(H > 0, H, np.nan)

    H_q1      = _hist(q1_emb)
    H_q1_desi = _hist(q1_desi_emb)
    H_ero_desi = _hist(ero_desi_emb)

    q1_norm = LogNorm(vmin=np.nanmin(H_q1), vmax=np.nanmax(H_q1))

    finite_desi = [h for h in [H_q1_desi, H_ero_desi] if not np.all(np.isnan(h))]
    shared_norm = LogNorm(
        vmin=np.nanmin([np.nanmin(h) for h in finite_desi]),
        vmax=np.nanmax([np.nanmax(h) for h in finite_desi]),
    )

    # ── Background helper ─────────────────────────────────────────────────────
    def _bg(ax):
        pts = _downsample(q1_emb)
        ax.scatter(pts[:, 0], pts[:, 1], c=Q1_BG, s=2, alpha=0.3,
                   linewidths=0, label=f"Q1 all (n={len(q1_y):,})", zorder=1)

    # ── Figure: 2 rows × 3 cols ───────────────────────────────────────────────
    scatter_h = 0.5;  density_h = 0.5
    height_ratios = [scatter_h, density_h]
    panel_frac = scatter_h / sum(height_ratios)
    fig = plt.figure(figsize=(22.5, 3.0 / panel_frac))
    gs  = GridSpec(2, 3, figure=fig, height_ratios=height_ratios,
                   hspace=0.22, wspace=0.25, top=0.88)

    ax_q1_d    = fig.add_subplot(gs[0:2, 0])   # Q1 density (full height)
    ax_q1d_sc  = fig.add_subplot(gs[0, 1])
    ax_erod_sc = fig.add_subplot(gs[0, 2])
    ax_q1d_dn  = fig.add_subplot(gs[1, 1])
    ax_erod_dn = fig.add_subplot(gs[1, 2])

    # Col 0: Q1 source density
    sc = ax_q1_d.pcolormesh(x_edges, y_edges, H_q1.T,
                             cmap=cmap_dens, norm=q1_norm, rasterized=True)
    fig.colorbar(sc, ax=ax_q1_d, label="Count (log)", pad=0.02, fraction=0.046)
    ax_q1_d.set_xlim(x_lo, x_hi); ax_q1_d.set_ylim(y_lo, y_hi)
    ax_q1_d.set_xlabel("UMAP 1", fontsize=FS_LABEL)
    ax_q1_d.set_ylabel("UMAP 2", fontsize=FS_LABEL)
    ax_q1_d.set_title(f"Q1: source density  (n={len(q1_y):,})",
                      fontsize=FS_TITLE, fontweight="bold")
    ax_q1_d.tick_params(labelsize=FS_TICK)
    ax_q1_d.grid(alpha=0.2)

    # Col 1: Q1 × DESI galaxies — scatter
    _bg(ax_q1d_sc)
    pts = _downsample(q1_desi_emb)
    ax_q1d_sc.scatter(pts[:, 0], pts[:, 1], c=Q1_DESI_C, s=6, alpha=0.6,
                      linewidths=0, label=f"Q1×DESI galaxies (n={n_q1_desi:,})", zorder=3)
    _style_sc(ax_q1d_sc, f"Q1 × DESI galaxies  (n={n_q1_desi:,})")

    # Col 2: ERO × DESI galaxies — scatter
    _bg(ax_erod_sc)
    pts = _downsample(ero_desi_emb)
    ax_erod_sc.scatter(pts[:, 0], pts[:, 1], c=ERO_DESI_C, s=6, alpha=0.6,
                       linewidths=0, label=f"ERO×DESI galaxies (n={n_ero_desi:,})", zorder=3)
    _style_sc(ax_erod_sc, f"ERO × DESI galaxies  (n={n_ero_desi:,})")

    # Density panels
    for ax_d, H in [(ax_q1d_dn, H_q1_desi), (ax_erod_dn, H_ero_desi)]:
        sc2 = ax_d.pcolormesh(x_edges, y_edges, H.T,
                              cmap=cmap_dens, norm=shared_norm, rasterized=True)
        _style_dn(ax_d)
        fig.colorbar(sc2, ax=ax_d, label="Count (log)", pad=0.02, fraction=0.046)

    fig.suptitle(
        "DESI-confirmed galaxies projected into Q1 individual UMAP space\n"
        "(Q1 mean/std normalisation  |  Q1 reducer.transform())",
        fontsize=12, fontweight="bold", y=0.995,
    )

    out = OUTPUT_DIR / "q1_space_desi.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
