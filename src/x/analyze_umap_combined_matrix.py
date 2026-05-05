#!/usr/bin/env python3
"""
5-col × 2-row matrix of combined UMAP panels.

Each row is one embedding (regular / rebalanced).
Cols per row:
  0: ERO stars in combined    (scatter + density)
  1: ERO galaxies in combined (scatter + density)
  2: combined source density  (full-height histogram, no scatter)
  3: Q1 stars in combined     (scatter + density)
  4: Q1 galaxies in combined  (scatter + density)

Colours (from combined_compare.png row 1):
  ERO stars    = lightpink    (square marker)
  ERO galaxies = deepskyblue  (square marker)
  Q1  stars    = coral        (circle marker)
  Q1  galaxies = #1976D2      (circle marker)

Reads:
  output/umap/arrays/combined.npz
  output/umap/arrays/combined_rebalanced.npz

Writes:
  output/umap/combined/combined_matrix.png

Usage:
  python src/analyze_umap_combined_matrix.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.gridspec import GridSpec

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "combined"

CMAP_DENSITY = "CET_L19"
MAX_PTS      = 15_000

# categories: 0=Q1-STAR, 1=Q1-GALAXY, 2=ERO-STAR, 3=ERO-GALAXY
ERO_STAR_C   = "lightpink"
ERO_GAL_C    = "deepskyblue"
Q1_STAR_C    = "coral"
Q1_GAL_C     = "#1976D2"


def _get_cmap(name: str, *, bad_alpha_zero: bool = False):
    normalized = name.replace("-", "_")
    try:
        cmap = plt.get_cmap(normalized).copy()
        if bad_alpha_zero:
            cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
        return cmap
    except ValueError:
        pass
    import colorcet as cc
    key = normalized
    if hasattr(cc, f"m_{key}"):
        cmap = getattr(cc, f"m_{key}").copy()
    elif key in cc.cm:
        cmap = cc.cm[key].copy()
    else:
        tail = key.replace("CET_", "", 1).lower() if key.startswith("CET_") else key.lower()
        cmap = cc.cm[tail].copy()
    if bad_alpha_zero:
        cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    return cmap


def _slice_cmap(cmap, minval=0.0, maxval=1.0, n=256):
    if minval <= 0.0 and maxval >= 1.0:
        return cmap.copy()
    return LinearSegmentedColormap.from_list(
        f"slice_{cmap.name}_{minval:.2f}_{maxval:.2f}",
        cmap(np.linspace(minval, maxval, n)),
    )


def _downsample(pts, rng, n=MAX_PTS):
    if len(pts) > n:
        return pts[rng.choice(len(pts), n, replace=False)]
    return pts


def _hist(pts, x_edges, y_edges):
    if len(pts) == 0:
        return np.full((len(x_edges) - 1, len(y_edges) - 1), np.nan)
    H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[x_edges, y_edges])
    return np.where(H > 0, H, np.nan)


def _shared_norm(*arrays):
    finite = [h for h in arrays if not np.all(np.isnan(h))]
    if not finite:
        return LogNorm(vmin=1, vmax=10)
    return LogNorm(
        vmin=np.nanmin([np.nanmin(h) for h in finite]),
        vmax=np.nanmax([np.nanmax(h) for h in finite]),
    )


def _draw_row(fig, gs, gs_sum_rows, gs_sc_row, gs_dn_row,
              emb, cat, x_edges, y_edges, cmap_dens, rng,
              x_lo, x_hi, y_lo, y_hi, FS_TITLE, FS_LABEL, FS_TICK,
              *, combined_density_label: str):
    """Draw one main row (7 panels) into the given GridSpec sub-rows."""

    m_ero      = cat >= 2
    m_q1       = cat < 2
    m_ero_star = cat == 2
    m_ero_gal  = cat == 3
    m_q1_star  = cat == 0
    m_q1_gal   = cat == 1
    n_total    = len(emb)

    def _style_sc(ax, title):
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("UMAP 1", fontsize=FS_LABEL)
        ax.set_ylabel("UMAP 2", fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=5, loc="upper right", markerscale=1.5)

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

    def _bg(ax, bg_emb, n_bg, label):
        pts = _downsample(bg_emb, rng)
        ax.scatter(pts[:, 0], pts[:, 1], c="lightgrey", s=2, alpha=0.3,
                   linewidths=0, label=f"{label} (n={n_bg:,})", zorder=1)

    # ── histograms ────────────────────────────────────────────────────────────
    H_all      = _hist(emb,           x_edges, y_edges)
    H_ero      = _hist(emb[m_ero],    x_edges, y_edges)
    H_ero_star = _hist(emb[m_ero_star], x_edges, y_edges)
    H_ero_gal  = _hist(emb[m_ero_gal],  x_edges, y_edges)
    H_q1       = _hist(emb[m_q1],    x_edges, y_edges)
    H_q1_star  = _hist(emb[m_q1_star],  x_edges, y_edges)
    H_q1_gal   = _hist(emb[m_q1_gal],   x_edges, y_edges)

    norm_sum = _shared_norm(H_all)
    norm_cat = _shared_norm(H_ero_star, H_ero_gal, H_q1_star, H_q1_gal)

    # ── col 2: summary density (full-height) ──────────────────────────────────
    ax_sum = fig.add_subplot(gs[gs_sum_rows, 2])
    sc = ax_sum.pcolormesh(x_edges, y_edges, H_all.T,
                           cmap=cmap_dens, norm=norm_sum, rasterized=True)
    fig.colorbar(sc, ax=ax_sum, label="Count (log)", pad=0.02, fraction=0.046)
    _style_sum(ax_sum, f"Combined ({combined_density_label})  (n={n_total:,})")

    # ── cols 0–1 and 3–4: scatter + density ──────────────────────────────────
    panel_specs = [
        # (col, label,            fg_mask,    fg_color,   fg_marker, bg_mask,     bg_label, H_dn)
        (0, "Q1 Stars",    m_q1_star,  Q1_STAR_C,  "o",       ~m_q1_star,  "other",  H_q1_star),
        (1, "Q1 Galaxies", m_q1_gal,   Q1_GAL_C,   "o",       ~m_q1_gal,   "other",  H_q1_gal),
        (3, "ERO Stars",   m_ero_star, ERO_STAR_C, "s",       ~m_ero_star, "other",  H_ero_star),
        (4, "ERO Galaxies",m_ero_gal,  ERO_GAL_C,  "s",       ~m_ero_gal,  "other",  H_ero_gal),
    ]

    for col, panel_label, fg_mask, fg_color, fg_marker, bg_mask, bg_label, H_dn in panel_specs:
        ax_sc = fig.add_subplot(gs[gs_sc_row, col])
        ax_dn = fig.add_subplot(gs[gs_dn_row, col])

        # scatter — alpha=0.25 matches combined_compare.png; large counts use 0.15
        n_fg = fg_mask.sum()
        alpha = 0.15 if n_fg > 10_000 else 0.25
        _bg(ax_sc, emb[bg_mask], bg_mask.sum(), bg_label)
        pts = _downsample(emb[fg_mask], rng)
        ax_sc.scatter(pts[:, 0], pts[:, 1], c=fg_color, s=6, alpha=alpha,
                      linewidths=0, marker=fg_marker,
                      label=f"{panel_label} (n={n_fg:,})", zorder=3)
        _style_sc(ax_sc, f"{panel_label}  (n={n_fg:,})")

        # density
        sc_dn = ax_dn.pcolormesh(x_edges, y_edges, H_dn.T,
                                 cmap=cmap_dens, norm=norm_cat, rasterized=True)
        _style_dn(ax_dn)
        fig.colorbar(sc_dn, ax=ax_dn, label="Count (log)", pad=0.02, fraction=0.046)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for p in [ARRAYS_DIR / "combined.npz", ARRAYS_DIR / "combined_rebalanced.npz"]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run analyze_umap_query.py first.")
            return 1

    print("Loading arrays...")
    comb   = np.load(ARRAYS_DIR / "combined.npz",            allow_pickle=True)
    comb_r = np.load(ARRAYS_DIR / "combined_rebalanced.npz", allow_pickle=True)

    emb_c  = comb["embedding"];   cat_c  = comb["categories"]
    emb_r  = comb_r["embedding"]; cat_r  = comb_r["categories"]

    for tag, emb, cat in [("regular",   emb_c, cat_c), ("rebalanced", emb_r, cat_r)]:
        print(f"  {tag}: {len(emb):,}  Q1-STAR={(cat==0).sum():,}  Q1-GAL={(cat==1).sum():,}  "
              f"ERO-STAR={(cat==2).sum():,}  ERO-GAL={(cat==3).sum():,}")

    # ── global axis limits ────────────────────────────────────────────────────
    all_pts = np.vstack([emb_c, emb_r])
    x_lo = all_pts[:, 0].min() - 1;  x_hi = all_pts[:, 0].max() + 1
    y_lo = all_pts[:, 1].min() - 1;  y_hi = all_pts[:, 1].max() + 1

    n_bins  = 180
    x_edges = np.linspace(x_lo, x_hi, n_bins + 1)
    y_edges = np.linspace(y_lo, y_hi, n_bins + 1)

    cmap_dens = _slice_cmap(_get_cmap(CMAP_DENSITY, bad_alpha_zero=True), 0.3, 1.0)
    rng = np.random.RandomState(0)

    FS_TITLE = 9;  FS_LABEL = 8;  FS_TICK = 7

    # ── GridSpec: 5 sub-rows × 7 cols ────────────────────────────────────────
    scatter_h = 0.5;  density_h = 0.5;  outer_gap = 0.02
    height_ratios = [scatter_h, density_h, outer_gap, scatter_h, density_h]
    panel_frac = scatter_h / sum(height_ratios)
    fig = plt.figure(figsize=(37.5, 3.0 / panel_frac))
    gs  = GridSpec(5, 5, figure=fig, height_ratios=height_ratios,
                   hspace=0.22, wspace=0.25, top=0.93)

    kw = dict(x_edges=x_edges, y_edges=y_edges, cmap_dens=cmap_dens, rng=rng,
              x_lo=x_lo, x_hi=x_hi, y_lo=y_lo, y_hi=y_hi,
              FS_TITLE=FS_TITLE, FS_LABEL=FS_LABEL, FS_TICK=FS_TICK)

    print("Drawing row 0 (regular)...")
    _draw_row(fig, gs,
              gs_sum_rows=slice(0, 2), gs_sc_row=0, gs_dn_row=1,
              emb=emb_c, cat=cat_c, combined_density_label="regular", **kw)

    print("Drawing row 1 (rebalanced)...")
    _draw_row(fig, gs,
              gs_sum_rows=slice(3, 5), gs_sc_row=3, gs_dn_row=4,
              emb=emb_r, cat=cat_r, combined_density_label="rebalanced", **kw)

    fig.suptitle(
        "Combined UMAP matrix  |  "
        "Cols: Q1 Stars · Q1 Galaxies · Combined density · ERO Stars · ERO Galaxies\n"
        "Row 0: regular embedding   |   Row 1: ERO-rebalanced embedding",
        fontsize=13, fontweight="bold", y=0.995,
    )

    out = OUTPUT_DIR / "combined_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
