#!/usr/bin/env python3
"""
Two slide-style figures (saved separately):

  1. umap_concept_features.png — high-dim Gaia feature space: scattered,
     overlapping sources with decorative perimeter feature labels.
  2. umap_concept_embedded.png — UMAP 2D embedding: well-separated clusters
     with dashed circles labelled "Stars" and "Galaxies".

Sources are rendered with star/galaxy iconography from xgboost_diagram.py
(yellow ★ for stars, blue ellipses for galaxies).

Reads:
  output/umap/arrays/q1_indiv.npz  (real Q1 UMAP for the embedded panel)

Writes:
  docs/slides/figs/umap_concept_features.png
  docs/slides/figs/umap_concept_embedded.png

Usage:
  python src/slides/analyze_umap_concept.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Circle, FancyBboxPatch

PROJECT_DIR = Path(__file__).resolve().parents[2]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "docs" / "slides" / "figs"

# Marker colours — match xgboost_diagram.py ────────────────────────────────
STAR_FACE   = "#F4D03F"
STAR_EDGE   = "#D4AC0D"
GAL_FACE    = "#AED6F1"
GAL_INNER   = "#85C1E9"
GAL_EDGE    = "#2E86C1"


def draw_star(ax, x, y, size=10, alpha=1.0):
    ax.plot(x, y, marker="*", markersize=size,
            color=STAR_FACE, markeredgecolor=STAR_EDGE,
            markeredgewidth=0.6, zorder=5, alpha=alpha)


def _apply_rounded_frame(ax, face="#E8F0FE", edge="#A6BBC8",
                         linewidth=1.2, rounding=0.025):
    """Replace ax's default rectangular spines with a rounded-corner background."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("none")
    patch = FancyBboxPatch(
        (0.005, 0.005), 0.99, 0.99,
        transform=ax.transAxes,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        facecolor=face, edgecolor=edge, linewidth=linewidth,
        zorder=-10, clip_on=False)
    ax.add_patch(patch)


def draw_galaxy(ax, x, y, w=0.55, h=0.32, angle=None, alpha=1.0,
                rng=None):
    """Single galaxy ellipse + faint inner ellipse, matching xgboost_diagram."""
    if angle is None:
        angle = (rng.uniform(-60, 60) if rng is not None else 30)
    e = Ellipse((x, y), w, h, angle=angle,
                facecolor=GAL_FACE, edgecolor=GAL_EDGE, lw=0.8,
                zorder=5, alpha=alpha)
    ax.add_patch(e)
    e2 = Ellipse((x, y), w * 0.55, h * 0.45, angle=angle,
                 facecolor=GAL_INNER, edgecolor=GAL_EDGE, lw=0.4,
                 zorder=6, alpha=alpha)
    ax.add_patch(e2)


# ── Left panel: feature space (overlapping scatter, decorative labels) ────

FEATURE_LABELS = [
    ("bp_rp",                        (0.06, 0.95), "left",   "top"),
    ("phot_g",                       (0.94, 0.95), "right",  "top"),
    ("ruwe",                         (0.04, 0.50), "left",   "center"),
    ("phot_bp_mean_flux_over_error", (0.96, 0.50), "right",  "center"),
    ("astrometric_gof_al",           (0.04, 0.78), "left",   "center"),
    ("ipd_gof_harmonic_amplitude",   (0.96, 0.25), "right",  "center"),
    ("parallax_over_error",          (0.06, 0.05), "left",   "bottom"),
    ("pm_significance",              (0.94, 0.05), "right",  "bottom"),
]


def plot_feature_space(ax, n_star=70, n_gal=70, seed=1):
    rng = np.random.RandomState(seed)
    # Heavy overlap: both classes share roughly the same Gaussian blob,
    # with mild structure so the scatter doesn't look completely random.
    star_xy = rng.normal(loc=(0.0, 0.0), scale=(1.5, 1.5), size=(n_star, 2))
    gal_xy  = rng.normal(loc=(0.3, -0.2), scale=(1.6, 1.4), size=(n_gal, 2))

    for x, y in star_xy:
        draw_star(ax, x, y, size=10)
    for x, y in gal_xy:
        draw_galaxy(ax, x, y, w=0.45, h=0.26, rng=rng)

    ax.set_xlim(-4.6, 4.6)
    ax.set_ylim(-4.6, 4.6)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    _apply_rounded_frame(ax)

    # Decorative perimeter feature labels
    for txt, (fx, fy), ha, va in FEATURE_LABELS:
        ax.text(fx, fy, txt, transform=ax.transAxes,
                fontsize=9, color="#5D6D7E", style="italic",
                ha=ha, va=va, zorder=2)


# ── Right panel: UMAP space with two clean clusters ───────────────────────

def plot_umap_space(ax, n_star=60, n_gal=60, seed=2):
    """Draw two diagonally-placed clusters in a square frame, styled to
    match the feature-space panel (same background + perimeter labels)."""
    rng = np.random.RandomState(seed)
    # Synthetic clusters: stars upper-left, galaxies lower-right
    star_xy = rng.normal(loc=(-2.2, +2.2), scale=(0.95, 0.85), size=(n_star, 2))
    gal_xy  = rng.normal(loc=(+2.2, -2.2), scale=(1.10, 0.85), size=(n_gal, 2))

    for x, y_ in star_xy:
        draw_star(ax, x, y_, size=10)
    for x, y_ in gal_xy:
        draw_galaxy(ax, x, y_, w=0.55, h=0.30, rng=rng)

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    _apply_rounded_frame(ax)

    # Dashed circles around each cluster
    for xy, color, label, label_offset in [
        (star_xy, "#5D6D7E", "Stars",    (0.0, 1.15)),
        (gal_xy,  "#5D6D7E", "Galaxies", (0.0, -1.15)),
    ]:
        cx, cy = xy.mean(axis=0)
        rx = 1.2 * np.std(xy[:, 0]) + 0.5
        ry = 1.2 * np.std(xy[:, 1]) + 0.5
        r  = max(rx, ry)
        c = Circle((cx, cy), r, facecolor="none",
                   edgecolor=color, linestyle=(0, (4, 3)),
                   linewidth=1.4, zorder=3, alpha=0.7)
        ax.add_patch(c)
        ax.text(cx + label_offset[0], cy + label_offset[1] * r,
                label, fontsize=12, fontweight="bold",
                color="#34495E", ha="center",
                va="bottom" if label_offset[1] > 0 else "top",
                zorder=6)



# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Panel 1: feature space
    fig1, ax1 = plt.subplots(figsize=(6.0, 5.4))
    plot_feature_space(ax1)
    out1 = OUTPUT_DIR / "umap_concept_features.png"
    fig1.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved -> {out1}")

    # Panel 2: UMAP embedding
    fig2, ax2 = plt.subplots(figsize=(6.0, 5.4))
    plot_umap_space(ax2)
    out2 = OUTPUT_DIR / "umap_concept_embedded.png"
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved -> {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
