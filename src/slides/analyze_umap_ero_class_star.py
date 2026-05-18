#!/usr/bin/env python3
"""
ERO UMAP slide figures (regular + Q1-ratio-rebalanced, in one run):
  1. UMAP coloured by 1 - CLASS_STAR (ERO's analog of PHZ P(galaxy)).
  2. UMAP coloured by binary STAR/GALAXY label (CLASS_STAR >= 0.65 → STAR).

Mirrors src/slides/analyze_umap_q1_phz_pgal.py but for the ERO base UMAP
produced by query_umap_ero_50k.py. Whichever of {ero_indiv.npz,
ero_indiv_rebal.npz} are present get plotted.

Reads:
  output/umap/ero_individual/arrays/ero_indiv.npz
  output/umap/ero_individual/arrays/ero_indiv_rebal.npz

Writes (per available npz):
  docs/slides/figs/ero_umap_class_star.png             (P(gal), regular)
  docs/slides/figs/ero_umap_class_star_label.png       (label, regular)
  docs/slides/figs/ero_umap_class_star_rebal.png       (P(gal), rebalanced)
  docs/slides/figs/ero_umap_class_star_label_rebal.png (label, rebalanced)

Usage:
  python src/slides/analyze_umap_ero_class_star.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "ero_individual" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "docs" / "slides" / "figs"

CLASS_COLORS = {0: "#f48db0", 1: "deepskyblue"}  # STAR red, GALAXY blue
CLASS_NAMES  = {0: "STAR", 1: "GALAXY"}


def plot_class_star(emb: np.ndarray, pgal: np.ndarray, out_path: Path,
                    title: str = "ERO 1 − CLASS_STAR") -> None:
    x_pad = 1
    xlim = (emb[:, 0].min() - x_pad, emb[:, 0].max() + x_pad)
    ylim = (emb[:, 1].min() - x_pad, emb[:, 1].max() + x_pad)

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(emb))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(emb[idx, 0], emb[idx, 1],
                    c=pgal[idx], cmap="RdYlBu_r",
                    vmin=0, vmax=1, s=0.6, alpha=0.6,
                    linewidths=0, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("1 − CLASS_STAR  (ERO P(galaxy))",
                 fontsize=12, fontweight="bold")
    cb.ax.tick_params(labelsize=11)

    ax.set_xlabel("UMAP 1", fontsize=13, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=12)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_labels(emb: np.ndarray, y: np.ndarray, out_path: Path,
                title: str = "CLASS_STAR label") -> None:
    """Scatter coloured by binary STAR/GALAXY label (from CLASS_STAR >= 0.65)."""
    from matplotlib.lines import Line2D

    x_pad = 1
    xlim = (emb[:, 0].min() - x_pad, emb[:, 0].max() + x_pad)
    ylim = (emb[:, 1].min() - x_pad, emb[:, 1].max() + x_pad)

    rng = np.random.RandomState(42)
    order = rng.permutation(len(emb))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    handles = []
    for cls_val in (0, 1):
        mask = y[order] == cls_val
        if not mask.any():
            continue
        ax.scatter(emb[order[mask], 0], emb[order[mask], 1],
                   c=CLASS_COLORS[cls_val], s=0.6, alpha=1,
                   linewidths=0, rasterized=True)
        n = int(mask.sum())
        handles.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=CLASS_COLORS[cls_val], markersize=8,
                   label=f"{CLASS_NAMES[cls_val]}: {n:,}"))
    ax.legend(handles=handles, fontsize=11, loc="upper right",
              framealpha=0.85, markerscale=1)

    ax.set_xlabel("UMAP 1", fontsize=13, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=12)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def _process_variant(suffix: str) -> bool:
    """Plot pgal + label figures for one variant. Returns True if processed."""
    npz_path = ARRAYS_DIR / f"ero_indiv{suffix}.npz"
    if not npz_path.exists():
        print(f"  [skip] {npz_path.name} not found.")
        return False

    out_pgal  = OUTPUT_DIR / f"ero_umap_class_star{suffix}.png"
    out_label = OUTPUT_DIR / f"ero_umap_class_star_label{suffix}.png"
    rebal_tag = " (rebalanced)" if suffix else ""
    title_pgal  = f"ERO UMAP — 1 − CLASS_STAR{rebal_tag}"
    title_label = f"ERO Dataset{rebal_tag}"

    print(f"\nLoading {npz_path.name}...")
    npz = np.load(npz_path, allow_pickle=True)
    emb = npz["embedding"]
    if "class_star" not in npz or "y" not in npz:
        print("  ERROR: 'class_star' or 'y' missing from npz.")
        return False
    cs = np.asarray(npz["class_star"], dtype=float)
    pgal = np.clip(1.0 - cs, 0.0, 1.0)
    y = np.asarray(npz["y"]).astype(int)
    print(f"  {emb.shape[0]:,} points  "
          f"(STAR: {(y==0).sum():,}, GALAXY: {(y==1).sum():,})  "
          f"P(gal) range [{np.nanmin(pgal):.3f}, {np.nanmax(pgal):.3f}]")

    print("  Plotting 1 − CLASS_STAR...")
    plot_class_star(emb, pgal, out_pgal, title=title_pgal)
    print("  Plotting STAR/GALAXY labels...")
    plot_labels(emb, y, out_label, title=title_label)
    return True


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    any_done = False
    for suffix in ("", "_rebal"):
        if _process_variant(suffix):
            any_done = True

    if not any_done:
        print("\nERROR: no ERO npz files found in", ARRAYS_DIR)
        print("  Run:  PGUSER=<user> python src/slides/query_umap_ero_50k.py")
        print("  And:  PGUSER=<user> python src/slides/query_umap_ero_50k.py --rebalance")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
