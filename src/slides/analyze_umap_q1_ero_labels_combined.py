#!/usr/bin/env python3
"""
Side-by-side Q1 + ERO UMAP slide figure, both coloured by binary STAR/GALAXY
label (left = Q1 PHZ classification, right = ERO CLASS_STAR threshold).

Combines q1_umap_phz_label.png and ero_umap_class_star_label.png into one
two-panel figure with shared styling. Produces both regular and Q1-ratio-
rebalanced ERO variants in one run (whichever ERO npz files are available).

Reads:
  output/umap/arrays/q1_indiv.npz                       (Q1, left panel)
  output/umap/ero_individual/arrays/ero_indiv.npz       (ERO regular)
  output/umap/ero_individual/arrays/ero_indiv_rebal.npz (ERO rebalanced)

Writes (per available ERO npz):
  docs/slides/figs/q1_ero_umap_labels.png         (regular)
  docs/slides/figs/q1_ero_umap_labels_rebal.png   (rebalanced)

Usage:
  python src/slides/analyze_umap_q1_ero_labels_combined.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJECT_DIR = Path(__file__).resolve().parents[2]
Q1_ARRAYS   = PROJECT_DIR / "output" / "umap" / "arrays"
ERO_ARRAYS  = PROJECT_DIR / "output" / "umap" / "ero_individual" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "docs" / "slides" / "figs"

CLASS_COLORS = {0: "#f4df3f", 1: "deepskyblue"}  # STAR pink, GALAXY blue
CLASS_NAMES  = {0: "STAR", 1: "GALAXY"}


def _load_emb_y(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not npz_path.exists():
        raise SystemExit(f"ERROR: {npz_path} not found.")
    npz = np.load(npz_path, allow_pickle=True)
    if "embedding" not in npz or "y" not in npz:
        raise SystemExit(f"ERROR: 'embedding' or 'y' missing from {npz_path.name}.")
    return npz["embedding"], np.asarray(npz["y"]).astype(int)


def _scatter_labels(ax, emb: np.ndarray, y: np.ndarray,
                    xlim: tuple, ylim: tuple, title: str,
                    legend_loc: str = "lower left") -> None:
    rng = np.random.RandomState(42)
    order = rng.permutation(len(emb))

    handles = []
    for cls_val in (0, 1):
        mask = y[order] == cls_val
        if not mask.any():
            continue
        ax.scatter(emb[order[mask], 0], emb[order[mask], 1],
                   c=CLASS_COLORS[cls_val], s=1.2, alpha=1,
                   linewidths=0, rasterized=True)
        n = int(mask.sum())
        handles.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=CLASS_COLORS[cls_val], markersize=8,
                   label=f"{CLASS_NAMES[cls_val]}: {n:,}"))
    ax.legend(handles=handles, fontsize=11, loc=legend_loc,
              framealpha=0.85, markerscale=1)

    ax.set_xlabel("UMAP 1", fontsize=13, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=12)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(alpha=0.2)


def _make_combined(q1_emb, q1_y, ero_npz: Path, suffix: str) -> bool:
    """Build the side-by-side figure for one ERO variant. Returns True if made."""
    if not ero_npz.exists():
        print(f"  [skip] {ero_npz.name} not found.")
        return False

    print(f"\nLoading ERO: {ero_npz.name}")
    ero_emb, ero_y = _load_emb_y(ero_npz)
    print(f"  {ero_emb.shape[0]:,} points  "
          f"(STAR: {(ero_y==0).sum():,}, GALAXY: {(ero_y==1).sum():,})")

    pad = 1
    q1_xlim  = (q1_emb[:, 0].min() - pad, q1_emb[:, 0].max() + pad)
    q1_ylim  = (q1_emb[:, 1].min() - pad, q1_emb[:, 1].max() + pad)
    ero_xlim = (ero_emb[:, 0].min() - pad, ero_emb[:, 0].max() + pad)
    ero_ylim = (ero_emb[:, 1].min() - pad, ero_emb[:, 1].max() + pad)
    ero_title = "ERO Dataset" + (" (rebalanced)" if suffix else "")
    out_path = OUTPUT_DIR / f"q1_ero_umap_labels{suffix}.png"

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.subplots_adjust(wspace=0.3)
    _scatter_labels(axes[0], q1_emb,  q1_y,  q1_xlim,  q1_ylim,
                    title="Q1 Dataset", legend_loc="lower left")
    _scatter_labels(axes[1], ero_emb, ero_y, ero_xlim, ero_ylim,
                    title=ero_title, legend_loc="upper right")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")
    return True


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    q1_npz = Q1_ARRAYS / "q1_indiv.npz"
    print(f"Loading Q1: {q1_npz.name}")
    q1_emb, q1_y = _load_emb_y(q1_npz)
    print(f"  {q1_emb.shape[0]:,} points  "
          f"(STAR: {(q1_y==0).sum():,}, GALAXY: {(q1_y==1).sum():,})")

    any_done = False
    for suffix in ("", "_rebal"):
        if _make_combined(q1_emb, q1_y,
                          ERO_ARRAYS / f"ero_indiv{suffix}.npz", suffix):
            any_done = True

    if not any_done:
        print("\nERROR: no ERO npz files found in", ERO_ARRAYS)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
