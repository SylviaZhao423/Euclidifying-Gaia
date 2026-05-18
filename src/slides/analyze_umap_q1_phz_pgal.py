#!/usr/bin/env python3
"""
Q1 UMAP slide figures:
  1. Single-panel UMAP coloured by PHZ P(galaxy) — right panel of
     q1_cluster_phz_combined.png, extracted as a standalone slide figure.
  2. Single-panel UMAP coloured by PHZ class label (STAR vs GALAXY).

Reads:
  output/umap/arrays/q1_indiv.npz   (needs 'embedding', 'phz_gal_prob', 'y')

Writes:
  docs/slides/figs/q1_umap_phz_pgal.png
  docs/slides/figs/q1_umap_phz_label.png

Usage:
  python src/slides/analyze_umap_q1_phz_pgal.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "docs" / "slides" / "figs"

CLASS_COLORS = {0: "#f48db0", 1: "deepskyblue"}  # STAR red, GALAXY blue
CLASS_NAMES  = {0: "STAR", 1: "GALAXY"}


def plot_phz_pgal(emb: np.ndarray, pgal: np.ndarray, out_path: Path) -> None:
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
    cb.set_label("PHZ P(galaxy)", fontsize=12, fontweight="bold")
    cb.ax.tick_params(labelsize=11)

    ax.set_xlabel("UMAP 1", fontsize=13, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=13, fontweight="bold")
    ax.set_title("PHZ P(galaxy)", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=12)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_labels(emb: np.ndarray, y: np.ndarray, out_path: Path,
                title: str = "PHZ class label") -> None:
    """Scatter coloured by binary STAR/GALAXY label."""
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
    ax.legend(handles=handles, fontsize=11, loc="lower left",
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


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    npz_path = ARRAYS_DIR / "q1_indiv.npz"
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found.")
        return 1

    print(f"[1] Loading {npz_path.name}...")
    npz = np.load(npz_path, allow_pickle=True)
    emb = npz["embedding"]
    if "phz_gal_prob" not in npz:
        print("ERROR: 'phz_gal_prob' missing from q1_indiv.npz.")
        return 1
    if "y" not in npz:
        print("ERROR: 'y' missing from q1_indiv.npz.")
        return 1
    pgal = npz["phz_gal_prob"]
    y = np.asarray(npz["y"]).astype(int)
    print(f"  {emb.shape[0]:,} points  "
          f"(STAR: {(y==0).sum():,}, GALAXY: {(y==1).sum():,})")

    print("\n[2] Plotting P(galaxy)...")
    plot_phz_pgal(emb, pgal, OUTPUT_DIR / "q1_umap_phz_pgal.png")

    print("\n[3] Plotting STAR/GALAXY labels...")
    plot_labels(emb, y, OUTPUT_DIR / "q1_umap_phz_label.png",
                title="Q1 Dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
