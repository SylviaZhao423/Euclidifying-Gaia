#!/usr/bin/env python3
"""
DESI-confirmed galaxies projected into Q1 UMAP space — insert layout.

Left panel (full height): Q1 × DESI galaxies scatter
Right column (stacked):   Q1 × DESI density (top) + ERO × DESI density (bottom)

Reads:
  output/umap/arrays/q1_indiv.npz
  output/umap/arrays/q1_indiv_reducer.joblib
  output/umap/arrays/q1_allphz_desi.npz
  output/umap/arrays/ero_desi.npz

Writes:
  output/umap/q1_individual/insert/q1_space_desi_insert.png

Usage:
  python src/analyze_umap_q1_space_desi_insert.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "q1_individual" / "insert"

Q1_BG      = "#cccccc"
Q1_DESI_C  = "#66bb6a"
ERO_DESI_C = "#9575cd"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Q1 embedding + reducer ──────────────────────────────────────────
    print("Loading Q1 UMAP arrays and reducer...")
    q1      = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    q1_emb  = q1["embedding"]
    q1_y    = q1["y"]
    q1_mean = q1["mean"]
    q1_std  = q1["std"]
    reducer = joblib.load(ARRAYS_DIR / "q1_indiv_reducer.joblib")
    print(f"  Q1: {len(q1_y):,} sources")

    std_safe = np.where(q1_std > 0, q1_std, 1.0)

    def _project(X_raw, label):
        X_scaled = (X_raw - q1_mean) / std_safe
        X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0.0)
        print(f"  Projecting {label}...")
        emb = reducer.transform(X_scaled)
        print(f"    -> {emb.shape[0]:,} points")
        return emb

    # ── Load DESI arrays ─────────────────────────────────────────────────────
    q1_desi_path  = ARRAYS_DIR / "q1_allphz_desi.npz"
    ero_desi_path = ARRAYS_DIR / "ero_desi.npz"
    for p in [q1_desi_path, ero_desi_path]:
        if not p.exists():
            print(f"ERROR: {p} not found.")
            return 1

    q1_desi_all  = np.load(q1_desi_path, allow_pickle=True)
    q1_desi_mask = np.isin(q1_desi_all["phz"].astype(int), [1, 2])
    q1_desi_X    = q1_desi_all["X_features"].astype(float)[q1_desi_mask]
    ero_desi     = np.load(ero_desi_path, allow_pickle=True)
    ero_desi_X   = ero_desi["X_features"].astype(float)

    n_q1_desi  = len(q1_desi_X)
    n_ero_desi = len(ero_desi_X)
    print(f"  Q1  × DESI galaxies: {n_q1_desi:,}")
    print(f"  ERO × DESI galaxies: {n_ero_desi:,}")

    q1_desi_emb  = _project(q1_desi_X,  f"Q1×DESI ({n_q1_desi:,})")
    ero_desi_emb = _project(ero_desi_X, f"ERO×DESI ({n_ero_desi:,})")

    # ── Axis limits ──────────────────────────────────────────────────────────
    x_all = np.concatenate([q1_emb[:, 0], q1_desi_emb[:, 0], ero_desi_emb[:, 0]])
    y_all = np.concatenate([q1_emb[:, 1], q1_desi_emb[:, 1], ero_desi_emb[:, 1]])
    xlim = (x_all.min() - 1, x_all.max() + 1)
    ylim = (y_all.min() - 1, y_all.max() + 1)

    rng = np.random.RandomState(0)
    MAX_PTS = 15_000

    def _downsample(arr, n=MAX_PTS):
        if len(arr) > n:
            return arr[rng.choice(len(arr), n, replace=False)]
        return arr

    # ── Figure: 2-panel scatter ────────────────────────────────────────────
    print("\nPlotting...")
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 7.5))

    def _bg(ax):
        pts = _downsample(q1_emb)
        ax.scatter(pts[:, 0], pts[:, 1], c=Q1_BG, s=2, alpha=0.3,
                   linewidths=0, label=f"Q1 all (n={len(q1_y):,})", zorder=1)

    def _style(ax, title):
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("UMAP 1", fontsize=13, fontweight="bold")
        ax.set_ylabel("UMAP 2", fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=17, fontweight="bold", y=1.02)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=12, loc="lower left", markerscale=4, framealpha=0.8)

    # Left: Q1 × DESI
    _bg(ax_left)
    pts = _downsample(q1_desi_emb)
    ax_left.scatter(pts[:, 0], pts[:, 1], c=Q1_DESI_C, s=6, alpha=0.6,
                    linewidths=0, label=f"Q1×DESI galaxies (n={n_q1_desi:,})", zorder=3)
    _style(ax_left, f"Q1 × DESI galaxies  (n={n_q1_desi:,})")

    # Right: ERO × DESI
    _bg(ax_right)
    pts = _downsample(ero_desi_emb)
    ax_right.scatter(pts[:, 0], pts[:, 1], c=ERO_DESI_C, s=6, alpha=0.6,
                     linewidths=0, label=f"ERO×DESI galaxies (n={n_ero_desi:,})", zorder=3)
    _style(ax_right, f"ERO × DESI galaxies  (n={n_ero_desi:,})")

    out = OUTPUT_DIR / "q1_space_desi.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  -> Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
