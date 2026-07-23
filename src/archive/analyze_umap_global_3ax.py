#!/usr/bin/env python3
"""
Overlay the 3 animation sort axes on the original 14-feature Q1 UMAP.
All 50k sources are plotted in the original UMAP coordinate space; each panel
is coloured by one axis. Cluster centroids are annotated.

  AEN sig     — astrometric_excess_noise_sig   (X_raw[:, 3])
  BPRP excess — corrected_bp_rp_excess         (X_raw[:, 11])
  variability — sqrt(phot_g_n_obs)/phot_g_mean_flux_over_error
                (taken from q1_addvar.npz column 14, already computed)

Reads:
  output/umap/arrays/q1_indiv.npz          — original 14-feat embedding
  output/umap/arrays/q1_cluster_labels.npy — cluster assignments
  output/umap/q1_addvar/q1_addvar.npz      — X_raw col 14 = variability

Writes:
  output/umap/q1_clusters_3ax/umap_global_3ax.png

Usage:
  python src/analyze_umap_global_3ax.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
ADDVAR_NPZ  = PROJECT_DIR / "output" / "umap" / "q1_addvar" / "q1_addvar.npz"
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "q1_clusters_3ax"

from selected_parameters_list import SELECTED_GAIA_PARAMETERS  # noqa: E402
import sys; sys.path.insert(0, str(Path(__file__).parent))

AEN_IDX    = SELECTED_GAIA_PARAMETERS.index("astrometric_excess_noise_sig")
BPRPEF_IDX = SELECTED_GAIA_PARAMETERS.index("corrected_bp_rp_excess")

AXIS_NAMES = ["AEN sig", "BPRP excess", "variability"]

CLUSTER_COLOURS = {
    0: "#e6194b",   # red    — 95% gal
    1: "#4363d8",   # blue
    2: "#f58231",   # orange
    3: "#3cb44b",   # green  — main star body
    4: "#911eb4",   # purple
}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    print("Loading arrays...")
    data    = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    labels  = np.load(ARRAYS_DIR / "q1_cluster_labels.npy")
    addvar  = np.load(ADDVAR_NPZ, allow_pickle=True)

    embedding = data["embedding"]       # (50000, 2) — original 14-feat UMAP
    X_raw     = data["X_raw"]           # (50000, 14)
    y         = data["y"]

    # variability is column 14 in q1_addvar.npz X_raw
    variability = addvar["X_raw"][:, 14]

    vals = [
        X_raw[:, AEN_IDX],
        X_raw[:, BPRPEF_IDX],
        variability,
    ]

    # Cluster centroids
    cluster_ids = sorted(set(labels[labels >= 0]))
    centroids = {
        cid: (embedding[labels == cid, 0].mean(),
              embedding[labels == cid, 1].mean())
        for cid in cluster_ids
    }
    cluster_galpct = {
        cid: 100 * (y[labels == cid] == 1).sum() / (labels == cid).sum()
        for cid in cluster_ids
    }

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.subplots_adjust(wspace=0.08, left=0.04, right=0.96, top=0.88, bottom=0.04)

    xlim = (embedding[:, 0].min() - 0.8, embedding[:, 0].max() + 0.8)
    ylim = (embedding[:, 1].min() - 0.8, embedding[:, 1].max() + 0.8)

    for ax, name, v in zip(axes, AXIS_NAMES, vals):
        lo, hi = np.nanpercentile(v, 1), np.nanpercentile(v, 99)
        vc = np.clip(v, lo, hi)

        sc = ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=vc, cmap="RdYlBu_r", vmin=lo, vmax=hi,
            s=2, alpha=0.55, linewidths=0, rasterized=True,
        )
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=13, fontweight="bold", pad=6)
        ax.grid(alpha=0.15)

        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=8)

        # Annotate cluster centroids
        for cid in cluster_ids:
            cx, cy = centroids[cid]
            gp = cluster_galpct[cid]
            txt = ax.text(
                cx, cy, f"C{cid}\n{gp:.0f}%gal",
                fontsize=8, fontweight="bold", ha="center", va="center",
                color=CLUSTER_COLOURS[cid],
            )
            txt.set_path_effects([
                pe.Stroke(linewidth=2, foreground="white"),
                pe.Normal(),
            ])

    fig.suptitle(
        "Q1 UMAP (original 14-feature embedding) coloured by animation sort axes\n"
        "Cluster centroids labelled with galaxy fraction",
        fontsize=12, fontweight="bold",
    )

    out = OUTPUT_DIR / "umap_global_3ax.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
