#!/usr/bin/env python3
"""Combined UMAP figure: the three projected samples on one Q1 UMAP, colour-coded
by sample — for the write-up.

  * SDSS+DESI QSOs        (optical/spectroscopic quasars)
  * Euclid+WISE QSOs      (mid-IR photometric quasars)
  * NSA galaxy nuclei     (central Gaia detections)

Two panels: (left) all matched sources; (right) 2-parameter (position-only) sources
only — which is where the clean C0(resolved)/C1(point-like) extragalactic story lives.

Reads the projected CSVs in output/umap/{qso,nsa}/.
Writes output/umap/combined/umap_combined_three_samples.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
QSO_DIR = PROJECT_DIR / "output" / "umap" / "qso"
NSA_DIR = PROJECT_DIR / "output" / "umap" / "nsa"
ARR = PROJECT_DIR / "output" / "umap" / "arrays"
OUT_DIR = PROJECT_DIR / "output" / "umap" / "combined"

# Okabe-Ito, distinct from the cluster palette
SAMPLE_COLOURS = {"sdssdesi": "#d55e00", "euclidwise": "#0072b2", "nsa": "#009e73",
                  "quaia": "#cc79a7"}
SAMPLE_LABELS = {"sdssdesi": "SDSS+DESI QSOs", "euclidwise": "Euclid+WISE QSOs",
                 "nsa": "NSA galaxy nuclei", "quaia": "Quaia QSOs (Gaia-native)"}


def _load(path: Path, cols=("umap1", "umap2", "astrometric_params_solved")) -> pd.DataFrame:
    if not path.exists():
        print(f"  [skip] {path.name} missing"); return pd.DataFrame(columns=cols)
    return pd.read_csv(path, usecols=lambda c: c in cols).dropna(subset=["umap1", "umap2"])


def _subsample(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    if len(df) <= k:
        return df
    return df.iloc[np.sort(np.random.default_rng(seed).choice(len(df), k, replace=False))]


def main() -> int:
    # samples
    sdss = _load(QSO_DIR / "arrays" / "qso_gaia_matches.csv")
    desi = _load(QSO_DIR / "arrays" / "qso_gaia_matches_desi.csv")
    sdssdesi = pd.concat([sdss, desi], ignore_index=True)
    euclidwise = _load(QSO_DIR / "arrays" / "qso_gaia_matches_euclidwise.csv")
    quaia = _load(QSO_DIR / "arrays" / "qso_gaia_matches_quaia.csv")
    nsa = _load(NSA_DIR / "nsa_gaia_matches.csv")
    samples = {"sdssdesi": sdssdesi, "quaia": quaia, "euclidwise": euclidwise, "nsa": nsa}
    for k, d in samples.items():
        print(f"  {SAMPLE_LABELS[k]}: {len(d):,}")

    # Q1 background + cluster centroids
    q1 = np.load(ARR / "q1_indiv.npz", allow_pickle=True)
    emb, y = q1["embedding"], q1["y"]
    labels = np.load(ARR / "q1_cluster_labels.npy")
    cids = sorted(set(int(c) for c in labels[labels >= 0]))
    cents = {c: (emb[labels == c, 0].mean(), emb[labels == c, 1].mean()) for c in cids}
    galpct = {c: 100 * (y[labels == c] == 1).mean() for c in cids}
    xlim = (emb[:, 0].min() - 0.8, emb[:, 0].max() + 0.8)
    ylim = (emb[:, 1].min() - 0.8, emb[:, 1].max() + 0.8)

    def _panel(ax, dfs: dict, title: str, sizes: dict):
        ax.scatter(emb[:, 0], emb[:, 1], s=2, c="#e8e8e8", alpha=0.5, linewidths=0, rasterized=True, zorder=1)
        # draw largest sample first so the rarer ones stay visible on top
        order = ["sdssdesi", "quaia", "nsa", "euclidwise"]
        for k in order:
            d = dfs[k]
            if not len(d):
                continue
            ax.scatter(d["umap1"], d["umap2"], s=sizes[k], c=SAMPLE_COLOURS[k], alpha=0.35,
                       linewidths=0, rasterized=True, zorder=2 + order.index(k),
                       label=f"{SAMPLE_LABELS[k]} ({len(d):,})")
        for c in cids:
            cx, cy = cents[c]
            t = ax.text(cx, cy, f"C{c}\n{galpct[c]:.0f}%gal", fontsize=9, fontweight="bold",
                        ha="center", va="center", color="#222", zorder=9)
            t.set_path_effects([pe.Stroke(linewidth=2.4, foreground="white"), pe.Normal()])
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=14, fontweight="bold")
        leg = ax.legend(loc="upper right", fontsize=9, framealpha=0.92, markerscale=3)
        for lh in leg.legend_handles:
            lh.set_alpha(1)

    fig, axes = plt.subplots(1, 2, figsize=(21, 10.5))
    all_ds = {k: _subsample(d, 45000, 1) for k, d in samples.items()}
    _panel(axes[0], all_ds, "All matched sources",
           {"sdssdesi": 3, "quaia": 3, "nsa": 3, "euclidwise": 7})

    two = {k: _subsample(d[d["astrometric_params_solved"] == 3], 45000, 1) for k, d in samples.items()}
    _panel(axes[1], two, "2-parameter (position-only) sources only",
           {"sdssdesi": 4, "quaia": 4, "nsa": 4, "euclidwise": 9})

    fig.suptitle("Four samples on the Q1 14-feature UMAP  —  optical QSOs · Gaia-native QSOs · "
                 "mid-IR QSOs · galaxy nuclei", fontsize=16, fontweight="bold", y=0.98)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "umap_combined_three_samples.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
