#!/usr/bin/env python3
"""Colour the projected QSOs on the Q1 UMAP by their Gaia astrometric SIGNIFICANCE —
parallax_over_error and pm_significance — to make the "5/6-param QSOs still have
parallax/pmra/pmdec" point visual:

  * 2-parameter QSOs (Clusters 0/1) have NO astrometric solution at all → the value is
    NaN. They are drawn in light grey ("no astrometry").
  * 5/6-parameter QSOs (mostly Cluster 3) DO carry parallax + PM, but the values are
    consistent with ZERO — near-white in a diverging parallax_over_error map, low in a
    sequential pm_significance map. So having a 5/6-param solution ≠ a real parallax/PM.

Companion to analyze_umap_qso.py — a SEPARATE script (does not modify it). Same overlay
style as umap_qso_{tag}_clusters.png (grey Q1 reference embedding + C0–C4 labels) and the
value-colour + 2D-binned-mean panels of analyze_nsa_nucleation.py.

Reads the projected CSV written by analyze_umap_qso.py:
    output/umap/qso/arrays/qso_gaia_matches{suffix}.csv
Writes (per catalog):
    output/umap/qso/umap_qso_{tag}_astrometry.png   (value-coloured scatter)

Usage:
    conda run -n base python3 src/analyze_umap_qso_astrometry.py               # desisdss + euclidwise
    conda run -n base python3 src/analyze_umap_qso_astrometry.py --catalog desisdss
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Reuse paths + cluster colours from the main projector (no modification to it).
from analyze_umap_qso import ARRAYS_DIR, OUTPUT_DIR, QSO_ARRAYS, CLUSTER_COLOURS  # noqa: E402

# tag -> projected CSV (the two catalogs that get overlays in analyze_umap_qso.py)
CATALOGS = {
    "desisdss":   QSO_ARRAYS / "qso_gaia_matches_desisdss.csv",
    "euclidwise": QSO_ARRAYS / "qso_gaia_matches_euclidwise.csv",
}

# (column, label, cmap, kind, vlim) — parallax_over_error is signed (diverging, white=0);
# pm_significance is a magnitude (sequential, ±vlim / [0,vlim] clip the spurious tail);
# phot_g_mean_mag is the G magnitude (sequential, vlim = (vmin, vmax)). The magnitude panel
# shows WHY the astrometry differs: bright QSOs get 5/6-param solutions (C3/C4), faint ones
# fall back to 2-param (C0/C1).
ASTRO_PANELS = [
    ("parallax_over_error", "parallax_over_error", "RdYlBu_r",  "div", 5.0),
    ("pm_significance",     "pm_significance",      "viridis_r", "seq", 5.0),
    ("phot_g_mean_mag",     "G magnitude",          "cividis",   "mag", (16.5, 21.5)),
]


def _q1_reference():
    """Grey Q1 embedding + cluster centroids for the background + C0–C4 labels."""
    q1 = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    emb = q1["embedding"]
    labels = np.load(ARRAYS_DIR / "q1_cluster_labels.npy")
    ids = sorted(int(c) for c in set(labels[labels >= 0]))
    centroids = {c: (emb[labels == c, 0].mean(), emb[labels == c, 1].mean()) for c in ids}
    xlim = (emb[:, 0].min() - 0.8, emb[:, 0].max() + 0.8)
    ylim = (emb[:, 1].min() - 0.8, emb[:, 1].max() + 0.8)
    return emb, ids, centroids, xlim, ylim


def _base(ax, emb, ids, centroids, xlim, ylim):
    ax.scatter(emb[:, 0], emb[:, 1], s=2, c="#e2e2e2", alpha=0.6,
               linewidths=0, rasterized=True, zorder=1)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_xticks([]); ax.set_yticks([])
    for c in ids:
        cx, cy = centroids[c]
        t = ax.text(cx, cy, f"C{c}", fontsize=9, fontweight="bold",
                    ha="center", va="center", color=CLUSTER_COLOURS[c], zorder=7)
        t.set_path_effects([pe.Stroke(linewidth=2, foreground="white"), pe.Normal()])


def _norm(kind, vlim):
    if kind == "div":
        return mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    if kind == "mag":
        return mcolors.Normalize(vlim[0], vlim[1])
    return mcolors.Normalize(0, vlim)


def _suptitle(det):
    return (f"{len(det):,} QSOs on the Q1 UMAP — Gaia astrometric significance & G magnitude   "
            f"(2-param = faint, no parallax/PM; 5/6-param = brighter, values ≈ 0)")


def make_scatter(det, out):
    """Value-coloured scatter: each QSO coloured by parallax_over_error / pm_significance /
    G magnitude. In the astrometry panels 2-param QSOs (no solution → NaN) are light grey;
    every QSO has a magnitude, so the G panel colours them all."""
    emb, ids, centroids, xlim, ylim = _q1_reference()
    aps = det["astrometric_params_solved"].to_numpy()
    extend = {"div": "both", "seq": "max", "mag": "both"}
    fig, axes = plt.subplots(1, len(ASTRO_PANELS), figsize=(6.7 * len(ASTRO_PANELS), 7.2))
    for ax, (col, label, cmap, kind, vlim) in zip(axes, ASTRO_PANELS):
        _base(ax, emb, ids, centroids, xlim, ylim)
        v = pd.to_numeric(det[col], errors="coerce").to_numpy()
        finite = np.isfinite(v)
        if kind != "mag":                       # astrometry: NaN = 2-param (no solution)
            nan = ~finite
            ax.scatter(det["umap1"][nan], det["umap2"][nan], s=4, c="#bdbdbd", alpha=0.35,
                       linewidths=0, rasterized=True, zorder=2,
                       label=f"2-param: no astrometry ({int(nan.sum()):,})")
            ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
        sc = ax.scatter(det["umap1"][finite], det["umap2"][finite], s=5, c=v[finite],
                        cmap=cmap, norm=_norm(kind, vlim), alpha=0.6, linewidths=0,
                        rasterized=True, zorder=3)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, extend=extend[kind])
        cb.set_label(f"{label}  ({'mag' if kind == 'mag' else 'σ'})", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax.set_title(label, fontsize=13, fontweight="bold")
        vf = v[finite]
        if kind == "div":
            box = (f"5/6-param (N={int(finite.sum()):,}):\n median {np.median(vf):+.2f}σ\n"
                   f" |·| < 3σ: {100*np.mean(np.abs(vf) < 3):.0f}%")
        elif kind == "seq":
            box = (f"5/6-param (N={int(finite.sum()):,}):\n median {np.median(vf):.2f}σ\n"
                   f" < 3σ: {100*np.mean(vf < 3):.0f}%")
        else:  # magnitude — the faint 2-param vs brighter 5/6-param split (the reason for it)
            box = (f"median G:\n 2-param: {np.nanmedian(v[aps == 3]):.1f}\n"
                   f" 5/6-param: {np.nanmedian(v[aps != 3]):.1f}")
        ax.text(0.03, 0.97, box, transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
                zorder=8, bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.92))
    fig.suptitle(_suptitle(det), fontsize=15, fontweight="bold", x=0.5, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  -> {out}")


def run(tag: str) -> int:
    csv = CATALOGS[tag]
    if not csv.exists():
        print(f"  {csv.name} not found — run analyze_umap_qso.py first."); return 1
    det = pd.read_csv(csv, usecols=lambda c: c in
                      {"umap1", "umap2", "cluster", "astrometric_params_solved",
                       "parallax_over_error", "pm_significance", "phot_g_mean_mag"}
                      ).dropna(subset=["umap1", "umap2"])
    print(f"{tag}: {len(det):,} projected QSOs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    make_scatter(det, OUTPUT_DIR / f"umap_qso_{tag}_astrometry.png")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", choices=["desisdss", "euclidwise", "both"], default="both",
                    help="which projected QSO sample to colour (default both).")
    args = ap.parse_args()
    tags = ["desisdss", "euclidwise"] if args.catalog == "both" else [args.catalog]
    for t in tags:
        run(t)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
