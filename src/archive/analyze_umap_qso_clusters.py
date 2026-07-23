#!/usr/bin/env python3
"""Cluster-labelled UMAP plots for the projected QSO samples (the analog of
analyze_umap_q1_clusters.py, but for the QSO overlays).

For each QSO catalogue projected onto the Q1 14-feature UMAP by query_umap_qso.py,
this reads the saved per-QSO CSV (which already carries `cluster`, `umap1/2`,
`model_pgal`, `astrometric_params_solved`) and produces, over the grey Q1
reference embedding:

    {catalog}_cluster_labels.png        QSO points coloured + labelled by C0..C4
    {catalog}_cluster_pgal_combined.png cluster labels | XGB P(gal)
    {catalog}_cluster_summary.csv       per-cluster composition table

Runs entirely off local arrays — no WSDB needed.

Usage:
    python src/analyze_umap_qso_clusters.py                 # all catalogues
    python src/analyze_umap_qso_clusters.py --catalog desi  # one
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR = PROJECT_DIR / "output" / "umap" / "arrays"          # Q1 reference
QSO_ARRAYS = PROJECT_DIR / "output" / "umap" / "qso" / "arrays"
OUT_DIR = PROJECT_DIR / "output" / "umap" / "qso" / "clusters"

# same palette as the QSO overlays (query_umap_qso.py)
CLUSTER_COLOURS = {0: "#e6194b", 1: "#4363d8", 2: "#f58231", 3: "#3cb44b", 4: "#911eb4"}
NOISE_COLOR = "#dddddd"

CATALOGS = {          # catalog -> (csv suffix, display label)
    "sdss":       ("",            "SDSS"),
    "desi":       ("_desi",       "DESI"),
    "euclidwise": ("_euclidwise", "Euclid+WISE"),
    "quaia":      ("_quaia",      "Quaia"),
}
APS_LABEL = {3: "2-param", 31: "5-param", 95: "6-param"}


def load_reference():
    """Q1 embedding + cluster labels → grey backdrop + C0..C4 centroids."""
    npz = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    emb = npz["embedding"]
    labels = np.load(ARRAYS_DIR / "q1_cluster_labels.npy")
    cluster_ids = sorted(int(c) for c in set(labels[labels >= 0]))
    centroids = {c: (emb[labels == c, 0].mean(), emb[labels == c, 1].mean())
                 for c in cluster_ids}
    return emb, cluster_ids, centroids


def _backdrop(ax, ref_emb):
    ax.scatter(ref_emb[:, 0], ref_emb[:, 1], s=1, c=NOISE_COLOR,
               alpha=0.5, linewidths=0, rasterized=True)


def _label_clusters(ax, cluster_ids, centroids):
    for c in cluster_ids:
        cx, cy = centroids[c]
        ax.annotate(f"C{c}", (cx, cy), fontsize=13, fontweight="bold",
                    ha="center", va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=CLUSTER_COLOURS.get(c, "k"), alpha=0.9, lw=1.5))


def _scatter_by_cluster(ax, df, cluster_ids):
    rng = np.random.RandomState(0)
    order = rng.permutation(len(df))
    d = df.iloc[order]
    for c in cluster_ids:
        m = d["cluster"].to_numpy() == c
        if not m.any():
            continue
        ax.scatter(d["umap1"].to_numpy()[m], d["umap2"].to_numpy()[m], s=4,
                   c=CLUSTER_COLOURS.get(c, "k"), alpha=0.55, linewidths=0,
                   rasterized=True)


def composition(df, cluster_ids, n_total):
    rows = []
    for c in cluster_ids:
        sub = df[df["cluster"] == c]
        n = len(sub)
        if n == 0:
            rows.append(dict(cluster=f"C{c}", n=0, pct=0.0, mean_pgal=np.nan,
                             frac_pgal_ge_0p5=np.nan, n_2param=0, n_5param=0,
                             n_6param=0, z_median=np.nan))
            continue
        aps = sub["astrometric_params_solved"]
        z = sub["qso_z"] if "qso_z" in sub else pd.Series([np.nan])
        rows.append(dict(
            cluster=f"C{c}", n=n, pct=100 * n / n_total,
            mean_pgal=float(sub["model_pgal"].mean()),
            frac_pgal_ge_0p5=100 * float((sub["model_pgal"] >= 0.5).mean()),
            n_2param=int((aps == 3).sum()), n_5param=int((aps == 31).sum()),
            n_6param=int((aps == 95).sum()),
            z_median=float(np.nanmedian(z)) if z.notna().any() else np.nan))
    return pd.DataFrame(rows)


def plot_catalog(catalog: str) -> int:
    suffix, label = CATALOGS[catalog]
    csv = QSO_ARRAYS / f"qso_gaia_matches{suffix}.csv"
    if not csv.exists():
        print(f"  [{catalog}] {csv.name} not found — skipping"); return 0
    df = pd.read_csv(csv, usecols=lambda c: c in {
        "umap1", "umap2", "cluster", "model_pgal", "astrometric_params_solved", "qso_z"})
    df = df[df["cluster"].notna()].copy()
    df["cluster"] = df["cluster"].astype(int)
    ref_emb, cluster_ids, centroids = load_reference()
    n_total = len(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pad = 1
    xlim = (ref_emb[:, 0].min() - pad, ref_emb[:, 0].max() + pad)
    ylim = (ref_emb[:, 1].min() - pad, ref_emb[:, 1].max() + pad)

    # per-cluster legend handles (count + % of catalogue + P(gal)>=0.5 fraction)
    comp = composition(df, cluster_ids, n_total)
    handles = []
    for _, r in comp.iterrows():
        c = int(r["cluster"][1:])
        handles.append(Line2D([0], [0], marker="o", color="w",
            markerfacecolor=CLUSTER_COLOURS.get(c, "k"), markersize=7,
            label=f"C{c}: {int(r['n']):,}  ({r['pct']:.0f}%,  "
                  f"{r['frac_pgal_ge_0p5']:.0f}% P(gal)≥0.5)"))

    # ── Fig 1: cluster labels ────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8.5, 6.5))
    _backdrop(ax1, ref_emb)
    _scatter_by_cluster(ax1, df, cluster_ids)
    _label_clusters(ax1, cluster_ids, centroids)
    ax1.legend(handles=handles, fontsize=9, loc="lower left", framealpha=0.85)
    ax1.set_xlabel("UMAP 1", fontsize=13, fontweight="bold")
    ax1.set_ylabel("UMAP 2", fontsize=13, fontweight="bold")
    ax1.set_title(f"{label} QSOs — UMAP cluster membership  (N={n_total:,})",
                  fontsize=13, fontweight="bold")
    ax1.set_xlim(*xlim); ax1.set_ylim(*ylim); ax1.grid(alpha=0.2)
    fig1.tight_layout()
    p1 = OUT_DIR / f"{catalog}_cluster_labels.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(fig1)

    # ── Fig 2: cluster labels | XGB P(gal) ───────────────────────────────────
    fig2, (axc, axp) = plt.subplots(1, 2, figsize=(13, 5.4))
    fig2.subplots_adjust(wspace=0.24)
    _backdrop(axc, ref_emb)
    _scatter_by_cluster(axc, df, cluster_ids)
    _label_clusters(axc, cluster_ids, centroids)
    axc.legend(handles=handles, fontsize=8, loc="lower left", framealpha=0.85)
    axc.set_title("UMAP cluster membership", fontsize=13, fontweight="bold")

    _backdrop(axp, ref_emb)
    rng = np.random.RandomState(42)
    o = rng.permutation(len(df))
    sc = axp.scatter(df["umap1"].to_numpy()[o], df["umap2"].to_numpy()[o],
                     c=df["model_pgal"].to_numpy()[o], cmap="RdYlBu_r",
                     vmin=0, vmax=1, s=4, alpha=0.6, linewidths=0, rasterized=True)
    _label_clusters(axp, cluster_ids, centroids)
    cb = fig2.colorbar(sc, ax=axp, pad=0.02, fraction=0.046)
    cb.set_label("XGB P(gal)", fontsize=12, fontweight="bold")
    axp.set_title("XGBoost P(gal)", fontsize=13, fontweight="bold")
    for ax in (axc, axp):
        ax.set_xlabel("UMAP 1", fontsize=12, fontweight="bold")
        ax.set_ylabel("UMAP 2", fontsize=12, fontweight="bold")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.grid(alpha=0.2)
    fig2.suptitle(f"{label} QSOs projected on the Q1 UMAP  (N={n_total:,})",
                  fontsize=14, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    p2 = OUT_DIR / f"{catalog}_cluster_pgal_combined.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(fig2)

    comp.to_csv(OUT_DIR / f"{catalog}_cluster_summary.csv", index=False)

    # console table
    print(f"\n=== {label} QSOs  (N={n_total:,}) ===")
    print(f"  {'Clust':>5} {'N':>9} {'%':>6} {'meanP(gal)':>11} {'P>=0.5%':>8} "
          f"{'2par':>7} {'5par':>7} {'6par':>7} {'z_med':>6}")
    for _, r in comp.iterrows():
        zt = "n/a" if np.isnan(r["z_median"]) else f"{r['z_median']:.2f}"
        mp = "n/a" if np.isnan(r["mean_pgal"]) else f"{r['mean_pgal']:.2f}"
        pg = "n/a" if np.isnan(r["frac_pgal_ge_0p5"]) else f"{r['frac_pgal_ge_0p5']:.0f}"
        print(f"  {r['cluster']:>5} {int(r['n']):>9,} {r['pct']:>5.1f}% {mp:>11} "
              f"{pg:>7}% {int(r['n_2param']):>7,} {int(r['n_5param']):>7,} "
              f"{int(r['n_6param']):>7,} {zt:>6}")
    print(f"  -> {p1.name}, {p2.name}, {catalog}_cluster_summary.csv")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", choices=list(CATALOGS) + ["all"], default="all")
    args = ap.parse_args()
    cats = list(CATALOGS) if args.catalog == "all" else [args.catalog]
    for c in cats:
        plot_catalog(c)
    print(f"\nDone. Outputs in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
