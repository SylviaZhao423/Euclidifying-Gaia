#!/usr/bin/env python3
"""Gaia-parameter DISTRIBUTION plots for the hand-picked QSOs on a DESI/SDSS/combined
QSO×Gaia UMAP.  Split out of analyze_picked_qso_umap.py so the distribution figures can be
regenerated on their own (they only need the build npz + a fresh projection of the picks).

Reuses the projection helpers from analyze_picked_qso_umap.py (fetch_features, reducer load,
the multi_gaia companion map, the path constants) so this stays a thin wrapper: it rebuilds
the picks' `ids` table (14 features + projected HDBSCAN cluster), then draws the two 4x4
distribution grids.

Outputs (output/picked_qso/); the default `combined` sample drops its name prefix, other
samples keep a `{sample}_` prefix (e.g. desi_qso_gaia_features_hist.png):
  qso_gaia_features_hist.png         4x4 grid, NO picks — the population DENSITY distribution
      split by HDBSCAN cluster.
  qso_gaia_features_hist_picked.png  same 4x4 grid, but each box = the population DENSITY
      distribution (left axis) + the picks' COUNT distribution split by QSO-UMAP cluster
      (separate right count axis, so ~10s of picks stay visible next to the pop).

Usage:
  PGUSER=… python src/analyze_picked_qso_dist.py                 # combined UMAP, n_all
  PGUSER=… python src/analyze_picked_qso_dist.py --sample desi --n-sample 480000
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from analyze_picked_qso_umap import (
    fetch_features, _load_reducer, MULTI_GAIA_PAIRS,
    DS_DIR, OUT_DIR, CSV, n_suffix, parse_n, build_arrays_dir,
)
from analyze_picked_qso_contact_sheet import PICKED
from analyze_qso_umap_ds import CLUSTER_COLOURS
from selected_parameters_list import SELECTED_GAIA_PARAMETERS


def plot_features_hist(sample, X_raw, out, ids=None):
    """Companion to plot_features_grid with the EXACT same 4x4 layout, but each of the 14
    boxes is the COUNT distribution of that feature instead of the UMAP scatter (feature-name
    x-axis; x-range = the same 1-99 percentile clip as the UMAP colour scale, so boxes line up
    when swapping figures). If `ids` is given, the whole population is a filled backdrop on the
    LEFT count axis and the picks are one step-line per QSO-UMAP cluster on a SEPARATE RIGHT
    count axis (so the ~10s of picks stay visible next to the ~10^5 population)."""
    nfeat = len(SELECTED_GAIA_PARAMETERS); ncols = 4
    nrows = int(np.ceil(nfeat / ncols))
    clusters = sorted(int(c) for c in np.unique(ids["cluster"])) if ids is not None else []
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.0))
    axes = np.atleast_2d(axes)
    for fi, feat in enumerate(SELECTED_GAIA_PARAMETERS):
        ax = axes[fi // ncols, fi % ncols]
        pop = X_raw[:, fi].astype(float)
        finite = pop[np.isfinite(pop)]
        lo, hi = (np.percentile(finite, [1, 99]) if len(finite) else (0.0, 1.0))
        if hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 60)
        ax.hist(np.clip(finite, lo, hi), bins=bins, density=True, color="#9ccc9c", alpha=0.30,
                edgecolor="none", label=f"population (n={len(finite):,})", zorder=1)
        from matplotlib.ticker import ScalarFormatter
        _fmt = ScalarFormatter(useMathText=True); _fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(_fmt)                # linear axis, sci-notation ticks (×10^N)
        ax.yaxis.get_offset_text().set_fontsize(6)
        ax.set_ylabel("population density", fontsize=8)
        if ids is not None:
            ax2 = ax.twinx()                              # separate count axis for the picks
            for c in clusters:                            # picks split by their QSO-UMAP cluster
                cv = ids.loc[ids["cluster"] == c, feat].to_numpy(float)
                cf = cv[np.isfinite(cv)]
                if len(cf):
                    col = CLUSTER_COLOURS.get(c, "k")
                    ax2.hist(np.clip(cf, lo, hi), bins=bins, color=col, alpha=0.2,
                             edgecolor="none", zorder=4)                        # soft fill
                    ax2.hist(np.clip(cf, lo, hi), bins=bins, histtype="step",
                             color=col, lw=1.2, alpha=1.0, zorder=5,
                             label=f"C{c} picks (n={len(cf)})")                 # bold outline
            ax2.set_ylim(0, 30)                           # fixed pick-count scale across all panels
            ax2.set_ylabel("pick count", fontsize=8); ax2.tick_params(labelsize=6)
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6, framealpha=0.9)
        else:
            ax.legend(loc="upper right", fontsize=6, framealpha=0.9)
        ax.set_xlim(lo, hi); ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
        ax.set_xlabel(feat, fontsize=8)
        ax.set_title(feat, fontsize=9, fontweight="bold", pad=4)
    for fi in range(nfeat, nrows * ncols):
        axes[fi // ncols, fi % ncols].set_visible(False)
    n_euclid_picks = int((ids["role"] == "primary").sum()) if ids is not None else 0
    tail = (f" — {n_euclid_picks} hand-picked candidate doubles by QSO-UMAP cluster"
            if ids is not None else "")
    fig.suptitle(f"{sample.upper()} QSO × Gaia — Gaia-parameter distributions"
                 f"{tail}",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


def plot_features_hist_by_cluster(sample, X_raw, labels, out):
    """14-feature grid (same layout as plot_features_hist) but NO picks — instead each box
    overlays the POPULATION distribution split by HDBSCAN cluster, as raw counts on the same
    linear/sci-notation y-axis used for the population panel of plot_features_hist."""
    from matplotlib.ticker import ScalarFormatter
    nfeat = len(SELECTED_GAIA_PARAMETERS); ncols = 4
    nrows = int(np.ceil(nfeat / ncols))
    cl_ids = sorted(int(c) for c in np.unique(labels) if c >= 0)
    counts = {c: int((labels == c).sum()) for c in cl_ids}
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.0))
    axes = np.atleast_2d(axes)
    for fi, feat in enumerate(SELECTED_GAIA_PARAMETERS):
        ax = axes[fi // ncols, fi % ncols]
        colv = X_raw[:, fi].astype(float)
        finite = colv[np.isfinite(colv)]
        lo, hi = (np.percentile(finite, [1, 99]) if len(finite) else (0.0, 1.0))
        if hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 60)
        # LEFT axis: all-population green backdrop EXACTLY as plot_features_hist (#9ccc9c, alpha
        # 0.30, no outline) on its OWN scale, so the broad population envelope stays prominent
        # instead of being squashed by the peaky small clusters. Drawn on ax (behind); the
        # per-cluster curves go on the twin axis in front.
        ax.hist(np.clip(finite, lo, hi), bins=bins, density=True, color="#9ccc9c", alpha=0.30,
                edgecolor="none", label=f"population (n={len(finite):,})", zorder=1)
        _fmt = ScalarFormatter(useMathText=True); _fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(_fmt); ax.yaxis.get_offset_text().set_fontsize(6)
        ax.set_ylabel("population density", fontsize=8)
        # RIGHT axis: the per-cluster distributions on their own shared density scale (soft fill +
        # bold step, same format as the picked figure); drawn in front of the green backdrop.
        ax2 = ax.twinx()
        for c in cl_ids:
            cf = colv[labels == c]; cf = cf[np.isfinite(cf)]
            if len(cf) > 10:
                col = CLUSTER_COLOURS.get(c, "k")
                ax2.hist(np.clip(cf, lo, hi), bins=bins, density=True, color=col, alpha=0.2,
                         edgecolor="none", zorder=4)
                ax2.hist(np.clip(cf, lo, hi), bins=bins, density=True, histtype="step",
                         color=col, lw=1.2, alpha=1.0, zorder=5, label=f"C{c} (n={counts[c]:,})")
        _fmt2 = ScalarFormatter(useMathText=True); _fmt2.set_powerlimits((0, 0))
        ax2.yaxis.set_major_formatter(_fmt2); ax2.yaxis.get_offset_text().set_fontsize(6)
        ax2.set_ylabel("per-cluster density", fontsize=8); ax2.tick_params(labelsize=6)
        ax.set_xlim(lo, hi); ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
        ax.set_xlabel(feat, fontsize=8)
        ax.set_title(feat, fontsize=9, fontweight="bold", pad=4)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6, framealpha=0.9)
    for fi in range(nfeat, nrows * ncols):
        axes[fi // ncols, fi % ncols].set_visible(False)
    fig.suptitle(f"{sample.upper()} QSO × Gaia — Gaia-parameter distributions BY HDBSCAN cluster  "
                 f"(G ≥ 17, {nfeat} features)",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


def build_ids(sample, npz_path, red_path, user):
    """Project the picks (primaries + multi_gaia companions) onto the {sample} UMAP and return
    (ids, pop_X_raw, labels): ids carries role, the 14 features, and the nearest-neighbour
    HDBSCAN cluster; pop_X_raw/labels are the build population arrays for the backdrop."""
    flat = pd.read_csv(CSV, dtype={"gaia_source_id": "int64"})
    wise_of = dict(zip(flat["gaia_source_id"], flat["qso_id"]))

    rows = [{"gaia_source_id": g, "role": "primary", "wise": wise_of.get(g, "")} for g in PICKED]
    for prim, comp in MULTI_GAIA_PAIRS.items():
        rows.append({"gaia_source_id": comp, "role": "companion", "wise": wise_of.get(prim, "")})
    ids = pd.DataFrame(rows).drop_duplicates("gaia_source_id").reset_index(drop=True)
    print(f"  projecting {len(ids)} gaia ids onto the {sample.upper()} QSO×Gaia UMAP "
          f"({(ids.role=='primary').sum()} primary + {(ids.role=='companion').sum()} companion)")

    feats = fetch_features(ids["gaia_source_id"].tolist(), user)
    feats = feats.rename(columns={"source_id": "gaia_source_id"})
    ids = ids.merge(feats, on="gaia_source_id", how="left")

    d = np.load(npz_path, allow_pickle=True)
    q1_emb = d["embedding"]; pop_X_raw = d["X_raw"]; labels = d["cluster"]
    mean, std = d["mean"], d["std"]
    std_safe = np.where(std > 0, std, 1.0)
    X  = ids[list(SELECTED_GAIA_PARAMETERS)].apply(pd.to_numeric, errors="coerce").values.astype(float)
    Xs = np.where(np.isfinite((X - mean) / std_safe), (X - mean) / std_safe, 0.0)
    emb = _load_reducer(red_path).transform(Xs).astype(float)

    valid = labels >= 0
    _, idx = NearestNeighbors(n_neighbors=1).fit(q1_emb[valid]).kneighbors(emb)
    ids["cluster"] = labels[np.where(valid)[0][idx.ravel()]]
    return ids, pop_X_raw, labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", choices=["combined", "desi", "sdss"], default="combined",
                    help="which QSO×Gaia UMAP to project onto (default combined).")
    ap.add_argument("--n-sample", type=parse_n, default="all",
                    help="ds_qso build size used as the UMAP background: int or 'all' (default).")
    args = ap.parse_args()
    sample, n = args.sample, args.n_sample
    user = os.environ.get("PGUSER", "siwei_zhao_2025")

    ds_arrays = build_arrays_dir(sample, n)
    npz_path = ds_arrays / f"{sample}_qso_gaia.npz"
    red_path = ds_arrays / f"{sample}_qso_gaia_reducer.joblib"
    if not (npz_path.exists() and red_path.exists()):
        print(f"  missing {n_suffix(n)} arrays for {sample} — build it first (query_qso_umap_ds.py).")
        return 1

    ids, pop_X_raw, labels = build_ids(sample, npz_path, red_path, user)
    # combined is the default sample -> drop the "combined_" prefix (matches analyze_picked_qso_umap.py)
    pfx = "" if sample == "combined" else f"{sample}_"

    # population backdrop + picks split by QSO-UMAP cluster (same 4x4 layout as the feature grid)
    plot_features_hist(sample, pop_X_raw, OUT_DIR / f"{pfx}qso_gaia_features_hist_picked.png", ids=ids)
    # same grid, no picks — population distributions split by HDBSCAN cluster
    plot_features_hist_by_cluster(sample, pop_X_raw, labels,
                                  OUT_DIR / f"{pfx}qso_gaia_features_hist.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
