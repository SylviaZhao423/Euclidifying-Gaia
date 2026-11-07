#!/usr/bin/env python3
"""
Analysis for the DESI/SDSS/combined QSO×Gaia UMAPs built by query_qso_umap_ds.py.
Three subcommands:

  features         14-feature grid of a sample's UMAP (grey background, points coloured by
                   each Gaia parameter; RdYlBu_r, 1-99 pct clip)
  picked           project the hand-picked sources onto the UMAP(s) and colour each by its
                   ORIGINAL Q1 cluster (marker+colour); grey background, per-panel legend
  picked-features  14-feature grid of the projected picks (big stars coloured by feature)

Artifacts live under output/qso_umap/ds_qso/n{N}/ (n_all for --n-sample all); the
n{N} folder encodes the sample size so filenames carry no size suffix.

Usage:
  python src/analyze_qso_umap_ds.py features --all-samples --n-sample 480000
  PGUSER=… python src/analyze_qso_umap_ds.py picked --n-sample 480000
  python src/analyze_qso_umap_ds.py picked --n-sample 480000 --replot
  PGUSER=… python src/analyze_qso_umap_ds.py picked-features --all-samples --n-sample 480000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_picked_qso_contact_sheet import PICKED   # authoritative picked list
from selected_parameters_list import SELECTED_GAIA_PARAMETERS
from builder_qso_gaia_crossmatch import WSDB_HOST, WSDB_DB, FEATURE_SQL

PROJECT_DIR = Path(__file__).resolve().parents[1]
DS_DIR      = PROJECT_DIR / "output" / "qso_umap" / "ds_qso"
FLAT_CSV    = PROJECT_DIR / "output" / "umap" / "qso" / "arrays" / "catwise_euclid_gaia.csv"
SAMPLES     = ["desi", "sdss", "combined"]
MAG_CUT     = 17
CLUSTER_COLOURS = {0: "#e6194b", 1: "#4363d8", 2: "#f58231", 3: "#3cb44b", 4: "#911eb4"}
CLUSTER_MARKERS = {0: "*", 1: "o", 2: "s", 3: "D", 4: "^", -1: "X"}


# ── output paths (n{N}/ folder encodes the size; 'all' -> n_all) ──
def n_suffix(n):   return "n_all" if str(n) == "all" else f"n{int(n)}"
def parse_n(v):    return "all" if str(v).lower() == "all" else int(v)
def nlabel(n):     return "all" if str(n) == "all" else f"{int(n):,}"
def out_dir(n):    return DS_DIR / n_suffix(n)
def arrays_dir(n): return out_dir(n) / "arrays"


def fetch_features(ids, user):
    """14 derived Gaia features per source_id (bare-name cols), from gaia_dr3.gaia_source."""
    import psycopg2
    feat_sql = FEATURE_SQL.replace(" AS g_", " AS ")
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30)
    cur = conn.cursor()
    cur.execute(f"SELECT g.source_id, {feat_sql} FROM gaia_dr3.gaia_source g "
                f"WHERE source_id = ANY(%s)", (list(ids),))
    df = pd.DataFrame(cur.fetchall(), columns=["source_id"] + list(SELECTED_GAIA_PARAMETERS))
    conn.close()
    return df


def _load_reducer(path):
    import pynndescent
    for attr, default in [("quantization", None), ("_quantized_data", None),
                          ("parallel_batch_queries", False)]:
        if not hasattr(pynndescent.NNDescent, attr):
            setattr(pynndescent.NNDescent, attr, default)
    return joblib.load(path)


# ════════════════════════════════ features ══════════════════════════════════
def _feature_grid(emb, colour_by, title, out, big_stars=False, star_xy=None):
    """Grid of UMAP panels, one per Gaia feature. If big_stars, only star_xy points are
    coloured (grey full-emb background); else the whole embedding is coloured."""
    nfeat = len(SELECTED_GAIA_PARAMETERS); ncols = 4
    nrows = int(np.ceil(nfeat / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.0))
    axes = np.atleast_2d(axes)
    for fi, feat in enumerate(SELECTED_GAIA_PARAMETERS):
        ax = axes[fi // ncols, fi % ncols]
        vals = colour_by[:, fi].astype(float)
        finite = vals[np.isfinite(vals)]
        lo, hi = (np.percentile(finite, [1, 99]) if len(finite) else (0.0, 1.0))
        if hi <= lo:
            hi = lo + 1.0
        ax.scatter(emb[:, 0], emb[:, 1], c="lightgrey", s=(3 if big_stars else 4),
                   alpha=0.18, marker="o", rasterized=True, linewidths=0, zorder=0)
        if big_stars:
            sc = ax.scatter(star_xy[:, 0], star_xy[:, 1], c=np.clip(vals, lo, hi),
                            cmap="RdYlBu_r", vmin=lo, vmax=hi, s=90, marker="*",
                            edgecolors="k", linewidths=0.5, zorder=2)
        else:
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=np.clip(vals, lo, hi), cmap="RdYlBu_r",
                            vmin=lo, vmax=hi, s=4, alpha=0.6, marker="o", rasterized=True,
                            linewidths=0, zorder=1)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.25)
        ax.set_title(feat, fontsize=9, fontweight="bold", pad=4)
        fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046).ax.tick_params(labelsize=7)
    for fi in range(nfeat, nrows * ncols):
        axes[fi // ncols, fi % ncols].set_visible(False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


def run_features(sample, n):
    p = arrays_dir(n) / f"{sample}_qso_gaia.npz"
    if not p.exists():
        print(f"  missing {p} — run query_qso_umap_ds.py --sample {sample} "
              f"--n-sample {n} first."); return
    d = np.load(p, allow_pickle=True)
    print(f"=== {sample} (n={nlabel(n)}): {len(d['embedding']):,} sources ===")
    title = (f"{sample.upper()} QSO × Gaia UMAP coloured by Gaia parameters  "
             f"(G ≥ {MAG_CUT},  {len(SELECTED_GAIA_PARAMETERS)} features)")
    _feature_grid(d["embedding"], d["X_raw"], title, out_dir(n) / f"{sample}_qso_gaia_features.png")


def cmd_features(args):
    for s in (SAMPLES if args.all_samples else [args.sample]):
        run_features(s, args.n_sample)
    return 0


# ═════════════════════════════════ picked ═══════════════════════════════════
def _pick_ids_features(user):
    flat = pd.read_csv(FLAT_CSV, dtype={"gaia_source_id": "int64"})
    wise_of = dict(zip(flat["gaia_source_id"], flat["qso_id"]))
    q1cl_of = dict(zip(flat["gaia_source_id"], flat["cluster"]))
    ids = pd.DataFrame({"gaia_source_id": list(PICKED)})
    ids["wise"] = ids["gaia_source_id"].map(wise_of)
    ids["q1cluster"] = ids["gaia_source_id"].map(q1cl_of)
    feats = fetch_features([int(x) for x in PICKED], user).rename(columns={"source_id": "gaia_source_id"})
    ids = ids.merge(feats, on="gaia_source_id", how="left")
    print(f"  fetched features for {len(ids)} picked gaia ids")
    return ids


def project_all(user, n):
    ids = _pick_ids_features(user)
    X = ids[list(SELECTED_GAIA_PARAMETERS)].apply(pd.to_numeric, errors="coerce").values.astype(float)
    out = {}
    for sample in SAMPLES:
        npz = arrays_dir(n) / f"{sample}_qso_gaia.npz"
        red = arrays_dir(n) / f"{sample}_qso_gaia_reducer.joblib"
        if not (npz.exists() and red.exists()):
            print(f"  missing n={nlabel(n)} arrays for {sample} — build it first."); continue
        d = np.load(npz, allow_pickle=True)
        emb, mean, std, labels = d["embedding"], d["mean"], d["std"], d["cluster"]
        std_safe = np.where(std > 0, std, 1.0)
        Xs = np.where(np.isfinite((X - mean) / std_safe), (X - mean) / std_safe, 0.0)
        proj = _load_reducer(red).transform(Xs).astype(float)
        valid = labels >= 0
        _, nbr = NearestNeighbors(n_neighbors=1).fit(emb[valid]).kneighbors(proj)
        pk = ids[["gaia_source_id", "wise", "q1cluster"]].copy()
        pk["umap1"], pk["umap2"] = proj[:, 0], proj[:, 1]
        pk["hdb_cluster"] = labels[np.where(valid)[0][nbr.ravel()]]
        out[sample] = pk
        print(f"  {sample}: projected {len(pk)} picks -> QSO-UMAP "
              + "  ".join(f"C{c}:{k}" for c, k in pk['hdb_cluster'].value_counts().sort_index().items()))
    return out


def _pick_panel(ax, sample, pick, n):
    emb = np.load(arrays_dir(n) / f"{sample}_qso_gaia.npz", allow_pickle=True)["embedding"]
    ax.scatter(emb[:, 0], emb[:, 1], s=2, c="#e2e2e2", alpha=0.6, lw=0, rasterized=True, zorder=1)
    ax.set_xlim(emb[:, 0].min() - 0.8, emb[:, 0].max() + 0.8)
    ax.set_ylim(emb[:, 1].min() - 0.8, emb[:, 1].max() + 0.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{sample.upper()} QSO×Gaia UMAP  (N_bg={len(emb):,})", fontsize=13, fontweight="bold")
    q1_clusters = sorted({int(c) for c in pick["q1cluster"].dropna().unique()})
    for c in q1_clusters:
        m = pick["q1cluster"] == c
        ax.scatter(pick.loc[m, "umap1"], pick.loc[m, "umap2"], s=180,
                   marker=CLUSTER_MARKERS.get(c, "X"), color=CLUSTER_COLOURS.get(c, "k"),
                   edgecolors="k", linewidths=0.9, zorder=6)
    miss = pick["q1cluster"].isna()
    if miss.any():
        ax.scatter(pick.loc[miss, "umap1"], pick.loc[miss, "umap2"], s=130, marker="X",
                   color="0.5", edgecolors="k", linewidths=0.9, zorder=6)
    for _, r in pick.iterrows():
        t = ax.text(r["umap1"] + 0.10, r["umap2"] + 0.10, str(r["wise"]), fontsize=5,
                    color="0.15", zorder=8, ha="left", va="bottom")
        t.set_path_effects([pe.Stroke(linewidth=1.2, foreground="white"), pe.Normal()])
    vc = pick["q1cluster"].value_counts()
    handles = [Line2D([0], [0], marker=CLUSTER_MARKERS.get(c, "X"), color="w",
                      markerfacecolor=CLUSTER_COLOURS.get(c, "k"), markeredgecolor="k",
                      markersize=11, label=f"Q1 C{c}  ({int(vc.get(c, 0))})") for c in q1_clusters]
    if miss.any():
        handles.append(Line2D([0], [0], marker="X", color="w", markerfacecolor="0.5",
                              markeredgecolor="k", markersize=10, label=f"no Q1 cluster ({int(miss.sum())})"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.92,
              title="marker = original Q1 cluster")


def plot_picked(proj, n):
    samples = [s for s in SAMPLES if s in proj]
    fig = plt.figure(figsize=(8.2 * len(samples), 7.8))
    gs = fig.add_gridspec(1, len(samples), left=0.03, right=0.965, top=0.86, bottom=0.05, wspace=0.22)
    for k, s in enumerate(samples):
        _pick_panel(fig.add_subplot(gs[0, k]), s, proj[s], n)
    fig.suptitle(f"{len(PICKED)} hand-picked QSO Gaia sources projected onto the "
                 f"{'/'.join(samples).upper()} QSO×Gaia UMAP(s) (n={nlabel(n)}) — "
                 f"marker/colour = original Q1 cluster (14 features → reducer.transform)",
                 fontsize=15, fontweight="bold", x=0.5, y=0.96)
    out = out_dir(n) / "all_project_picked.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


def cmd_picked(args):
    n = args.n_sample
    csv_out = arrays_dir(n) / "picked_double.csv"
    if args.replot:
        if not csv_out.exists():
            print(f"  {csv_out.name} not found — run without --replot first."); return 1
        long = pd.read_csv(csv_out)
        proj = {s: g.drop(columns="sample").reset_index(drop=True) for s, g in long.groupby("sample")}
        print(f"Replotting from {csv_out.name} ({len(long):,} rows)...")
        plot_picked(proj, n); return 0
    user = os.environ.get("PGUSER", "siwei_zhao_2025")
    proj = project_all(user, n)
    if not proj:
        return 1
    arrays_dir(n).mkdir(parents=True, exist_ok=True)
    pd.concat([g.assign(sample=s) for s, g in proj.items()], ignore_index=True).to_csv(csv_out, index=False)
    print(f"  -> {csv_out}")
    plot_picked(proj, n)
    return 0


# ═══════════════════════════ picked-features ════════════════════════════════
def _get_feature_vals(user, gaia_ids, replot, feat_cache):
    if feat_cache.exists() and replot:
        return pd.read_csv(feat_cache, dtype={"gaia_source_id": "int64"})
    if replot:
        print(f"  --replot needs {feat_cache.name} — run once without it."); sys.exit(1)
    feats = fetch_features([int(x) for x in gaia_ids], user).rename(columns={"source_id": "gaia_source_id"})
    for c in SELECTED_GAIA_PARAMETERS:
        feats[c] = pd.to_numeric(feats[c], errors="coerce")
    feat_cache.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(feat_cache, index=False)
    print(f"  cached feature values -> {feat_cache}")
    return feats


def cmd_picked_features(args):
    n = args.n_sample
    proj_csv = arrays_dir(n) / "picked_double.csv"
    if not proj_csv.exists():
        print(f"  {proj_csv.name} not found — run `picked --n-sample {n}` first."); return 1
    proj = pd.read_csv(proj_csv, dtype={"gaia_source_id": "int64"})
    user = os.environ.get("PGUSER", "siwei_zhao_2025")
    feats = _get_feature_vals(user, proj["gaia_source_id"].unique(), args.replot,
                              arrays_dir(n) / "picked_feature_vals.csv")
    for s in (SAMPLES if args.all_samples else [args.sample]):
        pk = proj[proj["sample"] == s].merge(feats, on="gaia_source_id", how="left")
        if not len(pk):
            print(f"  no projected picks for {s}"); continue
        print(f"=== {s} (n={nlabel(n)}): {len(pk)} picks ===")
        emb = np.load(arrays_dir(n) / f"{s}_qso_gaia.npz", allow_pickle=True)["embedding"]
        title = (f"{len(pk)} hand-picked QSO Gaia sources on the {s.upper()} QSO×Gaia UMAP, "
                 f"coloured by Gaia parameters  (G ≥ {MAG_CUT}, {len(SELECTED_GAIA_PARAMETERS)} features)")
        _feature_grid(emb, pk[list(SELECTED_GAIA_PARAMETERS)].to_numpy(float), title,
                      out_dir(n) / f"{s}_picked_features.png",
                      big_stars=True, star_xy=pk[["umap1", "umap2"]].to_numpy(float))
    return 0


CMDS = {"features": cmd_features, "picked": cmd_picked, "picked-features": cmd_picked_features}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=list(CMDS) + ["all"], default="all",
                    help="which output(s) to produce (default: all three).")
    ap.add_argument("--sample", choices=SAMPLES, default="desi",
                    help="features / picked-features: a single sample (default desi).")
    ap.add_argument("--all-samples", action="store_true",
                    help="features / picked-features: desi, sdss AND combined.")
    ap.add_argument("--n-sample", type=parse_n, default=480_000)
    ap.add_argument("--replot", action="store_true",
                    help="picked / picked-features: reuse cached CSV / feature values (no WSDB).")
    args = ap.parse_args()
    steps = list(CMDS) if args.cmd == "all" else [args.cmd]
    if args.cmd == "all":
        args.all_samples = True   # 'all' = everything: all 3 outputs over all 3 samples
    rc = 0
    for c in steps:
        rc = CMDS[c](args) or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
