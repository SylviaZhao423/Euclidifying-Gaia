#!/usr/bin/env python3
"""Project the WHOLE CatWISE×Gaia QSO sample (siwei_zhao_2025.crossmatch_qso_catwise_gaia) onto
the combined DESI/SDSS QSO×Gaia UMAP and show WHERE that photometric-QSO population lands —
the population-level analogue of analyze_picked_qso_umap.py (which projects only the hand-picks).

Same projection recipe as analyze_picked_qso_umap.py / analyze_qso_umap_ds.py:project_all():
the 14 SELECTED_GAIA_PARAMETERS are read straight from the crossmatch table's g_<name> columns
(no per-source WSDB fetch — the table already carries them), standardised with the build's train
mean/std (stored in its npz), run through the saved reducer.transform, and each projected point is
assigned the HDBSCAN cluster of its nearest labelled UMAP neighbour.

Reads:
  output/qso_umap/base/arrays/combined_qso_gaia.{npz,_reducer.joblib}   (combined build + reducer;
      desi/sdss come from ds_qso/n{N}/arrays — see build_arrays_dir())
  siwei_zhao_2025.crossmatch_qso_catwise_gaia                                    (the QSOs to project)
Each source also gets a Q1 star/galaxy XGBoost P(gal) (the same 10-seed ensemble
deploy_q1_models_on_gaia.py uses, on the RAW 14 features), used to colour the P(gal) panel.

Writes (output/qso_umap/base/ — the curated combined-only bundle, see query_qso_umap_base.py):
  arrays/catwise_gaia_projected.npz   cached projection (source_id, umap1/2, cluster, g_mag, w1_w2,
                                      pgal, feat) — re-plot instantly with --replot
  qso_space_catwise_umap.png          3-panel: 2D density (Blues, log) | all by XGB P(gal) | W1-W2,
                                      over the build UMAP as a grey backdrop (C0..C3 labelled)
  qso_space_catwise_gaia_features.png 14-feature grid (single panel per feature, like
                                      qso_gaia_features.png), projected points coloured by value

Because the reducer was trained on G>=17 spec QSOs, only G>=17 CatWISE QSOs are projected by
default (--gmag-min); use --gmag-min 0 to project everything. Rows are de-duplicated by Gaia
source_id. reducer.transform + nearest-cluster are done in chunks so the 4.4M-row sample fits.

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_qso_umap_space_catwise.py                # full sample
  PGUSER=siwei_zhao_2025 python src/analyze_qso_umap_space_catwise.py --limit 200000 # quick test
  python src/analyze_qso_umap_space_catwise.py --replot                              # plots from cache
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
import psycopg2
from sklearn.neighbors import NearestNeighbors

from analyze_picked_qso_umap import _load_reducer, n_suffix, parse_n, build_arrays_dir
from analyze_qso_umap_ds import CLUSTER_COLOURS
from selected_parameters_list import SELECTED_GAIA_PARAMETERS
from builder_qso_gaia_crossmatch import WSDB_HOST, WSDB_DB

PROJECT_DIR = Path(__file__).resolve().parent.parent
# outputs live under the build they were projected onto: ds_qso/n{N}/catwise/ (set in main())
CW_TABLE    = "crossmatch_qso_catwise_gaia"
# Q1 star/galaxy XGBoost ensemble (predicts P(gal) from the 14 RAW Gaia features), same models
# deploy_q1_models_on_gaia.py runs — averaged over the 10 seeds; galaxies at P(gal) >= 0.5.
XGB_MODEL_DIR = PROJECT_DIR / "output" / "xgboost_q1" / "seed_models"
GAL_THRESH = 0.5


def fetch_catwise(user, gmag_min, limit, table=CW_TABLE):
    """Read g_source_id + G + W1/W2 + the 14 g_<feature> columns from `table` (a CatWISE×Gaia
    crossmatch table), de-duplicated by Gaia source_id and cut at G>=gmag_min."""
    gcols = ", ".join(f"g_{p}" for p in SELECTED_GAIA_PARAMETERS)
    sql = (f"SELECT DISTINCT ON (g_source_id) g_source_id, g_phot_g_mean_mag, "
           f"w1mpro, w2mpro, {gcols} "
           f"FROM {user}.{table} "
           f"WHERE g_source_id IS NOT NULL AND g_phot_g_mean_mag >= %s "
           f"ORDER BY g_source_id")
    if limit:
        sql += f" LIMIT {int(limit)}"
    print(f"  querying {table} (G>={gmag_min}{', LIMIT '+str(limit) if limit else ''}) ...")
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=60)
    df = pd.read_sql_query(sql, conn, params=(gmag_min,))
    conn.close()
    print(f"  {len(df):,} unique Gaia sources read")
    return df


def _chunked(fn, n, chunk, label):
    """Apply fn(lo, hi) over [0,n) in chunks, vstack the results, with progress."""
    out = []
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        out.append(fn(lo, hi))
        print(f"    {label}: {hi:,}/{n:,}")
    return np.vstack(out)


def project(df, npz_path, red_path, chunk=200000):
    """Standardise the 14 features with the build mean/std, reducer.transform in chunks, then
    assign each point the HDBSCAN cluster of its nearest labelled build-UMAP neighbour."""
    d = np.load(npz_path, allow_pickle=True)
    emb = d["embedding"]; labels = d["cluster"]
    mean, std = d["mean"], np.where(d["std"] > 0, d["std"], 1.0)
    X = df[[f"g_{p}" for p in SELECTED_GAIA_PARAMETERS]].apply(pd.to_numeric, errors="coerce").values.astype(float)
    Xs = np.where(np.isfinite((X - mean) / std), (X - mean) / std, 0.0)

    reducer = _load_reducer(red_path)
    pemb = _chunked(lambda lo, hi: reducer.transform(Xs[lo:hi]).astype(np.float32),
                    len(Xs), chunk, "transform")
    valid = labels >= 0
    nn = NearestNeighbors(n_neighbors=1).fit(emb[valid])
    cl = _chunked(lambda lo, hi: labels[np.where(valid)[0][nn.kneighbors(pemb[lo:hi])[1].ravel()]][:, None],
                  len(pemb), chunk, "cluster").ravel()
    return emb, labels, pemb, cl


def _cluster_labels(ax, emb, labels, fontsize=13):
    """C0..C3 centroid labels (same style + small-cluster radial nudging as the picked/gmag
    figures) — small/compact clusters (e.g. the tiny C0 island) get their label pushed radially
    outward so it clears the handful of points instead of sitting on top of them."""
    cl_ids = sorted(int(x) for x in np.unique(labels) if x >= 0)
    if not cl_ids:
        return
    centroids = {c: (float(emb[labels == c, 0].mean()), float(emb[labels == c, 1].mean()))
                 for c in cl_ids}
    cradius = {c: float(np.percentile(np.hypot(emb[labels == c, 0] - centroids[c][0],
                                               emb[labels == c, 1] - centroids[c][1]), 95))
               for c in cl_ids}
    gcen = (float(np.mean([centroids[c][0] for c in cl_ids])),
            float(np.mean([centroids[c][1] for c in cl_ids])))
    span = min(emb[:, 0].max() - emb[:, 0].min(), emb[:, 1].max() - emb[:, 1].min())
    for c in cl_ids:
        cx, cy = centroids[c]; lx, ly = cx, cy
        if cradius[c] < 0.10 * span:
            vx, vy = cx - gcen[0], cy - gcen[1]
            nrm = np.hypot(vx, vy) or 1.0
            offd = cradius[c] + 0.045 * span
            lx, ly = cx + vx / nrm * offd, cy + vy / nrm * offd
        t = ax.text(lx, ly, f"C{c}", fontsize=fontsize, fontweight="bold", ha="center",
                    va="center", color=CLUSTER_COLOURS.get(c, "k"), zorder=9)
        t.set_path_effects([pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()])


def plot(emb, labels, pemb, pgal, w1_w2, n_total, gmag_min, out, sub=120000):
    """3-panel projection over the grey build UMAP: 2D density (Blues, log) | all coloured by
    XGB P(gal) | coloured by WISE W1-W2. Cluster centroids labelled C0..C3."""
    xlim = (emb[:, 0].min() - 0.8, emb[:, 0].max() + 0.8)
    ylim = (emb[:, 1].min() - 0.8, emb[:, 1].max() + 0.8)
    rng = np.arange(len(pemb))
    sidx = rng if len(pemb) <= sub else rng[:: max(1, len(pemb) // sub)]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # panel 1: 2D density of ALL projected CatWISE QSOs (hist2d, Blues, LogNorm)
    axD = axes[0]
    nb = 260
    xe = np.linspace(xlim[0], xlim[1], nb + 1); ye = np.linspace(ylim[0], ylim[1], nb + 1)
    H, _, _ = np.histogram2d(pemb[:, 0], pemb[:, 1], bins=[xe, ye])
    H = np.where(H > 0, H, np.nan)
    pm = axD.pcolormesh(xe, ye, H.T, cmap="Blues", norm=LogNorm(vmin=1, vmax=np.nanmax(H)),
                        rasterized=True)
    fig.colorbar(pm, ax=axD, fraction=0.046, pad=0.02).set_label("Count (log)", fontsize=10)
    axD.set_title(f"CatWISE QSOs projected density  (N={n_total:,})", fontsize=13, fontweight="bold")

    # panel 2: all points coloured by Q1 XGB P(gal)
    axP = axes[1]
    axP.scatter(emb[:, 0], emb[:, 1], s=2, c="lightgrey", alpha=0.15, lw=0, rasterized=True)
    if pgal is not None and np.isfinite(pgal).any():
        sc = axP.scatter(pemb[sidx, 0], pemb[sidx, 1], c=pgal[sidx], cmap="RdYlBu_r", vmin=0, vmax=1,
                         s=4, alpha=0.6, lw=0, rasterized=True)
        fig.colorbar(sc, ax=axP, fraction=0.046, pad=0.02).set_label("XGBoost P(gal)", fontsize=10)
        axP.set_title(f"all — coloured by XGB P(gal)  (N={n_total:,})", fontsize=13, fontweight="bold")
    else:
        axP.set_title("XGB P(gal) unavailable", fontsize=13, fontweight="bold")

    # panel 3: coloured by WISE W1-W2 (AGN heat)
    axW = axes[2]
    axW.scatter(emb[:, 0], emb[:, 1], s=2, c="lightgrey", alpha=0.15, lw=0, rasterized=True)
    ww = w1_w2[sidx]; mok = np.isfinite(ww)
    lo, hi = (np.nanpercentile(ww[mok], [2, 98]) if mok.any() else (0.0, 1.0))
    sc = axW.scatter(pemb[sidx][mok, 0], pemb[sidx][mok, 1], s=4, c=np.clip(ww[mok], lo, hi),
                     cmap="magma", vmin=lo, vmax=hi, alpha=0.6, lw=0, rasterized=True)
    fig.colorbar(sc, ax=axW, fraction=0.046, pad=0.02).set_label("W1 - W2 (mag)", fontsize=10)
    axW.set_title("coloured by WISE W1 - W2", fontsize=13, fontweight="bold")

    for ax in axes:
        _cluster_labels(ax, emb, labels)
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.suptitle(f"CatWISE×Gaia QSOs projected onto the combined QSO×Gaia UMAP  "
                 f"(G >= {gmag_min}, N={n_total:,}; 14 features -> reducer.transform)",
                 fontsize=15, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


def predict_pgal(df):
    """P(gal) per source from the 10-seed Q1 star/galaxy XGB ensemble (mean over seeds), applied
    to the RAW 14 Gaia features (XGBoost handles NaN natively — no standardisation), exactly like
    deploy_q1_models_on_gaia.py. Returns None if the models are missing."""
    import xgboost as xgb
    paths = sorted(XGB_MODEL_DIR.glob("xgb_classifier_q1_seed*.json"))
    if not paths:
        print(f"  [warn] no Q1 XGB seed models in {XGB_MODEL_DIR} — skipping P(gal) plots")
        return None
    boosters = []
    for p in paths:
        b = xgb.Booster(); b.load_model(str(p)); boosters.append(b)
    X = df[[f"g_{p}" for p in SELECTED_GAIA_PARAMETERS]].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    dmat = xgb.DMatrix(X)
    probs = np.vstack([b.predict(dmat) for b in boosters])
    print(f"  P(gal) from {len(boosters)}-seed Q1 XGB ensemble "
          f"(galaxies P(gal)>={GAL_THRESH}: {(probs.mean(0) >= GAL_THRESH).sum():,})")
    return probs.mean(axis=0)


def plot_features(emb, labels, pemb, feat, n_total, gmag_min, out, sub=120000):
    """14-feature grid (single panel per feature, like qso_gaia_features.png): the projected
    CatWISE QSOs coloured by each Gaia parameter (RdYlBu_r, 1-99 pct clip) over the grey build
    UMAP, with C0..C3 centroid labels."""
    xlim = (emb[:, 0].min() - 0.8, emb[:, 0].max() + 0.8)
    ylim = (emb[:, 1].min() - 0.8, emb[:, 1].max() + 0.8)
    rng = np.arange(len(pemb))
    sidx = rng if len(pemb) <= sub else rng[:: max(1, len(pemb) // sub)]
    E, F = pemb[sidx], feat[sidx]
    nfeat = len(SELECTED_GAIA_PARAMETERS); ncols = 4
    nrows = int(np.ceil(nfeat / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.0))
    axes = np.atleast_2d(axes)
    for fi, feat_name in enumerate(SELECTED_GAIA_PARAMETERS):
        ax = axes[fi // ncols, fi % ncols]
        vals = F[:, fi].astype(float); finite = vals[np.isfinite(vals)]
        lo, hi = (np.nanpercentile(finite, [1, 99]) if len(finite) else (0.0, 1.0))
        if hi <= lo:
            hi = lo + 1.0
        ax.scatter(emb[:, 0], emb[:, 1], s=2, c="#e9e9e9", alpha=0.5, lw=0, rasterized=True, zorder=0)
        v = np.isfinite(vals)
        sc = ax.scatter(E[v, 0], E[v, 1], c=np.clip(vals[v], lo, hi), cmap="RdYlBu_r", vmin=lo, vmax=hi,
                        s=5, alpha=0.7, lw=0, rasterized=True, zorder=1)
        _cluster_labels(ax, emb, labels, fontsize=8)   # match qso_gaia_features.png
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.25)
        ax.set_title(feat_name, fontsize=9, fontweight="bold", pad=4)
        fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046).ax.tick_params(labelsize=7)
    for fi in range(nfeat, nrows * ncols):
        axes[fi // ncols, fi % ncols].set_visible(False)
    fig.suptitle(f"CatWISE×Gaia QSOs in the combined QSO×Gaia UMAP coloured by Gaia parameters  "
                 f"(G >= {gmag_min}, {nfeat} features, N={n_total:,})",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


def make_argparser(doc=None) -> argparse.ArgumentParser:
    """Shared CLI (reused by analyze_qso_umap_space_catwise_e.py)."""
    ap = argparse.ArgumentParser(description=doc or __doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", choices=["combined", "desi", "sdss"], default="combined",
                    help="which QSO×Gaia UMAP build to project onto (default combined).")
    ap.add_argument("--n-sample", type=parse_n, default="all",
                    help="build size for the target npz (default 'all').")
    ap.add_argument("--gmag-min", type=float, default=17.0,
                    help="only project CatWISE QSOs with Gaia G >= this (default 17, the build "
                         "domain; use 0 to project everything).")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of CatWISE sources (for quick tests; default all).")
    ap.add_argument("--subsample-plot", type=int, default=120000,
                    help="max points drawn in the scatter panels (density panel always uses all).")
    ap.add_argument("--replot", action="store_true",
                    help="skip the WSDB query + projection and re-plot from the cached npz.")
    return ap


def run(user, args, table, arr_path, plot_path, feat_path, label="CatWISE") -> int:
    """Project the CatWISE×Gaia sources in `table` onto the {sample} build UMAP and write the
    3-panel + feature-grid plots. Shared by the catwise (crossmatch_qso_catwise_gaia) and the
    catwise×euclid (crossmatch_qso_catwise_euclid_gaia) variants — only `table` and the output
    paths differ."""
    adir = build_arrays_dir(args.sample, args.n_sample)
    npz_path = adir / f"{args.sample}_qso_gaia.npz"
    red_path = adir / f"{args.sample}_qso_gaia_reducer.joblib"
    if not (npz_path.exists() and red_path.exists()):
        print(f"  missing {n_suffix(args.n_sample)} arrays for {args.sample} — build it first "
              f"(query_qso_umap_ds.py).")
        return 1

    if args.replot:
        if not arr_path.exists():
            print(f"  no cached projection at {arr_path} — run once without --replot first.")
            return 1
        d = np.load(npz_path, allow_pickle=True)
        c = np.load(arr_path, allow_pickle=True)
        emb, labels = d["embedding"], d["cluster"]
        pemb = np.c_[c["umap1"], c["umap2"]]; n = len(c["umap1"]); gm = float(c["gmag_min"])
        pg = c["pgal"] if "pgal" in c.files else None
        plot(emb, labels, pemb, pg, c["w1_w2"], n, gm, plot_path, sub=args.subsample_plot)
        if "feat" in c.files:
            plot_features(emb, labels, pemb, c["feat"], n, gm, feat_path, sub=args.subsample_plot)
        return 0

    df = fetch_catwise(user, args.gmag_min, args.limit, table=table)
    emb, labels, pemb, cl = project(df, npz_path, red_path)
    pgal = predict_pgal(df)                      # Q1 XGB P(gal) from the raw 14 features (or None)
    feat = df[[f"g_{p}" for p in SELECTED_GAIA_PARAMETERS]].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)

    # cache the projection so --replot is cheap
    arr_path.parent.mkdir(parents=True, exist_ok=True)
    w1_w2 = (pd.to_numeric(df["w1mpro"], errors="coerce")
             - pd.to_numeric(df["w2mpro"], errors="coerce")).values.astype(float)
    g_mag = pd.to_numeric(df["g_phot_g_mean_mag"], errors="coerce").values.astype(float)
    np.savez_compressed(arr_path, source_id=df["g_source_id"].astype("int64").values,
                        umap1=pemb[:, 0], umap2=pemb[:, 1], cluster=cl, g_mag=g_mag,
                        w1_w2=w1_w2, gmag_min=args.gmag_min,
                        pgal=(pgal if pgal is not None else np.full(len(df), np.nan)), feat=feat)
    print(f"  -> cached {arr_path}")

    # cluster fractions to stdout
    tot = max(int((cl >= 0).sum()), 1)
    print(f"\n  nearest-cluster distribution of the projected {label} QSOs:")
    for c in sorted(int(x) for x in np.unique(labels) if x >= 0):
        n = int((cl == c).sum())
        print(f"    C{c}: {n:,}  ({100*n/tot:.1f}%)")

    plot(emb, labels, pemb, pgal, w1_w2, len(df), args.gmag_min, plot_path, sub=args.subsample_plot)
    plot_features(emb, labels, pemb, feat, len(df), args.gmag_min, feat_path, sub=args.subsample_plot)
    return 0


def main() -> int:
    args = make_argparser().parse_args()
    user = os.environ.get("PGUSER", "siwei_zhao_2025")
    base = PROJECT_DIR / "output" / "qso_umap" / "base"    # curated combined-only bundle
    return run(user, args, CW_TABLE,
               base / "arrays" / "catwise_gaia_projected.npz",
               base / "qso_space_catwise_umap.png",
               base / "qso_space_catwise_gaia_features.png",
               label="CatWISE")


if __name__ == "__main__":
    raise SystemExit(main())
