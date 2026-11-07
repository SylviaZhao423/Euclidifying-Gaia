#!/usr/bin/env python3
"""
Project the QSO × Gaia cross-match onto the Q1 14-feature UMAP and report which
cluster the QSOs land in (esp. Cluster 0, the 95%-galaxy / extragalactic locus).

Reads the per-catalog WSDB tables {PGUSER}.crossmatch_qso_sdss_dr16_gaia and
{PGUSER}.crossmatch_qso_desi_dr1_gaia — build them once with
    PGUSER=siwei_zhao_2025 python src/crossmatch_qso_sdss_dr16_gaia.py
    PGUSER=siwei_zhao_2025 python src/crossmatch_qso_desi_dr1_gaia.py
then this script just SELECTs from them (fast, no re-join).

Science
-------
QSOs are bona-fide extragalactic point sources. Gaia's standard pipeline often
cannot fit them with a full 5/6-parameter astrometric model, so they collapse to
2-parameter (position-only) solutions (astrometric_params_solved = 3). Those are
exactly the sources that populate UMAP Clusters 0 and 1. If known QSOs land in
Cluster 0, that validates C0 as a genuine extragalactic population that Gaia's
normal pipeline misses.

Note on 2-parameter sources: they have NO parallax / proper motion, so
parallax_over_error, pm_significance, and several astrometric-quality features are
NULL. The UMAP standardization imputes those to 0 — so among 2-parameter sources
C0-vs-C1 membership is driven by PHOTOMETRY (colours, flux ratios), which is what
makes "do QSO colours land them in C0?" a meaningful, non-circular test.

Catalogs (both matched, tagged by astrometric_params_solved)
------------------------------------------------------------
  SDSS DR16Q : sdssdr16qso.main                 (is_qso_final = 1)
  DESI DR1   : vasily.desi_dr1_qso_zpix_v1
Closest Gaia source per QSO is kept; n_gaia_within_1arcsec flags blends
(Vasily: ~80 SDSS / ~180 DESI QSOs have >1 Gaia detection within 1", many are
QSO-star blends identifiable by colour).

Requires WSDB — connect the Astronomy or Cambridge VPN first (plain eduroam
blocks port 5432). Auth via ~/.pgpass (no password argument).

Reads:
  output/umap/arrays/q1_indiv.npz          (mean, std, embedding, X_raw)
  output/umap/arrays/q1_cluster_labels.npy
  output/umap/arrays/q1_indiv_reducer.joblib
  output/xgboost_q1/seed_models/xgb_classifier_q1_seed*.json   (model_pgal)

Writes (per catalog with an OVERLAY_TAG; TWO figures each):
  output/umap/qso/arrays/qso_gaia_matches{suffix}.csv   (ids, features, umap, cluster, model_pgal)
  output/umap/qso/umap_qso_{tag}_clusters.png           (aps | cluster | streams)
  output/umap/qso/umap_qso_{tag}.png                    (blend flag | XGB P(gal))

Usage:
  python src/analyze_umap_qso.py                 # default: desisdss (DESI+SDSS) + euclidwise
  python src/analyze_umap_qso.py --catalog sdss  # a single catalog only
  python src/analyze_umap_qso.py --replot        # fast: just redraw overlay(s) from saved CSV(s)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS  # noqa: E402

ARRAYS_DIR      = PROJECT_DIR / "output" / "umap" / "arrays"
SEED_MODELS_DIR = PROJECT_DIR / "output" / "xgboost_q1" / "seed_models"
OUTPUT_DIR      = PROJECT_DIR / "output" / "umap" / "qso"      # overlay PNGs
QSO_ARRAYS      = OUTPUT_DIR / "arrays"                        # projected CSVs

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
WSDB_USER = "siwei_zhao_2025"

MATCH_RADIUS_ARCSEC = 1.0

CLUSTER_COLOURS = {0: "#e6194b", 1: "#4363d8", 2: "#f58231", 3: "#3cb44b", 4: "#911eb4"}
# Sequential colormap per cluster, hue-matched to CLUSTER_COLOURS (for per-cluster density).
CLUSTER_CMAPS = {0: "Reds", 1: "Blues", 2: "Oranges", 3: "Greens", 4: "Purples"}

# Inter-cluster "bridges"/streams: single source of truth in umap_streams.py, shared
# with analyze_umap_nsa.py and download_ds_g_bridges_cutouts.py. Edit corridors there.
from umap_streams import (outside_mask, stream_labels, STREAM_PAIRS,  # noqa: E402,F401
                          OUTSIDE_KNN_K, OUTSIDE_KNN_THR, STREAM_CAP)

# astrometric_params_solved: 3 = 2-param (pos only), 31 = 5-param, 95 = 6-param
APS_COLOURS = {3: "#7b2d8b", 31: "#1f9e89", 95: "#f5d000"}
APS_LABEL   = {3: "2-param (pos only)", 31: "5-param", 95: "6-param"}

# Per-catalog WSDB tables (under the PGUSER schema). Built by the two
# crossmatch_*_qso_gaia.py scripts.
CATALOG_TABLES = {
    "sdss": "crossmatch_qso_sdss_dr16_gaia",
    "desi": "crossmatch_qso_desi_dr1_gaia",
    "euclidwise": "crossmatch_qso_catwise_euclid_gaia",  # Euclid+WISE (W1-W2>=0.8) photometric QSOs
}
# Per-catalog projected-CSV suffix (sdss keeps the original bare name for back-compat).
CSV_SUFFIX = {"sdss": "", "desi": "_desi", "both": "_desisdss", "euclidwise": "_euclidwise"}

# Overlay-figure filename tag per catalog. sdss / desi build their CSVs but get NO standalone
# overlay — use --catalog both for the combined DESI+SDSS figures
# (umap_qso_desisdss.png + umap_qso_desisdss_clusters.png). None => no overlay.
OVERLAY_TAG = {"sdss": None, "desi": None, "both": "desisdss", "euclidwise": "euclidwise"}


def _csv_for(catalog: str) -> Path:
    return QSO_ARRAYS / f"qso_gaia_matches{CSV_SUFFIX[catalog]}.csv"


def _read_one_table(user: str, table_name: str) -> pd.DataFrame | None:
    """Read one per-catalog crossmatch table, aliasing g_<feature> → plain names."""
    import psycopg2
    table = f"{user}.{table_name}"
    feat_cols = ", ".join(f"g_{c} AS {c}" for c in SELECTED_GAIA_PARAMETERS)
    # The CATWise×Euclid×Gaia table carries WISE mags (w1mpro/w2mpro), the Q1/ERO tag
    # (euclid_source, aliased to 'euclid' for downstream compat with euclid_release()),
    # and the Euclid-deblend flag n_euclid_within_1arcsec (the blend-flag panel).
    euclid_blend = ""
    if "catwise_euclid" in table_name:
        euclid_blend = (", w1mpro AS w1, w2mpro AS w2, w1mpro - w2mpro AS w1_w2"
                        ", euclid_source AS euclid, n_euclid_within_1arcsec")
    sql = f"""
        SELECT qso_catalog, qso_id, qso_ra, qso_dec, qso_z,
               g_source_id                 AS gaia_source_id,
               gaia_sep_arcsec, n_gaia_within_1arcsec{euclid_blend},
               g_astrometric_params_solved AS astrometric_params_solved,
               g_phot_g_mean_mag           AS phot_g_mean_mag,
               g_phot_g_n_obs              AS phot_g_n_obs,
               {feat_cols}
        FROM {table}
    """
    print(f"  reading {table} ...")
    try:
        conn = psycopg2.connect(
            host=WSDB_HOST, database=WSDB_DB, user=user,
            connect_timeout=30, keepalives=1, keepalives_idle=30,
            keepalives_interval=10, keepalives_count=5,
        )
        df = pd.read_sql_query(sql, conn)
        conn.close()
    except Exception as exc:
        print(f"  skipped {table} ({exc}).")
        return None
    print(f"    {len(df):,} rows")
    return df


def load_matches_from_table(catalog: str = "both") -> pd.DataFrame | None:
    """Read the per-catalog crossmatch table(s). For 'both', concatenates whichever
    tables exist (missing ones are skipped with a warning). Returns None if none
    could be read."""
    import os
    user = os.getenv("PGUSER", WSDB_USER)
    which = ["sdss", "desi"] if catalog == "both" else [catalog]
    frames = [d for d in (_read_one_table(user, CATALOG_TABLES[k]) for k in which)
              if d is not None and len(d)]
    if not frames:
        print("  No crossmatch tables found. Build them with "
              "crossmatch_qso_sdss_dr16_gaia.py / crossmatch_qso_desi_dr1_gaia.py (needs VPN).")
        return None
    return pd.concat(frames, ignore_index=True)


def _load_reducer():
    """Load the saved UMAP reducer, shimming a pynndescent version skew:
    newer pynndescent's NNDescent.__setstate__ references attributes (e.g.
    `quantization`) absent on reducers pickled by older versions. Providing
    class-level defaults lets `self.<attr>` resolve during unpickling."""
    import pynndescent
    for attr, default in [("quantization", None), ("_quantized_data", None),
                          ("parallel_batch_queries", False)]:
        if not hasattr(pynndescent.NNDescent, attr):
            setattr(pynndescent.NNDescent, attr, default)
    return joblib.load(ARRAYS_DIR / "q1_indiv_reducer.joblib")


def _project_knn(Xs: np.ndarray, Xtr: np.ndarray, q1_emb: np.ndarray, k: int = 15) -> np.ndarray:
    """Project via distance-weighted average of the k nearest TRAINING points'
    embeddings in standardized feature space. This is exactly how UMAP.transform
    initializes new points, so it is a faithful, reducer-free projection — robust
    to the pynndescent version skew that breaks unpickling the saved reducer."""
    nn = NearestNeighbors(n_neighbors=k).fit(Xtr)
    dist, idx = nn.kneighbors(Xs)
    w = 1.0 / (dist + 1e-6)
    w /= w.sum(axis=1, keepdims=True)
    return np.einsum("nk,nkd->nd", w, q1_emb[idx])


def project_and_assign(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the 14 features with q1 mean/std, project onto the UMAP, assign
    the nearest q1 cluster, and score model_pgal with the XGB seed ensemble."""
    q1        = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    labels    = np.load(ARRAYS_DIR / "q1_cluster_labels.npy")
    q1_emb    = q1["embedding"]
    q1_X_raw  = q1["X_raw"].astype(float)
    mean, std = q1["mean"], q1["std"]
    std_safe  = np.where(std > 0, std, 1.0)

    X = df[list(SELECTED_GAIA_PARAMETERS)].apply(pd.to_numeric, errors="coerce").values.astype(float)

    # ── Standardize + impute (same as the allphz projection) ──
    Xs  = (X - mean) / std_safe
    Xs  = np.where(np.isfinite(Xs), Xs, 0.0)

    # ── Project: real UMAP.transform if the reducer loads, else k-NN fallback ──
    try:
        reducer = _load_reducer()
        emb = reducer.transform(Xs).astype(float)
        print("  projection: umap.transform (saved reducer)")
    except Exception as exc:
        print(f"  reducer unavailable ({type(exc).__name__}: {exc});")
        print("  projection: k-NN neighbour-average (UMAP transform init) — reducer-free fallback")
        Xtr = np.where(np.isfinite((q1_X_raw - mean) / std_safe), (q1_X_raw - mean) / std_safe, 0.0)
        emb = _project_knn(Xs, Xtr, q1_emb)

    valid = labels >= 0
    nn = NearestNeighbors(n_neighbors=1).fit(q1_emb[valid])
    _, idx = nn.kneighbors(emb)
    clusters = labels[np.where(valid)[0][idx.ravel()]]

    df = df.copy()
    df["umap1"], df["umap2"] = emb[:, 0], emb[:, 1]
    df["cluster"] = clusters

    # ── XGB model_pgal (raw features, NaN → q1 training-column median) ──
    df["model_pgal"] = _xgb_pgal(X, q1_X_raw)
    return df


def _xgb_pgal(X: np.ndarray, q1_X_raw: np.ndarray) -> np.ndarray:
    seed_paths = sorted(SEED_MODELS_DIR.glob("xgb_classifier_q1_seed*.json"))
    if not seed_paths:
        print(f"  WARNING: no seed models in {SEED_MODELS_DIR} — model_pgal = NaN")
        return np.full(len(X), np.nan)
    import xgboost as xgb
    medians = np.nanmedian(q1_X_raw, axis=0)
    Xi = X.astype(np.float32).copy()
    for c in range(Xi.shape[1]):
        bad = ~np.isfinite(Xi[:, c])
        if bad.any():
            Xi[bad, c] = medians[c]
    probs = []
    for p in seed_paths:
        m = xgb.XGBClassifier()
        m.load_model(str(p))
        probs.append(m.predict_proba(Xi)[:, 1])
    return np.mean(probs, axis=0)


def report(df: pd.DataFrame) -> None:
    """Print cluster-membership breakdowns, overall and sliced."""
    def _frac_table(sub: pd.DataFrame, title: str) -> None:
        n = len(sub)
        if n == 0:
            print(f"  {title}: (none)")
            return
        vc = sub["cluster"].value_counts().sort_index()
        parts = "  ".join(f"C{c}={vc.get(c, 0):>4} ({100*vc.get(c,0)/n:4.1f}%)" for c in range(5))
        print(f"  {title:<34} N={n:>6}   {parts}")

    print("\n" + "=" * 96)
    print("QSO → UMAP cluster membership   (C0 = 95%-galaxy extragalactic locus)")
    print("=" * 96)
    _frac_table(df, "ALL QSO-Gaia matches")
    print("\n— by catalog —")
    for tag, sub in df.groupby("qso_catalog"):
        _frac_table(sub, tag)
    print("\n— by astrometric_params_solved —")
    for aps, sub in df.groupby("astrometric_params_solved"):
        _frac_table(sub, APS_LABEL.get(int(aps), f"aps={int(aps)}"))
    print("\n— single-match (clean) vs multi-match (blends) —")
    _frac_table(df[df["n_gaia_within_1arcsec"] == 1], "single Gaia match")
    _frac_table(df[df["n_gaia_within_1arcsec"] > 1],  "multi-match (blend)")
    print("\n— high XGB P(gal) ≥ 0.5 —")
    _frac_table(df[df["model_pgal"] >= 0.5], "model_pgal ≥ 0.5")
    print("=" * 96)
    c0 = df[df["cluster"] == 0]
    print(f"\nHeadline: {len(c0):,}/{len(df):,} ({100*len(c0)/max(len(df),1):.1f}%) "
          f"of QSO-Gaia matches land in Cluster 0.")


def aps_density_panel(fig, ax, df, xlim, ylim, noun="QSOs") -> None:
    """The 'by astrometric_params_solved' panel drawn as per-class 2D DENSITY: each class
    (2/5/6-param) is hexbinned in its OWN sequential colormap (Purples / Greens / YlOrBr),
    so the denser a bin is in that class, the darker its hue — with a stacked 3-segment
    colorbar (one light→dark ramp per class, labelled 2 / 5 / 6). The classes are largely
    spatially segregated on the manifold, so the three layers don't muddy each other.
    (Caller still adds the centroid labels via its own _clusters(ax).)"""
    aps_cmaps = {3: "Purples", 31: "Greens", 95: "Blues"}   # 2 / 5 / 6-param
    ext = [xlim[0], xlim[1], ylim[0], ylim[1]]
    for aps in (95, 31, 3):          # draw 2-param last (compact + dense) → sits on top
        s = df[df["astrometric_params_solved"] == aps]
        if len(s):
            ax.hexbin(s["umap1"], s["umap2"], gridsize=120, cmap=aps_cmaps[aps],
                      bins="log", mincnt=1, alpha=0.9, extent=ext, zorder=2)
    ax.set_title(f"by astrometric_params_solved  (per-class density)",
                 fontsize=13, fontweight="bold")
    # stacked colorbar: three thin light→dark ramps to the right of the panel (6 / 5 / 2 top→bottom)
    pos = ax.get_position()
    order = [95, 31, 3]; lab = {3: "2", 31: "5", 95: "6"}
    gap, w = 0.004, 0.010
    h = (pos.height - (len(order) - 1) * gap) / len(order)
    x0 = pos.x1 + 0.006
    for i, aps in enumerate(order):
        cax = fig.add_axes([x0, pos.y1 - (i + 1) * h - i * gap, w, h])
        sm = mpl.cm.ScalarMappable(cmap=aps_cmaps[aps], norm=mpl.colors.Normalize(0, 1))
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax); cb.set_ticks([])
        cax.set_ylabel(lab[aps], rotation=0, fontsize=9, labelpad=8, va="center", ha="right")
    fig.text(x0 + w + 0.01, pos.y0 + pos.height / 2, "# params (shade = density)",
             rotation=90, va="center", ha="left", fontsize=7.5)


def cluster_density_panel(fig, ax, df, xlim, ylim, cluster_ids, noun="QSOs") -> None:
    """The 'by assigned UMAP cluster' panel drawn as per-cluster 2D DENSITY: each cluster
    is hexbinned in its OWN hue-matched sequential colormap (C0 Reds / C1 Blues / C2 Oranges
    / C3 Greens / C4 Purples — same hues as CLUSTER_COLOURS), so the denser a bin is in that
    cluster, the darker its hue. Adds a stacked colorbar (one light→dark ramp per cluster,
    labelled C0..C4). Caller still adds the centroid labels + fraction box."""
    ext = [xlim[0], xlim[1], ylim[0], ylim[1]]
    for c in cluster_ids:
        s = df[df["cluster"] == c]
        if len(s):
            ax.hexbin(s["umap1"], s["umap2"], gridsize=120, cmap=CLUSTER_CMAPS.get(c, "Greys"),
                      bins="log", mincnt=1, alpha=0.9, extent=ext, zorder=2)
    ax.set_title(f"by assigned UMAP cluster  (per-cluster density)",
                 fontsize=13, fontweight="bold")
    pos = ax.get_position()
    order = list(cluster_ids)
    gap, w = 0.003, 0.010
    h = (pos.height - (len(order) - 1) * gap) / len(order)
    x0 = pos.x1 + 0.006
    for i, c in enumerate(order):
        cax = fig.add_axes([x0, pos.y1 - (i + 1) * h - i * gap, w, h])
        sm = mpl.cm.ScalarMappable(cmap=CLUSTER_CMAPS.get(c, "Greys"), norm=mpl.colors.Normalize(0, 1))
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax); cb.set_ticks([])
        cax.set_ylabel(f"C{c}", rotation=0, fontsize=9, labelpad=15, va="center", ha="right")
    fig.text(x0 + w + 0.014, pos.y0 + pos.height / 2, "cluster (shade = density)",
             rotation=90, va="center", ha="left", fontsize=7.5)


def plot_overlay(df: pd.DataFrame, tag: str = "sdss") -> None:
    q1     = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    labels = np.load(ARRAYS_DIR / "q1_cluster_labels.npy")
    q1_emb = q1["embedding"]
    cluster_ids = sorted(set(labels[labels >= 0]))
    y = q1["y"]
    centroids = {c: (q1_emb[labels == c, 0].mean(), q1_emb[labels == c, 1].mean()) for c in cluster_ids}
    galpct    = {c: 100 * (y[labels == c] == 1).sum() / (labels == c).sum() for c in cluster_ids}
    xlim = (q1_emb[:, 0].min() - 0.8, q1_emb[:, 0].max() + 0.8)
    ylim = (q1_emb[:, 1].min() - 0.8, q1_emb[:, 1].max() + 0.8)

    def _base(ax):
        ax.scatter(q1_emb[:, 0], q1_emb[:, 1], s=2, c="#e2e2e2", alpha=0.6,
                   linewidths=0, rasterized=True, zorder=1)
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_xticks([]); ax.set_yticks([])

    def _clusters(ax):   # second-row centroid labels: cluster id only (no %gal)
        for c in cluster_ids:
            cx, cy = centroids[c]
            t = ax.text(cx, cy, f"C{c}", fontsize=9, fontweight="bold",
                        ha="center", va="center", color=CLUSTER_COLOURS[c], zorder=7)
            t.set_path_effects([pe.Stroke(linewidth=2, foreground="white"), pe.Normal()])

    def _frac_box(ax, sub, extra=None):
        n = len(sub)
        lines = [f"N = {n:,}"]
        if n:
            vc = sub["cluster"].value_counts()
            for c in sorted(cluster_ids, key=lambda k: -vc.get(k, 0)):
                p = 100 * vc.get(c, 0) / n
                if p >= 0.5:
                    lines.append(f"C{c}: {p:.0f}%")
        if extra:
            lines.append(extra)
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=9,
                va="top", ha="left", zorder=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.92))

    DISPLAY = {"desisdss": "DESI+SDSS", "euclidwise": "Euclid⨯WISE"}
    cats = DISPLAY.get(tag, "+".join(sorted(t.split("_")[0] for t in df["qso_catalog"].unique())))
    suptitle = f"{len(df):,} QSO–Gaia matches ({cats}) projected onto the Q1 UMAP"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ===================== Figure 1: umap_qso_{tag}_clusters.png =====================
    # Row-1 panels: astrometric_params_solved | assigned UMAP cluster | inter-cluster streams
    fig = plt.figure(figsize=(19.5, 7))
    gs = fig.add_gridspec(1, 3, left=0.03, right=0.965, top=0.86, bottom=0.05, wspace=0.17)
    aax = fig.add_subplot(gs[0, 0]); kax = fig.add_subplot(gs[0, 1]); sax = fig.add_subplot(gs[0, 2])

    # ── all QSOs by astrometric_params_solved — per-class 2D density (denser = darker) ──
    ax = aax; _base(ax)
    aps_density_panel(fig, ax, df, xlim, ylim, noun="QSOs")
    _clusters(ax)

    # ── QSOs by assigned UMAP cluster — per-cluster 2D density (denser = darker) ──
    ax = kax; _base(ax)
    cluster_density_panel(fig, ax, df, xlim, ylim, cluster_ids, noun="QSOs")
    _clusters(ax)
    _frac_box(ax, df)   # N + per-cluster membership fractions (≥0.5%)

    # ── OUTSIDE QSOs relabelled into named inter-cluster streams ──
    ax = sax; _base(ax)
    pts = df[["umap1", "umap2"]].to_numpy()
    outside = outside_mask(pts, q1_emb)
    slab = stream_labels(pts, outside, centroids)
    ax.scatter(pts[~outside, 0], pts[~outside, 1], s=2, c="#d8d8d8", alpha=0.2,
               linewidths=0, rasterized=True, zorder=2)      # core QSOs, faded
    for k, (name, _verts, col) in enumerate(STREAM_PAIRS):   # named streams
        m = slab == k
        ax.scatter(pts[m, 0], pts[m, 1], s=6, color=col, alpha=0.85, linewidths=0,
                   rasterized=True, zorder=5, label=f"{name}: {int(m.sum()):,}")
    mo = slab == -1                                          # outside but unassigned
    ax.scatter(pts[mo, 0], pts[mo, 1], s=3, c="#9a9a9a", alpha=0.4, linewidths=0,
               rasterized=True, zorder=4, label=f"other: {int(mo.sum()):,}")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title(f"inter-cluster streams  ({int(outside.sum()):,} outside Q1 cores)",
                 fontsize=12, fontweight="bold")
    _clusters(ax)

    fig.suptitle(suptitle, fontsize=16, fontweight="bold", x=0.5, y=0.97)
    out1 = OUTPUT_DIR / f"umap_qso_{tag}_clusters.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight"); plt.close(); print(f"  -> {out1}")

    # ===================== Figure 2: umap_qso_{tag}.png =====================
    # Row-2 panels, swapped order: blend flag | XGB P(gal)
    fig = plt.figure(figsize=(13, 6.8))
    gs = fig.add_gridspec(1, 2, left=0.04, right=0.955, top=0.86, bottom=0.05, wspace=0.17)
    bax = fig.add_subplot(gs[0, 0]); pax = fig.add_subplot(gs[0, 1])

    # ── blend flag — 2D density (log) background + blend stars on top ──
    # euclidwise QSO cut: the meaningful blend is the EUCLID deblend (one Gaia source Euclid
    # splits into >1 MER object; n_gaia is trivially 1 there). Everything else uses the Gaia
    # blend flag n_gaia_within_1arcsec.
    ax = bax; _base(ax)
    if "n_euclid_within_1arcsec" in df.columns:
        bcol, blabel, noun = "n_euclid_within_1arcsec", "Euclid deblend", "Euclid MER detections"
    else:
        bcol, blabel, noun = "n_gaia_within_1arcsec", "blend", "Gaia detections"
    multi = df[df[bcol] > 1]
    hb = ax.hexbin(df["umap1"], df["umap2"], gridsize=120, cmap="viridis", bins="log", alpha=0.6,
                   mincnt=1, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], zorder=2)
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("QSO count (log)", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.scatter(multi["umap1"], multi["umap2"], s=14, c="red", marker="*", alpha=0.7,
               edgecolors="w", linewidths=0.4, zorder=5,
               label=f"{blabel}: >1 within 1\" ({len(multi):,})")
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.92)   # lower-left: clears the frac box
    ax.set_title(f"blend flag (# {noun} within 1\")", fontsize=12, fontweight="bold")
    _clusters(ax)
    _frac_box(ax, df)   # N + per-cluster fractions (same box as the cluster panel)

    # ── XGB P(gal) ──
    ax = pax; _base(ax)
    sc = ax.scatter(df["umap1"], df["umap2"], s=4, c=df["model_pgal"], cmap="RdYlBu_r",
                    vmin=0, vmax=1, alpha=0.5, linewidths=0, rasterized=True, zorder=2)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("XGB P(gal)", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_title("XGB P(gal)", fontsize=13, fontweight="bold"); _clusters(ax)
    lines = ["QSOs with P(gal) ≥ 0.5:"]
    for c in cluster_ids:
        sub = df[df["cluster"] == c]
        frac = 100 * (sub["model_pgal"] >= 0.5).mean() if len(sub) else 0.0
        lines.append(f"  C{c}: {frac:>3.0f}%   (N={len(sub):,})")
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=8.5,
            va="top", ha="left", zorder=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.92))

    fig.suptitle(suptitle, fontsize=16, fontweight="bold", x=0.5, y=0.97)
    out2 = OUTPUT_DIR / f"umap_qso_{tag}.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(); print(f"  -> {out2}")


# Named catalog set. Bare `analyze_umap_qso.py` (default) draws the two catalogs we keep:
# desisdss (DESI+SDSS) and euclidwise (W1-W2>=0.8 QSO cut), two figures each.
CATALOG_SETS = {"default": ["both", "euclidwise"]}


def _write_and_plot(df: pd.DataFrame, catalog: str) -> None:
    """Save the projected CSV, then draw the overlay if this catalog has one.
    (The old per-catalog qso_gaia_umap*.npz mirror was dropped — nothing read it; the CSV
    is the single source of truth for the projection.)"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    QSO_ARRAYS.mkdir(parents=True, exist_ok=True)
    csv = _csv_for(catalog)
    df.to_csv(csv, index=False)
    print(f"  -> {csv}")
    _maybe_plot(df, catalog)


def _maybe_plot(df: pd.DataFrame, catalog: str) -> None:
    """Draw the overlay for this catalog, or note that it's CSV-only. Shared by the full
    run and --replot so the 'has an overlay?' decision lives in exactly one place."""
    tag = OVERLAY_TAG[catalog]
    if tag:
        plot_overlay(df, tag)
    else:
        print(f"  (no overlay for {catalog} — CSV only)")


def _run_full(catalog: str) -> int:
    print(f"\n=== {catalog} : WSDB → project → plot ===")
    df = load_matches_from_table(catalog)
    if df is None or not len(df):
        print(f"  no {catalog} matches (table missing/empty) — skipped."); return 1
    print(f"  {len(df):,} matches; projecting onto Q1 UMAP + clusters + model_pgal...")
    df = project_and_assign(df)
    report(df)
    _write_and_plot(df, catalog)
    return 0


def _run_replot(catalog: str) -> int:
    csv = _csv_for(catalog)
    if not csv.exists():
        print(f"  {csv.name} not found — run a full projection first (no --replot)."); return 1
    df = pd.read_csv(csv)
    print(f"Replotting {catalog} from {csv.name} ({len(df):,} rows)...")
    _maybe_plot(df, catalog)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Project QSO×Gaia matches onto the UMAP")
    ap.add_argument("--catalog",
                    choices=["sdss", "desi", "both", "euclidwise", "default"],
                    default="default",
                    help="'default' = both (DESI+SDSS → desisdss) + euclidwise (the two "
                         "overlays); or a single catalog. (default: default)")
    ap.add_argument("--replot", action="store_true",
                    help="Skip WSDB + projection; just redraw the overlay(s) from the saved CSV(s).")
    args = ap.parse_args()

    catalogs = CATALOG_SETS.get(args.catalog, [args.catalog])
    run = _run_replot if args.replot else _run_full
    for c in catalogs:
        run(c)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
