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

Writes:
  output/umap/qso/arrays/qso_gaia_matches.csv     (per-QSO: ids, features, umap, cluster, model_pgal)
  output/umap/qso/arrays/qso_gaia_umap.npz
  output/umap/qso/umap_qso_overlay.png

Usage:
  python src/analyze_umap_qso.py                 # default: desisdss (DESI+SDSS) + euclidwise
  python src/analyze_umap_qso.py --catalog all   # also adds quaia
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
QSO_ARRAYS      = OUTPUT_DIR / "arrays"                        # projected CSVs + NPZs

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
WSDB_USER = "siwei_zhao_2025"

MATCH_RADIUS_ARCSEC = 1.0

CLUSTER_COLOURS = {0: "#e6194b", 1: "#4363d8", 2: "#f58231", 3: "#3cb44b", 4: "#911eb4"}

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
    "euclidwise": "crossmatch_qso_euclid_wise_gaia",   # Euclid+WISE (W1-W2>=0.8) photometric QSOs
    "euclidwise_all": "crossmatch_euclid_wise_gaia",   # ALL Euclid×Gaia with a WISE match (no colour cut)
    "quaia": "crossmatch_quaia_gaia",                  # Quaia Gaia-native quasar catalogue
}
# Per-catalog projected-CSV suffix (sdss keeps the original bare name for back-compat).
CSV_SUFFIX = {"sdss": "", "desi": "_desi", "both": "_desisdss",
              "euclidwise": "_euclidwise", "euclidwise_all": "_euclidwise_all", "quaia": "_quaia"}

# Overlay-figure filename tag per catalog. sdss / desi build their CSVs but get NO
# standalone overlay — use --catalog both for the single combined DESI+SDSS figure
# (umap_qso_overlay_desisdss.png). None => don't draw an individual overlay.
OVERLAY_TAG = {"sdss": None, "desi": None, "both": "desisdss",
               "euclidwise": "euclidwise", "euclidwise_all": "euclidwise_all",
               "quaia": None}   # quaia: CSV only, no overlay (for now)


def _csv_for(catalog: str) -> Path:
    return QSO_ARRAYS / f"qso_gaia_matches{CSV_SUFFIX[catalog]}.csv"


def _read_one_table(user: str, table_name: str) -> pd.DataFrame | None:
    """Read one per-catalog crossmatch table, aliasing g_<feature> → plain names."""
    import psycopg2
    table = f"{user}.{table_name}"
    feat_cols = ", ".join(f"g_{c} AS {c}" for c in SELECTED_GAIA_PARAMETERS)
    # Both Euclid×WISE tables carry the WISE colour (w1/w2/w1_w2 — for the W1-W2 panel) and
    # the 'euclid' Q1/ERO tag. Only the QSO subset (crossmatch_qso_euclid_wise_gaia) also has
    # the Euclid-deblend flag n_euclid_within_1arcsec; the uncut base does not.
    euclid_blend = ""
    if "euclid_wise" in table_name:
        euclid_blend = ", w1, w2, w1_w2, euclid"
        if "qso_euclid_wise" in table_name:
            euclid_blend += ", n_euclid_within_1arcsec"
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

    fig = plt.figure(figsize=(26, 12.5))
    gs = fig.add_gridspec(2, 4, left=0.03, right=0.965, top=0.9, bottom=0.03,
                          wspace=0.17, hspace=0.14)
    # Row 1 (astrometry): all-QSO params-solved  +  the three per-class density facets.
    # Row 2 (assignments): P(gal) | blend | cluster | inter-cluster streams.
    aax = fig.add_subplot(gs[0, 0])       # all QSOs by astrometric_params_solved (row1 col1)
    dax = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])]
    pax = fig.add_subplot(gs[1, 0])       # XGB P(gal)
    bax = fig.add_subplot(gs[1, 1])       # blend flag
    kax = fig.add_subplot(gs[1, 2])       # assigned cluster
    sax = fig.add_subplot(gs[1, 3])       # inter-cluster streams

    # ── Row 1: density facets by astrometric_params_solved (SHARED log colour scale) ──
    hbs = []
    for ax, aps in zip(dax, (3, 31, 95)):
        _base(ax)
        sub = df[df["astrometric_params_solved"] == aps]
        if len(sub):
            hb = ax.hexbin(sub["umap1"], sub["umap2"], gridsize=150, cmap="viridis",
                           mincnt=1, alpha=0.7,
                           extent=[xlim[0], xlim[1], ylim[0], ylim[1]], zorder=2)
            hbs.append(hb)
        ax.set_title(f"{APS_LABEL[aps]}  density", fontsize=13, fontweight="bold")
        # centroid labels show THIS panel's cluster fractions (% of this class in each cluster)
        for c in cluster_ids:
            cx, cy = centroids[c]
            p = 100 * (sub["cluster"] == c).mean() if len(sub) else 0.0
            t = ax.text(cx, cy, f"C{c}\n{p:.0f}%", fontsize=8.5, fontweight="bold",
                        ha="center", va="center", color=CLUSTER_COLOURS[c], zorder=7)
            t.set_path_effects([pe.Stroke(linewidth=2, foreground="white"), pe.Normal()])
    # one shared log norm across all three facets → directly comparable densities
    if hbs:
        vmax = max(h.get_array().max() for h in hbs)
        shared = mpl.colors.LogNorm(vmin=1, vmax=vmax)
        for h in hbs:
            h.set_norm(shared)
        cb = fig.colorbar(hbs[-1], ax=dax, fraction=0.025, pad=0.02)
        cb.set_label("QSO count (log, shared scale)", fontsize=9)
        cb.ax.tick_params(labelsize=8)

    # ── Row 2a: XGB P(gal) ──
    ax = pax; _base(ax)
    sc = ax.scatter(df["umap1"], df["umap2"], s=4, c=df["model_pgal"], cmap="RdYlBu_r",
                    vmin=0, vmax=1, alpha=0.5, linewidths=0, rasterized=True, zorder=2)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("XGB P(gal)", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_title("XGB P(gal)", fontsize=13, fontweight="bold"); _clusters(ax)
    # per-cluster fraction the model calls galaxy (P(gal) ≥ 0.5)
    lines = ["QSOs with P(gal) ≥ 0.5:"]
    for c in cluster_ids:
        sub = df[df["cluster"] == c]
        frac = 100 * (sub["model_pgal"] >= 0.5).mean() if len(sub) else 0.0
        lines.append(f"  C{c}: {frac:>3.0f}%   (N={len(sub):,})")
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=8.5,
            va="top", ha="left", zorder=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.92))

    # ── Row 2b: blend flag, else the Euclid+WISE W1-W2 colour map ──
    # Panel routing by catalog, keyed on which columns the projection carries:
    #   euclidwise      (QSO cut, has n_euclid_within_1arcsec) -> EUCLID-DEBLEND blend flag
    #                    (n_gaia is trivially 1 here; the meaningful blend is one Gaia source
    #                     that Euclid splits into >1 MER object — the euclid_wise_g_blend set),
    #                    drawn in the same single/multi star layout as the desisdss panel.
    #   euclidwise_all  (all WISE matches, has w1_w2 but no n_euclid) -> W1-W2 colour map.
    #   desisdss/sdss/…  (neither) -> the Gaia blend flag n_gaia_within_1arcsec.
    ax = bax; _base(ax)
    if "n_euclid_within_1arcsec" in df.columns:
        bcol = "n_euclid_within_1arcsec"
        single = df[df[bcol] == 1]
        multi  = df[df[bcol] > 1]
        ax.scatter(single["umap1"], single["umap2"], s=4, c="#9ecae1", alpha=0.3,
                   linewidths=0, rasterized=True, zorder=2,
                   label=f"single Euclid MER ({len(single):,})")
        ax.scatter(multi["umap1"], multi["umap2"], s=60, c="#e6194b", marker="*",
                   edgecolors="k", linewidths=0.4, zorder=5,
                   label=f"Euclid deblend: >1 within 1\" ({len(multi):,})")
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
        ax.set_title("blend flag (# Euclid MER detections within 1\")",
                     fontsize=12, fontweight="bold")
        _clusters(ax)
    elif "w1_w2" in df.columns and df["w1_w2"].notna().any():
        import matplotlib.colors as mcolors
        w = df["w1_w2"].to_numpy(dtype=float)
        lo, hi = np.nanpercentile(w, [1, 99])
        lo = min(lo, 0.79); hi = max(hi, 0.81)          # keep the 0.8 cut inside the diverging norm
        norm = mcolors.TwoSlopeNorm(vmin=lo, vcenter=0.8, vmax=hi)   # white = 0.8 = Stern+12 AGN cut
        sc = ax.scatter(df["umap1"], df["umap2"], s=4, c=w, cmap="RdBu_r", norm=norm,
                        alpha=0.5, linewidths=0, rasterized=True, zorder=2)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("W1 - W2 (Vega)", fontsize=8); cb.ax.tick_params(labelsize=7)
        ax.set_title("Euclid+WISE colour  W1 - W2  (white = 0.8 AGN cut)",
                     fontsize=12, fontweight="bold")
        _clusters(ax)
        n_fin = int(np.isfinite(w).sum()); n_agn = int(np.nansum(w >= 0.8))
        ax.text(0.03, 0.97, f"W1-W2 ≥ 0.8 (AGN wedge): {n_agn:,}/{n_fin:,}",
                transform=ax.transAxes, fontsize=9, va="top", ha="left", zorder=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.92))
    else:
        bcol = "n_gaia_within_1arcsec"
        single = df[df[bcol] == 1]
        multi  = df[df[bcol] > 1]
        ax.scatter(single["umap1"], single["umap2"], s=4, c="#9ecae1", alpha=0.3,
                   linewidths=0, rasterized=True, zorder=2,
                   label=f"single Gaia match ({len(single):,})")
        ax.scatter(multi["umap1"], multi["umap2"], s=60, c="#e6194b", marker="*",
                   edgecolors="k", linewidths=0.4, zorder=5,
                   label=f"blend: >1 within 1\" ({len(multi):,})")
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
        ax.set_title("blend flag (# Gaia detections within 1\")", fontsize=12, fontweight="bold")
        _clusters(ax)

    # ── Row 2c: all QSOs coloured by astrometric_params_solved (viridis + discrete cbar) ──
    ax = aax; _base(ax)
    cmap3 = plt.get_cmap("viridis", 3)          # 3 = 2-param, 31 = 5-param, 95 = 6-param
    aps_color = {3: cmap3(0), 31: cmap3(1), 95: cmap3(2)}
    for aps in (95, 31, 3):  # draw 2-param last so it sits on top
        s = df[df["astrometric_params_solved"] == aps]
        ax.scatter(s["umap1"], s["umap2"], s=4, color=aps_color[aps], alpha=0.5,
                   linewidths=0, rasterized=True, zorder=2)
    ax.set_title("all QSOs by astrometric_params_solved", fontsize=13, fontweight="bold")
    _clusters(ax)
    # discrete colorbar, 3 segments labelled by # solved params (2 / 5 / 6)
    sm = mpl.cm.ScalarMappable(cmap=cmap3, norm=mpl.colors.BoundaryNorm([0, 1, 2, 3], 3))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, ticks=[0.5, 1.5, 2.5])
    cb.set_ticklabels(["2", "5", "6"])
    cb.set_label("# astrometric params solved", fontsize=8)
    cb.ax.tick_params(labelsize=8)

    # ── Row 2d: QSOs coloured by their assigned UMAP cluster (C0–C4) ──
    ax = kax; _base(ax)
    for c in cluster_ids:
        s = df[df["cluster"] == c]
        if len(s):
            ax.scatter(s["umap1"], s["umap2"], s=4, color=CLUSTER_COLOURS[c],
                       alpha=0.55, linewidths=0, rasterized=True, zorder=2)
    ax.set_title("QSOs by assigned UMAP cluster", fontsize=13, fontweight="bold")
    _clusters(ax)
    _frac_box(ax, df)   # N + per-cluster membership fractions (≥0.5%)

    # ── Row 2e: OUTSIDE QSOs relabelled into named inter-cluster streams ──
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

    cats = "+".join(sorted(t.split("_")[0] for t in df["qso_catalog"].unique()))
    n2 = int((df["astrometric_params_solved"] == 3).sum())
    n2c1 = int(((df["astrometric_params_solved"] == 3) & (df["cluster"] == 1)).sum())
    fig.suptitle(
        f"{len(df):,} QSO–Gaia matches ({cats}) projected onto the Q1 14-feature UMAP",
        fontsize=16, fontweight="bold", x=0.5, y=0.96)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"umap_qso_overlay_{tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


# Named catalog SETS. Bare `analyze_umap_qso.py` (default) compiles the DESI+SDSS combo,
# the Euclid+WISE QSO cut, AND the full WISE-matched base; 'all' also adds Quaia.
CATALOG_SETS = {"default": ["both", "euclidwise", "euclidwise_all"],
                "all":     ["both", "euclidwise", "euclidwise_all", "quaia"]}


def _write_and_plot(df: pd.DataFrame, catalog: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    QSO_ARRAYS.mkdir(parents=True, exist_ok=True)
    csv = _csv_for(catalog)
    df.to_csv(csv, index=False)
    print(f"  -> {csv}")
    np.savez_compressed(
        QSO_ARRAYS / f"qso_gaia_umap{CSV_SUFFIX[catalog]}.npz",
        source_ids=df["gaia_source_id"].astype(str).values,
        qso_catalog=df["qso_catalog"].values,
        umap=df[["umap1", "umap2"]].values,
        cluster=df["cluster"].values,
        astrometric_params_solved=df["astrometric_params_solved"].values,
        model_pgal=df["model_pgal"].values,
        n_gaia_within_1arcsec=df["n_gaia_within_1arcsec"].values,
    )
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
    tag = OVERLAY_TAG[catalog]
    if tag:
        plot_overlay(df, tag)
    else:
        print(f"  (no overlay for {catalog} — CSV only)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Project QSO×Gaia matches onto the UMAP")
    ap.add_argument("--catalog",
                    choices=["sdss", "desi", "both", "euclidwise", "euclidwise_all",
                             "quaia", "default", "all"],
                    default="default",
                    help="'default' = both (DESI+SDSS → desisdss) + euclidwise; "
                         "'euclidwise_all' = every WISE-matched Euclid×Gaia source (no colour "
                         "cut, ~439k); 'all' also adds quaia; or a single catalog. (default: default)")
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
