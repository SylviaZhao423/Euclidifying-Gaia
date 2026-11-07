#!/usr/bin/env python3
"""
Project the CATWise × Euclid × Gaia cross-match onto the Q1 14-feature UMAP.

Upstream table (built by crossmatch_qso_catwise_euclid_gaia.py):
    siwei_zhao_2025.crossmatch_qso_catwise_euclid_gaia

Reads (pre-built, no VPN):
  output/umap/arrays/q1_indiv.npz
  output/umap/arrays/q1_cluster_labels.npy
  output/umap/arrays/q1_indiv_reducer.joblib
  output/xgboost_q1/seed_models/xgb_classifier_q1_seed*.json

Writes:
  output/umap/qso/arrays/catwise_euclid_gaia.csv    — all sources with
      gaia_source_id, cluster, model_pgal, w1, w2, w1_w2, euclid, etc.
  output/umap/qso/catwise_euclid_gaia_clusters.png  — UMAP overlay by
      astrometric_params_solved | cluster | inter-cluster streams
  output/umap/qso/catwise_euclid_gaia.png           — blend flag | XGB P(gal)

Usage:
    PGUSER=siwei_zhao_2025 conda run -n base python3 src/analyze_umap_qso_catwise.py
    ... --replot    # fast: redraw figures from saved CSV, no WSDB needed
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.neighbors import NearestNeighbors

PROJECT_DIR     = Path(__file__).resolve().parents[1]
ARRAYS_DIR      = PROJECT_DIR / "output" / "umap" / "arrays"
SEED_MODELS_DIR = PROJECT_DIR / "output" / "xgboost_q1" / "seed_models"
OUTPUT_DIR      = PROJECT_DIR / "output" / "umap" / "qso"
QSO_ARRAYS      = OUTPUT_DIR / "arrays"

sys.path.insert(0, str(Path(__file__).parent))

from selected_parameters_list import SELECTED_GAIA_PARAMETERS   # noqa: E402
from umap_streams import outside_mask, stream_labels, STREAM_PAIRS  # noqa: E402

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

CLUSTER_COLOURS = {0: "#e6194b", 1: "#4363d8", 2: "#f58231", 3: "#3cb44b", 4: "#911eb4"}
CLUSTER_CMAPS   = {0: "Reds",    1: "Blues",   2: "Oranges", 3: "Greens",  4: "Purples"}
APS_LABEL       = {3: "2-param (pos only)", 31: "5-param", 95: "6-param"}


def _load_reducer():
    import pynndescent
    for attr, default in [("quantization", None), ("_quantized_data", None),
                          ("parallel_batch_queries", False)]:
        if not hasattr(pynndescent.NNDescent, attr):
            setattr(pynndescent.NNDescent, attr, default)
    return joblib.load(ARRAYS_DIR / "q1_indiv_reducer.joblib")


def _project_knn(Xs, Xtr, q1_emb, k=15):
    nn = NearestNeighbors(n_neighbors=k).fit(Xtr)
    dist, idx = nn.kneighbors(Xs)
    w = 1.0 / (dist + 1e-6); w /= w.sum(axis=1, keepdims=True)
    return np.einsum("nk,nkd->nd", w, q1_emb[idx])


def _xgb_pgal(X, q1_X_raw):
    seed_paths = sorted(SEED_MODELS_DIR.glob("xgb_classifier_q1_seed*.json"))
    if not seed_paths:
        return np.full(len(X), np.nan)
    import xgboost as xgb
    medians = np.nanmedian(q1_X_raw, axis=0)
    Xi = X.astype(np.float32).copy()
    for c in range(Xi.shape[1]):
        bad = ~np.isfinite(Xi[:, c])
        if bad.any(): Xi[bad, c] = medians[c]
    probs = []
    for p in seed_paths:
        m = xgb.XGBClassifier(); m.load_model(str(p))
        probs.append(m.predict_proba(Xi)[:, 1])
    return np.mean(probs, axis=0)


def project_and_assign(df: pd.DataFrame) -> pd.DataFrame:
    q1       = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    labels   = np.load(ARRAYS_DIR / "q1_cluster_labels.npy")
    q1_emb   = q1["embedding"]; q1_X_raw = q1["X_raw"].astype(float)
    mean, std = q1["mean"], q1["std"]; std_safe = np.where(std > 0, std, 1.0)
    X  = df[list(SELECTED_GAIA_PARAMETERS)].apply(pd.to_numeric, errors="coerce").values.astype(float)
    Xs = np.where(np.isfinite((X - mean) / std_safe), (X - mean) / std_safe, 0.0)
    try:
        emb = _load_reducer().transform(Xs).astype(float)
        print("  projection: umap.transform (saved reducer)")
    except Exception as exc:
        print(f"  reducer unavailable ({type(exc).__name__}: {exc}); using k-NN fallback")
        Xtr = np.where(np.isfinite((q1_X_raw - mean) / std_safe), (q1_X_raw - mean) / std_safe, 0.0)
        emb = _project_knn(Xs, Xtr, q1_emb)
    valid = labels >= 0
    nn = NearestNeighbors(n_neighbors=1).fit(q1_emb[valid])
    _, idx = nn.kneighbors(emb)
    df = df.copy()
    df["umap1"], df["umap2"] = emb[:, 0], emb[:, 1]
    df["cluster"]    = labels[np.where(valid)[0][idx.ravel()]]
    df["model_pgal"] = _xgb_pgal(X, q1_X_raw)
    return df


def report(df: pd.DataFrame) -> None:
    def _row(sub, title):
        n = len(sub)
        if not n: print(f"  {title}: (none)"); return
        vc = sub["cluster"].value_counts().sort_index()
        parts = "  ".join(f"C{c}={vc.get(c,0):>4} ({100*vc.get(c,0)/n:4.1f}%)" for c in range(5))
        print(f"  {title:<34} N={n:>6}   {parts}")
    print("\n" + "=" * 96)
    print("QSO → UMAP cluster membership   (C0 = 95%-galaxy extragalactic locus)")
    print("=" * 96)
    _row(df, "ALL QSO-Gaia matches")
    print("\n— by astrometric_params_solved —")
    for aps, sub in df.groupby("astrometric_params_solved"):
        _row(sub, APS_LABEL.get(int(aps), f"aps={int(aps)}"))
    print("\n— single vs multi-Gaia match —")
    _row(df[df["n_gaia_within_1arcsec"] == 1], "single Gaia match")
    _row(df[df["n_gaia_within_1arcsec"] >  1], "multi-match (blend)")
    print("\n— high XGB P(gal) ≥ 0.5 —")
    _row(df[df["model_pgal"] >= 0.5], "model_pgal ≥ 0.5")
    print("=" * 96)
    c0 = df[df["cluster"] == 0]
    print(f"\nHeadline: {len(c0):,}/{len(df):,} ({100*len(c0)/max(len(df),1):.1f}%) land in C0.")


def aps_density_panel(fig, ax, df, xlim, ylim, noun="QSOs") -> None:
    aps_cmaps = {3: "Purples", 31: "Greens", 95: "Blues"}
    ext = [xlim[0], xlim[1], ylim[0], ylim[1]]
    for aps in (95, 31, 3):
        s = df[df["astrometric_params_solved"] == aps]
        if len(s): ax.hexbin(s["umap1"], s["umap2"], gridsize=120, cmap=aps_cmaps[aps],
                             bins="log", mincnt=1, alpha=0.9, extent=ext, zorder=2)
    ax.set_title("by astrometric_params_solved  (per-class density)", fontsize=13, fontweight="bold")
    pos = ax.get_position(); order = [95, 31, 3]; lab = {3: "2", 31: "5", 95: "6"}
    gap, w = 0.004, 0.010; h = (pos.height - (len(order)-1)*gap) / len(order); x0 = pos.x1 + 0.006
    for i, aps in enumerate(order):
        cax = fig.add_axes([x0, pos.y1-(i+1)*h-i*gap, w, h])
        sm = mpl.cm.ScalarMappable(cmap=aps_cmaps[aps], norm=mpl.colors.Normalize(0,1)); sm.set_array([])
        cb = fig.colorbar(sm, cax=cax); cb.set_ticks([])
        cax.set_ylabel(lab[aps], rotation=0, fontsize=9, labelpad=8, va="center", ha="right")
    fig.text(x0+w+0.01, pos.y0+pos.height/2, "# params (shade = density)",
             rotation=90, va="center", ha="left", fontsize=7.5)


def cluster_density_panel(fig, ax, df, xlim, ylim, cluster_ids, noun="QSOs") -> None:
    ext = [xlim[0], xlim[1], ylim[0], ylim[1]]
    for c in cluster_ids:
        s = df[df["cluster"] == c]
        if len(s): ax.hexbin(s["umap1"], s["umap2"], gridsize=120, cmap=CLUSTER_CMAPS.get(c,"Greys"),
                             bins="log", mincnt=1, alpha=0.9, extent=ext, zorder=2)
    ax.set_title("by assigned UMAP cluster  (per-cluster density)", fontsize=13, fontweight="bold")
    pos = ax.get_position(); order = list(cluster_ids)
    gap, w = 0.003, 0.010; h = (pos.height-(len(order)-1)*gap)/len(order); x0 = pos.x1+0.006
    for i, c in enumerate(order):
        cax = fig.add_axes([x0, pos.y1-(i+1)*h-i*gap, w, h])
        sm = mpl.cm.ScalarMappable(cmap=CLUSTER_CMAPS.get(c,"Greys"), norm=mpl.colors.Normalize(0,1)); sm.set_array([])
        cb = fig.colorbar(sm, cax=cax); cb.set_ticks([])
        cax.set_ylabel(f"C{c}", rotation=0, fontsize=9, labelpad=15, va="center", ha="right")
    fig.text(x0+w+0.014, pos.y0+pos.height/2, "cluster (shade = density)",
             rotation=90, va="center", ha="left", fontsize=7.5)

TABLE      = "crossmatch_qso_catwise_euclid_gaia"
CSV_OUT    = QSO_ARRAYS / "catwise_euclid_gaia.csv"
PNG_CLUST  = OUTPUT_DIR / "catwise_euclid_gaia_clusters.png"
PNG_PGAL   = OUTPUT_DIR / "catwise_euclid_gaia.png"


# ── DB fetch ───────────────────────────────────────────────────────────────────

def load_table(user: str) -> pd.DataFrame | None:
    import psycopg2
    table     = f"{user}.{TABLE}"
    feat_cols = ", ".join(f"g_{c} AS {c}" for c in SELECTED_GAIA_PARAMETERS)
    sql = f"""
        SELECT qso_catalog, qso_id, qso_ra, qso_dec,
               g_source_id                  AS gaia_source_id,
               gaia_sep_arcsec,
               n_gaia_within_1arcsec,
               w1mpro                       AS w1,
               w2mpro                       AS w2,
               w1mpro - w2mpro              AS w1_w2,
               euclid_source                AS euclid,
               n_euclid_within_1arcsec,
               n_qso_per_euclid,
               n_qso_per_gaia,
               e_deblended_flag,
               g_astrometric_params_solved  AS astrometric_params_solved,
               g_phot_g_mean_mag            AS phot_g_mean_mag,
               g_phot_g_n_obs               AS phot_g_n_obs,
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
        print(f"  failed: {exc}")
        return None
    print(f"    {len(df):,} rows")
    return df


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot(df: pd.DataFrame) -> None:
    q1         = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    labels     = np.load(ARRAYS_DIR / "q1_cluster_labels.npy")
    q1_emb     = q1["embedding"]
    y          = q1["y"]
    cluster_ids = sorted(set(labels[labels >= 0]))
    centroids  = {c: (q1_emb[labels == c, 0].mean(), q1_emb[labels == c, 1].mean())
                  for c in cluster_ids}
    xlim = (q1_emb[:, 0].min() - 0.8, q1_emb[:, 0].max() + 0.8)
    ylim = (q1_emb[:, 1].min() - 0.8, q1_emb[:, 1].max() + 0.8)

    def _base(ax):
        ax.scatter(q1_emb[:, 0], q1_emb[:, 1], s=2, c="#e2e2e2", alpha=0.6,
                   linewidths=0, rasterized=True, zorder=1)
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_xticks([]); ax.set_yticks([])

    def _clusters(ax):
        for c in cluster_ids:
            cx, cy = centroids[c]
            t = ax.text(cx, cy, f"C{c}", fontsize=9, fontweight="bold",
                        ha="center", va="center", color=CLUSTER_COLOURS[c], zorder=7)
            t.set_path_effects([pe.Stroke(linewidth=2, foreground="white"), pe.Normal()])

    def _frac_box(ax, sub):
        n = len(sub)
        lines = [f"N = {n:,}"]
        if n:
            vc = sub["cluster"].value_counts()
            for c in sorted(cluster_ids, key=lambda k: -vc.get(k, 0)):
                p = 100 * vc.get(c, 0) / n
                if p >= 0.5:
                    lines.append(f"C{c}: {p:.0f}%")
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=9,
                va="top", ha="left", zorder=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.92))

    suptitle = f"{len(df):,} CATWise×Euclid×Gaia QSOs projected onto the Q1 UMAP"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: clusters (4 panels) ────────────────────────────────────────
    fig = plt.figure(figsize=(26, 7))
    gs  = fig.add_gridspec(1, 4, left=0.03, right=0.975, top=0.86, bottom=0.05, wspace=0.17)
    aax = fig.add_subplot(gs[0, 0])
    kax = fig.add_subplot(gs[0, 1])
    sax = fig.add_subplot(gs[0, 2])
    blax = fig.add_subplot(gs[0, 3])

    ax = aax; _base(ax)
    aps_density_panel(fig, ax, df, xlim, ylim, noun="QSOs")
    _clusters(ax)

    ax = kax; _base(ax)
    cluster_density_panel(fig, ax, df, xlim, ylim, cluster_ids, noun="QSOs")
    _clusters(ax); _frac_box(ax, df)

    ax = sax; _base(ax)
    pts     = df[["umap1", "umap2"]].to_numpy()
    outside = outside_mask(pts, q1_emb)
    slab    = stream_labels(pts, outside, centroids)
    ax.scatter(pts[~outside, 0], pts[~outside, 1], s=2, c="#d8d8d8", alpha=0.2,
               linewidths=0, rasterized=True, zorder=2)
    for k, (name, _verts, col) in enumerate(STREAM_PAIRS):
        m = slab == k
        ax.scatter(pts[m, 0], pts[m, 1], s=6, color=col, alpha=0.85, linewidths=0,
                   rasterized=True, zorder=5, label=f"{name}: {int(m.sum()):,}")
    mo = slab == -1
    ax.scatter(pts[mo, 0], pts[mo, 1], s=3, c="#9a9a9a", alpha=0.4, linewidths=0,
               rasterized=True, zorder=4, label=f"other: {int(mo.sum()):,}")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title(f"inter-cluster streams  ({int(outside.sum()):,} outside Q1 cores)",
                 fontsize=12, fontweight="bold")
    _clusters(ax)

    # ── Panel 4: blend types scatter (no density background) ──
    ax = blax; _base(ax)
    BLEND_GROUPS = [
        # (label, mask, colour, marker, size, alpha)
        ("euclid_deblend", df["e_deblended_flag"].fillna(0) > 0, "#e6194b", "o", 6, 0.6),
        ("multi_euclid",   df["n_euclid_within_1arcsec"] > 1,   "#B2D28D", "^", 20, 0.8),
        ("multi_gaia",     df["n_gaia_within_1arcsec"]   > 1,   "#4363d8", "s", 20, 0.8), 
    ]
    for label, mask, col, mkr, size, alp in BLEND_GROUPS:
        sub = df[mask]
        ax.scatter(sub["umap1"], sub["umap2"], s=size, c=col, marker=mkr,
                   alpha=alp, linewidths=0.3, zorder=5, edgecolors="w",
                   label=f"{label} ({len(sub):,})")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title("blend types", fontsize=12, fontweight="bold")
    _clusters(ax)

    fig.suptitle(suptitle, fontsize=16, fontweight="bold", x=0.5, y=0.97)
    plt.savefig(PNG_CLUST, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  -> {PNG_CLUST}")

    # ── Figure 2: blend flag | XGB P(gal) ────────────────────────────────────
    fig = plt.figure(figsize=(13, 6.8))
    gs  = fig.add_gridspec(1, 2, left=0.04, right=0.955, top=0.86, bottom=0.05, wspace=0.17)
    bax = fig.add_subplot(gs[0, 0])
    pax = fig.add_subplot(gs[0, 1])

    ax = bax; _base(ax)
    hb = ax.hexbin(df["umap1"], df["umap2"], gridsize=120, cmap="viridis", bins="log",
                   alpha=0.8, mincnt=1, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], zorder=2)
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("QSO count (log)", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_title("2D density", fontsize=12, fontweight="bold")
    _clusters(ax); _frac_box(ax, df)

    ax = pax; _base(ax)
    sc = ax.scatter(df["umap1"], df["umap2"], s=4, c=df["model_pgal"], cmap="RdYlBu_r",
                    vmin=0, vmax=1, alpha=0.5, linewidths=0, rasterized=True, zorder=2)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("XGB P(gal)", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_title("XGB P(gal)", fontsize=13, fontweight="bold"); _clusters(ax)
    lines = ["QSOs with P(gal) ≥ 0.5:"]
    for c in cluster_ids:
        sub  = df[df["cluster"] == c]
        frac = 100 * (sub["model_pgal"] >= 0.5).mean() if len(sub) else 0.0
        lines.append(f"  C{c}: {frac:>3.0f}%   (N={len(sub):,})")
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=8.5,
            va="top", ha="left", zorder=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.92))

    fig.suptitle(suptitle, fontsize=16, fontweight="bold", x=0.5, y=0.97)
    plt.savefig(PNG_PGAL, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  -> {PNG_PGAL}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replot", action="store_true",
                    help="Skip WSDB + projection; redraw figures from saved CSV.")
    args = ap.parse_args()

    if args.replot:
        if not CSV_OUT.exists():
            print(f"  {CSV_OUT.name} not found — run without --replot first.")
            return 1
        df = pd.read_csv(CSV_OUT)
        print(f"Replotting from {CSV_OUT.name} ({len(df):,} rows)...")
        plot(df)
        return 0

    user = os.environ.get("PGUSER", "siwei_zhao_2025")
    df   = load_table(user)
    if df is None or not len(df):
        return 1

    print(f"  {len(df):,} rows; projecting onto Q1 UMAP + clusters + model_pgal...")
    df = project_and_assign(df)
    report(df)

    QSO_ARRAYS.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False)
    print(f"  -> {CSV_OUT}")

    plot(df)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
