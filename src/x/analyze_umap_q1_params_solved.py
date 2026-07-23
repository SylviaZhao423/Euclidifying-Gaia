#!/usr/bin/env python3
"""
Random Gaia DR3 sources projected into Q1 UMAP space, broken down by
astrometric_params_solved (2 / 5 / 6-parameter solution).

Companion to analyze_umap_q1_space_gaia_xgb.py — same projected sample
(gaia_random_projected.csv: umap coords + assigned cluster), but instead of
colouring by XGBoost P(gal) it answers: **how much of each parameter-solved
class lands in which UMAP cluster?**  This is the general-Gaia baseline for the
QSO result (2-param QSOs → C1): does the *whole* 2-param Gaia population also
prefer C0/C1, or is the QSO concentration special?

astrometric_params_solved is NOT in the projected CSV (not one of the 14
features), so it is fetched from WSDB for the sample's source_ids (needs VPN).

Reads:
  output/umap/arrays/q1_indiv.npz                          (embedding, y)
  output/umap/arrays/q1_cluster_labels.npy
  output/umap/arrays/projected/gaia_random_projected.csv   (umap1, umap2, cluster, source_id)

Writes:
  output/umap/q1_individual/q1_space_gaia_params_solved.png

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_umap_q1_space_gaia_params_solved.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

PROJECT_DIR   = Path(__file__).resolve().parents[1]
ARRAYS_DIR    = PROJECT_DIR / "output" / "umap" / "arrays"
PROJECTED_DIR = ARRAYS_DIR / "projected"
OUTPUT_DIR    = PROJECT_DIR / "output" / "umap" / "q1_individual"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
WSDB_USER = "siwei_zhao_2025"

# Canonical Q1 cluster colours (match compile_umap / analyze_umap_q1_space_gaia_xgb)
CLUSTER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#F298B6", "mediumpurple",
                  "#17becf", "#d62728", "#8c564b", "#7f7f7f", "#bcbd22"]
APS_LABEL = {3: "2-param (pos only)", 31: "5-param", 95: "6-param"}


APS_CACHE = ARRAYS_DIR / "gaia_astrometric_params_solved.csv"   # persistent local cache


def _load_aps_cache() -> dict:
    if not APS_CACHE.exists():
        return {}
    try:
        d = pd.read_csv(APS_CACHE)
        return {str(int(s)): int(v) for s, v in zip(d["source_id"], d["astrometric_params_solved"])}
    except Exception:
        return {}


def _wsdb_fetch_aps(sids_int: list) -> dict:
    """Temp-table + index JOIN for astrometric_params_solved. ANALYZE the temp
    table so the planner does an index nested-loop against gaia_source's source_id
    PK (without it, it may sequential-scan the 1.8B-row table). Always closes."""
    import psycopg2
    print(f"  [aps] querying WSDB for {len(sids_int):,} missing source_ids...")
    conn = None
    try:
        conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=WSDB_USER,
                                connect_timeout=30, keepalives=1)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("CREATE TEMP TABLE _aps_sids (source_id BIGINT PRIMARY KEY)")
        for i in range(0, len(sids_int), 10000):
            b = sids_int[i:i + 10000]
            cur.execute("INSERT INTO _aps_sids VALUES " +
                        ",".join(f"({s})" for s in b) + " ON CONFLICT DO NOTHING")
        cur.execute("ANALYZE _aps_sids")        # <-- forces the fast index-join plan
        cur.execute("SELECT s.source_id, g.astrometric_params_solved "
                    "FROM _aps_sids s JOIN gaia_dr3.gaia_source g ON g.source_id = s.source_id")
        return {str(int(a)): int(b) for a, b in cur.fetchall() if b is not None}
    except Exception as exc:
        print(f"  [aps] WSDB fetch failed ({exc}).")
        return {}
    finally:
        if conn is not None:
            conn.close()


def fetch_params_solved(source_ids: np.ndarray) -> np.ndarray:
    """Return astrometric_params_solved aligned with source_ids (NaN where
    unknown). Disk-cached in output/umap/arrays/ — only missing ids are queried,
    so it's a one-time WSDB cost and later runs are instant (like the other saved
    arrays this pipeline reads)."""
    req = [str(int(s)) for s in source_ids]
    cache = _load_aps_cache()
    if cache:
        print(f"  [aps] loaded {len(cache):,} from cache ({APS_CACHE.name})")
    missing = sorted({int(s) for s in req if s not in cache})
    if missing:
        fetched = _wsdb_fetch_aps(missing)
        if fetched:
            cache.update(fetched)
            APS_CACHE.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"source_id": list(cache.keys()),
                          "astrometric_params_solved": list(cache.values())}).to_csv(APS_CACHE, index=False)
            print(f"  [aps] cached {len(cache):,} → {APS_CACHE.name}")
    n_ok = sum(1 for s in req if s in cache)
    print(f"  [aps] resolved {n_ok:,}/{len(req):,} sources")
    return np.array([float(cache.get(s, np.nan)) for s in req], dtype=float)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1] Loading Q1 UMAP + cluster labels...")
    q1        = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    q1_emb    = q1["embedding"]
    y         = q1["y"]
    labels    = np.load(ARRAYS_DIR / "q1_cluster_labels.npy")
    cluster_ids = sorted(set(labels[labels >= 0]))
    centroids = {c: (q1_emb[labels == c, 0].mean(), q1_emb[labels == c, 1].mean()) for c in cluster_ids}
    galpct    = {c: 100 * (y[labels == c] == 1).sum() / (labels == c).sum() for c in cluster_ids}

    print("[2] Loading projected Gaia random sample...")
    df = pd.read_csv(PROJECTED_DIR / "gaia_random_projected.csv",
                     usecols=["source_id", "umap1", "umap2", "cluster"])
    print(f"  {len(df):,} sources")

    print("[3] Fetching astrometric_params_solved from WSDB (needs VPN)...")
    df["aps"] = fetch_params_solved(df["source_id"].values)
    df = df[np.isfinite(df["aps"])].copy()
    df["aps"] = df["aps"].astype(int)

    # ── Cluster-composition table per params-solved class ──
    print("\n" + "=" * 78)
    print("Gaia random: cluster membership by astrometric_params_solved")
    print("=" * 78)
    aps_present = [a for a in (3, 31, 95) if (df["aps"] == a).any()]
    for a in aps_present:
        sub = df[df["aps"] == a]
        parts = "  ".join(f"C{c}={100*(sub['cluster']==c).mean():4.1f}%" for c in cluster_ids)
        print(f"  {APS_LABEL[a]:<20} N={len(sub):>6,}   {parts}")
    print("=" * 78)

    # ── Plot: 3 density facets by params-solved ──
    xlim = (q1_emb[:, 0].min() - 1, q1_emb[:, 0].max() + 1)
    ylim = (q1_emb[:, 1].min() - 1, q1_emb[:, 1].max() + 1)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))
    fig.subplots_adjust(wspace=0.08, left=0.02, right=0.97, top=0.86, bottom=0.06)

    hbs = []
    for ax, a in zip(axes, (3, 31, 95)):
        ax.scatter(q1_emb[:, 0], q1_emb[:, 1], s=2, c="#e2e2e2", alpha=0.5,
                   linewidths=0, rasterized=True, zorder=1)
        sub = df[df["aps"] == a]
        if len(sub):
            hb = ax.hexbin(sub["umap1"], sub["umap2"], gridsize=110, cmap="viridis",
                           mincnt=1, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                           alpha=0.8, zorder=2)
            hbs.append(hb)
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{APS_LABEL[a]}  (N={len(sub):,})", fontsize=12, fontweight="bold")
        # centroid labels show THIS panel's cluster fractions (% of this class in each cluster)
        for c in cluster_ids:
            cx, cy = centroids[c]
            fpct = 100 * (sub["cluster"] == c).mean() if len(sub) else 0.0
            t = ax.text(cx, cy, f"C{c}\n{fpct:.0f}%", fontsize=9, fontweight="bold",
                        ha="center", va="center", color=CLUSTER_COLORS[c], zorder=6)
            t.set_path_effects([pe.Stroke(linewidth=2, foreground="white"), pe.Normal()])
    if hbs:
        vmax = max(h.get_array().max() for h in hbs)
        norm = mpl.colors.LogNorm(vmin=1, vmax=vmax)
        for h in hbs:
            h.set_norm(norm)
        cb = fig.colorbar(hbs[-1], ax=list(axes), fraction=0.025, pad=0.02)
        cb.set_label("Gaia count (log, shared scale)", fontsize=9); cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Random Gaia DR3 in Q1 UMAP space — where each astrometric solution class lands",
        fontsize=13, fontweight="bold", y=0.98)

    out = OUTPUT_DIR / "q1_space_gaia_params_solved.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")

    # ── Training-sample comparison (Q1) → q1_params_solved.png ──────────────────────────
    # Same idea for the TRAINING set (q1_indiv), split upper=stars / lower=galaxies
    # like analyze_umap_q1_features.py, coloured by astrometric_params_solved.
    print("\n[4] Q1 training sample: fetching astrometric_params_solved...")
    q1_sids = q1["source_ids"].astype(str) if "source_ids" in q1 else None
    if q1_sids is None:
        print("  source_ids missing in q1_indiv.npz — skipping q1_params_solved.png")
        return 0
    aps_tr = fetch_params_solved(q1_sids)
    plot_q1_training(q1_emb, y, aps_tr, labels, cluster_ids, centroids, galpct)
    return 0


def plot_q1_training(q1_emb, y, aps, labels, cluster_ids, centroids, galpct):
    """Q1 training UMAP as 3 density facets by astrometric_params_solved
    (2 / 5 / 6-param), shared log colour scale → q1_params_solved.png."""
    xlim = (q1_emb[:, 0].min() - 1, q1_emb[:, 0].max() + 1)
    ylim = (q1_emb[:, 1].min() - 1, q1_emb[:, 1].max() + 1)

    print("\n" + "=" * 78)
    print("Q1 TRAINING: cluster membership by astrometric_params_solved")
    print("=" * 78)
    for a in (3, 31, 95):
        m = aps == a
        if not m.any():
            continue
        parts = "  ".join(f"C{c}={100*(labels[m] == c).mean():4.1f}%" for c in cluster_ids)
        print(f"  {APS_LABEL[a]:<20} N={int(m.sum()):>6,}   {parts}")
    print("=" * 78)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))
    fig.subplots_adjust(wspace=0.08, left=0.02, right=0.97, top=0.86, bottom=0.06)
    hbs = []
    for ax, a in zip(axes, (3, 31, 95)):
        ax.scatter(q1_emb[:, 0], q1_emb[:, 1], s=2, c="#e2e2e2", alpha=0.5,
                   linewidths=0, rasterized=True, zorder=1)
        m = aps == a
        if m.any():
            hb = ax.hexbin(q1_emb[m, 0], q1_emb[m, 1], gridsize=110, cmap="viridis",
                           mincnt=1, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                           alpha=0.8, zorder=2)
            hbs.append(hb)
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{APS_LABEL[a]}  (N={int(m.sum()):,})", fontsize=12, fontweight="bold")
        # centroid labels show THIS panel's cluster fractions (% of this class in each cluster)
        for c in cluster_ids:
            cx, cy = centroids[c]
            fpct = 100 * (labels[m] == c).mean() if m.any() else 0.0
            t = ax.text(cx, cy, f"C{c}\n{fpct:.0f}%", fontsize=9, fontweight="bold",
                        ha="center", va="center", color=CLUSTER_COLORS[c], zorder=6)
            t.set_path_effects([pe.Stroke(linewidth=2, foreground="white"), pe.Normal()])
    if hbs:
        vmax = max(h.get_array().max() for h in hbs)
        norm = mpl.colors.LogNorm(vmin=1, vmax=vmax)
        for h in hbs:
            h.set_norm(norm)
        cb = fig.colorbar(hbs[-1], ax=list(axes), fraction=0.025, pad=0.02)
        cb.set_label("Q1 training count (log, shared scale)", fontsize=9)
        cb.ax.tick_params(labelsize=8)
    fig.suptitle("Q1 training sample in UMAP space — density by astrometric_params_solved",
                 fontsize=13, fontweight="bold", y=0.98)
    out = OUTPUT_DIR / "q1_params_solved.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
