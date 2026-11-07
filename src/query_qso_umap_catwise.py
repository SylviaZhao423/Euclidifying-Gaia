#!/usr/bin/env python3
"""
Fit a fresh 14-feature Gaia UMAP on the CatWISE photometric-QSO × Gaia sample — same
recipe as query_qso_umap_ds.py (subsample -> median-fill NaN -> z-score ->
UMAP n_neighbors=15, min_dist=0.1, seed=42, then HDBSCAN), but the input is the
CatWISE (AllWISE/CatWISE colour-selected) QSO candidates instead of the spectroscopic
DESI/SDSS samples:

    crossmatch_qso_catwise_gaia   (CatWISE QSO × Gaia, 4.43M rows)

The table has all 14 g_<feature> columns but NO spectroscopic redshift (CatWISE QSOs are
photometric candidates), so the redshift panel is dropped — the quick look shows G
magnitude and HDBSCAN clusters only. 4.43M is far too many to UMAP directly, so the rows
are pulled with a Bernoulli TABLESAMPLE and then subsampled to SAMPLE_SIZE.

Outputs (output/umap/catwise_qso/arrays/):
    catwise_qso_gaia.npz          embedding, X_raw, mean, std, source_ids, qso_id, g_mag, ra, dec, cluster
    catwise_qso_gaia_reducer.joblib
Plot (output/umap/catwise_qso/):
    catwise_qso_gaia_umap.png

Usage:
    PGUSER=siwei_zhao_2025 python src/query_qso_umap_catwise.py
    PGUSER=siwei_zhao_2025 python src/query_qso_umap_catwise.py --n-sample 100000   # smaller/faster
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import umap

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
CW_DIR      = PROJECT_DIR / "output" / "qso_umap" / "catwise_qso"   # <subdir>/n{N}/[arrays]
WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
MAG_CUT     = 17
SAMPLE_SIZE = 480_000
TABLE       = "crossmatch_qso_catwise_gaia"


# ── output paths (n{N}/ folder encodes the size; 'all' -> n_all) ──
def n_suffix(n):    return "n_all" if str(n) == "all" else f"n{int(n)}"
def parse_n(v):     return "all" if str(v).lower() == "all" else int(v)
def out_dir(n):     return CW_DIR / n_suffix(n)
def arrays_dir(n):  return out_dir(n) / "arrays"


# ── DB query ────────────────────────────────────────────────────────────────
def query_catwise(username, mag_cut, n_sample):
    """14 Gaia features + ids + g_mag for the CatWISE-QSO×Gaia sample (TABLESAMPLE)."""
    import psycopg2
    gaia_cols = ", ".join(f"g_{c} AS {c}" for c in SELECTED_GAIA_PARAMETERS)
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cur = conn.cursor(); cur.execute("SET statement_timeout = 600000")
    cur.execute(f"SELECT count(*) FROM {username}.{TABLE}")
    total = cur.fetchone()[0]
    if n_sample == "all":
        tablesample = ""   # every row (heavy: full 4.4M -> UMAP)
        print(f"  {TABLE}: {total:,} rows -> ALL (no TABLESAMPLE) (G>={mag_cut})")
    else:
        # sample ~2x the target fraction so >= n_sample survive the G cut
        pct = min(100.0, 100.0 * n_sample * 2.0 / max(total, 1))
        tablesample = f"TABLESAMPLE BERNOULLI({pct})"
        print(f"  {TABLE}: {total:,} rows -> {tablesample} (G>={mag_cut})")
    df = pd.read_sql_query(f"""
        SELECT g_source_id, qso_id,
               g_phot_g_mean_mag AS g_mag, g_ra AS ra, g_dec AS dec,
               {gaia_cols}
        FROM {username}.{TABLE} {tablesample}
        WHERE g_phot_g_mean_mag >= {mag_cut}
    """, conn)
    conn.close()
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace("NOT_AVAILABLE", np.nan)
    for c in SELECTED_GAIA_PARAMETERS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"  -> {len(df):,} CatWISE QSO×Gaia sources sampled")
    return df


# ── Subsample + standardize (identical to query_qso_umap_ds.clean_for_umap) ──
def clean_for_umap(X, sample_size, rng):
    n_eff = len(X) if sample_size == "all" else min(sample_size, len(X))
    idx = rng.choice(len(X), n_eff, replace=False)
    X_sub = X[idx].copy()
    for col in range(X_sub.shape[1]):
        bad = np.isnan(X_sub[:, col])
        if bad.any():
            X_sub[bad, col] = np.nanmedian(X_sub[:, col])
    mean_ = np.nanmean(X_sub, axis=0)
    std_  = np.nanstd(X_sub,  axis=0)
    X_scaled = (X_sub - mean_) / np.where(std_ > 0, std_, 1.0)
    return X_scaled, idx, mean_, std_


def cluster_hdbscan(embedding, min_cluster_size=150):
    try:
        import hdbscan
        labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                 min_samples=max(10, min_cluster_size // 10),
                                 cluster_selection_epsilon=0.3).fit_predict(embedding)
        n = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"  HDBSCAN -> {n} clusters ({(labels == -1).sum():,} noise)")
    except ImportError:
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(embedding)
        print("  hdbscan unavailable -> KMeans(k=5)")
    return labels


# ── Plot (G magnitude + HDBSCAN clusters; no spec-z for CatWISE) ──────────────
def plot_umap(emb, g_mag, labels, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    s1 = axes[0].scatter(emb[:, 0], emb[:, 1], c=g_mag, s=3, cmap="magma_r",
                         alpha=0.5, lw=0, rasterized=True)
    axes[0].set_title("coloured by Gaia $G$", fontsize=13, fontweight="bold")
    fig.colorbar(s1, ax=axes[0], fraction=0.046, pad=0.02).set_label("G (mag)", fontsize=10)

    cl_ids = sorted(set(labels))
    cmap = plt.get_cmap("tab10")
    for c in cl_ids:
        m = labels == c
        col = "0.8" if c == -1 else cmap(c % 10)
        axes[1].scatter(emb[m, 0], emb[m, 1], s=3, color=col, alpha=0.5, lw=0,
                        rasterized=True, label=("noise" if c == -1 else f"C{c}"))
    axes[1].set_title(f"HDBSCAN clusters ({len([c for c in cl_ids if c >= 0])})",
                      fontsize=13, fontweight="bold")
    axes[1].legend(loc="best", fontsize=7, markerscale=2, framealpha=0.9, ncol=2)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.suptitle(f"CatWISE QSO × Gaia — 14-feature UMAP  (N={len(emb):,}, G≥{MAG_CUT})",
                 fontsize=15, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=parse_n, default=SAMPLE_SIZE,
                    help="subsample size, or 'all' to use every source (folder n_all; heavy).")
    ap.add_argument("--mag-cut", type=float, default=MAG_CUT)
    ap.add_argument("--min-cluster-size", type=int, default=150)
    args = ap.parse_args()
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set"); return 1
    odir = out_dir(args.n_sample); adir = arrays_dir(args.n_sample)
    adir.mkdir(parents=True, exist_ok=True)

    df = query_catwise(username, args.mag_cut, args.n_sample)
    X = df[SELECTED_GAIA_PARAMETERS].values
    rng = np.random.RandomState(42)
    X_scaled, idx, mean_, std_ = clean_for_umap(X, args.n_sample, rng)
    sub = df.iloc[idx].reset_index(drop=True)
    print(f"  {len(X_scaled):,} subsampled -> fitting UMAP ...")

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    emb = reducer.fit_transform(X_scaled)
    labels = cluster_hdbscan(emb, min_cluster_size=args.min_cluster_size)

    np.savez(adir / "catwise_qso_gaia.npz",
             embedding=emb, X_raw=X[idx], mean=mean_, std=std_,
             source_ids=sub["g_source_id"].astype(str).values,
             qso_id=sub["qso_id"].astype(str).values,
             g_mag=pd.to_numeric(sub["g_mag"], errors="coerce").values,
             ra=pd.to_numeric(sub["ra"], errors="coerce").values,
             dec=pd.to_numeric(sub["dec"], errors="coerce").values,
             cluster=labels)
    joblib.dump(reducer, adir / "catwise_qso_gaia_reducer.joblib")
    print(f"  -> catwise/{n_suffix(args.n_sample)}/catwise_qso_gaia.npz + reducer")

    plot_umap(emb, pd.to_numeric(sub["g_mag"], errors="coerce").values, labels,
              odir / "catwise_qso_gaia_umap.png")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
