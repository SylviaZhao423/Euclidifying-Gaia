#!/usr/bin/env python3
"""
Fit a fresh 14-feature Gaia UMAP on the SPECTROSCOPIC QSO × Gaia samples — same recipe
as query_umap_q1_50k.py (subsample -> median-fill NaN -> z-score -> UMAP n_neighbors=15,
min_dist=0.1, seed=42, then HDBSCAN), but the input is a known-QSO sample instead of Q1
PHZ 1/2. Always builds all THREE samples:

    desi      crossmatch_qso_desi_dr1_gaia      (DESI DR1 QSO × Gaia, 626k)
    sdss      crossmatch_qso_sdss_dr16_gaia     (SDSS DR16 QSO × Gaia, 489k)
    combined  both, de-duplicated on g_source_id

Each sample fits its OWN reducer (3 independent UMAPs, NOT projections onto the Q1
embedding) and saves arrays + a quick 3-panel look (coloured by QSO redshift, G, cluster).

Outputs (output/qso_umap/ds_qso/n{N}/):
    arrays/{sample}_qso_gaia.npz          embedding, X_raw, mean, std, source_ids, qso_id, qso_z, g_mag, ra, dec, survey, cluster
    arrays/{sample}_qso_gaia_reducer.joblib
    {sample}_qso_gaia_umap.png

Usage (always all 3 samples; only the sample SIZE is a knob):
    PGUSER=siwei_zhao_2025 python src/query_qso_umap_ds.py                 # n=480000 default
    PGUSER=siwei_zhao_2025 python src/query_qso_umap_ds.py --n-sample 50000
    PGUSER=siwei_zhao_2025 python src/query_qso_umap_ds.py --n-sample all  # every source (n_all)
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
DS_DIR      = PROJECT_DIR / "output" / "qso_umap" / "ds_qso"   # <subdir>/n{N}/[arrays]
WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
MAG_CUT     = 17         # same G>=17 floor as query_umap_q1_50k (comparable feature regime)
SAMPLE_SIZE = 480_000


# ── output paths (n{N}/ folder encodes the size; 'all' -> n_all, no suffix on names) ──
def n_suffix(n):    return "n_all" if str(n) == "all" else f"n{int(n)}"
def parse_n(v):     return "all" if str(v).lower() == "all" else int(v)
def out_dir(n):     return DS_DIR / n_suffix(n)
def arrays_dir(n):  return out_dir(n) / "arrays"

TABLES = {
    "desi": "crossmatch_qso_desi_dr1_gaia",
    "sdss": "crossmatch_qso_sdss_dr16_gaia",
}


# ── DB query ────────────────────────────────────────────────────────────────
def query_qso(username, sample, mag_cut):
    """14 Gaia features + ids + qso_z + g_mag for a known-QSO×Gaia sample."""
    import psycopg2
    gaia_cols = ", ".join(f"g_{c} AS {c}" for c in SELECTED_GAIA_PARAMETERS)
    tabs = list(TABLES.values()) if sample == "combined" else [TABLES[sample]]
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    frames = []
    for i, t in enumerate(tabs):
        src = "desi" if "desi" in t else "sdss"
        df = pd.read_sql_query(f"""
            SELECT g_source_id, qso_id, qso_z,
                   g_phot_g_mean_mag AS g_mag, g_ra AS ra, g_dec AS dec,
                   '{src}' AS survey,
                   {gaia_cols}
            FROM {username}.{t}
            WHERE g_phot_g_mean_mag >= {mag_cut}
        """, conn)
        print(f"    {t}: {len(df):,} (G>={mag_cut})")
        frames.append(df)
    conn.close()
    df = pd.concat(frames, ignore_index=True)
    if sample == "combined":
        n0 = len(df)
        df = df.drop_duplicates("g_source_id").reset_index(drop=True)
        print(f"    combined de-dup on g_source_id: {n0:,} -> {len(df):,}")

    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace("NOT_AVAILABLE", np.nan)
    for c in SELECTED_GAIA_PARAMETERS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"  -> {len(df):,} {sample} QSO×Gaia sources")
    return df


# ── Subsample + standardize (identical to query_umap_q1_50k.clean_for_umap) ──
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


def cluster_hdbscan(embedding, min_cluster_size=150, epsilon=0.3):
    try:
        import hdbscan
        labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                 min_samples=max(10, min_cluster_size // 10),
                                 cluster_selection_epsilon=epsilon).fit_predict(embedding)
        n = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"  HDBSCAN -> {n} clusters ({(labels == -1).sum():,} noise)")
    except ImportError:
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(embedding)
        print("  hdbscan unavailable -> KMeans(k=5)")
    return labels


def merge_clusters(labels, merges):
    """Merge clusters per 'src>dst' specs (src folded into dst), then renumber the non-noise
    labels to contiguous 0..k-1 (noise stays -1). e.g. merges=['2>3']."""
    labels = np.asarray(labels).copy()
    for spec in merges:
        src, dst = (int(x) for x in spec.split(">"))
        n = int((labels == src).sum())
        if n:
            labels[labels == src] = dst
            print(f"  merged C{src} ({n:,} pts) -> C{dst}")
    old = sorted(set(labels[labels >= 0]))
    out = labels.copy()
    for new, o in enumerate(old):
        out[labels == o] = new
    return out


# ── Plot ────────────────────────────────────────────────────────────────────
def plot_umap(emb, qso_z, g_mag, labels, sample, out):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.6))
    # (a) coloured by QSO redshift
    z = np.clip(qso_z, 0, 4)
    s0 = axes[0].scatter(emb[:, 0], emb[:, 1], c=z, s=4, cmap="viridis",
                         vmin=0, vmax=4, alpha=0.6, lw=0, rasterized=True)
    axes[0].set_title("coloured by spectroscopic $z$", fontsize=13, fontweight="bold")
    fig.colorbar(s0, ax=axes[0], fraction=0.046, pad=0.02).set_label("QSO redshift", fontsize=10)
    # (b) coloured by G magnitude
    s1 = axes[1].scatter(emb[:, 0], emb[:, 1], c=g_mag, s=4, cmap="magma_r",
                         alpha=0.6, lw=0, rasterized=True)
    axes[1].set_title("coloured by Gaia $G$", fontsize=13, fontweight="bold")
    fig.colorbar(s1, ax=axes[1], fraction=0.046, pad=0.02).set_label("G (mag)", fontsize=10)
    # (c) HDBSCAN clusters
    cl_ids = sorted(set(labels))
    cmap = plt.get_cmap("tab10")
    for c in cl_ids:
        m = labels == c
        col = "0.8" if c == -1 else cmap(c % 10)
        axes[2].scatter(emb[m, 0], emb[m, 1], s=4, color=col, alpha=0.6, lw=0,
                        rasterized=True, label=("noise" if c == -1 else f"C{c}"))
    axes[2].set_title(f"HDBSCAN clusters ({len([c for c in cl_ids if c >= 0])})",
                      fontsize=13, fontweight="bold")
    axes[2].legend(loc="best", fontsize=7, markerscale=2, framealpha=0.9, ncol=2)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.suptitle(f"{sample.upper()} QSO × Gaia — 14-feature UMAP  (N={len(emb):,}, G≥{MAG_CUT})",
                 fontsize=15, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


# ── Build one sample ─────────────────────────────────────────────────────────
def build(username, sample, n_sample, mag_cut, min_cluster_size, epsilon=0.3, merges=()):
    print(f"\n=== {sample} ===")
    df = query_qso(username, sample, mag_cut)
    X = df[SELECTED_GAIA_PARAMETERS].values
    rng = np.random.RandomState(42)
    X_scaled, idx, mean_, std_ = clean_for_umap(X, n_sample, rng)
    sub = df.iloc[idx].reset_index(drop=True)
    print(f"  {len(X_scaled):,} subsampled -> fitting UMAP ...")

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    emb = reducer.fit_transform(X_scaled)
    labels = cluster_hdbscan(emb, min_cluster_size=min_cluster_size, epsilon=epsilon)
    if merges:
        labels = merge_clusters(labels, merges)

    adir = arrays_dir(n_sample); odir = out_dir(n_sample)
    adir.mkdir(parents=True, exist_ok=True)
    np.savez(adir / f"{sample}_qso_gaia.npz",
             embedding=emb, X_raw=X[idx], mean=mean_, std=std_,
             source_ids=sub["g_source_id"].astype(str).values,
             qso_id=sub["qso_id"].astype(str).values,
             qso_z=pd.to_numeric(sub["qso_z"], errors="coerce").values,
             g_mag=pd.to_numeric(sub["g_mag"], errors="coerce").values,
             ra=pd.to_numeric(sub["ra"], errors="coerce").values,
             dec=pd.to_numeric(sub["dec"], errors="coerce").values,
             survey=sub["survey"].astype(str).values,
             cluster=labels)
    joblib.dump(reducer, adir / f"{sample}_qso_gaia_reducer.joblib")
    print(f"  -> {sample}/{n_suffix(n_sample)}/{sample}_qso_gaia.npz + reducer")

    plot_umap(emb, pd.to_numeric(sub["qso_z"], errors="coerce").values,
              pd.to_numeric(sub["g_mag"], errors="coerce").values, labels, sample,
              odir / f"{sample}_qso_gaia_umap.png")


# ── Re-cluster an already-built sample (no UMAP refit, no DB) ──────────────────
def recluster(sample, n_sample, min_cluster_size, epsilon, merges=()):
    """Re-run HDBSCAN on the stored embedding, overwrite the `cluster` array in the npz, and
    redraw {sample}_qso_gaia_umap.png. Everything else in the npz (embedding/reducer/features)
    is preserved. Use to retune clustering (e.g. lower --epsilon to keep the C-arm separate on
    big builds) without the expensive UMAP fit."""
    adir = arrays_dir(n_sample); odir = out_dir(n_sample)
    npz = adir / f"{sample}_qso_gaia.npz"
    if not npz.exists():
        print(f"  {npz} missing — build it first (drop --recluster)."); return
    d = dict(np.load(npz, allow_pickle=True))
    print(f"\n=== recluster {sample} (n={n_suffix(n_sample)}, {len(d['embedding']):,} pts, "
          f"eps={epsilon}) ===")
    labels = cluster_hdbscan(d["embedding"], min_cluster_size=min_cluster_size, epsilon=epsilon)
    if merges:
        labels = merge_clusters(labels, merges)
    d["cluster"] = labels
    np.savez(npz, **d)
    print(f"  updated cluster labels in {npz.name}")
    plot_umap(d["embedding"], d["qso_z"], d["g_mag"], labels, sample,
              odir / f"{sample}_qso_gaia_umap.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=parse_n, default=SAMPLE_SIZE,
                    help="subsample size, or 'all' to use every source (folder n_all).")
    ap.add_argument("--mag-cut", type=float, default=MAG_CUT)
    ap.add_argument("--min-cluster-size", type=int, default=150)
    ap.add_argument("--epsilon", type=float, default=0.3,
                    help="HDBSCAN cluster_selection_epsilon (larger = more merging). Lower it "
                         "(e.g. 0.1) on big builds to keep the C-arm/tail as its own cluster.")
    ap.add_argument("--recluster", action="store_true",
                    help="re-run HDBSCAN on the ALREADY-BUILT embeddings (no UMAP refit, no DB) "
                         "and overwrite the cluster labels + umap.png. Combine with --epsilon / "
                         "--min-cluster-size / --merge.")
    ap.add_argument("--merge", action="append", default=[], metavar="SRC>DST",
                    help="merge HDBSCAN clusters after fitting, e.g. --merge 2>4 (repeatable). "
                         "Applied to the selected sample(s); labels are then renumbered "
                         "contiguously. Cluster ids are per-sample, so scope with --sample.")
    ap.add_argument("--sample", choices=["desi", "sdss", "combined", "all"], default="all",
                    help="which sample(s) to build/recluster (default all three). Use a single "
                         "sample when applying a sample-specific --merge.")
    args = ap.parse_args()
    samples = ["desi", "sdss", "combined"] if args.sample == "all" else [args.sample]

    # re-cluster mode: operate on the stored embeddings only (no DB, no PGUSER needed)
    if args.recluster:
        for s in samples:
            recluster(s, args.n_sample, args.min_cluster_size, args.epsilon, merges=args.merge)
        print("\nDone (recluster).")
        return 0

    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set"); return 1

    for s in samples:
        build(username, s, args.n_sample, args.mag_cut, args.min_cluster_size,
              epsilon=args.epsilon, merges=args.merge)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
