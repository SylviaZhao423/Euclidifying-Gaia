#!/usr/bin/env python3
"""
Download Euclid Q1 VIS cutouts for non-QSO phz classes, stratified by UMAP cluster.

Handles phz = 1 (Star), 2 (Galaxy), 3 (Multi Star∩Gal), -1 (Rejected), 0 (Outlier).
QSO variants (4, 5, 6, 7) are handled by the separate download_q_q1_qso_cutouts.py.

Workflow
--------
1. Load Q1 UMAP embedding (q1_indiv.npz) + reducer.
2. Load saved cluster labels (q1_cluster_labels.npy).
3. Query selected phz classes from DB — RA/Dec + Gaia features.
4. Project each source into UMAP space via the fitted reducer.
5. Assign each source to the nearest Q1 base cluster.
6. Discover tiles once for all sources (single TAP query over full bounding box).
7. For each (phz × cluster) cell: sample up to --per-cluster sources,
   download VIS cutouts via save_q1_utils.make_q1_cutouts().

Output layout
-------------
output/euclid_cutouts/q1_cutouts/
  star/        (phz=1)
  galaxy/      (phz=2)
  multi_sg/    (phz=3)
  rej/         (phz=-1)   # may already exist from download_q_q1_rej_cutouts.py
  outlier/     (phz=0)
    cluster{k}/
      contact_sheet.png
      manifest.csv
      cutouts/
        cutout_NNNN.png

Usage
-----
  # Download all phz classes (default)
  PGUSER=siwei_zhao_2025 python src/download_q_q1_cutouts.py

  # Download only stars and rejected
  PGUSER=siwei_zhao_2025 python src/download_q_q1_cutouts.py --phz 1,-1

  # Download galaxies with 20 per cluster
  PGUSER=siwei_zhao_2025 python src/download_q_q1_cutouts.py --phz 2 --per-cluster 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS
from save_q1_utils import make_q1_cutouts, _discover_tiles

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
CUTOUTS_DIR = PROJECT_DIR / "output" / "euclid_cutouts" / "q1_cutouts"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
MAG_CUT   = 17

PHZ_LABELS = {
     1: ("Star",             "cyan"),
     2: ("Galaxy",           "#F45347"),
     3: ("Multi (Star∩Gal)", "#FFC107"),
    -1: ("Rejected",         "#9C27B0"),
     0: ("Outlier",          "#FF9800"),
}

PHZ_DIRS = {1: "star", 2: "galaxy", 3: "multi_sg", -1: "rej", 0: "outlier"}

DEFAULT_PHZ = [1, 2, 3, -1, 0]


# ── DB query ──────────────────────────────────────────────────────────────────

def query_phz_sources(username: str, phz_values: list[int]) -> pd.DataFrame:
    """Query selected phz classes with RA/Dec and Gaia features."""
    import psycopg2
    gaia_cols = ", ".join([f"g_{col} AS {col}" for col in SELECTED_GAIA_PARAMETERS])
    in_clause = ", ".join(str(v) for v in phz_values)
    query = f"""
        SELECT
            g_source_id AS source_id,
            m_right_ascension AS ra,
            m_declination     AS dec,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            p_phz_classification AS phz,
            {gaia_cols}
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE p_phz_classification IN ({in_clause})
          AND g_phot_g_mean_mag >= {MAG_CUT}
          AND m_right_ascension IS NOT NULL
          AND m_declination IS NOT NULL
        ORDER BY g_source_id
    """
    phz_str = ", ".join(str(v) for v in phz_values)
    print(f"  Querying phz IN ({phz_str}) sources...")
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    df = pd.read_sql_query(query, conn)
    conn.close()

    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace("NOT_AVAILABLE", np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["phz"] = df["phz"].astype(int)
    print(f"  -> {len(df):,} total sources")
    for v in phz_values:
        lbl = PHZ_LABELS[v][0]
        n = (df["phz"] == v).sum()
        if n:
            print(f"     phz={v}  {lbl}: {n:,}")
    return df


# ── Cluster assignment ────────────────────────────────────────────────────────

def assign_to_clusters(src_emb: np.ndarray,
                       q1_emb: np.ndarray,
                       q1_labels: np.ndarray) -> np.ndarray:
    valid_mask = q1_labels >= 0
    if valid_mask.sum() == 0:
        return np.full(len(src_emb), -1, dtype=int)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(q1_emb[valid_mask])
    _, idx = nn.kneighbors(src_emb)
    idx = idx.ravel()
    valid_indices = np.where(valid_mask)[0]
    return q1_labels[valid_indices[idx]]


# ── Title factory ─────────────────────────────────────────────────────────────

def _make_title(phz_val: int, lbl: str, k: int):
    def _title(row, size, _phz=phz_val, _lbl=lbl, _k=k):
        return (f"phz={_phz} {_lbl}  Cluster {_k}\n"
                f"RA={float(row['ra']):.4f}  Dec={float(row['dec']):+.4f}\n"
                f"G={float(row['phot_g_mean_mag']):.2f}  "
                f"UMAP({float(row['umap1']):.1f},{float(row['umap2']):.1f})  "
                f"[{size:.1f}\"]")
    return _title


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Q1 VIS cutouts for non-QSO phz classes, stratified by cluster"
    )
    parser.add_argument("--phz", type=str, default=None,
                        help="Comma-separated phz values to download, e.g. --phz 1,2,-1. "
                             "Default: all of 1,2,3,-1,0.")
    parser.add_argument("--per-cluster", type=int, default=30,
                        help="Max sources per (phz × cluster) cell (default: 30).")
    parser.add_argument("--size-arcsec", type=float, default=6.0,
                        help="VIS cutout size in arcsec (default: 6.0).")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--dpi",        type=int,   default=90)
    parser.add_argument("--no-cutouts", action="store_true",
                        help="Skip cutout downloads; just show cluster assignment stats.")
    args = parser.parse_args()

    # Parse --phz argument (handle negatives like -1)
    if args.phz is not None:
        try:
            phz_values = [int(x.strip()) for x in args.phz.split(",")]
        except ValueError:
            print("ERROR: --phz must be a comma-separated list of integers, e.g. --phz 1,2,-1")
            return 1
        unknown = [v for v in phz_values if v not in PHZ_LABELS]
        if unknown:
            print(f"ERROR: Unknown phz values: {unknown}. "
                  f"Supported: {sorted(PHZ_LABELS.keys())}")
            return 1
    else:
        phz_values = DEFAULT_PHZ

    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER environment variable not set.")
        return 1

    print("=" * 60)
    phz_desc = ", ".join(f"phz={v} ({PHZ_LABELS[v][0]})" for v in phz_values)
    print(f"Downloading cutouts for: {phz_desc}")
    print("=" * 60)

    # ── 1. Load Q1 UMAP ───────────────────────────────────────────────────────
    print("\n[1] Loading Q1 UMAP embedding...")
    q1 = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    q1_emb  = q1["embedding"]
    q1_mean = q1["mean"]
    q1_std  = q1["std"]
    reducer = joblib.load(ARRAYS_DIR / "q1_indiv_reducer.joblib")
    print(f"  Q1 embedding: {q1_emb.shape[0]:,} points")

    # ── 2. Load cluster labels ─────────────────────────────────────────────────
    print("\n[2] Loading cluster labels...")
    labels_path = ARRAYS_DIR / "q1_cluster_labels.npy"
    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found.")
        print("  Run first:  python src/compute_q1_clusters.py")
        return 1
    q1_labels = np.load(labels_path)
    print(f"  Loaded from: {labels_path}")
    cluster_ids = sorted(set(q1_labels[q1_labels >= 0]))
    print(f"  Cluster sizes: " +
          "  ".join(f"C{k}={(q1_labels==k).sum():,}" for k in cluster_ids))

    # ── 3. Query selected phz classes from DB ─────────────────────────────────
    print("\n[3] Querying sources from DB...")
    df_all = query_phz_sources(username, phz_values)

    # ── 4. Project into UMAP space ────────────────────────────────────────────
    print("\n[4] Projecting into UMAP space...")
    X = df_all[SELECTED_GAIA_PARAMETERS].values.astype(float)
    std_safe = np.where(q1_std > 0, q1_std, 1.0)
    X_scaled = (X - q1_mean) / std_safe
    X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0.0)
    emb = reducer.transform(X_scaled)
    print(f"  Projected {len(emb):,} sources")

    df_all = df_all.copy()
    df_all["umap1"]   = emb[:, 0]
    df_all["umap2"]   = emb[:, 1]

    # ── 5. Assign to nearest Q1 cluster ───────────────────────────────────────
    print("\n[5] Assigning to nearest cluster...")
    src_cluster = assign_to_clusters(emb, q1_emb, q1_labels)
    df_all["cluster"] = src_cluster

    print("\n  Cluster assignment summary:")
    for v in phz_values:
        lbl = PHZ_LABELS[v][0]
        sub = df_all[df_all["phz"] == v]
        if len(sub) == 0:
            continue
        print(f"  phz={v} {lbl}:")
        for k in cluster_ids:
            n = (sub["cluster"] == k).sum()
            if n:
                print(f"    C{k}: {n:,}")
        n_noise = (sub["cluster"] == -1).sum()
        if n_noise:
            print(f"    unassigned: {n_noise:,}")

    if args.no_cutouts:
        print("\n[6] --no-cutouts set, skipping downloads.")
        return 0

    # ── 6. Discover tiles once for all sources ────────────────────────────────
    print("\n[6] Discovering tiles (one TAP query for all sources)...")
    tiles = _discover_tiles(df_all["ra"].min(),  df_all["ra"].max(),
                            df_all["dec"].min(), df_all["dec"].max(),
                            field_name="EDF-North")
    if tiles:
        print(f"  {len(tiles)} tile(s) found — reusing for all clusters.")
    else:
        print("  ESA TAP failed; each cluster will fall back to per-source IRSA TAP.")
        tiles = None

    # ── 7. Download cutouts per (phz × cluster) cell ──────────────────────────
    print("\n[7] Downloading cutouts per (phz × cluster) cell...")
    rng = np.random.default_rng(args.seed)

    for v in phz_values:
        lbl      = PHZ_LABELS[v][0]
        subdir   = PHZ_DIRS[v]
        phz_dir  = CUTOUTS_DIR / subdir
        df_phz   = df_all[df_all["phz"] == v]
        if len(df_phz) == 0:
            print(f"\n  phz={v} {lbl}: no sources, skipping.")
            continue

        print(f"\n  ── phz={v} {lbl} ({len(df_phz):,} sources) ──")
        phz_dir.mkdir(parents=True, exist_ok=True)

        for k in cluster_ids:
            cell_df = df_phz[df_phz["cluster"] == k]
            if len(cell_df) == 0:
                continue

            if len(cell_df) > args.per_cluster:
                cell_df = cell_df.sample(args.per_cluster,
                                         random_state=int(rng.integers(1e6)))
            cell_df = cell_df.reset_index(drop=True)

            cluster_dir = phz_dir / f"cluster{k}"
            print(f"\n    [cluster{k}]  {len(cell_df)} sources")

            make_q1_cutouts(
                cell_df,
                output_dir=cluster_dir,
                size_arcsec=args.size_arcsec,
                title_fn=_make_title(v, lbl, k),
                tiles=tiles,
                contact_sheet_stem="contact_sheet",
                manifest_stem="manifest",
                individuals_dir=cluster_dir / "cutouts",
                dpi=args.dpi,
                save_individuals=True,
            )

    print("\n" + "=" * 60)
    print(f"Done. Outputs in: {CUTOUTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
