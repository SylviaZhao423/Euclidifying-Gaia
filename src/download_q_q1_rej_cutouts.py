#!/usr/bin/env python3
"""
Download Euclid Q1 VIS cutouts for Rejected (phz=-1) sources,
stratified by UMAP cluster.

Workflow
--------
1. Load Q1 UMAP embedding (q1_indiv.npz) + reducer.
2. Load saved cluster labels (q1_cluster_labels.npy).
3. Query Rejected (phz=-1) sources from DB — RA/Dec + Gaia features.
4. Project each source into UMAP space via the fitted reducer.
5. Assign each source to the nearest Q1 base cluster.
6. Discover tiles once for all sources (single TAP query).
7. For every cluster, sample up to --per-cluster sources
   and download VIS cutouts via save_q1_utils.make_q1_cutouts().

Output layout
-------------
output/euclid_cutouts/q1_cutouts/rej/
  cluster{k}/
    contact_sheet.png
    manifest.csv
    cutouts/
      cutout_NNNN.png     (individual cutouts, for hover cache)

Usage
-----
  PGUSER=siwei_zhao_2025 python src/download_q_q1_rej_cutouts.py
  PGUSER=siwei_zhao_2025 python src/download_q_q1_rej_cutouts.py --per-cluster 8 --size-arcsec 8
  PGUSER=siwei_zhao_2025 python src/download_q_q1_rej_cutouts.py --per-cluster 20
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
OUTPUT_DIR  = PROJECT_DIR / "output" / "euclid_cutouts" / "q1_cutouts" / "rej"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
MAG_CUT   = 17


# ── DB query ──────────────────────────────────────────────────────────────────

def query_rejected(username: str) -> pd.DataFrame:
    """Query phz=-1 sources with RA/Dec and Gaia features."""
    import psycopg2
    gaia_cols = ", ".join([f"g_{col} AS {col}" for col in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT
            g_source_id AS source_id,
            m_right_ascension AS ra,
            m_declination     AS dec,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            {gaia_cols}
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE p_phz_classification = -1
          AND g_phot_g_mean_mag >= {MAG_CUT}
          AND m_right_ascension IS NOT NULL
          AND m_declination IS NOT NULL
        ORDER BY g_source_id
    """
    print("  Querying Rejected (phz=-1) sources...")
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    df = pd.read_sql_query(query, conn)
    conn.close()

    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace("NOT_AVAILABLE", np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"  -> {len(df):,} rejected sources")
    return df


# ── Cluster assignment ────────────────────────────────────────────────────────

def assign_to_clusters(rej_emb: np.ndarray,
                       q1_emb: np.ndarray,
                       q1_labels: np.ndarray) -> np.ndarray:
    valid_mask = q1_labels >= 0
    if valid_mask.sum() == 0:
        return np.full(len(rej_emb), -1, dtype=int)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(q1_emb[valid_mask])
    _, idx = nn.kneighbors(rej_emb)
    idx = idx.ravel()
    valid_indices = np.where(valid_mask)[0]
    return q1_labels[valid_indices[idx]]


# ── Title factory ─────────────────────────────────────────────────────────────

def _make_title(k: int):
    def _title(row, size, _k=k):
        return (f"Rejected (phz=-1)  Cluster {_k}\n"
                f"RA={float(row['ra']):.4f}  Dec={float(row['dec']):+.4f}\n"
                f"G={float(row['phot_g_mean_mag']):.2f}  "
                f"UMAP({float(row['umap1']):.1f},{float(row['umap2']):.1f})  "
                f"[{size:.1f}\"]")
    return _title


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Rejected-source UMAP-clump cutouts")
    parser.add_argument("--per-cluster", type=int, default=5,
                        help="Max sources per cluster (default: 5)")
    parser.add_argument("--size-arcsec", type=float, default=6.0,
                        help="VIS cutout size in arcsec (default: 6)")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--dpi",         type=int,   default=90)
    parser.add_argument("--no-cutouts",  action="store_true",
                        help="Skip cutout download (useful for testing cluster assignment)")
    args = parser.parse_args()

    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER environment variable not set.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load Q1 UMAP ───────────────────────────────────────────────────────
    print("=" * 60)
    print("[1] Loading Q1 UMAP embedding...")
    q1 = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    q1_emb  = q1["embedding"]
    q1_mean = q1["mean"]
    q1_std  = q1["std"]
    reducer = joblib.load(ARRAYS_DIR / "q1_indiv_reducer.joblib")
    print(f"  Q1 embedding: {q1_emb.shape[0]:,} points")

    # ── 2. Load cluster labels ────────────────────────────────────────────────
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

    # ── 3. Query rejected sources ─────────────────────────────────────────────
    print("\n[3] Querying Rejected sources from DB...")
    df_rej = query_rejected(username)

    # ── 4. Project into UMAP space ────────────────────────────────────────────
    print("\n[4] Projecting into UMAP space...")
    X = df_rej[SELECTED_GAIA_PARAMETERS].values.astype(float)
    std_safe = np.where(q1_std > 0, q1_std, 1.0)
    X_scaled = (X - q1_mean) / std_safe
    X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0.0)
    rej_emb  = reducer.transform(X_scaled)
    print(f"  Projected {len(rej_emb):,} sources")

    # ── 5. Assign to clusters ─────────────────────────────────────────────────
    print("\n[5] Assigning to nearest cluster...")
    rej_cluster = assign_to_clusters(rej_emb, q1_emb, q1_labels)
    df_rej = df_rej.copy()
    df_rej["umap1"]   = rej_emb[:, 0]
    df_rej["umap2"]   = rej_emb[:, 1]
    df_rej["cluster"] = rej_cluster

    print("\n  Cluster assignment:")
    for k in cluster_ids:
        n = (rej_cluster == k).sum()
        print(f"    C{k}: {n:,}")
    n_noise = (rej_cluster == -1).sum()
    if n_noise:
        print(f"    unassigned: {n_noise:,}")

    if args.no_cutouts:
        print("\n[6] --no-cutouts set, skipping downloads.")
        return 0

    # ── 6. Discover tiles once ────────────────────────────────────────────────
    print("\n[6] Discovering tiles (one TAP query for all clusters)...")
    tiles = _discover_tiles(df_rej["ra"].min(),  df_rej["ra"].max(),
                            df_rej["dec"].min(), df_rej["dec"].max(),
                            field_name="EDF-North")
    if tiles:
        print(f"  {len(tiles)} tile(s) found — reusing for all clusters.")
    else:
        print("  ESA TAP failed; each cluster will fall back to per-source IRSA TAP.")
        tiles = None

    # ── 7. Download cutouts per cluster ───────────────────────────────────────
    print("\n[7] Downloading cutouts per cluster...")
    rng = np.random.default_rng(args.seed)

    for k in cluster_ids:
        cell_df = df_rej[df_rej["cluster"] == k]
        if len(cell_df) == 0:
            continue

        if len(cell_df) > args.per_cluster:
            cell_df = cell_df.sample(args.per_cluster,
                                     random_state=int(rng.integers(1e6)))
        cell_df = cell_df.reset_index(drop=True)

        cluster_dir = OUTPUT_DIR / f"cluster{k}"
        print(f"\n  [cluster{k}]  {len(cell_df)} sources")

        make_q1_cutouts(
            cell_df,
            output_dir=cluster_dir,
            size_arcsec=args.size_arcsec,
            title_fn=_make_title(k),
            tiles=tiles,
            contact_sheet_stem="contact_sheet",
            manifest_stem="manifest",
            individuals_dir=cluster_dir / "cutouts",
            dpi=args.dpi,
            save_individuals=True,
        )

    print("\n" + "=" * 60)
    print(f"Done. Outputs in: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
