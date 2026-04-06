#!/usr/bin/env python3
"""
Download Euclid Q1 VIS cutouts for QSO-variant sources (phz=4/5/6/7),
stratified by UMAP clump.

Workflow
--------
1. Load Q1 UMAP embedding (q1_indiv.npz) + reducer.
2. Load saved cluster labels (q1_cluster_labels.npy) from analyze_umap_q1_qso_3d.py,
   or fall back to re-clustering with HDBSCAN if not found.
3. Query QSO variants (phz=4,5,6,7) from DB — RA/Dec + Gaia features.
4. Project each QSO source into UMAP space via the fitted reducer.
5. Assign each QSO to the nearest Q1 base point's cluster.
6. Discover tiles once for all sources (single TAP query).
7. For every (cluster × phz_class) cell, sample up to --per-cell sources
   and download VIS cutouts via save_q1_utils.make_q1_cutouts().

Output layout
-------------
output/euclid_cutouts/q1_cutouts/qso/
  cluster{k}/
    contact_sheet_phz{v}.png
    manifest_phz{v}.csv
    cutout_phz{v}/
      cutout_NNNN.png     (individual cutouts, for hover cache)

Usage
-----
  PGUSER=siwei_zhao_2025 python src/download_q_q1_qso_cutouts.py
  PGUSER=siwei_zhao_2025 python src/download_q_q1_qso_cutouts.py --per-cell 8 --size-arcsec 8
  PGUSER=siwei_zhao_2025 python src/download_q_q1_qso_cutouts.py --min-cluster-size 100
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

PROJECT_DIR  = Path(__file__).resolve().parents[1]
ARRAYS_DIR   = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR   = PROJECT_DIR / "output" / "euclid_cutouts" / "q1_cutouts" / "qso"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
MAG_CUT   = 17

# PHZ QSO variant labels
QSO_LABELS = {
    4: ("QSO",              "green"),
    5: ("Multi (Star∩QSO)", "red"),
    6: ("Multi (Gal∩QSO)",  "steelblue"),
    7: ("Outlier∩QSO",      "orange"),
}

# ── Data loading ──────────────────────────────────────────────────────────────

def query_qso_variants(username: str) -> pd.DataFrame:
    """Query phz=4/5/6/7 sources with RA/Dec and Gaia features."""
    import psycopg2
    gaia_cols = ", ".join([f"g_{col} AS {col}" for col in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT
            g_source_id AS source_id,
            m_right_ascension AS ra,
            m_declination     AS dec,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            p_phz_classification AS phz,
            {gaia_cols}
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE p_phz_classification IN (4, 5, 6, 7)
          AND g_phot_g_mean_mag >= {MAG_CUT}
          AND m_right_ascension IS NOT NULL
          AND m_declination IS NOT NULL
        ORDER BY g_source_id
    """
    print("  Querying QSO variants (phz=4/5/6/7)...")
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    df = pd.read_sql_query(query, conn)
    conn.close()

    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace("NOT_AVAILABLE", np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["phz"] = df["phz"].astype(int)
    print(f"  -> {len(df):,} QSO-variant sources")
    for v, (lbl, _) in QSO_LABELS.items():
        n = (df["phz"] == v).sum()
        if n:
            print(f"     phz={v}  {lbl}: {n:,}")
    return df



def assign_to_clusters(qso_emb: np.ndarray,
                       q1_emb: np.ndarray,
                       q1_labels: np.ndarray) -> np.ndarray:
    """
    Assign each projected QSO point to the cluster of its nearest Q1 base point.
    Noise points (label=-1) in Q1 base are skipped; QSO points nearest only
    to noise are labelled -1.
    """
    valid_mask = q1_labels >= 0
    if valid_mask.sum() == 0:
        return np.full(len(qso_emb), -1, dtype=int)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(q1_emb[valid_mask])
    _, idx = nn.kneighbors(qso_emb)
    idx = idx.ravel()
    valid_indices = np.where(valid_mask)[0]
    return q1_labels[valid_indices[idx]]


# ── Title factory ─────────────────────────────────────────────────────────────

def _make_title(lbl: str):
    """Return a title_fn that captures the phz label by value."""
    def _title(row, size, _lbl=lbl):
        return (f"phz={int(row['phz'])} {_lbl}\n"
                f"RA={float(row['ra']):.4f}  Dec={float(row['dec']):+.4f}\n"
                f"G={float(row['phot_g_mean_mag']):.2f}  "
                f"UMAP({float(row['umap1']):.1f},{float(row['umap2']):.1f})  "
                f"[{size:.1f}\"]")
    return _title


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="QSO UMAP-clump cutouts")
    parser.add_argument("--per-cell",         type=int,   default=5,
                        help="Max sources per (cluster × phz) cell (default: 5)")
    parser.add_argument("--size-arcsec",      type=float, default=6.0,
                        help="VIS cutout size in arcsec (default: 6)")
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--dpi",              type=int,   default=90)
    parser.add_argument("--no-cutouts",       action="store_true",
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
          "  ".join(f"C{k}={( q1_labels==k).sum():,}" for k in cluster_ids))

    # ── 3. Query QSO variants ─────────────────────────────────────────────────
    print("\n[3] Querying QSO variants from DB...")
    df_qso = query_qso_variants(username)

    # ── 4. Project QSOs into UMAP space ──────────────────────────────────────
    print("\n[4] Projecting QSO sources into UMAP space...")
    X = df_qso[SELECTED_GAIA_PARAMETERS].values.astype(float)
    std_safe = np.where(q1_std > 0, q1_std, 1.0)
    X_scaled = (X - q1_mean) / std_safe
    X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0.0)
    qso_emb  = reducer.transform(X_scaled)
    print(f"  Projected {len(qso_emb):,} QSO sources")

    # ── 5. Assign to clusters ─────────────────────────────────────────────────
    print("\n[5] Assigning QSO sources to nearest cluster...")
    qso_cluster = assign_to_clusters(qso_emb, q1_emb, q1_labels)
    df_qso = df_qso.copy()
    df_qso["umap1"]   = qso_emb[:, 0]
    df_qso["umap2"]   = qso_emb[:, 1]
    df_qso["cluster"] = qso_cluster

    print("\n  Cluster × phz assignment:")
    print(f"  {'Cluster':>8}  " + "  ".join(f"phz={v:1d}" for v in QSO_LABELS))
    for k in cluster_ids + ([-1] if (qso_cluster == -1).any() else []):
        row_str = f"  {'C'+str(k):>8}  "
        row_str += "  ".join(
            f"{(df_qso[(df_qso['cluster']==k)&(df_qso['phz']==v)].shape[0]):>5}"
            for v in QSO_LABELS
        )
        print(row_str)

    if args.no_cutouts:
        print("\n[6] --no-cutouts set, skipping downloads.")
        return 0

    # ── 6. Discover tiles once for all sources ────────────────────────────────
    print("\n[6] Discovering tiles (one TAP query for all cells)...")
    tiles = _discover_tiles(df_qso["ra"].min(),  df_qso["ra"].max(),
                            df_qso["dec"].min(), df_qso["dec"].max(),
                            field_name="EDF-North")
    if tiles:
        print(f"  {len(tiles)} tile(s) found — reusing for all cells.")
    else:
        print("  ESA TAP failed; each cell will fall back to per-source IRSA TAP.")
        tiles = None

    # ── 7. Download cutouts per (cluster × phz) ───────────────────────────────
    print("\n[7] Downloading cutouts per cluster × phz class...")
    rng = np.random.default_rng(args.seed)

    for k in cluster_ids:
        for v, (lbl, _) in QSO_LABELS.items():
            cell_df = df_qso[(df_qso["cluster"] == k) & (df_qso["phz"] == v)]
            if len(cell_df) == 0:
                continue

            if len(cell_df) > args.per_cell:
                cell_df = cell_df.sample(args.per_cell,
                                         random_state=int(rng.integers(1e6)))
            cell_df = cell_df.reset_index(drop=True)

            cluster_dir = OUTPUT_DIR / f"cluster{k}"
            print(f"\n  [cluster{k} phz={v}]  {lbl}  — {len(cell_df)} sources")

            make_q1_cutouts(
                cell_df,
                output_dir=cluster_dir,
                size_arcsec=args.size_arcsec,
                title_fn=_make_title(lbl),
                tiles=tiles,
                contact_sheet_stem=f"contact_sheet_phz{v}",
                manifest_stem=f"manifest_phz{v}",
                individuals_dir=cluster_dir / f"cutout_phz{v}",
                dpi=args.dpi,
                save_individuals=True,
            )

    print("\n" + "=" * 60)
    print(f"Done. Outputs in: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
