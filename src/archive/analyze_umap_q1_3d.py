#!/usr/bin/env python3
"""
Interactive 3D UMAP plot for Q1 QSO-variant sources.

Option B: existing 2D UMAP x/y + Gaia G magnitude as z-axis.
Q1 base points are coloured by HDBSCAN/KMeans cluster.
QSO variants (phz=4/5/6/7) are overlaid as diamonds.

Reads:
  output/umap/17≤G/arrays/q1_indiv.npz
  output/umap/17≤G/arrays/q1_indiv_reducer.joblib

Writes:
  output/q1_qso_umap_cutouts/umap_clusters_3d.html

Usage:
  PGUSER=siwei_zhao_2025 python src/plot_q1_umap_3d.py
  PGUSER=siwei_zhao_2025 python src/plot_q1_umap_3d.py --min-cluster-size 100
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

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "17≤G" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "output" / "q1_qso_umap_cutouts"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
MAG_CUT   = 17

QSO_LABELS = {
    4: ("QSO",              "green"),
    5: ("Multi (Star∩QSO)", "red"),
    6: ("Multi (Gal∩QSO)",  "steelblue"),
    7: ("Outlier∩QSO",      "orange"),
}


def query_qso_variants(username: str) -> pd.DataFrame:
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


def cluster_q1_umap(embedding: np.ndarray, n_expected: int = 5,
                    min_cluster_size: int = 150) -> np.ndarray:
    try:
        import hdbscan
        print(f"  Clustering with HDBSCAN (min_cluster_size={min_cluster_size})...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                     min_samples=max(10, min_cluster_size // 10),
                                     cluster_selection_epsilon=0.3)
        labels = clusterer.fit_predict(embedding)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"  -> {n_clusters} clusters found  ({(labels==-1).sum():,} noise points)")
    except ImportError:
        print(f"  hdbscan not available — falling back to KMeans(k={n_expected})...")
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=n_expected, random_state=42,
                        n_init=10).fit_predict(embedding)
        print(f"  -> {n_expected} clusters (KMeans)")
    return labels


def plot_umap_3d_plotly(q1_emb, q1_labels, q1_gmag,
                        qso_emb, qso_phz, qso_gmag,
                        output_dir):
    import plotly.graph_objects as go

    cluster_ids = sorted(set(q1_labels[q1_labels >= 0]))
    tab10 = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    traces = []

    # Q1 base points, coloured by cluster
    noise_mask = q1_labels == -1
    if noise_mask.any():
        z_noise = q1_gmag[noise_mask] if q1_gmag is not None else np.zeros(noise_mask.sum())
        traces.append(go.Scatter3d(
            x=q1_emb[noise_mask, 0], y=q1_emb[noise_mask, 1], z=z_noise,
            mode="markers",
            marker=dict(size=1.5, color="lightgray", opacity=0.15),
            name="noise",
        ))
    for k in cluster_ids:
        m = q1_labels == k
        z_k = q1_gmag[m] if q1_gmag is not None else np.zeros(m.sum())
        traces.append(go.Scatter3d(
            x=q1_emb[m, 0], y=q1_emb[m, 1], z=z_k,
            mode="markers",
            marker=dict(size=2, color=tab10[k % len(tab10)], opacity=0.25),
            name=f"Q1 Cluster {k}",
        ))

    # QSO variants overlaid
    phz_colors = {4: "green", 5: "red", 6: "steelblue", 7: "orange"}
    for v, (lbl, _) in QSO_LABELS.items():
        m = qso_phz == v
        if not m.any():
            continue
        z_q = qso_gmag[m] if qso_gmag is not None else np.zeros(m.sum())
        traces.append(go.Scatter3d(
            x=qso_emb[m, 0], y=qso_emb[m, 1], z=z_q,
            mode="markers",
            marker=dict(size=5, color=phz_colors.get(v, "black"),
                        symbol="diamond", opacity=0.9,
                        line=dict(width=0.5, color="black")),
            name=f"phz={v} {lbl}",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Q1 UMAP — clusters + QSO variants<br>(UMAP 1 × UMAP 2 × G magnitude)",
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="G magnitude",
            zaxis=dict(autorange="reversed"),  # faint (large G) at top
        ),
        legend=dict(itemsizing="constant", font=dict(size=11)),
        margin=dict(l=0, r=0, b=0, t=50),
        width=1100, height=750,
    )

    out = output_dir / "umap_clusters_3d.html"
    fig.write_html(str(out))
    print(f"  -> {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive 3D UMAP for Q1 QSO variants")
    parser.add_argument("--n-clusters",       type=int, default=5,
                        help="Expected number of UMAP clumps (default: 5)")
    parser.add_argument("--min-cluster-size", type=int, default=150,
                        help="HDBSCAN min_cluster_size — smaller → more clusters (default: 150)")
    args = parser.parse_args()

    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER environment variable not set.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Q1 UMAP
    print("[1] Loading Q1 UMAP embedding...")
    q1 = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    q1_emb  = q1["embedding"]
    q1_mean = q1["mean"]
    q1_std  = q1["std"]
    q1_gmag = q1["g_mag"].astype(float) if "g_mag" in q1 else None
    reducer = joblib.load(ARRAYS_DIR / "q1_indiv_reducer.joblib")
    print(f"  Q1 embedding: {q1_emb.shape[0]:,} points"
          + ("" if q1_gmag is not None else "  (no G mag in npz — z-axis unavailable for Q1 base)"))

    # Cluster
    print("\n[2] Clustering Q1 UMAP...")
    q1_labels = cluster_q1_umap(q1_emb, n_expected=args.n_clusters,
                                 min_cluster_size=args.min_cluster_size)

    # Query QSO variants
    print("\n[3] Querying QSO variants from DB...")
    df_qso = query_qso_variants(username)

    # Project QSOs into UMAP space
    print("\n[4] Projecting QSO sources into UMAP space...")
    X = df_qso[SELECTED_GAIA_PARAMETERS].values.astype(float)
    std_safe = np.where(q1_std > 0, q1_std, 1.0)
    X_scaled = (X - q1_mean) / std_safe
    X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0.0)
    qso_emb  = reducer.transform(X_scaled)
    print(f"  Projected {len(qso_emb):,} QSO sources")

    # Save 3D HTML
    print("\n[5] Saving interactive 3D plot...")
    qso_gmag = df_qso["phot_g_mean_mag"].values.astype(float)
    plot_umap_3d_plotly(q1_emb, q1_labels, q1_gmag,
                        qso_emb, df_qso["phz"].values, qso_gmag,
                        OUTPUT_DIR)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
