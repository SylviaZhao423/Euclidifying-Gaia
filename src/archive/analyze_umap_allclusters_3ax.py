#!/usr/bin/env python3
"""
For each HDBSCAN cluster (0-4) in the Q1 individual UMAP, refit a new UMAP
using only those cluster's sources and only the 3 animation sort axes:
  AEN sig     — astrometric_excess_noise_sig   (X_raw[:, 3])
  BPRP excess — corrected_bp_rp_excess         (X_raw[:, 11])
  variability — sqrt(phot_g_n_obs) / phot_g_mean_flux_over_error  (WSDB)

phot_g_n_obs is fetched once for all 50k sources then filtered per cluster.

Reads:
  output/umap/arrays/q1_indiv.npz
  output/umap/arrays/q1_cluster_labels.npy

Writes (one file per cluster):
  output/umap/q1_cluster{i}/umap_cluster{i}_3ax.png

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_umap_allclusters_3ax.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import umap

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

AEN_IDX           = SELECTED_GAIA_PARAMETERS.index("astrometric_excess_noise_sig")
BPRPEF_IDX        = SELECTED_GAIA_PARAMETERS.index("corrected_bp_rp_excess")
FLUX_OVER_ERR_IDX = SELECTED_GAIA_PARAMETERS.index("phot_g_mean_flux_over_error")

AXIS_NAMES = ["AEN sig", "BPRP excess", "variability"]


# ── WSDB ─────────────────────────────────────────────────────────────────────

def fetch_phot_g_n_obs_all(source_ids: np.ndarray) -> dict[int, float]:
    """Fetch phot_g_n_obs for all source_ids in one WSDB query."""
    import psycopg2
    pguser = os.environ.get("PGUSER", "")
    if not pguser:
        raise RuntimeError("PGUSER not set")
    print(f"  Connecting to WSDB ({len(source_ids):,} sources)...")
    conn = psycopg2.connect(
        host=WSDB_HOST, dbname=WSDB_DB, user=pguser, connect_timeout=30,
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CREATE TEMP TABLE _allc_sids (source_id BIGINT);")
    chunk = 5000
    for i in range(0, len(source_ids), chunk):
        vals = ",".join(f"({int(sid)})" for sid in source_ids[i : i + chunk])
        cur.execute(f"INSERT INTO _allc_sids VALUES {vals};")
    cur.execute("""
        SELECT s.source_id, s.phot_g_n_obs
        FROM   gaia_dr3.gaia_source s
        JOIN   _allc_sids t USING (source_id)
        WHERE  s.phot_g_n_obs IS NOT NULL;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    result = {int(r[0]): float(r[1]) for r in rows}
    print(f"  Received {len(result):,} rows")
    return result


def compute_variability_array(source_ids: np.ndarray,
                               flux_over_err: np.ndarray,
                               n_obs_map: dict[int, float]) -> np.ndarray:
    out = np.full(len(source_ids), np.nan)
    for i, sid in enumerate(source_ids):
        n = n_obs_map.get(int(sid))
        if n is not None and flux_over_err[i] > 0:
            out[i] = np.sqrt(n) / flux_over_err[i]
    return out


# ── Helpers ───────────────────────────────────────────────────────────────────

def standardize(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    for col in range(X.shape[1]):
        bad = ~np.isfinite(X[:, col])
        if bad.any():
            X[bad, col] = np.nanmedian(X[:, col])
    mean_ = X.mean(axis=0)
    std_  = X.std(axis=0)
    return (X - mean_) / np.where(std_ > 0, std_, 1.0)


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_cluster(embedding: np.ndarray,
                 X3: np.ndarray,
                 y: np.ndarray,
                 cluster_id: int,
                 gal_pct: float,
                 output_path: Path) -> None:
    n_total = len(y)
    m_star  = y == 0
    m_gal   = y == 1

    panel_h   = 1.0
    inner_gap = 0.04
    height_ratios = [panel_h, inner_gap, panel_h]

    fig_width  = 3 * 4.0
    panel_fraction = panel_h / sum(height_ratios)
    fig_height = 1.5 / panel_fraction

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = GridSpec(3, 3, figure=fig,
                   height_ratios=height_ratios, hspace=0.04, wspace=0.35)

    xlim = (embedding[:, 0].min() - 0.5, embedding[:, 0].max() + 0.5)
    ylim = (embedding[:, 1].min() - 0.5, embedding[:, 1].max() + 0.5)

    for ci, name in enumerate(AXIS_NAMES):
        vals  = X3[:, ci].astype(float)
        lo, hi = np.nanpercentile(vals, 1), np.nanpercentile(vals, 99)
        vals_c = np.clip(vals, lo, hi)

        ax_top = fig.add_subplot(gs[0, ci])
        ax_bot = fig.add_subplot(gs[2, ci])

        for ax, mask, lbl in [
            (ax_top, m_star, f"Q1 Stars  (n={m_star.sum():,})"),
            (ax_bot, m_gal,  f"Q1 Galaxies  (n={m_gal.sum():,})"),
        ]:
            ax.scatter(embedding[:, 0], embedding[:, 1],
                       c="lightgrey", s=6, alpha=0.2, marker="o",
                       rasterized=True, linewidths=0, zorder=0)
            sc = ax.scatter(embedding[mask, 0], embedding[mask, 1],
                            c=vals_c[mask], cmap="RdYlBu_r",
                            vmin=lo, vmax=hi, s=6, alpha=0.65, marker="o",
                            rasterized=True, linewidths=0, zorder=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_ylabel(lbl, fontsize=8)
            ax.grid(alpha=0.2)

        fig.colorbar(sc, ax=[ax_top, ax_bot], pad=0.02, fraction=0.046,
                     ).ax.tick_params(labelsize=7)
        ax_top.set_title(name, fontsize=10, fontweight="bold", pad=5)

    fig.suptitle(
        f"Q1 Cluster {cluster_id} UMAP  (n={n_total:,}, {gal_pct:.0f}% galaxies)  "
        f"— 3-feature space: AEN sig · BPRP excess · variability  |  Upper = Stars, Lower = Galaxies",
        fontsize=11, fontweight="bold", y=0.98,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    # ── 1. Load all arrays ───────────────────────────────────────────────
    print("[1] Loading arrays...")
    data       = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    labels_all = np.load(ARRAYS_DIR / "q1_cluster_labels.npy")
    X_raw_all  = data["X_raw"]
    y_all      = data["y"]
    sids_all   = data["source_ids"]

    cluster_ids = sorted(set(labels_all[labels_all >= 0]))
    print(f"  Clusters found: {cluster_ids}")
    for cid in cluster_ids:
        mask = labels_all == cid
        n = mask.sum()
        ngal = (y_all[mask] == 1).sum()
        print(f"  C{cid}: {n:,}  ({100*ngal/n:.0f}% gal)")

    # ── 2. Fetch phot_g_n_obs once for all 50k sources ──────────────────
    print("\n[2] Fetching phot_g_n_obs from WSDB (all 50k, one query)...")
    n_obs_map = fetch_phot_g_n_obs_all(sids_all)

    # Precompute variability for all sources
    foe_all         = X_raw_all[:, FLUX_OVER_ERR_IDX]
    variability_all = compute_variability_array(sids_all, foe_all, n_obs_map)
    finite = np.isfinite(variability_all).sum()
    print(f"  variability finite for {finite:,}/{len(sids_all):,}")

    # ── 3. Loop over clusters ────────────────────────────────────────────
    for cid in cluster_ids:
        print(f"\n{'='*55}")
        print(f"[Cluster {cid}]")
        mask = labels_all == cid
        X_raw = X_raw_all[mask]
        y     = y_all[mask]
        var   = variability_all[mask]
        n     = mask.sum()
        ngal  = (y == 1).sum()
        gal_pct = 100 * ngal / n
        print(f"  {n:,} sources  (stars={int((y==0).sum()):,}, galaxies={ngal:,}, {gal_pct:.0f}% gal)")

        # Build 3-feature matrix
        X3 = np.column_stack([
            X_raw[:, AEN_IDX],
            X_raw[:, BPRPEF_IDX],
            var,
        ])

        # Standardize
        X_scaled = standardize(X3)

        # Fit UMAP
        print(f"  Fitting UMAP...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                            random_state=42)
        embedding = reducer.fit_transform(X_scaled)

        # Plot
        out_path = PROJECT_DIR / "output" / "umap" / "q1_clusters_3ax" / f"umap_cluster{cid}_3ax.png"
        plot_cluster(embedding, X3, y,
                     cluster_id=cid, gal_pct=gal_pct,
                     output_path=out_path)

    print(f"\n{'='*55}")
    print("Done — all clusters plotted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
