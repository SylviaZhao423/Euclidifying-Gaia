#!/usr/bin/env python3
"""
Refit UMAP on Cluster-0 sources using only the 3 animation sort axes:
  AEN_sig     — astrometric_excess_noise_sig    (X_raw[:, 3])
  BPRPEF      — corrected_bp_rp_excess          (X_raw[:, 11])
  variability — sqrt(phot_g_n_obs) / phot_g_mean_flux_over_error  (WSDB)

Cluster 0 is the 95%-galaxy cluster (4,289 sources) in the original Q1 UMAP.
Fitting UMAP on only these 3 axes for only these sources reveals their
internal structure in the 3-parameter space.

Reads:
  output/umap/arrays/q1_indiv.npz
  output/umap/arrays/q1_cluster_labels.npy

Writes:
  output/umap/q1_cluster0/umap_cluster0_3ax.png

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_umap_cluster0_3ax.py
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
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "q1_cluster0"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

CLUSTER_ID = 0

AEN_IDX          = SELECTED_GAIA_PARAMETERS.index("astrometric_excess_noise_sig")
BPRPEF_IDX       = SELECTED_GAIA_PARAMETERS.index("corrected_bp_rp_excess")
FLUX_OVER_ERR_IDX = SELECTED_GAIA_PARAMETERS.index("phot_g_mean_flux_over_error")

AXIS_NAMES = ["AEN sig", "BPRP excess", "variability"]


# ── WSDB ─────────────────────────────────────────────────────────────────────

def fetch_phot_g_n_obs(source_ids: np.ndarray) -> dict[int, float]:
    import psycopg2
    pguser = os.environ.get("PGUSER", "")
    if not pguser:
        raise RuntimeError("PGUSER not set")
    conn = psycopg2.connect(
        host=WSDB_HOST, dbname=WSDB_DB, user=pguser, connect_timeout=30,
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CREATE TEMP TABLE _c0_sids (source_id BIGINT);")
    chunk = 2000
    for i in range(0, len(source_ids), chunk):
        vals = ",".join(f"({int(sid)})" for sid in source_ids[i : i + chunk])
        cur.execute(f"INSERT INTO _c0_sids VALUES {vals};")
    cur.execute("""
        SELECT s.source_id, s.phot_g_n_obs
        FROM   gaia_dr3.gaia_source s
        JOIN   _c0_sids t USING (source_id)
        WHERE  s.phot_g_n_obs IS NOT NULL;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {int(r[0]): float(r[1]) for r in rows}


# ── Feature build ─────────────────────────────────────────────────────────────

def build_3features(X_raw: np.ndarray,
                    source_ids: np.ndarray) -> np.ndarray:
    """Return (N, 3) array: [AEN_sig, BPRPEF, variability]."""
    aen    = X_raw[:, AEN_IDX].copy()
    bprpef = X_raw[:, BPRPEF_IDX].copy()
    foe    = X_raw[:, FLUX_OVER_ERR_IDX]

    print(f"  Fetching phot_g_n_obs from WSDB for {len(source_ids):,} sources...")
    n_obs_map = fetch_phot_g_n_obs(source_ids)
    print(f"  Received {len(n_obs_map):,} rows")

    variability = np.full(len(source_ids), np.nan)
    for i, sid in enumerate(source_ids):
        n = n_obs_map.get(int(sid))
        if n is not None and foe[i] > 0:
            variability[i] = np.sqrt(n) / foe[i]

    print(f"  variability finite for {np.isfinite(variability).sum():,}/{len(source_ids):,}")
    return np.column_stack([aen, bprpef, variability])


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

def plot_3ax(embedding: np.ndarray,
             X3: np.ndarray,
             y: np.ndarray,
             n_total: int,
             output_path: Path) -> None:
    """3-column figure — one column per axis, stars top / galaxies bottom."""
    n_axes = 3
    m_star = y == 0
    m_gal  = y == 1

    panel_h   = 1.0
    inner_gap = 0.04

    height_ratios = [panel_h, inner_gap, panel_h]
    fig_width  = n_axes * 4.0
    panel_fraction = panel_h / sum(height_ratios)
    per_panel_height = 1.5
    fig_height = per_panel_height / panel_fraction

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = GridSpec(3, n_axes, figure=fig,
                   height_ratios=height_ratios, hspace=0.04, wspace=0.35)

    xlim = (embedding[:, 0].min() - 0.5, embedding[:, 0].max() + 0.5)
    ylim = (embedding[:, 1].min() - 0.5, embedding[:, 1].max() + 0.5)

    for ci, name in enumerate(AXIS_NAMES):
        vals = X3[:, ci].astype(float)
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
                            vmin=lo, vmax=hi,
                            s=6, alpha=0.65, marker="o",
                            rasterized=True, linewidths=0, zorder=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_ylabel(lbl, fontsize=8)
            ax.grid(alpha=0.2)

        fig.colorbar(sc, ax=[ax_top, ax_bot], pad=0.02, fraction=0.046,
                     ).ax.tick_params(labelsize=7)
        ax_top.set_title(name, fontsize=10, fontweight="bold", pad=5)

    fig.suptitle(
        f"Q1 Cluster 0 UMAP  (n={n_total:,}, 95% galaxies)  "
        f"— 3-feature space: AEN sig · BPRP excess · variability  |  Upper = Stars, Lower = Galaxies",
        fontsize=11, fontweight="bold", y=0.98,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ──────────────────────────────────────────────────────────
    print("[1] Loading arrays...")
    data   = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    labels = np.load(ARRAYS_DIR / "q1_cluster_labels.npy")

    mask_c0    = labels == CLUSTER_ID
    X_raw      = data["X_raw"][mask_c0]
    y          = data["y"][mask_c0]
    source_ids = data["source_ids"][mask_c0]

    n = mask_c0.sum()
    n_gal = (y == 1).sum()
    print(f"  Cluster {CLUSTER_ID}: {n:,} sources  "
          f"(stars={int((y==0).sum()):,}, galaxies={n_gal:,}, "
          f"{100*n_gal/n:.0f}% gal)")

    # ── 2. Build 3-feature matrix ────────────────────────────────────────
    print("\n[2] Building 3-feature matrix...")
    X3 = build_3features(X_raw, source_ids)

    # ── 3. Standardize ───────────────────────────────────────────────────
    print("\n[3] Standardizing...")
    X_scaled = standardize(X3)

    # ── 4. Fit UMAP ──────────────────────────────────────────────────────
    print("\n[4] Fitting UMAP on 3 features for cluster-0 sources...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                        random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    print(f"  Embedding shape: {embedding.shape}")

    # ── 5. Plot ──────────────────────────────────────────────────────────
    print("\n[5] Plotting...")
    plot_3ax(embedding, X3, y, n_total=n,
             output_path=OUTPUT_DIR / "umap_cluster0_3ax.png")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
