#!/usr/bin/env python3
"""
UMAP of DESI spectroscopic features, labelled by Q1 PHZ and ERO class_star.

All sources in both crossmatch tables are DESI-confirmed galaxies; this shows
how the DESI observation/spectral features distribute and where the two Euclid
classifiers go wrong.

UMAP features (11 numeric DESI columns):
  d_desi_z, d_desi_zerr, d_desi_chi2, d_desi_deltachi2,
  d_desi_coadd_numexp, d_desi_coadd_exptime, d_desi_coadd_numnight,
  d_desi_coadd_numtile, d_desi_tsnr2_lrg, d_desi_tsnr2_elg, d_desi_tsnr2_bgs

Reads:
  siwei_zhao_2025.crossmatch_q1_mer_phz_desi
  siwei_zhao_2025.crossmatch_ero_gaia_0_3_desi

Writes:
  output/umap/desi/desi_labels.png   — 3-panel: dataset | Q1 label | ERO label
  output/umap/desi/desi_features.png — per-feature grid (upper=Q1, lower=ERO)

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_umap_desi_features.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "desi"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

DESI_FEATURES = [
    "d_desi_z",
    "d_desi_zerr",
    "d_desi_chi2",
    "d_desi_deltachi2",
    "d_desi_coadd_numexp",
    "d_desi_coadd_exptime",
    "d_desi_coadd_numnight",
    "d_desi_coadd_numtile",
    "d_desi_tsnr2_lrg",
    "d_desi_tsnr2_elg",
    "d_desi_tsnr2_bgs",
]

PHZ_COLORS = {
    "GALAXY":       "steelblue",
    "STAR":         "tomato",
    "QSO":          "goldenrod",
    "GLOB_CL":      "mediumseagreen",
    "UNCLASSIFIED": "silver",
}
ERO_COLORS = {"GALAXY": "steelblue", "STAR": "tomato"}

Q1_COLOR  = "#BA68C8"
ERO_COLOR = "#66BB6A"


# ── Query ─────────────────────────────────────────────────────────────────────

def query(username: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)

    feat_cols = ", ".join(DESI_FEATURES)

    q1_sql = f"""
        SELECT
            {feat_cols},
            CASE p_phz_classification
                WHEN 0 THEN 'UNCLASSIFIED'
                WHEN 1 THEN 'STAR'
                WHEN 2 THEN 'GALAXY'
                WHEN 3 THEN 'QSO'
                WHEN 4 THEN 'GLOB_CL'
                ELSE 'UNKNOWN'
            END AS phz_label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_desi
        WHERE {' AND '.join(f'{c} IS NOT NULL' for c in DESI_FEATURES)}
    """
    ero_sql = f"""
        SELECT
            {feat_cols},
            CASE WHEN e_v_class_star >= 0.65 THEN 'STAR' ELSE 'GALAXY' END AS ero_label
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3_desi
        WHERE {' AND '.join(f'{c} IS NOT NULL' for c in DESI_FEATURES)}
          AND e_v_class_star IS NOT NULL
    """

    print("  Querying Q1 × DESI crossmatch...")
    q1_df = pd.read_sql_query(q1_sql, conn)
    print(f"    {len(q1_df):,} sources")

    print("  Querying ERO × DESI crossmatch...")
    ero_df = pd.read_sql_query(ero_sql, conn)
    print(f"    {len(ero_df):,} sources")

    conn.close()
    return q1_df, ero_df


# ── UMAP fit ──────────────────────────────────────────────────────────────────

def fit_umap(X: np.ndarray) -> np.ndarray:
    import umap as umap_lib
    print(f"  Fitting UMAP on {X.shape[0]:,} sources × {X.shape[1]} features...")
    reducer = umap_lib.UMAP(n_neighbors=30, min_dist=0.1, random_state=42, verbose=False)
    return reducer.fit_transform(X)


# ── Plot 1: label panels ──────────────────────────────────────────────────────

def plot_labels(embedding: np.ndarray,
                is_q1: np.ndarray,
                phz_labels: np.ndarray,
                ero_labels: np.ndarray,
                output_path: Path) -> None:
    """3-panel label plot: dataset | Q1 PHZ label | ERO label."""
    is_ero = ~is_q1
    s = 6

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.3)

    # ── Panel 0: dataset membership ──────────────────────────────────────────
    ax = axes[0]
    ax.scatter(embedding[is_q1, 0], embedding[is_q1, 1],
               c=Q1_COLOR, s=s, alpha=0.5, linewidths=0, rasterized=True, label=f"Q1 (n={is_q1.sum():,})")
    ax.scatter(embedding[is_ero, 0], embedding[is_ero, 1],
               c=ERO_COLOR, s=s, alpha=0.5, linewidths=0, rasterized=True, label=f"ERO (n={is_ero.sum():,})")
    ax.set_title("Dataset", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, markerscale=2)

    # ── Panel 1: Q1 PHZ label ─────────────────────────────────────────────────
    ax = axes[1]
    # ERO sources as grey background
    ax.scatter(embedding[is_ero, 0], embedding[is_ero, 1],
               c="lightgrey", s=s, alpha=0.3, linewidths=0, rasterized=True, zorder=0)
    for label, color in PHZ_COLORS.items():
        mask = is_q1 & (phz_labels == label)
        if mask.sum() == 0:
            continue
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=color, s=s, alpha=0.7, linewidths=0, rasterized=True,
                   zorder=1, label=f"{label} (n={mask.sum():,})")
    ax.set_title("Q1 PHZ label  (ERO=grey)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, markerscale=2)

    # ── Panel 2: ERO class_star label ─────────────────────────────────────────
    ax = axes[2]
    # Q1 sources as grey background
    ax.scatter(embedding[is_q1, 0], embedding[is_q1, 1],
               c="lightgrey", s=s, alpha=0.3, linewidths=0, rasterized=True, zorder=0)
    for label, color in ERO_COLORS.items():
        mask = is_ero & (ero_labels == label)
        if mask.sum() == 0:
            continue
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=color, s=s, alpha=0.7, linewidths=0, rasterized=True,
                   zorder=1, label=f"{label} (n={mask.sum():,})")
    ax.set_title("ERO class_star label  (Q1=grey)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, markerscale=2)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(alpha=0.2)

    fig.suptitle(
        f"UMAP of DESI spectroscopic features  ({len(DESI_FEATURES)} features, "
        f"{len(embedding):,} sources)\n"
        "All sources are DESI-confirmed galaxies",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Plot 2: feature panels ────────────────────────────────────────────────────

def plot_features(embedding: np.ndarray,
                  X_raw: np.ndarray,
                  is_q1: np.ndarray,
                  output_path: Path) -> None:
    """Per-feature grid.  Upper panel = Q1 sources, lower panel = ERO sources."""
    is_ero = ~is_q1
    n_features = len(DESI_FEATURES)
    ncols = 4
    nfeature_rows = int(np.ceil(n_features / ncols))

    panel_h   = 1.0
    inner_gap = 0.04
    outer_gap = 0.32
    per_panel_height = 1.2

    height_ratios: list[float] = []
    for i in range(nfeature_rows):
        height_ratios.extend([panel_h, inner_gap, panel_h])
        if i < nfeature_rows - 1:
            height_ratios.append(outer_gap)

    panel_fraction = panel_h / sum(height_ratios)
    fig_width  = ncols * 3.8
    fig_height = per_panel_height / panel_fraction

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = GridSpec(len(height_ratios), ncols,
                   figure=fig, height_ratios=height_ratios,
                   hspace=0, wspace=0.35)

    def _gs_top(feat_row): return feat_row * 4
    def _gs_bot(feat_row): return feat_row * 4 + 2

    for fi, feat_name in enumerate(DESI_FEATURES):
        feat_row = fi // ncols
        col      = fi % ncols
        ax_top = fig.add_subplot(gs[_gs_top(feat_row), col])
        ax_bot = fig.add_subplot(gs[_gs_bot(feat_row), col])

        vals = X_raw[:, fi].astype(float)
        lo, hi = np.nanpercentile(vals, 2), np.nanpercentile(vals, 98)
        vals_clipped = np.clip(vals, lo, hi)

        for ax, mask, label in [
            (ax_top, is_q1,  "Q1"),
            (ax_bot, is_ero, "ERO"),
        ]:
            # grey background
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c="lightgrey", s=4, alpha=0.18,
                       rasterized=True, linewidths=0, zorder=0)
            sc = ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=vals_clipped[mask], cmap="RdYlBu_r",
                vmin=lo, vmax=hi,
                s=4, alpha=0.6, rasterized=True,
                linewidths=0, zorder=1,
            )
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(alpha=0.25)
            ax.set_ylabel(f"{label}\n(n={mask.sum():,})", fontsize=7)

        fig.colorbar(sc, ax=[ax_top, ax_bot], pad=0.02, fraction=0.046).ax.tick_params(labelsize=7)
        ax_top.set_title(feat_name, fontsize=8, fontweight="bold", pad=4)

    # hide unused panels
    for fi in range(n_features, nfeature_rows * ncols):
        feat_row = fi // ncols
        col      = fi % ncols
        fig.add_subplot(gs[_gs_top(feat_row), col]).set_visible(False)
        fig.add_subplot(gs[_gs_bot(feat_row), col]).set_visible(False)

    fig.suptitle(
        f"UMAP coloured by DESI feature values  ({n_features} features)  |  "
        "Upper = Q1 sources, Lower = ERO sources",
        fontsize=12, fontweight="bold", y=0.93,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("UMAP of DESI spectroscopic features — Q1 + ERO")
    print(f"  Features : {', '.join(DESI_FEATURES)}")
    print("=" * 65)

    # ── Step 1: query ──────────────────────────────────────────────────────
    print("\n[Step 1] Querying DB...")
    q1_df, ero_df = query(username)

    q1_df["_dataset"]  = "Q1"
    ero_df["_dataset"] = "ERO"

    # align columns: Q1 has phz_label, ERO has ero_label — add missing as NaN
    q1_df["ero_label"]  = np.nan
    ero_df["phz_label"] = np.nan

    combined = pd.concat([q1_df, ero_df], ignore_index=True)

    # ── Step 2: prepare feature matrix ────────────────────────────────────
    print("\n[Step 2] Preparing feature matrix...")
    X = combined[DESI_FEATURES].to_numpy(dtype=float)
    valid = np.all(np.isfinite(X), axis=1)
    print(f"  Dropping {(~valid).sum():,} rows with non-finite values — "
          f"{valid.sum():,} remaining")
    combined = combined.loc[valid].reset_index(drop=True)
    X = X[valid]

    X_scaled = StandardScaler().fit_transform(X)

    is_q1     = combined["_dataset"].to_numpy() == "Q1"
    phz_labels = combined["phz_label"].to_numpy(dtype=str)
    ero_labels = combined["ero_label"].to_numpy(dtype=str)

    # ── Step 3: fit UMAP ───────────────────────────────────────────────────
    print("\n[Step 3] Fitting UMAP...")
    embedding = fit_umap(X_scaled)

    # ── Step 4: plot labels ────────────────────────────────────────────────
    print("\n[Step 4] Plotting label panels...")
    plot_labels(embedding, is_q1, phz_labels, ero_labels,
                OUTPUT_DIR / "desi_labels.png")

    # ── Step 5: plot features ──────────────────────────────────────────────
    print("\n[Step 5] Plotting feature panels...")
    plot_features(embedding, X, is_q1, OUTPUT_DIR / "desi_features.png")

    print(f"\nAll outputs saved under: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
