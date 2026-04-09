#!/usr/bin/env python3
"""
Q1 individual UMAP coloured by Euclid features.

Fixed single-row layout (5 columns):
  ellipticity | flux_vis_psf | fluxerr_vis_psf | fluxsig_vis_psf

Reads:
  output/umap/arrays/q1_indiv.npz            — UMAP embedding + source_ids
  output/umap/arrays/q1_features.csv  — cached features (optional)

Writes:
  output/umap/arrays/q1_features.csv
  output/umap/q1_individual/q1_features.png

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_umap_q1_features.py
  python src/analyze_umap_q1_features.py --csv   # re-plot from CSV
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "q1_individual"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
MAG_CUT   = 17

ORDERED_FEATURES = [
    "ellipticity",
    "flux_vis_psf",
    "fluxerr_vis_psf",
    "fluxsig_vis_psf",
]
NCOLS = 4

# DB columns to fetch (bare names, queried with m_ prefix)
DB_COLS = [
    "ellipticity",
    "flux_vis_psf",
    "fluxerr_vis_psf",
]


# ── DB query ──────────────────────────────────────────────────────────────────

def query_features(username: str) -> pd.DataFrame:
    """Query Euclid features from crossmatch table."""
    import psycopg2

    conn   = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    select = ",\n            ".join([f"m_{c} AS {c}" for c in DB_COLS])
    query  = f"""
        SELECT
            g_source_id          AS source_id,
            g_phot_g_mean_mag    AS phot_g_mean_mag,
            p_phz_classification AS phz,
            {select}
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE g_phot_g_mean_mag >= {MAG_CUT}
          AND g_source_id IS NOT NULL
        ORDER BY g_source_id
    """
    print("  Querying crossmatch_q1_mer_phz_gaia for Euclid features...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  -> {len(df):,} sources")
    return df


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_euclid_features(embedding, y, feat_vals: dict[str, np.ndarray],
                         output_dir: Path, filename="q1_q1features.png",
                         xlim=None, ylim=None, n_total: int = 0):
    """One-row × 5-column UMAP grid coloured by Euclid features (stars upper, galaxies lower)."""
    ncols         = NCOLS
    nfeature_rows = len(ORDERED_FEATURES) // ncols

    per_panel_height = 1.2
    panel_h   = 1.0
    inner_gap = 0.04
    outer_gap = 0.32

    height_ratios = []
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

    m_star = y == 0
    m_gal  = y == 1

    n_plotted = 0
    for fi, feat_name in enumerate(ORDERED_FEATURES):
        feat_row = fi // ncols
        col      = fi % ncols

        ax_top = fig.add_subplot(gs[_gs_top(feat_row), col])
        ax_bot = fig.add_subplot(gs[_gs_bot(feat_row), col])

        if feat_name not in feat_vals or not np.isfinite(feat_vals[feat_name]).any():
            ax_top.set_visible(False)
            ax_bot.set_visible(False)
            continue

        vals   = feat_vals[feat_name].astype(float)
        finite = vals[np.isfinite(vals)]
        lo, hi = np.nanpercentile(finite, 2), np.nanpercentile(finite, 98)
        if lo == hi:
            lo, hi = finite.min(), finite.max()
        vals_clipped = np.clip(vals, lo, hi)
        n_plotted += 1

        for ax, mask, label in [
                (ax_top, m_star, "Stars"),
                (ax_bot, m_gal,  "Galaxies")]:
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c="lightgrey", s=4, alpha=0.18, marker="o",
                rasterized=True, linewidths=0, zorder=0,
            )
            valid = mask & np.isfinite(vals)
            sc = ax.scatter(
                embedding[valid, 0], embedding[valid, 1],
                c=vals_clipped[valid], cmap="RdYlBu_r",
                vmin=lo, vmax=hi,
                s=4, alpha=0.6, marker="o", rasterized=True,
                linewidths=0, zorder=1,
            )
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(alpha=0.25)
            if xlim is not None: ax.set_xlim(xlim)
            if ylim is not None: ax.set_ylim(ylim)
            ax.set_ylabel(f"Q1 {label}\n(n={valid.sum():,})", fontsize=7)

        fig.colorbar(sc, ax=[ax_top, ax_bot],
                     pad=0.02, fraction=0.046).ax.tick_params(labelsize=7)
        ax_top.set_title(feat_name, fontsize=8, fontweight="bold", pad=4)

    fig.suptitle(
        f"Q1 Individual UMAP colored by Q1 features  "
        f"(G ≥ {MAG_CUT},  {n_plotted} features) |  Upper = ERO stars · Lower = ERO galaxies",
        fontsize=11, fontweight="bold", y=1.08,
    )
    out = output_dir / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot Q1 UMAP coloured by Euclid features"
    )
    parser.add_argument("--csv", action="store_true",
                        help="Skip DB query; re-plot from saved CSV.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load UMAP embedding ────────────────────────────────────────────────
    indiv_path = ARRAYS_DIR / "q1_indiv.npz"
    if not indiv_path.exists():
        print(f"ERROR: {indiv_path} not found. Run analyze_umap.py first.")
        return 1
    print("[1] Loading Q1 UMAP embedding...")
    data       = np.load(indiv_path, allow_pickle=True)
    embedding  = data["embedding"]
    y          = data["y"]
    source_ids = data["source_ids"].astype(str)
    print(f"  {embedding.shape[0]:,} sources  (0=STAR: {(y==0).sum():,}  1=GAL: {(y==1).sum():,})")

    csv_path = ARRAYS_DIR / "q1_features.csv"

    # ── 2. Load or query features ─────────────────────────────────────────────
    if args.csv:
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found. Run without --csv first.")
            return 1
        print(f"\n[2] Loading features from {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
        username = os.getenv("PGUSER")
        if not username:
            print("ERROR: PGUSER not set.")
            return 1
        print("\n[2] Querying Euclid features from DB...")
        df = query_features(username)
        df["source_id"] = df["source_id"].astype(str)

        # Compute flux significance
        df["fluxsig_vis_psf"] = (
            df["flux_vis_psf"] / df["fluxerr_vis_psf"].replace(0, np.nan)
        )

        save_cols = (["source_id", "phot_g_mean_mag", "phz"]
                     + DB_COLS + ["fluxsig_vis_psf"])
        df[save_cols].to_csv(csv_path, index=False)
        print(f"  Saved → {csv_path}")

    # Recompute fluxsig if missing from old CSV
    if "fluxsig_vis_psf" not in df.columns:
        df["fluxsig_vis_psf"] = (
            df["flux_vis_psf"] / df["fluxerr_vis_psf"].replace(0, np.nan)
        )

    # ── 3. Merge with UMAP source IDs ─────────────────────────────────────────
    print("\n[3] Merging features with UMAP source IDs...")
    df["source_id"] = df["source_id"].astype(str)
    sid_to_idx = {sid: i for i, sid in enumerate(df["source_id"])}
    umap_idx   = np.array([sid_to_idx.get(sid, -1) for sid in source_ids])
    n_matched  = (umap_idx >= 0).sum()
    print(f"  Matched {n_matched:,} / {len(source_ids):,} UMAP sources")

    feat_cols = [f for f in ORDERED_FEATURES if f is not None and f in df.columns]
    feat_vals: dict[str, np.ndarray] = {}
    for col in feat_cols:
        arr   = np.full(len(embedding), np.nan)
        valid = umap_idx >= 0
        arr[valid] = pd.to_numeric(df[col].values[umap_idx[valid]], errors="coerce")
        n_finite = np.isfinite(arr).sum()
        if n_finite > 0:
            feat_vals[col] = arr
            print(f"    {col:<40} {n_finite:>8,} finite values")
        else:
            print(f"    {col:<40}  (all NaN — skipped)")

    # ── 4. Plot ────────────────────────────────────────────────────────────────
    xlim = (embedding[:, 0].min() - 1.0, embedding[:, 0].max() + 1.0)
    ylim = (embedding[:, 1].min() - 1.0, embedding[:, 1].max() + 1.0)

    print("\n[4] Plotting Q1 UMAP coloured by Euclid features...")
    plot_euclid_features(
        embedding, y, feat_vals, OUTPUT_DIR,
        xlim=xlim, ylim=ylim,
        n_total=n_matched,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
