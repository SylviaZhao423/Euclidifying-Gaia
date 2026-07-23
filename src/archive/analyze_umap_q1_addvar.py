#!/usr/bin/env python3
"""
Refit UMAP on 15 features (14 model features + variability) and plot feature grid.

variability = sqrt(phot_g_n_obs) / phot_g_mean_flux_over_error
  — Belokurov-Erkal per-transit G-band scatter proxy; phot_g_n_obs fetched
    from WSDB since it is not stored in the original 14-feature matrix.

Reads:
  output/umap/arrays/q1_indiv.npz   (X_raw 14-feat, source_ids, y, embedding meta)

Writes (does NOT touch the original arrays/):
  output/umap/q1_addvar/q1_addvar.npz           — new 15-feat embedding
  output/umap/q1_addvar/q1_addvar_reducer.joblib — fitted 15-feat UMAP reducer
  output/umap/q1_addvar/q1_features_addvar.png  — 15-panel feature grid

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_umap_q1_addvar.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import umap

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "q1_addvar"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
MAG_CUT   = 17

FLUX_OVER_ERR_IDX = SELECTED_GAIA_PARAMETERS.index("phot_g_mean_flux_over_error")
FEATURE_NAMES     = list(SELECTED_GAIA_PARAMETERS) + ["variability"]


# ── WSDB fetch ───────────────────────────────────────────────────────────────

def fetch_phot_g_n_obs(source_ids: np.ndarray) -> dict[int, float]:
    import psycopg2
    pguser = os.environ.get("PGUSER", "")
    if not pguser:
        raise RuntimeError("PGUSER not set")
    print(f"  Connecting to WSDB as {pguser} ...")
    conn = psycopg2.connect(
        host=WSDB_HOST, dbname=WSDB_DB, user=pguser, connect_timeout=30,
    )
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("CREATE TEMP TABLE _addvar_sids (source_id BIGINT);")
    chunk = 5000
    for i in range(0, len(source_ids), chunk):
        vals = ",".join(f"({int(sid)})" for sid in source_ids[i : i + chunk])
        cur.execute(f"INSERT INTO _addvar_sids VALUES {vals};")

    cur.execute("""
        SELECT s.source_id, s.phot_g_n_obs
        FROM   gaia_dr3.gaia_source s
        JOIN   _addvar_sids t USING (source_id)
        WHERE  s.phot_g_n_obs IS NOT NULL;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {int(r[0]): float(r[1]) for r in rows}


def compute_variability(source_ids: np.ndarray,
                        flux_over_err: np.ndarray) -> np.ndarray:
    n_obs_map = fetch_phot_g_n_obs(source_ids)
    print(f"  phot_g_n_obs received for {len(n_obs_map):,}/{len(source_ids):,} sources")
    out = np.full(len(source_ids), np.nan)
    for i, sid in enumerate(source_ids):
        n = n_obs_map.get(int(sid))
        if n is not None and flux_over_err[i] > 0:
            out[i] = np.sqrt(n) / flux_over_err[i]
    print(f"  variability finite for {np.isfinite(out).sum():,} sources")
    return out


# ── Standardize ──────────────────────────────────────────────────────────────

def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median-fill NaNs, then Z-score. Returns (X_scaled, mean_, std_)."""
    X = X.copy()
    for col in range(X.shape[1]):
        bad = ~np.isfinite(X[:, col])
        if bad.any():
            X[bad, col] = np.nanmedian(X[:, col])
    mean_ = X.mean(axis=0)
    std_  = X.std(axis=0)
    X_scaled = (X - mean_) / np.where(std_ > 0, std_, 1.0)
    return X_scaled, mean_, std_


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_features(embedding, X_aug, y, output_dir,
                  filename="q1_features_addvar.png",
                  xlim=None, ylim=None, mag_label=None):
    n_features = len(FEATURE_NAMES)
    ncols = 4
    nfeature_rows = int(np.ceil(n_features / ncols))

    m_star = y == 0
    m_gal  = y == 1

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

    for fi, feat_name in enumerate(FEATURE_NAMES):
        feat_row = fi // ncols
        col      = fi % ncols
        ax_top = fig.add_subplot(gs[_gs_top(feat_row), col])
        ax_bot = fig.add_subplot(gs[_gs_bot(feat_row), col])

        vals = X_aug[:, fi].astype(float)
        lo, hi = np.nanpercentile(vals, 1), np.nanpercentile(vals, 99)
        vals_c = np.clip(vals, lo, hi)

        for ax, mask, label in [
            (ax_top, m_star, "Q1 Stars"),
            (ax_bot, m_gal,  "Q1 Galaxies"),
        ]:
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c="lightgrey", s=4, alpha=0.18, marker="o",
                       rasterized=True, linewidths=0, zorder=0)
            sc = ax.scatter(embedding[mask, 0], embedding[mask, 1],
                            c=vals_c[mask], cmap="RdYlBu_r",
                            vmin=lo, vmax=hi,
                            s=4, alpha=0.6, marker="o",
                            rasterized=True, linewidths=0, zorder=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(alpha=0.25)
            if xlim is not None: ax.set_xlim(xlim)
            if ylim is not None: ax.set_ylim(ylim)
            ax.set_ylabel(f"{label}\n(n={mask.sum():,})", fontsize=7)

        fig.colorbar(sc, ax=[ax_top, ax_bot], pad=0.02, fraction=0.046
                     ).ax.tick_params(labelsize=7)
        ax_top.set_title(feat_name, fontsize=8, fontweight="bold", pad=4)

    for fi in range(n_features, nfeature_rows * ncols):
        feat_row = fi // ncols
        col      = fi % ncols
        fig.add_subplot(gs[_gs_top(feat_row), col]).set_visible(False)
        fig.add_subplot(gs[_gs_bot(feat_row), col]).set_visible(False)

    ml = mag_label if mag_label is not None else f"G ≥ {MAG_CUT}"
    fig.suptitle(
        f"Q1 Individual UMAP coloured by Gaia parameters  ({ml},  {n_features} features)"
        f"  |  Upper = stars, Lower = galaxies",
        fontsize=12, fontweight="bold", y=0.93,
    )
    out = output_dir / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    npz_path = ARRAYS_DIR / "q1_indiv.npz"
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found")
        return 1

    # ── 1. Load existing arrays ──────────────────────────────────────────
    print("[1] Loading q1_indiv.npz...")
    data       = np.load(npz_path, allow_pickle=True)
    X_raw      = data["X_raw"]           # (50000, 14)
    y          = data["y"]
    source_ids = data["source_ids"]
    ra         = data["ra"]
    dec        = data["dec"]
    g_mag      = data["g_mag"]
    phz_gal_prob = data["phz_gal_prob"]
    model_pgal   = data["model_pgal"]
    mag_cut_faint = int(data["mag_cut_faint"]) if "mag_cut_faint" in data else 99
    mag_label  = f"G ≥ {MAG_CUT}" if mag_cut_faint >= 90 else f"{MAG_CUT} ≤ G ≤ {mag_cut_faint}"
    print(f"  {len(y):,} sources  (stars={int((y==0).sum()):,}, galaxies={int((y==1).sum()):,})")

    # ── 2. Fetch variability ─────────────────────────────────────────────
    print("\n[2] Fetching variability from WSDB...")
    flux_over_err = X_raw[:, FLUX_OVER_ERR_IDX]
    variability   = compute_variability(source_ids, flux_over_err)

    # ── 3. Build 15-feature matrix ───────────────────────────────────────
    print("\n[3] Building 15-feature matrix...")
    X_15 = np.column_stack([X_raw, variability])   # (50000, 15)
    print(f"  X_15 shape: {X_15.shape}")

    # ── 4. Standardize ───────────────────────────────────────────────────
    print("\n[4] Standardizing...")
    X_scaled, mean_15, std_15 = standardize(X_15)
    print(f"  NaN after fill: {(~np.isfinite(X_scaled)).sum()}")

    # ── 5. Refit UMAP on 15 features ────────────────────────────────────
    print("\n[5] Fitting UMAP on 15 features (this takes ~2-3 min)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                        random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    print(f"  Embedding shape: {embedding.shape}")

    # ── 6. Save arrays + reducer ─────────────────────────────────────────
    print("\n[6] Saving arrays...")
    np.savez(OUTPUT_DIR / "q1_addvar.npz",
             embedding=embedding,
             y=y,
             source_ids=source_ids,
             X_raw=X_15,
             mean=mean_15,
             std=std_15,
             ra=ra,
             dec=dec,
             g_mag=g_mag,
             phz_gal_prob=phz_gal_prob,
             model_pgal=model_pgal)
    joblib.dump(reducer, OUTPUT_DIR / "q1_addvar_reducer.joblib")
    print(f"  -> q1_addvar.npz")
    print(f"  -> q1_addvar_reducer.joblib")

    # ── 7. Plot ──────────────────────────────────────────────────────────
    print("\n[7] Plotting 15-feature grid...")
    xlim = (embedding[:, 0].min() - 1.0, embedding[:, 0].max() + 1.0)
    ylim = (embedding[:, 1].min() - 1.0, embedding[:, 1].max() + 1.0)
    plot_features(embedding, X_15, y, OUTPUT_DIR,
                  xlim=xlim, ylim=ylim, mag_label=mag_label)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
