#!/usr/bin/env python3
"""
Plot Q1 PHZ galaxies on the UMAP, split by whether Gaia DSC agrees.

DSC = Discrete Source Classifier from Gaia DR3 (De Angeli+ 2023),
which uses BP/RP spectra to classify sources as star/galaxy/quasar.

Panels:
  Row 0 (scatter + density):
    col 0  PHZ galaxies where DSC agrees (classprob_dsc_combmod_galaxy >= 0.5)
    col 1  PHZ galaxies where DSC disagrees (classprob_dsc_combmod_galaxy < 0.5)
    col 2  DSC P(galaxy) continuous colour (all Q1)
    col 3  XGBoost P(galaxy) continuous colour (all Q1)
    col 4  Sources in galaxy_candidates table

Reads:
  output/umap/arrays/q1_indiv.npz
  WSDB: gaia_dr3.astrophysical_parameters, gaia_dr3.galaxy_candidates

Writes:
  output/umap/q1_individual/q1_dsc_agreement.png

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_umap_q1_dsc_agreement.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "q1_individual"

MAG_CUT = 17


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_plot_cmap(name: str, *, bad_alpha_zero: bool = False):
    """Return a matplotlib or colorcet colormap."""
    try:
        cmap = plt.get_cmap(name.replace("-", "_")).copy()
    except ValueError:
        import colorcet as cc
        cmap = cc.cm[name.replace("-", "_")].copy()
    if bad_alpha_zero:
        cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    return cmap


def slice_cmap(cmap, lo=0.0, hi=1.0, n=256):
    if lo <= 0.0 and hi >= 1.0:
        return cmap.copy()
    return LinearSegmentedColormap.from_list(
        f"slice_{cmap.name}", cmap(np.linspace(lo, hi, n)))


def query_dsc_for_sources(source_ids, username):
    """Query DSC galaxy probability and galaxy_candidates membership."""
    import psycopg2
    conn = psycopg2.connect(host='wsdb.ast.cam.ac.uk', database='wsdb', user=username)
    cur = conn.cursor()

    # Create temp table with unique source_ids
    cur.execute("CREATE TEMP TABLE tmp_sids (source_id BIGINT PRIMARY KEY)")
    from io import StringIO
    buf = StringIO()
    seen = set()
    for sid in source_ids:
        if sid not in seen:
            buf.write(f"{sid}\n")
            seen.add(sid)
    buf.seek(0)
    cur.copy_from(buf, 'tmp_sids', columns=('source_id',))
    conn.commit()
    print(f"  Uploaded {len(source_ids):,} source_ids to temp table")

    # Query DSC probabilities
    cur.execute("""
        SELECT t.source_id,
               ap.classprob_dsc_combmod_galaxy,
               ap.classprob_dsc_combmod_star,
               ap.classprob_dsc_combmod_quasar,
               CASE WHEN gc.source_id IS NOT NULL THEN 1 ELSE 0 END AS in_galaxy_candidates
        FROM tmp_sids t
        LEFT JOIN gaia_dr3.astrophysical_parameters ap ON t.source_id = ap.source_id
        LEFT JOIN gaia_dr3.galaxy_candidates gc ON t.source_id = gc.source_id
    """)
    rows = cur.fetchall()
    conn.close()

    # Build lookup dict
    result = {}
    for sid, dsc_gal, dsc_star, dsc_qso, in_gc in rows:
        result[str(sid)] = {
            'dsc_pgal': float(dsc_gal) if dsc_gal is not None else np.nan,
            'dsc_pstar': float(dsc_star) if dsc_star is not None else np.nan,
            'dsc_pqso': float(dsc_qso) if dsc_qso is not None else np.nan,
            'in_galcand': bool(in_gc),
        }
    return result


def load_xgb_proba(X_raw):
    """Load Q1-only XGBoost ensemble and predict P(galaxy)."""
    import xgboost as xgb
    seed_dir = PROJECT_DIR / "output" / "xgboost_q1" / "seed_models"
    seed_paths = sorted(seed_dir.glob("xgb_classifier_q1_seed*.json"))
    if not seed_paths:
        raise FileNotFoundError(f"No seed models in {seed_dir}")
    all_probs = []
    for p in seed_paths:
        m = xgb.XGBClassifier()
        m.load_model(str(p))
        all_probs.append(m.predict_proba(X_raw.astype(np.float32))[:, 1])
    return np.mean(all_probs, axis=0)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER not set")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load UMAP data ───────────────────────────────────────────────────
    print("[1] Loading Q1 UMAP data...")
    npz = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    embedding  = npz["embedding"]
    y          = npz["y"]           # 0=STAR, 1=GALAXY (PHZ)
    source_ids = npz["source_ids"].astype(str)
    X_raw      = npz["X_raw"]
    print(f"  {len(y):,} sources  (STAR={( y==0).sum():,}  GALAXY={(y==1).sum():,})")

    # ── 2. Query DSC data ────────────────────────────────────────────────────
    print("\n[2] Querying DSC probabilities from WSDB...")
    dsc_data = query_dsc_for_sources(source_ids, username)
    print(f"  Got DSC data for {len(dsc_data):,} sources")

    # Build arrays aligned to our sources
    dsc_pgal    = np.array([dsc_data.get(s, {}).get('dsc_pgal', np.nan) for s in source_ids])
    dsc_pstar   = np.array([dsc_data.get(s, {}).get('dsc_pstar', np.nan) for s in source_ids])
    in_galcand  = np.array([dsc_data.get(s, {}).get('in_galcand', False) for s in source_ids])

    n_with_dsc = np.isfinite(dsc_pgal).sum()
    print(f"  DSC P(galaxy) available: {n_with_dsc:,} / {len(source_ids):,}")
    print(f"  In galaxy_candidates: {in_galcand.sum():,}")

    # ── 3. XGBoost predictions ───────────────────────────────────────────────
    print("\n[3] Running XGBoost ensemble...")
    xgb_pgal = load_xgb_proba(X_raw)

    # ── 4. Define masks ──────────────────────────────────────────────────────
    phz_gal = y == 1
    phz_star = y == 0

    dsc_agree   = phz_gal & (dsc_pgal >= 0.5)   # PHZ=galaxy, DSC=galaxy
    dsc_disagree = phz_gal & (dsc_pgal < 0.5) & np.isfinite(dsc_pgal)  # PHZ=galaxy, DSC=not galaxy

    n_agree   = dsc_agree.sum()
    n_disagree = dsc_disagree.sum()
    print(f"\n  PHZ galaxies: {phz_gal.sum():,}")
    print(f"    DSC agrees  (P_gal >= 0.5): {n_agree:,}  ({100*n_agree/phz_gal.sum():.1f}%)")
    print(f"    DSC disagrees (P_gal < 0.5): {n_disagree:,}  ({100*n_disagree/phz_gal.sum():.1f}%)")
    print(f"    In galaxy_candidates: {(phz_gal & in_galcand).sum():,}")

    # ── 5. Plot ──────────────────────────────────────────────────────────────
    print("\n[4] Plotting...")

    rng = np.random.RandomState(0)
    x_lo, x_hi = embedding[:, 0].min() - 1, embedding[:, 0].max() + 1
    y_lo, y_hi = embedding[:, 1].min() - 1, embedding[:, 1].max() + 1

    n_bins  = 200
    x_edges = np.linspace(x_lo, x_hi, n_bins + 1)
    y_edges = np.linspace(y_lo, y_hi, n_bins + 1)

    cmap_dens = slice_cmap(get_plot_cmap("CET_L19", bad_alpha_zero=True), 0.3, 1.0)
    cmap_pgal = get_plot_cmap("RdYlBu_r")

    MAX_PTS = 15_000
    def _ds(mask):
        idx = np.where(mask)[0]
        return rng.choice(idx, min(len(idx), MAX_PTS), replace=False) if len(idx) > MAX_PTS else idx

    # Layout: 2 rows (scatter + density) × 5 cols
    scatter_h = 1.0
    density_h = 1.0
    fig = plt.figure(figsize=(38, 8))
    gs = GridSpec(2, 5, figure=fig, height_ratios=[scatter_h, density_h],
                  hspace=0.08, wspace=0.25)

    def _style(ax, title):
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

    # Background indices
    bg_idx = _ds(np.ones(len(embedding), dtype=bool))

    # ── Col 0: DSC agrees (PHZ=galaxy, DSC=galaxy) ──────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(embedding[bg_idx, 0], embedding[bg_idx, 1],
               c="lightgrey", s=2, alpha=0.12, linewidths=0, zorder=0)
    idx = _ds(dsc_agree)
    ax.scatter(embedding[idx, 0], embedding[idx, 1],
               c="#1976D2", s=6, alpha=0.5, linewidths=0, zorder=3,
               label=f"DSC agrees (n={n_agree:,})")
    ax.legend(fontsize=7, loc="upper right", markerscale=2)
    _style(ax, f"PHZ galaxy + DSC agrees\n(DSC P_gal \u2265 0.5, n={n_agree:,})")

    ax_d = fig.add_subplot(gs[1, 0])
    H, _, _ = np.histogram2d(embedding[dsc_agree, 0], embedding[dsc_agree, 1],
                              bins=[x_edges, y_edges])
    H = np.where(H > 0, H, np.nan)
    ax_d.pcolormesh(x_edges, y_edges, H.T, cmap=cmap_dens,
                    norm=LogNorm(vmin=1, vmax=max(np.nanmax(H), 2)), rasterized=True)
    ax_d.set_xlim(x_lo, x_hi); ax_d.set_ylim(y_lo, y_hi)
    ax_d.set_xticks([]); ax_d.set_yticks([]); ax_d.grid(alpha=0.2)

    # ── Col 1: DSC disagrees (PHZ=galaxy, DSC=star) ─────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(embedding[bg_idx, 0], embedding[bg_idx, 1],
               c="lightgrey", s=2, alpha=0.12, linewidths=0, zorder=0)
    idx = _ds(dsc_disagree)
    ax.scatter(embedding[idx, 0], embedding[idx, 1],
               c="#E53935", s=6, alpha=0.15, linewidths=0, zorder=3,
               label=f"DSC disagrees (n={n_disagree:,})")
    ax.legend(fontsize=7, loc="upper right", markerscale=2)
    _style(ax, f"PHZ galaxy + DSC disagrees\n(DSC P_gal < 0.5, n={n_disagree:,})")

    ax_d = fig.add_subplot(gs[1, 1])
    H2, _, _ = np.histogram2d(embedding[dsc_disagree, 0], embedding[dsc_disagree, 1],
                               bins=[x_edges, y_edges])
    H2 = np.where(H2 > 0, H2, np.nan)
    ax_d.pcolormesh(x_edges, y_edges, H2.T, cmap=cmap_dens,
                    norm=LogNorm(vmin=1, vmax=max(np.nanmax(H2), 2)), rasterized=True)
    ax_d.set_xlim(x_lo, x_hi); ax_d.set_ylim(y_lo, y_hi)
    ax_d.set_xticks([]); ax_d.set_yticks([]); ax_d.grid(alpha=0.2)

    # ── Col 2: DSC P(galaxy) continuous ──────────────────────────────────────
    ax = fig.add_subplot(gs[0:2, 2])
    valid = np.isfinite(dsc_pgal)
    idx = _ds(valid)
    sc = ax.scatter(embedding[idx, 0], embedding[idx, 1],
                    c=dsc_pgal[idx], cmap=cmap_pgal, vmin=0, vmax=1,
                    s=3, alpha=0.6, linewidths=0, rasterized=True)
    fig.colorbar(sc, ax=ax, label="DSC P(galaxy)", pad=0.02, fraction=0.046)
    _style(ax, f"DSC P(galaxy) — all Q1\n(n={valid.sum():,})")

    # ── Col 3: XGBoost P(galaxy) continuous ──────────────────────────────────
    ax = fig.add_subplot(gs[0:2, 3])
    idx = _ds(np.ones(len(embedding), dtype=bool))
    sc = ax.scatter(embedding[idx, 0], embedding[idx, 1],
                    c=xgb_pgal[idx], cmap=cmap_pgal, vmin=0, vmax=1,
                    s=3, alpha=0.6, linewidths=0, rasterized=True)
    fig.colorbar(sc, ax=ax, label="XGB P(galaxy)", pad=0.02, fraction=0.046)
    _style(ax, f"XGBoost P(galaxy) — all Q1\n(n={len(embedding):,})")

    # ── Col 4: galaxy_candidates membership ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 4])
    ax.scatter(embedding[bg_idx, 0], embedding[bg_idx, 1],
               c="lightgrey", s=2, alpha=0.12, linewidths=0, zorder=0)
    # Split galaxy_candidates by PHZ label
    gc_gal  = in_galcand & phz_gal
    gc_star = in_galcand & phz_star
    idx_gg = _ds(gc_gal)
    idx_gs = _ds(gc_star) if gc_star.sum() > 0 else np.array([], dtype=int)
    ax.scatter(embedding[idx_gg, 0], embedding[idx_gg, 1],
               c="#1976D2", s=6, alpha=0.5, linewidths=0, zorder=3,
               label=f"PHZ gal in galcand (n={gc_gal.sum():,})")
    if len(idx_gs) > 0:
        ax.scatter(embedding[idx_gs, 0], embedding[idx_gs, 1],
                   c="orange", s=10, alpha=0.8, linewidths=0.3, edgecolors="k",
                   zorder=4, marker="x",
                   label=f"PHZ star in galcand (n={gc_star.sum():,})")
    ax.legend(fontsize=6, loc="upper right", markerscale=2)
    _style(ax, f"In galaxy_candidates table\n(n={in_galcand.sum():,})")

    ax_d = fig.add_subplot(gs[1, 4])
    H3, _, _ = np.histogram2d(embedding[in_galcand, 0], embedding[in_galcand, 1],
                               bins=[x_edges, y_edges])
    H3 = np.where(H3 > 0, H3, np.nan)
    ax_d.pcolormesh(x_edges, y_edges, H3.T, cmap=cmap_dens,
                    norm=LogNorm(vmin=1, vmax=max(np.nanmax(H3), 2)), rasterized=True)
    ax_d.set_xlim(x_lo, x_hi); ax_d.set_ylim(y_lo, y_hi)
    ax_d.set_xticks([]); ax_d.set_yticks([]); ax_d.grid(alpha=0.2)

    fig.suptitle(
        "Q1 UMAP — PHZ vs DSC galaxy classification agreement\n"
        f"PHZ galaxies: {phz_gal.sum():,}  |  "
        f"DSC agrees: {n_agree:,} ({100*n_agree/phz_gal.sum():.1f}%)  |  "
        f"DSC disagrees: {n_disagree:,} ({100*n_disagree/phz_gal.sum():.1f}%)  |  "
        f"In galaxy_candidates: {in_galcand.sum():,}",
        fontsize=12, fontweight="bold", y=0.99,
    )

    out = OUTPUT_DIR / "q1_dsc_agreement.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
