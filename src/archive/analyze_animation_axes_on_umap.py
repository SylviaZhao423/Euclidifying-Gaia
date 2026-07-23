#!/usr/bin/env python3
"""
UMAP coloured by the three animation sort axes:
  AEN_sig     — gaia_astrometric_excess_noise_sig   (X_raw[:, 3])
  BPRPEF      — corrected_bp_rp_excess              (X_raw[:, 11])
  variability — sqrt(phot_g_n_obs) / phot_g_mean_flux_over_error (WSDB query)

Reads:
  output/umap/arrays/q1_indiv.npz

Writes:
  output/animation_axis_analysis/umap_animation_axes.png

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_animation_axes_on_umap.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "output" / "animation_axis_analysis"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

AXES = {
    "AEN sig":     {"npz_idx": 3,  "col": "gaia_astrometric_excess_noise_sig"},
    "BPRP excess": {"npz_idx": 11, "col": "gaia_bp_rp_excess_corrected"},
    "Variability": {"npz_idx": None, "col": "gaia_g_flux_frac_error_sqrt_nobs"},
}


def fetch_variability(source_ids: np.ndarray) -> np.ndarray | None:
    """Query WSDB for phot_g_n_obs and compute variability for given source_ids."""
    try:
        import psycopg2
    except ImportError:
        print("  [warn] psycopg2 not installed — variability panel will be skipped")
        return None

    pguser = os.environ.get("PGUSER", "")
    if not pguser:
        print("  [warn] PGUSER not set — variability panel will be skipped")
        return None

    print(f"  Connecting to WSDB as {pguser} to fetch phot_g_n_obs for {len(source_ids):,} sources...")
    try:
        conn = psycopg2.connect(
            host=WSDB_HOST, dbname=WSDB_DB, user=pguser,
            connect_timeout=20,
        )
        conn.autocommit = True
        cur = conn.cursor()

        # Build temp table from source_id list
        cur.execute("CREATE TEMP TABLE _umap_sids (source_id BIGINT);")
        # Insert in chunks to avoid huge literals
        chunk = 5000
        for i in range(0, len(source_ids), chunk):
            vals = ",".join(f"({int(sid)})" for sid in source_ids[i:i+chunk])
            cur.execute(f"INSERT INTO _umap_sids VALUES {vals};")

        cur.execute("""
            SELECT s.source_id,
                   SQRT(s.phot_g_n_obs::real) / s.phot_g_mean_flux_over_error AS variability
            FROM   gaia_dr3.gaia_source s
            JOIN   _umap_sids t USING (source_id)
            WHERE  s.phot_g_mean_flux_over_error > 0;
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"  [warn] WSDB query failed: {e}")
        return None

    sid_to_var = {int(r[0]): float(r[1]) for r in rows}
    print(f"  Got variability for {len(sid_to_var):,}/{len(source_ids):,} sources")

    out = np.full(len(source_ids), np.nan, dtype=np.float64)
    for i, sid in enumerate(source_ids):
        v = sid_to_var.get(int(sid))
        if v is not None:
            out[i] = v
    return out


def make_umap_axes_plot(embedding, y, vals_dict, output_path):
    """
    3-column figure, each column = one animation axis.
    Each column: upper panel = Q1 Stars, lower panel = Q1 Galaxies.
    """
    axes_names = list(vals_dict.keys())
    n_axes = len(axes_names)

    m_star = y == 0
    m_gal  = y == 1

    # Gridspec: for each feature column, 2 data rows + 1 small gap between them
    # plus a top-label gap row at the very top
    per_panel_h = 1.0
    gap_inner   = 0.04
    gap_outer   = 0.22   # between feature columns (only relevant for colorbars)

    gs_rows = [per_panel_h, gap_inner, per_panel_h]  # stars, gap, galaxies (one feature row)
    fig_width  = n_axes * 4.0
    fig_height = (per_panel_h * 2 + gap_inner) * (fig_width / (n_axes * 4.0)) * 4.5

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = GridSpec(3, n_axes, figure=fig,
                   height_ratios=gs_rows, hspace=0.04, wspace=0.35)

    xlim = (embedding[:, 0].min() - 0.5, embedding[:, 0].max() + 0.5)
    ylim = (embedding[:, 1].min() - 0.5, embedding[:, 1].max() + 0.5)

    for ci, name in enumerate(axes_names):
        vals = vals_dict[name]
        ax_top = fig.add_subplot(gs[0, ci])
        ax_bot = fig.add_subplot(gs[2, ci])

        if vals is None or np.all(np.isnan(vals)):
            for ax, lbl in [(ax_top, "Stars"), (ax_bot, "Galaxies")]:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="grey")
                ax.set_xticks([]); ax.set_yticks([])
            ax_top.set_title(name, fontsize=10, fontweight="bold", pad=5)
            continue

        lo = np.nanpercentile(vals, 1)
        hi = np.nanpercentile(vals, 99)
        vals_c = np.clip(vals, lo, hi)

        for ax, mask, lbl in [
            (ax_top, m_star, f"Q1 Stars  (n={m_star.sum():,})"),
            (ax_bot, m_gal,  f"Q1 Galaxies  (n={m_gal.sum():,})"),
        ]:
            # grey background (all points)
            ax.scatter(embedding[:, 0], embedding[:, 1],
                       c="lightgrey", s=3, alpha=0.15, marker="o",
                       rasterized=True, linewidths=0, zorder=0)
            # colored foreground (this class)
            sc = ax.scatter(embedding[mask, 0], embedding[mask, 1],
                            c=vals_c[mask], cmap="RdYlBu_r",
                            vmin=lo, vmax=hi,
                            s=3, alpha=0.55, marker="o",
                            rasterized=True, linewidths=0, zorder=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_ylabel(lbl, fontsize=8)
            ax.grid(alpha=0.2)

        fig.colorbar(sc, ax=[ax_top, ax_bot], pad=0.02, fraction=0.046,
                     ).ax.tick_params(labelsize=7)
        ax_top.set_title(name, fontsize=10, fontweight="bold", pad=5)

    fig.suptitle(
        "Q1 UMAP coloured by animation sort axes  |  Upper = Stars, Lower = Galaxies",
        fontsize=12, fontweight="bold", y=1.01,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {output_path}")


def main() -> int:
    npz_path = ARRAYS_DIR / "q1_indiv.npz"
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found. Run query_umap_q1_50k.py first.")
        return 1

    print("Loading q1_indiv.npz...")
    data      = np.load(npz_path, allow_pickle=True)
    embedding  = data["embedding"]
    y          = data["y"]
    X_raw      = data["X_raw"]
    source_ids = data["source_ids"]
    print(f"  {len(y):,} sources  (stars={int((y==0).sum()):,}, galaxies={int((y==1).sum()):,})")

    vals_dict = {}
    for name, meta in AXES.items():
        if meta["npz_idx"] is not None:
            vals_dict[name] = X_raw[:, meta["npz_idx"]].astype(np.float64)
            print(f"  {name}: loaded from X_raw col {meta['npz_idx']}")
        else:
            vals_dict[name] = None  # filled below

    print("Fetching variability from WSDB...")
    vals_dict["Variability"] = fetch_variability(source_ids.astype(str if source_ids.dtype.kind in ('U','S') else int))

    out_path = OUTPUT_DIR / "umap_animation_axes.png"
    print("Plotting...")
    make_umap_axes_plot(embedding, y, vals_dict, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
