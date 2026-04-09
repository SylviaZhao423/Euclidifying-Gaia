#!/usr/bin/env python3
"""
Show where each Q1 field sits in the Q1 individual UMAP space.

Fields (from analyze_q1_mag_dist_fields.py):
  EDF-North  (RA=269.733, Dec=+66.018)
  EDF-South  (RA= 61.241, Dec=-48.423)
  EDF-Fornax (RA= 52.932, Dec=-28.088)

Each source in the existing q1_indiv.npz embedding is assigned to its
nearest field via haversine distance, then plotted.

Layout: 2 rows × 4 cols
  Col 0 (spans both rows): all Q1 sources coloured by field
  Cols 1–3:
    Row 0 (scatter): EDF-North | EDF-South | EDF-Fornax
    Row 1 (density): EDF-North | EDF-South | EDF-Fornax

Reads:
  output/umap/arrays/q1_indiv.npz
  DB: siwei_zhao_2025.crossmatch_q1_mer_phz_gaia  (RA/Dec lookup)

Writes:
  output/umap/q1_individual/q1_space_fields.png

Usage:
  python src/analyze_umap_q1_fields_overlay.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.gridspec import GridSpec

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "q1_individual"

MAG_CUT   = 17

# ── Field definitions (matching analyze_q1_mag_dist_fields.py) ────────────────
Q1_FIELDS = {
    "EDF-North":  (269.733, +66.018, 22.9),
    "EDF-South":  ( 61.241, -48.423, 28.1),
    "EDF-Fornax": ( 52.932, -28.088, 12.1),
}

FIELD_COLORS = {
    "EDF-North":  "#1976D2",   # blue
    "EDF-South":  "#E53935",   # red
    "EDF-Fornax": "#43A047",   # green
}

CMAP_DENSITY = "CET_L19"


def assign_fields(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Return field name array (nearest field per source, haversine)."""
    assignments = np.full(len(ra), None, dtype=object)
    min_dist    = np.full(len(ra), np.inf)
    for name, (f_ra, f_dec, _) in Q1_FIELDS.items():
        ra1  = np.radians(ra);     dec1 = np.radians(dec)
        ra2  = np.radians(f_ra);   dec2 = np.radians(f_dec)
        a    = (np.sin((dec2 - dec1) / 2)**2
                + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2)**2)
        dist = np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))
        mask = dist < min_dist
        assignments[mask] = name
        min_dist[mask]    = dist[mask]
    return assignments


def load_ra_dec_from_npz(npz_path):
    """Load RA/Dec arrays from a Q1 npz file (no DB needed)."""
    data = np.load(npz_path, allow_pickle=True)
    return data["ra"].astype(float), data["dec"].astype(float)


def get_plot_cmap(name: str, *, bad_alpha_zero: bool = False):
    normalized = name.replace("-", "_")
    try:
        cmap = plt.get_cmap(normalized).copy()
        if bad_alpha_zero:
            cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
        return cmap
    except ValueError:
        pass
    try:
        import colorcet as cc
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Install colorcet: pip install colorcet") from e
    key = normalized
    if hasattr(cc, f"m_{key}"):
        cmap = getattr(cc, f"m_{key}").copy()
    elif key in cc.cm:
        cmap = cc.cm[key].copy()
    else:
        tail = key.replace("CET_", "", 1).lower() if key.startswith("CET_") else key.lower()
        cmap = cc.cm[tail].copy()
    if bad_alpha_zero:
        cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    return cmap


def slice_cmap(cmap, minval=0.0, maxval=1.0, n=256):
    if minval <= 0.0 and maxval >= 1.0:
        return cmap.copy()
    return LinearSegmentedColormap.from_list(
        f"slice_{cmap.name}_{minval:.2f}_{maxval:.2f}",
        cmap(np.linspace(minval, maxval, n)),
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Q1 UMAP embedding ────────────────────────────────────────────────
    print("Loading Q1 UMAP arrays...")
    q1         = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    q1_emb     = q1["embedding"]
    q1_y       = q1["y"]
    source_ids = q1["source_ids"].astype(str)
    ra         = q1["ra"].astype(float)
    dec        = q1["dec"].astype(float)
    print(f"  Q1: {len(q1_y):,} sources")

    # ── Assign fields from RA/Dec ─────────────────────────────────────────────
    print("\nAssigning fields from RA/Dec (from npz)...")
    fields_arr = assign_fields(ra, dec)

    for fname in Q1_FIELDS:
        n = (fields_arr == fname).sum()
        print(f"  {fname}: {n:,} sources in embedding")

    # ── Plot setup ────────────────────────────────────────────────────────────
    rng = np.random.RandomState(0)

    x_lo = q1_emb[:, 0].min() - 1;  x_hi = q1_emb[:, 0].max() + 1
    y_lo = q1_emb[:, 1].min() - 1;  y_hi = q1_emb[:, 1].max() + 1

    n_bins  = 180
    x_edges = np.linspace(x_lo, x_hi, n_bins + 1)
    y_edges = np.linspace(y_lo, y_hi, n_bins + 1)

    cmap_dens = slice_cmap(get_plot_cmap(CMAP_DENSITY, bad_alpha_zero=True), 0.3, 1.0)

    FS_TITLE = 9;  FS_LABEL = 8;  FS_TICK = 7
    MAX_PTS  = 15_000

    def _downsample(mask, n=MAX_PTS):
        idx = np.where(mask)[0]
        if len(idx) > n:
            idx = rng.choice(idx, n, replace=False)
        return idx

    def _hist(mask):
        pts = q1_emb[mask]
        H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[x_edges, y_edges])
        return np.where(H > 0, H, np.nan)

    def _style_sc(ax, title):
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("UMAP 1", fontsize=FS_LABEL)
        ax.set_ylabel("UMAP 2", fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6, loc="upper right", markerscale=1.5)

    def _style_sum(ax, title):
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("UMAP 1", fontsize=FS_LABEL)
        ax.set_ylabel("UMAP 2", fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.2)

    def _style_dn(ax):
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(alpha=0.2)

    # ── Histograms ────────────────────────────────────────────────────────────
    field_names = list(Q1_FIELDS.keys())
    histograms  = {f: _hist(fields_arr == f) for f in field_names}

    finite_h = [h for h in histograms.values() if not np.all(np.isnan(h))]
    shared_norm = LogNorm(
        vmin=np.nanmin([np.nanmin(h) for h in finite_h]),
        vmax=np.nanmax([np.nanmax(h) for h in finite_h]),
    )

    # ── Figure: 2 rows × 4 cols ───────────────────────────────────────────────
    scatter_h = 0.5;  density_h = 0.5
    height_ratios = [scatter_h, density_h]
    panel_frac = scatter_h / sum(height_ratios)
    fig = plt.figure(figsize=(30.0, 3.0 / panel_frac))
    gs  = GridSpec(2, 4, figure=fig, height_ratios=height_ratios,
                   hspace=0.22, wspace=0.25, top=0.88)

    ax_sum = fig.add_subplot(gs[0:2, 0])   # col 0 spans both rows
    sc_axes = [fig.add_subplot(gs[0, c]) for c in range(1, 4)]
    dn_axes = [fig.add_subplot(gs[1, c]) for c in range(1, 4)]

    # ── Col 0: source density ─────────────────────────────────────────────────
    H_all, _, _ = np.histogram2d(q1_emb[:, 0], q1_emb[:, 1], bins=[x_edges, y_edges])
    H_all = np.where(H_all > 0, H_all, np.nan)
    all_norm = LogNorm(vmin=np.nanmin(H_all), vmax=np.nanmax(H_all))
    sc_sum = ax_sum.pcolormesh(x_edges, y_edges, H_all.T,
                               cmap=cmap_dens, norm=all_norm, rasterized=True)
    fig.colorbar(sc_sum, ax=ax_sum, label="Count (log)", pad=0.02, fraction=0.046)
    _style_sum(ax_sum, f"Q1: source density  (n={len(q1_y):,})")

    # ── Cols 1–3: per-field scatter + density ─────────────────────────────────
    for col, fname in enumerate(field_names):
        color = FIELD_COLORS[fname]
        mask  = fields_arr == fname
        n     = mask.sum()

        # scatter
        ax = sc_axes[col]
        # grey background (other fields)
        idx_bg = _downsample(fields_arr != fname)
        ax.scatter(q1_emb[idx_bg, 0], q1_emb[idx_bg, 1],
                   c="lightgrey", s=2, alpha=0.3, linewidths=0,
                   label=f"Q1 other (n={len(q1_y)-n:,})", zorder=1)
        idx = _downsample(mask)
        ax.scatter(q1_emb[idx, 0], q1_emb[idx, 1],
                   c=color, s=6, alpha=0.6, linewidths=0,
                   label=f"{fname} (n={n:,})", zorder=3)
        _style_sc(ax, f"{fname}  (n={n:,})")

        # density
        ax_d = dn_axes[col]
        H    = histograms[fname]
        sc2  = ax_d.pcolormesh(x_edges, y_edges, H.T,
                               cmap=cmap_dens, norm=shared_norm, rasterized=True)
        _style_dn(ax_d)
        fig.colorbar(sc2, ax=ax_d, label="Count (log)", pad=0.02, fraction=0.046)

    fig.suptitle(
        "Q1 sources by field in Q1 individual UMAP space\n"
        "(nearest-field assignment by haversine distance)",
        fontsize=12, fontweight="bold", y=0.995,
    )

    out = OUTPUT_DIR / "q1_fields.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
