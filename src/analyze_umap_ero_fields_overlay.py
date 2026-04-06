#!/usr/bin/env python3
"""
Show where each of the 17 ERO fields sits in the ERO individual UMAP space.

Fields (from analyze_ero_mag_dist_fields.py):
  Star-forming (red):      Horsehead, Messier 78, Taurus, Barnard 30
  Globular Clusters (green): NGC 6254, NGC 6397
  Nearby Galaxies (blue):  NGC 6822, Holmberg II, IC 10, IC 342, NGC 2403, NGC 6744
  Galaxy Groups (purple):  Fornax, Dorado
  Galaxy Clusters (orange): Perseus, Abell 2390, Abell 2764

Layout: 3 row-groups × 6 cols  (each field = scatter + density sub-panel)
  (row-group 0, col 0)     : summary — all sources coloured by category
  (row-group 0, cols 1–5)  : fields 0–4
  (row-group 1, cols 0–5)  : fields 5–10
  (row-group 2, cols 0–5)  : fields 11–16

Reads:
  output/umap/arrays/ero_indiv.npz
  output/umap/arrays/ero_indiv_rebalanced.npz
  DB: siwei_zhao_2025.crossmatch_ero_gaia_0_3  (RA/Dec lookup)

Writes:
  output/umap/ero_individual/ero_space_fields.png
  output/umap/ero_individual/ero_space_fields_rebalanced.png

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_umap_ero_fields_overlay.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.gridspec import GridSpec

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "ero_individual"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
MAG_CUT   = 17

# ── Field definitions (from analyze_ero_mag_dist_fields.py) ──────────────────
ERO_FIELDS = {
    "Horsehead":   ( 85.150,  -2.613, 0.58),
    "Messier 78":  ( 86.690,  -0.015, 0.60),
    "Taurus":      ( 64.983,  28.023, 0.61),
    "Barnard 30":  ( 82.880,  12.316, 0.60),
    "NGC 6254":    (254.303,  -4.100, 0.60),
    "NGC 6397":    (265.174, -53.658, 0.61),
    "NGC 6822":    (296.236, -14.788, 0.60),
    "Holmberg II": (124.790,  70.706, 0.60),
    "IC 10":       (  5.063,  59.288, 0.62),
    "IC 342":      ( 56.730,  68.084, 0.59),
    "NGC 2403":    (114.211,  65.586, 0.60),
    "NGC 6744":    (287.430, -63.842, 0.60),
    "Fornax":      ( 54.017, -35.267, 0.57),
    "Dorado":      ( 64.014, -55.780, 0.60),
    "Perseus":     ( 49.638,  41.651, 0.70),
    "Abell 2390":  (328.397,  17.709, 0.75),
    "Abell 2764":  (  5.713, -49.249, 0.75),
}

FIELD_CATEGORIES = {
    "Star-forming":      ["Horsehead", "Messier 78", "Taurus", "Barnard 30"],
    "Globular Clusters": ["NGC 6254", "NGC 6397"],
    "Nearby Galaxies":   ["NGC 6822", "Holmberg II", "IC 10", "IC 342", "NGC 2403", "NGC 6744"],
    "Galaxy Groups":     ["Fornax", "Dorado"],
    "Galaxy Clusters":   ["Perseus", "Abell 2390", "Abell 2764"],
}

CATEGORY_COLORS = {
    "Star-forming":      "#E53935",
    "Globular Clusters": "#43A047",
    "Nearby Galaxies":   "#1976D2",
    "Galaxy Groups":     "#8E24AA",
    "Galaxy Clusters":   "#FB8C00",
}

FIELD_LIST = list(ERO_FIELDS.keys())   # 17 fields in order

CMAP_DENSITY = "CET_L19"


def _field_color(field: str) -> str:
    for cat, fields in FIELD_CATEGORIES.items():
        if field in fields:
            return CATEGORY_COLORS[cat]
    return "grey"


def assign_fields(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    assignments = np.full(len(ra), None, dtype=object)
    min_dist    = np.full(len(ra), np.inf)
    for name, (f_ra, f_dec, _) in ERO_FIELDS.items():
        ra1  = np.radians(ra);    dec1 = np.radians(dec)
        ra2  = np.radians(f_ra);  dec2 = np.radians(f_dec)
        a    = (np.sin((dec2 - dec1) / 2)**2
                + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2)**2)
        dist = np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))
        mask = dist < min_dist
        assignments[mask] = name
        min_dist[mask]    = dist[mask]
    return assignments


def query_ra_dec(username: str):
    import psycopg2, pandas as pd
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    df = pd.read_sql_query(f"""
        SELECT g_source_id,
               e_v_alpha_j2000 AS ra,
               e_v_delta_j2000 AS dec
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_class_star IS NOT NULL
          AND g_phot_g_mean_mag >= {MAG_CUT}
        ORDER BY g_source_id
    """, conn)
    conn.close()
    df["g_source_id"] = df["g_source_id"].astype(str)
    return df


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


def _run_pipeline(emb, y, fields_arr, label: str, out_name: str):
    rng = np.random.RandomState(0)

    x_lo = emb[:, 0].min() - 1;  x_hi = emb[:, 0].max() + 1
    y_lo = emb[:, 1].min() - 1;  y_hi = emb[:, 1].max() + 1

    n_bins  = 180
    x_edges = np.linspace(x_lo, x_hi, n_bins + 1)
    y_edges = np.linspace(y_lo, y_hi, n_bins + 1)

    cmap_dens = slice_cmap(get_plot_cmap(CMAP_DENSITY, bad_alpha_zero=True), 0.3, 1.0)

    FS_TITLE = 8;  FS_LABEL = 7;  FS_TICK = 6
    MAX_PTS  = 10_000

    def _downsample(mask, n=MAX_PTS):
        idx = np.where(mask)[0]
        if len(idx) > n:
            idx = rng.choice(idx, n, replace=False)
        return idx

    def _hist(mask):
        if mask.sum() == 0:
            return np.full((n_bins, n_bins), np.nan)
        pts = emb[mask]
        H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[x_edges, y_edges])
        return np.where(H > 0, H, np.nan)

    def _style_sc(ax, title):
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("UMAP 1", fontsize=FS_LABEL)
        ax.set_ylabel("UMAP 2", fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=5, loc="upper right", markerscale=1.5)

    def _style_dn(ax):
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(alpha=0.2)

    # ── Per-field histograms + shared norm ────────────────────────────────────
    histograms = {f: _hist(fields_arr == f) for f in FIELD_LIST}
    finite_h   = [h for h in histograms.values() if not np.all(np.isnan(h))]
    shared_norm = LogNorm(
        vmin=np.nanmin([np.nanmin(h) for h in finite_h]),
        vmax=np.nanmax([np.nanmax(h) for h in finite_h]),
    )

    # ── GridSpec: 3 row-groups × 6 cols ──────────────────────────────────────
    # sub-rows: [sc0, dn0, gap, sc1, dn1, gap, sc2, dn2]
    s = 0.5;  d = 0.5;  g = 0.02
    height_ratios = [s, d, g, s, d, g, s, d]
    panel_frac = s / sum(height_ratios)
    fig = plt.figure(figsize=(45.0, 3.0 / panel_frac))
    gs  = GridSpec(8, 6, figure=fig, height_ratios=height_ratios,
                   hspace=0.23, wspace=0.25, top=0.92)

    # sub-row indices per row-group
    SC = [0, 3, 6]   # scatter sub-rows
    DN = [1, 4, 7]   # density sub-rows

    # Grid positions: (row_group, col) → field index
    # (0,0) = summary; (0,1..5) = fields 0-4; (1,0..5) = fields 5-10; (2,0..5) = fields 11-16
    field_positions = []   # list of (rg, col) for fields in FIELD_LIST order
    for col in range(1, 6):
        field_positions.append((0, col))
    for rg in [1, 2]:
        for col in range(6):
            field_positions.append((rg, col))
    # field_positions now has 5 + 6 + 6 = 17 entries

    # ── Summary panel at (rg=0, col=0) spanning scatter+density — density plot
    ax_sum = fig.add_subplot(gs[0:2, 0])
    H_all, _, _ = np.histogram2d(emb[:, 0], emb[:, 1], bins=[x_edges, y_edges])
    H_all = np.where(H_all > 0, H_all, np.nan)
    all_norm = LogNorm(vmin=np.nanmin(H_all), vmax=np.nanmax(H_all))
    sc_sum = ax_sum.pcolormesh(x_edges, y_edges, H_all.T,
                               cmap=cmap_dens, norm=all_norm, rasterized=True)
    fig.colorbar(sc_sum, ax=ax_sum, label="Count (log)", pad=0.02, fraction=0.046)
    ax_sum.set_xlim(x_lo, x_hi); ax_sum.set_ylim(y_lo, y_hi)
    ax_sum.set_xlabel("UMAP 1", fontsize=FS_LABEL)
    ax_sum.set_ylabel("UMAP 2", fontsize=FS_LABEL)
    ax_sum.set_title(f"ERO {label}: source density  (n={len(y):,})",
                     fontsize=FS_TITLE, fontweight="bold")
    ax_sum.tick_params(labelsize=FS_TICK)
    ax_sum.grid(alpha=0.2)

    # ── 17 field panels ───────────────────────────────────────────────────────
    for fi, field in enumerate(FIELD_LIST):
        rg, col = field_positions[fi]
        color   = _field_color(field)
        mask    = fields_arr == field
        n       = mask.sum()

        # scatter
        ax_sc = fig.add_subplot(gs[SC[rg], col])
        idx_bg = _downsample(~mask)
        ax_sc.scatter(emb[idx_bg, 0], emb[idx_bg, 1],
                      c="lightgrey", s=2, alpha=0.3, linewidths=0,
                      label=f"other (n={len(y)-n:,})", zorder=1)
        idx = _downsample(mask)
        ax_sc.scatter(emb[idx, 0], emb[idx, 1],
                      c=color, s=6, alpha=0.6, linewidths=0,
                      label=f"{field} (n={n:,})", zorder=3)
        _style_sc(ax_sc, f"{field}  (n={n:,})")

        # density
        ax_dn = fig.add_subplot(gs[DN[rg], col])
        H     = histograms[field]
        sc2   = ax_dn.pcolormesh(x_edges, y_edges, H.T,
                                 cmap=cmap_dens, norm=shared_norm, rasterized=True)
        _style_dn(ax_dn)
        fig.colorbar(sc2, ax=ax_dn, label="Count (log)", pad=0.02, fraction=0.046)

    fig.suptitle(
        f"ERO {label} sources by field in ERO individual UMAP space\n"
        "(nearest-field assignment by haversine distance)",
        fontsize=12, fontweight="bold", y=0.97,
    )

    out = OUTPUT_DIR / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {out}")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set.")
        return 1

    # ── Query RA/Dec and build field lookup ───────────────────────────────────
    print("Querying RA/Dec from DB...")
    df_pos = query_ra_dec(username)
    print(f"  Retrieved {len(df_pos):,} sources")

    fields_db = assign_fields(df_pos["ra"].values, df_pos["dec"].values)
    field_lookup = dict(zip(df_pos["g_source_id"], fields_db))

    for f in FIELD_LIST:
        n = (fields_db == f).sum()
        print(f"  {f}: {n:,}")

    def _make_field_arr(source_ids):
        return np.array([field_lookup.get(sid, None) for sid in source_ids], dtype=object)

    # ── Regular ERO ───────────────────────────────────────────────────────────
    indiv_path = ARRAYS_DIR / "ero_indiv.npz"
    if not indiv_path.exists():
        print(f"ERROR: {indiv_path} not found. Run analyze_umap.py first.")
        return 1

    print("\nLoading ERO individual UMAP arrays...")
    d  = np.load(indiv_path, allow_pickle=True)
    emb = d["embedding"];  y = d["y"];  sids = d["source_ids"].astype(str)
    print(f"  {len(y):,} sources")
    _run_pipeline(emb, y, _make_field_arr(sids), "individual", "ero_space_fields.png")

    # ── Rebalanced ERO ────────────────────────────────────────────────────────
    rebal_path = ARRAYS_DIR / "ero_indiv_rebalanced.npz"
    if not rebal_path.exists():
        print(f"WARNING: {rebal_path} not found — skipping rebalanced plot.")
        return 0

    print("\nLoading rebalanced ERO arrays...")
    d2  = np.load(rebal_path, allow_pickle=True)
    emb2 = d2["embedding"];  y2 = d2["y"];  sids2 = d2["source_ids"].astype(str)
    print(f"  {len(y2):,} sources")
    _run_pipeline(emb2, y2, _make_field_arr(sids2), "rebalanced", "ero_space_fields_rebalanced.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
