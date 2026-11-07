#!/usr/bin/env python3
"""
Project all PHZ classification categories into the Q1 individual UMAP space.

Layout (2 rows × 4 cols, each of the 3 class cols has scatter + density subpanel):
  Col 0 (summary, full-height per row):
    Row 0: 2D histogram of Q1 sources coloured by mean PHZ P(galaxy) per bin
    Row 1: 2D density histogram of all Q1 sources
  Cols 1–3:
    Row 0: phz=1 (Star) | phz=2 (Galaxy) | phz=3 (Multi Star∩Gal)
    Row 1: phz=-1 (Rejected) | phz=0 (Outlier) | phz=4+5+6+7 (QSO variants)

Reads:
  output/umap/arrays/q1_indiv.npz
  output/umap/arrays/q1_indiv_reducer.joblib

Writes:
  output/umap/q1_individual/allphz_in_q1_space.png

Usage:
  python src/analyze_umap_allphz_in_q1_space.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "q1_individual"

MAG_CUT   = 17
MAX_PER_CLASS = 5_000

# Individual class map (no merging of QSO codes)
PHZ_MAP = {
     1: ("Star",              "#1976D2"),   # blue
     2: ("Galaxy",            "#E53935"),   # red
     3: ("Multi (Star∩Gal)",  "#FFC107"),   # amber
    -1: ("Rejected",          "#9C27B0"),   # purple
     0: ("Outlier",           "#FF9800"),   # orange
}

# QSO variants shown together in one panel, each a distinct colour
QSO_MAP = {
    4: ("QSO",              "#4CAF50"),   # green
    5: ("Multi (Star∩QSO)", "#E53935"),   # red
    6: ("Multi (Gal∩QSO)",  "#00ACC1"),   # teal
    7: ("Multi (all three)", "#FF7043"),  # deep orange
}

# ── Colormap helpers ──────────────────────────────────────────────────────────
def _get_cmap(name: str, *, bad_alpha_zero: bool = False):
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


def _slice_cmap(cmap, minval: float = 0.0, maxval: float = 1.0, n: int = 256):
    if minval <= 0.0 and maxval >= 1.0:
        return cmap.copy()
    return LinearSegmentedColormap.from_list(
        f"slice_{cmap.name}_{minval:.2f}_{maxval:.2f}",
        cmap(np.linspace(minval, maxval, n)),
    )


def load_allphz_from_npz() -> pd.DataFrame:
    """Load all-PHZ data from q1_allphz.npz (no DB needed)."""
    data = np.load(ARRAYS_DIR / "q1_allphz.npz", allow_pickle=True)
    df = pd.DataFrame(data["X_features"], columns=SELECTED_GAIA_PARAMETERS)
    df["phz"] = data["phz"].astype(int)
    df["phz_gal_prob"] = data["phz_gal_prob"].astype(float)
    return df


def load_phz_gal_prob_from_npz(npz_data) -> np.ndarray:
    """Load p_phz_gal_prob from Q1 npz file (no DB needed)."""
    return npz_data["phz_gal_prob"].astype(float)


def _project(df_sub, q1_mean, q1_std, reducer, max_n=MAX_PER_CLASS):
    if len(df_sub) > max_n:
        df_sub = df_sub.sample(max_n, random_state=42)
    X = df_sub[SELECTED_GAIA_PARAMETERS].values.astype(float)
    std_safe = np.where(q1_std > 0, q1_std, 1.0)
    X_scaled = (X - q1_mean) / std_safe
    X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0.0)
    gal_prob = df_sub["phz_gal_prob"].values if "phz_gal_prob" in df_sub.columns else None
    return reducer.transform(X_scaled), len(df_sub), gal_prob


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Q1 normalisation + reducer ──────────────────────────────────────
    print("Loading Q1 reducer and normalisation stats...")
    q1 = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    mag_cut_faint = int(q1['mag_cut_faint']) if 'mag_cut_faint' in q1 else 99
    mag_label = f'G \u2265 {MAG_CUT}' if mag_cut_faint >= 90 else f'{MAG_CUT} \u2264 G \u2264 {mag_cut_faint}'
    q1_emb     = q1["embedding"]
    q1_y       = q1["y"]
    q1_mean    = q1["mean"]
    q1_std     = q1["std"]
    source_ids = q1["source_ids"].astype(str)
    reducer = joblib.load(ARRAYS_DIR / "q1_indiv_reducer.joblib")
    print(f"  Q1: {len(q1_y):,} sources")

    # ── Load PHZ P(galaxy) from npz ────────────────────────────────────────────
    print("\nLoading p_phz_gal_prob from npz...")
    phz_gal_prob = load_phz_gal_prob_from_npz(q1)
    n_miss = np.isnan(phz_gal_prob).sum()
    print(f"  Loaded {(~np.isnan(phz_gal_prob)).sum():,} values  ({n_miss:,} NaN)")

    # ── Load all PHZ classes from npz ────────────────────────────────────────
    print("\nLoading all PHZ classes from npz...")
    df = load_allphz_from_npz()
    print(f"  Total rows: {len(df):,}")
    all_vals = sorted(df["phz"].unique())
    for v in all_vals:
        print(f"  phz={v:>3}: {(df['phz']==v).sum():>8,}")

    # ── Project each class ────────────────────────────────────────────────────
    print("\nProjecting into Q1 UMAP space...")
    embeddings: dict[int, tuple[np.ndarray, int]] = {}
    for phz_val in list(PHZ_MAP.keys()) + list(QSO_MAP.keys()):
        sub = df[df["phz"] == phz_val]
        if len(sub) == 0:
            print(f"  phz={phz_val}: no sources, skipping")
            continue
        label = PHZ_MAP.get(phz_val, QSO_MAP.get(phz_val, (str(phz_val), None)))[0]
        emb, n_used, gal_prob = _project(sub, q1_mean, q1_std, reducer)
        embeddings[phz_val] = (emb, n_used, gal_prob)
        print(f"  phz={phz_val:>3} {label:22s}: {n_used:,} sources projected")

    # ── Plot setup ────────────────────────────────────────────────────────────
    print("\nPlotting...")
    x_lo = q1_emb[:, 0].min() - 1;  x_hi = q1_emb[:, 0].max() + 1
    y_lo = q1_emb[:, 1].min() - 1;  y_hi = q1_emb[:, 1].max() + 1
    Q1_BG = "#E0E0E0"

    n_bins  = 180
    x_edges = np.linspace(x_lo, x_hi, n_bins + 1)
    y_edges = np.linspace(y_lo, y_hi, n_bins + 1)
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])

    cmap_dens     = _slice_cmap(_get_cmap("CET_L19", bad_alpha_zero=True), 0.3, 1.0)
    cmap_dens_q1  = _slice_cmap(_get_cmap("CET_L19", bad_alpha_zero=True), 0.3, 1.0)
    cmap_pgal     = _get_cmap("RdYlBu_r", bad_alpha_zero=True)

    # ── Summary column histograms ─────────────────────────────────────────────
    # Row-1 summary: density of ALL projected PHZ sources (not the original Q1 50k)
    all_projected = [emb for emb, _, _ in embeddings.values()]
    if all_projected:
        all_proj_emb = np.vstack(all_projected)
    else:
        all_proj_emb = q1_emb  # fallback
    n_projected_total = len(all_proj_emb)
    H_q1, _, _ = np.histogram2d(all_proj_emb[:, 0], all_proj_emb[:, 1],
                                 bins=[x_edges, y_edges])
    H_q1 = np.where(H_q1 > 0, H_q1, np.nan)
    q1_dens_norm = LogNorm(vmin=np.nanmin(H_q1), vmax=np.nanmax(H_q1))

    # ── Per-class density histograms ──────────────────────────────────────────
    panel_keys = [1, 2, 3, -1, 0, "QSO"]
    histograms: dict = {}
    for key in panel_keys:
        if key == "QSO":
            parts = [embeddings[v][0] for v in QSO_MAP if v in embeddings]
            if parts:
                pts = np.vstack(parts)
                H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1],
                                         bins=[x_edges, y_edges])
                histograms["QSO"] = np.where(H > 0, H, np.nan)
            else:
                histograms["QSO"] = np.full((n_bins, n_bins), np.nan)
        elif key in embeddings:
            emb, _, _ = embeddings[key]
            H, _, _ = np.histogram2d(emb[:, 0], emb[:, 1],
                                     bins=[x_edges, y_edges])
            histograms[key] = np.where(H > 0, H, np.nan)
        else:
            histograms[key] = np.full((n_bins, n_bins), np.nan)

    all_h = [h for h in histograms.values() if not np.all(np.isnan(h))]
    shared_norm = LogNorm(
        vmin=np.nanmin([np.nanmin(h) for h in all_h]),
        vmax=np.nanmax([np.nanmax(h) for h in all_h]),
    )

    # ── GridSpec layout ───────────────────────────────────────────────────────
    # 5 rows: scatter_r0 | density_r0 | gap | scatter_r1 | density_r1
    # 4 cols: summary | phz=1,2,3 (row0) / phz=-1,0,QSO (row1)
    scatter_h = 0.5
    density_h = 0.5
    outer_gap = 0.02
    height_ratios = [scatter_h, density_h, outer_gap, scatter_h, density_h]
    panel_frac = scatter_h / sum(height_ratios)
    fig_height = 3.0 / panel_frac
    fig_width  = 30.0

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = GridSpec(5, 4, figure=fig, height_ratios=height_ratios,
                   hspace=0.22, wspace=0.25, top=0.93)

    # Summary column: full-height axes (span scatter+density rows)
    ax_sum0 = fig.add_subplot(gs[0:2, 0])   # row-0: phz_gal_prob 2D histogram
    ax_sum1 = fig.add_subplot(gs[3:5, 0])   # row-1: Q1 density

    # Class scatter axes (cols 1–3)
    sc_axes = [
        [fig.add_subplot(gs[0, c]) for c in range(1, 4)],  # row 0
        [fig.add_subplot(gs[3, c]) for c in range(1, 4)],  # row 1
    ]
    # Class density axes (cols 1–3)
    dn_axes = [
        [fig.add_subplot(gs[1, c]) for c in range(1, 4)],  # row 0
        [fig.add_subplot(gs[4, c]) for c in range(1, 4)],  # row 1
    ]

    FS_TITLE = 9
    FS_LABEL = 8
    FS_TICK  = 7

    def _style_sc(ax, title):
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("UMAP 1", fontsize=FS_LABEL)
        ax.set_ylabel("UMAP 2", fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6, loc="upper right", markerscale=1.5)

    def _style_dn(ax):
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(alpha=0.2)

    def _style_sum(ax, title):
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("UMAP 1", fontsize=FS_LABEL)
        ax.set_ylabel("UMAP 2", fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.2)

    def _bg(ax):
        ax.scatter(q1_emb[::3, 0], q1_emb[::3, 1],
                   c=Q1_BG, s=2, alpha=0.4, linewidths=0,
                   label=f"Q1 all (n={len(q1_y):,})", zorder=1)

    # ── Summary col, row 0: projected all-PHZ sources coloured by phz_gal_prob ─
    all_proj_emb_list = []
    all_proj_gal_prob_list = []
    for emb, _, gal_prob in embeddings.values():
        if gal_prob is not None:
            all_proj_emb_list.append(emb)
            all_proj_gal_prob_list.append(gal_prob)
    all_proj_emb_pgal = np.vstack(all_proj_emb_list) if all_proj_emb_list else q1_emb
    all_proj_gal_prob = np.concatenate(all_proj_gal_prob_list) if all_proj_gal_prob_list else phz_gal_prob

    rng_sum = np.random.RandomState(0)
    idx_sum = np.arange(len(all_proj_emb_pgal))
    if len(idx_sum) > 15_000:
        idx_sum = rng_sum.choice(idx_sum, 15_000, replace=False)
    sc0 = ax_sum0.scatter(all_proj_emb_pgal[idx_sum, 0], all_proj_emb_pgal[idx_sum, 1],
                          c=all_proj_gal_prob[idx_sum], cmap=cmap_pgal,
                          vmin=0, vmax=1, s=3, alpha=0.6, linewidths=0,
                          rasterized=True)
    fig.colorbar(sc0, ax=ax_sum0, label="PHZ P(galaxy)", pad=0.02, fraction=0.046)
    _style_sum(ax_sum0, f"All PHZ projected: P(galaxy)  (n={len(all_proj_emb_pgal):,})")

    # ── Summary col, row 1: Q1 density ────────────────────────────────────────
    sc1 = ax_sum1.pcolormesh(x_edges, y_edges, H_q1.T,
                              cmap=cmap_dens_q1, norm=q1_dens_norm, rasterized=True)
    fig.colorbar(sc1, ax=ax_sum1, label="Count (log)", pad=0.02, fraction=0.046)
    _style_sum(ax_sum1, f"All PHZ projected density  (n={n_projected_total:,})")

    # ── Row 0: phz=1, 2, 3 ───────────────────────────────────────────────────
    for col, phz_val in enumerate([1, 2, 3]):
        ax = sc_axes[0][col]
        _bg(ax)
        label, color = PHZ_MAP[phz_val]
        n = embeddings[phz_val][1] if phz_val in embeddings else 0
        if phz_val in embeddings:
            emb, _, _ = embeddings[phz_val]
            alpha = 0.3 if phz_val in (1, 2) else 0.7
            ax.scatter(emb[:, 0], emb[:, 1], c=color, s=6, alpha=alpha,
                       linewidths=0, label=f"{label} (n={n:,})", zorder=3)
        _style_sc(ax, f"phz={phz_val} — {label}  (n={n:,})")

        ax_d = dn_axes[0][col]
        H = histograms[phz_val]
        sc2 = ax_d.pcolormesh(x_edges, y_edges, H.T,
                              cmap=cmap_dens, norm=shared_norm, rasterized=True, alpha=0.7)
        _style_dn(ax_d)
        fig.colorbar(sc2, ax=ax_d, label="Count (log)", pad=0.02, fraction=0.046)

    # ── Row 1, cols 0–1: phz=-1, 0 ───────────────────────────────────────────
    for col, phz_val in enumerate([-1, 0]):
        ax = sc_axes[1][col]
        _bg(ax)
        label, color = PHZ_MAP[phz_val]
        n = embeddings[phz_val][1] if phz_val in embeddings else 0
        if phz_val in embeddings:
            emb, _, _ = embeddings[phz_val]
            ax.scatter(emb[:, 0], emb[:, 1], c=color, s=6, alpha=0.7,
                       linewidths=0, label=f"{label} (n={n:,})", zorder=3)
        _style_sc(ax, f"phz={phz_val} — {label}  (n={n:,})")

        ax_d = dn_axes[1][col]
        H = histograms[phz_val]
        sc2 = ax_d.pcolormesh(x_edges, y_edges, H.T,
                              cmap=cmap_dens, norm=shared_norm, rasterized=True, alpha=0.7)
        _style_dn(ax_d)
        fig.colorbar(sc2, ax=ax_d, label="Count (log)", pad=0.02, fraction=0.046)

    # ── Row 1, col 2: QSO variants ────────────────────────────────────────────
    ax = sc_axes[1][2]
    _bg(ax)
    for phz_val, (label, color) in QSO_MAP.items():
        if phz_val not in embeddings:
            continue
        emb, n, _ = embeddings[phz_val]
        ax.scatter(emb[:, 0], emb[:, 1], c=color, s=6, alpha=0.8,
                   linewidths=0, label=f"{label} phz={phz_val} (n={n:,})", zorder=3)
    n_qso_total = sum(embeddings[v][1] for v in QSO_MAP if v in embeddings)
    _style_sc(ax, f"phz=4/5/6/7 — QSO variants  (n={n_qso_total:,})")

    ax_d = dn_axes[1][2]
    H = histograms["QSO"]
    sc2 = ax_d.pcolormesh(x_edges, y_edges, H.T,
                          cmap=cmap_dens, norm=shared_norm, rasterized=True, alpha=0.7)
    _style_dn(ax_d)
    fig.colorbar(sc2, ax=ax_d, label="Count (log)", pad=0.02, fraction=0.046)

    fig.suptitle(
        f"PHZ classification categories projected into Q1 UMAP space\n"
        f"(Q1 mean/std normalisation  |  up to {MAX_PER_CLASS:,} per class  |  {mag_label})",
        fontsize=12, fontweight="bold", y=0.995,
    )

    out = OUTPUT_DIR / "q1_space_allphz.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
