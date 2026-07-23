#!/usr/bin/env python3
"""
Q1 individual UMAP coloured by grouped Gaia parameters.

Produces 5 plots:
  1. q1_features_quality.png   — astrometric quality (3 panels)
  2. q1_features_snr.png       — SNR features (3 panels: parallax/error, pm_sig, G flux/error)
  3. q1_features_color.png     — colour indices (2 panels)
  4. q1_features_excess.png    — excess / position (2 panels)
  5. q1_features_gmag.png      — G magnitude (1 panel)

Each plot has star (top) / galaxy (bottom) sub-rows per panel.

Reads:
  output/umap/arrays/q1_indiv.npz

Writes:
  output/umap/q1_individual/q1_features_*.png

Usage:
  python src/analyze_umap_q1_features_snr.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "output"

MAG_CUT = 17


def _add_cluster_labels(ax, centroids):
    """Add subtle cluster reference labels (C0–C4) at centroid positions."""
    import matplotlib.patheffects as pe
    # C0,C1,C2 shifted outside clusters -> dark text; C3,C4 inside -> white text
    y_offsets = {0: -4.4, 1: -4.2, 2: -4}
    outside = {0, 1, 2}
    for k, (cx, cy) in centroids.items():
        cy_adj = cy + y_offsets.get(k, 0)
        if k in outside:
            ax.text(cx, cy_adj, f"C{k}", fontsize=7, fontweight="bold",
                    color="dimgray", alpha=1, ha="center", va="center", zorder=5)
        else:
            ax.text(cx, cy_adj, f"C{k}", fontsize=7, fontweight="bold",
                    color="white", alpha=0.7, ha="center", va="center", zorder=5,
                    path_effects=[pe.withStroke(linewidth=2, foreground="black", alpha=0.3)])

FEATURE_GROUPS = [
    (
        "quality",
        "Astrometric quality features",
        ["astrometric_gof_al", "astrometric_chi2_al", "astrometric_excess_noise_sig"],
    ),
    (
        "color",
        "Colour features",
        ["bp_g", "g_rp"],
    ),
    (
        "excess",
        "Excess / position features",
        ["corrected_bp_rp_excess", "position_error"],
    ),
]

# SNR plot: 3 key SNR features in a single row
SNR_FEATURES = ["parallax_over_error", "pm_significance", "phot_g_mean_flux_over_error"]


def plot_all_features(embedding, X_raw, y, output_dir,
                      filename="q1_features.png",
                      xlim=None, ylim=None, cluster_centroids=None):
    """3-column grid of all 14 Gaia features, star (top) / galaxy (bottom) per panel."""
    n_features = len(SELECTED_GAIA_PARAMETERS)
    ncols = 3
    nfeature_rows = int(np.ceil(n_features / ncols))

    m_star = y == 0
    m_gal  = y == 1

    per_panel_height = 1.8
    panel_h   = 1.0
    inner_gap = 0.08
    outer_gap = 0.34

    height_ratios = []
    for i in range(nfeature_rows):
        height_ratios.extend([panel_h, inner_gap, panel_h])
        if i < nfeature_rows - 1:
            height_ratios.append(outer_gap)

    panel_fraction = panel_h / sum(height_ratios)
    fig_width  = ncols * 5
    fig_height = per_panel_height / panel_fraction

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = GridSpec(len(height_ratios), ncols,
                   figure=fig, height_ratios=height_ratios,
                   hspace=0, wspace=0.2)

    def _gs_top(feat_row): return feat_row * 4
    def _gs_bot(feat_row): return feat_row * 4 + 2

    for fi, feat_name in enumerate(SELECTED_GAIA_PARAMETERS):
        feat_row = fi // ncols
        col      = fi % ncols
        ax_top = fig.add_subplot(gs[_gs_top(feat_row), col])
        ax_bot = fig.add_subplot(gs[_gs_bot(feat_row), col])

        vals = X_raw[:, fi].astype(float)
        lo, hi = np.nanpercentile(vals, 1), np.nanpercentile(vals, 99)
        vals_clipped = np.clip(vals, lo, hi)

        for ax, mask, label in [
                (ax_top, m_star, "Stars"),
                (ax_bot, m_gal,  "Galaxies")]:
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c="lightgrey", s=4, alpha=0.18, marker="o",
                rasterized=True, linewidths=0, zorder=0,
            )
            sc = ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=vals_clipped[mask], cmap="RdYlBu_r",
                vmin=lo, vmax=hi,
                s=4, alpha=0.6, marker="o", rasterized=True,
                linewidths=0, zorder=1,
            )
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(alpha=0.25)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            if col == 0:
                ax.set_ylabel(label, fontsize=14, fontweight="bold", labelpad=12)
            else:
                ax.set_ylabel("")

        fig.colorbar(sc, ax=[ax_top, ax_bot], pad=0.02, fraction=0.046).ax.tick_params(labelsize=11)
        ax_top.set_title(feat_name, fontsize=14, fontweight="bold", pad=10)
        if cluster_centroids is not None:
            for ax in (ax_top, ax_bot):
                _add_cluster_labels(ax, cluster_centroids)

    for fi in range(n_features, nfeature_rows * ncols):
        feat_row = fi // ncols
        col      = fi % ncols
        fig.add_subplot(gs[_gs_top(feat_row), col]).set_visible(False)
        fig.add_subplot(gs[_gs_bot(feat_row), col]).set_visible(False)

    out = output_dir / filename
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")


def plot_feature_group(embedding, X_raw, y, features, group_label,
                       output_dir, filename,
                       xlim=None, ylim=None, mag_label=None,
                       cluster_centroids=None):
    """1 row × N columns, each with star (top) + galaxy (bottom) sub-panels."""
    ncols = len(features)
    feat_indices = [list(SELECTED_GAIA_PARAMETERS).index(f) for f in features]

    m_star = y == 0
    m_gal  = y == 1

    panel_h   = 1.0
    inner_gap = 0.08
    height_ratios = [panel_h, inner_gap, panel_h]

    per_panel_height = 1.5
    panel_fraction = panel_h / sum(height_ratios)
    fig_width  = ncols * 3.8
    fig_height = per_panel_height / panel_fraction

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = GridSpec(len(height_ratios), ncols,
                   figure=fig, height_ratios=height_ratios,
                   hspace=0, wspace=0.22)

    for col, (feat_name, fi) in enumerate(zip(features, feat_indices)):
        ax_top = fig.add_subplot(gs[0, col])
        ax_bot = fig.add_subplot(gs[2, col])

        vals = X_raw[:, fi].astype(float)
        lo, hi = np.nanpercentile(vals, 1), np.nanpercentile(vals, 99)
        vals_clipped = np.clip(vals, lo, hi)

        for ax, mask, label in [
                (ax_top, m_star, "Stars"),
                (ax_bot, m_gal,  "Galaxies")]:
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c="lightgrey", s=4, alpha=0.18, marker="o",
                rasterized=True, linewidths=0, zorder=0,
            )
            sc = ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=vals_clipped[mask], cmap="RdYlBu_r",
                vmin=lo, vmax=hi,
                s=4, alpha=0.6, marker="o", rasterized=True,
                linewidths=0, zorder=1,
            )
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(alpha=0.25)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            if col == 0:
                ax.set_ylabel(label, fontsize=9, fontweight="bold", labelpad=10)
            else:
                ax.set_ylabel("")

        fig.colorbar(sc, ax=[ax_top, ax_bot], pad=0.02, fraction=0.046).ax.tick_params(labelsize=9)
        ax_top.set_title(feat_name, fontsize=10, fontweight="bold", pad=6)
        if cluster_centroids is not None:
            for ax in (ax_top, ax_bot):
                _add_cluster_labels(ax, cluster_centroids)

    # fig.suptitle(
    #     f"Q1 UMAP — {group_label}  ({mag_label or f'G ≥ {MAG_CUT}'})  |  Upper = stars, Lower = galaxies",
    #     fontsize=12, fontweight="bold", y=1.1,
    # )
    out = output_dir / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")



def plot_gmag(embedding, g_mag, y, output_dir,
              filename="q1_features_gmag.png",
              xlim=None, ylim=None, mag_label=None,
              cluster_centroids=None):
    """Single panel pair (star top, galaxy bottom) coloured by G magnitude."""
    m_star = y == 0
    m_gal  = y == 1

    panel_h   = 1.0
    inner_gap = 0.04
    height_ratios = [panel_h, inner_gap, panel_h]

    per_panel_height = 1.3
    panel_fraction = panel_h / sum(height_ratios)
    fig_width  = 3.8
    fig_height = per_panel_height / panel_fraction

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = GridSpec(len(height_ratios), 1,
                   figure=fig, height_ratios=height_ratios,
                   hspace=0)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[2, 0])

    lo, hi = np.nanpercentile(g_mag, 1), np.nanpercentile(g_mag, 99)
    vals_clipped = np.clip(g_mag, lo, hi)

    for ax, mask, label in [
            (ax_top, m_star, "Stars"),
            (ax_bot, m_gal,  "Galaxies")]:
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c="lightgrey", s=4, alpha=0.18, marker="o",
            rasterized=True, linewidths=0, zorder=0,
        )
        sc = ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=vals_clipped[mask], cmap="RdYlBu_r",
            vmin=lo, vmax=hi,
            s=4, alpha=0.6, marker="o", rasterized=True,
            linewidths=0, zorder=1,
        )
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(alpha=0.25)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_ylabel(label, fontsize=9, fontweight="bold", labelpad=10)

    fig.colorbar(sc, ax=[ax_top, ax_bot], pad=0.02, fraction=0.046).ax.tick_params(labelsize=9)
    ax_top.set_title("phot_g_mean_mag", fontsize=10, fontweight="bold", pad=6)
    if cluster_centroids is not None:
        for ax in (ax_top, ax_bot):
            _add_cluster_labels(ax, cluster_centroids)

    # fig.suptitle(
    #     f"Q1 UMAP — G magnitude  ({mag_label or f'G ≥ {MAG_CUT}'})  |  Upper = stars, Lower = galaxies",
    #     fontsize=10, fontweight="bold", y=1.1,
    # )
    out = output_dir / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    indiv_path = ARRAYS_DIR / "q1_indiv.npz"
    if not indiv_path.exists():
        print(f"ERROR: missing file {indiv_path}")
        print("Run query_umap_array.py first.")
        return 1

    print("Loading Q1 individual UMAP arrays...")
    data      = np.load(indiv_path, allow_pickle=True)
    mag_cut_faint = int(data['mag_cut_faint']) if 'mag_cut_faint' in data else 99
    mag_label = f'G ≥ {MAG_CUT}' if mag_cut_faint >= 90 else f'{MAG_CUT} ≤ G ≤ {mag_cut_faint}'
    embedding = data['embedding']
    y         = data['y']
    X_raw     = data['X_raw']
    g_mag     = data['g_mag'].astype(float) if 'g_mag' in data else None
    print(f"  embedding shape: {embedding.shape}  (0=STAR, 1=GALAXY)")

    xlim = (embedding[:, 0].min() - 1.0, embedding[:, 0].max() + 1.0)
    ylim = (embedding[:, 1].min() - 1.0, embedding[:, 1].max() + 1.0)

    # Load cluster labels for reference annotations
    cluster_centroids = None
    clabel_path = ARRAYS_DIR / "q1_cluster_labels.npy"
    if clabel_path.exists():
        labels = np.load(clabel_path)
        cluster_centroids = {}
        for k in sorted(set(labels[labels >= 0])):
            mask_k = labels == k
            cluster_centroids[k] = (
                embedding[mask_k, 0].mean(),
                embedding[mask_k, 1].mean(),
            )
        print(f"  Loaded cluster centroids: {list(cluster_centroids.keys())}")
    else:
        print("  WARNING: q1_cluster_labels.npy not found, skipping cluster labels")

    print("\nPlotting Q1 UMAP — all 14 features grid...")
    plot_all_features(embedding, X_raw, y, OUTPUT_DIR, xlim=xlim, ylim=ylim)

    for group_key, group_label, features in FEATURE_GROUPS:
        print(f"\nPlotting Q1 UMAP — {group_label}...")
        plot_feature_group(
            embedding, X_raw, y, features, group_label,
            OUTPUT_DIR, filename=f"q1_features_{group_key}.png",
            xlim=xlim, ylim=ylim, mag_label=mag_label,
            cluster_centroids=cluster_centroids,
        )

    print("\nPlotting Q1 UMAP — SNR features...")
    plot_feature_group(
        embedding, X_raw, y, SNR_FEATURES, "SNR features",
        OUTPUT_DIR, filename="q1_features_snr.png",
        xlim=xlim, ylim=ylim, mag_label=mag_label,
        cluster_centroids=cluster_centroids,
    )

    if g_mag is not None:
        print("\nPlotting Q1 UMAP — G magnitude...")
        plot_gmag(embedding, g_mag, y, OUTPUT_DIR,
                  xlim=xlim, ylim=ylim, mag_label=mag_label,
                  cluster_centroids=cluster_centroids)
    else:
        print("\n  WARNING: g_mag not found in q1_indiv.npz, skipping G mag plot.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
