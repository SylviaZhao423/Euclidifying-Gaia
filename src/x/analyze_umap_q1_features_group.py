#!/usr/bin/env python3
"""
Q1 individual UMAP coloured by grouped Gaia parameters.

Produces 5 plots:
  1. q1_features_quality.png   — astrometric quality (3 panels)
  2. q1_features_motion.png    — motion SNR (2 panels)
  3. q1_features_flux.png      — flux SNR (3 panels)
  4. q1_features_color.png     — colour indices (2 panels)
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
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "q1_individual" / "insert"

MAG_CUT = 17

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

# Combined 2-row SNR plot: motion (top row) + flux (bottom row)
SNR_MOTION = ["parallax_over_error", "pm_significance"]
SNR_FLUX   = ["phot_g_mean_flux_over_error", "phot_bp_mean_flux_over_error",
              "phot_rp_mean_flux_over_error"]


def plot_feature_group(embedding, X_raw, y, features, group_label,
                       output_dir, filename,
                       xlim=None, ylim=None, mag_label=None):
    """1 row × N columns, each with star (top) + galaxy (bottom) sub-panels."""
    ncols = len(features)
    feat_indices = [list(SELECTED_GAIA_PARAMETERS).index(f) for f in features]

    m_star = y == 0
    m_gal  = y == 1

    panel_h   = 1.0
    inner_gap = 0.04
    height_ratios = [panel_h, inner_gap, panel_h]

    per_panel_height = 1.3
    panel_fraction = panel_h / sum(height_ratios)
    fig_width  = ncols * 3.8
    fig_height = per_panel_height / panel_fraction

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = GridSpec(len(height_ratios), ncols,
                   figure=fig, height_ratios=height_ratios,
                   hspace=0, wspace=0.35)

    for col, (feat_name, fi) in enumerate(zip(features, feat_indices)):
        ax_top = fig.add_subplot(gs[0, col])
        ax_bot = fig.add_subplot(gs[2, col])

        vals = X_raw[:, fi].astype(float)
        lo, hi = np.nanpercentile(vals, 1), np.nanpercentile(vals, 99)
        vals_clipped = np.clip(vals, lo, hi)

        for ax, mask, label in [
                (ax_top, m_star, "Q1 Stars"),
                (ax_bot, m_gal,  "Q1 Galaxies")]:
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
            ax.set_ylabel(f"{label}\n(n={mask.sum():,})", fontsize=7)

        fig.colorbar(sc, ax=[ax_top, ax_bot], pad=0.02, fraction=0.046).ax.tick_params(labelsize=7)
        ax_top.set_title(feat_name, fontsize=8, fontweight="bold", pad=4)

    # fig.suptitle(
    #     f"Q1 UMAP — {group_label}  ({mag_label or f'G ≥ {MAG_CUT}'})  |  Upper = stars, Lower = galaxies",
    #     fontsize=12, fontweight="bold", y=1.1,
    # )
    out = output_dir / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")


def plot_snr_combined(embedding, X_raw, y, output_dir,
                     filename="q1_features_snr.png",
                     xlim=None, ylim=None, mag_label=None):
    """2-row combined plot: motion SNR (top) + flux SNR (bottom).
    Each feature gets star/galaxy sub-panels stacked vertically.
    Top row: 3 cols (2 motion + 1 empty), bottom row: 3 cols (3 flux)."""
    ncols = 3  # max of len(SNR_MOTION), len(SNR_FLUX)
    motion_indices = [list(SELECTED_GAIA_PARAMETERS).index(f) for f in SNR_MOTION]
    flux_indices   = [list(SELECTED_GAIA_PARAMETERS).index(f) for f in SNR_FLUX]

    m_star = y == 0
    m_gal  = y == 1

    # Layout: 2 feature-rows, each with star+gap+galaxy sub-rows
    panel_h   = 1.0
    inner_gap = 0.04
    outer_gap = 0.32
    height_ratios = [
        panel_h, inner_gap, panel_h,   # row 0: motion
        outer_gap,
        panel_h, inner_gap, panel_h,   # row 1: flux
    ]

    per_panel_height = 1.3
    panel_fraction = panel_h / sum(height_ratios)
    fig_width  = ncols * 3.8
    fig_height = per_panel_height / panel_fraction

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = GridSpec(len(height_ratios), ncols,
                   figure=fig, height_ratios=height_ratios,
                   hspace=0, wspace=0.35)

    def _plot_row(gs_top, gs_bot, features, feat_indices):
        for col, (feat_name, fi) in enumerate(zip(features, feat_indices)):
            ax_top = fig.add_subplot(gs[gs_top, col])
            ax_bot = fig.add_subplot(gs[gs_bot, col])

            vals = X_raw[:, fi].astype(float)
            lo, hi = np.nanpercentile(vals, 1), np.nanpercentile(vals, 99)
            vals_clipped = np.clip(vals, lo, hi)

            for ax, mask, label in [
                    (ax_top, m_star, "Q1 Stars"),
                    (ax_bot, m_gal,  "Q1 Galaxies")]:
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
                ax.set_ylabel(f"{label}\n(n={mask.sum():,})", fontsize=7)

            fig.colorbar(sc, ax=[ax_top, ax_bot], pad=0.02, fraction=0.046).ax.tick_params(labelsize=7)
            ax_top.set_title(feat_name, fontsize=8, fontweight="bold", pad=4)

    # Row 0: motion (gs rows 0, 2)
    _plot_row(0, 2, SNR_MOTION, motion_indices)
    # Hide unused columns in motion row
    for col in range(len(SNR_MOTION), ncols):
        fig.add_subplot(gs[0, col]).set_visible(False)
        fig.add_subplot(gs[2, col]).set_visible(False)

    # Row 1: flux (gs rows 4, 6)
    _plot_row(4, 6, SNR_FLUX, flux_indices)

    out = output_dir / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")


def plot_gmag(embedding, g_mag, y, output_dir,
              filename="q1_features_gmag.png",
              xlim=None, ylim=None, mag_label=None):
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
            (ax_top, m_star, "Q1 Stars"),
            (ax_bot, m_gal,  "Q1 Galaxies")]:
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
        ax.set_ylabel(f"{label}\n(n={mask.sum():,})", fontsize=7)

    fig.colorbar(sc, ax=[ax_top, ax_bot], pad=0.02, fraction=0.046).ax.tick_params(labelsize=7)
    ax_top.set_title("phot_g_mean_mag", fontsize=8, fontweight="bold", pad=4)

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

    for group_key, group_label, features in FEATURE_GROUPS:
        print(f"\nPlotting Q1 UMAP — {group_label}...")
        plot_feature_group(
            embedding, X_raw, y, features, group_label,
            OUTPUT_DIR, filename=f"q1_features_{group_key}.png",
            xlim=xlim, ylim=ylim, mag_label=mag_label,
        )

    print("\nPlotting Q1 UMAP — SNR features (motion + flux)...")
    plot_snr_combined(embedding, X_raw, y, OUTPUT_DIR,
                      xlim=xlim, ylim=ylim, mag_label=mag_label)

    if g_mag is not None:
        print("\nPlotting Q1 UMAP — G magnitude...")
        plot_gmag(embedding, g_mag, y, OUTPUT_DIR,
                  xlim=xlim, ylim=ylim, mag_label=mag_label)
    else:
        print("\n  WARNING: g_mag not found in q1_indiv.npz, skipping G mag plot.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
