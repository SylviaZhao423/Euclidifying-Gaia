#!/usr/bin/env python3
"""
Load pre-fitted UMAP arrays and generate all plots for both regular combined
and rebalanced combined UMAP embeddings. No DB queries, no UMAP fitting.

Reads (from output/umap/arrays/):
  combined.npz, combined_reducer.joblib
  combined_rebalanced.npz, combined_rebalanced_reducer.joblib
  desi_q1_df.csv, desi_ero_df.csv

Writes (all to output/umap/combined/):
  combined_umap.png
  combined_umap_rebalanced.png
  combined_compare.png
  combined_features.png
  combined_features_density.png
  combined_features_density_sg.png
  combined_gmag.png             — 2-panel: regular + rebalanced side-by-side

  combined_compare_rebalanced.png
  combined_features_rebalanced.png

Usage:
    python src/analyze_umap_plot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / 'output' / 'umap' / 'arrays'
OUTPUT_DIR  = PROJECT_DIR / 'output' / 'umap' / 'combined'

MAG_CUT = 17


# ---------------------------------------------------------------------------
# Utility (mirrors the copy in analyze_umap.py — kept here for self-containment)
# ---------------------------------------------------------------------------

def _standardize(X, mean_=None, std_=None):
    """Z-score standardize X. If mean_/std_ not provided, compute from X."""
    if mean_ is None:
        mean_ = np.nanmean(X, axis=0)
    if std_ is None:
        std_ = np.nanstd(X, axis=0)
    std_safe = np.where(std_ > 0, std_, 1.0)
    return (X - mean_) / std_safe, mean_, std_


# ---------------------------------------------------------------------------
# 3-panel combined UMAP (moved from analyze_umap.py / plot_umap_comparison)
# ---------------------------------------------------------------------------

_INDIV_COLORS = {
    'Q1':  {0: 'coral',     1: '#1976D2'},
    'ERO': {0: 'lightpink', 1: 'deepskyblue'},
}
_LABEL_NAMES = {0: 'STAR', 1: 'GALAXY'}
_COMB_STYLES = [
    (0, 'Q1 STAR',    'coral',        'o'),
    (2, 'ERO STAR',   'lightpink',    's'),
    (1, 'Q1 GALAXY',  '#1976D2',      'o'),
    (3, 'ERO GALAXY', 'deepskyblue',  's'),
]


def _style_umap_ax(ax, title, legend_fontsize=10):
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(markerscale=3, fontsize=legend_fontsize, loc='upper right')
    ax.grid(alpha=0.3)


def plot_combined_umap(q1_emb, q1_y, ero_emb, ero_y, comb_emb, categories,
                       output_path: Path, title_suffix: str = '') -> None:
    """3-panel plot: Q1 individual | ERO individual | Combined (4 categories)."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    for label_val in [0, 1]:
        mask = q1_y == label_val
        axes[0].scatter(q1_emb[mask, 0], q1_emb[mask, 1],
                        c=_INDIV_COLORS['Q1'][label_val],
                        label=f'{_LABEL_NAMES[label_val]} (n={mask.sum():,})',
                        alpha=0.25, s=6, linewidths=0)
    _style_umap_ax(axes[0], f'Q1 — individual UMAP{title_suffix}')

    for label_val in [0, 1]:
        mask = ero_y == label_val
        axes[1].scatter(ero_emb[mask, 0], ero_emb[mask, 1],
                        c=_INDIV_COLORS['ERO'][label_val],
                        label=f'{_LABEL_NAMES[label_val]} (n={mask.sum():,})',
                        alpha=0.25, s=6, linewidths=0, marker='s')
    _style_umap_ax(axes[1], f'ERO — individual UMAP{title_suffix}')

    for cat_val, cat_label, color, marker in _COMB_STYLES:
        mask = categories == cat_val
        axes[2].scatter(comb_emb[mask, 0], comb_emb[mask, 1],
                        c=color, label=f'{cat_label} (n={mask.sum():,})',
                        alpha=0.25, s=6, linewidths=0, marker=marker)
    _style_umap_ax(axes[2], 'Combined Q1 + ERO', legend_fontsize=9)

    plt.suptitle(
        f'Combined UMAP Visualization of Gaia Feature Space'
        f'  (G \u2265 {MAG_CUT},  {len(SELECTED_GAIA_PARAMETERS)} features)',
        fontsize=14, fontweight='bold', y=1.0,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {output_path.name}")


# ---------------------------------------------------------------------------
# Moved from analyze_umap.py
# ---------------------------------------------------------------------------

def _load_agreement_id_sets():
    """Load 4 agreement/disagreement CSVs and return sets of g_source_id strings."""
    base = Path(__file__).parent.parent / 'output' / 'q1_to_ero'
    csv_map = {
        'agree_star':   base / 'agree_both_star.csv',
        'agree_galaxy': base / 'agree_both_galaxy.csv',
        'disagree_sg':  base / 'disagree_ero_star_q1_gal.csv',  # ERO star, Q1 gal
        'disagree_gs':  base / 'disagree_ero_gal_q1_star.csv',  # ERO gal, Q1 star
    }
    result = {}
    for key, path in csv_map.items():
        df = pd.read_csv(path)
        result[key] = set(df['g_source_id'].astype(str))
        print(f"  Agreement [{key}]: {len(result[key]):,} IDs loaded")
    return result


def plot_umap_corner_compare(embedding, categories, output_dir, max_points_per_group=15000,
                              ero_source_ids=None, agreement_id_sets=None,
                              desi_q1_df=None, desi_ero_df=None,
                              reducer=None,
                              q1_mean=None, q1_std=None,
                              ero_mean=None, ero_std=None,
                              suffix=''):
    """Three-row 5-panel grid: scatter+DESI / agreement (ID lookup) / 2D histogram.
    DESI overlay uses projection (reducer.transform). Agreement uses ID lookup on embedded ERO IDs."""
    rng = np.random.RandomState(42)

    def _sample_mask(cat_value):
        idx = np.where(categories == cat_value)[0]
        if len(idx) > max_points_per_group:
            idx = rng.choice(idx, max_points_per_group, replace=False)
        mask = np.zeros(len(categories), dtype=bool)
        mask[idx] = True
        return mask

    specs = [
        ("Combined",     None, None,               "o"),
        ("Q1 Stars",     0,    "coral",             "o"),
        ("ERO Stars",    2,    "lightpink",            "s"),
        ("Q1 Galaxies",  1,    "#1976D2",          "o"),
        ("ERO Galaxies", 3,    "deepskyblue",     "s"),
    ]
    cat_colors = {0: "coral", 1: "#1976D2", 2: "lightpink", 3: "deepskyblue"}

    x_all = embedding[:, 0]
    y_all = embedding[:, 1]
    n_bins = 150

    x_lo, x_hi = np.nanmin(x_all), np.nanmax(x_all)
    y_lo, y_hi = np.nanmin(y_all), np.nanmax(y_all)
    # Make both ranges equal so set_aspect('equal') produces a square box
    x_mid, y_mid = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
    half = max(x_hi - x_lo, y_hi - y_lo) / 2 * 1.03  # 3% padding
    x_lo, x_hi = x_mid - half, x_mid + half
    y_lo, y_hi = y_mid - half, y_mid + half
    x_edges = np.linspace(x_lo, x_hi, n_bins + 1)
    y_edges = np.linspace(y_lo, y_hi, n_bins + 1)

    from matplotlib.colors import LogNorm
    cmap_hist = plt.cm.Blues.copy()
    cmap_hist.set_bad(alpha=0)

    fig, axes = plt.subplots(3, 5, figsize=(36, 18), constrained_layout=True)

    for col, (title, cat_value, color, marker) in enumerate(specs):
        ax_sc  = axes[0, col]
        ax_ag  = axes[1, col]
        ax_h   = axes[2, col]

        # ── Row 1: scatter ────────────────────────────────────────────────────
        if cat_value is None:
            for cv, cc in cat_colors.items():
                m = _sample_mask(cv)
                ax_sc.scatter(embedding[m, 0], embedding[m, 1],
                              c=cc, s=4, alpha=0.15, linewidths=0)
        else:
            mask = _sample_mask(cat_value)
            ax_sc.scatter(embedding[mask, 0], embedding[mask, 1],
                          c=color, s=6, alpha=0.25, linewidths=0, marker=marker,
                          label=f'n={mask.sum():,}')

        # DESI overlay — project DESI sources via reducer.transform()
        if cat_value is not None and reducer is not None:
            desi_df = desi_q1_df if cat_value in (0, 1) else desi_ero_df
            mean_ = q1_mean if cat_value in (0, 1) else ero_mean
            std_  = q1_std  if cat_value in (0, 1) else ero_std
            if desi_df is not None and mean_ is not None:
                sub = desi_df[desi_df['cat_value'] == cat_value]
                if len(sub) > 0:
                    X_d = sub[SELECTED_GAIA_PARAMETERS].values.astype(float)
                    for j in range(X_d.shape[1]):
                        bad = np.isnan(X_d[:, j])
                        if bad.any():
                            X_d[bad, j] = np.nanmedian(X_d[:, j])
                    X_d_scaled, _, _ = _standardize(X_d, mean_=mean_, std_=std_)
                    emb_desi = reducer.transform(X_d_scaled)
                    desi_style = {
                        'GALAXY': ('#2196F3', 'o'),
                        'STAR':   ('#F44336', '^'),
                        'QSO':    ('#FF9800', 's'),
                    }
                    for spec, (clr, mkr) in desi_style.items():
                        m = sub['spectype'].values == spec
                        if m.sum() == 0:
                            continue
                        ax_sc.scatter(emb_desi[m, 0], emb_desi[m, 1],
                                      c=clr, marker=mkr, s=30, alpha=0.85,
                                      linewidths=0.4, edgecolors='white',
                                      label=f"DESI {spec}  n={m.sum():,}", zorder=5)
        ax_sc.legend(fontsize=7, loc='upper right', markerscale=1.5)
        ax_sc.set_title(title, fontsize=11, fontweight='bold')
        ax_sc.set_xlabel('UMAP 1', fontsize=10)
        ax_sc.set_ylabel('UMAP 2', fontsize=10)
        ax_sc.set_xlim(x_lo, x_hi)
        ax_sc.set_ylim(y_lo, y_hi)
        ax_sc.set_aspect('equal', adjustable='box')
        ax_sc.tick_params(labelsize=8)
        ax_sc.grid(alpha=0.25)

        # ── Row 2: same ERO dots as row 1, colored by Q1 agreement (ERO panels only) ──
        n_q1_emb = (categories < 2).sum()  # number of Q1 sources in embedding
        if cat_value in (2, 3) and ero_source_ids is not None and agreement_id_sets is not None:
            # Get the full-embedding mask for this category
            full_mask = _sample_mask(cat_value)
            # Map to ERO-local indices (ERO sources start at n_q1_emb in embedding)
            ero_local_idx = np.where(full_mask[n_q1_emb:])[0]
            ids_disp = ero_source_ids[ero_local_idx]
            emb_disp = embedding[full_mask]

            if cat_value == 2:
                key_agree, key_disagree = 'agree_star', 'disagree_sg'
                lbl_agree, lbl_disagree = 'Q1 agree (star)', 'Q1 disagree (gal)'
            else:
                key_agree, key_disagree = 'agree_galaxy', 'disagree_gs'
                lbl_agree, lbl_disagree = 'Q1 agree (gal)', 'Q1 disagree (star)'

            set_agree    = agreement_id_sets[key_agree]
            set_disagree = agreement_id_sets[key_disagree]
            m_agree    = np.array([sid in set_agree    for sid in ids_disp])
            m_disagree = np.array([sid in set_disagree for sid in ids_disp])
            m_neither  = ~m_agree & ~m_disagree

            # Plot: no-match first (background), then agree, then disagree on top
            ax_ag.scatter(emb_disp[m_neither,  0], emb_disp[m_neither,  1],
                          c='lightgrey', s=6, alpha=0.25, linewidths=0, marker='s',
                          label=f'no Q1 match  n={m_neither.sum():,}')
            ax_ag.scatter(emb_disp[m_agree,    0], emb_disp[m_agree,    1],
                          c='limegreen', s=6, alpha=0.25, linewidths=0, marker='s',
                          label=f'{lbl_agree}  n={m_agree.sum():,}')
            ax_ag.scatter(emb_disp[m_disagree, 0], emb_disp[m_disagree, 1],
                          c='#E74C3C', marker='s', s=6, alpha=0.25,
                          linewidths=0, label=f'{lbl_disagree}  n={m_disagree.sum():,}', zorder=5)

            ax_ag.legend(fontsize=7, loc='upper right', markerscale=1.5)
            ax_ag.set_title(f'{title} — Q1 agreement', fontsize=11, fontweight='bold')
            ax_ag.set_xlim(x_lo, x_hi)
            ax_ag.set_ylim(y_lo, y_hi)
            ax_ag.set_aspect('equal', adjustable='box')
            ax_ag.set_xlabel('UMAP 1', fontsize=10)
            ax_ag.set_ylabel('UMAP 2', fontsize=10)
            ax_ag.tick_params(labelsize=8)
            ax_ag.grid(alpha=0.25)
        else:
            ax_ag.axis('off')

        # ── Row 3: 2D histogram ───────────────────────────────────────────────
        if cat_value is None:
            hmask = np.ones(len(x_all), dtype=bool)
        else:
            hmask = categories == cat_value
        H, _, _ = np.histogram2d(x_all[hmask], y_all[hmask], bins=[x_edges, y_edges])
        H = np.where(H > 0, H, np.nan)
        sc = ax_h.pcolormesh(x_edges, y_edges, H.T,
                             cmap=cmap_hist, norm=LogNorm(), rasterized=True)
        ax_h.set_xlabel('UMAP 1', fontsize=10)
        ax_h.set_ylabel('UMAP 2', fontsize=10)
        ax_h.set_xlim(x_lo, x_hi)
        ax_h.set_ylim(y_lo, y_hi)
        ax_h.set_aspect('equal', adjustable='box')
        # Colorbar anchored to axes box (not full cell height)
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        cax = inset_axes(ax_h, width='5%', height='100%', loc='lower left',
                         bbox_to_anchor=(1.02, 0, 1, 1),
                         bbox_transform=ax_h.transAxes, borderpad=0)
        fig.colorbar(sc, cax=cax, label='Count (log)')
        ax_h.tick_params(labelsize=8)
        ax_h.grid(alpha=0.25)

    fig.suptitle(
        f'Combined UMAP — class-wise comparison  (G ≥ {MAG_CUT},  {len(SELECTED_GAIA_PARAMETERS)} features)\n'
        f'Row 1: scatter (DESI overlay)   |   Row 2: Q1 agreement overlay (ERO panels only)'
        f'   |   Row 3: 2D count histogram (Blues, log scale)',
        fontsize=13, fontweight='bold'
    )
    out = output_dir / f'combined_compare{suffix}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out.name} saved")


def plot_umap_2d_hist(embedding, categories, output_dir, n_bins=100):
    """2D histogram of UMAP1 vs UMAP2, one panel per class and one combined.
    Output: combined_umap_2d_hist.png
    """
    from matplotlib.colors import LogNorm

    x_all, y_all = embedding[:, 0], embedding[:, 1]
    x_edges = np.linspace(np.nanpercentile(x_all, 0.5), np.nanpercentile(x_all, 99.5), n_bins + 1)
    y_edges = np.linspace(np.nanpercentile(y_all, 0.5), np.nanpercentile(y_all, 99.5), n_bins + 1)

    specs = [
        ('All sources',  None,        'Blues'),
        ('Q1 Stars',     0,           'Oranges'),
        ('Q1 Galaxies',  1,           'Blues'),
        ('ERO Stars',    2,           'Oranges'),
        ('ERO Galaxies', 3,           'Blues'),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(28, 5), constrained_layout=True)

    for ax, (title, cat_val, cmap_name) in zip(axes, specs):
        if cat_val is None:
            mask = np.ones(len(x_all), dtype=bool)
        else:
            mask = categories == cat_val

        H, _, _ = np.histogram2d(x_all[mask], y_all[mask], bins=[x_edges, y_edges])
        H = np.where(H > 0, H, np.nan)

        sc = ax.pcolormesh(x_edges, y_edges, H.T,
                           cmap=cmap_name, norm=LogNorm(), rasterized=True)
        fig.colorbar(sc, ax=ax, label='Count (log)')
        ax.set_title(f'{title}\n(n={mask.sum():,})', fontsize=10, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=10)
        ax.set_ylabel('UMAP 2', fontsize=10)
        ax.grid(alpha=0.25)

    fig.suptitle(
        f'Combined UMAP — 2D histogram  '
        f'(G ≥ {MAG_CUT},  {len(x_all):,} sources)',
        fontsize=12, fontweight='bold'
    )
    out = output_dir / 'combined_umap_2d_hist.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> combined_umap_2d_hist.png saved")


def plot_umap_density_gmag(embedding, g_mag, output_dir, n_bins=150):
    """Two-panel plot: source count density (log) + mean Gaia G mag, per 2D UMAP bin.
    Output: combined_umap_density_gmag.png
    """
    x_all, y_all = embedding[:, 0], embedding[:, 1]
    x_edges = np.linspace(np.nanpercentile(x_all, 0.5), np.nanpercentile(x_all, 99.5), n_bins + 1)
    y_edges = np.linspace(np.nanpercentile(y_all, 0.5), np.nanpercentile(y_all, 99.5), n_bins + 1)

    xi = np.clip(np.digitize(x_all, x_edges) - 1, 0, n_bins - 1)
    yi = np.clip(np.digitize(y_all, y_edges) - 1, 0, n_bins - 1)

    # Mean G mag
    gm = g_mag.astype(float)
    H_sum  = np.zeros((n_bins, n_bins))
    H_cnt2 = np.zeros((n_bins, n_bins))
    valid = np.isfinite(gm)
    np.add.at(H_sum,  (xi[valid], yi[valid]), gm[valid])
    np.add.at(H_cnt2, (xi[valid], yi[valid]), 1)
    H_gmag = np.where(H_cnt2 > 0, H_sum / H_cnt2, np.nan)
    gmag_vmin = np.nanpercentile(H_gmag[np.isfinite(H_gmag)], 2)
    gmag_vmax = np.nanpercentile(H_gmag[np.isfinite(H_gmag)], 98)

    cmap_gmag = plt.cm.YlOrRd_r.copy(); cmap_gmag.set_bad(alpha=0)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)

    sc = ax.pcolormesh(x_edges, y_edges, H_gmag.T,
                       cmap=cmap_gmag, vmin=gmag_vmin, vmax=gmag_vmax, rasterized=True)
    fig.colorbar(sc, ax=ax, label='Mean Gaia G mag')
    ax.set_title('Mean Gaia G mag', fontsize=12, fontweight='bold')
    ax.set_xlabel('UMAP 1', fontsize=11)
    ax.set_ylabel('UMAP 2', fontsize=11)
    ax.grid(alpha=0.25)

    fig.suptitle(
        f'Combined UMAP — mean Gaia G mag per 2D bin\n'
        f'(G ≥ {MAG_CUT},  {len(x_all):,} sources,  {len(SELECTED_GAIA_PARAMETERS)} features)',
        fontsize=12, fontweight='bold'
    )
    out = output_dir / 'combined_gmag.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> combined_gmag.png saved")


def plot_umap_gmag_comparison(embedding, g_mag, embedding_rebal, g_mag_rebal,
                               output_dir, n_bins=150):
    """Two-panel side-by-side: mean Gaia G mag for regular vs rebalanced UMAP.
    Output: combined_gmag.png (replaces both single-panel versions)
    """
    cmap_gmag = plt.cm.YlOrRd_r.copy()
    cmap_gmag.set_bad(alpha=0)

    def _compute(emb, gm):
        x_all, y_all = emb[:, 0], emb[:, 1]
        x_edges = np.linspace(np.nanpercentile(x_all, 0.5), np.nanpercentile(x_all, 99.5), n_bins + 1)
        y_edges = np.linspace(np.nanpercentile(y_all, 0.5), np.nanpercentile(y_all, 99.5), n_bins + 1)
        xi = np.clip(np.digitize(x_all, x_edges) - 1, 0, n_bins - 1)
        yi = np.clip(np.digitize(y_all, y_edges) - 1, 0, n_bins - 1)
        gm = gm.astype(float)
        H_sum  = np.zeros((n_bins, n_bins))
        H_cnt  = np.zeros((n_bins, n_bins))
        valid  = np.isfinite(gm)
        np.add.at(H_sum, (xi[valid], yi[valid]), gm[valid])
        np.add.at(H_cnt, (xi[valid], yi[valid]), 1)
        H_gmag = np.where(H_cnt > 0, H_sum / H_cnt, np.nan)
        return x_edges, y_edges, H_gmag

    x_edges, y_edges, H_gmag   = _compute(embedding,       g_mag)
    x_edges_r, y_edges_r, H_r  = _compute(embedding_rebal, g_mag_rebal)

    # Shared colour scale across both panels
    all_vals = np.concatenate([H_gmag[np.isfinite(H_gmag)], H_r[np.isfinite(H_r)]])
    vmin = np.nanpercentile(all_vals, 2)
    vmax = np.nanpercentile(all_vals, 98)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    for ax, (xe, ye, H, label, n) in zip(axes, [
        (x_edges,   y_edges,   H_gmag, 'Combined',            len(embedding)),
        (x_edges_r, y_edges_r, H_r,    'Combined (rebalanced)', len(embedding_rebal)),
    ]):
        sc = ax.pcolormesh(xe, ye, H.T,
                           cmap=cmap_gmag, vmin=vmin, vmax=vmax, rasterized=True)
        fig.colorbar(sc, ax=ax, label='Mean Gaia G mag')
        ax.set_title(f'{label}\n(n={n:,})', fontsize=12, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=11)
        ax.set_ylabel('UMAP 2', fontsize=11)
        ax.grid(alpha=0.25)

    fig.suptitle(
        f'Combined UMAP — mean Gaia G mag per 2D bin  '
        f'(G ≥ {MAG_CUT},  {len(SELECTED_GAIA_PARAMETERS)} features)',
        fontsize=12, fontweight='bold'
    )
    out = output_dir / 'combined_gmag.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> combined_gmag.png saved")


# ---------------------------------------------------------------------------

def _load_g_mag(arr):
    """Return g_mag array or None if it was stored as an empty placeholder."""
    if arr is None or len(arr) == 0:
        return None
    return arr


def main() -> int:
    # Verify arrays exist
    required = [
        ARRAYS_DIR / 'combined.npz',
        ARRAYS_DIR / 'combined_reducer.joblib',
        ARRAYS_DIR / 'combined_rebalanced.npz',
        ARRAYS_DIR / 'combined_rebalanced_reducer.joblib',
        ARRAYS_DIR / 'desi_q1_df.csv',
        ARRAYS_DIR / 'desi_ero_df.csv',
    ]
    for p in required:
        if not p.exists():
            print(f"ERROR: missing file {p}")
            print("Run analyze_umap_query.py first to generate all arrays.")
            return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UMAP Plot Generation")
    print("=" * 60)

    # ── 3-panel combined UMAP ────────────────────────────────────────────────
    print("\n[1] Plotting combined_umap.png / combined_umap_rebalanced.png...")
    q1_arr      = np.load(ARRAYS_DIR / 'q1_indiv.npz',             allow_pickle=True)
    ero_arr     = np.load(ARRAYS_DIR / 'ero_indiv.npz',            allow_pickle=True)
    comb_arr    = np.load(ARRAYS_DIR / 'combined.npz',             allow_pickle=True)
    plot_combined_umap(
        q1_emb=q1_arr['embedding'],    q1_y=q1_arr['y'],
        ero_emb=ero_arr['embedding'],  ero_y=ero_arr['y'],
        comb_emb=comb_arr['embedding'], categories=comb_arr['categories'],
        output_path=OUTPUT_DIR / 'combined_umap.png',
    )
    rebal_ero_path  = ARRAYS_DIR / 'ero_indiv_rebalanced.npz'
    rebal_comb_path = ARRAYS_DIR / 'combined_rebalanced.npz'
    if rebal_ero_path.exists() and rebal_comb_path.exists():
        ero_rebal_arr  = np.load(rebal_ero_path,  allow_pickle=True)
        comb_rebal_arr = np.load(rebal_comb_path, allow_pickle=True)
        plot_combined_umap(
            q1_emb=q1_arr['embedding'],           q1_y=q1_arr['y'],
            ero_emb=ero_rebal_arr['embedding'],   ero_y=ero_rebal_arr['y'],
            comb_emb=comb_rebal_arr['embedding'], categories=comb_rebal_arr['categories'],
            output_path=OUTPUT_DIR / 'combined_umap_rebalanced.png',
            title_suffix=' (rebalanced)',
        )
    else:
        print("  WARNING: rebalanced arrays not found — skipping combined_umap_rebalanced.png")

    # Load DESI DataFrames
    print("\n[2] Loading DESI DataFrames...")
    desi_q1_df  = pd.read_csv(ARRAYS_DIR / 'desi_q1_df.csv')
    desi_ero_df = pd.read_csv(ARRAYS_DIR / 'desi_ero_df.csv')
    print(f"  Q1xDESI: {len(desi_q1_df):,}   EROxDESI: {len(desi_ero_df):,}")

    # Load agreement ID sets
    print("\n[3] Loading agreement ID sets...")
    agreement_id_sets = _load_agreement_id_sets()

    # ── Regular combined ────────────────────────────────────────────────────
    print("\n[4] Loading regular combined arrays...")
    data = np.load(ARRAYS_DIR / 'combined.npz', allow_pickle=True)
    embedding_comb      = data['embedding']
    X_raw_comb          = data['X_raw']
    categories_comb     = data['categories']
    g_mag_comb          = _load_g_mag(data['g_mag'])
    source_ids_ero_comb = data['source_ids_ero']
    q1_mean_comb        = data['q1_mean']
    q1_std_comb         = data['q1_std']
    ero_mean_comb       = data['ero_mean']
    ero_std_comb        = data['ero_std']
    reducer_comb        = joblib.load(ARRAYS_DIR / 'combined_reducer.joblib')
    print(f"  Loaded {len(embedding_comb):,} embedded sources")

    print("\n  [combined] plot_umap_corner_compare...")
    plot_umap_corner_compare(
        embedding_comb, categories_comb, OUTPUT_DIR,
        ero_source_ids=source_ids_ero_comb,
        agreement_id_sets=agreement_id_sets,
        desi_q1_df=desi_q1_df, desi_ero_df=desi_ero_df,
        reducer=reducer_comb,
        q1_mean=q1_mean_comb, q1_std=q1_std_comb,
        ero_mean=ero_mean_comb, ero_std=ero_std_comb,
    )

    # ── Rebalanced combined ─────────────────────────────────────────────────
    print("\n[5] Loading rebalanced combined arrays...")
    data_rebal = np.load(ARRAYS_DIR / 'combined_rebalanced.npz', allow_pickle=True)
    embedding_rebal      = data_rebal['embedding']
    X_raw_rebal          = data_rebal['X_raw']
    categories_rebal     = data_rebal['categories']
    g_mag_rebal          = _load_g_mag(data_rebal['g_mag'])
    source_ids_ero_rebal = data_rebal['source_ids_ero']
    q1_mean_rebal        = data_rebal['q1_mean']
    q1_std_rebal         = data_rebal['q1_std']
    ero_mean_rebal       = data_rebal['ero_mean']
    ero_std_rebal        = data_rebal['ero_std']
    reducer_rebal        = joblib.load(ARRAYS_DIR / 'combined_rebalanced_reducer.joblib')
    print(f"  Loaded {len(embedding_rebal):,} embedded sources (rebalanced)")

    print("\n  [rebalanced] plot_umap_corner_compare...")
    plot_umap_corner_compare(
        embedding_rebal, categories_rebal, OUTPUT_DIR,
        ero_source_ids=source_ids_ero_rebal,
        agreement_id_sets=agreement_id_sets,
        desi_q1_df=desi_q1_df, desi_ero_df=desi_ero_df,
        reducer=reducer_rebal,
        q1_mean=q1_mean_rebal, q1_std=q1_std_rebal,
        ero_mean=ero_mean_rebal, ero_std=ero_std_rebal,
        suffix='_rebalanced',
    )

    # ── Combined gmag: 2-panel regular + rebalanced (or 1-panel fallback) ───
    if g_mag_comb is not None and g_mag_rebal is not None:
        print("\n[6] plot_umap_gmag_comparison (2-panel)...")
        plot_umap_gmag_comparison(
            embedding_comb, g_mag_comb,
            embedding_rebal, g_mag_rebal,
            OUTPUT_DIR,
        )
    elif g_mag_comb is not None:
        print("\n[6] plot_umap_density_gmag (1-panel, rebalanced g_mag unavailable)...")
        plot_umap_density_gmag(embedding_comb, g_mag_comb, OUTPUT_DIR)
    else:
        print("\n[6] Skipping gmag plot (g_mag not available)")

    print("\n  -> All plots saved.")
    print("=" * 60)
    print("Plot generation complete!")
    print(f"  Output dir: {OUTPUT_DIR}")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
