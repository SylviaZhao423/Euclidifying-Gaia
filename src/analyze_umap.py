#!/usr/bin/env python3
"""
UMAP visualization of Gaia feature space for Q1 and ERO datasets.

Outputs:
  combined_umap.png               — 3-panel individual + combined UMAP
  combined_umap_corner_compare.png — 4-panel Q1/ERO class comparison in combined UMAP space
  combined_umap_features.png      — per-feature colour maps (stars / galaxies)

Usage:
    PGUSER=<user> python src/analyze_umap.py [--test-mode]
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output' / 'umap'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

CLASS_MAPPING = {'STAR': 0, 'GALAXY': 1}

MAG_CUT = 17


def query_dataset(username, data_source, test_mode=False):
    """Query dataset. Returns X, y arrays."""
    import psycopg2

    conn = psycopg2.connect(
        host=WSDB_HOST, database=WSDB_DB, user=username
    )

    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])

    if data_source == 'q1':
        query = f"""
            SELECT
                {gaia_cols},
                CASE
                    WHEN p_phz_classification = 1 THEN 'STAR'
                    WHEN p_phz_classification = 2 THEN 'GALAXY'
                END as label
            FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
            WHERE p_phz_classification IN (1, 2)
                AND g_phot_g_mean_mag >= {MAG_CUT}
            ORDER BY g_source_id
        """
    else:  # ero
        query = f"""
            SELECT
                {gaia_cols},
                CASE
                    WHEN e_v_class_star >= 0.65 THEN 'STAR'
                    WHEN e_v_class_star < 0.65 THEN 'GALAXY'
                END as label
            FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
            WHERE e_v_class_star IS NOT NULL
                AND g_phot_g_mean_mag >= {MAG_CUT}
            ORDER BY g_source_id
        """

    if test_mode:
        query += " LIMIT 10000"

    print(f"  Querying {data_source.upper()} dataset...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Clean features
    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    X = df[SELECTED_GAIA_PARAMETERS].values
    y = df['label'].map(CLASS_MAPPING).values

    n_star = (y == 0).sum()
    n_galaxy = (y == 1).sum()
    print(f"  -> {len(X):,} sources — STAR: {n_star:,}, GALAXY: {n_galaxy:,}")

    return X, y


def _clean_for_umap(X, sample_size, rng):
    """Subsample, fill NaN with median, standardize.
    Returns (X_scaled, X_raw_filled, idx) where X_raw_filled has NaNs
    replaced by median but is NOT standardized (for coloring plots)."""
    if len(X) > sample_size:
        idx = rng.choice(len(X), sample_size, replace=False)
    else:
        idx = np.arange(len(X))

    X_raw = X[idx].copy()
    for col in range(X_raw.shape[1]):
        col_data = X_raw[:, col]
        mask = np.isnan(col_data)
        if mask.any():
            col_data[mask] = np.nanmedian(col_data)

    X_clean = X_raw.copy()
    for col in range(X_clean.shape[1]):
        std  = np.nanstd(X_clean[:, col])
        mean = np.nanmean(X_clean[:, col])
        if std > 0:
            X_clean[:, col] = (X_clean[:, col] - mean) / std

    return X_clean, X_raw, idx


def plot_umap_comparison(X_q1, y_q1, X_ero, y_ero, output_dir, sample_size=50000):
    """2×3 UMAP grid.
    Row 1: Q1 individual | ERO individual | (empty)
    Row 2: Q1 in combined space | ERO in combined space | All combined (4 categories)
    """
    rng = np.random.RandomState(42)

    # --- Row 1: individual UMAP fits ---
    embeddings = []
    for X, y, name in [(X_q1, y_q1, 'Q1'), (X_ero, y_ero, 'ERO')]:
        X_clean, _, idx = _clean_for_umap(X, sample_size, rng)
        y_sub = y[idx]
        print(f"  Fitting UMAP for {name} ({len(X_clean):,} sources)...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding = reducer.fit_transform(X_clean)
        embeddings.append((embedding, y_sub, name))

    # --- Row 2: combined UMAP fit ---
    half_sample = sample_size // 2
    X_q1_clean, X_q1_raw, idx_q1 = _clean_for_umap(X_q1, half_sample, rng)
    X_ero_clean, X_ero_raw, idx_ero = _clean_for_umap(X_ero, half_sample, rng)
    y_q1_sub = y_q1[idx_q1]
    y_ero_sub = y_ero[idx_ero]
    n_q1 = len(X_q1_clean)

    X_combined = np.vstack([X_q1_clean, X_ero_clean])
    # Category labels: 0=Q1-STAR, 1=Q1-GALAXY, 2=ERO-STAR, 3=ERO-GALAXY
    cat_q1 = np.where(y_q1_sub == 0, 0, 1)
    cat_ero = np.where(y_ero_sub == 0, 2, 3)
    categories = np.concatenate([cat_q1, cat_ero])

    print(f"  Fitting UMAP for Combined ({len(X_combined):,} sources)...")
    reducer_comb = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding_comb = reducer_comb.fit_transform(X_combined)

    emb_q1_comb = embedding_comb[:n_q1]
    emb_ero_comb = embedding_comb[n_q1:]

    # --- Plot: 1 row × 3 panels ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    label_names = {0: 'STAR', 1: 'GALAXY'}
    indiv_colors = {
        'Q1':  {0: 'coral',      1: 'steelblue'},
        'ERO': {0: 'orange',     1: 'mediumturquoise'},
    }

    def _style_ax(ax, title, legend_fontsize=10):
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(markerscale=3, fontsize=legend_fontsize, loc='upper right')
        ax.grid(alpha=0.3)

    # Panel 0: Q1 individual UMAP
    for label_val in [0, 1]:
        mask = embeddings[0][1] == label_val
        axes[0].scatter(embeddings[0][0][mask, 0], embeddings[0][0][mask, 1],
                        c=indiv_colors['Q1'][label_val],
                        label=f'{label_names[label_val]} (n={mask.sum():,})',
                        alpha=0.3, s=5)
    _style_ax(axes[0], 'Q1 — individual UMAP')

    # Panel 1: ERO individual UMAP
    for label_val in [0, 1]:
        mask = embeddings[1][1] == label_val
        axes[1].scatter(embeddings[1][0][mask, 0], embeddings[1][0][mask, 1],
                        c=indiv_colors['ERO'][label_val],
                        label=f'{label_names[label_val]} (n={mask.sum():,})',
                        alpha=0.3, s=5)
    _style_ax(axes[1], 'ERO — individual UMAP')

    # Panel 2: combined (4 categories, stars before galaxies)
    for cat_val, cat_label, color, marker in [
            (0, 'Q1 STAR',    'coral',           'o'),
            (2, 'ERO STAR',   'orange',           's'),
            (1, 'Q1 GALAXY',  'steelblue',        'o'),
            (3, 'ERO GALAXY', 'mediumturquoise',  's')]:
        mask = categories == cat_val
        axes[2].scatter(embedding_comb[mask, 0], embedding_comb[mask, 1],
                        c=color, label=f'{cat_label} (n={mask.sum():,})',
                        alpha=0.3, s=5, marker=marker)
    _style_ax(axes[2], 'Combined Q1 + ERO', legend_fontsize=9)

    plt.suptitle(f'Combined UMAP Visualization of Gaia Feature Space  (G ≥ {MAG_CUT},  {len(SELECTED_GAIA_PARAMETERS)} features)',
                 fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_umap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Return combined embedding and raw feature values for feature coloring plot
    X_combined_raw = np.vstack([X_q1_raw, X_ero_raw])
    return embedding_comb, X_combined_raw, categories


def plot_umap_features(embedding, X_raw, categories, output_dir):
    """Grid of UMAP panels coloured by individual Gaia parameter values.
    Each feature occupies two stacked panels: upper = stars, lower = galaxies.
    Layout: 4 feature rows × 4 feature cols; each feature slot has stacked
    stars/galaxies panels.
    Output: combined_umap_features.png
    """
    n_features = len(SELECTED_GAIA_PARAMETERS)
    ncols = 4
    nfeature_rows = int(np.ceil(n_features / ncols))

    # categories: 0=Q1-STAR, 1=Q1-GALAXY, 2=ERO-STAR, 3=ERO-GALAXY
    is_ero   = categories >= 2
    is_star  = (categories == 0) | (categories == 2)
    is_galaxy = (categories == 1) | (categories == 3)

    # GridSpec with two independently-controlled gaps:
    #   inner_gap  — between the stars / galaxies panels of the SAME feature
    #   outer_gap  — between DIFFERENT features (where the title sits)
    # Layout per feature row:  [stars | inner_gap | galaxies]  +  [outer_gap]  (except last)
    per_feature_width = 3.8
    per_panel_height  = 1.2   # inches per individual scatter panel
    panel_h   = 1.0           # relative height unit (keep at 1.0)
    inner_gap = 0.04          # relative — tight join between stars & galaxies
    outer_gap = 0.32          # relative — breathing room between features for title

    height_ratios = []
    for i in range(nfeature_rows):
        height_ratios.extend([panel_h, inner_gap, panel_h])
        if i < nfeature_rows - 1:
            height_ratios.append(outer_gap)

    # Scale fig_height so each panel row is exactly per_panel_height inches
    panel_fraction = panel_h / sum(height_ratios)
    fig_width  = ncols * per_feature_width
    fig_height = per_panel_height / panel_fraction

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = GridSpec(len(height_ratios), ncols,
                   figure=fig, height_ratios=height_ratios,
                   hspace=0, wspace=0.35)

    # gs row indices:  stars = feat_row*4,  inner_gap = feat_row*4+1,  galaxies = feat_row*4+2
    # (outer_gap occupies feat_row*4+3, skipped automatically)
    def _gs_top(feat_row): return feat_row * 4
    def _gs_bot(feat_row): return feat_row * 4 + 2

    for fi, feat_name in enumerate(SELECTED_GAIA_PARAMETERS):
        feat_row = fi // ncols
        col      = fi % ncols
        ax_top = fig.add_subplot(gs[_gs_top(feat_row), col])
        ax_bot = fig.add_subplot(gs[_gs_bot(feat_row), col])

        vals = X_raw[:, fi].astype(float)
        lo, hi = np.nanpercentile(vals, 2), np.nanpercentile(vals, 98)
        vals_clipped = np.clip(vals, lo, hi)

        for ax, class_mask, label in [
                (ax_top, is_star,   'Stars'),
                (ax_bot, is_galaxy, 'Galaxies')]:
            # ERO squares first, Q1 circles on top
            for ero_flag, marker in [(True, 's'), (False, 'o')]:
                mask = class_mask & (is_ero == ero_flag)
                sc = ax.scatter(
                    embedding[mask, 0], embedding[mask, 1],
                    c=vals_clipped[mask], cmap='RdYlBu_r',
                    vmin=lo, vmax=hi,
                    s=4, alpha=0.6, marker=marker, rasterized=True,
                    linewidths=0,
                )
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(alpha=0.2)
            n = class_mask.sum()
            ax.set_ylabel(f'{label}\n(n={n:,})', fontsize=7)

        # Shared colorbar spanning both panels
        cb = fig.colorbar(sc, ax=[ax_top, ax_bot], pad=0.02, fraction=0.046)
        cb.ax.tick_params(labelsize=7)

        # Feature name above each top subplot.
        ax_top.set_title(feat_name, fontsize=8, fontweight='bold', pad=4)

    # Hide unused panel pairs
    for fi in range(n_features, nfeature_rows * ncols):
        feat_row = fi // ncols
        col      = fi % ncols
        fig.add_subplot(gs[_gs_top(feat_row), col]).set_visible(False)
        fig.add_subplot(gs[_gs_bot(feat_row), col]).set_visible(False)

    fig.suptitle(
        f'Combined UMAP coloured by Gaia parameters  (G ≥ {MAG_CUT},  {len(SELECTED_GAIA_PARAMETERS)} features)  |  Upper = stars, Lower = galaxies',
        fontsize=12, fontweight='bold', y=0.93
    )
    out = output_dir / 'combined_umap_features.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> combined_umap_features.png saved")


def plot_umap_features_density(embedding, X_raw, categories, output_dir, n_bins=60):
    """Same figure geometry as plot_umap_features (identical GridSpec, figure size,
    column widths, outer gaps) but each feature slot is a single pcolormesh panel
    spanning the full height of what was the stars+galaxies pair.
    Colour = mean feature value per 2D UMAP bin, all sources combined.
    Output: combined_umap_features_density.png
    """
    n_features = len(SELECTED_GAIA_PARAMETERS)
    ncols = 4
    nfeature_rows = int(np.ceil(n_features / ncols))

    x_all, y_all = embedding[:, 0], embedding[:, 1]
    x_edges = np.linspace(np.nanpercentile(x_all, 0.5), np.nanpercentile(x_all, 99.5), n_bins + 1)
    y_edges = np.linspace(np.nanpercentile(y_all, 0.5), np.nanpercentile(y_all, 99.5), n_bins + 1)

    # Shared bin indices and total source count — determines which bins are "populated"
    # across ALL panels (same populated area regardless of per-feature NaN rates)
    xi_all = np.clip(np.digitize(x_all, x_edges) - 1, 0, n_bins - 1)
    yi_all = np.clip(np.digitize(y_all, y_edges) - 1, 0, n_bins - 1)
    H_count_all = np.zeros((n_bins, n_bins))
    np.add.at(H_count_all, (xi_all, yi_all), 1)
    # 1.0 for populated bins, NaN for empty — used as gray background in each panel
    H_bg = np.where(H_count_all > 0, 1.0, np.nan)

    def _mean_in_bins(feat_vals):
        H_sum   = np.zeros((n_bins, n_bins))
        H_count = np.zeros((n_bins, n_bins))
        valid = np.isfinite(feat_vals)
        np.add.at(H_sum,   (xi_all[valid], yi_all[valid]), feat_vals[valid])
        np.add.at(H_count, (xi_all[valid], yi_all[valid]), 1)
        return np.where(H_count > 0, H_sum / H_count, np.nan)

    # ── Identical GridSpec geometry to plot_umap_features ──────────────────────
    per_feature_width = 3.8
    per_panel_height  = 1.2
    panel_h   = 1.0
    inner_gap = 0.04
    outer_gap = 0.32

    height_ratios = []
    for i in range(nfeature_rows):
        height_ratios.extend([panel_h, inner_gap, panel_h])
        if i < nfeature_rows - 1:
            height_ratios.append(outer_gap)

    panel_fraction = panel_h / sum(height_ratios)
    fig_width  = ncols * per_feature_width
    fig_height = per_panel_height / panel_fraction

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = GridSpec(len(height_ratios), ncols,
                   figure=fig, height_ratios=height_ratios,
                   hspace=0, wspace=0.35)

    # Each feature spans rows [feat_row*4 : feat_row*4+3] (top panel + inner_gap + bottom panel)
    def _gs_span(feat_row): return slice(feat_row * 4, feat_row * 4 + 3)

    n_total = len(x_all)

    for fi, feat_name in enumerate(SELECTED_GAIA_PARAMETERS):
        feat_row = fi // ncols
        col      = fi % ncols
        ax = fig.add_subplot(gs[_gs_span(feat_row), col])

        vals = X_raw[:, fi].astype(float)

        # Gray background for all populated bins (identical across all panels)
        cmap_bg = plt.cm.Greys.copy()
        cmap_bg.set_bad(alpha=0)   # empty bins → transparent (white ax background)
        ax.pcolormesh(x_edges, y_edges, H_bg.T,
                      cmap=cmap_bg, vmin=0.5, vmax=2.0,
                      rasterized=True)

        # Feature mean values on top; NaN → transparent so gray shows through
        H  = _mean_in_bins(vals)
        # Compute colorscale from the actual per-bin means in UMAP space,
        # not from the raw feature distribution, so every panel uses its
        # full dynamic range and panels look visually comparable.
        populated_vals = H[np.isfinite(H)]
        if len(populated_vals) > 0:
            lo, hi = np.nanpercentile(populated_vals, 2), np.nanpercentile(populated_vals, 98)
        else:
            lo, hi = 0, 1
        cmap_feat = plt.cm.Blues.copy()
        cmap_feat.set_bad(alpha=0)
        sc = ax.pcolormesh(x_edges, y_edges, H.T,
                           cmap=cmap_feat, vmin=lo, vmax=hi,
                           rasterized=True)
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cb.ax.tick_params(labelsize=7)

        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(alpha=0.2)
        ax.set_title(feat_name, fontsize=8, fontweight='bold', pad=4)

    # Hide unused slots
    for fi in range(n_features, nfeature_rows * ncols):
        feat_row = fi // ncols
        col      = fi % ncols
        fig.add_subplot(gs[_gs_span(feat_row), col]).set_visible(False)

    fig.suptitle(
        f'Combined UMAP — mean feature value per 2D bin  (G ≥ {MAG_CUT},  {n_total:,} sources,  {n_features} features)',
        fontsize=12, fontweight='bold', y=0.93
    )
    out = output_dir / 'combined_umap_features_density.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> combined_umap_features_density.png saved")


def plot_umap_corner_compare(embedding, categories, output_dir, max_points_per_group=15000):
    """Four-panel row: Q1 and ERO shown separately for stars and galaxies."""
    rng = np.random.RandomState(42)

    def _sample_mask(cat_value):
        idx = np.where(categories == cat_value)[0]
        if len(idx) > max_points_per_group:
            idx = rng.choice(idx, max_points_per_group, replace=False)
        mask = np.zeros(len(categories), dtype=bool)
        mask[idx] = True
        return mask

    specs = [
        ("Q1 Stars", 0, "coral", "o"),
        ("ERO Stars", 2, "orange", "s"),
        ("Q1 Galaxies", 1, "steelblue", "o"),
        ("ERO Galaxies", 3, "mediumturquoise", "s"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)

    for ax, (title, cat_value, color, marker) in zip(axes, specs):
        mask = _sample_mask(cat_value)
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=color, s=6, alpha=0.20, linewidths=0, marker=marker,
            label=f'n={mask.sum():,}'
        )
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=10)
        ax.set_ylabel('UMAP 2', fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc='upper right')

    fig.suptitle(
        f'Extracted views from the combined UMAP map: class-wise comparison (G >= {MAG_CUT},  {len(SELECTED_GAIA_PARAMETERS)} features)',
        fontsize=13, fontweight='bold'
    )
    out = output_dir / 'combined_umap_corner_compare.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  -> combined_umap_corner_compare.png saved")


def main():
    parser = argparse.ArgumentParser(
        description='UMAP visualization of Gaia feature space'
    )
    parser.add_argument('--test-mode', action='store_true',
                        help='Use small sample for quick testing (10k sources)')
    args = parser.parse_args()

    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UMAP Visualization of Gaia Feature Space")
    print("=" * 60)
    print(f"  Features: {len(SELECTED_GAIA_PARAMETERS)}")
    print(f"  Magnitude cut: Gaia G >= {MAG_CUT}")
    print(f"  Output: {OUTPUT_DIR}")
    if args.test_mode:
        print(f"  TEST MODE: limited to 10k sources per dataset")

    print("\n[Step 1] Querying datasets...")
    X_q1, y_q1 = query_dataset(username, 'q1', test_mode=args.test_mode)
    X_ero, y_ero = query_dataset(username, 'ero', test_mode=args.test_mode)

    print("\n[Step 2] Generating UMAP comparison...")
    embedding_comb, X_raw_comb, categories = plot_umap_comparison(
        X_q1, y_q1, X_ero, y_ero, OUTPUT_DIR)

    print("\n[Step 3] Generating 4-panel class comparison plot...")
    plot_umap_corner_compare(embedding_comb, categories, OUTPUT_DIR)

    print("\n[Step 4] Generating UMAP coloured by Gaia features (scatter)...")
    plot_umap_features(embedding_comb, X_raw_comb, categories, OUTPUT_DIR)

    print("\n[Step 5] Generating UMAP feature density (2D histogram)...")
    plot_umap_features_density(embedding_comb, X_raw_comb, categories, OUTPUT_DIR)

    print("\n  -> combined_umap.png + combined_umap_corner_compare.png + combined_umap_features.png + combined_umap_features_density.png saved")
    print("=" * 60)
    print("UMAP visualization complete!")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
