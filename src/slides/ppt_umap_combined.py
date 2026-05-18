#!/usr/bin/env python3
"""
Reproduce the old-style 3-panel combined UMAP plot.

Panels:
  1. Q1 Dataset      — individual UMAP fit, STAR=steelblue / GALAXY=coral
  2. ERO Dataset     — individual UMAP fit, same colours
  3. Combined (Q1 + ERO) — joint UMAP fit, 4 categories

Output: output/umap/combined_umap_recovered.png

Usage:
    PGUSER=<user> python src/temp/ppt_umap_combined.py [--test-mode]
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'docs' / 'slides' / 'figs'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'
MAG_CUT   = 17
CLASS_MAPPING = {'STAR': 0, 'GALAXY': 1}


def query_dataset(username, data_source, test_mode=False):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])

    if data_source == 'q1':
        query = f"""
            SELECT {gaia_cols},
                CASE WHEN p_phz_classification = 1 THEN 'STAR'
                     WHEN p_phz_classification = 2 THEN 'GALAXY' END AS label
            FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
            WHERE p_phz_classification IN (1, 2)
              AND g_phot_g_mean_mag >= {MAG_CUT}
            ORDER BY g_source_id
        """
    else:
        query = f"""
            SELECT {gaia_cols},
                CASE WHEN e_v_class_star >= 0.65 THEN 'STAR'
                     ELSE 'GALAXY' END AS label
            FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
            WHERE e_v_class_star IS NOT NULL
              AND g_phot_g_mean_mag >= {MAG_CUT}
            ORDER BY g_source_id
        """
    if test_mode:
        query += " LIMIT 10000"

    print(f"  Querying {data_source.upper()} ...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    X = df[SELECTED_GAIA_PARAMETERS].values
    y = df['label'].map(CLASS_MAPPING).values
    print(f"  -> {len(X):,}  STAR: {(y==0).sum():,}  GALAXY: {(y==1).sum():,}")
    return X, y


def _clean_for_umap(X, sample_size, rng):
    idx = rng.choice(len(X), min(sample_size, len(X)), replace=False)
    X_sub = X[idx].copy()
    for c in range(X_sub.shape[1]):
        col = X_sub[:, c]
        mask = np.isnan(col)
        if mask.any():
            col[mask] = np.nanmedian(col)
    X_scaled = X_sub.copy()
    for c in range(X_scaled.shape[1]):
        std = np.nanstd(X_scaled[:, c])
        if std > 0:
            X_scaled[:, c] = (X_scaled[:, c] - np.nanmean(X_scaled[:, c])) / std
    return X_scaled, idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-mode', action='store_true')
    args = parser.parse_args()

    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: set PGUSER"); return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)
    sample_size = 50000

    print("[1] Querying datasets ...")
    X_q1, y_q1 = query_dataset(username, 'q1',  test_mode=args.test_mode)
    X_ero, y_ero = query_dataset(username, 'ero', test_mode=args.test_mode)

    # ── Individual UMAP fits ────────────────────────────────────────────────
    print("[2] Fitting individual UMAPs ...")
    embeddings = []
    for X, y, name in [(X_q1, y_q1, 'Q1'), (X_ero, y_ero, 'ERO')]:
        X_clean, idx = _clean_for_umap(X, sample_size, rng)
        y_sub = y[idx]
        print(f"    {name}: {len(X_clean):,} sources")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        emb = reducer.fit_transform(X_clean)
        embeddings.append((emb, y_sub, name))

    # ── Combined UMAP fit ───────────────────────────────────────────────────
    print("[3] Fitting combined UMAP ...")
    half = sample_size // 2
    X_q1_clean,  idx_q1  = _clean_for_umap(X_q1,  half, rng)
    X_ero_clean, idx_ero = _clean_for_umap(X_ero, half, rng)
    y_q1_sub  = y_q1[idx_q1]
    y_ero_sub = y_ero[idx_ero]
    n_q1 = len(X_q1_clean)

    X_comb = np.vstack([X_q1_clean, X_ero_clean])
    # categories: 0=Q1-STAR, 1=Q1-GALAXY, 2=ERO-STAR, 3=ERO-GALAXY
    cat_q1  = np.where(y_q1_sub  == 0, 0, 1)
    cat_ero = np.where(y_ero_sub == 0, 2, 3)
    categories = np.concatenate([cat_q1, cat_ero])

    print(f"    Combined: {len(X_comb):,} sources")
    reducer_comb = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    emb_comb = reducer_comb.fit_transform(X_comb)

    # ── Plot ────────────────────────────────────────────────────────────────
    print("[4] Plotting ...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    indiv_colors = {0: '#1976D2', 1: '#E53935'}   # STAR, GALAXY
    label_names  = {0: 'STAR',      1: 'GALAXY'}

    def _style_ax(ax, title, legend_fontsize=10):
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(markerscale=3, fontsize=legend_fontsize, loc='upper right')
        ax.grid(alpha=0.3)

    # Panel 0 — Q1 Dataset  (GALAXY first = underneath, STAR last = on top)
    for lv in [1, 0]:
        mask = embeddings[0][1] == lv
        axes[0].scatter(embeddings[0][0][mask, 0], embeddings[0][0][mask, 1],
                        c=indiv_colors[lv],
                        label=f'{label_names[lv]} (n={mask.sum():,})',
                        alpha=0.3, s=5)
    _style_ax(axes[0], 'Q1 Dataset')

    # Panel 1 — ERO Dataset
    for lv in [1, 0]:
        mask = embeddings[1][1] == lv
        axes[1].scatter(embeddings[1][0][mask, 0], embeddings[1][0][mask, 1],
                        c=indiv_colors[lv],
                        label=f'{label_names[lv]} (n={mask.sum():,})',
                        alpha=0.3, s=5)
    _style_ax(axes[1], 'ERO Dataset')

    # Panel 2 — Combined (Q1 + ERO): galaxies first, then stars on top
    for cat_val, cat_label, color, marker in [
            (1, 'Q1 GALAXY',  '#E53935', 'o'),
            (3, 'ERO GALAXY', '#FF7043', 's'),
            (0, 'Q1 STAR',    '#1976D2', 'o'),
            (2, 'ERO STAR',   '#42A5F5', 's')]:
        mask = categories == cat_val
        axes[2].scatter(emb_comb[mask, 0], emb_comb[mask, 1],
                        c=color, label=f'{cat_label} ({mask.sum():,})',
                        alpha=0.3, s=5, marker=marker)
    _style_ax(axes[2], 'Combined (Q1 + ERO)', legend_fontsize=9)

    plt.suptitle('UMAP Visualization of Gaia Feature Space',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    out = OUTPUT_DIR / 'combined_umap_recovered.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> saved: {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
