#!/usr/bin/env python3
"""
Combined 2-row UMAP figure:
  Row 1: 2D count histograms  (All | Q1 Stars | Q1 Galaxies | ERO Stars | ERO Galaxies)
  Row 2: DESI overlay + Q1-ERO agreement scatter
         (empty | Q1 Stars | Q1 Galaxies | ERO Stars | ERO Galaxies)

Output: output/umap/combined_umap_2d_hist.png

Usage:
    PGUSER=siwei_zhao_2025 python src/plot_umap_2d_hist.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'output' / 'umap'
CSV_DIR     = PROJECT_DIR / 'output' / 'q1_to_ero'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'
MAG_CUT   = 17

SPECTYPE_STYLE = {
    'GALAXY': dict(color='#2196F3', marker='o', label='DESI GALAXY', zorder=5),
    'STAR':   dict(color='#F44336', marker='^', label='DESI STAR',   zorder=5),
    'QSO':    dict(color='#FF9800', marker='s', label='DESI QSO',    zorder=5),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _connect(username):
    import psycopg2
    return psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)


def _prep_features(df):
    X = df[SELECTED_GAIA_PARAMETERS].copy().replace('NOT_AVAILABLE', np.nan)
    for c in SELECTED_GAIA_PARAMETERS:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.values.astype(float)
    for j in range(X.shape[1]):
        bad = np.isnan(X[:, j])
        if bad.any():
            X[bad, j] = np.nanmedian(X[:, j])
    return X


def _standardize(X, mean_=None, std_=None):
    if mean_ is None:
        mean_ = np.nanmean(X, axis=0)
        std_  = np.nanstd(X,  axis=0)
    std_safe = np.where(std_ > 0, std_, 1.0)
    return (X - mean_) / std_safe, mean_, std_


def _project(df, reducer, mean_, std_):
    X = _prep_features(df)
    X_scaled, _, _ = _standardize(X, mean_, std_)
    print(f"  Projecting {len(X_scaled):,} sources...")
    return reducer.transform(X_scaled)


# ── queries ───────────────────────────────────────────────────────────────────

def query_base(username):
    gaia_cols = ', '.join([f'g_{c} AS {c}' for c in SELECTED_GAIA_PARAMETERS])
    q1_sql = f"""
        SELECT {gaia_cols},
               CASE WHEN p_phz_classification = 1 THEN 0 ELSE 1 END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE p_phz_classification IN (1, 2)
          AND g_phot_g_mean_mag >= {MAG_CUT}
        ORDER BY g_source_id
    """
    ero_sql = f"""
        SELECT {gaia_cols},
               CASE WHEN e_v_class_star >= 0.65 THEN 0 ELSE 1 END AS label
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_class_star IS NOT NULL
          AND g_phot_g_mean_mag >= {MAG_CUT}
        ORDER BY g_source_id
    """
    conn = _connect(username)
    print("  Querying Q1..."); df_q1 = pd.read_sql_query(q1_sql, conn)
    print(f"  -> {len(df_q1):,}")
    print("  Querying ERO..."); df_ero = pd.read_sql_query(ero_sql, conn)
    print(f"  -> {len(df_ero):,}")
    conn.close()
    return df_q1, df_ero


def query_desi_q1(username):
    gaia_cols = ', '.join([f'g_{c} AS {c}' for c in SELECTED_GAIA_PARAMETERS])
    sql = f"""
        SELECT g_source_id, p_phz_classification, d_desi_spectype, {gaia_cols}
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia_desi
        WHERE d_desi_spectype IN ('GALAXY','STAR','QSO')
          AND d_desi_z IS NOT NULL
          AND g_phot_g_mean_mag >= {MAG_CUT}
          AND p_phz_classification IN (1, 2)
    """
    conn = _connect(username)
    df = pd.read_sql_query(sql, conn); conn.close()
    print(f"  Q1×DESI: {len(df):,}  {df['d_desi_spectype'].value_counts().to_dict()}")
    return df


def query_desi_ero(username):
    gaia_cols = ', '.join([f'g_{c} AS {c}' for c in SELECTED_GAIA_PARAMETERS])
    sql = f"""
        SELECT g_source_id, e_v_class_star, d_desi_spectype, {gaia_cols}
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3_desi
        WHERE d_desi_spectype IN ('GALAXY','STAR','QSO')
          AND d_desi_z IS NOT NULL
          AND g_phot_g_mean_mag >= {MAG_CUT}
          AND e_v_class_star IS NOT NULL
    """
    conn = _connect(username)
    df = pd.read_sql_query(sql, conn); conn.close()
    print(f"  ERO×DESI: {len(df):,}  {df['d_desi_spectype'].value_counts().to_dict()}")
    return df


def load_agreement_features(username):
    groups = {
        'agree_star':   CSV_DIR / 'agree_both_star.csv',
        'agree_galaxy': CSV_DIR / 'agree_both_galaxy.csv',
        'disagree_sg':  CSV_DIR / 'disagree_ero_star_q1_gal.csv',
        'disagree_gs':  CSV_DIR / 'disagree_ero_gal_q1_star.csv',
    }
    dfs = []
    for group_name, path in groups.items():
        df = pd.read_csv(path, usecols=['g_source_id'])
        df['agreement_class'] = group_name
        dfs.append(df)
    df_labels = pd.concat(dfs, ignore_index=True)
    ids = ','.join(str(i) for i in df_labels['g_source_id'].tolist())
    gaia_cols = ', '.join([f'g_{c} AS {c}' for c in SELECTED_GAIA_PARAMETERS])
    sql = f"""
        SELECT g_source_id, {gaia_cols}
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE g_source_id IN ({ids})
    """
    conn = _connect(username)
    print(f"  Fetching features for {len(df_labels):,} agreement sources...")
    df_feats = pd.read_sql_query(sql, conn); conn.close()
    merged = df_labels.merge(df_feats, on='g_source_id', how='inner')
    print(f"  Agreement: {merged['agreement_class'].value_counts().to_dict()}")
    return merged


# ── UMAP fit ──────────────────────────────────────────────────────────────────

def fit_umap(df_q1, df_ero, sample_size=50000):
    rng = np.random.RandomState(42)
    half = sample_size // 2

    X_q1_raw = _prep_features(df_q1)
    X_ero_raw = _prep_features(df_ero)
    idx_q1 = rng.choice(len(X_q1_raw), min(half, len(X_q1_raw)), replace=False)
    idx_ero = rng.choice(len(X_ero_raw), min(half, len(X_ero_raw)), replace=False)

    X_combined_raw = np.vstack([X_q1_raw[idx_q1], X_ero_raw[idx_ero]])
    X_combined, mean_, std_ = _standardize(X_combined_raw)

    cat_q1  = np.where(df_q1['label'].values[idx_q1]  == 0, 0, 1)
    cat_ero = np.where(df_ero['label'].values[idx_ero] == 0, 2, 3)
    categories = np.concatenate([cat_q1, cat_ero])

    print(f"  Fitting UMAP on {len(X_combined):,} sources...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_combined)
    return embedding, categories, reducer, mean_, std_


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: set PGUSER"); return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("[1] Querying data...")
    df_q1, df_ero = query_base(username)
    print("  Q1×DESI:"); df_desi_q1  = query_desi_q1(username)
    print("  ERO×DESI:"); df_desi_ero = query_desi_ero(username)
    df_agreement = load_agreement_features(username)

    print("\n[2] Fitting UMAP...")
    embedding, categories, reducer, mean_, std_ = fit_umap(df_q1, df_ero)

    print("\n[3] Projecting DESI + agreement sources...")
    emb_desi_q1   = _project(df_desi_q1,   reducer, mean_, std_)
    emb_desi_ero  = _project(df_desi_ero,  reducer, mean_, std_)
    emb_agreement = _project(df_agreement, reducer, mean_, std_)

    print("\n[4] Plotting...")

    x_all, y_all = embedding[:, 0], embedding[:, 1]
    n_bins  = 120
    x_edges = np.linspace(np.nanpercentile(x_all, 0.5), np.nanpercentile(x_all, 99.5), n_bins + 1)
    y_edges = np.linspace(np.nanpercentile(y_all, 0.5), np.nanpercentile(y_all, 99.5), n_bins + 1)

    # Row 1: 2D histograms
    hist_specs = [
        ('All sources', None, 'Blues'),
        ('Q1 Stars',    0,    'Oranges'),
        ('Q1 Galaxies', 1,    'Blues'),
        ('ERO Stars',   2,    'Oranges'),
        ('ERO Galaxies',3,    'Blues'),
    ]
    histograms = []
    for title, cat_val, cmap_name in hist_specs:
        mask = np.ones(len(x_all), dtype=bool) if cat_val is None else (categories == cat_val)
        H, _, _ = np.histogram2d(x_all[mask], y_all[mask], bins=[x_edges, y_edges])
        histograms.append((title, cat_val, cmap_name, mask, H))

    class_vmax  = max(H[H > 0].max() for _, cv, _, _, H in histograms if cv is not None)
    shared_norm = LogNorm(vmin=1, vmax=class_vmax)

    # Row 2: scatter panels (same column order as row 1, skip col 0)
    scatter_specs = [
        ('Q1 Stars',     0, 'coral',          'o'),
        ('Q1 Galaxies',  1, 'steelblue',      'o'),
        ('ERO Stars',    2, 'orange',          's'),
        ('ERO Galaxies', 3, 'mediumturquoise', 's'),
    ]
    q1_phz   = df_desi_q1['p_phz_classification'].values
    ero_cs   = df_desi_ero['e_v_class_star'].values
    ag_class = df_agreement['agreement_class'].values

    desi_panel = {
        0: (df_desi_q1,  emb_desi_q1,  q1_phz == 1),
        1: (df_desi_q1,  emb_desi_q1,  q1_phz == 2),
        2: (df_desi_ero, emb_desi_ero, ero_cs >= 0.65),
        3: (df_desi_ero, emb_desi_ero, ero_cs <  0.65),
    }
    agreement_overlays = {
        2: [('#4CAF50', 'o', 'Q1 agree star',     ag_class == 'agree_star'),
            ('#F44336', '^', 'Q1 disagree (gal)',  ag_class == 'disagree_sg')],
        3: [('#4CAF50', 'o', 'Q1 agree galaxy',   ag_class == 'agree_galaxy'),
            ('#9C27B0', '^', 'Q1 disagree (star)', ag_class == 'disagree_gs')],
    }

    rng2    = np.random.RandomState(42)
    max_pts = 15000

    fig, axes = plt.subplots(2, 5, figsize=(34, 12), constrained_layout=True)

    # ── Row 1 ─────────────────────────────────────────────────────────────────
    for ax, (title, cat_val, cmap_name, mask, H) in zip(axes[0], histograms):
        H_plot = np.where(H > 0, H, np.nan)
        norm = LogNorm() if cat_val is None else shared_norm
        sc = ax.pcolormesh(x_edges, y_edges, H_plot.T, cmap=cmap_name, norm=norm, rasterized=True)
        fig.colorbar(sc, ax=ax, label='Count (log)')
        ax.set_title(f'{title}\n(n={mask.sum():,})', fontsize=10, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=10)
        ax.set_ylabel('UMAP 2', fontsize=10)
        ax.grid(alpha=0.2)

    # ── Row 2, col 0: empty ───────────────────────────────────────────────────
    axes[1, 0].set_visible(False)

    # ── Row 2, cols 1-4 ───────────────────────────────────────────────────────
    for ax, (title, cat_val, color, marker) in zip(axes[1, 1:], scatter_specs):
        idx = np.where(categories == cat_val)[0]
        if len(idx) > max_pts:
            idx = rng2.choice(idx, max_pts, replace=False)
        ax.scatter(embedding[idx, 0], embedding[idx, 1],
                   c=color, s=6, alpha=0.20, linewidths=0, marker=marker,
                   label=f'n={len(idx):,}', zorder=2)

        df_p, emb_p, pmask = desi_panel[cat_val]
        for spectype, style in SPECTYPE_STYLE.items():
            m = pmask & (df_p['d_desi_spectype'].values == spectype)
            if m.sum() == 0:
                continue
            ax.scatter(emb_p[m, 0], emb_p[m, 1],
                       c=style['color'], marker=style['marker'],
                       s=30, alpha=0.80, linewidths=0.4, edgecolors='white',
                       label=f"{style['label']}  n={m.sum():,}", zorder=style['zorder'])

        if cat_val in agreement_overlays:
            for ag_color, ag_marker, ag_label, ag_mask in agreement_overlays[cat_val]:
                ag_idx = np.where(ag_mask)[0]
                if len(ag_idx) > max_pts:
                    ag_idx = rng2.choice(ag_idx, max_pts, replace=False)
                ax.scatter(emb_agreement[ag_idx, 0], emb_agreement[ag_idx, 1],
                           c=ag_color, marker=ag_marker, s=12, alpha=0.50,
                           linewidths=0, label=f'{ag_label}  n={ag_mask.sum():,}', zorder=4)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=10)
        ax.set_ylabel('UMAP 2', fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc='upper right', markerscale=1.5)

    fig.suptitle(
        f'Combined UMAP  (G ≥ {MAG_CUT},  {len(x_all):,} sources)\n'
        f'Row 1: 2D count histogram   |   Row 2: DESI overlay + Q1–ERO agreement',
        fontsize=13, fontweight='bold'
    )
    out = OUTPUT_DIR / 'combined_umap_2d_hist.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  -> Saved: {out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
