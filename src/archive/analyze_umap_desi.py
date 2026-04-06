#!/usr/bin/env python3
"""
Reproduce combined_umap_corner_compare.png with DESI-matched sources overlaid.

Same 4-panel layout (Q1 Stars | ERO Stars | Q1 Galaxies | ERO Galaxies) using
a combined Q1+ERO UMAP fit.  DESI-confirmed sources are projected into the
same UMAP space via transform() and overlaid coloured by DESI spectype
(GALAXY / STAR / QSO).

Output: output/umap/combined_umap_corner_compare_desi.png

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_umap_desi.py [--test-mode]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'output'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

MAG_CUT = 17

SPECTYPE_STYLE = {
    'GALAXY': dict(color='#2196F3', marker='o', label='DESI GALAXY', zorder=5),
    'STAR':   dict(color='#F44336', marker='^', label='DESI STAR',   zorder=5),
    'QSO':    dict(color='#FF9800', marker='s', label='DESI QSO',    zorder=5),
}


# ── Queries ────────────────────────────────────────────────────────────────────

def query_q1(username, test_mode=False):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    sql = f"""
        SELECT {gaia_cols},
               CASE WHEN p_phz_classification = 1 THEN 0
                    WHEN p_phz_classification = 2 THEN 1
               END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE p_phz_classification IN (1, 2)
          AND g_phot_g_mean_mag >= {MAG_CUT}
        ORDER BY g_source_id
    """
    if test_mode:
        sql += " LIMIT 10000"
    print("  Querying Q1...")
    df = pd.read_sql_query(sql, conn)
    conn.close()
    print(f"  -> {len(df):,}  STAR:{(df['label']==0).sum():,}  GALAXY:{(df['label']==1).sum():,}")
    return df


def query_ero(username, test_mode=False):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    sql = f"""
        SELECT g_source_id,
               {gaia_cols},
               CASE WHEN e_v_class_star >= 0.65 THEN 0
                    ELSE 1
               END AS label
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_class_star IS NOT NULL
          AND g_phot_g_mean_mag >= {MAG_CUT}
        ORDER BY g_source_id
    """
    if test_mode:
        sql += " LIMIT 10000"
    print("  Querying ERO...")
    df = pd.read_sql_query(sql, conn)
    conn.close()
    print(f"  -> {len(df):,}  STAR:{(df['label']==0).sum():,}  GALAXY:{(df['label']==1).sum():,}")
    return df


def query_desi_ero(username, test_mode=False):
    """Query ERO×DESI crossmatch (crossmatch_ero_gaia_0_3_desi).
    Panel assignment: e_v_class_star >= 0.65 → ERO Stars, else → ERO Galaxies.
    Sum across both ERO panels = 450 DESI galaxies.
    """
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    sql = f"""
        SELECT g_source_id,
               e_v_class_star,
               d_desi_spectype,
               d_desi_z,
               {gaia_cols}
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3_desi
        WHERE d_desi_spectype IN ('GALAXY', 'STAR', 'QSO')
          AND d_desi_z IS NOT NULL
          AND g_phot_g_mean_mag >= {MAG_CUT}
          AND e_v_class_star IS NOT NULL
    """
    if test_mode:
        sql += " LIMIT 2000"
    print("  Querying ERO×DESI crossmatch...")
    df = pd.read_sql_query(sql, conn)
    conn.close()
    vc = df['d_desi_spectype'].value_counts()
    print(f"  -> {len(df):,}  " + "  ".join(f"{k}:{v}" for k, v in vc.items()))
    return df


def query_desi_q1(username, test_mode=False):
    """Query Q1×DESI crossmatch (crossmatch_q1_mer_phz_gaia_desi).
    Panel assignment: p_phz_classification=1 → Q1 Stars, =2 → Q1 Galaxies.
    Sum across both Q1 panels = 3,395 DESI galaxies.
    """
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    sql = f"""
        SELECT g_source_id,
               p_phz_classification,
               d_desi_spectype,
               d_desi_z,
               {gaia_cols}
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia_desi
        WHERE d_desi_spectype IN ('GALAXY', 'STAR', 'QSO')
          AND d_desi_z IS NOT NULL
          AND g_phot_g_mean_mag >= {MAG_CUT}
          AND p_phz_classification IN (1, 2)
    """
    if test_mode:
        sql += " LIMIT 5000"
    print("  Querying Q1×DESI crossmatch...")
    df = pd.read_sql_query(sql, conn)
    conn.close()
    vc = df['d_desi_spectype'].value_counts()
    print(f"  -> {len(df):,}  " + "  ".join(f"{k}:{v}" for k, v in vc.items()))
    return df


def load_agreement_features(username, test_mode=False):
    """Load agree/disagree CSVs, fetch Gaia features by source_id, return combined df."""
    import psycopg2
    csv_dir = Path(__file__).parent.parent / 'output' / 'q1_to_ero'
    groups = {
        'agree_star':   csv_dir / 'agree_both_star.csv',
        'agree_galaxy': csv_dir / 'agree_both_galaxy.csv',
        'disagree_sg':  csv_dir / 'disagree_ero_star_q1_gal.csv',
        'disagree_gs':  csv_dir / 'disagree_ero_gal_q1_star.csv',
    }
    dfs = []
    for group_name, path in groups.items():
        df = pd.read_csv(path, usecols=['g_source_id'])
        df['agreement_class'] = group_name
        dfs.append(df)
    df_labels = pd.concat(dfs, ignore_index=True)
    if test_mode:
        df_labels = df_labels.groupby('agreement_class', group_keys=False).apply(
            lambda x: x.sample(min(500, len(x)), random_state=42))

    ids = df_labels['g_source_id'].tolist()
    id_str = ','.join(str(i) for i in ids)

    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    sql = f"""
        SELECT g_source_id, {gaia_cols}
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE g_source_id IN ({id_str})
    """
    print(f"  Fetching Gaia features for {len(ids):,} agreement sources...")
    df_feats = pd.read_sql_query(sql, conn)
    conn.close()

    merged = df_labels.merge(df_feats, on='g_source_id', how='inner')
    counts = merged['agreement_class'].value_counts()
    print(f"  Merged {len(merged):,} sources:")
    for k, v in counts.items():
        print(f"    {k}: {v:,}")
    return merged


def query_q1_ero_overlap(username, test_mode=False):
    """Sources matched in both Q1 and ERO (inner join on g_source_id).
    Returns a df with q1_label (0=star,1=galaxy) and ero_label (0=star,1=galaxy).
    """
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'q.g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    sql = f"""
        SELECT q.g_source_id,
               CASE WHEN q.p_phz_classification = 1 THEN 0 ELSE 1 END AS q1_label,
               CASE WHEN e.e_v_class_star >= 0.65   THEN 0 ELSE 1 END AS ero_label,
               {gaia_cols}
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia q
        JOIN siwei_zhao_2025.crossmatch_ero_gaia_0_3 e USING (g_source_id)
        WHERE q.p_phz_classification IN (1, 2)
          AND e.e_v_class_star IS NOT NULL
          AND q.g_phot_g_mean_mag >= {MAG_CUT}
    """
    if test_mode:
        sql += " LIMIT 5000"
    print("  Querying Q1∩ERO overlap...")
    df = pd.read_sql_query(sql, conn)
    conn.close()
    agree_s  = ((df['q1_label'] == 0) & (df['ero_label'] == 0)).sum()
    agree_g  = ((df['q1_label'] == 1) & (df['ero_label'] == 1)).sum()
    dis_sg   = ((df['q1_label'] == 0) & (df['ero_label'] == 1)).sum()
    dis_gs   = ((df['q1_label'] == 1) & (df['ero_label'] == 0)).sum()
    print(f"  -> {len(df):,} overlap sources")
    print(f"     Agree Star:{agree_s:,}  Agree Galaxy:{agree_g:,}  "
          f"Q1★/ERO🌌:{dis_sg:,}  Q1🌌/ERO★:{dis_gs:,}")
    return df


# ── Feature preparation ────────────────────────────────────────────────────────

def _prep_features(df, cols=SELECTED_GAIA_PARAMETERS):
    """Coerce to numeric, return float array."""
    X = df[cols].copy()
    X = X.replace('NOT_AVAILABLE', np.nan)
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    return X.values.astype(float)


def _fill_nan_median(X):
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        bad = np.isnan(col)
        if bad.any():
            col[bad] = np.nanmedian(col)
    return X


def _standardize(X, mean_=None, std_=None):
    X = X.copy()
    if mean_ is None:
        mean_ = np.nanmean(X, axis=0)
        std_  = np.nanstd(X,  axis=0)
    std_safe = np.where(std_ > 0, std_, 1.0)
    return (X - mean_) / std_safe, mean_, std_


# ── UMAP fit ───────────────────────────────────────────────────────────────────

def fit_combined_umap(df_q1, df_ero, sample_size=50000):
    """
    Subsample Q1 and ERO equally, fit a combined UMAP.
    Returns:
        embedding   — (N_q1_sub + N_ero_sub, 2)
        categories  — 0=Q1-STAR, 1=Q1-GALAXY, 2=ERO-STAR, 3=ERO-GALAXY
        reducer     — fitted umap.UMAP (for transform)
        scale_      — (mean_, std_) used for standardisation
        idx_q1, idx_ero  — subsample indices into original dfs
    """
    rng = np.random.RandomState(42)
    half = sample_size // 2

    X_q1_raw = _prep_features(df_q1)
    X_ero_raw = _prep_features(df_ero)

    idx_q1 = rng.choice(len(X_q1_raw), min(half, len(X_q1_raw)), replace=False)
    idx_ero = rng.choice(len(X_ero_raw), min(half, len(X_ero_raw)), replace=False)

    X_q1_sub = _fill_nan_median(X_q1_raw[idx_q1])
    X_ero_sub = _fill_nan_median(X_ero_raw[idx_ero])

    X_combined_raw = np.vstack([X_q1_sub, X_ero_sub])
    X_combined, mean_, std_ = _standardize(X_combined_raw)

    y_q1_sub  = df_q1['label'].values[idx_q1]
    y_ero_sub = df_ero['label'].values[idx_ero]
    cat_q1  = np.where(y_q1_sub  == 0, 0, 1)   # 0=Q1-STAR,  1=Q1-GALAXY
    cat_ero = np.where(y_ero_sub == 0, 2, 3)   # 2=ERO-STAR, 3=ERO-GALAXY
    categories = np.concatenate([cat_q1, cat_ero])

    print(f"  Fitting combined UMAP on {len(X_combined):,} sources...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_combined)

    return embedding, categories, reducer, (mean_, std_), idx_q1, idx_ero


def project_desi(df, reducer, scale_):
    """Project a DESI DataFrame into the fitted UMAP space."""
    mean_, std_ = scale_
    X_raw = _prep_features(df)
    X_filled = _fill_nan_median(X_raw)
    X_scaled, _, _ = _standardize(X_filled, mean_=mean_, std_=std_)
    print(f"  Projecting {len(X_scaled):,} sources via transform()...")
    return reducer.transform(X_scaled)


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_corner_compare_desi(embedding, categories,
                              df_desi_q1, emb_desi_q1,
                              df_desi_ero, emb_desi_ero,
                              df_agreement, emb_agreement,
                              output_dir, max_points=15000):
    """3-row × 5-col UMAP grid (Combined | Q1 Stars | ERO Stars | Q1 Gal | ERO Gal).

    Row 1: scatter + DESI overlay
    Row 2: Q1∩ERO agreement overlay — ALL ERO sources projected (ERO panels only)
    Row 3: 2D count histogram (Blues, log scale)
    """
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    rng = np.random.RandomState(42)
    ag_class = df_agreement['agreement_class'].values
    q1_phz   = df_desi_q1['p_phz_classification'].values
    ero_cs   = df_desi_ero['e_v_class_star'].values

    specs = [
        ("Combined",     None, None,          None),
        ("Q1 Stars",     0,    "coral",        "o"),
        ("ERO Stars",    2,    "orange",       "s"),
        ("Q1 Galaxies",  1,    "steelblue",    "o"),
        ("ERO Galaxies", 3,    "darkcyan",     "s"),
    ]
    cat_colors = {0: "coral", 1: "steelblue", 2: "orange", 3: "darkcyan"}

    # Shared square axis limits
    x_all, y_all = embedding[:, 0], embedding[:, 1]
    x_mid = (x_all.min() + x_all.max()) / 2
    y_mid = (y_all.min() + y_all.max()) / 2
    half  = max(x_all.max() - x_all.min(), y_all.max() - y_all.min()) / 2 * 1.03
    x_lo, x_hi = x_mid - half, x_mid + half
    y_lo, y_hi = y_mid - half, y_mid + half

    n_bins  = 150
    x_edges = np.linspace(x_lo, x_hi, n_bins + 1)
    y_edges = np.linspace(y_lo, y_hi, n_bins + 1)
    cmap_hist = plt.cm.Blues.copy()
    cmap_hist.set_bad(alpha=0)

    def _sample_mask(cat_value):
        idx = np.where(categories == cat_value)[0]
        if len(idx) > max_points:
            idx = rng.choice(idx, max_points, replace=False)
        m = np.zeros(len(categories), dtype=bool)
        m[idx] = True
        return m

    def _style(ax, title):
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('UMAP 1', fontsize=10)
        ax.set_ylabel('UMAP 2', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, loc='upper right', markerscale=1.5)

    fig, axes = plt.subplots(3, 5, figsize=(36, 18), constrained_layout=True)

    for col, (title, cat_val, color, marker) in enumerate(specs):
        ax_sc = axes[0, col]
        ax_ag = axes[1, col]
        ax_h  = axes[2, col]

        # ── Row 1: scatter + DESI overlay ────────────────────────────────────
        if cat_val is None:
            for cv, cc in cat_colors.items():
                m = _sample_mask(cv)
                ax_sc.scatter(embedding[m, 0], embedding[m, 1],
                              c=cc, s=4, alpha=0.15, linewidths=0)
        elif cat_val in (0, 1):
            # Q1 panels: use embedded sources
            mask = _sample_mask(cat_val)
            ax_sc.scatter(embedding[mask, 0], embedding[mask, 1],
                          c=color, s=6, alpha=0.25, linewidths=0, marker=marker,
                          label=f'n={mask.sum():,}')
            # DESI overlay
            df_desi, emb_desi = df_desi_q1, emb_desi_q1
            pmask = (q1_phz == 1) if cat_val == 0 else (q1_phz == 2)
            for spectype, style in SPECTYPE_STYLE.items():
                m = pmask & (df_desi['d_desi_spectype'].values == spectype)
                if m.sum() == 0:
                    continue
                ax_sc.scatter(emb_desi[m, 0], emb_desi[m, 1],
                              c=style['color'], marker=style['marker'],
                              s=30, alpha=0.80, linewidths=0.4, edgecolors='white',
                              label=f"{style['label']}  n={m.sum():,}",
                              zorder=style['zorder'])
        else:
            # ERO panels: use same projected agreement sources as Row 2 (same distribution)
            if cat_val == 2:
                ero_mask = (ag_class == 'agree_star') | (ag_class == 'disagree_sg')
            else:
                ero_mask = (ag_class == 'agree_galaxy') | (ag_class == 'disagree_gs')
            ero_idx = np.where(ero_mask)[0]
            if len(ero_idx) > max_points:
                ero_idx = rng.choice(ero_idx, max_points, replace=False)
            ax_sc.scatter(emb_agreement[ero_idx, 0], emb_agreement[ero_idx, 1],
                          c=color, s=6, alpha=0.25, linewidths=0, marker=marker,
                          label=f'n={ero_mask.sum():,}')
            # DESI overlay
            df_desi, emb_desi = df_desi_ero, emb_desi_ero
            pmask = (ero_cs >= 0.65) if cat_val == 2 else (ero_cs < 0.65)
            for spectype, style in SPECTYPE_STYLE.items():
                m = pmask & (df_desi['d_desi_spectype'].values == spectype)
                if m.sum() == 0:
                    continue
                ax_sc.scatter(emb_desi[m, 0], emb_desi[m, 1],
                              c=style['color'], marker=style['marker'],
                              s=30, alpha=0.80, linewidths=0.4, edgecolors='white',
                              label=f"{style['label']}  n={m.sum():,}",
                              zorder=style['zorder'])
        _style(ax_sc, title)

        # ── Row 2: agreement (ERO panels only, ALL sources projected) ─────────
        if cat_val in (2, 3):
            if cat_val == 2:
                agree_m    = ag_class == 'agree_star'
                disagree_m = ag_class == 'disagree_sg'
                lbl_ag, lbl_dis = 'Q1 agree (star)', 'Q1 disagree (gal)'
            else:
                agree_m    = ag_class == 'agree_galaxy'
                disagree_m = ag_class == 'disagree_gs'
                lbl_ag, lbl_dis = 'Q1 agree (gal)', 'Q1 disagree (star)'

            for ag_mask, ag_color, ag_lbl in [
                (agree_m,    'limegreen', lbl_ag),
                (disagree_m, '#E74C3C',   lbl_dis),
            ]:
                ag_idx = np.where(ag_mask)[0]
                if len(ag_idx) > max_points:
                    ag_idx = rng.choice(ag_idx, max_points, replace=False)
                if len(ag_idx) == 0:
                    continue
                ax_ag.scatter(emb_agreement[ag_idx, 0], emb_agreement[ag_idx, 1],
                              c=ag_color, s=6, alpha=0.20, linewidths=0, marker='s',
                              label=f'{ag_lbl}  n={ag_mask.sum():,}')
            _style(ax_ag, f'{title} — Q1 agreement')
        else:
            ax_ag.axis('off')

        # ── Row 3: 2D histogram ───────────────────────────────────────────────
        hmask = np.ones(len(x_all), dtype=bool) if cat_val is None else (categories == cat_val)
        H, _, _ = np.histogram2d(x_all[hmask], y_all[hmask], bins=[x_edges, y_edges])
        H = np.where(H > 0, H, np.nan)
        sc = ax_h.pcolormesh(x_edges, y_edges, H.T,
                             cmap=cmap_hist, norm=LogNorm(), rasterized=True)
        cax = inset_axes(ax_h, width='5%', height='100%', loc='lower left',
                         bbox_to_anchor=(1.02, 0, 1, 1),
                         bbox_transform=ax_h.transAxes, borderpad=0)
        fig.colorbar(sc, cax=cax, label='Count (log)')
        _style(ax_h, '')

    fig.suptitle(
        f'Combined UMAP — DESI overlay + Q1∩ERO agreement  (G ≥ {MAG_CUT},  {len(SELECTED_GAIA_PARAMETERS)} features)\n'
        f'Row 1: scatter (DESI overlay)   |   Row 2: Q1 agreement overlay (ERO panels only, all sources projected)'
        f'   |   Row 3: 2D count histogram (Blues, log scale)',
        fontsize=13, fontweight='bold'
    )
    out = output_dir / 'combined_umap_2d_hist.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-mode', action='store_true',
                        help='Limit queries to 10k sources for quick testing')
    args = parser.parse_args()

    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: set PGUSER"); return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("UMAP corner compare — DESI overlay")
    print("=" * 65)

    print("\n[1] Querying datasets...")
    df_q1       = query_q1(username,           test_mode=args.test_mode)
    df_ero      = query_ero(username,          test_mode=args.test_mode)
    df_desi_ero = query_desi_ero(username,     test_mode=args.test_mode)
    df_desi_q1  = query_desi_q1(username,      test_mode=args.test_mode)
    df_agreement = load_agreement_features(username, test_mode=args.test_mode)

    print("\n[2] Fitting combined UMAP...")
    embedding, categories, reducer, scale_, _, _ = fit_combined_umap(df_q1, df_ero)

    print("\n[3] Projecting DESI sources and Q1∩ERO overlap...")
    print("  ERO×DESI:")
    emb_desi_ero = project_desi(df_desi_ero, reducer, scale_)
    print("  Q1×DESI:")
    emb_desi_q1  = project_desi(df_desi_q1,  reducer, scale_)
    print("  Agreement sources:")
    emb_agreement = project_desi(df_agreement, reducer, scale_)

    # Save projections for use by plot_umap_2d_hist.py
    umap_dir = OUTPUT_DIR / 'umap'
    umap_dir.mkdir(parents=True, exist_ok=True)
    np.save(umap_dir / 'desi_q1_emb.npy',       emb_desi_q1)
    np.save(umap_dir / 'desi_q1_phz.npy',        df_desi_q1['p_phz_classification'].values)
    np.save(umap_dir / 'desi_q1_spectype.npy',   df_desi_q1['d_desi_spectype'].values.astype(str))
    np.save(umap_dir / 'desi_ero_emb.npy',       emb_desi_ero)
    np.save(umap_dir / 'desi_ero_cs.npy',        df_desi_ero['e_v_class_star'].values)
    np.save(umap_dir / 'desi_ero_spectype.npy',  df_desi_ero['d_desi_spectype'].values.astype(str))
    np.save(umap_dir / 'agreement_emb.npy',      emb_agreement)
    np.save(umap_dir / 'agreement_class.npy',    df_agreement['agreement_class'].values.astype(str))
    print(f"  Saved DESI projection arrays to {umap_dir}")

    print("\n[4] Plotting...")
    plot_corner_compare_desi(
        embedding, categories,
        df_desi_q1,   emb_desi_q1,
        df_desi_ero,  emb_desi_ero,
        df_agreement, emb_agreement,
        OUTPUT_DIR,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
