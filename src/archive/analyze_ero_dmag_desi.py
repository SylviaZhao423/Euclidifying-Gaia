#!/usr/bin/env python3
"""
Overlay DESI spectroscopic labels on the ERO Δmag distributions.

Same 4×2 layout as ero_dmag_vs_gmag_xgb.png, but each panel has the full
ERO background (grey density) plus DESI-confirmed sources coloured by
spectroscopic type (GALAXY / STAR / QSO).

Panels (columns: ERO VIS galaxy cs<0.65  |  ERO VIS star cs≥0.65):
  Row 0: Δmag vs Gaia G — grey density + DESI scatter coloured by spectype
  Row 1: Δmag vs Gaia G — grey density + DESI scatter coloured by DESI redshift
  Row 2: Gaia G magnitude histogram — background + DESI step histograms by spectype
  Row 3: Q1 XGBoost prob histogram  — background + DESI step histograms by spectype

Output: output/ero_dmag/ero_dmag_desi_overlay.png

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_ero_dmag_desi.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR        = Path(__file__).parent.parent
OUTPUT_DIR         = PROJECT_DIR / 'output'
MODEL_Q1_SEEDS_DIR = PROJECT_DIR / 'output' / 'xgboost_q1' / 'seed_models'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

SPECTYPE_STYLE = {
    'GALAXY': dict(color='#2196F3', marker='o', label='DESI GALAXY'),
    'STAR':   dict(color='#F44336', marker='^', label='DESI STAR'),
    'QSO':    dict(color='#FF9800', marker='s', label='DESI QSO'),
}


def query_ero(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'c.g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    sql = f"""
        SELECT
            e_v_mag_auto               AS v_mag_auto,
            e_v_class_star             AS class_star,
            c.g_corrected_bp_rp_excess AS c_star,
            c.g_phot_g_mean_mag        AS phot_g_mean_mag,
            {gaia_cols}
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3 c
        WHERE e_v_class_star IS NOT NULL
    """
    print("  Querying ERO crossmatch (background)...")
    df = pd.read_sql_query(sql, conn)
    conn.close()
    print(f"  -> {len(df):,} sources")
    return df


def query_desi(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    sql = f"""
        SELECT
            e_v_mag_auto               AS v_mag_auto,
            e_v_class_star             AS class_star,
            g_corrected_bp_rp_excess   AS c_star,
            g_phot_g_mean_mag          AS phot_g_mean_mag,
            d_desi_spectype,
            d_desi_z,
            d_desi_deltachi2,
            {gaia_cols}
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3_desi
        WHERE d_desi_z IS NOT NULL
          AND d_desi_spectype IN ('GALAXY', 'STAR', 'QSO')
    """
    print("  Querying ERO-DESI crossmatch (overlay)...")
    df = pd.read_sql_query(sql, conn)
    conn.close()
    print(f"  -> {len(df):,} sources  "
          + "  ".join(f"{k}:{v}" for k, v in df['d_desi_spectype'].value_counts().items()))
    return df


def apply_xgb(df):
    import xgboost as xgb
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    X = df[SELECTED_GAIA_PARAMETERS].values
    seed_paths = sorted(MODEL_Q1_SEEDS_DIR.glob('xgb_classifier_q1_seed*.json'))
    if not seed_paths:
        raise FileNotFoundError(f'No seed models in {MODEL_Q1_SEEDS_DIR}')
    probs = []
    for p in seed_paths:
        m = xgb.XGBClassifier()
        m.load_model(str(p))
        probs.append(m.predict_proba(X)[:, 1])
    df['xgb_q1_prob'] = np.mean(np.vstack(probs), axis=0)
    return df, len(seed_paths)


def make_plot(ero, desi, n_seeds, output_dir):
    # ── Prepare ──────────────────────────────────────────────────────────────
    for df in (ero, desi):
        df['mag_diff'] = df['v_mag_auto'] - df['phot_g_mean_mag']

    xbins     = np.arange(6,  23.1, 0.1)
    ybins     = np.arange(-6.5, 7.6, 0.1)
    mag_bins  = np.arange(14, 22.1, 0.2)
    prob_bins = np.arange(0, 1.01, 0.02)

    # For the star panel use P(star) = 1 - P(galaxy)
    desi['xgb_q1_prob_star'] = 1 - desi['xgb_q1_prob']
    ero['xgb_q1_prob_star']  = 1 - ero['xgb_q1_prob']

    cs_splits = [
        (ero[ero['class_star'] <  0.65],  desi[desi['class_star'] <  0.65],
         'VIS galaxy (cs<0.65)',  'P (galaxy | Q1)',  'xgb_q1_prob'),
        (ero[ero['class_star'] >= 0.65],  desi[desi['class_star'] >= 0.65],
         'VIS star (cs≥0.65)',    'P (star | Q1)',    'xgb_q1_prob_star'),
    ]

    fig, axes_arr = plt.subplots(3, 2, figsize=(15, 13))
    axes = [list(row) for row in axes_arr]
    fig.subplots_adjust(hspace=0.42, wspace=0.32)

    # Shared density colour scale
    vmaxes = []
    histograms = []
    for ero_sub, _, _, _, _ in cs_splits:
        ero_clean = ero_sub.dropna(subset=['phot_g_mean_mag', 'mag_diff'])
        h, xe, ye = np.histogram2d(ero_clean['phot_g_mean_mag'].values,
                                   ero_clean['mag_diff'].values, bins=[xbins, ybins])
        hm = np.ma.masked_where(h == 0, h)
        if hm.count() > 0:
            vmaxes.append(hm.max())
        histograms.append((h, xe, ye))
    global_vmax = max(max(vmaxes, default=10), 2)

    for ci, ((ero_sub, desi_sub, title, prob_label, prob_col),
             (h, xe, ye)) in enumerate(zip(cs_splits, histograms)):

        ero_clean = ero_sub.dropna(subset=['phot_g_mean_mag', 'mag_diff'])
        desi_clean = desi_sub.dropna(subset=['phot_g_mean_mag', 'mag_diff'])

        # ── Row 0: grey density + DESI scatter coloured by redshift ─────
        ax = axes[0][ci]
        hm = np.ma.masked_where(h == 0, h)
        ax.pcolormesh(xe, ye, hm.T, cmap='Greys', norm=LogNorm(vmin=1, vmax=global_vmax),
                      zorder=1)
        desi_z = desi_clean.dropna(subset=['d_desi_z'])
        if len(desi_z):
            sc = ax.scatter(desi_z['phot_g_mean_mag'], desi_z['mag_diff'],
                            c=desi_z['d_desi_z'], cmap='plasma',
                            vmin=0, vmax=1.5, s=18, alpha=0.85,
                            linewidths=0.3, edgecolors='white', zorder=3,
                            label=f'DESI GALAXY (n={len(desi_z):,})')
            cax = ax.inset_axes([1.02, 0, 0.03, 1])
            fig.colorbar(sc, cax=cax).set_label('DESI redshift z', fontsize=8, labelpad=8)
        ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.6, zorder=2)
        ax.set_xlim(6, 23); ax.set_ylim(-6.5, 5.5)
        ax.set_xlabel('Gaia G', fontsize=11)
        if ci == 0:
            ax.set_ylabel('Δmag  (VIS − Gaia G)', fontsize=11)
        ax.set_title(f'{title}  —  coloured by DESI redshift\nn={len(ero_clean):,} ERO  |  {len(desi_clean):,} DESI',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left', markerscale=1.5)
        ax.grid(alpha=0.3)

        # ── Row 1: grey density + DESI scatter coloured by C* ────────────
        ax = axes[1][ci]
        ax.pcolormesh(xe, ye, hm.T, cmap='Greys', norm=LogNorm(vmin=1, vmax=global_vmax),
                      zorder=1)
        desi_cs = desi_clean.dropna(subset=['c_star'])
        if len(desi_cs):
            sc = ax.scatter(desi_cs['phot_g_mean_mag'], desi_cs['mag_diff'],
                            c=desi_cs['c_star'], cmap='RdYlBu_r',
                            vmin=-1, vmax=1, s=18, alpha=0.85,
                            linewidths=0.3, edgecolors='white', zorder=3,
                            label=f'DESI GALAXY (n={len(desi_cs):,})')
            cax = ax.inset_axes([1.02, 0, 0.03, 1])
            fig.colorbar(sc, cax=cax).set_label('C* (corrected BP-RP excess)', fontsize=8, labelpad=8)
        ax.legend(fontsize=8, loc='upper left', markerscale=1.5)
        ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.6, zorder=2)
        ax.set_xlim(6, 23); ax.set_ylim(-6.5, 5.5)
        ax.set_xlabel('Gaia G', fontsize=11)
        if ci == 0:
            ax.set_ylabel('Δmag  (VIS − Gaia G)', fontsize=11)
        ax.set_title(f'{title}  —  coloured by C*', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)

        # ── Row 2: grey density + DESI scatter coloured by Δχ² ──────────
        ax = axes[2][ci]
        ax.pcolormesh(xe, ye, hm.T, cmap='Greys', norm=LogNorm(vmin=1, vmax=global_vmax),
                      zorder=1)
        desi_dc2 = desi_clean.dropna(subset=['d_desi_deltachi2'])
        if len(desi_dc2):
            dc2_vals = desi_dc2['d_desi_deltachi2'].clip(lower=1)
            vmax_dc2 = float(dc2_vals.quantile(0.99))
            sc = ax.scatter(desi_dc2['phot_g_mean_mag'], desi_dc2['mag_diff'],
                            c=dc2_vals, cmap='viridis',
                            norm=LogNorm(vmin=1, vmax=max(vmax_dc2, 2)),
                            s=18, alpha=0.85,
                            linewidths=0.3, edgecolors='white', zorder=3,
                            label=f'DESI GALAXY (n={len(desi_dc2):,})')
            cax = ax.inset_axes([1.02, 0, 0.03, 1])
            fig.colorbar(sc, cax=cax).set_label('DESI Δχ²', fontsize=8, labelpad=8)
        ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.6, zorder=2)
        ax.set_xlim(6, 23); ax.set_ylim(-6.5, 5.5)
        ax.set_xlabel('Gaia G', fontsize=11)
        if ci == 0:
            ax.set_ylabel('Δmag  (VIS − Gaia G)', fontsize=11)
        ax.set_title(f'{title}  —  coloured by DESI Δχ²',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left', markerscale=1.5)
        ax.grid(alpha=0.3)

    fig.suptitle(
        'ERO Δmag distributions with DESI spectroscopic labels overlaid',
        fontsize=13, fontweight='bold', y=0.98,
    )
    out = output_dir / 'ero_dmag_desi_overlay.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {out}")


def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: set PGUSER"); return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ERO Δmag — DESI label overlay")
    print("=" * 65)

    print("\n[1] Querying data...")
    ero  = query_ero(username)
    desi = query_desi(username)

    print("\n[2] Computing Δmag and applying Q1 XGBoost model...")
    ero,  _       = apply_xgb(ero)
    desi, n_seeds = apply_xgb(desi)

    print("\n[3] Plotting...")
    make_plot(ero, desi, n_seeds, OUTPUT_DIR)

    return 0


if __name__ == '__main__':
    sys.exit(main())
