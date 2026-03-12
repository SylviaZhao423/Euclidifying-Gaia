#!/usr/bin/env python3
"""
VIS − Gaia G magnitude difference vs class_star and XGBoost model — produces:
  1. ero_dmag_vs_gmag_by_class_star.png — Δmag vs Gaia G split by class_star < / >= 0.65
  2. ero_dmag_xgb.png        — Q1 XGBoost probs split by Δmag sign (VIS galaxy/star)

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_ero_class_star_dmag.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR   = Path(__file__).parent.parent
OUTPUT_DIR    = PROJECT_DIR / 'output' / 'ero_dmag'
MODEL_Q1_SEEDS_DIR = PROJECT_DIR / 'output' / 'xgboost_q1' / 'seed_models'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

# --- Data query ---------------------------------------------------------------

def query_data(username):
    """Query ERO crossmatch with Gaia features for Δmag and XGBoost plots."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_feat_cols = ', '.join([f'c.g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT
            e_v_alpha_j2000            AS ra,
            e_v_delta_j2000            AS dec,
            e_v_mag_auto               AS v_mag_auto,
            e_v_class_star             AS class_star,
            c.g_corrected_bp_rp_excess AS c_star,
            {gaia_feat_cols}
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3 c
        WHERE e_v_class_star IS NOT NULL
    """
    print("  Querying ERO crossmatch + Gaia features...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} sources")
    return df


# --- Plots --------------------------------------------------------------------

def plot_dmag_vs_gmag_galaxy(df, output_dir):
    """2×2 grid: Gaia G vs Δmag split by class_star < / >= 0.65.
    Row 1: 2D density histogram (Blues, LogNorm).
    Row 2: scatter colored by corrected BP-RP excess factor C* (Riello et al. 2021).
    """
    base = df[df['phot_g_mean_mag'].notna()
              & df['v_mag_auto'].notna()
              & df['class_star'].notna()].copy()
    base['mag_diff'] = base['v_mag_auto'] - base['phot_g_mean_mag']

    xbins = np.arange(6, 23.1, 0.1)
    ybins = np.arange(-6.5, 7.6, 0.1)

    panels = [
        (base[base['class_star'] <  0.65], 'VIS galaxy', 'cs<0.65'),
        (base[base['class_star'] >= 0.65], 'VIS star',   'cs≥0.65'),
    ]

    # ── Pre-compute density histograms ────────────────────────────────────
    vmaxes, histograms = [], []
    for sub, _, _ in panels:
        h, xe, ye = np.histogram2d(sub['phot_g_mean_mag'].values,
                                   sub['mag_diff'].values, bins=[xbins, ybins])
        hm = np.ma.masked_where(h == 0, h)
        if hm.count() > 0:
            vmaxes.append(hm.max())
        histograms.append((h, xe, ye))
    global_vmax = max(max(vmaxes, default=10), 2)

    # ── Mean class_star per bin (full dataset, no split) ─────────────────
    cs_all = base.dropna(subset=['class_star', 'phot_g_mean_mag', 'mag_diff'])
    xi = np.searchsorted(xbins[:-1], cs_all['phot_g_mean_mag'].values, side='right') - 1
    yi = np.searchsorted(ybins[:-1], cs_all['mag_diff'].values, side='right') - 1
    valid = (xi >= 0) & (xi < len(xbins) - 1) & (yi >= 0) & (yi < len(ybins) - 1)
    h_sum   = np.zeros((len(xbins) - 1, len(ybins) - 1))
    h_count = np.zeros((len(xbins) - 1, len(ybins) - 1))
    np.add.at(h_sum,   (xi[valid], yi[valid]), cs_all['class_star'].values[valid])
    np.add.at(h_count, (xi[valid], yi[valid]), 1)
    mean_cs = np.where(h_count >= 3, h_sum / h_count, np.nan)
    mean_cs_m = np.ma.masked_invalid(mean_cs)

    fig = plt.figure(figsize=(14, 18))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.35)
    ax_cs   = fig.add_subplot(gs[0, :])   # top row, full width
    axes_r1 = [fig.add_subplot(gs[1, c]) for c in range(2)]
    axes_r2 = [fig.add_subplot(gs[2, c]) for c in range(2)]

    # ── Row 0: mean class_star per bin (full width) ───────────────────────
    im2 = ax_cs.pcolormesh(xbins, ybins, mean_cs_m.T, cmap='RdYlBu', vmin=0, vmax=1)
    ax_cs.axhline(y=0, color='gray', ls='--', linewidth=1.0, alpha=0.7)
    cax2 = ax_cs.inset_axes([1.01, 0, 0.015, 1])
    cb2 = fig.colorbar(im2, cax=cax2)
    cb2.ax.invert_yaxis()
    cb2.set_label('Mean class_star', fontsize=10, labelpad=10)
    ax_cs.set_xlabel('Gaia G', fontsize=11)
    ax_cs.set_ylabel('v_mag_auto − phot_g_mean_mag', fontsize=11)
    ax_cs.set_title(f'Mean class_star per bin (all ERO, n={len(cs_all):,})',
                    fontsize=12, fontweight='bold')
    ax_cs.set_xlim(6, 23)
    ax_cs.set_ylim(-6.5, 5.5)
    ax_cs.grid(alpha=0.3)

    # ── Row 1: density histograms ─────────────────────────────────────────
    for col_idx, (ax, (sub, short, cs_tag), (h, xe, ye)) in enumerate(
            zip(axes_r1, panels, histograms)):
        hm = np.ma.masked_where(h == 0, h)
        im = ax.pcolormesh(xe, ye, hm.T, cmap='Blues',
                           norm=LogNorm(vmin=1, vmax=global_vmax))
        ax.axhline(y=0, color='gray', ls='--', linewidth=1.0, alpha=0.7, label='Δmag = 0')
        cax = ax.inset_axes([1.02, 0, 0.03, 1])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('Count', fontsize=9, labelpad=10)
        ax.set_xlabel('Gaia G', fontsize=11)
        if col_idx == 0:
            ax.set_ylabel('v_mag_auto − phot_g_mean_mag', fontsize=11)
        ax.set_title(f'{short} ({cs_tag}, n={len(sub):,})', fontsize=12, fontweight='bold')
        ax.set_xlim(6, 23)
        ax.set_ylim(-6.5, 5.5)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(alpha=0.3)

    # ── Row 2: scatter colored by C* ──────────────────────────────────────
    for col_idx, (ax, (sub, short, _)) in enumerate(zip(axes_r2, panels)):
        sub_c = sub.dropna(subset=['c_star', 'phot_g_mean_mag', 'mag_diff'])
        sc = ax.scatter(
            sub_c['phot_g_mean_mag'], sub_c['mag_diff'],
            c=sub_c['c_star'], cmap='RdYlBu_r',
            vmin=-1, vmax=1,
            s=1, alpha=0.7, rasterized=True,
        )
        ax.axhline(y=0, color='gray', ls='--', linewidth=1.0, alpha=0.7, label='Δmag = 0')
        cax = ax.inset_axes([1.02, 0, 0.03, 1])
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label('C* (corrected BP-RP excess)', fontsize=9, labelpad=10)
        cb.set_ticks(np.arange(-1.0, 1.01, 0.2))
        ax.set_xlabel('Gaia G', fontsize=11)
        if col_idx == 0:
            ax.set_ylabel('v_mag_auto − phot_g_mean_mag', fontsize=11)
        ax.set_title(f'{short} — C*', fontsize=12, fontweight='bold')
        ax.set_xlim(6, 23)
        ax.set_ylim(-6.5, 5.5)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(alpha=0.3)

    fig.suptitle('ERO: Δmag = (v_mag_auto − Gaia G) split by class_star',
                 fontsize=13, fontweight='bold', y=0.92)
    out = output_dir / 'ero_dmag_vs_gmag.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_xgb_q1_on_ero_galaxies(df, output_dir):
    """4×2 combined figure — columns: VIS galaxy (cs<0.65) | VIS star (cs≥0.65).
    Row 0: Δmag vs Gaia G density 2D histogram.
    Row 1: Δmag vs Gaia G scatter coloured by C*.
    Row 2: Gaia G magnitude histogram split by Δmag sign.
    Row 3: Q1 XGBoost model probability split by Δmag sign.
    """
    import xgboost as xgb

    base = df[df['phot_g_mean_mag'].notna()
              & df['v_mag_auto'].notna()
              & df['class_star'].notna()].copy()
    base['mag_diff'] = base['v_mag_auto'] - base['phot_g_mean_mag']

    for col in SELECTED_GAIA_PARAMETERS:
        base[col] = pd.to_numeric(base[col], errors='coerce')

    X = base[SELECTED_GAIA_PARAMETERS].values

    seed_model_paths = sorted(MODEL_Q1_SEEDS_DIR.glob('xgb_classifier_q1_seed*.json'))
    if seed_model_paths:
        probs = []
        for model_path in seed_model_paths:
            q1_model = xgb.XGBClassifier()
            q1_model.load_model(str(model_path))
            probs.append(q1_model.predict_proba(X)[:, 1])
        prob_arr = np.vstack(probs)
        base['xgb_q1_prob'] = np.mean(prob_arr, axis=0)
        base['xgb_q1_prob_std'] = np.std(prob_arr, axis=0)
        model_label = f'Q1 ensemble mean ({len(seed_model_paths)} seeds)'
    else:
        raise FileNotFoundError(
            f'No seed models found in {MODEL_Q1_SEEDS_DIR}. '
            'Run train_xgboost_q1.py first.'
        )

    panels = [
        (base[base['class_star'] <  0.65].copy(), 'VIS galaxy', 'cs<0.65'),
        (base[base['class_star'] >= 0.65].copy(), 'VIS star',   'cs≥0.65'),
    ]

    xbins        = np.arange(6, 23.1, 0.1)
    ybins        = np.arange(-6.5, 7.6, 0.1)
    mag_bins     = np.arange(14, 22.1, 0.1)
    prob_bins    = np.arange(0, 1.01, 0.02)
    # Pre-compute density histograms for shared colour scales
    histograms, vmaxes = [], []
    for sub, _, _ in panels:
        h, xe, ye = np.histogram2d(sub['phot_g_mean_mag'].values, sub['mag_diff'].values,
                                   bins=[xbins, ybins])
        hm = np.ma.masked_where(h == 0, h)
        if hm.count() > 0:
            vmaxes.append(hm.max())
        histograms.append((h, xe, ye))
    global_vmax = max(max(vmaxes, default=10), 2)

    # Simple 4×2 grid — colorbars placed as inset_axes so all rows stay perfectly aligned
    fig, axes_arr = plt.subplots(4, 2, figsize=(15, 16))
    axes = [list(row) for row in axes_arr]
    fig.subplots_adjust(hspace=0.35, wspace=0.30)

    for ci, (sub, short, cs_tag) in enumerate(panels):
        h, xe, ye = histograms[ci]
        neg = sub[sub['mag_diff'] < 0]
        pos = sub[sub['mag_diff'] > 0]

        # Row 0: density 2D histogram
        ax = axes[0][ci]
        hm = np.ma.masked_where(h == 0, h)
        im = ax.pcolormesh(xe, ye, hm.T, cmap='Blues',
                           norm=LogNorm(vmin=1, vmax=global_vmax))
        ax.axhline(y=0, color='gray', ls='--', linewidth=1.0, alpha=0.7, label='Δmag = 0')
        cax = ax.inset_axes([1.02, 0, 0.03, 1])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('Count', fontsize=9, labelpad=10)
        ax.set_xlabel('Gaia G', fontsize=11)
        if ci == 0:
            ax.set_ylabel('v_mag_auto − phot_g_mean_mag', fontsize=11)
        ax.set_title(f'{short} ({cs_tag}, n={len(sub):,})', fontsize=11, fontweight='bold')
        ax.set_xlim(6, 23); ax.set_ylim(-6.5, 5.5)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(alpha=0.3)

        # Row 1: C* scatter
        ax = axes[1][ci]
        sub_c = sub.dropna(subset=['c_star', 'phot_g_mean_mag', 'mag_diff'])
        sc = ax.scatter(sub_c['phot_g_mean_mag'], sub_c['mag_diff'],
                        c=sub_c['c_star'], cmap='RdYlBu_r',
                        vmin=-1, vmax=1, s=1, alpha=0.7, rasterized=True)
        ax.axhline(y=0, color='gray', ls='--', linewidth=1.0, alpha=0.7, label='Δmag = 0')
        cax = ax.inset_axes([1.02, 0, 0.03, 1])
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label('C* (corrected BP-RP excess)', fontsize=9, labelpad=10)
        cb.set_ticks(np.arange(-1.0, 1.01, 0.2))
        ax.set_xlabel('Gaia G', fontsize=11)
        if ci == 0:
            ax.set_ylabel('v_mag_auto − phot_g_mean_mag', fontsize=11)
        ax.set_xlim(6, 23); ax.set_ylim(-6.5, 5.5)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(alpha=0.3)

        # Row 2: G magnitude histogram split by Δmag sign
        ax = axes[2][ci]
        ax.hist(neg['phot_g_mean_mag'], bins=mag_bins, alpha=0.6, color='steelblue',
                label=f'Δmag < 0 (VIS brighter)  N={len(neg):,}')
        ax.hist(pos['phot_g_mean_mag'], bins=mag_bins, alpha=0.6, color='coral',
                label=f'Δmag > 0 (VIS fainter)   N={len(pos):,}')
        ax.set_xlabel('Gaia G', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Row 3: Q1 probability split by Δmag sign
        ax = axes[3][ci]
        if ci == 0:
            vals_neg, vals_pos = neg['xgb_q1_prob'], pos['xgb_q1_prob']
            xlabel = 'P (galaxy | Q1 model)'
        else:
            vals_neg, vals_pos = 1 - neg['xgb_q1_prob'], 1 - pos['xgb_q1_prob']
            xlabel = 'P (star | Q1 model)'
        ax.hist(vals_neg, bins=prob_bins, alpha=0.6, color='steelblue',
                label=f'Δmag < 0  N={len(neg):,}')
        ax.hist(vals_pos, bins=prob_bins, alpha=0.6, color='coral',
                label=f'Δmag > 0  N={len(pos):,}')
        ax.axvline(x=1, color='red', ls='--', linewidth=1.5, label='prob = 1')
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(f'ERO: Δmag = (v_mag_auto − Gaia G) split by class_star',
                 fontsize=13, fontweight='bold', y=0.93)
    out = output_dir / 'ero_dmag_vs_gmag_xgb.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

    return base, model_label


# --- Main ---------------------------------------------------------------------

def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ERO VIS − Gaia G Δmag vs class_star")
    print("=" * 70)

    print("\n[1] Querying data...")
    df = query_data(username)

    print("\n[2] Generating Δmag vs Gaia G split by class_star...")
    plot_dmag_vs_gmag_galaxy(df, OUTPUT_DIR)

    print("\n[3] Applying Q1 XGBoost model by class_star and Δmag sign...")
    base, model_label = plot_xgb_q1_on_ero_galaxies(df, OUTPUT_DIR)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
