#!/usr/bin/env python3
"""
mAB − Gaia G magnitude difference vs p_phz_star_prob and PHZ label — produces:
  1. q1_mag_diff_vs_star_prob.png  — Δmag vs p_phz_star_prob 2D hist in G magnitude slices
  2. q1_dmag_vs_gmag_by_phz.png   — Δmag vs Gaia G split by PHZ label (galaxy/star/unclassified)
  3. q1_dmag_xgb.png       — Q1 + Combined XGBoost probs split by Δmag sign (PHZ galaxy/star)

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_q1_phz_dmag.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR      = PROJECT_DIR / 'output' / 'q1_dmag'
MODEL_Q1_SEEDS_DIR = PROJECT_DIR / 'output' / 'xgboost_q1' / 'seed_models'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

ZPC_IE = 1.05
ZP_REF = 23.9
STAR_PROB_CUT = 0.5


# --- Data query ---------------------------------------------------------------

def query_data(username):
    """Query Gaia-matched Q1+PHZ sources with mAB, Gaia G, star_prob, phz_class, and Gaia features."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_feat_cols = ', '.join([f'c.g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT
            -2.5 * LOG10({ZPC_IE} * c.m_flux_detection_total) + {ZP_REF} AS ie_mag,
            c.g_phot_g_mean_mag          AS gaia_g,
            c.p_phz_star_prob            AS star_prob,
            c.p_phz_classification       AS phz_class,
            c.g_corrected_bp_rp_excess   AS c_star,
            {gaia_feat_cols}
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia c
        WHERE c.m_flux_detection_total > 0
          AND c.m_flux_detection_total IS NOT NULL
          AND c.m_vis_det = 1
          AND c.g_phot_g_mean_mag IS NOT NULL
          AND c.p_phz_star_prob IS NOT NULL
    """
    print("  Querying Q1 PHZ + Gaia...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['mag_diff'] = df['ie_mag'] - df['gaia_g']
    print(f"  Retrieved {len(df):,} sources")
    return df


# --- Plots --------------------------------------------------------------------

def plot_dmag_vs_gmag_by_phz(df, output_dir):
    """3-row grid: Gaia G vs Δmag split by PHZ label + mean PHZ P_galaxy per bin.
    Row 0: 2D density histogram split by PHZ label (3 cols).
    Row 1: Mean PHZ P_galaxy per bin — all sources, full width.
    Row 2: C* scatter split by PHZ label (3 cols).
    Uses p_phz_classification: phz=2 (galaxy), phz=1 (star), unclassified (other/NULL).
    """
    from matplotlib.gridspec import GridSpec

    base = df[df['mag_diff'].notna() & df['gaia_g'].notna()].copy()

    xbins = np.arange(6, 23.1, 0.1)
    ybins = np.arange(-6.5, 7.6, 0.1)

    panels = [
        (base[base['phz_class'] == 2],          'PHZ galaxy',   'phz=2'),
        (base[base['phz_class'] == 1],          'PHZ star',     'phz=1'),
        (base[~base['phz_class'].isin([1, 2])], 'Unclassified', 'phz≠1,2'),
    ]

    vmaxes, histograms = [], []
    for sub, _, _ in panels:
        h, xe, ye = np.histogram2d(sub['gaia_g'].values, sub['mag_diff'].values,
                                   bins=[xbins, ybins])
        hm = np.ma.masked_where(h == 0, h)
        if hm.count() > 0:
            vmaxes.append(hm.max())
        histograms.append((h, xe, ye))
    global_vmax = max(max(vmaxes, default=10), 2)

    # Mean PHZ P_galaxy per bin (all sources with star_prob available, min 3 per bin)
    base_p = base.dropna(subset=['star_prob'])
    gx = base_p['gaia_g'].values
    gy = base_p['mag_diff'].values
    gp = 1.0 - base_p['star_prob'].values   # PHZ P_galaxy
    xi = np.digitize(gx, xbins) - 1
    yi = np.digitize(gy, ybins) - 1
    valid = (xi >= 0) & (xi < len(xbins)-1) & (yi >= 0) & (yi < len(ybins)-1)
    h_sum   = np.zeros((len(xbins)-1, len(ybins)-1))
    h_count = np.zeros_like(h_sum)
    np.add.at(h_sum,   (xi[valid], yi[valid]), gp[valid])
    np.add.at(h_count, (xi[valid], yi[valid]), 1)
    mean_pgal = np.where(h_count >= 3, h_sum / h_count, np.nan)
    mean_pgal_m = np.ma.masked_invalid(mean_pgal)

    fig = plt.figure(figsize=(21, 18))
    gs = GridSpec(3, 3, figure=fig, hspace=0.25, wspace=0.20)
    ax_pgal = fig.add_subplot(gs[0, :])   # spans all 3 columns — top row
    axes_r1 = [fig.add_subplot(gs[1, c]) for c in range(3)]
    axes_r2 = [fig.add_subplot(gs[2, c]) for c in range(3)]

    # Row 0: mean PHZ P_galaxy per bin (all sources, full width)
    im2 = ax_pgal.pcolormesh(xbins, ybins, mean_pgal_m.T,
                              cmap='RdYlBu_r', vmin=0, vmax=1)
    ax_pgal.axhline(y=0, color='gray', ls='--', lw=1.0, alpha=0.7, label='Δmag = 0')
    cax2 = ax_pgal.inset_axes([1.005, 0, 0.008, 1])
    cb2 = fig.colorbar(im2, cax=cax2)
    cb2.set_label('Mean PHZ P_galaxy', fontsize=10, labelpad=10)
    cb2.set_ticks(np.arange(0, 1.01, 0.1))
    ax_pgal.set_xlabel('Gaia G', fontsize=11)
    ax_pgal.set_ylabel('I_E − phot_g_mean_mag', fontsize=11)
    ax_pgal.set_title(
        f'Mean PHZ P_galaxy per bin — all sources (n={len(base_p):,})',
        fontsize=12, fontweight='bold')
    ax_pgal.set_xlim(6, 23); ax_pgal.set_ylim(-6.5, 5.5)
    ax_pgal.legend(fontsize=9, loc='upper left'); ax_pgal.grid(alpha=0.3)

    # Row 1: density split by PHZ label
    for col_idx, (ax, (sub, short, phz_tag), (h, xe, ye)) in enumerate(
            zip(axes_r1, panels, histograms)):
        hm = np.ma.masked_where(h == 0, h)
        im = ax.pcolormesh(xe, ye, hm.T, cmap='Blues',
                           norm=LogNorm(vmin=1, vmax=global_vmax))
        ax.axhline(y=0, color='gray', ls='--', linewidth=1.0, alpha=0.7, label='Δmag = 0')
        fig.colorbar(im, ax=ax, pad=0.02).set_label('Count', fontsize=10)
        ax.set_xlabel('Gaia G', fontsize=11)
        ax.set_ylabel('I_E − phot_g_mean_mag' if col_idx == 0 else '', fontsize=11)
        ax.set_title(f'{short} ({phz_tag}, n={len(sub):,})', fontsize=12, fontweight='bold')
        ax.set_xlim(6, 23); ax.set_ylim(-6.5, 5.5)
        ax.legend(fontsize=9, loc='upper left'); ax.grid(alpha=0.3)

    # Row 2: C* scatter split by PHZ label
    for col_idx, (ax, (sub, short, _)) in enumerate(zip(axes_r2, panels)):
        sub_c = sub.dropna(subset=['c_star', 'gaia_g', 'mag_diff'])
        sc = ax.scatter(sub_c['gaia_g'], sub_c['mag_diff'],
                        c=sub_c['c_star'], cmap='RdYlBu_r',
                        vmin=-1, vmax=1, s=1, alpha=0.7, rasterized=True)
        ax.axhline(y=0, color='gray', ls='--', linewidth=1.0, alpha=0.7, label='Δmag = 0')
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label('C* (corrected BP-RP excess)', fontsize=10)
        cb.set_ticks(np.arange(-1.0, 1.01, 0.2))
        ax.set_xlabel('Gaia G', fontsize=11)
        ax.set_ylabel('I_E − phot_g_mean_mag' if col_idx == 0 else '', fontsize=11)
        ax.set_title(f'{short} — C*', fontsize=12, fontweight='bold')
        ax.set_xlim(6, 23); ax.set_ylim(-6.5, 5.5)
        ax.legend(fontsize=9, loc='upper left'); ax.grid(alpha=0.3)

    plt.suptitle('Q1: Δmag = (I_E − Gaia G) split by PHZ label',
                 fontsize=13, fontweight='bold', y=0.92)
    plt.tight_layout()
    out = output_dir / 'q1_dmag_vs_gmag.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_xgb_on_q1_by_phz(df, output_dir):
    """4×2 combined figure — columns: PHZ galaxy (phz=2) | PHZ star (phz=1).
    Row 0: Δmag vs Gaia G density 2D histogram.
    Row 1: Δmag vs Gaia G scatter coloured by C*.
    Row 2: Gaia G magnitude histogram split by Δmag sign.
    Row 3: Q1 XGBoost model probability split by Δmag sign.
    """
    import xgboost as xgb

    base = df[df['mag_diff'].notna() & df['gaia_g'].notna()].copy()

    for col in SELECTED_GAIA_PARAMETERS:
        base[col] = pd.to_numeric(base[col], errors='coerce')

    X = base[SELECTED_GAIA_PARAMETERS].values
    seed_model_paths = sorted(MODEL_Q1_SEEDS_DIR.glob('xgb_classifier_q1_seed*.json'))
    if not seed_model_paths:
        raise FileNotFoundError(f'No seed models found in {MODEL_Q1_SEEDS_DIR}. Run train_xgboost_q1.py first.')
    probs = []
    for model_path in seed_model_paths:
        m = xgb.XGBClassifier()
        m.load_model(str(model_path))
        probs.append(m.predict_proba(X)[:, 1])
    base['xgb_q1_prob'] = np.mean(np.vstack(probs), axis=0)
    model_label = f'Q1 ensemble mean ({len(seed_model_paths)} seeds)'

    panels = [
        (base[base['phz_class'] == 2].copy(), 'PHZ galaxy', 'phz=2'),
        (base[base['phz_class'] == 1].copy(), 'PHZ star',   'phz=1'),
    ]

    xbins     = np.arange(6, 23.1, 0.1)
    ybins     = np.arange(-6.5, 7.6, 0.1)
    mag_bins  = np.arange(14, 22.1, 0.1)
    prob_bins = np.arange(0, 1.01, 0.02)

    # Pre-compute density histograms for shared colour scale
    histograms, vmaxes = [], []
    for sub, _, _ in panels:
        h, xe, ye = np.histogram2d(sub['gaia_g'].values, sub['mag_diff'].values,
                                   bins=[xbins, ybins])
        hm = np.ma.masked_where(h == 0, h)
        if hm.count() > 0:
            vmaxes.append(hm.max())
        histograms.append((h, xe, ye))
    global_vmax = max(max(vmaxes, default=10), 2)

    rng = np.random.default_rng(42)

    # Simple 4×2 grid — colorbars placed as inset_axes so all rows stay perfectly aligned
    fig, axes_arr = plt.subplots(4, 2, figsize=(15, 16))
    axes = [list(row) for row in axes_arr]
    fig.subplots_adjust(hspace=0.35, wspace=0.30)

    for ci, (sub, short, phz_tag) in enumerate(panels):
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
            ax.set_ylabel('I_E − phot_g_mean_mag', fontsize=11)
        else:
            ax.set_ylabel('')
        ax.set_title(f'{short} ({phz_tag}, n={len(sub):,})', fontsize=11, fontweight='bold', linespacing=1.5)
        ax.set_xlim(6, 23); ax.set_ylim(-6.5, 5.5)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(alpha=0.3)

        # Row 1: C* scatter
        ax = axes[1][ci]
        sub_c = sub.dropna(subset=['c_star', 'gaia_g', 'mag_diff'])
        n_plot = min(len(sub_c), 50_000)
        idx = rng.choice(len(sub_c), n_plot, replace=False)
        sub_s = sub_c.iloc[idx]
        sc = ax.scatter(sub_s['gaia_g'], sub_s['mag_diff'],
                        c=sub_s['c_star'], cmap='RdYlBu_r',
                        vmin=-1, vmax=1, s=1, alpha=0.7, rasterized=True)
        ax.axhline(y=0, color='gray', ls='--', linewidth=1.0, alpha=0.7, label='Δmag = 0')
        cax = ax.inset_axes([1.02, 0, 0.03, 1])
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label('C* (corrected BP-RP excess)', fontsize=9, labelpad=10)
        cb.set_ticks(np.arange(-1.0, 1.01, 0.2))
        ax.set_xlabel('Gaia G', fontsize=11)
        if ci == 0:
            ax.set_ylabel('I_E − phot_g_mean_mag', fontsize=11)
        else:
            ax.set_ylabel('')
        ax.set_xlim(6, 23); ax.set_ylim(-6.5, 5.5)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(alpha=0.3)

        # Row 2: G magnitude histogram split by Δmag sign
        ax = axes[2][ci]
        ax.hist(neg['gaia_g'], bins=mag_bins, alpha=0.6, color='steelblue',
                label=f'Δmag < 0 (I_E brighter)  N={len(neg):,}')
        ax.hist(pos['gaia_g'], bins=mag_bins, alpha=0.6, color='coral',
                label=f'Δmag > 0 (I_E fainter)   N={len(pos):,}')
        ax.set_xlabel('Gaia G', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Row 3: probability split by Δmag sign
        ax = axes[3][ci]
        if ci == 0:
            vals_neg, vals_pos = neg['xgb_q1_prob'], pos['xgb_q1_prob']
            xlabel = 'PHZ galaxy probability'
        else:
            vals_neg, vals_pos = neg['star_prob'], pos['star_prob']
            xlabel = 'PHZ star probability'
        ax.hist(vals_neg, bins=prob_bins, alpha=0.6, color='steelblue',
                label=f'Δmag < 0  N={len(neg):,}')
        ax.hist(vals_pos, bins=prob_bins, alpha=0.6, color='coral',
                label=f'Δmag > 0  N={len(pos):,}')
        ax.axvline(x=1, color='red', ls='--', linewidth=1.5, label='prob = 1')
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle('Q1: Δmag = (I_E − Gaia G) split by PHZ label',
                 fontsize=13, fontweight='bold', y=0.93)
    out = output_dir / 'q1_dmag_vs_gmag_xgb.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# --- Main ---------------------------------------------------------------------

def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Q1 mAB − Gaia G Δmag vs p_phz_star_prob, PHZ label, and XGBoost")
    print("=" * 70)

    print("\n[1] Querying data...")
    df = query_data(username)

    print("\n[2] Generating I_E−Gaia Δmag vs Gaia G split by PHZ label...")
    plot_dmag_vs_gmag_by_phz(df, OUTPUT_DIR)

    print("\n[4] Applying Q1 + Combined XGBoost models by PHZ label and Δmag sign...")
    plot_xgb_on_q1_by_phz(df, OUTPUT_DIR)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
