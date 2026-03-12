#!/usr/bin/env python3
"""
VIS − Gaia G magnitude difference vs spread_model criterion — produces:
  1. ero_dmag_vs_gmag_by_spread_model.png — 2×2 plot:
       Row 1: Δmag vs Gaia G split by SM + 5/3×SME >= / < 0.005  (density)
       Row 2: same split, scatter colored by corrected BP-RP excess C*
  2. ero_spread_model_dmag_xgb.png — 4×2 Q1 XGBoost model probs split by Δmag sign

Mirrors analyze_ero_class_star_dmag.py but using SM + 5/3×SME criterion.

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_ero_spread_model_dmag.py
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

PROJECT_DIR   = Path(__file__).parent.parent
OUTPUT_DIR    = PROJECT_DIR / 'output' / 'ero_dmag_sm53'
MODEL_Q1_PATH = PROJECT_DIR / 'output' / 'xgboost_q1' / 'xgb_classifier_q1.json'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

ERO_FIELDS = {
    'Horsehead':   (85.150,  -2.613,  0.58),
    'Messier 78':  (86.690,  -0.015,  0.60),
    'Taurus':      (64.983,  28.023,  0.61),
    'Barnard 30':  (82.880,  12.316,  0.60),
    'NGC 6254':    (254.303,  -4.100, 0.60),
    'NGC 6397':    (265.174, -53.658, 0.61),
    'NGC 6822':    (296.236, -14.788, 0.60),
    'Holmberg II': (124.790,  70.706, 0.60),
    'IC 10':       (  5.063,  59.288, 0.62),
    'IC 342':      ( 56.730,  68.084, 0.59),
    'NGC 2403':    (114.211,  65.586, 0.60),
    'NGC 6744':    (287.430, -63.842, 0.60),
    'Fornax':      ( 54.017, -35.267, 0.57),
    'Dorado':      ( 64.014, -55.780, 0.60),
    'Perseus':     ( 49.638,  41.651, 0.70),
    'Abell 2390':  (328.397,  17.709, 0.75),
    'Abell 2764':  (  5.713, -49.249, 0.75),
}


def assign_fields(df):
    """Assign each source to its nearest ERO field (within field radius)."""
    ra  = df['ra'].values
    dec = df['dec'].values
    n_sources = len(df)

    field_assignments = np.full(n_sources, None, dtype=object)
    min_distances     = np.full(n_sources, np.inf)

    for field_name, (field_ra, field_dec, area) in ERO_FIELDS.items():
        radius = np.sqrt(area / np.pi)
        ra1  = np.radians(ra);       dec1 = np.radians(dec)
        ra2  = np.radians(field_ra); dec2 = np.radians(field_dec)
        a    = (np.sin((dec2 - dec1) / 2)**2
                + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2)**2)
        dist = np.degrees(2 * np.arcsin(np.sqrt(a)))
        mask = (dist < radius) & (dist < min_distances)
        field_assignments[mask] = field_name
        min_distances[mask]     = dist[mask]

    df['field'] = field_assignments
    return df


# ─── Data query ───────────────────────────────────────────────────────────────

def query_data(username):
    """Query ERO Gaia-matched sources with spread_model, spreaderr_model, magnitudes, and Gaia features."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_feat_cols = ', '.join([f'g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT
            e_v_alpha_j2000          AS ra,
            e_v_delta_j2000          AS dec,
            e_v_mag_auto             AS v_mag_auto,
            e_v_spread_model         AS spread_model,
            e_v_spreaderr_model      AS spread_model_err,
            g_corrected_bp_rp_excess AS c_star,
            {gaia_feat_cols}
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_spread_model    IS NOT NULL
          AND e_v_spreaderr_model IS NOT NULL
          AND e_v_mag_auto        IS NOT NULL
          AND g_phot_g_mean_mag   IS NOT NULL
    """
    print("  Querying ERO crossmatch + spread_model + Gaia features...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} sources")
    return df


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_dmag_vs_gmag_by_spread_model(df, output_dir):
    """2×2: Δmag vs Gaia G split by SM + 5/3×SME >= / < 0.005.
    Row 1: 2D density histogram (Blues, LogNorm).
    Row 2: scatter colored by corrected BP-RP excess factor C* (Riello et al. 2021).
    """
    base = df.copy()
    base['sm53sme']  = base['spread_model'] + 5/3 * base['spread_model_err']
    base['mag_diff'] = base['v_mag_auto'] - base['phot_g_mean_mag']

    xbins = np.arange(6, 23.1, 0.1)
    ybins = np.arange(-6.5, 7.6, 0.1)

    crit_label = r'SM + 5/3$\times$SME'
    panels = [
        (base[base['sm53sme'] >= 0.005], 'VIS galaxy', f'{crit_label} ≥ 0.005'),
        (base[base['sm53sme'] <  0.005], 'VIS star',   f'{crit_label} < 0.005'),
    ]

    # ── Row 1: density histograms ──────────────────────────────────────────
    vmaxes, histograms = [], []
    for sub, _, _ in panels:
        h, xe, ye = np.histogram2d(sub['phot_g_mean_mag'].values,
                                   sub['mag_diff'].values, bins=[xbins, ybins])
        hm = np.ma.masked_where(h == 0, h)
        if hm.count() > 0:
            vmaxes.append(hm.max())
        histograms.append((h, xe, ye))
    global_vmax = max(max(vmaxes, default=10), 2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharey=True)
    fig.subplots_adjust(hspace=0.35, wspace=0.35)

    for col_idx, (ax, (sub, short, crit_tag), (h, xe, ye)) in enumerate(
            zip(axes[0], panels, histograms)):
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
        ax.set_title(f'{short} ({crit_tag}, n={len(sub):,})', fontsize=12, fontweight='bold')
        ax.set_xlim(6, 23)
        ax.set_ylim(-6.5, 5.5)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(alpha=0.3)

    # ── Row 2: C* scatter ─────────────────────────────────────────────────
    for col_idx, (ax, (sub, short, _)) in enumerate(zip(axes[1], panels)):
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

    plt.suptitle(r'ERO: Δmag = (v_mag_auto − Gaia G) split by spread_model',
                 fontsize=13, fontweight='bold', y=0.92)
    plt.tight_layout()
    out = output_dir / 'ero_dmag_vs_gmag.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_xgb_q1_on_ero_spread_model(df, output_dir):
    """4×2 combined figure — columns: VIS galaxy (SM+5/3×SME≥0.005) | VIS star (<0.005).
    Row 0: Δmag vs Gaia G density 2D histogram.
    Row 1: Δmag vs Gaia G scatter coloured by C*.
    Row 2: Gaia G magnitude histogram split by Δmag sign.
    Row 3: Q1 XGBoost model probability split by Δmag sign.
    """
    import xgboost as xgb

    base = df.copy()
    base['sm53sme']  = base['spread_model'] + 5/3 * base['spread_model_err']
    base['mag_diff'] = base['v_mag_auto'] - base['phot_g_mean_mag']

    for col in SELECTED_GAIA_PARAMETERS:
        base[col] = pd.to_numeric(base[col], errors='coerce')

    X = base[SELECTED_GAIA_PARAMETERS].values
    q1_model = xgb.XGBClassifier()
    q1_model.load_model(str(MODEL_Q1_PATH))
    base['xgb_q1_prob'] = q1_model.predict_proba(X)[:, 1]

    crit_label = r'SM + 5/3$\times$SME'
    panels = [
        (base[base['sm53sme'] >= 0.005].copy(), 'VIS galaxy', f'{crit_label} ≥ 0.005'),
        (base[base['sm53sme'] <  0.005].copy(), 'VIS star',   f'{crit_label} < 0.005'),
    ]

    xbins     = np.arange(6, 23.1, 0.1)
    ybins     = np.arange(-6.5, 7.6, 0.1)
    mag_bins  = np.arange(14, 22.1, 0.1)
    prob_bins = np.arange(0, 1.01, 0.02)

    # Pre-compute density histograms for shared colour scale
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

    for ci, (sub, short, crit_tag) in enumerate(panels):
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
        ax.set_title(f'{short} ({crit_tag}, n={len(sub):,})', fontsize=11, fontweight='bold')
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

    fig.suptitle(r'ERO: Δmag = (v_mag_auto − Gaia G) split by spread_model & Q1 XGBoost model',
                 fontsize=13, fontweight='bold', y=0.93)
    out = output_dir / 'ero_dmag_vs_gmag_xgb.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ERO VIS − Gaia G Δmag vs spread_model criteria")
    print("=" * 70)

    print("\n[1] Querying data...")
    df = query_data(username)
    df = assign_fields(df)

    print("\n[2] Generating Δmag vs Gaia G split by spread_model criteria...")
    plot_dmag_vs_gmag_by_spread_model(df, OUTPUT_DIR)

    print("\n[3] Applying Q1 XGBoost model by spread_model criterion and Δmag sign...")
    plot_xgb_q1_on_ero_spread_model(df, OUTPUT_DIR)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
