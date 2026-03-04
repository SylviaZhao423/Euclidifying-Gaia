#!/usr/bin/env python3
"""
class_star vs magnitude 2D histograms split by Gaia neighbour count
at two radii (cnt4 = 4", cnt0_25 = 0.25").

Produces:
  1. class_star_neighbours_2d.png — 2x4 grid (rows = radii, cols = count bins):
     Row 1 (cnt4):    =1 (isolated), =2 (one neighbour), 3–4 (moderate), >=5 (crowded)
     Row 2 (cnt0_25): =1 (isolated), =2 (one neighbour), 3–4 (moderate), >=5 (crowded)
  2. class_star_vs_cnt.png — class_star vs neighbour count, one panel per radius

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_class_star_neighbours.py
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
OUTPUT_DIR = PROJECT_DIR / 'output'
MODEL_Q1_PATH   = PROJECT_DIR / 'output' / 'xgboost_combined' / 'xgb_classifier_q1.json'
MODEL_COMB_PATH = PROJECT_DIR / 'output' / 'xgboost_combined' / 'xgb_classifier_combined.json'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

ERO_FIELDS = {
    'Horsehead':    (85.150, -2.613,  0.58),
    'Messier 78':   (86.690, -0.015,  0.60),
    'Taurus':       (64.983, +28.023, 0.61),
    'Barnard 30':   (82.880, +12.316, 0.60),
    'NGC 6254':     (254.303, -4.100, 0.60),
    'NGC 6397':     (265.174, -53.658, 0.61),
    'NGC 6822':     (296.236, -14.788, 0.60),
    'Holmberg II':  (124.790, +70.706, 0.60),
    'IC 10':        (5.063, +59.288, 0.62),
    'IC 342':       (56.730, +68.084, 0.59),
    'NGC 2403':     (114.211, +65.586, 0.60),
    'NGC 6744':     (287.430, -63.842, 0.60),
    'Fornax':       (54.017, -35.267, 0.57),
    'Dorado':       (64.014, -55.780, 0.60),
    'Perseus':      (49.638, +41.651, 0.70),
    'Abell 2390':   (328.397, +17.709, 0.75),
    'Abell 2764':   (5.713, -49.249, 0.75),
}

RADII = [
    ('cnt4',  '4"'),
    ('cnt8',  '8"'),
    ('cnt16', '16"'),
    ('cnt32', '32"'),
]


def assign_fields(df):
    """Assign each source to its nearest ERO field (within field radius)."""
    ra = df['ra'].values
    dec = df['dec'].values
    n_sources = len(df)

    field_assignments = np.full(n_sources, None, dtype=object)
    min_distances = np.full(n_sources, np.inf)

    for field_name, (field_ra, field_dec, area) in ERO_FIELDS.items():
        radius = np.sqrt(area / np.pi)
        ra1 = np.radians(ra)
        dec1 = np.radians(dec)
        ra2 = np.radians(field_ra)
        dec2 = np.radians(field_dec)
        ddec = dec2 - dec1
        dra = ra2 - ra1
        a = np.sin(ddec/2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra/2)**2
        dist = np.degrees(2 * np.arcsin(np.sqrt(a)))
        mask = (dist < radius) & (dist < min_distances)
        field_assignments[mask] = field_name
        min_distances[mask] = dist[mask]

    df['field'] = field_assignments
    return df


# --- Data query ---------------------------------------------------------------

def query_data(username):
    """Query ERO crossmatch joined with Gaia neighbour counts at all radii."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    gaia_feat_cols = ', '.join([f'c.g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT
            e_v_alpha_j2000 AS ra,
            e_v_delta_j2000 AS dec,
            e_v_mag_auto AS v_mag_auto,
            e_v_class_star AS class_star,
            n.cnt4, n.cnt8, n.cnt16, n.cnt32,
            {gaia_feat_cols}
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3 c
        JOIN gaia_edr3_aux.gaia_source_neighbour_count n
            ON c.g_source_id = n.source_id
        WHERE e_v_class_star IS NOT NULL
    """
    print("  Querying crossmatch + neighbour counts + Gaia features...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    # phot_g_mean_mag is now in SELECTED_GAIA_PARAMETERS; alias for convenience
    df.rename(columns={'phot_g_mean_mag': 'phot_g_mean_mag'}, inplace=True)
    print(f"  Retrieved {len(df):,} sources")
    return df


# --- Plot ---------------------------------------------------------------------

def plot_class_star_by_neighbours(df, output_dir):
    """6×4 grid: rows = radii (4"–128"), cols = neighbour count bins. Raw log counts."""
    sub = df[(df['field'].notna())
             & df['phot_g_mean_mag'].notna()
             & df['class_star'].notna()]

    xbins = np.arange(14, 22.05, 0.05)
    ybins = np.arange(0, 1.01, 0.01)

    count_bins = [
        (lambda s, c: s[c] == 1,            '= 1 (isolated)'),
        (lambda s, c: s[c] == 2,            '= 2'),
        (lambda s, c: s[c].between(3, 4),   '= 3–4'),
        (lambda s, c: s[c] >= 5,            '≥ 5 (crowded)'),
    ]

    n_rows = len(RADII)
    n_cols = len(count_bins)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 6 * n_rows))

    for row, (cnt_col, radius_label) in enumerate(RADII):
        s = sub[sub[cnt_col].notna()]

        for col, (mask_fn, bin_label) in enumerate(count_bins):
            ax = axes[row, col]
            panel_df = s[mask_fn(s, cnt_col)]
            n = len(panel_df)

            h, xedges, yedges = np.histogram2d(
                panel_df['phot_g_mean_mag'].values,
                panel_df['class_star'].values,
                bins=[xbins, ybins]
            )

            h_masked = np.ma.masked_where(h == 0, h)
            vmax = h_masked.max() if h_masked.count() > 0 else 10
            im = ax.pcolormesh(xedges, yedges, h_masked.T,
                               cmap='Blues', norm=LogNorm(vmin=1, vmax=max(2, vmax)))
            cbar = fig.colorbar(im, ax=ax, pad=0.02)
            cbar.set_label('Count', fontsize=10)

            ax.axhline(y=0.65, color='red', ls='--', linewidth=1.5, label='cut=0.65')
            # Column header: only on top row
            if row == 0:
                ax.set_title(f'cnt {bin_label}', fontsize=12, fontweight='bold')
            else:
                ax.set_title('')
            # Row label: horizontal text on left column
            if col == 0:
                ax.set_ylabel('class_star', fontsize=10)
                ax.annotate(f'r = {radius_label}', xy=(0, 0.5),
                            xycoords='axes fraction', xytext=(-0.17, 0.5),
                            textcoords='axes fraction', fontsize=12,
                            fontweight='bold', ha='center', va='center',
                            rotation=0)
            else:
                ax.set_ylabel('class_star', fontsize=10)
            # N count inside panel
            ax.text(0.03, 0.97, f'N={n:,}', transform=ax.transAxes,
                    fontsize=9, va='top', ha='left')
            ax.set_xlabel('Gaia G' if row == n_rows - 1 else '', fontsize=10)
            ax.set_xlim(14, 22)
            ax.set_ylim(0, 1)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(alpha=0.3)

    plt.suptitle('class_star vs Gaia G by neighbour count (raw counts)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.subplots_adjust(left=0.08)
    output_path = output_dir / 'class_star_neighbours_2d.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_class_star_vs_cnt(df, output_dir):
    """1x5 row of class_star vs neighbour count at each radius."""
    base = df[(df['field'].notna()) & df['class_star'].notna()]
    ybins = np.arange(0, 1.01, 0.01)

    fig, axes = plt.subplots(1, len(RADII), figsize=(6 * len(RADII), 6))

    for ax, (cnt_col, radius_label) in zip(axes, RADII):
        sub = base[base[cnt_col].notna()]
        cnt_max = int(sub[cnt_col].quantile(0.99)) if len(sub) > 0 else 5
        cnt_max = max(cnt_max, 2)
        xbins = np.arange(0.5, cnt_max + 1.5, 1)

        h, xedges, yedges = np.histogram2d(
            sub[cnt_col].values, sub['class_star'].values, bins=[xbins, ybins])
        h_masked = np.ma.masked_where(h == 0, h)
        vmax = h_masked.max() if h_masked.count() > 0 else 10
        im = ax.pcolormesh(xedges, yedges, h_masked.T,
                           cmap='Blues', norm=LogNorm(vmin=1, vmax=max(2, vmax)))
        ax.axhline(y=0.65, color='red', ls='--', linewidth=1.5, label='cut=0.65')
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Count', fontsize=10)
        ax.set_xlabel(f'{cnt_col} (neighbours within {radius_label})',
                      fontsize=10, fontweight='bold')
        ax.set_ylabel('class_star', fontsize=10)
        ax.set_title(f'r = {radius_label}  (N={len(sub):,})',
                     fontsize=12, fontweight='bold')
        ax.set_xlim(xbins[0], xbins[-1])
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'class_star_vs_cnt.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")





def plot_mag_diff_vs_class_star_combined(df, output_dir):
    """1×4 row: Δmag (v_mag_auto − Gaia G) vs class_star 2D histogram.
    All G / G<18 / 18–18.5 / G>18.5. Y-axes shared.
    """
    base = df[df['phot_g_mean_mag'].notna()
              & df['v_mag_auto'].notna()
              & df['class_star'].notna()].copy()
    base['mag_diff'] = base['v_mag_auto'] - base['phot_g_mean_mag']

    xbins = np.arange(0, 1.01, 0.01)
    ybins = np.arange(-2.5, 1.51, 0.05)

    # Combined: all G / G<18 / 18–18.5 / G<20 / G<21 / G>21
    sub_panels = [
        # (base,                                                    'All G'),
        (base[base['phot_g_mean_mag'] > 15.0],                    'All G > 15.0'),
        (base[base['phot_g_mean_mag'].between(17.5, 18.5)],       '17.5 ≤ G < 18.5'),
        (base[base['phot_g_mean_mag'].between(18.5, 19.5)],       '18.5 ≤ G < 19.5'),
        (base[base['phot_g_mean_mag'].between(19.5, 21.0)],       '19.5 ≤ G < 21.0'),
        (base[base['phot_g_mean_mag'] > 21.0],                    'G > 21.0'),
    ]
    # Pre-compute all histograms to find a shared global vmax
    histograms = []
    for panel_df, _ in sub_panels:
        h, xe, ye = np.histogram2d(
            panel_df['class_star'].values, panel_df['mag_diff'].values,
            bins=[xbins, ybins])
        histograms.append((h, xe, ye))
    global_vmax = max(
        (np.ma.masked_where(h == 0, h).max()
         for h, _, _ in histograms
         if np.ma.masked_where(h == 0, h).count() > 0),
        default=10)
    global_vmax = max(2, global_vmax)

    fig2, axes2 = plt.subplots(1, 5, figsize=(40, 7), sharey=True)
    for j, (ax2, (panel_df, label), (h2, xedges2, yedges2)) in enumerate(
            zip(axes2, sub_panels, histograms)):
        n_total = len(panel_df)
        n_gal   = (panel_df['class_star'] < 0.65).sum()
        n_star  = (panel_df['class_star'] >= 0.65).sum()
        h2_masked = np.ma.masked_where(h2 == 0, h2)
        im2 = ax2.pcolormesh(xedges2, yedges2, h2_masked.T,
                             cmap='Blues', norm=LogNorm(vmin=1, vmax=global_vmax))
        ax2.axvline(x=0.65, color='red',  ls='--', linewidth=1.5, label='class_star = 0.65')
        ax2.axhline(y=0,    color='gray', ls='--', linewidth=1.0, alpha=0.7, label='Δmag = 0')
        fig2.colorbar(im2, ax=ax2, pad=0.02).set_label('Count', fontsize=10)
        ax2.set_xlabel('class_star', fontsize=12, fontweight='bold')
        if j == 0:
            ax2.set_ylabel('v_mag_auto − phot_g_mean_mag', fontsize=11, fontweight='bold')
        ax2.set_title(
            f'{label}\n'
            f'N={n_total:,}  (gal: {n_gal:,},  star: {n_star:,})',
            fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(-2.5, 1.5)
        ax2.legend(fontsize=9, loc='upper right')
        ax2.grid(alpha=0.3)
    fig2.suptitle('VIS − Gaia G magnitude difference vs class_star  (ERO)',
                  fontsize=13, fontweight='bold', y=1.02)
    fig2.tight_layout()
    out2 = output_dir / 'mag_diff_vs_class_star_combined.png'
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {out2}")


def plot_dmag_vs_gmag_galaxy(df, output_dir):
    """2D histogram of Gaia G vs Δmag for class_star < 0.65 and class_star >= 0.65,
    side by side. Directly shows which magnitude range drives the negative-Δmag blob."""
    base = df[df['phot_g_mean_mag'].notna()
              & df['v_mag_auto'].notna()
              & df['class_star'].notna()].copy()
    base['mag_diff'] = base['v_mag_auto'] - base['phot_g_mean_mag']

    xbins = np.arange(14, 22.1, 0.1)
    ybins = np.arange(-2.5, 1.51, 0.05)

    panels = [
        (base[base['class_star'] < 0.65],  'class_star < 0.65  (VIS galaxy)'),
        (base[base['class_star'] >= 0.65], 'class_star ≥ 0.65  (VIS star)'),
    ]

    # Shared colour scale
    vmaxes = []
    histograms = []
    for sub, _ in panels:
        h, xe, ye = np.histogram2d(sub['phot_g_mean_mag'].values,
                                   sub['mag_diff'].values,
                                   bins=[xbins, ybins])
        hm = np.ma.masked_where(h == 0, h)
        if hm.count() > 0:
            vmaxes.append(hm.max())
        histograms.append((h, xe, ye))
    global_vmax = max(max(vmaxes, default=10), 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, (sub, label), (h, xe, ye) in zip(axes, panels, histograms):
        hm = np.ma.masked_where(h == 0, h)
        im = ax.pcolormesh(xe, ye, hm.T, cmap='Blues',
                           norm=LogNorm(vmin=1, vmax=global_vmax))
        ax.axhline(y=0, color='gray', ls='--', linewidth=1.0, alpha=0.7, label='Δmag = 0')
        fig.colorbar(im, ax=ax, pad=0.02).set_label('Count', fontsize=10)
        ax.set_xlabel('Gaia G', fontsize=12, fontweight='bold')
        ax.set_ylabel('v_mag_auto − phot_g_mean_mag', fontsize=11)
        ax.set_title(f'{label}\nN={len(sub):,}', fontsize=12, fontweight='bold')
        ax.set_xlim(14, 22)
        ax.set_ylim(-2.5, 1.5)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)

    plt.suptitle('Δmag vs Gaia G — split by class_star  (ERO)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = output_dir / 'dmag_vs_gmag_by_class_star.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_xgb_q1_on_ero_galaxies(df, output_dir):
    """2×3 grid: rows = class_star < 0.65 / class_star >= 0.65,
    cols = G magnitude histogram / Q1 model prob / Combo model prob,
    each split by sign of Δmag."""
    import xgboost as xgb

    base = df[df['phot_g_mean_mag'].notna()
              & df['v_mag_auto'].notna()
              & df['class_star'].notna()].copy()
    base['mag_diff'] = base['v_mag_auto'] - base['phot_g_mean_mag']

    for col in SELECTED_GAIA_PARAMETERS:
        base[col] = pd.to_numeric(base[col], errors='coerce')

    X = base[SELECTED_GAIA_PARAMETERS].values

    q1_model = xgb.XGBClassifier()
    q1_model.load_model(str(MODEL_Q1_PATH))
    base['xgb_q1_prob'] = q1_model.predict_proba(X)[:, 1]

    comb_model = xgb.XGBClassifier()
    comb_model.load_model(str(MODEL_COMB_PATH))
    base['xgb_comb_prob'] = comb_model.predict_proba(X)[:, 1]

    rows = [
        (base[base['class_star'] < 0.65].copy(),  'class_star < 0.65  (VIS galaxy)'),
        (base[base['class_star'] >= 0.65].copy(), 'class_star ≥ 0.65  (VIS star)'),
    ]

    mag_bins  = np.arange(14, 22.1, 0.1)
    prob_bins = np.arange(0, 1.01, 0.02)

    fig, axes = plt.subplots(2, 3, figsize=(21, 10))

    for row_idx, (sub, row_label) in enumerate(rows):
        neg = sub[sub['mag_diff'] < 0]
        pos = sub[sub['mag_diff'] > 0]

        # Col 0: G magnitude histogram
        ax = axes[row_idx, 0]
        ax.hist(neg['phot_g_mean_mag'], bins=mag_bins, alpha=0.6, color='steelblue',
                label=f'Δmag < 0 (VIS brighter)  N={len(neg):,}')
        ax.hist(pos['phot_g_mean_mag'], bins=mag_bins, alpha=0.6, color='coral',
                label=f'Δmag > 0 (VIS fainter)   N={len(pos):,}')
        ax.set_xlabel('Gaia G', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'G magnitude — {row_label}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Row 0 (galaxies): show galaxy prob, ideal at 1.
        # Row 1 (stars):    show star prob = 1 - galaxy prob, ideal at 1.
        if row_idx == 0:
            q1_vals   = neg['xgb_q1_prob'],   pos['xgb_q1_prob']
            comb_vals = neg['xgb_comb_prob'],  pos['xgb_comb_prob']
            prob_xlabel_q1   = 'Q1 model galaxy probability'
            prob_xlabel_comb = 'Combined model galaxy probability'
            vline_x, vline_label = 1, 'prob = 1'
        else:
            q1_vals   = 1 - neg['xgb_q1_prob'],   1 - pos['xgb_q1_prob']
            comb_vals = 1 - neg['xgb_comb_prob'],  1 - pos['xgb_comb_prob']
            prob_xlabel_q1   = 'Q1 model star probability (1 − galaxy prob)'
            prob_xlabel_comb = 'Combined model star probability (1 − galaxy prob)'
            vline_x, vline_label = 1, 'prob = 1'

        # Col 1: Q1 model
        ax = axes[row_idx, 1]
        ax.hist(q1_vals[0], bins=prob_bins, alpha=0.6, color='steelblue',
                label=f'Δmag < 0  N={len(neg):,}')
        ax.hist(q1_vals[1], bins=prob_bins, alpha=0.6, color='coral',
                label=f'Δmag > 0  N={len(pos):,}')
        ax.axvline(x=vline_x, color='red', ls='--', linewidth=1.5, label=vline_label)
        ax.set_xlabel(prob_xlabel_q1, fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Q1 model — {row_label}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Col 2: Combined model
        ax = axes[row_idx, 2]
        ax.hist(comb_vals[0], bins=prob_bins, alpha=0.6, color='steelblue',
                label=f'Δmag < 0  N={len(neg):,}')
        ax.hist(comb_vals[1], bins=prob_bins, alpha=0.6, color='coral',
                label=f'Δmag > 0  N={len(pos):,}')
        ax.axvline(x=vline_x, color='red', ls='--', linewidth=1.5, label=vline_label)
        ax.set_xlabel(prob_xlabel_comb, fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Combined model — {row_label}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle('ERO sources split by Δmag sign — Q1 vs Combined XGBoost model (Gaia features only)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = output_dir / 'galaxy_dmag_xgb_q1.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
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
    print("class_star vs Gaia G — split by neighbour count (multi-radius)")
    print("=" * 70)

    print("\n[1] Querying data...")
    df = query_data(username)
    df = assign_fields(df)

    print("\n[2] Generating class_star vs Gaia G by neighbour count (raw + normalised)...")
    plot_class_star_by_neighbours(df, OUTPUT_DIR)

    print("\n[3] Generating class_star vs cnt (all radii)...")
    plot_class_star_vs_cnt(df, OUTPUT_DIR)

    print("\n[4] Generating VIS−Gaia mag difference vs class_star...")
    plot_mag_diff_vs_class_star_combined(df, OUTPUT_DIR)

    print("\n[5] Generating Δmag vs Gaia G by class_star...")
    plot_dmag_vs_gmag_galaxy(df, OUTPUT_DIR)

    print("\n[6] Applying Q1 XGBoost model to ERO sources by dM sign...")
    plot_xgb_q1_on_ero_galaxies(df, OUTPUT_DIR)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
