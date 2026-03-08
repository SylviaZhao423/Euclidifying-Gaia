#!/usr/bin/env python3
"""
VIS − Gaia G magnitude difference vs spread_model criterion — produces:
  ero_dmag_vs_gmag_by_spread_model.png — 2×2 plot:
    Row 1: Δmag vs Gaia G split by SM + 3×SME  >= / < 0.003
    Row 2: Δmag vs Gaia G split by SM + 5/3×SME >= / < 0.005

Mirrors analyze_ero_class_star_dmag.py but using spread_model criteria.

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

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'output'

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
    """Query ERO Gaia-matched sources with spread_model, spreaderr_model, and magnitudes."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = """
        SELECT
            e_v_alpha_j2000     AS ra,
            e_v_delta_j2000     AS dec,
            e_v_mag_auto        AS v_mag_auto,
            e_v_spread_model    AS spread_model,
            e_v_spreaderr_model AS spread_model_err,
            g_phot_g_mean_mag   AS phot_g_mean_mag
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_spread_model    IS NOT NULL
          AND e_v_spreaderr_model IS NOT NULL
          AND e_v_mag_auto        IS NOT NULL
          AND g_phot_g_mean_mag   IS NOT NULL
    """
    print("  Querying ERO crossmatch + spread_model...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} sources")
    return df


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_dmag_vs_gmag_by_spread_model(df, output_dir):
    """2×2: Δmag vs Gaia G split by two spread_model criteria.
    Row 1: SM + 3×SME   >= / < 0.003  (galaxy left, star right)
    Row 2: SM + 5/3×SME >= / < 0.005  (galaxy left, star right)
    Mirrors ero_dmag_vs_gmag_by_class_star.png.
    """
    base = df.copy()
    base['sm3sme']   = base['spread_model'] + 3     * base['spread_model_err']
    base['sm53sme']  = base['spread_model'] + 5/3   * base['spread_model_err']
    base['mag_diff'] = base['v_mag_auto'] - base['phot_g_mean_mag']

    xbins = np.arange(14, 22.1, 0.1)
    ybins = np.arange(-2.5, 1.51, 0.05)

    criteria = [
        ('sm3sme',  0.003, r'SM + 3$\times$SME',   r'SM + 3$\times$SME'),
        ('sm53sme', 0.005, r'SM + 5/3$\times$SME', r'SM + 5/3$\times$SME'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharey=True)

    for row_idx, (col, cut, crit_label, _) in enumerate(criteria):
        panels = [
            (base[base[col] >= cut],
             f'{crit_label} ≥ {cut}  (VIS galaxy)'),
            (base[base[col] <  cut],
             f'{crit_label} < {cut}  (VIS star)'),
        ]

        vmaxes, histograms = [], []
        for sub, _ in panels:
            h, xe, ye = np.histogram2d(sub['phot_g_mean_mag'].values,
                                       sub['mag_diff'].values,
                                       bins=[xbins, ybins])
            hm = np.ma.masked_where(h == 0, h)
            if hm.count() > 0:
                vmaxes.append(hm.max())
            histograms.append((h, xe, ye))
        global_vmax = max(max(vmaxes, default=10), 2)

        for col_idx, (ax, (sub, label), (h, xe, ye)) in enumerate(
                zip(axes[row_idx], panels, histograms)):
            hm = np.ma.masked_where(h == 0, h)
            im = ax.pcolormesh(xe, ye, hm.T, cmap='Blues',
                               norm=LogNorm(vmin=1, vmax=global_vmax))
            ax.axhline(y=0, color='gray', ls='--', linewidth=1.0,
                       alpha=0.7, label='Δmag = 0')
            fig.colorbar(im, ax=ax, pad=0.02).set_label('Count', fontsize=10)
            ax.set_xlabel('Gaia G', fontsize=12, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel('v_mag_auto − phot_g_mean_mag', fontsize=11)
            ax.set_title(f'{label}\nN={len(sub):,}', fontsize=12, fontweight='bold')
            ax.set_xlim(14, 22)
            ax.set_ylim(-2.5, 1.5)
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(alpha=0.3)

    plt.suptitle('Δmag vs Gaia G — split by spread_model criteria  (ERO)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = output_dir / 'ero_dmag_vs_gmag_by_spread_model.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
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

    print(f"\nPlot saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
