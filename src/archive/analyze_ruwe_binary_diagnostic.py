#!/usr/bin/env python3
"""
Diagnostic: are Q1-model vs class_star disagreement cases on ERO
enriched in unresolved binary candidates?

Method:
  1. Query ERO sources (ra, dec, ruwe, class_star + 27 Gaia features).
  2. Apply the trained Q1 XGBoost model to get galaxy probability.
  3. Use GaiaUnlimited BinarySystemsSelectionFunction to obtain the
     sky-position-dependent RUWE threshold for each source.
  4. Compute ruwe_ratio = ruwe / ruwe_threshold  (>1 → binary candidate).
  5. Split sources into 4 groups by agreement/disagreement between the
     Q1 model and class_star, and plot the ruwe_ratio distributions.

Output:
  output/xgboost_combined/ruwe_binary_diagnostic.png

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_ruwe_binary_diagnostic.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from astropy.coordinates import SkyCoord
import astropy.units as u

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR    = Path(__file__).parent.parent
OUTPUT_DIR     = PROJECT_DIR / 'output'
MODEL_Q1_PATH  = PROJECT_DIR / 'output' / 'xgboost_combined' / 'xgb_classifier_q1.json'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'
MAG_CUT   = 18.5
CLASS_STAR_THRESH = 0.65   # same threshold as training labels


# ---------------------------------------------------------------------------
# 1. Query
# ---------------------------------------------------------------------------

def query_ero(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)

    gaia_cols = ', '.join([f'c.g_{col} AS {col}' for col in SELECTED_GAIA_PARAMETERS])

    query = f"""
        SELECT
            c.g_ra  AS ra,
            c.g_dec AS dec,
            c.g_ruwe AS ruwe_raw,
            c.e_v_class_star AS class_star,
            n.cnt4, n.cnt8, n.cnt16, n.cnt32,
            {gaia_cols}
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3 c
        JOIN gaia_edr3_aux.gaia_source_neighbour_count n
            ON c.g_source_id = n.source_id
        WHERE c.e_v_class_star IS NOT NULL
          AND c.g_phot_g_mean_mag >= {MAG_CUT}
          AND c.g_ra  IS NOT NULL
          AND c.g_dec IS NOT NULL
          AND c.g_ruwe IS NOT NULL
        ORDER BY c.g_source_id
    """
    print("  Querying ERO sources...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace('NOT_AVAILABLE', np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['ra']         = pd.to_numeric(df['ra'],         errors='coerce')
    df['dec']        = pd.to_numeric(df['dec'],        errors='coerce')
    df['ruwe_raw']   = pd.to_numeric(df['ruwe_raw'],   errors='coerce')
    df['class_star'] = pd.to_numeric(df['class_star'], errors='coerce')
    for col in ['cnt4', 'cnt8', 'cnt16', 'cnt32']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # class_star >= 0.65 → STAR,  < 0.65 → GALAXY
    n_star = (df['class_star'] >= CLASS_STAR_THRESH).sum()
    n_gal  = (df['class_star'] <  CLASS_STAR_THRESH).sum()
    print(f"  Retrieved {len(df):,} ERO sources "
          f"(STAR: {n_star:,}, GALAXY: {n_gal:,})")
    return df


# ---------------------------------------------------------------------------
# 2. Apply Q1 model
# ---------------------------------------------------------------------------

def apply_q1_model(df):
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_Q1_PATH))

    X = df[SELECTED_GAIA_PARAMETERS].values.astype(float)
    # predict_proba returns [P(star), P(galaxy)] — galaxy is class 1
    proba = model.predict_proba(X)
    df = df.copy()
    df['q1_galaxy_prob'] = proba[:, 1]
    print(f"  Q1 model applied to {len(df):,} sources")
    return df


# ---------------------------------------------------------------------------
# 3. GaiaUnlimited RUWE threshold
# ---------------------------------------------------------------------------

def add_ruwe_threshold(df):
    from gaiaunlimited.selectionfunctions import binaries

    print("  Loading GaiaUnlimited BinarySystemsSelectionFunction...")
    bsf = binaries.BinarySystemsSelectionFunction()

    coords = SkyCoord(ra=df['ra'].values * u.deg,
                      dec=df['dec'].values * u.deg)

    print(f"  Querying RUWE thresholds for {len(df):,} sources...")
    ruwe_threshold = bsf.query_RUWE(coords, crowding=True)

    df = df.copy()
    df['ruwe_threshold'] = np.array(ruwe_threshold)
    df['ruwe_ratio']     = df['ruwe_raw'] / df['ruwe_threshold']  # >1 → binary candidate
    return df


# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------

def plot_diagnostic(df, output_dir):
    # Drop rows with missing ruwe_ratio
    sub = df.dropna(subset=['ruwe_ratio', 'q1_galaxy_prob', 'class_star'])

    # Q1 model prediction: galaxy prob > 0.5 → galaxy
    # class_star: >= 0.65 → STAR,  < 0.65 → GALAXY
    q1_says_galaxy       = sub['q1_galaxy_prob'] > 0.5
    class_star_says_gal  = sub['class_star'] < CLASS_STAR_THRESH

    groups = {
        'Both agree: GALAXY\n(Q1 model gal & class_star gal)':
            sub[ q1_says_galaxy &  class_star_says_gal],
        'Both agree: STAR\n(Q1 model star & class_star star)':
            sub[~q1_says_galaxy & ~class_star_says_gal],
        'Disagree: Q1=GAL, class_star=STAR\n(model says galaxy, class_star says star)':
            sub[ q1_says_galaxy & ~class_star_says_gal],
        'Disagree: Q1=STAR, class_star=GAL\n(model says star, class_star says galaxy)':
            sub[~q1_says_galaxy &  class_star_says_gal],
    }

    colors = ['steelblue', 'coral', 'mediumseagreen', 'orchid']
    bins   = np.linspace(0, 4, 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for ax, (label, grp), color in zip(axes, groups.items(), colors):
        ratio = grp['ruwe_ratio'].clip(upper=bins[-1])
        n_binary = (grp['ruwe_ratio'] > 1).sum()
        frac_binary = n_binary / len(grp) * 100 if len(grp) > 0 else 0

        ax.hist(ratio, bins=bins, color=color, alpha=0.75, density=True)
        ax.axvline(x=1, color='black', ls='--', linewidth=1.5,
                   label='ruwe_ratio = 1 (threshold)')
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('RUWE / local threshold', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.text(0.97, 0.95,
                f'N = {len(grp):,}\nruwe_ratio > 1: {frac_binary:.1f}%',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis='y')

    plt.suptitle(
        'RUWE / local threshold by Q1 model vs class_star agreement (ERO sources, G ≥ 18.5)\n'
        'ruwe_ratio > 1 → likely unresolved binary candidate (GaiaUnlimited threshold)',
        fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    out = output_dir / 'ruwe_binary_diagnostic.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 5. Gaia selection function S(G) per source
# ---------------------------------------------------------------------------

def add_selection_function(df):
    """Compute Gaia DR3 S(G) at each source's sky position and G magnitude."""
    from gaiaunlimited import DR3SelectionFunctionTCG

    print("  Loading DR3SelectionFunctionTCG...")
    sf = DR3SelectionFunctionTCG(mode='multi')

    mask = df['phot_g_mean_mag'].notna() & df['ra'].notna() & df['dec'].notna()
    idx  = df.index[mask]
    ra_arr   = df.loc[idx, 'ra'].values
    dec_arr  = df.loc[idx, 'dec'].values
    gmag_arr = df.loc[idx, 'phot_g_mean_mag'].values.astype(float)

    n          = len(idx)
    batch_size = 50000
    sg_vals    = np.full(n, np.nan)

    print(f"  Computing S(G) for {n:,} sources in batches of {batch_size:,}...")
    for start in range(0, n, batch_size):
        end    = min(start + batch_size, n)
        coords = SkyCoord(ra=ra_arr[start:end] * u.deg,
                          dec=dec_arr[start:end] * u.deg)
        sg_vals[start:end] = sf.query(coords, gmag_arr[start:end])
        print(f"    Batch {start:,}–{end:,} done")

    df = df.copy()
    df['sg'] = np.nan
    df.loc[idx, 'sg'] = sg_vals

    print(f"  Median S(G): {np.nanmedian(sg_vals):.4f}")
    print(f"  S(G) < 0.9:  {(sg_vals < 0.9).sum():,}  "
          f"({(sg_vals < 0.9).mean()*100:.1f}%)")
    return df


# ---------------------------------------------------------------------------
# 6. Crowding check: cnt distributions by agreement group
# ---------------------------------------------------------------------------

def plot_cnt_by_agreement(df, output_dir):
    """1×4 panels (cnt4, cnt8, cnt16, cnt32): fraction histograms for the
    4 Q1-model vs class_star agreement groups. Tests whether disagreement
    cases are preferentially in crowded environments."""
    sub = df.dropna(subset=['q1_galaxy_prob', 'class_star',
                            'cnt4', 'cnt8', 'cnt16', 'cnt32'])

    q1_says_galaxy      = sub['q1_galaxy_prob'] > 0.5
    class_star_says_gal = sub['class_star'] < CLASS_STAR_THRESH

    groups = {
        'Both agree: GALAXY':            sub[ q1_says_galaxy &  class_star_says_gal],
        'Both agree: STAR':              sub[~q1_says_galaxy & ~class_star_says_gal],
        'Disagree: Q1=GAL, cs=STAR':     sub[ q1_says_galaxy & ~class_star_says_gal],
        'Disagree: Q1=STAR, cs=GAL':     sub[~q1_says_galaxy &  class_star_says_gal],
    }
    colors = ['steelblue', 'coral', 'mediumseagreen', 'orchid']

    cnt_cols   = ['cnt4',  'cnt8',  'cnt16',  'cnt32']
    cnt_labels = ['4"',    '8"',    '16"',    '32"']

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    for ax, cnt_col, radius_label in zip(axes, cnt_cols, cnt_labels):
        # Integer bins up to 99th percentile across all groups
        all_vals = sub[cnt_col].dropna()
        cnt_max  = max(int(all_vals.quantile(0.99)), 2)
        bins     = np.arange(0.5, cnt_max + 1.5, 1)

        for (label, grp), color in zip(groups.items(), colors):
            vals = grp[cnt_col].dropna()
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=bins, alpha=0.55, color=color,
                    weights=np.ones(len(vals)) / len(vals),
                    label=f'{label} (N={len(grp):,})')

        ax.set_xlabel(f'Neighbours within {radius_label}', fontsize=11)
        ax.set_ylabel('Fraction', fontsize=11)
        ax.set_title(f'cnt within {radius_label}', fontsize=12, fontweight='bold')
        ax.set_xticks(np.arange(1, cnt_max + 1))
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3, axis='y')

    plt.suptitle(
        'Neighbour count distributions by Q1 model vs class_star agreement (ERO, G ≥ 18.5)\n'
        'Are disagreement cases more crowded?',
        fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    out = output_dir / 'cnt_by_agreement.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

    # Print median crowding per group
    print("\n  Median cnt4 by agreement group:")
    for label, grp in groups.items():
        med = grp['cnt4'].median()
        frac_crowded = (grp['cnt4'] > 1).mean() * 100
        print(f"    {label:<40s}  median={med:.0f}  cnt4>1: {frac_crowded:.1f}%")


# ---------------------------------------------------------------------------
# 7. Selection function S(G) distributions by agreement group
# ---------------------------------------------------------------------------

def plot_sg_by_agreement(df, output_dir):
    """S(G) distributions for the 4 Q1-model vs class_star agreement groups.
    Tests whether disagreement cases preferentially occupy regions of low
    Gaia completeness (crowded / scanning-law-sparse positions)."""
    sub = df.dropna(subset=['q1_galaxy_prob', 'class_star', 'sg'])

    q1_says_galaxy      = sub['q1_galaxy_prob'] > 0.5
    class_star_says_gal = sub['class_star'] < CLASS_STAR_THRESH

    groups = {
        'Both agree: GALAXY':         sub[ q1_says_galaxy &  class_star_says_gal],
        'Both agree: STAR':           sub[~q1_says_galaxy & ~class_star_says_gal],
        'Disagree: Q1=GAL, cs=STAR':  sub[ q1_says_galaxy & ~class_star_says_gal],
        'Disagree: Q1=STAR, cs=GAL':  sub[~q1_says_galaxy &  class_star_says_gal],
    }
    colors = ['steelblue', 'coral', 'mediumseagreen', 'orchid']
    bins = np.linspace(0, 1, 51)

    fig, ax = plt.subplots(figsize=(9, 6))

    for (label, grp), color in zip(groups.items(), colors):
        vals = grp['sg'].dropna()
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, alpha=0.55, color=color,
                weights=np.ones(len(vals)) / len(vals),
                label=f'{label} (N={len(grp):,})')

    ax.axvline(x=0.9, color='black', ls='--', linewidth=1.2,
               label='S(G) = 0.9')
    ax.set_xlabel('S(G)  —  Gaia DR3 completeness at source position & G magnitude',
                  fontsize=11)
    ax.set_ylabel('Fraction', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3, axis='y')
    plt.suptitle(
        'Gaia S(G) completeness by Q1 model vs class_star agreement (ERO, G ≥ 18.5)\n'
        'Are disagreement cases in lower-completeness (more crowded) Gaia regions?',
        fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    out = output_dir / 'sg_by_agreement.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

    # Print summary
    print("\n  Median S(G) and low-completeness fraction by agreement group:")
    for label, grp in groups.items():
        vals = grp['sg'].dropna()
        med  = vals.median()
        frac_low = (vals < 0.9).mean() * 100
        print(f"    {label:<40s}  median S(G)={med:.4f}  S(G)<0.9: {frac_low:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER not set")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("RUWE binary diagnostic: Q1 model vs class_star disagreements")
    print("=" * 65)

    print("\n[1] Querying ERO data...")
    df = query_ero(username)

    print("\n[2] Applying Q1 model...")
    df = apply_q1_model(df)

    print("\n[3] Computing GaiaUnlimited RUWE thresholds...")
    df = add_ruwe_threshold(df)

    print("\n[4] Plotting RUWE diagnostic...")
    plot_diagnostic(df, OUTPUT_DIR)

    print("\n[5] Computing Gaia selection function S(G)...")
    df = add_selection_function(df)

    print("\n[6] Plotting crowding (cnt) by agreement group...")
    plot_cnt_by_agreement(df, OUTPUT_DIR)

    print("\n[7] Plotting S(G) completeness by agreement group...")
    plot_sg_by_agreement(df, OUTPUT_DIR)

    # Print summary fractions
    sub = df.dropna(subset=['ruwe_ratio', 'q1_galaxy_prob', 'class_star'])
    q1_gal         = sub['q1_galaxy_prob'] > 0.5
    class_star_gal = sub['class_star'] < CLASS_STAR_THRESH
    disagree = sub[q1_gal & ~class_star_gal]
    print(f"\n  Disagreement cases (Q1 model=gal, class_star=star): {len(disagree):,}")
    print(f"  Of those with ruwe_ratio > 1: "
          f"{(disagree['ruwe_ratio'] > 1).sum():,} "
          f"({(disagree['ruwe_ratio'] > 1).mean()*100:.1f}%)")
    print(f"  Overall ERO ruwe_ratio > 1: "
          f"{(sub['ruwe_ratio'] > 1).mean()*100:.1f}%")

    print("\n" + "=" * 65)
    return 0


if __name__ == '__main__':
    sys.exit(main())
