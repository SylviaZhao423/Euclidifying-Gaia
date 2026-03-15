#!/usr/bin/env python3
"""
Gaia DR3 completeness analysis for Q1 fields — produces:
  1. q1_completeness_multi_G_fields.png        — Mollweide maps at G=18,19,20,21,22
  2. q1_completeness_summary.png               — 3-panel summary using Gaia inclusion S(G)
  3. q1_per_field_sf.png                       — median S(G) vs Gaia G per Q1 field
  4. q1_per_field_galaxy_fraction_sf_corrected.png — raw vs Gaia-inclusion-corrected galaxy fraction

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_q1_completeness.py
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import healpy as hp
from astropy.coordinates import SkyCoord

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'output' / 'q1_completeness'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

ZPC_IE = 1.05
ZP_REF = 23.9

G_VALUES = [18, 19, 20, 21, 22]

Q1_FIELDS = {
    'EDF-North':  (269.733, +66.018, 22.9),
    'EDF-South':  ( 61.241, -48.423, 28.1),
    'EDF-Fornax': ( 52.932, -28.088, 12.1),
}

ERO_FIELDS = {
    'Horsehead':    (85.150,  -2.613,  0.58),
    'Messier 78':   (86.690,  -0.015,  0.60),
    'Taurus':       (64.983,  +28.023, 0.61),
    'Barnard 30':   (82.880,  +12.316, 0.60),
    'NGC 6254':     (254.303, -4.100,  0.60),
    'NGC 6397':     (265.174, -53.658, 0.61),
    'NGC 6822':     (296.236, -14.788, 0.60),
    'Holmberg II':  (124.790, +70.706, 0.60),
    'IC 10':        (  5.063, +59.288, 0.62),
    'IC 342':       ( 56.730, +68.084, 0.59),
    'NGC 2403':     (114.211, +65.586, 0.60),
    'NGC 6744':     (287.430, -63.842, 0.60),
    'Fornax':       ( 54.017, -35.267, 0.57),
    'Dorado':       ( 64.014, -55.780, 0.60),
    'Perseus':      ( 49.638, +41.651, 0.70),
    'Abell 2390':   (328.397, +17.709, 0.75),
    'Abell 2764':   (  5.713, -49.249, 0.75),
}

FIELD_COLORS = {
    'EDF-North':  '#1f77b4',
    'EDF-South':  '#ff7f0e',
    'EDF-Fornax': '#2ca02c',
}

DEFAULT_MIN_COMPLETENESS = 0.05


def get_galactic_b(ra, dec):
    return SkyCoord(ra=ra, dec=dec, unit='deg').galactic.b.deg


def assign_fields(df):
    """Assign each source to its nearest Q1 field (no radius cutoff — fields are large)."""
    ra  = df['ra'].values
    dec = df['dec'].values
    field_assignments = np.full(len(df), None, dtype=object)
    min_distances     = np.full(len(df), np.inf)

    for field_name, (field_ra, field_dec, _area) in Q1_FIELDS.items():
        ra1  = np.radians(ra);       dec1 = np.radians(dec)
        ra2  = np.radians(field_ra); dec2 = np.radians(field_dec)
        a    = (np.sin((dec2 - dec1) / 2)**2
                + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2)**2)
        dist = np.degrees(2 * np.arcsin(np.sqrt(a)))
        mask = dist < min_distances
        field_assignments[mask] = field_name
        min_distances[mask]     = dist[mask]

    df['field'] = field_assignments
    return df


# =============================================================================
# Data query
# =============================================================================

def query_q1_data(username):
    """Query Gaia-matched Q1+PHZ sources with Gaia G, I_E, RA/Dec, and PHZ label."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = f"""
        SELECT
            m_right_ascension AS ra,
            m_declination     AS dec,
            -2.5 * LOG10({ZPC_IE} * m_flux_detection_total) + {ZP_REF} AS ie_mag,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            CASE
                WHEN p_phz_classification = 1 THEN 'STAR'
                WHEN p_phz_classification = 2 THEN 'GALAXY'
                ELSE 'UNCLASSIFIED'
            END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE m_flux_detection_total > 0
          AND m_flux_detection_total IS NOT NULL
          AND m_vis_det = 1
    """
    print("  Querying Gaia-matched Q1 data...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  Retrieved {len(df):,} sources")
    return df


# =============================================================================
# Completeness maps (Mollweide)
# =============================================================================

def build_completeness_maps(sf, nside=64):
    """Build full-sky completeness maps for all G_VALUES."""
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra_deg  = np.degrees(phi)
    dec_deg = 90.0 - np.degrees(theta)

    batch = 10000
    coord_batches = []
    for i in range(0, npix, batch):
        j = min(i + batch, npix)
        coord_batches.append((i, j, SkyCoord(ra=ra_deg[i:j], dec=dec_deg[i:j], unit='deg')))

    maps = {}
    for g_val in G_VALUES:
        print(f"  Computing S(G={g_val}) for {npix} pixels...")
        completeness = np.zeros(npix)
        for i, j, coords in coord_batches:
            completeness[i:j] = sf.query(coords, np.full(j - i, float(g_val)))
        maps[g_val] = completeness
        print(f"    Done (min={completeness.min():.3f}, median={np.median(completeness):.3f})")
    return maps


def draw_field_circle(ra_center, dec_center, area_deg2, color, lw=1.0, n_pts=72):
    radius   = np.sqrt(area_deg2 / np.pi)
    theta    = np.linspace(0, 2 * np.pi, n_pts)
    cos_dec  = max(np.cos(np.radians(dec_center)), 0.01)
    ra_pts   = ra_center + radius * np.cos(theta) / cos_dec
    dec_pts  = np.clip(dec_center + radius * np.sin(theta), -89.9, 89.9)
    hp.projplot(ra_pts, dec_pts, lonlat=True, coord='CG',
                color=color, linewidth=lw, zorder=10)


def plot_completeness_map(maps, nside, output_dir):
    """1×5 Mollweide completeness maps with Q1 and ERO fields overlaid."""
    from matplotlib.lines import Line2D
    n   = len(G_VALUES)
    fig = plt.figure(figsize=(6 * n, 5))

    for idx, g_val in enumerate(G_VALUES):
        hp.mollview(maps[g_val], coord='CG', title=f'G = {g_val}',
                    cmap='viridis', min=0, max=1,
                    sub=(1, n, idx + 1), fig=fig,
                    notext=True, cbar=False)
        for name, (ra, dec, area) in Q1_FIELDS.items():
            draw_field_circle(ra, dec, area, 'red', lw=1.5)
        for name, (ra, dec, area) in ERO_FIELDS.items():
            draw_field_circle(ra, dec, area, '#FF69B4', lw=0.8)

    fig.subplots_adjust(bottom=0.12)
    cax = fig.add_axes([0.25, 0.05, 0.5, 0.025])
    sm  = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, cax=cax, orientation='horizontal').set_label(
        'Completeness S(G)', fontsize=11)

    legend_elements = [
        Line2D([0], [0], color='red',    linewidth=2, label='Q1 fields'),
        Line2D([0], [0], color='#FF69B4', linewidth=2, label='ERO fields'),
    ]
    fig.legend(handles=legend_elements, loc='lower right', fontsize=9,
               framealpha=0.9, edgecolor='gray', bbox_to_anchor=(0.92, 0.02))

    fig.suptitle('Gaia DR3 Completeness by Magnitude with Euclid Q1 & ERO Fields',
                 fontsize=14, fontweight='bold', y=0.99)
    out = output_dir / 'q1_completeness_multi_G_fields.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out.name}")


def print_completeness_table(maps, nside):
    header = f"{'Field':<15}" + ''.join(f"  G={g:>2}" for g in G_VALUES)
    print(f"\n{'='*70}")
    print("COMPLETENESS S(G) AT Q1 FIELD POSITIONS")
    print(f"{'='*70}")
    print(header)
    print("-" * 70)
    for name, (ra, dec, area) in Q1_FIELDS.items():
        pix  = hp.ang2pix(nside, np.radians(90.0 - dec), np.radians(ra))
        vals = ''.join(f"  {maps[g][pix]:>5.3f}" for g in G_VALUES)
        print(f"  {name:<15}{vals}")


# =============================================================================
# Per-source completeness
# =============================================================================

def compute_per_source_completeness(df, sf):
    """Compute S(G) at each source's actual RA/Dec."""
    mask = df['field'].notna() & df['phot_g_mean_mag'].notna()
    idx  = df.index[mask]
    ra_arr   = df.loc[idx, 'ra'].values
    dec_arr  = df.loc[idx, 'dec'].values
    gmag_arr = df.loc[idx, 'phot_g_mean_mag'].values.astype(float)

    n = len(idx)
    print(f"  Querying S(G) for {n:,} sources...")
    batch_size   = 50000
    completeness = np.full(n, np.nan)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        coords = SkyCoord(ra=ra_arr[start:end], dec=dec_arr[start:end], unit='deg')
        completeness[start:end] = sf.query(coords, gmag_arr[start:end])
        print(f"    Batch {start:,}–{end:,} done")

    df['completeness'] = np.nan
    df.loc[idx, 'completeness'] = completeness
    print(f"  Median S(G): {np.nanmedian(completeness):.3f}  |  "
          f"S(G)<0.5: {(completeness < 0.5).sum():,}")


def build_ipw_weights(s_values, min_completeness):
    """
    Clipped inverse-probability weights for Gaia catalogue inclusion.

    This avoids zeroing faint-end sources entirely while still preventing
    arbitrarily large weights when S(G) becomes very small.
    """
    s = np.asarray(s_values, dtype=float)
    valid = np.isfinite(s) & (s > 0)
    weights = np.zeros_like(s, dtype=float)
    if not np.any(valid):
        return weights
    s_eff = np.maximum(s[valid], float(min_completeness))
    weights[valid] = 1.0 / s_eff
    return weights


# =============================================================================
# Plot 2: completeness summary (3 panels)
# =============================================================================

def plot_completeness_summary(df, output_dir, min_completeness):
    """3-panel: S(G) scatter and Gaia-inclusion-corrected summary statistics."""
    df_gal = df[(df['field'].notna()) & (df['label'] == 'GALAXY')
                & df['phot_g_mean_mag'].notna() & df['completeness'].notna()].copy()

    bins = np.arange(14, 22.5, 0.25)
    bc   = (bins[:-1] + bins[1:]) / 2

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- Panel 1: per-source S(G) scatter ---
    ax = axes[0]
    for field_name in Q1_FIELDS:
        fd = df_gal[df_gal['field'] == field_name]
        if len(fd):
            ax.scatter(fd['phot_g_mean_mag'], fd['completeness'],
                       s=1, alpha=0.3, color=FIELD_COLORS[field_name],
                       label=f'{field_name} (n={len(fd):,})', rasterized=True)
    ax.set_xlabel('Gaia G', fontsize=11, fontweight='bold')
    ax.set_ylabel('S(G) at source position', fontsize=11, fontweight='bold')
    ax.set_title('Per-Source Completeness', fontsize=12, fontweight='bold')
    ax.set_xlim(14, 22); ax.set_ylim(-0.02, 1.05)
    ax.axhline(y=0.5, color='gray', ls='--', alpha=0.5)
    ax.legend(fontsize=8, loc='lower left', markerscale=5)
    ax.grid(axis='y', alpha=0.3)

    # --- Panel 2: raw vs Gaia-inclusion-corrected galaxy fraction ---
    ax = axes[1]
    df_field = df[(df['field'].notna()) & df['phot_g_mean_mag'].notna()
                  & df['completeness'].notna()].copy()
    df_star  = df_field[df_field['label'] == 'STAR']

    raw_gal,  _ = np.histogram(df_gal['phot_g_mean_mag'], bins=bins)
    raw_star, _ = np.histogram(df_star['phot_g_mean_mag'], bins=bins)
    raw_all = raw_gal + raw_star
    with np.errstate(divide='ignore', invalid='ignore'):
        raw_frac = np.where(raw_all > 0, raw_gal / raw_all, np.nan)

    def _whist(sub):
        s = sub['completeness'].values
        g = sub['phot_g_mean_mag'].values
        w = build_ipw_weights(s, min_completeness)
        c, _ = np.histogram(g, bins=bins, weights=w)
        return c

    corr_gal  = _whist(df_gal)
    corr_star = _whist(df_star)
    corr_all  = corr_gal + corr_star
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_frac = np.where(corr_all > 0, corr_gal / corr_all, np.nan)

    ax.plot(bc[raw_all >= 10],  raw_frac[raw_all >= 10],  'o-', color='steelblue',
            markersize=3, linewidth=2, label='Raw')
    ax.plot(bc[corr_all > 0],   corr_frac[corr_all > 0],  's--', color='coral',
            markersize=3, linewidth=2,
            label=f'Gaia-inclusion-corrected (1/max[S,{min_completeness:.2f}])')
    ax.set_xlabel('Gaia G', fontsize=11, fontweight='bold')
    ax.set_ylabel('Galaxy fraction', fontsize=11, fontweight='bold')
    ax.set_title('Raw vs Gaia-Inclusion-Corrected Galaxy Fraction',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(14, 22); ax.set_ylim(-0.02, 1.02)
    ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)

    # --- Panel 3: raw vs Gaia-inclusion-corrected galaxy density ---
    ax = axes[2]
    raw_gal3, _  = np.histogram(df_gal['phot_g_mean_mag'], bins=bins)
    s_gal = df_gal['completeness'].values
    g_gal = df_gal['phot_g_mean_mag'].values
    w_gal = build_ipw_weights(s_gal, min_completeness)
    corr_gal3, _ = np.histogram(g_gal, bins=bins, weights=w_gal)
    raw_norm  = raw_gal3  / raw_gal3.sum()  if raw_gal3.sum()  > 0 else raw_gal3
    corr_norm = corr_gal3 / corr_gal3.sum() if corr_gal3.sum() > 0 else corr_gal3
    bw = bins[1] - bins[0]
    ax.bar(bc, raw_norm,  width=bw, alpha=0.5, color='steelblue',
           edgecolor='white', linewidth=0.5, label='Raw')
    ax.bar(bc, corr_norm, width=bw, alpha=0.5, color='coral',
           edgecolor='white', linewidth=0.5,
           label=f'Gaia-inclusion-corrected (1/max[S,{min_completeness:.2f}])')
    ax.set_xlabel('Gaia G', fontsize=11, fontweight='bold')
    ax.set_ylabel('Normalized density', fontsize=11, fontweight='bold')
    ax.set_title('Raw vs Gaia-Inclusion-Corrected Galaxy Density',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(14, 22); ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)

    plt.suptitle(
        f'Gaia Completeness & Q1 Galaxy Fraction '
        f'(Gaia-inclusion correction; weight floor S={min_completeness:.2f})',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = output_dir / 'q1_completeness_summary.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out.name}")


# =============================================================================
# Plot 3: per-field median S(G)
# =============================================================================

def plot_selection_function_per_field(df, output_dir):
    """1×3: median S(G) vs Gaia G per Q1 field (median ± 16–84%)."""
    bins = np.arange(14, 22.5, 0.25)
    bc   = (bins[:-1] + bins[1:]) / 2

    sub = df[(df['field'].notna()) & df['phot_g_mean_mag'].notna()
             & df['completeness'].notna()].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, field_name in zip(axes, Q1_FIELDS):
        gal_b = get_galactic_b(*Q1_FIELDS[field_name][:2])
        color = FIELD_COLORS[field_name]
        fd    = sub[sub['field'] == field_name]
        g_arr = fd['phot_g_mean_mag'].values
        s_arr = fd['completeness'].values

        median_s = np.full(len(bins) - 1, np.nan)
        p16      = np.full(len(bins) - 1, np.nan)
        p84      = np.full(len(bins) - 1, np.nan)
        for i in range(len(bins) - 1):
            in_bin = (g_arr >= bins[i]) & (g_arr < bins[i + 1])
            if in_bin.sum() >= 5:
                median_s[i] = np.median(s_arr[in_bin])
                p16[i]      = np.percentile(s_arr[in_bin], 16)
                p84[i]      = np.percentile(s_arr[in_bin], 84)

        mask = ~np.isnan(median_s)
        ax.plot(bc[mask], median_s[mask], 'o-', color=color, markersize=3, linewidth=1.5)
        ax.fill_between(bc[mask], p16[mask], p84[mask], alpha=0.2, color=color)
        ax.set_title(f'{field_name}  (|b|={abs(gal_b):.1f}°)\nN={len(fd):,}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Gaia G', fontsize=10)
        ax.set_ylabel('S(G)', fontsize=10)
        ax.set_xlim(14, 22); ax.set_ylim(-0.02, 1.05)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Q1 Fields: Gaia Selection Function S(G) vs Gaia G (median ± 16–84%)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = output_dir / 'q1_per_field_sf.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out.name}")


# =============================================================================
# Plot 4: per-field galaxy fraction raw vs S(G)-corrected
# =============================================================================

def plot_galaxy_fraction_sg_per_field(df, output_dir, min_completeness):
    """1×3: galaxy fraction vs Gaia G, raw and Gaia-inclusion-corrected per Q1 field."""
    bins     = np.arange(14, 22.5, 0.25)
    bc       = (bins[:-1] + bins[1:]) / 2

    sub = df[(df['field'].notna()) & df['phot_g_mean_mag'].notna()
             & df['completeness'].notna() & df['label'].notna()].copy()
    # Restrict to PHZ-classified only for a cleaner fraction
    sub = sub[sub['label'].isin(['STAR', 'GALAXY'])]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, field_name in zip(axes, Q1_FIELDS):
        gal_b = get_galactic_b(*Q1_FIELDS[field_name][:2])
        fd    = sub[sub['field'] == field_name]
        g_arr = fd['phot_g_mean_mag'].values
        s_arr = fd['completeness'].values
        is_gal = fd['label'].values == 'GALAXY'

        n_gal_raw, _ = np.histogram(g_arr[is_gal], bins=bins)
        n_all_raw, _ = np.histogram(g_arr, bins=bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            raw_frac = np.where(n_all_raw >= 10, n_gal_raw / n_all_raw, np.nan)

        weights = build_ipw_weights(s_arr, min_completeness)
        corr_gal, _ = np.histogram(g_arr[is_gal], bins=bins, weights=weights[is_gal])
        corr_all, _ = np.histogram(g_arr, bins=bins, weights=weights)
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_frac = np.where(corr_all > 0, corr_gal / corr_all, np.nan)

        mask_raw  = ~np.isnan(raw_frac)
        mask_corr = ~np.isnan(corr_frac)
        mask_both = mask_raw & mask_corr

        ax.plot(bc[mask_raw],  raw_frac[mask_raw],   'o-', color='steelblue',
                markersize=3, linewidth=1.5, label='Raw')
        ax.plot(bc[mask_corr], corr_frac[mask_corr], 's-', color='coral',
                markersize=3, linewidth=1.5,
                label=f'Gaia-inclusion-corrected (1/max[S,{min_completeness:.2f}])')
        if mask_both.any():
            ax.fill_between(bc[mask_both], raw_frac[mask_both], corr_frac[mask_both],
                            alpha=0.2, color='coral')

        n_gal  = is_gal.sum()
        n_star = (~is_gal).sum()
        ax.set_title(f'{field_name}  (|b|={abs(gal_b):.1f}°)\n'
                     f'Stars {n_star:,}  Galaxies {n_gal:,}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Gaia G', fontsize=10)
        ax.set_ylabel('Galaxy fraction', fontsize=10)
        ax.set_xlim(14, 22); ax.set_ylim(-0.02, 1.02)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(
        f'Q1 Fields: Galaxy Fraction vs Gaia G '
        f'(Raw & Gaia-inclusion-corrected; weight floor S={min_completeness:.2f})',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = output_dir / 'q1_per_field_galaxy_fraction_sf_corrected.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out.name}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Q1 Gaia completeness analysis using GaiaUnlimited DR3SelectionFunctionTCG.')
    parser.add_argument(
        '--min-completeness',
        type=float,
        default=DEFAULT_MIN_COMPLETENESS,
        help=('Floor used in clipped inverse-probability weights: '
              'weight = 1 / max(S(G), floor). Default: %(default)s'))
    return parser.parse_args()


def main():
    args = parse_args()
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Q1 Gaia DR3 Completeness Analysis")
    print("=" * 70)

    from gaiaunlimited import DR3SelectionFunctionTCG
    print("\n[1] Loading DR3SelectionFunctionTCG...")
    sf = DR3SelectionFunctionTCG(mode='multi')

    nside = 64
    print("\n[2] Building full-sky completeness maps...")
    maps = build_completeness_maps(sf, nside=nside)
    print_completeness_table(maps, nside)

    print("\n[3] Generating Mollweide plot...")
    plot_completeness_map(maps, nside, OUTPUT_DIR)

    print("\n[4] Querying Q1 Gaia-matched data...")
    df = query_q1_data(username)
    df = assign_fields(df)
    print(f"  Assigned to fields: {df['field'].notna().sum():,} sources")

    print("\n[5] Computing per-source completeness...")
    compute_per_source_completeness(df, sf)

    print("\n[6] Generating completeness summary...")
    plot_completeness_summary(df, OUTPUT_DIR, args.min_completeness)

    print("\n[7] Generating per-field selection function...")
    plot_selection_function_per_field(df, OUTPUT_DIR)

    print("\n[8] Generating per-field galaxy fraction (Gaia-inclusion-corrected)...")
    plot_galaxy_fraction_sg_per_field(df, OUTPUT_DIR, args.min_completeness)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
