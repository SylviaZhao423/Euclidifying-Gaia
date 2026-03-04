#!/usr/bin/env python3
"""
Plot the match rate per ERO field as a function of magnitude:
  1. match_rate_gaia.png     — N_matched / N_gaia vs Gaia G per field
  2. match_rate_ero.png      — N_matched / N_ero vs Euclid VIS per field
  3. match_rate_combined.png — R_gaia × R_ero combined rate per field
  4. galaxy_fraction_corrected.png — galaxy fraction vs Gaia G, raw & corrected
     by 1/(R_gaia × R_ero) joint selection

Usage:
    PGUSER=siwei_zhao_2025 python src/plot_match_rate.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / 'output' / 'ero_completeness' / 'match_rate_test'

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

# ERO field definitions: name -> (RA, Dec, area_deg2)
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

FIELD_CATEGORIES = {
    'Star-forming': ['Horsehead', 'Messier 78', 'Taurus', 'Barnard 30'],
    'Globular Clusters': ['NGC 6254', 'NGC 6397'],
    'Nearby Galaxies': ['NGC 6822', 'Holmberg II', 'IC 10', 'IC 342', 'NGC 2403', 'NGC 6744'],
    'Galaxy Groups': ['Fornax', 'Dorado'],
    'Galaxy Clusters': ['Perseus', 'Abell 2390', 'Abell 2764'],
}

CATEGORY_COLORS = {
    'Star-forming': '#d62728',
    'Globular Clusters': '#2ca02c',
    'Nearby Galaxies': '#1f77b4',
    'Galaxy Groups': '#9467bd',
    'Galaxy Clusters': '#ff7f0e',
}


def get_field_color(field_name):
    for category, fields in FIELD_CATEGORIES.items():
        if field_name in fields:
            return CATEGORY_COLORS[category]
    return 'gray'


def get_galactic_b(ra, dec):
    c = SkyCoord(ra=ra, dec=dec, unit='deg')
    return c.galactic.b.deg


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


# =============================================================================
# Queries
# =============================================================================

def query_matched(username):
    """Gaia-ERO crossmatch sources."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = """
        SELECT
            e_v_alpha_j2000 AS ra,
            e_v_delta_j2000 AS dec,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            e_v_mag_auto AS v_mag_auto,
            e_v_class_star AS class_star,
            CASE
                WHEN e_v_class_star >= 0.65 THEN 'STAR'
                WHEN e_v_class_star < 0.65 THEN 'GALAXY'
            END as label
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_class_star IS NOT NULL
    """
    print("  Querying matched sources...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  ✓ {len(df):,} matched sources")
    return df


def query_all_ero(username):
    """All ERO sources."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    query = """
        SELECT
            v_alpha_j2000 AS ra,
            v_delta_j2000 AS dec,
            v_mag_auto
        FROM euclid_ero.merge_cat
        WHERE v_class_star IS NOT NULL
    """
    print("  Querying all ERO sources...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"  ✓ {len(df):,} ERO sources")
    return df


def query_all_gaia_in_fields(username):
    """All Gaia DR3 sources within ERO field footprints via q3c cone search."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)

    frames = []
    for field_name, (field_ra, field_dec, area) in ERO_FIELDS.items():
        radius = np.sqrt(area / np.pi)
        query = f"""
            SELECT ra, dec, phot_g_mean_mag
            FROM gaia_dr3.gaia_source
            WHERE q3c_radial_query(ra, dec, {field_ra}, {field_dec}, {radius})
        """
        print(f"    {field_name} (r={radius:.3f}°)...", end='', flush=True)
        df_f = pd.read_sql_query(query, conn)
        df_f['field'] = field_name
        print(f" {len(df_f):,} sources")
        frames.append(df_f)

    conn.close()
    df = pd.concat(frames, ignore_index=True)
    print(f"  ✓ {len(df):,} total Gaia sources in ERO fields")
    return df


# =============================================================================
# Plotting
# =============================================================================

def plot_match_rates(df_matched, df_gaia, df_ero, output_dir):
    """Plot match rate vs magnitude per field."""

    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols

    # --- Panel A: N_matched / N_gaia vs Gaia G ---
    bins_g = np.arange(14, 22.5, 0.25)
    bc_g = (bins_g[:-1] + bins_g[1:]) / 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])
        color = get_field_color(field_name)

        gaia_f = df_gaia[df_gaia['field'] == field_name]['phot_g_mean_mag'].dropna()
        match_f = df_matched[df_matched['field'] == field_name]['phot_g_mean_mag'].dropna()

        n_gaia, _ = np.histogram(gaia_f, bins=bins_g)
        n_match, _ = np.histogram(match_f, bins=bins_g)

        with np.errstate(divide='ignore', invalid='ignore'):
            rate = np.where(n_gaia > 0, n_match / n_gaia, np.nan)

        mask = n_gaia >= 5
        ax.plot(bc_g[mask], rate[mask], 'o-', color=color, markersize=3, linewidth=1.5)

        n_g_tot = len(gaia_f)
        n_m_tot = len(match_f)
        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)\n'
                     f'Gaia: {n_g_tot:,}, matched: {n_m_tot:,}',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Gaia G', fontsize=9)
        ax.set_ylabel('N_matched / N_Gaia', fontsize=9)
        ax.set_xlim(14, 22)
        # Auto-scale y to data range with padding
        valid_rates = rate[mask]
        valid_rates = valid_rates[~np.isnan(valid_rates)]
        if len(valid_rates) > 0:
            ymin = max(0, valid_rates.min() - 0.05 * (valid_rates.max() - valid_rates.min() + 0.01))
            ymax = min(1.02, valid_rates.max() + 0.05 * (valid_rates.max() - valid_rates.min() + 0.01))
            ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Match Rate: Fraction of Gaia Sources with ERO Match (vs Gaia G)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'match_rate_gaia.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ match_rate_gaia.png")

    # --- Panel B: N_matched / N_ero vs Euclid VIS ---
    bins_v = np.arange(14, 26.75, 0.25)
    bc_v = (bins_v[:-1] + bins_v[1:]) / 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])
        color = get_field_color(field_name)

        ero_f = df_ero[df_ero['field'] == field_name]['v_mag_auto'].dropna()
        match_f = df_matched[df_matched['field'] == field_name]['v_mag_auto'].dropna()

        n_ero, _ = np.histogram(ero_f, bins=bins_v)
        n_match, _ = np.histogram(match_f, bins=bins_v)

        with np.errstate(divide='ignore', invalid='ignore'):
            rate = np.where(n_ero > 0, n_match / n_ero, np.nan)

        mask = n_ero >= 5
        ax.plot(bc_v[mask], rate[mask], 'o-', color=color, markersize=2, linewidth=1.5)

        n_e_tot = len(ero_f)
        n_m_tot = len(match_f)
        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)\n'
                     f'ERO: {n_e_tot:,}, matched: {n_m_tot:,}',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Euclid VIS mag', fontsize=9)
        ax.set_ylabel('N_matched / N_ERO', fontsize=9)
        ax.set_xlim(14, 26.5)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.grid(alpha=0.3)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Match Rate: Fraction of ERO Sources with Gaia Match (vs Euclid VIS)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'match_rate_ero.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ match_rate_ero.png")

    # --- Combined per-field: R_gaia, R_ero, and R_combined = R_gaia × R_ero ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])
        color = get_field_color(field_name)

        # Compute R_gaia(G) per G bin
        gaia_f = df_gaia[df_gaia['field'] == field_name]['phot_g_mean_mag'].dropna()
        match_f_g = df_matched[df_matched['field'] == field_name]['phot_g_mean_mag'].dropna()
        n_gaia, _ = np.histogram(gaia_f, bins=bins_g)
        n_match_g, _ = np.histogram(match_f_g, bins=bins_g)
        with np.errstate(divide='ignore', invalid='ignore'):
            r_gaia = np.where(n_gaia > 0, n_match_g / n_gaia, np.nan)

        # Compute R_ero(VIS) per VIS bin
        ero_f = df_ero[df_ero['field'] == field_name]['v_mag_auto'].dropna()
        match_f_v = df_matched[df_matched['field'] == field_name]['v_mag_auto'].dropna()
        n_ero, _ = np.histogram(ero_f, bins=bins_v)
        n_match_v, _ = np.histogram(match_f_v, bins=bins_v)
        with np.errstate(divide='ignore', invalid='ignore'):
            r_ero = np.where(n_ero > 0, n_match_v / n_ero, np.nan)

        # Compute combined rate per VIS bin:
        # For matched sources in each VIS bin, look up their R_gaia from their G bin
        match_field = df_matched[df_matched['field'] == field_name].dropna(
            subset=['phot_g_mean_mag', 'v_mag_auto'])
        g_vals = match_field['phot_g_mean_mag'].values
        v_vals = match_field['v_mag_auto'].values

        # Assign each matched source its R_gaia based on G bin
        g_bin_idx = np.digitize(g_vals, bins_g) - 1
        g_bin_idx = np.clip(g_bin_idx, 0, len(r_gaia) - 1)
        per_source_r_gaia = r_gaia[g_bin_idx]

        # Assign each matched source its R_ero based on VIS bin
        v_bin_idx = np.digitize(v_vals, bins_v) - 1
        v_bin_idx = np.clip(v_bin_idx, 0, len(r_ero) - 1)
        per_source_r_ero = r_ero[v_bin_idx]

        # Combined rate per source
        per_source_combined = per_source_r_gaia * per_source_r_ero

        # Average combined rate per VIS bin
        combined_rate = np.full(len(bins_v) - 1, np.nan)
        for i in range(len(bins_v) - 1):
            in_bin = (v_vals >= bins_v[i]) & (v_vals < bins_v[i + 1])
            if in_bin.sum() >= 5:
                combined_rate[i] = np.nanmean(per_source_combined[in_bin])

        # Plot
        mask_ero = n_ero >= 5
        mask_comb = ~np.isnan(combined_rate)
        ax.plot(bc_v[mask_ero], r_ero[mask_ero], '-', color='steelblue',
                alpha=0.7, linewidth=1.5, label='R_ero')
        ax.plot(bc_v[mask_comb], combined_rate[mask_comb], '-', color='coral',
                alpha=0.7, linewidth=1.5, label='R_gaia × R_ero')

        n_e_tot = len(ero_f)
        n_m_tot = len(match_f_v)
        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)\n'
                     f'ERO: {n_e_tot:,}, matched: {n_m_tot:,}',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Euclid VIS mag', fontsize=9)
        ax.set_ylabel('Match rate', fontsize=9)
        ax.set_xlim(14, 26.5)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.3)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Combined Match Rate per Field (R_gaia × R_ero vs Euclid VIS)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'match_rate_combined.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ match_rate_combined.png")


def plot_corrected_galaxy_fraction(df_matched, df_gaia, df_ero, output_dir):
    """Galaxy fraction vs Gaia G per field, with combined selection correction."""
    n_fields = len(ERO_FIELDS)
    n_cols = 3
    n_rows = (n_fields + n_cols - 1) // n_cols

    bins_g = np.arange(14, 22.5, 0.25)
    bins_v = np.arange(14, 26.75, 0.25)
    bc_g = (bins_g[:-1] + bins_g[1:]) / 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()

    for idx, field_name in enumerate(ERO_FIELDS.keys()):
        ax = axes[idx]
        gal_b = get_galactic_b(*ERO_FIELDS[field_name][:2])
        color = get_field_color(field_name)

        # Field subsets
        match_f = df_matched[df_matched['field'] == field_name].dropna(
            subset=['phot_g_mean_mag', 'v_mag_auto', 'label'])
        gaia_f = df_gaia[df_gaia['field'] == field_name]['phot_g_mean_mag'].dropna()
        ero_f = df_ero[df_ero['field'] == field_name]['v_mag_auto'].dropna()

        # Compute R_gaia(G) and R_ero(VIS) per bin
        n_gaia, _ = np.histogram(gaia_f, bins=bins_g)
        n_match_g, _ = np.histogram(match_f['phot_g_mean_mag'], bins=bins_g)
        with np.errstate(divide='ignore', invalid='ignore'):
            r_gaia = np.where(n_gaia > 0, n_match_g / n_gaia, np.nan)

        n_ero, _ = np.histogram(ero_f, bins=bins_v)
        n_match_v, _ = np.histogram(match_f['v_mag_auto'], bins=bins_v)
        with np.errstate(divide='ignore', invalid='ignore'):
            r_ero = np.where(n_ero > 0, n_match_v / n_ero, np.nan)

        # --- Raw galaxy fraction per G bin ---
        gal_g = match_f[match_f['label'] == 'GALAXY']['phot_g_mean_mag'].values
        all_g = match_f['phot_g_mean_mag'].values
        n_gal_raw, _ = np.histogram(gal_g, bins=bins_g)
        n_all_raw, _ = np.histogram(all_g, bins=bins_g)
        with np.errstate(divide='ignore', invalid='ignore'):
            raw_frac = np.where(n_all_raw >= 10, n_gal_raw / n_all_raw, np.nan)

        # --- Corrected galaxy fraction: weight each source by 1/(R_gaia × R_ero) ---
        g_vals = match_f['phot_g_mean_mag'].values
        v_vals = match_f['v_mag_auto'].values
        labels = match_f['label'].values

        # Look up R_gaia for each source from its G bin
        g_idx = np.clip(np.digitize(g_vals, bins_g) - 1, 0, len(r_gaia) - 1)
        src_r_gaia = r_gaia[g_idx]

        # Look up R_ero for each source from its VIS bin
        v_idx = np.clip(np.digitize(v_vals, bins_v) - 1, 0, len(r_ero) - 1)
        src_r_ero = r_ero[v_idx]

        # Combined weight = 1 / (R_gaia × R_ero), skip where rate is too low
        src_r_combined = src_r_gaia * src_r_ero
        valid = (src_r_combined > 0.01) & np.isfinite(src_r_combined)
        weights = np.where(valid, 1.0 / src_r_combined, 0.0)

        is_gal = labels == 'GALAXY'
        corr_gal, _ = np.histogram(g_vals[valid & is_gal], bins=bins_g,
                                    weights=weights[valid & is_gal])
        corr_all, _ = np.histogram(g_vals[valid], bins=bins_g,
                                    weights=weights[valid])
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_frac = np.where(corr_all > 0, corr_gal / corr_all, np.nan)

        # Plot with distinct colors and shaded difference
        mask_raw = ~np.isnan(raw_frac)
        mask_corr = ~np.isnan(corr_frac)
        mask_both = mask_raw & mask_corr

        ax.plot(bc_g[mask_raw], raw_frac[mask_raw], 'o-', color='steelblue',
                markersize=3, linewidth=1.5, label='Raw')
        ax.plot(bc_g[mask_corr], corr_frac[mask_corr], 's-', color='coral',
                markersize=3, linewidth=1.5, label='Corrected')

        # Shade the difference
        if mask_both.any():
            ax.fill_between(bc_g[mask_both], raw_frac[mask_both], corr_frac[mask_both],
                            alpha=0.2, color='coral')

        ax.set_title(f'{field_name} (|b|={abs(gal_b):.1f}°)',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Gaia G', fontsize=9)
        ax.set_ylabel('Galaxy fraction', fontsize=9)
        ax.set_xlim(14, 22)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(alpha=0.3)

    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Galaxy Fraction vs Gaia G by Field (Raw & Selection-Corrected)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_field_galaxy_fraction_mr_corrected.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ galaxy_fraction_corrected.png")


# =============================================================================
# Main
# =============================================================================

def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ERO–Gaia Match Rate Analysis")
    print("=" * 70)

    # Query data
    print("\n[1] Querying matched sources...")
    df_matched = query_matched(username)
    df_matched = assign_fields(df_matched)

    print("\n[2] Querying all ERO sources...")
    df_ero = query_all_ero(username)
    df_ero = assign_fields(df_ero)

    print("\n[3] Querying all Gaia sources in ERO fields...")
    df_gaia = query_all_gaia_in_fields(username)

    # Summary
    for name in ERO_FIELDS:
        n_g = (df_gaia['field'] == name).sum()
        n_e = (df_ero['field'] == name).sum()
        n_m = (df_matched['field'] == name).sum()
        print(f"  {name:<15} Gaia: {n_g:>8,}  ERO: {n_e:>8,}  Matched: {n_m:>7,}")

    # Plot
    print("\n[4] Plotting match rates...")
    plot_match_rates(df_matched, df_gaia, df_ero, OUTPUT_DIR)

    print("\n[5] Plotting corrected galaxy fraction...")
    plot_corrected_galaxy_fraction(df_matched, df_gaia, df_ero, OUTPUT_DIR)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
