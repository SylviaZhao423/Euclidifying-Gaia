#!/usr/bin/env python3
"""
Download relevant ERO-Gaia cross-matched data for morphology analysis.

Extracts morphology-relevant columns from the cross-matched table
(siwei_zhao_2025.ero_gaia_crossmatch) and saves to parquet for local analysis.

Columns include:
- Positions (ERO and Gaia)
- Morphology classifiers (ERO: class_star, spread_model; Gaia: classprob_star)
- Magnitudes (for magnitude-dependent analysis)
- Photometry (for SNR calculation)
- Astrometry quality metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'output'

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# WSDB Configuration
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'


def download_morphology_subset(username):
    """
    Download morphology-relevant columns from ERO-Gaia cross-match table.

    Args:
        username: WSDB username (from PGUSER environment variable)

    Returns:
        DataFrame with morphology data
    """
    try:
        import sqlutilpy
    except ImportError:
        print("ERROR: sqlutilpy not installed!")
        print("Install it using: pip install sqlutilpy")
        return None

    table_name = f'{username}.ero_gaia_crossmatch'

    print(f"Downloading morphology subset from {table_name}...")

    # SQL query to get relevant columns for morphology analysis
    query = f"""
        SELECT
            -- Position (primary identifier)
            v_alpha_j2000 as ero_ra,
            v_delta_j2000 as ero_dec,
            ra as gaia_ra,
            dec as gaia_dec,

            -- Identifiers
            source_id as gaia_source_id,

            -- ERO morphology classifiers
            v_class_star as ero_class_star,
            v_spread_model as ero_spread_model,
            v_spreaderr_model as ero_spreaderr_model,

            -- Gaia morphology classifiers
            classprob_dsc_combmod_star as gaia_classprob_star,
            classprob_dsc_combmod_galaxy as gaia_classprob_galaxy,
            classprob_dsc_combmod_quasar as gaia_classprob_quasar,

            -- Magnitudes (for magnitude-dependent analysis)
            v_mag_auto as ero_mag_auto,
            phot_g_mean_mag as gaia_mag_g,
            phot_bp_mean_mag as gaia_mag_bp,
            phot_rp_mean_mag as gaia_mag_rp,

            -- Photometry (for SNR calculation)
            v_flux_auto as ero_flux_auto,
            v_fluxerr_auto as ero_fluxerr_auto,
            phot_g_mean_flux as gaia_flux_g,
            phot_g_mean_flux_error as gaia_flux_error_g,

            -- Astrometry quality
            astrometric_excess_noise as gaia_astrometric_excess_noise,
            ruwe as gaia_ruwe
        FROM {table_name}
        WHERE source_id IS NOT NULL
    """

    try:
        print("Executing query...")
        result = sqlutilpy.get(
            query,
            db=WSDB_DB,
            host=WSDB_HOST
        )

        if result and len(result[0]) > 0:
            print(f"✓ Query successful! Downloaded {len(result[0])} rows")

            # Define column names in order matching the query
            columns = [
                'ero_ra', 'ero_dec', 'gaia_ra', 'gaia_dec',
                'gaia_source_id',
                'ero_class_star', 'ero_spread_model', 'ero_spreaderr_model',
                'gaia_classprob_star', 'gaia_classprob_galaxy', 'gaia_classprob_quasar',
                'ero_mag_auto', 'gaia_mag_g', 'gaia_mag_bp', 'gaia_mag_rp',
                'ero_flux_auto', 'ero_fluxerr_auto', 'gaia_flux_g', 'gaia_flux_error_g',
                'gaia_astrometric_excess_noise', 'gaia_ruwe'
            ]

            # Create DataFrame
            df = pd.DataFrame({
                col: result[i] for i, col in enumerate(columns)
            })

            print(f"\nDataFrame shape: {df.shape}")
            print(f"Columns: {len(df.columns)}")

            return df
        else:
            print("No results returned from query")
            return None

    except Exception as e:
        print(f"ERROR executing query: {e}")
        return None


def calculate_derived_columns(df):
    """
    Calculate derived quantities useful for morphology analysis.

    Args:
        df: DataFrame with morphology data

    Returns:
        DataFrame with additional derived columns
    """
    print("\nCalculating derived columns...")

    # ERO SNR
    mask_valid = df['ero_fluxerr_auto'] > 0
    df.loc[mask_valid, 'ero_snr'] = (
        df.loc[mask_valid, 'ero_flux_auto'] /
        df.loc[mask_valid, 'ero_fluxerr_auto']
    )
    df.loc[~mask_valid, 'ero_snr'] = np.nan

    # ERO SNR log
    mask_positive = df['ero_snr'] > 0
    df.loc[mask_positive, 'ero_snr_log'] = np.log10(df.loc[mask_positive, 'ero_snr'])
    df.loc[~mask_positive, 'ero_snr_log'] = np.nan

    # Gaia SNR
    mask_valid_gaia = df['gaia_flux_error_g'] > 0
    df.loc[mask_valid_gaia, 'gaia_snr'] = (
        df.loc[mask_valid_gaia, 'gaia_flux_g'] /
        df.loc[mask_valid_gaia, 'gaia_flux_error_g']
    )
    df.loc[~mask_valid_gaia, 'gaia_snr'] = np.nan

    # Gaia SNR log
    mask_positive_gaia = df['gaia_snr'] > 0
    df.loc[mask_positive_gaia, 'gaia_snr_log'] = np.log10(
        df.loc[mask_positive_gaia, 'gaia_snr']
    )
    df.loc[~mask_positive_gaia, 'gaia_snr_log'] = np.nan

    # Angular separation (in arcseconds)
    dlon = np.radians(df['gaia_ra'] - df['ero_ra'])
    dlat = np.radians(df['gaia_dec'] - df['ero_dec'])
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(df['ero_dec'])) *
         np.cos(np.radians(df['gaia_ra'])) *
         np.sin(dlon/2)**2)
    c = 2 * np.arcsin(np.sqrt(a))
    df['angular_separation_arcsec'] = np.degrees(c) * 3600.0

    # Color (BP - RP)
    mask_color = (df['gaia_mag_bp'].notna()) & (df['gaia_mag_rp'].notna())
    df.loc[mask_color, 'gaia_bp_minus_rp'] = (
        df.loc[mask_color, 'gaia_mag_bp'] -
        df.loc[mask_color, 'gaia_mag_rp']
    )

    print("✓ Derived columns calculated:")
    print(f"  - ero_snr, ero_snr_log")
    print(f"  - gaia_snr, gaia_snr_log")
    print(f"  - angular_separation_arcsec")
    print(f"  - gaia_bp_minus_rp")

    return df


def analyze_data(df):
    """
    Print summary statistics about the downloaded data.

    Args:
        df: DataFrame with morphology data
    """
    print("\n" + "="*70)
    print("Data Summary")
    print("="*70)

    print(f"\nShape: {df.shape}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print(f"\nERO Morphology Statistics:")
    print(f"  class_star:")
    print(f"    Mean: {df['ero_class_star'].mean():.4f}")
    print(f"    Median: {df['ero_class_star'].median():.4f}")
    print(f"    Range: {df['ero_class_star'].min():.4f} - {df['ero_class_star'].max():.4f}")
    print(f"  spread_model:")
    print(f"    Mean: {df['ero_spread_model'].mean():.4f}")
    print(f"    Median: {df['ero_spread_model'].median():.4f}")
    print(f"    Range: {df['ero_spread_model'].min():.4f} - {df['ero_spread_model'].max():.4f}")

    print(f"\nGaia Morphology Statistics:")
    print(f"  classprob_star:")
    gaia_star = df['gaia_classprob_star'].dropna()
    if len(gaia_star) > 0:
        print(f"    Mean: {gaia_star.mean():.4f}")
        print(f"    Median: {gaia_star.median():.4f}")
        print(f"    Range: {gaia_star.min():.4f} - {gaia_star.max():.4f}")
    print(f"  classprob_galaxy:")
    gaia_galaxy = df['gaia_classprob_galaxy'].dropna()
    if len(gaia_galaxy) > 0:
        print(f"    Mean: {gaia_galaxy.mean():.4f}")
        print(f"    Median: {gaia_galaxy.median():.4f}")
        print(f"    Range: {gaia_galaxy.min():.4f} - {gaia_galaxy.max():.4f}")

    print(f"\nMagnitude Ranges:")
    print(f"  ERO mag_auto: {df['ero_mag_auto'].min():.2f} - {df['ero_mag_auto'].max():.2f}")
    print(f"  Gaia mag_g: {df['gaia_mag_g'].min():.2f} - {df['gaia_mag_g'].max():.2f}")

    print(f"\nSNR Statistics:")
    ero_snr_valid = df['ero_snr'].dropna()
    if len(ero_snr_valid) > 0:
        print(f"  ERO SNR: {ero_snr_valid.min():.2f} - {ero_snr_valid.max():.2f} (median: {ero_snr_valid.median():.2f})")
    gaia_snr_valid = df['gaia_snr'].dropna()
    if len(gaia_snr_valid) > 0:
        print(f"  Gaia SNR: {gaia_snr_valid.min():.2f} - {gaia_snr_valid.max():.2f} (median: {gaia_snr_valid.median():.2f})")

    print(f"\nAngular Separation:")
    sep = df['angular_separation_arcsec'].dropna()
    if len(sep) > 0:
        print(f"  Mean: {sep.mean():.4f} arcsec")
        print(f"  Median: {sep.median():.4f} arcsec")
        print(f"  Max: {sep.max():.4f} arcsec (should be ~1.0 arcsec)")
        pct_within_1arcsec = 100 * (sep <= 1.0).sum() / len(sep)
        print(f"  Within 1.0 arcsec: {pct_within_1arcsec:.1f}%")


def main():
    """Main execution."""
    import os

    print("="*70)
    print("Download ERO-Gaia Morphology Subset for Analysis")
    print("="*70)

    # Get username
    username = os.getenv('PGUSER')
    if not username:
        print("\nERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return

    print(f"\nLogged in as: {username}")
    print(f"WSDB Host: {WSDB_HOST}")
    print(f"Database: {WSDB_DB}")

    # Download data
    print("\n" + "="*70)
    print("Step 1: Download Data")
    print("="*70)

    df = download_morphology_subset(username)
    if df is None:
        print("\nFailed to download data. Exiting.")
        return

    # Calculate derived columns
    print("\n" + "="*70)
    print("Step 2: Calculate Derived Columns")
    print("="*70)

    df = calculate_derived_columns(df)

    # Analyze
    print("\n" + "="*70)
    print("Step 3: Analyze Data")
    print("="*70)

    analyze_data(df)

    # Save to parquet
    print("\n" + "="*70)
    print("Step 4: Save to Parquet")
    print("="*70)

    output_path = DATA_DIR / 'ero_gaia_subset.parquet'
    df.to_parquet(output_path)
    print(f"\n✓ Data saved to: {output_path}")
    file_size_mb = output_path.stat().st_size / 1e6
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    print("\n" + "="*70)
    print("Download complete!")
    print("="*70)
    print("\nYou can now load the data with:")
    print(f"  df = pd.read_parquet('{output_path}')")


if __name__ == '__main__':
    main()
