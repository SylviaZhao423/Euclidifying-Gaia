#!/usr/bin/env python3
"""
Download relevant Q1-Gaia cross-matched data for morphology analysis.

Extracts morphology-relevant columns from the cross-matched table
(siwei_zhao_2025.q1_gaia_crossmatch) and saves to parquet for local analysis.

Columns include:
- Positions (Q1 and Gaia)
- Morphology classifiers (Q1: point_like_flag, extended_flag; Gaia: classprob_star)
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
    Download morphology-relevant columns from Q1-Gaia cross-match table.

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

    table_name = f'{username}.q1_gaia_crossmatch'

    print(f"Downloading morphology subset from {table_name}...")

    # SQL query to get relevant columns for morphology analysis
    query = f"""
        SELECT
            -- Position (primary identifier)
            right_ascension as q1_ra,
            declination as q1_dec,
            ra as gaia_ra,
            dec as gaia_dec,

            -- Identifiers
            object_id as q1_object_id,
            source_id as gaia_source_id,

            -- Q1 morphology classifiers
            point_like_flag as q1_point_like_flag,
            point_like_prob as q1_point_like_prob,
            extended_flag as q1_extended_flag,
            extended_prob as q1_extended_prob,

            -- Q1 magnitudes (mag_stargal_sep is used for POINT_LIKE_PROB calibration)
            mag_stargal_sep as q1_mag_stargal_sep,
            mumax_minus_mag as q1_mumax_minus_mag,

            -- Gaia morphology classifiers
            classprob_dsc_combmod_star as gaia_classprob_star,
            classprob_dsc_combmod_galaxy as gaia_classprob_galaxy,
            classprob_dsc_combmod_quasar as gaia_classprob_quasar,

            -- Gaia magnitudes (for magnitude-dependent analysis)
            phot_g_mean_mag as gaia_mag_g,
            phot_bp_mean_mag as gaia_mag_bp,
            phot_rp_mean_mag as gaia_mag_rp,

            -- Photometry (for SNR calculation)
            phot_g_mean_flux as gaia_flux_g,
            phot_g_mean_flux_error as gaia_flux_error_g,

            -- Astrometry quality
            astrometric_excess_noise as gaia_astrometric_excess_noise,
            ruwe as gaia_ruwe,

            -- VIS aperture flux (primary Q1 photometry)
            flux_vis_1fwhm_aper as q1_flux_vis_1fwhm,
            vis_det as q1_vis_detected
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
                'q1_ra', 'q1_dec', 'gaia_ra', 'gaia_dec',
                'q1_object_id', 'gaia_source_id',
                'q1_point_like_flag', 'q1_point_like_prob',
                'q1_extended_flag', 'q1_extended_prob',
                'q1_mag_stargal_sep', 'q1_mumax_minus_mag',
                'gaia_classprob_star', 'gaia_classprob_galaxy', 'gaia_classprob_quasar',
                'gaia_mag_g', 'gaia_mag_bp', 'gaia_mag_rp',
                'gaia_flux_g', 'gaia_flux_error_g',
                'gaia_astrometric_excess_noise', 'gaia_ruwe',
                'q1_flux_vis_1fwhm', 'q1_vis_detected'
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
    dlon = np.radians(df['gaia_ra'] - df['q1_ra'])
    dlat = np.radians(df['gaia_dec'] - df['q1_dec'])
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(df['q1_dec'])) *
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

    print(f"\nQ1 Morphology Statistics:")
    print(f"  point_like_flag:")
    q1_pl_flag = df['q1_point_like_flag'].dropna()
    if len(q1_pl_flag) > 0:
        print(f"    Values: {q1_pl_flag.unique()}")
        print(f"    Counts: {q1_pl_flag.value_counts().to_dict()}")
    print(f"  point_like_prob:")
    q1_pl_prob = df['q1_point_like_prob'].dropna()
    if len(q1_pl_prob) > 0:
        print(f"    Mean: {q1_pl_prob.mean():.4f}")
        print(f"    Median: {q1_pl_prob.median():.4f}")
        print(f"    Range: {q1_pl_prob.min():.4f} - {q1_pl_prob.max():.4f}")
    print(f"  extended_flag:")
    q1_ext_flag = df['q1_extended_flag'].dropna()
    if len(q1_ext_flag) > 0:
        print(f"    Values: {q1_ext_flag.unique()}")
        print(f"    Counts: {q1_ext_flag.value_counts().to_dict()}")

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
    print(f"  Gaia mag_g: {df['gaia_mag_g'].min():.2f} - {df['gaia_mag_g'].max():.2f}")

    print(f"\nSNR Statistics:")
    gaia_snr_valid = df['gaia_snr'].dropna()
    if len(gaia_snr_valid) > 0:
        print(f"  Gaia SNR: {gaia_snr_valid.min():.2f} - {gaia_snr_valid.max():.2f} (median: {gaia_snr_valid.median():.2f})")

    print(f"\nAngular Separation:")
    sep = df['angular_separation_arcsec'].dropna()
    if len(sep) > 0:
        print(f"  Mean: {sep.mean():.4f} arcsec")
        print(f"  Median: {sep.median():.4f} arcsec")
        print(f"  Max: {sep.max():.4f} arcsec")


def main():
    """Main execution."""
    import os

    print("="*70)
    print("Download Q1-Gaia Morphology Subset for Analysis")
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

    output_path = DATA_DIR / 'q1_gaia_subset.parquet'
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
