#!/usr/bin/env python3
"""
Extract ERO-Gaia crossmatch data from WSDB for different radii
and save as local parquet files.

Downloads crossmatch tables from WSDB for different matching radii
and saves them as:
  - ero_gaia_subset_0_05.parquet
  - ero_gaia_subset_0_08.parquet
  - ero_gaia_subset_0_1.parquet
  - ero_gaia_subset_0_2.parquet
  - ero_gaia_subset_1_0.parquet (renamed from existing ero_gaia_subset.parquet)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
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


def get_radii_selection():
    """
    Interactive prompt to select which radii to extract.

    Returns:
        list of float: selected radii in arcseconds
    """
    print("\n" + "="*70)
    print("Select Matching Radius to Extract")
    print("="*70)
    print("Enter one or more radius values in arcseconds, separated by commas.")
    print("Examples:")
    print("  - Single value: 0.1")
    print("  - Multiple values: 0.05,0.08,0.1,0.2,1.0")

    while True:
        user_input = input("\nEnter radius value(s) in arcseconds: ").strip()
        try:
            # Parse comma-separated values
            radii = [float(r.strip()) for r in user_input.split(',')]

            # Validate: all must be positive
            if all(r > 0 for r in radii):
                radii = sorted(set(radii))  # Remove duplicates and sort
                print(f"\nSelected radii: {radii} arcseconds")
                return radii
            else:
                print("ERROR: All radius values must be positive. Please try again.")
        except ValueError:
            print("ERROR: Invalid input. Please enter numeric values separated by commas.")
            print("Example: 0.1 or 0.05,0.08,0.1,0.2,1.0")


def download_subset(table_name, username):
    """
    Download morphology-relevant columns from ERO-Gaia cross-match table.

    Args:
        table_name: full table name (username.ero_gaia_crossmatch_X_Y)
        username: WSDB username

    Returns:
        DataFrame with crossmatch data, or None on failure
    """
    try:
        import sqlutilpy
    except ImportError:
        print("ERROR: sqlutilpy not installed!")
        return None

    print(f"Downloading subset from {table_name}...")

    # SQL query to get all relevant columns
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
        print("  Executing query...")
        result = sqlutilpy.get(
            query,
            db=WSDB_DB,
            host=WSDB_HOST
        )

        if result and len(result[0]) > 0:
            print(f"  ✓ Query successful! Downloaded {len(result[0]):,} rows")

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

            return df
        else:
            print("  No results returned from query")
            return None

    except Exception as e:
        print(f"  ERROR executing query: {e}")
        return None


def calculate_derived_columns(df):
    """
    Calculate derived quantities useful for morphology analysis.

    Args:
        df: DataFrame with morphology data

    Returns:
        DataFrame with additional derived columns
    """
    print("  Calculating derived columns...")

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
         np.cos(np.radians(df['gaia_dec'])) *
         np.sin(dlon/2)**2)
    c = 2 * np.arcsin(np.sqrt(a))
    df['angular_separation_arcsec'] = np.degrees(c) * 3600.0

    # Color (BP - RP)
    mask_color = (df['gaia_mag_bp'].notna()) & (df['gaia_mag_rp'].notna())
    df.loc[mask_color, 'gaia_bp_minus_rp'] = (
        df.loc[mask_color, 'gaia_mag_bp'] -
        df.loc[mask_color, 'gaia_mag_rp']
    )

    return df


def extract_and_save_subset(radius_arcsec, username):
    """
    Extract crossmatch data from WSDB and save as parquet.

    Args:
        radius_arcsec: matching radius in arcseconds
        username: WSDB username

    Returns:
        bool: success or failure
    """
    # Build table and filename
    radius_suffix = str(radius_arcsec).replace('.', '_')
    table_name = f'{username}.ero_gaia_crossmatch_{radius_suffix}'
    output_file = DATA_DIR / f'ero_gaia_subset_{radius_suffix}.parquet'

    print(f"\n  Extracting {radius_arcsec} arcsec subset...")

    # Download data
    df = download_subset(table_name, username)
    if df is None:
        return False

    # Calculate derived columns
    df = calculate_derived_columns(df)

    # Save to parquet
    print(f"  Saving to {output_file.name}...")
    df.to_parquet(output_file, index=False, compression='snappy')

    print(f"  ✓ Saved successfully!")
    print(f"    File: {output_file}")
    print(f"    Size: {output_file.stat().st_size / (1024**2):.1f} MB")
    print(f"    Rows: {len(df):,}")
    print(f"    Columns: {len(df.columns)}")

    return True


def main():
    """Main execution."""
    import os

    print("="*70)
    print("Extract ERO-Gaia Crossmatch Subsets from WSDB")
    print("="*70)
    print("\nConfiguration:")
    print(f"  WSDB Host: {WSDB_HOST}")
    print(f"  Database: {WSDB_DB}")
    print(f"  Data directory: {DATA_DIR}")
    print("\nNote: Requires WSDB credentials via environment variables:")
    print("  PGUSER, PGPASSWORD")

    # Get username from environment
    username = os.getenv('PGUSER')
    if not username:
        print("\nERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return

    print(f"\nLogged in as: {username}")

    # Get radius selection from user
    selected_radii = get_radii_selection()

    # Extract and save subsets
    print("\n" + "="*70)
    print("Step 1: Extract and Save Subsets")
    print("="*70)

    results = {}
    for i, radius in enumerate(selected_radii, 1):
        print(f"\n{'-'*70}")
        print(f"Subset {i}/{len(selected_radii)}: {radius} arcsecond radius")
        print(f"{'-'*70}")

        if extract_and_save_subset(radius, username):
            radius_suffix = str(radius).replace('.', '_')
            results[radius] = True
        else:
            results[radius] = False

    # Results summary
    print("\n" + "="*70)
    print("Step 2: Results Summary")
    print("="*70)

    successful = sum(1 for v in results.values() if v)
    print(f"\nExtraction complete: {successful}/{len(results)} successful")

    print(f"\nFiles created in {DATA_DIR}:")
    for radius in selected_radii:
        if results[radius]:
            radius_suffix = str(radius).replace('.', '_')
            filename = f'ero_gaia_subset_{radius_suffix}.parquet'
            filepath = DATA_DIR / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024**2)
                print(f"  ✓ {filename} ({size_mb:.1f} MB)")
        else:
            radius_suffix = str(radius).replace('.', '_')
            filename = f'ero_gaia_subset_{radius_suffix}.parquet'
            print(f"  ✗ {filename} (FAILED)")

    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == '__main__':
    main()
