#!/usr/bin/env python3
"""
Extract Euclid ERO data for all 17 fields from WSDB and store in a single parquet file.

This script:
1. Loads ERO field reference data (coordinates, areas, etc.)
2. Queries WSDB for each field
3. Stores all data in a single parquet file for combined analysis
4. Calculates magnitude-dependent spread_model threshold for each source
"""

import pandas as pd
import numpy as np
import psycopg2
from pathlib import Path

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'output'

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def load_ero_reference():
    """Load ERO reference data with field coordinates and areas."""
    filepath = DATA_DIR / 'ero_fields.parquet'
    df = pd.read_parquet(filepath)
    print(f"Loaded {len(df)} ERO fields from reference data")
    return df


def extract_ero_field_data(field_name, ra_center, dec_center, radius):
    """Query WSDB and extract data for a single ERO field."""
    try:
        conn = psycopg2.connect("dbname=wsdb")
        cursor = conn.cursor()

        # SQL query - same columns as before
        query = f"""
        SELECT
          v_alpha_j2000 as ra,
          v_delta_j2000 as dec,
          v_mag_auto as magnitude,
          v_class_star as class_star,
          v_spread_model as spread_model,
          v_spreaderr_model as spreaderr_model,
          v_flux_auto as flux_auto,
          v_fluxerr_auto as fluxerr_auto
        FROM euclid_ero.merge_cat
        WHERE v_alpha_j2000 IS NOT NULL
          AND v_delta_j2000 IS NOT NULL
          AND v_mag_auto IS NOT NULL
          AND v_class_star IS NOT NULL
          AND v_spread_model IS NOT NULL
          AND v_spreaderr_model IS NOT NULL
          AND v_flux_auto IS NOT NULL
          AND v_fluxerr_auto IS NOT NULL
          AND SQRT(POWER(v_alpha_j2000 - {ra_center}, 2) + POWER(v_delta_j2000 - {dec_center}, 2)) < {radius}
        """

        cursor.execute(query)
        columns = ['ra', 'dec', 'magnitude', 'class_star', 'spread_model', 'spreaderr_model', 'flux_auto', 'fluxerr_auto']
        data = cursor.fetchall()
        df = pd.DataFrame(data, columns=columns)

        # Add field identifier
        df['field_name'] = field_name

        cursor.close()
        conn.close()

        return df

    except Exception as e:
        print(f"Error extracting {field_name}: {e}")
        return None


def calculate_spread_model_threshold(df):
    """
    Calculate magnitude-dependent spread_model threshold.

    Formula from SExtractor documentation:
    threshold = sqrt(ε² + (κ × SPREADERR_MODEL)²)
    where ε ≈ 0.005 and κ ≈ 4
    """
    EPSILON = 0.005
    KAPPA = 4.0
    df['spread_model_threshold'] = np.sqrt(EPSILON**2 + (KAPPA * df['spreaderr_model'])**2)
    return df


def calculate_snr(df):
    """
    Calculate Signal-to-Noise Ratio from AUTO photometry.

    SNR = flux_auto / fluxerr_auto
    Uses SExtractor AUTO aperture (Kron-like elliptical).
    """
    # Avoid division by zero
    mask_valid = df['fluxerr_auto'] > 0
    df.loc[mask_valid, 'snr'] = df.loc[mask_valid, 'flux_auto'] / df.loc[mask_valid, 'fluxerr_auto']
    df.loc[~mask_valid, 'snr'] = np.nan

    # Log-transform for better visualization (avoid log(0) and log(negative))
    mask_positive = df['snr'] > 0
    df.loc[mask_positive, 'snr_log'] = np.log10(df.loc[mask_positive, 'snr'])
    df.loc[~mask_positive, 'snr_log'] = np.nan

    return df


def filter_anomalies(df):
    """Remove unrealistic data points.

    Filters:
    - magnitude <= 31: Remove sources fainter than realistic limit
    - spread_model in [-0.5, 1.0]: Remove artifacts (cosmic rays, bad pixels)
      * spread_model < -0.5 indicates clear measurement errors/artifacts
        (removes only 24 sources, ~0.0002% of data)
      * spread_model > 1.0 indicates capped/saturated values (already at max)

    Note: Values in range -0.5 to 0.0 are legitimate faint point sources (~2.4M sources)
    """
    df_filtered = df[(df['magnitude'] <= 31) &
                     (df['spread_model'] >= -0.5) &
                     (df['spread_model'] <= 1.0)].copy()
    return df_filtered


def main():
    """Main execution."""
    print("="*60)
    print("Extracting All Euclid ERO Fields")
    print("="*60)

    # Load reference data
    ero_reference = load_ero_reference()

    all_data = []
    successful_fields = []
    failed_fields = []

    # Extract data for each field
    for idx, row in ero_reference.iterrows():
        field_name = row['field_name']
        ra_center = row['ra_deg']
        dec_center = row['dec_deg']
        radius = row['radius_deg']

        print(f"\nExtracting {field_name} (RA={ra_center:.3f}, Dec={dec_center:.3f}, radius={radius:.3f}°)...")

        df = extract_ero_field_data(field_name, ra_center, dec_center, radius)

        if df is not None and len(df) > 0:
            # Calculate threshold
            df = calculate_spread_model_threshold(df)

            # Calculate SNR
            df = calculate_snr(df)

            # Filter anomalies
            df_filtered = filter_anomalies(df)

            print(f"  Retrieved {len(df)} sources, {len(df_filtered)} after filtering")
            print(f"  Magnitude range: {df_filtered['magnitude'].min():.2f} - {df_filtered['magnitude'].max():.2f}")
            print(f"  Retention rate: {(len(df_filtered)/len(df))*100:.1f}%")

            all_data.append(df_filtered)
            successful_fields.append(field_name)
        else:
            print(f"  Failed or no data retrieved for {field_name}")
            failed_fields.append(field_name)

    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        print("\n" + "="*60)
        print("Combined Data Summary")
        print("="*60)
        print(f"Total sources from {len(successful_fields)} fields: {len(combined_df)}")
        print(f"Columns: {', '.join(combined_df.columns)}")
        print(f"\nSources per field:")
        for field_name in combined_df['field_name'].unique():
            count = len(combined_df[combined_df['field_name'] == field_name])
            print(f"  {field_name}: {count}")

        # Save to parquet
        output_path = DATA_DIR / 'all_ero_fields.parquet'
        combined_df.to_parquet(output_path, compression='snappy')
        print(f"\nSaved to: {output_path}")
        print(f"File size: {output_path.stat().st_size / (1024**2):.1f} MB")
    else:
        print("No data retrieved. Check WSDB connection.")

    if failed_fields:
        print(f"\nFailed fields: {', '.join(failed_fields)}")

    print("\n" + "="*60)
    print("Extraction complete!")
    print("="*60)


if __name__ == '__main__':
    main()
