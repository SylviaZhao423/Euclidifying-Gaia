#!/usr/bin/env python3
"""
Extract Euclid Q1 data for all 4 fields from WSDB and store in a single parquet file.

This script:
1. Loads Q1 field reference data (coordinates, areas, etc.)
2. Queries WSDB for each field using Q1 catalog columns
3. Stores all data in a single parquet file for combined analysis
4. Calculates Signal-to-Noise Ratio from VIS PSF photometry
5. Calculates magnitude from flux for magnitude-dependent analysis

Euclid Q1 Fields (correct coordinates from Euclid documentation):
- EDF-N: RA=269.733°, Dec=66.018°, area=22.9 deg²
- EDF-S: RA=61.241°, Dec=-48.423°, area=28.1 deg²
- EDF-F: RA=52.932°, Dec=-28.088°, area=12.1 deg²
- DC: RA=85.75°, Dec=-8.367°, area=0.5 deg²

Key difference from ERO: Q1 uses probability-based morphology flags
(point_like_prob, extended_prob) instead of continuous classifiers
(class_star, spread_model).

Reference: https://euclid.esac.esa.int/dr/q1/dpdd/merdpd/dpcards/mer_finalcatalog.html
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

# Q1 field definitions with correct coordinates from Euclid documentation
# RA [deg], Dec [deg], and area in deg²
Q1_FIELDS = {
    'EDF-N': {'ra': 269.733, 'dec': 66.018, 'area_deg2': 22.9},
    'EDF-S': {'ra': 61.241, 'dec': -48.423, 'area_deg2': 28.1},
    'EDF-F': {'ra': 52.932, 'dec': -28.088, 'area_deg2': 12.1},
    'DC': {'ra': 85.75, 'dec': -8.367, 'area_deg2': 0.5}
}

# Calculate radius from area (A = π * r²)
for field_name, data in Q1_FIELDS.items():
    data['radius'] = np.sqrt(data['area_deg2'] / np.pi)


def extract_q1_field_data(field_name, ra_center, dec_center, radius):
    """Query WSDB and extract data for a single Q1 field.

    Uses euclid_q1.mer_final_cat with Q1-specific column names:
    - Position: right_ascension, declination
    - Magnitude: mag_stargal_sep (magnitude used for star/galaxy separation)
    - Morphology: point_like_prob, extended_prob, point_like_flag, extended_flag
    - Photometry: flux_vis_psf, fluxerr_vis_psf (VIS PSF photometry for SNR)
    """
    try:
        conn = psycopg2.connect("dbname=wsdb")
        cursor = conn.cursor()

        # SQL query - using Q1-specific lowercase underscore column names
        query = f"""
        SELECT
          right_ascension as ra,
          declination as dec,
          mag_stargal_sep as magnitude,
          point_like_prob,
          point_like_flag,
          extended_prob,
          extended_flag,
          flux_vis_psf as flux_auto,
          fluxerr_vis_psf as fluxerr_auto
        FROM euclid_q1.mer_final_cat
        WHERE right_ascension IS NOT NULL
          AND declination IS NOT NULL
          AND mag_stargal_sep IS NOT NULL
          AND point_like_prob IS NOT NULL
          AND extended_prob IS NOT NULL
          AND flux_vis_psf IS NOT NULL
          AND fluxerr_vis_psf IS NOT NULL
          AND SQRT(POWER(right_ascension - {ra_center}, 2) + POWER(declination - {dec_center}, 2)) < {radius}
        """

        cursor.execute(query)
        columns = ['ra', 'dec', 'magnitude', 'point_like_prob', 'point_like_flag',
                   'extended_prob', 'extended_flag', 'flux_auto', 'fluxerr_auto']
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


def calculate_snr(df):
    """
    Calculate Signal-to-Noise Ratio from VIS PSF photometry.

    SNR = flux_vis_psf / fluxerr_vis_psf
    Uses SExtractor-equivalent PSF photometry.
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
    - flux_auto > 0: Remove sources with invalid/negative flux

    Note: Q1 catalog is already well-filtered with quality flags,
    so we apply minimal additional filtering.
    """
    df_filtered = df[(df['magnitude'] <= 31) &
                     (df['flux_auto'] > 0)].copy()
    return df_filtered


def main():
    """Main execution."""
    print("="*60)
    print("Extracting All Euclid Q1 Fields")
    print("="*60)

    all_data = []
    successful_fields = []
    failed_fields = []

    # Extract data for each Q1 field
    for field_name, field_info in Q1_FIELDS.items():
        ra_center = field_info['ra']
        dec_center = field_info['dec']
        radius = field_info['radius']

        print(f"\nExtracting {field_name} (RA={ra_center:.3f}, Dec={dec_center:.3f}, radius={radius:.3f}°)...")

        df = extract_q1_field_data(field_name, ra_center, dec_center, radius)

        if df is not None and len(df) > 0:
            # Calculate SNR
            df = calculate_snr(df)

            # Filter anomalies
            df_filtered = filter_anomalies(df)

            print(f"  Retrieved {len(df)} sources, {len(df_filtered)} after filtering")
            print(f"  Magnitude range: {df_filtered['magnitude'].min():.2f} - {df_filtered['magnitude'].max():.2f}")
            print(f"  Retention rate: {(len(df_filtered)/len(df))*100:.1f}%")
            print(f"  SNR range: {df_filtered['snr'].min():.1f} - {df_filtered['snr'].max():.1f}")

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

        print(f"\nExtended source classification:")
        print(f"  By extended_flag: {combined_df['extended_flag'].sum()} sources")
        print(f"  By point_like_flag: {combined_df['point_like_flag'].sum()} sources")
        print(f"  Extended probability range: {combined_df['extended_prob'].min():.3f} - {combined_df['extended_prob'].max():.3f}")
        print(f"  Point-like probability range: {combined_df['point_like_prob'].min():.3f} - {combined_df['point_like_prob'].max():.3f}")

        # Save to parquet
        output_path = DATA_DIR / 'all_q1_fields.parquet'
        combined_df.to_parquet(output_path, compression='snappy')
        print(f"\nSaved to: {output_path}")
        print(f"File size: {output_path.stat().st_size / (1024**2):.1f} MB")
    else:
        print("No data retrieved. Check WSDB connection and Q1 table availability.")

    if failed_fields:
        print(f"\nFailed fields: {', '.join(failed_fields)}")

    print("\n" + "="*60)
    print("Extraction complete!")
    print("="*60)


if __name__ == '__main__':
    main()
