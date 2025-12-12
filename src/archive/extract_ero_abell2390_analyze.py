#!/usr/bin/env python3
"""
Extract Abell 2390 Euclid ERO data and create simplified 2D histogram analysis.

This script:
1. Queries WSDB for Abell 2390 data
2. Stores data in Parquet format
3. Creates 2D histograms with linear and power law (γ=0.5) normalization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import os
from pathlib import Path
from matplotlib.colors import LogNorm, PowerNorm

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'output'

# Abell 2390 coordinates and area (from Euclid ERO Table B.2)
RA_CENTER = 328.397
DEC_CENTER = 17.709
AREA_DEG2 = 0.75  # Area in square degrees from Euclid ERO paper

# Calculate radius from area (A = π * r²)
RADIUS = np.sqrt(AREA_DEG2 / np.pi)

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def extract_abell2390_data():
    """Query WSDB and extract Abell 2390 data."""
    print("Connecting to WSDB...")

    try:
        conn = psycopg2.connect("dbname=wsdb")
        cursor = conn.cursor()

        # SQL query
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
          AND SQRT(POWER(v_alpha_j2000 - {RA_CENTER}, 2) + POWER(v_delta_j2000 - {DEC_CENTER}, 2)) < {RADIUS}
        """

        print(f"Querying Abell 2390 (RA={RA_CENTER}, Dec={DEC_CENTER}, radius={RADIUS}°)...")
        cursor.execute(query)

        # Fetch data
        columns = ['ra', 'dec', 'magnitude', 'class_star', 'spread_model', 'spreaderr_model', 'flux_auto', 'fluxerr_auto']
        data = cursor.fetchall()
        df = pd.DataFrame(data, columns=columns)

        cursor.close()
        conn.close()

        print(f"Retrieved {len(df)} sources")
        return df

    except Exception as e:
        print(f"Error connecting to WSDB: {e}")
        print("Make sure PostgreSQL/WSDB is running")
        return None

def save_data_parquet(df, filename='abell2390_euclid_ero.parquet'):
    """Save dataframe to Parquet format."""
    filepath = DATA_DIR / filename
    df.to_parquet(filepath, compression='snappy')
    print(f"Saved to: {filepath}")
    print(f"File size: {filepath.stat().st_size / (1024**2):.1f} MB")
    return filepath

def load_data_parquet(filename='abell2390_euclid_ero.parquet'):
    """Load data from Parquet format."""
    filepath = DATA_DIR / filename
    df = pd.read_parquet(filepath)
    print(f"Loaded {len(df)} sources from {filepath}")
    return df

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
    """Remove unrealistic data points."""
    print(f"\nOriginal data: {len(df)} sources")
    print(f"  Magnitude range: {df['magnitude'].min():.2f} - {df['magnitude'].max():.2f}")
    print(f"  Spread_model range: {df['spread_model'].min():.6f} - {df['spread_model'].max():.6f}")

    df_filtered = df[(df['magnitude'] <= 31) & (df['spread_model'] <= 1)].copy()

    print(f"\nAfter filtering (mag <= 31, spread_model <= 1): {len(df_filtered)} sources")
    print(f"  Magnitude range: {df_filtered['magnitude'].min():.2f} - {df_filtered['magnitude'].max():.2f}")
    print(f"  Spread_model range: {df_filtered['spread_model'].min():.6f} - {df_filtered['spread_model'].max():.6f}")
    print(f"  Retention rate: {(len(df_filtered)/len(df))*100:.1f}%")

    return df_filtered

def create_2d_histograms(df):
    """Create 2D histograms with logarithmic color normalization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes = axes.flatten()

    # 2D Histogram 1: Class Star
    ax1 = axes[0]
    h1 = ax1.hist2d(
        df['magnitude'],
        df['class_star'],
        bins=[50, 50],
        cmap='viridis',
        norm=LogNorm()
    )
    ax1.set_xlabel('Magnitude', fontsize=12)
    ax1.set_ylabel('Class Star', fontsize=12)
    ax1.invert_yaxis()
    ax1.set_yticklabels(['0\n(extended)', '0.2', '0.4', '0.6', '0.8', '1\n(point-like)'])
    ax1.set_title('Class Star (Log Scale)', fontsize=13, fontweight='bold')
    cbar1 = plt.colorbar(h1[3], ax=ax1)
    cbar1.set_label('Count (log scale)')
    ax1.grid(True, alpha=0.3)

    # 2D Histogram 2: Spread Model
    ax2 = axes[1]
    h2 = ax2.hist2d(
        df['magnitude'],
        df['spread_model'],
        bins=[50, 50],
        cmap='viridis',
        range=[[df['magnitude'].min(), df['magnitude'].max()], [-0.01, 0.05]],
        norm=LogNorm()
    )
    ax2.set_xlabel('Magnitude', fontsize=12)
    ax2.set_ylabel('Spread Model', fontsize=12)
    ax2.set_yticklabels(['-0.01\n(saturated)', '', '0\n(point-like)', '', '0.05\n(extended)'])
    ax2.set_title('Spread Model (Log Scale)', fontsize=13, fontweight='bold')
    cbar2 = plt.colorbar(h2[3], ax=ax2)
    cbar2.set_label('Count (log scale)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'abell2390_2d_histograms_logscale.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    # plt.show()

def main():
    """Main execution."""
    print("="*60)
    print("Abell 2390 Data Extraction and Simplified Analysis")
    print("="*60)

    # Check if data already exists
    parquet_file = DATA_DIR / 'abell2390_euclid_ero.parquet'

    if parquet_file.exists():
        print(f"\nParquet file already exists: {parquet_file}")
        response = input("Load existing data? (y/n): ").strip().lower()
        if response == 'y':
            df = load_data_parquet()
        else:
            df = extract_abell2390_data()
            if df is not None:
                # Calculate threshold before saving
                print("Calculating spread_model threshold...")
                df = calculate_spread_model_threshold(df)
                print("Calculating SNR...")
                df = calculate_snr(df)
                save_data_parquet(df)
    else:
        # Extract from WSDB
        df = extract_abell2390_data()
        if df is not None:
            # Calculate threshold before saving
            print("Calculating spread_model threshold...")
            df = calculate_spread_model_threshold(df)
            print("Calculating SNR...")
            df = calculate_snr(df)
            save_data_parquet(df)
        else:
            print("Failed to extract data. Exiting.")
            return

    if df is None:
        return

    # Filter anomalies
    df = filter_anomalies(df)

    # Create visualizations
    print("\nCreating 2D histograms...")
    create_2d_histograms(df)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

if __name__ == '__main__':
    main()
