#!/usr/bin/env python3
"""
Fast extraction of Q1 MER-PHZ subset using COPY command.

Uses PostgreSQL COPY for 10-50x faster extraction compared to INSERT.
Extracts to CSV then converts to parquet.
"""

import pandas as pd
import io
import os
from pathlib import Path

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'output'

# WSDB Configuration
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


def extract_q1_mer_phz_fast(username):
    """
    Fast extraction using COPY command.

    Args:
        username: WSDB username

    Returns:
        DataFrame with extracted subset
    """
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed!")
        return None

    table_name = f'{username}.q1_mer_phz_crossmatch'

    print(f"\nConnecting to WSDB...")
    try:
        conn = psycopg2.connect(
            host=WSDB_HOST,
            database=WSDB_DB,
            user=username
        )
        cursor = conn.cursor()

        # Use COPY for fast extraction
        query = f"""
            COPY (
                SELECT
                    object_id,
                    right_ascension,
                    declination,
                    mag_stargal_sep,
                    vis_det,
                    flux_vis_1fwhm_aper,
                    flux_y_1fwhm_aper,
                    flux_j_1fwhm_aper,
                    flux_h_1fwhm_aper,
                    point_like_flag,
                    point_like_prob,
                    extended_flag,
                    extended_prob,
                    phz_classification,
                    phz_star_prob,
                    phz_gal_prob,
                    phz_qso_prob,
                    phz_glob_cl_prob,
                    star_threshold,
                    galaxy_threshold,
                    globc_threshold
                FROM {table_name}
            ) TO STDOUT WITH CSV HEADER
        """

        print(f"Extracting {table_name} using COPY (fast method)...")

        # Read directly into a StringIO buffer
        buffer = io.StringIO()
        cursor.copy_expert(query, buffer)

        # Get the CSV data
        buffer.seek(0)
        csv_data = buffer.getvalue()

        print(f"  ✓ COPY completed ({len(csv_data) / (1024**2):.1f} MB)")

        # Convert to DataFrame
        print(f"  Converting to DataFrame...")
        csv_buffer = io.StringIO(csv_data)
        df = pd.read_csv(csv_buffer)

        cursor.close()
        conn.close()

        print(f"  ✓ Loaded {len(df):,} rows")
        print(f"  ✓ {len(df.columns)} columns")

        return df

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main execution."""
    print("="*80)
    print("Extract Q1 MER-PHZ Subset (FAST - Using COPY)")
    print("="*80)

    # Get username from environment
    username = os.getenv('PGUSER')
    if not username:
        print("\nERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return

    print(f"\nLogged in as: {username}")
    print(f"WSDB Host: {WSDB_HOST}")
    print(f"Database: {WSDB_DB}")

    # Extract subset from WSDB
    print("\n" + "="*80)
    print("Step 1: Extract Subset from WSDB (using COPY)")
    print("="*80)

    df = extract_q1_mer_phz_fast(username)
    if df is None:
        print("Failed to extract data from WSDB")
        return

    # Save to parquet
    print("\n" + "="*80)
    print("Step 2: Save to Parquet")
    print("="*80)

    output_path = DATA_DIR / 'q1_mer_phz_crossmatch_subset.parquet'
    print(f"Saving to {output_path}...")
    df.to_parquet(output_path, compression='snappy', index=False)

    print(f"\n✓ Subset saved to: {output_path}")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  File size: {output_path.stat().st_size / (1024**2):.1f} MB")

    # Print summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    print(f"\nExtracted {len(df.columns)} columns:")
    print(f"\nPosition & Identification:")
    print(f"  - object_id, RA (right_ascension), Dec (declination)")
    print(f"\nMagnitudes:")
    print(f"  - mag_stargal_sep (Q1 native separation magnitude)")
    print(f"\nPhotometry:")
    print(f"  - vis_det: VIS band detection flag")
    print(f"  - flux_*_1fwhm_aper: Flux in 1-FWHM aperture (VIS, Y, J, H bands)")
    print(f"\nMorphology Indicators (Q1):")
    print(f"  - point_like_flag, point_like_prob (point-like classification)")
    print(f"  - extended_flag, extended_prob (extended source classification)")
    print(f"\nClassification (PHZ):")
    print(f"  - phz_classification (dominant class: 1=STAR, 2=GALAXY, 4=QSO, 8=GC)")
    print(f"  - phz_star_prob, phz_gal_prob, phz_qso_prob, phz_glob_cl_prob")
    print(f"  - Classification thresholds (star, galaxy, globular cluster)")

    print("\n" + "="*80)
    print("Extraction Complete!")
    print("="*80)
    print(f"\nLoad the subset with:")
    print(f"  import pandas as pd")
    print(f"  df = pd.read_parquet('{output_path}')")


if __name__ == '__main__':
    main()
