#!/usr/bin/env python3
"""
Extract Q1 MER-PHZ subset to parquet file.

Extracts useful columns from the WSDB table q1_mer_phz_crossmatch and saves to parquet.

This subset includes:
- Positions (RA/Dec from Q1)
- Identifiers (object_id)
- Morphology indicators (point-like/extended flags and probabilities)
- Magnitudes (VIS, BP, RP)
- Photometry (flux measurements)
- Classification (Q1 flags, PHZ classification and probabilities)

Output: q1_mer_phz_crossmatch_subset.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'output'

# WSDB Configuration
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


def extract_q1_mer_phz_subset(username):
    """
    Extract useful subset of columns from Q1 MER-PHZ crossmatch table.

    Args:
        username: WSDB username

    Returns:
        DataFrame with extracted subset
    """
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed!")
        print("Install with: pip install psycopg2")
        return None

    table_name = f'{username}.q1_mer_phz_crossmatch'

    print(f"\nConnecting to WSDB...")
    try:
        conn = psycopg2.connect(
            host=WSDB_HOST,
            database=WSDB_DB,
            user=username
        )

        # Select useful columns from Q1 MER and PHZ
        query = f"""
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
        """

        print(f"Querying {table_name}...")
        df = pd.read_sql(query, conn)
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
    print("Extract Q1 MER-PHZ Useful Subset to Parquet")
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
    print("Step 1: Extract Subset from WSDB")
    print("="*80)

    df = extract_q1_mer_phz_subset(username)
    if df is None:
        print("Failed to extract data from WSDB")
        return

    # Save to parquet
    print("\n" + "="*80)
    print("Step 2: Save to Parquet")
    print("="*80)

    output_path = DATA_DIR / 'q1_mer_phz_crossmatch_subset.parquet'
    df.to_parquet(output_path, compression='snappy', index=False)

    print(f"\n✓ Subset saved to: {output_path}")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Print summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    print(f"\nExtracted {len(df.columns)} columns:")
    print(f"\nPosition & Identification:")
    print(f"  - object_id, RA, Dec")
    print(f"\nMagnitudes:")
    print(f"  - mag_stargal_sep (Q1 native separation magnitude)")
    print(f"  - VIS, BP, RP photometric magnitudes")
    print(f"\nPhotometry:")
    print(f"  - VIS flux (total and 1-FWHM aperture)")
    print(f"  - BP, RP flux measurements")
    print(f"\nMorphology Indicators (Q1):")
    print(f"  - pointlike, pointlike_prob")
    print(f"  - vis_extended, vis_extended_prob")
    print(f"\nClassification:")
    print(f"  - phz_classification (dominant class from PHZ)")
    print(f"  - phz_star_prob, phz_gal_prob, phz_qso_prob, phz_globc_prob")
    print(f"  - Classification thresholds (star, galaxy, globular cluster)")

    print("\n" + "="*80)
    print("Extraction Complete!")
    print("="*80)
    print(f"\nLoad the subset with:")
    print(f"  import pandas as pd")
    print(f"  df = pd.read_parquet('{output_path}')")


if __name__ == '__main__':
    main()
