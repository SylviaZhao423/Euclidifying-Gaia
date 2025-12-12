#!/usr/bin/env python3
"""
Extract Q1-Gaia-PHZ subset to parquet file.

Reads the parameter list from gaia_q1_subset_para.csv and extracts those columns
from the WSDB table q1_gaia_phz_crossmatch, saving to parquet format locally.

This subset includes:
- Positions (RA/Dec from Q1 and Gaia)
- Identifiers (object_id, gaia_source_id)
- Morphology indicators (point-like/extended probabilities and flags)
- Magnitudes (Q1 and Gaia)
- Photometry (VIS flux, Gaia flux)
- Astrometry (RUWE, excess noise)
- Classification probabilities (Gaia class_prob_*, PHZ probabilities)
- PHZ classifications and thresholds

Output: q1_gaia_phz_crossmatch_subset.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# Configuration
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'output'

# Parameter reference file
PARAM_CSV = OUTPUT_DIR / 'gaia_q1_subset_para.csv'

# WSDB Configuration
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'


def load_parameter_list(param_csv_path):
    """
    Load parameter list from CSV file.

    Returns dict mapping parameter names to their WSDB column names and metadata.
    """
    if not param_csv_path.exists():
        print(f"ERROR: Parameter CSV not found: {param_csv_path}")
        return None

    df_params = pd.read_csv(param_csv_path)
    print(f"Loaded {len(df_params)} parameters from {param_csv_path}")

    # Create mapping of parameter names to WSDB column names
    param_dict = {}
    for _, row in df_params.iterrows():
        param_name = row['Parameter']
        wsdb_col_name = row['WSDB_Column']
        source = row['Source']
        category = row['Category']
        description = row['Description']

        param_dict[param_name] = {
            'wsdb_column': wsdb_col_name,
            'source': source,
            'category': category,
            'description': description
        }

    return param_dict, df_params


def extract_subset_from_wsdb(param_dict, username):
    """
    Extract subset of columns from WSDB table q1_gaia_phz_crossmatch.

    Args:
        param_dict: Dictionary with 'wsdb_column' for each parameter
        username: WSDB username

    Returns:
        DataFrame with extracted subset, and mapping of param names to wsdb columns
    """
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed!")
        print("Install with: pip install psycopg2")
        return None, None

    table_name = f'{username}.q1_gaia_phz_crossmatch'

    print(f"\nConnecting to WSDB...")
    try:
        conn = psycopg2.connect(
            host=WSDB_HOST,
            database=WSDB_DB,
            user=username
        )

        # Build SELECT statement with all WSDB columns
        wsdb_cols_to_select = [info['wsdb_column'] for info in param_dict.values()]
        select_list = ', '.join(wsdb_cols_to_select)

        query = f"""
            SELECT {select_list}
            FROM {table_name}
        """

        print(f"Extracting {len(wsdb_cols_to_select)} columns from {table_name}...")
        df = pd.read_sql(query, conn)

        conn.close()

        print(f"  ✓ Loaded {len(df):,} rows × {len(df.columns)} columns")

        # Create mapping for renaming
        wsdb_to_param = {info['wsdb_column']: param_name
                         for param_name, info in param_dict.items()}

        return df, wsdb_to_param

    except Exception as e:
        print(f"ERROR: {e}")
        return None, None


def rename_columns_to_param_names(df, wsdb_to_param):
    """
    Rename WSDB columns to parameter names from CSV.

    Maps from wsdb names (m_*, g_*, p_*) back to CSV names (q1_*, gaia_*, phz_*).
    """
    df_renamed = df.rename(columns=wsdb_to_param)
    return df_renamed


def main():
    """Main execution."""
    print("="*70)
    print("Extract Q1-Gaia-PHZ Subset from WSDB to Parquet")
    print("="*70)

    # Get username from environment
    username = os.getenv('PGUSER')
    if not username:
        print("\nERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return

    print(f"\nLogged in as: {username}")
    print(f"WSDB Host: {WSDB_HOST}")
    print(f"Database: {WSDB_DB}")

    # Load parameter list from CSV
    print("\n" + "="*70)
    print("Step 1: Load Parameter List")
    print("="*70)

    param_dict, df_params = load_parameter_list(PARAM_CSV)
    if param_dict is None:
        return

    print(f"  Loaded {len(param_dict)} parameters")
    print(f"    M (MER/Q1) columns: {sum(1 for info in param_dict.values() if info['wsdb_column'].startswith('m_'))}")
    print(f"    G (Gaia) columns: {sum(1 for info in param_dict.values() if info['wsdb_column'].startswith('g_'))}")
    print(f"    P (PHZ) columns: {sum(1 for info in param_dict.values() if info['wsdb_column'].startswith('p_'))}")

    # Extract from WSDB
    print("\n" + "="*70)
    print("Step 2: Extract from WSDB")
    print("="*70)

    df, wsdb_to_param = extract_subset_from_wsdb(param_dict, username)
    if df is None:
        print("Failed to extract data from WSDB")
        return

    # Rename columns back to CSV parameter names
    print("\nRenaming columns to parameter names from CSV...")
    df_final = rename_columns_to_param_names(df, wsdb_to_param)

    # Save to parquet
    print("\n" + "="*70)
    print("Step 3: Save to Parquet")
    print("="*70)

    output_path = DATA_DIR / 'q1_gaia_phz_crossmatch_subset.parquet'
    df_final.to_parquet(output_path, compression='snappy', index=False)

    print(f"\n✓ Subset saved to: {output_path}")
    print(f"  Shape: {df_final.shape[0]:,} rows × {df_final.shape[1]} columns")

    # Print summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)

    print(f"\nParameters extracted: {len(df_final.columns)}")
    print(f"  M (MER/Q1): {sum(1 for col in df_final.columns if col.startswith('q1_'))}")
    print(f"  G (Gaia): {sum(1 for col in df_final.columns if col.startswith('gaia_'))}")
    print(f"  P (PHZ): {sum(1 for col in df_final.columns if col.startswith('phz_'))}")

    print(f"\nCategories:")
    for category in sorted(df_params['Category'].unique()):
        count = (df_params['Category'] == category).sum()
        print(f"  {category}: {count}")

    print(f"\nColumn reference:")
    print(f"  See {PARAM_CSV} for full parameter descriptions and WSDB column mappings")

    print(f"\nData types:")
    for col in sorted(df_final.columns):
        print(f"  {col:30s} {str(df_final[col].dtype):15s}")

    print("\n" + "="*70)
    print("Extraction Complete!")
    print("="*70)
    print(f"\nOutput file: {output_path}")
    print(f"Shape: {df_final.shape[0]:,} rows × {df_final.shape[1]} columns")
    print(f"\nLoad the subset with:")
    print(f"  import pandas as pd")
    print(f"  df = pd.read_parquet('{output_path}')")


if __name__ == '__main__':
    main()
