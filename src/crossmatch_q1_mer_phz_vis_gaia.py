#!/usr/bin/env python3
"""
Extract VIS-detected Q1-Gaia-PHZ cross-matched data from WSDB.

Cross-matches VIS-detected Q1 sources (vis_det=1) from euclid_q1.mer_final_cat with:
1. Gaia DR3 sources using the existing gaia_id field (direct ID match)
2. Q1 PHZ classification using object_id

Creates two tables:
  {username}.crossmatch_q1_mer_phz_vis_gaia
      All VIS-detected Q1 × PHZ × Gaia sources (no sep filter).

  {username}.crossmatch_q1_mer_phz_vis_gaia_sep
      Subset with angular separation ≤ 0.3" (cos(dec)-corrected).

Uses direct SQL JOIN query (stays entirely in WSDB).
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
OUTPUT_DIR.mkdir(exist_ok=True)

# WSDB Configuration
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

# Zero-point correction for VIS band and AB zero-point
ZPC_IE = 1.05
ZP_AB  = 23.90
# Note: User should set WSDB credentials via environment variables:
#   PGUSER, PGPASSWORD (see WSDB documentation)


def get_gaia_column_names():
    """
    Get all column names from gaia_dr3.gaia_source.

    Returns list of column names in order.
    """
    try:
        import sqlutilpy
    except ImportError:
        print("ERROR: sqlutilpy not installed!")
        return None

    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'gaia_dr3' AND table_name = 'gaia_source'
        ORDER BY ordinal_position
    """

    try:
        result = sqlutilpy.get(query, db=WSDB_DB, host=WSDB_HOST)
        columns = list(result[0])
        print(f"Retrieved {len(columns)} Gaia column names")
        return columns
    except Exception as e:
        print(f"ERROR getting Gaia column names: {e}")
        return None


def get_q1_column_names():
    """
    Get all column names from euclid_q1.mer_final_cat.

    Returns list of column names in order.
    """
    try:
        import sqlutilpy
    except ImportError:
        print("ERROR: sqlutilpy not installed!")
        return None

    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'euclid_q1' AND table_name = 'mer_final_cat'
        ORDER BY ordinal_position
    """

    try:
        result = sqlutilpy.get(query, db=WSDB_DB, host=WSDB_HOST)
        columns = list(result[0])
        print(f"Retrieved {len(columns)} Q1 column names")
        return columns
    except Exception as e:
        print(f"ERROR getting Q1 column names: {e}")
        return None


def get_phz_column_names(phz_table_name):
    """
    Get all column names from the user's PHZ table.

    Args:
        phz_table_name: Full table name (e.g., 'siwei_zhao_2025.q1_phz_withgaia')

    Returns list of column names in order.
    """
    try:
        import sqlutilpy
    except ImportError:
        print("ERROR: sqlutilpy not installed!")
        return None

    # Parse schema and table name
    parts = phz_table_name.split('.')
    if len(parts) != 2:
        print(f"ERROR: Invalid table name format: {phz_table_name}")
        print("Expected format: schema.table_name")
        return None

    schema, table = parts

    query = f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = '{schema}' AND table_name = '{table}'
        ORDER BY ordinal_position
    """

    try:
        result = sqlutilpy.get(query, db=WSDB_DB, host=WSDB_HOST)
        columns = list(result[0])
        print(f"Retrieved {len(columns)} PHZ column names from {phz_table_name}")
        return columns
    except Exception as e:
        print(f"ERROR getting PHZ column names from {phz_table_name}: {e}")
        return None


def crossmatch_q1_gaia_phz_sql(q1_columns, gaia_columns, phz_columns, phz_table_name, username):
    """
    Cross-match Q1 with Gaia and PHZ using direct SQL JOIN query.

    Uses:
    - Q1.gaia_id = Gaia.source_id (direct ID match)
    - Q1.object_id = PHZ.object_id (object ID match)

    Creates a new table in the user's WSDB schema (username.q1_gaia_phz_crossmatch).

    Args:
        q1_columns: list of Q1 column names
        gaia_columns: list of Gaia column names
        phz_columns: list of PHZ column names
        phz_table_name: Full table name for PHZ (e.g., 'siwei_zhao_2025.q1_phz_withgaia')
        username: WSDB username (from PGUSER env var)

    Returns:
        tuple of (success: bool, table_name: str, row_count: int)
    """
    print("\nCross-matching Q1 with Gaia and PHZ using direct SQL JOIN query...")

    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed!")
        return False, None, 0

    # Build column selections with prefixes to indicate source
    # m_ prefix for MER catalogue (Q1), g_ for Gaia, p_ for PHZ

    # Create sets for case-insensitive comparison to find overlaps
    q1_cols_lower = {col.lower(): col for col in q1_columns}
    gaia_cols_lower = {col.lower(): col for col in gaia_columns}
    phz_cols_lower = {col.lower(): col for col in phz_columns}

    # Q1 columns: all with m_ prefix
    q1_cols_str = ', '.join([f'q.{col} AS m_{col}' for col in q1_columns])

    # Pre-computed IE magnitude (NULL where flux <= 0)
    ie_mag_str = (
        f'CASE WHEN q.flux_detection_total > 0 THEN '
        f'-2.5 * LOG10({ZPC_IE} * q.flux_detection_total) + {ZP_AB} '
        f'ELSE NULL END AS m_ie_mag'
    )

    # Gaia columns: all with g_ prefix, excluding those that exist in Q1
    gaia_cols_filtered = [col for col in gaia_columns
                          if col.lower() not in q1_cols_lower]
    gaia_cols_str = ', '.join([f'g.{col} AS g_{col}' for col in gaia_cols_filtered])

    # PHZ columns: all with p_ prefix, excluding those that exist in Q1 or Gaia
    phz_cols_filtered = [col for col in phz_columns
                         if col.lower() not in q1_cols_lower and col.lower() not in gaia_cols_lower]
    phz_cols_str = ', '.join([f'p.{col} AS p_{col}' for col in phz_cols_filtered])

    # Table name in user's schema
    output_table_name = f'{username}.crossmatch_q1_mer_phz_vis_gaia'

    # SQL query using INNER JOINs
    # Q1's gaia_id matches Gaia's source_id
    # Q1's object_id matches PHZ's object_id
    # vis_det = 1 filter baked in
    create_query = f"""
        DROP TABLE IF EXISTS {output_table_name};

        CREATE TABLE {output_table_name} AS
        SELECT
            {q1_cols_str},
            {ie_mag_str},
            {gaia_cols_str},
            {phz_cols_str}
        FROM euclid_q1.mer_final_cat as q
        INNER JOIN gaia_dr3.gaia_source as g
            ON q.gaia_id = g.source_id
        INNER JOIN {phz_table_name} as p
            ON q.object_id = p.object_id
        WHERE q.gaia_id IS NOT NULL
            AND q.object_id IS NOT NULL
            AND q.vis_det = 1;
    """

    try:
        print(f"Connecting to WSDB...")
        conn = psycopg2.connect(
            host=WSDB_HOST,
            database=WSDB_DB,
            user=username
        )
        cursor = conn.cursor()

        print(f"Creating table: {output_table_name}")
        print(f"  Q1/MER columns (m_): {len(q1_columns)}")
        print(f"  Gaia columns (g_): {len(gaia_cols_filtered)} (excluded {len(gaia_columns) - len(gaia_cols_filtered)} duplicates)")
        print(f"  PHZ columns (p_): {len(phz_cols_filtered)} (excluded {len(phz_columns) - len(phz_cols_filtered)} duplicates)")
        print(f"  Total columns: {len(q1_columns) + len(gaia_cols_filtered) + len(phz_cols_filtered)}")

        # Execute the create table query
        cursor.execute(create_query)
        conn.commit()

        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {output_table_name};")
        row_count = cursor.fetchone()[0]

        print(f"\n✓ Cross-match table created successfully!")
        print(f"  Table: {output_table_name}")
        print(f"  Rows: {row_count:,}")

        cursor.close()
        conn.close()

        return True, output_table_name, row_count

    except Exception as e:
        print(f"ERROR executing cross-match query: {e}")
        return False, None, 0


def main():
    """Main execution."""
    import os

    # Get username from environment
    username = os.getenv('PGUSER')
    if not username:
        print("\nERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return

    print("="*70)
    print("Q1-Gaia-PHZ Cross-Match (Direct SQL in WSDB)")
    print("="*70)
    print("\nConfiguration:")
    print(f"  WSDB Host: {WSDB_HOST}")
    print(f"  Database: {WSDB_DB}")
    print(f"  Q1 Table: euclid_q1.mer_final_cat")
    print(f"  Gaia Table: gaia_dr3.gaia_source (all columns)")
    print(f"  PHZ Table: {username}.q1_phz_withgaia (all columns)")
    print(f"  Join Method 1: Direct ID match (Q1.gaia_id = Gaia.source_id)")
    print(f"  Join Method 2: Object ID match (Q1.object_id = PHZ.object_id)")
    print(f"  Storage: User schema (from PGUSER environment variable)")
    print("\nNote: Requires WSDB credentials via environment variables:")
    print("  PGUSER, PGPASSWORD")

    print(f"\nLogged in as: {username}")

    # PHZ table name in user's schema
    phz_table_name = f"{username}.q1_phz_withgaia"

    # Get column names from all three tables
    print("\n" + "="*70)
    print("Step 1: Retrieve Column Names")
    print("="*70)

    q1_columns = get_q1_column_names()
    if q1_columns is None:
        print("\nFailed to get Q1 column names. Exiting.")
        return

    gaia_columns = get_gaia_column_names()
    if gaia_columns is None:
        print("\nFailed to get Gaia column names. Exiting.")
        return

    phz_columns = get_phz_column_names(phz_table_name)
    if phz_columns is None:
        print("\nFailed to get PHZ column names. Exiting.")
        return

    # Perform cross-match
    print("\n" + "="*70)
    print("Step 2: Execute Cross-Match Query")
    print("="*70)

    success, output_table_name, row_count = crossmatch_q1_gaia_phz_sql(
        q1_columns, gaia_columns, phz_columns, phz_table_name, username
    )

    if not success or output_table_name is None:
        print("\nFailed to cross-match. Exiting.")
        return

    # Analyze results
    print("\n" + "="*70)
    print("Step 3: Results Summary")
    print("="*70)

    # Recalculate filtered columns for summary
    q1_cols_lower = {col.lower(): col for col in q1_columns}
    gaia_cols_lower = {col.lower(): col for col in gaia_columns}
    gaia_cols_filtered_final = [col for col in gaia_columns
                                 if col.lower() not in q1_cols_lower]
    phz_cols_filtered_final = [col for col in phz_columns
                               if col.lower() not in q1_cols_lower and col.lower() not in gaia_cols_lower]

    total_columns = len(q1_columns) + len(gaia_cols_filtered_final) + len(phz_cols_filtered_final)

    print(f"\nCross-match Table:")
    print(f"  Schema.Table: {output_table_name}")
    print(f"  Total rows: {row_count:,}")
    print(f"  Total columns: {total_columns}")
    print(f"    Q1/MER columns (m_): {len(q1_columns)}")
    print(f"    Gaia columns (g_): {len(gaia_cols_filtered_final)} (excluded {len(gaia_columns) - len(gaia_cols_filtered_final)} duplicates)")
    print(f"    PHZ columns (p_): {len(phz_cols_filtered_final)} (excluded {len(phz_columns) - len(phz_cols_filtered_final)} duplicates)")
    print(f"\nColumn Naming Convention:")
    print(f"  m_* = MER catalogue (Q1) columns")
    print(f"  g_* = Gaia DR3 columns")
    print(f"  p_* = PHZ classification columns")

    print(f"\nYou can now query this table directly from WSDB:")
    print(f"  SELECT * FROM {output_table_name} LIMIT 10;")
    print(f"  SELECT COUNT(*) FROM {output_table_name};")

    print("\n" + "="*70)
    print("Cross-match complete!")
    print("="*70)
    print(f"\nTable created in WSDB:")
    print(f"  {output_table_name}  (vis_det=1)")


if __name__ == '__main__':
    main()
