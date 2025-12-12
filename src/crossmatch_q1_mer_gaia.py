#!/usr/bin/env python3
"""
Extract full Q1-Gaia cross-matched data from WSDB using direct SQL query.

Cross-matches all Q1 sources from euclid_q1.mer_final_cat with Gaia DR3 sources
using the existing gaia_id field (direct ID match, simpler than spatial matching).

This is more efficient than ERO cross-match since Q1 already has gaia_id field.

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


def crossmatch_q1_gaia_sql(q1_columns, gaia_columns, username):
    """
    Cross-match Q1 with Gaia using direct SQL JOIN query.

    Uses the existing gaia_id field in Q1 to join with Gaia source_id.
    Creates a new table in the user's WSDB schema (username.q1_gaia_crossmatch).

    Args:
        q1_columns: list of Q1 column names
        gaia_columns: list of Gaia column names
        username: WSDB username (from PGUSER env var)

    Returns:
        tuple of (success: bool, table_name: str, row_count: int)
    """
    print("\nCross-matching Q1 with Gaia using direct SQL JOIN query...")

    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed!")
        return False, None, 0

    # Build column selections with prefixes to indicate source
    # m_ prefix for MER catalogue (Q1), g_ for Gaia

    # Create sets for case-insensitive comparison to find overlaps
    q1_cols_lower = {col.lower(): col for col in q1_columns}
    gaia_cols_lower = {col.lower(): col for col in gaia_columns}

    # Q1 columns: all with m_ prefix
    q1_cols_str = ', '.join([f'q.{col} AS m_{col}' for col in q1_columns])

    # Gaia columns: all with g_ prefix, excluding those that exist in Q1
    gaia_cols_filtered = [col for col in gaia_columns
                          if col.lower() not in q1_cols_lower]
    gaia_cols_str = ', '.join([f'g.{col} AS g_{col}' for col in gaia_cols_filtered])

    # Table name in user's schema
    table_name = f'{username}.crossmatch_q1_mer_gaia'

    # SQL query using INNER JOIN on gaia_id
    # Q1's gaia_id matches Gaia's source_id
    create_query = f"""
        DROP TABLE IF EXISTS {table_name};

        CREATE TABLE {table_name} AS
        SELECT
            {q1_cols_str},
            {gaia_cols_str}
        FROM euclid_q1.mer_final_cat as q
        INNER JOIN gaia_dr3.gaia_source as g
            ON q.gaia_id = g.source_id
        WHERE q.gaia_id IS NOT NULL;
    """

    try:
        print(f"Connecting to WSDB...")
        conn = psycopg2.connect(
            host=WSDB_HOST,
            database=WSDB_DB,
            user=username
        )
        cursor = conn.cursor()

        print(f"Creating table: {table_name}")
        print(f"  Q1/MER columns (m_): {len(q1_columns)}")
        print(f"  Gaia columns (g_): {len(gaia_cols_filtered)} (excluded {len(gaia_columns) - len(gaia_cols_filtered)} duplicates)")
        print(f"  Total columns: {len(q1_columns) + len(gaia_cols_filtered)}")

        # Execute the create table query
        cursor.execute(create_query)
        conn.commit()

        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cursor.fetchone()[0]

        print(f"\n✓ Cross-match table created successfully!")
        print(f"  Table: {table_name}")
        print(f"  Rows: {row_count:,}")

        cursor.close()
        conn.close()

        return True, table_name, row_count

    except Exception as e:
        print(f"ERROR executing cross-match query: {e}")
        return False, None, 0


def main():
    """Main execution."""
    import os

    print("="*70)
    print("Q1-Gaia Cross-Match (Direct SQL in WSDB)")
    print("="*70)
    print("\nConfiguration:")
    print(f"  WSDB Host: {WSDB_HOST}")
    print(f"  Database: {WSDB_DB}")
    print(f"  Q1 Table: euclid_q1.mer_final_cat")
    print(f"  Gaia Table: gaia_dr3.gaia_source (all columns)")
    print(f"  Join Method: Direct ID match (Q1.gaia_id = Gaia.source_id)")
    print(f"  Storage: User schema (from PGUSER environment variable)")
    print("\nNote: Requires WSDB credentials via environment variables:")
    print("  PGUSER, PGPASSWORD")

    # Get username from environment
    username = os.getenv('PGUSER')
    if not username:
        print("\nERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return

    print(f"\nLogged in as: {username}")

    # Get column names from both tables
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

    # Perform cross-match
    print("\n" + "="*70)
    print("Step 2: Execute Cross-Match Query")
    print("="*70)

    success, table_name, row_count = crossmatch_q1_gaia_sql(
        q1_columns, gaia_columns, username
    )

    if not success or table_name is None:
        print("\nFailed to cross-match. Exiting.")
        return

    # Analyze results
    print("\n" + "="*70)
    print("Step 3: Results Summary")
    print("="*70)

    # Recalculate filtered columns for summary
    q1_cols_lower = {col.lower(): col for col in q1_columns}
    gaia_cols_filtered_final = [col for col in gaia_columns
                                 if col.lower() not in q1_cols_lower]

    total_columns = len(q1_columns) + len(gaia_cols_filtered_final)

    print(f"\nCross-match Table:")
    print(f"  Schema.Table: {table_name}")
    print(f"  Total rows: {row_count:,}")
    print(f"  Total columns: {total_columns}")
    print(f"    Q1/MER columns (m_): {len(q1_columns)}")
    print(f"    Gaia columns (g_): {len(gaia_cols_filtered_final)} (excluded {len(gaia_columns) - len(gaia_cols_filtered_final)} duplicates)")
    print(f"\nColumn Naming Convention:")
    print(f"  m_* = MER catalogue (Q1) columns")
    print(f"  g_* = Gaia DR3 columns")

    print(f"\nYou can now query this table directly from WSDB:")
    print(f"  SELECT * FROM {table_name} LIMIT 10;")
    print(f"  SELECT COUNT(*) FROM {table_name};")
    print(f"  SELECT * FROM {table_name} WHERE source_id IS NOT NULL;")

    print("\n" + "="*70)
    print("Cross-match complete!")
    print("="*70)
    print(f"\nData is stored in WSDB at: {table_name}")
    print("Use sqlutilpy or psycopg2 to query it directly.")


if __name__ == '__main__':
    main()
