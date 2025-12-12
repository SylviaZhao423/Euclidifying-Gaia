#!/usr/bin/env python3
"""
Cross-match ERO with Gaia using direct SQL query with proper prefixing.

Cross-matches all ERO sources from euclid_ero.merge_cat with ALL Gaia DR3
sources using spatial position matching (configurable radius) via direct
SQL LATERAL JOIN query (stays entirely in WSDB).

Creates cross-matched tables with proper column prefixing:
  - e_* prefix for ERO columns
  - g_* prefix for Gaia columns

Supports three matching radii:
  - 0.05 arcsecond (crossmatch_ero_gaia_0_05)
  - 0.2 arcsecond (crossmatch_ero_gaia_0_2)
  - 1.0 arcsecond (crossmatch_ero_gaia_1_0)

This is more efficient than uploading data locally - the cross-match happens
entirely in the database.

Uses q3c spatial join functions available in WSDB PostgreSQL.
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


def get_gaia_column_names(cursor):
    """
    Get all column names from gaia_dr3.gaia_source.

    Returns list of column names in order.
    """
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'gaia_dr3' AND table_name = 'gaia_source'
        ORDER BY ordinal_position
    """
    cursor.execute(query)
    columns = [row[0] for row in cursor.fetchall()]
    return columns


def get_ero_column_names(cursor):
    """
    Get all column names from euclid_ero.merge_cat.

    Returns list of column names in order.
    """
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'euclid_ero' AND table_name = 'merge_cat'
        ORDER BY ordinal_position
    """
    cursor.execute(query)
    columns = [row[0] for row in cursor.fetchall()]
    return columns


def crossmatch_ero_gaia_sql(ero_columns, gaia_columns, username, radius_arcsec=1.0):
    """
    Cross-match ERO with Gaia using direct SQL LATERAL JOIN query with proper prefixing.

    Creates a new table in the user's WSDB schema with configurable matching radius.
    Options:
      - 0.05 arcsecond: crossmatch_ero_gaia_0_05
      - 0.2 arcsecond: crossmatch_ero_gaia_0_2
      - 1.0 arcsecond: crossmatch_ero_gaia_1_0

    This is more efficient than downloading locally - the cross-match happens
    entirely in the database.

    Args:
        ero_columns: list of ERO column names
        gaia_columns: list of Gaia column names
        username: WSDB username (from PGUSER env var)
        radius_arcsec: matching radius in arcseconds (default: 1.0)

    Returns:
        tuple of (success: bool, table_name: str, row_count: int)
    """
    print(f"\nCross-matching ERO with Gaia using direct SQL query (radius: {radius_arcsec} arcsec)...")

    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed!")
        return False, None, 0

    # Build column selections with prefixes
    # e_ prefix for ERO columns, g_ prefix for Gaia columns
    ero_cols_str = ', '.join([f'm.{col} AS e_{col}' for col in ero_columns])
    gaia_cols_str = ', '.join([f'g.{col} AS g_{col}' for col in gaia_columns])

    # Table name in user's schema - format: crossmatch_ero_gaia_[radius]
    radius_suffix = str(radius_arcsec).replace('.', '_')
    table_name = f'{username}.crossmatch_ero_gaia_{radius_suffix}'

    # Convert arcseconds to degrees
    radius_degrees = radius_arcsec / 3600.0

    # SQL query using LATERAL JOIN with q3c spatial functions
    create_query = f"""
        DROP TABLE IF EXISTS {table_name};

        CREATE TABLE {table_name} AS
        SELECT
            {ero_cols_str},
            {gaia_cols_str}
        FROM euclid_ero.merge_cat as m
        LEFT JOIN LATERAL (
            SELECT {', '.join([col.split(' AS ')[0] for col in gaia_cols_str.split(', ')])}
            FROM gaia_dr3.gaia_source as g
            WHERE q3c_join(m.v_alpha_j2000, m.v_delta_j2000,
                          g.ra, g.dec, {radius_degrees})
            ORDER BY q3c_dist(m.v_alpha_j2000, m.v_delta_j2000,
                             g.ra, g.dec) ASC
            LIMIT 1
        ) as g ON true;
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
        print(f"  Matching radius: {radius_arcsec} arcsecond")
        print(f"  ERO columns (e_): {len(ero_columns)}")
        print(f"  Gaia columns (g_): {len(gaia_columns)}")
        print(f"  Total columns: {len(ero_columns) + len(gaia_columns)}")

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
    print("ERO-Gaia Cross-Match with Three Radii (Direct SQL in WSDB)")
    print("="*70)
    print("\nConfiguration:")
    print(f"  WSDB Host: {WSDB_HOST}")
    print(f"  Database: {WSDB_DB}")
    print(f"  ERO Table: euclid_ero.merge_cat")
    print(f"  Gaia Table: gaia_dr3.gaia_source (all columns)")
    print(f"  Cross-match radii: 0.05, 0.2, and 1.0 arcseconds")
    print(f"  Method: Direct SQL LATERAL JOIN (table stored in WSDB)")
    print(f"  Naming Convention: e_* (ERO), g_* (Gaia)")
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

    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed!")
        return

    try:
        conn = psycopg2.connect(
            host=WSDB_HOST,
            database=WSDB_DB,
            user=username
        )
        cursor = conn.cursor()

        print(f"Retrieving ERO column names...")
        ero_columns = get_ero_column_names(cursor)
        print(f"  Retrieved {len(ero_columns)} ERO columns")

        print(f"Retrieving Gaia column names...")
        gaia_columns = get_gaia_column_names(cursor)
        print(f"  Retrieved {len(gaia_columns)} Gaia columns")

        cursor.close()
        conn.close()
    except Exception as e:
        print(f"ERROR getting column names: {e}")
        return

    # Step 2: Perform cross-match with 0.05 arcsecond radius
    print("\n" + "="*70)
    print("Step 2: Create 0.05 arcsecond Cross-Match")
    print("="*70)

    success_005, table_name_005, row_count_005 = crossmatch_ero_gaia_sql(
        ero_columns, gaia_columns, username, radius_arcsec=0.05
    )

    if not success_005 or table_name_005 is None:
        print("\nFailed to cross-match with 0.05 arcsecond radius. Exiting.")
        return

    # Step 3: Perform cross-match with 0.2 arcsecond radius
    print("\n" + "="*70)
    print("Step 3: Create 0.2 arcsecond Cross-Match")
    print("="*70)

    success_02, table_name_02, row_count_02 = crossmatch_ero_gaia_sql(
        ero_columns, gaia_columns, username, radius_arcsec=0.2
    )

    if not success_02 or table_name_02 is None:
        print("\nFailed to cross-match with 0.2 arcsecond radius. Exiting.")
        return

    # Step 4: Create 1.0 arcsecond cross-match
    print("\n" + "="*70)
    print("Step 4: Create 1.0 arcsecond Cross-Match")
    print("="*70)

    success_1, table_name_1, row_count_1 = crossmatch_ero_gaia_sql(
        ero_columns, gaia_columns, username, radius_arcsec=1.0
    )

    # Analyze results
    print("\n" + "="*70)
    print("Step 5: Results Summary")
    print("="*70)

    print(f"\nCross-match Tables Created:")
    print(f"\n1. 0.05 arcsecond radius (STRICT):")
    print(f"   Table: {table_name_005}")
    print(f"   Rows: {row_count_005:,}")
    print(f"   Columns: {len(ero_columns) + len(gaia_columns)}")

    print(f"\n2. 0.2 arcsecond radius (MODERATE):")
    print(f"   Table: {table_name_02}")
    print(f"   Rows: {row_count_02:,}")
    print(f"   Columns: {len(ero_columns) + len(gaia_columns)}")

    if success_1:
        print(f"\n3. 1.0 arcsecond radius (LOOSE):")
        print(f"   Table: {table_name_1}")
        print(f"   Rows: {row_count_1:,}")
        print(f"   Columns: {len(ero_columns) + len(gaia_columns)}")

    print(f"\nColumn Naming Convention:")
    print(f"  e_* = ERO columns (e.g., e_v_alpha_j2000, e_v_mag_iso)")
    print(f"  g_* = Gaia DR3 columns (e.g., g_source_id, g_phot_g_mean_mag)")

    print(f"\nColumn breakdown:")
    print(f"  ERO columns: {len(ero_columns)}")
    print(f"  Gaia columns: {len(gaia_columns)}")

    print(f"\nYou can now query these tables directly from WSDB:")
    print(f"  SELECT * FROM {table_name_005} LIMIT 10;")
    print(f"  SELECT COUNT(*) FROM {table_name_005};")
    print(f"  SELECT * FROM {table_name_02} LIMIT 10;")
    print(f"  SELECT COUNT(*) FROM {table_name_02};")
    if success_1:
        print(f"  SELECT * FROM {table_name_1} LIMIT 10;")
        print(f"  SELECT COUNT(*) FROM {table_name_1};")

    print("\n" + "="*70)
    print("Cross-match complete!")
    print("="*70)
    print(f"\nMatching radii summary:")
    print(f"  Strict (0.05 arcsec): {table_name_005} ({row_count_005:,} rows)")
    print(f"  Moderate (0.2 arcsec): {table_name_02} ({row_count_02:,} rows)")
    if success_1:
        print(f"  Loose (1.0 arcsec): {table_name_1} ({row_count_1:,} rows)")
    print(f"\nUse sqlutilpy or psycopg2 to query these tables directly.")


if __name__ == '__main__':
    main()
