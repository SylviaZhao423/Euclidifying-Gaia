#!/usr/bin/env python3
"""
Interactive ERO-Gaia cross-match script with configurable matching radius.

Allows user to select which matching radius (0.1, 0.2, or 1.0 arcseconds)
to use for cross-matching ERO sources with Gaia DR3 sources.

Cross-matches all ERO sources from euclid_ero.merge_cat with ALL Gaia DR3
sources using spatial position matching via direct SQL LATERAL JOIN query
(stays entirely in WSDB).

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
    Table naming format: crossmatch_ero_gaia_[radius]
    Example: crossmatch_ero_gaia_0_05, crossmatch_ero_gaia_0_2, crossmatch_ero_gaia_1_0

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

        # Check if table already exists
        check_table_query = f"""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = '{username}' AND table_name = 'crossmatch_ero_gaia_{radius_suffix}'
            );
        """
        cursor.execute(check_table_query)
        table_exists = cursor.fetchone()[0]

        if table_exists:
            print(f"Table {table_name} already exists. Dropping and recreating...")
            drop_query = f"DROP TABLE IF EXISTS {table_name};"
            cursor.execute(drop_query)
            conn.commit()
            print(f"  ✓ Old table dropped")

        print(f"Creating table: {table_name}")
        print(f"  Matching radius: {radius_arcsec} arcsecond")
        print(f"  ERO columns: {len(ero_columns)}")
        print(f"  Gaia columns: {len(gaia_columns)}")
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


def get_radii_selection():
    """
    Interactive prompt to get custom radius value(s) for cross-matching.

    Returns:
        list of float: selected radii in arcseconds
    """
    return get_custom_radii()


def get_custom_radii():
    """
    Prompt user to enter custom radius value(s).

    Returns:
        list of float: custom radii in arcseconds
    """
    print("\n" + "="*70)
    print("Enter Custom Radius Value(s)")
    print("="*70)
    print("Enter one or more radius values in arcseconds, separated by commas.")
    print("Examples:")
    print("  - Single value: 0.08")
    print("  - Multiple values: 0.05,0.08,0.15")

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
            print("Example: 0.08 or 0.05,0.08,0.15")


def main():
    """Main execution."""
    import os

    print("="*70)
    print("ERO-Gaia Cross-Match - Interactive Mode")
    print("="*70)
    print("\nConfiguration:")
    print(f"  WSDB Host: {WSDB_HOST}")
    print(f"  Database: {WSDB_DB}")
    print(f"  ERO Table: euclid_ero.merge_cat")
    print(f"  Gaia Table: gaia_dr3.gaia_source (all columns)")
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

    # Get radius selection from user
    selected_radii = get_radii_selection()
    print(f"\nSelected radii: {', '.join([str(r) for r in selected_radii])} arcseconds")

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

    # Perform cross-matches for selected radii
    print("\n" + "="*70)
    print("Step 2: Execute Cross-Matches")
    print("="*70)

    results = {}
    step_num = 1
    for radius in selected_radii:
        print(f"\n{'-'*70}")
        print(f"Cross-match {step_num}/{len(selected_radii)}: {radius} arcsecond radius")
        print(f"{'-'*70}")

        success, table_name, row_count = crossmatch_ero_gaia_sql(
            ero_columns, gaia_columns, username, radius_arcsec=radius
        )

        if success and table_name is not None:
            results[radius] = {'table': table_name, 'rows': row_count}
        else:
            print(f"\nFailed to cross-match with {radius} arcsecond radius.")
            return

        step_num += 1

    # Results summary
    print("\n" + "="*70)
    print("Step 3: Results Summary")
    print("="*70)

    print(f"\nCross-match Tables Created:")
    for i, radius in enumerate(selected_radii, 1):
        if radius in results:
            table_name = results[radius]['table']
            row_count = results[radius]['rows']
            print(f"\n{i}. {radius} arcsecond radius:")
            print(f"   Table: {table_name}")
            print(f"   Rows: {row_count:,}")
            print(f"   Columns: {len(ero_columns) + len(gaia_columns)}")

    print(f"\nColumn Naming Convention:")
    print(f"  e_* = ERO columns (e.g., e_v_alpha_j2000, e_v_mag_iso)")
    print(f"  g_* = Gaia DR3 columns (e.g., g_source_id, g_phot_g_mean_mag)")

    print(f"\nColumn breakdown:")
    print(f"  ERO columns: {len(ero_columns)}")
    print(f"  Gaia columns: {len(gaia_columns)}")

    print(f"\nYou can now query these tables directly from WSDB:")
    for radius in selected_radii:
        if radius in results:
            table_name = results[radius]['table']
            print(f"  SELECT * FROM {table_name} LIMIT 10;")
            print(f"  SELECT COUNT(*) FROM {table_name};")

    print("\n" + "="*70)
    print("Cross-match complete!")
    print("="*70)
    print(f"\nCreated {len(results)} cross-match table(s):")
    for radius in selected_radii:
        if radius in results:
            print(f"  {radius} arcsec: {results[radius]['table']} ({results[radius]['rows']:,} rows)")
    print("\nUse sqlutilpy or psycopg2 to query these tables directly.")


if __name__ == '__main__':
    main()
