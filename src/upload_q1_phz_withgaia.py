#!/usr/bin/env python3
"""
Upload Q1 PHZ data with Gaia IDs from VOTable file to WSDB.

Reads the VOTable file and uploads it to WSDB as a new table
in the user's schema named 'q1_phz_withgaia'.
"""

import pandas as pd
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# WSDB Configuration
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

# Input file
VOTABLE_FILE = Path('/Users/sylviazhao/Downloads/III/Euclidifying-Gaia/data/q1_phz_withgaia.vot')


def upload_votable_to_wsdb(votable_path, table_name, username):
    """
    Read VOTable file and upload to WSDB.

    Args:
        votable_path: Path to VOTable file
        table_name: Name for new table in WSDB (schema.table)
        username: WSDB username

    Returns:
        bool: Success or failure
    """
    try:
        import psycopg2
        from astropy.io import votable
    except ImportError:
        print("ERROR: Required packages not installed!")
        print("Install with: pip install psycopg2 astropy")
        return False

    print("=" * 80)
    print(f"Uploading VOTable to WSDB")
    print("=" * 80)

    # Check file exists
    if not votable_path.exists():
        print(f"ERROR: File not found: {votable_path}")
        return False

    print(f"\nReading VOTable file...")
    print(f"  File: {votable_path}")
    print(f"  Size: {votable_path.stat().st_size / (1024**2):.1f} MB")

    try:
        # Read VOTable
        print(f"  Parsing VOTable...")
        vot = votable.parse(str(votable_path))

        # Get first table from VOTable
        if len(vot.resources) > 0 and len(vot.resources[0].tables) > 0:
            table = vot.resources[0].tables[0]
            df = table.to_table().to_pandas()
        else:
            print("ERROR: Could not find table in VOTable")
            return False

        print(f"  ✓ Loaded {len(df):,} rows")
        print(f"  ✓ {len(df.columns)} columns")

        # Show column names
        print(f"\nColumn names:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col} ({df[col].dtype})")

        # Connect to WSDB
        print(f"\nConnecting to WSDB...")
        conn = psycopg2.connect(
            host=WSDB_HOST,
            database=WSDB_DB,
            user=username
        )
        cursor = conn.cursor()

        # Drop table if exists
        print(f"Dropping existing table if it exists...")
        cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
        conn.commit()

        # Create table from dataframe
        print(f"Creating table: {table_name}")

        # Build CREATE TABLE statement
        col_defs = []
        for col in df.columns:
            dtype = df[col].dtype

            # Map pandas dtype to PostgreSQL type
            if dtype == 'object':
                pg_type = 'TEXT'
            elif dtype == 'int64':
                pg_type = 'BIGINT'
            elif dtype == 'int32':
                pg_type = 'INTEGER'
            elif dtype == 'float64':
                pg_type = 'DOUBLE PRECISION'
            elif dtype == 'float32':
                pg_type = 'REAL'
            elif dtype == 'bool':
                pg_type = 'BOOLEAN'
            else:
                pg_type = 'TEXT'

            col_defs.append(f'"{col}" {pg_type}')

        create_sql = f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(col_defs) + "\n);"

        print(f"Executing CREATE TABLE...")
        cursor.execute(create_sql)
        conn.commit()

        # Insert data
        print(f"Inserting {len(df):,} rows...")

        # Build INSERT statement
        col_names = ', '.join([f'"{col}"' for col in df.columns])
        placeholders = ', '.join(['%s'] * len(df.columns))
        insert_sql = f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders})"

        # Insert in batches
        batch_size = 1000
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            rows = [tuple(row) for row in batch.values]
            cursor.executemany(insert_sql, rows)
            conn.commit()
            print(f"  Inserted {min(i + batch_size, len(df)):,}/{len(df):,} rows")

        # Get final count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        final_count = cursor.fetchone()[0]

        print(f"\n✓ Upload successful!")
        print(f"  Table: {table_name}")
        print(f"  Rows: {final_count:,}")
        print(f"  Columns: {len(df.columns)}")

        cursor.close()
        conn.close()

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution."""
    import os

    print("=" * 80)
    print("Upload Q1 PHZ with Gaia IDs to WSDB")
    print("=" * 80)

    # Get username from environment
    username = os.getenv('PGUSER')
    if not username:
        print("\nERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return

    print(f"\nLogged in as: {username}")
    print(f"WSDB Host: {WSDB_HOST}")
    print(f"Database: {WSDB_DB}")

    # Check file exists
    if not VOTABLE_FILE.exists():
        print(f"\nERROR: VOTable file not found!")
        print(f"  Expected at: {VOTABLE_FILE}")
        return

    # Upload to WSDB
    table_name = f"{username}.q1_phz_withgaia"

    if upload_votable_to_wsdb(VOTABLE_FILE, table_name, username):
        print("\n" + "=" * 80)
        print("Upload Complete!")
        print("=" * 80)
        print(f"\nYou can now query the table with:")
        print(f"  SELECT * FROM {table_name} LIMIT 10;")
        print(f"  SELECT COUNT(*) FROM {table_name};")
    else:
        print("\nUpload failed. Please check the error messages above.")


if __name__ == '__main__':
    main()
