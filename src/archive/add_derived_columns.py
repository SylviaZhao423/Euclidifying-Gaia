#!/usr/bin/env python3
"""
Add derived columns to crossmatch tables in WSDB.

This script adds computed columns (like corrected_bp_rp_excess) directly to the
crossmatch tables so they don't need to be recomputed on every query.

Features:
- Checks if column already exists before adding
- Skips computation if column already populated
- Can be run multiple times safely (idempotent)

Usage:
    python src/add_derived_columns.py
"""

import os
import sys
import psycopg2

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'

# Tables to update
TABLES = [
    'siwei_zhao_2025.crossmatch_q1_mer_phz_gaia',
    'siwei_zhao_2025.crossmatch_ero_gaia_0_3',
]

# Define derived columns to add
# Each entry: (column_name, column_type, sql_expression, dependencies)
# dependencies: list of columns that must be non-null for computation
DERIVED_COLUMNS = [
    {
        'name': 'g_corrected_bp_rp_excess',
        'type': 'real',
        'expression': """
            g_phot_bp_rp_excess_factor - (
                CASE
                    WHEN g_bp_rp < 0.5 THEN
                        1.154360 + 0.033772 * g_bp_rp + 0.032277 * g_bp_rp * g_bp_rp
                    WHEN g_bp_rp < 4.0 THEN
                        1.162004 + 0.011464 * g_bp_rp + 0.049255 * g_bp_rp * g_bp_rp
                        - 0.005879 * g_bp_rp * g_bp_rp * g_bp_rp
                    ELSE
                        1.057572 + 0.140537 * g_bp_rp
                END
            )
        """,
        'dependencies': ['g_phot_bp_rp_excess_factor', 'g_bp_rp'],
        'description': 'Corrected BP/RP excess factor C* from Riello et al. 2021',
    },
    # Add more derived columns here as needed, e.g.:
    # {
    #     'name': 'g_some_other_feature',
    #     'type': 'real',
    #     'expression': 'g_col1 / g_col2',
    #     'dependencies': ['g_col1', 'g_col2'],
    #     'description': 'Description of the feature',
    # },
]


def column_exists(cursor, table, column):
    """Check if a column exists in a table."""
    schema, table_name = table.split('.')
    cursor.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s AND column_name = %s
        )
    """, (schema, table_name, column))
    return cursor.fetchone()[0]


def column_is_populated(cursor, table, column):
    """Check if a column has any non-null values."""
    cursor.execute(f"SELECT EXISTS (SELECT 1 FROM {table} WHERE {column} IS NOT NULL LIMIT 1)")
    return cursor.fetchone()[0]


def add_column(cursor, table, column_name, column_type):
    """Add a column to a table."""
    cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column_name} {column_type}")


def populate_column(cursor, table, column_name, expression, dependencies):
    """Populate a column with computed values."""
    # Build WHERE clause to only update rows where dependencies are available
    where_clause = ' AND '.join([f"{dep} IS NOT NULL" for dep in dependencies])

    sql = f"""
        UPDATE {table}
        SET {column_name} = {expression}
        WHERE {where_clause}
    """
    cursor.execute(sql)
    return cursor.rowcount


def process_table(conn, table):
    """Process a single table, adding all derived columns."""
    cursor = conn.cursor()
    print(f"\n{'='*60}")
    print(f"Processing: {table}")
    print('='*60)

    for col_def in DERIVED_COLUMNS:
        col_name = col_def['name']
        col_type = col_def['type']
        expression = col_def['expression']
        dependencies = col_def['dependencies']
        description = col_def.get('description', '')

        print(f"\n  Column: {col_name}")
        if description:
            print(f"    ({description})")

        # Check if column exists
        if column_exists(cursor, table, col_name):
            print(f"    ✓ Column already exists")

            # Check if already populated
            if column_is_populated(cursor, table, col_name):
                print(f"    ✓ Column already populated - skipping")
                continue
            else:
                print(f"    → Column exists but empty - populating...")
        else:
            print(f"    → Adding column...")
            add_column(cursor, table, col_name, col_type)
            conn.commit()
            print(f"    ✓ Column added")

        # Populate the column
        print(f"    → Computing values...")
        rows_updated = populate_column(cursor, table, col_name, expression, dependencies)
        conn.commit()
        print(f"    ✓ Updated {rows_updated:,} rows")

    cursor.close()


def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return 1

    print("="*60)
    print("Add Derived Columns to WSDB Tables")
    print("="*60)
    print(f"Username: {username}")
    print(f"Tables: {len(TABLES)}")
    print(f"Derived columns to add: {len(DERIVED_COLUMNS)}")
    for col in DERIVED_COLUMNS:
        print(f"  - {col['name']}: {col.get('description', '')}")

    try:
        conn = psycopg2.connect(
            host=WSDB_HOST,
            database=WSDB_DB,
            user=username
        )
        print(f"\n✓ Connected to {WSDB_HOST}")

        for table in TABLES:
            process_table(conn, table)

        conn.close()

        print("\n" + "="*60)
        print("Done! All derived columns added successfully.")
        print("="*60)
        print("\nNext steps:")
        print("  1. Add 'corrected_bp_rp_excess' to GAIA_QUERY_COLUMNS in selected_parameters_list.py")
        print("  2. Remove it from DERIVED_FEATURES")
        print("  3. Remove compute_derived_features() calls from training scripts")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
