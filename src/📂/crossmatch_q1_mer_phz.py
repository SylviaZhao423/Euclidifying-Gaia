#!/usr/bin/env python3
"""
Cross-match of Q1 MER and Q1 PHZ in WSDB using object_id with proper prefixing.

Creates a new table crossmatch_q1_mer_phz by joining:
- euclid_q1.mer_final_cat (Q1 MER catalogue) with m_* prefix
- {username}.q1_phz_all (Q1 PHZ classification) with p_* prefix

Join key: object_id

Column naming convention:
- m_* = Q1 MER columns
- p_* = PHZ classification columns
- m_ie_mag: pre-computed AB magnitude = -2.5 * log10(ZPC * m_flux_detection_total) + 23.90
            ZPC = 1.05 (VIS band), only defined where flux_detection_total > 0
"""

# Zero-point correction for VIS band (ZPC_IE) and AB zero-point
ZPC_IE = 1.05
ZP_AB  = 23.90

import os
from pathlib import Path

# WSDB Configuration
WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB = 'wsdb'


def get_mer_column_names(cursor, username):
    """
    Get all column names from euclid_q1.mer_final_cat.

    Returns list of column names in order.
    """
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'euclid_q1' AND table_name = 'mer_final_cat'
        ORDER BY ordinal_position
    """
    cursor.execute(query)
    columns = [row[0] for row in cursor.fetchall()]
    return columns


def crossmatch_q1_mer_phz(username):
    """
    Cross-match Q1 MER with PHZ using object_id with proper prefixing.

    Args:
        username: WSDB username

    Returns:
        tuple of (success: bool, row_count: int)
    """
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed!")
        return False, 0

    phz_table = f"{username}.q1_phz_all"
    output_table = f"{username}.crossmatch_q1_mer_phz"  # Table name matches filename

    print("="*80)
    print("Q1 MER-PHZ Cross-Match (With Proper Prefixing)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Q1 MER Table: euclid_q1.mer_final_cat")
    print(f"  PHZ Table: {phz_table}")
    print(f"  Output Table: {output_table}")
    print(f"  Join Key: object_id")
    print(f"  Naming Convention: m_* (MER), p_* (PHZ)")

    try:
        print(f"\nConnecting to WSDB as {username}...")
        conn = psycopg2.connect(
            host=WSDB_HOST,
            database=WSDB_DB,
            user=username
        )
        cursor = conn.cursor()

        # Get MER column names
        print(f"Retrieving MER column names...")
        mer_columns = get_mer_column_names(cursor, username)
        print(f"  Retrieved {len(mer_columns)} Q1 MER columns")

        # Build column selections with prefixes
        # m_ prefix for MER columns
        mer_cols_str = ', '.join([f'q.{col} AS m_{col}' for col in mer_columns])

        # Pre-computed IE magnitude (NULL where flux <= 0)
        ie_mag_str = (
            f'CASE WHEN q.flux_detection_total > 0 THEN '
            f'-2.5 * LOG10({ZPC_IE} * q.flux_detection_total) + {ZP_AB} '
            f'ELSE NULL END AS m_ie_mag'
        )

        # PHZ columns with p_ prefix
        phz_cols_str = ', '.join([
            f'p.phz_classification AS p_phz_classification',
            f'p.phz_star_prob AS p_phz_star_prob',
            f'p.phz_gal_prob AS p_phz_gal_prob',
            f'p.phz_qso_prob AS p_phz_qso_prob',
            f'p.phz_glob_cl_prob AS p_phz_glob_cl_prob',
            f'p.star_threshold AS p_star_threshold',
            f'p.galaxy_threshold AS p_galaxy_threshold',
            f'p.globc_threshold AS p_globc_threshold'
        ])

        # Drop table if exists
        print(f"Dropping existing table if it exists...")
        cursor.execute(f"DROP TABLE IF EXISTS {output_table} CASCADE;")
        conn.commit()

        # Create cross-matched table with proper prefixes
        print(f"Creating cross-match table with prefixes...")
        create_query = f"""
            CREATE TABLE {output_table} AS
            SELECT
                {mer_cols_str},
                {ie_mag_str},
                {phz_cols_str}
            FROM euclid_q1.mer_final_cat q
            INNER JOIN {phz_table} p
                ON q.object_id = p.object_id
            WHERE q.object_id IS NOT NULL
        """

        cursor.execute(create_query)
        conn.commit()

        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {output_table};")
        row_count = cursor.fetchone()[0]

        print(f"\n✓ Cross-match table created successfully!")
        print(f"  Table: {output_table}")
        print(f"  Rows: {row_count:,}")

        # Get column count
        cursor.execute(f"""
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = 'crossmatch_q1_mer_phz'
        """, (username,))
        col_count = cursor.fetchone()[0]
        print(f"  Columns: {col_count}")
        print(f"    MER columns (m_): {len(mer_columns)}")
        print(f"    Derived columns:  1 (m_ie_mag)")
        print(f"    PHZ columns (p_): 8")

        cursor.close()
        conn.close()

        return True, row_count

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def main():
    """Main execution."""
    # Get username from environment
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return

    # Perform cross-match
    success, row_count = crossmatch_q1_mer_phz(username)

    if success:
        output_table = f"{username}.crossmatch_q1_mer_phz"
        print("\n" + "="*80)
        print("Cross-match Complete!")
        print("="*80)
        print(f"\nData is stored in WSDB at: {output_table}")
        print(f"\nColumn Naming Convention:")
        print(f"  m_* = Q1 MER columns (e.g., m_object_id, m_mag_stargal_sep)")
        print(f"  p_* = PHZ classification columns (e.g., p_phz_classification, p_phz_star_prob)")
        print(f"\nYou can query it with:")
        print(f"  SELECT * FROM {output_table} LIMIT 10;")
        print(f"  SELECT COUNT(*) FROM {output_table};")
        print(f"  SELECT p_phz_classification, COUNT(*) FROM {output_table}")
        print(f"    GROUP BY p_phz_classification ORDER BY p_phz_classification;")
    else:
        print("\nCross-match failed. Please check the error messages above.")


if __name__ == '__main__':
    main()
