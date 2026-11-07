#!/usr/bin/env python3
"""
Create crossmatch_q1_mer_phz_vis: VIS-detected subset of crossmatch_q1_mer_phz.

  {username}.crossmatch_q1_mer_phz_vis
      SELECT * FROM crossmatch_q1_mer_phz WHERE m_vis_det = 1

Prerequisite: crossmatch_q1_mer_phz must exist (run crossmatch_q1_mer_phz.py first).

Usage:
    PGUSER=siwei_zhao_2025 python src/crossmatch_q1_mer_phz_vis.py
"""

import os
import sys

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'


def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return 1

    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()

    source_table = f"{username}.crossmatch_q1_mer_phz"
    output_table = f"{username}.crossmatch_q1_mer_phz_vis"

    print("=" * 65)
    print(f"Creating VIS-detected subset")
    print(f"  Source: {source_table}")
    print(f"  Output: {output_table}")
    print(f"  Filter: m_vis_det = 1")
    print("=" * 65)

    cursor.execute(f"DROP TABLE IF EXISTS {output_table} CASCADE;")
    cursor.execute(f"""
        CREATE TABLE {output_table} AS
        SELECT * FROM {source_table}
        WHERE m_vis_det = 1
    """)
    conn.commit()

    cursor.execute(f"SELECT COUNT(*) FROM {output_table};")
    row_count = cursor.fetchone()[0]

    cursor.close()
    conn.close()

    print(f"\n  Table: {output_table}")
    print(f"  Rows:  {row_count:,}")
    print("=" * 65)
    return 0


if __name__ == '__main__':
    sys.exit(main())
