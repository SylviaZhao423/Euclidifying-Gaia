#!/usr/bin/env python3
"""
Create crossmatch_q1_mer_phz_vis_gaia_sep_uncor: uncorrected sep <= 0.3"
subset of crossmatch_q1_mer_phz_vis_gaia.

This is the legacy / pre-PM-correction pipeline preserved for traceability.
The canonical sep-cut table (PM-corrected) is now produced by
`crossmatch_q1_mer_phz_vis_gaia_sep.py`; this script exists only to
regenerate the uncorrected snapshot under the `_uncor` name.

  {username}.crossmatch_q1_mer_phz_vis_gaia_sep_uncor
      SELECT * FROM crossmatch_q1_mer_phz_vis_gaia
      WHERE angular_separation <= 0.3"  (cos(dec)-corrected, small-angle)

Prerequisite: crossmatch_q1_mer_phz_vis_gaia must exist
              (run crossmatch_q1_mer_phz_vis_gaia.py first).

Usage:
    PGUSER=siwei_zhao_2025 python src/crossmatch_q1_mer_phz_vis_gaia_sep_uncor.py
"""

import os
import sys

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

SEP_EXPR = ("SQRT(POWER((m_right_ascension - g_ra) "
            "* COS(RADIANS(g_dec)), 2) "
            "+ POWER(m_declination - g_dec, 2)) * 3600")


def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        print("Set it with: export PGUSER=your_username")
        return 1

    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()

    source_table = f"{username}.crossmatch_q1_mer_phz_vis_gaia"
    output_table = f"{username}.crossmatch_q1_mer_phz_vis_gaia_sep_uncor"

    print("=" * 65)
    print(f"Creating sep <= 0.3\" subset")
    print(f"  Source: {source_table}")
    print(f"  Output: {output_table}")
    print(f"  Filter: {SEP_EXPR} <= 0.3")
    print("=" * 65)

    cursor.execute(f"DROP TABLE IF EXISTS {output_table} CASCADE;")
    cursor.execute(f"""
        CREATE TABLE {output_table} AS
        SELECT * FROM {source_table}
        WHERE {SEP_EXPR} <= 0.3
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
