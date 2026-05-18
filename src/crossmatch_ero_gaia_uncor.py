#!/usr/bin/env python3
"""
ERO-Gaia cross-match (uncorrected) with configurable matching radius.

Legacy / pre-PM-correction pipeline preserved for traceability. The
canonical PM-corrected ERO-Gaia cross-match is now produced by
`crossmatch_ero_gaia.py`; this script exists only to regenerate the
uncorrected snapshot tables under the `_uncor` suffix.

Cross-matches all ERO sources from euclid_ero.merge_cat with Gaia DR3
using a q3c INNER JOIN LATERAL (closest match, LIMIT 1) against the raw
Gaia 2016 positions (no PM propagation).

Output table name: crossmatch_ero_gaia_<radius_with_underscores>_uncor
e.g. for radius 0.3″ → crossmatch_ero_gaia_0_3_uncor.

Usage:
    PGUSER=siwei_zhao_2025 python src/crossmatch_ero_gaia_uncor.py --radii 0.3
    PGUSER=siwei_zhao_2025 python src/crossmatch_ero_gaia_uncor.py --radii 0.3 1.0
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'


def get_column_names(cursor, schema, table):
    cursor.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """, (schema, table))
    return [r[0] for r in cursor.fetchall()]


def crossmatch(ero_columns, gaia_columns, username, radius_arcsec):
    import psycopg2

    ero_cols_str  = ', '.join([f'm.{c} AS e_{c}' for c in ero_columns])
    gaia_cols_str = ', '.join([f'g.{c} AS g_{c}' for c in gaia_columns])
    gaia_bare     = ', '.join([c.split(' AS ')[0] for c in gaia_cols_str.split(', ')])

    radius_suffix = str(radius_arcsec).replace('.', '_')
    table_name    = f'{username}.crossmatch_ero_gaia_{radius_suffix}_uncor'
    radius_deg    = radius_arcsec / 3600.0

    query = f"""
        DROP TABLE IF EXISTS {table_name};
        CREATE TABLE {table_name} AS
        SELECT
            {ero_cols_str},
            {gaia_cols_str}
        FROM euclid_ero.merge_cat AS m
        INNER JOIN LATERAL (
            SELECT {gaia_bare}
            FROM gaia_dr3.gaia_source AS g
            WHERE q3c_join(m.v_alpha_j2000, m.v_delta_j2000,
                           g.ra, g.dec, {radius_deg})
            ORDER BY q3c_dist(m.v_alpha_j2000, m.v_delta_j2000,
                              g.ra, g.dec) ASC
            LIMIT 1
        ) AS g ON true;
    """

    print(f"\n  Matching radius: {radius_arcsec}\"  →  table: {table_name}")
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    row_count = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    print(f"  Rows: {row_count:,}")
    return table_name, row_count


def main():
    parser = argparse.ArgumentParser(
        description="Cross-match ERO merge_cat with Gaia DR3 at one or more radii."
    )
    parser.add_argument(
        "--radii", nargs="+", type=float, required=True, metavar="ARCSEC",
        help="One or more matching radii in arcseconds, e.g. --radii 0.3 1.0"
    )
    args = parser.parse_args()

    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER not set"); return 1

    import psycopg2
    print(f"Connecting to WSDB as {username}...")
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()
    ero_columns  = get_column_names(cursor, 'euclid_ero', 'merge_cat')
    gaia_columns = get_column_names(cursor, 'gaia_dr3',   'gaia_source')
    cursor.close(); conn.close()
    print(f"  ERO columns:  {len(ero_columns)}")
    print(f"  Gaia columns: {len(gaia_columns)}")

    radii = sorted(set(args.radii))
    for radius in radii:
        crossmatch(ero_columns, gaia_columns, username, radius)

    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
