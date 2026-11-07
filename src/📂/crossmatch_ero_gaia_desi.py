#!/usr/bin/env python3
"""
Build {username}.crossmatch_ero_gaia_0_3_desi — the Euclid ERO + Gaia cross-match
(crossmatch_ero_gaia_0_3) restricted to DESI-confirmed galaxies.

Direct ID join (no positional matching):
  desi_ero_galaxies.v_objid  =  crossmatch_ero_gaia_0_3.e_v_objid

DESI columns are carried through prefixed with d_ (e.g. d_desi_z, d_desi_spectype).
Use for: ERO transfer-learning evaluation with spectroscopic truth.

(Split out of the old crossmatch_desi.py; the Q1 half is
crossmatch_q1_mer_phz_vis_gaia_sep_desi.py.)

Requires WSDB (VPN). Auth via ~/.pgpass.

Usage:
  PGUSER=siwei_zhao_2025 python src/crossmatch_ero_gaia_0_3_desi.py
"""

import os
import sys

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB = "wsdb"


def create_table(conn, table_name: str, sql: str) -> int:
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
    conn.commit()
    cur.execute(sql)
    conn.commit()
    cur.execute(f"SELECT COUNT(*) FROM {table_name};")
    n = cur.fetchone()[0]
    cur.close()
    return n


def get_columns(conn, schema: str, table: str) -> list[str]:
    cur = conn.cursor()
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """, (schema, table))
    cols = [r[0] for r in cur.fetchall()]
    cur.close()
    return cols


def main():
    username = os.environ.get("PGUSER")
    if not username:
        print("ERROR: set PGUSER"); sys.exit(1)

    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    out = f"{username}.crossmatch_ero_gaia_0_3_desi"

    print("=" * 65)
    print(f"Building {out}")
    print("  ERO + Gaia  ×  DESI ERO (join on ero_objid = v_objid)")
    print("=" * 65)

    # d_*-prefixed DESI columns (drop the join key to avoid a duplicate column)
    cols = get_columns(conn, username, "desi_ero_galaxies")
    desi_ero_cols = ", ".join(f'd."{c}" AS d_{c}' for c in cols if c != "ero_objid")

    n = create_table(conn, out, f"""
        CREATE TABLE {out} AS
        SELECT
            e.*,
            {desi_ero_cols}
        FROM {username}.crossmatch_ero_gaia_0_3 AS e
        INNER JOIN {username}.desi_ero_galaxies AS d
            ON e.e_v_objid = d.v_objid
    """)
    conn.close()

    print(f"\n  -> {out}  ({n:,} rows)")
    print("  DESI columns prefixed d_  (e.g. d_desi_z, d_desi_spectype)")
    print("=" * 65)


if __name__ == "__main__":
    main()
