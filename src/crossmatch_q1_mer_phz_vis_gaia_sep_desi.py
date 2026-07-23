#!/usr/bin/env python3
"""
Build {username}.crossmatch_q1_mer_phz_vis_gaia_sep_desi — the Euclid Q1 MER + PHZ +
Gaia cross-match (vis_det=1, sep ≤ 0.3") restricted to DESI-confirmed galaxies.

Direct ID join (no positional matching):
  desi_q1_galaxies.euclid_object_id  =  crossmatch_q1_mer_phz_vis_gaia_sep.m_object_id

DESI columns are carried through prefixed with d_ (e.g. d_desi_z, d_desi_spectype).
Use for: XGBoost training/testing with spectroscopic ground truth.

(Split out of the old crossmatch_desi.py; the ERO half is crossmatch_ero_gaia_0_3_desi.py.)

Requires WSDB (VPN). Auth via ~/.pgpass.

Usage:
  PGUSER=siwei_zhao_2025 python src/crossmatch_q1_mer_phz_vis_gaia_sep_desi.py
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
    out = f"{username}.crossmatch_q1_mer_phz_vis_gaia_sep_desi"

    print("=" * 65)
    print(f"Building {out}")
    print("  Q1 MER + PHZ + Gaia (vis, sep)  ×  DESI Q1 (join on object_id)")
    print("=" * 65)

    # d_*-prefixed DESI columns (drop the join key to avoid a duplicate column)
    cols = get_columns(conn, username, "desi_q1_galaxies")
    desi_q1_cols = ", ".join(f'd."{c}" AS d_{c}' for c in cols if c != "euclid_object_id")

    n = create_table(conn, out, f"""
        CREATE TABLE {out} AS
        SELECT
            q.*,
            {desi_q1_cols}
        FROM {username}.crossmatch_q1_mer_phz_vis_gaia_sep AS q
        INNER JOIN {username}.desi_q1_galaxies AS d
            ON q.m_object_id = d.euclid_object_id
    """)
    conn.close()

    print(f"\n  -> {out}  ({n:,} rows)")
    print("  DESI columns prefixed d_  (e.g. d_desi_z, d_desi_spectype)")
    print("=" * 65)


if __name__ == "__main__":
    main()
