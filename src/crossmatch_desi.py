#!/usr/bin/env python3
"""
Join DESI-confirmed galaxy tables with existing Euclid/Gaia crossmatch tables.

All joins are direct ID matches (no positional matching needed):
  Q1 : desi_q1_galaxies.euclid_object_id  =  crossmatch_q1_*.m_object_id
  ERO: desi_ero_galaxies.ero_objid         =  crossmatch_ero_gaia_0_3.e_ero_objid

Creates three tables (DESI columns prefixed with d_):

  {username}.crossmatch_q1_mer_phz_desi
      Q1 MER + PHZ classifications for DESI-confirmed galaxies.
      Use for: studying PHZ selection effects against spectroscopic truth.

  {username}.crossmatch_q1_mer_phz_gaia_desi
      Q1 MER + PHZ + Gaia features for DESI-confirmed galaxies.
      Use for: XGBoost training/testing with spectroscopic ground truth.

  {username}.crossmatch_ero_gaia_0_3_desi
      ERO + Gaia features for DESI-confirmed galaxies.
      Use for: ERO transfer learning evaluation with spectroscopic truth.

Usage:
  PGUSER=siwei_zhao_2025 python src/crossmatch_desi.py
"""

import os
import sys

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"


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

    print("=" * 65)
    print("Crossmatching DESI galaxies with Euclid/Gaia tables")
    print(f"  User: {username}")
    print("=" * 65)

    # Build d_* prefixed column list for DESI tables, excluding join key
    # to avoid duplicate columns in the output
    def desi_cols(conn, table_suffix: str, exclude: list[str]) -> str:
        cols = get_columns(conn, username, table_suffix)
        return ", ".join([f'd."{c}" AS d_{c}' for c in cols if c not in exclude])

    # ── 1. crossmatch_q1_mer_phz_desi ────────────────────────────────
    print("\n[1] crossmatch_q1_mer_phz_desi")
    print("    Q1 MER + PHZ  ×  DESI Q1 (join on object_id)")
    desi_q1_cols = desi_cols(conn, "desi_q1_galaxies", exclude=["euclid_object_id"])
    n = create_table(conn,
        f"{username}.crossmatch_q1_mer_phz_desi",
        f"""
        CREATE TABLE {username}.crossmatch_q1_mer_phz_desi AS
        SELECT
            q.*,
            {desi_q1_cols}
        FROM {username}.crossmatch_q1_mer_phz AS q
        INNER JOIN {username}.desi_q1_galaxies AS d
            ON q.m_object_id = d.euclid_object_id
        """,
    )
    print(f"    -> {n:,} rows")

    # ── 2. crossmatch_q1_mer_phz_gaia_desi ───────────────────────────
    print("\n[2] crossmatch_q1_mer_phz_gaia_desi")
    print("    Q1 MER + PHZ + Gaia  ×  DESI Q1 (join on object_id)")
    n = create_table(conn,
        f"{username}.crossmatch_q1_mer_phz_gaia_desi",
        f"""
        CREATE TABLE {username}.crossmatch_q1_mer_phz_gaia_desi AS
        SELECT
            q.*,
            {desi_q1_cols}
        FROM {username}.crossmatch_q1_mer_phz_gaia AS q
        INNER JOIN {username}.desi_q1_galaxies AS d
            ON q.m_object_id = d.euclid_object_id
        """,
    )
    print(f"    -> {n:,} rows")

    # ── 3. crossmatch_ero_gaia_0_3_desi ──────────────────────────────
    print("\n[3] crossmatch_ero_gaia_0_3_desi")
    print("    ERO + Gaia  ×  DESI ERO (join on ero_objid)")
    desi_ero_cols = desi_cols(conn, "desi_ero_galaxies", exclude=["ero_objid"])
    n = create_table(conn,
        f"{username}.crossmatch_ero_gaia_0_3_desi",
        f"""
        CREATE TABLE {username}.crossmatch_ero_gaia_0_3_desi AS
        SELECT
            e.*,
            {desi_ero_cols}
        FROM {username}.crossmatch_ero_gaia_0_3 AS e
        INNER JOIN {username}.desi_ero_galaxies AS d
            ON e.e_v_objid = d.v_objid
        """,
    )
    print(f"    -> {n:,} rows")

    conn.close()

    print("\n" + "=" * 65)
    print("Done. Tables created:")
    print(f"  {username}.crossmatch_q1_mer_phz_desi")
    print(f"  {username}.crossmatch_q1_mer_phz_gaia_desi")
    print(f"  {username}.crossmatch_ero_gaia_0_3_desi")
    print("\nAll DESI columns are prefixed d_  (e.g. d_desi_z, d_desi_spectype)")
    print("=" * 65)


if __name__ == "__main__":
    main()
