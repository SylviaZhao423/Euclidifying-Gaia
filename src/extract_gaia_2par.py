#!/usr/bin/env python3
"""Materialise the Gaia DR3 2-parameter (position-only) sources into your own
WSDB table.

    astrometric_params_solved = 3  →  ~343,964,953 rows  (18.99% of gaia_source)

This is a large CREATE TABLE AS SELECT against the 1.8-billion-row
gaia_dr3.gaia_source on a *shared* server — expect tens of minutes and a table
of order tens of GB (depending on how many columns you keep). A q3c spatial
index is built by default so the table is usable for cone / neighbour queries
(e.g. an all-sky PM proxy); building it on ~344M rows is itself a long step.

Usage:
    # default: source_id, ra, dec, phot_g_mean_mag  +  q3c index  → siwei_zhao_2025.gaia_2par
    PGUSER=siwei_zhao_2025 python src/extract_gaia_2par.py

    # just see the SQL, run nothing
    PGUSER=siwei_zhao_2025 python src/extract_gaia_2par.py --dry-run

    # custom column set / table name / no index
    PGUSER=siwei_zhao_2025 python src/extract_gaia_2par.py \
        --table gaia_2par --columns source_id ra dec pmra pmdec phot_g_mean_mag l b
    PGUSER=siwei_zhao_2025 python src/extract_gaia_2par.py --columns ALL --no-index
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import psycopg2

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB = "wsdb"
SOURCE = "gaia_dr3.gaia_source"
DEFAULT_COLUMNS = ["source_id", "ra", "dec", "phot_g_mean_mag"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="gaia_2par",
                    help="output table name in your PGUSER schema (default: gaia_2par)")
    ap.add_argument("--columns", nargs="+", default=DEFAULT_COLUMNS,
                    help="columns to copy (default: source_id ra dec phot_g_mean_mag). "
                         "Pass ALL for every column (much larger).")
    ap.add_argument("--aps", type=int, default=3,
                    help="astrometric_params_solved to select (default 3 = 2-param; "
                         "31 = 5-param, 95 = 6-param)")
    ap.add_argument("--index", dest="index", action="store_true", default=True,
                    help="build a q3c spatial index on (ra, dec) [default]")
    ap.add_argument("--no-index", dest="index", action="store_false",
                    help="skip the q3c index")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the SQL and exit without touching the DB")
    args = ap.parse_args()

    schema = os.getenv("PGUSER")
    if not schema:
        print("ERROR: PGUSER not set"); return 1

    out = f"{schema}.{args.table}"
    all_cols = len(args.columns) == 1 and args.columns[0].upper() == "ALL"
    cols_sql = "*" if all_cols else ", ".join(args.columns)
    have_radec = all_cols or ({"ra", "dec"} <= set(c.lower() for c in args.columns))

    ctas = (f"CREATE TABLE {out} AS\n"
            f"SELECT {cols_sql}\n"
            f"FROM {SOURCE}\n"
            f"WHERE astrometric_params_solved = {args.aps};")
    idx = (f"CREATE INDEX ON {out} (q3c_ang2ipix(ra, dec));" if (args.index and have_radec)
           else None)

    print("=" * 70)
    print(f"Target : {out}")
    print(f"Filter : astrometric_params_solved = {args.aps}")
    print(f"Columns: {cols_sql}")
    print(f"q3c idx: {'yes' if idx else 'no'}"
          + ("" if have_radec or not args.index else "  (skipped: ra/dec not in column list)"))
    print("=" * 70)
    print("\n-- SQL --\nDROP TABLE IF EXISTS " + out + " CASCADE;\n" + ctas
          + (("\n" + idx) if idx else "") + f"\nANALYZE {out};")

    if args.dry_run:
        print("\n[dry-run] nothing executed.")
        return 0

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=schema)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 0;")   # this is a long job — no timeout

    print(f"\n[1/3] DROP + CREATE TABLE AS SELECT ...  (this is the long one)")
    t0 = time.time()
    cur.execute(f"DROP TABLE IF EXISTS {out} CASCADE;")
    cur.execute(ctas)
    cur.execute(f"SELECT count(*) FROM {out};")
    n = cur.fetchone()[0]
    print(f"      → {n:,} rows written  ({time.time()-t0:.0f} s)")

    if idx:
        print(f"[2/3] Building q3c index on (ra, dec) ...")
        t1 = time.time()
        cur.execute(idx)
        print(f"      → index built  ({time.time()-t1:.0f} s)")
    else:
        print("[2/3] q3c index skipped")

    print(f"[3/3] ANALYZE ...")
    cur.execute(f"ANALYZE {out};")

    cur.close(); conn.close()
    print(f"\nDone. {out}  ({n:,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
