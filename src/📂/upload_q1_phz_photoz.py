#!/usr/bin/env python3
"""
Pull the FULL Euclid Q1 catalogue.phz_photo_z (~29.6M rows) straight from the ESA archive
TAP (anonymous — Q1 is public) into WSDB, no manual VOTable download.

Companion to upload_q1_phz_all.py (which loaded the phz *classification*). This loads the
official photo-z catalogue: phz_median / phz_mode_1 / phz_mode_2 / phz_classification, plus
the flux_{vis,y,j,h}_unif + external u/g/r/i/z _unif fluxes PHZ actually used.

Method: keyset pagination on object_id (index-efficient, RESUMABLE) via astroquery.esa.euclid
async, COPY each chunk into {PGUSER}.q1_phz_photo_z. Re-running resumes from MAX(object_id)
already loaded. ~5.9k rows/s measured -> ~1.5-2 h for the full table.

Pulls the 58 SCALAR columns only; the 5 array columns are skipped (phz_pdf = 601-elt PDF,
phz_70/90/95_int = interval pairs, ie_cuts_weights) — arrays don't COPY cleanly and phz_pdf
would balloon the table. The scalar point estimates + classification + fluxes are what we need.

Needs BOTH: WSDB (Astronomy/Cambridge VPN, ~/.pgpass) and internet (ESA TAP).

Usage:
  PGUSER=siwei_zhao_2025 python src/📂/upload_q1_phz_photoz.py                 # full run (resumes)
  PGUSER=siwei_zhao_2025 python src/📂/upload_q1_phz_photoz.py --max-chunks 2  # smoke test
  PGUSER=siwei_zhao_2025 python src/📂/upload_q1_phz_photoz.py --chunk 100000
  PGUSER=siwei_zhao_2025 python src/📂/upload_q1_phz_photoz.py --fresh          # DROP + reload
"""
import argparse
import io
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB = "wsdb"
SRC = "catalogue.phz_photo_z"
TABLE = "q1_phz_photo_z"
PGTYPE = {"long": "BIGINT", "int": "INTEGER", "double": "DOUBLE PRECISION",
          "boolean": "BOOLEAN"}


def scalar_columns():
    """The 58 scalar columns of phz_photo_z (skip the 5 array columns), + their PG types."""
    from astroquery.esa.euclid import Euclid
    c = Euclid.launch_job(
        f"SELECT column_name, datatype, arraysize FROM tap_schema.columns "
        f"WHERE table_name='{SRC}'").get_results().to_pandas()
    c = c[c["arraysize"].isna()].sort_values("column_name")
    cols = list(c["column_name"])
    assert "object_id" in cols, "object_id not found among scalar columns"
    return cols, {r.column_name: r.datatype for r in c.itertuples()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk", type=int, default=200000, help="rows per TAP page (default 200k).")
    ap.add_argument("--max-chunks", type=int, default=None, help="stop after N chunks (smoke test).")
    ap.add_argument("--fresh", action="store_true", help="DROP the table and reload from scratch.")
    args = ap.parse_args()

    user = os.getenv("PGUSER")
    if not user:
        print("ERROR: set PGUSER"); return 1

    import psycopg2
    from astroquery.esa.euclid import Euclid

    cols, types = scalar_columns()
    colsql = ", ".join(cols)
    full = f"{user}.{TABLE}"
    print(f"{len(cols)} scalar columns -> {full}")

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                            connect_timeout=30, keepalives=1, keepalives_idle=30,
                            keepalives_interval=10, keepalives_count=5)
    cur = conn.cursor()
    if args.fresh:
        cur.execute(f"DROP TABLE IF EXISTS {full} CASCADE;"); conn.commit()
    coldefs = ", ".join(f'"{c}" {PGTYPE.get(types[c], "double precision")}' for c in cols)
    cur.execute(f"CREATE TABLE IF NOT EXISTS {full} ({coldefs});"); conn.commit()

    cur.execute(f"SELECT COUNT(*), MAX(object_id) FROM {full}")
    done, last = cur.fetchone()
    print(f"resume: {done:,} rows already loaded" + (f", last object_id={last}" if last is not None else " (empty)"))

    t0 = time.time(); nchunk = 0
    while True:
        where = f"WHERE object_id > {int(last)} " if last is not None else ""
        q = f"SELECT TOP {args.chunk} {colsql} FROM {SRC} {where}ORDER BY object_id"
        r = Euclid.launch_job_async(q).get_results().to_pandas()
        if len(r) == 0:
            break
        r = r[cols].replace([np.inf, -np.inf], np.nan)
        buf = io.StringIO()
        r.to_csv(buf, index=False, header=False, sep="\t", na_rep="\\N")
        buf.seek(0)
        cur.copy_expert(
            f"COPY {full} ({colsql}) FROM STDIN WITH (FORMAT csv, DELIMITER E'\t', NULL '\\N')", buf)
        conn.commit()
        done += len(r); last = int(r["object_id"].max()); nchunk += 1
        rate = done / max(1e-9, time.time() - t0)
        print(f"  chunk {nchunk}: +{len(r):,} -> {done:,} rows  (last={last}, {rate:,.0f} r/s)", flush=True)
        if len(r) < args.chunk:
            break
        if args.max_chunks and nchunk >= args.max_chunks:
            print(f"  stopped after {nchunk} chunks (--max-chunks)"); cur.close(); conn.close(); return 0

    print("  building index on object_id ...")
    cur.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_object_id_idx ON {full}(object_id);")
    conn.commit()
    print(f"DONE: {done:,} rows in {full}  ({time.time()-t0:.0f}s)")
    cur.close(); conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
