#!/usr/bin/env python3
"""Euclid-side blend flag for the Euclid+WISE quasar sample.

The Gaia-side count (n_gaia_within_1arcsec) is trivially 1 here — every source is
one Euclid object matched to its closest Gaia detection. The physically meaningful
question is the REVERSE: Euclid (VIS ~0.16") resolves finer than Gaia (~0.4"), so a
single Gaia detection can be a blend of several Euclid objects. This counts, for
each Gaia detection in the sample, how many Euclid MER objects lie within 1", and
writes n_euclid_within_1arcsec back into the projected CSV.

Reads:  output/umap/qso/arrays/qso_gaia_matches_euclidwise.csv  (gaia_source_id)
        WSDB {PGUSER}.crossmatch_euclid_wise_qso_gaia + gaia_dr3 + euclid_q1.mer_final_cat
Writes: adds n_euclid_within_1arcsec column back to the CSV
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
CSV = PROJECT_DIR / "output" / "umap" / "qso" / "arrays" / "qso_gaia_matches_euclidwise.csv"
WSDB_HOST, WSDB_DB = "wsdb.ast.cam.ac.uk", "wsdb"


def main() -> int:
    import psycopg2
    user = os.getenv("PGUSER", "siwei_zhao_2025")
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 0")
    print("Building temp table of the sample's Gaia positions...")
    cur.execute("""
        CREATE TEMP TABLE _ew AS
        SELECT ct.g_source_id, g.ra AS gra, g.dec AS gdec
        FROM {u}.crossmatch_euclid_wise_qso_gaia ct
        JOIN gaia_dr3.gaia_source g ON g.source_id = ct.g_source_id
    """.format(u=user))
    cur.execute("ANALYZE _ew")
    cur.execute("SELECT count(*) FROM _ew")
    print(f"  {cur.fetchone()[0]:,} Gaia positions; counting Euclid MER objects within 1\"...")
    cur.execute("""
        SELECT e.g_source_id, count(*) AS n_euclid
        FROM _ew e
        JOIN euclid_q1.mer_final_cat m
          ON q3c_join(e.gra, e.gdec, m.right_ascension, m.declination, 1.0/3600.0)
        GROUP BY e.g_source_id
    """)
    counts = pd.DataFrame(cur.fetchall(), columns=["gaia_source_id", "n_euclid_within_1arcsec"])
    counts["gaia_source_id"] = counts["gaia_source_id"].astype("int64")
    cur.close(); conn.close()

    df = pd.read_csv(CSV, dtype={"gaia_source_id": "int64"})
    df = df.drop(columns=["n_euclid_within_1arcsec"], errors="ignore").merge(
        counts, on="gaia_source_id", how="left")
    df["n_euclid_within_1arcsec"] = df["n_euclid_within_1arcsec"].fillna(1).astype(int)
    df.to_csv(CSV, index=False)

    n = len(df); nb = int((df["n_euclid_within_1arcsec"] > 1).sum())
    print(f"\n  {nb}/{n} ({100*nb/n:.1f}%) Gaia detections are Euclid-deblended "
          f"(>1 Euclid object within 1\")")
    print("  distribution:")
    print(df["n_euclid_within_1arcsec"].value_counts().sort_index().to_string())
    print(f"\n  -> updated {CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
