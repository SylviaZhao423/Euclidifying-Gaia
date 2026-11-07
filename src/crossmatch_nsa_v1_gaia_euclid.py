#!/usr/bin/env python3
"""
NSA-v1 nuclei that also have Euclid imaging  →  {schema}.crossmatch_nsa_v1_gaia_euclid

Exact Gaia source_id join (no second positional match): keep the crossmatch_nsa_v1_gaia
rows whose central Gaia source (has_gaia) also appears in a Euclid×Gaia table — Q1 or ERO
— and tag each row 'Q1'/'ERO'. The output carries ALL the NSA + Gaia feature columns of
crossmatch_nsa_v1_gaia, so it projects onto the Q1 UMAP (analyze_umap_nsa.py --catalog
v1_euclid) and later drives the NSA Euclid cutout scripts.

    crossmatch_nsa_v1_gaia
  ⋈ (crossmatch_q1_mer_phz_vis_gaia_sep  ∪  crossmatch_ero_gaia_0_3)   by g_source_id

Requires WSDB — Astronomy/Cambridge VPN. Auth via ~/.pgpass.

Usage:
    PGUSER=siwei_zhao_2025 python src/crossmatch_nsa_v1_gaia_euclid.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from builder_qso_gaia_crossmatch import WSDB_HOST, WSDB_DB  # noqa: E402

NSA = "crossmatch_nsa_v1_gaia"
Q1  = "crossmatch_q1_mer_phz_vis_gaia_sep"
ERO = "crossmatch_ero_gaia_0_3"
OUTPUT_TABLE = "crossmatch_nsa_v1_gaia_euclid"


def build_sql(user: str, out: str) -> str:
    nsa, q1, ero = f"{user}.{NSA}", f"{user}.{Q1}", f"{user}.{ERO}"
    return f"""
    CREATE TABLE {out} AS
    SELECT n.*,
           CASE WHEN EXISTS (SELECT 1 FROM {q1} e WHERE e.g_source_id = n.g_source_id)
                THEN 'Q1' ELSE 'ERO' END AS euclid
    FROM {nsa} n
    WHERE n.has_gaia AND n.g_source_id IS NOT NULL
      AND (EXISTS (SELECT 1 FROM {q1}  e WHERE e.g_source_id = n.g_source_id)
        OR EXISTS (SELECT 1 FROM {ero} e WHERE e.g_source_id = n.g_source_id))
    """


def main() -> int:
    import psycopg2
    user = os.getenv("PGUSER")
    if not user:
        print("ERROR: PGUSER not set (e.g. PGUSER=siwei_zhao_2025 python ...)."); return 1
    out = f"{user}.{OUTPUT_TABLE}"
    print("=" * 78)
    print(f"Building {out}  (NSA-v1 nuclei with Euclid imaging, Gaia source_id join)")
    print("=" * 78)
    try:
        conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 0")
        cur.execute(f"DROP TABLE IF EXISTS {out} CASCADE;")
        cur.execute(build_sql(user, out))
        conn.commit()
        cur.execute(f"CREATE INDEX ON {out} (g_source_id);")
        conn.commit()
    except Exception as exc:
        msg = str(exc).lower()
        if "permission denied" in msg:
            print(f"NO ACCESS: {exc}")
        elif "does not exist" in msg or "column" in msg:
            print(f"SCHEMA MISMATCH: {exc}")
        else:
            print(f"ERROR: {exc}\n  Is the VPN up? Plain eduroam blocks WSDB:5432.")
        return 1

    cur.execute(f"SELECT count(*) FROM {out};")
    n = cur.fetchone()[0]
    print(f"\n  {out}: {n:,} NSA nuclei with Euclid imaging")
    cur.execute(f"SELECT euclid, COUNT(*) FROM {out} GROUP BY 1 ORDER BY 1;")
    for rel, k in cur.fetchall():
        print(f"    {rel:<4} {k:>6,}")
    cur.execute(f"""SELECT g_astrometric_params_solved, COUNT(*) FROM {out} GROUP BY 1 ORDER BY 1;""")
    print("  by astrometric_params_solved:")
    for aps, k in cur.fetchall():
        lbl = {3: "2-param", 31: "5-param", 95: "6-param"}.get(aps, str(aps))
        print(f"    {str(aps):<4} ({lbl:<7}) {k:>6,}")
    cur.close(); conn.close()
    print("=" * 78)
    print("Done. Project it:  python src/analyze_umap_nsa.py --catalog v1_euclid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
