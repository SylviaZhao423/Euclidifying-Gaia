#!/usr/bin/env python3
"""Does seeding the PM-proxy builder from the 0.3" raw ERO-Gaia match give the
SAME 2-parameter candidate list as the 1.0" raw match?

build_gaia_pm_proxy.py enumerates its 2-par candidates as the distinct
`g_source_id` with NULL PM in crossmatch_ero_gaia_1_0_uncor (the wide net). This
script (1) lists which crossmatch_ero_gaia* / gaia_pm_proxy tables actually exist,
then (2) compares the 2-par candidate set from the narrow 0.3" raw match against
the wide 1.0" raw match, and (3) checks how many wide-net-only sources survive into
the final canonical crossmatch_ero_gaia_0_3 — i.e. whether the width matters.

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_pm_proxy_candidate_overlap.py
"""
import os
import sys

import psycopg2

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB = "wsdb"


def exists(cur, schema, table) -> bool:
    cur.execute("SELECT to_regclass(%s) IS NOT NULL", (f"{schema}.{table}",))
    return cur.fetchone()[0]


def par2_set_sql(schema, table):
    return (f"SELECT DISTINCT g_source_id FROM {schema}.{table} "
            f"WHERE g_pmra IS NULL AND g_pmdec IS NULL")


def main() -> int:
    schema = os.getenv("PGUSER")
    if not schema:
        print("ERROR: PGUSER not set", flush=True); return 1

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=schema,
                            connect_timeout=20)
    cur = conn.cursor()

    # ── 0. what actually exists ──────────────────────────────────────────────
    cur.execute("""
        SELECT tablename FROM pg_tables
        WHERE schemaname = %s
          AND (tablename LIKE 'crossmatch_ero_gaia%%' OR tablename = 'gaia_pm_proxy')
        ORDER BY tablename
    """, (schema,))
    tabs = [r[0] for r in cur.fetchall()]
    print("=" * 68, flush=True)
    print("Existing ERO-Gaia / proxy tables in", schema, flush=True)
    print("=" * 68, flush=True)
    for t in tabs:
        cur.execute(f"SELECT count(*) FROM {schema}.{t}")
        print(f"  {t:<42} {cur.fetchone()[0]:>12,} rows", flush=True)

    narrow, wide = "crossmatch_ero_gaia_0_3_uncor", "crossmatch_ero_gaia_1_0_uncor"
    have_narrow, have_wide = exists(cur, schema, narrow), exists(cur, schema, wide)
    if not (have_narrow and have_wide):
        print(f"\nCannot compare: narrow({narrow})={have_narrow}, "
              f"wide({wide})={have_wide} — one is missing.", flush=True)
        cur.close(); conn.close()
        return 0

    # ── 1. candidate 2-par set overlap: 0.3" raw vs 1.0" raw ──────────────────
    cur.execute(f"""
        WITH c03 AS ({par2_set_sql(schema, narrow)}),
             c10 AS ({par2_set_sql(schema, wide)})
        SELECT (SELECT count(*) FROM c03),
               (SELECT count(*) FROM c10),
               (SELECT count(*) FROM c03 JOIN c10 USING (g_source_id)),
               (SELECT count(*) FROM c10 WHERE g_source_id NOT IN (SELECT g_source_id FROM c03)),
               (SELECT count(*) FROM c03 WHERE g_source_id NOT IN (SELECT g_source_id FROM c10))
    """)
    n03, n10, common, only10, only03 = cur.fetchone()
    print("\n" + "=" * 68, flush=True)
    print("2-par candidate lists the PM-proxy builder would enumerate", flush=True)
    print("=" * 68, flush=True)
    print(f"  from {narrow} (narrow) : {n03:>8,}", flush=True)
    print(f"  from {wide} (wide)   : {n10:>8,}", flush=True)
    print(f"  in BOTH                                     : {common:>8,}", flush=True)
    print(f"  in 1.0\" only  (wide net adds these)         : {only10:>8,}", flush=True)
    print(f"  in 0.3\" only  (should be ~0; 0.3  subset 1.0): {only03:>8,}", flush=True)
    print(f"\n  -> identical set? {only10 == 0 and only03 == 0}", flush=True)

    # ── 2. does the difference reach the FINAL canonical table? ───────────────
    if exists(cur, schema, "crossmatch_ero_gaia_0_3"):
        cur.execute(f"""
            WITH c03 AS ({par2_set_sql(schema, narrow)}),
                 canon2 AS ({par2_set_sql(schema, 'crossmatch_ero_gaia_0_3')})
            SELECT (SELECT count(*) FROM canon2),
                   (SELECT count(*) FROM canon2 WHERE g_source_id NOT IN (SELECT g_source_id FROM c03))
        """)
        ncanon2, canon2_only = cur.fetchone()
        print("\n" + "=" * 68, flush=True)
        print("Do the wide-net-only sources reach the final canonical table?", flush=True)
        print("=" * 68, flush=True)
        print(f"  2-par sources in canonical crossmatch_ero_gaia_0_3 : {ncanon2:>8,}", flush=True)
        print(f"  ...of those, NOT in the narrow 0.3\" seed list      : {canon2_only:>8,}", flush=True)
        print("  (that last number = final 2-par matches that seeding from 0.3\"", flush=True)
        print("   could have missed / left zero-PM — the payoff of the wide net)", flush=True)

    cur.close(); conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
