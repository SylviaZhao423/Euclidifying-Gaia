#!/usr/bin/env python3
"""
Extract the Euclid+WISE photometric-quasar subset  →  {schema}.crossmatch_qso_euclid_wise_gaia

This EXTRACTS from the base cross-match table
    siwei_zhao_2025.crossmatch_euclid_wise_gaia   (built by crossmatch_euclid_wise_gaia.py)
which holds every Euclid×Gaia source that has a CatWISE match (W1/W2 measured, no
colour cut). Here we keep only the rows that satisfy the mid-IR AGN colour cut —
hence `extract_`, not `crossmatch_` (no new Gaia/WISE match is done).

We ALSO add the Euclid-deblend blend flag n_euclid_within_1arcsec here rather than in
the base: it is a per-row q3c count over the 30M-row Euclid MER catalogues, so running
it only for the ~6k W1-W2>=0.8 survivors (not the whole uncut base) keeps the query
short enough to survive the WSDB connection.

Selection (Stern et al. 2012 mid-IR AGN wedge):
    W1 - W2 >= 0.8   (CatWISE2020 Vega magnitudes w1mpro - w2mpro)

Supervisor's QSO check (2026-07-16): select quasars *photometrically* from Euclid
+ WISE mid-IR colour (independent of the SDSS/DESI spectroscopic lists) and project
onto the Q1 UMAP to see which clusters they land in — especially whether IR-selected
Euclid quasars pile into C1 (point-like extragalactic) and surface dust-reddened AGN
that optical selection misses.

The output keeps the base table's full column schema (incl. the 14 g_<feature>
columns), so analyze_umap_qso.py --catalog euclidwise reads it with no changes.

Requires WSDB (VPN). Auth via ~/.pgpass.

Usage:
    PGUSER=siwei_zhao_2025 python src/extract_qso_euclid_wise_gaia.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from qso_gaia_crossmatch_builder import WSDB_HOST, WSDB_DB  # noqa: E402

BASE_TABLE = "crossmatch_euclid_wise_gaia"           # {user}.<this>, built upstream
OUTPUT_TABLE = "crossmatch_qso_euclid_wise_gaia"
W1_W2_CUT = 0.8


def build_sql(base_table: str, output_table: str) -> str:
    return f"""
    CREATE TABLE {output_table} AS
    WITH cut AS MATERIALIZED (
        SELECT * FROM {base_table} WHERE w1_w2 >= {W1_W2_CUT}
    )
    SELECT c.*,
           -- Euclid deblends finer than Gaia: how many Euclid MER objects fall within
           -- 1" of THIS Gaia detection (Q1 → mer_final_cat, ERO → merge_cat). Computed
           -- HERE (not in the base) so the q3c count runs only for the ~6k survivors,
           -- keeping the query short enough to survive the WSDB connection.
           CASE WHEN c.euclid = 'Q1' THEN
                (SELECT count(*) FROM euclid_q1.mer_final_cat me
                   WHERE q3c_join(c.g_ra, c.g_dec, me.right_ascension, me.declination, 1.0/3600.0))
           ELSE (SELECT count(*) FROM euclid_ero.merge_cat me
                   WHERE q3c_join(c.g_ra, c.g_dec, me.v_alpha_j2000, me.v_delta_j2000, 1.0/3600.0))
           END                                            AS n_euclid_within_1arcsec
    FROM cut c
    """


def main() -> int:
    import psycopg2
    user = os.getenv("PGUSER")
    if not user:
        print("ERROR: PGUSER not set (e.g. PGUSER=siwei_zhao_2025 python ...).")
        return 1
    base = f"{user}.{BASE_TABLE}"
    out = f"{user}.{OUTPUT_TABLE}"
    print("=" * 78)
    print(f"Extracting {out}")
    print(f"  From    : {base}  (base Euclid×WISE×Gaia cross-match)")
    print(f"  Select  : W1 - W2 >= {W1_W2_CUT}  (Stern+12 mid-IR AGN wedge, Vega)")
    print("=" * 78)
    try:
        conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                                connect_timeout=30, keepalives=1, keepalives_idle=30,
                                keepalives_interval=10, keepalives_count=5)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 0")
        cur.execute("SELECT to_regclass(%s)", (base,))
        if cur.fetchone()[0] is None:
            print(f"MISSING BASE: {base} not found — run crossmatch_euclid_wise_gaia.py first.")
            return 1
        print("Applying colour cut + CREATE TABLE ...")
        cur.execute(f"DROP TABLE IF EXISTS {out} CASCADE;")
        cur.execute(build_sql(base, out))
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

    cur.execute(f"SELECT COUNT(*) FROM {out};")
    n = cur.fetchone()[0]
    print(f"\n  {out}: {n:,} Euclid+WISE quasar candidates with a Gaia match")
    cur.execute(f"SELECT euclid, COUNT(*) FROM {out} GROUP BY 1 ORDER BY 1;")
    print("  by Euclid release:")
    for rel, k in cur.fetchall():
        print(f"    {rel:<4} {k:>9,}")
    cur.execute(f"""SELECT g_astrometric_params_solved, COUNT(*)
                    FROM {out} GROUP BY 1 ORDER BY 1;""")
    print("  by astrometric_params_solved:")
    for aps, k in cur.fetchall():
        lbl = {3: "2-param", 31: "5-param", 95: "6-param"}.get(aps, str(aps))
        print(f"    {str(aps):<4} ({lbl:<7}) {k:>9,}")
    cur.execute(f"SELECT round(avg(w1_w2)::numeric,2), round(min(w1_w2)::numeric,2), "
                f"round(max(w1_w2)::numeric,2) FROM {out};")
    a, lo, hi = cur.fetchone()
    print(f"  W1-W2: mean {a}, range [{lo}, {hi}]")
    cur.close(); conn.close()
    print("=" * 78)
    print(f"Done. Project it:  python src/analyze_umap_qso.py --catalog euclidwise")
    return 0


if __name__ == "__main__":
    sys.exit(main())
