#!/usr/bin/env python3
"""
Quaia quasar catalogue × Gaia DR3  →  {schema}.crossmatch_quaia_gaia

Quaia (Storey-Fisher et al. 2024, vasily.quaia_g20_v1, G<20, 755,850 sources) is a
Gaia-NATIVE quasar catalogue: every entry already carries a Gaia DR3 source_id and
a photometric redshift. So there is no coordinate cross-match — we simply JOIN on
source_id to pull the 14 UMAP features. A fourth, independent QSO sample to cross-
check the C1 point-like-extragalactic locus (after SDSS, DESI, Euclid+WISE).

Output uses the SAME column schema as the other QSO crossmatch tables, so
    python src/query_umap_qso.py --catalog quaia
reads it with no changes.

Requires WSDB (VPN). Auth via ~/.pgpass.

Usage:
    PGUSER=siwei_zhao_2025 python src/crossmatch_quaia_gaia.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from qso_gaia_crossmatch_builder import FEATURE_SQL, WSDB_HOST, WSDB_DB  # noqa: E402

QUAIA = "vasily.quaia_g20_v1"
OUTPUT_TABLE = "crossmatch_quaia_gaia"


def build_sql(out: str) -> str:
    return f"""
    CREATE TABLE {out} AS
    SELECT 'QUAIA'::text                    AS qso_catalog,
           q.source_id                      AS qso_id,
           q.ra                             AS qso_ra,
           q.dec                            AS qso_dec,
           q.redshift_quaia                 AS qso_z,
           g.source_id                      AS g_source_id,
           0.0                              AS gaia_sep_arcsec,
           1                                AS n_gaia_within_1arcsec,
           g.astrometric_params_solved      AS g_astrometric_params_solved,
           g.phot_g_mean_mag                AS g_phot_g_mean_mag,
           g.phot_g_n_obs                   AS g_phot_g_n_obs,
{FEATURE_SQL}
    FROM {QUAIA} q
    JOIN gaia_dr3.gaia_source g ON g.source_id = q.source_id
    """


def main() -> int:
    import psycopg2
    user = os.getenv("PGUSER")
    if not user:
        print("ERROR: PGUSER not set (e.g. PGUSER=siwei_zhao_2025 python ...).")
        return 1
    out = f"{user}.{OUTPUT_TABLE}"
    print("=" * 74)
    print(f"Building {out}")
    print(f"  Catalog : QUAIA  ({QUAIA})")
    print(f"  Match   : source_id JOIN to gaia_dr3.gaia_source (Gaia-native, no q3c)")
    print("=" * 74)
    try:
        conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                                connect_timeout=30, keepalives=1, keepalives_idle=30,
                                keepalives_interval=10, keepalives_count=5)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 0")
        print("Running JOIN + CREATE TABLE ...")
        cur.execute(f"DROP TABLE IF EXISTS {out} CASCADE;")
        cur.execute(build_sql(out))
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
    print(f"\n  {out}: {cur.fetchone()[0]:,} Quaia quasars with Gaia features")
    cur.execute(f"""SELECT g_astrometric_params_solved, COUNT(*)
                    FROM {out} GROUP BY 1 ORDER BY 1;""")
    print("  by astrometric_params_solved:")
    for aps, k in cur.fetchall():
        lbl = {3: "2-param", 31: "5-param", 95: "6-param"}.get(aps, str(aps))
        print(f"    {str(aps):<4} ({lbl:<7}) {k:>9,}")
    cur.close(); conn.close()
    print("=" * 74)
    print(f"Done. Project it:  python src/query_umap_qso.py --catalog quaia")
    return 0


if __name__ == "__main__":
    sys.exit(main())
