#!/usr/bin/env python3
"""
Euclid × WISE × Gaia base cross-match  →  {schema}.crossmatch_euclid_wise_gaia

This is the BASE table: every Euclid×Gaia source that also has a CatWISE match
within 1" (its W1/W2 measured), with NO colour cut applied. The QSO/AGN subsample
(W1-W2 >= 0.8, Stern+12 mid-IR wedge) is then pulled out of this table by
    extract_qso_euclid_wise_gaia.py  →  crossmatch_qso_euclid_wise_gaia

Mechanism
---------
We do NOT re-cross-match the raw 30M-row Euclid catalogues. The two existing
Euclid×Gaia canonical tables already carry the 14 g_<feature> columns +
g_astrometric_params_solved etc.:
    siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep   (Euclid Q1 MER+PHZ × Gaia)
    siwei_zhao_2025.crossmatch_ero_gaia_0_3              (Euclid ERO × Gaia)
Their Gaia sources are disjoint (Q1 EDF fields and ERO fields don't overlap on-sky),
so we UNION ALL them and q3c-match each source to catwise_202003.main (closest
within 1") for W1/W2. A `euclid` column tags each row 'Q1' or 'ERO'. Only sources
with a CatWISE match survive (that is what makes this the Euclid×WISE cross-match);
no W1-W2 threshold is applied here. Column schema matches the SDSS/DESI QSO
crossmatch tables, so downstream code reads it with no changes.

Requires WSDB (VPN). Auth via ~/.pgpass.

Usage:
    PGUSER=siwei_zhao_2025 python src/crossmatch_euclid_wise_gaia.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS  # noqa: E402
from qso_gaia_crossmatch_builder import WSDB_HOST, WSDB_DB  # noqa: E402

# Both Euclid×Gaia canonical tables feed the sample (their Gaia sources are disjoint:
# Q1 EDF fields and ERO fields don't overlap on-sky).
Q1_TABLE = "siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep"   # Euclid Q1 MER+PHZ × Gaia
ERO_TABLE = "siwei_zhao_2025.crossmatch_ero_gaia_0_3"             # Euclid ERO × Gaia (canonical)
CATWISE = "catwise_202003.main"
OUTPUT_TABLE = "crossmatch_euclid_wise_gaia"
WISE_RADIUS_ARCSEC = 1.0


def build_sql(output_table: str) -> str:
    r_deg = WISE_RADIUS_ARCSEC / 3600.0
    gbare = ", ".join(f"g_{c}" for c in SELECTED_GAIA_PARAMETERS)
    gfeat = ",\n             ".join(f"b.g_{c}" for c in SELECTED_GAIA_PARAMETERS)
    gcommon = ("g_source_id, g_ra, g_dec, g_astrometric_params_solved, "
               "g_phot_g_mean_mag, g_phot_g_n_obs")
    return f"""
    CREATE TABLE {output_table} AS
    WITH base AS MATERIALIZED (
        -- Euclid Q1 (MER + PHZ) × Gaia
        SELECT 'Q1'::text          AS euclid,
               m_object_id::bigint AS qso_id,
               m_right_ascension   AS qso_ra,
               m_declination       AS qso_dec,
               p_phz_qso_prob      AS phz_qso_prob,
               m_point_like_prob   AS point_like_prob,
               {gcommon}, {gbare}
        FROM {Q1_TABLE}
        WHERE m_right_ascension IS NOT NULL AND g_source_id IS NOT NULL
        UNION ALL
        -- Euclid ERO × Gaia (ERO has no PHZ → NULL)
        SELECT 'ERO'::text,
               e_v_objid::bigint,
               e_v_alpha_j2000, e_v_delta_j2000,
               NULL::real, NULL::real,
               {gcommon}, {gbare}
        FROM {ERO_TABLE}
        WHERE e_v_alpha_j2000 IS NOT NULL AND g_source_id IS NOT NULL
    )
    SELECT 'EUCLID_WISE'::text                            AS qso_catalog,
           b.euclid, b.qso_id, b.qso_ra, b.qso_dec,
           NULL::real                                     AS qso_z,
           b.g_source_id, b.g_ra, b.g_dec,
           q3c_dist(b.qso_ra, b.qso_dec, b.g_ra, b.g_dec) * 3600.0 AS gaia_sep_arcsec,
           1                                              AS n_gaia_within_1arcsec,
           -- NOTE: the Euclid-deblend blend flag (n_euclid_within_1arcsec) is NOT
           -- computed here — over the whole uncut base it is a per-row q3c count that
           -- makes the query too long to survive the WSDB connection. It is computed
           -- in extract_qso_euclid_wise_gaia.py for the ~6k W1-W2>=0.8 survivors only
           -- (which is why g_ra/g_dec are carried through here).
           b.g_astrometric_params_solved, b.g_phot_g_mean_mag, b.g_phot_g_n_obs,
             {gfeat},
           b.phz_qso_prob, b.point_like_prob,
           w.w1mpro AS w1, w.w2mpro AS w2, (w.w1mpro - w.w2mpro) AS w1_w2
    FROM base b
    CROSS JOIN LATERAL (
        SELECT cw.w1mpro, cw.w2mpro
        FROM {CATWISE} cw
        WHERE q3c_join(b.qso_ra, b.qso_dec, cw.ra, cw.dec, {r_deg})
        ORDER BY q3c_dist(b.qso_ra, b.qso_dec, cw.ra, cw.dec) ASC
        LIMIT 1
    ) w
    WHERE w.w1mpro IS NOT NULL AND w.w2mpro IS NOT NULL
    """


def main() -> int:
    import psycopg2
    user = os.getenv("PGUSER")
    if not user:
        print("ERROR: PGUSER not set (e.g. PGUSER=siwei_zhao_2025 python ...).")
        return 1
    out = f"{user}.{OUTPUT_TABLE}"
    print("=" * 78)
    print(f"Building {out}  (base Euclid×WISE×Gaia cross-match, no colour cut)")
    print(f"  Sources : {Q1_TABLE}  (Euclid Q1 × Gaia)")
    print(f"            {ERO_TABLE}  (Euclid ERO × Gaia)")
    print(f"  WISE    : {CATWISE}  (closest within {WISE_RADIUS_ARCSEC}\")")
    print("=" * 78)
    try:
        conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                                connect_timeout=30, keepalives=1, keepalives_idle=30,
                                keepalives_interval=10, keepalives_count=5)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 0")
        print("Running WISE match + CREATE TABLE ...")
        cur.execute(f"DROP TABLE IF EXISTS {out} CASCADE;")
        cur.execute(build_sql(out))
        conn.commit()
        cur.execute(f"CREATE INDEX ON {out} (g_source_id);")
        cur.execute(f"CREATE INDEX ON {out} (w1_w2);")
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
    print(f"\n  {out}: {n:,} Euclid×Gaia sources with a CatWISE match (no colour cut)")
    cur.execute(f"SELECT euclid, COUNT(*) FROM {out} GROUP BY 1 ORDER BY 1;")
    print("  by Euclid release:")
    for rel, k in cur.fetchall():
        print(f"    {rel:<4} {k:>9,}")
    cur.execute(f"SELECT COUNT(*) FROM {out} WHERE w1_w2 >= 0.8;")
    print(f"  of which W1-W2 >= 0.8 (QSO/AGN wedge): {cur.fetchone()[0]:,}")
    cur.execute(f"SELECT round(avg(w1_w2)::numeric,2), round(min(w1_w2)::numeric,2), "
                f"round(max(w1_w2)::numeric,2) FROM {out};")
    a, lo, hi = cur.fetchone()
    print(f"  W1-W2: mean {a}, range [{lo}, {hi}]")
    cur.close(); conn.close()
    print("=" * 78)
    print("Done. Extract the QSO subset:  python src/extract_qso_euclid_wise_gaia.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
