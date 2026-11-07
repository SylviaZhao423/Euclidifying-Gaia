#!/usr/bin/env python3
"""
QSO catalog × CatWISE 2020 cross-match table builder.

Companion to builder_qso_gaia_crossmatch.py / builder_qso_euclid_crossmatch.py, but the
TARGET is catwise_202003.main instead of Gaia/Euclid. Used by:
    crossmatch_qso_desi_dr1_catwise.py  →  {schema}.crossmatch_qso_desi_dr1_catwise

Matching is a q3c SPATIAL cross-match: for each source in the (small) input catalog,
every CatWISE source within MATCH_RADIUS_ARCSEC is found; DISTINCT ON keeps the CLOSEST
(then brightest in W1) CatWISE source per input source. W1/W2 magnitudes are carried
through. Unlike crossmatch_qso_catwise_*.py — where CatWISE is the driver, pre-filtered
by the Stern W1-W2>=0.8 AGN cut — here CatWISE is the TARGET and is matched in FULL (no
colour cut), so the W1/W2 photometry attached to a spectroscopic sample is unbiased.

The input catalog is the driver (small), so this is a one-shot CREATE TABLE (no chunking):
each input source is a single indexed q3c cone search into catwise_202003.main.

CFG dict keys (same interface as the Gaia/Euclid builders)
---------------------------------------------------------
Required: tag, table, id_col, ra_col, dec_col
Optional: z_col, where

Output columns (one row per input source, closest CatWISE match):
  qso_catalog, qso_id, qso_ra, qso_dec, [qso_z]
  catwise_source_name, catwise_ra, catwise_dec, catwise_sep_arcsec
  w1mpro, w2mpro, w1mpro_pm, w2mpro_pm, w1_w2
  n_catwise_within_1arcsec, all_catwise_ids_within_1arcsec, all_catwise_seps_within_1arcsec
  n_qso_per_catwise, all_qso_ids_per_catwise, is_mutual_blend

Requires WSDB — VPN up (plain eduroam blocks port 5432). Auth via ~/.pgpass.
"""
from __future__ import annotations

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
CATWISE_TABLE = "catwise_202003.main"
MATCH_RADIUS_ARCSEC = 1.0

_SESSION_SETS = (
    "SET statement_timeout = 0",
    "SET max_parallel_workers_per_gather = 4",
    "SET parallel_setup_cost = 0",
    "SET parallel_tuple_cost = 0",
    "SET min_parallel_table_scan_size = 0",
    "SET min_parallel_index_scan_size = 0",
)


def crossmatch_select(cfg: dict, extra_where: str | None = None) -> str:
    """Cross-match SELECT: one row per input source (closest CatWISE match), with
    bidirectional blend columns (CatWISE sources near each QSO, and QSOs sharing the
    same closest CatWISE source)."""
    r_deg = MATCH_RADIUS_ARCSEC / 3600.0
    z_col = cfg.get("z_col")
    where = cfg.get("where", "TRUE")
    if extra_where:
        where = f"({where}) AND ({extra_where})"
    z_xmatch = f"q.{z_col} AS qso_z," if z_col else ""
    z_select = "x.qso_z," if z_col else ""

    return f"""
      WITH
      xmatch AS (
          SELECT q.{cfg['id_col']}  AS qso_id,
                 q.{cfg['ra_col']}  AS qso_ra,
                 q.{cfg['dec_col']} AS qso_dec,
                 {z_xmatch}
                 c.source_name AS cw_source_name,
                 c.ra          AS cw_ra,
                 c.dec         AS cw_dec,
                 c.w1mpro      AS w1mpro,
                 c.w2mpro      AS w2mpro,
                 c.w1mpro_pm   AS w1mpro_pm,
                 c.w2mpro_pm   AS w2mpro_pm,
                 q3c_dist(q.{cfg['ra_col']}, q.{cfg['dec_col']}, c.ra, c.dec)*3600.0 AS catwise_sep_arcsec
          FROM {cfg['table']} AS q
          JOIN {CATWISE_TABLE} AS c
              ON q3c_join(q.{cfg['ra_col']}, q.{cfg['dec_col']}, c.ra, c.dec, {r_deg})
          WHERE {where}
      ),
      cw_agg AS (
          SELECT qso_id,
                 COUNT(*)                                                 AS n_catwise_within_1arcsec,
                 array_agg(cw_source_name ORDER BY catwise_sep_arcsec)    AS all_catwise_ids_within_1arcsec,
                 array_agg(catwise_sep_arcsec ORDER BY catwise_sep_arcsec) AS all_catwise_seps_within_1arcsec
          FROM xmatch GROUP BY qso_id
      ),
      closest AS (
          SELECT DISTINCT ON (qso_id) qso_id, cw_source_name
          FROM xmatch
          ORDER BY qso_id, catwise_sep_arcsec, w1mpro NULLS LAST, cw_source_name
      ),
      qso_agg AS (
          SELECT cw_source_name,
                 COUNT(*)          AS n_qso_per_catwise,
                 array_agg(qso_id) AS all_qso_ids_per_catwise
          FROM closest GROUP BY cw_source_name
      )
      SELECT DISTINCT ON (x.qso_id)
          '{cfg['tag']}'::text        AS qso_catalog,
          x.qso_id,
          x.qso_ra,
          x.qso_dec,
          {z_select}
          x.cw_source_name            AS catwise_source_name,
          x.cw_ra                     AS catwise_ra,
          x.cw_dec                    AS catwise_dec,
          x.catwise_sep_arcsec,
          x.w1mpro,
          x.w2mpro,
          x.w1mpro_pm,
          x.w2mpro_pm,
          (x.w1mpro - x.w2mpro)       AS w1_w2,
          ca.n_catwise_within_1arcsec,
          ca.all_catwise_ids_within_1arcsec,
          ca.all_catwise_seps_within_1arcsec,
          qa.n_qso_per_catwise,
          qa.all_qso_ids_per_catwise,
          (ca.n_catwise_within_1arcsec > 1 AND qa.n_qso_per_catwise > 1) AS is_mutual_blend
      FROM xmatch AS x
      JOIN cw_agg  AS ca ON ca.qso_id          = x.qso_id
      JOIN qso_agg AS qa ON qa.cw_source_name  = x.cw_source_name
      ORDER BY x.qso_id, x.catwise_sep_arcsec, x.w1mpro NULLS LAST, x.cw_source_name
    """


def build_table(username: str, cfg: dict, table_name: str) -> int:
    """DROP + CREATE {username}.{table_name}, index it, print a summary."""
    import psycopg2
    output_table = f"{username}.{table_name}"
    print("=" * 74)
    print(f"Building {output_table}")
    print(f"  Catalog : {cfg['tag']}  ({cfg['table']})  ->  {CATWISE_TABLE} (FULL, no AGN cut)")
    print(f"  Match   : closest CatWISE source within {MATCH_RADIUS_ARCSEC}\" per source (q3c)")
    print("=" * 74)
    try:
        conn = psycopg2.connect(
            host=WSDB_HOST, database=WSDB_DB, user=username,
            connect_timeout=30, keepalives=1, keepalives_idle=30,
            keepalives_interval=10, keepalives_count=5,
        )
        cur = conn.cursor()
        for stmt in _SESSION_SETS:
            cur.execute(stmt)
        print("Running cross-match + CREATE TABLE ...")
        cur.execute(f"DROP TABLE IF EXISTS {output_table} CASCADE;")
        cur.execute(f"CREATE TABLE {output_table} AS\n{crossmatch_select(cfg)};")
        conn.commit()
        cur.execute(f"CREATE INDEX ON {output_table} (qso_id);")
        cur.execute(f"CREATE INDEX ON {output_table} (catwise_source_name);")
        conn.commit()
    except Exception as exc:
        msg = str(exc).lower()
        if "permission denied" in msg:
            print(f"NO ACCESS: {exc}".rstrip())
            print(f"  {cfg['table']} or {CATWISE_TABLE} not readable by this role.")
        elif "does not exist" in msg or "no such" in msg:
            print(f"MISSING TABLE: {exc}")
        else:
            print(f"ERROR: {exc}")
            print("  Is the Astronomy/Cambridge VPN up? Plain eduroam blocks WSDB:5432.")
        return 1

    _print_summary(cur, output_table)
    cur.close(); conn.close()
    print("=" * 74)
    print(f"Done. Query it, e.g.:  SELECT * FROM {output_table} LIMIT 5;")
    return 0


def _print_summary(cur, output_table: str) -> None:
    cur.execute(f"SELECT COUNT(*) FROM {output_table};")
    print(f"\n  {output_table}: {cur.fetchone()[0]:,} rows (QSOs with a CatWISE match within 1\")")
    cur.execute(f"SELECT COUNT(*) FROM {output_table} WHERE n_catwise_within_1arcsec > 1;")
    print(f"  QSOs with >1 CatWISE within 1\": {cur.fetchone()[0]:,}")
    cur.execute(f"SELECT COUNT(*) FROM {output_table} WHERE n_qso_per_catwise > 1;")
    print(f"  QSOs sharing a CatWISE source:  {cur.fetchone()[0]:,}")
    cur.execute(f"SELECT COUNT(*) FROM {output_table} WHERE is_mutual_blend;")
    print(f"  Mutual blends (both):           {cur.fetchone()[0]:,}")
    cur.execute(f"SELECT round(percentile_cont(0.5) WITHIN GROUP (ORDER BY w1_w2)::numeric, 3) "
                f"FROM {output_table} WHERE w1_w2 IS NOT NULL;")
    print(f"  median W1-W2 (Vega):            {cur.fetchone()[0]}")
