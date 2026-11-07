#!/usr/bin/env python3
"""
Shared NSA-galaxy × Euclid cross-match table builder (galaxy-nucleation project).

Used by the per-catalog entry points:
    crossmatch_nsa_v1_euclid.py  →  {schema}.crossmatch_nsa_v1_euclid   (primary)
    crossmatch_nsa_v0_euclid.py  →  {schema}.crossmatch_nsa_v0_euclid   (cross-check)

This is the Euclid-target twin of builder_nsa_gaia_crossmatch.py: same NSA-galaxy
semantics, but matched against the combined Euclid footprint instead of Gaia.

SEMANTICS (inherited from the NSA-Gaia builder, NOT the QSO-Euclid one)
-----------------------------------------------------------------------
1. LEFT JOIN — every galaxy is kept (matched or not). The science output is a
   *footprint/detection fraction* = (galaxies with a central Euclid source) / (all
   galaxies); has_euclid flags detection, and the denominator must be preserved.
   Because Euclid only covers a few fields, most NSA galaxies are unmatched
   (has_euclid = FALSE, all Euclid cols NULL).
2. "Central" match — closest Euclid source within MATCH_RADIUS_ARCSEC of the galaxy
   photometric centre (nucleus / galaxy core candidate).
3. Galaxy properties travel with each row (log mass, Sérsic n, Re, axis ratio,
   concentration, distance) — the axes the fraction is measured against.

EUCLID MACHINERY (inherited from builder_qso_euclid_crossmatch.py)
--------------------------------------------------------------
The Euclid target is euclid_q1.mer_final_cat UNION ALL euclid_ero.merge_cat. Each
UNION arm q3c-joins the galaxies DIRECTLY to a base Euclid table so the join uses
that table's q3c index (joining a pre-UNIONed CTE instead defeats the index — the
bug that made the first Euclid build scan 42M rows per band). The standardised
e_* columns (_ECOLS) and per-branch geometry (_EBRANCHES) are imported from
builder_qso_euclid_crossmatch so the two stay in lock-step. euclid_source ('Q1'/'ERO')
tags each match's origin.

Two-way blend columns (same idea as the Gaia builder):
  n_euclid_within_1arcsec, all_euclid_ids_within_1arcsec, all_euclid_seps_within_1arcsec
      — Euclid sources near this galaxy's centre
  n_nsa_per_euclid, all_nsa_ids_per_euclid
      — galaxies claiming the same closest Euclid source
  is_mutual_blend — both > 1
All NULL/0 for unmatched galaxies, preserving the denominator.

Requires WSDB — VPN must be up (plain eduroam blocks port 5432). PGUSER selects
the schema/role; auth via ~/.pgpass.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from builder_qso_euclid_crossmatch import (  # noqa: E402
    _ECOLS, _EBRANCHES, MATCH_RADIUS_ARCSEC, WSDB_HOST, WSDB_DB, _SESSION_SETS,
)


def crossmatch_select(cfg: dict) -> str:
    """LEFT-JOIN cross-match SELECT: every galaxy, its properties, and the closest
    central Euclid source within MATCH_RADIUS_ARCSEC (NULL Euclid cols if none).

    cfg keys:
      tag, table, id_col, ra_col, dec_col : NSA table + identity/coordinate columns
      props : {output_alias: sql_expr}  galaxy-property expressions (q.<col> ...)
      where : optional extra row filter (without leading AND)
    """
    r_deg = MATCH_RADIUS_ARCSEC / 3600.0
    prop_sel = ",\n                 ".join(
        f"{expr} AS {alias}" for alias, expr in cfg["props"].items())
    extra_where = f" AND ({cfg['where']})" if cfg.get("where") else ""

    def _branch(source, etable, eid, era, edec, is_q1):
        # One UNION arm: q3c-join galaxies straight to a base Euclid table so the
        # join uses that table's q3c index; carry the standardised e_* columns.
        ecols = ",\n".join(
            f"                 {(q1 if is_q1 else ero)} AS {alias}"
            for alias, q1, ero in _ECOLS
        )
        return f"""
          SELECT gal.nsaid,
                 '{source}'::text   AS euclid_source,
                 e.{eid}            AS euclid_id,
                 e.{era}            AS euclid_ra,
                 e.{edec}           AS euclid_dec,
                 q3c_dist(gal.gal_ra, gal.gal_dec, e.{era}, e.{edec}) * 3600.0 AS euclid_sep_arcsec,
{ecols}
          FROM gal
          JOIN {etable} AS e
              ON q3c_join(gal.gal_ra, gal.gal_dec, e.{era}, e.{edec}, {r_deg})"""

    xmatch_union  = "\n          UNION ALL\n".join(_branch(*b) for b in _EBRANCHES)
    closest_ecols = ",\n".join(f"                 x.{alias}" for alias, _, _ in _ECOLS)
    final_ecols   = ",\n".join(f"          cl.{alias}" for alias, _, _ in _ECOLS)

    return f"""
      WITH gal AS MATERIALIZED (
          SELECT q.{cfg['id_col']} AS nsaid,
                 q.iauname          AS iauname,
                 q.{cfg['ra_col']}  AS gal_ra,
                 q.{cfg['dec_col']} AS gal_dec,
                 {prop_sel}
          FROM {cfg['table']} AS q
          WHERE q.{cfg['ra_col']} IS NOT NULL AND q.{cfg['dec_col']} IS NOT NULL{extra_where}
      ),
      -- All galaxy×Euclid pairs within the match radius (matched pairs only; the
      -- final LEFT JOIN re-adds unmatched galaxies). Each arm hits a q3c index.
      xmatch AS MATERIALIZED (
{xmatch_union}
      ),
      -- Per-galaxy Euclid blend info: count + arrays (closest first)
      euclid_agg AS MATERIALIZED (
          SELECT nsaid,
                 COUNT(*)                                               AS n_euclid_within_1arcsec,
                 array_agg(euclid_id ORDER BY euclid_sep_arcsec)       AS all_euclid_ids_within_1arcsec,
                 array_agg(euclid_sep_arcsec ORDER BY euclid_sep_arcsec) AS all_euclid_seps_within_1arcsec
          FROM xmatch GROUP BY nsaid
      ),
      -- Closest central Euclid per galaxy (tiebreak: flux_vis descending, then id).
      -- Carries the e_* columns so the final LEFT JOIN needs no re-join.
      closest AS MATERIALIZED (
          SELECT DISTINCT ON (x.nsaid)
                 x.nsaid, x.euclid_id, x.euclid_source, x.euclid_ra, x.euclid_dec,
                 x.euclid_sep_arcsec,
{closest_ecols}
          FROM xmatch x
          ORDER BY x.nsaid, x.euclid_sep_arcsec, x.e_flux_vis DESC NULLS LAST, x.euclid_id
      ),
      -- Reverse: per closest-Euclid-source, how many galaxies claim it as centre.
      -- euclid_id is globally unique within each source tag (Q1 object_id and
      -- ERO v_objid are both globally unique within their catalogs).
      nsa_agg AS MATERIALIZED (
          SELECT euclid_id, euclid_source,
                 COUNT(*)          AS n_nsa_per_euclid,
                 array_agg(nsaid)  AS all_nsa_ids_per_euclid
          FROM closest GROUP BY euclid_id, euclid_source
      )
      SELECT
          '{cfg['tag']}'::text                    AS nsa_catalog,
          gal.*,
          (cl.euclid_id IS NOT NULL)              AS has_euclid,
          COALESCE(ea.n_euclid_within_1arcsec, 0) AS n_euclid_within_1arcsec,
          ea.all_euclid_ids_within_1arcsec,
          ea.all_euclid_seps_within_1arcsec,
          cl.euclid_sep_arcsec,
          na.n_nsa_per_euclid,
          na.all_nsa_ids_per_euclid,
          (COALESCE(ea.n_euclid_within_1arcsec, 0) > 1
           AND COALESCE(na.n_nsa_per_euclid, 0) > 1) AS is_mutual_blend,
          cl.euclid_id                            AS e_id,
          cl.euclid_ra                            AS e_ra,
          cl.euclid_dec                           AS e_dec,
          cl.euclid_source,
{final_ecols}
      FROM gal
      LEFT JOIN euclid_agg AS ea ON ea.nsaid         = gal.nsaid
      LEFT JOIN closest    AS cl ON cl.nsaid         = gal.nsaid
      LEFT JOIN nsa_agg    AS na ON na.euclid_id     = cl.euclid_id
                                AND na.euclid_source  = cl.euclid_source
    """


def build_table(username: str, cfg: dict, table_name: str) -> int:
    """DROP + CREATE {username}.{table_name}, index it, print a footprint summary.
    Returns 0 on success, 1 on failure (graceful on permission-denied)."""
    import psycopg2
    output_table = f"{username}.{table_name}"
    print("=" * 78)
    print(f"Building {output_table}")
    print(f"  Catalog : {cfg['tag']}  ({cfg['table']})")
    print(f"  Match   : closest Euclid source within {MATCH_RADIUS_ARCSEC}\" of galaxy centre (q3c)")
    print(f"  Target  : euclid_q1.mer_final_cat UNION ALL euclid_ero.merge_cat")
    print(f"  Join    : LEFT (all galaxies kept — has_euclid flags central detection)")
    print("=" * 78)
    try:
        conn = psycopg2.connect(
            host=WSDB_HOST, database=WSDB_DB, user=username,
            connect_timeout=30, keepalives=1, keepalives_idle=30,
            keepalives_interval=10, keepalives_count=5,
        )
        cur = conn.cursor()
        for stmt in _SESSION_SETS:
            cur.execute(stmt)
        print("Running cross-match + CREATE TABLE (slow, one-time step)...")
        cur.execute(f"DROP TABLE IF EXISTS {output_table} CASCADE;")
        cur.execute(f"CREATE TABLE {output_table} AS\n{crossmatch_select(cfg)};")
        conn.commit()
        cur.execute(f"CREATE INDEX ON {output_table} (nsaid);")
        cur.execute(f"CREATE INDEX ON {output_table} (has_euclid);")
        conn.commit()
    except Exception as exc:
        msg = str(exc).lower()
        if "permission denied" in msg:
            print(f"NO ACCESS: {exc.__class__.__name__}: {exc}".rstrip())
            print(f"  {cfg['table']} is not readable by this role — no table built.")
            print(f"  Ask for  GRANT SELECT ON {cfg['table']} TO {username};")
        elif "does not exist" in msg or "no such" in msg or "column" in msg:
            print(f"SCHEMA MISMATCH: {exc}")
            print("  Check the NSA column names in this script's config against the table.")
        else:
            print(f"ERROR: {exc}")
            print("  Is the Astronomy/Cambridge VPN up? Plain eduroam blocks WSDB:5432.")
        return 1

    cur.execute(f"SELECT COUNT(*), COUNT(*) FILTER (WHERE has_euclid) FROM {output_table};")
    n_tot, n_det = cur.fetchone()
    frac = 100 * n_det / n_tot if n_tot else 0.0
    print(f"\n  {output_table}: {n_tot:,} galaxies, {n_det:,} with a central Euclid source")
    print(f"  fraction in Euclid footprint (matched): {frac:.2f}%")
    cur.execute(f"""SELECT euclid_source, COUNT(*)
                    FROM {output_table} WHERE has_euclid GROUP BY 1 ORDER BY 1;""")
    print("  matched by euclid_source:")
    for src, n in cur.fetchall():
        print(f"    {src:<5} {n:>9,}")
    cur.execute(f"SELECT COUNT(*) FROM {output_table} WHERE n_euclid_within_1arcsec > 1;")
    print(f"  galaxies with >1 Euclid source within 1\" (crowding/blend): {cur.fetchone()[0]:,}")
    cur.execute(f"SELECT COUNT(*) FROM {output_table} WHERE n_nsa_per_euclid > 1;")
    print(f"  galaxies sharing a closest Euclid source:                  {cur.fetchone()[0]:,}")
    cur.execute(f"SELECT COUNT(*) FROM {output_table} WHERE is_mutual_blend;")
    print(f"  mutual blends (both directions):                           {cur.fetchone()[0]:,}")
    cur.close(); conn.close()
    print("=" * 78)
    print(f"Done. Query it, e.g.:  SELECT * FROM {output_table} LIMIT 5;")
    return 0
