#!/usr/bin/env python3
"""
Shared NSA-galaxy × Gaia DR3 cross-match table builder (galaxy-nucleation project).

Used by the per-catalog entry points:
    crossmatch_nsa_v1_gaia.py  →  {schema}.crossmatch_nsa_v1_gaia   (primary)
    crossmatch_nsa_v0_gaia.py  →  {schema}.crossmatch_nsa_v0_gaia   (cross-check)

KEY DIFFERENCES FROM THE QSO CROSS-MATCH (qso_gaia_crossmatch_builder.py)
------------------------------------------------------------------------
1. LEFT JOIN — we keep EVERY galaxy (matched or not), because the science output
   is a *nucleation fraction* = (galaxies with a central Gaia detection) / (all
   galaxies). has_gaia flags detection; the denominator must be preserved.
2. "Central" match — galaxies are extended, so a Gaia source at the centre is a
   nucleus candidate (nuclear star cluster / AGN) whereas one off-centre is a
   foreground star / HII knot. We match within a SMALL radius of the photometric
   centre (default 1"; closest source kept) and store n_gaia_within so blends /
   crowding can be filtered.
3. Galaxy properties travel with each row (log stellar mass, Sérsic n, effective
   radius, axis ratio, concentration, distance) — these are the axes the
   nucleation fraction is measured against.
4. The matched Gaia source's astrometric_params_solved is diagnostic: a true
   nucleus tends to be 2-param (like QSOs → UMAP C0/C1); a foreground star is
   5/6-param (→ C3). So the QSO/UMAP machinery becomes a nucleus-vs-interloper
   filter downstream.

The 14 Gaia features are the SAME as the QSO case — imported from
qso_gaia_crossmatch_builder.FEATURE_SQL (single source of truth). They are NULL
for unmatched galaxies.

Requires WSDB — VPN must be up (plain eduroam blocks port 5432). PGUSER selects
the schema/role; auth via ~/.pgpass.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from qso_gaia_crossmatch_builder import FEATURE_SQL, WSDB_HOST, WSDB_DB  # noqa: E402

MATCH_RADIUS_ARCSEC = 1.0


def crossmatch_select(cfg: dict) -> str:
    """LEFT-JOIN cross-match SELECT: every galaxy, its properties, and the closest
    central Gaia source within MATCH_RADIUS_ARCSEC (NULL Gaia cols if none).

    cfg keys:
      table, id_col, ra_col, dec_col : NSA table + identity/coordinate columns
      props : {output_alias: sql_expr}  galaxy-property expressions (q.<col> ...),
              normalized to a version-agnostic set of aliases so downstream code
              is identical for v0 and v1.
      where : optional extra row filter (without leading AND)
    """
    r_deg = MATCH_RADIUS_ARCSEC / 3600.0
    prop_sel = ",\n                 ".join(f"{expr} AS {alias}" for alias, expr in cfg["props"].items())
    extra_where = f" AND ({cfg['where']})" if cfg.get("where") else ""
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
      xmatch AS MATERIALIZED (
          SELECT gal.nsaid, g.source_id,
                 q3c_dist(gal.gal_ra, gal.gal_dec, g.ra, g.dec)*3600.0 AS gaia_sep_arcsec
          FROM gal
          JOIN gaia_dr3.gaia_source AS g
              ON q3c_join(gal.gal_ra, gal.gal_dec, g.ra, g.dec, {r_deg})
      ),
      counts AS MATERIALIZED (
          SELECT nsaid, COUNT(*) AS n_gaia_within_1arcsec FROM xmatch GROUP BY nsaid
      ),
      closest AS MATERIALIZED (
          SELECT DISTINCT ON (nsaid) nsaid, source_id, gaia_sep_arcsec
          FROM xmatch ORDER BY nsaid, gaia_sep_arcsec
      )
      SELECT
          '{cfg['tag']}'::text                  AS nsa_catalog,
          gal.*,
          (cl.source_id IS NOT NULL)            AS has_gaia,
          COALESCE(c.n_gaia_within_1arcsec, 0)  AS n_gaia_within_1arcsec,
          cl.gaia_sep_arcsec,
          g.source_id                           AS g_source_id,
          g.astrometric_params_solved           AS g_astrometric_params_solved,
          g.phot_g_mean_mag                     AS g_phot_g_mean_mag,
          g.phot_g_n_obs                        AS g_phot_g_n_obs,
{FEATURE_SQL}
      FROM gal
      LEFT JOIN closest AS cl ON cl.nsaid = gal.nsaid
      LEFT JOIN counts  AS c  ON c.nsaid  = gal.nsaid
      LEFT JOIN gaia_dr3.gaia_source AS g ON g.source_id = cl.source_id
    """


def build_table(username: str, cfg: dict, table_name: str) -> int:
    """DROP + CREATE {username}.{table_name}, index it, print a nucleation summary.
    Returns 0 on success, 1 on failure (graceful on permission-denied)."""
    import psycopg2
    output_table = f"{username}.{table_name}"
    print("=" * 78)
    print(f"Building {output_table}")
    print(f"  Catalog : {cfg['tag']}  ({cfg['table']})")
    print(f"  Match   : closest Gaia source within {MATCH_RADIUS_ARCSEC}\" of galaxy centre (q3c)")
    print(f"  Join    : LEFT (all galaxies kept — has_gaia flags central detection)")
    print("=" * 78)
    try:
        conn = psycopg2.connect(
            host=WSDB_HOST, database=WSDB_DB, user=username,
            connect_timeout=30, keepalives=1, keepalives_idle=30,
            keepalives_interval=10, keepalives_count=5,
        )
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 0")
        print("Running cross-match + CREATE TABLE (slow, one-time step)...")
        cur.execute(f"DROP TABLE IF EXISTS {output_table} CASCADE;")
        cur.execute(f"CREATE TABLE {output_table} AS\n{crossmatch_select(cfg)};")
        conn.commit()
        cur.execute(f"CREATE INDEX ON {output_table} (nsaid);")
        cur.execute(f"CREATE INDEX ON {output_table} (has_gaia);")
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

    cur.execute(f"SELECT COUNT(*), COUNT(*) FILTER (WHERE has_gaia) FROM {output_table};")
    n_tot, n_det = cur.fetchone()
    frac = 100 * n_det / n_tot if n_tot else 0.0
    print(f"\n  {output_table}: {n_tot:,} galaxies, {n_det:,} with a central Gaia detection")
    print(f"  overall nucleation-detection fraction: {frac:.2f}%")
    cur.execute(f"""SELECT g_astrometric_params_solved, COUNT(*)
                    FROM {output_table} WHERE has_gaia GROUP BY 1 ORDER BY 1;""")
    print("  detected nuclei by astrometric_params_solved (2-param ⇒ likely true nucleus):")
    for aps, n in cur.fetchall():
        lbl = {3: "2-param", 31: "5-param", 95: "6-param"}.get(aps, str(aps))
        print(f"    {str(aps):<4} ({lbl:<7}) {n:>9,}")
    cur.execute(f"SELECT COUNT(*) FROM {output_table} WHERE n_gaia_within_1arcsec > 1;")
    print(f"  galaxies with >1 Gaia source within 1\" (crowding/blend): {cur.fetchone()[0]:,}")
    cur.close(); conn.close()
    print("=" * 78)
    print(f"Done. Query it, e.g.:  SELECT * FROM {output_table} LIMIT 5;")
    return 0
