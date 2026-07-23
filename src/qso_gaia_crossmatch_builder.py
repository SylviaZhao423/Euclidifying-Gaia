#!/usr/bin/env python3
"""
Shared QSO × Gaia DR3 cross-match table builder.

Used by the per-catalog entry points:
    crossmatch_qso_sdss_dr16_gaia.py  →  {schema}.crossmatch_qso_sdss_dr16_gaia
    crossmatch_qso_desi_dr1_gaia.py   →  {schema}.crossmatch_qso_desi_dr1_gaia

Matching is a q3c SPATIAL cross-match (coordinate, not ID): for each QSO, every
Gaia source within MATCH_RADIUS_ARCSEC is found; n_gaia_within_1arcsec counts them
(blend flag), and DISTINCT ON keeps the CLOSEST (then brightest) Gaia source per
QSO. The 14 XGBoost/UMAP features are computed inline and stored as g_<feature>
columns; the 3 derived features use the exact formulas from add_derived_columns.py
(keep in sync with that file — it is the source of truth for the derivations).

Requires WSDB — VPN must be up (plain eduroam blocks port 5432). PGUSER selects
the schema/role; auth via ~/.pgpass.
"""

from __future__ import annotations

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
MATCH_RADIUS_ARCSEC = 1.0

# 14 features as g_<name>. Derived expressions verbatim from add_derived_columns.py.
FEATURE_SQL = """
        g.parallax_over_error            AS g_parallax_over_error,
        g.astrometric_gof_al             AS g_astrometric_gof_al,
        g.astrometric_chi2_al            AS g_astrometric_chi2_al,
        g.astrometric_excess_noise_sig   AS g_astrometric_excess_noise_sig,
        g.ipd_gof_harmonic_amplitude     AS g_ipd_gof_harmonic_amplitude,
        g.ruwe                           AS g_ruwe,
        g.phot_g_mean_flux_over_error    AS g_phot_g_mean_flux_over_error,
        g.phot_bp_mean_flux_over_error   AS g_phot_bp_mean_flux_over_error,
        g.phot_rp_mean_flux_over_error   AS g_phot_rp_mean_flux_over_error,
        g.bp_g                           AS g_bp_g,
        g.g_rp                           AS g_g_rp,
        (g.phot_bp_rp_excess_factor - (
            CASE
                WHEN g.bp_rp < 0.5 THEN 1.154360 + 0.033772*g.bp_rp + 0.032277*g.bp_rp*g.bp_rp
                WHEN g.bp_rp < 4.0 THEN 1.162004 + 0.011464*g.bp_rp + 0.049255*g.bp_rp*g.bp_rp
                                       - 0.005879*g.bp_rp*g.bp_rp*g.bp_rp
                ELSE 1.057572 + 0.140537*g.bp_rp
            END
        ))                               AS g_corrected_bp_rp_excess,
        SQRT(
            (g.pmra*g.pmra*g.pmdec_error*g.pmdec_error
             - 2.0*g.pmra_pmdec_corr*g.pmra_error*g.pmdec_error*g.pmra*g.pmdec
             + g.pmdec*g.pmdec*g.pmra_error*g.pmra_error)
            / NULLIF(g.pmra_error*g.pmra_error*g.pmdec_error*g.pmdec_error
                     * (1.0 - g.pmra_pmdec_corr*g.pmra_pmdec_corr), 0)
        )                                AS g_pm_significance,
        SQRT(g.ra_error*g.ra_error*g.dec_error*g.dec_error
             * (1.0 - g.ra_dec_corr*g.ra_dec_corr))   AS g_position_error
"""


def crossmatch_select(cfg: dict) -> str:
    """One catalog's DISTINCT-ON q3c cross-match SELECT (closest Gaia per QSO)."""
    r_deg = MATCH_RADIUS_ARCSEC / 3600.0
    return f"""
      WITH xmatch AS MATERIALIZED (
          SELECT q.{cfg['id_col']} AS qso_id,
                 q.{cfg['ra_col']} AS qso_ra, q.{cfg['dec_col']} AS qso_dec,
                 q.{cfg['z_col']}  AS qso_z,
                 g.source_id       AS gaia_source_id,
                 q3c_dist(q.{cfg['ra_col']}, q.{cfg['dec_col']}, g.ra, g.dec)*3600.0 AS gaia_sep_arcsec
          FROM {cfg['table']} AS q
          JOIN gaia_dr3.gaia_source AS g
              ON q3c_join(q.{cfg['ra_col']}, q.{cfg['dec_col']}, g.ra, g.dec, {r_deg})
          WHERE {cfg['where']}
      ),
      counts AS MATERIALIZED (
          SELECT qso_id, COUNT(*) AS n_gaia_within_1arcsec FROM xmatch GROUP BY qso_id
      )
      SELECT DISTINCT ON (x.qso_id)
          '{cfg['tag']}'::text        AS qso_catalog,
          x.qso_id                    AS qso_id,
          x.qso_ra, x.qso_dec, x.qso_z,
          g.source_id                 AS g_source_id,
          g.ra                        AS g_ra,
          g.dec                       AS g_dec,
          x.gaia_sep_arcsec,
          c.n_gaia_within_1arcsec,
          g.astrometric_params_solved AS g_astrometric_params_solved,
          g.phot_g_mean_mag           AS g_phot_g_mean_mag,
          g.phot_g_n_obs              AS g_phot_g_n_obs,
{FEATURE_SQL}
      FROM xmatch AS x
      JOIN counts AS c ON c.qso_id = x.qso_id
      JOIN gaia_dr3.gaia_source AS g ON g.source_id = x.gaia_source_id
      ORDER BY x.qso_id, x.gaia_sep_arcsec, g.phot_g_mean_mag NULLS LAST, g.source_id
    """


def build_table(username: str, cfg: dict, table_name: str) -> int:
    """DROP + CREATE {username}.{table_name} from the cross-match, add indexes,
    and print a summary. Returns 0 on success, 1 on failure."""
    import psycopg2
    output_table = f"{username}.{table_name}"
    print("=" * 74)
    print(f"Building {output_table}")
    print(f"  Catalog : {cfg['tag']}  ({cfg['table']})")
    print(f"  Match   : closest Gaia source within {MATCH_RADIUS_ARCSEC}\" per QSO (q3c)")
    print("=" * 74)
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
        cur.execute(f"CREATE INDEX ON {output_table} (g_source_id);")
        conn.commit()
    except Exception as exc:
        msg = str(exc).lower()
        if "permission denied" in msg:
            print(f"NO ACCESS: {exc.__class__.__name__}: {exc}".rstrip())
            print(f"  {cfg['table']} is not readable by this role — no table built.")
            print("  Ask for  GRANT SELECT  on it, or switch CFG to a readable catalog.")
        elif "does not exist" in msg or "no such" in msg:
            print(f"MISSING TABLE: {exc}")
            print(f"  {cfg['table']} not found — check the schema/table name.")
        else:
            print(f"ERROR: {exc}")
            print("  Is the Astronomy/Cambridge VPN up? Plain eduroam blocks WSDB:5432.")
        return 1

    cur.execute(f"SELECT COUNT(*) FROM {output_table};")
    print(f"\n  {output_table}: {cur.fetchone()[0]:,} rows")
    cur.execute(f"""SELECT g_astrometric_params_solved, COUNT(*)
                    FROM {output_table} GROUP BY 1 ORDER BY 1;""")
    print("  by astrometric_params_solved:")
    for aps, n in cur.fetchall():
        lbl = {3: "2-param", 31: "5-param", 95: "6-param"}.get(aps, str(aps))
        print(f"    {str(aps):<4} ({lbl:<7}) {n:>10,}")
    cur.execute(f"SELECT COUNT(*) FROM {output_table} WHERE n_gaia_within_1arcsec > 1;")
    print(f"  multi-match (blend) rows: {cur.fetchone()[0]:,}")
    cur.close(); conn.close()
    print("=" * 74)
    print(f"Done. Query it, e.g.:  SELECT * FROM {output_table} LIMIT 5;")
    return 0
