#!/usr/bin/env python3
"""
QSO catalog × Gaia DR3 cross-match table builder.

Used by per-catalog entry points:
    crossmatch_qso_sdss_dr16_gaia.py  →  {schema}.crossmatch_qso_sdss_dr16_gaia
    crossmatch_qso_desi_dr1_gaia.py   →  {schema}.crossmatch_qso_desi_dr1_gaia
    crossmatch_qso_catwise_gaia.py    →  {schema}.crossmatch_qso_catwise_gaia

Matching is a q3c SPATIAL cross-match (coordinate, not ID): for each source in the
input catalog, every Gaia source within MATCH_RADIUS_ARCSEC is found; DISTINCT ON
keeps the CLOSEST (then brightest) Gaia source per input source. The 14 XGBoost/UMAP
features are computed inline as g_<feature> columns; the 3 derived features use the
exact formulas from add_derived_columns.py (keep in sync — it is the source of truth).

CFG dict keys
-------------
Required:
  tag        str   Label stored in the `catalog` output column.
  table      str   Fully-qualified source table, e.g. "sdssdr16qso.main".
  id_col     str   Primary ID column in the source table.
  ra_col     str   RA column name in the source table (degrees).
  dec_col    str   Dec column name in the source table (degrees).

Optional:
  z_col      str   Redshift column name. Omit (or None) to skip source_z.
  where      str   SQL WHERE clause applied to the source table.
                   Defaults to "TRUE" (all rows).
  extra_cols list  Additional source-table columns to carry through.
                   List of (source_col, output_alias) tuples, e.g.:
                     [("w1mpro", "w1mpro"), ("w2mpro", "w2mpro")]
                   Each becomes x.{output_alias} in the final SELECT.

Output columns (one row per QSO/candidate, closest Gaia match):
  qso_catalog                 — cfg tag string
  qso_id                      — input catalog ID
  qso_ra, qso_dec             — input catalog position
  qso_z                       — redshift (only if z_col provided)
  <extra_cols aliases>        — any extra catalog-specific columns
  g_source_id, g_ra, g_dec   — matched Gaia source
  gaia_sep_arcsec             — angular separation (arcsec)
  n_gaia_within_1arcsec       — how many Gaia sources within 1" of this QSO
  all_gaia_ids_within_1arcsec — array of those Gaia source_ids (closest first)
  all_gaia_seps_within_1arcsec— array of their separations (arcsec)
  n_qso_per_gaia              — how many QSOs share this closest Gaia source
  all_qso_ids_per_gaia        — array of those QSO IDs
  is_mutual_blend             — TRUE when n_gaia_within_1arcsec > 1 AND
                                n_qso_per_gaia > 1 (bidirectional confusion)
  g_astrometric_params_solved — Gaia astrometric solution flag
  g_phot_g_mean_mag           — Gaia G magnitude
  g_phot_g_n_obs              — Gaia G-band observation count
  g_<14 features>             — XGBoost/UMAP Gaia features

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


def crossmatch_select(cfg: dict, extra_where: str | None = None) -> str:
    """Return the cross-match SELECT SQL for one catalog configuration.

    One row per input source (closest Gaia match). Blend columns track both
    directions: Gaia sources near each input source, and input sources sharing
    the same closest Gaia match.

    extra_where : optional extra predicate ANDed onto the source-table scan,
                  used by the chunked builder to restrict to one RA band
                  (e.g. "q.ra >= 150 AND q.ra < 165"). None = whole catalog.
    """
    r_deg        = MATCH_RADIUS_ARCSEC / 3600.0
    z_col        = cfg.get("z_col")
    where        = cfg.get("where", "TRUE")
    if extra_where:
        where = f"({where}) AND ({extra_where})"
    extra_cols   = cfg.get("extra_cols", [])   # list of (src_col, out_alias)

    z_xmatch   = f"q.{z_col} AS qso_z," if z_col else ""
    z_select   = "x.qso_z,"            if z_col else ""
    extra_xmatch = "\n".join(
        f"                 q.{src} AS {alias}," for src, alias in extra_cols
    )
    extra_select = "\n".join(
        f"          x.{alias}," for _, alias in extra_cols
    )

    return f"""
      WITH
      -- All QSO×Gaia pairs within the match radius
      xmatch AS (
          SELECT q.{cfg['id_col']}  AS qso_id,
                 q.{cfg['ra_col']}  AS qso_ra,
                 q.{cfg['dec_col']} AS qso_dec,
                 {z_xmatch}
                 {extra_xmatch}
                 g.source_id        AS gaia_source_id,
                 q3c_dist(q.{cfg['ra_col']}, q.{cfg['dec_col']}, g.ra, g.dec)*3600.0 AS gaia_sep_arcsec
          FROM {cfg['table']} AS q
          JOIN gaia_dr3.gaia_source AS g
              ON q3c_join(q.{cfg['ra_col']}, q.{cfg['dec_col']}, g.ra, g.dec, {r_deg})
          WHERE {where}
      ),
      -- Per-QSO Gaia blend info: count + arrays (closest first)
      gaia_agg AS (
          SELECT qso_id,
                 COUNT(*)                                              AS n_gaia_within_1arcsec,
                 array_agg(gaia_source_id ORDER BY gaia_sep_arcsec)   AS all_gaia_ids_within_1arcsec,
                 array_agg(gaia_sep_arcsec ORDER BY gaia_sep_arcsec)  AS all_gaia_seps_within_1arcsec
          FROM xmatch GROUP BY qso_id
      ),
      -- Closest Gaia per QSO (same tiebreaker as the final DISTINCT ON)
      closest AS (
          SELECT DISTINCT ON (x.qso_id) x.qso_id, x.gaia_source_id
          FROM xmatch x
          JOIN gaia_dr3.gaia_source g ON g.source_id = x.gaia_source_id
          ORDER BY x.qso_id, x.gaia_sep_arcsec, g.phot_g_mean_mag NULLS LAST, g.source_id
      ),
      -- Reverse: per closest-Gaia-source, how many QSOs claim it
      qso_agg AS (
          SELECT gaia_source_id,
                 COUNT(*)           AS n_qso_per_gaia,
                 array_agg(qso_id)  AS all_qso_ids_per_gaia
          FROM closest GROUP BY gaia_source_id
      )
      SELECT DISTINCT ON (x.qso_id)
          '{cfg['tag']}'::text        AS qso_catalog,
          x.qso_id,
          x.qso_ra,
          x.qso_dec,
          {z_select}
          {extra_select}
          g.source_id                 AS g_source_id,
          g.ra                        AS g_ra,
          g.dec                       AS g_dec,
          x.gaia_sep_arcsec,
          ga.n_gaia_within_1arcsec,
          ga.all_gaia_ids_within_1arcsec,
          ga.all_gaia_seps_within_1arcsec,
          qa.n_qso_per_gaia,
          qa.all_qso_ids_per_gaia,
          (ga.n_gaia_within_1arcsec > 1 AND qa.n_qso_per_gaia > 1) AS is_mutual_blend,
          g.astrometric_params_solved AS g_astrometric_params_solved,
          g.phot_g_mean_mag           AS g_phot_g_mean_mag,
          g.phot_g_n_obs              AS g_phot_g_n_obs,
{FEATURE_SQL}
      FROM xmatch AS x
      JOIN gaia_agg  AS ga ON ga.qso_id          = x.qso_id
      JOIN gaia_dr3.gaia_source AS g             ON g.source_id = x.gaia_source_id
      JOIN qso_agg   AS qa ON qa.gaia_source_id  = x.gaia_source_id
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
    print(f"  Match   : closest Gaia source within {MATCH_RADIUS_ARCSEC}\" per source (q3c)")
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

    _print_summary(cur, output_table)
    cur.close(); conn.close()
    print("=" * 74)
    print(f"Done. Query it, e.g.:  SELECT * FROM {output_table} LIMIT 5;")
    return 0


# Session GUCs applied to every builder connection: no statement timeout (the
# cross-match is a long one-time job) and force parallelism even on small
# per-band scans (min_parallel_*_size = 0) so chunks parallelise too.
_SESSION_SETS = (
    "SET statement_timeout = 0",
    "SET max_parallel_workers_per_gather = 4",
    "SET parallel_setup_cost = 0",
    "SET parallel_tuple_cost = 0",
    "SET min_parallel_table_scan_size = 0",
    "SET min_parallel_index_scan_size = 0",
)


def _print_summary(cur, output_table: str) -> None:
    """Print the standard row-count / blend summary for a finished output table."""
    cur.execute(f"SELECT COUNT(*) FROM {output_table};")
    print(f"\n  {output_table}: {cur.fetchone()[0]:,} rows")
    cur.execute(f"""SELECT g_astrometric_params_solved, COUNT(*)
                    FROM {output_table} GROUP BY 1 ORDER BY 1;""")
    print("  by astrometric_params_solved:")
    for aps, n in cur.fetchall():
        lbl = {3: "2-param", 31: "5-param", 95: "6-param"}.get(aps, str(aps))
        print(f"    {str(aps):<4} ({lbl:<7}) {n:>10,}")
    cur.execute(f"SELECT COUNT(*) FROM {output_table} WHERE n_gaia_within_1arcsec > 1;")
    print(f"  QSOs with >1 Gaia within 1\":  {cur.fetchone()[0]:,}")
    cur.execute(f"SELECT COUNT(*) FROM {output_table} WHERE n_qso_per_gaia > 1;")
    print(f"  QSOs sharing a Gaia source:   {cur.fetchone()[0]:,}")
    cur.execute(f"SELECT COUNT(*) FROM {output_table} WHERE is_mutual_blend;")
    print(f"  Mutual blends (both):          {cur.fetchone()[0]:,}")


def build_table_chunked(username: str, cfg: dict, table_name: str,
                        n_bands: int = 24, resume: bool = False,
                        ra_col: str | None = None) -> int:
    """Build {username}.{table_name} in N RA bands, each its own committed
    transaction — robust to VPN drops and resumable.

    Instead of one giant CREATE TABLE AS (which restarts from scratch if the
    connection dies), this splits the sky into `n_bands` equal RA slices and
    INSERTs each independently. A companion `{table_name}_progress` table records
    which bands are done; a band's data INSERT and its progress row commit in the
    SAME transaction, so every band is all-or-nothing. On `resume=True`, bands
    already in the progress table are skipped, so you can restart after any
    interruption and only the unfinished bands re-run.

    n_bands : number of equal RA slices (default 24 → 15° each).
    resume  : True keeps existing output + progress and skips done bands;
              False (default) drops both and starts clean.
    ra_col  : source RA column to band on; defaults to cfg['ra_col'].

    Caveat: the QSO-side blend columns (n_qso_per_gaia, all_qso_ids_per_gaia)
    are computed within each band, so two QSOs within 1" that straddle a band
    boundary (a ~1"/360° chance at each of the N meridians) are not counted as
    sharing a Gaia source. The closest-match columns are unaffected (Gaia is
    never banded). Negligible for diagnostics; use build_table for exactness.
    """
    import psycopg2
    output_table = f"{username}.{table_name}"
    progress_table = f"{username}.{table_name}_progress"
    ra = ra_col or cfg["ra_col"]
    band_w = 360.0 / n_bands

    print("=" * 74)
    print(f"Building {output_table}  (chunked, {n_bands} RA bands of {band_w:g}°)")
    print(f"  Catalog : {cfg['tag']}  ({cfg['table']})")
    print(f"  Match   : closest Gaia source within {MATCH_RADIUS_ARCSEC}\" per source (q3c)")
    print(f"  Mode    : {'RESUME (skip done bands)' if resume else 'fresh (drop & rebuild)'}")
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

        # Setup: empty output table with correct column types (WHERE-false band)
        # + progress tracker. IF NOT EXISTS so resume keeps prior work.
        if not resume:
            cur.execute(f"DROP TABLE IF EXISTS {output_table} CASCADE;")
            cur.execute(f"DROP TABLE IF EXISTS {progress_table} CASCADE;")
        empty_select = crossmatch_select(cfg, extra_where="1 = 0")
        cur.execute(f"CREATE TABLE IF NOT EXISTS {output_table} AS\n{empty_select};")
        cur.execute(f"""CREATE TABLE IF NOT EXISTS {progress_table} (
                            band     integer PRIMARY KEY,
                            ra_lo    double precision,
                            ra_hi    double precision,
                            n_rows   bigint,
                            done_at  timestamptz DEFAULT now());""")
        conn.commit()

        cur.execute(f"SELECT band FROM {progress_table};")
        done = {r[0] for r in cur.fetchall()}
        if done:
            print(f"  {len(done)} band(s) already done: {sorted(done)}")

        for i in range(n_bands):
            if i in done:
                continue
            lo, hi = i * band_w, (i + 1) * band_w
            band_pred = f"q.{ra} >= {lo} AND q.{ra} < {hi}"
            print(f"  band {i:2d}/{n_bands}  ra [{lo:6.1f}, {hi:6.1f}) ...", end="", flush=True)
            sel = crossmatch_select(cfg, extra_where=band_pred)
            # Data insert + progress marker in ONE transaction → all-or-nothing.
            cur.execute(f"INSERT INTO {output_table}\n{sel};")
            n = cur.rowcount
            cur.execute(
                f"INSERT INTO {progress_table} (band, ra_lo, ra_hi, n_rows) "
                f"VALUES (%s, %s, %s, %s);", (i, lo, hi, n))
            conn.commit()
            print(f" {n:,} rows committed")

        print("\nAll bands done — building index...")
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {table_name}_g_source_id_idx "
            f"ON {output_table} (g_source_id);")
        conn.commit()
    except Exception as exc:
        msg = str(exc).lower()
        if "permission denied" in msg:
            print(f"\nNO ACCESS: {exc}".rstrip())
        elif "does not exist" in msg or "no such" in msg:
            print(f"\nMISSING TABLE: {exc}")
        else:
            print(f"\nERROR: {exc}")
            print("  VPN may have dropped. Rerun with resume=True to finish the "
                  "remaining bands (committed bands are kept).")
        return 1

    _print_summary(cur, output_table)
    cur.close(); conn.close()
    print("=" * 74)
    print(f"Done. Query it, e.g.:  SELECT * FROM {output_table} LIMIT 5;")
    return 0
