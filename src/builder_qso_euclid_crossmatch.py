#!/usr/bin/env python3
"""
Generic catalog × Euclid cross-match table builder.

Matches an input source catalog against the combined Euclid footprint
(euclid_q1.mer_final_cat UNION ALL euclid_ero.merge_cat) within 1" using q3c.

Used by per-catalog entry points:
    crossmatch_qso_catwise_euclid.py  →  {schema}.crossmatch_qso_catwise_euclid

CFG dict keys — same interface as gaia_crossmatch_builder.py
-----------------------------------------------------------------
Required:
  tag        str   Label stored in the `qso_catalog` output column.
  table      str   Fully-qualified source table, e.g. "catwise_202003.main".
  id_col     str   Primary ID column in the source table.
  ra_col     str   RA column name in the source table (degrees).
  dec_col    str   Dec column name in the source table (degrees).

Optional:
  z_col      str   Redshift column name. Omit (or None) to skip qso_z.
  where      str   SQL WHERE clause applied to the source table.
                   Defaults to "TRUE" (all rows).
  extra_cols list  Additional source-table columns to carry through.
                   List of (source_col, output_alias) tuples.

Output columns (one row per source, closest Euclid match):
  qso_catalog, qso_id, qso_ra, qso_dec, [qso_z], [extra_cols...]
  e_id                  — Euclid object_id (Q1) or v_objid (ERO)
  e_ra, e_dec           — Euclid position
  euclid_source         — 'Q1' or 'ERO'
  euclid_sep_arcsec     — angular separation (arcsec)
  n_euclid_within_1arcsec, all_euclid_ids_within_1arcsec, all_euclid_seps_within_1arcsec
  n_qso_per_euclid, all_qso_ids_per_euclid
  is_mutual_blend       — n_euclid_within_1arcsec > 1 AND n_qso_per_euclid > 1
  e_vis_det             — VIS detection flag (Q1 only)
  e_flux_vis            — Q1: flux_vis_2fwhm_aper; ERO: v_flux_auto
  e_fluxerr_vis
  e_flux_vis_psf, e_fluxerr_vis_psf — PSF (point-source) flux
  e_flux_y, e_fluxerr_y — Y-band (Q1: 2fwhm aper; ERO: y_flux_auto)
  e_flux_j, e_fluxerr_j — J-band
  e_flux_h, e_fluxerr_h — H-band
  e_flag_vis, e_flag_y, e_flag_j, e_flag_h — Q1 quality flags (NULL for ERO)
  e_point_like_prob     — Q1: point_like_prob; ERO: v_class_star ∈ [0,1]
  e_point_like_flag     — Q1 only
  e_extended_prob       — Q1 only
  e_extended_flag       — Q1 only
  e_spurious_flag       — Q1 only
  e_det_quality_flag    — Q1 only
  e_deblended_flag      — Q1 only
  e_blended_prob        — Q1 only
  e_fwhm                — Q1: fwhm; ERO: v_fwhm_image
  e_mumax_minus_mag     — Q1 only (compactness proxy)
  e_gal_ebv             — Q1 only (E(B-V) for dust correction)
  e_variable_flag       — Q1 only
  e_gaia_id             — Q1 only (Euclid's own Gaia cross-match)
  e_gaia_match_quality  — Q1 only

Requires WSDB — VPN must be up (plain eduroam blocks port 5432). Auth via ~/.pgpass.
"""

from __future__ import annotations

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
MATCH_RADIUS_ARCSEC = 1.0

# Standardised e_* columns, per Euclid table. Each entry is (alias, Q1 expr, ERO
# expr); base-table columns are prefixed `e.` (the joined Euclid alias). Q1 uses
# 2-FWHM fixed apertures; ERO uses AUTO (total) flux — closest equivalent. Q1-only
# columns are NULL in ERO rows. UNION ALL requires the two exprs to share a type,
# hence the explicit ::casts on the NULLs. These become x.<alias> in the output.
_ECOLS = [
    ("e_vis_det",           "e.vis_det",                                 "NULL::smallint"),
    ("e_flux_vis",          "e.flux_vis_2fwhm_aper::double precision",    "e.v_flux_auto"),
    ("e_fluxerr_vis",       "e.fluxerr_vis_2fwhm_aper::double precision", "e.v_fluxerr_auto"),
    ("e_flux_vis_psf",      "e.flux_vis_psf::double precision",           "e.v_flux_psf"),
    ("e_fluxerr_vis_psf",   "e.fluxerr_vis_psf::double precision",        "e.v_fluxerr_psf"),
    ("e_flux_y",            "e.flux_y_2fwhm_aper::double precision",      "e.y_flux_auto"),
    ("e_fluxerr_y",         "e.fluxerr_y_2fwhm_aper::double precision",   "e.y_fluxerr_auto"),
    ("e_flux_j",            "e.flux_j_2fwhm_aper::double precision",      "e.j_flux_auto"),
    ("e_fluxerr_j",         "e.fluxerr_j_2fwhm_aper::double precision",   "e.j_fluxerr_auto"),
    ("e_flux_h",            "e.flux_h_2fwhm_aper::double precision",      "e.h_flux_auto"),
    ("e_fluxerr_h",         "e.fluxerr_h_2fwhm_aper::double precision",   "e.h_fluxerr_auto"),
    ("e_flag_vis",          "e.flag_vis",                                 "NULL::integer"),
    ("e_flag_y",            "e.flag_y",                                   "NULL::integer"),
    ("e_flag_j",            "e.flag_j",                                   "NULL::integer"),
    ("e_flag_h",            "e.flag_h",                                   "NULL::integer"),
    ("e_point_like_prob",   "e.point_like_prob::double precision",        "e.v_class_star"),
    ("e_point_like_flag",   "e.point_like_flag",                          "NULL::smallint"),
    ("e_extended_prob",     "e.extended_prob::double precision",          "NULL::double precision"),
    ("e_extended_flag",     "e.extended_flag",                            "NULL::smallint"),
    ("e_spurious_flag",     "e.spurious_flag",                            "NULL::smallint"),
    ("e_det_quality_flag",  "e.det_quality_flag",                         "NULL::smallint"),
    ("e_deblended_flag",    "e.deblended_flag",                           "NULL::smallint"),
    ("e_blended_prob",      "e.blended_prob::double precision",           "NULL::double precision"),
    ("e_fwhm",              "e.fwhm::double precision",                   "e.v_fwhm_image"),
    ("e_mumax_minus_mag",   "e.mumax_minus_mag::double precision",        "NULL::double precision"),
    ("e_gal_ebv",           "e.gal_ebv::double precision",                "NULL::double precision"),
    ("e_variable_flag",     "e.variable_flag",                            "NULL::smallint"),
    ("e_gaia_id",           "e.gaia_id",                                  "NULL::bigint"),
    ("e_gaia_match_quality","e.gaia_match_quality::double precision",     "NULL::double precision"),
]

# Per-branch Euclid geometry/identity: (source_tag, table, id_col, ra_col, dec_col,
# is_q1). The q3c_join goes directly against these base tables so it uses each
# table's q3c index (mer_final_cat / merge_cat q3c_ang2ipix indexes) — the whole
# point: joining a pre-UNIONed CTE instead defeats the index (full-scan + filter).
_EBRANCHES = [
    ("Q1",  "euclid_q1.mer_final_cat", "object_id", "right_ascension", "declination",   True),
    ("ERO", "euclid_ero.merge_cat",    "v_objid",   "v_alpha_j2000",   "v_delta_j2000", False),
]


def crossmatch_select(cfg: dict, extra_where: str | None = None) -> str:
    """Return the cross-match SELECT SQL for one catalog configuration.

    One row per input source (closest Euclid match). Blend columns track both
    directions: Euclid sources near each input source, and input sources sharing
    the same closest Euclid match.

    extra_where : optional extra predicate ANDed onto the source-table scan,
                  used by the chunked builder to restrict to one RA band
                  (e.g. "q.ra >= 150 AND q.ra < 165"). None = whole catalog.
    """
    r_deg        = MATCH_RADIUS_ARCSEC / 3600.0
    z_col        = cfg.get("z_col")
    where        = cfg.get("where", "TRUE")
    if extra_where:
        where = f"({where}) AND ({extra_where})"
    extra_cols   = cfg.get("extra_cols", [])

    z_xmatch   = f"q.{z_col} AS qso_z," if z_col else ""
    z_select   = "x.qso_z,"            if z_col else ""
    extra_xmatch = "\n".join(
        f"                 q.{src} AS {alias}," for src, alias in extra_cols
    )
    extra_select = "\n".join(
        f"          x.{alias}," for _, alias in extra_cols
    )
    # Final SELECT list of e_* columns, straight from the materialized xmatch.
    e_select = ",\n".join(f"          x.{alias}" for alias, _, _ in _ECOLS)

    def _branch(source, etable, eid, era, edec, is_q1):
        # One UNION arm: q3c-join the source catalog directly to a base Euclid
        # table so the join uses that table's q3c index. Carries qso cols + the
        # standardised e_* cols (Q1 or ERO expression per column).
        ecols = ",\n".join(
            f"                 {(q1 if is_q1 else ero)} AS {alias}"
            for alias, q1, ero in _ECOLS
        )
        return f"""
          SELECT q.{cfg['id_col']}  AS qso_id,
                 q.{cfg['ra_col']}  AS qso_ra,
                 q.{cfg['dec_col']} AS qso_dec,
                 {z_xmatch}
                 {extra_xmatch}
                 '{source}'::text   AS euclid_source,
                 e.{eid}            AS euclid_id,
                 e.{era}            AS euclid_ra,
                 e.{edec}           AS euclid_dec,
                 q3c_dist(q.{cfg['ra_col']}, q.{cfg['dec_col']}, e.{era}, e.{edec}) * 3600.0 AS euclid_sep_arcsec,
{ecols}
          FROM {cfg['table']} AS q
          JOIN {etable} AS e
              ON q3c_join(q.{cfg['ra_col']}, q.{cfg['dec_col']}, e.{era}, e.{edec}, {r_deg})
          WHERE {where}"""

    xmatch_union = "\n          UNION ALL\n".join(_branch(*b) for b in _EBRANCHES)

    return f"""
      WITH
      -- All source×Euclid pairs within the match radius. Each UNION arm joins
      -- the source catalog straight to a base Euclid table so q3c uses that
      -- table's index; every e_* column is carried here (no later re-join).
      xmatch AS MATERIALIZED (
{xmatch_union}
      ),
      -- Per-source Euclid blend info: count + arrays (closest first)
      euclid_agg AS MATERIALIZED (
          SELECT qso_id,
                 COUNT(*)                                               AS n_euclid_within_1arcsec,
                 array_agg(euclid_id ORDER BY euclid_sep_arcsec)       AS all_euclid_ids_within_1arcsec,
                 array_agg(euclid_sep_arcsec ORDER BY euclid_sep_arcsec) AS all_euclid_seps_within_1arcsec
          FROM xmatch GROUP BY qso_id
      ),
      -- Closest Euclid per source (tiebreak: flux_vis descending — brightest)
      closest AS MATERIALIZED (
          SELECT DISTINCT ON (qso_id) qso_id, euclid_id, euclid_source,
                                      euclid_ra, euclid_dec
          FROM xmatch
          ORDER BY qso_id, euclid_sep_arcsec, e_flux_vis DESC NULLS LAST, euclid_id
      ),
      -- Reverse: per closest-Euclid-source, how many input sources claim it.
      -- euclid_id is globally unique within each source tag (Q1 object_id and
      -- ERO v_objid are both globally unique within their catalogs), so grouping
      -- by (euclid_id, euclid_source) is sufficient.
      qso_agg AS MATERIALIZED (
          SELECT euclid_id, euclid_source,
                 COUNT(*)           AS n_qso_per_euclid,
                 array_agg(qso_id)  AS all_qso_ids_per_euclid
          FROM closest GROUP BY euclid_id, euclid_source
      )
      SELECT DISTINCT ON (x.qso_id)
          '{cfg['tag']}'::text         AS qso_catalog,
          x.qso_id,
          x.qso_ra,
          x.qso_dec,
          {z_select}
          {extra_select}
          x.euclid_id                  AS e_id,
          x.euclid_ra                  AS e_ra,
          x.euclid_dec                 AS e_dec,
          x.euclid_source,
          x.euclid_sep_arcsec,
          ea.n_euclid_within_1arcsec,
          ea.all_euclid_ids_within_1arcsec,
          ea.all_euclid_seps_within_1arcsec,
          qa.n_qso_per_euclid,
          qa.all_qso_ids_per_euclid,
          (ea.n_euclid_within_1arcsec > 1 AND qa.n_qso_per_euclid > 1) AS is_mutual_blend,
{e_select}
      FROM xmatch AS x
      JOIN euclid_agg AS ea ON ea.qso_id         = x.qso_id
      JOIN qso_agg    AS qa ON qa.euclid_id       = x.euclid_id
                           AND qa.euclid_source    = x.euclid_source
      ORDER BY x.qso_id, x.euclid_sep_arcsec, x.e_flux_vis DESC NULLS LAST, x.euclid_id
    """


def build_table(username: str, cfg: dict, table_name: str) -> int:
    """DROP + CREATE {username}.{table_name} from the cross-match, add indexes,
    and print a summary. Returns 0 on success, 1 on failure."""
    import psycopg2
    output_table = f"{username}.{table_name}"
    print("=" * 74)
    print(f"Building {output_table}")
    print(f"  Catalog : {cfg['tag']}  ({cfg['table']})")
    print(f"  Match   : closest Euclid source within {MATCH_RADIUS_ARCSEC}\" per source (q3c)")
    print(f"  Target  : euclid_q1.mer_final_cat UNION ALL euclid_ero.merge_cat")
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
        cur.execute(f"CREATE INDEX ON {output_table} (e_id);")
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
# per-band scans so chunks parallelise too.
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
    cur.execute(f"SELECT euclid_source, COUNT(*) FROM {output_table} GROUP BY 1 ORDER BY 1;")
    print("  by euclid_source:")
    for src, n in cur.fetchall():
        print(f"    {src:<5} {n:>10,}")
    cur.execute(f"SELECT COUNT(*) FROM {output_table} WHERE n_euclid_within_1arcsec > 1;")
    print(f"  Sources with >1 Euclid within 1\":  {cur.fetchone()[0]:,}")
    cur.execute(f"SELECT COUNT(*) FROM {output_table} WHERE n_qso_per_euclid > 1;")
    print(f"  Sources sharing an Euclid match:   {cur.fetchone()[0]:,}")
    cur.execute(f"SELECT COUNT(*) FROM {output_table} WHERE is_mutual_blend;")
    print(f"  Mutual blends (both):               {cur.fetchone()[0]:,}")


def build_table_chunked(username: str, cfg: dict, table_name: str,
                        n_bands: int = 24, resume: bool = False,
                        ra_col: str | None = None) -> int:
    """Build {username}.{table_name} in N RA bands, each its own committed
    transaction — robust to VPN drops and resumable. See the Gaia builder's
    build_table_chunked for the full rationale; this is the Euclid-target twin
    (output index on e_id, summary by euclid_source).

    n_bands : number of equal RA slices (default 24 → 15° each).
    resume  : True keeps existing output + progress and skips done bands;
              False (default) drops both and starts clean.
    ra_col  : source RA column to band on; defaults to cfg['ra_col'].

    Caveat: the source-side blend columns (n_qso_per_euclid, all_qso_ids_per_euclid)
    are computed within each band, so two sources within 1" straddling a band
    boundary are not counted as sharing an Euclid match (negligible for diagnostics;
    use build_table for exactness). Closest-match columns are unaffected.
    """
    import psycopg2
    output_table = f"{username}.{table_name}"
    progress_table = f"{username}.{table_name}_progress"
    ra = ra_col or cfg["ra_col"]
    band_w = 360.0 / n_bands

    print("=" * 74)
    print(f"Building {output_table}  (chunked, {n_bands} RA bands of {band_w:g}°)")
    print(f"  Catalog : {cfg['tag']}  ({cfg['table']})")
    print(f"  Match   : closest Euclid source within {MATCH_RADIUS_ARCSEC}\" per source (q3c)")
    print(f"  Target  : euclid_q1.mer_final_cat UNION ALL euclid_ero.merge_cat")
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
            cur.execute(f"INSERT INTO {output_table}\n{sel};")
            n = cur.rowcount
            cur.execute(
                f"INSERT INTO {progress_table} (band, ra_lo, ra_hi, n_rows) "
                f"VALUES (%s, %s, %s, %s);", (i, lo, hi, n))
            conn.commit()
            print(f" {n:,} rows committed")

        print("\nAll bands done — building index...")
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {table_name}_e_id_idx "
            f"ON {output_table} (e_id);")
        conn.commit()
        # Fully built + indexed — the progress tracker has served its purpose,
        # so drop it. Only reached on full success; a mid-run failure leaves it
        # in place so --resume can pick up the unfinished bands.
        cur.execute(f"DROP TABLE IF EXISTS {progress_table};")
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
