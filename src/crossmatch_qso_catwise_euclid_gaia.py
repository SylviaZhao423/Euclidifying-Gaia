#!/usr/bin/env python3
"""
Join crossmatch_qso_catwise_euclid × crossmatch_qso_catwise_gaia
→  {schema}.crossmatch_qso_catwise_euclid_gaia

NOT a new catalog scan — INNER JOIN of the two already-built tables on qso_id
(CatWISE source_name). Carries ALL columns from both source tables; the only
renames are the two is_mutual_blend columns (one per source table):
  euclid_is_mutual_blend  — n_euclid_within_1arcsec > 1 AND n_qso_per_euclid > 1
  gaia_is_mutual_blend    — n_gaia_within_1arcsec   > 1 AND n_qso_per_gaia   > 1

Shared identity columns (qso_id, qso_ra, qso_dec, w1mpro, w2mpro, qso_catalog)
are kept once from the Euclid table.

Requires VPN. Both source tables must already be built:
  PGUSER=siwei_zhao_2025 python src/crossmatch_qso_catwise_euclid.py
  PGUSER=siwei_zhao_2025 python src/crossmatch_qso_catwise_gaia.py

Usage:
  PGUSER=siwei_zhao_2025 python src/crossmatch_qso_catwise_euclid_gaia.py
"""

import os
import sys

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
TABLE_E   = "crossmatch_qso_catwise_euclid"
TABLE_G   = "crossmatch_qso_catwise_gaia"
TABLE_OUT = "crossmatch_qso_catwise_euclid_gaia"

# Full SELECT — all columns from both tables, conflicts resolved by prefix/rename.
# Shared cols (qso_id, qso_ra, qso_dec, w1mpro, w2mpro, qso_catalog) taken from e.
# is_mutual_blend exists in both → renamed to euclid_is_mutual_blend / gaia_is_mutual_blend.
_SQL = """
    SELECT
        -- shared identity (from Euclid side; both tables have the same values)
        e.qso_catalog,
        e.qso_id,
        e.qso_ra,
        e.qso_dec,
        e.w1mpro,
        e.w2mpro,

        -- Euclid match columns
        e.e_id,
        e.e_ra,
        e.e_dec,
        e.euclid_source,
        e.euclid_sep_arcsec,
        e.n_euclid_within_1arcsec,
        e.all_euclid_ids_within_1arcsec,
        e.all_euclid_seps_within_1arcsec,
        e.n_qso_per_euclid,
        e.all_qso_ids_per_euclid,
        e.is_mutual_blend               AS euclid_is_mutual_blend,

        -- Euclid photometry + morphology (all e_* columns)
        e.e_vis_det,
        e.e_flux_vis,
        e.e_fluxerr_vis,
        e.e_flux_vis_psf,
        e.e_fluxerr_vis_psf,
        e.e_flux_y,
        e.e_fluxerr_y,
        e.e_flux_j,
        e.e_fluxerr_j,
        e.e_flux_h,
        e.e_fluxerr_h,
        e.e_flag_vis,
        e.e_flag_y,
        e.e_flag_j,
        e.e_flag_h,
        e.e_point_like_prob,
        e.e_point_like_flag,
        e.e_extended_prob,
        e.e_extended_flag,
        e.e_spurious_flag,
        e.e_det_quality_flag,
        e.e_deblended_flag,
        e.e_blended_prob,
        e.e_fwhm,
        e.e_mumax_minus_mag,
        e.e_gal_ebv,
        e.e_variable_flag,
        e.e_gaia_id,
        e.e_gaia_match_quality,

        -- Gaia match columns
        g.g_source_id,
        g.g_ra,
        g.g_dec,
        g.gaia_sep_arcsec,
        g.n_gaia_within_1arcsec,
        g.all_gaia_ids_within_1arcsec,
        g.all_gaia_seps_within_1arcsec,
        g.n_qso_per_gaia,
        g.all_qso_ids_per_gaia,
        g.is_mutual_blend               AS gaia_is_mutual_blend,

        -- Gaia metadata + 14 XGBoost features
        g.g_astrometric_params_solved,
        g.g_phot_g_mean_mag,
        g.g_phot_g_n_obs,
        g.g_parallax_over_error,
        g.g_astrometric_gof_al,
        g.g_astrometric_chi2_al,
        g.g_astrometric_excess_noise_sig,
        g.g_ipd_gof_harmonic_amplitude,
        g.g_ruwe,
        g.g_phot_g_mean_flux_over_error,
        g.g_phot_bp_mean_flux_over_error,
        g.g_phot_rp_mean_flux_over_error,
        g.g_bp_g,
        g.g_g_rp,
        g.g_corrected_bp_rp_excess,
        g.g_pm_significance,
        g.g_position_error

    FROM {src_e} AS e
    JOIN {src_g} AS g ON g.qso_id = e.qso_id
"""


def _build(username: str) -> int:
    import psycopg2

    output_table = f"{username}.{TABLE_OUT}"
    src_e = f"{username}.{TABLE_E}"
    src_g = f"{username}.{TABLE_G}"
    sql = _SQL.format(src_e=src_e, src_g=src_g)

    print("=" * 70)
    print(f"Building {output_table}")
    print(f"  {src_e}")
    print(f"  INNER JOIN {src_g}  ON qso_id")
    print("=" * 70)

    try:
        conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username,
                                connect_timeout=30, keepalives=1, keepalives_idle=30,
                                keepalives_interval=10, keepalives_count=5)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 0")

        for tbl in (src_e, src_g):
            cur.execute(f"SELECT COUNT(*) FROM {tbl}")
            print(f"  {tbl}: {cur.fetchone()[0]:,} rows")

        print(f"\n  DROP + CREATE {output_table} ...")
        cur.execute(f"DROP TABLE IF EXISTS {output_table} CASCADE;")
        cur.execute(f"CREATE TABLE {output_table} AS {sql};")
        conn.commit()

        # ── summary ──────────────────────────────────────────────────────────
        cur.execute(f"SELECT COUNT(*) FROM {output_table}")
        n_total = cur.fetchone()[0]

        cur.execute(f"SELECT COUNT(*) FROM {src_e}")
        n_e = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*) FROM {src_g}")
        n_g = cur.fetchone()[0]

        print(f"\n  crossmatch_qso_catwise_euclid : {n_e:,}")
        print(f"  crossmatch_qso_catwise_gaia   : {n_g:,}")
        print(f"  overlap (inner join)          : {n_total:,}  "
              f"({100*n_total/max(n_e,1):.1f}% of Euclid, "
              f"{100*n_total/max(n_g,1):.1f}% of Gaia)")

        cur.execute(f"SELECT euclid_source, COUNT(*) FROM {output_table} "
                    f"GROUP BY 1 ORDER BY 1")
        print("  by euclid_source:")
        for src, n in cur.fetchall():
            print(f"    {src:<5} {n:>8,}")

        cur.execute(f"SELECT COUNT(*) FROM {output_table} "
                    f"WHERE n_euclid_within_1arcsec > 1")
        print(f"  n_euclid > 1 (crowded Euclid):       {cur.fetchone()[0]:,}")
        cur.execute(f"SELECT COUNT(*) FROM {output_table} "
                    f"WHERE n_gaia_within_1arcsec > 1")
        print(f"  n_gaia > 1   (crowded Gaia):         {cur.fetchone()[0]:,}")
        cur.execute(f"SELECT COUNT(*) FROM {output_table} "
                    f"WHERE n_qso_per_euclid > 1")
        print(f"  n_qso_per_euclid > 1 (shared Euclid match): {cur.fetchone()[0]:,}")
        cur.execute(f"SELECT COUNT(*) FROM {output_table} "
                    f"WHERE n_qso_per_gaia > 1")
        print(f"  n_qso_per_gaia > 1   (shared Gaia match):   {cur.fetchone()[0]:,}")
        cur.execute(f"SELECT COUNT(*) FROM {output_table} WHERE euclid_is_mutual_blend")
        print(f"  euclid mutual blend:                 {cur.fetchone()[0]:,}")
        cur.execute(f"SELECT COUNT(*) FROM {output_table} WHERE gaia_is_mutual_blend")
        print(f"  gaia mutual blend:                   {cur.fetchone()[0]:,}")

        # 2x2 crowding breakdown
        cells = {}
        for ng in (1, 2):
            for ne in (1, 2):
                op_g = ">" if ng == 2 else "="
                op_e = ">" if ne == 2 else "="
                val_g = "1" if ng == 2 else "1"
                val_e = "1" if ne == 2 else "1"
                cur.execute(
                    f"SELECT COUNT(*) FROM {output_table} "
                    f"WHERE n_gaia_within_1arcsec {op_g} 1 "
                    f"AND n_euclid_within_1arcsec {op_e} 1"
                )
                cells[(ng, ne)] = cur.fetchone()[0]
        print(f"\n  crowding 2×2 (within 1\" of WISE):")
        print(f"                       n_euclid=1     n_euclid>1")
        print(f"    n_gaia=1    {cells[(1,1)]:>12,}   {cells[(1,2)]:>12,}")
        print(f"    n_gaia>1    {cells[(2,1)]:>12,}   {cells[(2,2)]:>12,}")

        # verify all columns landed
        cur.execute("SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema=%s AND table_name=%s "
                    "ORDER BY ordinal_position",
                    (username, TABLE_OUT))
        cols = [r[0] for r in cur.fetchall()]
        print(f"\n  {len(cols)} columns: {', '.join(cols)}")

        # ── indexes ───────────────────────────────────────────────────────────
        print("\n  Creating indexes ...")
        cur.execute(f"CREATE INDEX ON {output_table} (qso_id);")
        cur.execute(f"CREATE INDEX ON {output_table} (g_source_id);")
        cur.execute(f"CREATE INDEX ON {output_table} (e_id);")
        conn.commit()
        cur.close(); conn.close()

    except Exception as exc:
        msg = str(exc).lower()
        if "does not exist" in msg or "no such" in msg:
            print(f"\nMISSING TABLE: {exc}")
            print("  Run crossmatch_qso_catwise_euclid.py and "
                  "crossmatch_qso_catwise_gaia.py first.")
        elif "permission denied" in msg:
            print(f"\nNO ACCESS: {exc}")
        else:
            print(f"\nERROR: {exc}")
            print("  VPN up? Plain eduroam blocks WSDB:5432.")
        return 1

    print("=" * 70)
    print(f"Done.  SELECT * FROM {output_table} LIMIT 5;")
    return 0


def main() -> int:
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: set PGUSER, e.g.  PGUSER=siwei_zhao_2025 python ...")
        return 1
    return _build(username)


if __name__ == "__main__":
    sys.exit(main())
