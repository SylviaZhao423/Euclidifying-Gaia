#!/usr/bin/env python3
"""
ERO-Gaia cross-match (PM-correction-only variant) with configurable
matching radius. Writes to `crossmatch_ero_gaia_<r>_pmcor` table names.

This is the PM-correction-only variant. The CANONICAL default-named
tables `crossmatch_ero_gaia_<r>` are now built by `crossmatch_ero_gaia.py`
and apply an additional "raw-sep also ≤ R″" filter on top of this
PM-corrected closest-match (intersection logic — see that script).

The closest-match selection here is computed against the per-source
PM-propagated Gaia position (Gaia DR3 epoch 2016.0 → ERO epoch 2023.5)
instead of the raw Gaia 2016 position. Sources without published PM
(Gaia 2-par solutions) fall back to a neighbour-PM proxy
({username}.gaia_pm_proxy_ero); 2-par sources with no proxy use zero PM.

The legacy uncorrected pipeline is preserved as
`crossmatch_ero_gaia_uncor.py`, which writes to `..._uncor` table names.

Strategy
--------
We want to keep the q3c spatial index for speed but match against
PM-corrected positions. We do this in two layers inside the same SQL
query:

  1. Fast **wide pre-filter** with `q3c_join(m.ra, m.dec, g.ra, g.dec,
     wide_radius)` using the raw Gaia coordinates — this hits the q3c
     index on gaia_source and prunes ~1.8 B Gaia rows down to a small
     candidate pool per ERO source.
  2. **Per-candidate PM correction** computed in the SELECT and used in
     both the WHERE final-radius cut and the ORDER BY. We pick the
     closest candidate by *PM-corrected* sep.

The wide pre-filter radius is generous enough (5×final_radius, capped
at 2.0″) that no high-PM source whose corrected sep is within
final_radius can be missed by the pre-filter. For typical Gaia DR3 PMs
(median ≪ 10 mas/yr), a 7.5-yr propagation moves a source by ≪ 0.1″,
so 2″ comfortably contains anything that lands within a sub-arcsec
final_radius after correction.

Output table name:
    crossmatch_ero_gaia_<radius_with_underscores>_pmcor
e.g. for radius 0.3″ → crossmatch_ero_gaia_0_3_pmcor.

Usage:
    PGUSER=siwei_zhao_2025 python src/crossmatch_ero_gaia_pmcor.py --radii 0.3
    PGUSER=siwei_zhao_2025 python src/crossmatch_ero_gaia_pmcor.py --radii 0.3 1.0
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

# Epoch baselines (match analyze_ero_sep_rayleigh_corrected_per_source_proxy.py)
GAIA_REF_EPOCH = 2016.0
ERO_OBS_EPOCH  = 2023.5
DELTA_T_YEARS  = ERO_OBS_EPOCH - GAIA_REF_EPOCH    # = 7.5

# Cap on the wide pre-filter radius (q3c index lookup before PM correction)
WIDE_RADIUS_CAP_ARCSEC = 2.0
WIDE_RADIUS_FACTOR     = 5.0


# PM-aware great-circle distance via q3c's native q3c_dist_pm().
# Signature:
#   q3c_dist_pm(ra1, dec1, pmra1, pmdec1, cos_dec_pmra_flag, epoch1,
#               ra2, dec2, epoch2)  →  degrees
# Propagates (ra1, dec1) from epoch1 to epoch2 using (pmra1, pmdec1)
# and returns the great-circle distance to (ra2, dec2). flag=1 marks
# pmra1 as cos(dec)-corrected (Gaia's pmra* convention).
#
# 2-par sources (no published PM) get a neighbour-PM proxy via the
# pre-built {username}.gaia_pm_proxy_ero table (built by
# crossmatch_ero_gaia_1_0_pm_proxy.py — median of 5/6-par neighbour PMs within 30").
# 2-par sources without enough neighbours fall back to zero PM.
def _pm_dist_expr(g_alias='g', proxy_alias='proxy',
                   m_ra_col='m.v_alpha_j2000',
                   m_dec_col='m.v_delta_j2000'):
    return (
        f"q3c_dist_pm("
        f"{g_alias}.ra, {g_alias}.dec, "
        f"COALESCE({g_alias}.pmra,  {proxy_alias}.proxy_pmra,  0), "
        f"COALESCE({g_alias}.pmdec, {proxy_alias}.proxy_pmdec, 0), "
        f"1::int, {GAIA_REF_EPOCH}::real, "
        f"{m_ra_col}, {m_dec_col}, {ERO_OBS_EPOCH}::real)"
    )


def get_column_names(cursor, schema, table):
    cursor.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """, (schema, table))
    return [r[0] for r in cursor.fetchall()]


def crossmatch_corrected(ero_columns, gaia_columns, username, radius_arcsec):
    import psycopg2

    ero_cols_str  = ', '.join([f'm.{c} AS e_{c}' for c in ero_columns])
    gaia_cols_str = ', '.join([f'g.{c} AS g_{c}' for c in gaia_columns])
    gaia_bare     = ', '.join([c.split(' AS ')[0] for c in gaia_cols_str.split(', ')])

    radius_suffix = str(radius_arcsec).replace('.', '_')
    table_name    = f'{username}.crossmatch_ero_gaia_{radius_suffix}_pmcor'
    proxy_table   = f'{username}.gaia_pm_proxy_ero'
    radius_deg    = radius_arcsec / 3600.0

    # Pre-filter radius: 5× final_radius, capped at WIDE_RADIUS_CAP_ARCSEC
    wide_radius_arcsec = min(WIDE_RADIUS_FACTOR * radius_arcsec,
                              WIDE_RADIUS_CAP_ARCSEC)
    wide_radius_deg = wide_radius_arcsec / 3600.0

    pm_dist = _pm_dist_expr(g_alias='g', proxy_alias='proxy',
                             m_ra_col='m.v_alpha_j2000',
                             m_dec_col='m.v_delta_j2000')

    query = f"""
        DROP TABLE IF EXISTS {table_name};
        CREATE TABLE {table_name} AS
        SELECT
            {ero_cols_str},
            {gaia_cols_str}
        FROM euclid_ero.merge_cat AS m
        INNER JOIN LATERAL (
            SELECT {gaia_bare}
            FROM gaia_dr3.gaia_source AS g
            LEFT JOIN {proxy_table} AS proxy
              ON proxy.g_source_id = g.source_id
            WHERE q3c_join(m.v_alpha_j2000, m.v_delta_j2000,
                           g.ra, g.dec, {wide_radius_deg})
              AND {pm_dist} <= {radius_deg}
            ORDER BY {pm_dist} ASC
            LIMIT 1
        ) AS g ON true;
    """

    print(f"\n  Final radius: {radius_arcsec}\"  "
          f"(pre-filter {wide_radius_arcsec}\")")
    print(f"  Epoch       : Gaia {GAIA_REF_EPOCH} → ERO {ERO_OBS_EPOCH}  "
          f"(Δt = {DELTA_T_YEARS:.1f} yr)")
    print(f"  Table       : {table_name}")

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    row_count = cursor.fetchone()[0]

    # Comparison with the preserved uncorrected sibling table, if it exists
    sibling = f'{username}.crossmatch_ero_gaia_{radius_suffix}_uncor'
    cursor.execute("""
        SELECT to_regclass(%s) IS NOT NULL
    """, (sibling,))
    sibling_exists = cursor.fetchone()[0]
    sibling_count = None
    if sibling_exists:
        cursor.execute(f"SELECT COUNT(*) FROM {sibling};")
        sibling_count = cursor.fetchone()[0]

    cursor.close(); conn.close()

    print(f"  Rows        : {row_count:>10,}  (PM-corrected)")
    if sibling_count is not None:
        delta = row_count - sibling_count
        sign = "+" if delta >= 0 else ""
        print(f"  Rows (sibling uncorrected {sibling.split('.')[-1]}): "
              f"{sibling_count:>10,}  (Δ vs corrected: {sign}{delta:,})")
    return table_name, row_count


def main():
    parser = argparse.ArgumentParser(
        description="PM-corrected cross-match of ERO merge_cat with Gaia DR3."
    )
    parser.add_argument(
        "--radii", nargs="+", type=float, required=True, metavar="ARCSEC",
        help="One or more (post-correction) matching radii in arcseconds, "
             "e.g. --radii 0.3 1.0"
    )
    args = parser.parse_args()

    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER not set"); return 1

    import psycopg2
    print(f"Connecting to WSDB as {username}...")
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()
    ero_columns  = get_column_names(cursor, 'euclid_ero', 'merge_cat')
    gaia_columns = get_column_names(cursor, 'gaia_dr3',   'gaia_source')
    cursor.close(); conn.close()
    print(f"  ERO columns:  {len(ero_columns)}")
    print(f"  Gaia columns: {len(gaia_columns)}")

    radii = sorted(set(args.radii))
    for radius in radii:
        crossmatch_corrected(ero_columns, gaia_columns, username, radius)

    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
