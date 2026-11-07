#!/usr/bin/env python3
"""
Canonical ERO-Gaia cross-match — INTERSECTION variant.

Builds default-named tables {username}.crossmatch_ero_gaia_<radius>
defined as the INTERSECTION of two cuts on the PM-correction-only
sibling table {username}.crossmatch_ero_gaia_<radius>_pmcor:

  (a) PM-corrected sep ≤ R″   — already enforced by the _pmcor sibling,
                                via q3c_dist_pm with own PM, neighbour-PM
                                proxy fallback (gaia_pm_proxy), or 0.
  (b) Raw sep ≤ R″            — q3c_dist with no PM applied.

Output tables:
    {username}.crossmatch_ero_gaia_<r>     ← canonical (intersection)
    {username}.crossmatch_ero_gaia_<r>_pmcor   (built by _pmcor.py)
    {username}.crossmatch_ero_gaia_<r>_uncor   (legacy raw-only)

Why intersection? See crossmatch_q1_mer_phz_vis_gaia_sep.py for the
full argument: the intersection is the consensus cross-match between
PM-corrected and raw matching, dropping the small set of newly added
high-PM matches that tend to come with noisier labels.

Prerequisite: the corresponding _pmcor table must exist
(run crossmatch_ero_gaia_pmcor.py --radii <R> first).

Usage:
    PGUSER=siwei_zhao_2025 python src/crossmatch_ero_gaia.py --radii 0.3
    PGUSER=siwei_zhao_2025 python src/crossmatch_ero_gaia.py --radii 0.3 1.0
"""

import argparse
import os
import sys


WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'


def build_intersection(username, radius_arcsec):
    import psycopg2
    radius_suffix = str(radius_arcsec).replace('.', '_')
    pmcor_table   = f'{username}.crossmatch_ero_gaia_{radius_suffix}_pmcor'
    output_table  = f'{username}.crossmatch_ero_gaia_{radius_suffix}'

    print(f"\n  Radius : {radius_arcsec}\"")
    print(f"  Source : {pmcor_table}  (PM-corrected sep ≤ {radius_arcsec}\")")
    print(f"  Output : {output_table}  (intersection: + raw sep ≤ {radius_arcsec}\")")

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()

    # ERO position columns from the merge_cat join: e_v_alpha_j2000, e_v_delta_j2000
    # Gaia position columns: g_ra, g_dec
    cursor.execute(f"DROP TABLE IF EXISTS {output_table} CASCADE;")
    cursor.execute(f"""
        CREATE TABLE {output_table} AS
        SELECT *
        FROM {pmcor_table}
        WHERE q3c_dist(g_ra, g_dec,
                       e_v_alpha_j2000, e_v_delta_j2000) * 3600 <= {radius_arcsec}
    """)
    conn.commit()

    cursor.execute(f"SELECT COUNT(*) FROM {output_table};")
    row_count = cursor.fetchone()[0]
    cursor.execute(f"SELECT COUNT(*) FROM {pmcor_table};")
    pmcor_count = cursor.fetchone()[0]

    sibling_uncor = f'{username}.crossmatch_ero_gaia_{radius_suffix}_uncor'
    cursor.execute("SELECT to_regclass(%s) IS NOT NULL", (sibling_uncor,))
    if cursor.fetchone()[0]:
        cursor.execute(f"SELECT COUNT(*) FROM {sibling_uncor};")
        uncor_count = cursor.fetchone()[0]
    else:
        uncor_count = None

    cursor.close(); conn.close()

    print(f"  Rows   :")
    print(f"    PM-corrected only       : {pmcor_count:>10,}  (sibling _pmcor)")
    if uncor_count is not None:
        print(f"    Raw only                : {uncor_count:>10,}  (sibling _uncor)")
    print(f"    INTERSECTION (canonical): {row_count:>10,}  ← this script")
    delta_pm = row_count - pmcor_count
    print(f"    Δ vs PM-corrected only  : {delta_pm:+,}")
    if uncor_count is not None:
        delta_uncor = row_count - uncor_count
        print(f"    Δ vs raw only           : {delta_uncor:+,}")
    return output_table, row_count


def main():
    parser = argparse.ArgumentParser(
        description="Canonical ERO-Gaia (intersection) cross-match table builder."
    )
    parser.add_argument(
        "--radii", nargs="+", type=float, required=True, metavar="ARCSEC",
        help="One or more matching radii in arcseconds (e.g. --radii 0.3 1.0). "
             "The corresponding _pmcor table must already exist for each radius."
    )
    args = parser.parse_args()

    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER not set"); return 1

    print("=" * 70)
    print("Canonical ERO-Gaia cross-match (INTERSECTION variant)")
    print(f"  Radii  : {args.radii}")
    print("=" * 70)

    radii = sorted(set(args.radii))
    for radius in radii:
        build_intersection(username, radius)

    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
