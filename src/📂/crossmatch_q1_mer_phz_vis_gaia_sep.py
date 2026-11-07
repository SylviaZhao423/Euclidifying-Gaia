#!/usr/bin/env python3
"""
Canonical Q1-Gaia sep <= 0.3" cross-match table — INTERSECTION variant.

This builds the default-named table:
    {username}.crossmatch_q1_mer_phz_vis_gaia_sep

It is defined as the INTERSECTION of two cuts:

  (a) PM-corrected sep ≤ 0.3″   — q3c_dist_pm with own PM, neighbour-PM
                                  proxy fallback (gaia_pm_proxy), or 0.
                                  Already enforced by the PM-correction-only
                                  sibling table
                                  crossmatch_q1_mer_phz_vis_gaia_sep_pmcor.
  (b) Raw sep ≤ 0.3″            — q3c_dist with no PM applied. Same
                                  definition used by the legacy uncorrected
                                  sibling table
                                  crossmatch_q1_mer_phz_vis_gaia_sep_uncor.

Implementation: starts from the _pmcor sibling (which already enforces
cut (a) AND carries the post-processing derived columns like
g_corrected_bp_rp_excess) and applies the raw-sep cut (b) on top.

Why intersection (instead of just PM-corrected, or just raw)?
-------------------------------------------------------------
PM correction adds ~5,688 sources to the Q1-Gaia 0.3″ sample that the
uncorrected match missed (high-PM stars whose PM-propagated 2024 position
falls within 0.3″ of an Euclid source even though the raw 2016 position
is > 0.3″). Empirically these added sources concentrate in the
low-confidence PHZ-galaxy tail (median p_phz_gal_prob = 0.60 vs 0.72
for typical galaxies) — likely high-PM stars that PHZ misclassified
as galaxies. Training a star/galaxy classifier on the PM-corrected
pool produces measurably more false positives on real stars (FP rate
0.0179 vs 0.0145 for the intersection — see
output/xgboost_q1_precision_eval/).

The intersection — sources where BOTH the PM-corrected AND the raw match
are ≤ 0.3″ — is the *consensus* cross-match. It keeps the bulk of
high-quality matches while dropping the noisy newly-matched sources.
For ML training in particular, the intersection is the cleanest source
pool and gives the best metrics on every test we ran (PHZ F1, ROC AUC,
DESI galaxy recall, and proxy-truth star precision).

The two component cuts are still available individually:
  - PM-corrected only :  crossmatch_q1_mer_phz_vis_gaia_sep_pmcor
  - Raw only          :  crossmatch_q1_mer_phz_vis_gaia_sep_uncor

Prerequisite: the _pmcor sibling table must exist
(run crossmatch_q1_mer_phz_vis_gaia_sep_pmcor.py first; or rename the
old PM-corrected canonical _sep table via ALTER TABLE).

Usage:
    PGUSER=siwei_zhao_2025 python src/crossmatch_q1_mer_phz_vis_gaia_sep.py
"""

import os
import sys

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

GAIA_REF_EPOCH = 2016.0
Q1_OBS_EPOCH   = 2024.0
DELTA_T_YEARS  = Q1_OBS_EPOCH - GAIA_REF_EPOCH    # = 8.0


def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!"); return 1

    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()

    pmcor_table  = f"{username}.crossmatch_q1_mer_phz_vis_gaia_sep_pmcor"
    output_table = f"{username}.crossmatch_q1_mer_phz_vis_gaia_sep"

    print("=" * 70)
    print("Creating canonical sep <= 0.3\" subset (INTERSECTION variant)")
    print(f"  Source : {pmcor_table}  (PM-corrected sep ≤ 0.3\" already enforced)")
    print(f"  Output : {output_table}")
    print(f"  Filter : + raw sep ≤ 0.3\"  (q3c_dist, no PM)")
    print("=" * 70)

    cursor.execute(f"DROP TABLE IF EXISTS {output_table} CASCADE;")
    cursor.execute(f"""
        CREATE TABLE {output_table} AS
        SELECT *
        FROM {pmcor_table}
        WHERE q3c_dist(g_ra, g_dec,
                       m_right_ascension, m_declination) * 3600 <= 0.3
    """)
    conn.commit()

    cursor.execute(f"SELECT COUNT(*) FROM {output_table};")
    row_count = cursor.fetchone()[0]

    # Sibling counts for context
    cursor.execute(f"SELECT COUNT(*) FROM {pmcor_table};")
    pmcor_count = cursor.fetchone()[0]
    uncor_count = None
    sib_uncor = f"{username}.crossmatch_q1_mer_phz_vis_gaia_sep_uncor"
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {sib_uncor};")
        uncor_count = cursor.fetchone()[0]
    except Exception:
        conn.rollback()

    cursor.close(); conn.close()

    print()
    print(f"  PM-corrected only (≤ 0.3″)         : {pmcor_count:>10,}  (sibling _pmcor)")
    if uncor_count is not None:
        print(f"  Raw only          (≤ 0.3″)         : {uncor_count:>10,}  (sibling _uncor)")
    print(f"  INTERSECTION (canonical _sep)       : {row_count:>10,}  ← this script")
    delta_pm = row_count - pmcor_count
    print(f"    Δ vs PM-corrected only            : {delta_pm:+,}")
    if uncor_count is not None:
        delta_uncor = row_count - uncor_count
        print(f"    Δ vs raw only                     : {delta_uncor:+,}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
