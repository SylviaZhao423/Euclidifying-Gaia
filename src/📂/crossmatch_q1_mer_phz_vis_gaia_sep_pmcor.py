#!/usr/bin/env python3
"""
Create crossmatch_q1_mer_phz_vis_gaia_sep_pmcor: PM-correction-only
sep <= 0.3" subset of crossmatch_q1_mer_phz_vis_gaia.

This is the PM-correction-only variant. The CANONICAL default-named
table `crossmatch_q1_mer_phz_vis_gaia_sep` is now built by
`crossmatch_q1_mer_phz_vis_gaia_sep.py` and applies an additional
"raw-sep also ≤ 0.3″" filter on top of this PM-corrected cut
(intersection logic — see that script for rationale).

Methodological choices vs the legacy
`crossmatch_q1_mer_phz_vis_gaia_sep_uncor.py`:

  1. The angular separation is computed against the per-source
     PM-propagated Gaia position (Gaia DR3 epoch 2016.0 → Q1 epoch
     2024.0) instead of the raw Gaia catalogue position. Sources without
     published PM (Gaia 2-par solutions) fall back to a neighbour-PM
     proxy ({username}.gaia_pm_proxy_q1); 2-par sources with no proxy use
     zero PM (no shift).

  2. The separation is computed with q3c_dist_pm (exact great-circle
     with PM propagation) — the same method used by
     crossmatch_ero_gaia_pmcor.py — replacing the small-angle
     approximation in the legacy uncorrected script.

  {username}.crossmatch_q1_mer_phz_vis_gaia_sep_pmcor
      SELECT * FROM crossmatch_q1_mer_phz_vis_gaia
      WHERE q3c_dist_pm(g_ra, g_dec, COALESCE(g_pmra, proxy.proxy_pmra, 0), ...,
                        m_ra, m_dec, q1_epoch) * 3600  <=  0.3

Usage:
    PGUSER=siwei_zhao_2025 python src/crossmatch_q1_mer_phz_vis_gaia_sep_pmcor.py
"""

import os
import sys

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

# Epoch baselines (match analyze_q1_sep_rayleigh_corrected_per_source_proxy.py)
GAIA_REF_EPOCH = 2016.0
Q1_OBS_EPOCH   = 2024.0
DELTA_T_YEARS  = Q1_OBS_EPOCH - GAIA_REF_EPOCH    # = 8.0

# Exact great-circle separation with per-source PM propagation, computed
# via q3c's native q3c_dist_pm(). Signature:
#   q3c_dist_pm(ra1, dec1, pmra1, pmdec1, cos_dec_pmra_flag, epoch1,
#               ra2, dec2, epoch2)  →  degrees
# It propagates (ra1, dec1) from epoch1 to epoch2 using (pmra1, pmdec1)
# and returns the great-circle distance to (ra2, dec2). The flag = 1
# tells q3c that pmra1 already includes the cos(dec) factor (Gaia's
# pmra* convention).
#
# For 2-par Gaia sources (no published PM), we COALESCE into the
# pre-computed neighbour-PM proxy table {username}.gaia_pm_proxy_q1
# (built by crossmatch_q1_mer_phz_vis_gaia_pm_proxy.py — median pmra/pmdec of 5/6-par
# neighbours within 30"). 2-par sources without enough neighbours fall
# back to zero PM (no propagation).
SEPCOR_EXPR = (
    f"q3c_dist_pm(g_ra, g_dec, "
    f"COALESCE(g_pmra,  proxy.proxy_pmra,  0), "
    f"COALESCE(g_pmdec, proxy.proxy_pmdec, 0), "
    f"1::int, {GAIA_REF_EPOCH}::real, "
    f"m_right_ascension, m_declination, "
    f"{Q1_OBS_EPOCH}::real) * 3600"
)


def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER environment variable not set!"); return 1

    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()

    source_table = f"{username}.crossmatch_q1_mer_phz_vis_gaia"
    proxy_table  = f"{username}.gaia_pm_proxy_q1"
    output_table = f"{username}.crossmatch_q1_mer_phz_vis_gaia_sep_pmcor"

    print("=" * 70)
    print(f"Creating PM-correction-only sep <= 0.3\" subset (variant: _pmcor)")
    print(f"  Source : {source_table}")
    print(f"  Proxy  : {proxy_table}  (neighbour-PM proxy for 2-par sources)")
    print(f"  Output : {output_table}")
    print(f"  Epoch  : Gaia {GAIA_REF_EPOCH} → Q1 {Q1_OBS_EPOCH}  "
          f"(Δt = {DELTA_T_YEARS:.1f} yr)")
    print(f"  Filter : sep_corr <= 0.3\"  (q3c_dist_pm great-circle)")
    print(f"  PM     : per-source pmra/pmdec; 2-par sources use proxy "
          f"(median of neighbour 5/6-par PMs within 30\")")
    print("=" * 70)

    # Drop sep_*_uncor's siblings? — only the canonical _sep, leave _uncor alone.
    cursor.execute(f"DROP TABLE IF EXISTS {output_table} CASCADE;")
    cursor.execute(f"""
        CREATE TABLE {output_table} AS
        SELECT src.*
        FROM {source_table} AS src
        LEFT JOIN {proxy_table} AS proxy
          ON proxy.g_source_id = src.g_source_id
        WHERE q3c_dist_pm(
                  src.g_ra, src.g_dec,
                  COALESCE(src.g_pmra,  proxy.proxy_pmra,  0),
                  COALESCE(src.g_pmdec, proxy.proxy_pmdec, 0),
                  1::int, {GAIA_REF_EPOCH}::real,
                  src.m_right_ascension, src.m_declination,
                  {Q1_OBS_EPOCH}::real
              ) * 3600 <= 0.3
    """)
    conn.commit()

    cursor.execute(f"SELECT COUNT(*) FROM {output_table};")
    row_count = cursor.fetchone()[0]

    # Compare with the preserved uncorrected snapshot for context
    cursor.execute(f"SELECT COUNT(*) FROM {source_table};")
    src_count = cursor.fetchone()[0]
    sep_count = None
    try:
        cursor.execute(
            f"SELECT COUNT(*) FROM {username}."
            f"crossmatch_q1_mer_phz_vis_gaia_sep_uncor;")
        sep_count = cursor.fetchone()[0]
    except Exception:
        conn.rollback()

    cursor.close(); conn.close()

    print()
    print(f"  Source pool   :   {src_count:>10,}")
    if sep_count is not None:
        print(f"  Uncorrected (≤ 0.3″) :   {sep_count:>10,}  "
              f"(crossmatch_q1_mer_phz_vis_gaia_sep_uncor)")
    print(f"  PM-corrected (≤ 0.3″):   {row_count:>10,}  "
          f"(crossmatch_q1_mer_phz_vis_gaia_sep_pmcor)")
    if sep_count is not None:
        delta = row_count - sep_count
        sign = "+" if delta >= 0 else ""
        print(f"  Δ vs uncorrected      :   {sign}{delta:,}")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
