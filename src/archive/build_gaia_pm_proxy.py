#!/usr/bin/env python3
"""
Build {username}.gaia_pm_proxy:
    For every Gaia DR3 2-parameter source (no published proper motion)
    that lies within 5" of an Euclid source (ERO merge_cat OR Q1
    mer_final_cat), compute a proxy PM as the per-axis median (pmra,
    pmdec) of its 5- or 6-parameter neighbours within 30" on the sky.
    Sources with fewer than 5 neighbours are excluded (no proxy).

This is the SQL counterpart of the Python ``neighbour_pm_proxy`` used
in the analysis scripts. Building it once and storing it as a table
means the SQL cross-match scripts (``crossmatch_q1_mer_phz_vis_gaia_sep.py``
and ``crossmatch_ero_gaia.py``) can fold proxy PMs into their
``COALESCE(pmra, proxy.proxy_pmra, 0)`` chain via a single LEFT JOIN.

Strategy
--------
Doing a full Gaia 2-par × 5/6-par self-join over the entire DR3
catalogue is wasteful — only ~100k 2-par sources can ever become
Euclid cross-match candidates (out of ~200M 2-par sources globally).
The existing canonical cross-match tables already enumerate every
Gaia source that survived a wide pre-filter against Euclid, so we
take the candidate 2-par set directly from there:

  ``candidate_2par`` (TEMP): the union of distinct g_source_id from
      crossmatch_q1_mer_phz_vis_gaia (Q1 — direct ID match, includes
      every 2-par Gaia near a Q1 source) and crossmatch_ero_gaia_1_0
      (ERO 1.0″ q3c match — wider than the 0.3″ table so it captures
      every 2-par source that could become a candidate at any radius
      we currently use).
  ``gaia_pm_proxy`` (PERSISTED): for each candidate, find its 5/6-par
      neighbours within 30", aggregate per-axis median (pmra, pmdec),
      keep only those with ≥ 5 neighbours.

Output
------
{username}.gaia_pm_proxy
    g_source_id (bigint, PK)
    proxy_pmra  (real, mas/yr)
    proxy_pmdec (real, mas/yr)
    n_neighbours (int)

Usage
-----
    PGUSER=siwei_zhao_2025 python src/build_gaia_pm_proxy.py
"""

import os
import sys
import time

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

NEIGHBOUR_RADIUS_ARCSEC = 30.0  # neighbour search radius for proxy
MIN_NEIGHBOURS          = 5     # min 5/6-par neighbours to define a proxy


def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER not set"); return 1

    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cur = conn.cursor()

    proxy_table = f'{username}.gaia_pm_proxy'

    print('=' * 70)
    print(f'Building {proxy_table}')
    print(f'  Candidate 2-par sources: from existing Q1+ERO cross-match tables')
    print(f'  Neighbour radius       : {NEIGHBOUR_RADIUS_ARCSEC}"')
    print(f'  Min 5/6-par neighbours : {MIN_NEIGHBOURS}')
    print('=' * 70)

    # ---- step 1: candidate 2-par sources from existing cross-match tables -
    # (Q1 direct-ID-matched parent + ERO 1.0" q3c match — together cover all
    # 2-par Gaia sources that could ever become a candidate at any radius.)
    print('\n[1/3] Building candidate 2-par set from existing crossmatch tables...')
    t0 = time.time()
    cur.execute(f"""
        DROP TABLE IF EXISTS pg_temp.candidate_2par;
        CREATE TEMP TABLE candidate_2par AS
        SELECT DISTINCT g_source_id AS source_id, g_ra AS ra, g_dec AS dec
        FROM (
            SELECT g_source_id, g_ra, g_dec
              FROM {username}.crossmatch_q1_mer_phz_vis_gaia
              WHERE g_pmra IS NULL AND g_pmdec IS NULL
            UNION
            SELECT g_source_id, g_ra, g_dec
              FROM {username}.crossmatch_ero_gaia_1_0
              WHERE g_pmra IS NULL AND g_pmdec IS NULL
        ) AS s;
        CREATE INDEX ON pg_temp.candidate_2par
            (q3c_ang2ipix(ra, dec));
        ANALYZE pg_temp.candidate_2par;
    """)
    conn.commit()
    cur.execute("SELECT COUNT(*) FROM pg_temp.candidate_2par")
    n_candidate = cur.fetchone()[0]
    print(f'  → {n_candidate:,} candidate 2-par sources  ({time.time()-t0:.1f} s)')

    # ---- step 2: aggregate neighbour PMs ----------------------------------
    print(f'\n[2/3] Aggregating 5/6-par neighbour PMs (radius {NEIGHBOUR_RADIUS_ARCSEC}")...')
    t0 = time.time()
    cur.execute(f"""
        DROP TABLE IF EXISTS {proxy_table};
        CREATE TABLE {proxy_table} AS
        SELECT c.source_id AS g_source_id,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY g5.pmra)
                   ::real AS proxy_pmra,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY g5.pmdec)
                   ::real AS proxy_pmdec,
               COUNT(*)::int AS n_neighbours
        FROM pg_temp.candidate_2par c
        JOIN gaia_dr3.gaia_source g5
          ON q3c_join(c.ra, c.dec, g5.ra, g5.dec,
                      {NEIGHBOUR_RADIUS_ARCSEC}/3600.0)
        WHERE g5.astrometric_params_solved IN (31, 95)
        GROUP BY c.source_id
        HAVING COUNT(*) >= {MIN_NEIGHBOURS};
    """)
    conn.commit()
    cur.execute(f"SELECT COUNT(*) FROM {proxy_table}")
    n_proxy = cur.fetchone()[0]
    print(f'  → {n_proxy:,} 2-par sources got a valid proxy '
          f'({100*n_proxy/n_candidate:.1f}%)  ({time.time()-t0:.1f} s)')

    # ---- step 3: index for downstream LEFT JOINs --------------------------
    print('\n[3/3] Indexing proxy table on g_source_id ...')
    cur.execute(f"""
        ALTER TABLE {proxy_table}
            ADD CONSTRAINT gaia_pm_proxy_pkey PRIMARY KEY (g_source_id);
        ANALYZE {proxy_table};
    """)
    conn.commit()
    print(f'  → primary key on g_source_id added; statistics gathered')

    # ---- diagnostics ------------------------------------------------------
    cur.execute(f"""
        SELECT MIN(n_neighbours), AVG(n_neighbours)::int,
               PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY n_neighbours)::int,
               MAX(n_neighbours)
        FROM {proxy_table}
    """)
    nmin, nmean, nmed, nmax = cur.fetchone()
    print()
    print('  Neighbour count stats: '
          f'min={nmin}  median={nmed}  mean={nmean}  max={nmax}')

    cur.close(); conn.close()
    print('\nDone.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
