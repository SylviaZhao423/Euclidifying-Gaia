#!/usr/bin/env python3
"""Build {username}.gaia_pm_proxy_ero — the neighbour-PM proxy for the ERO 2-parameter
Gaia sources, self-contained (no separate crossmatch_ero_gaia_1_0_uncor step).

It (1) does the wide 1" ERO→Gaia position match *inline* to enumerate the 2-par
candidates (the closest Gaia within 1" of each ERO source, kept if 2-param), then
(2) computes the 30" median-neighbour proxy via gaia_pm_proxy_builder.build_proxy.

Consumed by crossmatch_ero_gaia_pmcor.py (LEFT JOIN gaia_pm_proxy_ero). Together with
gaia_pm_proxy_q1 this partitions the old shared gaia_pm_proxy (ERO/Q1 fields disjoint).

Requires WSDB (VPN). Auth via ~/.pgpass.

Usage:
    PGUSER=siwei_zhao_2025 python src/crossmatch_ero_gaia_1_0_pm_proxy.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gaia_pm_proxy_builder import WSDB_HOST, WSDB_DB, ERO_FIELDS, build_proxy  # noqa: E402

MATCH_RADIUS_ARCSEC = 1.0


def ero_2par_candidates(cur):
    """Closest Gaia source within 1" of each ERO source (euclid_ero.merge_cat),
    kept where it is a 2-parameter solution — the same set as the 2-par rows of the
    old crossmatch_ero_gaia_1_0_uncor, computed inline."""
    r_deg = MATCH_RADIUS_ARCSEC / 3600.0
    cur.execute(f"""
        SELECT DISTINCT g_source_id, g_ra, g_dec FROM (
            SELECT g.source_id AS g_source_id, g.ra AS g_ra, g.dec AS g_dec,
                   g.astrometric_params_solved AS aps
            FROM euclid_ero.merge_cat AS m
            INNER JOIN LATERAL (
                SELECT source_id, ra, dec, astrometric_params_solved
                FROM gaia_dr3.gaia_source AS g
                WHERE q3c_join(m.v_alpha_j2000, m.v_delta_j2000, g.ra, g.dec, {r_deg})
                ORDER BY q3c_dist(m.v_alpha_j2000, m.v_delta_j2000, g.ra, g.dec) ASC
                LIMIT 1
            ) AS g ON true
        ) x
        WHERE aps = 3
    """)
    return cur.fetchall()


def main() -> int:
    user = os.getenv("PGUSER")
    if not user:
        print("ERROR: PGUSER not set"); return 1
    out = f"{user}.gaia_pm_proxy_ero"

    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user)
    cur = conn.cursor()

    print("=" * 66)
    print(f"Building {out}   (ERO 2-par neighbour-PM proxy)")
    print(f"  candidates: closest Gaia ≤ {MATCH_RADIUS_ARCSEC}\" of each ERO source, 2-param (inline)")
    print("=" * 66)

    t0 = time.time()
    rows = ero_2par_candidates(cur)
    cur.close()
    print(f"  → {len(rows):,} ERO 2-par candidates  ({time.time()-t0:.0f}s)")
    if not rows:
        print("  no candidates — nothing to build."); conn.close(); return 1

    cand_id = np.array([r[0] for r in rows], dtype=np.int64)
    cand_ra = np.array([r[1] for r in rows], dtype=float)
    cand_dec = np.array([r[2] for r in rows], dtype=float)
    build_proxy(conn, cand_id, cand_ra, cand_dec, ERO_FIELDS, out)
    conn.close()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
