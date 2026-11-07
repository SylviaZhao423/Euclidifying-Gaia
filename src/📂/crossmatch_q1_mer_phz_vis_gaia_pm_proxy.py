#!/usr/bin/env python3
"""Build {username}.gaia_pm_proxy_q1 — the neighbour-PM proxy for the Q1 2-parameter
Gaia sources.

Q1 already carries pre-computed Gaia associations, so — unlike ERO — no 1" position
match is needed: the 2-par candidates come straight from the NULL-PM rows of
{username}.crossmatch_q1_mer_phz_vis_gaia. The 30" median-neighbour proxy is then
computed via gaia_pm_proxy_builder.build_proxy over the Q1 (EDF) fields.

Consumed by crossmatch_q1_mer_phz_vis_gaia_sep_pmcor.py (LEFT JOIN gaia_pm_proxy_q1).
Together with gaia_pm_proxy_ero this partitions the old shared gaia_pm_proxy.

(Note: the Q1 EDF fields are deep/high-latitude and sparse in bright 5/6-par stars,
so only a small fraction of Q1 2-par sources reach ≥5 neighbours — the rest fall back
to 0 PM in the pmcor step, which is fine for faint high-latitude sources.)

Requires WSDB (VPN). Auth via ~/.pgpass.

Usage:
    PGUSER=siwei_zhao_2025 python src/crossmatch_q1_mer_phz_vis_gaia_pm_proxy.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gaia_pm_proxy_builder import WSDB_HOST, WSDB_DB, Q1_FIELDS, build_proxy  # noqa: E402

SOURCE_TABLE = "crossmatch_q1_mer_phz_vis_gaia"   # pre-computed Q1×Gaia association


def q1_2par_candidates(cur, user):
    cur.execute(f"""
        SELECT DISTINCT g_source_id, g_ra, g_dec
        FROM {user}.{SOURCE_TABLE}
        WHERE g_pmra IS NULL AND g_pmdec IS NULL
    """)
    return cur.fetchall()


def main() -> int:
    user = os.getenv("PGUSER")
    if not user:
        print("ERROR: PGUSER not set"); return 1
    out = f"{user}.gaia_pm_proxy_q1"

    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user)
    cur = conn.cursor()

    print("=" * 66)
    print(f"Building {out}   (Q1 2-par neighbour-PM proxy)")
    print(f"  candidates: 2-param rows of {user}.{SOURCE_TABLE} (pre-computed assoc)")
    print("=" * 66)

    t0 = time.time()
    rows = q1_2par_candidates(cur, user)
    cur.close()
    print(f"  → {len(rows):,} Q1 2-par candidates  ({time.time()-t0:.0f}s)")
    if not rows:
        print("  no candidates — nothing to build."); conn.close(); return 1

    cand_id = np.array([r[0] for r in rows], dtype=np.int64)
    cand_ra = np.array([r[1] for r in rows], dtype=float)
    cand_dec = np.array([r[2] for r in rows], dtype=float)
    build_proxy(conn, cand_id, cand_ra, cand_dec, Q1_FIELDS, out)
    conn.close()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
