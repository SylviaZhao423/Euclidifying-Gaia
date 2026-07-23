#!/usr/bin/env python3
"""
How many NSA-v1 central-Gaia nuclei also have Euclid imaging?

An exact-ID cross-match (Gaia source_id join — no second positional match): the NSA
nucleus×Gaia table against the two Euclid×Gaia tables, to see how many NSA nuclei fall in
a Euclid field (Q1 Deep Field or ERO) — i.e. whether the NSA×Euclid sample is big enough
to study galaxy nucleation at Euclid resolution.

    siwei_zhao_2025.crossmatch_nsa_v1_gaia               (NSA nucleus × Gaia; has_gaia = central detection)
  ⋈ siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep   (Euclid Q1 × Gaia)   by g_source_id
  ⋈ siwei_zhao_2025.crossmatch_ero_gaia_0_3              (Euclid ERO × Gaia)  by g_source_id

Requires WSDB — Astronomy/Cambridge VPN (plain eduroam blocks port 5432). Auth via ~/.pgpass.

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_nsa_euclid_overlap.py
    ... --preview 40      # also print a preview of the surviving nuclei
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from qso_gaia_crossmatch_builder import WSDB_HOST, WSDB_DB  # noqa: E402

NSA = "crossmatch_nsa_v1_gaia"
Q1  = "crossmatch_q1_mer_phz_vis_gaia_sep"
ERO = "crossmatch_ero_gaia_0_3"


def main() -> int:
    import psycopg2
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", type=int, default=20,
                    help="print this many surviving NSA×Euclid nuclei (default 20)")
    args = ap.parse_args()
    user = os.getenv("PGUSER")
    if not user:
        print("ERROR: PGUSER not set (e.g. PGUSER=siwei_zhao_2025 python ...)."); return 1
    nsa, q1, ero = f"{user}.{NSA}", f"{user}.{Q1}", f"{user}.{ERO}"
    try:
        conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=20)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 0")
    except Exception as exc:
        print(f"CONNECT FAILED (VPN up? eduroam blocks WSDB:5432): {exc}"); return 1

    print("=" * 78)
    print("NSA-v1 central-Gaia nuclei that also have Euclid imaging (Gaia source_id join)")
    print("=" * 78)

    cur.execute(f"SELECT count(*) FROM {nsa} WHERE has_gaia AND g_source_id IS NOT NULL")
    n_nuclei = cur.fetchone()[0]
    print(f"  NSA nuclei with a central Gaia detection: {n_nuclei:,}")

    def _in(euclid_table: str) -> int:
        cur.execute(f"""
            SELECT count(*) FROM {nsa} n
            WHERE n.has_gaia
              AND EXISTS (SELECT 1 FROM {euclid_table} e WHERE e.g_source_id = n.g_source_id)
        """)
        return cur.fetchone()[0]

    n_q1, n_ero = _in(q1), _in(ero)
    cur.execute(f"""
        SELECT count(*) FROM {nsa} n
        WHERE n.has_gaia
          AND (EXISTS (SELECT 1 FROM {q1}  e WHERE e.g_source_id = n.g_source_id)
            OR EXISTS (SELECT 1 FROM {ero} e WHERE e.g_source_id = n.g_source_id))
    """)
    n_either = cur.fetchone()[0]
    print(f"    also in Euclid Q1  ({Q1}): {n_q1:,}")
    print(f"    also in Euclid ERO ({ERO}): {n_ero:,}")
    print(f"    in EITHER (survive the Euclid cross-match): {n_either:,}"
          f"  ({100*n_either/max(n_nuclei,1):.2f}% of nuclei)")

    if args.preview and n_either:
        print(f"\n  — preview of up to {args.preview} surviving NSA×Euclid nuclei —")
        cur.execute(f"""
            SELECT n.nsaid, n.iauname, n.gal_ra, n.gal_dec,
                   n.g_astrometric_params_solved AS aps,
                   CASE WHEN EXISTS (SELECT 1 FROM {q1}  e WHERE e.g_source_id=n.g_source_id) THEN 'Q1'
                        ELSE 'ERO' END AS euclid
            FROM {nsa} n
            WHERE n.has_gaia
              AND (EXISTS (SELECT 1 FROM {q1}  e WHERE e.g_source_id=n.g_source_id)
                OR EXISTS (SELECT 1 FROM {ero} e WHERE e.g_source_id=n.g_source_id))
            ORDER BY n.nsaid
            LIMIT {int(args.preview)}
        """)
        print(f"    {'nsaid':>8}  {'iauname':<22} {'ra':>9} {'dec':>9}  {'aps':>4}  euclid")
        for nsaid, iau, ra, dec, aps, euclid in cur.fetchall():
            print(f"    {nsaid:>8}  {str(iau):<22} {ra:>9.4f} {dec:>+9.4f}  {aps:>4}  {euclid}")
    cur.close(); conn.close()
    print("=" * 78)
    verdict = ("enough for a dedicated NSA×Euclid nucleation study" if n_either >= 100 else
               "a small sample — likely case-by-case rather than statistical")
    print(f"Verdict: {n_either:,} NSA nuclei have Euclid imaging — {verdict}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
