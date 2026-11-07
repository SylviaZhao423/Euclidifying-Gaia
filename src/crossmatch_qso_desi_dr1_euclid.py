#!/usr/bin/env python3
"""
DESI DR1 QSO × Euclid cross-match  →  {schema}.crossmatch_qso_desi_dr1_euclid

Spatial (q3c) cross-match of the DESI DR1 QSO sample with the combined Euclid
footprint (euclid_q1.mer_final_cat UNION ALL euclid_ero.merge_cat) within 1";
closest Euclid source per QSO, bidirectional blend columns, and key Euclid
photometry/morphology (e_* columns). See builder_qso_euclid_crossmatch.py for the
mechanism. This is the Euclid twin of crossmatch_qso_desi_dr1_gaia.py.

DESI source table
-----------------
vasily.desi_dr1_qso_zpix_v1 — Vasily's curated DESI DR1 QSO catalog
(q.zcat_primary selects the primary redshift per target).

Euclid target
-------------
euclid_q1.mer_final_cat UNION ALL euclid_ero.merge_cat, unified by the builder's
EUCLID_UNION_SQL with standardised e_* columns; euclid_source ('Q1'/'ERO') tags origin.

Blend columns (both directions, as for the Gaia matches):
  n_euclid_within_1arcsec, all_euclid_ids_within_1arcsec, all_euclid_seps_within_1arcsec
      — Euclid sources within 1" of each QSO
  n_qso_per_euclid, all_qso_ids_per_euclid
      — QSOs sharing the same closest Euclid source
  is_mutual_blend — both > 1

Default is --chunked (RA bands, each its own committed transaction, resumable after
a VPN drop). Euclid covers small fields, so most bands return ~0 matches quickly.

Requires VPN (WSDB port 5432 blocked on plain eduroam). Auth via ~/.pgpass.

Usage:
    PGUSER=siwei_zhao_2025 python src/crossmatch_qso_desi_dr1_euclid.py           # chunked (default)
    PGUSER=siwei_zhao_2025 python src/crossmatch_qso_desi_dr1_euclid.py --resume   # finish leftover bands
    PGUSER=siwei_zhao_2025 python src/crossmatch_qso_desi_dr1_euclid.py --bands 36
    PGUSER=siwei_zhao_2025 python src/crossmatch_qso_desi_dr1_euclid.py --single    # one-shot CREATE TABLE AS
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from builder_qso_euclid_crossmatch import build_table, build_table_chunked  # noqa: E402

CFG = dict(
    tag="DESI_DR1Q",
    table="vasily.desi_dr1_qso_zpix_v1",
    id_col="targetid",
    ra_col="ra",
    dec_col="dec",
    z_col="z",
    where="q.zcat_primary AND q.ra IS NOT NULL AND q.dec IS NOT NULL",
)

TABLE_NAME = "crossmatch_qso_desi_dr1_euclid"


def main() -> int:
    ap = argparse.ArgumentParser(description="DESI DR1 QSO × Euclid cross-match")
    ap.add_argument("--single", action="store_true",
                    help="one-shot CREATE TABLE AS instead of chunked (fragile to VPN drops)")
    ap.add_argument("--resume", action="store_true",
                    help="chunked: keep committed bands, run only the unfinished ones")
    ap.add_argument("--bands", type=int, default=24,
                    help="chunked: number of equal RA bands (default 24 = 15° each)")
    args = ap.parse_args()

    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set (e.g. PGUSER=siwei_zhao_2025 python ...).")
        return 1
    if args.single:
        return build_table(username, CFG, TABLE_NAME)
    return build_table_chunked(username, CFG, TABLE_NAME,
                               n_bands=args.bands, resume=args.resume)


if __name__ == "__main__":
    sys.exit(main())
