#!/usr/bin/env python3
"""
SDSS DR16Q × Gaia DR3 cross-match  →  {schema}.crossmatch_qso_sdss_dr16_gaia

Spatial (q3c) cross-match of the SDSS DR16Q QSOs with Gaia DR3 within 1"; closest
Gaia source per QSO, n_gaia_within_1arcsec blend flag, 14 g_<feature> columns.
See builder_qso_gaia_crossmatch.py for the mechanism.

Source table
------------
siwei_zhao_2025.sdss_dr16_qso — a pre-filtered (is_qso_final = 1, ra/dec NOT NULL)
copy of sdssdr16qso.main in our schema, q3c-indexed with parallel_workers = 4.
Built once with:
    CREATE TABLE siwei_zhao_2025.sdss_dr16_qso AS
      SELECT thing_id, ra, dec, z FROM sdssdr16qso.main
      WHERE is_qso_final = 1 AND ra IS NOT NULL AND dec IS NOT NULL;
    CREATE INDEX ON siwei_zhao_2025.sdss_dr16_qso (q3c_ang2ipix(ra, dec));
    ALTER TABLE siwei_zhao_2025.sdss_dr16_qso SET (parallel_workers = 4);
    ANALYZE siwei_zhao_2025.sdss_dr16_qso;
The plain copy dodges the original table's is_qso_final index (which steered the
planner into a slow single-threaded plan) and lets it parallelise.

The SDSS cross-match is intrinsically a longer q3c job than DESI (its QSOs sit in
denser Gaia regions, so each 1" cone rechecks more candidates) and a single giant
CREATE TABLE AS repeatedly died to VPN drops. Default here is therefore --chunked:
build in RA bands, each its own committed transaction, resumable after any drop.

Requires VPN (WSDB port 5432 blocked on plain eduroam). Auth via ~/.pgpass.

Usage:
    PGUSER=siwei_zhao_2025 python src/crossmatch_qso_sdss_dr16_gaia.py           # chunked (default)
    PGUSER=siwei_zhao_2025 python src/crossmatch_qso_sdss_dr16_gaia.py --resume   # finish leftover bands
    PGUSER=siwei_zhao_2025 python src/crossmatch_qso_sdss_dr16_gaia.py --bands 36 # finer bands
    PGUSER=siwei_zhao_2025 python src/crossmatch_qso_sdss_dr16_gaia.py --single    # one-shot CREATE TABLE AS
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from builder_qso_gaia_crossmatch import build_table, build_table_chunked  # noqa: E402

CFG = dict(
    tag="SDSS_DR16Q",
    table="siwei_zhao_2025.sdss_dr16_qso",   # filtered copy with parallel_workers=4
    id_col="thing_id",
    ra_col="ra",
    dec_col="dec",
    z_col="z",
    where="TRUE",   # pre-filtered at copy time
)

TABLE_NAME = "crossmatch_qso_sdss_dr16_gaia"


def main() -> int:
    ap = argparse.ArgumentParser(description="SDSS DR16Q × Gaia DR3 cross-match")
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
