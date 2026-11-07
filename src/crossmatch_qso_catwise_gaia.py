#!/usr/bin/env python3
"""
CatWISE 2020 × Gaia DR3 cross-match  →  {schema}.crossmatch_qso_catwise_gaia

Spatial (q3c) cross-match of catwise_202003.main with Gaia DR3 within 1";
closest Gaia source per CatWISE source, blend flags, 14 g_<feature> columns.
Also carries through W1/W2 magnitudes from CatWISE.
See builder_qso_gaia_crossmatch.py for the mechanism.

CatWISE 2020 source table
--------------------------
catwise_202003.main — full CatWISE 2020 catalog (~1.89 billion sources),
pre-filtered with the Stern et al. (2012) AGN colour cut: W1−W2 >= 0.8.
This selects AGN/QSO candidates by their distinctively red mid-infrared colours,
reducing the full catalog to a tractable AGN-candidate subset before matching.

CatWISE is ~1.89 billion rows (174.7M pass the W1−W2 filter) — the biggest of the
cross-matches. Default is therefore --chunked: build in RA bands, each its own
committed transaction, resumable after any VPN drop. Many bands here (default 72 =
5°) keep each transaction short. The band predicate rides the (ra, dec) btree index
so each band only scans its RA slice of the 1.89B-row source table.

Requires VPN (WSDB port 5432 blocked on plain eduroam). Auth via ~/.pgpass.

Usage:
    PGUSER=siwei_zhao_2025 python src/crossmatch_qso_catwise_gaia.py            # chunked (default, 72 bands)
    PGUSER=siwei_zhao_2025 python src/crossmatch_qso_catwise_gaia.py --resume    # finish leftover bands
    PGUSER=siwei_zhao_2025 python src/crossmatch_qso_catwise_gaia.py --bands 120 # even shorter transactions
    PGUSER=siwei_zhao_2025 python src/crossmatch_qso_catwise_gaia.py --single     # one-shot (fragile, not advised)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from builder_qso_gaia_crossmatch import build_table, build_table_chunked  # noqa: E402

CFG = dict(
    tag="CATWISE2020",
    table="catwise_202003.main",
    id_col="source_name",
    ra_col="ra",
    dec_col="dec",
    # No redshift for CatWISE — z_col omitted
    # Stern et al. (2012) AGN colour cut: W1−W2 >= 0.8 selects AGN/QSO candidates
    # by exploiting their distinctively red mid-infrared WISE colours.
    where=(
        "q.ra IS NOT NULL AND q.dec IS NOT NULL "
        "AND q.w1mpro IS NOT NULL AND q.w2mpro IS NOT NULL "
        "AND (q.w1mpro - q.w2mpro) >= 0.8"
    ),
    extra_cols=[
        ("w1mpro",  "w1mpro"),   # W1 magnitude (3.4 µm)
        ("w2mpro",  "w2mpro"),   # W2 magnitude (4.6 µm)
    ],
)

TABLE_NAME = "crossmatch_qso_catwise_gaia"


def main() -> int:
    ap = argparse.ArgumentParser(description="CatWISE 2020 × Gaia DR3 cross-match")
    ap.add_argument("--single", action="store_true",
                    help="one-shot CREATE TABLE AS instead of chunked (fragile to VPN drops)")
    ap.add_argument("--resume", action="store_true",
                    help="chunked: keep committed bands, run only the unfinished ones")
    ap.add_argument("--bands", type=int, default=72,
                    help="chunked: number of equal RA bands (default 72 = 5° each)")
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
