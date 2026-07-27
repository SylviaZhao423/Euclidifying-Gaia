#!/usr/bin/env python3
"""
CatWISE 2020 × Euclid cross-match  →  {schema}.crossmatch_qso_catwise_euclid

Spatial (q3c) cross-match of catwise_202003.main (W1−W2 >= 0.8 AGN candidates)
against the combined Euclid footprint (Q1 + ERO) within 1". Closest Euclid source
per CatWISE source, blend flags, and key Euclid photometry/morphology columns.
See euclid_crossmatch_builder.py for the mechanism.

CatWISE source filter
---------------------
Stern et al. (2012) AGN colour cut: W1−W2 >= 0.8 (Vega), selecting ~174.7M of
~1.89B CatWISE 2020 sources as AGN/QSO candidates by their mid-infrared colours.

Euclid target
-------------
euclid_q1.mer_final_cat UNION ALL euclid_ero.merge_cat, unified via the builder's
EUCLID_UNION_SQL with standardised e_* column names. Q1 is the primary data set;
ERO (early release observations) has limited area. The `euclid_source` column
('Q1'/'ERO') distinguishes the origin of each match.

CatWISE is ~1.89 billion rows (174.7M pass the W1−W2 filter). Default is --chunked:
build in RA bands, each its own committed transaction, resumable after a VPN drop.
Euclid covers only small fields, so most RA bands return ~0 matches quickly — the
few bands overlapping Euclid Q1/ERO do the real work.

Requires VPN (WSDB port 5432 blocked on plain eduroam). Auth via ~/.pgpass.

Usage:
    PGUSER=siwei_zhao_2025 python src/crossmatch_qso_catwise_euclid.py            # chunked (default, 72 bands)
    PGUSER=siwei_zhao_2025 python src/crossmatch_qso_catwise_euclid.py --resume    # finish leftover bands
    PGUSER=siwei_zhao_2025 python src/crossmatch_qso_catwise_euclid.py --bands 120 # even shorter transactions
    PGUSER=siwei_zhao_2025 python src/crossmatch_qso_catwise_euclid.py --single     # one-shot (fragile)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from euclid_crossmatch_builder import build_table, build_table_chunked  # noqa: E402

CFG = dict(
    tag="CATWISE2020",
    table="catwise_202003.main",
    id_col="source_name",
    ra_col="ra",
    dec_col="dec",
    # No redshift for CatWISE — z_col omitted
    # Stern et al. (2012) AGN colour cut: W1−W2 >= 0.8 selects AGN/QSO candidates
    where=(
        "q.ra IS NOT NULL AND q.dec IS NOT NULL "
        "AND q.w1mpro IS NOT NULL AND q.w2mpro IS NOT NULL "
        "AND (q.w1mpro - q.w2mpro) >= 0.8"
    ),
    extra_cols=[
        ("w1mpro", "w1mpro"),   # W1 magnitude (3.4 µm)
        ("w2mpro", "w2mpro"),   # W2 magnitude (4.6 µm)
    ],
)

TABLE_NAME = "crossmatch_qso_catwise_euclid"


def main() -> int:
    ap = argparse.ArgumentParser(description="CatWISE 2020 × Euclid cross-match")
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
