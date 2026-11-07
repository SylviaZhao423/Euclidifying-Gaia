#!/usr/bin/env python3
"""
DESI DR1 QSO × CatWISE 2020 cross-match  →  {schema}.crossmatch_qso_desi_dr1_catwise

Spatial (q3c) cross-match of the DESI DR1 QSO sample with the FULL CatWISE 2020 catalog
within 1"; closest CatWISE source per QSO, W1/W2 magnitudes carried through, blend flags.
See builder_qso_catwise_crossmatch.py for the mechanism.

Unlike crossmatch_qso_catwise_gaia/euclid.py (CatWISE is the driver, pre-filtered to the
Stern W1-W2>=0.8 AGN cut), here DESI QSOs are the driver and CatWISE is matched in FULL —
so we attach UNBIASED W1/W2 photometry to the spectroscopic training sample (needed to
test whether a WISE colour helps the Euclid photo-z; see experiment_photz_features.py).

DESI source table
-----------------
vasily.desi_dr1_qso_zpix_v1 — Vasily's curated DESI DR1 QSO catalog (same as the
crossmatch_qso_desi_dr1_gaia / _euclid entry points).

Requires VPN (WSDB port 5432 blocked on plain eduroam). Auth via ~/.pgpass.

Usage:
    PGUSER=siwei_zhao_2025 python src/crossmatch_qso_desi_dr1_catwise.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from builder_qso_catwise_crossmatch import build_table  # noqa: E402

CFG = dict(
    tag="DESI_DR1Q",
    table="vasily.desi_dr1_qso_zpix_v1",
    id_col="targetid",
    ra_col="ra",
    dec_col="dec",
    z_col="z",
    where="q.zcat_primary AND q.ra IS NOT NULL AND q.dec IS NOT NULL",
)
TABLE_NAME = "crossmatch_qso_desi_dr1_catwise"


def main() -> int:
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set (e.g. PGUSER=siwei_zhao_2025 python ...).")
        return 1
    return build_table(username, CFG, TABLE_NAME)


if __name__ == "__main__":
    sys.exit(main())
