#!/usr/bin/env python3
"""
DESI DR1 QSO × Gaia DR3 cross-match  →  {schema}.crossmatch_qso_desi_dr1_gaia

Spatial (q3c) cross-match of the DESI DR1 QSO sample with Gaia DR3 within 1";
closest Gaia source per QSO, n_gaia_within_1arcsec blend flag, 14 g_<feature>
columns. See builder_qso_gaia_crossmatch.py for the mechanism.

DESI source table
-----------------
vasily.desi_dr1_qso_zpix_v1 — Vasily's curated DESI DR1 QSO catalog.

Requires VPN (WSDB port 5432 blocked on plain eduroam). Auth via ~/.pgpass.

Usage:
    PGUSER=siwei_zhao_2025 python src/crossmatch_qso_desi_dr1_gaia.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from builder_qso_gaia_crossmatch import build_table  # noqa: E402

CFG = dict(
    tag="DESI_DR1Q",
    table="vasily.desi_dr1_qso_zpix_v1",
    id_col="targetid",
    ra_col="ra",
    dec_col="dec",
    z_col="z",
    where="q.zcat_primary AND q.ra IS NOT NULL AND q.dec IS NOT NULL",
)
TABLE_NAME = "crossmatch_qso_desi_dr1_gaia"


def main() -> int:
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set (e.g. PGUSER=siwei_zhao_2025 python ...).")
        return 1
    return build_table(username, CFG, TABLE_NAME)


if __name__ == "__main__":
    sys.exit(main())
