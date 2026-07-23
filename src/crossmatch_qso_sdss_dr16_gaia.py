#!/usr/bin/env python3
"""
SDSS DR16Q × Gaia DR3 cross-match  →  {schema}.crossmatch_qso_sdss_dr16_gaia

Spatial (q3c) cross-match of sdssdr16qso.main (is_qso_final = 1) with Gaia DR3
within 1"; closest Gaia source per QSO, n_gaia_within_1arcsec blend flag, 14
g_<feature> columns. See qso_gaia_crossmatch_builder.py for the mechanism.

Requires VPN (WSDB port 5432 blocked on plain eduroam). Auth via ~/.pgpass.

Usage:
    PGUSER=siwei_zhao_2025 python src/crossmatch_qso_sdss_dr16_gaia.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from qso_gaia_crossmatch_builder import build_table  # noqa: E402

CFG = dict(
    tag="SDSS_DR16Q",
    table="sdssdr16qso.main",
    id_col="thing_id",
    ra_col="ra",
    dec_col="dec",
    z_col="z",
    where="q.is_qso_final = 1 AND q.ra IS NOT NULL AND q.dec IS NOT NULL",
)

TABLE_NAME = "crossmatch_qso_sdss_dr16_gaia"


def main() -> int:
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set (e.g. PGUSER=siwei_zhao_2025 python ...).")
        return 1
    return build_table(username, CFG, TABLE_NAME)


if __name__ == "__main__":
    sys.exit(main())
