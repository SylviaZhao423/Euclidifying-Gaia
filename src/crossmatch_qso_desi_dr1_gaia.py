#!/usr/bin/env python3
"""
DESI DR1 QSO × Gaia DR3 cross-match  →  {schema}.crossmatch_qso_desi_dr1_gaia

Spatial (q3c) cross-match of the DESI DR1 QSO sample with Gaia DR3 within 1";
closest Gaia source per QSO, n_gaia_within_1arcsec blend flag, 14 g_<feature>
columns. See qso_gaia_crossmatch_builder.py for the mechanism.

DESI source table
-----------------
Default: koposov.desi_dr1_agngal filtered to spectype='QSO' AND zcat_primary AND
zwarn=0 → 1,713,651 QSOs, identical to the DESI DR1 zpix primary-QSO count, with
proper target_ra/target_dec. This works today.

Preferred (pending access): vasily.desi_dr1_qso_zpix_v1 — Vasily's curated catalog.
It is currently PERMISSION-DENIED for siwei_zhao_2025. Once he grants
  GRANT SELECT ON vasily.desi_dr1_qso_zpix_v1 TO siwei_zhao_2025;
switch CFG to VASILY_CFG below (and confirm its column names — we could not read
its schema without access; the guesses are ra/dec/z/targetid).

Requires VPN (WSDB port 5432 blocked on plain eduroam). Auth via ~/.pgpass.

Usage:
    PGUSER=siwei_zhao_2025 python src/crossmatch_qso_desi_dr1_gaia.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from qso_gaia_crossmatch_builder import build_table  # noqa: E402

# Working default — Sergey Koposov's DESI DR1 AGN/galaxy catalog, QSO subset.
KOPOSOV_CFG = dict(
    tag="DESI_DR1Q",
    table="koposov.desi_dr1_agngal",
    id_col="targetid",
    ra_col="target_ra",
    dec_col="target_dec",
    z_col="z",
    where="q.spectype = 'QSO' AND q.zcat_primary AND q.zwarn = 0 "
          "AND q.target_ra IS NOT NULL AND q.target_dec IS NOT NULL",
)

# Preferred (default) — Vasily's curated DESI DR1 QSO catalog. Column names below
# are CONFIRMED from pg_catalog (targetid / ra / dec / z / zcat_primary), even
# though we can't yet read the DATA (permission denied). Once he runs
#   GRANT SELECT ON vasily.desi_dr1_qso_zpix_v1 TO siwei_zhao_2025;
# this will build with no further changes. Until then build_table() fails
# gracefully (no table built), and analyze_umap_qso.py --catalog both simply
# proceeds with whatever tables exist (SDSS-only for now).
VASILY_CFG = dict(
    tag="DESI_DR1Q",
    table="vasily.desi_dr1_qso_zpix_v1",
    id_col="targetid",
    ra_col="ra",
    dec_col="dec",
    z_col="z",
    where="q.zcat_primary AND q.ra IS NOT NULL AND q.dec IS NOT NULL",
)

CFG = VASILY_CFG    # Vasily's curated pure-QSO catalogue (granted 2026-07-17); KOPOSOV_CFG is the fallback
TABLE_NAME = "crossmatch_qso_desi_dr1_gaia"


def main() -> int:
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set (e.g. PGUSER=siwei_zhao_2025 python ...).")
        return 1
    return build_table(username, CFG, TABLE_NAME)


if __name__ == "__main__":
    sys.exit(main())
