#!/usr/bin/env python3
"""
NSA v1_0_1 galaxies × Gaia DR3 central cross-match (PRIMARY nucleation sample)
    →  {schema}.crossmatch_nsa_v1_gaia

LEFT join (all galaxies kept), closest Gaia source within 1" of the photometric
centre = nucleus candidate. See nsa_gaia_crossmatch_builder.py for the mechanism
and how this differs from the QSO cross-match.

Galaxy-property columns are normalized to version-agnostic aliases
(gal_log_mass / gal_sersic_n / gal_re_arcsec / gal_ba / gal_conc / gal_zdist) so
the downstream nucleation analysis is identical for v0 and v1.

Access: vasily.nsa_v1_0_1 is currently PERMISSION-DENIED. Column names below are
confirmed from pg_catalog; once
    GRANT SELECT ON vasily.nsa_v1_0_1 TO siwei_zhao_2025;
lands this builds with no changes. (Confirm then that `ra`/`dec` is the intended
photometric centre vs racat/deccat.)

Requires VPN. Auth via ~/.pgpass.

Usage:
    PGUSER=siwei_zhao_2025 python src/crossmatch_nsa_v1_gaia.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from nsa_gaia_crossmatch_builder import build_table  # noqa: E402

CFG = dict(
    tag="NSA_v1",
    table="vasily.nsa_v1_0_1",
    id_col="nsaid",
    ra_col="ra",
    dec_col="dec",
    props={
        "gal_log_mass":  "q.log_stellar_mass",
        "gal_sersic_n":  "q.sersic_n",
        "gal_re_arcsec": "q.sersic_th50",
        "gal_ba":        "q.sersic_ba",
        "gal_conc":      "q.petro_th90 / NULLIF(q.petro_th50, 0)",
        "gal_zdist":     "q.zdist",
    },
)

TABLE_NAME = "crossmatch_nsa_v1_gaia"


def main() -> int:
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set (e.g. PGUSER=siwei_zhao_2025 python ...).")
        return 1
    return build_table(username, CFG, TABLE_NAME)


if __name__ == "__main__":
    sys.exit(main())
