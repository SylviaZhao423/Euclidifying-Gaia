#!/usr/bin/env python3
"""
NSA v0_1_2 galaxies × Gaia DR3 central cross-match (CROSS-CHECK nucleation sample)
    →  {schema}.crossmatch_nsa_v0_gaia

Same as crossmatch_nsa_v1_gaia.py but for the older v0_1_2 catalog. Vasily flagged
that v0 has some flaky/spurious coordinates, and that this correlates with
morphology (fluffier galaxies → less well-defined centres) — so treat v0 as a
SYSTEMATICS cross-check on the v1 result, not the headline sample.

NB: v0 uses petroth50/petroth90 (no underscore); the concentration column differs
from v1. Property aliases are the same version-agnostic set as v1.

Access: vasily.nsa_v0_1_2 is currently PERMISSION-DENIED. Once
    GRANT SELECT ON vasily.nsa_v0_1_2 TO siwei_zhao_2025;
lands this builds with no changes.

Requires VPN. Auth via ~/.pgpass.

Usage:
    PGUSER=siwei_zhao_2025 python src/crossmatch_nsa_v0_gaia.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from builder_nsa_gaia_crossmatch import build_table  # noqa: E402

CFG = dict(
    tag="NSA_v0",
    table="vasily.nsa_v0_1_2",
    id_col="nsaid",
    ra_col="ra",
    dec_col="dec",
    props={
        "gal_log_mass":  "q.log_stellar_mass",
        "gal_sersic_n":  "q.sersic_n",
        "gal_re_arcsec": "q.sersic_th50",
        "gal_ba":        "q.sersic_ba",
        "gal_conc":      "q.petroth90 / NULLIF(q.petroth50, 0)",
        "gal_zdist":     "q.zdist",
    },
)

TABLE_NAME = "crossmatch_nsa_v0_gaia"


def main() -> int:
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set (e.g. PGUSER=siwei_zhao_2025 python ...).")
        return 1
    return build_table(username, CFG, TABLE_NAME)


if __name__ == "__main__":
    sys.exit(main())
