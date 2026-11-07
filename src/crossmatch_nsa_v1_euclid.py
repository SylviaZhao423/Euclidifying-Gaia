#!/usr/bin/env python3
"""
NSA v1_0_1 galaxies × Euclid central cross-match (PRIMARY nucleation sample)
    →  {schema}.crossmatch_nsa_v1_euclid

LEFT join (all galaxies kept), closest Euclid source within 1" of the photometric
centre. Euclid target is euclid_q1.mer_final_cat UNION ALL euclid_ero.merge_cat.
See builder_nsa_euclid_crossmatch.py for the mechanism; this is the Euclid twin of
crossmatch_nsa_v1_gaia.py and shares the same galaxy-property config.

Galaxy-property columns are normalized to version-agnostic aliases
(gal_log_mass / gal_sersic_n / gal_re_arcsec / gal_ba / gal_conc / gal_zdist) so
the downstream analysis is identical for v0 and v1.

Because Euclid covers only a few fields, most NSA galaxies are unmatched
(has_euclid = FALSE) — the matched fraction measures NSA coverage by Euclid.

Requires VPN. Auth via ~/.pgpass.

Usage:
    PGUSER=siwei_zhao_2025 python src/crossmatch_nsa_v1_euclid.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from builder_nsa_euclid_crossmatch import build_table  # noqa: E402

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

TABLE_NAME = "crossmatch_nsa_v1_euclid"


def main() -> int:
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set (e.g. PGUSER=siwei_zhao_2025 python ...).")
        return 1
    return build_table(username, CFG, TABLE_NAME)


if __name__ == "__main__":
    sys.exit(main())
