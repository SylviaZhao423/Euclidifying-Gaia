#!/usr/bin/env python3
"""Project the CatWISE×Euclid×Gaia QSO sample (siwei_zhao_2025.crossmatch_qso_catwise_euclid_gaia)
onto the combined DESI/SDSS QSO×Gaia UMAP — the Euclid-matched counterpart of
analyze_qso_umap_space_catwise.py (which projects the full CatWISE×Gaia sample).

Only the source table differs (crossmatch_qso_catwise_euclid_gaia instead of
crossmatch_qso_catwise_gaia): these are the CatWISE QSOs that have BOTH a Euclid detection AND a
Gaia DR3 match (~6.4k unique Gaia sources). Everything else — the projection recipe, the Q1 XGB
P(gal), and the two plots — is identical, reusing run()/make_argparser() from that script.

Reads:
  output/qso_umap/base/arrays/combined_qso_gaia.{npz,_reducer.joblib}          (combined build)
  siwei_zhao_2025.crossmatch_qso_catwise_euclid_gaia                            (the QSOs to project)
Writes (output/qso_umap/base/):
  arrays/catwise_e_gaia_projected.npz     cached projection (--replot re-plots instantly)
  qso_space_catwise_e_umap.png            2D density (Blues, log) | XGB P(gal) | W1-W2
  qso_space_catwise_e_gaia_features.png   14-feature grid, projected points coloured by value

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_qso_umap_space_catwise_e.py
  python src/analyze_qso_umap_space_catwise_e.py --replot
"""
import os
from pathlib import Path

from analyze_qso_umap_space_catwise import run, make_argparser

PROJECT_DIR = Path(__file__).resolve().parent.parent
CW_TABLE_E = "crossmatch_qso_catwise_euclid_gaia"


def main() -> int:
    args = make_argparser(__doc__).parse_args()
    user = os.environ.get("PGUSER", "siwei_zhao_2025")
    base = PROJECT_DIR / "output" / "qso_umap" / "base"
    return run(user, args, CW_TABLE_E,
               base / "arrays" / "catwise_e_gaia_projected.npz",
               base / "qso_space_catwise_e_umap.png",
               base / "qso_space_catwise_e_gaia_features.png",
               label="CatWISE×Euclid")


if __name__ == "__main__":
    raise SystemExit(main())
