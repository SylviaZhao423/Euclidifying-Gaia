#!/usr/bin/env python3
"""
Same as train_xgb_qso_photz.py but with an extra 17th feature: the XGBoost P(gal)
score from the Gaia-based Q1 star/galaxy classifier (output/xgboost_q1).

The P(gal) is computed by running the 10 Q1 seed models (xgb_classifier_q1_seed*.json)
on the 14 Gaia astrometry/photometry features for each DESI spec QSO, fetched via:
  crossmatch_qso_desi_dr1_euclid.e_id  →  crossmatch_q1_mer_phz_vis_gaia_sep.m_object_id
QSOs with no Gaia counterpart in that crossmatch get pgal=NaN; XGBoost handles missing
values natively with tree_method=hist.

Motivation: P(gal) encodes whether Gaia "sees" this source as stellar or extended — a
different information axis from Euclid aperture photometry alone. Adding it as a feature
tests whether astrometric morphology from Gaia improves Euclid-based photo-z estimation.

Data source
-----------
DESI DR1 QSOs matched to Euclid Q1 via crossmatch_qso_desi_dr1_euclid (euclid_source='Q1').
16 Euclid aperture mags (4 bands × 4 FWHM) + pgal from Gaia Q1 classifier = 17 features.

Reads (WSDB; Astronomy/Cambridge VPN; auth via ~/.pgpass):
  {PGUSER}.crossmatch_qso_desi_dr1_euclid
  {PGUSER}.crossmatch_q1_mer_phz_vis_gaia_sep
  euclid_q1.mer_final_cat
  output/xgboost_q1/seed_models/xgb_classifier_q1_seed*.json  (local, no WSDB)
Writes:
  output/xgb_qso_photz_wp/models/xgb_zphot_seed{seed}.json
  output/xgb_qso_photz_wp/models/features.json
  output/xgb_qso_photz_wp/qso_euclid_zphot_oof.csv
  output/xgb_qso_photz_wp/qso_euclid_zphot_metrics.json
  output/xgb_qso_photz_wp/qso_euclid_zphot.png
  output/xgb_qso_photz_wp/qso_euclid_zphot_importance.png

Usage:
  PGUSER=siwei_zhao_2025 python src/train_xgb_wp_qso_photz.py
  PGUSER=siwei_zhao_2025 python src/train_xgb_wp_qso_photz.py --log1pz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_DIR     = Path(__file__).resolve().parents[1]
OUTPUT_DIR      = PROJECT_DIR / "output" / "xgb_qso_photz_wp"
MODELS_DIR      = OUTPUT_DIR / "models"
PGAL_MODELS_DIR = PROJECT_DIR / "output" / "xgboost_q1" / "seed_models"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_xgb_qso_photz import (  # noqa: E402
    train_pipeline, zphot_metrics, OUTLIER_DZ,
    ALL_FEATURES, BANDS, APERS, MER_TABLE,
)
from analyze_qso_desi_q1_color_z import (  # noqa: E402
    WSDB_HOST, WSDB_DB, WSDB_USER, _flux_to_mag, ZPC_VIS,
)
from selected_parameters_list import SELECTED_GAIA_PARAMETERS  # noqa: E402

WP_FEATURES = ALL_FEATURES + ["pgal"]


def _compute_pgal(gaia_X: np.ndarray) -> np.ndarray:
    """Score the 10 Q1 XGB classifier seeds on gaia_X; return mean P(galaxy)."""
    seed_paths = sorted(PGAL_MODELS_DIR.glob("xgb_classifier_q1_seed*.json"))
    if not seed_paths:
        print(f"  WARNING: no Q1 seed models in {PGAL_MODELS_DIR} — pgal = NaN")
        return np.full(len(gaia_X), np.nan)
    Xi = gaia_X.astype(np.float32)
    probs = []
    for p in seed_paths:
        m = xgb.XGBClassifier()
        m.load_model(str(p))
        probs.append(m.predict_proba(Xi)[:, 1])
    return np.mean(probs, axis=0)


def build_training_data(user: str) -> pd.DataFrame:
    """Pull DESI×Euclid Q1 QSOs, join mer_final_cat + Gaia features; add pgal."""
    import psycopg2
    conn = psycopg2.connect(
        host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30,
        keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5,
    )
    xmatch   = f"{user}.crossmatch_qso_desi_dr1_euclid"
    gaia_xm  = f"{user}.crossmatch_q1_mer_phz_vis_gaia_sep"
    gaia_sel = ", ".join(f"gx.g_{c} AS {c}" for c in SELECTED_GAIA_PARAMETERS)

    print(f"  reading {xmatch} (Q1, z>0) + Gaia features from {gaia_xm} ...")
    ids = pd.read_sql_query(
        f"""SELECT q.e_id::bigint AS object_id, q.qso_id, q.qso_z,
                   {gaia_sel}
            FROM {xmatch} q
            LEFT JOIN {gaia_xm} gx ON gx.m_object_id = q.e_id::bigint
            WHERE q.euclid_source = 'Q1' AND q.qso_z IS NOT NULL AND q.qso_z > 0
              AND q.e_id IS NOT NULL""",
        conn,
    ).drop_duplicates("object_id")
    n_gaia = int(ids[SELECTED_GAIA_PARAMETERS[0]].notna().sum())
    print(f"  {len(ids):,} unique Q1 QSO Euclid ids  ({n_gaia:,} with Gaia match)")

    # compute pgal from Gaia features
    gaia_X = ids[list(SELECTED_GAIA_PARAMETERS)].to_numpy(dtype=np.float32)
    n_seeds = len(list(PGAL_MODELS_DIR.glob("xgb_classifier_q1_seed*.json")))
    print(f"  scoring pgal with {n_seeds} Q1 seed models ...")
    ids["pgal"] = _compute_pgal(gaia_X)
    print(f"  pgal: mean={ids['pgal'].mean():.3f}  NaN={ids['pgal'].isna().sum():,}")

    # Euclid aperture fluxes from mer_final_cat
    fcols, ecols = [], []
    for _, s in BANDS:
        for a in APERS:
            fcols.append(f"flux_{s}_{a}fwhm_aper")
            ecols.append(f"fluxerr_{s}_{a}fwhm_aper")
    sel = ", ".join(["object_id"] + fcols + ecols)
    print(f"  querying {MER_TABLE} for all 4×4 aperture fluxes ...")
    mer = pd.read_sql_query(
        f"SELECT {sel} FROM {MER_TABLE} WHERE object_id = ANY(%(ids)s)",
        conn, params={"ids": ids["object_id"].tolist()},
    )
    conn.close()

    df = ids.merge(mer, on="object_id", how="inner")
    print(f"  matched {len(df):,} in mer_final_cat")

    for band, s in BANDS:
        zpc = ZPC_VIS if s == "vis" else 1.0
        for a in APERS:
            df[f"{band}_{a}"] = _flux_to_mag(
                df[f"flux_{s}_{a}fwhm_aper"], df[f"fluxerr_{s}_{a}fwhm_aper"], zpc
            )

    return df[["qso_id", "qso_z"] + WP_FEATURES]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log1pz", action="store_true",
                    help="predict log(1+z) internally (compresses the high-z tail)")
    args = ap.parse_args()

    user = os.getenv("PGUSER", WSDB_USER)
    df   = build_training_data(user)
    train_pipeline(df, WP_FEATURES, OUTPUT_DIR, MODELS_DIR, args.log1pz,
                   title_note="+ Gaia P(gal)")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
