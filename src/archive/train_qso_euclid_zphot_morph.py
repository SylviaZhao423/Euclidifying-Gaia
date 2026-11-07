#!/usr/bin/env python3
"""
STEP 2 (variant): Euclid photo-z regressor that ADDS the morphology feature
`point_like_prob` on top of the 3 colours + H_E.

Ablation (experiment_qso_zphot_features.py, 10-seed OOF) showed point_like_prob is the
one extra feature COMMON to both Euclid tables (Q1 point_like_prob / ERO v_class_star,
standardised as e_point_like_prob) that meaningfully helps — it drops catastrophic
outliers 31.4% -> 29.1% (σ_NMAD 0.141 -> 0.132). It works because at low z the QSO host
is resolved (less point-like), a weak redshift proxy. (compactness was marginal, fwhm
useless due to Q1-arcsec vs ERO-pixel inhomogeneity — both excluded.)

This is a SEPARATE model from the colour-only train_xgb_qso_photz.py: it saves to its
own folder so the two can be compared side by side. Both write a feature-importance plot.

Features : IE_YE, YE_JE, JE_HE, HE, point_like_prob
Reads    : {PGUSER}.crossmatch_qso_desi_dr1_euclid + crossmatch_qso_sdss_dr16_euclid (WSDB, VPN)
Writes   : output/qso_zphot_morph/
             models/xgb_zphot_seed*.json (+ features.json)
             qso_euclid_zphot.png                (z_pred vs z_spec + residuals)
             qso_euclid_zphot_importance.png     (gain importance, 10-seed mean)
             qso_euclid_zphot_oof.csv, qso_euclid_zphot_metrics.json

Usage:
  PGUSER=siwei_zhao_2025 python src/train_qso_euclid_zphot_morph.py
  PGUSER=siwei_zhao_2025 python src/train_qso_euclid_zphot_morph.py --log1pz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_qso_desi_q1_color_z import (COLORS, WSDB_HOST, WSDB_DB, WSDB_USER,  # noqa: E402
                                        _flux_to_mag, ZPC_VIS)
from train_xgb_qso_photz import train_pipeline, COLOR_FEATURES, MAG_ANCHOR  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "output" / "qso_zphot_morph"
MODELS_DIR = OUTPUT_DIR / "models"

FEATURES = COLOR_FEATURES + [MAG_ANCHOR, "point_like_prob"]   # 3 colours + H_E + morphology
TABLES = ["crossmatch_qso_desi_dr1_euclid", "crossmatch_qso_sdss_dr16_euclid"]


def pull(user: str) -> pd.DataFrame:
    """Both spec-QSO × Euclid tables with the 4 fluxes + point_like_prob → colours + morph."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30,
                            keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5)
    frames = []
    for t in TABLES:
        print(f"  reading {user}.{t} ...")
        frames.append(pd.read_sql_query(
            f"""SELECT qso_z, e_flux_vis, e_fluxerr_vis, e_flux_y, e_fluxerr_y,
                       e_flux_j, e_fluxerr_j, e_flux_h, e_fluxerr_h, e_point_like_prob
                FROM {user}.{t} WHERE qso_z IS NOT NULL AND qso_z > 0
                  AND euclid_source = 'Q1'""", conn))   # Q1-only: ERO flux is raw ADU
    conn.close()
    d = pd.concat(frames, ignore_index=True)
    d["IE"] = _flux_to_mag(d["e_flux_vis"], d["e_fluxerr_vis"], ZPC_VIS)
    d["YE"] = _flux_to_mag(d["e_flux_y"], d["e_fluxerr_y"])
    d["JE"] = _flux_to_mag(d["e_flux_j"], d["e_fluxerr_j"])
    d["HE"] = _flux_to_mag(d["e_flux_h"], d["e_fluxerr_h"])
    for _, b, r in COLORS:
        d[f"{b}_{r}"] = d[b] - d[r]
    d["point_like_prob"] = pd.to_numeric(d["e_point_like_prob"], errors="coerce")
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log1pz", action="store_true",
                    help="predict log(1+z) internally (compresses the high-z tail)")
    args = ap.parse_args()
    df = pull(os.getenv("PGUSER", WSDB_USER))
    train_pipeline(df, FEATURES, OUTPUT_DIR, MODELS_DIR, args.log1pz)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
