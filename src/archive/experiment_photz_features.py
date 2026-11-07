#!/usr/bin/env python3
"""
Feature-set experiment for the Euclid QSO photo-z (train_xgboost_qso_photz.py).

Tests two questions raised by feature_importance.png:
  (1) Is the model leaning on ABSOLUTE brightness (magnitude<->z / Malmquist) rather than
      pure colour?  -> retrain on COLOURS ONLY and compare.
  (2) Do flux/error (S/N) indices help?  -> add them as features.

All variants use the IDENTICAL sample (the baseline all-16-mag-detected QSOs) and the same
10-seed × 5-fold z-stratified OOF CV, so the metrics are directly comparable.

Feature sets:
  mags16        : the 16 aperture mags  (VIS/Y/J/H × 1/2/3/4 FWHM)          [current baseline]
  colours12     : 12 within-aperture adjacent colours (I-Y, Y-J, J-H)×4     [no absolute scale]
  colours12+sn16: colours + 16 flux/err S/N indices
  mags16+sn16   : mags + S/N indices

Reads WSDB (VPN; ~/.pgpass). Writes:
  output/xgboost_qso_photz/feature_experiment.csv        (metrics table)
  output/xgboost_qso_photz/feature_experiment_data.csv   (pulled sample cache, --replot)

Usage:
  PGUSER=siwei_zhao_2025 python src/experiment_photz_features.py
  python src/experiment_photz_features.py --replot        # reuse cached sample
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_qso_color_vs_specz import (  # noqa: E402
    WSDB_HOST, WSDB_DB, WSDB_USER, _flux_to_mag, ZPC_VIS,
)
from train_xgboost_qso_photz import (  # noqa: E402
    BANDS, APERS, MER_TABLE, oof_predict, photz_metrics,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
XGB_DIR     = PROJECT_DIR / "output" / "xgboost_qso_photz"
DATA_CACHE  = XGB_DIR / "feature_experiment_data.csv"
OUT_CSV     = XGB_DIR / "feature_experiment.csv"

MAGS16   = [f"{b}_{a}" for a in APERS for b, _ in BANDS]
ADJ      = [("IE", "YE"), ("YE", "JE"), ("JE", "HE")]
COLOURS12 = [f"{x}_{y}_{a}" for a in APERS for x, y in ADJ]
SN16     = [f"SN_{b}_{a}" for a in APERS for b, _ in BANDS]


def build_full(user: str) -> pd.DataFrame:
    """DESI×Euclid Q1 QSOs -> 16 mags + 12 colours + 16 S/N, on the all-16-mag sample."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30,
                            keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5)
    ids = pd.read_sql_query(
        """SELECT e_id::bigint AS object_id, qso_id, qso_z
           FROM {u}.crossmatch_qso_desi_dr1_euclid
           WHERE euclid_source='Q1' AND qso_z IS NOT NULL AND qso_z>0 AND e_id IS NOT NULL"""
        .format(u=user), conn).drop_duplicates("object_id")
    fcols = [f"flux_{s}_{a}fwhm_aper" for _, s in BANDS for a in APERS]
    ecols = [f"fluxerr_{s}_{a}fwhm_aper" for _, s in BANDS for a in APERS]
    mer = pd.read_sql_query(
        f"SELECT object_id, {', '.join(fcols + ecols)} FROM {MER_TABLE} WHERE object_id = ANY(%(i)s)",
        conn, params={"i": ids["object_id"].tolist()})
    conn.close()
    df = ids.merge(mer, on="object_id", how="inner")

    for b, s in BANDS:
        zpc = ZPC_VIS if s == "vis" else 1.0
        for a in APERS:
            f, e = df[f"flux_{s}_{a}fwhm_aper"], df[f"fluxerr_{s}_{a}fwhm_aper"]
            df[f"{b}_{a}"] = _flux_to_mag(f, e, zpc)
            with np.errstate(divide="ignore", invalid="ignore"):
                df[f"SN_{b}_{a}"] = pd.to_numeric(f, errors="coerce") / pd.to_numeric(e, errors="coerce")
    for a in APERS:
        for x, y in ADJ:
            df[f"{x}_{y}_{a}"] = df[f"{x}_{a}"] - df[f"{y}_{a}"]

    df = df.dropna(subset=MAGS16 + ["qso_z"]).reset_index(drop=True)   # baseline sample
    print(f"  {len(df):,} QSOs (all 16 mags detected)")
    return df[["qso_id", "qso_z"] + MAGS16 + COLOURS12 + SN16]


def run(df: pd.DataFrame, feats: list, name: str) -> dict:
    zs = df["qso_z"].to_numpy(float)
    X  = df[feats].to_numpy(float)
    print(f"\n=== {name}  ({len(feats)} features) ===")
    zp  = oof_predict(X, zs, False, zs)
    met = photz_metrics(zs, zp)
    m3  = photz_metrics(zs[zs <= 3], zp[zs <= 3])
    print(f"  sigma_NMAD={met['sigma_nmad']:.3f}  outliers={100*met['outlier_frac']:.1f}%  "
          f"bias={met['bias']:+.3f}  sigma(z<=3)={m3['sigma_nmad']:.3f}")
    return {"features": name, "n_feat": len(feats), "N": met["N"],
            "sigma_nmad": met["sigma_nmad"], "outlier_frac": met["outlier_frac"],
            "bias": met["bias"], "sigma_nmad_zle3": m3["sigma_nmad"],
            "outlier_frac_zle3": m3["outlier_frac"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true", help="reuse cached sample (no WSDB).")
    args = ap.parse_args()

    if args.replot and DATA_CACHE.exists():
        df = pd.read_csv(DATA_CACHE); print(f"  loaded {len(df):,} from cache")
    else:
        df = build_full(os.getenv("PGUSER", WSDB_USER))
        DATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_CACHE, index=False)

    rows = [
        run(df, MAGS16,               "mags16 (baseline)"),
        run(df, COLOURS12,            "colours12 (no abs scale)"),
        run(df, COLOURS12 + SN16,     "colours12 + sn16"),
        run(df, MAGS16 + SN16,        "mags16 + sn16"),
    ]
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT_CSV, index=False)
    print("\n================= SUMMARY (OOF, all z) =================")
    print(tab[["features", "n_feat", "sigma_nmad", "outlier_frac", "bias", "sigma_nmad_zle3"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n  -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
