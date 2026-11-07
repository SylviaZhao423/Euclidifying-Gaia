#!/usr/bin/env python3
"""
Feature-importance ranks for the experimental feature sets (companion to
experiment_photz_features.py / _disentangle.py, which only produced METRICS).

For each feature set it fits the 10 seed models on the full sample and makes the same
gain-importance stability plot + CSV as train_xgboost_qso_photz.py. Reuses the cached
sample so it runs OFFLINE.

Feature sets:
  colours12            : 12 within-aperture adjacent colours (the transfer-safe candidate)
  colours12_mag_mean   : colours + one overall-brightness scalar (best metrics)
  colours12_sn16       : colours + full per-band S/N

Writes (all in output/photz_feature_experiment/):
  feature_importance_<set>.png  (+ .csv)
  models_<set>/xgb_photz_seed*.json

Usage:
  python src/experiment_photz_importance.py                 # offline (cached sample)
  PGUSER=siwei_zhao_2025 python src/experiment_photz_importance.py   # pulls if no cache
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_qso_color_vs_specz import WSDB_USER  # noqa: E402
from experiment_photz_features import build_full, MAGS16, COLOURS12, SN16, DATA_CACHE  # noqa: E402
from train_xgboost_qso_photz import fit_final_models, feature_importance_plot  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR     = PROJECT_DIR / "output" / "photz_feature_experiment"


def main() -> int:
    if DATA_CACHE.exists():
        df = pd.read_csv(DATA_CACHE); print(f"  loaded {len(df):,} from cache {DATA_CACHE.name}")
    else:
        df = build_full(os.getenv("PGUSER", WSDB_USER))
        DATA_CACHE.parent.mkdir(parents=True, exist_ok=True); df.to_csv(DATA_CACHE, index=False)

    df["mag_mean"] = df[MAGS16].mean(axis=1)

    sets = {
        "colours12":          COLOURS12,
        "colours12_mag_mean": COLOURS12 + ["mag_mean"],
        "colours12_sn16":     COLOURS12 + SN16,
    }
    zs = df["qso_z"].to_numpy(float)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, feats in sets.items():
        print(f"\n=== {name}  ({len(feats)} features) ===")
        mdir = OUT_DIR / f"models_{name}"
        X = df[feats].to_numpy(float)
        fit_final_models(X, zs, False, mdir, feats)
        out_png = OUT_DIR / f"feature_importance_{name}.png"
        feature_importance_plot(mdir, feats, out_png)
        top = pd.read_csv(out_png.with_suffix(".csv")).head(6)
        print(top[["rank", "feature", "importance_mean", "importance_cv"]].to_string(index=False))
    print(f"\n  -> {OUT_DIR}/feature_importance_*.png (+ .csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
