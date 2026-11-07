#!/usr/bin/env python3
"""
Disentangle WHY flux/error (S/N) helps the Euclid QSO photo-z (see experiment_photz_features.py:
colours12+sn16 cut outliers 36% -> 27%). Is the gain from...
  (A) a single OVERALL brightness/depth scalar  -> Malmquist, selection-specific, may NOT
      transfer to CatWISE (different depth/selection); or
  (B) the PER-BAND S/N pattern (measurement quality / which bands to trust) -> generalizable.

Same 2,232-QSO sample and 10-seed × 5-fold OOF CV as the feature experiment; reuses its
pulled data cache so this runs OFFLINE. Feature sets (all start from the 12 colours):
  colours12                 : reference, no brightness at all
  colours12 + mag_mean      : + ONE overall-brightness scalar (mean of 16 mags)
  colours12 + sn_mean       : + ONE overall-depth scalar (mean of 16 S/N)
  colours12 + sn16          : + full per-band S/N (the good config)
  colours12 + sn16_rel      : + per-band S/N NORMALISED by its mean (shape only, level removed)

Read: interpret by how much of the colours12 -> colours12+sn16 gap each variant recovers.
  mag_mean/sn_mean recover most  -> gain is overall brightness (A, selection-specific).
  sn16_rel recovers most         -> gain is per-band quality shape (B, generalizable).

Writes:
  output/photz_feature_experiment/disentangle.csv

Usage:
  python src/experiment_photz_disentangle.py                 # offline (uses cached sample)
  PGUSER=siwei_zhao_2025 python src/experiment_photz_disentangle.py   # pulls if no cache
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_qso_color_vs_specz import WSDB_USER  # noqa: E402
from experiment_photz_features import build_full, run, MAGS16, COLOURS12, SN16, DATA_CACHE  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR     = PROJECT_DIR / "output" / "photz_feature_experiment"
OUT_CSV     = OUT_DIR / "disentangle.csv"


def main() -> int:
    if DATA_CACHE.exists():
        df = pd.read_csv(DATA_CACHE); print(f"  loaded {len(df):,} from cache {DATA_CACHE.name}")
    else:
        df = build_full(os.getenv("PGUSER", WSDB_USER))
        DATA_CACHE.parent.mkdir(parents=True, exist_ok=True); df.to_csv(DATA_CACHE, index=False)

    # derived brightness / quality features
    df["mag_mean"] = df[MAGS16].mean(axis=1)          # overall brightness (magnitudes)
    df["sn_mean"]  = df[SN16].mean(axis=1)            # overall depth / detection strength
    sn_rel = [f"{c}_rel" for c in SN16]
    for c in SN16:
        df[f"{c}_rel"] = df[c] / df["sn_mean"]         # per-band S/N with overall level divided out

    rows = [
        run(df, COLOURS12,                 "colours12 (reference)"),
        run(df, COLOURS12 + ["mag_mean"],  "colours12 + mag_mean (brightness)"),
        run(df, COLOURS12 + ["sn_mean"],   "colours12 + sn_mean (depth)"),
        run(df, COLOURS12 + SN16,          "colours12 + sn16 (full per-band)"),
        run(df, COLOURS12 + sn_rel,        "colours12 + sn16_rel (shape only)"),
    ]
    tab = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT_CSV, index=False)

    # fraction of the colours12 -> colours12+sn16 outlier gap each variant recovers
    ref = tab.loc[0, "outlier_frac"]; full = tab.loc[3, "outlier_frac"]; gap = ref - full
    tab["gap_recovered_%"] = (ref - tab["outlier_frac"]) / gap * 100 if gap else np.nan

    print("\n================= DISENTANGLING (OOF, all z) =================")
    print(tab[["features", "n_feat", "sigma_nmad", "outlier_frac", "bias", "gap_recovered_%"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n  -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
