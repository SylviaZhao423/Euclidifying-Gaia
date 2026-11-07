#!/usr/bin/env python3
"""
Does keeping all four apertures help the colours photo-z, or is one enough?

QSOs are point sources so a colour is ~aperture-invariant — but the 4 apertures are
independent noise realisations at different S/N, so keeping them could still help by
averaging. This weighs that (modest) precision gain against the practical cost: requiring
all 4 apertures detected shrinks the applicable sample (CatWISE 265k -> 158k). Fewer
apertures = looser detection cut = more candidates get a photo-z.

Same 2,232-QSO sample + 10-seed × 5-fold OOF CV; reuses the cached sample -> OFFLINE.

Usage:
  python src/experiment_photz_apertures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_photz_features import run, DATA_CACHE  # noqa: E402

ADJ = [("IE", "YE"), ("YE", "JE"), ("JE", "HE")]
OUT_CSV = Path(__file__).resolve().parents[1] / "output" / "photz_feature_experiment" / "apertures.csv"


def cols(apers):
    return [f"{x}_{y}_{a}" for a in apers for x, y in ADJ]


def main() -> int:
    if not DATA_CACHE.exists():
        print(f"  missing {DATA_CACHE} — run experiment_photz_features.py once first."); return 1
    df = pd.read_csv(DATA_CACHE)
    print(f"  loaded {len(df):,} from cache")

    rows = [
        run(df, cols([2]),          "colours @ 2FWHM only (3 feat)"),
        run(df, cols([1]),          "colours @ 1FWHM only (3 feat)"),
        run(df, cols([1, 2]),       "colours @ 1+2 FWHM (6 feat)"),
        run(df, cols([2, 3]),       "colours @ 2+3 FWHM (6 feat)"),
        run(df, cols([1, 2, 3, 4]), "colours @ all 4 (12 feat) [current]"),
    ]
    tab = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT_CSV, index=False)
    print("\n================= APERTURES (OOF, all z) =================")
    print(tab[["features", "n_feat", "sigma_nmad", "outlier_frac", "bias", "sigma_nmad_zle3"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n  -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
