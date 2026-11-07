#!/usr/bin/env python3
"""Apply XGBoost photo-z to Euclid deblend siblings and plot their colour rivers.

Reads output/qso_color_vs_z/arrays/deblend_sibling_photometry.csv (produced by
extract_qso_deblend_sibling_photometry.py), applies the 10 seed photoz models,
and draws the Euclid colour(z) rivers overlaid on the DESI spec-z QSO locus.

The goal: see whether deblend siblings fall on the QSO colour locus, making
them candidates for the companion in a QSO pair.

Writes:
  output/qso_color_vs_z/arrays/deblend_sibling_xgbz.csv
  output/qso_color_vs_z/deblend_sibling_color_vs_photz.png

Usage:
  python src/apply_photz_deblend_siblings.py
  python src/apply_photz_deblend_siblings.py --replot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_qso_color_vs_specz import COLORS, ZMAX_PLOT       # noqa: E402
from analyze_qso_color_vs_photz  import _draw_row, _spec_rivers  # noqa: E402
from train_xgboost_qso_photz     import add_colours              # noqa: E402

PROJECT_DIR  = Path(__file__).resolve().parents[1]
XGB_DIR      = PROJECT_DIR / "output" / "xgboost_qso_photz"
MODELS_DIR   = XGB_DIR / "models"
ARRAYS_DIR   = PROJECT_DIR / "output" / "qso_color_vs_z" / "arrays"
PHOT_CSV     = ARRAYS_DIR / "deblend_sibling_photometry.csv"
CACHE_XGBZ   = ARRAYS_DIR / "deblend_sibling_xgbz.csv"
POP_CSV      = ARRAYS_DIR / "rivers_pop_2fwhm.csv"
SPEC_CSV     = ARRAYS_DIR / "rivers_qso_2fwhm.csv"
PNG_OUT      = PROJECT_DIR / "output" / "qso_color_vs_z" / "deblend_sibling_color_vs_photz.png"


def apply_models(df: pd.DataFrame) -> pd.DataFrame:
    import xgboost as xgb
    feats  = json.loads((MODELS_DIR / "features.json").read_text())["features"]
    target = json.loads((XGB_DIR    / "metrics.json" ).read_text())["target"]

    df = df.dropna(subset=feats).reset_index(drop=True)
    print(f"  {len(df):,} siblings with all bands detected (in-domain)")

    X = df[feats].to_numpy(float)
    preds = []
    for p in sorted(MODELS_DIR.glob("xgb_photz_seed*.json")):
        m = xgb.XGBRegressor(); m.load_model(str(p))
        pr = m.predict(X)
        preds.append(np.expm1(pr) if target == "log1p_z" else pr)
    df["z_xgb"] = np.mean(preds, axis=0)
    print(f"  applied {len(preds)} seed models  (target={target})")

    for _, b, r in COLORS:
        df[f"{b}_{r}"] = df[f"{b}_2"] - df[f"{r}_2"]
    return df


def make_plot(df: pd.DataFrame, pop: pd.DataFrame, spec_rivers: dict, out: Path) -> None:
    ylim = (0.0, ZMAX_PLOT)
    ncol = 1 + len(COLORS)
    fig, axes = plt.subplots(1, ncol, figsize=(6.8 * ncol, 6.8))
    _draw_row(fig, axes, df, "z_xgb", "XGBoost photo-$z$",
              pop, ylim, show_poplegend=True, spec_rivers=spec_rivers)
    fig.suptitle(
        f"Euclid deblend siblings: colour vs XGBoost photo-z  ·  2×FWHM aperture photometry\n"
        f"N={len(df):,} in-domain siblings  (median sep={df['pair_sep_arcsec'].median():.2f}\")  "
        f"—  dashed = DESI spec-z QSO median (known locus)",
        fontsize=14, fontweight="bold", y=1.01)
    try:
        fig.tight_layout()
    except Exception:
        fig.subplots_adjust(wspace=0.4)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replot", action="store_true",
                    help="redraw from cache only (no model inference)")
    args = ap.parse_args()

    if args.replot and not CACHE_XGBZ.exists():
        print(f"--replot needs {CACHE_XGBZ} — run once without it first."); return 1

    if args.replot:
        df = pd.read_csv(CACHE_XGBZ)
        print(f"  loaded {len(df):,} sibling XGB predictions from cache")
    else:
        phot = pd.read_csv(PHOT_CSV)
        print(f"Loaded {len(phot):,} sibling photometry rows")
        df = apply_models(phot)
        keep = (["sibling_e_id", "primary_e_id", "primary_gaia_id",
                 "pair_sep_arcsec", "z_xgb"]
                + [f"{b}_{r}" for _, b, r in COLORS])
        keep = [c for c in keep if c in df.columns]
        CACHE_XGBZ.parent.mkdir(parents=True, exist_ok=True)
        df[keep].to_csv(CACHE_XGBZ, index=False)
        print(f"  cached -> {CACHE_XGBZ}")
        df = df[keep]

    # Load cached population and spec rivers (no WSDB needed)
    pop = pd.read_csv(POP_CSV) if POP_CSV.exists() else pd.DataFrame()
    spec_df = pd.read_csv(SPEC_CSV) if SPEC_CSV.exists() else pd.DataFrame()
    spec_rivers = _spec_rivers(spec_df) if not spec_df.empty else {}
    print(f"  pop={len(pop):,}  spec_rivers={len(spec_rivers)} colours")

    make_plot(df, pop, spec_rivers, PNG_OUT)

    # Quick summary
    print(f"\n{'='*60}")
    print(f"Deblend sibling photo-z summary")
    print(f"  In-domain siblings    : {len(df):,}")
    print(f"  Median z_xgb          : {df['z_xgb'].median():.3f}")
    print(f"  z_xgb < 0.5           : {(df['z_xgb'] < 0.5).sum():,} ({100*(df['z_xgb']<0.5).mean():.1f}%)")
    print(f"  0.5 ≤ z_xgb < 1.5    : {((df['z_xgb']>=0.5)&(df['z_xgb']<1.5)).sum():,}")
    print(f"  z_xgb ≥ 1.5          : {(df['z_xgb']>=1.5).sum():,}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
