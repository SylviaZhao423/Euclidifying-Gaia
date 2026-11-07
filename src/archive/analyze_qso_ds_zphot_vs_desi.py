#!/usr/bin/env python3
"""
Head-to-head: OUR Euclid 4-band XGB photo-z vs EUCLID's OFFICIAL PHZ photo-z, both against
the DESI spectroscopic redshift, on the SAME matched set of spec QSOs.

This is the direct "how do they compare with the DESI z" figure: same objects, same axes,
same metrics (train_xgb_qso_photz.zphot_metrics), so the two estimators can be read side
by side against the ground truth.

Reads : data/spec_qso_object_ids.csv                   (object_id, qso_id, qso_z, qso_catalog)
        data/phz_photoz_specqso.csv                    (Euclid official phz_median; from TAP)
        output/qso_zphot/qso_euclid_zphot_oof.csv      (our colour-XGB OOF z_phot_oof, keyed qso_id)
Writes: output/qso_zphot_official/zphot_vs_desi_headtohead.png
        output/qso_zphot_official/zphot_vs_desi_headtohead_metrics.json

Usage: python src/analyze_qso_ds_zphot_vs_desi.py [--desi-only]
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
from train_xgb_qso_photz import zphot_metrics, OUTLIER_DZ  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
IDS_CSV   = PROJECT_DIR / "data" / "spec_qso_object_ids.csv"
PHZ_CSV   = PROJECT_DIR / "data" / "phz_photoz_specqso.csv"
OOF_CSV   = PROJECT_DIR / "output" / "xgb_qso_photz" / "qso_euclid_zphot_oof.csv"
OUT_DIR   = PROJECT_DIR / "output" / "qso_zphot_official"
PNG_PATH  = OUT_DIR / "zphot_vs_desi_headtohead.png"
JSON_PATH = OUT_DIR / "zphot_vs_desi_headtohead_metrics.json"


def _panel(ax, zs, zp, title, color):
    ax.scatter(zs, zp, s=10, alpha=0.4, lw=0, color=color, rasterized=True)
    zl = np.linspace(0, max(zs.max(), zp.max()) * 1.02, 100)
    ax.plot(zl, zl, "k-", lw=0.9)
    ax.plot(zl, zl + OUTLIER_DZ * (1 + zl), "r--", lw=0.9, label=f"|Δz|={OUTLIER_DZ}")
    ax.plot(zl, zl - OUTLIER_DZ * (1 + zl), "r--", lw=0.9)
    m = zphot_metrics(zs, zp); m3 = zphot_metrics(zs[zs <= 3], zp[zs <= 3])
    ax.text(0.03, 0.97,
            f"N={m['N']:,}\nσ_NMAD={m['sigma_nmad']:.3f}\noutliers={100*m['outlier_frac']:.1f}%\n"
            f"bias={m['bias']:+.3f}\n—— z≤3 ——\nσ_NMAD={m3['sigma_nmad']:.3f}\n"
            f"outliers={100*m3['outlier_frac']:.1f}%",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    ax.set_xlabel("DESI/SDSS spectroscopic $z$"); ax.set_ylabel("photo-$z$")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, zl.max()); ax.set_ylim(0, zl.max())
    return m, m3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desi-only", action="store_true", help="drop the handful of SDSS QSOs.")
    args = ap.parse_args()

    ids = pd.read_csv(IDS_CSV)
    if args.desi_only:
        ids = ids[ids["qso_catalog"].str.startswith("DESI")]
    phz = pd.read_csv(PHZ_CSV)[["object_id", "phz_median"]]
    oof = pd.read_csv(OOF_CSV)[["qso_id", "z_phot_oof"]]

    # align both estimators to the SAME objects: ids carries qso_id<->object_id
    df = (ids.merge(oof, on="qso_id", how="inner")
             .merge(phz, on="object_id", how="inner"))
    df = df[df["qso_z"] > 0].dropna(subset=["z_phot_oof", "phz_median"]).reset_index(drop=True)
    print(f"  {len(df):,} spec QSOs with BOTH our XGB z and Euclid official phz (matched)")

    zs = df["qso_z"].to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6))
    metrics = {}
    metrics["our_xgb"] = dict(zip(("all", "z_le_3"),
        _panel(axes[0], zs, df["z_phot_oof"].to_numpy(float),
               "OUR 4-band XGB  vs  DESI $z$", "#4363d8")))
    metrics["euclid_official"] = dict(zip(("all", "z_le_3"),
        _panel(axes[1], zs, df["phz_median"].to_numpy(float),
               "Euclid OFFICIAL PHZ  vs  DESI $z$", "#e6194b")))

    fig.suptitle(f"Euclid QSO photo-z — ours vs Euclid's official — on the SAME {len(df):,} spec QSOs "
                 f"(matched to DESI/SDSS $z$)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=130, bbox_inches="tight"); plt.close(fig)
    JSON_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"\n  {'estimator':22}{'σ_NMAD':>9}{'outliers%':>11}{'bias':>9}")
    for k, lbl in [("our_xgb", "our 4-band XGB"), ("euclid_official", "Euclid official PHZ")]:
        m = metrics[k]["all"]
        print(f"  {lbl:22}{m['sigma_nmad']:>9.3f}{100*m['outlier_frac']:>10.1f}%{m['bias']:>+9.3f}")
    print(f"  -> {PNG_PATH}\n  -> {JSON_PATH}\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
