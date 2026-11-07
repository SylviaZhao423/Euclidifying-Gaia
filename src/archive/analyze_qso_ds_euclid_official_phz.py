#!/usr/bin/env python3
"""
Euclid OFFICIAL Q1 PHZ photo-z vs DESI/SDSS spectroscopic z, for the spec QSOs.

The Q1 PHZ pipeline (NNPZ) publishes a photo-z (catalogue.phz_photo_z: PHZ_MEDIAN,
PHZ_MODE_1, ...) that is NOT loaded in WSDB. Q1 is a PUBLIC release, so we fetch photo-z for
ONLY our ~2,835 spec QSOs straight from the ESA archive TAP (anonymous, no login, no web
form) via astroquery.esa.euclid — chunked object_id IN(...) queries — and compare Euclid's
official photo-z against the DESI/SDSS spectroscopic z, scored the SAME way as our XGB
(train_xgboost_qso_photz.py). This tells us whether our 4-band Euclid-only regressor
(σ_NMAD~0.13, ~30% outliers) is near the ceiling of Euclid photometry or leaving a lot on
the table (Euclid's photo-z also uses the external u/g/r/i/z bands).

Reads : data/spec_qso_object_ids.csv           (object_id, qso_catalog, qso_id, qso_z)
        data/phz_photoz_specqso.csv            (TAP cache; auto-fetched if absent)
        or any data/phz_photoz_specqso*.vot    (a manually-downloaded VOTable, if present)
Writes: data/phz_photoz_specqso.csv            (TAP fetch cache)
        output/qso_zphot_official/qso_euclid_official_phz.png
        output/qso_zphot_official/qso_euclid_official_phz_metrics.json

Usage:
  python src/analyze_qso_ds_euclid_official_phz.py            # fetch (or use cache) + compare
  python src/analyze_qso_ds_euclid_official_phz.py --refetch  # force a fresh TAP fetch
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_xgboost_qso_photz import photz_metrics, OUTLIER_DZ  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
IDS_CSV   = PROJECT_DIR / "data" / "spec_qso_object_ids.csv"
PHZ_CACHE = PROJECT_DIR / "data" / "phz_photoz_specqso.csv"
VOT_GLOB  = str(PROJECT_DIR / "data" / "phz_photoz_specqso*.vot")
OUT_DIR   = PROJECT_DIR / "output" / "qso_zphot_official"
PNG_PATH  = OUT_DIR / "qso_euclid_official_phz.png"
JSON_PATH = OUT_DIR / "qso_euclid_official_phz_metrics.json"

PHZ_TABLE = "catalogue.phz_photo_z"
COLS = "object_id, phz_median, phz_mode_1"


def _fetch_tap(object_ids: list[int], chunk: int = 500) -> pd.DataFrame:
    """Anonymous chunked TAP fetch of phz_photo_z for the given object_ids (public Q1)."""
    from astroquery.esa.euclid import Euclid
    frames = []
    for i in range(0, len(object_ids), chunk):
        inlist = ",".join(map(str, object_ids[i:i + chunk]))
        q = f"SELECT {COLS} FROM {PHZ_TABLE} WHERE object_id IN ({inlist})"
        r = Euclid.launch_job(q).get_results().to_pandas()
        frames.append(r)
        print(f"    chunk {i//chunk + 1}: {len(r)} rows", flush=True)
    return pd.concat(frames, ignore_index=True).drop_duplicates("object_id")


def load_phz(ids: pd.DataFrame, refetch: bool) -> pd.DataFrame:
    """Get phz_photo_z for our ids: CSV cache -> manual VOTable(s) -> anonymous TAP fetch."""
    vots = sorted(glob.glob(VOT_GLOB))
    if PHZ_CACHE.exists() and not refetch:
        print(f"  using cache {PHZ_CACHE.name}")
        return pd.read_csv(PHZ_CACHE)
    if vots and not refetch:
        from astropy.io import votable
        frames = []
        for f in vots:
            t = votable.parse(f).get_first_table().to_table().to_pandas()
            t.columns = [c.lower() for c in t.columns]
            frames.append(t); print(f"  read {f} ({len(t):,})")
        df = pd.concat(frames, ignore_index=True).drop_duplicates("object_id")
        df.to_csv(PHZ_CACHE, index=False)
        return df
    print(f"  fetching {PHZ_TABLE} via anonymous ESA TAP for {len(ids):,} object_ids ...")
    df = _fetch_tap(ids["object_id"].astype("int64").tolist())
    df.to_csv(PHZ_CACHE, index=False)
    print(f"  -> {PHZ_CACHE} ({len(df):,} rows)")
    return df


def _panel(ax, zs, zp, title):
    ax.scatter(zs, zp, s=9, alpha=0.4, lw=0, color="#4363d8", rasterized=True)
    zl = np.linspace(0, max(zs.max(), zp.max()) * 1.02, 100)
    ax.plot(zl, zl, "k-", lw=0.9)
    ax.plot(zl, zl + OUTLIER_DZ * (1 + zl), "r--", lw=0.9, label=f"|Δz|={OUTLIER_DZ}")
    ax.plot(zl, zl - OUTLIER_DZ * (1 + zl), "r--", lw=0.9)
    m = photz_metrics(zs, zp)
    m3 = photz_metrics(zs[zs <= 3], zp[zs <= 3])
    ax.text(0.03, 0.97,
            f"N={m['N']:,}\nσ_NMAD={m['sigma_nmad']:.3f}\noutliers={100*m['outlier_frac']:.1f}%\n"
            f"bias={m['bias']:+.3f}\n—— z≤3 ——\nσ_NMAD={m3['sigma_nmad']:.3f}\n"
            f"outliers={100*m3['outlier_frac']:.1f}%",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    ax.set_xlabel("DESI/SDSS spectroscopic $z$"); ax.set_ylabel("Euclid PHZ photo-$z$")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, zl.max()); ax.set_ylim(0, zl.max())
    return m, m3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refetch", action="store_true", help="force a fresh TAP fetch (ignore cache).")
    args = ap.parse_args()

    ids = pd.read_csv(IDS_CSV)
    phz = load_phz(ids, args.refetch)
    df = ids.merge(phz, on="object_id", how="inner")
    df = df[df["qso_z"] > 0].dropna(subset=["phz_median"]).reset_index(drop=True)
    print(f"  {len(df):,} spec QSOs matched to an official Euclid photo-z "
          f"({len(phz):,} fetched, {len(ids):,} requested)")

    estimators = [("phz_median", "PHZ median")]
    if "phz_mode_1" in df.columns and df["phz_mode_1"].notna().any():
        estimators.append(("phz_mode_1", "PHZ mode 1"))

    fig, axes = plt.subplots(1, len(estimators), figsize=(6.0 * len(estimators), 5.6), squeeze=False)
    metrics = {}
    for ax, (col, lbl) in zip(axes[0], estimators):
        sub = df.dropna(subset=[col])
        m, m3 = _panel(ax, sub["qso_z"].to_numpy(float), sub[col].to_numpy(float),
                       f"Euclid official {lbl}  vs  spec-$z$")
        metrics[col] = {"all": m, "z_le_3": m3}
        print(f"  {lbl:12}: N={m['N']:,}  σ_NMAD={m['sigma_nmad']:.3f}  "
              f"outliers={100*m['outlier_frac']:.1f}%  bias={m['bias']:+.3f}")

    fig.suptitle(f"Euclid Q1 OFFICIAL PHZ photo-z vs DESI/SDSS spec-z — {len(df):,} spec QSOs\n"
                 f"(compare to our 4-band XGB: σ_NMAD~0.13, ~30% outliers)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=130, bbox_inches="tight"); plt.close(fig)
    JSON_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"  -> {PNG_PATH}\n  -> {JSON_PATH}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
