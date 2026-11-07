#!/usr/bin/env python3
"""
Aperture-size comparison for the spec-QSO Euclid photometry.

The colour pipeline uses the 2-FWHM aperture (builder_qso_euclid_crossmatch `_ECOLS`).
This script pulls ALL FOUR aperture sizes (1/2/3/4 FWHM) for every band directly from
euclid_q1.mer_final_cat, joined by Euclid object_id to our Q1 spec QSOs, so we can SEE
how the choice of aperture shifts the magnitude and colour distributions.

Expectation: on PSF-matched images a larger aperture captures more total light (brighter
mag) — a real per-band shift — but for point-like QSOs that shift is nearly the SAME in
every band, so it largely CANCELS in colours. This plot tests exactly that: if the four
colour histograms overlap, the 2-FWHM choice is harmless for colours.

Join key : e_id (crossmatch tables) = object_id (mer_final_cat), Q1 rows only.
Reads    : {PGUSER}.crossmatch_qso_desi_dr1_euclid + crossmatch_qso_sdss_dr16_euclid
           euclid_q1.mer_final_cat  (flux_{vis,y,j,h}_{1,2,3,4}fwhm_aper + errors)
Writes   : output/qso_desi_q1_color_z/qso_desi_q1_aperture_fluxes.png
           output/qso_desi_q1_color_z/arrays/qso_desi_q1_aperture_fluxes.csv  (cache)

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_qso_desi_q1_aperture_fluxes.py
  PGUSER=siwei_zhao_2025 python src/analyze_qso_desi_q1_aperture_fluxes.py --cached  # replot
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_qso_desi_q1_color_z import (WSDB_HOST, WSDB_DB, WSDB_USER,  # noqa: E402
                                        _flux_to_mag, ZPC_VIS)

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_DIR / "output" / "qso_desi_q1_color_z"
ARR_DIR = OUT_DIR / "arrays"
CSV_PATH = ARR_DIR / "qso_desi_q1_aperture_fluxes.csv"
PNG_PATH = OUT_DIR / "qso_desi_q1_aperture_fluxes.png"

TABLES = ["crossmatch_qso_desi_dr1_euclid", "crossmatch_qso_sdss_dr16_euclid"]
APERS = [1, 2, 3, 4]
BANDS = [("VIS", "vis"), ("Y", "y"), ("J", "j"), ("H", "h")]     # label, db suffix
COLOR_DEFS = [("$I_E-Y_E$", "vis", "y"), ("$Y_E-J_E$", "y", "j"), ("$J_E-H_E$", "j", "h")]
APER_COLORS = plt.cm.viridis(np.linspace(0.12, 0.9, len(APERS)))  # 1FWHM light -> 4FWHM dark


def pull(user: str) -> pd.DataFrame:
    """Q1 spec-QSO Euclid ids + z, joined to mer_final_cat for all 4 aperture fluxes."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30,
                            keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5)
    # 1) Q1 spec-QSO Euclid ids + redshift
    frames = []
    for t in TABLES:
        print(f"  reading {user}.{t} ...")
        frames.append(pd.read_sql_query(
            f"""SELECT e_id::bigint AS object_id, qso_z
                FROM {user}.{t}
                WHERE euclid_source = 'Q1' AND qso_z IS NOT NULL AND qso_z > 0
                  AND e_id IS NOT NULL""", conn))
    ids = pd.concat(frames, ignore_index=True).drop_duplicates("object_id")
    print(f"  {len(ids):,} unique Q1 QSO Euclid ids")

    # 2) all aperture fluxes from mer_final_cat for those ids
    flux_cols = [f"flux_{s}_{a}fwhm_aper" for _, s in BANDS for a in APERS]
    err_cols = [f"fluxerr_{s}_{a}fwhm_aper" for _, s in BANDS for a in APERS]
    sel = ", ".join(["object_id"] + flux_cols + err_cols)
    id_list = ids["object_id"].tolist()
    print(f"  querying mer_final_cat for {len(flux_cols)} flux + {len(err_cols)} err columns ...")
    mer = pd.read_sql_query(
        f"SELECT {sel} FROM euclid_q1.mer_final_cat WHERE object_id = ANY(%(ids)s)",
        conn, params={"ids": id_list})
    conn.close()
    print(f"  matched {len(mer):,} rows in mer_final_cat")

    df = ids.merge(mer, on="object_id", how="inner")
    # flux -> AB mag per band/aperture (ZPC_VIS on VIS only, matching the pipeline)
    for lbl, s in BANDS:
        zpc = ZPC_VIS if s == "vis" else 1.0
        for a in APERS:
            df[f"mag_{s}_{a}"] = _flux_to_mag(df[f"flux_{s}_{a}fwhm_aper"],
                                              df[f"fluxerr_{s}_{a}fwhm_aper"], zpc)
    # colours per aperture
    for _, bl, rl in COLOR_DEFS:
        for a in APERS:
            df[f"col_{bl}_{rl}_{a}"] = df[f"mag_{bl}_{a}"] - df[f"mag_{rl}_{a}"]
    return df


def make_plot(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 8.2))

    # ---- Row 0: magnitude distribution per band, one histogram per aperture ----
    for k, (lbl, s) in enumerate(BANDS):
        ax = axes[0, k]
        allmags = np.concatenate([df[f"mag_{s}_{a}"].dropna().to_numpy() for a in APERS])
        lo, hi = np.nanpercentile(allmags, [1, 99])
        bins = np.linspace(lo, hi, 45)
        for a, c in zip(APERS, APER_COLORS):
            m = df[f"mag_{s}_{a}"].dropna().to_numpy()
            ax.hist(m, bins=bins, histtype="step", lw=1.8, color=c,
                    label=f"{a} FWHM (med {np.median(m):.2f})")
        ax.set_xlabel(f"${lbl}$ AB mag"); ax.set_ylabel("count" if k == 0 else "")
        ax.set_title(f"${lbl}$ band — aperture magnitudes")
        ax.legend(fontsize=7.5, frameon=False)

    # ---- Row 1 (first 3): colour distribution per aperture ----
    for k, (clbl, bl, rl) in enumerate(COLOR_DEFS):
        ax = axes[1, k]
        allc = np.concatenate([df[f"col_{bl}_{rl}_{a}"].dropna().to_numpy() for a in APERS])
        lo, hi = np.nanpercentile(allc, [1, 99])
        bins = np.linspace(lo, hi, 45)
        for a, c in zip(APERS, APER_COLORS):
            cc = df[f"col_{bl}_{rl}_{a}"].dropna().to_numpy()
            ax.hist(cc, bins=bins, histtype="step", lw=1.8, color=c,
                    label=f"{a} FWHM (med {np.median(cc):.3f})")
        ax.set_xlabel(f"{clbl}"); ax.set_ylabel("count" if k == 0 else "")
        ax.set_title(f"{clbl} — vs aperture")
        ax.legend(fontsize=7.5, frameon=False)

    # ---- Row 1 (last): aperture growth curve — median mag offset vs 2FWHM per band ----
    ax = axes[1, 3]
    for lbl, s in BANDS:
        med = [np.nanmedian(df[f"mag_{s}_{a}"]) for a in APERS]
        med = np.array(med) - med[1]        # offset relative to 2 FWHM (the pipeline choice)
        ax.plot(APERS, med, "o-", lw=1.8, label=f"${lbl}$")
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.axvline(2, color="0.6", lw=0.8, ls=":")
    ax.set_xticks(APERS)
    ax.set_xlabel("aperture size (FWHM)")
    ax.set_ylabel(r"median $\Delta$mag  vs 2 FWHM")
    ax.set_title("aperture growth (brighter with size)")
    ax.legend(fontsize=8, frameon=False)
    ax.invert_yaxis()                        # brighter (more light) upward

    fig.suptitle(f"Euclid Q1 aperture photometry of {len(df):,} spec QSOs — "
                 f"1/2/3/4 FWHM diameter (PSF-matched)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  -> {out}")


def _print_summary(df: pd.DataFrame) -> None:
    print("\n  median AB mag by band x aperture:")
    print(f"    {'band':5}" + "".join(f"{a}FWHM".rjust(9) for a in APERS))
    for lbl, s in BANDS:
        print(f"    {lbl:5}" + "".join(f"{np.nanmedian(df[f'mag_{s}_{a}']):9.3f}" for a in APERS))
    print("\n  median colour by aperture (should barely move if aperture cancels):")
    print(f"    {'colour':10}" + "".join(f"{a}FWHM".rjust(9) for a in APERS))
    for clbl, bl, rl in COLOR_DEFS:
        name = clbl.replace("$", "").replace("_E", "").replace("\\", "")
        print(f"    {name:10}" + "".join(f"{np.nanmedian(df[f'col_{bl}_{rl}_{a}']):9.3f}" for a in APERS))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cached", action="store_true", help="replot from the CSV cache")
    args = ap.parse_args()

    if args.cached and CSV_PATH.exists():
        print(f"  loading cache {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
    else:
        df = pull(os.getenv("PGUSER", WSDB_USER))
        ARR_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CSV_PATH, index=False)
        print(f"  -> {CSV_PATH}  ({len(df):,} QSOs)")

    _print_summary(df)
    make_plot(df, PNG_PATH)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
