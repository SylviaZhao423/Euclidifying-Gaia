#!/usr/bin/env python3
"""Euclid cutouts for the Euclid×Gaia QSOs that PASS the "2-parameter (no parallax/PM) & z > 0.7"
test — i.e. the sources highlighted in analyze_picked_qso_other.py's row-2 panel (the 159 of the
1,004 Euclid-matched: 156 Q1 + 3 ERO).

Same rendering as download_qso_ds_e_g_cutouts.py (identical Euclid VIS+NISP stamp, same companion
rings, same spec-QSO title, same per-survey (desi / sdss) grouping — Q1 + ERO combined) — this
script just adds the pass filter on top of that script's fetch:

  pass = (g_astrometric_params_solved == 3)   # Gaia 2-param solution: no parallax/PM
         AND (qso_z > 0.7)                     # spec-z > 0.7
         AND (g_phot_g_mean_mag >= 17)         # same G cut as the combined UMAP build

De-duplicated by Gaia source_id (Q1 kept over ERO if a source has both), so each Gaia source is
rendered once — exactly the set counted in the qso_other_features.png panel.

Companion rings (same code as the ds_e_g cutouts):
  Red crosshair  — Euclid primary centroid (cutout centre)   Orange + — secondary Euclid (<1")
  Magenta x      — QSO (DESI/SDSS) catalogue position          Bright blue — Gaia matched (<1")
  Pale blue      — Gaia in the stamp FOV (>1")

Outputs (output/qso_cutouts/ds_e_g_pass/); one folder per survey — Q1 + ERO combined:
  desi/ sdss/                             cutout_0000.png ...   (individual stamps, sorted by G)
  contact_sheet_{survey}.png                                    per-survey contact sheet
  manifest.csv                                                  every rendered source + blend cols
      (the Q1/ERO release stays in each stamp title and the manifest euclid_source column)

Usage:
    PGUSER=siwei_zhao_2025 conda run -n base python3 src/download_qso_ds_e_g_pass_cutouts.py
    ... --size 6          # cutout FOV arcsec (default 6)
    ... --survey desi     # only one survey
    ... --n 40            # cap sources per group (0 = all, default)
    ... --no-gaia         # skip WSDB ring queries (stamp only)
    ... --force           # re-fetch cached stamps
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_qso_ds_e_g_cutouts import fetch_survey, bake_stamp, contact_sheet   # noqa: E402
from download_qso_catwise_e_g_cutouts import (                                    # noqa: E402
    fetch_euclid, fetch_gaia_positions, fetch_euclid_secondary_positions,
    EUCLID_RING, WISE_RING, CACHE_DIR,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output" / "qso_cutouts" / "ds_e_g_pass"

GMAG_MIN = 17.0    # same G cut as the combined QSO×Gaia UMAP build


def select_passing(merged: pd.DataFrame) -> pd.DataFrame:
    """Keep only the Euclid×Gaia QSOs passing 2-param (aps==3) & z>0.7 & G>=17, de-duplicated by
    Gaia source_id (Q1 preferred over ERO), brightest first — the qso_other_features.png pass set."""
    aps = pd.to_numeric(merged["aps"], errors="coerce")
    z   = pd.to_numeric(merged["qso_z"], errors="coerce")
    g   = pd.to_numeric(merged["g_phot_g_mean_mag"], errors="coerce")
    keep = merged[(aps == 3) & (z > 0.7) & (g >= GMAG_MIN)].copy()
    keep["_r"] = (keep["euclid_source"] != "Q1").astype(int)          # Q1 (0) before ERO (1)
    keep = (keep.sort_values(["_r", "g_phot_g_mean_mag"])
                .drop_duplicates("sid_str").reset_index(drop=True))
    print(f"  passing 2-param & z>0.7 & G>={GMAG_MIN:g}: {len(keep)} unique Gaia sources "
          f"({(keep.euclid_source=='Q1').sum()} Q1, {(keep.euclid_source=='ERO').sum()} ERO)")
    return keep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size",   type=float, default=6.0, help="cutout FOV arcsec (default 6)")
    ap.add_argument("--survey", choices=["desi", "sdss", "both"], default="both")
    ap.add_argument("--n",      type=int,   default=0,
                    help="max sources per group, 0 = all (default 0)")
    ap.add_argument("--no-gaia", action="store_true", help="skip all WSDB ring queries")
    ap.add_argument("--force",   action="store_true",
                    help="delete cached stamps before rendering (re-fetches from IRSA)")
    args = ap.parse_args()

    user = os.environ.get("PGUSER", "siwei_zhao_2025")
    surveys = ["desi", "sdss"] if args.survey == "both" else [args.survey]

    print("[1] fetching Euclid∩Gaia matched QSOs ...")
    merged = pd.concat([fetch_survey(s, user) for s in surveys], ignore_index=True)
    print("[2] applying the 2-param & z>0.7 pass filter ...")
    merged = select_passing(merged)
    if merged.empty:
        print("  no passing sources — nothing to render.")
        return 0

    # Ring inputs (shared across groups) — build once for the whole passing set.
    blend_info: dict = {}
    gaia: dict = {}
    euclid_sec: dict = {}
    if not args.no_gaia:
        print("[3] fetching Gaia + secondary-Euclid ring positions ...")
        blend_info = {r["sid_str"]: r.to_dict() for _, r in merged.iterrows()}
        gaia       = fetch_gaia_positions(merged, args.size, blend_info)
        euclid_sec = fetch_euclid_secondary_positions(merged, blend_info, user)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    # one folder per survey (Q1 + ERO combined): desi, sdss
    for group in surveys:
        sub = merged[merged["group"] == group]
        if args.n > 0:
            sub = sub.sort_values("g_phot_g_mean_mag").head(args.n)
        sub = sub.reset_index(drop=True)
        print(f"\n{group}: {len(sub)} sources")
        if sub.empty:
            print("  (none — skipping)")
            continue

        gdir = OUT_DIR / group
        gdir.mkdir(parents=True, exist_ok=True)
        for old in gdir.glob("*.png"):
            old.unlink()

        items = []
        n_ok = 0
        for i, (_, row) in enumerate(sub.iterrows()):
            sid = row["sid_str"]

            # Euclid centroid = cutout centre; QSO + Gaia + secondary Euclid as companions
            e_ra  = row.get("e_ra")  if pd.notna(row.get("e_ra"))  else float(row["qso_ra"])
            e_dec = row.get("e_dec") if pd.notna(row.get("e_dec")) else float(row["qso_dec"])
            companions = None
            if not args.no_gaia:
                sec_euclid = [(ra, dec, lbl, EUCLID_RING, "crosshair")
                              for ra, dec, lbl, *_ in euclid_sec.get(sid, [])]
                companions = (list(gaia.get(sid, []))
                              + sec_euclid
                              + [(float(row["qso_ra"]), float(row["qso_dec"]), None, WISE_RING, "x")])

            if args.force:
                for _p in CACHE_DIR.glob(f"{sid}_*.png"):
                    _p.unlink(missing_ok=True)
            arr   = fetch_euclid(row, args.size, companions=companions,
                                 center_ra=e_ra, center_dec=e_dec)
            n_ok += arr is not None
            baked = bake_stamp(row, arr)
            fn    = f"cutout_{i:04d}.png"
            plt.imsave(gdir / fn, np.clip(baked, 0, 1))
            items.append((baked, float(row["g_phot_g_mean_mag"])
                          if pd.notna(row.get("g_phot_g_mean_mag")) else np.nan))

            def _arr_str(v):
                return "|".join(str(x) for x in (v or [])) if v is not None else ""

            manifest.append({
                "group":            group,
                "survey":           row["survey"],
                "euclid_source":    row["euclid_source"],
                "cutout_file":      f"{group}/{fn}",
                "gaia_source_id":   sid,
                "qso_id":           int(row["qso_id"]),
                "qso_catalog":      row.get("qso_catalog", ""),
                "qso_ra":           row["qso_ra"],
                "qso_dec":          row["qso_dec"],
                "qso_z":            row.get("qso_z", ""),
                "e_id":             int(row["e_id"]) if pd.notna(row.get("e_id")) else "",
                "e_ra":             e_ra,
                "e_dec":            e_dec,
                "euclid_sep_arcsec": row.get("euclid_sep_arcsec", ""),
                "aps":              int(row["aps"]) if pd.notna(row.get("aps")) else "",
                "g_phot_g_mean_mag": row.get("g_phot_g_mean_mag", ""),
                "euclid_ok":        arr is not None,
                "gaia_sep_arcsec":  row.get("gaia_sep_arcsec", ""),
                "n_gaia_within_1arcsec":   row.get("n_gaia_within_1arcsec", ""),
                "all_gaia_ids":     _arr_str(row.get("all_gaia_ids_within_1arcsec")),
                "gaia_is_blend":    row.get("gaia_is_mutual_blend", ""),
                "n_euclid_within_1arcsec": row.get("n_euclid_within_1arcsec", ""),
                "all_euclid_ids":   _arr_str(row.get("all_euclid_ids_within_1arcsec")),
                "euclid_is_blend":  row.get("euclid_is_mutual_blend", ""),
                "e_deblended_flag": row.get("e_deblended_flag", ""),
                "n_gaia_in_fov":    len(gaia.get(sid, [])),
                "n_euclid_secondary": len(euclid_sec.get(sid, [])),
            })
            if (i + 1) % 25 == 0 or (i + 1) == len(sub):
                print(f"  {i+1}/{len(sub)}  ({n_ok} Euclid ok)", flush=True)

        contact_sheet(items, OUT_DIR / f"contact_sheet_{group}.png",
                      f"QSO Euclid+Gaia — 2-param & z>0.7 pass — {group}")
        print(f"  {group}: {n_ok}/{len(sub)} Euclid ok")

    pd.DataFrame(manifest).to_csv(OUT_DIR / "manifest.csv", index=False)
    print(f"\n  -> {OUT_DIR / 'manifest.csv'}  ({len(manifest)} sources)")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
