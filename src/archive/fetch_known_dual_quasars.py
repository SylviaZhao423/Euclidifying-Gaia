#!/usr/bin/env python3
"""
Fetch published, Gaia-based dual/binary-quasar catalogs from VizieR into one clean seed
list — the reference/test set for building a Gaia dual-QSO finder.

Physical dual/binary quasars (two real QSOs) are what we target. This pulls the machine-
readable, Gaia-astrometry-selected samples that live in our feature space:

  VODKA  — Chen et al. 2022, J/ApJ/925/162  (HST-resolved varstrometry double quasars)
           table3: 45 resolved systems with a Class label (dual/lensed quasar,
                   quasar+star, ...), separation (Sep, arcsec) and rp (kpc).
           table2: coords (_RA/_DE), z, Gaia astrometric excess noise (epsi1/epsi2).
  Hwang  — Hwang et al. 2020, J/ApJ/888/73   (varstrometry candidate parent sample,
           49 sources with coords + plx/pm; candidates, not confirmed).

Output (one row per system):
  name, ra, dec, z, sep_arcsec, rp_kpc, gaia_excess_noise_mas, class, tier, source

  tier   'resolved'  = VODKA HST-resolved (has a Class label)
         'candidate' = varstrometry-selected, unconfirmed
  class  raw VODKA Class string (dual/lensed quasar | quasar+star | ...), else ''

NOTE: the very recent Deng et al. "Search for quasar pairs with Gaia astrometric data"
series (I-IV, DESI-confirmed dual vs projected) is not on VizieR yet — add from the
papers when available; it is the best-matched RESOLVED confirmed sample.

Writes:  output/qso_pairs/known_dual_quasars.csv

Usage:   python src/fetch_known_dual_quasars.py     # VizieR only, no VPN/WSDB
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_CSV = PROJECT_DIR / "output" / "qso_pairs" / "known_dual_quasars.csv"


def _num(col):
    return pd.to_numeric(col, errors="coerce")


def fetch() -> pd.DataFrame:
    from astroquery.vizier import Vizier
    v = Vizier(columns=["**"], row_limit=-1)

    # ---- VODKA (Chen+ 2022) : table3 (labels+sep) merged with table2 (coords/z/epsi) ----
    t2, t3 = None, None
    for t in v.get_catalogs("J/ApJ/925/162"):
        nm = t.meta.get("name", "")
        if nm.endswith("table2"): t2 = t.to_pandas()
        if nm.endswith("table3"): t3 = t.to_pandas()
    vod = t3.merge(t2[["Name", "_RA", "_DE", "z", "epsi1", "epsi2"]], on="Name",
                   how="left", suffixes=("", "_t2"))
    vod = pd.DataFrame({
        "name": vod["Name"].astype(str).str.strip(),
        "ra":   _num(vod["_RA"]),
        "dec":  _num(vod["_DE"]),
        "z":    _num(vod["z"]),
        "sep_arcsec": _num(vod["Sep"]),
        "rp_kpc":     _num(vod["rp"]),
        "gaia_excess_noise_mas": _num(vod[["epsi1", "epsi2"]].apply(_num).max(axis=1)),
        "class": vod["Class"].astype(str).str.strip(),
        "tier":  "resolved",
        "source": "VODKA_Chen2022",
    })
    print(f"  VODKA (Chen+2022): {len(vod)} resolved systems")

    # ---- Hwang+ 2020 varstrometry candidate parent sample (coords only) ----
    cand = []
    for t in v.get_catalogs("J/ApJ/888/73"):
        d = t.to_pandas()
        rac = "_RA.icrs" if "_RA.icrs" in d else "RA_ICRS"
        dec = "_DE.icrs" if "_DE.icrs" in d else "DE_ICRS"
        cand.append(pd.DataFrame({
            "name": d["SDSS"].astype(str).str.strip(), "ra": _num(d[rac]), "dec": _num(d[dec]),
            "z": np.nan, "sep_arcsec": np.nan, "rp_kpc": np.nan,
            "gaia_excess_noise_mas": np.nan, "class": "", "tier": "candidate",
            "source": "Hwang2020_varstrometry"}))
    hw = pd.concat(cand, ignore_index=True).drop_duplicates("name")
    print(f"  Hwang+2020 varstrometry candidates: {len(hw)}")

    out = pd.concat([vod, hw], ignore_index=True)
    out = out[out["ra"].notna() & out["dec"].notna()].reset_index(drop=True)
    return out


def main() -> int:
    df = fetch()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  total {len(df)} systems with coords -> {OUT_CSV}")
    print("\n  by tier:");   print(df["tier"].value_counts().to_string())
    print("\n  VODKA Class breakdown:")
    print(df[df.tier == "resolved"]["class"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
