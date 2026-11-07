#!/usr/bin/env python3
"""
STEP 3 of the QSO-companion pipeline (Gaia-matched variant): APPLY the trained Euclid
XGBoost photo-z to the CatWISE×Euclid×Gaia Q1 QSO candidates and plot their Euclid
colour(z) rivers against the PREDICTED redshift.

Same as apply_photz_catwise_euclid.py but restricted to candidates that ALSO have a Gaia
counterpart (crossmatch_qso_catwise_euclid_gaia) — a smaller, more point-source-like and
likely purer subset. The 10 seed models in output/xgboost_qso_photz/models/ were trained
on DESI spec QSOs (train_xgboost_qso_photz.py); here we run them on the Gaia-matched Q1
sources, pulling the same 16 aperture mags (VIS/Y/J/H × 1/2/3/4 FWHM) from mer_final_cat,
average the 10 predictions, and draw the colour(photo-z) rivers exactly like
qso_color_vs_specz.png / qso_color_vs_photz.png. The DESI spec-z QSO median is overlaid
(dashed) so you can see whether the candidates trace the known QSO locus.

Reads (WSDB; Astronomy/Cambridge VPN; auth via ~/.pgpass):
  {PGUSER}.crossmatch_qso_catwise_euclid_gaia   (Q1 candidates; e_id -> Euclid object_id)
  euclid_q1.mer_final_cat                       (16 aperture fluxes, joined by e_id)
  output/xgboost_qso_photz/models/*.json        (10 seed models + features.json)
  {PGUSER}.q1_phz_photo_z                        (Euclid official PHZ mode_1, joined by e_id)
Writes:
  output/qso_color_vs_z/catwise_e_g_color_vs_xgbz.png    (colour vs XGBoost photo-z, single row)
  output/qso_color_vs_z/arrays/catwise_euclid_gaia_xgbz.csv   (XGB predictions + colours cache)
  output/qso_color_vs_z/arrays/catwise_euclid_gaia_phz.csv    (PHZ mode_1 cache)

Usage:
  PGUSER=siwei_zhao_2025 python src/apply_photz_catwise_euclid_gaia.py     # apply + both plots (caches)
  python src/apply_photz_catwise_euclid_gaia.py --replot                   # redraw from caches (no WSDB)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_qso_color_vs_specz import (  # noqa: E402
    COLORS, ZMAX_PLOT, WSDB_HOST, WSDB_DB, WSDB_USER, ARRAYS_DIR,
    _flux_to_mag, ZPC_VIS, fetch_pop,
)
from analyze_qso_color_vs_photz import _draw_row, _spec_rivers  # noqa: E402
from train_xgboost_qso_photz import add_colours  # noqa: E402  (model features = 12 colours)

PROJECT_DIR = Path(__file__).resolve().parents[1]
XGB_DIR     = PROJECT_DIR / "output" / "xgboost_qso_photz"
MODELS_DIR  = XGB_DIR / "models"
ARRAYS_DIR  = PROJECT_DIR / "output" / "qso_color_vs_z" / "arrays"
PNG_XGBZ    = PROJECT_DIR / "output" / "qso_color_vs_z" / "catwise_e_g_color_vs_xgbz.png"
CACHE_XGBZ  = ARRAYS_DIR / "catwise_euclid_gaia_xgbz.csv"  # object_id, qso_id, z_xgb, colours
CACHE_PHZ   = ARRAYS_DIR / "catwise_euclid_gaia_phz.csv"   # object_id, phz_mode_1
CATWISE_TABLE = "crossmatch_qso_catwise_euclid_gaia"
MER_TABLE   = "euclid_q1.mer_final_cat"
PHZ_TABLE   = "q1_phz_photo_z"
_BAND_DB    = {"IE": "vis", "YE": "y", "JE": "j", "HE": "h"}


def fetch_phz(user: str, object_ids) -> pd.DataFrame:
    """Euclid PHZ mode_1 for the candidate Euclid object_ids (chunked index lookups)."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30,
                            keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5)
    conn.cursor().execute("SET statement_timeout = 600000")
    ids, chunk, parts = list(map(int, object_ids)), 20000, []
    for i in range(0, len(ids), chunk):
        parts.append(pd.read_sql_query(
            f"SELECT object_id, phz_mode_1 FROM {user}.{PHZ_TABLE} WHERE object_id = ANY(%(b)s)",
            conn, params={"b": ids[i:i + chunk]}))
        print(f"    phz {min(i + chunk, len(ids)):,}/{len(ids):,}")
    conn.close()
    df = pd.concat(parts, ignore_index=True).drop_duplicates("object_id")
    print(f"  {len(df):,} of {len(ids):,} candidates have a PHZ entry")
    return df


# --------------------------------------------------------------------- apply model
def build_and_predict(user: str) -> pd.DataFrame:
    """Pull CatWISE×Euclid×Gaia Q1 candidates, join mer_final_cat for the 16 aperture
    mags, apply the 10 seed models (mean), and return {qso_id, z_xgb, colours}."""
    import psycopg2, xgboost as xgb
    feats = json.loads((MODELS_DIR / "features.json").read_text())["features"]
    target = json.loads((XGB_DIR / "metrics.json").read_text())["target"]

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30,
                            keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5)
    conn.cursor().execute("SET statement_timeout = 600000")
    fcols = [f"flux_{s}_{a}fwhm_aper" for s in _BAND_DB.values() for a in (1, 2, 3, 4)]
    ecols = [f"fluxerr_{s}_{a}fwhm_aper" for s in _BAND_DB.values() for a in (1, 2, 3, 4)]
    sel = ", ".join(fcols + ecols)

    print(f"  reading {CATWISE_TABLE} (Q1) e_ids ...")
    ids = pd.read_sql_query(
        f"""SELECT DISTINCT ON (e_id) e_id AS object_id, qso_id
            FROM {user}.{CATWISE_TABLE} WHERE euclid_source = 'Q1' AND e_id IS NOT NULL
            ORDER BY e_id""", conn)
    print(f"  {len(ids):,} unique CatWISE×Gaia Q1 Euclid ids")

    # chunked index-lookups into the wide mer_final_cat (a single big join can time out)
    id_list = ids["object_id"].tolist()
    chunk, parts = 20000, []
    for i in range(0, len(id_list), chunk):
        batch = id_list[i:i + chunk]
        parts.append(pd.read_sql_query(
            f"SELECT object_id, {sel} FROM {MER_TABLE} WHERE object_id = ANY(%(b)s)",
            conn, params={"b": batch}))
        print(f"    mer fluxes {min(i + chunk, len(id_list)):,}/{len(id_list):,}")
    conn.close()
    mer = pd.concat(parts, ignore_index=True)
    df = ids.merge(mer, on="object_id", how="inner").drop_duplicates("object_id").reset_index(drop=True)
    print(f"  {len(df):,} candidates matched in mer_final_cat")

    for band, s in _BAND_DB.items():
        zpc = ZPC_VIS if s == "vis" else 1.0
        for a in (1, 2, 3, 4):
            df[f"{band}_{a}"] = _flux_to_mag(df[f"flux_{s}_{a}fwhm_aper"],
                                             df[f"fluxerr_{s}_{a}fwhm_aper"], zpc)
    df = add_colours(df)   # 12 within-aperture colours = the model features
    # in-domain: all colours finite <=> all 16 bands detected (S/N>=3), same as training
    df = df.dropna(subset=feats).reset_index(drop=True)
    print(f"  {len(df):,} with all bands detected (S/N>=3) -> predicting")

    X = df[feats].to_numpy(float)
    preds = []
    for p in sorted(MODELS_DIR.glob("xgb_photz_seed*.json")):
        m = xgb.XGBRegressor(); m.load_model(str(p))
        pr = m.predict(X)
        preds.append(np.expm1(pr) if target == "log1p_z" else pr)
    df["z_xgb"] = np.mean(preds, axis=0)
    print(f"  applied {len(preds)} seed models (target={target})")

    # colours from the 2FWHM mags (canonical, matches qso_color_vs_specz / _photz)
    for _, b, r in COLORS:
        df[f"{b}_{r}"] = df[f"{b}_2"] - df[f"{r}_2"]
    keep = ["object_id", "qso_id", "z_xgb"] + [f"{b}_{r}" for _, b, r in COLORS]
    return df[keep]


# ------------------------------------------------------------------------- plot
def make_plot(df_xgb, pop, spec_rivers, out: Path, ylim):
    """Single-row rivers plot: colour vs XGBoost photo-z."""
    ncol = 1 + len(COLORS)
    fig, axes = plt.subplots(1, ncol, figsize=(6.8 * ncol, 6.8))
    _draw_row(fig, axes, df_xgb, "z_xgb", "XGBoost photo-$z$",
              pop, ylim, show_poplegend=True, spec_rivers=spec_rivers)
    fig.suptitle(
        f"CatWISE×Euclid×Gaia Q1 QSO candidates: Euclid colour vs XGBoost photo-z  (N={len(df_xgb):,})"
        f"  ·  2×FWHM aperture photometry\n"
        f"dashed = DESI spec-z QSO median (known locus);  left marginal = candidates vs full Q1 population",
        fontsize=15, fontweight="bold", y=0.98)
    try:
        fig.tight_layout(rect=[0, 0, 1, 0.92])
    except Exception:
        fig.subplots_adjust(top=0.88, wspace=0.4)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


# ------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replot", action="store_true", help="redraw from caches only (no WSDB).")
    args = ap.parse_args()
    user = os.getenv("PGUSER", WSDB_USER)

    if args.replot and not (CACHE_XGBZ.exists() and CACHE_PHZ.exists()):
        print("  --replot needs both catwise_euclid_gaia_{xgbz,phz}.csv — run once without it."); return 1

    # XGB photo-z predictions (reuse cache if present; the mer join is the slow part)
    if CACHE_XGBZ.exists():
        df_cat = pd.read_csv(CACHE_XGBZ)
        print(f"  loaded {len(df_cat):,} candidate XGB predictions from cache")
    else:
        df_cat = build_and_predict(user)
        CACHE_XGBZ.parent.mkdir(parents=True, exist_ok=True)
        df_cat.to_csv(CACHE_XGBZ, index=False); print(f"  cached -> {CACHE_XGBZ}")

    # Euclid PHZ mode_1 for the same Euclid sources
    if CACHE_PHZ.exists():
        phz = pd.read_csv(CACHE_PHZ)
        print(f"  loaded {len(phz):,} candidate PHZ rows from cache")
    else:
        phz = fetch_phz(user, df_cat["object_id"].to_numpy())
        phz.to_csv(CACHE_PHZ, index=False); print(f"  cached -> {CACHE_PHZ}")

    # DESI spec-z reference rivers + Q1 population marginals (shared with the spec-z plot)
    qcsv = ARRAYS_DIR / "rivers_qso_2fwhm.csv"
    pcsv = ARRAYS_DIR / "rivers_pop_2fwhm.csv"
    if qcsv.exists():
        df_desi = pd.read_csv(qcsv)
    else:
        from analyze_qso_color_vs_specz import build_dataframe
        df_desi = build_dataframe(user, 2)
    pop = pd.read_csv(pcsv) if pcsv.exists() else fetch_pop(user, 2)
    spec_rivers = _spec_rivers(df_desi)

    # shared colour y-range from the DESI reference (same scale as qso_color_vs_specz)
    pooled = []
    for _, b, r in COLORS:
        c = df_desi[f"{b}_{r}"].to_numpy(float); c = c[np.isfinite(c)]
        pooled += list(np.percentile(c, [1, 99]))
    cpad = 0.15 * (max(pooled) - min(pooled))
    ylim = (min(pooled) - cpad, max(pooled) + cpad)

    # single-row figure: colour vs XGBoost photo-z only
    # (`phz` above is still fetched/cached so apply_photz_catwise_euclid_gaia_pm.py has its PHZ cache)
    make_plot(df_cat, pop, spec_rivers, PNG_XGBZ, ylim)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
