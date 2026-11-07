#!/usr/bin/env python3
"""TEMPORARY — Euclid×Gaia PHZ colour(z) rivers, sourced entirely from
crossmatch_q1_mer_phz_vis_gaia_sep (the full Q1 MER×PHZ×VIS×Gaia crossmatch, 327,953
Gaia-matched Euclid sources — NOT restricted to the CatWISE QSO candidates).

Everything comes from that one table: the 16 Euclid aperture fluxes give z_xgb (10 seed
QSO photo-z models) + the 2FWHM colours, and p_phz_classification gives the PHZ label.
The full crossmatch is the background colour(z) log-density; the chosen PHZ class
(star=phz 1 / galaxy=phz 2 / qso=phz 4) is over-plotted as a scatter (coloured by that
class probability) + its own running median, on the same canvas as
picked_qso_color_vs_xgbz.png (DESI spec-z dashed + DESI spec-QSO XGB-z dotted references).

Reads : {PGUSER}.crossmatch_q1_mer_phz_vis_gaia_sep
Writes: output/qso_color_vs_z/arrays/e_phz_colorz.csv   (derived z_xgb + colours + PHZ class cache)
        output/e_phz_{class}_color_vs_xgbz.png

Usage:
  PGUSER=siwei_zhao_2025 python src/tmp_e_phz_color_vs_z.py               # build cache + star plot
  python src/tmp_e_phz_color_vs_z.py --phz qso --replot                   # redraw a class (no WSDB)
  PGUSER=... python src/tmp_e_phz_color_vs_z.py --phz galaxy --label argmax
"""
import os
import sys
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_qso_color_vs_specz import (COLORS, PALETTE, COLOR_CMAPS, ZMAX_PLOT, _flux_to_mag,
                                        ZPC_VIS, _running_stats, _hexbin_count_range,
                                        WSDB_HOST, WSDB_DB, WSDB_USER)
from analyze_qso_color_vs_photz import _spec_rivers
from train_xgboost_qso_photz import add_colours
from tmp_picked_color_vs_z import ARRAYS_DIR, XGB_DIR, MODELS_DIR, _xgbz_rivers

PROJECT_DIR = Path(__file__).resolve().parents[1]
TABLE       = "crossmatch_q1_mer_phz_vis_gaia_sep"
CACHE       = ARRAYS_DIR / "e_phz_colorz.csv"
_BAND_DB    = {"IE": "vis", "YE": "y", "JE": "j", "HE": "h"}

CLASS_BIT     = {"star": 1, "galaxy": 2, "qso": 4}
CLASS_PROBCOL = {"star": "phz_star", "galaxy": "phz_gal", "qso": "phz_qso"}
CLASS_ARGMAX  = {"galaxy": 0, "qso": 1, "star": 2}   # order of [phz_gal, phz_qso, phz_star]
PROB_CMAP     = "cividis"
PROB_NORM     = plt.Normalize(0.5, 1.0)
SCATTER_CAP   = 8000                                  # subsample big classes for the scatter only


def build_table(user):
    """z_xgb + 2FWHM colours + PHZ label for every Gaia-matched Euclid source (chunked)."""
    import psycopg2, xgboost as xgb
    feats  = json.loads((MODELS_DIR / "features.json").read_text())["features"]
    target = json.loads((XGB_DIR / "metrics.json").read_text())["target"]
    fcols = [f"m_flux_{s}_{a}fwhm_aper"    for s in _BAND_DB.values() for a in (1, 2, 3, 4)]
    ecols = [f"m_fluxerr_{s}_{a}fwhm_aper" for s in _BAND_DB.values() for a in (1, 2, 3, 4)]
    sel = ", ".join(["m_object_id", "p_phz_classification AS bm", "p_phz_star_prob AS phz_star",
                     "p_phz_gal_prob AS phz_gal", "p_phz_qso_prob AS phz_qso",
                     "g_pm_significance AS pm_sig"] + fcols + ecols)

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30,
                            keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5)
    cur = conn.cursor(); cur.execute("SET statement_timeout = 600000")
    cur.execute(f"SELECT count(*) FROM {user}.{TABLE}"); ntot = cur.fetchone()[0]
    parts, off, chunk = [], 0, 50000
    while off < ntot:
        d = pd.read_sql_query(
            f"SELECT {sel} FROM {user}.{TABLE} ORDER BY m_object_id LIMIT %(c)s OFFSET %(o)s",
            conn, params={"c": chunk, "o": off})
        if not len(d):
            break
        parts.append(d); off += len(d)
        print(f"    fetched {off:,}/{ntot:,}")
    conn.close()
    df = pd.concat(parts, ignore_index=True).rename(columns={"m_object_id": "object_id"})
    assert len(df) == ntot, f"fetched {len(df)} != {ntot} rows"

    for band, s in _BAND_DB.items():
        zpc = ZPC_VIS if s == "vis" else 1.0
        for a in (1, 2, 3, 4):
            df[f"{band}_{a}"] = _flux_to_mag(df[f"m_flux_{s}_{a}fwhm_aper"],
                                             df[f"m_fluxerr_{s}_{a}fwhm_aper"], zpc)
    df = add_colours(df)
    ok = df.dropna(subset=feats).reset_index(drop=True)
    print(f"  {len(ok):,}/{len(df):,} rows with all 16 bands detected -> predicting z_xgb")
    X = ok[feats].to_numpy(float)
    preds = []
    for p in sorted(MODELS_DIR.glob("xgb_photz_seed*.json")):
        m = xgb.XGBRegressor(); m.load_model(str(p))
        pr = m.predict(X); preds.append(np.expm1(pr) if target == "log1p_z" else pr)
    ok["z_xgb"] = np.mean(preds, axis=0)
    for _, b, r in COLORS:
        ok[f"{b}_{r}"] = ok[f"{b}_2"] - ok[f"{r}_2"]
    keep = ["object_id", "z_xgb", "bm", "phz_star", "phz_gal", "phz_qso", "pm_sig"] + \
           [f"{b}_{r}" for _, b, r in COLORS]
    return ok[keep]


def select_class(df, cls, label_mode):
    if label_mode == "argmax":
        probs = df[["phz_gal", "phz_qso", "phz_star"]].to_numpy(float)
        sel = np.argmax(probs, axis=1) == CLASS_ARGMAX[cls]
    else:
        sel = df["bm"] == CLASS_BIT[cls]
    out = df[sel].copy()
    out["phz_prob"] = out[CLASS_PROBCOL[cls]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true", help="redraw from the cache only (no WSDB).")
    ap.add_argument("--phz", choices=["star", "galaxy", "qso"], default="star")
    ap.add_argument("--label", choices=["bitmask", "argmax"], default="bitmask")
    args = ap.parse_args()
    user = os.getenv("PGUSER", WSDB_USER)
    cls = args.phz
    out = PROJECT_DIR / "output" / f"e_phz_{cls}_color_vs_xgbz.png"

    if CACHE.exists():
        df = pd.read_csv(CACHE)
        print(f"  loaded {len(df):,} Euclid×Gaia colour(z) rows from cache")
    elif args.replot:
        print(f"  --replot needs {CACHE.name} — run once without it."); return 1
    else:
        df = build_table(user)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE, index=False); print(f"  cached -> {CACHE}")

    sub = select_class(df, cls, args.label)
    print(f"  PHZ {cls} ({args.label}): {len(sub):,} of {len(df):,}")

    qcsv = ARRAYS_DIR / "rivers_qso_2fwhm.csv"
    df_desi = pd.read_csv(qcsv)
    spec_rivers = _spec_rivers(df_desi)
    xgbz_rivers = _xgbz_rivers(df_desi)
    pooled = []
    for _, b, r in COLORS:
        c = df_desi[f"{b}_{r}"].to_numpy(float); c = c[np.isfinite(c)]
        pooled += list(np.percentile(c, [1, 99]))
    cpad = 0.15 * (max(pooled) - min(pooled))
    ylim = (min(pooled) - cpad, max(pooled) + cpad)

    ncol = len(COLORS)
    fig, axes = plt.subplots(1, ncol, figsize=(7.2 * ncol, 6.8), squeeze=False)
    axes = axes[0]
    zp = df["z_xgb"].to_numpy(float)
    extent = (0, ZMAX_PLOT, ylim[0], ylim[1])

    panel, dvmin, dvmax = [], np.inf, -np.inf
    for label, b, r in COLORS:
        col = f"{b}_{r}"
        c = df[col].to_numpy(float)
        m = np.isfinite(c) & np.isfinite(zp) & (zp <= ZMAX_PLOT)
        z, cc = zp[m], c[m]
        lo, hi = _hexbin_count_range(z, cc, extent)
        dvmin = min(dvmin, lo); dvmax = max(dvmax, hi)
        panel.append((label, col, z, cc))

    # subsample the scatter of large classes (median/stats use the FULL sample)
    zs_all = sub["z_xgb"].to_numpy(float)
    rng = np.arange(len(sub))
    if len(sub) > SCATTER_CAP:
        step = int(np.ceil(len(sub) / SCATTER_CAP))
        scat_idx = rng[::step]
        print(f"  scatter subsampled to {len(scat_idx):,} of {len(sub):,} (every {step}th) for legibility")
    else:
        scat_idx = rng

    sc = None
    for k, (label, col, z, cc) in enumerate(panel):
        ax = axes[k]
        ax.hexbin(z, cc, gridsize=100, bins="log", cmap=COLOR_CMAPS[col], mincnt=1,
                  extent=extent, vmin=dvmin, vmax=dvmax, alpha=0.45, zorder=1, rasterized=True)
        mid, med, lo, hi = _running_stats(z, cc)
        ax.fill_between(mid, lo, hi, color=PALETTE[col], alpha=0.20, lw=0, zorder=2,
                        label="Euclid×Gaia 16–84%")
        ln, = ax.plot(mid, med, color=PALETTE[col], lw=2.4, zorder=6, label="Euclid×Gaia median")
        ln.set_path_effects([pe.Stroke(linewidth=4, foreground="white"), pe.Normal()])
        smid, smed = spec_rivers[col]
        sl, = ax.plot(smid, smed, color="k", lw=1.6, ls="--", zorder=5, label="DESI spec-z median")
        sl.set_path_effects([pe.Stroke(linewidth=3, foreground="white"), pe.Normal()])
        if xgbz_rivers is not None:
            xmid, xmed = xgbz_rivers[col]
            xl, = ax.plot(xmid, xmed, color="k", lw=1.6, ls=":", zorder=5,
                          label="DESI spec-QSO XGB-z median")
            xl.set_path_effects([pe.Stroke(linewidth=3, foreground="white"), pe.Normal()])

        cs_all = sub[col].to_numpy(float)
        zsp, csp = zs_all[scat_idx], cs_all[scat_idx]
        msp = np.isfinite(zsp) & np.isfinite(csp) & (zsp <= ZMAX_PLOT)
        sc = ax.scatter(zsp[msp], csp[msp], s=10, c=sub["phz_prob"].to_numpy(float)[scat_idx][msp],
                        cmap=PROB_CMAP, norm=PROB_NORM, alpha=0.7, lw=0, zorder=7)
        mall = np.isfinite(zs_all) & np.isfinite(cs_all) & (zs_all <= ZMAX_PLOT)
        smid2, smed2, _, _ = _running_stats(zs_all[mall], cs_all[mall])
        stl, = ax.plot(smid2, smed2, color="k", lw=2.4, zorder=8, label=f"PHZ-{cls} median")
        stl.set_path_effects([pe.Stroke(linewidth=4, foreground="white"), pe.Normal()])

        ax.set_xlim(0, 3.5); ax.set_ylim(*ylim)
        ax.set_xlabel("XGBoost photo-$z$", fontsize=11)
        ax.set_ylabel(f"{label}  (AB)", fontsize=11)
        ax.set_title(label, fontsize=13, fontweight="bold")
        if k >= 1:
            ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    n = int(np.isfinite(zs_all).sum())
    fig.suptitle(f"Euclid×Gaia PHZ {cls} on the colour(z) rivers  (crossmatch_q1_mer_phz_vis_gaia_sep) "
                 f"— {n:,} PHZ {cls} ({args.label})\n"
                 f"background = full Euclid×Gaia colour(z) log-density; solid black = PHZ-{cls} median; "
                 f"dashed = DESI spec-z median, dotted = DESI spec-QSO XGB-z median",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.94, 0.95])
    ptop = axes[-1].get_position()
    cax = fig.add_axes([0.955, ptop.y0, 0.011, ptop.height])
    fig.colorbar(sc, cax=cax).set_label(f"Euclid PHZ {cls} probability", fontsize=10)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> {out}  ({n:,} PHZ {cls} placed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
