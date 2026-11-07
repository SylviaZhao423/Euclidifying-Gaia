#!/usr/bin/env python3
"""Label the hand-picked sources on the Euclid colour(z) rivers.

Base = catwise_e_g_color_vs_xgbz_pm.png (Euclid colour vs XGBoost photo-z for the
CatWISE×Euclid×Gaia Q1 candidate population), but the 16-84% spread is drawn as a
FILLED coloured band like qso_by_phz_labels.png, and the hand-picked sources are
over-plotted as cluster-coloured stars with WISE-id labels.

z_xgb + 2FWHM colours are computed FRESH per picked source from the 16 aperture
fluxes in euclid_q1.mer_final_cat (same recipe as apply_photz_catwise_euclid_gaia.py).
Those apertures only exist for Q1 sources, so the 13 ERO picks cannot be placed on
this Q1 XGB-photo-z plot — they are listed in the corner instead.

Output: output/picked_qso/picked_qso_color_vs_xgbz.png
"""
import os
import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm, Normalize
import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_qso_color_vs_specz import (COLORS, PALETTE, COLOR_CMAPS, ZMAX_PLOT, _flux_to_mag,
                                        ZPC_VIS, _running_stats, _hexbin_count_range,
                                        WSDB_HOST, WSDB_DB, WSDB_USER)
from analyze_qso_color_vs_photz import _spec_rivers
from train_xgboost_qso_photz import add_colours
from analyze_picked_qso_contact_sheet import PICKED, CLUSTER_COLOURS

PROJECT_DIR = Path(__file__).resolve().parents[1]
XGB_DIR     = PROJECT_DIR / "output" / "xgboost_qso_photz"
MODELS_DIR  = XGB_DIR / "models"
ARRAYS_DIR  = PROJECT_DIR / "output" / "qso_color_vs_z" / "arrays"
FLAT_CSV    = PROJECT_DIR / "output" / "umap" / "qso" / "arrays" / "catwise_euclid_gaia.csv"
POP_CACHE   = ARRAYS_DIR / "catwise_euclid_gaia_xgbz.csv"    # full Q1 candidate colour(z)
OOF_CSV     = PROJECT_DIR / "output" / "xgboost_qso_photz" / "oof.csv"   # DESI QSO XGB OOF photo-z
OUT         = PROJECT_DIR / "output" / "picked_qso" / "picked_qso_color_vs_xgbz.png"
CATWISE_TABLE = "crossmatch_qso_catwise_euclid_gaia"
MER_TABLE   = "euclid_q1.mer_final_cat"
_BAND_DB    = {"IE": "vis", "YE": "y", "JE": "j", "HE": "h"}

# picked sources: marker shape = UMAP cluster, fill colour = Gaia PM significance.
# CLUSTER_SIZES is per-cluster so shapes with different visual weight (a ★ looks bigger
# than a ◆ at the same s) can be balanced independently — tweak any value here.
CLUSTER_MARKERS = {0: "*", 1: "o", 2: "s", 3: "D", 4: "^", -1: "X"}
CLUSTER_SIZES   = {0: 160, 1: 70, 2: 50, 3: 50, 4: 70, -1: 50}
# row 1: Gaia PM significance (diverging, pale-yellow midpoint at the 3σ threshold)
PM_CMAP   = "RdYlBu_r"
PM_NORM   = TwoSlopeNorm(vmin=0.0, vcenter=3.0, vmax=10.0)
PM_LABEL  = "Gaia PM significance"
# row 2: XGBoost P(gal) in [0, 1] (star -> galaxy), same cmap as the contact-sheet borders
PGAL_CMAP = "RdYlBu_r"
PGAL_NORM = Normalize(vmin=0.0, vmax=1.0)
PGAL_LABEL = "XGBoost P(gal)   (0 = star, 1 = galaxy)"


def predict_picked(user):
    """z_xgb + 2FWHM colours for the picked Q1 sources, straight from mer_final_cat."""
    feats  = json.loads((MODELS_DIR / "features.json").read_text())["features"]
    target = json.loads((XGB_DIR / "metrics.json").read_text())["target"]

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30)
    conn.cursor().execute("SET statement_timeout = 300000")
    # picked gaia -> Euclid e_id (Q1 only)
    idmap = pd.read_sql_query(
        f"""SELECT g_source_id, e_id AS object_id, qso_id, euclid_source,
                   g_pm_significance AS pm_sig
            FROM {user}.{CATWISE_TABLE}
            WHERE g_source_id = ANY(%(g)s) AND e_id IS NOT NULL""",
        conn, params={"g": [int(x) for x in PICKED]})
    q1 = idmap[idmap["euclid_source"] == "Q1"].drop_duplicates("object_id")
    print(f"  picked: {len(idmap)} matched  ({(idmap.euclid_source=='Q1').sum()} Q1, "
          f"{(idmap.euclid_source=='ERO').sum()} ERO)")

    fcols = [f"flux_{s}_{a}fwhm_aper" for s in _BAND_DB.values() for a in (1, 2, 3, 4)]
    ecols = [f"fluxerr_{s}_{a}fwhm_aper" for s in _BAND_DB.values() for a in (1, 2, 3, 4)]
    mer = pd.read_sql_query(
        f"SELECT object_id, {', '.join(fcols + ecols)} FROM {MER_TABLE} "
        f"WHERE object_id = ANY(%(b)s)", conn, params={"b": q1["object_id"].tolist()})
    conn.close()

    df = q1.merge(mer, on="object_id", how="inner").drop_duplicates("object_id").reset_index(drop=True)
    for band, s in _BAND_DB.items():
        zpc = ZPC_VIS if s == "vis" else 1.0
        for a in (1, 2, 3, 4):
            df[f"{band}_{a}"] = _flux_to_mag(df[f"flux_{s}_{a}fwhm_aper"],
                                             df[f"fluxerr_{s}_{a}fwhm_aper"], zpc)
    df = add_colours(df)
    ok = df.dropna(subset=feats).reset_index(drop=True)
    X = ok[feats].to_numpy(float)
    preds = []
    for p in sorted(MODELS_DIR.glob("xgb_photz_seed*.json")):
        m = xgb.XGBRegressor(); m.load_model(str(p))
        pr = m.predict(X)
        preds.append(np.expm1(pr) if target == "log1p_z" else pr)
    ok["z_xgb"] = np.mean(preds, axis=0)
    for _, b, r in COLORS:
        ok[f"{b}_{r}"] = ok[f"{b}_2"] - ok[f"{r}_2"]
    print(f"  {len(ok)}/{len(q1)} Q1 picks have all 16 bands -> z_xgb assigned")
    return ok[["object_id", "qso_id", "z_xgb", "pm_sig"] + [f"{b}_{r}" for _, b, r in COLORS]], idmap


def _xgbz_rivers(df_desi):
    """DESI spec-QSO running-median colour vs XGBoost photo-z — the solid 'QSO median'
    locus of qso_color_vs_photz.png row 1, reduced to a reference curve. Same x-axis
    (XGB photo-z) as the picked candidates, so it is directly comparable."""
    if not OOF_CSV.exists():
        return None
    oof = pd.read_csv(OOF_CSV)[["qso_id", "z_phot_oof"]].rename(columns={"z_phot_oof": "z_xgb"})
    m = df_desi.merge(oof, on="qso_id", how="inner")
    z = m["z_xgb"].to_numpy(float)
    out = {}
    for _, b, r in COLORS:
        col = f"{b}_{r}"
        c = m[col].to_numpy(float)
        ok = np.isfinite(c) & np.isfinite(z) & (z <= ZMAX_PLOT)
        mid, med, _, _ = _running_stats(z[ok], c[ok])
        out[col] = (np.asarray(mid), np.asarray(med))
    return out


def _draw_row(axrow, panel, picked, cval_col, cmap, norm, grey_label,
              spec_rivers, xgbz_rivers, ylim, extent, dvmin, dvmax):
    """One row of the 3 colour panels: identical rivers background/bands/medians, with
    the picked sources coloured by `cval_col` (missing values drawn grey)."""
    for k, (label, col, z, cc) in enumerate(panel):
        ax = axrow[k]
        ax.hexbin(z, cc, gridsize=100, bins="log", cmap=COLOR_CMAPS[col], mincnt=1,
                  extent=extent, vmin=dvmin, vmax=dvmax, alpha=0.5, zorder=1, rasterized=True)
        mid, med, lo, hi = _running_stats(z, cc)
        ax.fill_between(mid, lo, hi, color=PALETTE[col], alpha=0.22, lw=0, zorder=2,
                        label="candidate 16–84%")
        ln, = ax.plot(mid, med, color=PALETTE[col], lw=2.4, zorder=6, label="candidate median")
        ln.set_path_effects([pe.Stroke(linewidth=4, foreground="white"), pe.Normal()])
        smid, smed = spec_rivers[col]
        sl, = ax.plot(smid, smed, color="k", lw=1.6, ls="--", zorder=5, label="DESI spec-z median")
        sl.set_path_effects([pe.Stroke(linewidth=3, foreground="white"), pe.Normal()])
        if xgbz_rivers is not None:
            xmid, xmed = xgbz_rivers[col]
            xl, = ax.plot(xmid, xmed, color="k", lw=1.6, ls=":", zorder=5,
                          label="DESI spec-QSO XGB-z median")
            xl.set_path_effects([pe.Stroke(linewidth=3, foreground="white"), pe.Normal()])

        # picked sources: marker shape = UMAP cluster, fill colour = cval_col
        for _, s in picked.iterrows():
            zz, ccc = s["z_xgb"], s[col]
            if not (np.isfinite(zz) and np.isfinite(ccc)):
                continue
            cl = int(s["cluster"]) if pd.notna(s["cluster"]) else -1
            mk = CLUSTER_MARKERS.get(cl, "X"); msz = CLUSTER_SIZES.get(cl, 150)
            val = s[cval_col] if pd.notna(s[cval_col]) else np.nan
            fc = "0.6" if not np.isfinite(val) else None   # grey if value missing
            ax.scatter([zz], [ccc], s=msz, marker=mk,
                       c=([val] if fc is None else None), facecolors=fc,
                       cmap=cmap, norm=norm, edgecolors="k", linewidths=0.8, zorder=8)

        ax.set_xlim(0, 3.5); ax.set_ylim(*ylim)
        ax.set_xlabel("XGBoost photo-$z$", fontsize=11)
        ax.set_ylabel(f"{label}  (AB)", fontsize=11)
        ax.set_title(label, fontsize=13, fontweight="bold")
        # curve legend on the 2nd & 3rd panels (1st panel carries the cluster legend)
        if k >= 1:
            ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # marker-shape = cluster legend (colour-neutral) + grey "missing value" entry, panel 0
    clusters_present = sorted({int(c) for c in picked["cluster"].dropna().unique()})
    handles = [Line2D([0], [0], marker=CLUSTER_MARKERS.get(c, "X"), color="w",
                      markerfacecolor="0.6", markeredgecolor="k",
                      markersize=np.sqrt(CLUSTER_SIZES.get(c, 150)) * 0.55,
                      label=f"Cluster {c}")
               for c in clusters_present]
    handles.append(Line2D([0], [0], marker="", color="w", markerfacecolor="0.6",
                          markeredgecolor="k", markersize=7, label=grey_label))
    axrow[0].legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)


def main():
    user = os.getenv("PGUSER", WSDB_USER)

    picked, idmap = predict_picked(user)
    flat = pd.read_csv(FLAT_CSV, dtype={"gaia_source_id": "int64"})
    cl_of = dict(zip(flat["qso_id"], flat["cluster"]))
    pgal_of = dict(zip(flat["qso_id"], flat["model_pgal"]))
    picked["cluster"] = picked["qso_id"].map(cl_of)
    picked["pgal"]    = picked["qso_id"].map(pgal_of)

    pop = pd.read_csv(POP_CACHE)                       # full Q1 candidate colour(z)
    qcsv = ARRAYS_DIR / "rivers_qso_2fwhm.csv"
    df_desi = pd.read_csv(qcsv)
    spec_rivers = _spec_rivers(df_desi)
    xgbz_rivers = _xgbz_rivers(df_desi)

    # shared colour y-range from the DESI reference (same scale as the pipeline plots)
    pooled = []
    for _, b, r in COLORS:
        c = df_desi[f"{b}_{r}"].to_numpy(float); c = c[np.isfinite(c)]
        pooled += list(np.percentile(c, [1, 99]))
    cpad = 0.15 * (max(pooled) - min(pooled))
    ylim = (min(pooled) - cpad, max(pooled) + cpad)

    # ERO picks that can't be placed on the Q1 XGB-photo-z plot
    ero = idmap[idmap["euclid_source"] == "ERO"]["qso_id"].tolist()

    ncol = len(COLORS)
    fig, axes = plt.subplots(2, ncol, figsize=(7.2 * ncol, 6.8 * 2), squeeze=False)
    zp = pop["z_xgb"].to_numpy(float)
    extent = (0, ZMAX_PLOT, ylim[0], ylim[1])

    # shared log-density colour range across the three panels (same as the base plot)
    panel, dvmin, dvmax = [], np.inf, -np.inf
    for label, b, r in COLORS:
        col = f"{b}_{r}"
        c = pop[col].to_numpy(float)
        m = np.isfinite(c) & np.isfinite(zp) & (zp <= ZMAX_PLOT)
        z, cc = zp[m], c[m]
        lo, hi = _hexbin_count_range(z, cc, extent)
        dvmin = min(dvmin, lo); dvmax = max(dvmax, hi)
        panel.append((label, col, z, cc))

    # row 1 = coloured by Gaia PM significance;  row 2 = coloured by XGBoost P(gal)
    _draw_row(axes[0], panel, picked, "pm_sig", PM_CMAP, PM_NORM,
              "no PM (2-param astrometry)", spec_rivers, xgbz_rivers, ylim, extent, dvmin, dvmax)
    _draw_row(axes[1], panel, picked, "pgal", PGAL_CMAP, PGAL_NORM,
              "no P(gal)", spec_rivers, xgbz_rivers, ylim, extent, dvmin, dvmax)

    n_shown = int(picked["z_xgb"].notna().sum())
    fig.suptitle(f"Hand-picked QSO sources on the CatWISE×Euclid×Gaia Q1 colour(z) rivers "
                 f"— {n_shown} Q1 picks placed  ·  marker = UMAP cluster\n"
                 f"row 1 colour = Gaia PM significance   ·   row 2 colour = XGBoost P(gal)   ·   "
                 f"dashed = DESI spec-z median, dotted = DESI spec-QSO XGB-z median",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.94, 0.95])   # reserve top for title, right for colour bars
    # two colour bars in the reserved right margin, each aligned to its own row's axes
    ptop = axes[0][-1].get_position(); pbot = axes[1][-1].get_position()
    sm_pm = plt.cm.ScalarMappable(cmap=PM_CMAP, norm=PM_NORM); sm_pm.set_array([])
    cax1 = fig.add_axes([0.955, ptop.y0, 0.011, ptop.height])
    fig.colorbar(sm_pm, cax=cax1).set_label(PM_LABEL, fontsize=10)
    sm_pg = plt.cm.ScalarMappable(cmap=PGAL_CMAP, norm=PGAL_NORM); sm_pg.set_array([])
    cax2 = fig.add_axes([0.955, pbot.y0, 0.011, pbot.height])
    fig.colorbar(sm_pg, cax=cax2).set_label(PGAL_LABEL, fontsize=10)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> {OUT}  ({n_shown} placed, {len(ero)} ERO not placeable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
