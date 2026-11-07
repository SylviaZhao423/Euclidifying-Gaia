#!/usr/bin/env python3
"""TEMPORARY — project the Euclid-PHZ "star" sample onto the colour(z) rivers.

Same canvas as picked_qso_color_vs_xgbz.png (candidate colour(z) log-density background,
coloured 16-84% band, candidate median, DESI spec-z dashed + DESI spec-QSO XGB-z dotted
references), but instead of the hand-picks we over-plot every CatWISE×Euclid×Gaia Q1
source that Euclid PHZ classifies as a STAR (p_phz_classification bitmask = 1).

These are the point-source contaminants: if the QSO colour(z) selection is clean they
should sit OFF the QSO rivers. z_xgb + 2FWHM colours are reused straight from the cached
candidate table (catwise_euclid_gaia_xgbz.csv) — the only DB hit is the star id list.

Reads : {PGUSER}.crossmatch_qso_catwise_euclid_gaia  x  crossmatch_q1_mer_phz_vis
Writes: output/qso_color_vs_z/arrays/phz_star_ids.csv   (object_id, phz_star_prob cache)
        output/picked_qso_phz_stars_color_vs_xgbz.png

Usage:
  PGUSER=siwei_zhao_2025 python src/tmp_phz_stars_color_vs_z.py            # fetch ids + plot
  python src/tmp_phz_stars_color_vs_z.py --replot                         # redraw from caches
  PGUSER=... python src/tmp_phz_stars_color_vs_z.py --label argmax        # argmax star instead of bitmask==1
"""
import os
import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_qso_color_vs_specz import (COLORS, PALETTE, COLOR_CMAPS, ZMAX_PLOT,
                                        _running_stats, _hexbin_count_range,
                                        WSDB_HOST, WSDB_DB, WSDB_USER)
from analyze_qso_color_vs_photz import _spec_rivers
from tmp_picked_color_vs_z import POP_CACHE, ARRAYS_DIR, _xgbz_rivers

PROJECT_DIR = Path(__file__).resolve().parents[1]
CATWISE_TABLE = "crossmatch_qso_catwise_euclid_gaia"
PHZ_TABLE     = "crossmatch_q1_mer_phz_vis"

# PHZ p_phz_classification is a bitmask: bit0(1)=star, bit1(2)=galaxy, bit2(4)=QSO.
CLASS_BIT     = {"star": 1, "galaxy": 2, "qso": 4}
CLASS_PROBCOL = {"star": "phz_star", "galaxy": "phz_gal", "qso": "phz_qso"}
CLASS_ARGMAX  = {"galaxy": 0, "qso": 1, "star": 2}   # matches [phz_gal, phz_qso, phz_star]
PROB_CMAP     = "RdYlBu_r"
PROB_NORM     = plt.Normalize(0.5, 1.0)              # class probability of the selected sources


def fetch_phzclass(user, label_mode, cls):
    """Q1 CatWISE×Euclid×Gaia sources PHZ assigns to class `cls`, with that class prob."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30,
                            keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5)
    df = pd.read_sql_query(
        f"""SELECT DISTINCT ON (c.e_id) c.e_id AS object_id,
                   p.p_phz_star_prob AS phz_star,
                   p.p_phz_gal_prob  AS phz_gal,
                   p.p_phz_qso_prob  AS phz_qso,
                   p.p_phz_classification AS bm
            FROM {user}.{CATWISE_TABLE} c
            JOIN {user}.{PHZ_TABLE} p ON p.m_object_id = c.e_id
            WHERE c.euclid_source = 'Q1' AND c.e_id IS NOT NULL
            ORDER BY c.e_id""", conn)
    conn.close()
    if label_mode == "argmax":
        probs = df[["phz_gal", "phz_qso", "phz_star"]].to_numpy(float)
        sel = np.argmax(probs, axis=1) == CLASS_ARGMAX[cls]
    else:                                  # bitmask == class bit (the literal phz=N label)
        sel = df["bm"] == CLASS_BIT[cls]
    out = df.loc[sel, ["object_id", CLASS_PROBCOL[cls]]].rename(
        columns={CLASS_PROBCOL[cls]: "phz_prob"}).reset_index(drop=True)
    print(f"  PHZ-labelled ({label_mode}) {cls} in the Q1 crossmatch: {len(out):,} "
          f"/ {len(df):,} with a PHZ label")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true", help="redraw from caches only (no WSDB).")
    ap.add_argument("--label", choices=["bitmask", "argmax"], default="bitmask",
                    help="'bitmask' = p_phz_classification==bit (default), or 'argmax' of the 3 probs.")
    ap.add_argument("--phz", choices=["star", "galaxy", "qso"], default="star",
                    help="which PHZ class to overplot (star=phz 1, galaxy=phz 2, qso=phz 4).")
    args = ap.parse_args()
    user = os.getenv("PGUSER", WSDB_USER)
    cls = args.phz
    cache = ARRAYS_DIR / f"phz_{cls}_ids.csv"
    out   = PROJECT_DIR / "output" / f"picked_qso_phz_{cls}_color_vs_xgbz.png"

    if args.replot:
        if not cache.exists():
            print(f"  --replot needs {cache.name} — run once without it."); return 1
        sel_id = pd.read_csv(cache)
        print(f"  loaded {len(sel_id):,} PHZ-{cls} ids from cache")
    else:
        sel_id = fetch_phzclass(user, args.label, cls)
        cache.parent.mkdir(parents=True, exist_ok=True)
        sel_id.to_csv(cache, index=False); print(f"  cached -> {cache}")

    pop = pd.read_csv(POP_CACHE)                    # full Q1 candidate colour(z) (z_xgb + colours)
    stars = pop.merge(sel_id, on="object_id", how="inner")
    print(f"  {len(stars):,} of the PHZ-{cls} have cached z_xgb + colours (plotted)")

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
    zp = pop["z_xgb"].to_numpy(float)
    extent = (0, ZMAX_PLOT, ylim[0], ylim[1])

    panel, dvmin, dvmax = [], np.inf, -np.inf
    for label, b, r in COLORS:
        col = f"{b}_{r}"
        c = pop[col].to_numpy(float)
        m = np.isfinite(c) & np.isfinite(zp) & (zp <= ZMAX_PLOT)
        z, cc = zp[m], c[m]
        lo, hi = _hexbin_count_range(z, cc, extent)
        dvmin = min(dvmin, lo); dvmax = max(dvmax, hi)
        panel.append((label, col, z, cc))

    zs = stars["z_xgb"].to_numpy(float)
    sc = None
    for k, (label, col, z, cc) in enumerate(panel):
        ax = axes[k]
        ax.hexbin(z, cc, gridsize=100, bins="log", cmap=COLOR_CMAPS[col], mincnt=1,
                  extent=extent, vmin=dvmin, vmax=dvmax, alpha=0.45, zorder=1, rasterized=True)
        mid, med, lo, hi = _running_stats(z, cc)
        ax.fill_between(mid, lo, hi, color=PALETTE[col], alpha=0.20, lw=0, zorder=2,
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

        # PHZ-class sample: scatter (coloured by class prob) + their own running median
        cs = stars[col].to_numpy(float)
        ms = np.isfinite(zs) & np.isfinite(cs) & (zs <= ZMAX_PLOT)
        sc = ax.scatter(zs[ms], cs[ms], s=14, c=stars["phz_prob"].to_numpy(float)[ms],
                        cmap=PROB_CMAP, norm=PROB_NORM, alpha=0.8, lw=0.2,
                        edgecolors="k", zorder=7)
        smid2, smed2, slo2, shi2 = _running_stats(zs[ms], cs[ms])
        stl, = ax.plot(smid2, smed2, color="k", lw=2.2, zorder=8, label=f"PHZ-{cls} median")
        stl.set_path_effects([pe.Stroke(linewidth=4, foreground="white"), pe.Normal()])

        ax.set_xlim(0, 3.5); ax.set_ylim(*ylim)
        ax.set_xlabel("XGBoost photo-$z$", fontsize=11)
        ax.set_ylabel(f"{label}  (AB)", fontsize=11)
        ax.set_title(label, fontsize=13, fontweight="bold")
        if k >= 1:
            ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    n = int(np.isfinite(zs).sum())
    fig.suptitle(f"Euclid-PHZ {cls} sample on the CatWISE×Euclid×Gaia Q1 colour(z) rivers "
                 f"— {n:,} PHZ {cls} ({args.label})\n"
                 f"background = candidate colour(z) log-density; solid black = PHZ-{cls} median; "
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
