#!/usr/bin/env python3
"""
Variant of apply_photz_catwise_euclid_gaia.py: the same CatWISE×Euclid×Gaia Q1 colour vs
photo-z rivers (row 1 = XGBoost, row 2 = Euclid PHZ mode 1), but each colour panel is a
SCATTER (like qso_by_phz_labels.png) with every point COLOURED BY its Gaia proper-motion
significance (g_pm_significance = |μ|/σ_μ from crossmatch_qso_catwise_euclid_gaia).

Why: real QSOs are extragalactic and have PM consistent with zero (low significance),
while stellar contaminants move (high significance). Colouring the colour(z) locus by PM
significance shows WHERE on the locus the point-source contaminants sit — the colour scale
is centred on 3σ (the usual "significant PM" threshold): blue = QSO-like, red = stellar.

Reuses the cached predictions from apply_photz_catwise_euclid_gaia.py (run that first).

Reads:
  output/qso_color_vs_z/arrays/catwise_euclid_gaia_xgbz.csv  (object_id, z_xgb, colours)
  {PGUSER}.crossmatch_qso_catwise_euclid_gaia            (g_pm_significance, by e_id)
Writes:
  output/qso_color_vs_z/catwise_e_g_color_vs_xgbz_pm.png
  output/qso_color_vs_z/arrays/catwise_euclid_gaia_pm.csv    (PM cache)

Usage:
  PGUSER=siwei_zhao_2025 python src/apply_photz_catwise_euclid_gaia_pm.py    # fetch PM + plot
  python src/apply_photz_catwise_euclid_gaia_pm.py --replot                  # redraw (no WSDB)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_qso_color_vs_specz import (  # noqa: E402
    COLORS, PALETTE, ZMAX_PLOT, WSDB_HOST, WSDB_DB, WSDB_USER, ARRAYS_DIR,
    fetch_pop, _running_stats,
)
from analyze_qso_color_vs_photz import _combined, _spec_rivers  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "qso_color_vs_z" / "arrays"
PNG_PATH    = PROJECT_DIR / "output" / "qso_color_vs_z" / "catwise_e_g_color_vs_xgbz_pm.png"
CACHE_XGBZ  = ARRAYS_DIR / "catwise_euclid_gaia_xgbz.csv"
CACHE_PM    = ARRAYS_DIR / "catwise_euclid_gaia_pm.csv"
CATWISE_TABLE = "crossmatch_qso_catwise_euclid_gaia"
PM_CMAP     = "RdYlBu_r"
PM_NORM     = TwoSlopeNorm(vmin=0.0, vcenter=3.0, vmax=10.0)   # white at the 3σ PM threshold


def fetch_pm(user: str) -> pd.DataFrame:
    """Gaia PM significance for the CatWISE×Gaia Q1 Euclid sources (keyed by e_id)."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30,
                            keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5)
    df = pd.read_sql_query(
        f"""SELECT DISTINCT ON (e_id) e_id AS object_id, g_pm_significance AS pm_sig
            FROM {user}.{CATWISE_TABLE}
            WHERE euclid_source = 'Q1' AND e_id IS NOT NULL ORDER BY e_id""", conn)
    conn.close()
    print(f"  {df['pm_sig'].notna().sum():,} of {len(df):,} sources have a Gaia PM significance")
    return df


# ------------------------------------------------------------------------- plot
def _draw_row(fig, axrow, df, zcol, xlabel, pop, ylim, spec_rivers, show_poplegend):
    """One 4-panel row: combined locus + 3 colour SCATTER panels coloured by PM significance."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    z_all = df[zcol].to_numpy(float)
    pm_all = df["pm_sig"].to_numpy(float)
    _combined(axrow[0], df, zcol, xlabel, ylim, spec_rivers)
    sc = None
    for k, (label, b, r) in enumerate(COLORS):
        col = f"{b}_{r}"
        ax = axrow[1 + k]
        c = df[col].to_numpy(float)
        m = np.isfinite(c) & np.isfinite(z_all) & (z_all <= ZMAX_PLOT) & np.isfinite(pm_all)
        z, cc, pm = z_all[m], c[m], pm_all[m]
        div = make_axes_locatable(ax)
        axL = div.append_axes("left", size="27%", pad=0.10, sharey=ax)
        # scatter coloured by PM significance; draw high-PM (contaminants) on top
        order = np.argsort(pm)
        sc = ax.scatter(z[order], cc[order], c=pm[order], s=11, alpha=0.75, lw=0,
                        cmap=PM_CMAP, norm=PM_NORM, rasterized=True)
        # dashed spec-z median (deeper) + candidate running median (on top)
        smid, smed = spec_rivers[col]
        sl, = ax.plot(smid, smed, color="k", lw=1.6, ls="--", label="spec-z median", zorder=6)
        sl.set_path_effects([pe.Stroke(linewidth=3, foreground="white"), pe.Normal()])
        mid, med, lo, hi = _running_stats(z, cc); w = float(np.median(hi - lo))
        ln, = ax.plot(mid, med, color=PALETTE[col], lw=2.4, label="candidate median", zorder=7)
        ln.set_path_effects([pe.Stroke(linewidth=4, foreground="white"), pe.Normal()])
        ax.text(0.97, 0.03, f"N = {m.sum():,}\n16–84% width = {w:.2f} mag",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.92))
        ax.set_xlim(0, 3.5); ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel(""); ax.tick_params(labelleft=False)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        # left marginal = colour distribution (candidates vs full Q1 population), raw counts
        qc = cc[(cc >= ylim[0]) & (cc <= ylim[1])]
        mbins = np.linspace(ylim[0], ylim[1], 56)
        if pop is not None and col in pop:
            pcv = pop[col].to_numpy(float); pcv = pcv[np.isfinite(pcv) & (pcv >= ylim[0]) & (pcv <= ylim[1])]
            if len(pcv) > 20:
                axL.hist(pcv, bins=mbins, orientation="horizontal", color="0.72", alpha=0.55, histtype="stepfilled")
                axL.hist(pcv, bins=mbins, orientation="horizontal", color="0.40", lw=1.0, histtype="step")
        axL.hist(qc, bins=mbins, orientation="horizontal", color=PALETTE[col], alpha=0.40, histtype="stepfilled")
        axL.hist(qc, bins=mbins, orientation="horizontal", color=PALETTE[col], lw=1.5, histtype="step")
        axL.invert_xaxis(); axL.set_xticks([]); axL.set_ylim(*ylim)
        axL.set_ylabel(f"{label}  (AB)", fontsize=11); axL.tick_params(labelbottom=False, bottom=False)
        if k == 0 and show_poplegend:
            from matplotlib.lines import Line2D
            axL.legend([Line2D([0], [0], color="0.4", lw=2), Line2D([0], [0], color=PALETTE[col], lw=2)],
                       ["Q1 population", "candidates"], loc="lower left", fontsize=7, framealpha=0.85)
    return sc


def make_plot(df_xgb, pop, spec_rivers, out: Path, ylim):
    ncol = 1 + len(COLORS)
    fig, axes = plt.subplots(1, ncol, figsize=(6.8 * ncol, 6.8))
    sc = _draw_row(fig, axes, df_xgb, "z_xgb", "XGBoost photo-$z$", pop, ylim, spec_rivers, True)
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), location="right",
                        fraction=0.015, pad=0.02, extend="max")
    cbar.set_label("Gaia PM significance  |μ| / σ$_μ$  (white = 3σ threshold)", fontsize=11)
    fig.suptitle(
        f"CatWISE×Euclid×Gaia Q1 QSO candidates: colour vs XGBoost photo-z, coloured by Gaia PM significance  (N={len(df_xgb):,})\n"
        f"blue = QSO-like (no PM), red = stellar (moving);  dashed = DESI spec-z QSO median",
        fontsize=15, fontweight="bold", y=0.99)
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

    if not CACHE_XGBZ.exists():
        print("  missing catwise_euclid_gaia_xgbz.csv — run apply_photz_catwise_euclid_gaia.py first.")
        return 1
    df_cat = pd.read_csv(CACHE_XGBZ)

    if CACHE_PM.exists():
        pm = pd.read_csv(CACHE_PM); print(f"  loaded {len(pm):,} PM rows from cache")
    elif args.replot:
        print("  --replot needs catwise_euclid_gaia_pm.csv — run once without it."); return 1
    else:
        pm = fetch_pm(user); pm.to_csv(CACHE_PM, index=False); print(f"  cached -> {CACHE_PM}")

    df_xgb = df_cat.merge(pm, on="object_id", how="left")

    # DESI spec-z reference rivers + Q1 population marginals + shared colour y-range
    qcsv = ARRAYS_DIR / "rivers_qso_2fwhm.csv"; pcsv = ARRAYS_DIR / "rivers_pop_2fwhm.csv"
    if qcsv.exists():
        df_desi = pd.read_csv(qcsv)
    else:
        from analyze_qso_color_vs_specz import build_dataframe
        df_desi = build_dataframe(user, 2)
    pop = pd.read_csv(pcsv) if pcsv.exists() else fetch_pop(user, 2)
    spec_rivers = _spec_rivers(df_desi)
    pooled = []
    for _, b, r in COLORS:
        c = df_desi[f"{b}_{r}"].to_numpy(float); c = c[np.isfinite(c)]
        pooled += list(np.percentile(c, [1, 99]))
    cpad = 0.15 * (max(pooled) - min(pooled)); ylim = (min(pooled) - cpad, max(pooled) + cpad)

    print(f"  XGB row: {len(df_xgb):,}  (coloured where PM available)")
    make_plot(df_xgb, pop, spec_rivers, PNG_PATH, ylim)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
