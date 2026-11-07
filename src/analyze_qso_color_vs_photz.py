#!/usr/bin/env python3
"""
Companion to analyze_qso_color_vs_specz.py: the Euclid colour(z) "rivers" plot, but with
the x-axis redshift replaced by a PREDICTED z instead of the spectroscopic z. Two rows:

  row 1 : colour vs XGBoost photo-z      (our regressor, output/xgboost_qso_photz/oof.csv)
  row 2 : colour vs Euclid PHZ mode_1    (official template photo-z, q1_phz_photo_z)

Same sources, same 2×FWHM Euclid colours and same full-Q1 population marginals as the
spec-z rivers plot — only the horizontal axis changes. This shows how well each photo-z
"unfolds" the emission-line wiggles: if a photo-z is good, the coloured rivers stay as
narrow and wiggly as in the spec-z version; if it is degenerate (e.g. galaxy-template
PHZ, see analyze_qso_phz_diagnostics.py) the rivers smear and the wiggles wash out.

The colours per QSO come from analyze_qso_color_vs_specz.build_dataframe (2FWHM), the
XGB z from the training OOF file, and the PHZ z is bridged qso_id -> e_id -> object_id
via analyze_qso_photz_vs_specz.fetch_phz — exactly how qso_photz_vs_specz.png pulls them.

Reads (WSDB; Astronomy/Cambridge VPN; auth via ~/.pgpass):
  {PGUSER}.crossmatch_qso_desi_dr1_euclid, euclid_q1.mer_final_cat   (colours + population)
  {PGUSER}.q1_phz_photo_z                                            (PHZ mode_1)
  output/xgboost_qso_photz/oof.csv                                   (XGB OOF photo-z)
Writes:
  output/qso_color_vs_z/qso_color_vs_photz.png
  output/qso_color_vs_z/arrays/phz_colorz.csv                        (PHZ cache, --replot)

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_qso_color_vs_photz.py     # pull + plot (caches)
  python src/analyze_qso_color_vs_photz.py --replot                   # redraw from caches (no WSDB)
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
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_qso_color_vs_specz import (  # noqa: E402
    COLORS, PALETTE, COLOR_CMAPS, ZMAX_PLOT, WSDB_USER, ARRAYS_DIR,
    build_dataframe, fetch_pop, _running_stats, _hexbin_count_range,
)
from analyze_qso_photz_vs_specz import fetch_phz  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
PNG_PATH    = PROJECT_DIR / "output" / "qso_color_vs_z" / "qso_color_vs_photz.png"
OOF_CSV     = PROJECT_DIR / "output" / "xgboost_qso_photz" / "oof.csv"
PHZ_CACHE   = PROJECT_DIR / "output" / "qso_color_vs_z" / "arrays" / "phz_colorz.csv"
APER        = 2   # 2×FWHM canonical (matches the trained photo-z features)


# ----------------------------------------------------------------------------- plot
def _spec_rivers(df_col) -> dict:
    """Spec-z running median per colour (the ORIGINAL rivers) — a fixed reference to
    overlay on every photo-z panel. Built from qso_z, so identical for both rows."""
    out = {}
    z = df_col["qso_z"].to_numpy(float)
    for _, b, r in COLORS:
        col = f"{b}_{r}"
        c = df_col[col].to_numpy(float)
        m = np.isfinite(c) & np.isfinite(z) & (z <= ZMAX_PLOT)
        mid, med, _, _ = _running_stats(z[m], c[m])
        out[col] = (np.asarray(mid), np.asarray(med))
    return out


def _combined(ax, df, zcol, xlabel, ylim, spec_rivers=None):
    """All three colour(z) rivers (μ ± σ, solid) overlaid — the row's overview panel."""
    z = df[zcol].to_numpy(float)
    for label, b, r in COLORS:
        col = f"{b}_{r}"
        c = df[col].to_numpy(float)
        m = np.isfinite(c) & np.isfinite(z) & (z <= ZMAX_PLOT)
        mid, med, lo, hi = _running_stats(z[m], c[m])
        ax.fill_between(mid, lo, hi, color=PALETTE[col], alpha=0.20, lw=0)
        ax.plot(mid, med, color=PALETTE[col], lw=1.9, label=label)
    ax.set_xlim(0, 3.5); ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel("Euclid colour (AB)", fontsize=11)
    ax.set_title("QSO colour(z) locus  μ ± σ", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)


def _draw_row(fig, axrow, df, zcol, xlabel, pop, ylim, show_poplegend, spec_rivers):
    """One 4-panel row: combined locus + 3 colour density panels with left marginals.
    Mirrors analyze_qso_color_vs_specz.make_plot, but colour is plotted against df[zcol].
    The original spec-z median (spec_rivers) is overlaid as a dashed black reference."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    z_all = df[zcol].to_numpy(float)
    extent = (0, ZMAX_PLOT, ylim[0], ylim[1])

    panel_data, dens_vmin, dens_vmax = [], np.inf, -np.inf
    for label, b, r in COLORS:
        col = f"{b}_{r}"
        c = df[col].to_numpy(float)
        m = np.isfinite(c) & np.isfinite(z_all) & (z_all <= ZMAX_PLOT)
        z, cc = z_all[m], c[m]
        lo, hi = _hexbin_count_range(z, cc, extent)
        dens_vmin = min(dens_vmin, lo); dens_vmax = max(dens_vmax, hi)
        panel_data.append((label, col, z, cc, m))

    _combined(axrow[0], df, zcol, xlabel, ylim, spec_rivers)
    for k, (label, col, z, cc, m) in enumerate(panel_data):
        ax = axrow[1 + k]
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="4.5%", pad=0.06)
        axL = div.append_axes("left", size="27%", pad=0.10, sharey=ax)
        hb = ax.hexbin(z, cc, gridsize=100, bins="log", cmap=COLOR_CMAPS[col], mincnt=1,
                       extent=extent, vmin=dens_vmin, vmax=dens_vmax)
        cb = fig.colorbar(hb, cax=cax)
        cb.set_label("QSO density (log)", fontsize=8, labelpad=0, loc="center"); cb.ax.tick_params(labelsize=7)
        mid, med, lo, hi = _running_stats(z, cc)
        w = float(np.median(hi - lo))
        # original spec-z median as a dashed black reference, on a DEEPER layer so the
        # solid photo-z median stays on top (white-haloed for legibility over the density)
        smid, smed = spec_rivers[col]
        sl, = ax.plot(smid, smed, color="k", lw=1.6, ls="--", label="spec-z median", zorder=4)
        sl.set_path_effects([pe.Stroke(linewidth=3, foreground="white"), pe.Normal()])
        ln, = ax.plot(mid, med, color=PALETTE[col], lw=2.4, label="QSO median", zorder=7)
        ln.set_path_effects([pe.Stroke(linewidth=4, foreground="white"), pe.Normal()])
        ax.text(0.97, 0.03, f"N = {m.sum():,}\nQSO 16–84% width = {w:.2f} mag",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.92))
        ax.set_xlim(0, 3.5); ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(""); ax.tick_params(labelleft=False)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        # left marginal = colour distribution, RAW counts on a single shared LINEAR axis
        qc = cc[(cc >= ylim[0]) & (cc <= ylim[1])]
        mbins = np.linspace(ylim[0], ylim[1], 56)
        if pop is not None and col in pop:
            pc = pop[col].to_numpy(float); pc = pc[np.isfinite(pc) & (pc >= ylim[0]) & (pc <= ylim[1])]
            if len(pc) > 20:
                axL.hist(pc, bins=mbins, orientation="horizontal", color="0.72", alpha=0.55, histtype="stepfilled")
                axL.hist(pc, bins=mbins, orientation="horizontal", color="0.40", lw=1.0, histtype="step")
        axL.hist(qc, bins=mbins, orientation="horizontal", color=PALETTE[col], alpha=0.40, histtype="stepfilled")
        axL.hist(qc, bins=mbins, orientation="horizontal", color=PALETTE[col], lw=1.5, histtype="step")
        axL.invert_xaxis(); axL.set_xticks([]); axL.set_ylim(*ylim)
        axL.set_ylabel(f"{label}  (AB)", fontsize=11)
        axL.tick_params(labelbottom=False, bottom=False)
        if k == 0 and show_poplegend:
            from matplotlib.lines import Line2D
            axL.legend([Line2D([0], [0], color="0.4", lw=2), Line2D([0], [0], color=PALETTE[col], lw=2)],
                       ["Q1 population", "spec QSO"], loc="lower left", fontsize=7, framealpha=0.85)


def make_plot(df_xgb, df_phz, pop, out: Path, ylim, spec_rivers):
    ncol = 1 + len(COLORS)
    fig, axes = plt.subplots(2, ncol, figsize=(6.8 * ncol, 6.8 * 2))
    _draw_row(fig, axes[0], df_xgb, "z_xgb", "XGBoost photo-$z$",
              pop, ylim, show_poplegend=True, spec_rivers=spec_rivers)
    _draw_row(fig, axes[1], df_phz, "z_phz", "Euclid PHZ mode 1 photo-$z$",
              pop, ylim, show_poplegend=False, spec_rivers=spec_rivers)
    fig.suptitle(
        f"DESI×Euclid Q1 QSO colour vs PREDICTED redshift  ·  {APER}×FWHM aperture photometry\n"
        f"row 1: XGBoost photo-z (N={len(df_xgb):,})   ·   row 2: Euclid PHZ mode 1 (N={len(df_phz):,})"
        f"   —   dashed = DESI spec-z QSO median (known locus)",
        fontsize=15, fontweight="bold", y=0.97)
    try:
        fig.tight_layout(rect=[0, 0, 1, 0.94])
    except Exception:
        fig.subplots_adjust(top=0.92, wspace=0.4, hspace=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replot", action="store_true",
                    help="redraw from cached colours/population/PHZ (no WSDB).")
    ap.add_argument("--out", default=str(PNG_PATH), help="output figure path.")
    args = ap.parse_args()
    user = os.getenv("PGUSER", WSDB_USER)

    # --- colours (2FWHM) + full-Q1 population marginals: reuse the spec-z rivers cache ---
    qcsv = ARRAYS_DIR / f"rivers_qso_{APER}fwhm.csv"
    pcsv = ARRAYS_DIR / f"rivers_pop_{APER}fwhm.csv"
    if args.replot:
        if not (qcsv.exists() and pcsv.exists()):
            print(f"  no colour cache ({qcsv.name}) — run analyze_qso_color_vs_specz.py once first."); return 1
        df_col = pd.read_csv(qcsv); pop = pd.read_csv(pcsv)
        print(f"  loaded {len(df_col):,} QSO colours + {len(pop):,} population from cache")
    else:
        df_col = build_dataframe(user, APER)
        pop = fetch_pop(user, APER)
        ARRAYS_DIR.mkdir(parents=True, exist_ok=True)
        df_col.to_csv(qcsv, index=False); pop.to_csv(pcsv, index=False)

    # --- XGB OOF photo-z ---
    if not OOF_CSV.exists():
        print(f"  missing {OOF_CSV} — run train_xgboost_qso_photz.py first."); return 1
    oof = pd.read_csv(OOF_CSV)[["qso_id", "z_phot_oof"]].rename(columns={"z_phot_oof": "z_xgb"})

    # --- Euclid PHZ mode_1 (bridged qso_id -> e_id -> object_id) ---
    phz = fetch_phz("" if args.replot else user, df_col["qso_id"].to_numpy(), cache=PHZ_CACHE)
    if phz is None or not len(phz):
        print("  no PHZ available (need VPN on first run) — aborting."); return 1
    phz = phz.rename(columns={"phz_mode_1": "z_phz"})

    df_xgb = df_col.merge(oof, on="qso_id", how="inner")
    df_phz = df_col.merge(phz, on="qso_id", how="inner")
    df_phz = df_phz[df_phz["z_phz"].notna()]
    print(f"  XGB row: {len(df_xgb):,} QSOs   ·   PHZ row: {len(df_phz):,} QSOs")

    # shared colour y-range across both rows (union of the three colours' 1–99 pct)
    pooled = []
    for _, b, r in COLORS:
        c = df_col[f"{b}_{r}"].to_numpy(float); c = c[np.isfinite(c)]
        pooled += list(np.percentile(c, [1, 99]))
    cpad = 0.15 * (max(pooled) - min(pooled))
    ylim = (min(pooled) - cpad, max(pooled) + cpad)

    spec_rivers = _spec_rivers(df_col)   # original spec-z wiggles, shared by both rows
    make_plot(df_xgb, df_phz, pop, Path(args.out), ylim, spec_rivers)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
