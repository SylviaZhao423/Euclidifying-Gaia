#!/usr/bin/env python3
"""
How does the Euclid Q1 PHZ classifier label the KNOWN spectroscopic QSOs, and where do
those labels sit in Euclid colour(z) space?

Motivation
----------
There is no simple star/galaxy/QSO label — that is the premise of the whole project. So
rather than invent an ad-hoc "star" cut, we use the classification we DO trust, Euclid
Q1 PHZ (crossmatch_q1_mer_phz_vis), and ask two things of the 2,843 DESI DR1 spec QSOs:
  1. What does PHZ call them? (spoiler: ~73% "galaxy", ~25% "QSO", ~1% "star" — even the
     trusted classifier does not cleanly recover QSOs.)
  2. Do the PHZ labels separate in the Euclid colour(z) plane, or do they overlap the
     QSO locus (which they must, since every one of these IS a spectroscopic QSO)?

PHZ classification
------------------
p_phz_classification is a BITMASK: bit0(1)=star, bit1(2)=galaxy, bit2(4)=QSO (so 6 =
galaxy+QSO, etc.). We label each source by the ARGMAX of the three class probabilities
(p_phz_gal_prob / p_phz_qso_prob / p_phz_star_prob) — a single, defensible best-guess
label. PHZ is Q1-only, so the 72 ERO + any unmatched spec QSOs have no PHZ label.

Reads:
  (none — spec-QSO colours+z built live via analyze_qso_color_vs_specz.build_dataframe())    (colours + spec z; from Step 1)
  {PGUSER}.crossmatch_q1_mer_phz_vis  joined by e_id = m_object_id (WSDB, VPN)
Writes:
  output/qso_color_vs_z/arrays/qso_by_phz_labels.csv (qso_id, probs, argmax label — PHZ cache)
  output/qso_color_vs_z/qso_by_phz_labels[_Nfwhm].png (colour(z) by PHZ label + fractions;
    swept over 1/2/3/4 FWHM apertures — 2FWHM keeps the canonical name, others get _Nfwhm)

Default is 2×FWHM (the pipeline canonical). --aperture N selects a single aperture;
--all-apertures sweeps 1/2/3/4. The PHZ classification is aperture-independent (fetched once).

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_qso_by_phz_labels.py               # 2FWHM (default)
  PGUSER=siwei_zhao_2025 python src/analyze_qso_by_phz_labels.py --aperture 3
  PGUSER=siwei_zhao_2025 python src/analyze_qso_by_phz_labels.py --all-apertures  # 1..4
  python src/analyze_qso_by_phz_labels.py --replot                             # cached PHZ labels
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
from analyze_qso_color_vs_specz import (COLORS, ZMAX_PLOT, PALETTE, build_dataframe,  # noqa: E402
                                        build_dataframe_aper, _running_stats, fetch_pop,
                                        WSDB_HOST, WSDB_DB, WSDB_USER)

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR   = PROJECT_DIR / "output" / "qso_color_vs_z"
ARRAYS_DIR = OUT_DIR / "arrays"
PHZ_CSV   = ARRAYS_DIR / "qso_by_phz_labels.csv"
PNG_PATH  = OUT_DIR / "qso_by_phz_labels.png"
APERS = [1, 2, 3, 4]                              # Euclid aperture sweep (FWHM diameters)

# Grid rows = OFFICIAL Euclid PHZ classification (p_phz_classification bitmask): the three
# populated classes. Others (-1,0,1,3,5 = 87 objects total) are too few to plot.
PHZ_ROWS = [(2, "galaxy"), (6, "gal+QSO"), (4, "QSO")]


def _png_for(aper: int) -> Path:
    return PNG_PATH if aper == 2 else OUT_DIR / f"qso_by_phz_labels_{aper}fwhm.png"


def fetch_phz(user: str) -> pd.DataFrame:
    """PHZ class probabilities for DESI spec QSOs, joined e_id -> m_object_id (Q1 only)."""
    import psycopg2
    sql = f"""
        SELECT q.qso_id,
               p.p_phz_gal_prob  AS phz_gal,
               p.p_phz_qso_prob  AS phz_qso,
               p.p_phz_star_prob AS phz_star,
               p.p_phz_classification AS phz_bitmask
        FROM {user}.crossmatch_qso_desi_dr1_euclid q
        JOIN {user}.crossmatch_q1_mer_phz_vis p ON p.m_object_id = q.e_id
        WHERE q.euclid_source = 'Q1'
    """
    print("  reading PHZ labels (join e_id -> m_object_id) ...")
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                            connect_timeout=30, keepalives=1, keepalives_idle=30,
                            keepalives_interval=10, keepalives_count=5)
    df = pd.read_sql_query(sql, conn); conn.close()
    probs = df[["phz_gal", "phz_qso", "phz_star"]].to_numpy(float)
    df["phz_label"] = np.array(["galaxy", "QSO", "star"])[np.argmax(probs, axis=1)]
    PHZ_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PHZ_CSV, index=False)
    print(f"    {len(df):,} spec QSOs with a PHZ label -> {PHZ_CSV}")
    return df


def _adaptive_width(z):
    """z-bin width targeting ~15 pts/bin, so a sparse class (few hundred pts) still gets a
    median spanning its whole z-range instead of only the locally-dense middle."""
    z = z[np.isfinite(z)]
    if len(z) < 4:
        return 0.3
    span = max(0.5, np.percentile(z, 98) - np.percentile(z, 2))
    return float(np.clip(15 * span / len(z), 0.05, 0.35))


def make_plot(df: pd.DataFrame, out: Path, pop=None, aper: int = 2):
    """Grid: rows = PHZ argmax class, columns = [combined locus, I-Y, Y-J, J-H] (the
    qso_color_vs_specz.png format). Each colour panel: that class's scatter + its own
    median (colour) + the overall spec-QSO median (black dashed) reference, and a LEFT
    MARGINAL of the colour distribution (Q1 population grey + this PHZ class coloured)."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    z_all = df["qso_z"].to_numpy(float)
    rows = [(v, L) for v, L in PHZ_ROWS if (df["phz_bitmask"] == v).any()]
    ncols = 1 + len(COLORS)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(5.6 * ncols, 4.6 * len(rows)),
                             squeeze=False)

    # ONE y-range shared by every panel (union of the three colours' 1–99 percentiles)
    pooled = []
    for _, b, r in COLORS:
        c = df[f"{b}_{r}"].to_numpy(float); c = c[np.isfinite(c)]
        pooled += list(np.percentile(c, [1, 99]))
    cpad = 0.15 * (max(pooled) - min(pooled)); YLIM = (min(pooled) - cpad, max(pooled) + cpad)

    # overall spec-QSO median per colour (the black dashed reference)
    ref = {}
    for _, b, r in COLORS:
        col = f"{b}_{r}"; c = df[col].to_numpy(float); mm = np.isfinite(c) & (z_all <= ZMAX_PLOT)
        ref[col] = tuple(_running_stats(z_all[mm], c[mm]))   # materialise (reused across rows)

    for i, (val, L) in enumerate(rows):
        sub = df[df["phz_bitmask"] == val]; zc = sub["qso_z"].to_numpy(float); n = len(sub)
        w_bin = _adaptive_width(zc)                           # wider bins for sparser classes
        # column 0: this class's combined colour(z) rivers
        ax = axes[i][0]
        for label, b, r in COLORS:
            col = f"{b}_{r}"; c = sub[col].to_numpy(float); mm = np.isfinite(c) & (zc <= ZMAX_PLOT)
            mid, med, lo, hi = _running_stats(zc[mm], c[mm], width=w_bin, minpts=5)
            ax.fill_between(mid, lo, hi, color=PALETTE[col], alpha=0.20, lw=0)
            ax.plot(mid, med, color=PALETTE[col], lw=1.8, label=label)
        ax.set_xlim(0, 3.5); ax.set_ylim(*YLIM)
        ax.set_ylabel(f"PHZ {L} (class {val}, N={n:,})\nEuclid colour (AB)", fontsize=11, fontweight="bold")
        if i == 0:
            ax.set_title("QSO colour(z) locus", fontsize=12, fontweight="bold")
        if i == len(rows) - 1:
            ax.set_xlabel("spectroscopic $z$", fontsize=11)
        ax.legend(fontsize=8)
        # columns 1..: per-colour density + class median + overall reference median
        for k, (label, b, r) in enumerate(COLORS):
            col = f"{b}_{r}"; c = sub[col].to_numpy(float)
            mm = np.isfinite(c) & np.isfinite(zc) & (zc <= ZMAX_PLOT)
            axk = axes[i][1 + k]
            # ---- left marginal: colour distribution — Q1 population (grey) + this PHZ class (coloured),
            # raw counts, single shared LINEAR axis (like the qso_color_vs_specz.py marginals) ----
            div = make_axes_locatable(axk)
            axL = div.append_axes("left", size="26%", pad=0.08, sharey=axk)
            mbins = np.linspace(YLIM[0], YLIM[1], 48)
            if pop is not None and col in pop:
                pc = pd.to_numeric(pop[col], errors="coerce").to_numpy()
                pc = pc[np.isfinite(pc) & (pc >= YLIM[0]) & (pc <= YLIM[1])]
                if len(pc) > 20:
                    axL.hist(pc, bins=mbins, orientation="horizontal", color="0.72", alpha=0.45, histtype="stepfilled")
                    axL.hist(pc, bins=mbins, orientation="horizontal", color="0.45", lw=0.8, histtype="step")
            ccm = c[mm]; ccm = ccm[(ccm >= YLIM[0]) & (ccm <= YLIM[1])]
            axL.hist(ccm, bins=mbins, orientation="horizontal", color=PALETTE[col], alpha=0.50, histtype="stepfilled")
            axL.hist(ccm, bins=mbins, orientation="horizontal", color=PALETTE[col], lw=1.4, histtype="step")
            axL.invert_xaxis(); axL.set_xticks([]); axL.set_ylim(*YLIM)
            axL.set_ylabel(f"{label} (AB)", fontsize=9)
            axL.tick_params(labelbottom=False, bottom=False)
            # ---- main colour panel ----
            axk.scatter(zc[mm], c[mm], s=16, color=PALETTE[col], alpha=0.5,
                        lw=0, rasterized=True)
            rmid, rmed, _, _ = ref[col]
            axk.plot(rmid, rmed, color="k", lw=1.5, ls="--", label="all spec-QSO")
            mid, med, lo, hi = _running_stats(zc[mm], c[mm], width=w_bin, minpts=5)
            axk.fill_between(mid, lo, hi, color=PALETTE[col], alpha=0.22, lw=0,
                             label=f"PHZ {L} 16–84%")
            ln, = axk.plot(mid, med, color=PALETTE[col], lw=2.6, label=f"PHZ {L} median")
            ln.set_path_effects([pe.Stroke(linewidth=4.5, foreground="white"), pe.Normal()])
            axk.set_xlim(0, 3.5); axk.set_ylim(*YLIM)
            if i == 0:
                axk.set_title(label, fontsize=12, fontweight="bold")
            if i == len(rows) - 1:
                axk.set_xlabel("spectroscopic $z$", fontsize=11)
            axk.set_ylabel(""); axk.tick_params(labelleft=False)   # colour label lives on the marginal
            axk.legend(loc="upper left", fontsize=7, framealpha=0.9)

    parts = "  |  ".join(f"{L}(class {v})={int((df['phz_bitmask']==v).sum()):,}" for v, L in rows)
    fig.suptitle(f"{len(df):,} known spec QSOs Euclid colour(z) by Q1 PHZ class  ·  {aper}×FWHM aperture"
                 f"\n{parts}",
                 fontsize=14, fontweight="bold", y=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aperture", type=int, choices=APERS, default=2,
                    help="Euclid aperture in FWHM (default 2 = pipeline canonical).")
    ap.add_argument("--all-apertures", action="store_true",
                    help="generate the PHZ-label plot for all of 1/2/3/4 FWHM in one run.")
    ap.add_argument("--replot", action="store_true", help="use cached PHZ CSV, no WSDB.")
    args = ap.parse_args()
    user = os.getenv("PGUSER", WSDB_USER)

    # PHZ classification is aperture-independent — fetch (or load) it once, reuse per aperture.
    if args.replot and PHZ_CSV.exists():
        phz = pd.read_csv(PHZ_CSV); print(f"  loaded {len(phz):,} PHZ labels from cache")
    else:
        phz = fetch_phz(user)

    apers = APERS if args.all_apertures else [args.aperture]
    for a in apers:
        print(f"\n=== {a} FWHM aperture ===")
        qcsv = ARRAYS_DIR / f"rivers_qso_{a}fwhm.csv"   # shared cache with analyze_qso_color_vs_specz.py
        pcsv = ARRAYS_DIR / f"rivers_pop_{a}fwhm.csv"
        if args.replot:
            if not (qcsv.exists() and pcsv.exists()):
                print(f"  no colour/pop cache for {a}FWHM — run once without --replot first."); continue
            colz = pd.read_csv(qcsv); pop = pd.read_csv(pcsv)
            print(f"  loaded cache: {len(colz):,} spec QSOs + {len(pop):,} population ({a}FWHM)")
        else:
            colz = build_dataframe(user, a); pop = fetch_pop(user, a)
            ARRAYS_DIR.mkdir(parents=True, exist_ok=True)
            colz.to_csv(qcsv, index=False); pop.to_csv(pcsv, index=False)
        df = colz.merge(phz, on="qso_id", how="inner").dropna(subset=["qso_z"])
        print(f"  {len(df):,} spec QSOs with colours + PHZ label")
        for v, L in PHZ_ROWS:
            n = int((df["phz_bitmask"] == v).sum())
            print(f"    PHZ {L:<8} (class {v}): {n:5,} ({100*n/len(df):.0f}%)")
        make_plot(df, _png_for(a), pop, a)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
