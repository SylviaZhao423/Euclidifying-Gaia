#!/usr/bin/env python3
"""
Diagnose the Euclid official PHZ (mode_1) catastrophic-outlier structure for DESI QSOs.

analyze_qso_photz_vs_specz.py showed the PHZ mode_1 vs spec-z plane has a bright 1:1
ridge PLUS a second, lower diagonal (photo-z systematically < spec-z) and horizontal
"attractor" stripes. This script colours the SAME scatter by suspect features to see
what drives the failure. The leading suspect is phz_classification: ~80% of these
spectroscopic QSOs are template-classified as GALAXY (=2), and galaxy templates
under-predict QSO redshift.

Panels (all: PHZ mode_1 vs spectroscopic z, with 1:1 and |Δz|=0.15 lines), coloured by:
  1. phz_classification   (1=star 2=galaxy 4=QSO 6=gal+QSO ...)   <- the main hypothesis
  2. VIS magnitude        (brightness, from PHZ unified flux)
  3. phz_mode_1_area      (probability mass in the primary peak = confidence)
  4. best_chi2            (template fit quality, log)
  5. phz_mode_2 vs spec-z (does the SECONDARY peak recover the 1:1?)
  6. mode_2-is-closer     (outliers where |mode_2 - z| < |mode_1 - z| = the "wrong peak")

Reads (WSDB; Astronomy/Cambridge VPN; auth via ~/.pgpass):
  {PGUSER}.crossmatch_qso_desi_dr1_euclid   (qso_id, e_id, qso_z; euclid_source='Q1')
  {PGUSER}.q1_phz_photo_z                   (PHZ point estimates + diagnostics, by object_id)
Writes:
  output/qso_phz_diagnostics.png
  output/qso_color_vs_z/arrays/phz_diagnostics.csv   (--replot cache)

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_qso_phz_diagnostics.py       # pull + plot (caches)
  python src/analyze_qso_phz_diagnostics.py --replot                     # redraw from cache (no WSDB)
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
from analyze_qso_color_vs_specz import WSDB_HOST, WSDB_DB, ZPC_VIS  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
PNG_PATH    = PROJECT_DIR / "output" / "qso_phz_diagnostics.png"
CACHE       = PROJECT_DIR / "output" / "qso_color_vs_z" / "arrays" / "phz_diagnostics.csv"
PHZ_TABLE   = "q1_phz_photo_z"
OUTLIER_DZ  = 0.15
ZMAX        = 4.2

# phz_classification is a bitmask: bit0=star(1) bit1=galaxy(2) bit2=QSO(4)
CLASS_NAMES = {-1: "flagged", 0: "none", 1: "star", 2: "galaxy", 3: "star+gal",
               4: "QSO", 5: "star+QSO", 6: "gal+QSO", 7: "all three"}
CLASS_COLORS = {2: "#e6194B", 6: "#4363d8", 4: "#3cb44b", 0: "#9A9A9A",
                -1: "#000000", 1: "#f58231", 3: "#911eb4", 5: "#42d4f4", 7: "#bfef45"}
# phz_flags is near-discrete (mostly 0); colour 0 as neutral grey, flagged values bold
FLAG_COLORS = {0: "#C8C8C8", 10: "#e6194B", 11: "#4363d8", 12: "#3cb44b",
               13: "#f58231", 14: "#911eb4"}


# ----------------------------------------------------------------------- fetch
def fetch(user: str) -> pd.DataFrame:
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                            connect_timeout=30, keepalives=1, keepalives_idle=30,
                            keepalives_interval=10, keepalives_count=5)
    conn.cursor().execute("SET statement_timeout = 120000")
    print("  querying crossmatch x q1_phz_photo_z ...")
    df = pd.read_sql_query(
        f"""SELECT x.qso_id, x.qso_z,
                   p.phz_mode_1, p.phz_mode_2, p.phz_median,
                   p.phz_mode_1_area, p.phz_mode_2_area,
                   p.phz_classification, p.phz_flags, p.best_chi2, p.phz_weight,
                   p.flux_vis_unif
            FROM {user}.crossmatch_qso_desi_dr1_euclid x
            JOIN {user}.{PHZ_TABLE} p ON p.object_id = x.e_id::bigint
            WHERE x.euclid_source = 'Q1' AND x.qso_z IS NOT NULL AND x.qso_z > 0
              AND x.e_id IS NOT NULL""", conn)
    conn.close()
    df = df.drop_duplicates("qso_id").reset_index(drop=True)
    print(f"  {len(df):,} DESI Q1 QSOs with a PHZ entry")
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        df["IE"] = 23.90 - 2.5 * np.log10(ZPC_VIS * df["flux_vis_unif"].to_numpy(float))
    z, m1, m2 = df["qso_z"], df["phz_mode_1"], df["phz_mode_2"]
    df["dz1"] = (m1 - z) / (1.0 + z)
    df["is_outlier"]    = df["dz1"].abs() > OUTLIER_DZ
    df["mode2_closer"]  = (m2 - z).abs() < (m1 - z).abs()
    return df


# ------------------------------------------------------------------------ plot
def _ref_lines(ax):
    zl = np.linspace(0, ZMAX, 100)
    ax.plot(zl, zl, "k-", lw=1.2, zorder=5)
    ax.plot(zl, zl + OUTLIER_DZ * (1 + zl), "r--", lw=0.9, zorder=5)
    ax.plot(zl, zl - OUTLIER_DZ * (1 + zl), "r--", lw=0.9, zorder=5)
    ax.set_xlim(0, ZMAX); ax.set_ylim(0, ZMAX)
    ax.set_xlabel("spectroscopic $z$", fontsize=11)
    ax.set_ylabel("PHZ photo-$z$ (mode 1)", fontsize=11)


def _cont_panel(fig, ax, df, col, title, cmap, log=False, vmin=None, vmax=None):
    c = df[col].to_numpy(float)
    if log:
        c = np.log10(np.clip(c, 1e-6, None))
    ok = np.isfinite(c)
    sc = ax.scatter(df["qso_z"][ok], df["phz_mode_1"][ok], c=c[ok], s=9, alpha=0.55,
                    lw=0, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    _ref_lines(ax)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(("log10 " if log else "") + col, fontsize=10)


def make_plot(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(2, 3, figsize=(21, 13))

    # 1 — classification (discrete): the main hypothesis --------------------------
    ax = axes[0, 0]
    order = df["phz_classification"].value_counts().index.tolist()
    n = len(df)
    for cls in order:
        sub = df[df["phz_classification"] == cls]
        lab = f"{CLASS_NAMES.get(cls, cls)} ({cls}): {len(sub)} = {100*len(sub)/n:.0f}%"
        ax.scatter(sub["qso_z"], sub["phz_mode_1"], s=10, alpha=0.55, lw=0,
                   color=CLASS_COLORS.get(cls, "#777777"), label=lab, rasterized=True)
    _ref_lines(ax)
    ax.set_title("PHZ classification (template family)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, markerscale=1.6)

    # 2 — brightness -------------------------------------------------------------
    _cont_panel(fig, axes[0, 1], df, "IE", "VIS magnitude (brightness)",
                cmap="viridis", vmin=np.nanpercentile(df["IE"], 2),
                vmax=np.nanpercentile(df["IE"], 98))
    # 3 — PHZ flags (discrete; rare flagged points drawn on top) -----------------
    ax = axes[0, 2]
    for i, flag in enumerate(df["phz_flags"].value_counts().index.tolist()):
        sub = df[df["phz_flags"] == flag]
        lab = f"flag {flag}: {len(sub)} = {100*len(sub)/n:.0f}%"
        ax.scatter(sub["qso_z"], sub["phz_mode_1"], s=10, alpha=0.55, lw=0,
                   color=FLAG_COLORS.get(flag, "#777777"), label=lab,
                   zorder=3 + i, rasterized=True)
    _ref_lines(ax)
    ax.set_title("PHZ flags", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, markerscale=1.6)
    # 4 — fit quality ------------------------------------------------------------
    _cont_panel(fig, axes[1, 0], df, "best_chi2", "best $\\chi^2$ (template fit quality)",
                cmap="magma", log=True)

    # 5 — secondary peak vs spec-z ----------------------------------------------
    ax = axes[1, 1]
    ax.scatter(df["qso_z"], df["phz_mode_2"], s=9, alpha=0.5, lw=0,
               color="#f58231", rasterized=True)
    zl = np.linspace(0, ZMAX, 100)
    ax.plot(zl, zl, "k-", lw=1.2); ax.plot(zl, zl + OUTLIER_DZ*(1+zl), "r--", lw=0.9)
    ax.plot(zl, zl - OUTLIER_DZ*(1+zl), "r--", lw=0.9)
    ax.set_xlim(0, ZMAX); ax.set_ylim(0, ZMAX)
    ax.set_xlabel("spectroscopic $z$", fontsize=11)
    ax.set_ylabel("PHZ photo-$z$ (mode 2, secondary peak)", fontsize=11)
    ax.set_title("Secondary peak (mode 2) vs spec-$z$", fontsize=13, fontweight="bold", pad=12)

    # 6 — wrong-peak population --------------------------------------------------
    ax = axes[1, 2]
    out_bad  = df[df["is_outlier"] & df["mode2_closer"]]
    out_othr = df[df["is_outlier"] & ~df["mode2_closer"]]
    good     = df[~df["is_outlier"]]
    ax.scatter(good["qso_z"], good["phz_mode_1"], s=8, alpha=0.35, lw=0,
               color="#B0B0B0", label=f"within |Δz|≤0.15 ({len(good)})", rasterized=True)
    ax.scatter(out_othr["qso_z"], out_othr["phz_mode_1"], s=9, alpha=0.5, lw=0,
               color="#4363d8", label=f"outlier, mode 2 no better ({len(out_othr)})", rasterized=True)
    ax.scatter(out_bad["qso_z"], out_bad["phz_mode_1"], s=11, alpha=0.6, lw=0,
               color="#e6194B", label=f"outlier, mode 2 CLOSER ({len(out_bad)})", rasterized=True)
    _ref_lines(ax)
    nout = int(df["is_outlier"].sum())
    frac = 100 * len(out_bad) / max(nout, 1)
    ax.set_title(f"'Wrong peak' outliers  ({frac:.0f}% of outliers have a better mode 2)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, markerscale=1.6)

    galfrac = 100 * df["phz_classification"].isin([2, 6]).mean()
    fig.suptitle(f"Euclid PHZ mode 1 failure diagnostics for DESI QSOs  "
                 f"(N={len(df):,};  {galfrac:.0f}% classified galaxy)",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=132, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replot", action="store_true", help="redraw from cache (no WSDB).")
    ap.add_argument("--out", default=str(PNG_PATH), help="output figure path.")
    args = ap.parse_args()

    if args.replot and CACHE.exists():
        df = pd.read_csv(CACHE)
        print(f"  loaded {len(df):,} rows from cache {CACHE.name}")
    else:
        user = os.getenv("PGUSER", "")
        if not user:
            print("  PGUSER not set and no --replot cache — set PGUSER=siwei_zhao_2025.")
            return 1
        df = fetch(user)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE, index=False)
        print(f"  cached -> {CACHE}")

    df = add_derived(df)
    make_plot(df, Path(args.out))
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
