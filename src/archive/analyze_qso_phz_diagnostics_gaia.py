#!/usr/bin/env python3
"""
Gaia variant of analyze_qso_phz_diagnostics.py: the Euclid PHZ mode_1 vs spec-z plane for
DESI QSOs that ALSO have a Gaia match (crossmatch_qso_desi_dr1_gaia), so we can bring in
their Gaia astrometry — chiefly g_pm_significance (proper-motion |μ|/σ_μ).

Spec-confirmed QSOs are extragalactic, so their PM should be consistent with zero. Any
high-PM object among them is a likely blend / mis-association, and colouring the PHZ
scatter by PM significance tests whether the catastrophic PHZ outliers (the second, lower
diagonal) are contaminated by moving sources. Panels (all PHZ mode_1 vs spec-z unless
noted), coloured by:
  1. Gaia PM significance     (|μ|/σ_μ; white = 3σ threshold)   <- the main new diagnostic
  2. phz_classification       (template family: galaxy/QSO/...)
  3. Gaia parallax/error      (another astrometric stellar flag; diverging about 0)
  4. VIS magnitude            (brightness)
  5. phz_mode_2 vs spec-z     (does the secondary peak recover the 1:1?)
  6. PM significance vs Δz    (do PHZ outliers move? correlation panel)

Reads (WSDB; Astronomy/Cambridge VPN; auth via ~/.pgpass):
  {PGUSER}.crossmatch_qso_desi_dr1_euclid  (qso_id, e_id, qso_z; euclid_source='Q1')
  {PGUSER}.q1_phz_photo_z                  (PHZ point estimates + diagnostics, by object_id)
  {PGUSER}.crossmatch_qso_desi_dr1_gaia    (g_pm_significance, g_parallax_over_error, by qso_id)
Writes:
  output/qso_phz_diagnostics_gaia.png
  output/qso_color_vs_z/arrays/phz_diagnostics_gaia.csv   (--replot cache)

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_qso_phz_diagnostics_gaia.py     # pull + plot (caches)
  python src/analyze_qso_phz_diagnostics_gaia.py --replot                   # redraw from cache
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_qso_color_vs_specz import WSDB_HOST, WSDB_DB, WSDB_USER  # noqa: E402
from analyze_qso_phz_diagnostics import (  # noqa: E402
    CLASS_NAMES, CLASS_COLORS, ZMAX, OUTLIER_DZ, PHZ_TABLE, add_derived, _ref_lines, _cont_panel,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
PNG_PATH    = PROJECT_DIR / "output" / "qso_phz_diagnostics_gaia.png"
CACHE       = PROJECT_DIR / "output" / "qso_color_vs_z" / "arrays" / "phz_diagnostics_gaia.csv"
GAIA_TABLE  = "crossmatch_qso_desi_dr1_gaia"
PM_CMAP, PM_NORM = "coolwarm", TwoSlopeNorm(vmin=0.0, vcenter=3.0, vmax=10.0)   # white at 3σ


# ----------------------------------------------------------------------- fetch
def fetch(user: str) -> pd.DataFrame:
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30,
                            keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5)
    conn.cursor().execute("SET statement_timeout = 120000")
    print("  querying DESI×Euclid PHZ × DESI×Gaia ...")
    df = pd.read_sql_query(
        f"""SELECT x.qso_id, x.qso_z,
                   p.phz_mode_1, p.phz_mode_2, p.phz_median,
                   p.phz_mode_1_area, p.phz_classification, p.phz_flags, p.best_chi2,
                   p.flux_vis_unif,
                   g.g_pm_significance   AS pm_sig,
                   g.g_parallax_over_error AS plx_over_err
            FROM {user}.crossmatch_qso_desi_dr1_euclid x
            JOIN {user}.{PHZ_TABLE} p ON p.object_id = x.e_id::bigint
            JOIN {user}.{GAIA_TABLE} g ON g.qso_id = x.qso_id
            WHERE x.euclid_source = 'Q1' AND x.qso_z IS NOT NULL AND x.qso_z > 0
              AND x.e_id IS NOT NULL""", conn)
    conn.close()
    df = df.drop_duplicates("qso_id").reset_index(drop=True)
    print(f"  {len(df):,} DESI Q1 QSOs with PHZ + Gaia ({df['pm_sig'].notna().sum():,} have a PM)")
    return df


# ------------------------------------------------------------------------ plot
def _norm_scatter(fig, ax, df, col, title, cmap, norm, cbar_label):
    """PHZ mode_1 vs spec-z coloured by df[col] with an explicit norm (high values on top)."""
    c = df[col].to_numpy(float)
    ok = np.isfinite(c)
    zx = df["qso_z"].to_numpy(float)[ok]; zy = df["phz_mode_1"].to_numpy(float)[ok]; cc = c[ok]
    order = np.argsort(cc)
    sc = ax.scatter(zx[order], zy[order], c=cc[order], s=14, alpha=0.85, lw=0,
                    cmap=cmap, norm=norm, rasterized=True)
    _ref_lines(ax)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, extend="both")
    cb.set_label(cbar_label, fontsize=10)


def make_plot(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(2, 3, figsize=(21, 13))
    n = len(df)

    # 1 — Gaia PM significance (the main new diagnostic) --------------------------
    _norm_scatter(fig, axes[0, 0], df, "pm_sig",
                  "coloured by Gaia PM significance", PM_CMAP, PM_NORM,
                  "PM significance |μ|/σ  (white = 3σ)")

    # 2 — phz_classification (discrete) ------------------------------------------
    ax = axes[0, 1]
    for cls in df["phz_classification"].value_counts().index.tolist():
        sub = df[df["phz_classification"] == cls]
        lab = f"{CLASS_NAMES.get(cls, cls)} ({cls}): {len(sub)} = {100*len(sub)/n:.0f}%"
        ax.scatter(sub["qso_z"], sub["phz_mode_1"], s=13, alpha=0.6, lw=0,
                   color=CLASS_COLORS.get(cls, "#777777"), label=lab, rasterized=True)
    _ref_lines(ax)
    ax.set_title("PHZ classification (template family)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, markerscale=1.6)

    # 3 — Gaia parallax / error (diverging about 0) ------------------------------
    _norm_scatter(fig, axes[0, 2], df, "plx_over_err",
                  "coloured by Gaia parallax / error", "RdBu",
                  TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5), "parallax / error")

    # 4 — brightness -------------------------------------------------------------
    _cont_panel(fig, axes[1, 0], df, "IE", "VIS magnitude (brightness)", cmap="viridis",
                vmin=np.nanpercentile(df["IE"], 2), vmax=np.nanpercentile(df["IE"], 98))

    # 5 — secondary peak vs spec-z ----------------------------------------------
    ax = axes[1, 1]
    ax.scatter(df["qso_z"], df["phz_mode_2"], s=13, alpha=0.6, lw=0, color="#f58231", rasterized=True)
    zl = np.linspace(0, ZMAX, 100)
    ax.plot(zl, zl, "k-", lw=1.2); ax.plot(zl, zl + OUTLIER_DZ*(1+zl), "r--", lw=0.9)
    ax.plot(zl, zl - OUTLIER_DZ*(1+zl), "r--", lw=0.9)
    ax.set_xlim(0, ZMAX); ax.set_ylim(0, ZMAX)
    ax.set_xlabel("spectroscopic $z$", fontsize=11); ax.set_ylabel("PHZ photo-$z$ (mode 2)", fontsize=11)
    ax.set_title("Secondary peak (mode 2) vs spec-$z$", fontsize=13, fontweight="bold", pad=12)

    # 6 — does PHZ failure correlate with proper motion? -------------------------
    ax = axes[1, 2]
    ok = np.isfinite(df["pm_sig"].to_numpy(float))
    pm = df["pm_sig"].to_numpy(float)[ok]; dz = df["dz1"].to_numpy(float)[ok]
    outl = np.abs(dz) > OUTLIER_DZ
    ax.scatter(pm[~outl], dz[~outl], s=13, alpha=0.5, lw=0, color="#4363d8",
               label=f"PHZ ok ({(~outl).sum()})", rasterized=True)
    ax.scatter(pm[outl], dz[outl], s=15, alpha=0.7, lw=0, color="#e6194B",
               label=f"PHZ outlier ({outl.sum()})", rasterized=True)
    ax.axhline(0, color="k", lw=1); ax.axhline(OUTLIER_DZ, color="r", ls="--", lw=0.9)
    ax.axhline(-OUTLIER_DZ, color="r", ls="--", lw=0.9)
    ax.axvline(3, color="0.4", ls=":", lw=1.2, label="PM = 3σ")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlim(0, max(50, np.nanpercentile(pm, 99.5))); ax.set_ylim(-1, 1)
    ax.set_xlabel("Gaia PM significance |μ|/σ  (symlog)", fontsize=11)
    ax.set_ylabel(r"$\Delta z=(z_{\rm phz}-z_{\rm spec})/(1+z_{\rm spec})$", fontsize=11)
    ax.set_title("PHZ residual vs proper motion", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    hipm = 100 * np.mean(df["pm_sig"].dropna() > 3)
    fig.suptitle(
        f"Euclid PHZ diagnostics for DESI QSOs WITH a Gaia match  (N={n:,};  {hipm:.0f}% have PM > 3σ)"
        f"  —  spec-confirmed QSOs should have PM ≈ 0",
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
        df = pd.read_csv(CACHE); print(f"  loaded {len(df):,} rows from cache")
    else:
        user = os.getenv("PGUSER", WSDB_USER)
        df = fetch(user)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE, index=False); print(f"  cached -> {CACHE}")

    df = add_derived(df)
    make_plot(df, Path(args.out))
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
