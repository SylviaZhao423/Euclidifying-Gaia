#!/usr/bin/env python3
"""
Vasily's test: does the Euclid colour(z) "rivers" plot look the SAME when the x-axis is a
PHOTO-z instead of the DESI spectroscopic z?  If yes, photo-z can stand in for spec-z to
define the QSO colour locus for objects WITHOUT spectroscopy (the pair candidates).

For the SAME DESI Q1 spec QSOs, we draw the colour(z) rivers (as in analyze_qso_desi_q1_
color_z.py) with THREE different x-axes as COLUMNS (rows = the 3 colours), same axes:
  - DESI spectroscopic z               (qso_z; the truth / reference)
  - our XGB photo-z                    (out-of-fold, from train_xgboost_qso_photz.py)
  - Euclid OFFICIAL PHZ  phz_mode_1    (most-probable peak of the PHZ PDF)
The spec-z running median is over-plotted (black dashed) on every panel as the fixed
reference, so you can see how much each photo-z SMEARS the river (16-84% width reported).

DESI DR1 Q1 only (2,843; SDSS 18 + ERO too small / not worth unified params). Colours come
from the OG euclid_q1.mer_final_cat (native 2FWHM aperture); photo-z from the WSDB table
siwei_zhao_2025.q1_phz_photo_z — all joined on object_id (needs the full load for coverage).

Reads : {PGUSER}.crossmatch_qso_desi_dr1_euclid  (DESI Q1 spec QSOs: qso_id, qso_z, e_id)
        euclid_q1.mer_final_cat                  (flux_{vis,y,j,h}_2fwhm_aper, by object_id)
        {PGUSER}.q1_phz_photo_z                  (phz_median, phz_mode_1, ... by object_id)
        output/xgboost_qso_photz/oof.csv   (our XGB OOF z, keyed qso_id)
Writes: output/qso_photz_rivers/qso_color_photz_rivers.png

Usage: PGUSER=siwei_zhao_2025 python src/analyze_qso_photz_rivers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_qso_color_vs_specz import (COLORS, PALETTE, ZMAX_PLOT, _flux_to_mag,  # noqa: E402
                                         ZPC_VIS, _running_stats, WSDB_HOST, WSDB_DB,
                                         WSDB_USER)
from train_xgboost_qso_photz import photz_metrics, OUTLIER_DZ  # noqa: E402

import os

# DESI DR1 Q1 only (2,843 sources). SDSS (18) and ERO (raw ADU) dropped — too small / not
# worth unified-parameter pain. Colour fluxes pulled from the OG euclid_q1.mer_final_cat
# (native 2FWHM aperture), matching analyze_qso_color_vs_specz.py etc.
DESI_TABLE = "crossmatch_qso_desi_dr1_euclid"
MER_TABLE = "euclid_q1.mer_final_cat"
PHZ_TABLE = "q1_phz_photo_z"

PROJECT_DIR = Path(__file__).resolve().parents[1]
OOF_CSV = PROJECT_DIR / "output" / "xgboost_qso_photz" / "oof.csv"
OUT_DIR = PROJECT_DIR / "output" / "qso_photz_rivers"
PNG_PATH = OUT_DIR / "qso_color_photz_rivers.png"
SCATTER_PNG = OUT_DIR / "qso_photz_scatter_resid.png"
# photo-z estimators for the scatter+residuals figure (spec-z is the x-axis truth)
SCATTER_ESTIMATORS = [("z_phot_oof", "our XGB photo-$z$ (OOF)"), ("phz_mode_1", "Euclid PHZ mode 1")]

# (column label, dataframe column, colour for the river) — DESI Q1 only, no SDSS
ESTIMATORS = [
    ("DESI spec-$z$",      "qso_z",      "#000000"),
    ("our XGB photo-$z$",  "z_phot_oof", "#4363d8"),
    ("PHZ mode 1",       "phz_mode_1", "#f58231"),
]


def pull(user: str) -> pd.DataFrame:
    """DESI Q1 spec QSOs -> OG mer_final_cat 2FWHM colours + q1_phz_photo_z z's, all by object_id."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30,
                            keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5)
    print(f"  {DESI_TABLE} (Q1) x {MER_TABLE} x {PHZ_TABLE} ...")
    d = pd.read_sql_query(
        f"""SELECT x.qso_id, x.qso_z, x.e_id::bigint AS object_id,
                   m.flux_vis_2fwhm_aper AS fv, m.fluxerr_vis_2fwhm_aper AS ev,
                   m.flux_y_2fwhm_aper   AS fy, m.fluxerr_y_2fwhm_aper   AS ey,
                   m.flux_j_2fwhm_aper   AS fj, m.fluxerr_j_2fwhm_aper   AS ej,
                   m.flux_h_2fwhm_aper   AS fh, m.fluxerr_h_2fwhm_aper   AS eh,
                   z.phz_median, z.phz_mode_1, z.phz_classification, z.phz_mode_1_area
            FROM {user}.{DESI_TABLE} x
            JOIN {MER_TABLE} m           ON m.object_id = x.e_id::bigint
            JOIN {user}.{PHZ_TABLE} z    ON z.object_id = x.e_id::bigint
            WHERE x.euclid_source = 'Q1' AND x.qso_z > 0 AND x.e_id IS NOT NULL""", conn)
    conn.close()
    d = d.drop_duplicates("object_id")
    d["IE"] = _flux_to_mag(d["fv"], d["ev"], ZPC_VIS)
    d["YE"] = _flux_to_mag(d["fy"], d["ey"])
    d["JE"] = _flux_to_mag(d["fj"], d["ej"])
    d["HE"] = _flux_to_mag(d["fh"], d["eh"])
    for _, b, r in COLORS:
        d[f"{b}_{r}"] = d[b] - d[r]
    if OOF_CSV.exists():
        oof = pd.read_csv(OOF_CSV)[["qso_id", "z_phot_oof"]]
        d = d.merge(oof, on="qso_id", how="left")
    else:
        d["z_phot_oof"] = np.nan
        print(f"  (no {OOF_CSV.name} — XGB row will be empty)")
    print(f"  {len(d):,} spec QSOs matched to q1_phz_photo_z")
    return d


def make_plot(df: pd.DataFrame, out: Path):
    # shared y-range (union of the three colours' 1-99 percentiles)
    pooled = []
    for _, b, r in COLORS:
        c = df[f"{b}_{r}"].to_numpy(float); c = c[np.isfinite(c)]
        pooled += list(np.percentile(c, [1, 99]))
    pad = 0.15 * (max(pooled) - min(pooled)); YLIM = (min(pooled) - pad, max(pooled) + pad)

    # spec-z reference river per colour (computed once)
    ref = {}
    zs = df["qso_z"].to_numpy(float)
    for _, b, r in COLORS:
        col = f"{b}_{r}"; c = df[col].to_numpy(float)
        m = np.isfinite(c) & np.isfinite(zs) & (zs <= ZMAX_PLOT)
        ref[col] = tuple(_running_stats(zs[m], c[m]))

    # TRANSPOSED: rows = colours, columns = z-estimators
    nrow, ncol = len(COLORS), len(ESTIMATORS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.7 * nrow), squeeze=False)
    widths = {}
    for i, (label, b, r) in enumerate(COLORS):            # rows = colours
        col = f"{b}_{r}"; c = df[col].to_numpy(float)
        for j, (col_lbl, zcol, rc) in enumerate(ESTIMATORS):   # columns = z-estimators
            ax = axes[i][j]; z = df[zcol].to_numpy(float)
            m = np.isfinite(c) & np.isfinite(z) & (z >= 0) & (z <= ZMAX_PLOT)
            n = int(m.sum())
            if n > 20:
                ax.scatter(z[m], c[m], s=7, alpha=0.28, lw=0, color=rc, rasterized=True)
                mid, med, lo, hi = _running_stats(z[m], c[m])
                ax.fill_between(mid, lo, hi, color=rc, alpha=0.20, lw=0)
                ln, = ax.plot(mid, med, color=rc, lw=2.2)
                ln.set_path_effects([pe.Stroke(linewidth=4, foreground="white"), pe.Normal()])
                w = float(np.median(hi - lo)); widths[(col_lbl, col)] = w
            else:
                w = np.nan
            # spec-z reference (dashed black) on every panel
            rmid, rmed, _, _ = ref[col]
            ax.plot(rmid, rmed, color="k", lw=1.2, ls="--", alpha=0.7)
            ax.set_xlim(0, ZMAX_PLOT); ax.set_ylim(*YLIM)
            if i == 0:
                ax.set_title(col_lbl, fontsize=12, fontweight="bold")
            if i == nrow - 1:
                ax.set_xlabel("redshift $z$", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{label}  (AB)", fontsize=11, fontweight="bold")
            ax.text(0.97, 0.04, f"N={n:,}\nwidth={w:.2f}" if n > 20 else f"N={n:,}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))

    fig.suptitle("QSO Euclid colour vs redshift — does the river survive photo-z?  "
                 "(DESI Q1 spec QSOs; black dashed = spec-z reference river on every panel)\n"
                 "columns = which z on the x-axis; a tight coloured river tracking the dashed line "
                 "= photo-z preserves the colour(z) locus",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")

    # summary: 16-84% river width per estimator per colour
    print("\n  median 16-84% river width (mag) — smaller = tighter locus:")
    hdr = "  ".join(f"{f'{b}_{r}':>8}" for _, b, r in COLORS)
    print(f"    {'estimator':20} {hdr}")
    for row_lbl, _, _ in ESTIMATORS:
        cells = "  ".join(f"{widths.get((row_lbl, f'{b}_{r}'), float('nan')):>8.2f}" for _, b, r in COLORS)
        print(f"    {row_lbl:20} {cells}")


def _running_median(x, y, xmax=4.2, w=0.2):
    edges = np.arange(0, xmax + w, w); mid, med = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() >= 10:
            mid.append(0.5 * (a + b)); med.append(np.median(y[m]))
    return np.array(mid), np.array(med)


def make_scatter_resid(df: pd.DataFrame, out: Path):
    """photo-z vs spec-z scatter + residuals-vs-redshift, in the train_xgboost_qso_photz.py layout
    (image #14), one ROW per photo-z estimator (our XGB, PHZ mode_1)."""
    from scipy.stats import gaussian_kde
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    zmax = 4.2
    rows = [(zc, lbl) for zc, lbl in SCATTER_ESTIMATORS if zc in df.columns and df[zc].notna().any()]
    fig, axes = plt.subplots(len(rows), 2, figsize=(13.5, 6.2 * len(rows)), squeeze=False)
    for i, (zcol, lbl) in enumerate(rows):
        sub = df.dropna(subset=["qso_z", zcol])
        zs = sub["qso_z"].to_numpy(float); zp = sub[zcol].to_numpy(float)
        met = photz_metrics(zs, zp); met3 = photz_metrics(zs[zs <= 3], zp[zs <= 3])
        dz = (zp - zs) / (1.0 + zs)

        # ---- col 1: 2D KDE + marginal histograms (jointplot style) ----
        a1 = axes[i][0]
        m0 = (zs >= 0) & (zs <= zmax) & (zp >= 0) & (zp <= zmax)
        kde = gaussian_kde(np.vstack([zs[m0], zp[m0]]))
        g = np.linspace(0, zmax, 110); GX, GY = np.meshgrid(g, g)
        ZZ = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
        a1.contourf(GX, GY, ZZ, levels=14, cmap="viridis")
        zl = np.linspace(0, zmax, 100)
        a1.plot(zl, zl, "k-", lw=1)
        a1.plot(zl, zl + OUTLIER_DZ * (1 + zl), "r--", lw=0.9, label=f"|Δz|={OUTLIER_DZ}")
        a1.plot(zl, zl - OUTLIER_DZ * (1 + zl), "r--", lw=0.9)
        a1.text(0.03, 0.97,
                f"all z (N={met['N']:,}):\n bias={met['bias']:+.3f}\n σ_NMAD={met['sigma_nmad']:.3f}\n"
                f" outliers={100*met['outlier_frac']:.1f}%\nz≤3 (N={met3['N']:,}):\n"
                f" σ_NMAD={met3['sigma_nmad']:.3f}\n outliers={100*met3['outlier_frac']:.1f}%",
                transform=a1.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
        a1.set_xlim(0, zmax); a1.set_ylim(0, zmax)
        a1.set_xlabel("spectroscopic $z$"); a1.set_ylabel(f"{lbl}")
        a1.legend(loc="lower right", fontsize=8)
        # marginal histograms: top = spec-z, right = photo-z
        div = make_axes_locatable(a1)
        axhx = div.append_axes("top", size="20%", pad=0.06, sharex=a1)
        axhy = div.append_axes("right", size="20%", pad=0.06, sharey=a1)
        _g = np.linspace(0, zmax, 200)
        axhx.hist(zs[m0], bins=45, density=True, color="#4363d8", alpha=0.55)
        axhx.plot(_g, gaussian_kde(zs[m0])(_g), color="#1f3a93", lw=1.8)      # spec-z KDE
        axhy.hist(zp[m0], bins=45, density=True, orientation="horizontal", color="#e6194b", alpha=0.55)
        axhy.plot(gaussian_kde(zp[m0])(_g), _g, color="#8b0000", lw=1.8)       # photo-z KDE
        axhx.tick_params(labelbottom=False, left=False, labelleft=False)
        axhy.tick_params(labelleft=False, bottom=False, labelbottom=False)
        for s in ("top", "right", "left"):
            axhx.spines[s].set_visible(False); axhy.spines[s].set_visible(False)
        axhy.spines["bottom"].set_visible(False)
        axhx.set_title(f"{lbl} — vs spec-$z$  (spec-z hist top, photo-z hist right)",
                       fontsize=11, fontweight="bold")

        a2 = axes[i][1]
        a2.scatter(zs, dz, s=7, alpha=0.3, lw=0, color="#4363d8", rasterized=True)
        a2.axhline(0, color="k", lw=1)
        a2.axhline(OUTLIER_DZ, color="r", ls="--", lw=0.9); a2.axhline(-OUTLIER_DZ, color="r", ls="--", lw=0.9)
        mid, med = _running_median(zs, dz)
        a2.plot(mid, med, color="#1f9e9e", lw=2.4, label="median Δz")
        a2.set_xlim(0, zmax); a2.set_ylim(-1, 1)
        a2.set_xlabel("spectroscopic $z$")
        a2.set_ylabel(r"$\Delta z = (z_{\rm phot}-z_{\rm spec})/(1+z_{\rm spec})$")
        a2.set_title("residuals vs redshift", fontsize=12, fontweight="bold")
        a2.legend(loc="upper right", fontsize=9)

    fig.suptitle("Euclid QSO photo-z vs DESI spec-z — scatter + residuals  (our XGB vs Euclid PHZ)\n"
                 "same layout as the XGB figure; PHZ median Δz plunges negative at high z "
                 "(galaxy-template bias), XGB stays flat",
                 fontsize=13, fontweight="bold")
    try:
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    except Exception:
        fig.subplots_adjust(top=0.9, hspace=0.35, wspace=0.3)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")
    for zcol, lbl in rows:
        sub = df.dropna(subset=["qso_z", zcol])
        m = photz_metrics(sub["qso_z"].to_numpy(float), sub[zcol].to_numpy(float))
        print(f"  {lbl:26}: N={m['N']:,}  σ_NMAD={m['sigma_nmad']:.3f}  "
              f"outliers={100*m['outlier_frac']:.1f}%  bias={m['bias']:+.3f}")


def main() -> int:
    df = pull(os.getenv("PGUSER", WSDB_USER))
    make_plot(df, PNG_PATH)
    make_scatter_resid(df, SCATTER_PNG)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
