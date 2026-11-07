#!/usr/bin/env python3
"""
Plotting / analysis for the Euclid XGBoost photo-z — DECOUPLED from training.

train_xgboost_qso_photz.py now only trains (OOF predictions, 10 seed models, metrics JSON) and
no longer plots. This script reads those saved outputs and (re)makes the figures, so plots
can be restyled or regenerated without re-running the (slow) 10-seed cross-validation:
  qso_photz_vs_specz.png   row 1: XGBoost photo-z vs spec-z (KDE + scatter + residuals)
                        row 2: Euclid PHZ mode_1  vs spec-z (same layout, if PHZ available)
(the feature-importance plot is made by train_xgboost_qso_photz.py and lives in models/.)

Reads : <dir>/oof.csv       (qso_id, qso_z, z_phot_oof, ...)
        <dir>/metrics.json  (features, all_z + z_le_3 metrics)
        <dir>/phz_full.csv  (qso_id, qso_z, phz_mode_1; ALL Q1 QSOs) — fetched if PGUSER set
Writes: <plot-dir>/qso_photz_vs_specz.png

Usage:
  python src/analyze_qso_photz_vs_specz.py.py                          # default output/xgboost_qso_photz
  PGUSER=siwei_zhao_2025 python src/analyze_qso_photz_vs_specz.py.py   # also fetches PHZ row
  python src/analyze_qso_photz_vs_specz.py.py --dir output/xgboost_qso_photz_wp
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
from train_xgboost_qso_photz import OUTLIER_DZ, photz_metrics  # noqa: E402
from analyze_qso_color_vs_specz import WSDB_HOST, WSDB_DB, WSDB_USER  # noqa: E402

PROJECT_DIR  = Path(__file__).resolve().parents[1]
DEFAULT_DIR  = PROJECT_DIR / "output" / "xgboost_qso_photz"
DEFAULT_PLOT_DIR = PROJECT_DIR / "output" / "qso_color_vs_z"
PHZ_TABLE    = "q1_phz_photo_z"


# ----------------------------------------------------------------------- PHZ fetch

def fetch_phz(user: str, qso_ids: np.ndarray, cache: Path) -> pd.DataFrame | None:
    if cache.exists():
        df = pd.read_csv(cache)
        print(f"  PHZ loaded from cache ({len(df):,} rows)")
        return df
    try:
        import psycopg2
        ids_str = ",".join(str(int(i)) for i in qso_ids)
        conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                                connect_timeout=30, keepalives=1, keepalives_idle=30,
                                keepalives_interval=10, keepalives_count=5)
        conn.cursor().execute("SET statement_timeout = 120000")
        # PHZ is keyed by Euclid object_id; our OOF is keyed by DESI qso_id.
        # Bridge them through the crossmatch (qso_id -> e_id -> phz.object_id).
        df = pd.read_sql_query(
            f"""SELECT x.qso_id, p.phz_mode_1
                FROM {user}.crossmatch_qso_desi_dr1_euclid x
                JOIN {user}.{PHZ_TABLE} p ON p.object_id = x.e_id::bigint
                WHERE x.euclid_source = 'Q1' AND x.qso_id IN ({ids_str})""", conn)
        conn.close()
        df = df.drop_duplicates("qso_id")
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        print(f"  PHZ fetched ({len(df):,} rows) -> {cache.name}")
        return df
    except Exception as exc:
        print(f"  PHZ fetch failed ({exc}) — row 2 skipped")
        return None


def fetch_phz_full(user: str, cache: Path) -> pd.DataFrame | None:
    """PHZ mode_1 + spec-z for ALL DESI Q1 QSOs (not just the XGB dropna sample), so the
    PHZ row shows every source Euclid has a photo-z for. Keyed via the crossmatch bridge."""
    if cache.exists():
        df = pd.read_csv(cache)
        print(f"  PHZ (full) loaded from cache ({len(df):,} rows)")
        return df
    try:
        import psycopg2
        conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                                connect_timeout=30, keepalives=1, keepalives_idle=30,
                                keepalives_interval=10, keepalives_count=5)
        conn.cursor().execute("SET statement_timeout = 120000")
        df = pd.read_sql_query(
            f"""SELECT x.e_id::bigint AS object_id, x.qso_id, x.qso_z, p.phz_mode_1
                FROM {user}.crossmatch_qso_desi_dr1_euclid x
                JOIN {user}.{PHZ_TABLE} p ON p.object_id = x.e_id::bigint
                WHERE x.euclid_source = 'Q1' AND x.qso_z IS NOT NULL AND x.qso_z > 0
                  AND x.e_id IS NOT NULL""", conn)
        conn.close()
        # dedup on the Euclid object_id (not qso_id) to match build_dataframe / the colour
        # plot: PHZ is keyed per Euclid source, so 2 DESI ids sharing one e_id count once.
        df = df.drop_duplicates("object_id")
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        print(f"  PHZ (full) fetched ({len(df):,} rows) -> {cache.name}")
        return df
    except Exception as exc:
        print(f"  PHZ (full) fetch failed ({exc}) — row 2 skipped")
        return None


# ----------------------------------------------------------------------- plot helpers

def _metrics_box(met: dict, met3: dict) -> str:
    return (f"all z (N={met['N']:,}):\n bias={met['bias']:+.3f}\n"
            f" σ_NMAD={met['sigma_nmad']:.3f}\n outliers={100*met['outlier_frac']:.1f}%\n"
            f"z≤3 (N={met3['N']:,}):\n σ_NMAD={met3['sigma_nmad']:.3f}\n"
            f" outliers={100*met3['outlier_frac']:.1f}%")


def _draw_row(fig, gs, row: int, zs, zp, row_label: str,
              met: dict, met3: dict, zmax: float,
              pt_color: str, kde_cmap: str):
    from scipy.stats import gaussian_kde
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    zl  = np.linspace(0, zmax, 100)
    dz  = (zp - zs) / (1.0 + zs)
    box = _metrics_box(met, met3)

    ax_kde = fig.add_subplot(gs[row, 0])
    ax_sc  = fig.add_subplot(gs[row, 2])
    ax_res = fig.add_subplot(gs[row, 4])

    # ── KDE contour + marginals ───────────────────────────────────────────────
    mask = np.isfinite(zs) & np.isfinite(zp) & (zs <= zmax) & (zp <= zmax)
    zs_k, zp_k = zs[mask], zp[mask]
    g = np.linspace(0, zmax, 110); GX, GY = np.meshgrid(g, g)
    ZZ = gaussian_kde(np.vstack([zs_k, zp_k]))(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    ax_kde.contourf(GX, GY, ZZ, levels=14, cmap=kde_cmap)
    ax_kde.plot(zl, zl, "k-", lw=1)
    ax_kde.plot(zl, zl + OUTLIER_DZ * (1 + zl), "r--", lw=0.9)
    ax_kde.plot(zl, zl - OUTLIER_DZ * (1 + zl), "r--", lw=0.9, label=f"|Δz|={OUTLIER_DZ}")
    ax_kde.set_xlim(0, zmax); ax_kde.set_ylim(0, zmax)
    ax_kde.set_xlabel("spectroscopic $z$", fontsize=11)
    ax_kde.set_ylabel("photo-$z$", fontsize=11)
    # ax_kde.text(0.03, 0.97, box, transform=ax_kde.transAxes, fontsize=9, va="top",
    #             bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    # ax_kde.legend(loc="lower right", fontsize=8)

    div  = make_axes_locatable(ax_kde)
    axhx = div.append_axes("top",   size="20%", pad=0.06, sharex=ax_kde)
    axhy = div.append_axes("right", size="20%", pad=0.06, sharey=ax_kde)
    _g   = np.linspace(0, zmax, 200)
    axhx.hist(zs_k, bins=45, density=True, color="#4363d8", alpha=0.55)
    axhx.plot(_g, gaussian_kde(zs_k)(_g), color="#1f3a93", lw=1.8)
    axhy.hist(zp_k, bins=45, density=True, orientation="horizontal", color=pt_color, alpha=0.55)
    axhy.plot(gaussian_kde(zp_k)(_g), _g, color=pt_color, lw=1.8)
    axhx.tick_params(labelbottom=False, left=False, labelleft=False)
    axhy.tick_params(labelleft=False, bottom=False, labelbottom=False)
    for s in ("top", "right", "left"):
        axhx.spines[s].set_visible(False); axhy.spines[s].set_visible(False)
    axhy.spines["bottom"].set_visible(False)
    axhx.set_title(f"{row_label}: KDE Contour", fontsize=13, fontweight="bold", pad=16)

    # ── scatter ───────────────────────────────────────────────────────────────
    ax_sc.scatter(zs, zp, s=7, color=pt_color, alpha=0.3, lw=0, rasterized=True)
    ax_sc.plot(zl, zl, "k-", lw=1)
    ax_sc.plot(zl, zl + OUTLIER_DZ * (1 + zl), "r--", lw=0.9, label=f"|Δz|={OUTLIER_DZ}")
    ax_sc.plot(zl, zl - OUTLIER_DZ * (1 + zl), "r--", lw=0.9)
    ax_sc.set_xlim(0, zmax); ax_sc.set_ylim(0, zmax)
    ax_sc.set_xlabel("spectroscopic $z$", fontsize=11)
    ax_sc.set_ylabel("photo-$z$", fontsize=11)
    ax_sc.set_title(f"{row_label}: Scatter", fontsize=13, fontweight="bold", pad=16)
    ax_sc.text(0.03, 0.97, box, transform=ax_sc.transAxes, fontsize=9, va="top",
               bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    ax_sc.legend(loc="lower right", fontsize=8)

    # ── residuals ─────────────────────────────────────────────────────────────
    ax_res.scatter(zs, dz, s=7, color=pt_color, alpha=0.3, lw=0, rasterized=True)
    ax_res.axhline(0, color="k", lw=1)
    ax_res.axhline( OUTLIER_DZ, color="r", ls="--", lw=0.9)
    ax_res.axhline(-OUTLIER_DZ, color="r", ls="--", lw=0.9)
    edges = np.arange(0, zmax + 0.2, 0.2)
    mid, med_dz = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (zs >= lo) & (zs < hi)
        if m.sum() >= 15:
            mid.append(0.5 * (lo + hi)); med_dz.append(np.median(dz[m]))
    ax_res.plot(mid, med_dz, color="#1f9e9e", lw=2.4, label="median Δz")
    ax_res.set_xlim(0, zmax); ax_res.set_ylim(-1, 1)
    ax_res.set_xlabel("spectroscopic $z$", fontsize=11)
    ax_res.set_ylabel(r"$\Delta z=(z_{\rm phot}-z_{\rm spec})/(1+z_{\rm spec})$", fontsize=11)
    ax_res.set_title(f"{row_label}: Residuals", fontsize=13, fontweight="bold", pad=16)
    ax_res.legend(loc="upper right", fontsize=9)


# ----------------------------------------------------------------------- main plot

def make_scatter_plot(zs_xgb, zp_xgb, met_xgb, met3_xgb,
                      out: Path, features: list, title_note: str = "",
                      zs_phz=None, zp_phz=None, met_phz=None, met3_phz=None):
    from matplotlib.gridspec import GridSpec

    zmax  = 4.2
    nrows = 2 if zp_phz is not None else 1
    gap1, gap2 = 0.16, 0.26   # independent wspace between panels 1-2 and 2-3

    fig = plt.figure(figsize=(22, 6.8 * nrows + (1.0 if nrows == 2 else 0)))
    gs  = GridSpec(nrows, 5, figure=fig,
                   width_ratios=[1.1, gap1, 1.0, gap2, 1.0],
                   left=0.06, right=0.97, wspace=0,
                   top=0.90, bottom=0.08,
                   hspace=0.24 if nrows == 2 else 0)

    note_str = f"  [{title_note}]" if title_note else ""
    xgb_label = f"XGBoost z{note_str}"
    _draw_row(fig, gs, 0, zs_xgb, zp_xgb, xgb_label,
              met_xgb, met3_xgb, zmax, pt_color="#4363d8", kde_cmap="viridis")
    if zp_phz is not None:
        _draw_row(fig, gs, 1, zs_phz, zp_phz, "Euclid PHZ mode 1",
                  met_phz, met3_phz, zmax, pt_color="#f58231", kde_cmap="plasma")

    feat_desc = f"(VIS/Y/J/H × 1/2/3/4 FWHM{(', ' + title_note) if title_note else ''})"
    fig.suptitle(
        f"XGBoost Euclid photo-z  ·  {len(features)} features  {feat_desc}",
        fontsize=15, fontweight="bold", y=0.97)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=132, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


# ----------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(DEFAULT_DIR),
                    help="training-output directory (default output/xgboost_qso_photz).")
    ap.add_argument("--plot-dir", default=str(DEFAULT_PLOT_DIR),
                    help="where to write the figure (default: output/qso_color_vs_z/).")
    ap.add_argument("--title-note", default="",
                    help="optional note in figure titles (e.g. 'morph').")
    args = ap.parse_args()
    d = Path(args.dir); pdir = Path(args.plot_dir); pdir.mkdir(parents=True, exist_ok=True)

    oof_csv  = d / "oof.csv"
    met_json = d / "metrics.json"
    if not (oof_csv.exists() and met_json.exists()):
        print(f"  missing {oof_csv.name} / {met_json.name} in {d} — run train_xgboost_qso_photz.py first.")
        return 1

    oof      = pd.read_csv(oof_csv)
    meta     = json.loads(met_json.read_text())
    features = meta["features"]
    met_xgb  = meta["all_z"]
    met3_xgb = meta["z_le_3"]
    zs_xgb   = oof["qso_z"].to_numpy(float)
    zp_xgb   = oof["z_phot_oof"].to_numpy(float)
    print(f"  {len(zs_xgb):,} OOF QSOs, {len(features)} features (from {d.name})")

    # ── PHZ row ───────────────────────────────────────────────────────────────
    zs_phz = zp_phz = met_phz = met3_phz = None
    user = os.getenv("PGUSER", "")
    if user:
        # Full PHZ sample: every DESI Q1 QSO Euclid has a photo-z for (not limited to
        # the XGB dropna sample), so the PHZ row is fuller than the XGB row.
        phz_df = fetch_phz_full(user, cache=pdir / "arrays" / "phz_full.csv")
        if phz_df is not None and len(phz_df):
            valid = (np.isfinite(phz_df["phz_mode_1"].to_numpy(float))
                     & np.isfinite(phz_df["qso_z"].to_numpy(float)))
            zs_phz = phz_df["qso_z"].to_numpy(float)[valid]
            zp_phz = phz_df["phz_mode_1"].to_numpy(float)[valid]
            if len(zs_phz) >= 2:
                met_phz  = photz_metrics(zs_phz, zp_phz)
                met3_phz = photz_metrics(zs_phz[zs_phz <= 3], zp_phz[zs_phz <= 3])
                print(f"  PHZ row: {len(zs_phz):,} sources with phz_mode_1 (full Q1 QSO sample)")
            else:
                zs_phz = zp_phz = None
                print(f"  PHZ row: {len(phz_df):,} matched but <2 valid phz_mode_1 — row 2 skipped")
        else:
            print("  PHZ row: 0 crossmatch/PHZ rows — row 2 skipped")
    else:
        print("  PGUSER not set — PHZ row skipped (run with PGUSER=... to add it)")

    make_scatter_plot(zs_xgb, zp_xgb, met_xgb, met3_xgb,
                      pdir / "qso_photz_vs_specz.png", features, args.title_note,
                      zs_phz=zs_phz, zp_phz=zp_phz, met_phz=met_phz, met3_phz=met3_phz)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
