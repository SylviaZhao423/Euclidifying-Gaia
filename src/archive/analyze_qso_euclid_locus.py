#!/usr/bin/env python3
"""
STEP 3a of the QSO-companion colour pipeline: the QSO colour(z) LOCUS and a
marginalized-over-z "is-this-a-QSO?" consistency test.

Because the Euclid-only photo-z is degeneracy-limited (Step 2, ~31% outliers), we do
NOT rely on a single predicted z. Instead we ask a more robust question: is an object's
Euclid colour consistent with the QSO colour(z) locus at ANY redshift?

Method
------
1. Build the empirical locus from the spec QSOs (Step 1 CSV): for each colour c and a
   fine z-grid, mu_c(z) = running median, sig_c(z) = (P84-P16)/2 (intrinsic+photometric
   scatter, measured from the data). These smooth "rivers" are the QSO template track.
2. For any object with Euclid colours, define
       D2(z)   = sum_c [ (c_obs - mu_c(z))^2 / (sig_c(z)^2 + floor^2) ]
       D2_min  = min_z D2(z),   z_best = argmin_z D2(z)
   D2_min / n_colours is a reduced chi^2: ~1 if the object sits on the QSO locus at some
   z, large if it never matches (a star / non-QSO). z_best is a degeneracy-aware
   locus-fit photo-z (the GLOBAL best-matching redshift).
3. Validate the test: D2_min for spec QSOs (should be small) vs an unambiguous STAR
   control (Gaia parallax_over_error > 10, definitely Galactic; should be large). Report
   the ROC/AUC and the star contamination at fixed QSO completeness.

The locus (z, mu_c, sig_c) is saved for Step 3b (applying the test to the real
WISE+Euclid QSO-pair candidates, individually and at a common z).

Inputs:
  (none — spec-QSO colours+z built live via analyze_qso_color_vs_specz.build_dataframe())   (spec QSOs; from Step 1)
  {PGUSER}.crossmatch_q1_mer_phz_vis_gaia_sep        (star control; WSDB, VPN)
Outputs:
  output/qso_locus/qso_color_locus.csv               (z, mu/sig per colour — for Step 3b)
  output/qso_locus/qso_euclid_locus.png              (locus + D2_min separation + z_best)
  output/qso_locus/qso_euclid_locus_metrics.json

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_qso_euclid_locus.py            # pull control + run
  python src/analyze_qso_euclid_locus.py --replot                         # from cached control
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

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "output" / "qso_locus"
LOCUS_CSV  = OUTPUT_DIR / "qso_color_locus.csv"
CONTROL_CSV = OUTPUT_DIR / "arrays" / "star_control_colors.csv"
PNG_PATH   = OUTPUT_DIR / "qso_euclid_locus.png"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
WSDB_USER = "siwei_zhao_2025"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_qso_color_vs_specz import build_dataframe  # noqa: E402  (spec-QSO colours+z, live)

ZP_AB, ZPC_VIS, SNR_MIN = 23.90, 1.05, 3.0
COLORS = [("IE_YE", "$I_E-Y_E$"), ("YE_JE", "$Y_E-J_E$"), ("JE_HE", "$J_E-H_E$")]
NCOL = len(COLORS)

ZGRID = np.arange(0.0, 4.0 + 1e-9, 0.02)   # z sampling for the locus + D2(z) minimisation
SIG_FLOOR = 0.02                            # mag; guards against locus points with tiny sig
N_CONTROL = 6000                            # star-control sample size


# --------------------------------------------------------------------------- locus
def build_locus(dfq: pd.DataFrame) -> pd.DataFrame:
    """mu_c(z) (running median) and sig_c(z) ((P84-P16)/2) on ZGRID, per colour."""
    z = dfq["qso_z"].to_numpy(float)
    loc = {"z": ZGRID}
    half_win = 0.12                          # +/- z window for each grid point
    for col, _ in COLORS:
        c = dfq[col].to_numpy(float)
        mu = np.full(ZGRID.shape, np.nan)
        sg = np.full(ZGRID.shape, np.nan)
        for i, zc in enumerate(ZGRID):
            m = np.isfinite(c) & (np.abs(z - zc) <= half_win)
            if m.sum() >= 15:
                cc = c[m]
                mu[i] = np.median(cc)
                sg[i] = 0.5 * (np.percentile(cc, 84) - np.percentile(cc, 16))
        # fill any thin-coverage gaps by interpolation so D2(z) is defined on all ZGRID
        ok = np.isfinite(mu)
        mu = np.interp(ZGRID, ZGRID[ok], mu[ok])
        sg = np.interp(ZGRID, ZGRID[ok], sg[ok])
        loc[f"mu_{col}"] = mu
        loc[f"sig_{col}"] = sg
    return pd.DataFrame(loc)


def d2_min(colors: np.ndarray, locus: pd.DataFrame):
    """For each row of `colors` (N x NCOL), D2_min over ZGRID and the argmin z_best.
    colors columns must align with COLORS order."""
    mu  = np.stack([locus[f"mu_{c}"].to_numpy()  for c, _ in COLORS], axis=1)   # (Z, NCOL)
    sig = np.stack([locus[f"sig_{c}"].to_numpy() for c, _ in COLORS], axis=1)
    inv = 1.0 / (sig ** 2 + SIG_FLOOR ** 2)                                     # (Z, NCOL)
    # D2[n, z] = sum_c (colors[n,c] - mu[z,c])^2 * inv[z,c]
    diff = colors[:, None, :] - mu[None, :, :]                                  # (N, Z, NCOL)
    d2 = np.einsum("nzc,zc->nz", diff ** 2, inv)                                # (N, Z)
    zi = np.nanargmin(d2, axis=1)
    return d2[np.arange(len(colors)), zi], ZGRID[zi]


# --------------------------------------------------------------------------- control
def _flux_to_mag(flux, err, zpc=1.0):
    flux = pd.to_numeric(flux, errors="coerce").to_numpy(float)
    err  = pd.to_numeric(err,  errors="coerce").to_numpy(float)
    good = np.isfinite(flux) & (flux > 0) & np.isfinite(err) & (err > 0) & (flux / err >= SNR_MIN)
    mag = np.full(flux.shape, np.nan)
    mag[good] = ZP_AB - 2.5 * np.log10(zpc * flux[good])
    return mag


def fetch_star_control(user: str) -> pd.DataFrame:
    """Unambiguous Gaia stars (parallax_over_error>10) in Q1 with Euclid I/Y/J/H → colours.
    A random subsample of N_CONTROL. Cached to CONTROL_CSV."""
    import psycopg2
    sql = f"""
        SELECT m_flux_vis_2fwhm_aper AS fv, m_fluxerr_vis_2fwhm_aper AS ev,
               m_flux_y_2fwhm_aper   AS fy, m_fluxerr_y_2fwhm_aper   AS ey,
               m_flux_j_2fwhm_aper   AS fj, m_fluxerr_j_2fwhm_aper   AS ej,
               m_flux_h_2fwhm_aper   AS fh, m_fluxerr_h_2fwhm_aper   AS eh
        FROM {user}.crossmatch_q1_mer_phz_vis_gaia_sep
        WHERE g_parallax_over_error > 10
          AND m_flux_vis_2fwhm_aper > 0 AND m_flux_y_2fwhm_aper > 0
          AND m_flux_j_2fwhm_aper > 0 AND m_flux_h_2fwhm_aper > 0
        ORDER BY random() LIMIT {N_CONTROL}
    """
    print("  reading star control (Gaia parallax_over_error>10 × Euclid) ...")
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                            connect_timeout=30, keepalives=1, keepalives_idle=30,
                            keepalives_interval=10, keepalives_count=5)
    df = pd.read_sql_query(sql, conn); conn.close()
    df["IE"] = _flux_to_mag(df["fv"], df["ev"], ZPC_VIS)
    df["YE"] = _flux_to_mag(df["fy"], df["ey"])
    df["JE"] = _flux_to_mag(df["fj"], df["ej"])
    df["HE"] = _flux_to_mag(df["fh"], df["eh"])
    df["IE_YE"] = df["IE"] - df["YE"]
    df["YE_JE"] = df["YE"] - df["JE"]
    df["JE_HE"] = df["JE"] - df["HE"]
    CONTROL_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[["IE_YE", "YE_JE", "JE_HE"]].dropna().to_csv(CONTROL_CSV, index=False)
    print(f"    {len(df):,} stars -> {CONTROL_CSV}")
    return df


# -------------------------------------------------------------------------- metrics
def roc_auc(score_pos, score_neg):
    """ROC of a score where HIGHER = more QSO-like (we use -D2_min). Returns fpr, tpr, auc."""
    s = np.concatenate([score_pos, score_neg])
    y = np.concatenate([np.ones(len(score_pos)), np.zeros(len(score_neg))])
    order = np.argsort(-s)
    y = y[order]
    tpr = np.cumsum(y) / max(1, y.sum())
    fpr = np.cumsum(1 - y) / max(1, (1 - y).sum())
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def _zphot_stats(zs, zp):
    dz = (zp - zs) / (1 + zs)
    return {"sigma_nmad": float(1.48 * np.median(np.abs(dz - np.median(dz)))),
            "outlier_frac": float(np.mean(np.abs(dz) > 0.15))}


# ----------------------------------------------------------------------------- plot
def make_plot(locus, dfq, d2q, zbest_q, d2s, out):
    rq, rs = d2q / NCOL, d2s / NCOL          # reduced chi^2
    fpr, tpr, auc = roc_auc(-d2q, -d2s)
    # star contamination at 95% QSO completeness
    thr = np.percentile(rq, 95)              # keep 95% of QSOs (reduced-chi2 <= thr)
    star_contam = float(np.mean(rs <= thr))
    zst = _zphot_stats(dfq["qso_z"].to_numpy(float), zbest_q)

    fig, axes = plt.subplots(1, 4, figsize=(23, 5.6))
    # (1) the locus rivers
    ax = axes[0]
    zg = locus["z"].to_numpy()
    palette = ["#e6194b", "#3cb44b", "#4363d8"]
    for (col, lab), cpal in zip(COLORS, palette):
        mu, sg = locus[f"mu_{col}"].to_numpy(), locus[f"sig_{col}"].to_numpy()
        ax.plot(zg, mu, color=cpal, lw=1.8, label=lab)
        ax.fill_between(zg, mu - sg, mu + sg, color=cpal, alpha=0.2, lw=0)
    ax.set_xlabel("redshift $z$"); ax.set_ylabel("Euclid colour (AB)")
    ax.set_title("QSO colour(z) locus  μ ± σ", fontweight="bold")
    ax.legend(fontsize=9)
    # (2) D2_min separation
    ax = axes[1]
    bins = np.linspace(0, 30, 60)
    ax.hist(rq, bins=bins, color="#e6194b", alpha=0.6, density=True, label=f"QSOs (N={len(rq):,})")
    ax.hist(rs, bins=bins, color="#555555", alpha=0.6, density=True, label=f"stars (N={len(rs):,})")
    ax.axvline(thr, color="k", ls="--", lw=1, label=f"95% QSO cut (χ²ᵣ={thr:.1f})")
    ax.set_xlabel(r"reduced $\chi^2_{\min}=D^2_{\min}/3$"); ax.set_ylabel("density")
    ax.set_title("QSO-consistency separation", fontweight="bold")
    ax.legend(fontsize=9)
    ax.text(0.97, 0.55, f"star contam.\n@95% compl.\n= {100*star_contam:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    # (3) ROC
    ax = axes[2]
    ax.plot(fpr, tpr, color="#4363d8", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("star false-positive rate"); ax.set_ylabel("QSO true-positive rate")
    ax.set_title(f"ROC  (AUC = {auc:.3f})", fontweight="bold")
    # (4) locus-fit z_best vs z_spec
    ax = axes[3]
    zs = dfq["qso_z"].to_numpy(float)
    ax.hexbin(zs, zbest_q, gridsize=50, bins="log", cmap="viridis",
              extent=(0, 4.2, 0, 4.2), mincnt=1)
    ax.plot([0, 4.2], [0, 4.2], "k-", lw=1)
    ax.set_xlim(0, 4.2); ax.set_ylim(0, 4.2)
    ax.set_xlabel("spectroscopic $z$"); ax.set_ylabel("locus-fit $z_{best}$ (argmin)")
    ax.set_title("degeneracy-aware locus photo-z", fontweight="bold")
    ax.text(0.03, 0.97, f"σ_NMAD={zst['sigma_nmad']:.3f}\noutliers={100*zst['outlier_frac']:.1f}%",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    fig.suptitle("QSO Euclid colour(z) locus & marginalized-over-z QSO-consistency test",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")
    return {"auc": auc, "chi2r_95_cut": float(thr), "star_contam_at_95compl": star_contam,
            "locus_zphot": zst}


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replot", action="store_true",
                    help="use the cached star control instead of re-querying WSDB.")
    args = ap.parse_args()

    dfq = build_dataframe(os.getenv("PGUSER", WSDB_USER)).dropna(subset=[c for c, _ in COLORS] + ["qso_z"])
    dfq = dfq[dfq["qso_z"] > 0].reset_index(drop=True)
    print(f"  {len(dfq):,} spec QSOs")

    locus = build_locus(dfq)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    locus.to_csv(LOCUS_CSV, index=False)
    print(f"  -> {LOCUS_CSV}")

    # star control
    if args.replot and CONTROL_CSV.exists():
        dfs = pd.read_csv(CONTROL_CSV)
        print(f"  loaded {len(dfs):,} control stars from cache")
    else:
        dfs = fetch_star_control(os.getenv("PGUSER", WSDB_USER)).dropna(subset=[c for c, _ in COLORS])

    cq = dfq[[c for c, _ in COLORS]].to_numpy(float)
    cs = dfs[[c for c, _ in COLORS]].to_numpy(float)
    d2q, zbest_q = d2_min(cq, locus)
    d2s, _       = d2_min(cs, locus)

    met = make_plot(locus, dfq, d2q, zbest_q, d2s, PNG_PATH)
    (OUTPUT_DIR / "qso_euclid_locus_metrics.json").write_text(json.dumps(met, indent=2))
    print(f"  AUC={met['auc']:.3f}  star contam @95% QSO compl={100*met['star_contam_at_95compl']:.1f}%  "
          f"locus z σ_NMAD={met['locus_zphot']['sigma_nmad']:.3f}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
