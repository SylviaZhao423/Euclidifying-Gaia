#!/usr/bin/env python3
"""One-off ablation: do extra Euclid features (COMMON to Q1 & ERO, beyond the 4 bands)
improve the Euclid-colour photo-z? Base = 3 colours + H_E; add candidates and compare
OOF-CV metrics. Candidates are the non-NULL-in-ERO extras from builder _ECOLS:
  compactness = I_E(2fwhm aper) - I_E(psf)   (aperture vs PSF; extended host at low z)
  point_like_prob   (Q1 point_like_prob / ERO v_class_star)
  fwhm              (CAVEAT: Q1 arcsec vs ERO pixels — inhomogeneous; sample ~97% Q1)
Uses a reduced 3-seed OOF scan for speed (ranking); rerun the winner with full 10 seeds.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_qso_desi_q1_color_z import WSDB_HOST, WSDB_DB, WSDB_USER, _flux_to_mag, ZPC_VIS
import train_qso_euclid_zphot as T           # reuse oof_predict / zphot_metrics
T.SEEDS = [42, 123, 456]                      # 3-seed scan (speed); winner -> full 10

TABLES = ["crossmatch_qso_desi_dr1_euclid", "crossmatch_qso_sdss_dr16_euclid"]


def pull(user):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30,
                            keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5)
    frames = []
    for t in TABLES:
        frames.append(pd.read_sql_query(
            f"""SELECT qso_z, e_flux_vis, e_fluxerr_vis, e_flux_y, e_fluxerr_y,
                       e_flux_j, e_fluxerr_j, e_flux_h, e_fluxerr_h,
                       e_flux_vis_psf, e_fluxerr_vis_psf, e_point_like_prob, e_fwhm
                FROM {user}.{t} WHERE qso_z > 0
                  AND euclid_source = 'Q1'""", conn))   # Q1-only: ERO flux is raw ADU
    conn.close()
    d = pd.concat(frames, ignore_index=True)
    d["IE"] = _flux_to_mag(d.e_flux_vis, d.e_fluxerr_vis, ZPC_VIS)
    d["YE"] = _flux_to_mag(d.e_flux_y, d.e_fluxerr_y)
    d["JE"] = _flux_to_mag(d.e_flux_j, d.e_fluxerr_j)
    d["HE"] = _flux_to_mag(d.e_flux_h, d.e_fluxerr_h)
    d["IEpsf"] = _flux_to_mag(d.e_flux_vis_psf, d.e_fluxerr_vis_psf, ZPC_VIS)
    d["IE_YE"] = d.IE - d.YE; d["YE_JE"] = d.YE - d.JE; d["JE_HE"] = d.JE - d.HE
    d["compactness"] = d.IE - d.IEpsf
    d["plp"] = pd.to_numeric(d.e_point_like_prob, errors="coerce")
    d["fwhm"] = pd.to_numeric(d.e_fwhm, errors="coerce")
    return d


def run(d, feats):
    dd = d.dropna(subset=feats + ["qso_z"])
    X = dd[feats].to_numpy(float); zs = dd.qso_z.to_numpy(float)
    zp = T.oof_predict(X, zs, False)
    return len(dd), T.zphot_metrics(zs, zp)


def main():
    d = pull(os.getenv("PGUSER", WSDB_USER))
    print(f"  pulled {len(d):,} spec QSOs\n")
    BASE = ["IE_YE", "YE_JE", "JE_HE", "HE"]
    sets = {
        "base (3col + H_E)":  BASE,
        "+ compactness":      BASE + ["compactness"],
        "+ point_like_prob":  BASE + ["plp"],
        "+ fwhm":             BASE + ["fwhm"],
        "+ all three":        BASE + ["compactness", "plp", "fwhm"],
    }
    print(f"{'feature set':22} {'N':>6} {'sigma_NMAD':>11} {'outliers%':>10}")
    for name, feats in sets.items():
        n, m = run(d, feats)
        print(f"{name:22} {n:>6,} {m['sigma_nmad']:>11.3f} {100*m['outlier_frac']:>9.1f}", flush=True)


if __name__ == "__main__":
    main()
