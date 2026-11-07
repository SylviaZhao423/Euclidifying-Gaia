#!/usr/bin/env python3
"""
STAGE 2: Euclid colour(z) rivers from PHOTO-Z ALONE over the full Q1 QSO-classified
population — no spectroscopy.

Euclid's PHZ classifies 697,773 Q1 sources as QSO (phz_classification = 4). Using ONLY their
Euclid photo-z (phz_mode_1, the most-probable peak) as the x-axis — no spec-z — do their
Euclid colours trace the SAME rivers as the ~2,828 spectroscopically-confirmed QSOs? This is
the real application: the WISE+Euclid pair candidates have no spec-z, so the locus must be
recoverable from Euclid photo-z on a photo-selected sample.

Colours from euclid_q1.mer_final_cat (native 2FWHM aperture, S/N>=3). Population shown as a
2D density (hexbin); the spec-confirmed QSO spec-z river (from crossmatch_qso_desi_dr1_euclid)
is over-plotted (black dashed) as the ground-truth reference on each colour panel.

Reads : {PGUSER}.q1_phz_photo_z  (phz_classification=4; phz_mode_1, phz_median)
        euclid_q1.mer_final_cat  (flux_{vis,y,j,h}_2fwhm_aper, by object_id)
        {PGUSER}.crossmatch_qso_desi_dr1_euclid (Q1 spec QSOs -> spec-z reference river)
Writes: output/qso_photz_rivers/q1_qso_photz_rivers_pop.png

Usage: PGUSER=siwei_zhao_2025 python src/analyze_q1_qso_photz_rivers_pop.py [--zcol phz_median]
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
from analyze_qso_color_vs_specz import (COLORS, PALETTE, COLOR_CMAPS, ZMAX_PLOT,  # noqa: E402
                                         _flux_to_mag, ZPC_VIS, _running_stats,
                                         WSDB_HOST, WSDB_DB, WSDB_USER)

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_DIR / "output" / "qso_photz_rivers"
PNG_PATH = OUT_DIR / "q1_qso_photz_rivers_pop.png"
MER = "euclid_q1.mer_final_cat"


def _colours(d: pd.DataFrame) -> pd.DataFrame:
    d["IE"] = _flux_to_mag(d["fv"], d["ev"], ZPC_VIS)
    d["YE"] = _flux_to_mag(d["fy"], d["ey"])
    d["JE"] = _flux_to_mag(d["fj"], d["ej"])
    d["HE"] = _flux_to_mag(d["fh"], d["eh"])
    for _, b, r in COLORS:
        d[f"{b}_{r}"] = d[b] - d[r]
    return d


_FLUX = ("m.flux_vis_2fwhm_aper AS fv, m.fluxerr_vis_2fwhm_aper AS ev, "
         "m.flux_y_2fwhm_aper AS fy, m.fluxerr_y_2fwhm_aper AS ey, "
         "m.flux_j_2fwhm_aper AS fj, m.fluxerr_j_2fwhm_aper AS ej, "
         "m.flux_h_2fwhm_aper AS fh, m.fluxerr_h_2fwhm_aper AS eh")


def pull_pop(user: str) -> pd.DataFrame:
    """Euclid QSO-classified (phz_classification=4) sources + 2FWHM colours + PHZ z's."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30,
                            keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5)
    print(f"  q1_phz_photo_z (class=4 QSO) x {MER} ...")
    d = pd.read_sql_query(
        f"""SELECT z.object_id, z.phz_median, z.phz_mode_1, {_FLUX}
            FROM {user}.q1_phz_photo_z z
            JOIN {MER} m ON m.object_id = z.object_id
            WHERE z.phz_classification = 4""", conn)
    conn.close()
    print(f"    {len(d):,} Euclid-QSO sources")
    return _colours(d)


def pull_spec(user: str) -> pd.DataFrame:
    """DESI Q1 spec QSOs + 2FWHM colours + spec-z (the reference river)."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30,
                            keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5)
    d = pd.read_sql_query(
        f"""SELECT x.qso_z, {_FLUX}
            FROM {user}.crossmatch_qso_desi_dr1_euclid x
            JOIN {MER} m ON m.object_id = x.e_id::bigint
            WHERE x.euclid_source='Q1' AND x.qso_z>0 AND x.e_id IS NOT NULL""", conn)
    conn.close()
    print(f"    {len(d):,} DESI Q1 spec QSOs (reference)")
    return _colours(d)


def make_plot(pop: pd.DataFrame, spec: pd.DataFrame, zcol: str, out: Path):
    zp = pop[zcol].to_numpy(float)
    zs = spec["qso_z"].to_numpy(float)
    # shared y-range from the population 1-99 pct
    pooled = []
    for _, b, r in COLORS:
        c = pop[f"{b}_{r}"].to_numpy(float); c = c[np.isfinite(c)]
        pooled += list(np.percentile(c, [1, 99]))
    pad = 0.12 * (max(pooled) - min(pooled)); YLIM = (min(pooled) - pad, max(pooled) + pad)

    fig, axes = plt.subplots(1, len(COLORS), figsize=(6.0 * len(COLORS), 5.6), squeeze=False)
    for j, (label, b, r) in enumerate(COLORS):
        ax = axes[0][j]; col = f"{b}_{r}"
        cp = pop[col].to_numpy(float)
        mp = np.isfinite(cp) & np.isfinite(zp) & (zp >= 0) & (zp <= ZMAX_PLOT)
        hb = ax.hexbin(zp[mp], cp[mp], gridsize=90, bins="log", cmap=COLOR_CMAPS[col],
                       mincnt=1, extent=(0, ZMAX_PLOT, YLIM[0], YLIM[1]))
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02).set_label("QSO-pop density (log)", fontsize=8)
        # population running median (coloured, on its OWN photo-z)
        mid, med, lo, hi = _running_stats(zp[mp], cp[mp])
        ln, = ax.plot(mid, med, color=PALETTE[col], lw=2.4, label=f"Euclid-QSO pop median ({zcol})")
        ln.set_path_effects([pe.Stroke(linewidth=4, foreground="white"), pe.Normal()])
        wpop = float(np.median(hi - lo))
        # spec-confirmed spec-z reference river (black dashed)
        cs = spec[col].to_numpy(float); ms = np.isfinite(cs) & np.isfinite(zs) & (zs <= ZMAX_PLOT)
        rmid, rmed, rlo, rhi = _running_stats(zs[ms], cs[ms])
        ax.plot(rmid, rmed, color="k", lw=2.0, ls="--", label="spec-QSO median (spec-$z$)")
        wspec = float(np.median(rhi - rlo))
        ax.set_xlim(0, ZMAX_PLOT); ax.set_ylim(*YLIM)
        ax.set_xlabel(f"Euclid PHZ photo-$z$ ({zcol})", fontsize=10)
        ax.set_ylabel(f"{label}  (AB)", fontsize=11)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.text(0.97, 0.03, f"N={mp.sum():,}\npop width={wpop:.2f}\nspec width={wspec:.2f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.92))
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle(f"Rivers from photo-z ALONE — {len(pop):,} Euclid PHZ-QSO sources (class=4), "
                 f"x = {zcol}, NO spectroscopy\n"
                 f"colour density + population median vs the spec-confirmed QSO river "
                 f"(black dashed): does the photo-selected population trace the same locus?",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=104, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zcol", default="phz_mode_1", choices=["phz_mode_1", "phz_median"],
                    help="which PHZ point estimate on the x-axis (default phz_mode_1).")
    args = ap.parse_args()
    user = os.getenv("PGUSER", WSDB_USER)
    pop = pull_pop(user)
    spec = pull_spec(user)
    make_plot(pop, spec, args.zcol, PNG_PATH)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
