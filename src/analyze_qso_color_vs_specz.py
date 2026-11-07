#!/usr/bin/env python3
"""
STEP 1 of the QSO-companion colour pipeline: the Euclid colour(z) "rivers" plot.

Spectroscopically confirmed QSOs (DESI DR1) that fall in the Euclid Q1 footprint have
BOTH a spectroscopic redshift and Euclid VIS/Y/J/H photometry. Plot their Euclid
colours as a function of redshift and you see narrow, wiggly "rivers": at fixed z,
QSOs all look alike so the colour distribution is narrow; the wiggles come from strong
emission lines (Lyα, CIV, MgII, Hα ...) sweeping in and out of the broad Euclid
passbands as z changes (Belokurov, 2026-07).

Why this matters for the science goal
-------------------------------------
The WISE+Euclid QSO-pair candidates have NO spectroscopy, so we don't know their z.
But if the colour(z) locus is narrow, then (Step 2) an XGBoost regressor can predict
z from Euclid colours, and (Step 3) we can test whether a close companion's Euclid
colours fall inside the QSO locus at that z — i.e. "is the companion also a QSO?".
This script is the sanity check that the locus really IS narrow enough to build on.

Sample (verified 2026-07-27, on WSDB)
-------------------------------------
  DESI DR1 × Euclid Q1: 2,843 spec QSOs  (SDSS×Euclid = 18, too few to add signal)
Redshift coverage is dense z~0.5-3, thin z>3.5 (few high-z anchors).

Photometry
----------
All fluxes are pulled fresh from euclid_q1.mer_final_cat (joined by e_id from the
DESI crossmatch table), so every aperture size uses the same code path. The --aperture
{1,2,3,4} flag selects flux_{vis,y,j,h}_{N}fwhm_aper; default is 2 (the pipeline
canonical, PSF-matched). See analyze_qso_desi_q1_aperture_fluxes.py for side-by-side
aperture comparison (NISP colours aperture-invariant; I_E-Y_E drifts ~0.1 mag).
AB mag = 23.90 - 2.5*log10(ZPC * flux); ZPC = 1.05 for VIS (project convention, see
crossmatch_q1_mer_phz.py), 1.0 for NISP. Only fluxes > 0 with S/N >= SNR_MIN used.

Requires WSDB (Astronomy/Cambridge VPN; auth via ~/.pgpass, no password arg). Reads:
  {PGUSER}.crossmatch_qso_desi_dr1_euclid   (build: src/crossmatch_qso_desi_dr1_euclid.py)
  euclid_q1.mer_final_cat                    (always; joined by e_id for aperture fluxes)
Writes (2FWHM keeps the canonical png name, other apertures get a _NfWhm suffix):
  output/qso_color_vs_z/qso_color_vs_specz[_Nfwhm].png          (the rivers plot)
  output/qso_color_vs_z/arrays/rivers_{qso,pop}_Nfwhm.csv        (--replot cache)

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_qso_color_vs_specz.py            # 2FWHM pull+plot (caches)
  PGUSER=siwei_zhao_2025 python src/analyze_qso_color_vs_specz.py --aperture 3
  PGUSER=siwei_zhao_2025 python src/analyze_qso_color_vs_specz.py --all-apertures  # 1..4
  python src/analyze_qso_color_vs_specz.py --replot                          # redraw from cache (no WSDB)
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

PROJECT_DIR = Path(__file__).resolve().parents[1]

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
WSDB_USER = "siwei_zhao_2025"

OUTPUT_DIR = PROJECT_DIR / "output" / "qso_color_vs_z"
PNG_PATH   = OUTPUT_DIR / "qso_color_vs_specz.png"
ARRAYS_DIR = OUTPUT_DIR / "arrays"   # --replot cache (spec-QSO frame + population sample per aperture)

# DESI DR1 × Euclid crossmatch (SDSS×Euclid = 18 rows — too few to add signal, dropped).
DESI_TABLE = "crossmatch_qso_desi_dr1_euclid"

ZP_AB   = 23.90
ZPC_VIS = 1.05        # VIS-band zero-point correction (project convention)
SNR_MIN = 3.0         # min flux/fluxerr per band to trust a magnitude

# Euclid colours to plot (label, blue band, red band). Adjacent bands = classic QSO
# colour sequence spanning VIS -> NISP.
COLORS = [
    ("$I_E - Y_E$", "IE", "YE"),
    ("$Y_E - J_E$", "YE", "JE"),
    ("$J_E - H_E$", "JE", "HE"),
]

ZMAX_PLOT = 4.0       # x-axis cap (few QSOs beyond, see docstring)


# ----------------------------------------------------------------------------- data
MER_TABLE = "euclid_q1.mer_final_cat"
_BAND_DB  = {"IE": "vis", "YE": "y", "JE": "j", "HE": "h"}   # our band -> Euclid db suffix
KEEP_COLS = ["qso_catalog", "qso_id", "qso_z", "IE", "YE", "JE", "HE",
             *[f"{b}_{r}" for _, b, r in COLORS]]


def _flux_to_mag(flux: np.ndarray, err: np.ndarray, zpc: float = 1.0):
    """AB mag from an aperture flux, only where flux>0 and S/N>=SNR_MIN (else NaN)."""
    flux = pd.to_numeric(flux, errors="coerce").to_numpy(dtype=float)
    err  = pd.to_numeric(err,  errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(flux) & (flux > 0) & np.isfinite(err) & (err > 0) & (flux / err >= SNR_MIN)
    mag = np.full(flux.shape, np.nan)
    mag[good] = ZP_AB - 2.5 * np.log10(zpc * flux[good])
    return mag


def _png_for(aper: int) -> Path:
    return PNG_PATH if aper == 2 else OUTPUT_DIR / f"qso_color_vs_specz_{aper}fwhm.png"


def build_dataframe(user: str, aper: int = 2) -> pd.DataFrame:
    """Pull DESI×Euclid Q1 ids from the crossmatch table, join to mer_final_cat for
    {aper}FWHM aperture fluxes (default 2), convert to AB mags, return colour(z) frame.
    Works identically for aper in {1, 2, 3, 4} — always reads from the original Q1 table."""
    import psycopg2
    conn = psycopg2.connect(
        host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30,
        keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5,
    )
    xmatch = f"{user}.{DESI_TABLE}"
    print(f"  reading {xmatch} (Q1 ids) ...")
    ids = pd.read_sql_query(
        f"""SELECT e_id::bigint AS object_id, qso_catalog, qso_id, qso_z
            FROM {xmatch}
            WHERE euclid_source = 'Q1' AND qso_z IS NOT NULL AND qso_z > 0
              AND e_id IS NOT NULL""", conn)
    ids = ids.drop_duplicates("object_id")
    print(f"  {len(ids):,} unique Q1 QSO Euclid ids")

    fcols = [f"flux_{s}_{aper}fwhm_aper" for s in _BAND_DB.values()]
    ecols = [f"fluxerr_{s}_{aper}fwhm_aper" for s in _BAND_DB.values()]
    print(f"  querying {MER_TABLE} for {aper}×FWHM fluxes ...")
    mer = pd.read_sql_query(
        f"SELECT object_id, {', '.join(fcols + ecols)} FROM {MER_TABLE} "
        f"WHERE object_id = ANY(%(ids)s)",
        conn, params={"ids": ids["object_id"].tolist()})
    conn.close()

    df = ids.merge(mer, on="object_id", how="inner")
    print(f"  matched {len(df):,} in mer_final_cat")

    for band, s in _BAND_DB.items():
        zpc = ZPC_VIS if s == "vis" else 1.0
        df[band] = _flux_to_mag(df[f"flux_{s}_{aper}fwhm_aper"],
                                df[f"fluxerr_{s}_{aper}fwhm_aper"], zpc)
    for _, b, r in COLORS:
        df[f"{b}_{r}"] = df[b] - df[r]

    print(f"  built {len(df):,} QSO colour(z) rows ({aper}×FWHM)")
    return df[KEEP_COLS]


def build_dataframe_aper(user: str, aper: int) -> pd.DataFrame:
    """Alias kept for backward compatibility — delegates to build_dataframe(user, aper)."""
    return build_dataframe(user, aper)


# ----------------------------------------------------------------------------- plot
def _running_stats(z, c, zmax=ZMAX_PLOT, width=0.05, minpts=15):
    """Median + 16/84 percentile of colour c in fine z-bins (the river + its width)."""
    edges = np.arange(0.0, zmax + width, width)
    mid, med, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (z >= a) & (z < b)
        if m.sum() >= minpts:
            cc = c[m]
            mid.append(0.5 * (a + b))
            med.append(np.median(cc))
            lo.append(np.percentile(cc, 16))
            hi.append(np.percentile(cc, 84))
    return map(np.asarray, (mid, med, lo, hi))


# Per-colour palette (matches the combined locus panel and analyze_qso_euclid_locus.py).
PALETTE = {"IE_YE": "#e6194b", "YE_JE": "#3cb44b", "JE_HE": "#4363d8"}
# Sequential colormap per colour for the 2D density panels (darker = more QSOs).
COLOR_CMAPS = {"IE_YE": "Reds", "YE_JE": "Greens", "JE_HE": "Blues"}


def _hexbin_count_range(z, cc, extent, gridsize=80, mincnt=1) -> tuple[float, float]:
    """Min/max hexbin counts (for a shared log-density colorbar across panels)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    hb = ax.hexbin(z, cc, gridsize=gridsize, bins="log", mincnt=mincnt, extent=extent)
    arr = hb.get_array()
    if arr is None or len(arr) == 0:
        return 1.0, 1.0
    return float(np.min(arr)), float(np.max(arr))


def _river(ax, z, cc, col):
    """QSO running median (white-haloed so it reads over scatter) + 16–84% band. Returns
    the median 16–84% width."""
    mid, med, lo, hi = _running_stats(z, cc)
    cpal = PALETTE[col]
    ax.fill_between(mid, lo, hi, color=cpal, alpha=0.28, lw=0, label="QSO 16–84%")
    ln, = ax.plot(mid, med, color=cpal, lw=2.4, label="QSO median")
    ln.set_path_effects([pe.Stroke(linewidth=4, foreground="white"), pe.Normal()])
    return float(np.median(hi - lo))


def _combined_panel(ax, df):
    """All three QSO colour(z) rivers overlaid (μ ± σ) — the overview panel."""
    z = df["qso_z"].to_numpy(float)
    for label, b, r in COLORS:
        col = f"{b}_{r}"
        c = df[col].to_numpy(float)
        m = np.isfinite(c) & np.isfinite(z) & (z <= ZMAX_PLOT)
        mid, med, lo, hi = _running_stats(z[m], c[m])
        ax.fill_between(mid, lo, hi, color=PALETTE[col], alpha=0.20, lw=0)
        ax.plot(mid, med, color=PALETTE[col], lw=1.9, label=label)
    ax.set_xlim(0, 3.5)
    ax.set_xlabel("spectroscopic $z$", fontsize=11); ax.set_ylabel("Euclid colour (AB)", fontsize=11)
    ax.set_title("QSO colour(z) locus  μ ± σ", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)


def fetch_pop(user: str, aper: int, pct: float = 0.05) -> pd.DataFrame:
    """Full Q1 MER population colours at aperture N via fast page-level TABLESAMPLE — the
    background colour distribution for the left marginals (cf. analyze_q1_color_dist.py)."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30,
                            keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5)
    cur = conn.cursor()
    cur.execute("SELECT reltuples::bigint FROM pg_class WHERE oid='euclid_q1.mer_final_cat'::regclass")
    total = int(cur.fetchone()[0]); cur.close()
    bands = [("IE", "vis"), ("YE", "y"), ("JE", "j"), ("HE", "h")]
    cols = ", ".join(f"flux_{s}_{aper}fwhm_aper AS f_{s}, fluxerr_{s}_{aper}fwhm_aper AS e_{s}"
                     for _, s in bands)
    where = " AND ".join(f"flux_{s}_{aper}fwhm_aper > 0" for _, s in bands)
    print(f"  population TABLESAMPLE {pct}% of euclid_q1.mer_final_cat ({aper}FWHM) ...")
    d = pd.read_sql_query(
        f"SELECT {cols} FROM euclid_q1.mer_final_cat TABLESAMPLE SYSTEM ({pct}) WHERE {where}", conn)
    conn.close()
    d["IE"] = _flux_to_mag(d["f_vis"], d["e_vis"], ZPC_VIS); d["YE"] = _flux_to_mag(d["f_y"], d["e_y"])
    d["JE"] = _flux_to_mag(d["f_j"], d["e_j"]);              d["HE"] = _flux_to_mag(d["f_h"], d["e_h"])
    for _, b, r in COLORS:
        d[f"{b}_{r}"] = d[b] - d[r]
    d["_scale"] = total / max(1, len(d))   # sample -> full-catalogue count factor (à la q1_color_dist)
    print(f"    {len(d):,} population sources (×{d['_scale'].iloc[0]:.0f} to full ~{total/1e6:.0f}M)")
    return d


def make_plot(df: pd.DataFrame, out: Path, pop=None, aper: int = 2):
    z_all = df["qso_z"].to_numpy(dtype=float)
    # ONE y-range shared by every panel (union of the three colours' 1–99 percentiles)
    pooled = []
    for _, b, r in COLORS:
        c = df[f"{b}_{r}"].to_numpy(float); c = c[np.isfinite(c)]
        pooled += list(np.percentile(c, [1, 99]))
    cpad = 0.15 * (max(pooled) - min(pooled)); YLIM = (min(pooled) - cpad, max(pooled) + cpad)

    extent = (0, ZMAX_PLOT, YLIM[0], YLIM[1])
    panel_data = []
    dens_vmin, dens_vmax = np.inf, -np.inf
    for label, b, r in COLORS:
        col = f"{b}_{r}"
        c = df[col].to_numpy(dtype=float)
        m = np.isfinite(c) & np.isfinite(z_all) & (z_all <= ZMAX_PLOT)
        z, cc = z_all[m], c[m]
        lo, hi = _hexbin_count_range(z, cc, extent)
        dens_vmin = min(dens_vmin, lo)
        dens_vmax = max(dens_vmax, hi)
        panel_data.append((label, col, z, cc, m))

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, axes = plt.subplots(1, 1 + len(COLORS), figsize=(6.8 * (1 + len(COLORS)), 6.8))
    _combined_panel(axes[0], df); axes[0].set_ylim(*YLIM)   # first panel = combined locus
    for k, (label, col, z, cc, m) in enumerate(panel_data):
        ax = axes[1 + k]
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="4.5%", pad=0.06)
        axL = div.append_axes("left", size="27%", pad=0.10, sharey=ax)   # colour-distribution marginal
        # 2D density (darker = more QSOs) — shared log count range on all three panels
        hb = ax.hexbin(z, cc, gridsize=100, bins="log", cmap=COLOR_CMAPS[col], mincnt=1,
                       extent=extent, vmin=dens_vmin, vmax=dens_vmax)
        cb = fig.colorbar(hb, cax=cax)
        cb.set_label("QSO density (log)", fontsize=8, labelpad=0, loc="center"); cb.ax.tick_params(labelsize=7)
        mid, med, lo, hi = _running_stats(z, cc)
        w = float(np.median(hi - lo))
        ln, = ax.plot(mid, med, color=PALETTE[col], lw=2.4, label="QSO median")
        ln.set_path_effects([pe.Stroke(linewidth=4, foreground="white"), pe.Normal()])
        ax.text(0.97, 0.03, f"N = {m.sum():,}\nQSO 16–84% width = {w:.2f} mag",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.92))
        ax.set_xlim(0, 3.5); ax.set_ylim(*YLIM)
        ax.set_xlabel("spectroscopic $z$", fontsize=11)
        ax.set_ylabel(""); ax.tick_params(labelleft=False)   # colour label/ticks live on the marginal
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        # ---- left marginal = colour distribution, RAW counts on a SINGLE shared LINEAR axis
        # (like q1_color_dist.py, rotated). Same bins for pop & spec-QSO so they're directly
        # comparable; the concentrated QSO peak is naturally similar in height to the broad pop. ----
        qc = cc[(cc >= YLIM[0]) & (cc <= YLIM[1])]
        mbins = np.linspace(YLIM[0], YLIM[1], 56)   # shared bins
        if pop is not None and col in pop:
            pc = pop[col].to_numpy(float); pc = pc[np.isfinite(pc) & (pc >= YLIM[0]) & (pc <= YLIM[1])]
            if len(pc) > 20:
                axL.hist(pc, bins=mbins, orientation="horizontal", color="0.72", alpha=0.55, histtype="stepfilled")
                axL.hist(pc, bins=mbins, orientation="horizontal", color="0.40", lw=1.0, histtype="step")
        axL.hist(qc, bins=mbins, orientation="horizontal", color=PALETTE[col], alpha=0.40, histtype="stepfilled")
        axL.hist(qc, bins=mbins, orientation="horizontal", color=PALETTE[col], lw=1.5, histtype="step")
        axL.invert_xaxis(); axL.set_xticks([]); axL.set_ylim(*YLIM)   # single shared linear counts axis
        axL.set_ylabel(f"{label}  (AB)", fontsize=11)
        axL.tick_params(labelbottom=False, bottom=False)
        if k == 0:
            from matplotlib.lines import Line2D
            axL.legend([Line2D([0], [0], color="0.4", lw=2), Line2D([0], [0], color=PALETTE[col], lw=2)],
                       ["Q1 population", "spec QSO"], loc="lower left", fontsize=7, framealpha=0.85)

    fig.suptitle(
        f"{len(df):,} Spectroscopic QSOs (DESI DR1 × Euclid Q1) colour vs redshift  ·  {aper}×FWHM aperture photometry"
        f"\nwiggles = emission lines crossing passbands; left marginal = spec QSO vs full Q1 population colour distribution",
        fontsize=16, fontweight="bold", y=0.94)
    try:
        fig.tight_layout(rect=[0, 0, 1, 0.94])
    except Exception:
        fig.subplots_adjust(top=0.9, wspace=0.4)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


# ----------------------------------------------------------------------------- main
def _run_one(user: str, aper: int, replot: bool = False) -> None:
    """Draw the rivers plot for one aperture. Default pulls live from WSDB and caches both the
    spec-QSO frame and the population sample to arrays/; --replot reloads them (no WSDB)."""
    qcsv = ARRAYS_DIR / f"rivers_qso_{aper}fwhm.csv"
    pcsv = ARRAYS_DIR / f"rivers_pop_{aper}fwhm.csv"
    if replot:
        if not (qcsv.exists() and pcsv.exists()):
            print(f"  no cache for {aper}FWHM ({qcsv.name}) — run once without --replot first."); return
        df = pd.read_csv(qcsv); pop = pd.read_csv(pcsv)
        print(f"  loaded cache: {len(df):,} spec QSOs + {len(pop):,} population ({aper}FWHM)")
    else:
        df = build_dataframe(user, aper)
        pop = fetch_pop(user, aper)
        ARRAYS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(qcsv, index=False); pop.to_csv(pcsv, index=False)
        print(f"  cached -> {qcsv.name}, {pcsv.name}")
    make_plot(df, _png_for(aper), pop, aper=aper)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aperture", type=int, choices=[1, 2, 3, 4], default=2,
                    help="Euclid aperture size in FWHM (default 2 = pipeline canonical).")
    ap.add_argument("--all-apertures", action="store_true",
                    help="generate the rivers plot for all of 1/2/3/4 FWHM in one run.")
    ap.add_argument("--replot", action="store_true",
                    help="redraw from the cached spec-QSO + population CSVs (no WSDB, no VPN).")
    args = ap.parse_args()

    user = os.getenv("PGUSER", WSDB_USER)
    apers = [1, 2, 3, 4] if args.all_apertures else [args.aperture]
    for a in apers:
        print(f"\n=== {a} FWHM aperture ===")
        _run_one(user, a, args.replot)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
