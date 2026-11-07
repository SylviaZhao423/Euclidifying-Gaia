#!/usr/bin/env python3
"""
Colour distribution of the Euclid **Q1** source population — the background against which
the QSO colour locus lives. Histograms I_E-Y_E, Y_E-J_E, J_E-H_E and the RAW colour ranges,
so we can judge whether the spectroscopic-QSO locus is actually a narrow "river"
relative to how wide Euclid colours spread in general.

Q1 ONLY. ERO is dropped here: it has only Kron-AUTO photometry (no per-FWHM aperture) and
its fluxes are raw ADU (not µJy), so its colours are neither aperture-comparable nor on the
same system — mixing it in would just be misleading.

Population (default): the FULL euclid_q1.mer_final_cat (~30.0M) via a fast page-level
TABLESAMPLE (not an ORDER BY random() full sort). This is unbiased. With --gaia it instead
uses the Gaia-matched crossmatch_q1_mer_phz_vis_gaia_sep — fast but magnitude-biased (Gaia
only detects the brighter Euclid sources), for comparison.

Same flux->AB-mag convention as the pipeline (23.90 - 2.5*log10; VIS ×1.05 ZPC; S/N>=3).
The histogram is scaled by total_rows/n_sampled to estimate full-catalogue counts per bin.
Overlays the spec-QSO colour distribution (built live) to show where QSOs sit vs the bulk.

APERTURE SWEEP (default): colours are computed at ALL of 1/2/3/4 FWHM aperture diameters
(flux_{vis,y,j,h}_{N}fwhm_aper) — one plot per aperture — so the population distribution can
be compared aperture-by-aperture, matching analyze_qso_color_vs_specz.py. Same sources
each time (selected by the 2FWHM detection). --aperture N restricts to a single size.

Reads:  euclid_q1.mer_final_cat  (or the --gaia crossmatch_q1_mer_phz_vis_gaia_sep)
        spec-QSO overlay built live via analyze_qso_color_vs_specz.build_dataframe()
Writes: output/qso_color_vs_z/q1_color_dist[_Nfwhm].png  (2FWHM keeps the
        canonical name; 1/3/4 get a _Nfwhm suffix)
        output/qso_color_vs_z/arrays/q1_color_dist_sample.csv (cache;
        holds all apertures so --replot redraws any/all offline)

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_q1_color_dist.py           # full MER, all apertures
  PGUSER=siwei_zhao_2025 python src/analyze_q1_color_dist.py --aperture 2
  PGUSER=siwei_zhao_2025 python src/analyze_q1_color_dist.py --gaia     # Gaia-matched (biased)
  python src/analyze_q1_color_dist.py --replot                          # redraw from cache
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
from analyze_qso_color_vs_specz import (COLORS, build_dataframe, _flux_to_mag, ZPC_VIS,  # noqa: E402
                                        WSDB_HOST, WSDB_DB, WSDB_USER)

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR  = PROJECT_DIR / "output" / "qso_color_vs_z"
PNG_PATH = OUT_DIR / "q1_color_dist.png"
CACHE    = OUT_DIR / "arrays" / "q1_color_dist_sample.csv"

CCOLS = [f"{b}_{r}" for _, b, r in COLORS]
Q1_COLOR = "#4363d8"
APERS = [1, 2, 3, 4]                         # Q1 aperture sweep (FWHM diameters)
Q1_BANDS = [("IE", "vis"), ("YE", "y"), ("JE", "j"), ("HE", "h")]   # our band -> Euclid suffix

# Q1 ONLY (ERO dropped: Kron-AUTO photometry, no per-FWHM aperture, and raw ADU — not
# comparable). (table, TABLESAMPLE %, column prefix).
SOURCE_ORIG = ("euclid_q1.mer_final_cat", 0.05, "")
SOURCE_GAIA = ("siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep", None, "m_")


def _q1_select(prefix: str) -> tuple[str, str]:
    """SELECT cols (all 4 apertures) + WHERE (2FWHM detection) for a Q1-style table."""
    cols = []
    for _, s in Q1_BANDS:
        for a in APERS:
            cols.append(f"{prefix}flux_{s}_{a}fwhm_aper AS f_{s}_{a}")
            cols.append(f"{prefix}fluxerr_{s}_{a}fwhm_aper AS e_{s}_{a}")
    where = " AND ".join(f"{prefix}flux_{s}_2fwhm_aper>0" for _, s in Q1_BANDS)
    return ", ".join(cols), where


def _colours(fv, ev, fy, ey, fj, ej, fh, eh) -> pd.DataFrame:
    """Four fluxes/errs -> the three adjacent-band colours (AB), NaN where S/N<3."""
    IE = _flux_to_mag(fv, ev, ZPC_VIS); YE = _flux_to_mag(fy, ey)
    JE = _flux_to_mag(fj, ej);          HE = _flux_to_mag(fh, eh)
    return pd.DataFrame({"IE_YE": IE - YE, "YE_JE": YE - JE, "JE_HE": JE - HE})


def _to_long(raw: pd.DataFrame, total: int) -> pd.DataFrame:
    """Per-aperture Q1 colour blocks in LONG format (3 colours + aper/total/scale)."""
    out = []
    for a in APERS:
        d = _colours(raw[f"f_vis_{a}"], raw[f"e_vis_{a}"], raw[f"f_y_{a}"], raw[f"e_y_{a}"],
                     raw[f"f_j_{a}"], raw[f"e_j_{a}"], raw[f"f_h_{a}"], raw[f"e_h_{a}"]).dropna()
        d["aper"], d["total"] = a, total
        d["scale"] = total / max(1, len(d))
        out.append(d)
        print(f"    Q1 {a}FWHM: {len(d):,} good-photometry sources", flush=True)
    return pd.concat(out, ignore_index=True)


def fetch(user: str, use_gaia: bool) -> pd.DataFrame:
    import psycopg2
    table, pct, prefix = SOURCE_GAIA if use_gaia else SOURCE_ORIG
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                            connect_timeout=30, keepalives=1, keepalives_idle=30,
                            keepalives_interval=10, keepalives_count=5)
    cur = conn.cursor(); cur.execute("SET statement_timeout = 900000")
    cur.execute(f"SELECT reltuples::bigint FROM pg_class WHERE oid='{table}'::regclass")
    total = int(cur.fetchone()[0])
    samp = f"TABLESAMPLE SYSTEM ({pct})" if pct else ""
    cols, where = _q1_select(prefix)
    print(f"  reading {table} (~{total/1e6:.1f}M){' at SYSTEM '+str(pct)+'%' if pct else ' (full)'} ...",
          flush=True)
    raw = pd.read_sql_query(f"SELECT {cols} FROM {table} {samp} WHERE {where}", conn)
    cur.close(); conn.close()
    out = _to_long(raw, total)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE, index=False)
    print(f"  -> {CACHE}  ({len(out):,} rows, all apertures)")
    return out


# viridis colours for the 4 apertures (purple→teal→green→yellow)
APER_COLORS = [plt.cm.viridis(x) for x in np.linspace(0.1, 0.88, len(APERS))]


def _fetch_qso_frames(user: str) -> dict:
    """Pull spec-QSO colours at all 4 apertures from mer_final_cat. Returns {aper: df|None}."""
    frames = {}
    for a in APERS:
        try:
            frames[a] = build_dataframe(user, a)
            print(f"  spec-QSO {a}×FWHM: {len(frames[a]):,} rows")
        except Exception as exc:
            print(f"  spec-QSO {a}×FWHM skipped: {exc}")
            frames[a] = None
    return frames


def print_ranges(df: pd.DataFrame):
    """Raw colour spread per aperture vs the spec-QSO (marginalised over z)."""
    for a in APERS:
        sub = df[df["aper"] == a]
        print(f"  Q1 {a}×FWHM (N={len(sub):,}):")
        for _, b, r in COLORS:
            v = sub[f"{b}_{r}"].to_numpy(float); v = v[np.isfinite(v)]
            if len(v) == 0: continue
            p = np.percentile(v, [1, 16, 50, 84, 99])
            print(f"    {b}-{r}: med={p[2]:+.2f}  16-84%={p[3]-p[1]:.2f}  1-99%={p[4]-p[0]:.2f}")


def make_plot(df: pd.DataFrame, qso_by_aper: dict, out: Path):
    """All 4 apertures overlaid in each panel — viridis step lines for Q1 (solid) and
    spec-QSO (dashed, density on right axis). One combined legend per panel."""
    from matplotlib.lines import Line2D
    total_m = int(df["total"].iloc[0]) / 1e6
    fig, axes = plt.subplots(1, len(COLORS), figsize=(7 * len(COLORS), 4.3))
    for ax, (label, b, r) in zip(axes, COLORS):
        col = f"{b}_{r}"
        allvals = df[col].to_numpy(float); allvals = allvals[np.isfinite(allvals)]
        lo, hi = np.percentile(allvals, [0.5, 99.5])
        bins = np.arange(lo, hi + 0.05, 0.05)
        for a in reversed(APERS):
            i = APERS.index(a)   # keep colour mapping: purple=1, yellow=4
            c = APER_COLORS[i]
            sub = df[df["aper"] == a]
            # Q1 population = coloured LINES only (dashed, no fill)
            ax.hist(sub[col], bins=bins, histtype="step", color=c, lw=1.7, ls="--", label="_nolegend_")
            # spec-QSO = coloured FILLED (+ solid outline)
            qdf = qso_by_aper.get(a)
            if qdf is not None and col in qdf.columns:
                qc = pd.to_numeric(qdf[col], errors="coerce").dropna()
                ax.hist(qc, bins=bins, histtype="stepfilled", color=c, alpha=0.1, lw=0, label="_nolegend_")
                ax.hist(qc, bins=bins, histtype="step", color=c, lw=1.7, ls="-",
                        label=f"{a}×FWHM QSO  (n={len(qc):,})")
        h, l = ax.get_legend_handles_labels()
        h.append(Line2D([0], [0], color="0.4", lw=1.2, ls="--")); l.append("Q1 population (dashed lines)")
        ax.legend(h, l, loc="upper right", fontsize=7.5, framealpha=0.92)
        # ax.set_yscale("log"); ax.set_xlim(lo, hi)
        ax.set_xlabel(f"{label}  (AB)", fontsize=11)
        # ax.set_ylabel("counts / 0.05 mag bin", fontsize=10)
        ax.set_title(label, fontsize=13, fontweight="bold")
    fig.suptitle(
        f"Euclid Q1 colour distribution (full MER ~{total_m:.1f}M) vs spec-QSOs  ·  "
        f"all apertures overlaid",
        fontsize=13, fontweight="bold", y=0.92)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=105, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gaia", action="store_true",
                    help="use the Gaia-matched cross-matches (fast, magnitude-biased) instead of full MER.")
    ap.add_argument("--replot", action="store_true",
                    help="redraw from the cached Q1 sample (no DB fetch for Q1); still pulls QSO overlay.")
    args = ap.parse_args()
    user = os.getenv("PGUSER", WSDB_USER)

    if args.replot and CACHE.exists():
        df = pd.read_csv(CACHE); print(f"  loaded {len(df):,} cached rows from {CACHE.name}")
    else:
        df = fetch(user, args.gaia)

    print("\nfetching spec-QSO overlay (all apertures) ...")
    qso_by_aper = _fetch_qso_frames(user)

    print("\nRAW colour ranges:")
    print_ranges(df)
    make_plot(df, qso_by_aper, PNG_PATH)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
