#!/usr/bin/env python3
"""Distribution of the 5/6-par neighbour count used to build each proxy PM.

build_gaia_pm_proxy.py stores, per 2-par source, how many 5/6-parameter Gaia
neighbours within 30" went into its median-PM proxy — the `n_neighbours` column
of {PGUSER}.gaia_pm_proxy (a row is only written when n_neighbours >= 5). This
script summarises and plots that column.

Outputs (output/gaia_2par_neighbours/):
    pm_proxy_n_neighbours.png        2-panel: rebinned histogram + cumulative curve
    pm_proxy_n_neighbours.csv        raw (n_neighbours, n_sources) bins

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_pm_proxy_neighbours.py       # query WSDB + plot
    python src/analyze_pm_proxy_neighbours.py --replot                     # re-plot from saved CSV (no DB)
    python src/analyze_pm_proxy_neighbours.py --replot --binwidth 10       # coarser histogram bins
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

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB = "wsdb"
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output" / "gaia_2par_neighbours"
CSV_PATH = OUT_DIR / "pm_proxy_n_neighbours.csv"
PNG_PATH = OUT_DIR / "pm_proxy_n_neighbours.png"


def _wq(x, w, q):
    """Weighted quantile of value-array x with integer weights w."""
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, q * c[-1])])


def stats_from_hist(x, w):
    total = w.sum()
    return dict(n=int(total), lo=int(x.min()), hi=int(x.max()),
                mean=float((x * w).sum() / total),
                med=_wq(x, w, 0.5), p25=_wq(x, w, 0.25),
                p75=_wq(x, w, 0.75), p95=_wq(x, w, 0.95))


def fetch_from_db() -> pd.DataFrame:
    import psycopg2
    schema = os.getenv("PGUSER")
    if not schema:
        raise SystemExit("ERROR: PGUSER not set")
    # the proxy is now split per-catalogue; union the two (rows are disjoint by field)
    tabs = [f"{schema}.gaia_pm_proxy_ero", f"{schema}.gaia_pm_proxy_q1"]
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=schema)
    cur = conn.cursor()
    present = []
    for t in tabs:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (t,))
        if cur.fetchone()[0]:
            present.append(t)
    if not present:
        raise SystemExit("gaia_pm_proxy_ero / gaia_pm_proxy_q1 not found — run the "
                         "crossmatch_*_pm_proxy.py builders first.")
    union = " UNION ALL ".join(f"SELECT n_neighbours FROM {t}" for t in present)
    cur.execute(f"SELECT n_neighbours, count(*) FROM ({union}) u "
                f"GROUP BY n_neighbours ORDER BY n_neighbours")
    hist = pd.DataFrame(cur.fetchall(), columns=["n_neighbours", "n_sources"])
    conn.close()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hist.to_csv(CSV_PATH, index=False)
    return hist


def plot(hist: pd.DataFrame, st: dict, binwidth: int, out: Path) -> None:
    x = hist["n_neighbours"].to_numpy(dtype=float)
    w = hist["n_sources"].to_numpy(dtype=float)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.4))

    # ── left: rebinned histogram (smooths the noisy tail), log y ──────────────
    edges = np.arange(5, st["hi"] + binwidth, binwidth)
    counts, _ = np.histogram(x, bins=edges, weights=w)
    centers = edges[:-1] + binwidth / 2.0
    axL.bar(centers, counts, width=binwidth * 0.92, align="center",
            color="#3cb44b", edgecolor="white", linewidth=0.3)
    axL.set_yscale("log")
    axL.axvline(5, color="k", ls=":", lw=1.3, label="min = 5 (proxy threshold)")
    axL.axvline(st["med"], color="r", ls="--", lw=1.6, label=f"median = {st['med']:g}")
    axL.axvline(st["mean"], color="#ff7f0e", ls="-.", lw=1.6, label=f"mean = {st['mean']:.0f}")
    axL.set_xlabel('5/6-par neighbours within 30"  (n_neighbours)')
    axL.set_ylabel(f"2-par sources with a proxy  (per {binwidth}-neighbour bin)")
    axL.set_title(f"Neighbours used per proxy PM   (N = {st['n']:,})")
    axL.legend()

    # ── right: cumulative fraction, with percentile guides ────────────────────
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cum = np.cumsum(ws) / ws.sum()
    axR.step(xs, cum, where="post", color="#3cb44b", lw=2.0)
    for q, lbl in [(0.25, "p25"), (0.50, "median"), (0.75, "p75"), (0.95, "p95")]:
        xv = _wq(xs, ws, q)
        axR.hlines(q, 0, xv, color="0.6", ls=":", lw=1.0)
        axR.vlines(xv, 0, q, color="0.6", ls=":", lw=1.0)
        axR.annotate(f"{lbl} = {xv:g}", (xv, q), textcoords="offset points",
                     xytext=(6, -12), fontsize=9, color="0.25")
    axR.set_xlim(0, st["hi"]); axR.set_ylim(0, 1.02)
    axR.set_xlabel('5/6-par neighbours within 30"  (n_neighbours)')
    axR.set_ylabel("cumulative fraction of proxied 2-par sources")
    axR.set_title("Cumulative distribution")

    fig.suptitle("gaia_pm_proxy — 5/6-par neighbours per proxy PM",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replot", action="store_true",
                    help="re-plot from the saved CSV instead of querying WSDB")
    ap.add_argument("--binwidth", type=int, default=5,
                    help="histogram bin width in neighbours (default 5)")
    args = ap.parse_args()

    if args.replot:
        if not CSV_PATH.exists():
            print(f"{CSV_PATH} not found — run without --replot once first."); return 1
        hist = pd.read_csv(CSV_PATH)
    else:
        hist = fetch_from_db()

    x = hist["n_neighbours"].to_numpy(dtype=float)
    w = hist["n_sources"].to_numpy(dtype=float)
    st = stats_from_hist(x, w)
    print("=" * 60)
    print("gaia_pm_proxy.n_neighbours  (5/6-par neighbours per proxy)")
    print("=" * 60)
    print(f"  rows (2-par sources with a proxy) : {st['n']:,}")
    print(f"  min / median / max                : {st['lo']} / {st['med']:g} / {st['hi']}")
    print(f"  mean                              : {st['mean']:.1f}")
    print(f"  p25 / p75 / p95                   : {st['p25']:g} / {st['p75']:g} / {st['p95']:g}")

    plot(hist, st, args.binwidth, PNG_PATH)
    print(f"\n  -> {PNG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
