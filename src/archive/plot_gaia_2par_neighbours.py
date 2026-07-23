#!/usr/bin/env python3
"""Distribution of Gaia neighbour counts (cnt4/8/16/32) for the Gaia DR3
2-parameter (position-only) sources.

Joins {PGUSER}.gaia_2par (built by extract_gaia_2par.py) to the precomputed
gaia_edr3_aux.gaia_source_neighbour_count catalogue and histograms the neighbour
counts at 4/8/16/32". The histogram is aggregated IN SQL (GROUP BY the count
value, single join scan via a LATERAL unnest), so only the ~few-thousand binned
rows are transferred — never the 344M sources.

Outputs (output/gaia_2par_neighbours/):
    gaia_2par_cnt_distribution.png     4-panel neighbour-count distribution (log y)
    gaia_2par_cnt_histogram.csv        raw (radius, count_value, n) bins

Usage:
    PGUSER=siwei_zhao_2025 python src/plot_gaia_2par_neighbours.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB = "wsdb"
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output" / "gaia_2par_neighbours"
NB_TABLE = "gaia_edr3_aux.gaia_source_neighbour_count"
RADII = [("cnt4", '4"'), ("cnt8", '8"'), ("cnt16", '16"'), ("cnt32", '32"')]


def _wq(x, w, q):
    """Weighted quantile of integer values x with weights w."""
    order = np.argsort(x)
    x, w = x[order], w[order]
    c = np.cumsum(w)
    return x[np.searchsorted(c, q * c[-1])]


def main() -> int:
    schema = os.getenv("PGUSER")
    if not schema:
        print("ERROR: PGUSER not set"); return 1
    src = f"{schema}.gaia_2par"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=schema)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 0;")

    cur.execute("SELECT to_regclass(%s) IS NOT NULL", (src,))
    if not cur.fetchone()[0]:
        print(f"{src} not found — run extract_gaia_2par.py first."); return 1

    print(f"Aggregating neighbour-count histogram in SQL ({src} ⋈ {NB_TABLE})...")
    t0 = time.time()
    cur.execute(f"""
        SELECT v.radius, v.cntval, count(*) AS n
        FROM {src} g
        JOIN {NB_TABLE} nb ON nb.source_id = g.source_id
        CROSS JOIN LATERAL (VALUES
            ('cnt4', nb.cnt4), ('cnt8', nb.cnt8),
            ('cnt16', nb.cnt16), ('cnt32', nb.cnt32)
        ) AS v(radius, cntval)
        GROUP BY v.radius, v.cntval
        ORDER BY v.radius, v.cntval
    """)
    hist = pd.DataFrame(cur.fetchall(), columns=["radius", "cntval", "n"])
    print(f"  {len(hist):,} (radius, count) bins in {time.time()-t0:.0f}s")
    hist.to_csv(OUT_DIR / "gaia_2par_cnt_histogram.csv", index=False)

    matched = int(hist.loc[hist.radius == "cnt4", "n"].sum())
    cur.execute(f"SELECT count(*) FROM {src}")
    ntot = cur.fetchone()[0]
    print(f"  2-par sources: {ntot:,};  with neighbour-count row: {matched:,} "
          f"({100*matched/ntot:.1f}%)")
    conn.close()

    # ── 4-panel distribution ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.3))
    for ax, (col, lbl) in zip(axes, RADII):
        h = hist[hist.radius == col].sort_values("cntval")
        x, y = h["cntval"].to_numpy(dtype=float), h["n"].to_numpy(dtype=float)
        med = _wq(x, y, 0.5)
        xhi = _wq(x, y, 0.999)
        ax.bar(x, y, width=1.0, align="center", color="#4363d8", edgecolor="none")
        ax.set_yscale("log")
        ax.set_xlim(-0.5, max(xhi, 5))
        ax.set_xlabel(f"{col}  (neighbours within {lbl})")
        ax.set_ylabel("Gaia 2-par sources")
        ax.set_title(f"r = {lbl}   (N={matched:,})")
        ax.axvline(med, color="r", ls="--", lw=1.5, label=f"median = {med:g}")
        ax.legend(loc="upper right")
    fig.suptitle("Gaia DR3 2-parameter (position-only) sources — neighbour-count "
                 "distribution", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "gaia_2par_cnt_distribution.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
