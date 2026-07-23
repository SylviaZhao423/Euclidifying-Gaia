#!/usr/bin/env python3
"""
Euclid×Gaia → CatWISE match completeness & ambiguity vs match radius.

For a random sample of the two Euclid×Gaia source tables that feed the Euclid+WISE
QSO selection —
    siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep   (Q1)
    siwei_zhao_2025.crossmatch_ero_gaia_0_3              (ERO)
— this measures, as a function of the CatWISE match radius:

    MATCH RATE:  fraction with >= 1 CatWISE source within r   (completeness)
    MULTI RATE:  fraction with >= 2 CatWISE sources within r  (ambiguous / blended match)

The production cross-match (crossmatch_euclid_wise_gaia.py → extract_qso_euclid_wise_gaia.py)
takes the closest CatWISE within WISE_RADIUS_ARCSEC = 1"; this script quantifies what
tightening / loosening that radius would do to completeness and to match ambiguity.

Sampling keeps the query short (per-source q3c cone search over the 1.89B-row CatWISE
catalogue), so it never approaches the WSDB connection timeout a full-table pass hits.
Rates are scaled to the full population (COUNT of each source table) for absolute counts.

Requires WSDB (VPN). Auth via ~/.pgpass.

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_euclid_wise_match_radius.py
    ... --radii 0.5,1,2,3   --n 20000   --seed 42   --plot
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from qso_gaia_crossmatch_builder import WSDB_HOST, WSDB_DB  # noqa: E402

CATWISE = "catwise_202003.main"
SRC = {
    "Q1":  ("siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep", "m_right_ascension", "m_declination"),
    "ERO": ("siwei_zhao_2025.crossmatch_ero_gaia_0_3", "e_v_alpha_j2000", "e_v_delta_j2000"),
}
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output" / "crossmatch_diagnostics"


def sample_neighbours(cur, tab: str, ra: str, dec: str, n: int, rmax: float):
    """Return an (n_sampled × 2) float array of the separations (arcsec) to the 1st and
    2nd nearest CatWISE source within `rmax` per sampled source (NaN where absent).

    Using the 2 nearest (LIMIT 2) instead of counting ALL neighbours is what keeps this
    fast: q3c can early-stop at the 2nd hit, so dense ERO fields don't blow up. From the
    two separations, ANY radius r <= rmax gives both rates:
        match(r) = 1st-nearest <= r        (>=1 CatWISE within r)
        multi(r) = 2nd-nearest <= r        (>=2 CatWISE within r  ==  ambiguous)
    """
    import numpy as np
    cur.execute(f"""
        SELECT min(m.sep)                          AS sep1,
               (array_agg(m.sep ORDER BY m.sep))[2] AS sep2
        FROM (SELECT row_number() OVER () AS rn, {ra} AS ra, {dec} AS dec
              FROM {tab} WHERE {ra} IS NOT NULL ORDER BY random() LIMIT {n}) s
        LEFT JOIN LATERAL (
            SELECT q3c_dist(s.ra, s.dec, cw.ra, cw.dec) * 3600.0 AS sep
            FROM {CATWISE} cw
            WHERE q3c_join(s.ra, s.dec, cw.ra, cw.dec, {rmax} / 3600.0)
            ORDER BY q3c_dist(s.ra, s.dec, cw.ra, cw.dec) ASC
            LIMIT 2
        ) m ON true
        GROUP BY s.rn
    """)
    return np.array([[r[0] if r[0] is not None else np.nan,
                      r[1] if r[1] is not None else np.nan] for r in cur.fetchall()],
                    dtype=float)


def population(cur, tab: str, ra: str) -> int:
    cur.execute(f"SELECT count(*) FROM {tab} WHERE {ra} IS NOT NULL")
    return cur.fetchone()[0]


def qso_cut_full(cur, radii: list[float], cut: float):
    """EXACT (not sampled) full-population pass: for every Euclid×Gaia source, take its 2
    nearest CatWISE within max(radii) (LIMIT 2, so it early-stops). Returns per matched
    source (rel, sep1, w1w2_closest, sep2):
        sep1          separation to the closest CatWISE  (the one the QSO cut uses)
        w1w2_closest  W1-W2 of that closest source
        sep2          separation to the 2nd-nearest      (NaN if none within rmax)
    so the caller can count, at each radius r: QSO passers (sep1<=r AND w1w2>=cut) AND how
    many of those are ambiguous (sep2<=r, i.e. >=2 CatWISE within r). This reproduces the
    production rule of extract_qso_euclid_wise_gaia.py (closest within r, W1-W2>=cut)."""
    import numpy as np
    rmax = max(radii)
    union = "\n                UNION ALL\n                ".join(
        f"SELECT {ra} AS ra, {dec} AS dec, '{tag}'::text AS rel "
        f"FROM {tab} WHERE {ra} IS NOT NULL"
        for tag, (tab, ra, dec) in SRC.items())
    cur.execute(f"""
        SELECT s.rel,
               (array_agg(m.sep  ORDER BY m.sep))[1] AS sep1,
               (array_agg(m.w1w2 ORDER BY m.sep))[1] AS w1w2_closest,
               (array_agg(m.sep  ORDER BY m.sep))[2] AS sep2
        FROM (SELECT row_number() OVER () AS rn, ra, dec, rel FROM (
                {union}
              ) u) s
        CROSS JOIN LATERAL (
            SELECT q3c_dist(s.ra, s.dec, cw.ra, cw.dec) * 3600.0 AS sep,
                   (cw.w1mpro - cw.w2mpro) AS w1w2
            FROM {CATWISE} cw
            WHERE q3c_join(s.ra, s.dec, cw.ra, cw.dec, {rmax} / 3600.0)
            ORDER BY q3c_dist(s.ra, s.dec, cw.ra, cw.dec) ASC
            LIMIT 2
        ) m
        GROUP BY s.rn, s.rel
    """)
    rows = cur.fetchall()
    rel = np.array([r[0] for r in rows])
    sep1 = np.array([r[1] if r[1] is not None else np.nan for r in rows], dtype=float)
    w1w2 = np.array([r[2] if r[2] is not None else np.nan for r in rows], dtype=float)
    sep2 = np.array([r[3] if r[3] is not None else np.nan for r in rows], dtype=float)
    return rel, sep1, w1w2, sep2


def plot(radii, rates, out: Path):
    """Two panels: completeness (>=1) and ambiguity (>=2) vs radius, per release + combined."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"Q1": "#4363d8", "ERO": "#e6194b", "combined": "#111111"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for tag in ["Q1", "ERO", "combined"]:
        a1.plot(radii, [rates[tag][r]["match"] * 100 for r in radii], "o-",
                color=col[tag], label=tag, lw=1.8)
        a2.plot(radii, [rates[tag][r]["multi"] * 100 for r in radii], "s--",
                color=col[tag], label=tag, lw=1.8)
    a1.axvline(1.0, color="0.6", ls=":", lw=1)
    a2.axvline(1.0, color="0.6", ls=":", lw=1)
    a1.set_title("Match rate  (≥1 CatWISE within r)  — completeness")
    a2.set_title("Multi rate  (≥2 CatWISE within r)  — ambiguity")
    for ax in (a1, a2):
        ax.set_xlabel('match radius (")'); ax.set_ylabel("% of Euclid×Gaia sources")
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Euclid×Gaia → CatWISE match vs radius  (dotted = production 1\")",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def run_qso_cut(cur, conn, radii: list[float], cut: float) -> int:
    """EXACT (full-population, no sampling) combined table at each radius:
    completeness (>=1 CatWISE), population ambiguity (>=2 CatWISE), the QSO-cut yield
    (closest within r AND W1-W2>=cut), and ambiguity among those QSOs."""
    import csv as _csv
    import numpy as np
    pop = {tag: population(cur, tab, ra) for tag, (tab, ra, dec) in SRC.items()}
    tot = sum(pop.values())
    print(f"EXACT full match: 2 nearest CatWISE within {max(radii)}\" for every Euclid×Gaia "
          f"source (of {tot:,}) ... (may take a few min; LIMIT 2 keeps it bounded)", flush=True)
    rel, sep1, w1w2, sep2 = qso_cut_full(cur, radii, cut)
    cur.close(); conn.close()
    print(f"  {len(rel):,} have a CatWISE match within {max(radii)}\"\n")

    def cnt(mask, tag=None):
        return int(np.sum(mask & (rel == tag))) if tag else int(np.sum(mask))

    print("=" * 118)
    print(f"Euclid×Gaia → CatWISE vs match radius  (EXACT full population; W1-W2 QSO cut = {cut})")
    print(f'{"radius":>6} | {"Q1 match":>15} | {"ERO match":>15} | {"combined match":>16} | '
          f'{"multi>=2 (pop)":>15} | {"#QSO":>6} | {"QSO m>=2":>8}')
    print("-" * 118)
    rows_csv = []
    for r in radii:
        within = sep1 <= r
        multi = sep2 <= r                                   # >=2 CatWISE within r
        passing = within & (w1w2 >= cut)                    # production QSO selection at r
        m = {t: cnt(within, t) for t in SRC}
        mc = cnt(within)
        pm = cnt(multi)                                     # population ambiguity
        nq = cnt(passing)
        qm = cnt(passing & multi)                           # ambiguity among QSOs
        def mcell(t): return f'{100*m[t]/pop[t]:4.1f}% ({m[t]:>8,})'
        print(f'{r:5.2f}" | {mcell("Q1")} | {mcell("ERO")} | '
              f'{100*mc/tot:4.1f}% ({mc:>8,}) | {100*pm/tot:4.2f}% ({pm:>7,}) | '
              f'{nq:>6,} | {qm:>8,}')
        rows_csv.append([r, m["Q1"], m["ERO"], mc, f"{mc/tot:.4f}",
                         pm, f"{pm/tot:.4f}", nq, qm])
    print("=" * 118)
    print("match = >=1 CatWISE within r (completeness) · multi>=2 = >=2 within r (ambiguity)")
    print(f"#QSO = closest within r AND W1-W2>={cut} · QSO m>=2 = those QSOs with >=2 within r")
    print("(production radius 1.00\": #QSO should be 6,265 — matches crossmatch_qso_euclid_wise_gaia)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "euclid_wise_match_and_qso_vs_radius.csv"
    with open(out, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["radius_arcsec", "match_Q1", "match_ERO", "match_total", "match_frac",
                    "multi_ge2_total", "multi_ge2_frac", "qso_total", "qso_multi_ge2"])
        w.writerows(rows_csv)
    print(f"  -> {out}")
    return 0


def main() -> int:
    import numpy as np
    import psycopg2
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radii", default="0.5,1,2,3", help="comma-separated radii in arcsec")
    ap.add_argument("--n", type=int, default=20000, help="sample size per release (default 20000)")
    ap.add_argument("--seed", type=int, default=42, help="reproducible sample seed")
    ap.add_argument("--plot", action="store_true", help="also save a PNG of the two curves")
    ap.add_argument("--qso-cut", action="store_true",
                    help="EXACT full-population count of how many pass the QSO colour cut "
                         "(closest CatWISE within r AND W1-W2 >= --w1w2) at each radius, "
                         "instead of the sampled match/multi curves")
    ap.add_argument("--w1w2", type=float, default=0.8, help="W1-W2 QSO cut (default 0.8)")
    args = ap.parse_args()
    radii = [float(x) for x in args.radii.split(",")]

    user = os.getenv("PGUSER")
    if not user:
        print("ERROR: PGUSER not set (e.g. PGUSER=siwei_zhao_2025 python ...)."); return 1
    try:
        conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                                connect_timeout=30, keepalives=1, keepalives_idle=30,
                                keepalives_interval=10, keepalives_count=5)
    except Exception as exc:
        print(f"CONNECT FAILED (VPN up? eduroam blocks WSDB:5432): {exc}"); return 1
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 0")

    if args.qso_cut:
        return run_qso_cut(cur, conn, radii, args.w1w2)

    cur.execute("SELECT setseed(%s)", ((args.seed % 1000) / 1000.0,))
    rmax = max(radii)
    pop, neigh = {}, {}
    for tag, (tab, ra, dec) in SRC.items():
        pop[tag] = population(cur, tab, ra)
        print(f"sampling {args.n:,} {tag} sources (of {pop[tag]:,}); "
              f"2 nearest CatWISE within {rmax}\" ...", flush=True)
        neigh[tag] = sample_neighbours(cur, tab, ra, dec, args.n, rmax)
    cur.close(); conn.close()

    tot = sum(pop.values())
    rates: dict = {t: {} for t in ["Q1", "ERO", "combined"]}
    for r in radii:
        for tag in SRC:
            sep1, sep2 = neigh[tag][:, 0], neigh[tag][:, 1]
            rates[tag][r] = {"match": float(np.mean(sep1 <= r)),   # >=1 within r (NaN<=r False)
                             "multi": float(np.mean(sep2 <= r))}   # >=2 within r
        rates["combined"][r] = {
            k: sum(rates[t][r][k] * pop[t] for t in SRC) / tot for k in ("match", "multi")}

    # ── report ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("MATCH RATE  (>=1 CatWISE within r) — completeness")
    print(f'{"radius":>7} | {"Q1":>18} | {"ERO":>18} | {"combined":>18}')
    print("-" * 90)
    for r in radii:
        def c(t): return f'{rates[t][r]["match"]*100:5.1f}%  ({int(round(rates[t][r]["match"]*pop.get(t,tot))):>8,})'
        print(f'{r:5.2f}" | {c("Q1")} | {c("ERO")} | '
              f'{rates["combined"][r]["match"]*100:5.1f}%  ({int(round(rates["combined"][r]["match"]*tot)):>8,})')
    print("\n" + "=" * 90)
    print("MULTI RATE  (>=2 CatWISE within r) — ambiguous / blended match")
    print(f'{"radius":>7} | {"Q1":>18} | {"ERO":>18} | {"combined":>18}')
    print("-" * 90)
    for r in radii:
        def c(t): return f'{rates[t][r]["multi"]*100:5.1f}%  ({int(round(rates[t][r]["multi"]*pop.get(t,tot))):>8,})'
        print(f'{r:5.2f}" | {c("Q1")} | {c("ERO")} | '
              f'{rates["combined"][r]["multi"]*100:5.1f}%  ({int(round(rates["combined"][r]["multi"]*tot)):>8,})')
    print("=" * 90)
    print("(cells: % of Euclid×Gaia sources  (count scaled to full population))")

    # ── save CSV ────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv = OUT_DIR / "euclid_wise_match_radius.csv"
    import csv as _csv
    with open(csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["radius_arcsec", "release", "population", "match_rate", "match_count",
                    "multi_rate", "multi_count"])
        for r in radii:
            for tag in ["Q1", "ERO"]:
                mr, mu = rates[tag][r]["match"], rates[tag][r]["multi"]
                w.writerow([r, tag, pop[tag], f"{mr:.4f}", int(round(mr*pop[tag])),
                            f"{mu:.4f}", int(round(mu*pop[tag]))])
            mr, mu = rates["combined"][r]["match"], rates["combined"][r]["multi"]
            w.writerow([r, "combined", tot, f"{mr:.4f}", int(round(mr*tot)),
                        f"{mu:.4f}", int(round(mu*tot))])
    print(f"  -> {csv}")
    if args.plot:
        plot(radii, rates, OUT_DIR / "euclid_wise_match_radius.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
