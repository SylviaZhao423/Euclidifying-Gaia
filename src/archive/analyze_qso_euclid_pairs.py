#!/usr/bin/env python3
"""
STEP 3b of the QSO-companion colour pipeline: the JOINT common-z QSO-pair test.

The single-object Euclid-colour test is weak (Step 3a: ~49% star contamination) because
QSO and stellar loci overlap. The pair geometry fixes this: a genuine QSO pair has BOTH
members on the QSO colour(z) locus at ONE common redshift, whereas a chance star
projection almost never does. So for each Euclid "pair" (a WISE+Euclid QSO source with
>=2 MER detections within 1" of the Gaia position) we test

    joint reduced chi^2 = min_z [ D2_A(z) + D2_B(z) ] / (2 * n_colours)

on the two BRIGHTEST members (A, B), where D2_X(z) is the Step-3a distance of member X
from the locus at redshift z. Small joint chi^2 => both members look like QSOs at the
same z => strong QSO-pair candidate; z_common is that redshift.

Calibration
-----------
We compare the candidate pairs against two simulated populations built from the same
colour data:
  * star+star pairs  (control stars, no shared z)         -> the null / contaminants
  * QSO+QSO pairs at matched z (|dz|<0.05, spec sample)    -> the signal / true pairs
The star-pair distribution sets the threshold; candidates below it are the pairs whose
two members are jointly QSO-consistent. Where both members have Gaia, we also report the
proper-motion significance (a real QSO pair should show no significant motion).

Inputs:
  output/qso_locus/qso_color_locus.csv               (locus mu/sig; Step 3a)
  output/qso_locus/arrays/star_control_colors.csv    (control stars; Step 3a)
  (none — spec-QSO colours+z built live via analyze_qso_color_vs_specz.build_dataframe())   (spec QSOs w/ z; Step 1)
  {PGUSER}.crossmatch_qso_euclid_wise_gaia  + euclid_q1.mer_final_cat / euclid_ero.merge_cat
Outputs:
  output/qso_locus/qso_euclid_pairs.csv              (per-pair scores, ranked)
  output/qso_locus/qso_euclid_pairs.png              (calibration + candidate diagnostics)

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_qso_euclid_pairs.py            # pull + score
  python src/analyze_qso_euclid_pairs.py --replot                         # from cached pair CSV
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
from analyze_qso_euclid_locus import (COLORS, NCOL, ZGRID, SIG_FLOOR,  # noqa: E402
                                      _flux_to_mag, ZPC_VIS)
from analyze_qso_color_vs_specz import build_dataframe  # noqa: E402  (spec-QSO colours+z, live)

PROJECT_DIR = Path(__file__).resolve().parents[1]
LOCUS_CSV   = PROJECT_DIR / "output" / "qso_locus" / "qso_color_locus.csv"
STAR_CSV    = PROJECT_DIR / "output" / "qso_locus" / "arrays" / "star_control_colors.csv"
OUT_CSV     = PROJECT_DIR / "output" / "qso_locus" / "qso_euclid_pairs.csv"
PNG_PATH    = PROJECT_DIR / "output" / "qso_locus" / "qso_euclid_pairs.png"

WSDB_HOST, WSDB_DB, WSDB_USER = "wsdb.ast.cam.ac.uk", "wsdb", "siwei_zhao_2025"
MATCH_R_DEG = 1.0 / 3600.0        # 1" member search around the Gaia position
CCOLS = [c for c, _ in COLORS]

# Per-catalog MER member query (columns aliased to a common flux/err set fv/ev,...).
MEMBER_SQL = {
    "Q1": """
        SELECT object_id AS mid, right_ascension AS ra, declination AS dec,
               flux_vis_2fwhm_aper AS fv, fluxerr_vis_2fwhm_aper AS ev,
               flux_y_2fwhm_aper   AS fy, fluxerr_y_2fwhm_aper   AS ey,
               flux_j_2fwhm_aper   AS fj, fluxerr_j_2fwhm_aper   AS ej,
               flux_h_2fwhm_aper   AS fh, fluxerr_h_2fwhm_aper   AS eh
        FROM euclid_q1.mer_final_cat
        WHERE q3c_join(%(ra)s, %(dec)s, right_ascension, declination, %(r)s)""",
    "ERO": """
        SELECT v_objid AS mid, v_alpha_j2000 AS ra, v_delta_j2000 AS dec,
               v_flux_auto AS fv, v_fluxerr_auto AS ev,
               y_flux_auto AS fy, y_fluxerr_auto AS ey,
               j_flux_auto AS fj, j_fluxerr_auto AS ej,
               h_flux_auto AS fh, h_fluxerr_auto AS eh
        FROM euclid_ero.merge_cat
        WHERE q3c_join(%(ra)s, %(dec)s, v_alpha_j2000, v_delta_j2000, %(r)s)""",
}


# --------------------------------------------------------------------------- locus
def load_locus():
    loc = pd.read_csv(LOCUS_CSV)
    mu  = np.stack([loc[f"mu_{c}"].to_numpy()  for c in CCOLS], axis=1)   # (Z, NCOL)
    sig = np.stack([loc[f"sig_{c}"].to_numpy() for c in CCOLS], axis=1)
    inv = 1.0 / (sig ** 2 + SIG_FLOOR ** 2)
    return mu, inv


def d2_curves(colors: np.ndarray, mu, inv) -> np.ndarray:
    """D2_X(z) for each row of colors (N x NCOL) -> (N, Z)."""
    diff = colors[:, None, :] - mu[None, :, :]
    return np.einsum("nzc,zc->nz", diff ** 2, inv)


def _colors_from_fluxes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["IE"] = _flux_to_mag(df["fv"], df["ev"], ZPC_VIS)
    df["YE"] = _flux_to_mag(df["fy"], df["ey"])
    df["JE"] = _flux_to_mag(df["fj"], df["ej"])
    df["HE"] = _flux_to_mag(df["fh"], df["eh"])
    df["IE_YE"] = df["IE"] - df["YE"]
    df["YE_JE"] = df["YE"] - df["JE"]
    df["JE_HE"] = df["JE"] - df["HE"]
    return df


# ----------------------------------------------------------------------------- data
def fetch_pairs(user: str, mu, inv) -> pd.DataFrame:
    """Pull the >=2-member pairs, query each pair's members, and score the brightest two."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                            connect_timeout=30, keepalives=1, keepalives_idle=30,
                            keepalives_interval=10, keepalives_count=5)
    pairs = pd.read_sql_query(f"""
        SELECT g_source_id AS sid, qso_ra, qso_dec, g_ra, g_dec, euclid,
               n_euclid_within_1arcsec AS n_euclid,
               g_parallax_over_error AS plx_sig, g_pm_significance AS pm_sig, w1_w2
        FROM {user}.crossmatch_qso_euclid_wise_gaia
        WHERE n_euclid_within_1arcsec >= 2
          AND euclid = 'Q1'   -- Q1-only: ERO member fluxes are raw ADU, not µJy (broken vs Q1 locus)
    """, conn)
    print(f"  {len(pairs):,} pairs (n_euclid>=2)")
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 120000")
    rows = []
    for _, p in pairs.iterrows():
        cat = str(p["euclid"])
        cur.execute(MEMBER_SQL.get(cat, MEMBER_SQL["Q1"]),
                    {"ra": float(p["g_ra"]), "dec": float(p["g_dec"]), "r": MATCH_R_DEG})
        cols = [d[0] for d in cur.description]
        mem = _colors_from_fluxes(pd.DataFrame(cur.fetchall(), columns=cols))
        mem = mem.dropna(subset=CCOLS)
        if len(mem) < 2:
            continue
        mem = mem.sort_values("fv", ascending=False).head(2).reset_index(drop=True)   # brightest 2
        cc = mem[CCOLS].to_numpy(float)
        dc = d2_curves(cc, mu, inv)                       # (2, Z)
        indiv = dc.min(axis=1) / NCOL                     # per-member reduced chi2
        joint_curve = dc.sum(axis=0)
        zi = int(np.argmin(joint_curve))
        rows.append({
            "sid": p["sid"], "euclid": cat, "n_euclid": int(p["n_euclid"]),
            "ra": float(p["g_ra"]), "dec": float(p["g_dec"]),
            "a_IE_YE": cc[0, 0], "a_YE_JE": cc[0, 1], "a_JE_HE": cc[0, 2],
            "b_IE_YE": cc[1, 0], "b_YE_JE": cc[1, 1], "b_JE_HE": cc[1, 2],
            "chi2r_A": indiv[0], "chi2r_B": indiv[1],
            "chi2r_joint": joint_curve[zi] / (2 * NCOL), "z_common": ZGRID[zi],
            "plx_sig": p["plx_sig"], "pm_sig": p["pm_sig"], "w1_w2": p["w1_w2"],
        })
    cur.close(); conn.close()
    df = pd.DataFrame(rows)
    print(f"  scored {len(df):,} pairs with >=2 valid-colour members")
    return df


# ---------------------------------------------------------------------- calibration
def _sim_joint(colors_a, colors_b, mu, inv):
    """Joint reduced chi2 for aligned pairs (colors_a[i] with colors_b[i])."""
    da = d2_curves(colors_a, mu, inv)
    db = d2_curves(colors_b, mu, inv)
    return (da + db).min(axis=1) / (2 * NCOL)


def simulate(mu, inv, m=5000, seed=42):
    """star+star (null) and QSO+QSO-at-matched-z (signal) joint reduced-chi2 samples."""
    rng = np.random.default_rng(seed)
    out = {}
    if STAR_CSV.exists():
        s = pd.read_csv(STAR_CSV).dropna(subset=CCOLS)[CCOLS].to_numpy(float)
        i = rng.integers(0, len(s), m); j = rng.integers(0, len(s), m)
        out["star+star"] = _sim_joint(s[i], s[j], mu, inv)
    q = build_dataframe(os.getenv("PGUSER", WSDB_USER)).dropna(subset=CCOLS + ["qso_z"])
    q = q[q["qso_z"] > 0]
    if len(q):
        z = q["qso_z"].to_numpy(float); cq = q[CCOLS].to_numpy(float)
        order = np.argsort(z); z, cq = z[order], cq[order]
        # matched-z partner: neighbour in the z-sorted list with |dz|<0.05
        ia, ib = [], []
        for _ in range(m * 3):
            k = int(rng.integers(1, len(z) - 1))
            nb = k + (1 if rng.random() < 0.5 else -1)
            if abs(z[k] - z[nb]) < 0.05:
                ia.append(k); ib.append(nb)
            if len(ia) >= m:
                break
        out["QSO+QSO (same z)"] = _sim_joint(cq[ia], cq[ib], mu, inv)
    return out


# ----------------------------------------------------------------------------- plot
def make_plot(pairs, sims, thr, out):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    bins = np.linspace(0, 20, 60)
    # (1) calibration: candidate pairs vs simulated populations
    ax = axes[0]
    ax.hist(sims["star+star"], bins=bins, density=True, color="#555555", alpha=0.55,
            label=f"star+star (null, N={len(sims['star+star']):,})")
    ax.hist(sims["QSO+QSO (same z)"], bins=bins, density=True, color="#e6194b", alpha=0.55,
            label=f"QSO+QSO same z (N={len(sims['QSO+QSO (same z)']):,})")
    ax.hist(pairs["chi2r_joint"], bins=bins, density=True, color="#4363d8", alpha=0.7,
            histtype="step", lw=2, label=f"candidate pairs (N={len(pairs):,})")
    ax.axvline(thr, color="k", ls="--", lw=1, label=f"star-pair 5% cut ({thr:.1f})")
    ax.set_xlabel(r"joint reduced $\chi^2_{\min}$ (common z)"); ax.set_ylabel("density")
    ax.set_title("common-z pair calibration", fontweight="bold"); ax.legend(fontsize=8)
    # (2) does the common-z requirement help? single-worst-member vs joint
    ax = axes[1]
    worst = pairs[["chi2r_A", "chi2r_B"]].max(axis=1)
    passed = pairs["chi2r_joint"] <= thr
    ax.scatter(worst[~passed], pairs["chi2r_joint"][~passed], s=14, c="#999999",
               alpha=0.6, label="rejected")
    ax.scatter(worst[passed], pairs["chi2r_joint"][passed], s=20, c="#4363d8",
               alpha=0.85, label=f"QSO-pair candidates ({int(passed.sum())})")
    ax.axhline(thr, color="k", ls="--", lw=1)
    ax.plot([0, 20], [0, 20], "k:", lw=0.8)
    ax.set_xlim(0, 20); ax.set_ylim(0, 20)
    ax.set_xlabel(r"worst single-member reduced $\chi^2$")
    ax.set_ylabel(r"joint reduced $\chi^2$")
    ax.set_title("joint vs single-object", fontweight="bold"); ax.legend(fontsize=9)
    # (3) candidates: z_common vs Gaia PM significance
    ax = axes[2]
    cand = pairs[passed]
    sc = ax.scatter(cand["z_common"], cand["pm_sig"], c=cand["chi2r_joint"],
                    cmap="viridis_r", s=30, edgecolor="k", linewidth=0.3)
    ax.axhline(3, color="r", ls="--", lw=0.9, label="PM 3σ")
    ax.set_xlabel("common redshift $z_{common}$"); ax.set_ylabel("Gaia PM significance (σ)")
    ax.set_title("QSO-pair candidates", fontweight="bold"); ax.legend(fontsize=9)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label=r"joint $\chi^2$")
    fig.suptitle("Euclid joint common-z QSO-pair test (brightest two members per pair)",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replot", action="store_true",
                    help="reuse the cached pair CSV instead of re-querying WSDB.")
    args = ap.parse_args()
    if not LOCUS_CSV.exists():
        print(f"  {LOCUS_CSV} not found — run analyze_qso_euclid_locus.py first."); return 1
    mu, inv = load_locus()

    if args.replot and OUT_CSV.exists():
        pairs = pd.read_csv(OUT_CSV)
        print(f"  loaded {len(pairs):,} scored pairs from cache")
    else:
        pairs = fetch_pairs(os.getenv("PGUSER", WSDB_USER), mu, inv)
        if not len(pairs):
            print("  no scorable pairs."); return 1

    sims = simulate(mu, inv)
    thr = float(np.percentile(sims["star+star"], 5))     # keep 5% of star-pairs
    pairs["qso_pair_candidate"] = pairs["chi2r_joint"] <= thr
    pairs = pairs.sort_values("chi2r_joint").reset_index(drop=True)
    pairs.to_csv(OUT_CSV, index=False)

    n_cand = int(pairs["qso_pair_candidate"].sum())
    print(f"  star-pair 5% threshold: joint χ²ᵣ <= {thr:.2f}")
    print(f"  QSO-pair candidates: {n_cand}/{len(pairs)} "
          f"({100*n_cand/len(pairs):.0f}%); star+star pass rate at this cut = 5% by construction")
    frac_qso = float(np.mean(sims["QSO+QSO (same z)"] <= thr))
    print(f"  (true QSO+QSO pairs recovered at same cut: {100*frac_qso:.0f}%)")
    make_plot(pairs, sims, thr, PNG_PATH)
    print(f"  -> {OUT_CSV}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
