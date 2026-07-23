#!/usr/bin/env python3
"""Central-Gaia detection ("nucleation") fraction of NSA v1 galaxies as a function of
stellar mass, morphology and environment.

The projected crossmatch_nsa_v1_gaia LEFT JOIN keeps EVERY NSA galaxy (denominator) with
a has_gaia flag (numerator = a Gaia DR3 source within 1" of the photometric centre), so
    f_detect(X) = <has_gaia | X>
is a well-defined fraction in any bin of X. We compute it, with Wilson 95% binomial
intervals, versus:
    i)   stellar mass      gal_log_mass
    ii)  morphology        gal_sersic_n (late n<2.5 / early n>2.5), gal_conc (R90/R50), gal_ba
    iii) environment       Sigma_5 — projected surface density to the 5th nearest neighbour
                           within +/-1000 km/s (Baldry-style), computed here from
                           gal_ra/gal_dec/gal_zdist (not a native NSA column; cached).

IMPORTANT — Gaia is magnitude-limited (G<~21), so a central detection is more likely for
NEARER (brighter-nucleus) galaxies, and mass/environment both correlate with distance
(Malmquist; clusters are farther). A raw fraction-vs-X curve therefore mixes real
nucleation with detectability. Every panel is drawn twice:
    RAW           full sample (grey)
    VOL-LIMITED   a fixed low-z shell zmin<=zdist<=zmax (colour) — distance-controlled
so the astrophysical trend can be read off the volume-limited curve. A mass x z
detection-fraction heat-map panel makes the confound explicit.

Outputs (output/umap/nsa/):
    nsa_nucleation_fractions.png            6-panel figure (raw + vol-limited overlaid)
    arrays/nsa_env_sigma5.csv               cached Sigma_5 per nsaid (expensive; reused)
    arrays/nsa_nucleation_binned.csv        every binned fraction + Wilson CI (all panels)

Usage:
    conda run -n base python3 src/analyze_nsa_nucleation.py
    ... --zmin 0.005 --zmax 0.03      # volume-limited shell (default)
    ... --env-n 5 --env-dv 1000       # Sigma_N definition
    ... --nbins 12                    # quantile bins per variable
    ... --recompute-env               # force-recompute Sigma_5 (else load cache)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
NSA_DIR = PROJECT_DIR / "output" / "umap" / "nsa"
CSV = NSA_DIR / "arrays" / "nsa_gaia_matches.csv"
ENV_CACHE = NSA_DIR / "arrays" / "nsa_env_sigma5.csv"
BINNED_CSV = NSA_DIR / "arrays" / "nsa_nucleation_binned.csv"
OUT_PNG = NSA_DIR / "nsa_nucleation_fractions.png"

C_KMS = 299_792.458   # speed of light, km/s


# --------------------------------------------------------------------------- #
#  data                                                                        #
# --------------------------------------------------------------------------- #
def load_nsa() -> pd.DataFrame:
    if not CSV.exists():
        raise SystemExit(f"{CSV} not found — run analyze_umap_nsa.py (v1) first.")
    df = pd.read_csv(CSV, dtype={"gaia_source_id": "str"})
    if "has_gaia" not in df.columns:
        df["has_gaia"] = df["gaia_source_id"].notna()
    df["has_gaia"] = df["has_gaia"].astype(bool)
    return df


def add_nucleus_flag(df: pd.DataFrame, sep_max: float, star_sig: float) -> pd.DataFrame:
    """is_nucleus = a CONFIRMED central source (cleaner numerator than raw has_gaia):
      has_gaia  AND  well-centred (gaia_sep_arcsec <= sep_max)
                AND  NOT a foreground Milky-Way star — i.e. it does NOT show a *measured*
                     significant parallax or proper motion (|parallax_over_error| < star_sig
                     and pm_significance < star_sig). 2-param Gaia sources have no
                     parallax/PM (NaN) → they pass (as expected for extragalactic nuclei).
    This removes chance-projected Galactic stars and off-centre matches, which otherwise
    inflate the fraction preferentially for large (massive, nearby) galaxies."""
    hg = df["has_gaia"].to_numpy(bool)
    sep = pd.to_numeric(df.get("gaia_sep_arcsec"), errors="coerce").to_numpy()
    plx = pd.to_numeric(df.get("parallax_over_error"), errors="coerce").to_numpy()
    pms = pd.to_numeric(df.get("pm_significance"), errors="coerce").to_numpy()
    centred = hg & (sep <= sep_max)
    # star only if it HAS a measured significant parallax OR PM (NaN → not a star here)
    is_star = (np.abs(plx) >= star_sig) | (pms >= star_sig)
    is_star = np.where(np.isnan(plx) & np.isnan(pms), False, is_star)
    df = df.copy()
    df["is_nucleus"] = centred & ~is_star
    n_hg, n_nuc = int(hg.sum()), int(df["is_nucleus"].sum())
    n_off = int((hg & (sep > sep_max)).sum())
    n_star = int((centred & is_star).sum())
    print(f"  confirmed nuclei: {n_nuc:,}/{n_hg:,} candidates kept "
          f"(dropped {n_off:,} off-centre >{sep_max}\", {n_star:,} foreground stars "
          f"|plx/err| or pm_sig >= {star_sig})")
    return df


# --------------------------------------------------------------------------- #
#  environment: Sigma_N (projected density to the N-th neighbour within dv)     #
# --------------------------------------------------------------------------- #
def compute_sigma_n(df: pd.DataFrame, n: int, dv_kms: float,
                    k_ang: int = 100, chunk: int = 40_000) -> pd.DataFrame:
    """Sigma_N = N / (pi * r_N^2) [gal / Mpc^2], r_N = projected physical distance to the
    N-th nearest neighbour within |c dz| < dv. Angular neighbours from a 3D-unit-vector
    cKDTree (k_ang candidates/gal, chunked for memory); the velocity cut is applied on cz.
    Galaxies with <N qualifying neighbours in the k_ang candidates are flagged isolated
    (their own lowest-density category). Returns a frame keyed by nsaid."""
    from scipy.spatial import cKDTree
    try:
        from astropy.cosmology import FlatLambdaCDM
        _cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        def ang_diam_dist(z):        # Mpc
            return _cosmo.angular_diameter_distance(np.clip(z, 1e-6, None)).value
    except Exception:                # low-z fallback: D_A ~ (c z / H0)/(1+z)
        def ang_diam_dist(z):
            z = np.clip(z, 1e-6, None); return (C_KMS * z / 70.0) / (1.0 + z)

    ra = np.radians(df["gal_ra"].to_numpy(float))
    dec = np.radians(df["gal_dec"].to_numpy(float))
    cz = C_KMS * df["gal_zdist"].to_numpy(float)
    xyz = np.column_stack([np.cos(dec) * np.cos(ra),
                           np.cos(dec) * np.sin(ra),
                           np.sin(dec)])
    tree = cKDTree(xyz)
    dA = ang_diam_dist(df["gal_zdist"].to_numpy(float))
    N = len(df)
    sigma = np.full(N, np.nan)
    isolated = np.zeros(N, bool)
    k = min(k_ang + 1, N)            # +1: query returns self first
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        chord, idx = tree.query(xyz[s:e], k=k)         # (m, k) sorted by angular sep
        for j in range(e - s):
            gi = s + j
            nb = idx[j]
            ch = chord[j]
            keep = (nb != gi) & (np.abs(cz[nb] - cz[gi]) < dv_kms)
            if keep.sum() < n:
                isolated[gi] = True
                continue
            theta = 2.0 * np.arcsin(np.clip(ch[keep] / 2.0, 0, 1))   # rad (sorted)
            r_p = np.sort(theta * dA[gi])                            # projected Mpc
            r_N = r_p[n - 1]
            if r_N > 0:
                sigma[gi] = n / (np.pi * r_N ** 2)
    out = pd.DataFrame({"nsaid": df["nsaid"].to_numpy(),
                        f"sigma{n}": sigma, "env_isolated": isolated})
    n_ok = np.isfinite(sigma).sum()
    print(f"  Sigma_{n}: {n_ok:,}/{N:,} galaxies with >={n} neighbours within "
          f"{dv_kms:.0f} km/s ({isolated.sum():,} isolated)")
    return out


def get_sigma_n(df: pd.DataFrame, n: int, dv_kms: float, recompute: bool) -> pd.DataFrame:
    col = f"sigma{n}"
    if ENV_CACHE.exists() and not recompute:
        cached = pd.read_csv(ENV_CACHE)
        if col in cached.columns:
            print(f"  loaded cached {ENV_CACHE.name}")
            return cached
    env = compute_sigma_n(df, n, dv_kms)
    ENV_CACHE.parent.mkdir(parents=True, exist_ok=True)
    env.to_csv(ENV_CACHE, index=False)
    print(f"  -> cached {ENV_CACHE}")
    return env


# --------------------------------------------------------------------------- #
#  binomial fraction with Wilson 95% interval                                  #
# --------------------------------------------------------------------------- #
def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96):
    k = np.asarray(k, float); n = np.asarray(n, float)
    p = np.where(n > 0, k / n, np.nan)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def binned_fraction(x: np.ndarray, hit: np.ndarray, nbins: int,
                    edges: np.ndarray | None = None):
    """Quantile-binned detection fraction (equal counts) with Wilson CIs. Returns a frame:
    bin centre (median x), p, lo, hi, n, k."""
    m = np.isfinite(x)
    x, hit = x[m], hit[m]
    if len(x) < nbins * 5:
        nbins = max(2, len(x) // 20)
    if edges is None:
        edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
        edges = np.unique(edges)
    bi = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        sel = bi == b
        n = int(sel.sum())
        if n == 0:
            continue
        k = int(hit[sel].sum())
        xc = float(np.median(x[sel]))
        rows.append((xc, k, n))
    xc = np.array([r[0] for r in rows]); k = np.array([r[1] for r in rows]); n = np.array([r[2] for r in rows])
    p, lo, hi = wilson(k, n)
    return pd.DataFrame({"x": xc, "p": p, "lo": lo, "hi": hi, "n": n, "k": k})


# --------------------------------------------------------------------------- #
#  plotting                                                                    #
# --------------------------------------------------------------------------- #
def _panel(ax, col, label, nbins, series, logx=False, divider=None, div_label=""):
    """One fraction-vs-X panel. `series` = list of (sub_df, numerator_col, colour, name).
    Each is drawn as a Wilson-CI fraction curve; every bin is also logged to the CSV."""
    for sub, numcol, colour, name in series:
        x = sub[col].to_numpy(float)
        if logx:
            x = np.where(x > 0, np.log10(x), np.nan)
        b = binned_fraction(x, sub[numcol].to_numpy(bool), nbins)
        ax.plot(b["x"], b["p"], "-o", color=colour, ms=4, lw=1.4,
                label=f"{name} (N={len(sub):,})", zorder=4)
        ax.fill_between(b["x"], b["lo"], b["hi"], color=colour, alpha=0.18, zorder=2)
        b.insert(0, "variable", (("log " if logx else "") + col))
        b.insert(1, "series", name)
        _panel.tables.append(b)
    if divider is not None:
        dx = np.log10(divider) if logx else divider
        ax.axvline(dx, ls="--", color="crimson", lw=1, alpha=0.7)
        ax.text(dx, 0.02, f" {div_label}", color="crimson", fontsize=8,
                ha="left", va="bottom", transform=ax.get_xaxis_transform())
    ax.set_xlabel(("log₁₀ " if logx else "") + label, fontsize=10)
    ax.set_ylabel("nucleation fraction", fontsize=10)
    ax.set_ylim(0, 1); ax.grid(alpha=0.25); ax.legend(fontsize=7.5, loc="best")
_panel.tables = []


def make_figure(df, vl, envcol, nbins):
    _panel.tables = []
    RAW, VL, EARLY, LATE, CAND = "0.55", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"
    # numerator is the CONFIRMED nucleus (is_nucleus); candidate (has_gaia) shown for contrast.
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # (a) mass split into early / late morphology — the disentangling panel
    vl_e = vl[vl["gal_sersic_n"] >= 2.5]; vl_l = vl[vl["gal_sersic_n"] < 2.5]
    _panel(axes[0, 0], "gal_log_mass", "stellar mass  M⋆ [M☉]  (vol-limited)", nbins,
           [(vl, "is_nucleus", VL, "all types"),
            (vl_e, "is_nucleus", EARLY, "early  n≥2.5"),
            (vl_l, "is_nucleus", LATE, "late  n<2.5")])
    axes[0, 0].set_title("f(mass) split by morphology  — mass vs type", fontsize=10)

    _panel(axes[0, 1], "gal_sersic_n", "Sérsic index  n", nbins,
           [(df, "is_nucleus", RAW, "raw (all z)"), (vl, "is_nucleus", VL, "vol-limited")],
           divider=2.5, div_label="late | early")
    _panel(axes[0, 2], "gal_conc", "concentration  R90/R50", nbins,
           [(df, "is_nucleus", RAW, "raw (all z)"), (vl, "is_nucleus", VL, "vol-limited")],
           divider=2.6, div_label="late | early")
    _panel(axes[1, 0], "gal_ba", "axis ratio  b/a", nbins,
           [(df, "is_nucleus", RAW, "raw (all z)"), (vl, "is_nucleus", VL, "vol-limited")])
    _panel(axes[1, 1], envcol, f"environment  {envcol.replace('sigma','Σ_')}  [gal/Mpc²]", nbins,
           [(df, "is_nucleus", RAW, "raw (all z)"), (vl, "is_nucleus", VL, "vol-limited")],
           logx=True)
    # Row2 col3: candidate (raw has_gaia) vs confirmed (is_nucleus) fraction vs mass —
    # how much the centred + non-foreground-star cut removes (the interloper cleaning).
    _panel(axes[1, 2], "gal_log_mass", "stellar mass  M⋆ [M☉]  (vol-limited)", nbins,
           [(vl, "has_gaia", CAND, "candidate  (any Gaia <1\")"),
            (vl, "is_nucleus", VL, "confirmed  (centred, non-star)")])
    axes[1, 2].set_title("candidate vs confirmed nucleus  — interloper cleaning", fontsize=10)

    fig.suptitle("NSA v1 nucleation fraction — confirmed nuclei (centred + non-foreground-star), "
                 "raw vs volume-limited", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT_PNG}")
    if _panel.tables:
        pd.concat(_panel.tables, ignore_index=True).to_csv(BINNED_CSV, index=False)
        print(f"  -> {BINNED_CSV}")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zmin", type=float, default=0.005, help="volume-limited lower z (default 0.005)")
    ap.add_argument("--zmax", type=float, default=0.030, help="volume-limited upper z (default 0.03)")
    ap.add_argument("--env-n", type=int, default=5, help="N for Sigma_N (default 5)")
    ap.add_argument("--env-dv", type=float, default=1000.0, help="velocity window km/s (default 1000)")
    ap.add_argument("--nbins", type=int, default=12, help="quantile bins per variable (default 12)")
    ap.add_argument("--nucleus-sep", type=float, default=0.5,
                    help="max gaia_sep_arcsec for a confirmed (centred) nucleus (default 0.5)")
    ap.add_argument("--star-sig", type=float, default=5.0,
                    help="reject as foreground star if |parallax_over_error| or pm_significance "
                         ">= this (default 5)")
    ap.add_argument("--recompute-env", action="store_true", help="force recompute Sigma_N")
    args = ap.parse_args()

    df = load_nsa()
    print(f"NSA v1 galaxies: {len(df):,}  |  central Gaia: {int(df['has_gaia'].sum()):,}  "
          f"(candidate fraction {df['has_gaia'].mean():.3f})")
    df = add_nucleus_flag(df, args.nucleus_sep, args.star_sig)
    print(f"  confirmed nucleation fraction (all z): {df['is_nucleus'].mean():.3f}")

    env = get_sigma_n(df, args.env_n, args.env_dv, args.recompute_env)
    df = df.merge(env, on="nsaid", how="left")
    envcol = f"sigma{args.env_n}"

    vl = df[(df["gal_zdist"] >= args.zmin) & (df["gal_zdist"] <= args.zmax)].copy()
    print(f"volume-limited shell {args.zmin} <= z <= {args.zmax}: {len(vl):,} galaxies "
          f"(central Gaia fraction {vl['has_gaia'].mean():.3f})")

    make_figure(df, vl, envcol, args.nbins)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
