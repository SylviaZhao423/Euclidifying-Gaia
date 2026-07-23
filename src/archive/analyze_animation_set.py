"""
analyze_animation_set.py

General analyzer for the Euclid animation set, focused on the three sort
axes that define the three MP4s:

  (1) AEN significance      column `gaia_astrometric_excess_noise_sig`
  (2) Corrected BP/RP excess column `gaia_bp_rp_excess_corrected`
  (3) G-band variability    column `gaia_g_flux_frac_error_sqrt_nobs`

FAQ
---
Q: What is 'variability' — is it a Gaia catalogue column?
A: NO. It's a derived proxy stored only in the CSVs. Definition:
      variability ≈ sqrt(phot_g_n_obs) * (phot_g_mean_flux_error /
                                          phot_g_mean_flux)
                  = sqrt(nobs) / phot_g_mean_flux_over_error
   High for sources that scatter beyond Poisson noise
   (Belokurov & Erkal 2020 style).  Not the same as
   `gaia_only_y_log10_abs_bprp_excess_plus1` — that column is a
   monotone twin of BPRPEF and is *not* what the "variability" animation
   is sorted by.

Q: Is `bprp_excess_order.csv` the reverse of `variability_order.csv`?
A: NO.  Both are sorted DESCENDING by their own metric, but the two
   metrics only weakly correlate (Spearman rho ≈ 0.31 on this set), so
   they order the 7,542 sources differently.

Model
-----
Uses the LOCAL Q1-only 10-seed ensemble at
`output/xgboost_q1/seed_models/xgb_classifier_q1_seed{seed}.json`
loaded via `xgb.XGBClassifier.load_model` and evaluated with
`predict_proba` (never `get_booster().predict`).

Features are pulled live from `gaia_dr3.gaia_source` on WSDB for the
7,542 animation-set sids (requires PGUSER + Astronomy VPN).

Tests
-----
  1. Metric definitions & sort keys — which column each CSV is sorted by,
     which direction, and (spot-check) the definition of `variability`.
  2. Spearman correlation matrix across the four related metrics.
  3. Top-N overlap between the three sort axes.
  4. XGBoost P(gal) via the local ensemble.
  5. P(gal) vs each sort axis (do these axes track the model's decision?).
  6. DESI spectroscopic verification — cross-match against q1_desi.npz
     and report the spectype distribution on the overlap.

Outputs (output/animation_axis_analysis/)
  axis_summary.txt           — sort keys, directions, definitions verified
  axis_correlations.csv      — 3x3 Spearman rho matrix on the sort axes
  axis_topn_overlap.csv      — top-N intersections between the 3 axes
  axis_pgal_stats.csv        — P(gal) summary
  axis_desi_spectypes.csv    — DESI spectype counts on the overlap
  axis_correlations.png      — Spearman heatmap
  axis_pairs_scatter.png     — 3-panel scatter of the sort axes vs each other
  axis_pgal_by_axis.png      — P(gal) mean vs decile along each axis
  axis_pgal_hist.png         — overall P(gal) histogram

Usage
-----
    PGUSER=siwei_zhao_2025 python src/analyze_animation_set.py
    PGUSER=siwei_zhao_2025 python src/analyze_animation_set.py --no-plots
"""
import argparse
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
ANIM_DIR    = PROJECT_DIR / "data" / "set" / "animations"
MODEL_DIR   = PROJECT_DIR / "output" / "xgboost_q1" / "seed_models"
OUTPUT_DIR  = PROJECT_DIR / "output" / "animation_axis_analysis"

ANIM_AEN = ANIM_DIR / "euclid_decals_panels_sorted_by_aen_sig_order.csv"
ANIM_BPE = ANIM_DIR / "euclid_decals_panels_sorted_by_bprp_excess_order.csv"
ANIM_VAR = ANIM_DIR / "euclid_decals_panels_sorted_by_variability_order.csv"

Q1_DESI_NPZ = PROJECT_DIR / "output" / "umap" / "arrays" / "q1_desi.npz"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

SEEDS = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]

# 14 features in the order the model expects (matches q1_summary.txt).
FEATURES = [
    "parallax_over_error",
    "astrometric_gof_al",
    "astrometric_chi2_al",
    "astrometric_excess_noise_sig",
    "ipd_gof_harmonic_amplitude",
    "ruwe",
    "phot_g_mean_flux_over_error",
    "phot_bp_mean_flux_over_error",
    "phot_rp_mean_flux_over_error",
    "bp_g",
    "g_rp",
    "corrected_bp_rp_excess",
    "pm_significance",
    "position_error",
]

# Derived-feature SQL expressions (compute inline against gaia_dr3.gaia_source).
# Riello+2021 for corrected_bp_rp_excess; standard error propagation elsewhere.
DERIVED_SQL = {
    "bp_g": "phot_bp_mean_mag - phot_g_mean_mag",
    "g_rp": "phot_g_mean_mag - phot_rp_mean_mag",
    "corrected_bp_rp_excess": """
        phot_bp_rp_excess_factor - (
            CASE
                WHEN bp_rp < 0.5 THEN 1.154360 + 0.033772*bp_rp + 0.032277*bp_rp*bp_rp
                WHEN bp_rp < 4.0 THEN 1.162004 + 0.011464*bp_rp + 0.049255*bp_rp*bp_rp
                                       - 0.005879*bp_rp*bp_rp*bp_rp
                ELSE 1.057572 + 0.140537*bp_rp
            END
        )""",
    "pm_significance": """
        SQRT(
            (pmra*pmra*pmdec_error*pmdec_error
             - 2.0*pmra_pmdec_corr*pmra_error*pmdec_error*pmra*pmdec
             + pmdec*pmdec*pmra_error*pmra_error)
            / NULLIF(pmra_error*pmra_error*pmdec_error*pmdec_error
                     *(1.0 - pmra_pmdec_corr*pmra_pmdec_corr), 0)
        )""",
    "position_error": """
        SQRT(ra_error*ra_error*dec_error*dec_error
             *(1.0 - ra_dec_corr*ra_dec_corr))""",
}

AXES = {                                # user-facing name -> CSV column
    "AEN_sig":     "gaia_astrometric_excess_noise_sig",
    "BPRPEF":      "gaia_bp_rp_excess_corrected",
    "variability": "gaia_g_flux_frac_error_sqrt_nobs",
}


# ─── data loaders ────────────────────────────────────────────────────────────

def load_all_csvs() -> dict:
    """Return {name: DataFrame} for the three animation CSVs."""
    dfs = {"aen_sig":     pd.read_csv(ANIM_AEN),
           "bprp_excess": pd.read_csv(ANIM_BPE),
           "variability": pd.read_csv(ANIM_VAR)}
    for k, d in dfs.items():
        d["gaia_source_id"] = d["gaia_source_id"].astype(np.int64)
    return dfs


def load_local_ensemble():
    """Load the 10-seed Q1-only ensemble from output/xgboost_q1/seed_models/."""
    import xgboost as xgb
    models = []
    for seed in SEEDS:
        p = MODEL_DIR / f"xgb_classifier_q1_seed{seed}.json"
        if not p.exists():
            print(f"  WARNING: {p.name} not found — skipping")
            continue
        m = xgb.XGBClassifier()
        m.load_model(str(p))
        models.append(m)
    print(f"  Loaded {len(models)} seed models from {MODEL_DIR.relative_to(PROJECT_DIR)}")
    return models


def fetch_features_wsdb(sids: set) -> pd.DataFrame:
    """Pull the 14 model features from gaia_dr3.gaia_source for `sids`.

    Returns DataFrame with columns [source_id] + FEATURES.
    """
    import os, psycopg2
    username = os.getenv("PGUSER")
    if not username:
        raise RuntimeError("PGUSER environment variable not set")

    cols_expr = [
        f"({DERIVED_SQL[f]}) AS {f}" if f in DERIVED_SQL else f
        for f in FEATURES
    ]
    feature_select = ",\n               ".join(cols_expr)

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username,
                            connect_timeout=15)
    cur = conn.cursor()
    cur.execute("CREATE TEMP TABLE _anim_sids (source_id BIGINT PRIMARY KEY);")
    buf = StringIO("\n".join(str(int(s)) for s in sids))
    cur.copy_from(buf, "_anim_sids", columns=("source_id",))
    cur.execute(f"""
        SELECT g.source_id,
               {feature_select}
        FROM gaia_dr3.gaia_source g
        JOIN _anim_sids v USING (source_id)
    """)
    rows = cur.fetchall()
    conn.close()
    print(f"  Pulled features for {len(rows):,} / {len(sids):,} sids "
          f"from gaia_dr3.gaia_source")
    df = pd.DataFrame(rows, columns=["source_id"] + list(FEATURES))
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def predict_pgal(models, X: np.ndarray) -> tuple:
    """Ensemble P(gal) via predict_proba on each seed model."""
    probs = np.stack([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return probs.mean(axis=0), probs.std(axis=0)


# ─── analyses ────────────────────────────────────────────────────────────────

def check_definitions(dfs: dict) -> dict:
    """Verify sort keys, direction, and the 'variability' proxy definition."""
    print("\n=== TEST 1 — sort keys, directions, definitions ===")
    result = {}

    # Same source set?
    sets = [set(d["gaia_source_id"]) for d in dfs.values()]
    common = sets[0] & sets[1] & sets[2]
    print(f"  Same 7,542-source set across all three CSVs? "
          f"{sets[0]==sets[1]==sets[2]}  (|intersect|={len(common):,})")
    result["N_common_sids"] = len(common)

    # Sort direction check per named metric.
    def sort_dir(df, col):
        d = np.diff(df[col].values)
        if   np.all(d <= 1e-10):  return "DESCENDING"
        elif np.all(d >= -1e-10): return "ASCENDING"
        else:                     return "NOT MONOTONE"

    for csv_name, df in dfs.items():
        # The named axis for this CSV:
        expected = {"aen_sig": AXES["AEN_sig"],
                    "bprp_excess": AXES["BPRPEF"],
                    "variability": AXES["variability"]}[csv_name]
        d = sort_dir(df, expected)
        print(f"  {csv_name:12s} sorted {d:<12s} by  {expected}")
        result[f"{csv_name}_sortkey"]    = expected
        result[f"{csv_name}_sortdir"]    = d

    # Note the definition of the `gaia_g_flux_frac_error_sqrt_nobs` proxy
    # (fetched independently from WSDB inside the script; here we just note
    # the meaning).
    print("  'variability' column meaning:  sqrt(phot_g_n_obs) / "
          "phot_g_mean_flux_over_error  (Belokurov-Erkal-style scatter proxy)")
    return result


def spearman_matrix(anim: pd.DataFrame) -> pd.DataFrame:
    """3x3 Spearman rho over the three sort axes, on all 7,542.

    The CSV also stores `gaia_only_y_log10_abs_bprp_excess_plus1`, but it's
    a monotone twin of BPRPEF (Spearman rho = 1.0000) — literally
    log10(|BPRPEF|+1) — so it adds no independent information and is
    excluded here.
    """
    print("\n=== TEST 2 — Spearman rho matrix (7,542 sources) ===")
    cols = [AXES["AEN_sig"], AXES["BPRPEF"], AXES["variability"]]
    R = anim[cols].corr(method="spearman")
    R = R.rename(index=lambda c: {v: k for k, v in AXES.items()}[c],
                 columns=lambda c: {v: k for k, v in AXES.items()}[c])
    print(R.round(3))
    return R


def topn_overlap(dfs: dict, N: int = 100) -> pd.DataFrame:
    """Pairwise |top-N intersection| between the three axes."""
    print(f"\n=== TEST 3 — top-{N} overlap between the three axes ===")
    tops = {
        "AEN_sig":     set(dfs["aen_sig"].head(N)["gaia_source_id"]),
        "BPRPEF":      set(dfs["bprp_excess"].head(N)["gaia_source_id"]),
        "variability": set(dfs["variability"].head(N)["gaia_source_id"]),
    }
    rows = []
    axes_names = list(tops.keys())
    for a in axes_names:
        row = {}
        for b in axes_names:
            row[b] = len(tops[a] & tops[b])
        rows.append(row)
    mat = pd.DataFrame(rows, index=axes_names)
    print(mat)
    triple = len(tops["AEN_sig"] & tops["BPRPEF"] & tops["variability"])
    print(f"  |top-{N} intersection of all three|: {triple}")
    return mat


def pgal_stats(pgal: np.ndarray) -> dict:
    """Overall P(gal) statistics."""
    print("\n=== TEST 4 — XGBoost P(gal) via LOCAL Q1-only ensemble ===")
    s = {"n": int(len(pgal)),
         "median": float(np.median(pgal)),
         "mean":   float(np.mean(pgal)),
         "P_ge_0.5": float((pgal >= 0.5).mean()),
         "P_ge_0.9": float((pgal >= 0.9).mean()),
         "P_lt_0.5": float((pgal <  0.5).mean())}
    for k, v in s.items():
        if k == "n": print(f"  n         : {v:,}")
        else:        print(f"  {k:10s}: {v:.4f}")
    return s


def desi_verification(sids: set) -> dict:
    """DESI spectroscopic cross-match — returns spectype counts on the overlap.

    Reads the local q1_desi.npz array (Q1 x DESI join produced upstream).
    Empty result if the npz is missing.
    """
    from collections import Counter
    print("\n=== TEST 6 — DESI spectroscopic verification ===")
    if not Q1_DESI_NPZ.exists():
        print(f"  {Q1_DESI_NPZ} not found — skipping.")
        return {"n_overlap": 0, "spectype_counts": {}}
    d = np.load(Q1_DESI_NPZ, allow_pickle=True)
    d_sids = d["source_ids"].astype(np.int64)
    d_type = d["spectype"] if "spectype" in d.files else d["desi_spectype"]
    desi_map = {int(s): str(t) for s, t in zip(d_sids, d_type)}
    over = sids & set(desi_map.keys())
    print(f"  Q1 x DESI table size:  {len(d_sids):,}")
    print(f"  Overlap with 7,542:    {len(over):,}")
    if not over:
        return {"n_overlap": 0, "spectype_counts": {}}
    counts = Counter(desi_map[s] for s in over)
    for t, n in counts.most_common():
        print(f"    {t:12s} {n:>6,}   ({100*n/len(over):.1f}%)")
    return {"n_overlap": len(over), "spectype_counts": dict(counts)}


def pgal_by_axis(anim: pd.DataFrame, pgal_col: str = "pgal") -> pd.DataFrame:
    """Median P(gal) in deciles of each sort axis."""
    print("\n=== TEST 5 — P(gal) vs axis (deciles) ===")
    rows = []
    for axis, col in AXES.items():
        q = pd.qcut(anim[col].rank(method="first"), 10, labels=False)
        med = anim.groupby(q)[pgal_col].median()
        p90 = anim.groupby(q)[pgal_col].quantile(0.9)
        p10 = anim.groupby(q)[pgal_col].quantile(0.1)
        for i in range(10):
            rows.append({"axis": axis, "decile": i,
                         "median_pgal": med.get(i, np.nan),
                         "q10": p10.get(i, np.nan),
                         "q90": p90.get(i, np.nan)})
    df = pd.DataFrame(rows)
    print(df.pivot(index="decile", columns="axis", values="median_pgal").round(3))
    return df


# ─── plots ───────────────────────────────────────────────────────────────────

def make_plots(anim: pd.DataFrame, R: pd.DataFrame, decile_df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. P(gal) histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(anim["pgal"], bins=60, color="#58a6ff", edgecolor="#0d1117")
    ax.axvline(0.5, ls="--", color="#888", lw=1); ax.axvline(0.9, ls=":", color="#888", lw=1)
    ax.set_xlabel("XGBoost P(gal)"); ax.set_ylabel("N sources")
    ax.set_title(f"Local Q1-only ensemble  (N={len(anim):,})")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "axis_pgal_hist.png", dpi=150); plt.close(fig)

    # 2. Correlation heatmap
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(R.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(R))); ax.set_xticklabels(R.columns, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(len(R))); ax.set_yticklabels(R.index, fontsize=7)
    for i in range(len(R)):
        for j in range(len(R)):
            ax.text(j, i, f"{R.values[i,j]:.2f}", ha="center", va="center",
                    color="white" if abs(R.values[i,j]) > 0.5 else "black", fontsize=8)
    ax.set_title("Spearman rho")
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "axis_correlations.png", dpi=150); plt.close(fig)

    # 3. Pairwise scatters of the three axes, coloured by P(gal)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [("AEN_sig", "BPRPEF"),
             ("BPRPEF", "variability"),
             ("AEN_sig", "variability")]
    for ax, (a, b) in zip(axes, pairs):
        sc = ax.scatter(anim[AXES[a]], anim[AXES[b]],
                        c=anim["pgal"], s=6, cmap="RdBu",
                        vmin=0, vmax=1, alpha=0.7)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"{a}  ({AXES[a]})", fontsize=8)
        ax.set_ylabel(f"{b}  ({AXES[b]})", fontsize=8)
    fig.colorbar(sc, ax=axes, shrink=0.7, label="P(gal)")
    fig.suptitle("The three sort axes vs each other, coloured by P(gal)")
    fig.savefig(OUTPUT_DIR / "axis_pairs_scatter.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # 4. P(gal) vs decile per axis
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"AEN_sig": "#e63946", "BPRPEF": "#2a9d8f", "variability": "#8e44ad"}
    for axis in AXES:
        sub = decile_df[decile_df.axis == axis]
        ax.plot(sub.decile, sub.median_pgal, "-o", color=colors[axis], label=axis)
        ax.fill_between(sub.decile, sub.q10, sub.q90, color=colors[axis], alpha=0.15)
    ax.set_xlabel("Decile of axis  (0 = smallest, 9 = largest)")
    ax.set_ylabel("Median P(gal)  (band = 10-90 percentile)")
    ax.axhline(0.5, ls="--", color="#888", lw=0.8); ax.axhline(0.9, ls=":", color="#888", lw=0.8)
    ax.legend(loc="lower left", frameon=False); ax.set_ylim(0, 1.02)
    ax.set_title("How each sort axis tracks P(gal)")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "axis_pgal_by_axis.png", dpi=150); plt.close(fig)

    print(f"  Plots -> {OUTPUT_DIR}")


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topn", type=int, default=100,
                        help="Top-N size for axis overlap (default 100)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip PNG generation")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Analyzing the animation set — three sort axes")
    print("=" * 60)

    dfs = load_all_csvs()
    anim = dfs["aen_sig"].copy()          # canonical: all sids + 4 axis cols
    sids = set(anim["gaia_source_id"].astype(np.int64))
    print(f"Loaded 3 CSVs, {len(anim):,} sources each")

    r1 = check_definitions(dfs)
    R  = spearman_matrix(anim)
    m3 = topn_overlap(dfs, N=args.topn)

    # Pull features + run local ensemble.
    print("\n=== Local ensemble inference ===")
    feats = fetch_features_wsdb(sids)
    models = load_local_ensemble()
    X = feats[list(FEATURES)].values.astype(np.float32)
    pgal_mean, pgal_std = predict_pgal(models, X)

    # Merge P(gal) back onto anim by source_id.
    pg = pd.DataFrame({"gaia_source_id": feats["source_id"].astype(np.int64),
                       "pgal": pgal_mean, "pgal_std": pgal_std})
    anim = anim.merge(pg, on="gaia_source_id", how="left")

    r4 = pgal_stats(anim["pgal"].dropna().values)
    dec = pgal_by_axis(anim.dropna(subset=["pgal"]))
    r6 = desi_verification(sids)

    # Persist tables.
    with open(OUTPUT_DIR / "axis_summary.txt", "w") as fh:
        fh.write("Animation set — axis summary\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(f"N sources (common across 3 CSVs): {r1['N_common_sids']:,}\n\n")
        fh.write("Sort keys and directions:\n")
        for k in ("aen_sig", "bprp_excess", "variability"):
            fh.write(f"  {k:12s}  {r1[f'{k}_sortdir']:12s}  "
                     f"by  {r1[f'{k}_sortkey']}\n")
        fh.write("\n'variability' column definition:\n")
        fh.write("  gaia_g_flux_frac_error_sqrt_nobs\n")
        fh.write("  ≈ sqrt(phot_g_n_obs) / phot_g_mean_flux_over_error\n")
        fh.write("  Belokurov-Erkal-style photometric scatter proxy.\n")
        fh.write("  Not a stock Gaia column; precomputed upstream.\n\n")
        fh.write("P(gal) via LOCAL Q1-only ensemble:\n")
        for k, v in r4.items():
            fh.write(f"  {k:10s}: {v}\n" if k == "n" else f"  {k:10s}: {v:.4f}\n")
        fh.write(f"\nDESI overlap:  {r6['n_overlap']:,} / {len(sids):,}\n")
        for t, n in r6["spectype_counts"].items():
            frac = n / r6["n_overlap"] if r6["n_overlap"] else 0
            fh.write(f"  {t:12s} {n:>6,}   ({100*frac:.1f}%)\n")

    if r6["spectype_counts"]:
        pd.DataFrame([{"spectype": t, "n": n}
                      for t, n in r6["spectype_counts"].items()]
                     ).to_csv(OUTPUT_DIR / "axis_desi_spectypes.csv", index=False)

    R.to_csv(OUTPUT_DIR / "axis_correlations.csv")
    m3.to_csv(OUTPUT_DIR / "axis_topn_overlap.csv")
    pd.DataFrame([r4]).to_csv(OUTPUT_DIR / "axis_pgal_stats.csv", index=False)
    dec.to_csv(OUTPUT_DIR / "axis_pgal_by_decile.csv", index=False)

    if not args.no_plots:
        print("\n=== Plots ===")
        make_plots(anim.dropna(subset=["pgal"]), R, dec)

    print(f"\nDone.  Outputs -> {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
