#!/usr/bin/env python3
"""Inspect a single Gaia source against the QSO×Gaia UMAP build — the per-source version of the
qso_gaia_features_hist_picked.png / qso_other_features.png panels.

Given one gaia source_id it prints, all pulled fresh from WSDB (+ the n_all build npz for context):
  * the 14 selected Gaia features, each with its value and where it sits in the full population
    AND in its own HDBSCAN cluster (percentiles) — the numeric form of the features_hist panels;
    values in the extreme tails (<=2 / >=98 pct of the whole population) are flagged.
  * the raw gaia_source astrometry/photometry row (parallax, excess noise, params_solved, ...).
  * the "other-features" context: which HDBSCAN cluster it lands in, Euclid survey (Q1/ERO),
    P(gal), W1-W2, and — if the source is a spectroscopic QSO in the Euclid×Gaia join tables —
    its spec-z + Euclid morphology (point_like / extended / blended probs, mumax-mag, VIS/Y/J/H).
  * Gaia neighbours and Euclid MER neighbours within --radius arcsec (to catch a companion /
    the second blob in a cutout that Gaia may not resolve).

The cluster label comes from the build npz if the source is in it; otherwise the source is
projected through the saved reducer and assigned to the nearest labelled UMAP point.

All file outputs go to a per-source subfolder output/picked_qso/inspect/source_<id>/ so different
sources never mix. By DEFAULT it writes features.csv AND features_hist.png there (the latter is the
single-source analogue of qso_gaia_features_hist_picked.png: the population density + the hand-picked
candidate doubles by QSO-UMAP cluster, with this source marked by a dashed line). Use --no-csv /
--no-hist to skip either.

Usage:
  PGUSER=siwei_zhao_2025 python src/inspect_qso_source.py --gaia 5084798037839071104
  PGUSER=siwei_zhao_2025 python src/inspect_qso_source.py --gaia <id> --no-hist   # print + csv only
  PGUSER=siwei_zhao_2025 python src/inspect_qso_source.py --gaia <id> --radius 10 --sample combined
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

from analyze_picked_qso_umap import (fetch_features, _load_reducer, CSV, OUT_DIR,
                                     n_suffix, parse_n, build_arrays_dir)
from selected_parameters_list import SELECTED_GAIA_PARAMETERS
from builder_qso_gaia_crossmatch import WSDB_HOST, WSDB_DB

EUCLID_GAIA_TABLES = ["crossmatch_qso_desi_dr1_euclid_gaia", "crossmatch_qso_sdss_dr16_euclid_gaia"]


def _pct(v, arr):
    """Percentile of value v within the finite entries of arr (fraction below v, in %)."""
    a = arr[np.isfinite(arr)]
    return 100.0 * (a < v).mean() if (len(a) and np.isfinite(v)) else np.nan


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _read(conn, sql, params):
    """Run one read; roll back on error so a bad column name doesn't poison the next query."""
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception as e:                                    # noqa: BLE001
        conn.rollback()
        print(f"    [query failed: {repr(e)[:140]}]")
        return pd.DataFrame()


def cluster_of(sid, d, user):
    """Return (cluster, umap_xy) for `sid`: read straight from the build if present, else project
    the 14 features through the saved reducer and snap to the nearest labelled UMAP point."""
    sids = d["source_ids"].astype(str)
    hit = np.where(sids == str(sid))[0]
    if len(hit):
        i = hit[0]
        return int(d["cluster"][i]), tuple(float(x) for x in d["embedding"][i]), "in build"
    return None, None, "not in build (projected)"


def project_cluster(sid, d, red_path, user):
    from sklearn.neighbors import NearestNeighbors
    f = fetch_features([sid], user)
    if not len(f):
        return None, None
    X = np.array([[_num(f.iloc[0][p]) for p in SELECTED_GAIA_PARAMETERS]], float)
    mean, std = d["mean"], np.where(d["std"] > 0, d["std"], 1.0)
    Xs = np.where(np.isfinite((X - mean) / std), (X - mean) / std, 0.0)
    emb = d["embedding"]; labels = d["cluster"]; valid = labels >= 0
    pemb = _load_reducer(red_path).transform(Xs).astype(float)
    _, idx = NearestNeighbors(n_neighbors=1).fit(emb[valid]).kneighbors(pemb)
    return int(labels[np.where(valid)[0][idx.ravel()][0]]), tuple(float(x) for x in pemb[0])


def plot_source_hist(sample, X_raw, ids, values, pcts_all, sid, cl, out):
    """Per-source version of qso_gaia_features_hist_picked.png: SAME backdrop as that figure —
    the population density on the LEFT axis (#9ccc9c fill) + the hand-picked candidate doubles'
    COUNT split by QSO-UMAP cluster on the RIGHT axis — with a bold dashed line marking THIS
    source's value; the panel title carries the value and its whole-population percentile. Values
    off the 1-99 pct axis are pinned to the edge and flagged with an arrow."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    from analyze_qso_umap_ds import CLUSTER_COLOURS
    nfeat = len(SELECTED_GAIA_PARAMETERS); ncols = 4
    nrows = int(np.ceil(nfeat / ncols))
    clusters = sorted(int(c) for c in np.unique(ids["cluster"]))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.0))
    axes = np.atleast_2d(axes)
    for fi, feat in enumerate(SELECTED_GAIA_PARAMETERS):
        ax = axes[fi // ncols, fi % ncols]
        colv = X_raw[:, fi].astype(float); finite = colv[np.isfinite(colv)]
        lo, hi = (np.percentile(finite, [1, 99]) if len(finite) else (0.0, 1.0))
        if hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 60)
        # LEFT axis: all-population green backdrop, same as qso_gaia_features_hist(_picked)
        # (#9ccc9c, alpha 0.30, no outline) on its own scale so it stays prominent
        ax.hist(np.clip(finite, lo, hi), bins=bins, density=True, color="#9ccc9c", alpha=0.30,
                edgecolor="none", label=f"population (n={len(finite):,})", zorder=1)
        _fmt = ScalarFormatter(useMathText=True); _fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(_fmt); ax.yaxis.get_offset_text().set_fontsize(6)
        ax.set_ylabel("population density", fontsize=7)
        # RIGHT axis: the hand-picked candidate doubles' COUNT split by QSO-UMAP cluster (same
        # backdrop as qso_gaia_features_hist_picked.png), then this source's line on top
        ax2 = ax.twinx()
        for c in clusters:
            cv = ids.loc[ids["cluster"] == c, feat].to_numpy(float); cf = cv[np.isfinite(cv)]
            if len(cf):
                col = CLUSTER_COLOURS.get(c, "k")
                ax2.hist(np.clip(cf, lo, hi), bins=bins, color=col, alpha=0.2,
                         edgecolor="none", zorder=4)
                ax2.hist(np.clip(cf, lo, hi), bins=bins, histtype="step",
                         color=col, lw=1.2, zorder=5, label=f"C{c} picks (n={len(cf)})")
        ax2.set_ylim(0, 30)
        v = values[fi]; pa = pcts_all[fi]
        if np.isfinite(v):
            vp = float(np.clip(v, lo, hi))
            ax2.axvline(vp, color="k", lw=2, ls="--", zorder=10, label=f"this source (C{cl})")
            if v < lo:
                ax2.annotate("", xy=(lo, 0.5), xycoords=("data", "axes fraction"),
                             xytext=(0.12, 0.5), textcoords="axes fraction",
                             arrowprops=dict(arrowstyle="->", color="k", lw=1.6))
            elif v > hi:
                ax2.annotate("", xy=(hi, 0.5), xycoords=("data", "axes fraction"),
                             xytext=(0.88, 0.5), textcoords="axes fraction",
                             arrowprops=dict(arrowstyle="->", color="k", lw=1.6))
        ax2.set_ylabel("pick count", fontsize=7); ax2.tick_params(labelsize=6)
        vtxt = f"{v:.3g}" if np.isfinite(v) else "NaN"
        ptxt = f"  ({pa:.0f} pct)" if np.isfinite(pa) else ""
        ax.set_title(f"{feat}\nsource = {vtxt}{ptxt}", fontsize=8, fontweight="bold", pad=3)
        ax.set_xlim(lo, hi); ax.tick_params(labelsize=6); ax.grid(alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=5, framealpha=0.9)
    for fi in range(nfeat, nrows * ncols):
        axes[fi // ncols, fi % ncols].set_visible(False)
    fig.suptitle(f"Gaia {sid} (cluster C{cl}) — feature values vs the "
                 f"population  (dashed line = this source)",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gaia", type=int, required=True, help="Gaia DR3 source_id to inspect.")
    ap.add_argument("--sample", choices=["combined", "desi", "sdss"], default="combined",
                    help="which QSO×Gaia UMAP build to use for the population context.")
    ap.add_argument("--n-sample", type=parse_n, default="all",
                    help="build size for the context npz (default 'all').")
    ap.add_argument("--radius", type=float, default=8.0,
                    help="neighbour search radius in arcsec (default 8).")
    ap.add_argument("--csv", action=argparse.BooleanOptionalAction, default=True,
                    help="write the feature table to inspect/source_<id>/features.csv "
                         "(on by default; disable with --no-csv)")
    ap.add_argument("--hist", action=argparse.BooleanOptionalAction, default=True,
                    help="render the per-source features_hist plot to inspect/source_<id>/"
                         "features_hist.png (on by default; disable with --no-hist)")
    args = ap.parse_args()
    sid = args.gaia
    user = os.environ.get("PGUSER", "siwei_zhao_2025")

    adir = build_arrays_dir(args.sample, args.n_sample)
    npz_path = adir / f"{args.sample}_qso_gaia.npz"
    red_path = adir / f"{args.sample}_qso_gaia_reducer.joblib"
    if not npz_path.exists():
        print(f"  missing build npz {npz_path} — build it first (query_qso_umap_ds.py).")
        return 1
    d = np.load(npz_path, allow_pickle=True)
    X = d["X_raw"]; labels = d["cluster"]

    # cluster assignment (in-build lookup, else project)
    cl, xy, how = cluster_of(sid, d, user)
    if cl is None:
        cl, xy = project_cluster(sid, d, red_path, user)
        how = "not in build (projected via reducer)"
    in_cl = labels == cl if cl is not None else np.zeros(len(labels), bool)

    # flat-table metadata (P(gal), W1-W2, Euclid Q1/ERO, catalogue) if this source is a pick / in it
    flat = pd.read_csv(CSV, dtype={"gaia_source_id": "int64"})
    frow = flat[flat["gaia_source_id"] == sid]
    meta = frow.iloc[0].to_dict() if len(frow) else {}

    # 14 features fresh from Gaia
    f = fetch_features([sid], user)
    if not len(f):
        print(f"  Gaia source_id {sid} not found in gaia_dr3.gaia_source.")
        return 1
    frow_feat = f.iloc[0]

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30)

    # raw gaia_source row (for RA/Dec + the astrometry that the derived features hide)
    raw = _read(conn, "SELECT ra, dec, phot_g_mean_mag, bp_rp, parallax, parallax_over_error, "
                      "pmra, pmdec, ruwe, astrometric_excess_noise, astrometric_excess_noise_sig, "
                      "phot_bp_rp_excess_factor, ipd_frac_multi_peak, ipd_frac_odd_win, "
                      "non_single_star, astrometric_params_solved "
                      "FROM gaia_dr3.gaia_source WHERE source_id=%s", (sid,))
    ra = float(raw.iloc[0]["ra"]) if len(raw) else float(meta.get("qso_ra", np.nan))
    dec = float(raw.iloc[0]["dec"]) if len(raw) else float(meta.get("qso_dec", np.nan))

    # ---- header ----
    print("=" * 78)
    print(f"Gaia {sid}   ({args.sample.upper()} build, n={n_suffix(args.n_sample)})")
    print(f"  RA, Dec        = {ra:.6f}, {dec:.6f}")
    print(f"  G              = {frow_feat.get('phot_g_mean_mag', np.nan) if False else meta.get('phot_g_mean_mag', float(raw.iloc[0]['phot_g_mean_mag']) if len(raw) else np.nan)}")
    print(f"  HDBSCAN cluster= C{cl}   [{how}]   at UMAP {tuple(round(v,2) for v in xy) if xy else '—'}")
    if meta:
        print(f"  WISE qso_id    = {meta.get('qso_id','')}   catalogue={meta.get('qso_catalog','')}")
        print(f"  Euclid source  = {meta.get('euclid','')}   P(gal)={meta.get('model_pgal',np.nan):.3f}"
              f"   W1-W2={meta.get('w1_w2',np.nan):.3f}")
    if len(raw):
        aps = int(raw.iloc[0]["astrometric_params_solved"])
        apsn = {3: "2-param (position only; no parallax/PM)", 31: "5-param", 95: "6-param"}.get(aps, str(aps))
        print(f"  params_solved  = {aps}  ({apsn})")

    # ---- 14-feature table with population + cluster percentiles ----
    print("\n14 Gaia features  (pct = percentile within that population; * = <=2 or >=98 pct of ALL)")
    print(f"  {'feature':32s} {'value':>12s} {'pct(all)':>9s} {'pct(C'+str(cl)+')':>9s}")
    rows = []; values = np.full(len(SELECTED_GAIA_PARAMETERS), np.nan)
    pcts_all = np.full(len(SELECTED_GAIA_PARAMETERS), np.nan)
    for i, p in enumerate(SELECTED_GAIA_PARAMETERS):
        v = _num(frow_feat[p])
        pa = _pct(v, X[:, i]); pc = _pct(v, X[in_cl, i]) if in_cl.any() else np.nan
        values[i] = v; pcts_all[i] = pa
        flag = " *" if (np.isfinite(pa) and (pa <= 2 or pa >= 98)) else ""
        print(f"  {p:32s} {v:12.4g} {pa:9.1f} {pc:9.1f}{flag}")
        rows.append({"feature": p, "value": v, "pct_all": pa, f"pct_C{cl}": pc})

    # ---- raw astrometry/photometry ----
    if len(raw):
        print("\nraw gaia_source:")
        for k in ["parallax", "parallax_over_error", "pmra", "pmdec", "ruwe",
                  "astrometric_excess_noise", "astrometric_excess_noise_sig",
                  "phot_bp_rp_excess_factor", "ipd_frac_multi_peak", "ipd_frac_odd_win",
                  "non_single_star", "bp_rp"]:
            print(f"  {k:30s} = {raw.iloc[0][k]}")

    # ---- Euclid×Gaia join morphology (spectroscopic QSOs only) ----
    jcols = ("qso_catalog,qso_id,qso_z,euclid_source,euclid_sep_arcsec,n_euclid_within_1arcsec,"
             "n_qso_per_euclid,euclid_is_mutual_blend,n_gaia_within_1arcsec,gaia_is_mutual_blend,"
             "e_point_like_prob,e_extended_prob,e_blended_prob,e_deblended_flag,e_spurious_flag,"
             "e_fwhm,e_mumax_minus_mag,e_flux_vis,e_flux_y,e_flux_j,e_flux_h")
    found_spec = False
    for t in EUCLID_GAIA_TABLES:
        j = _read(conn, f"SELECT {jcols} FROM {user}.{t} WHERE g_source_id=%s", (sid,))
        if len(j):
            found_spec = True
            print(f"\nEuclid×Gaia join [{t}] — spectroscopic QSO:")
            for k, v in j.iloc[0].items():
                print(f"  {k:30s} = {v}")
    if not found_spec:
        print("\nEuclid×Gaia join: no row (not a DESI/SDSS spec-QSO in the join — "
              "e.g. a CatWISE-selected pick, so no spec-z).")

    # ---- Gaia neighbours ----
    r = args.radius
    gn = _read(conn, "SELECT source_id, phot_g_mean_mag, parallax_over_error, ruwe, "
                     "astrometric_params_solved, q3c_dist(ra,dec,%s,%s)*3600 AS sep_arcsec "
                     "FROM gaia_dr3.gaia_source "
                     "WHERE q3c_radial_query(ra,dec,%s,%s,%s/3600) ORDER BY sep_arcsec",
               (ra, dec, ra, dec, r))
    print(f"\nGaia neighbours within {r:.0f}\" ({len(gn)}):")
    print(gn.to_string(index=False) if len(gn) else "  none")

    # ---- Euclid MER neighbours ----
    en = _read(conn, "SELECT object_id, point_like_prob, extended_prob, mumax_minus_mag, "
                     "spurious_flag, flux_detection_total, "
                     "q3c_dist(right_ascension,declination,%s,%s)*3600 AS sep_arcsec "
                     "FROM euclid_q1.mer_final_cat "
                     "WHERE q3c_radial_query(right_ascension,declination,%s,%s,%s/3600) "
                     "ORDER BY sep_arcsec", (ra, dec, ra, dec, r))
    print(f"\nEuclid MER neighbours within {r:.0f}\" ({len(en)}):")
    print(en.to_string(index=False) if len(en) else "  none")
    conn.close()

    # per-source subfolder so each inspected source's artifacts stay isolated
    src_dir = OUT_DIR / "inspect" / f"source_{sid}"
    if args.csv:
        src_dir.mkdir(parents=True, exist_ok=True)
        out = src_dir / "features.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n  -> wrote {out}")
    if args.hist:
        from analyze_picked_qso_dist import build_ids
        picks_ids, _, _ = build_ids(args.sample, npz_path, red_path, user)  # picks + companions
        src_dir.mkdir(parents=True, exist_ok=True)
        plot_source_hist(args.sample, X, picks_ids, values, pcts_all, sid, cl,
                         src_dir / "features_hist.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
