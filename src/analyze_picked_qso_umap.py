#!/usr/bin/env python3
"""Project the hand-picked QSO Gaia sources onto a DESI/SDSS/combined QSO×Gaia UMAP,
not the Q1 UMAP.

Same idea as analyze_picked_umap.py, but the target embedding + reducer + cluster labels
come from one self-contained npz under
    output/qso_umap/base/arrays/combined_qso_gaia.{npz,_reducer.joblib}    (combined, default)
    output/qso_umap/ds_qso/n{N}/arrays/{sample}_qso_gaia.{npz,_reducer.joblib}  (desi/sdss)
(default sample = combined; the combined build is served from the curated base/ bundle, see
build_arrays_dir(); desi/sdss still come from ds_qso).
The 14 Gaia features are pulled FRESH per gaia_source_id,
standardised with THAT sample's train mean/std (stored in its npz), and run through the
saved reducer — exactly like analyze_qso_umap_ds.py:project_all().

Projected ids:
  * the hand-picked primaries (analyze_picked_qso_contact_sheet.PICKED)
  * every companion Gaia id inside the 10 multi_gaia blends (the 2nd source within 1"),
    so each blend's two Gaia components are projected individually.

Reads ONLY {sample}_qso_gaia.npz + {sample}_qso_gaia_reducer.joblib (combined from the base/
bundle; desi/sdss from ds_qso); every plot below is generated fresh (no PNGs are copied).

(The picked-projection figures picked_qso_umap.png + picked_qso_umap_og.png — coloured by cluster
+ colour(z) river deviation — were retired and archived along with analyze_picked_color_vs_z.py.)

Outputs (output/picked_qso/); the default `combined` sample drops its name prefix, other
samples keep a `{sample}_` prefix (e.g. desi_qso_gaia_features.png):
  qso_gaia_features.png            14-feature grid, population coloured by each param
  qso_gaia_features_picked.png     same grid + the picks over-plotted (same colour scale)
  qso_gaia_clusters.png            standalone HDBSCAN-cluster UMAP
  qso_gaia_gmag.png                2-panel: UMAP coloured by Gaia G | G-mag distribution
      by HDBSCAN cluster (density-normalised)

The Gaia-parameter DISTRIBUTION grids (qso_gaia_features_hist{,_by_cluster}.png) now live in
analyze_picked_qso_dist.py; the population "other-features" quick-look (qso_other_features.png)
now lives in analyze_picked_qso_other.py.

Usage:
  PGUSER=… python src/analyze_picked_qso_umap.py                 # combined UMAP
  PGUSER=… python src/analyze_picked_qso_umap.py --sample desi
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from pathlib import Path
import psycopg2
import joblib
from sklearn.neighbors import NearestNeighbors

from analyze_picked_qso_contact_sheet import PICKED
from analyze_qso_umap_ds import CLUSTER_COLOURS
from builder_qso_gaia_crossmatch import WSDB_HOST, WSDB_DB, FEATURE_SQL
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parent.parent
DS_DIR      = PROJECT_DIR / "output" / "qso_umap" / "ds_qso"   # <n{N}>/arrays/{sample}_qso_gaia.*
BASE_ARRAYS = PROJECT_DIR / "output" / "qso_umap" / "base" / "arrays"  # curated combined-only bundle
OUT_DIR     = PROJECT_DIR / "output" / "picked_qso"
CSV         = PROJECT_DIR / "output" / "umap" / "qso" / "arrays" / "catwise_euclid_gaia.csv"
Q1_ARRAYS   = PROJECT_DIR / "output" / "umap" / "arrays"   # original Q1 UMAP (q1_indiv.*)
RIVERS_CSV  = PROJECT_DIR / "output" / "qso_color_vs_z" / "arrays" / "rivers_qso_2fwhm.csv"
# Euclid∩Gaia join tables (built by crossmatch_qso_*_euclid_gaia.py): the QSOs that have
# BOTH a Euclid and a Gaia match — i.e. exactly the population that can appear in this UMAP.
# Reading these instead of the Euclid-only tables restricts the metric set to the Gaia-matched
# sources for free (no separate Gaia filter needed).
EUCLID_XM   = ["crossmatch_qso_desi_dr1_euclid_gaia", "crossmatch_qso_sdss_dr16_euclid_gaia"]
GAIA_XM     = ["crossmatch_qso_desi_dr1_gaia", "crossmatch_qso_sdss_dr16_gaia"]  # carry aps
MER_TABLE   = "euclid_q1.mer_final_cat"


def n_suffix(n):  return "n_all" if str(n) == "all" else f"n{int(n)}"
def parse_n(v):   return "all" if str(v).lower() == "all" else int(v)
def nlabel(n):    return "all" if str(n) == "all" else f"{int(n):,}"


def build_arrays_dir(sample, n):
    """Arrays folder holding {sample}_qso_gaia.{npz,_reducer.joblib}. The COMBINED build now lives
    in the curated base/ bundle (output/qso_umap/base/arrays/); desi/sdss stay under ds_qso/n{N}/."""
    return BASE_ARRAYS if sample == "combined" else DS_DIR / n_suffix(n) / "arrays"

# multi_gaia blends: primary gaia id -> companion gaia id (2nd source within 1")
MULTI_GAIA_PAIRS = {
    4182450736372664320: 4182450736372664576,
    5057350452333817344: 5057350452336259072,
    1634449293392314624: 1634449293393717632,
    3219151075810716800: 3219151071516780544,
    4830604663329901568: 4830604663328497920,
    4974123297627607936: 4974123297629291008,
    2161082515187886208: 2161082515186818688,
    4182448842295887616: 4182448846589599744,
    4778637895508914048: 4778637895509908608,
    4829578127489714304: 4829578131784739968,
}


def _load_reducer(path):
    import pynndescent
    for attr, default in [("quantization", None), ("_quantized_data", None),
                          ("parallel_batch_queries", False)]:
        if not hasattr(pynndescent.NNDescent, attr):
            setattr(pynndescent.NNDescent, attr, default)
    return joblib.load(path)


def fetch_features(ids, user):
    """Pull the 14 derived Gaia features per source_id (bare-name columns)."""
    feat_sql = FEATURE_SQL.replace(" AS g_", " AS ")   # g_<name> -> <name>
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30)
    cur = conn.cursor()
    cur.execute(f"SELECT g.source_id, {feat_sql} FROM gaia_dr3.gaia_source g "
                f"WHERE source_id = ANY(%s)", (list(ids),))
    cols = ["source_id"] + list(SELECTED_GAIA_PARAMETERS)
    df = pd.DataFrame(cur.fetchall(), columns=cols)
    conn.close()
    return df


def euclid_matched_metrics(user):
    """Euclid∩Gaia spectroscopic QSOs (DESI+SDSS), read from the crossmatch_qso_*_euclid_gaia
    join tables and restricted to euclid_source='Q1' (the XGB photo-z model + the colour(z)
    locus both require Q1 mer_final_cat aperture photometry; ERO has no comparable apertures).
    For each Q1 source, from its Euclid photometry: (a) RMS colour distance from the spec-z QSO
    median colour(z) locus at its own spec-z, and (b) its XGBoost photo-z (10 seed models) ->
    residual dz = z_xgb - spec-z. Returns a DataFrame keyed by qso_id. These are the only
    population sources with Euclid colours; everything else in the UMAP has none."""
    import json
    import xgboost as xgb
    from analyze_qso_color_vs_specz import (_flux_to_mag, ZPC_VIS, COLORS, _running_stats, ZMAX_PLOT)
    from train_xgboost_qso_photz import add_colours
    _BAND = {"IE": "vis", "YE": "y", "JE": "j", "HE": "h"}
    XGB_DIR = PROJECT_DIR / "output" / "xgboost_qso_photz"; MODELS_DIR = XGB_DIR / "models"
    feats = json.loads((MODELS_DIR / "features.json").read_text())["features"]
    target = json.loads((XGB_DIR / "metrics.json").read_text())["target"]

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30)
    conn.cursor().execute("SET statement_timeout = 300000")
    frames = [pd.read_sql_query(
        f"SELECT qso_id::text AS qso_id, qso_z, e_id AS object_id FROM {user}.{T} "
        f"WHERE euclid_source = 'Q1' AND e_id IS NOT NULL", conn) for T in EUCLID_XM]
    # keep EVERY Q1 QSO (one row per qso_id). Dedup ONLY the mer fetch, not idm — a handful of
    # QSO pairs share one Euclid source (n_qso_per_euclid > 1); both are distinct QSOs with their
    # own spec-z, so both get a metric (same photometry evaluated at each one's spec-z).
    idm = pd.concat(frames, ignore_index=True)
    fcols = [f"flux_{s}_{a}fwhm_aper" for s in _BAND.values() for a in (1, 2, 3, 4)]
    ecols = [f"fluxerr_{s}_{a}fwhm_aper" for s in _BAND.values() for a in (1, 2, 3, 4)]
    mer = pd.read_sql_query(
        f"SELECT object_id, {', '.join(fcols + ecols)} FROM {MER_TABLE} "
        f"WHERE object_id = ANY(%(b)s)", conn, params={"b": idm["object_id"].unique().tolist()})
    conn.close()
    df = idm.merge(mer, on="object_id", how="inner").reset_index(drop=True)
    for band, s in _BAND.items():
        zpc = ZPC_VIS if s == "vis" else 1.0
        for a in (1, 2, 3, 4):
            df[f"{band}_{a}"] = _flux_to_mag(df[f"flux_{s}_{a}fwhm_aper"],
                                             df[f"fluxerr_{s}_{a}fwhm_aper"], zpc)
    df = add_colours(df)                                   # 12 within-aperture model colours
    for _, b, r in COLORS:                                 # 3 canonical 2FWHM colours for the locus dist
        df[f"{b}_{r}"] = df[f"{b}_2"] - df[f"{r}_2"]

    # XGB photo-z (rows with all model features), then dz = z_xgb - spec-z
    df["z_xgb"] = np.nan
    m = df[feats].notna().all(axis=1)
    if m.any():
        X = df.loc[m, feats].to_numpy(float)
        preds = []
        for p in sorted(MODELS_DIR.glob("xgb_photz_seed*.json")):
            mo = xgb.XGBRegressor(); mo.load_model(str(p)); pr = mo.predict(X)
            preds.append(np.expm1(pr) if target == "log1p_z" else pr)
        df.loc[m, "z_xgb"] = np.mean(preds, axis=0)
    df["dz"] = df["z_xgb"] - pd.to_numeric(df["qso_z"], errors="coerce")

    # colour(z) distance from the spec-z QSO median (rivers reference)
    riv = pd.read_csv(RIVERS_CSV); zr = riv["qso_z"].to_numpy(float)
    med = {}
    for _, b, r in COLORS:
        col = f"{b}_{r}"; c = riv[col].to_numpy(float)
        mm = np.isfinite(c) & np.isfinite(zr) & (zr <= ZMAX_PLOT)
        mid, mmed, _, _ = _running_stats(zr[mm], c[mm])
        med[col] = (np.asarray(mid), np.asarray(mmed))
    dists = []
    for _, s in df.iterrows():
        z = s["qso_z"]; sq, nok = 0.0, 0
        if np.isfinite(z):
            for _, b, r in COLORS:
                cval = s[f"{b}_{r}"]
                if np.isfinite(cval):
                    sq += (cval - np.interp(z, med[f"{b}_{r}"][0], med[f"{b}_{r}"][1])) ** 2; nok += 1
        dists.append(np.sqrt(sq / nok) if nok else np.nan)
    df["dist"] = dists
    print(f"  Euclid-matched population QSOs: {len(df):,} "
          f"({int(df['dist'].notna().sum()):,} with colour dist, {int(df['z_xgb'].notna().sum()):,} with z_xgb)")
    return df[["qso_id", "qso_z", "dist", "z_xgb", "dz"]]


def _add_cluster_labels(ax, emb, labels, xlim=None, ylim=None, fontsize=13):
    """Overlay C0..C3 centroid labels on a UMAP scatter (same style as plot_gmag / the picked
    projection). Labels of small/compact clusters are nudged radially outward so they clear the
    handful of points there instead of sitting on top of them."""
    if labels is None:
        return
    cl_ids = sorted(int(c) for c in np.unique(labels) if c >= 0)
    if not cl_ids:
        return
    if xlim is None:
        xlim = (emb[:, 0].min(), emb[:, 0].max())
    if ylim is None:
        ylim = (emb[:, 1].min(), emb[:, 1].max())
    centroids = {c: (float(emb[labels == c, 0].mean()), float(emb[labels == c, 1].mean()))
                 for c in cl_ids}
    cradius = {c: float(np.percentile(np.hypot(emb[labels == c, 0] - centroids[c][0],
                                               emb[labels == c, 1] - centroids[c][1]), 95))
               for c in cl_ids}
    gcen = (float(np.mean([centroids[c][0] for c in cl_ids])),
            float(np.mean([centroids[c][1] for c in cl_ids])))
    span = min(xlim[1] - xlim[0], ylim[1] - ylim[0])
    for c in cl_ids:
        cx, cy = centroids[c]
        lx, ly = cx, cy
        if cradius[c] < 0.10 * span:
            vx, vy = cx - gcen[0], cy - gcen[1]
            nrm = np.hypot(vx, vy) or 1.0
            offd = cradius[c] + 0.045 * span
            lx, ly = cx + vx / nrm * offd, cy + vy / nrm * offd
        t = ax.text(lx, ly, f"C{c}", fontsize=fontsize, fontweight="bold", ha="center",
                    va="center", color=CLUSTER_COLOURS.get(c, "k"), zorder=8)
        t.set_path_effects([pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()])


def plot_umap_clusters(sample, emb, labels, out):
    """Standalone HDBSCAN-cluster UMAP (split out of the quick-look figure)."""
    from matplotlib import cm
    cl_ids = sorted(set(int(c) for c in labels))
    fig, ax = plt.subplots(figsize=(8.0, 7.6))
    cmap = cm.get_cmap("tab10")
    for c in cl_ids:
        mm = labels == c
        col = "0.8" if c < 0 else cmap(c % 10)
        ax.scatter(emb[mm, 0], emb[mm, 1], s=4, color=col, alpha=0.6, lw=0, rasterized=True,
                   label=("noise" if c < 0 else f"C{c}"))
    ax.set_title(f"{sample.upper()} QSO × Gaia UMAP — HDBSCAN clusters "
                 f"({len([c for c in cl_ids if c >= 0])})", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=8, markerscale=2, framealpha=0.9, ncol=2)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


def plot_features_grid(sample, emb, X_raw, out, ids=None, bg_alpha=0.6, labels=None):
    """14-feature grid: the full UMAP population coloured by each Gaia parameter
    (RdYlBu_r, 1-99 pct clip) at `bg_alpha`. If `ids` is given, the picks are over-plotted
    as big black-edged stars in the SAME colour scale. If `labels` is given, each panel gets
    C0..C3 centroid labels. Everything is generated fresh from the build npz — no PNGs are
    read/copied."""
    nfeat = len(SELECTED_GAIA_PARAMETERS); ncols = 4
    nrows = int(np.ceil(nfeat / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.0))
    axes = np.atleast_2d(axes)
    star_xy = ids[["umap1", "umap2"]].to_numpy(float) if ids is not None else None
    for fi, feat in enumerate(SELECTED_GAIA_PARAMETERS):
        ax = axes[fi // ncols, fi % ncols]
        pop = X_raw[:, fi].astype(float)
        finite = pop[np.isfinite(pop)]
        lo, hi = (np.percentile(finite, [1, 99]) if len(finite) else (0.0, 1.0))
        if hi <= lo:
            hi = lo + 1.0
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=np.clip(pop, lo, hi), cmap="RdYlBu_r",
                        vmin=lo, vmax=hi, s=4, alpha=bg_alpha, marker="o", rasterized=True,
                        linewidths=0, zorder=1)
        if ids is not None:
            pv = ids[feat].to_numpy(float)
            ax.scatter(star_xy[:, 0], star_xy[:, 1], c=np.clip(pv, lo, hi), cmap="RdYlBu_r",
                       vmin=lo, vmax=hi, s=95, marker="*", edgecolors="k", linewidths=0.6, zorder=3)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.25)
        ax.set_title(feat, fontsize=9, fontweight="bold", pad=4)
        _add_cluster_labels(ax, emb, labels, fontsize=8)
        fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046).ax.tick_params(labelsize=7)
    for fi in range(nfeat, nrows * ncols):
        axes[fi // ncols, fi % ncols].set_visible(False)
    tail = f" — {len(ids)} hand-picks over-plotted" if ids is not None else ""
    fig.suptitle(f"{sample.upper()} QSO × Gaia UMAP coloured by Gaia parameters  "
                 f"(G ≥ 17, {nfeat} features){tail}", fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


def plot_gmag(sample, emb, g_mag_pop, labels, out):
    """2-panel: UMAP coloured by Gaia G (left) + G-mag distribution split by HDBSCAN cluster
    (density-normalised) (right)."""
    cl_ids = sorted(int(c) for c in np.unique(labels) if c >= 0)
    counts = {c: int((labels == c).sum()) for c in cl_ids}
    xlim = (emb[:, 0].min() - 0.8, emb[:, 0].max() + 0.8)
    ylim = (emb[:, 1].min() - 0.8, emb[:, 1].max() + 0.8)
    gm = np.asarray(g_mag_pop, float); m = np.isfinite(gm)
    lo, hi = np.percentile(gm[m], [1, 99]); bins = np.linspace(lo, hi, 60)
    fig, (axU, axC) = plt.subplots(1, 2, figsize=(17, 8))

    # panel 1: UMAP coloured by Gaia G (magma_r: bright/low-G = pale, faint/high-G = dark)
    sc = axU.scatter(emb[m, 0], emb[m, 1], c=np.clip(gm[m], lo, hi), s=4, cmap="magma_r",
                     vmin=lo, vmax=hi, alpha=0.6, lw=0, rasterized=True)
    axU.set_title("coloured by Gaia $G$", fontsize=13, fontweight="bold")
    fig.colorbar(sc, ax=axU, fraction=0.046, pad=0.02).set_label("G (mag)", fontsize=10)
    axU.set_xlim(xlim); axU.set_ylim(ylim); axU.set_xticks([]); axU.set_yticks([])
    axU.set_xlabel("UMAP-1"); axU.set_ylabel("UMAP-2")
    # cluster centroid labels (C0..C3) — same style + small-cluster nudging as everywhere else
    _add_cluster_labels(axU, emb, labels, xlim, ylim)

    # panel 2: G distribution split by HDBSCAN cluster (density-normalised)
    for c in cl_ids:
        cf = gm[labels == c]; cf = cf[np.isfinite(cf)]
        if len(cf) > 10:
            axC.hist(np.clip(cf, lo, hi), bins=bins, density=True, histtype="step",
                     color=CLUSTER_COLOURS.get(c, "k"), lw=1.6, label=f"C{c} (n={counts[c]:,})")
    axC.set_xlim(lo, hi); axC.set_xlabel("Gaia G (mag)", fontsize=11); axC.set_ylabel("density", fontsize=9)
    axC.grid(alpha=0.3); axC.legend(loc="upper right", fontsize=8, framealpha=0.9, title="cluster")
    axC.set_title("Gaia G by HDBSCAN cluster (density)", fontsize=13, fontweight="bold")

    fig.suptitle(f"{sample.upper()} QSO × Gaia — Gaia G magnitude  (G ≥ 17, N={int(m.sum()):,})",
                 fontsize=16, fontweight="bold", y=0.96)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", choices=["combined", "desi", "sdss"], default="combined",
                    help="which QSO×Gaia UMAP to project onto (default combined).")
    ap.add_argument("--n-sample", type=parse_n, default="all",
                    help="ds_qso build size used as the UMAP background: int or 'all' (default). "
                         "n480000 is a subsample (only ~half the Euclid∩Gaia sources land in it); "
                         "'all' uses every source so the Euclid-matched panels recover the full "
                         "Euclid∩Gaia set.")
    args = ap.parse_args()
    sample = args.sample
    n = args.n_sample
    user = os.environ.get("PGUSER", "siwei_zhao_2025")

    ds_arrays = build_arrays_dir(sample, n)
    npz_path = ds_arrays / f"{sample}_qso_gaia.npz"
    red_path = ds_arrays / f"{sample}_qso_gaia_reducer.joblib"
    if not (npz_path.exists() and red_path.exists()):
        print(f"  missing {n_suffix(n)} arrays for {sample} — build it first "
              f"(query_qso_umap_ds.py).")
        return 1
    # combined is the default sample -> drop the "combined_" prefix from the plot names
    pfx = "" if sample == "combined" else f"{sample}_"

    # WISE label per primary (for annotation), from the flat table
    flat = pd.read_csv(CSV, dtype={"gaia_source_id": "int64"})
    wise_of = dict(zip(flat["gaia_source_id"], flat["qso_id"]))

    # Build the id table: primaries + multi_gaia companions
    rows = [{"gaia_source_id": g, "role": "primary",
             "wise": wise_of.get(g, ""), "pair": g} for g in PICKED]
    for prim, comp in MULTI_GAIA_PAIRS.items():
        rows.append({"gaia_source_id": comp, "role": "companion",
                     "wise": wise_of.get(prim, ""), "pair": prim})
    ids = pd.DataFrame(rows).drop_duplicates("gaia_source_id").reset_index(drop=True)
    print(f"  projecting {len(ids)} gaia ids onto the {sample.upper()} QSO×Gaia UMAP "
          f"({(ids.role=='primary').sum()} primary + {(ids.role=='companion').sum()} companion)")

    # 14 features fresh from gaia_dr3
    feats = fetch_features(ids["gaia_source_id"].tolist(), user)
    feats = feats.rename(columns={"source_id": "gaia_source_id"})
    ids = ids.merge(feats, on="gaia_source_id", how="left")
    # Gaia G per pick (not one of the 14 features) — from the flat table, for the G-mag panel
    ids["g_mag"] = ids["gaia_source_id"].map(dict(zip(flat["gaia_source_id"], flat["phot_g_mean_mag"])))

    # Standardise with THIS sample's train mean/std, then reducer.transform (== project_all)
    d = np.load(npz_path, allow_pickle=True)
    q1_emb = d["embedding"]
    pop_X_raw = d["X_raw"]        # population 14-feature values (for the feature-grid backdrop)
    pop_g_mag = d["g_mag"]        # population Gaia G magnitude
    labels = d["cluster"]
    mean, std = d["mean"], d["std"]
    std_safe = np.where(std > 0, std, 1.0)
    X  = ids[list(SELECTED_GAIA_PARAMETERS)].apply(pd.to_numeric, errors="coerce").values.astype(float)
    Xs = np.where(np.isfinite((X - mean) / std_safe), (X - mean) / std_safe, 0.0)
    emb = _load_reducer(red_path).transform(Xs).astype(float)
    ids["umap1"], ids["umap2"] = emb[:, 0], emb[:, 1]

    # nearest-QSO-UMAP-neighbour cluster assignment
    valid = labels >= 0
    nn = NearestNeighbors(n_neighbors=1).fit(q1_emb[valid])
    _, idx = nn.kneighbors(emb)
    ids["cluster"] = labels[np.where(valid)[0][idx.ravel()]]

    cluster_ids = sorted(set(labels[labels >= 0]))
    centroids = {c: (q1_emb[labels == c, 0].mean(), q1_emb[labels == c, 1].mean())
                 for c in cluster_ids}
    # per-cluster radius (95th-pct spread) so tiny clusters' labels can be nudged clear of them
    cradius = {c: (float(np.percentile(np.hypot(q1_emb[labels == c, 0] - centroids[c][0],
                                                q1_emb[labels == c, 1] - centroids[c][1]), 95))
                   if (labels == c).any() else 0.0) for c in cluster_ids}
    gcen = (float(np.mean([centroids[c][0] for c in cluster_ids])),
            float(np.mean([centroids[c][1] for c in cluster_ids])))
    xlim = (q1_emb[:, 0].min() - 0.8, q1_emb[:, 0].max() + 0.8)
    ylim = (q1_emb[:, 1].min() - 0.8, q1_emb[:, 1].max() + 0.8)

    pos = dict(zip(ids["gaia_source_id"], zip(ids["umap1"], ids["umap2"])))
    prim = ids[ids.role == "primary"]
    comp = ids[ids.role == "companion"]

    def _base(ax):
        """grey QSO-UMAP background + blend-connecting lines + centroid labels + limits."""
        ax.scatter(q1_emb[:, 0], q1_emb[:, 1], s=2, c="#e2e2e2", alpha=0.6,
                   linewidths=0, rasterized=True, zorder=1)
        for p, csrc in MULTI_GAIA_PAIRS.items():
            if p in pos and csrc in pos:
                (x1, y1), (x2, y2) = pos[p], pos[csrc]
                ax.plot([x1, x2], [y1, y2], color="0.5", lw=0.8, zorder=4)
        span = min(xlim[1] - xlim[0], ylim[1] - ylim[0])
        for c in cluster_ids:
            cx, cy = centroids[c]
            lx, ly = cx, cy
            # compact/small clusters: push the label radially outward (past the cluster's own
            # radius) so it doesn't sit on top of the handful of points there
            if cradius[c] < 0.10 * span:
                vx, vy = cx - gcen[0], cy - gcen[1]
                nrm = np.hypot(vx, vy) or 1.0
                offd = cradius[c] + 0.045 * span
                lx, ly = cx + vx / nrm * offd, cy + vy / nrm * offd
            t = ax.text(lx, ly, f"C{c}", fontsize=13, fontweight="bold", ha="center", va="center",
                        color=CLUSTER_COLOURS.get(c, "k"), zorder=8)
            t.set_path_effects([pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()])
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_xticks([]); ax.set_yticks([])

    def _labels(ax):
        for _, r in prim.iterrows():
            t = ax.text(r["umap1"] + 0.12, r["umap2"] + 0.12, str(r["wise"]), fontsize=5,
                        color="0.15", zorder=7, ha="left", va="bottom")
            t.set_path_effects([pe.Stroke(linewidth=1.3, foreground="white"), pe.Normal()])

    # how many picks (primary + companion) land in each cluster, for the legend
    n_by_cluster = ids["cluster"].value_counts().to_dict()

    def _panelA(ax, legend_loc="lower left", labels_on=True):
        """picks coloured by QSO-UMAP cluster over the grey population (shared by the
        picked-projection figure and the standalone clusters figure)."""
        _base(ax)
        for c in cluster_ids:
            s = prim[prim.cluster == c]
            if len(s):
                ax.scatter(s["umap1"], s["umap2"], s=160, marker="*", c=CLUSTER_COLOURS.get(c, "k"),
                           edgecolors="k", linewidths=0.8, zorder=6)
            s = comp[comp.cluster == c]
            if len(s):
                ax.scatter(s["umap1"], s["umap2"], s=50, marker="D", c=CLUSTER_COLOURS.get(c, "k"),
                           edgecolors="k", linewidths=0.8, zorder=6)
        if labels_on:
            _labels(ax)
        handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=CLUSTER_COLOURS.get(c, "k"),
                          markeredgecolor="k", markersize=6,
                          label=f"Cluster {c}  ({int(n_by_cluster.get(c, 0))} picks)")
                   for c in cluster_ids]
        handles += [Line2D([0], [0], marker="*", color="w", markerfacecolor="0.4",
                           markeredgecolor="k", markersize=10, label="picked double (primary)"),
                    Line2D([0], [0], marker="D", color="w", markerfacecolor="0.4",
                           markeredgecolor="k", markersize=6, label="picked double (blend companion)"),
                    Line2D([0], [0], color="0.5", lw=1.8, label="same blend")]
        ax.legend(handles=handles, loc=legend_loc, fontsize=8, framealpha=0.92, ncol=2)
        ax.set_title(f"coloured by {sample.upper()} QSO-UMAP cluster", fontsize=13, fontweight="bold")

    # NB: the picked-projection figures (picked_qso_umap.png + picked_qso_umap_og.png, coloured by
    # cluster + colour(z) river deviation) were retired and archived along with
    # analyze_picked_color_vs_z.py; _base/_panelA below are still used by the clusters figure.

    # generate both feature grids fresh from the build npz (nothing is copied):
    #   population-only (like the builder's features grid) + picks over-plotted
    plot_features_grid(sample, q1_emb, pop_X_raw, OUT_DIR / f"{pfx}qso_gaia_features.png",
                       labels=labels)
    plot_features_grid(sample, q1_emb, pop_X_raw, OUT_DIR / f"{pfx}qso_gaia_features_picked.png",
                       ids=ids, bg_alpha=0.6, labels=labels)   # fainter backdrop so the picks stand out

    # NB: the Gaia-parameter distribution grids (features_hist / features_hist_by_cluster) now
    # live in analyze_picked_qso_dist.py — run that script to regenerate them.

    # Gaia G: UMAP coloured by G + G-mag distribution by HDBSCAN cluster
    plot_gmag(sample, q1_emb, pop_g_mag, labels, OUT_DIR / f"{pfx}qso_gaia_gmag.png")

    # NB: the population "other-features" quick-look (qso_other_features.png: spec-z | Euclid
    # Q1/ERO | params_solved | 2-param&z>0.7 density) now lives in analyze_picked_qso_other.py.

    # standalone clusters figure: panel 1 = population HDBSCAN clusters (CLUSTER_COLOURS),
    # panel 2 = picks coloured by QSO-UMAP cluster (== panel A of the picked projection)
    clusters_out = OUT_DIR / f"{pfx}qso_gaia_clusters.png"
    figC, (axC1, axC2) = plt.subplots(1, 2, figsize=(17, 8))
    cl_pop = sorted(set(int(c) for c in labels))
    for c in cl_pop:
        mm = labels == c
        col = "0.8" if c < 0 else CLUSTER_COLOURS.get(c, "k")
        axC1.scatter(q1_emb[mm, 0], q1_emb[mm, 1], s=4, color=col, alpha=0.6, lw=0,
                     rasterized=True, label=("noise" if c < 0 else f"C{c}"))
    axC1.set_title(f"all QSOs — HDBSCAN clusters ({len([c for c in cl_pop if c >= 0])})",
                   fontsize=13, fontweight="bold")
    axC1.legend(loc="upper right", fontsize=8, markerscale=2, framealpha=0.9, ncol=2)
    axC1.set_xlim(xlim); axC1.set_ylim(ylim)
    axC1.set_xticks([]); axC1.set_yticks([]); axC1.set_xlabel("UMAP-1"); axC1.set_ylabel("UMAP-2")
    _add_cluster_labels(axC1, q1_emb, labels, xlim, ylim)
    _panelA(axC2, legend_loc="upper right", labels_on=False)
    axC2.set_title(f"hand-picked candidate DOUBLES ({len(prim)}) — by QSO-UMAP cluster",
                   fontsize=13, fontweight="bold")
    figC.suptitle(f"{sample.upper()} QSO × Gaia UMAP —  LEFT: all QSOs  ·  "
                  f"RIGHT: the {len(prim)} hand-picked candidate doubles",
                  fontsize=16, fontweight="bold", y=0.96)
    figC.tight_layout(rect=[0, 0, 1, 0.95])
    clusters_out.parent.mkdir(parents=True, exist_ok=True)
    figC.savefig(clusters_out, dpi=150, bbox_inches="tight"); plt.close(figC)
    print(f"  -> {clusters_out}")

    # report companions that jump clusters vs their primary
    print("\n  multi_gaia components (primary cluster -> companion cluster):")
    pc = dict(zip(prim["gaia_source_id"], prim["cluster"]))
    cc = dict(zip(comp["gaia_source_id"], comp["cluster"]))
    for p, csrc in MULTI_GAIA_PAIRS.items():
        pcl = pc.get(p); ccl = cc.get(csrc)
        flag = "  <-- different cluster" if (pcl is not None and ccl is not None and pcl != ccl) else ""
        print(f"    {wise_of.get(p,''):<22} C{pcl} -> C{ccl}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
