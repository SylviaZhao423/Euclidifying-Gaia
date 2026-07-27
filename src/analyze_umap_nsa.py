#!/usr/bin/env python3
"""
Project the NSA-galaxy × Gaia central cross-match onto the Q1 14-feature UMAP
(galaxy-nucleation project). The QSO analogue is analyze_umap_qso.py — this mirrors
it exactly so the two projects share one workflow.

Reads the WSDB table built by the crossmatch script:
    {PGUSER}.crossmatch_nsa_v1_gaia   (v1_0_1, PRIMARY)
    PGUSER=siwei_zhao_2025 python src/crossmatch_nsa_v1_gaia.py    # build once (needs grant)

Science
-------
Each NSA galaxy is matched to the closest Gaia source within 1" of its
photometric centre. has_gaia flags a *central* detection = a nucleus candidate
(nuclear star cluster / AGN). The astrometric solution of that nucleus is
diagnostic, exactly like the QSO case: a genuine unresolved nucleus is often
2-parameter (position-only, astrometric_params_solved = 3) and lands in UMAP
Clusters 0/1 (the extragalactic locus), whereas a foreground interloper star is
5/6-parameter and lands in the stellar body (C3/C4). So projecting the detected
nuclei onto the Q1 UMAP separates true nuclei from interlopers.

Because the cross-match is a LEFT JOIN (every galaxy kept, for the nucleation-
fraction denominator), only the has_gaia subset is projectable — unmatched
galaxies have NULL Gaia features. The overlay therefore shows the detected
nuclei; the console report gives the nucleation fraction (detected / all).

Requires WSDB — Astronomy/Cambridge VPN first (plain eduroam blocks port 5432).
Auth via ~/.pgpass.

Reads:
  output/umap/arrays/q1_indiv.npz, q1_cluster_labels.npy, q1_indiv_reducer.joblib
  output/xgboost_q1/seed_models/xgb_classifier_q1_seed*.json   (model_pgal)

Writes (CSV in an arrays/ subfolder, like analyze_umap_qso.py; TWO overlay figures):
  output/umap/nsa/arrays/nsa_gaia_matches.csv   (all galaxies; umap/cluster set for nuclei)
  output/umap/nsa/umap_nsa_v1_clusters.png      (aps | cluster | streams)
  output/umap/nsa/umap_nsa_v1.png               (blend flag | XGB P(gal))

Usage:
  python src/analyze_umap_nsa.py                 # v1 (full: WSDB + project + plot)
  python src/analyze_umap_nsa.py --replot        # fast: redraw from the saved CSV
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS  # noqa: E402
# Reuse the QSO projection + XGB machinery verbatim (identical 14-feature pipeline).
from analyze_umap_qso import (project_and_assign, WSDB_HOST, WSDB_DB, WSDB_USER,  # noqa: E402
                            CLUSTER_COLOURS, APS_LABEL, aps_density_panel,
                            cluster_density_panel)
from umap_streams import outside_mask, stream_labels, STREAM_PAIRS  # noqa: E402

ARRAYS_DIR = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR = PROJECT_DIR / "output" / "umap" / "nsa"      # overlay PNGs
NSA_ARRAYS = OUTPUT_DIR / "arrays"                        # projected CSVs + NPZs

# NSA galaxy-property columns carried by the crossmatch tables (version-agnostic aliases).
GAL_PROPS = ["gal_log_mass", "gal_sersic_n", "gal_re_arcsec", "gal_ba", "gal_conc", "gal_zdist"]

CATALOG_TABLES = {"v1": "crossmatch_nsa_v1_gaia"}
CSV_SUFFIX = {"v1": ""}


def _csv_for(catalog: str) -> Path:
    return NSA_ARRAYS / f"nsa_gaia_matches{CSV_SUFFIX[catalog]}.csv"


def _read_one_table(user: str, table_name: str) -> pd.DataFrame | None:
    """Read one NSA crossmatch table, aliasing g_<feature> → plain feature names."""
    import psycopg2
    table = f"{user}.{table_name}"
    feat_cols = ", ".join(f"g_{c} AS {c}" for c in SELECTED_GAIA_PARAMETERS)
    props = ", ".join(GAL_PROPS)
    sql = f"""
        SELECT nsa_catalog, nsaid, iauname, gal_ra, gal_dec, {props},
               has_gaia, n_gaia_within_1arcsec, gaia_sep_arcsec,
               g_source_id                 AS gaia_source_id,
               g_astrometric_params_solved AS astrometric_params_solved,
               g_phot_g_mean_mag           AS phot_g_mean_mag,
               g_phot_g_n_obs              AS phot_g_n_obs,
               {feat_cols}
        FROM {table}
    """
    print(f"  reading {table} ...")
    try:
        conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                                connect_timeout=30, keepalives=1, keepalives_idle=30,
                                keepalives_interval=10, keepalives_count=5)
        df = pd.read_sql_query(sql, conn)
        conn.close()
    except Exception as exc:
        print(f"  skipped {table} ({exc}).")
        return None
    print(f"    {len(df):,} galaxies ({int(df['has_gaia'].sum()):,} with a central Gaia detection)")
    return df


def load_matches_from_table(catalog: str = "v1") -> pd.DataFrame | None:
    import os
    user = os.getenv("PGUSER", WSDB_USER)
    df = _read_one_table(user, CATALOG_TABLES[catalog])
    if df is None or not len(df):
        print("  No NSA crossmatch table found. Build it with crossmatch_nsa_v1_gaia.py "
              "(needs VPN + a GRANT on the NSA table).")
        return None
    return df


def project_nuclei(df: pd.DataFrame) -> pd.DataFrame:
    """Project only the Gaia-detected nuclei (has_gaia); merge umap/cluster/pgal
    back onto the full galaxy table (unmatched galaxies keep NaN)."""
    det = df[df["has_gaia"]].copy()
    if not len(det):
        print("  no central Gaia detections to project.");
        for c in ["umap1", "umap2", "cluster", "model_pgal"]:
            df[c] = np.nan
        return df
    det = project_and_assign(det)
    keys = det[["nsaid", "umap1", "umap2", "cluster", "model_pgal"]]
    return df.merge(keys, on="nsaid", how="left")


def report(df: pd.DataFrame) -> None:
    det = df[df["has_gaia"]]
    n, nd = len(df), len(det)
    print("\n" + "=" * 96)
    print("NSA nucleation → UMAP   (2-param nuclei ⇒ extragalactic C0/C1; interlopers ⇒ C3/C4)")
    print("=" * 96)
    print(f"Overall: {nd:,}/{n:,} galaxies have a central Gaia detection "
          f"({100*nd/max(n,1):.2f}% nucleation-detection fraction)")

    def _frac_table(sub, title):
        m = sub.dropna(subset=["cluster"])
        if not len(m):
            print(f"  {title:<32} (no projected nuclei)"); return
        vc = m["cluster"].value_counts()
        parts = "  ".join(f"C{c}={int(vc.get(c,0)):>4} ({100*vc.get(c,0)/len(m):4.1f}%)" for c in range(5))
        print(f"  {title:<32} N={len(m):>6}   {parts}")

    print("\n— detected nuclei, cluster membership —")
    _frac_table(det, "ALL detected nuclei")
    for tag, sub in det.groupby("nsa_catalog"):
        _frac_table(sub, tag)
    print("\n— by astrometric_params_solved —")
    for aps, sub in det.groupby("astrometric_params_solved"):
        _frac_table(sub, APS_LABEL.get(int(aps), f"aps={int(aps)}"))
    print("\n— nucleation-detection fraction vs stellar mass —")
    if df["gal_log_mass"].notna().any():
        bins = [7, 9, 9.5, 10, 10.5, 11, 13]
        df_m = df.dropna(subset=["gal_log_mass"])
        cats = pd.cut(df_m["gal_log_mass"], bins)
        for b, sub in df_m.groupby(cats, observed=True):
            f = 100 * sub["has_gaia"].mean()
            print(f"  logM {str(b):<16} N={len(sub):>6}   nucleation {f:4.1f}%")
    print("=" * 96)


def plot_overlay(df: pd.DataFrame, tag: str = "v1") -> None:
    """Two overlay figures (SAME format as the QSO ones), for the detected nuclei:
      umap_nsa_{tag}_clusters.png : astrometric_params_solved | assigned cluster | streams
      umap_nsa_{tag}.png          : blend flag | XGB P(gal)"""
    det = df.dropna(subset=["umap1", "umap2"])
    q1     = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    labels = np.load(ARRAYS_DIR / "q1_cluster_labels.npy")
    q1_emb, y = q1["embedding"], q1["y"]
    cluster_ids = sorted(set(labels[labels >= 0]))
    centroids = {c: (q1_emb[labels == c, 0].mean(), q1_emb[labels == c, 1].mean()) for c in cluster_ids}
    xlim = (q1_emb[:, 0].min() - 0.8, q1_emb[:, 0].max() + 0.8)
    ylim = (q1_emb[:, 1].min() - 0.8, q1_emb[:, 1].max() + 0.8)

    def _base(ax):
        ax.scatter(q1_emb[:, 0], q1_emb[:, 1], s=2, c="#e2e2e2", alpha=0.6,
                   linewidths=0, rasterized=True, zorder=1)
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_xticks([]); ax.set_yticks([])

    def _clusters(ax):   # second-row centroid labels: cluster id only
        for c in cluster_ids:
            cx, cy = centroids[c]
            t = ax.text(cx, cy, f"C{c}", fontsize=9, fontweight="bold",
                        ha="center", va="center", color=CLUSTER_COLOURS[c], zorder=7)
            t.set_path_effects([pe.Stroke(linewidth=2, foreground="white"), pe.Normal()])

    def _frac_box(ax, sub, extra=None):
        n = len(sub)
        lines = [f"N = {n:,}"]
        if n:
            vc = sub["cluster"].value_counts()
            for c in sorted(cluster_ids, key=lambda k: -vc.get(k, 0)):
                p = 100 * vc.get(c, 0) / n
                if p >= 0.5:
                    lines.append(f"C{c}: {p:.0f}%")
        if extra:
            lines.append(extra)
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=9,
                va="top", ha="left", zorder=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.92))

    cats = "+".join(sorted(det["nsa_catalog"].astype(str).unique())) if len(det) else tag
    nd, n = len(det), len(df)
    suptitle = (f"{nd:,} central Gaia nuclei ({cats}) of {n:,} NSA galaxies "
                f"projected onto the Q1 UMAP")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ===================== Figure 1: umap_nsa_{tag}_clusters.png =====================
    # Row-1 panels: astrometric_params_solved | assigned UMAP cluster | inter-cluster streams
    fig = plt.figure(figsize=(19.5, 7))
    gs = fig.add_gridspec(1, 3, left=0.03, right=0.965, top=0.86, bottom=0.05, wspace=0.17)
    aax = fig.add_subplot(gs[0, 0]); kax = fig.add_subplot(gs[0, 1]); sax = fig.add_subplot(gs[0, 2])

    # ── all nuclei by astrometric_params_solved — per-class 2D density (denser = darker) ──
    ax = aax; _base(ax)
    aps_density_panel(fig, ax, det, xlim, ylim, noun="nuclei")
    _clusters(ax)

    # ── nuclei by assigned UMAP cluster — per-cluster 2D density (denser = darker) ──
    ax = kax; _base(ax)
    cluster_density_panel(fig, ax, det, xlim, ylim, cluster_ids, noun="nuclei")
    _clusters(ax); _frac_box(ax, det)

    # ── OUTSIDE nuclei relabelled into named inter-cluster streams ──
    ax = sax; _base(ax)
    pts = det[["umap1", "umap2"]].to_numpy()
    outside = outside_mask(pts, q1_emb)
    slab = stream_labels(pts, outside, centroids)
    ax.scatter(pts[~outside, 0], pts[~outside, 1], s=2, c="#d8d8d8", alpha=0.2,
               linewidths=0, rasterized=True, zorder=2)      # core nuclei, faded
    for k, (name, _verts, col) in enumerate(STREAM_PAIRS):   # named streams
        m = slab == k
        ax.scatter(pts[m, 0], pts[m, 1], s=6, color=col, alpha=0.85, linewidths=0,
                   rasterized=True, zorder=5, label=f"{name}: {int(m.sum()):,}")
    mo = slab == -1                                          # outside but unassigned
    ax.scatter(pts[mo, 0], pts[mo, 1], s=3, c="#9a9a9a", alpha=0.4, linewidths=0,
               rasterized=True, zorder=4, label=f"other: {int(mo.sum()):,}")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title(f"inter-cluster streams  ({int(outside.sum()):,} outside Q1 cores)",
                 fontsize=12, fontweight="bold")
    _clusters(ax)

    fig.suptitle(suptitle, fontsize=16, fontweight="bold", x=0.5, y=0.97)
    out1 = OUTPUT_DIR / f"umap_nsa_{tag}_clusters.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight"); plt.close(); print(f"  -> {out1}")

    # ===================== Figure 2: umap_nsa_{tag}.png =====================
    # Row-2 panels, swapped order: blend flag | XGB P(gal)
    fig = plt.figure(figsize=(13, 6.8))
    gs = fig.add_gridspec(1, 2, left=0.04, right=0.955, top=0.86, bottom=0.05, wspace=0.17)
    bax = fig.add_subplot(gs[0, 0]); pax = fig.add_subplot(gs[0, 1])

    # ── blend flag (# Gaia detections within 1" of the galaxy centre) ──
    # background = 2D density (log) of the detected nuclei; blends starred on top
    ax = bax; _base(ax)
    multi = det[det["n_gaia_within_1arcsec"] > 1]
    hb = ax.hexbin(det["umap1"], det["umap2"], gridsize=120, cmap="viridis", bins="log", alpha=0.6,
                   mincnt=1, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], zorder=2)
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("nucleus count (log)", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.scatter(multi["umap1"], multi["umap2"], s=14, c="red", marker="*", alpha=0.7,
               edgecolors="w", linewidths=0.4, zorder=5,
               label=f"blend: >1 Gaia within 1\" ({len(multi):,})")
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.92)   # lower-left: clears the frac box
    ax.set_title("blend flag (# Gaia detections within 1\")", fontsize=12, fontweight="bold")
    _clusters(ax)
    _frac_box(ax, det)   # N + per-cluster fractions (same box as the cluster panel)

    # ── XGB P(gal) ──
    ax = pax; _base(ax)
    sc = ax.scatter(det["umap1"], det["umap2"], s=6, c=det["model_pgal"], cmap="RdYlBu_r",
                    vmin=0, vmax=1, alpha=0.6, linewidths=0, rasterized=True, zorder=2)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("XGB P(gal)", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_title("XGB P(gal)", fontsize=13, fontweight="bold"); _clusters(ax)
    lines = ["nuclei with P(gal) ≥ 0.5:"]
    for c in cluster_ids:
        sub = det[det["cluster"] == c]
        frac = 100 * (sub["model_pgal"] >= 0.5).mean() if len(sub) else 0.0
        lines.append(f"  C{c}: {frac:>3.0f}%   (N={len(sub):,})")
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=8.5, va="top",
            ha="left", zorder=8, bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.92))

    fig.suptitle(suptitle, fontsize=16, fontweight="bold", x=0.5, y=0.97)
    out2 = OUTPUT_DIR / f"umap_nsa_{tag}.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(); print(f"  -> {out2}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Project NSA nuclei × Gaia onto the UMAP")
    ap.add_argument("--catalog", choices=["v1"], default="v1",
                    help="NSA v1 nuclei × Gaia (the only catalog kept).")
    ap.add_argument("--replot", action="store_true",
                    help="Skip WSDB + projection; redraw the umap_nsa_v1 figures from the saved CSV.")
    args = ap.parse_args()

    if args.replot:
        csv = _csv_for(args.catalog)
        if not csv.exists():
            print(f"{csv} not found — run a full projection first (no --replot)."); return 1
        df = pd.read_csv(csv)
        print(f"Replotting {args.catalog} ({len(df):,} galaxies)...")
        plot_overlay(df, args.catalog)
        return 0

    print(f"Loading NSA-Gaia matches ({args.catalog}) from the crossmatch table...")
    df = load_matches_from_table(args.catalog)
    if df is None or not len(df):
        print("No NSA matches loaded. Build the table first (needs a GRANT on the NSA table): "
              "crossmatch_nsa_v1_gaia.py.")
        return 1

    print("\nProjecting the detected nuclei onto the Q1 UMAP...")
    df = project_nuclei(df)
    report(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NSA_ARRAYS.mkdir(parents=True, exist_ok=True)
    csv = _csv_for(args.catalog)
    df.to_csv(csv, index=False)
    print(f"\n  -> {csv}")
    plot_overlay(df, args.catalog)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
