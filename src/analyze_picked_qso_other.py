#!/usr/bin/env python3
"""Population "other-features" quick-look for the combined QSO×Gaia UMAP — the qso_other_features.png
figure, split out of analyze_picked_qso_umap.py (same spirit as analyze_picked_qso_dist.py holding
the distribution grids).

6-panel (2 rows × 3), all over the build UMAP:
  row 1: spectroscopic z | Gaia astrometric_params_solved | 2D density of ALL 2-parameter
         (no parallax/PM) sources with z > 0.7
  row 2: Euclid source (Q1 vs ERO; Euclid-matched coloured, rest grey) | Euclid sources passing
         the 2-param & z>0.7 cut | 2D density of those passing Euclid sources

Reads:
  output/qso_umap/base/arrays/combined_qso_gaia.npz   (embedding, cluster, qso_z, source_ids;
      desi/sdss come from ds_qso/n{N}/arrays — see build_arrays_dir())
  siwei_zhao_2025.crossmatch_qso_*_euclid_gaia   (Euclid Q1/ERO per gaia source)
  siwei_zhao_2025.crossmatch_qso_*_gaia          (astrometric_params_solved per gaia source)
Writes (output/picked_qso/):  qso_other_features.png  (`{sample}_` prefix for non-combined samples)

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_picked_qso_other.py
  PGUSER=siwei_zhao_2025 python src/analyze_picked_qso_other.py --sample desi --n-sample all
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psycopg2

from analyze_picked_qso_umap import (_add_cluster_labels, OUT_DIR, n_suffix, parse_n,
                                     build_arrays_dir, EUCLID_XM, GAIA_XM)
from builder_qso_gaia_crossmatch import WSDB_HOST, WSDB_DB


def fetch_euclid_source(user):
    """g_source_id -> 'Q1'/'ERO' for EVERY Euclid∩Gaia matched QSO (both survey types). Keyed by
    the GAIA source_id (each UMAP point is a Gaia source), not qso_id: a Gaia source can carry a
    different qso_id in the build than in the Euclid×Gaia join (n_qso_per_gaia > 1), so qso_id
    keying silently drops those matches. Q1 takes precedence if a source has both."""
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30)
    frames = [pd.read_sql_query(
        f"SELECT g_source_id::text AS gsid, euclid_source FROM {user}.{T} WHERE e_id IS NOT NULL",
        conn) for T in EUCLID_XM]
    conn.close()
    df = pd.concat(frames, ignore_index=True)
    # Q1 before ERO so drop_duplicates keeps Q1 when a source appears as both
    df["_r"] = (df["euclid_source"] != "Q1").astype(int)
    df = df.sort_values("_r").drop_duplicates("gsid")
    print(f"  Euclid source per gaia source: {(df['euclid_source']=='Q1').sum():,} Q1, "
          f"{(df['euclid_source']=='ERO').sum():,} ERO")
    return dict(zip(df["gsid"], df["euclid_source"]))


def fetch_params_solved(user, source_ids):
    """Gaia astrometric_params_solved per UMAP source (g_source_id), from the ds_qso gaia
    crossmatch tables. 3 = 2-param (no parallax/PM), 31 = 5-param, 95 = 6-param."""
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30)
    frames = [pd.read_sql_query(
        f"SELECT g_source_id, g_astrometric_params_solved AS aps FROM {user}.{T}", conn)
        for T in GAIA_XM]
    conn.close()
    m = pd.concat(frames, ignore_index=True).drop_duplicates("g_source_id")
    aps_of = dict(zip(m["g_source_id"].astype(str), m["aps"]))
    out = np.array([aps_of.get(str(s), -1) for s in source_ids], float)
    for v in (3, 31, 95):
        print(f"  astrometric_params_solved == {v}: {int((out == v).sum()):,}")
    return out


def plot_umap_quicklook(sample, emb, qso_z, src_ids, src_of, aps, out, labels=None):
    """6-panel population quick-look (2 rows × 3):
      row 1: spectroscopic z | Gaia astrometric_params_solved | 2D density of ALL 2-parameter
             (no parallax/PM) sources with z > 0.7
      row 2: Euclid source (Q1 vs ERO) | Euclid sources passing the 2-param & z>0.7 cut |
             2D density of those passing Euclid sources
    The Euclid panels colour only the Euclid-matched subset (rest grey; keyed by GAIA source_id)."""
    from matplotlib.lines import Line2D
    from matplotlib.colors import LogNorm
    pop_src = np.array([src_of.get(str(s), "") for s in src_ids])
    m_q1 = pop_src == "Q1"; m_ero = pop_src == "ERO"; m_eucl = m_q1 | m_ero
    zf = np.asarray(qso_z, float)
    cut = (aps == 3) & (zf > 0.7)                 # 2-param (no parallax/PM) & z > 0.7
    xlim = (emb[:, 0].min() - 0.8, emb[:, 0].max() + 0.8)
    ylim = (emb[:, 1].min() - 0.8, emb[:, 1].max() + 0.8)
    fig, axes = plt.subplots(2, 3, figsize=(21, 14.4))

    def _density(ax, mask, title, nb=250):
        ax.scatter(emb[:, 0], emb[:, 1], s=3, c="#e2e2e2", alpha=0.4, lw=0, rasterized=True)
        if mask.any():
            xe = np.linspace(xlim[0], xlim[1], nb + 1); ye = np.linspace(ylim[0], ylim[1], nb + 1)
            H, _, _ = np.histogram2d(emb[mask, 0], emb[mask, 1], bins=[xe, ye])
            H = np.where(H > 0, H, np.nan)
            pm = ax.pcolormesh(xe, ye, H.T, cmap="Blues", norm=LogNorm(vmin=1, vmax=np.nanmax(H)),
                               rasterized=True, alpha=0.85)
            fig.colorbar(pm, ax=ax, fraction=0.046, pad=0.02).set_label("count (log)", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")

    # ── row 1, panel 1: spectroscopic z ──
    z = np.clip(zf, 0, 4)
    s0 = axes[0, 0].scatter(emb[:, 0], emb[:, 1], c=z, s=4, cmap="RdYlBu_r", vmin=0, vmax=4,
                            alpha=0.6, lw=0, rasterized=True)
    axes[0, 0].set_title("coloured by spectroscopic $z$", fontsize=13, fontweight="bold")
    fig.colorbar(s0, ax=axes[0, 0], fraction=0.046, pad=0.02).set_label("QSO redshift", fontsize=10)

    # ── row 1, panel 2: Gaia astrometric_params_solved ──
    axG = axes[0, 1]
    APS_C = {3: "#e6194b", 31: "#4363d8", 95: "#3cb44b"}
    APS_L = {3: "2-param (no parallax/PM)", 31: "5-param", 95: "6-param"}
    for v in (31, 95, 3):   # 2-param drawn last so it sits on top
        mm = aps == v
        if mm.any():
            axG.scatter(emb[mm, 0], emb[mm, 1], s=3, c=APS_C[v], alpha=0.5, lw=0,
                        rasterized=True, label=f"{APS_L[v]}  ({int(mm.sum()):,})")
    axG.set_title("Gaia astrometric_params_solved", fontsize=12, fontweight="bold")
    axG.legend(loc="best", fontsize=8, markerscale=2.5, framealpha=0.9)

    # ── row 1, panel 3: 2D density of ALL 2-param & z>0.7 sources ──
    _density(axes[0, 2], cut,
             f"2-param (no parallax/PM) & $z$ > 0.7\n2D density  ({int(cut.sum()):,} sources)")

    # ── row 2, panel 1: Euclid survey of the match — Q1 vs ERO ──
    axS = axes[1, 0]
    axS.scatter(emb[:, 0], emb[:, 1], s=3, c="#e2e2e2", alpha=0.5, lw=0, rasterized=True)
    axS.scatter(emb[m_q1, 0], emb[m_q1, 1], s=10, c="#1f77b4", alpha=0.85, lw=0.2,
                edgecolors="k", rasterized=True)
    axS.scatter(emb[m_ero, 0], emb[m_ero, 1], s=20, marker="D", c="#d62728", alpha=0.85, lw=0.2,
                edgecolors="k", rasterized=True)   # ERO few → bigger diamonds so they show
    axS.set_title(f"Euclid source: Q1 vs ERO\n"
                  f"({int(m_eucl.sum()):,} Euclid-matched; rest grey)", fontsize=12, fontweight="bold")
    axS.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markeredgecolor="k",
               markersize=6, label=f"Q1  ({int(m_q1.sum()):,})"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#d62728", markeredgecolor="k",
               markersize=4, label=f"ERO  ({int(m_ero.sum()):,})")],
        loc="best", fontsize=8, framealpha=0.9)

    # ── row 2, panel 2: Euclid sources that PASS the 2-param & z>0.7 cut ──
    axP = axes[1, 1]
    pass_q1 = m_q1 & cut; pass_ero = m_ero & cut; npass = int((pass_q1 | pass_ero).sum())
    axP.scatter(emb[:, 0], emb[:, 1], s=3, c="#e2e2e2", alpha=0.5, lw=0, rasterized=True)
    axP.scatter(emb[m_eucl, 0], emb[m_eucl, 1], s=8, c="#b8d0e6", alpha=0.5, lw=0,
                rasterized=True)                                   # all Euclid-matched, faint
    axP.scatter(emb[pass_q1, 0], emb[pass_q1, 1], s=10, marker="o", c="#1f77b4", alpha=0.85, 
                edgecolors="k", lw=0.2, zorder=6)                  # passing Q1
    axP.scatter(emb[pass_ero, 0], emb[pass_ero, 1], s=20, marker="D", c="#d62728", alpha=0.85,
                edgecolors="k", lw=0.2, zorder=6)                  # passing ERO
    axP.set_title(f"Euclid sources passing 2-param & $z$>0.7\n"
                  f"({npass} of {int(m_eucl.sum()):,} Euclid-matched)", fontsize=12, fontweight="bold")
    axP.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markeredgecolor="k",
               markersize=6, label=f"Q1 pass ({int(pass_q1.sum())})"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#d62728", markeredgecolor="k",
               markersize=4, label=f"ERO pass ({int(pass_ero.sum())})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#b8d0e6", markeredgecolor="none",
               markersize=5, label=f"other Euclid ({int(m_eucl.sum()) - npass})")],
        loc="best", fontsize=8, framealpha=0.9)

    # ── row 2, panel 3: 2D density of the passing Euclid sources ──
    _density(axes[1, 2], m_eucl & cut,
             f"Euclid & 2-param & $z$>0.7\n2D density  ({int((m_eucl & cut).sum())} sources)")

    for ax in axes.ravel():
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        _add_cluster_labels(ax, emb, labels, xlim, ylim, fontsize=11)
    fig.suptitle(f"{sample.upper()} QSO × Gaia — 14-feature UMAP  (N={len(emb):,}, G≥17)",
                 fontsize=15, fontweight="bold", y=0.96)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")
    print(f"  Euclid sources passing 2-param & z>0.7: {npass} "
          f"(Q1 {int(pass_q1.sum())}, ERO {int(pass_ero.sum())})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", choices=["combined", "desi", "sdss"], default="combined",
                    help="which QSO×Gaia UMAP build to use (default combined).")
    ap.add_argument("--n-sample", type=parse_n, default="all",
                    help="build size for the target npz (default 'all').")
    args = ap.parse_args()
    user = os.environ.get("PGUSER", "siwei_zhao_2025")

    adir = build_arrays_dir(args.sample, args.n_sample)
    npz_path = adir / f"{args.sample}_qso_gaia.npz"
    if not npz_path.exists():
        print(f"  missing {n_suffix(args.n_sample)} arrays for {args.sample} — build it first "
              f"(query_qso_umap_ds.py).")
        return 1
    d = np.load(npz_path, allow_pickle=True)
    pfx = "" if args.sample == "combined" else f"{args.sample}_"
    src_ids = d["source_ids"].astype(str)

    src_of = fetch_euclid_source(user)
    aps = fetch_params_solved(user, src_ids)
    plot_umap_quicklook(args.sample, d["embedding"], d["qso_z"], src_ids, src_of, aps,
                        OUT_DIR / f"{pfx}qso_other_features.png", labels=d["cluster"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
