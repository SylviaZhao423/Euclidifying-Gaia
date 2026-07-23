#!/usr/bin/env python3
"""
Euclid VIS cutouts for the SDSS QSOs that fall in the Euclid footprint.

The QSO×Gaia matches (crossmatch_qso_sdss_dr16_gaia) were joined to the existing
Euclid cross-match tables by shared Gaia source_id:
    Euclid Q1  : crossmatch_q1_mer_phz_vis_gaia_sep   → 4 QSOs
    Euclid ERO : crossmatch_ero_gaia_0_3              → 9 QSOs
giving 13 QSOs with Euclid imaging (the match list is folded into manifest.csv,
built by the WSDB join).

Cutout sources (reusing the existing cutout machinery):
    Q1  → save_q1_utils: IRSA VIS MER mosaic tiles fetched via HTTP range requests
    ERO → download_e_desi_cutouts helpers: cached ERO VIS Flattened FITS
          (output/euclid_ero_fits_cache/; run save_ero_fits.py if a field is
          missing)

If manifest.csv is missing the matches are queried from WSDB (needs VPN + PGUSER);
rendering the cutouts themselves needs only internet (IRSA/ESA CDN).

Output (output/qso_cutouts/euclid_desi+sdss/):
    cutouts/cutout_NNNN.png               individual VIS+NISP stamps (cluster, then P(gal) order)
    manifest.csv                          ONE comprehensive file: cutout_file + full match row
                                          (ids, coords, cluster, aps, P(gal), G, z, Q1/ERO);
                                          also serves as the WSDB-match cache
    qso_euclid_contact_sheet.png          contact sheet, rows = UMAP cluster, sorted by P(gal),
                                          border = XGB P(gal)

Usage:
    PGUSER=siwei_zhao_2025 python src/download_euclid_ds_cutouts.py   # builds matches if missing
    python src/download_euclid_ds_cutouts.py --size 6
    PGUSER=siwei_zhao_2025 python src/download_euclid_ds_cutouts.py --rebuild-matches
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

OUT_DIR = PROJECT_DIR / "output" / "qso_cutouts" / "euclid_desi+sdss"
MANIFEST = OUT_DIR / "manifest.csv"   # single comprehensive file (also the WSDB-match cache)

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"


def build_matches() -> "pd.DataFrame | None":
    """Query the QSO×Euclid matches: join {schema}.crossmatch_qso_sdss_dr16_gaia to
    the Euclid Q1/ERO cross-match tables by Gaia source_id, then merge cluster /
    model_pgal / G-mag from output/umap/qso/arrays/qso_gaia_matches_desisdss.csv. Needs WSDB (VPN);
    PGUSER selects the schema. Returns the df (folded into manifest.csv) or None."""
    import os
    import psycopg2
    user = os.getenv("PGUSER", "siwei_zhao_2025")
    sql = f"""
        SELECT 'Q1'::text AS euclid, q.qso_catalog, q.qso_id::text AS qso_id,
               q.g_source_id::text AS gaia_source_id, q.g_ra, q.g_dec,
               q.g_astrometric_params_solved AS aps, q.qso_z
        FROM {user}.crossmatch_qso_sdss_dr16_gaia q
        JOIN {user}.crossmatch_q1_mer_phz_vis_gaia_sep e ON e.g_source_id = q.g_source_id
        UNION ALL
        SELECT 'ERO'::text, q.qso_catalog, q.qso_id::text, q.g_source_id::text, q.g_ra, q.g_dec,
               q.g_astrometric_params_solved, q.qso_z
        FROM {user}.crossmatch_qso_sdss_dr16_gaia q
        JOIN {user}.crossmatch_ero_gaia_0_3 e ON e.g_source_id = q.g_source_id
    """
    print(f"  [matches] WSDB join ({user}.crossmatch_qso_sdss_dr16_gaia ⋈ Euclid Q1/ERO)...")
    try:
        conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                                connect_timeout=30, keepalives=1)
        df = pd.read_sql_query(sql, conn)
        conn.close()
    except Exception as exc:
        print(f"  [matches] WSDB join failed ({exc}).")
        print("  Needs the Astronomy/Cambridge VPN + the Euclid crossmatch tables built.")
        return None
    proj = PROJECT_DIR / "output" / "umap" / "qso" / "arrays" / "qso_gaia_matches_desisdss.csv"
    if proj.exists():
        p = pd.read_csv(proj, usecols=["gaia_source_id", "cluster", "model_pgal", "phot_g_mean_mag"])
        p["gaia_source_id"] = p["gaia_source_id"].astype("int64").astype(str)
        df = df.merge(p, on="gaia_source_id", how="left")
    print(f"  [matches] {len(df)} QSO×Euclid matches from WSDB")
    return df

# UMAP cluster colours — match umap_qso_overlay.png (analyze_umap_qso.CLUSTER_COLOURS)
CLUSTER_COLOURS = {0: "#e6194b", 1: "#4363d8", 2: "#f58231", 3: "#3cb44b", 4: "#911eb4"}
APS_LABEL = {3: "2-param", 31: "5-param", 95: "6-param"}
# stamp-border colour by astrometric_params_solved (viridis, matching the params-solved plots)
_CMAP3 = plt.get_cmap("viridis", 3)
APS_COLOURS = {3: _CMAP3(0), 31: _CMAP3(1), 95: _CMAP3(2)}


def plot_umap_positions(df: pd.DataFrame) -> None:
    """Mini UMAP plot: where the Euclid-imaged QSOs sit, coloured by
    astrometric_params_solved (2/5/6). n_gaia_within_1arcsec is reported in the
    title — for these all-in-footprint QSOs it is uniformly 1 (no blends)."""
    import matplotlib as mpl
    import matplotlib.patheffects as pe
    arrays = PROJECT_DIR / "output" / "umap" / "arrays"
    proj_csv = PROJECT_DIR / "output" / "umap" / "qso" / "arrays" / "qso_gaia_matches_desisdss.csv"
    if not (arrays / "q1_indiv.npz").exists() or not proj_csv.exists():
        print("  [umap] missing q1_indiv.npz or qso_gaia_matches_desisdss.csv — skipping position plot")
        return
    proj = pd.read_csv(proj_csv, usecols=["gaia_source_id", "umap1", "umap2", "n_gaia_within_1arcsec"])
    proj["gaia_source_id"] = proj["gaia_source_id"].astype("int64").astype(str)
    m = df.merge(proj, on="gaia_source_id", how="left").dropna(subset=["umap1", "umap2"])
    if m.empty:
        print("  [umap] no matched UMAP coords — skipping"); return
    n_blend = int((m["n_gaia_within_1arcsec"] > 1).sum())

    q1 = np.load(arrays / "q1_indiv.npz", allow_pickle=True)
    q1_emb, y = q1["embedding"], q1["y"]
    labels = np.load(arrays / "q1_cluster_labels.npy")
    cluster_ids = sorted(set(labels[labels >= 0]))
    centroids = {c: (q1_emb[labels == c, 0].mean(), q1_emb[labels == c, 1].mean()) for c in cluster_ids}
    galpct = {c: 100 * (y[labels == c] == 1).sum() / (labels == c).sum() for c in cluster_ids}

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.scatter(q1_emb[:, 0], q1_emb[:, 1], s=3, c="#dddddd", alpha=0.4,
               linewidths=0, rasterized=True, zorder=1)
    allq = pd.read_csv(proj_csv, usecols=["umap1", "umap2"])
    ax.scatter(allq["umap1"], allq["umap2"], s=2, c="#9ecae1", alpha=0.15,
               linewidths=0, rasterized=True, zorder=2, label=f"all SDSS QSOs ({len(allq):,})")
    cmap3 = plt.get_cmap("viridis", 3)   # 3→2-param, 31→5-param, 95→6-param
    idx = m["aps"].map({3: 0, 31: 1, 95: 2})
    ax.scatter(m["umap1"], m["umap2"], c=idx, cmap=cmap3, vmin=-0.5, vmax=2.5,
               s=260, marker="*", edgecolors="k", linewidths=0.7, zorder=5,
               label=f"Euclid-imaged QSOs ({len(m)})")
    sm = mpl.cm.ScalarMappable(cmap=cmap3, norm=mpl.colors.BoundaryNorm([0, 1, 2, 3], 3))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, ticks=[0.5, 1.5, 2.5])
    cb.set_ticklabels(["2", "5", "6"]); cb.set_label("# astrometric params solved", fontsize=10)
    for c in cluster_ids:
        cx, cy = centroids[c]
        t = ax.text(cx, cy, f"C{c}\n{galpct[c]:.0f}%gal", fontsize=9, fontweight="bold",
                    ha="center", va="center", color=CLUSTER_COLOURS[c], zorder=6)
        t.set_path_effects([pe.Stroke(linewidth=2, foreground="white"), pe.Normal()])
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title(f"Where the {len(m)} Euclid-imaged QSOs sit in the UMAP  "
                 f"(colour = params solved)\nall single Gaia matches — {n_blend} blends "
                 "(>1 within 1\")", fontsize=12, fontweight="bold")
    out = OUT_DIR / "qso_euclid_umap_positions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def _title(r: pd.Series) -> str:
    """Standard stamp title: classification/cluster, coords, G, P(gal), Gaia id."""
    aps = int(r["aps"]) if pd.notna(r["aps"]) else 0
    g = r.get("phot_g_mean_mag", np.nan)
    gtxt = f"{g:.2f}" if pd.notna(g) else "?"
    pg = r.get("model_pgal", np.nan)
    pgtxt = f"{pg:.2f}" if pd.notna(pg) else "?"
    return (f"QSO  {r['euclid']}  Cluster {int(r['cluster'])}  ({APS_LABEL.get(aps, '?')})\n"
            f"RA={float(r['g_ra']):.4f}  Dec={float(r['g_dec']):+.4f}  P(gal)={pgtxt}\n"
            f"G={gtxt}  id={r['gaia_source_id']}")


def render_stamps(df: pd.DataFrame, size_arcsec: float) -> dict:
    """Render each QSO to a standard stamp (P(gal) crosshair, scale bar,
    coord/G/id title) with an RGB VIS+NISP composite where the source is bright
    enough (grayscale fallback otherwise). Q1 → IRSA VIS+NISP tile URLs; ERO →
    cached VIS + NISP-J/H FITS. Returns {qso_id: (image, aps)}."""
    from save_q1_utils import (_discover_tiles, _assign_to_tiles, _render_cutout,
                               _nisp_filename, IRSA_IBE_DATA)
    from download_e_desi_cutouts import (nearest_field, get_cached_fits,
                                         get_nisp_fits, make_cutout_image)
    stamps = {}

    q1 = df[df["euclid"] == "Q1"]
    if len(q1):
        print(f"  [Q1] discovering IRSA VIS/NISP tiles for {len(q1)} sources...")
        tiles = _discover_tiles(q1["g_ra"].min(), q1["g_ra"].max(),
                                q1["g_dec"].min(), q1["g_dec"].max(), field_name="qso")
        print(f"  [Q1] {len(tiles)} tiles found")
        for _, r in q1.iterrows():
            try:
                ti = _assign_to_tiles(float(r["g_ra"]), float(r["g_dec"]), tiles, "qso")
                if ti is None:
                    print(f"    {r['qso_id']}: no tile covers it — skip"); continue
                njf = _nisp_filename(ti.tile_id, "J")
                nhf = _nisp_filename(ti.tile_id, "H")
                nju = f"{IRSA_IBE_DATA}/{ti.tile_id}/NISP/{njf}" if njf else None
                nhu = f"{IRSA_IBE_DATA}/{ti.tile_id}/NISP/{nhf}" if nhf else None
                img = _render_cutout(ti.url, float(r["g_ra"]), float(r["g_dec"]),
                                     size_arcsec, _title(r), c_star=float(r["model_pgal"]),
                                     nisp_j_url=nju, nisp_h_url=nhu, draw_crosshair=False)
                stamps[str(r["qso_id"])] = (img, int(r["aps"]), float(r["model_pgal"]), int(r["cluster"]))
                print(f"    {r['qso_id']}: OK  (RGB={'yes' if nju and nhu else 'no'})")
            except Exception as exc:
                print(f"    {r['qso_id']}: FAILED ({exc})")

    ero = df[df["euclid"] == "ERO"]
    for _, r in ero.iterrows():
        try:
            field = nearest_field(float(r["g_ra"]), float(r["g_dec"]))
            vis = get_cached_fits(field)
            if vis is None:
                print(f"    {r['qso_id']}: {field} VIS FITS not cached — run save_ero_fits.py"); continue
            nj = get_nisp_fits(field, "J")
            nh = get_nisp_fits(field, "H")
            img = make_cutout_image(vis, float(r["g_ra"]), float(r["g_dec"]),
                                    size_arcsec, _title(r), c_star=float(r["model_pgal"]),
                                    fits_path_nisp_j=nj, fits_path_nisp_h=nh, draw_crosshair=False)
            stamps[str(r["qso_id"])] = (img, int(r["aps"]), float(r["model_pgal"]), int(r["cluster"]))
            print(f"    {r['qso_id']}: OK  field={field}  (RGB={'yes' if nj and nh else 'no'})")
        except Exception as exc:
            print(f"    {r['qso_id']}: FAILED ({exc})")
    return stamps


def _pgal_color(p):
    """Outer-border colour by XGB P(gal) (RdYlBu_r: red=galaxy, blue=star)."""
    return plt.cm.RdYlBu_r(float(np.clip(p, 0, 1))) if np.isfinite(p) else "0.5"


def contact_sheet(stamps: dict, size_arcsec: float) -> None:
    """Grid grouped into rows by UMAP cluster (C0 row(s), then C1, C2, ...); WITHIN
    each cluster ordered by descending XGB P(gal). Outer box coloured by P(gal)
    (red=galaxy, blue=star); no crosshair."""
    n = len(stamps)
    if not n:
        print("  no stamps to sheet."); return
    # group by UMAP cluster, sort each group by descending P(gal)
    by_cl: dict = {}
    for (img, _ap, pg, cl) in stamps.values():
        by_cl.setdefault(cl, []).append((img, pg))
    for cl in by_cl:
        by_cl[cl].sort(key=lambda t: -t[1])
    present = sorted(by_cl)
    ncols = 5
    nrows = sum(int(np.ceil(len(by_cl[cl]) / ncols)) for cl in present)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    row = 0
    for cl in present:
        imgs = by_cl[cl]
        for j, (img, pg) in enumerate(imgs):
            ax = axes[row + j // ncols][j % ncols]
            ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(img)
            for sp in ax.spines.values():
                sp.set_edgecolor(_pgal_color(pg)); sp.set_linewidth(2.6)
            if j == 0:   # label the cluster row at its first stamp
                ax.set_ylabel(f"Cluster {cl}", fontsize=14, fontweight="bold",
                              color=CLUSTER_COLOURS.get(cl, "#222"))
        row += int(np.ceil(len(imgs) / ncols))

    fig.subplots_adjust(top=0.93, bottom=0.02, left=0.04, right=0.99, hspace=0.05, wspace=0.05)
    # P(gal) border-colour legend (top-left)
    cbar_ax = fig.add_axes([0.10, 0.955, 0.26, 0.012])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=plt.Normalize(0, 1)),
                        cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 0.5, 1]); cbar.set_ticklabels(["star", "0.5", "galaxy"])
    cbar.ax.tick_params(labelsize=7, pad=1)
    cbar_ax.set_title("border = XGBoost P(gal)", fontsize=8, pad=3, loc="left")
    fig.suptitle(f"SDSS+DESI QSOs with Euclid VIS+NISP imaging ({n}) — {size_arcsec:.0f}\" stamps  "
                 f"(rows = UMAP cluster, sorted by P(gal))", fontsize=13, fontweight="bold", x=0.5, y=0.99)
    out = OUT_DIR / "qso_euclid_contact_sheet.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Euclid VIS cutouts for QSOs in the Euclid footprint")
    ap.add_argument("--size", type=float, default=3.0, help="stamp size in arcsec (default 3)")
    ap.add_argument("--rebuild-matches", action="store_true",
                    help="Force re-querying the QSO×Euclid matches from WSDB (needs VPN).")
    args = ap.parse_args()

    # manifest.csv doubles as the WSDB-match cache: read it if present, else query WSDB
    if MANIFEST.exists() and not args.rebuild_matches:
        df = pd.read_csv(MANIFEST, dtype={"qso_id": str, "gaia_source_id": str})
    else:
        print("manifest.csv not found — building matches from WSDB (needs VPN)...")
        df = build_matches()
        if df is None:
            return 1
        df["qso_id"] = df["qso_id"].astype(str)
        df["gaia_source_id"] = df["gaia_source_id"].astype(str)
    # merge G-mag from the projected QSO matches (for the stamp titles)
    proj_csv = PROJECT_DIR / "output" / "umap" / "qso" / "arrays" / "qso_gaia_matches_desisdss.csv"
    if "phot_g_mean_mag" not in df.columns and proj_csv.exists():
        proj = pd.read_csv(proj_csv, usecols=["gaia_source_id", "phot_g_mean_mag"])
        proj["gaia_source_id"] = proj["gaia_source_id"].astype("int64").astype(str)
        df = df.merge(proj, on="gaia_source_id", how="left")
    print(f"{len(df)} QSO×Euclid matches  "
          f"(Q1={int((df['euclid']=='Q1').sum())}, ERO={int((df['euclid']=='ERO').sum())})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\nRendering standard VIS+NISP stamps (crosshair / scale bar / coord / G / id)...")
    stamps = render_stamps(df, args.size)
    print(f"\n{len(stamps)}/{len(df)} stamps rendered; saving individuals + contact sheet...")
    cut_dir = OUT_DIR / "cutouts"
    cut_dir.mkdir(parents=True, exist_ok=True)
    for old in cut_dir.glob("*.png"):
        old.unlink()                                   # clear any old id-named files
    dfi = df.set_index("qso_id")
    # sheet order: UMAP cluster group, then descending P(gal)
    ordered = sorted(stamps.items(), key=lambda kv: (kv[1][3], -kv[1][2]))
    rows = []
    for i, (qid, (img, aps, _pg, _cl)) in enumerate(ordered):
        fn = f"cutout_{i:04d}.png"
        plt.imsave(cut_dir / fn, np.clip(img, 0, 1))
        rec = dfi.loc[qid].to_dict()               # full match row (comprehensive manifest)
        rec.pop("cutout_file", None)               # drop a stale cutout_file if re-reading manifest
        rec["qso_id"] = qid
        rows.append({"cutout_file": fn, **rec})
    pd.DataFrame(rows).to_csv(MANIFEST, index=False)
    print(f"  -> {len(rows)} cutouts (cutout_NNNN.png) + comprehensive manifest.csv")
    contact_sheet(stamps, args.size)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
