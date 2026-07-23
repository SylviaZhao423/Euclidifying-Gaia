#!/usr/bin/env python3
"""Euclid cutouts for the Euclid+WISE "blend" candidates (n_euclid_within_1arcsec > 1).

The Euclid-side analog of download_ds_g_blend_cutouts.py. In the Euclid+WISE sample
each row is one Euclid MER object matched to its closest Gaia source, so the Gaia-side
blend flag n_gaia_within_1arcsec is always 1. The meaningful blend here is the REVERSE:
n_euclid_within_1arcsec > 1 — a single Gaia detection that Euclid's fine VIS resolution
(0.1"/px) deblends into several MER objects.

Mirroring download_ds_g_blend_cutouts.py (which circles the several Gaia detections that
make up an SDSS blend), we overlay the several Euclid MER detections that Euclid deblends
this one Gaia source into — a cyan circle at each MER component within 1" of the Gaia
position (the exact objects counted by n_euclid_within_1arcsec). The outer box is coloured
by XGB P(gal); a P(gal)-coloured crosshair (coolwarm_r: red=star, blue=galaxy) marks the
Gaia/source position, and the title reports the Euclid multiplicity (n_euclid):

  output/qso_cutouts/euclid_wise_g_blend/
    cluster{c}/cutout_NNNN.png           individual Euclid stamps per cluster (sorted by P(gal))
    contact_sheet_cluster{c}.png         one contact sheet per UMAP cluster (sorted by P(gal))
    manifest.csv

Euclid tiles: per-source IRSA IBE range reads (cached, shared with the other Euclid
cutout scripts). The MER-companion positions come from WSDB (VPN) — the same
euclid_q1.mer_final_cat / euclid_ero.merge_cat catalogs and 1" radius that produced
n_euclid_within_1arcsec; if WSDB is unreachable the stamps still render (no circles).

Usage:
    conda run -n base python3 src/download_euclid_wise_g_blend_cutouts.py            # all blends
    ... --size 6            # cutout FOV in arcsec (default 6)
    ... --limit 40          # only the top-N blends by P(gal)
"""
from __future__ import annotations

import argparse
import io
import math
import signal
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from save_q1_utils import (_assign_to_tiles, _nisp_filename, _render_cutout,  # noqa: E402
                           IRSA_IBE_DATA)
from download_qso_euclid_wise_g_cutouts import (_tiles_near, pgal_color,       # noqa: E402
                                                CLUSTER_COLOURS, APS_LABEL,
                                                CACHE_DIR, FETCH_TIMEOUT,
                                                _FetchTimeout, _on_alarm)
from euclid_ero_cutout import render_ero_cutout, euclid_release  # noqa: E402
from qso_gaia_crossmatch_builder import WSDB_HOST, WSDB_DB       # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV = PROJECT_DIR / "output" / "umap" / "qso" / "arrays" / "qso_gaia_matches_euclidwise.csv"
OUT_DIR = PROJECT_DIR / "output" / "qso_cutouts" / "euclid_wise_g_blend"


# --------------------------------------------------------------------------- #
#  data                                                                        #
# --------------------------------------------------------------------------- #
def load_blends(limit: int | None) -> pd.DataFrame:
    """Euclid+WISE sources that Euclid deblends (n_euclid_within_1arcsec > 1),
    grouped by UMAP cluster, then descending XGB P(gal)."""
    df = pd.read_csv(CSV, dtype={"gaia_source_id": "int64"})
    b = df[df["n_euclid_within_1arcsec"] > 1].copy()
    b["sid_str"] = b["gaia_source_id"].astype("int64").astype(str)
    b = b.sort_values(["cluster", "model_pgal"], ascending=[True, False]).reset_index(drop=True)
    return b.head(limit) if limit else b


def fetch_mer_companions(blends: pd.DataFrame) -> tuple[dict, dict]:
    """(companions, wise_names):
      companions[sid] = [(ra, dec, mer_object_id_str), ...]  the Euclid MER detections
        within 1" of the source — the very objects counted by n_euclid_within_1arcsec,
        each labelled with its MER object id (mirrors the Gaia-blend corner labels).
      wise_names[sid] = catwise source_name (WISE designation) for the title.
    Same catalogs / 1" radius as the base + extract builds:
        Q1  -> euclid_q1.mer_final_cat (object_id, right_ascension, declination)
        ERO -> euclid_ero.merge_cat    (v_objid,   v_alpha_j2000,   v_delta_j2000)
        WISE-> catwise_202003.main     (source_name, ra, dec; closest within 1")
    The MER search is centred on the GAIA position (g_ra/g_dec, pulled from the crossmatch
    table by g_source_id) — exactly as extract_qso_euclid_wise_gaia.py counted n_euclid — so
    the number of circles matches n_euclid (the MER-primary position qso_ra/qso_dec is offset
    from Gaia by gaia_sep, and centring there would miss the >1"-from-primary components).
    The WISE match stays on the MER-primary position (that is how the base matched CatWISE).
    Needs WSDB (VPN). On failure returns ({}, {}) so the stamps still render (no overlay)."""
    import os
    import psycopg2
    spec = {"Q1":  ("euclid_q1.mer_final_cat", "object_id", "right_ascension", "declination"),
            "ERO": ("euclid_ero.merge_cat",     "v_objid",   "v_alpha_j2000",   "v_delta_j2000")}
    comps: dict = {}
    wise: dict = {}
    user = os.environ.get("PGUSER", "siwei_zhao_2025")
    try:
        conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                                connect_timeout=30, keepalives=1, keepalives_idle=30,
                                keepalives_interval=10, keepalives_count=5)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 120000")
        # Gaia positions used for the n_euclid count (indexed on g_source_id).
        sids = [int(s) for s in blends["sid_str"]]
        cur.execute("SELECT g_source_id, g_ra, g_dec FROM crossmatch_qso_euclid_wise_gaia "
                    "WHERE g_source_id = ANY(%s)", (sids,))
        gpos = {str(r[0]): (float(r[1]), float(r[2])) for r in cur.fetchall()}
        for _, row in blends.iterrows():
            rel = euclid_release(row)
            if rel not in spec:
                continue
            tbl, idc, rac, decc = spec[rel]
            sid = row["sid_str"]
            qra, qdec = float(row["qso_ra"]), float(row["qso_dec"])   # MER primary → WISE match
            gra, gdec = gpos.get(sid, (qra, qdec))                    # Gaia → MER (n_euclid def)
            cur.execute(
                f"SELECT me.{idc}, me.{rac}, me.{decc} FROM {tbl} me "
                f"WHERE q3c_join(%s, %s, me.{rac}, me.{decc}, 1.0/3600.0)", (gra, gdec))
            comps[sid] = [(float(r[1]), float(r[2]), str(int(r[0]))) for r in cur.fetchall()]
            cur.execute(
                "SELECT cw.source_name FROM catwise_202003.main cw "
                "WHERE q3c_join(%s, %s, cw.ra, cw.dec, 1.0/3600.0) "
                "ORDER BY q3c_dist(%s, %s, cw.ra, cw.dec) ASC LIMIT 1",
                (qra, qdec, qra, qdec))
            r = cur.fetchone()
            wise[sid] = r[0] if r else None
        cur.close(); conn.close()
        got = sum(1 for v in comps.values() if v)
        # Circles should reproduce n_euclid (both counted within 1" of the Gaia position).
        nmatch = sum(1 for _, row in blends.iterrows()
                     if len(comps.get(row["sid_str"], [])) == int(row["n_euclid_within_1arcsec"]))
        print(f"  MER companions from WSDB: {got}/{len(comps)} blends have >=1 MER detection; "
              f"{nmatch}/{len(comps)} match n_euclid exactly; "
              f"{sum(1 for v in wise.values() if v)} have a WISE designation")
    except Exception as exc:
        print(f"  WSDB companion/WISE query failed ({exc}); rendering without overlay. "
              f"Is the VPN up?")
        return {}, {}
    return comps, wise


# --------------------------------------------------------------------------- #
#  imaging                                                                     #
# --------------------------------------------------------------------------- #
def fetch_euclid(row: pd.Series, size: float, companions: list | None = None) -> np.ndarray | None:
    """Rendered Euclid stamp WITH the P(gal) crosshair AND the deblended MER companions
    circled (raw figure array). Cached to disk under an '_xhc' key (crosshair+companions),
    so it never collides with the plain '_xh' blend stamps or the crosshair-free per-cluster
    stamps; a hard SIGALRM timeout skips stalls."""
    sid = row["sid_str"]
    cache = CACHE_DIR / f"{sid}_{size:.0f}_xhc.png"
    if cache.exists():
        return plt.imread(cache)
    ra, dec = float(row["qso_ra"]), float(row["qso_dec"])
    ti = _assign_to_tiles(ra, dec, _tiles_near(ra, dec), "euclidwise")
    if ti is None:
        # No Q1 MER tile → ERO source: render from the cached ERO VIS full-field FITS.
        arr = render_ero_cutout(ra, dec, size, c_star=float(row["model_pgal"]),
                                draw_crosshair=True, companions=companions)
        if arr is not None:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            plt.imsave(cache, np.clip(arr, 0, 1))
        return arr
    njf, nhf = _nisp_filename(ti.tile_id, "J"), _nisp_filename(ti.tile_id, "H")
    nju = f"{IRSA_IBE_DATA}/{ti.tile_id}/NISP/{njf}" if njf else None
    nhu = f"{IRSA_IBE_DATA}/{ti.tile_id}/NISP/{nhf}" if nhf else None
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(FETCH_TIMEOUT)
    try:
        arr = _render_cutout(ti.url, ra, dec, size, "", c_star=float(row["model_pgal"]),
                             nisp_j_url=nju, nisp_h_url=nhu, draw_crosshair=True,
                             companions=companions)
    except _FetchTimeout:
        print(f"    euclid {sid}: TIMEOUT ({FETCH_TIMEOUT}s) — skipping"); return None
    except Exception as exc:
        print(f"    euclid {sid}: FAILED ({exc})"); return None
    finally:
        signal.alarm(0)
    if arr is not None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        plt.imsave(cache, np.clip(arr, 0, 1))
    return arr


# --------------------------------------------------------------------------- #
#  rendering                                                                   #
# --------------------------------------------------------------------------- #
def render_to_array(row: pd.Series, arr, dpi: int = 100, wise_name: str | None = None) -> np.ndarray:
    """Bake one Euclid blend stamp (black title + image, no border) into an array so
    the contact sheet tiles them tightly. Title carries the WISE designation (the single
    WISE source that Euclid deblends) + the Euclid multiplicity; the per-MER Euclid object
    ids are labelled at the stamp corners (like the Gaia-blend source_id labels)."""
    aps = int(row["astrometric_params_solved"]) if pd.notna(row["astrometric_params_solved"]) else 0
    apsl = APS_LABEL.get(aps, str(aps))
    g = row.get("phot_g_mean_mag", np.nan)
    pg = row.get("model_pgal", np.nan)
    ne = int(row["n_euclid_within_1arcsec"])
    gt = f"{g:.2f}" if pd.notna(g) else "?"
    pt = f"{pg:.2f}" if pd.notna(pg) else "?"
    wt = f"WISE {wise_name}" if wise_name else f"gaia_id={row['sid_str']}"
    title = (f"QSO Euclid {euclid_release(row)} + WISE BLEND  Cluster {int(row['cluster'])}  ({apsl})\n"
             f"RA={float(row['qso_ra']):.4f}  Dec={float(row['qso_dec']):+.4f}  P(gal)={pt}\n"
             f"G={gt}  n_euclid={ne}  {wt}")
    fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=dpi)
    if arr is not None:
        ax.imshow(np.clip(arr, 0, 1))
    else:
        ax.set_facecolor("#111")
        ax.text(0.5, 0.5, "no Euclid\ntile", color="w", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8, color="black", pad=10)
    for sp in ax.spines.values():
        sp.set_visible(False)                 # border added by the contact sheet
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def contact_sheet(items: list, out: Path, cluster: int) -> None:
    """One PER-CLUSTER Euclid blend sheet: stamps sorted by descending P(gal); outer box
    coloured by P(gal); cluster-coloured suptitle. Same construction as the per-cluster
    Euclid cutout sheets. `items` = list of (baked_array, pgal)."""
    items = sorted(items, key=lambda t: -t[1])
    n = len(items)
    if n == 0:
        return
    ncols = 8
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (img, pg) in enumerate(items):
        ax = axes[i // ncols][i % ncols]
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(img)
        for sp in ax.spines.values():
            sp.set_edgecolor(pgal_color(pg)); sp.set_linewidth(2.6)
    # Fixed-height header (inches) so the colour-bar + suptitle never collide with the
    # stamps — critical for a single-row sheet, where a fractional top would overlap.
    fig_h_in = fig.get_figheight()
    header_in = 1.0
    top_frac = 1.0 - header_in / fig_h_in
    fig.subplots_adjust(top=top_frac, bottom=0.01, left=0.02, right=0.99, hspace=0.05, wspace=0.05)
    cbar_ax = fig.add_axes([0.06, top_frac + 0.30 / fig_h_in, 0.20, 0.10 / fig_h_in])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=plt.Normalize(0, 1)),
                        cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 0.5, 1]); cbar.set_ticklabels(["star", "0.5", "galaxy"])
    cbar.ax.tick_params(labelsize=7, pad=1)
    cbar_ax.set_title("border = XGBoost P(gal)", fontsize=8, pad=3, loc="left")
    fig.suptitle(f"Euclid+WISE blend candidates — Cluster {cluster} — Euclid VIS+NISP "
                 f"({n} sources, Euclid deblends 1 Gaia into >1 MER, sorted by P(gal))",
                 fontsize=13, fontweight="bold", x=0.5, y=top_frac + 0.72 / fig_h_in,
                 color=CLUSTER_COLOURS.get(cluster, "k"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=float, default=6.0, help="cutout FOV in arcsec (default 6)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only render the top-N blends by P(gal)")
    ap.add_argument("--no-circles", action="store_true",
                    help="skip the WSDB MER-companion query (no deblend circles)")
    args = ap.parse_args()
    if not CSV.exists():
        print(f"{CSV} not found — run analyze_umap_qso.py --catalog euclidwise first."); return 1

    blends = load_blends(args.limit)
    print(f"Euclid+WISE blend candidates (n_euclid_within_1arcsec > 1): {len(blends)}")
    if blends.empty:
        print("  none — nothing to do."); return 0

    # The multiple MER detections to circle+label on each stamp (mirrors ds_g_blend
    # circling the multiple Gaia detections) + the WISE designation for the title.
    # WSDB/VPN; empty dicts => plain stamps.
    companions, wise_names = ({}, {}) if args.no_circles else fetch_mer_companions(blends)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    ne_ok = 0
    # Per-cluster folders (cluster{c}/cutout_NNNN.png), like output/qso_cutouts/euclid_wise/
    for cl in sorted(blends["cluster"].unique()):
        cint = int(cl)
        rows = blends[blends["cluster"] == cl].sort_values(
            "model_pgal", ascending=False).reset_index(drop=True)
        cdir = OUT_DIR / f"cluster{cint}"; cdir.mkdir(parents=True, exist_ok=True)
        for old in cdir.glob("*.png"):        # clear stale cutouts before renumbering
            old.unlink()
        items = []
        n_ok = 0
        for i, (_, row) in enumerate(rows.iterrows()):
            arr = fetch_euclid(row, args.size, companions=companions.get(row["sid_str"]))
            n_ok += arr is not None
            baked = render_to_array(row, arr, wise_name=wise_names.get(row["sid_str"]))
            fn = f"cutout_{i:04d}.png"                        # numbered by P(gal) within cluster
            plt.imsave(cdir / fn, np.clip(baked, 0, 1))
            items.append((baked, float(row["model_pgal"])))
            comp = companions.get(row["sid_str"]) or []
            manifest.append({"cutout_file": f"cluster{cint}/{fn}", "cluster": cint,
                             "gaia_source_id": row["sid_str"],
                             "wise_source_name": wise_names.get(row["sid_str"]),
                             "euclid_mer_ids": ";".join(c[2] for c in comp),
                             "aps": int(row["astrometric_params_solved"]),
                             "qso_ra": row["qso_ra"], "qso_dec": row["qso_dec"],
                             "model_pgal": row["model_pgal"],
                             "n_euclid_within_1arcsec": int(row["n_euclid_within_1arcsec"]),
                             "n_mer_detected": len(comp),
                             "euclid_ok": arr is not None})
        ne_ok += n_ok
        contact_sheet(items, OUT_DIR / f"contact_sheet_cluster{cint}.png", cint)
        print(f"  cluster{cint}: {len(rows)} cutouts ({n_ok} Euclid ok)")
    print(f"  -> {len(manifest)} individual Euclid blend cutouts in {OUT_DIR}/cluster*/  "
          f"({ne_ok} Euclid ok)")
    pd.DataFrame(manifest).to_csv(OUT_DIR / "manifest.csv", index=False)
    print(f"  -> {OUT_DIR/'manifest.csv'}  ({len(manifest)} sources)")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
