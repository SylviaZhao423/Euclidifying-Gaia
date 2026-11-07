#!/usr/bin/env python3
"""Download Euclid cutouts for all blended sources in crossmatch_qso_catwise_euclid_gaia.

A "blend" is any source where more than one detection from the same instrument
falls within 1" of the WISE position:

  multi_euclid — n_euclid_within_1arcsec > 1  (~338 sources; default 80 downloaded)
      Multiple Euclid objects lie within 1" of the WISE centroid.
      Stamps ordered by n_euclid desc, then separation asc.

  multi_gaia     — n_gaia_within_1arcsec > 1    ( ~10 sources)
      Multiple Gaia DR3 objects lie within 1" of the WISE centroid.
      Stamps ordered by n_gaia desc, then separation asc.

  euclid_deblend — e_deblended_flag > 0         (~2647 sources; default 80 downloaded)
      Euclid internal deblend flag set (Q1 only; NULL for ERO sources).
      Stamps ordered by Euclid separation asc.

A source may appear in more than one group; it receives a cutout in each
applicable subfolder.  No cluster grouping — all sources per blend type fill
one contact sheet, sorted as above.

Companion rings on each stamp:
  Red crosshair   — Euclid primary centroid (cutout centre)
  Orange rings    — Secondary Euclid sources within 1" (n_euclid > 1)
  Magenta ring    — WISE source position (the catalog selector)
  Bright blue     — Gaia sources within 1" of the WISE source
  Pale blue       — Gaia sources in stamp FOV but outside 1"

Outputs:
  output/qso_cutouts/catwise_e_g_blend/
    multi_euclid/cutout_0000.png ...
    contact_sheet_multi_euclid.png
    multi_gaia/cutout_0000.png ...
    contact_sheet_multi_gaia.png
    manifest.csv

Usage:
    PGUSER=siwei_zhao_2025 conda run -n base python3 \\
        src/download_qso_catwise_euclid_g_blend_cutouts.py
    ... --size 6
    ... --no-gaia    # skip ring overlays entirely
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
src = Path(__file__).resolve().parent
sys.path.insert(0, str(src))

from builder_qso_gaia_crossmatch import WSDB_HOST, WSDB_DB  # noqa: E402
from download_qso_catwise_e_g_cutouts import (              # noqa: E402
    pgal_color, fetch_euclid, render_to_array,
    fetch_gaia_positions, fetch_euclid_secondary_positions,
    GAIA_RING, GAIA_FOV_RING, EUCLID_RING, WISE_RING, CACHE_DIR,
)

EUCLID_SIBLING_RING = "#ff8c00"  # dark orange — raw-catalog Euclid siblings of a deblend

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV = PROJECT_DIR / "output" / "umap" / "qso" / "arrays" / "catwise_euclid_gaia.csv"
OUT_DIR = PROJECT_DIR / "output" / "qso_cutouts" / "catwise_e_g_blend"

# Blend groups: (folder_name, human_title, filter_expr, sort_cols, ascending)
BLEND_GROUPS = [
    (
        "multi_euclid",
        "Multiple Euclid within 1\" of WISE",
        "n_euclid_within_1arcsec > 1",
        ["n_euclid_within_1arcsec", "euclid_sep_arcsec"],
        [False, True],
    ),
    (
        "multi_gaia",
        "Multiple Gaia within 1\" of WISE",
        "n_gaia_within_1arcsec > 1",
        ["n_gaia_within_1arcsec", "gaia_sep_arcsec"],
        [False, True],
    ),
    (
        "euclid_deblend",
        "Euclid internal deblend flag set",
        "e_deblended_flag > 0",
        ["euclid_sep_arcsec"],
        [True],
    ),
]


# ── DB fetch ───────────────────────────────────────────────────────────────────

def fetch_euclid_sibling_positions(subset: pd.DataFrame, size_arcsec: float, user: str) -> dict:
    """For euclid_deblend sources, query euclid_q1.mer_final_cat for all Euclid
    sources within the stamp FOV, excluding the primary match.
    Returns {sid_str: [(ra, dec, None, EUCLID_SIBLING_RING, 'crosshair'), ...]}."""
    import psycopg2
    out: dict = {}
    q1_rows = subset[subset["euclid_source"] == "Q1"].copy()
    if q1_rows.empty:
        return out
    radius_deg = (size_arcsec / 2.0) / 3600.0
    try:
        conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                                connect_timeout=30, keepalives=1,
                                keepalives_idle=30, keepalives_interval=10,
                                keepalives_count=5)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 120000")
        for _, row in q1_rows.iterrows():
            sid   = row["sid_str"]
            e_ra  = float(row["e_ra"])
            e_dec = float(row["e_dec"])
            e_id  = int(row["e_id"])
            cur.execute("""
                SELECT object_id, right_ascension, declination
                FROM euclid_q1.mer_final_cat
                WHERE q3c_radial_query(right_ascension, declination, %s, %s, %s)
                  AND object_id != %s
            """, (e_ra, e_dec, radius_deg, e_id))
            rows = cur.fetchall()
            if rows:
                out[sid] = [(float(ra), float(dec), None, EUCLID_SIBLING_RING, "crosshair")
                            for _, ra, dec in rows]
        conn.close()
        n_total = sum(len(v) for v in out.values())
        print(f"  Euclid deblend siblings: {n_total} rings across {len(out)} stamps")
    except Exception as exc:
        print(f"  fetch_euclid_sibling_positions failed ({exc})")
    return out


def fetch_blend_sources(user: str) -> pd.DataFrame:
    """Pull all blended rows from crossmatch_qso_catwise_euclid_gaia.
    Returns empty DataFrame on failure (VPN down etc.)."""
    import psycopg2
    t = f"{user}.crossmatch_qso_catwise_euclid_gaia"
    try:
        conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                                connect_timeout=30, keepalives=1,
                                keepalives_idle=30, keepalives_interval=10,
                                keepalives_count=5)
        conn.cursor().execute("SET statement_timeout = 60000")
        df = pd.read_sql_query(f"""
            SELECT g_source_id,
                   qso_id                        AS wise_id,
                   qso_ra,
                   qso_dec,
                   euclid_source,
                   gaia_sep_arcsec,
                   n_gaia_within_1arcsec,
                   all_gaia_ids_within_1arcsec,
                   all_gaia_seps_within_1arcsec,
                   n_qso_per_gaia,
                   all_qso_ids_per_gaia,
                   gaia_is_mutual_blend           AS gaia_is_blend,
                   e_id,
                   e_ra,
                   e_dec,
                   euclid_sep_arcsec,
                   n_euclid_within_1arcsec,
                   all_euclid_ids_within_1arcsec,
                   all_euclid_seps_within_1arcsec,
                   n_qso_per_euclid,
                   all_qso_ids_per_euclid,
                   euclid_is_mutual_blend         AS euclid_is_blend,
                   e_deblended_flag,
                   w1mpro,
                   w2mpro
            FROM {t}
            WHERE n_euclid_within_1arcsec > 1
               OR n_gaia_within_1arcsec   > 1
               OR n_qso_per_euclid        > 1
               OR n_qso_per_gaia          > 1
               OR e_deblended_flag        > 0
        """, conn)
        conn.close()
        print(f"  {len(df):,} blended sources from {t}")
        return df
    except Exception as exc:
        print(f"  fetch_blend_sources failed ({exc})")
        return pd.DataFrame()


# ── Sample loader ──────────────────────────────────────────────────────────────

def load_sample(user: str) -> pd.DataFrame:
    """Merge flat CSV (model_pgal, cluster, euclid release) with DB blend info."""
    if not CSV.exists():
        raise FileNotFoundError(
            f"{CSV} not found — run analyze_umap_qso.py --catalog euclidwise first.")

    flat = pd.read_csv(
        CSV, dtype={"gaia_source_id": "int64"},
        usecols=["gaia_source_id", "qso_ra", "qso_dec", "cluster",
                 "astrometric_params_solved", "model_pgal", "phot_g_mean_mag", "euclid"])
    flat["sid_str"] = flat["gaia_source_id"].astype(str)

    blends = fetch_blend_sources(user)
    if blends.empty:
        raise RuntimeError("Could not fetch blend sources from DB — is VPN up?")
    blends["sid_str"] = blends["g_source_id"].astype("int64").astype(str)

    # qso_ra/dec live on flat side; drop from blends to avoid suffix collision
    blends = blends.drop(columns=["qso_ra", "qso_dec"], errors="ignore")

    merged = flat.merge(blends, on="sid_str", how="inner")
    print(f"  {len(merged):,} blended sources matched in flat CSV")
    return merged.reset_index(drop=True)


# ── Contact sheet ──────────────────────────────────────────────────────────────

def blend_contact_sheet(items: list, out: Path, title: str):
    """Contact sheet for one blend group. items = [(baked_array, pgal)]."""
    items = sorted(items, key=lambda t: -t[1])
    n = len(items)
    if n == 0:
        return
    ncols = 8
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.4 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (arr, pg) in enumerate(items):
        ax = axes[i // ncols][i % ncols]
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(arr)
        for sp in ax.spines.values():
            sp.set_edgecolor(pgal_color(pg)); sp.set_linewidth(2.6)

    fig_h_in = fig.get_figheight()
    header_in = 1.0
    top_frac = 1.0 - header_in / fig_h_in
    fig.subplots_adjust(top=top_frac, bottom=0.01, left=0.02, right=0.99,
                        hspace=0.05, wspace=0.05)

    cbar_ax = fig.add_axes([0.06, top_frac + 0.30 / fig_h_in,
                            0.20, 0.10 / fig_h_in])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=plt.Normalize(0, 1)),
        cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["star", "0.5", "galaxy"])
    cbar.ax.tick_params(labelsize=7, pad=1)
    cbar_ax.set_title("border = XGBoost P(gal)", fontsize=8, pad=3, loc="left")

    legend_lines = [
        plt.Line2D([0], [0], color=WISE_RING,          lw=0, marker="x", ms=7, mew=1.5, label="WISE centroid"),
        plt.Line2D([0], [0], color=EUCLID_RING,        lw=0, marker="+", ms=8, mew=1.5, label="Secondary Euclid (<1\")"),
        plt.Line2D([0], [0], color=EUCLID_SIBLING_RING, lw=0, marker="+", ms=8, mew=1.5, label="Euclid deblend sibling"),
        plt.Line2D([0], [0], color=GAIA_RING,          lw=0, marker="o", ms=8, markerfacecolor="none", mew=1.2, label="Gaia matched (<1\")"),
        plt.Line2D([0], [0], color=GAIA_FOV_RING,      lw=0, marker="o", ms=8, markerfacecolor="none", mew=1.2, label="Gaia in FOV (>1\")"),
    ]
    fig.legend(handles=legend_lines, loc="upper right",
               bbox_to_anchor=(0.99, top_frac + 0.90 / fig_h_in),
               fontsize=7, framealpha=0.9, ncol=4)

    fig.suptitle(f"Blend: {title}  ({n} sources, sorted by P(gal))",
                 fontsize=13, fontweight="bold",
                 x=0.5, y=top_frac + 0.72 / fig_h_in, color="k")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n",       type=int,   default=80,
                    help="max sources per blend group, 0 = all (default 80)")
    ap.add_argument("--size",    type=float, default=6.0,
                    help="cutout FOV arcsec (default 6)")
    ap.add_argument("--no-gaia", action="store_true",
                    help="skip all WSDB ring queries")
    ap.add_argument("--force", action="store_true",
                    help="delete cached stamps before rendering (re-fetches from IRSA)")
    args = ap.parse_args()

    user = os.environ.get("PGUSER", "siwei_zhao_2025")
    merged = load_sample(user)

    # Build blend_info dict once — used by ring-fetch functions
    blend_info: dict = {}
    if not args.no_gaia:
        blend_info = {row["sid_str"]: row.to_dict() for _, row in merged.iterrows()}

    # Fetch Gaia rings and secondary Euclid rings for the full blended set
    gaia: dict = {}
    euclid_sec: dict = {}
    euclid_sib: dict = {}
    if not args.no_gaia:
        gaia         = fetch_gaia_positions(merged, args.size, blend_info)
        euclid_sec   = fetch_euclid_secondary_positions(merged, blend_info, user)
        deblend_sub  = merged.query("e_deblended_flag > 0").copy()
        euclid_sib   = fetch_euclid_sibling_positions(deblend_sub, args.size, user)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for group_name, group_title, filter_expr, sort_cols, sort_asc in BLEND_GROUPS:
        subset = merged.query(filter_expr).sort_values(
            sort_cols, ascending=sort_asc).reset_index(drop=True)
        if args.n > 0:
            subset = subset.head(args.n)
        print(f"\n{group_name}: {len(subset)} sources  [{filter_expr}]")
        if subset.empty:
            print("  (none — skipping)")
            continue

        gdir = OUT_DIR / group_name
        gdir.mkdir(parents=True, exist_ok=True)
        for old in gdir.glob("*.png"):
            old.unlink()

        items = []
        for i, (_, row) in enumerate(subset.iterrows()):
            sid = row["sid_str"]
            b   = blend_info.get(sid, row.to_dict())

            # Euclid centroid = cutout centre; WISE + Gaia + secondary Euclid as companions
            e_ra  = b.get("e_ra")  or float(row["qso_ra"])
            e_dec = b.get("e_dec") or float(row["qso_dec"])
            companions = None
            if not args.no_gaia:
                sec_euclid = [(ra, dec, lbl, EUCLID_RING, "crosshair")
                              for ra, dec, lbl, *_ in euclid_sec.get(sid, [])]
                siblings   = list(euclid_sib.get(sid, [])) if group_name == "euclid_deblend" else []
                companions = (list(gaia.get(sid, []))
                              + sec_euclid
                              + siblings
                              + [(float(row["qso_ra"]), float(row["qso_dec"]), None, WISE_RING, "x")])

            if args.force:
                for _p in CACHE_DIR.glob(f"{sid}_*.png"):
                    _p.unlink(missing_ok=True)
            arr   = fetch_euclid(row, args.size, companions=companions,
                                 center_ra=e_ra, center_dec=e_dec)
            baked = render_to_array(row, arr, b)
            fn    = f"cutout_{i:04d}.png"
            plt.imsave(gdir / fn, np.clip(baked, 0, 1))
            items.append((baked, float(row["model_pgal"])))

            def _arr_str(v):
                return "|".join(str(x) for x in (v or [])) if v else ""

            manifest_rows.append({
                "blend_group":     group_name,
                "cutout_file":     f"{group_name}/{fn}",
                "gaia_source_id":  sid,
                "cluster":         int(row.get("cluster", -1)),
                "euclid_release":  row.get("euclid", ""),
                "qso_ra":          row["qso_ra"],
                "qso_dec":         row["qso_dec"],
                "model_pgal":      row["model_pgal"],
                "euclid_ok":       arr is not None,
                "wise_id":         b.get("wise_id", ""),
                "w1mpro":          b.get("w1mpro", ""),
                "w2mpro":          b.get("w2mpro", ""),
                "n_euclid_within_1arcsec": b.get("n_euclid_within_1arcsec", ""),
                "euclid_sep_arcsec":       b.get("euclid_sep_arcsec", ""),
                "all_euclid_ids":  _arr_str(b.get("all_euclid_ids_within_1arcsec")),
                "n_qso_per_euclid":        b.get("n_qso_per_euclid", ""),
                "n_gaia_within_1arcsec":   b.get("n_gaia_within_1arcsec", ""),
                "gaia_sep_arcsec":         b.get("gaia_sep_arcsec", ""),
                "all_gaia_ids":    _arr_str(b.get("all_gaia_ids_within_1arcsec")),
                "n_qso_per_gaia":          b.get("n_qso_per_gaia", ""),
                "euclid_is_blend":   b.get("euclid_is_blend", ""),
                "gaia_is_blend":     b.get("gaia_is_blend", ""),
                "e_deblended_flag":  b.get("e_deblended_flag", ""),
                "n_euclid_in_fov": len(euclid_sec.get(sid, [])),
                "n_gaia_in_fov":   len(gaia.get(sid, [])),
            })
            if (i + 1) % 20 == 0 or (i + 1) == len(subset):
                n_ok = sum(1 for it in items if it[0] is not None)
                print(f"  {i+1}/{len(subset)}  ({n_ok} ok)", flush=True)

        blend_contact_sheet(
            items,
            OUT_DIR / f"contact_sheet_{group_name}.png",
            group_title,
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(OUT_DIR / "manifest.csv", index=False)
    print(f"\n  -> {OUT_DIR / 'manifest.csv'}  ({len(manifest_rows)} rows, "
          f"{manifest_df['gaia_source_id'].nunique()} unique sources)")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
