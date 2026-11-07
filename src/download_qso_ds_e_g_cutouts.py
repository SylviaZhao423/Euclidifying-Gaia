#!/usr/bin/env python3
"""Euclid cutouts for the spectroscopic (DESI/SDSS) QSO × Euclid × Gaia matches.

Unlike download_qso_catwise_e_g_cutouts.py (which works off the CatWISE-selected flat
CSV), the sources here come straight from the Euclid∩Gaia join tables built by
crossmatch_qso_{desi_dr1,sdss_dr16}_euclid_gaia.py — i.e. every spectroscopic QSO that
has BOTH a Euclid detection and a Gaia DR3 match:

  crossmatch_qso_desi_dr1_euclid_gaia    1,007  (Q1: 976, ERO: 31)
  crossmatch_qso_sdss_dr16_euclid_gaia      13  (Q1:   4, ERO:  9)

Each source is rendered exactly like the CatWISE cutouts (same Euclid VIS+NISP stamp,
same companion rings) — reusing fetch_euclid / fetch_gaia_positions /
fetch_euclid_secondary_positions from download_qso_catwise_e_g_cutouts — but the title
is spec-QSO-specific (survey, Euclid release, spec-z, QSO id, Gaia id; NO WISE/P(gal)
fields, which don't exist for these tables).

Companion rings on each stamp (same colour code as the CatWISE cutouts):
  Red crosshair   — Euclid primary centroid (cutout centre = e_ra/e_dec)
  Orange +        — Secondary Euclid sources within 1" (n_euclid > 1)
  Magenta x       — QSO (DESI/SDSS) catalogue position (the selector)
  Bright blue     — Gaia sources within 1" of the QSO (matched in the crossmatch)
  Pale blue       — Gaia sources in the stamp FOV but outside 1"

Outputs, one folder per survey — Q1 + ERO combined (output/qso_cutouts/ds_e_g/):
  desi/  sdss/
      cutout_0000.png ...                individual Euclid stamps (sorted by Gaia G)
  contact_sheet_{survey}.png             per-survey contact sheet
  manifest.csv                           every rendered source + blend columns
      (the Q1/ERO release stays in each stamp title and the manifest euclid_source column)

Usage:
    PGUSER=siwei_zhao_2025 conda run -n base python3 src/download_qso_ds_e_g_cutouts.py
    ... --size 6
    ... --survey desi          # only one survey
    ... --n 40                 # cap sources per group (0 = all, default)
    ... --no-gaia              # skip WSDB ring queries (stamp only)
    ... --force                # re-fetch cached stamps
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from builder_qso_gaia_crossmatch import WSDB_HOST, WSDB_DB          # noqa: E402
from download_qso_catwise_e_g_cutouts import (                      # noqa: E402
    fetch_euclid, fetch_gaia_positions, fetch_euclid_secondary_positions,
    GAIA_RING, GAIA_FOV_RING, EUCLID_RING, WISE_RING, CACHE_DIR, APS_LABEL,
)
from render_ero_cutout import euclid_release                        # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "output" / "qso_cutouts" / "ds_e_g"

# survey -> Euclid∩Gaia join table
SURVEY_TABLES = {
    "desi": "crossmatch_qso_desi_dr1_euclid_gaia",
    "sdss": "crossmatch_qso_sdss_dr16_euclid_gaia",
}

# columns needed for rendering + rings + manifest (bare Euclid∩Gaia join-table names)
FETCH_COLS = [
    "qso_catalog", "qso_id", "qso_ra", "qso_dec", "qso_z",
    "e_id", "e_ra", "e_dec", "euclid_source", "euclid_sep_arcsec",
    "n_euclid_within_1arcsec", "all_euclid_ids_within_1arcsec",
    "all_euclid_seps_within_1arcsec", "euclid_is_mutual_blend",
    "g_source_id", "gaia_sep_arcsec", "n_gaia_within_1arcsec",
    "all_gaia_ids_within_1arcsec", "all_gaia_seps_within_1arcsec",
    "gaia_is_mutual_blend", "g_astrometric_params_solved", "g_phot_g_mean_mag",
    "e_deblended_flag",
]


def fetch_survey(survey: str, user: str) -> pd.DataFrame:
    """Every Euclid∩Gaia matched QSO for one survey, with the columns needed to render."""
    import psycopg2
    t = f"{user}.{SURVEY_TABLES[survey]}"
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                            connect_timeout=30, keepalives=1, keepalives_idle=30,
                            keepalives_interval=10, keepalives_count=5)
    conn.cursor().execute("SET statement_timeout = 120000")
    df = pd.read_sql_query(f"SELECT {', '.join(FETCH_COLS)} FROM {t}", conn)
    conn.close()
    df["survey"] = survey
    df["sid_str"] = df["g_source_id"].astype("int64").astype(str)
    # render helpers reuse the CatWISE row API: fetch_euclid reads model_pgal (crosshair
    # colour) — these tables have none, so NaN -> default crosshair; euclid_release reads
    # `euclid`; render title reads aps.
    df["model_pgal"] = np.nan
    df["euclid"] = df["euclid_source"]
    df["aps"] = df["g_astrometric_params_solved"]
    df["group"] = survey                             # one folder per survey: desi / sdss
                                                     # (Q1 + ERO combined; release kept in the title + manifest)
    print(f"  {survey}: {len(df):,} sources "
          f"({(df.euclid_source=='Q1').sum()} Q1, {(df.euclid_source=='ERO').sum()} ERO)")
    return df


def bake_stamp(row: pd.Series, arr, dpi: int = 100) -> np.ndarray:
    """Bake one Euclid stamp (title + image, no border) into an array — spec-QSO title:
    survey, Euclid release, spec-z, QSO id, Gaia id (no WISE/P(gal) fields)."""
    import io
    aps = int(row["aps"]) if pd.notna(row.get("aps")) else 0
    apsl = APS_LABEL.get(aps, str(aps))
    g  = row.get("g_phot_g_mean_mag", np.nan)
    gt = f"{float(g):.2f}" if pd.notna(g) else "?"
    z  = row.get("qso_z", np.nan)
    zt = f"{float(z):.3f}" if pd.notna(z) else "?"

    n_gaia   = int(row.get("n_gaia_within_1arcsec") or 1)
    n_euclid = int(row.get("n_euclid_within_1arcsec") or 1)
    parts = []
    if n_gaia   > 1: parts.append(f"nG={n_gaia}")
    if n_euclid > 1: parts.append(f"nE={n_euclid}")
    if bool(row.get("gaia_is_mutual_blend")) or bool(row.get("euclid_is_mutual_blend")):
        parts.append("★blend")
    bstr = f"  [{', '.join(parts)}]" if parts else ""

    rel   = euclid_release(row)                       # 'Q1' / 'ERO' from row['euclid']
    surv  = str(row["survey"]).upper()
    e_ra  = row.get("e_ra")  if pd.notna(row.get("e_ra"))  else row["qso_ra"]
    e_dec = row.get("e_dec") if pd.notna(row.get("e_dec")) else row["qso_dec"]
    title = (f"QSO Euclid {rel} · {surv}  ({apsl}){bstr}\n"
             f"E_RA={float(e_ra):.4f}  E_Dec={float(e_dec):+.4f}  z={zt}\n"
             f"G={gt}  gaia={row['sid_str']}  {surv.lower()}={int(row['qso_id'])}")

    fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=dpi)
    if arr is not None:
        ax.imshow(np.clip(arr, 0, 1))
    else:
        ax.set_facecolor("#111")
        ax.text(0.5, 0.5, "no Euclid\ntile", color="w", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=7, color="black", pad=10)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def contact_sheet(items: list, out: Path, title: str):
    """One contact sheet per group. items = [(baked_array, G_mag)], drawn brightest-first."""
    items = sorted(items, key=lambda t: (np.inf if not np.isfinite(t[1]) else t[1]))
    n = len(items)
    if n == 0:
        return
    ncols = 8
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (arr, _g) in enumerate(items):
        ax = axes[i // ncols][i % ncols]
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(arr)
        for sp in ax.spines.values():
            sp.set_edgecolor("0.55"); sp.set_linewidth(2.0)
    fig_h_in = fig.get_figheight()
    header_in = 1.0
    top_frac = 1.0 - header_in / fig_h_in
    fig.subplots_adjust(top=top_frac, bottom=0.01, left=0.02, right=0.99, hspace=0.05, wspace=0.05)

    legend_lines = [
        plt.Line2D([0], [0], color=WISE_RING,     lw=0, marker="x", ms=7, mew=1.5, label="QSO (DESI/SDSS) centroid"),
        plt.Line2D([0], [0], color=EUCLID_RING,   lw=0, marker="+", ms=8, mew=1.5, label="Secondary Euclid (<1\")"),
        plt.Line2D([0], [0], color=GAIA_RING,     lw=0, marker="o", ms=8, markerfacecolor="none", mew=1.2, label="Gaia matched (<1\")"),
        plt.Line2D([0], [0], color=GAIA_FOV_RING, lw=0, marker="o", ms=8, markerfacecolor="none", mew=1.2, label="Gaia in FOV (>1\")"),
    ]
    fig.legend(handles=legend_lines, loc="upper right",
               bbox_to_anchor=(0.99, top_frac + 0.90 / fig_h_in),
               fontsize=7, framealpha=0.9, ncol=4)
    fig.suptitle(f"{title}  ({n} sources, sorted by Gaia G)", fontsize=13, fontweight="bold",
                 x=0.5, y=top_frac + 0.72 / fig_h_in, color="k")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}  ({n} stamps)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size",   type=float, default=6.0, help="cutout FOV arcsec (default 6)")
    ap.add_argument("--survey", choices=["desi", "sdss", "both"], default="both")
    ap.add_argument("--n",      type=int,   default=0,
                    help="max sources per group, 0 = all (default 0)")
    ap.add_argument("--no-gaia", action="store_true", help="skip all WSDB ring queries")
    ap.add_argument("--force",   action="store_true",
                    help="delete cached stamps before rendering (re-fetches from IRSA)")
    args = ap.parse_args()

    user = os.environ.get("PGUSER", "siwei_zhao_2025")
    surveys = ["desi", "sdss"] if args.survey == "both" else [args.survey]

    print("[1] fetching Euclid∩Gaia matched QSOs ...")
    merged = pd.concat([fetch_survey(s, user) for s in surveys], ignore_index=True)

    # Ring inputs (shared across groups) — build once for the whole set.
    blend_info: dict = {}
    gaia: dict = {}
    euclid_sec: dict = {}
    if not args.no_gaia:
        print("[2] fetching Gaia + secondary-Euclid ring positions ...")
        blend_info = {r["sid_str"]: r.to_dict() for _, r in merged.iterrows()}
        gaia       = fetch_gaia_positions(merged, args.size, blend_info)
        euclid_sec = fetch_euclid_secondary_positions(merged, blend_info, user)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    # one folder per survey (Q1 + ERO combined): desi, sdss
    for group in surveys:
        sub = merged[merged["group"] == group]
        if args.n > 0:
            sub = sub.sort_values("g_phot_g_mean_mag").head(args.n)
        sub = sub.reset_index(drop=True)
        print(f"\n{group}: {len(sub)} sources")
        if sub.empty:
            print("  (none — skipping)")
            continue

        gdir = OUT_DIR / group
        gdir.mkdir(parents=True, exist_ok=True)
        for old in gdir.glob("*.png"):
            old.unlink()

        items = []
        n_ok = 0
        for i, (_, row) in enumerate(sub.iterrows()):
            sid = row["sid_str"]

            # Euclid centroid = cutout centre; QSO + Gaia + secondary Euclid as companions
            e_ra  = row.get("e_ra")  if pd.notna(row.get("e_ra"))  else float(row["qso_ra"])
            e_dec = row.get("e_dec") if pd.notna(row.get("e_dec")) else float(row["qso_dec"])
            companions = None
            if not args.no_gaia:
                sec_euclid = [(ra, dec, lbl, EUCLID_RING, "crosshair")
                              for ra, dec, lbl, *_ in euclid_sec.get(sid, [])]
                companions = (list(gaia.get(sid, []))
                              + sec_euclid
                              + [(float(row["qso_ra"]), float(row["qso_dec"]), None, WISE_RING, "x")])

            if args.force:
                for _p in CACHE_DIR.glob(f"{sid}_*.png"):
                    _p.unlink(missing_ok=True)
            arr   = fetch_euclid(row, args.size, companions=companions,
                                 center_ra=e_ra, center_dec=e_dec)
            n_ok += arr is not None
            baked = bake_stamp(row, arr)
            fn    = f"cutout_{i:04d}.png"
            plt.imsave(gdir / fn, np.clip(baked, 0, 1))
            items.append((baked, float(row["g_phot_g_mean_mag"])
                          if pd.notna(row.get("g_phot_g_mean_mag")) else np.nan))

            def _arr_str(v):
                return "|".join(str(x) for x in (v or [])) if v is not None else ""

            manifest.append({
                "group":            group,
                "survey":           row["survey"],
                "euclid_source":    row["euclid_source"],
                "cutout_file":      f"{group}/{fn}",
                "gaia_source_id":   sid,
                "qso_id":           int(row["qso_id"]),
                "qso_catalog":      row.get("qso_catalog", ""),
                "qso_ra":           row["qso_ra"],
                "qso_dec":          row["qso_dec"],
                "qso_z":            row.get("qso_z", ""),
                "e_id":             int(row["e_id"]) if pd.notna(row.get("e_id")) else "",
                "e_ra":             e_ra,
                "e_dec":            e_dec,
                "euclid_sep_arcsec": row.get("euclid_sep_arcsec", ""),
                "aps":              int(row["aps"]) if pd.notna(row.get("aps")) else "",
                "g_phot_g_mean_mag": row.get("g_phot_g_mean_mag", ""),
                "euclid_ok":        arr is not None,
                "gaia_sep_arcsec":  row.get("gaia_sep_arcsec", ""),
                "n_gaia_within_1arcsec":   row.get("n_gaia_within_1arcsec", ""),
                "all_gaia_ids":     _arr_str(row.get("all_gaia_ids_within_1arcsec")),
                "gaia_is_blend":    row.get("gaia_is_mutual_blend", ""),
                "n_euclid_within_1arcsec": row.get("n_euclid_within_1arcsec", ""),
                "all_euclid_ids":   _arr_str(row.get("all_euclid_ids_within_1arcsec")),
                "euclid_is_blend":  row.get("euclid_is_mutual_blend", ""),
                "e_deblended_flag": row.get("e_deblended_flag", ""),
                "n_gaia_in_fov":    len(gaia.get(sid, [])),
                "n_euclid_secondary": len(euclid_sec.get(sid, [])),
            })
            if (i + 1) % 25 == 0 or (i + 1) == len(sub):
                print(f"  {i+1}/{len(sub)}  ({n_ok} Euclid ok)", flush=True)

        contact_sheet(items, OUT_DIR / f"contact_sheet_{group}.png",
                      f"QSO Euclid+Gaia — {group}")
        print(f"  {group}: {n_ok}/{len(sub)} Euclid ok")

    pd.DataFrame(manifest).to_csv(OUT_DIR / "manifest.csv", index=False)
    print(f"\n  -> {OUT_DIR / 'manifest.csv'}  ({len(manifest)} sources)")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
