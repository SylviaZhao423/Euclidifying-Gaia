#!/usr/bin/env python3
"""Euclid VIS(+NISP) cutouts of the NSA-v1 nuclei that have Euclid imaging (151).

Reads the projected crossmatch_nsa_v1_gaia_euclid sample (nsa_gaia_matches_v1_euclid.csv)
and renders a Euclid stamp per nucleus — Q1 sources from IRSA MER tiles, ERO sources from
the cached ERO full-field FITS (euclid_ero_cutout) — grouped per UMAP cluster and ordered
by descending XGB P(gal). Title notes the Euclid release (Q1/ERO), nsaid and log stellar
mass. The ground-based counterpart is download_nsa_v1_g_cutouts.py (Legacy).

  output/nsa_cutouts/euclid_v1_g/
    cluster{c}/cutout_NNNN.png
    contact_sheet_cluster{c}.png
    manifest.csv

Euclid Q1 tiles: IRSA range reads (cached). ERO: local FITS. No Legacy/WSDB needed.

Usage:
    conda run -n base python3 src/download_nsa_euclid_v1_g_cutouts.py --size 6
"""
from __future__ import annotations

import argparse
import io
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_qso_euclid_wise_g_cutouts import (fetch_euclid, pgal_color,      # noqa: E402
                                                CLUSTER_COLOURS, APS_LABEL)
from euclid_ero_cutout import euclid_release  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV = PROJECT_DIR / "output" / "umap" / "nsa" / "arrays" / "nsa_gaia_matches_v1_euclid.csv"
OUT_DIR = PROJECT_DIR / "output" / "nsa_cutouts" / "euclid_v1_g"


def load_nuclei(limit: int | None) -> pd.DataFrame:
    if not CSV.exists():
        raise SystemExit(f"{CSV} not found — run analyze_umap_nsa.py --catalog v1_euclid first.")
    df = pd.read_csv(CSV, dtype={"gaia_source_id": "int64"}).dropna(subset=["umap1", "umap2"])
    df["sid_str"] = df["gaia_source_id"].astype("int64").astype(str)
    df["qso_ra"], df["qso_dec"] = df["gal_ra"], df["gal_dec"]   # aliases for the shared fetch_euclid
    df = df.sort_values(["cluster", "model_pgal"], ascending=[True, False]).reset_index(drop=True)
    return df.head(limit) if limit else df


def render_to_array(row: pd.Series, arr, dpi: int = 100) -> np.ndarray:
    """Bake one Euclid NSA-nucleus stamp (black title + image, no border) into an array."""
    aps = int(row["astrometric_params_solved"]) if pd.notna(row["astrometric_params_solved"]) else 0
    apsl = APS_LABEL.get(aps, str(aps))
    g, pg, lm = row.get("phot_g_mean_mag", np.nan), row.get("model_pgal", np.nan), row.get("gal_log_mass", np.nan)
    gt = f"{g:.2f}" if pd.notna(g) else "?"
    pt = f"{pg:.2f}" if pd.notna(pg) else "?"
    lmt = f"{lm:.2f}" if pd.notna(lm) else "?"
    title = (f"NSA nucleus  Euclid {euclid_release(row)}  Cluster {int(row['cluster'])}  ({apsl})\n"
             f"RA={float(row['gal_ra']):.4f}  Dec={float(row['gal_dec']):+.4f}  P(gal)={pt}\n"
             f"G={gt}  logM={lmt}  nsaid={int(row['nsaid'])}")
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
        sp.set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def contact_sheet(items: list, out: Path, cluster: int) -> None:
    """Per-cluster sheet: stamps sorted by descending P(gal); P(gal) border; fixed-height
    header (so single-row sheets don't overlap the title). `items` = [(array, pgal)]."""
    items = sorted(items, key=lambda t: -t[1])
    n = len(items)
    if n == 0:
        return
    ncols = 8
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (arr, pg) in enumerate(items):
        ax = axes[i // ncols][i % ncols]
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(arr)
        for sp in ax.spines.values():
            sp.set_edgecolor(pgal_color(pg)); sp.set_linewidth(2.6)
    fig_h = fig.get_figheight(); top = 1.0 - 1.0 / fig_h
    fig.subplots_adjust(top=top, bottom=0.01, left=0.02, right=0.99, hspace=0.05, wspace=0.05)
    cax = fig.add_axes([0.06, top + 0.30 / fig_h, 0.20, 0.10 / fig_h])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=plt.Normalize(0, 1)),
                      cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["star", "0.5", "galaxy"]); cb.ax.tick_params(labelsize=7, pad=1)
    cax.set_title("border = XGBoost P(gal)", fontsize=8, pad=3, loc="left")
    fig.suptitle(f"NSA-v1 nuclei with Euclid — Cluster {cluster} — Euclid VIS+NISP "
                 f"({n} nuclei, sorted by P(gal))", fontsize=13, fontweight="bold",
                 x=0.5, y=top + 0.72 / fig_h, color=CLUSTER_COLOURS.get(cluster, "k"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=float, default=6.0, help="cutout FOV arcsec (default 6)")
    ap.add_argument("--limit", type=int, default=None, help="only the top-N nuclei by P(gal)")
    args = ap.parse_args()
    df = load_nuclei(args.limit)
    print(f"NSA-v1 nuclei with Euclid imaging: {len(df)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest, ne_ok = [], 0
    for cl in sorted(df["cluster"].unique()):
        cint = int(cl)
        rows = df[df["cluster"] == cl].reset_index(drop=True)
        cdir = OUT_DIR / f"cluster{cint}"; cdir.mkdir(parents=True, exist_ok=True)
        for old in cdir.glob("*.png"):
            old.unlink()
        items, n_ok = [], 0
        for i, (_, row) in enumerate(rows.iterrows()):
            arr = fetch_euclid(row, args.size)          # Q1 IRSA tile, else ERO local FITS
            n_ok += arr is not None
            baked = render_to_array(row, arr)
            fn = f"cutout_{i:04d}.png"
            plt.imsave(cdir / fn, np.clip(baked, 0, 1))
            items.append((baked, float(row["model_pgal"])))
            manifest.append({"cutout_file": f"cluster{cint}/{fn}", "cluster": cint,
                             "nsaid": int(row["nsaid"]), "euclid": euclid_release(row),
                             "gaia_source_id": row["sid_str"], "gal_ra": row["gal_ra"],
                             "gal_dec": row["gal_dec"], "gal_log_mass": row.get("gal_log_mass"),
                             "aps": int(row["astrometric_params_solved"]),
                             "model_pgal": row["model_pgal"], "euclid_ok": arr is not None})
        ne_ok += n_ok
        contact_sheet(items, OUT_DIR / f"contact_sheet_cluster{cint}.png", cint)
        print(f"  cluster{cint}: {len(rows)} cutouts ({n_ok} Euclid ok)")
    pd.DataFrame(manifest).to_csv(OUT_DIR / "manifest.csv", index=False)
    print(f"  -> {OUT_DIR/'manifest.csv'}  ({len(manifest)} nuclei, {ne_ok} Euclid ok)")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
