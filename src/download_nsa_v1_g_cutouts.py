#!/usr/bin/env python3
"""Legacy/DECaLS cutouts of the NSA-v1 nuclei that have Euclid imaging (151).

The ground-based counterpart to download_nsa_euclid_v1_g_cutouts.py (Euclid) — same 151
nuclei, so you can compare Legacy (~1" seeing) vs Euclid (0.1" VIS) side by side for the
nucleation question. Reads the projected crossmatch_nsa_v1_gaia_euclid sample
(nsa_gaia_matches_v1_euclid.csv); per UMAP cluster, downloads a Legacy cutout for each
nucleus (P(gal)-sorted), P(gal) border, title with Euclid release / nsaid / log mass:

  output/nsa_cutouts/v1_g/
    cluster{c}/cutout_NNNN.png
    contact_sheet_cluster{c}.png
    manifest.csv

Legacy needs legacysurvey.org reachable (VPN OFF — it black-holes NERSC). No WSDB.

Usage:
    conda run -n base python3 src/download_nsa_v1_g_cutouts.py --size 12
    ... --workers 24        # more parallel fetches
"""
from __future__ import annotations

import argparse
import io
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_qso_ds_g_cutouts import fetch_ls_jpg, pgal_color, _place_block, _finish_sheet  # noqa: E402
from euclid_ero_cutout import euclid_release  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV = PROJECT_DIR / "output" / "umap" / "nsa" / "arrays" / "nsa_gaia_matches_v1_euclid.csv"
OUT_DIR = PROJECT_DIR / "output" / "nsa_cutouts" / "v1_g"


def load_nuclei(limit: int | None) -> pd.DataFrame:
    if not CSV.exists():
        raise SystemExit(f"{CSV} not found — run analyze_umap_nsa.py --catalog v1_euclid first.")
    df = pd.read_csv(CSV, dtype={"gaia_source_id": "int64"}).dropna(subset=["umap1", "umap2"])
    df["sid_str"] = df["gaia_source_id"].astype("int64").astype(str)
    df = df.sort_values(["cluster", "model_pgal"], ascending=[True, False]).reset_index(drop=True)
    return df.head(limit) if limit else df


def render_to_array(row: pd.Series, img, size_arcsec: float, dpi: int = 100) -> np.ndarray:
    """Bake one Legacy NSA-nucleus stamp (title + image + scale bar, no border) into an array."""
    half = size_arcsec / 2.0
    fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=dpi)
    if img is not None:
        ax.imshow(np.asarray(img), extent=[half, -half, -half, half], origin="upper")
    else:
        ax.set_facecolor("#111111")
        ax.text(0, 0, "no\nimage", color="w", ha="center", va="center", fontsize=7)
    ax.plot([-half + 1.3, -half + 0.3], [-half + 0.7, -half + 0.7], "-", color="w", lw=2.0)
    ax.text(-half + 0.8, -half + 1.05, '1"', color="w", fontsize=7, ha="center", va="bottom")
    aps = int(row["astrometric_params_solved"]) if pd.notna(row["astrometric_params_solved"]) else 0
    apsl = {3: "2-param", 31: "5-param", 95: "6-param"}.get(aps, str(aps))
    g, pg, lm = row.get("phot_g_mean_mag", np.nan), row.get("model_pgal", np.nan), row.get("gal_log_mass", np.nan)
    gt = f"{g:.2f}" if pd.notna(g) else "?"
    pt = f"{pg:.2f}" if pd.notna(pg) else "?"
    lmt = f"{lm:.2f}" if pd.notna(lm) else "?"
    title = (f"NSA nucleus  Euclid {euclid_release(row)}  Cluster {int(row['cluster'])}  ({apsl})\n"
             f"RA={float(row['gal_ra']):.4f}  Dec={float(row['gal_dec']):+.4f}  P(gal)={pt}\n"
             f"G={gt}  logM={lmt}  nsaid={int(row['nsaid'])}")
    ax.set_title(title, fontsize=8, color="black", pad=10)
    ax.set_xlim(half, -half); ax.set_ylim(-half, half); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def contact_sheet(items: list, cluster: int, out: Path) -> None:
    """Per-cluster sheet: stamps sorted by descending P(gal); P(gal) border (via the shared
    _place_block / _finish_sheet — same fixed-header layout as the QSO Legacy sheets)."""
    items = sorted(items, key=lambda t: -t[1])
    n = len(items)
    if n == 0:
        return
    ncols = 8
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    _place_block(axes, items, 0, ncols)
    _finish_sheet(fig, out, f"NSA-v1 nuclei with Euclid — Cluster {cluster} — Legacy/DECaLS "
                  f"({n} nuclei, sorted by P(gal))", cluster)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=float, default=12.0, help="cutout FOV arcsec (default 12)")
    ap.add_argument("--limit", type=int, default=None, help="only the top-N nuclei by P(gal)")
    ap.add_argument("--workers", type=int, default=12, help="parallel Legacy fetches per cluster")
    args = ap.parse_args()
    df = load_nuclei(args.limit)
    print(f"NSA-v1 nuclei with Euclid imaging: {len(df)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for cl in sorted(df["cluster"].unique()):
        cint = int(cl)
        rows = df[df["cluster"] == cl].reset_index(drop=True)
        cdir = OUT_DIR / f"cluster{cint}"; cdir.mkdir(parents=True, exist_ok=True)
        for old in cdir.glob("*.png"):
            old.unlink()
        print(f"  cluster{cint}: fetching {len(rows)} Legacy cutouts ({args.workers} parallel)...")
        row_list = [row for _, row in rows.iterrows()]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            imgs = list(ex.map(
                lambda row: fetch_ls_jpg(float(row["gal_ra"]), float(row["gal_dec"]), args.size),
                row_list))
        items = []
        for i, (row, img) in enumerate(zip(row_list, imgs)):
            arr = render_to_array(row, img, args.size)
            fn = f"cutout_{i:04d}.png"
            plt.imsave(cdir / fn, np.clip(arr, 0, 1))
            items.append((arr, float(row["model_pgal"])))
            manifest.append({"cutout_file": f"cluster{cint}/{fn}", "cluster": cint,
                             "nsaid": int(row["nsaid"]), "euclid": euclid_release(row),
                             "gaia_source_id": row["sid_str"], "gal_ra": row["gal_ra"],
                             "gal_dec": row["gal_dec"], "gal_log_mass": row.get("gal_log_mass"),
                             "aps": int(row["astrometric_params_solved"]),
                             "model_pgal": row["model_pgal"], "has_image": img is not None})
        print(f"    cluster{cint}: {sum(im is not None for im in imgs)}/{len(rows)} imaged")
        contact_sheet(items, cint, OUT_DIR / f"contact_sheet_cluster{cint}.png")
    pd.DataFrame(manifest).to_csv(OUT_DIR / "manifest.csv", index=False)
    print(f"  -> {OUT_DIR/'manifest.csv'}  ({len(manifest)} nuclei)")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
