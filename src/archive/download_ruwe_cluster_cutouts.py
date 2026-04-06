#!/usr/bin/env python3
"""
Extract Euclid VIS cutouts for the RUWE-driven separate cluster visible in
ero_umap_features_rebalanced.png.

The cluster is defined by a UMAP bounding box on the rebalanced ERO individual
embedding (ero_indiv_rebalanced.npz): UMAP-x ∈ [1.5, 6.5], UMAP-y ∈ [0.5, 2.8].

Sources are matched to RA/Dec/field via the existing ERO classification CSVs,
then cutouts are rendered with crosshair colour encoding RUWE (plasma, 1–4).

Reads:
  output/umap/arrays/ero_indiv_rebalanced.npz
  output/q1_to_ero/{agree_both_star,agree_both_galaxy,
                    disagree_ero_gal_q1_star,disagree_ero_star_q1_gal}.csv

Writes:
  output/euclid_cutouts/ruwe_cluster/contact_sheet_ruwe_cluster_px.png
  output/euclid_cutouts/ruwe_cluster/manifest.csv

Usage:
  python src/download_ruwe_cluster_cutouts.py [--limit 300] [--size-arcsec 10]
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARRAYS_DIR   = PROJECT_ROOT / "output" / "umap" / "arrays"
ERO_DIR      = PROJECT_ROOT / "output" / "q1_to_ero"
OUTPUT_DIR   = PROJECT_ROOT / "output" / "euclid_cutouts" / "ruwe_cluster"

sys.path.insert(0, str(Path(__file__).parent))
from download_ero_euclid_cutouts import (
    FITS_CACHE, FIELD_TO_TARGET, VIS_PIXEL_SCALE,
    ensure_fits, make_contact_sheet,
)

# UMAP bounding box that isolates the high-RUWE cluster
CLUSTER_X_LO, CLUSTER_X_HI = 1.5, 6.5
CLUSTER_Y_LO, CLUSTER_Y_HI = 0.5, 2.8

# RUWE colourscale for crosshair
RUWE_VMIN, RUWE_VMAX = 1.0, 9.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--limit",       type=int,   default=300,
                   help="Max sources to render (sorted by RUWE descending)")
    p.add_argument("--size-arcsec", type=float, default=10.0,
                   help="Cutout size in arcsec (default 10)")
    p.add_argument("--dpi",         type=int,   default=75)
    p.add_argument("--overwrite",   action="store_true")
    return p.parse_args()


def load_ero_meta() -> pd.DataFrame:
    """Load RA/Dec/field from all ERO classification CSVs."""
    csvs = [
        ERO_DIR / "agree_both_star.csv",
        ERO_DIR / "agree_both_galaxy.csv",
        ERO_DIR / "disagree_ero_gal_q1_star.csv",
        ERO_DIR / "disagree_ero_star_q1_gal.csv",
    ]
    parts = []
    for p in csvs:
        if p.exists():
            parts.append(pd.read_csv(p)[["g_source_id", "ra", "dec", "field"]])
    meta = pd.concat(parts).drop_duplicates("g_source_id")
    meta["g_source_id"] = meta["g_source_id"].astype(str)
    return meta


def make_cutout_image_ruwe(fits_path: Path, ra: float, dec: float,
                            size_arcsec: float, title: str,
                            dpi: int = 75, ruwe: float | None = None) -> np.ndarray:
    """VIS stamp with crosshair coloured by RUWE (plasma, 1–4)."""
    from astropy.io import fits as afits
    from astropy.coordinates import SkyCoord
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    import astropy.units as u

    with afits.open(fits_path, memmap=True) as hdul:
        hdu = None
        for h in hdul:
            if hasattr(h, "data") and isinstance(h.data, np.ndarray) and h.data.ndim == 2:
                hdu = h
                break
        if hdu is None:
            raise RuntimeError(f"No 2D image HDU in {fits_path.name}")
        wcs    = WCS(hdu.header)
        coord  = SkyCoord(ra=ra, dec=dec, unit="deg")
        size   = u.Quantity([size_arcsec, size_arcsec], unit=u.arcsec)
        cutout = Cutout2D(hdu.data, coord, size, wcs=wcs)

    image  = cutout.data.astype(float)
    finite = np.isfinite(image)
    if not finite.any():
        raise RuntimeError("No finite pixels in cutout")

    try:
        from astropy.visualization import ZScaleInterval
        vmin, vmax = ZScaleInterval(contrast=0.25).get_limits(image[finite])
    except Exception:
        vmin, vmax = np.nanpercentile(image[finite], [0.5, 99.5])
    if not (np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin):
        vmin, vmax = np.nanmin(image[finite]), np.nanmax(image[finite])

    if ruwe is not None and np.isfinite(ruwe):
        crosshair_color = plt.cm.plasma(
            np.clip((ruwe - RUWE_VMIN) / (RUWE_VMAX - RUWE_VMIN), 0, 1)
        )
    else:
        crosshair_color = "white"

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
              interpolation="nearest")
    ax.set_title(title, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    n = image.shape[0]
    ax.axhline(n / 2, color=crosshair_color, lw=1.5)
    ax.axvline(n / 2, color=crosshair_color, lw=1.5)
    ax.plot(n / 2, n / 2, "o", color=crosshair_color,
            markersize=7, markeredgecolor="black", markeredgewidth=0.5)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("RUWE cluster cutouts — Euclid ERO VIS")
    print(f"  Cluster box: UMAP-x [{CLUSTER_X_LO}, {CLUSTER_X_HI}]  "
          f"UMAP-y [{CLUSTER_Y_LO}, {CLUSTER_Y_HI}]")
    print(f"  RUWE colourscale: {RUWE_VMIN}–{RUWE_VMAX} (plasma)")
    print("=" * 65)

    # ── Load rebalanced ERO UMAP ──────────────────────────────────────────────
    npz_path = ARRAYS_DIR / "ero_indiv_rebalanced.npz"
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found. Run analyze_umap.py first.")
        return 1

    data       = np.load(npz_path, allow_pickle=True)
    embedding  = data["embedding"]
    X_raw      = data["X_raw"]
    y          = data["y"]
    source_ids = data["source_ids"].astype(str)

    ruwe_col = 5   # index in SELECTED_GAIA_PARAMETERS
    ruwe_vals = X_raw[:, ruwe_col]

    in_cluster = (
        (embedding[:, 0] > CLUSTER_X_LO) & (embedding[:, 0] < CLUSTER_X_HI) &
        (embedding[:, 1] > CLUSTER_Y_LO) & (embedding[:, 1] < CLUSTER_Y_HI)
    )
    print(f"\nCluster sources: {in_cluster.sum():,}  "
          f"(stars={( y[in_cluster]==0).sum():,}, "
          f"gal={(y[in_cluster]==1).sum():,})")
    print(f"RUWE in cluster: {ruwe_vals[in_cluster].min():.2f} – "
          f"{ruwe_vals[in_cluster].max():.2f}  "
          f"(median {np.nanmedian(ruwe_vals[in_cluster]):.2f})")

    # ── Match to RA/Dec/field ─────────────────────────────────────────────────
    print("\nLoading ERO metadata (RA/Dec/field)...")
    meta = load_ero_meta()
    print(f"  Total ERO meta rows: {len(meta):,}")

    cluster_ids    = source_ids[in_cluster]
    cluster_ruwe   = ruwe_vals[in_cluster]
    cluster_y      = y[in_cluster]
    cluster_umap   = embedding[in_cluster]

    df_cluster = pd.DataFrame({
        "source_id": cluster_ids,
        "ruwe":      cluster_ruwe,
        "label":     np.where(cluster_y == 0, "STAR", "GALAXY"),
        "umap1":     cluster_umap[:, 0],
        "umap2":     cluster_umap[:, 1],
    })
    df_cluster = df_cluster.merge(meta, left_on="source_id",
                                  right_on="g_source_id", how="left")
    n_missing = df_cluster["ra"].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} sources have no RA/Dec (not in ERO CSVs) — skipping")
    df_cluster = df_cluster.dropna(subset=["ra", "dec", "field"]).reset_index(drop=True)
    print(f"  Matched: {len(df_cluster):,} sources")

    # Sort by RUWE descending, cap at limit
    df_cluster = df_cluster.sort_values("ruwe", ascending=False).reset_index(drop=True)
    if len(df_cluster) > args.limit:
        df_cluster = df_cluster.head(args.limit).reset_index(drop=True)
        print(f"  Capped to top {args.limit} by RUWE")

    # ── Download FITS per field ───────────────────────────────────────────────
    needed_fields = [f for f in df_cluster["field"].unique() if f in FIELD_TO_TARGET]
    unknown = set(df_cluster["field"].unique()) - set(FIELD_TO_TARGET.keys())
    if unknown:
        print(f"  WARNING: unknown fields: {unknown}")

    fits_cache: dict[str, Path | None] = {}
    print(f"\nFields needed: {needed_fields}")
    for field in needed_fields:
        print(f"\n  Field: {field}")
        fits_cache[field] = ensure_fits(field)

    # ── Render cutouts ────────────────────────────────────────────────────────
    records, cell_images = [], []
    n = len(df_cluster)
    for idx, row in df_cluster.iterrows():
        field  = row.get("field", "")
        ra, dec = float(row["ra"]), float(row["dec"])
        ruwe_v  = float(row["ruwe"])
        label   = row["label"]
        status, message = "ok", ""

        fits_path = fits_cache.get(field)
        if fits_path is None:
            status, message = "skipped", f"no FITS for field '{field}'"
        elif not fits_path.exists():
            status, message = "error", "FITS file missing from cache"
        else:
            try:
                title = (f"{field}  {label}  RUWE={ruwe_v:.2f}\n"
                         f"RA={ra:.4f}  Dec={dec:+.4f}  "
                         f"[{args.size_arcsec:.1f}\"]")
                img = make_cutout_image_ruwe(
                    fits_path, ra, dec, args.size_arcsec, title,
                    dpi=args.dpi, ruwe=ruwe_v,
                )
                cell_images.append(img)
            except Exception as exc:
                status, message = "error", str(exc)

        base = f"{sanitize(field)}_ra{ra:.6f}_dec{dec:+.6f}"
        print(f"  {idx+1:>4}/{n}  {base}  {status}"
              + (f"  — {message}" if message else ""))
        records.append({**row.to_dict(), "status": status, "message": message})

    pd.DataFrame(records).to_csv(OUTPUT_DIR / "manifest.csv", index=False)
    print(f"\n  Manifest: {OUTPUT_DIR / 'manifest.csv'}")

    make_contact_sheet(cell_images, OUTPUT_DIR / "contact_sheet_ruwe_cluster")
    print(f"\nOutputs saved under: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
