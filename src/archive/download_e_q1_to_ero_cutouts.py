#!/usr/bin/env python3
"""
Download Euclid ERO VIS cutouts for ERO×Q1 agreement/disagreement groups,
stratified by Q1 UMAP cluster.

Cutouts use log stretch with crosshair coloured by Gaia corrected BP/RP
excess (c_star, coolwarm colourmap: blue=star, red=galaxy).

Each ERO source is projected into Q1 UMAP space via the fitted reducer,
then assigned to the nearest Q1 cluster.  Per-cluster contact sheets and
manifests are saved under {group}/cluster{k}/.

Reads:
  output/q1_to_ero/disagree_ero_gal_q1_star.csv
  output/q1_to_ero/disagree_ero_star_q1_gal.csv
  output/q1_to_ero/agree_both_galaxy.csv
  output/q1_to_ero/agree_both_star.csv
  output/umap/arrays/ero_indiv.npz              (ERO Gaia features + source IDs)
  output/umap/arrays/q1_indiv.npz               (Q1 UMAP embedding + normalisation)
  output/umap/arrays/q1_indiv_reducer.joblib    (fitted UMAP reducer)
  output/umap/arrays/q1_cluster_labels.npy      (Q1 cluster labels)
  output/euclid_ero_fits_cache/*.fits

Writes:
  output/euclid_cutouts/ero_cutouts/q1_to_ero/{group}/manifest.csv
  output/euclid_cutouts/ero_cutouts/q1_to_ero/{group}/cluster{k}/contact_sheet.png
  output/euclid_cutouts/ero_cutouts/q1_to_ero/{group}/cluster{k}/manifest.csv
  output/euclid_cutouts/ero_cutouts/q1_to_ero/{group}/cluster{k}/cutouts/cutout_NNNN.png

Usage:
  python src/download_e_q1_to_ero_cutouts.py
  python src/download_e_q1_to_ero_cutouts.py --per-cell 20
  python src/download_e_q1_to_ero_cutouts.py --size-arcsec 6
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT   = Path(__file__).resolve().parents[1]
INPUT_DIR      = PROJECT_ROOT / "output" / "q1_to_ero"
OUTPUT_DIR     = PROJECT_ROOT / "output" / "euclid_cutouts" / "ero_cutouts" / "q1_to_ero"
FITS_CACHE     = PROJECT_ROOT / "output" / "euclid_ero_fits_cache"
ARRAYS_DIR     = PROJECT_ROOT / "output" / "umap" / "arrays"
Q1_INDIV_NPZ   = ARRAYS_DIR / "q1_indiv.npz"
Q1_LABELS_NPY  = ARRAYS_DIR / "q1_cluster_labels.npy"

CSV_GROUPS = {
    "disagree_ero_gal_q1_star": INPUT_DIR / "disagree_ero_gal_q1_star.csv",
    "disagree_ero_star_q1_gal": INPUT_DIR / "disagree_ero_star_q1_gal.csv",
    "agree_both_galaxy":        INPUT_DIR / "agree_both_galaxy.csv",
    "agree_both_star":          INPUT_DIR / "agree_both_star.csv",
}

FIELD_TO_FILENAME = {
    "Horsehead":   "Euclid-VIS-ERO-Horsehead-Flattened.DR3.fits",
    "Messier 78":  "Euclid-VIS-ERO-Messier78-Flattened.DR3.fits",
    "Taurus":      "Euclid-VIS-ERO-Taurus-Flattened.DR3.fits",
    "Barnard 30":  "Euclid-VIS-ERO-Barnard30-Flattened.DR3.fits",
    "NGC 6254":    "Euclid-VIS-ERO-NGC6254-Flattened.DR3.fits",
    "NGC 6397":    "Euclid-VIS-ERO-NGC6397-Flattened.DR3.fits",
    "NGC 6822":    "Euclid-VIS-ERO-NGC6822-Flattened.DR3.fits",
    "Holmberg II": "Euclid-VIS-ERO-HolmbergII-Flattened.DR3.fits",
    "IC 10":       "Euclid-VIS-ERO-IC10-Flattened.DR3.fits",
    "IC 342":      "Euclid-VIS-ERO-IC342-Flattened.DR3.fits",
    "NGC 2403":    "Euclid-VIS-ERO-NGC2403-Flattened.DR3.fits",
    "NGC 6744":    "Euclid-VIS-ERO-NGC6744-Flattened.DR3.fits",
    "Fornax":      "Euclid-VIS-ERO-Fornax-Flattened.DR3.fits",
    "Dorado":      "Euclid-VIS-ERO-Dorado-Flattened.DR3.fits",
    "Perseus":     "Euclid-VIS-ERO-Perseus-Flattened.DR3.fits",
    "Abell 2390":  "Euclid-VIS-ERO-Abell2390-Flattened.DR3.fits",
    "Abell 2764":  "Euclid-VIS-ERO-Abell2764-Flattened.DR3.fits",
}

VIS_PIXEL_SCALE = 0.1  # arcsec / pixel


# ── FITS cache ────────────────────────────────────────────────────────────────

def get_cached_fits(field: str) -> Path | None:
    filename = FIELD_TO_FILENAME.get(field)
    if filename is None:
        print(f"  WARNING: unknown field '{field}'")
        return None
    path = FITS_CACHE / filename
    if not path.exists():
        print(f"  WARNING: FITS not in cache: {path.name}")
        print(f"  Run: python src/save_ero_fits.py --fields \"{field}\"")
        return None
    print(f"  [cache] {path.name}  ({path.stat().st_size / 1e9:.1f} GB)")
    return path


# ── Cutout sizing ─────────────────────────────────────────────────────────────

def dynamic_cutout_size(row: pd.Series, args: argparse.Namespace) -> float:
    if args.size_arcsec is not None:
        return args.size_arcsec
    radius_px = None
    for col in ("flux_radius", "v_flux_radius", "e_v_flux_radius"):
        if col in row.index and pd.notna(row[col]):
            try:
                radius_px = float(row[col])
                break
            except (ValueError, TypeError):
                pass
    if radius_px is not None and np.isfinite(radius_px) and radius_px > 0:
        size = radius_px * VIS_PIXEL_SCALE * args.size_scale
    else:
        size = args.size_max
    return float(np.clip(size, args.size_min, args.size_max))


# ── Cutout rendering (log stretch) ────────────────────────────────────────────

def make_cutout_image(fits_path: Path, ra: float, dec: float,
                      size_arcsec: float, title: str,
                      dpi: int = 75,
                      c_star: float | None = None) -> np.ndarray:
    import io
    from astropy.io import fits as afits
    from astropy.coordinates import SkyCoord
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    from astropy.visualization import LogStretch, ImageNormalize, ManualInterval
    import astropy.units as u

    with afits.open(fits_path, memmap=True) as hdul:
        hdu = None
        for h in hdul:
            if hasattr(h, "data") and isinstance(h.data, np.ndarray) and h.data.ndim == 2:
                hdu = h
                break
        if hdu is None:
            raise RuntimeError(f"No 2D image HDU found in {fits_path.name}")
        wcs    = WCS(hdu.header)
        coord  = SkyCoord(ra=ra, dec=dec, unit="deg")
        size   = u.Quantity([size_arcsec, size_arcsec], unit=u.arcsec)
        cutout = Cutout2D(hdu.data, coord, size, wcs=wcs)

    image  = cutout.data.astype(float)
    finite = np.isfinite(image)
    if not finite.any():
        raise RuntimeError("No finite pixels in cutout")

    # Fill NaN/inf pixels with the local median so they don't render as black
    # patches after log stretch (log stretch expands dynamic range enough to
    # reveal masked pixels that ZScale hid by saturating the nucleus to white).
    image = np.where(finite, image, float(np.nanmedian(image[finite])))
    finite = np.ones_like(image, dtype=bool)

    # ── Log stretch ───────────────────────────────────────────────────────────
    # Shift image so that the 1st-percentile maps to 1.0 (ensures all log inputs > 0).
    # The shift preserves relative brightness differences; log then compresses
    # bright nucleus spikes and lifts faint outer structure simultaneously.
    p01  = float(np.nanpercentile(image, 1.0))
    p995 = float(np.nanpercentile(image, 99.5))
    shift = -p01 + 1.0          # p01 → 1.0 after shift
    img_s = np.clip(image + shift, 1e-6, None)
    vmin_s = 1.0                # shifted p01
    vmax_s = max(p995 + shift, vmin_s * 10)

    norm = ImageNormalize(img_s, interval=ManualInterval(vmin_s, vmax_s),
                          stretch=LogStretch())

    # ── Crosshair colour by Gaia corrected BP/RP excess (c_star) ────────────
    # coolwarm: blue = point-like (star), red = extended (galaxy)
    if c_star is not None and np.isfinite(c_star):
        crosshair_color = plt.cm.coolwarm(np.clip((c_star + 0.5) / 1.5, 0, 1))
    else:
        crosshair_color = "white"

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_s, origin="lower", cmap="gray", norm=norm, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    n   = img_s.shape[0]
    gap = n * 0.10   # clear gap radius around nucleus (10 % of stamp width)
    cx  = cy = n / 2
    kw  = dict(color=crosshair_color, lw=1.0, alpha=0.8, solid_capstyle="butt")
    ax.plot([0,      cx - gap], [cy, cy],      **kw)   # left
    ax.plot([cx + gap, n - 1], [cy, cy],       **kw)   # right
    ax.plot([cx, cx], [0,      cy - gap],      **kw)   # bottom
    ax.plot([cx, cx], [cy + gap, n - 1],       **kw)   # top
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


# ── Contact sheet ─────────────────────────────────────────────────────────────

def save_cmap_legend(output_dir: Path) -> None:
    """Save colourbar legend explaining the BP/RP excess crosshair colour encoding."""
    fig, ax = plt.subplots(figsize=(4, 0.6))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.55, bottom=0.0)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap="coolwarm",
                              norm=plt.Normalize(vmin=0, vmax=1)),
        cax=ax, orientation="horizontal",
    )
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["point-like (star)", "intermediate", "extended (galaxy)"])
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("Crosshair colour = Gaia corrected BP/RP excess (c_star)",
                 fontsize=8, pad=4)
    out = output_dir / "crosshair_cmap_legend.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved cmap legend: {out}")


def make_contact_sheet(cell_images: list, out_stem: Path,
                       ncols: int = 5, max_rows: int = 6) -> None:
    if not cell_images:
        return
    per_page = ncols * max_rows
    pages = [cell_images[i:i + per_page] for i in range(0, len(cell_images), per_page)]
    for pi, page_imgs in enumerate(pages, start=1):
        out_path = (Path(str(out_stem) + ".png") if len(pages) == 1
                    else Path(str(out_stem) + f"_p{pi}.png"))
        nrows = (len(page_imgs) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                                 squeeze=False)
        for i, ax in enumerate(axes.flat):
            if i < len(page_imgs):
                ax.imshow(page_imgs[i])
            ax.axis("off")
        plt.suptitle(f"Page {pi} / {len(pages)}  ({len(page_imgs)} sources)",
                     fontsize=10, y=1.002)
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Contact sheet: {out_path}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def source_basename(row: pd.Series) -> str:
    field = sanitize(row.get("field", "unknown"))
    ra    = f"{float(row['ra']):.6f}"
    dec   = f"{float(row['dec']):+.6f}".replace("+", "p").replace("-", "m")
    return f"{field}_ra{ra}_dec{dec}"


# ── Cluster assignment ────────────────────────────────────────────────────────

def build_cluster_map() -> dict[str, int]:
    """Build {source_id_str → cluster_int} by projecting ERO sources into Q1 UMAP space.

    Loads ERO Gaia features from ero_indiv.npz, projects via the Q1 reducer,
    then assigns each source to the nearest Q1 cluster (nearest-neighbour in
    UMAP space among points with valid cluster labels ≥ 0).
    """
    import joblib
    from sklearn.neighbors import NearestNeighbors

    ero_npz_path = ARRAYS_DIR / "ero_indiv.npz"
    reducer_path = ARRAYS_DIR / "q1_indiv_reducer.joblib"
    for p in [ero_npz_path, Q1_INDIV_NPZ, Q1_LABELS_NPY, reducer_path]:
        if not p.exists():
            print(f"  WARNING: {p.name} not found — cluster info unavailable.")
            return {}

    ero_data  = np.load(ero_npz_path, allow_pickle=True)
    q1_data   = np.load(Q1_INDIV_NPZ, allow_pickle=True)
    q1_emb    = q1_data["embedding"]
    q1_mean   = q1_data["mean"]
    q1_std    = q1_data["std"]
    q1_labels = np.load(Q1_LABELS_NPY)
    reducer   = joblib.load(reducer_path)

    valid_mask = q1_labels >= 0
    if valid_mask.sum() == 0:
        print("  WARNING: No valid Q1 cluster points — cluster info unavailable.")
        return {}

    # Project ERO features into Q1 UMAP space
    X_ero = ero_data["X_raw"].astype(float)
    std_safe = np.where(q1_std > 0, q1_std, 1.0)
    X_scaled = (X_ero - q1_mean) / std_safe
    X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0.0)
    ero_emb = reducer.transform(X_scaled)

    # Assign to nearest valid Q1 cluster
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(q1_emb[valid_mask])
    _, idx = nn.kneighbors(ero_emb)
    idx = idx.ravel()
    clusters = q1_labels[np.where(valid_mask)[0][idx]].astype(int)

    result = dict(zip(ero_data["source_ids"].astype(str), clusters))
    unique_clusters = sorted(set(clusters))
    print(f"  Cluster assignments: {len(result):,} sources → clusters {unique_clusters}")
    return result


# ── Per-group processing ──────────────────────────────────────────────────────

def process_group(name: str, csv_path: Path, args: argparse.Namespace,
                  cluster_map: dict[str, int] | None = None) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    if name in ("disagree_ero_gal_q1_star", "agree_both_star"):
        df = df.sort_values(["xgb_q1_prob", "class_star"], ascending=[True, True])
    else:
        df = df.sort_values(["xgb_q1_prob", "class_star"], ascending=[False, False])
    df = df.reset_index(drop=True)

    group_dir = OUTPUT_DIR / name
    group_dir.mkdir(parents=True, exist_ok=True)

    # Assign cluster to each source
    if cluster_map:
        df["cluster"] = df["g_source_id"].astype(str).map(cluster_map).fillna(-1).astype(int)
    else:
        df["cluster"] = -1

    # Load FITS for all needed fields upfront
    needed_fields = df["field"].dropna().unique().tolist()
    print(f"\n[{name}]  {len(df):,} total sources, fields: {needed_fields}")
    fits_cache: dict[str, Path | None] = {f: get_cached_fits(f) for f in needed_fields}

    all_records: list[dict] = []

    cluster_ids = sorted(df["cluster"].unique())

    for k in cluster_ids:
        cl_label = f"cluster{k}" if k >= 0 else "noise"
        cl_df    = df[df["cluster"] == k].copy().reset_index(drop=True)

        # Per-cluster limit (take highest-confidence sources first — already sorted)
        if args.per_cell is not None and len(cl_df) > args.per_cell:
            cl_df = cl_df.head(args.per_cell)

        cl_dir       = group_dir / cl_label
        cutouts_subdir = cl_dir / "cutouts"
        cl_dir.mkdir(parents=True, exist_ok=True)
        cutouts_subdir.mkdir(exist_ok=True)

        cl_records: list[dict] = []
        cl_cell_imgs: list     = []
        local_idx = 0
        n_cl = len(cl_df)
        print(f"  [{cl_label}]  {n_cl} sources")

        for i, row in cl_df.iterrows():
            field  = row.get("field", "")
            base   = source_basename(row)
            status, message, cutout_file = "ok", "", ""

            fits_path = fits_cache.get(field)
            if fits_path is None:
                status, message = "skipped", f"no FITS for field '{field}'"
            elif not fits_path.exists():
                status, message = "error", "FITS file missing from cache"
            else:
                try:
                    size_arcsec = dynamic_cutout_size(row, args)
                    c_star_val  = row.get("c_star", None)
                    c_star_val  = float(c_star_val) if pd.notna(c_star_val) else None
                    sid = str(row.get('g_source_id', ''))
                    title = (f"{name}  {cl_label}\n"
                             f"RA={float(row['ra']):.4f}  Dec={float(row['dec']):+.4f}\n"
                             f"G={row['phot_g_mean_mag']:.2f}  id={sid}")
                    img = make_cutout_image(fits_path, float(row["ra"]), float(row["dec"]),
                                            size_arcsec, title, dpi=args.dpi,
                                            c_star=c_star_val)
                    # ── Save individual cutout PNG ────────────────────────────
                    indiv_path = cutouts_subdir / f"cutout_{local_idx:04d}.png"
                    plt.imsave(str(indiv_path), img)
                    cutout_file = f"cutouts/cutout_{local_idx:04d}.png"
                    cl_cell_imgs.append(img)
                    local_idx += 1
                except Exception as exc:
                    status, message = "error", str(exc)

            print(f"    {i+1:>4}/{n_cl}  {base}  {status}"
                  + (f"  — {message}" if message else ""))

            rec = {**row.to_dict(),
                   "source_id":   row.get("g_source_id", ""),   # alias for load_cutout_cache
                   "status":      status,
                   "message":     message,
                   "cutout_file": cutout_file}
            cl_records.append(rec)
            all_records.append(rec)

        # ── Per-cluster contact sheet + manifest ──────────────────────────────
        make_contact_sheet(cl_cell_imgs, cl_dir / "contact_sheet")
        pd.DataFrame(cl_records).to_csv(cl_dir / "manifest.csv", index=False)
        print(f"    → {len(cl_cell_imgs)} cutouts, contact_sheet.png saved")

    # ── Overall group manifest (all clusters combined) ────────────────────────
    pd.DataFrame(all_records).to_csv(group_dir / "manifest.csv", index=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Euclid ERO VIS cutouts — log stretch + c_star crosshair.")
    p.add_argument("--size-arcsec",  type=float, default=None,
                   help="Fixed cutout size in arcsec (overrides dynamic sizing).")
    p.add_argument("--size-min",     type=float, default=4.0)
    p.add_argument("--size-max",     type=float, default=8.0)
    p.add_argument("--size-scale",   type=float, default=8.0)
    p.add_argument("--per-cell",  type=int,   default=30,
                   help="Max sources per (group × cluster) cell (default: 30).")
    p.add_argument("--dpi",          type=int,   default=75)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Euclid ERO VIS cutouts — log stretch + c_star crosshair")
    print(f"  Per cell    : {args.per_cell}")
    print("=" * 65)

    save_cmap_legend(OUTPUT_DIR)

    print("\n[cluster] Building ERO → Q1 cluster map...")
    cluster_map = build_cluster_map()
    if not cluster_map:
        print("  (No cluster info — contact sheets will not be split by cluster)")

    for name, csv_path in CSV_GROUPS.items():
        process_group(name, csv_path, args, cluster_map=cluster_map)

    print(f"\nAll outputs saved under: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
