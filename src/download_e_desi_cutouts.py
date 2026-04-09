#!/usr/bin/env python3
"""
Download Euclid ERO VIS cutouts for ERO×DESI cross-matched sources,
organised by cat_value (ERO-Star vs ERO-Galaxy) and Q1 UMAP cluster.

Each source is projected into Q1 UMAP space (via ero_desi.npz) and assigned
to its nearest Q1 cluster.  Cutouts are rendered with log stretch and BP/RP
excess crosshair colouring, using cached ERO FITS tiles.

Category layout (from cat_value in ero_desi.npz):
  2 = ERO Star   (e_v_class_star >= 0.65)
  3 = ERO Galaxy  (e_v_class_star < 0.65)

Output layout
-------------
output/euclid_cutouts/ero_cutouts/ero_desi/
  {category}/
    cluster{k}/
      contact_sheet.png
      manifest.csv
      cutouts/
        cutout_NNNN.png

Reads:
  output/umap/arrays/ero_desi.npz
  output/umap/arrays/q1_indiv.npz
  output/umap/arrays/q1_indiv_reducer.joblib
  output/umap/arrays/q1_cluster_labels.npy
  output/euclid_ero_fits_cache/*.fits

Usage:
  python src/download_e_ero_desi_cutouts.py
  python src/download_e_ero_desi_cutouts.py --per-cell 20 --size-arcsec 8
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARRAYS_DIR   = PROJECT_ROOT / "output" / "umap" / "arrays"
OUTPUT_DIR   = PROJECT_ROOT / "output" / "euclid_cutouts" / "ero_cutouts" / "desi"
FITS_CACHE   = PROJECT_ROOT / "output" / "euclid_ero_fits_cache"

CATEGORY_DIRS = {
    2: "disagree_desi_gal_ero_star",
    3: "agree_both_gal",
}
CATEGORY_LABELS = {
    2: "Disagree: DESI Galaxy, ERO Star (cs >= 0.65)",
    3: "Agree: Both Galaxy (cs < 0.65)",
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

ERO_FIELDS = {
    "Horsehead":   (85.245,   -2.458),
    "Messier 78":  (86.755,    0.072),
    "Taurus":      (68.000,   28.000),
    "Barnard 30":  (83.830,   12.030),
    "NGC 6254":    (254.288, -4.101),
    "NGC 6397":    (265.175, -53.674),
    "NGC 6822":    (296.235, -14.803),
    "Holmberg II": (124.768,  70.717),
    "IC 10":       (5.087,    59.304),
    "IC 342":      (56.702,   68.096),
    "NGC 2403":    (114.214,  65.601),
    "NGC 6744":    (287.444, -63.857),
    "Fornax":      (54.000,  -35.500),
    "Dorado":      (64.014,  -55.780),
    "Perseus":     (49.950,   41.512),
    "Abell 2390":  (328.403,  17.695),
    "Abell 2764":  (5.080,   -60.616),
}

_FIELD_NAMES = list(ERO_FIELDS.keys())
_FIELD_RA    = np.radians([v[0] for v in ERO_FIELDS.values()])
_FIELD_DEC   = np.radians([v[1] for v in ERO_FIELDS.values()])

VIS_PIXEL_SCALE = 0.1  # arcsec / pixel


# ── Helpers ──────────────────────────────────────────────────────────────────

def nearest_field(ra_deg: float, dec_deg: float) -> str:
    ra  = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    a = (np.sin((_FIELD_DEC - dec) / 2) ** 2
         + np.cos(dec) * np.cos(_FIELD_DEC) * np.sin((_FIELD_RA - ra) / 2) ** 2)
    return _FIELD_NAMES[int(np.argmin(a))]


def get_cached_fits(field: str) -> Path | None:
    filename = FIELD_TO_FILENAME.get(field)
    if filename is None:
        return None
    path = FITS_CACHE / filename
    if not path.exists():
        print(f"  WARNING: FITS not in cache: {path.name}")
        return None
    return path


def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def assign_to_clusters(src_emb: np.ndarray,
                       q1_emb: np.ndarray,
                       q1_labels: np.ndarray) -> np.ndarray:
    valid_mask = q1_labels >= 0
    if valid_mask.sum() == 0:
        return np.full(len(src_emb), -1, dtype=int)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(q1_emb[valid_mask])
    _, idx = nn.kneighbors(src_emb)
    idx = idx.ravel()
    valid_indices = np.where(valid_mask)[0]
    return q1_labels[valid_indices[idx]]


# ── Cutout rendering (log stretch) ──────────────────────────────────────────

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

    image = np.where(finite, image, float(np.nanmedian(image[finite])))

    p01  = float(np.nanpercentile(image, 1.0))
    p995 = float(np.nanpercentile(image, 99.5))
    shift = -p01 + 1.0
    img_s = np.clip(image + shift, 1e-6, None)
    vmin_s = 1.0
    vmax_s = max(p995 + shift, vmin_s * 10)

    norm = ImageNormalize(img_s, interval=ManualInterval(vmin_s, vmax_s),
                          stretch=LogStretch())

    if c_star is not None and np.isfinite(c_star):
        crosshair_color = plt.cm.coolwarm(np.clip((c_star + 0.5) / 1.5, 0, 1))
    else:
        crosshair_color = "white"

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_s, origin="lower", cmap="gray", norm=norm, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    n   = img_s.shape[0]
    gap = n * 0.10
    cx  = cy = n / 2
    kw  = dict(color=crosshair_color, lw=1.0, alpha=0.8, solid_capstyle="butt")
    ax.plot([0,      cx - gap], [cy, cy],      **kw)
    ax.plot([cx + gap, n - 1], [cy, cy],       **kw)
    ax.plot([cx, cx], [0,      cy - gap],      **kw)
    ax.plot([cx, cx], [cy + gap, n - 1],       **kw)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


# ── Contact sheet ────────────────────────────────────────────────────────────

def save_cmap_legend(output_dir: Path) -> None:
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


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ERO VIS cutouts for ERO×DESI sources, by cat_value × cluster"
    )
    p.add_argument("--per-cell", type=int, default=30,
                   help="Max sources per (category × cluster) cell (default: 30).")
    p.add_argument("--size-arcsec", type=float, default=6.0,
                   help="VIS cutout size in arcsec (default: 6.0).")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--dpi",        type=int, default=75)
    p.add_argument("--no-cutouts", action="store_true",
                   help="Skip cutout downloads; just show cluster assignment stats.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ERO VIS cutouts for ERO×DESI sources — by cat_value × cluster")
    print("=" * 65)

    # ── 1. Load Q1 UMAP + reducer ────────────────────────────────────────────
    print("\n[1] Loading Q1 UMAP embedding and reducer...")
    q1_npz  = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    q1_emb  = q1_npz["embedding"]
    q1_mean = q1_npz["mean"]
    q1_std  = q1_npz["std"]
    reducer = joblib.load(ARRAYS_DIR / "q1_indiv_reducer.joblib")
    print(f"  Q1 embedding: {q1_emb.shape[0]:,} points")

    # ── 2. Load cluster labels ────────────────────────────────────────────────
    print("\n[2] Loading cluster labels...")
    labels_path = ARRAYS_DIR / "q1_cluster_labels.npy"
    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found.")
        print("  Run: python src/compute_q1_clusters.py")
        return 1
    q1_labels = np.load(labels_path)
    cluster_ids = sorted(set(q1_labels[q1_labels >= 0]))
    print(f"  Clusters: " + "  ".join(f"C{k}={(q1_labels==k).sum():,}" for k in cluster_ids))

    # ── 3. Load ERO-DESI data ────────────────────────────────────────────────
    print("\n[3] Loading ERO-DESI data from ero_desi.npz...")
    npz_path = ARRAYS_DIR / "ero_desi.npz"
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found.")
        print("  Run: PGUSER=siwei_zhao_2025 python src/analyze_umap_query.py")
        return 1

    ed = np.load(npz_path, allow_pickle=True)
    ed_X     = ed["X_features"].astype(float)
    cat_vals = ed["cat_value"].astype(int)
    sids     = ed["source_ids"].astype(str) if "source_ids" in ed else np.full(len(cat_vals), "")
    spectype = ed["spectype"].astype(str) if "spectype" in ed else np.full(len(cat_vals), "")
    gmag     = ed["phot_g_mean_mag"].astype(float) if "phot_g_mean_mag" in ed else None
    ra       = ed["ra"].astype(float) if "ra" in ed else None
    dec      = ed["dec"].astype(float) if "dec" in ed else None
    desi_z   = ed["desi_z"].astype(float) if "desi_z" in ed else None
    cls_star = ed["class_star"].astype(float) if "class_star" in ed else None
    c_star   = ed["corrected_bp_rp_excess"].astype(float) if "corrected_bp_rp_excess" in ed else None

    if ra is None or dec is None:
        print("ERROR: ero_desi.npz missing RA/Dec. Re-run analyze_umap_query.py.")
        return 1

    n_total = len(cat_vals)
    print(f"  {n_total:,} sources  (cat=2: {(cat_vals==2).sum():,},  cat=3: {(cat_vals==3).sum():,})")

    # ── 4. Project into UMAP space ───────────────────────────────────────────
    print("\n[4] Projecting into UMAP space...")
    std_safe = np.where(q1_std > 0, q1_std, 1.0)
    X_scaled = (ed_X - q1_mean) / std_safe
    X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0.0)
    emb = reducer.transform(X_scaled)
    print(f"  Projected {len(emb):,} sources")

    # ── 5. Assign to nearest Q1 cluster ──────────────────────────────────────
    print("\n[5] Assigning to nearest Q1 cluster...")
    clusters = assign_to_clusters(emb, q1_emb, q1_labels)

    # Assign nearest ERO field from coordinates
    fields = np.array([nearest_field(ra[i], dec[i]) for i in range(n_total)])

    # Build a DataFrame for convenience
    df = pd.DataFrame({
        "source_id": sids,
        "ra": ra, "dec": dec,
        "cat_value": cat_vals,
        "spectype": spectype,
        "class_star": cls_star if cls_star is not None else np.nan,
        "desi_z": desi_z if desi_z is not None else np.nan,
        "corrected_bp_rp_excess": c_star if c_star is not None else np.nan,
        "phot_g_mean_mag": gmag if gmag is not None else np.nan,
        "umap1": emb[:, 0], "umap2": emb[:, 1],
        "cluster": clusters,
        "field": fields,
    })

    print("\n  Cluster assignment summary:")
    for v in sorted(CATEGORY_DIRS.keys()):
        lbl = CATEGORY_LABELS.get(v, str(v))
        sub = df[df["cat_value"] == v]
        if len(sub) == 0:
            continue
        print(f"  cat={v} {lbl}:")
        for k in cluster_ids:
            n = (sub["cluster"] == k).sum()
            if n:
                print(f"    C{k}: {n:,}")

    if args.no_cutouts:
        print("\n[6] --no-cutouts set, skipping downloads.")
        return 0

    # ── 6. Download cutouts per (category × cluster) ─────────────────────────
    print("\n[6] Downloading cutouts per (category × cluster) cell...")
    save_cmap_legend(OUTPUT_DIR)
    rng = np.random.default_rng(args.seed)

    for cat_val in sorted(CATEGORY_DIRS.keys()):
        cat_label = CATEGORY_LABELS[cat_val]
        cat_dir   = OUTPUT_DIR / CATEGORY_DIRS[cat_val]
        df_cat    = df[df["cat_value"] == cat_val]
        if len(df_cat) == 0:
            print(f"\n  cat={cat_val} {cat_label}: no sources, skipping.")
            continue

        print(f"\n  -- cat={cat_val} {cat_label} -> {CATEGORY_DIRS[cat_val]} "
              f"({len(df_cat):,} sources) --")

        # Sort: stars by class_star desc, galaxies by class_star asc
        if cat_val == 2:
            df_cat = df_cat.sort_values("class_star", ascending=False)
        else:
            df_cat = df_cat.sort_values("class_star", ascending=True)

        # Pre-cache FITS for all needed fields
        needed_fields = [f for f in df_cat["field"].unique() if f in FIELD_TO_FILENAME]
        fits_map: dict[str, Path | None] = {f: get_cached_fits(f) for f in needed_fields}

        for k in cluster_ids:
            cell_df = df_cat[df_cat["cluster"] == k]
            if len(cell_df) == 0:
                continue

            if len(cell_df) > args.per_cell:
                cell_df = cell_df.sample(args.per_cell,
                                          random_state=int(rng.integers(1e6)))
            cell_df = cell_df.reset_index(drop=True)

            cluster_dir    = cat_dir / f"cluster{k}"
            cutouts_subdir = cluster_dir / "cutouts"
            cluster_dir.mkdir(parents=True, exist_ok=True)
            cutouts_subdir.mkdir(exist_ok=True)

            n_cell = len(cell_df)
            print(f"\n    [cluster{k}]  {n_cell} sources")

            records:     list[dict] = []
            cell_images: list       = []
            local_idx = 0

            for idx, row in cell_df.iterrows():
                field = row["field"]
                status, message, cutout_file = "ok", "", ""

                fits_path = fits_map.get(field)
                if fits_path is None:
                    status, message = "skipped", f"no FITS for field '{field}'"
                else:
                    try:
                        c_val = row["corrected_bp_rp_excess"]
                        c_val = float(c_val) if pd.notna(c_val) and np.isfinite(c_val) else None
                        g_str = f"G={row['phot_g_mean_mag']:.2f}" if pd.notna(row['phot_g_mean_mag']) else ""
                        sid = str(row.get('g_source_id', row.get('source_id', '')))
                        title = (f"{cat_label}  cluster{k}\n"
                                 f"RA={float(row['ra']):.4f}  Dec={float(row['dec']):+.4f}\n"
                                 f"{g_str}  id={sid}")
                        img = make_cutout_image(
                            fits_path, float(row["ra"]), float(row["dec"]),
                            args.size_arcsec, title, dpi=args.dpi, c_star=c_val,
                        )
                        indiv_path = cutouts_subdir / f"cutout_{local_idx:04d}.png"
                        plt.imsave(str(indiv_path), img)
                        cutout_file = f"cutouts/cutout_{local_idx:04d}.png"
                        cell_images.append(img)
                        local_idx += 1
                    except Exception as exc:
                        status, message = "error", str(exc)

                ra_s  = f"{float(row['ra']):.4f}"
                dec_s = f"{float(row['dec']):+.4f}"
                print(f"      {idx+1:>4}/{n_cell}  {field} ({ra_s},{dec_s})  {status}"
                      + (f"  -- {message}" if message else ""))

                records.append({**row.to_dict(), "status": status,
                                "message": message, "cutout_file": cutout_file})

            make_contact_sheet(cell_images, cluster_dir / "contact_sheet")
            pd.DataFrame(records).to_csv(cluster_dir / "manifest.csv", index=False)
            print(f"      -> {len(cell_images)} cutouts saved")

    print("\n" + "=" * 65)
    print(f"Done. Outputs in: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
