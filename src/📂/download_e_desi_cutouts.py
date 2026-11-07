#!/usr/bin/env python3
"""
Download Euclid ERO VIS cutouts for ERO×DESI cross-matched sources,
organised by cat_value (ERO-Star vs ERO-Galaxy) and Q1 UMAP cluster.

Each source is projected into Q1 UMAP space (via ero_desi.npz) and assigned
to its nearest Q1 cluster.  Cutouts are rendered with log stretch, a scale bar,
and BP/RP excess crosshair colouring, using cached ERO FITS tiles.

Pass --color to produce RGB composites (R=H, G=J, B=VIS) using Lupton arcsinh
stretch.  NISP-J and NISP-H tiles are downloaded and cached automatically.

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
  python src/download_e_desi_cutouts.py                  # color composites, 30 per cell
  python src/download_e_desi_cutouts.py --per-cell 10    # fewer sources per cell
  python src/download_e_desi_cutouts.py --no-color       # grayscale VIS only
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
import xgboost as xgb
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT     = Path(__file__).resolve().parents[1]
ARRAYS_DIR       = PROJECT_ROOT / "output" / "umap" / "arrays"
SEED_MODELS_DIR  = PROJECT_ROOT / "output" / "xgboost_q1" / "seed_models"
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

VIS_PIXEL_SCALE  = 0.1   # arcsec / pixel
NISP_PIXEL_SCALE = 0.3   # arcsec / pixel

# Dynamic cutout sizing: size_arcsec = flux_radius (px) * VIS_PIXEL_SCALE * SIZE_SCALE
SIZE_SCALE = 8.0   # show 8× the half-light radius
SIZE_MIN   = 5.0   # arcsec — never smaller than this (50 VIS pixels min)
SIZE_MAX   = 15.0  # arcsec — cap to avoid pulling in bright neighbors

CDN_BASE = "https://cdn.euclid.esac.esa.int/Stack"

# NISP-J and NISP-H filename maps (same TARGET tokens as VIS)
FIELD_TO_TARGET = {
    "Horsehead":   "ERO-Horsehead",
    "Messier 78":  "ERO-Messier78",
    "Taurus":      "ERO-Taurus",
    "Barnard 30":  "ERO-Barnard30",
    "NGC 6254":    "ERO-NGC6254",
    "NGC 6397":    "ERO-NGC6397",
    "NGC 6822":    "ERO-NGC6822",
    "Holmberg II": "ERO-HolmbergII",
    "IC 10":       "ERO-IC10",
    "IC 342":      "ERO-IC342",
    "NGC 2403":    "ERO-NGC2403",
    "NGC 6744":    "ERO-NGC6744",
    "Fornax":      "ERO-Fornax",
    "Dorado":      "ERO-Dorado",
    "Perseus":     "ERO-Perseus",
    "Abell 2390":  "ERO-Abell2390",
    "Abell 2764":  "ERO-Abell2764",
}


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


# ── NISP download helper ──────────────────────────────────────────────────────

def _find_tar_entry(tar_url: str, pattern: str) -> tuple[str, int, int]:
    """Scan a remote tar via HTTP range requests to find an entry whose name
    matches *pattern* (case-insensitive substring).  Returns (name, data_offset,
    data_size).  Makes one 512-byte request per entry header — efficient for
    tars with a small number of large entries (e.g. Euclid NISP combined tar)."""
    import re, urllib.request

    def _fetch_header(byte_offset: int) -> bytes:
        hdrs = {"Range": f"bytes={byte_offset}-{byte_offset + 511}",
                "User-Agent": "euclid-fits-downloader"}
        req = urllib.request.Request(tar_url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()

    offset = 0
    long_name: str | None = None          # GNU long-name extension
    while True:
        block = _fetch_header(offset)
        if len(block) < 512 or block == b"\x00" * 512:
            break  # end of archive

        raw_name  = block[0:100].rstrip(b"\x00").decode("utf-8", errors="replace")
        typeflag  = block[156:157]
        size_oct  = block[124:136].rstrip(b"\x00 ").decode("ascii", errors="replace")
        size      = int(size_oct, 8) if size_oct else 0
        padded    = ((size + 511) // 512) * 512

        if typeflag == b"L":
            # GNU long-name: next 'size' bytes are the real name, then the actual entry
            name_hdrs = {"Range": f"bytes={offset+512}-{offset+512+size-1}",
                         "User-Agent": "euclid-fits-downloader"}
            name_req  = urllib.request.Request(tar_url, headers=name_hdrs)
            with urllib.request.urlopen(name_req, timeout=60) as r:
                long_name = r.read().rstrip(b"\x00").decode("utf-8", errors="replace")
            offset += 512 + padded
            continue

        name = long_name if long_name else raw_name
        long_name = None  # reset after use

        if typeflag not in (b"5", b"L", b"K") and size > 0:
            if re.search(pattern, name, re.IGNORECASE):
                return name, offset + 512, size

        offset += 512 + padded

    raise RuntimeError(f"No entry matching '{pattern}' found in tar: {tar_url}")


def _stream_download(url: str, dest: Path, start: int, size: int,
                     chunk: int = 4 << 20) -> None:
    import urllib.request
    headers = {"Range": f"bytes={start}-{start + size - 1}",
               "User-Agent": "euclid-fits-downloader"}
    req = urllib.request.Request(url, headers=headers)
    downloaded = 0
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        while True:
            block = r.read(chunk)
            if not block:
                break
            f.write(block)
            downloaded += len(block)
            print(f"    {downloaded/1e6:,.0f} / {size/1e6:,.0f} MB  "
                  f"({100*downloaded/size:.1f}%)", end="\r", flush=True)
    print()
    if downloaded != size:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Incomplete download: got {downloaded:,} / {size:,} bytes")


def get_nisp_fits(field: str, band: str) -> Path | None:
    """Return cached NISP FITS path for (field, band='J'|'H'|'Y'), downloading if needed.

    All three NISP bands live in a single combined tar:
      Euclid-NISP-Stack-ERO-{TARGET}.DR3.tar
    We scan the tar headers to find the entry matching the requested band and
    download only that entry's bytes via HTTP Range.
    """
    import gzip, shutil
    target = FIELD_TO_TARGET.get(field)
    if target is None:
        return None
    fits_path = FITS_CACHE / f"Euclid-NISP-{band}-{target}-Flattened.DR3.fits"
    if fits_path.exists():
        return fits_path
    # Combined NISP tar (contains Y, J, H band stacks + ancillary files)
    tar_url = f"{CDN_BASE}/Euclid-NISP-Stack-{target}.DR3.tar"
    gz_path = FITS_CACHE / f"Euclid-NISP-{band}-{target}-Flattened.DR3.fits.gz"
    # Pattern to locate the right band entry, e.g. "NISP-J-ERO-Fornax-Flattened"
    entry_pattern = rf"NISP-{band}-.*Flattened"
    print(f"  Scanning NISP tar for band {band} ({field})...")
    try:
        name, data_offset, data_size = _find_tar_entry(tar_url, entry_pattern)
        print(f"    Found: {name}  ({data_size/1e9:.2f} GB)  offset={data_offset:,}")
        print(f"  Downloading...")
        _stream_download(tar_url, gz_path, data_offset, data_size)
        print(f"  Decompressing {gz_path.name}...")
        with gzip.open(gz_path, "rb") as f_in, open(fits_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=32 << 20)
        gz_path.unlink()
        print(f"  -> {fits_path.name}")
        return fits_path
    except Exception as exc:
        print(f"  WARNING: NISP-{band} download failed for {field}: {exc}")
        gz_path.unlink(missing_ok=True)
        return None


def _extract_fits_cutout(fits_path: Path, ra: float, dec: float,
                         size_arcsec: float,
                         max_nan_frac: float = 0.30) -> np.ndarray:
    """Extract a square cutout from a FITS file, returned as float array.

    Uses mode='partial' so edge sources get NaN-padded rather than silently
    trimmed.  If more than *max_nan_frac* of pixels are NaN (source near tile
    edge), raises RuntimeError so the caller can fall back or skip.
    """
    from astropy.io import fits as afits
    from astropy.coordinates import SkyCoord
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    import astropy.units as u

    with afits.open(fits_path, memmap=True) as hdul:
        hdu = next((h for h in hdul
                    if hasattr(h, "data") and isinstance(h.data, np.ndarray)
                    and h.data.ndim == 2), None)
        if hdu is None:
            raise RuntimeError(f"No 2D HDU in {fits_path.name}")
        wcs    = WCS(hdu.header)
        coord  = SkyCoord(ra=ra, dec=dec, unit="deg")
        size   = u.Quantity([size_arcsec, size_arcsec], unit=u.arcsec)
        cutout = Cutout2D(hdu.data, coord, size, wcs=wcs,
                          mode="partial", fill_value=np.nan)
    data = cutout.data.astype(float)
    nan_frac = np.isnan(data).mean()
    if nan_frac > max_nan_frac:
        raise RuntimeError(
            f"Source near tile edge: {nan_frac:.0%} of cutout is off-tile "
            f"(>{max_nan_frac:.0%} threshold)"
        )
    return data


# ── LRGB composite ───────────────────────────────────────────────────────────

def _lrgb_composite(vis: np.ndarray,
                    nisp_j: np.ndarray,
                    nisp_h: np.ndarray,
                    colour_alpha: float = 0.25) -> np.ndarray:
    """LRGB composite via luminance-layer blending.

    VIS log-stretch provides the full-fidelity luminance base (identical to
    grayscale mode — no structural information is lost).  Smoothed NISP H/J
    colour channels are then blended on top at low opacity so that only
    large-scale warm/cool gradients appear.

    Steps:
      1. Log-stretch VIS → lum [0,1]  (base layer, R=G=B=lum)
      2. Smooth NISP H, J to NISP's native resolution (sigma ≈ 9 VIS px)
      3. Build a colour-tinted image: rescale each channel so its luminance
         matches the VIS luminance (no brightness change, only hue added)
      4. Alpha-blend:  result = (1-α)·gray + α·colour_tinted
         Background pixels are naturally dark in both layers → no noise issue.

    Returns (H, W, 3) float32 in [0, 1].
    """
    from scipy.ndimage import gaussian_filter
    from astropy.visualization import LogStretch, ImageNormalize, ManualInterval

    # ── 1. VIS log luminance — identical to grayscale mode ────────────────────
    vis_f  = np.where(np.isfinite(vis), vis, 0.0)
    p01    = float(np.nanpercentile(vis_f, 1.0))
    p995   = float(np.nanpercentile(vis_f, 99.5))
    shift  = -p01 + 1.0
    vis_s  = np.clip(vis_f + shift, 1e-6, None)
    vmax_s = max(p995 + shift, 10.0)
    norm   = ImageNormalize(vis_s,
                            interval=ManualInterval(1.0, vmax_s),
                            stretch=LogStretch())
    lum    = norm(vis_s).filled(0).astype(float)   # [0, 1]

    # ── 2. Base layer: pure VIS grayscale ─────────────────────────────────────
    gray = np.stack([lum, lum, lum], axis=-1)

    # ── 3. Colour layer: smoothed NISP, brightness-matched to VIS ─────────────
    # Smooth to NISP's native resolution (3 NISP px × 3 upsample = 9 VIS px)
    smooth_sigma = 9.0

    def _lin(img, lo_pct=1.0, hi_pct=99.0):
        img = np.where(np.isfinite(img), img, 0.0)
        lo  = float(np.nanpercentile(img, lo_pct))
        hi  = float(np.nanpercentile(img, hi_pct))
        return np.clip((img - lo) / (hi - lo + 1e-30), 0, 1)

    r_raw = gaussian_filter(_lin(nisp_h), sigma=smooth_sigma)  # H → warm/red
    g_raw = gaussian_filter(_lin(nisp_j), sigma=smooth_sigma)  # J → mid/green
    b_raw = gaussian_filter(_lin(vis_f),  sigma=smooth_sigma)  # VIS → cool/blue

    colour = np.stack([r_raw, g_raw, b_raw], axis=-1).clip(0, 1)

    # Rescale each pixel's colour so its perceived luminance equals VIS lum.
    # Perceived lum = 0.299R + 0.587G + 0.114B  (Rec.601 weights)
    colour_lum = (0.299 * colour[..., 0]
                + 0.587 * colour[..., 1]
                + 0.114 * colour[..., 2]) + 1e-10
    scale = lum / colour_lum           # per-pixel brightness correction
    colour_matched = (colour * scale[..., None]).clip(0, 1)

    # ── 4. Alpha blend: gray base + colour tint ───────────────────────────────
    result = (1.0 - colour_alpha) * gray + colour_alpha * colour_matched
    return result.clip(0, 1).astype(np.float32)


def predict_pgal(X: np.ndarray) -> np.ndarray:
    """Run the Q1 XGBoost seed ensemble and return mean P(galaxy) for each row."""
    seed_paths = sorted(SEED_MODELS_DIR.glob("xgb_classifier_q1_seed*.json"))
    if not seed_paths:
        # Fall back to single model
        single = PROJECT_ROOT / "output" / "xgboost_q1" / "xgb_classifier_q1.json"
        if single.exists():
            seed_paths = [single]
        else:
            print("  WARNING: no XGBoost model found — P(gal) will be NaN")
            return np.full(len(X), np.nan)
    probs = []
    for p in seed_paths:
        m = xgb.XGBClassifier()
        m.load_model(str(p))
        probs.append(m.predict_proba(X.astype(np.float32))[:, 1])
    return np.mean(probs, axis=0)


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


# ── Cutout rendering ─────────────────────────────────────────────────────────

def _add_scale_bar(ax: "plt.Axes", n_pix: int, size_arcsec: float,
                   pixel_scale: float, color: str = "white") -> None:
    """Draw a scale bar in the bottom-left corner of the cutout axes.

    Chooses the smallest round bar length (0.5 / 1 / 2 / 5 / 10 / 20 arcsec)
    that occupies at least 15% of the cutout width.
    """
    bar_as = 20.0  # fallback for very large cutouts
    for candidate in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
        if candidate / size_arcsec >= 0.15:
            bar_as = candidate
            break
    import matplotlib.patheffects as pe
    bar_pix  = bar_as / pixel_scale
    x0 = n_pix * 0.06
    y0 = n_pix * 0.08
    outline = [pe.withStroke(linewidth=2.5, foreground="black")]
    ax.plot([x0, x0 + bar_pix], [y0, y0], color=color, lw=2.5,
            solid_capstyle="butt",
            path_effects=[pe.withStroke(linewidth=4.0, foreground="black")])
    label = f'{bar_as:.0f}"' if bar_as >= 1 else f'{bar_as:.1f}"'
    ax.text(x0 + bar_pix / 2, y0 + n_pix * 0.04, label,
            ha="center", va="bottom", color=color,
            fontsize=9, fontweight="bold", path_effects=outline)


def make_cutout_image(fits_path_vis: Path, ra: float, dec: float,
                      size_arcsec: float, title: str,
                      dpi: int = 75,
                      c_star: float | None = None,
                      fits_path_nisp_j: Path | None = None,
                      fits_path_nisp_h: Path | None = None,
                      draw_crosshair: bool = True) -> np.ndarray:
    """Render a cutout stamp.

    Grayscale (log stretch) by default.  When fits_path_nisp_j and
    fits_path_nisp_h are both provided, produces an RGB composite
    (R=H, G=J, B=VIS) using Lupton arcsinh stretch.

    A scale bar and crosshair are always overlaid.
    """
    import io
    from astropy.visualization import LogStretch, ImageNormalize, ManualInterval

    color_mode = (fits_path_nisp_j is not None and fits_path_nisp_h is not None)

    # ── Extract VIS cutout ────────────────────────────────────────────────────
    vis = _extract_fits_cutout(fits_path_vis, ra, dec, size_arcsec)
    finite = np.isfinite(vis)
    if not finite.any():
        raise RuntimeError("No finite pixels in VIS cutout")
    vis = np.where(finite, vis, float(np.nanmedian(vis[finite])))
    n = vis.shape[0]

    if c_star is not None and np.isfinite(c_star):
        crosshair_color = plt.cm.coolwarm_r(float(np.clip(c_star, 0, 1)))
    else:
        crosshair_color = "white"

    fig, ax = plt.subplots(figsize=(4, 4))

    if color_mode:
        # ── RGB composite: R=H, G=J, B=VIS ───────────────────────────────────
        try:
            nisp_j = _extract_fits_cutout(fits_path_nisp_j, ra, dec, size_arcsec,
                                          max_nan_frac=0.30)
            nisp_h = _extract_fits_cutout(fits_path_nisp_h, ra, dec, size_arcsec,
                                          max_nan_frac=0.30)
        except Exception as exc:
            print(f"      WARNING: NISP cutout failed ({exc}), falling back to grayscale")
            color_mode = False

        if color_mode:
            # Skip colour for very faint sources: NISP noise would degrade the image.
            # Estimate peak SNR from the central 25% of the VIS cutout.
            cy, cx = n // 2, n // 2
            cy0, cy1 = max(0, cy - n // 4), min(n, cy + n // 4)
            cx0, cx1 = max(0, cx - n // 4), min(n, cx + n // 4)
            center_vis = vis[cy0:cy1, cx0:cx1]
            vis_peak = float(np.nanpercentile(center_vis, 99.0))
            vis_bg   = float(np.nanpercentile(vis, 50.0))
            vis_rms  = float(np.nanstd(vis[vis < vis_bg + 3 * np.nanstd(vis)]))
            if vis_rms > 0 and (vis_peak - vis_bg) / vis_rms < 10.0:
                print(f"      (faint source — using grayscale, peak SNR "
                      f"{(vis_peak-vis_bg)/vis_rms:.1f} < 10)")
                color_mode = False

        if color_mode:
            # Upsample NISP (0.3"/pix) to VIS resolution (0.1"/pix)
            from scipy.ndimage import zoom
            scale = NISP_PIXEL_SCALE / VIS_PIXEL_SCALE   # = 3.0
            nisp_j_up = zoom(nisp_j, scale, order=3)
            nisp_h_up = zoom(nisp_h, scale, order=3)

            # Crop / pad to match VIS shape
            def _match_shape(arr, target):
                th, tw = target.shape
                h, w   = arr.shape
                arr = arr[:th, :tw]
                if arr.shape[0] < th or arr.shape[1] < tw:
                    pad = np.full(target.shape, np.nanmedian(arr))
                    pad[:arr.shape[0], :arr.shape[1]] = arr
                    arr = pad
                return arr

            nisp_j_up = _match_shape(nisp_j_up, vis)
            nisp_h_up = _match_shape(nisp_h_up, vis)

            rgb = _lrgb_composite(vis, nisp_j_up, nisp_h_up)
            ax.imshow(rgb, origin="lower", interpolation="nearest")

    if not color_mode:
        # ── Grayscale log stretch ─────────────────────────────────────────────
        p01   = float(np.nanpercentile(vis, 1.0))
        p995  = float(np.nanpercentile(vis, 99.5))
        shift = -p01 + 1.0
        img_s = np.clip(vis + shift, 1e-6, None)
        vmin_s = 1.0
        vmax_s = max(p995 + shift, vmin_s * 10)
        norm = ImageNormalize(img_s, interval=ManualInterval(vmin_s, vmax_s),
                              stretch=LogStretch())
        ax.imshow(img_s, origin="lower", cmap="gray", norm=norm,
                  interpolation="nearest")

    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    # ── Crosshair ─────────────────────────────────────────────────────────────
    gap = n * 0.10
    cx  = cy = n / 2
    if draw_crosshair:
        kw  = dict(color=crosshair_color, lw=1.0, alpha=0.8, solid_capstyle="butt")
        ax.plot([0,        cx - gap], [cy, cy],      **kw)
        ax.plot([cx + gap, n - 1],   [cy, cy],       **kw)
        ax.plot([cx, cx],  [0,        cy - gap],     **kw)
        ax.plot([cx, cx],  [cy + gap, n - 1],        **kw)

    # ── Scale bar ─────────────────────────────────────────────────────────────
    _add_scale_bar(ax, n, size_arcsec, VIS_PIXEL_SCALE, color="white")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


# ── Contact sheet ────────────────────────────────────────────────────────────

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
        # Fixed-height header: 0.55" regardless of figure height, so the
        # colorbar and suptitle look the same on tall (many-row) and short sheets.
        fig_h_in  = 5.0 * nrows
        header_in = 1.2                       # physical header height in inches
        top_frac  = 1.0 - header_in / fig_h_in
        _ih       = header_in / fig_h_in       # header as figure fraction

        fig.subplots_adjust(top=top_frac, bottom=0.01, left=0.01, right=0.99,
                            hspace=0.05, wspace=0.05)

        # ── P(gal) colorbar — left side of header, same height as suptitle ────
        # All sizes anchored to physical inches so every sheet looks identical.
        legend_w  = 0.28
        legend_h  = 0.15 / fig_h_in           # ~0.15" bar height → ~15 px at 100 dpi
        legend_x  = 0.07
        # Leave 0.20" of clearance above subplots for tick labels below the bar
        legend_y  = top_frac + 0.20 / fig_h_in

        # Suptitle anchored to bar centre so they always sit on the same line
        sup_y = legend_y + legend_h * 0.5

        plt.suptitle(f"Page {pi} / {len(pages)}  ({len(page_imgs)} sources)",
                     fontsize=10, y=sup_y)

        cbar_ax = fig.add_axes([legend_x, legend_y, legend_w, legend_h])
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap="coolwarm_r",
                                  norm=plt.Normalize(vmin=0, vmax=1)),
            cax=cbar_ax, orientation="horizontal",
        )
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["star", "0.5", "galaxy"])
        cbar.ax.tick_params(labelsize=7, pad=1)
        cbar_ax.set_title("XGBoost P(gal) — crosshair colour",
                           fontsize=7, pad=3, loc="left")

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
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--dpi",        type=int, default=75)
    p.add_argument("--no-cutouts", action="store_true",
                   help="Skip cutout downloads; just show cluster assignment stats.")
    p.add_argument("--no-color", action="store_true",
                   help="Skip NISP color composites; produce grayscale VIS cutouts only.")
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
        print("  Run: python src/analyze_umap_q1_clusters.py")
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
    cls_star   = ed["class_star"].astype(float) if "class_star" in ed else None
    c_star     = ed["corrected_bp_rp_excess"].astype(float) if "corrected_bp_rp_excess" in ed else None
    flux_radius = ed["flux_radius"].astype(float) if "flux_radius" in ed else None

    if ra is None or dec is None:
        print("ERROR: ero_desi.npz missing RA/Dec. Re-run analyze_umap_query.py.")
        return 1

    n_total = len(cat_vals)
    print(f"  {n_total:,} sources  (cat=2: {(cat_vals==2).sum():,},  cat=3: {(cat_vals==3).sum():,})")

    # ── XGBoost P(galaxy) predictions ───────────────────────────────────────
    print("\n  Running XGBoost ensemble predictions...")
    xgb_pgal = predict_pgal(ed_X)
    print(f"  P(gal): min={np.nanmin(xgb_pgal):.3f}  median={np.nanmedian(xgb_pgal):.3f}  max={np.nanmax(xgb_pgal):.3f}")

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
        "flux_radius": flux_radius if flux_radius is not None else np.nan,
        "xgb_pgal": xgb_pgal,
        "umap1": emb[:, 0], "umap2": emb[:, 1],
        "cluster": clusters,
        "field": fields,
    })
    _has_flux_radius = flux_radius is not None and np.any(np.isfinite(flux_radius))
    if not _has_flux_radius:
        print("  NOTE: flux_radius not in npz — using fixed --size-arcsec for all sources.")
        print("        Re-run query_umap_array.py with PGUSER to enable dynamic sizing.")

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

        # Pre-cache VIS FITS (and NISP if --color) for all needed fields
        needed_fields = [f for f in df_cat["field"].unique() if f in FIELD_TO_FILENAME]
        fits_map:   dict[str, Path | None] = {f: get_cached_fits(f) for f in needed_fields}
        nisp_j_map: dict[str, Path | None] = {}
        nisp_h_map: dict[str, Path | None] = {}
        if not args.no_color:
            print(f"    Pre-caching NISP-J and NISP-H for {len(needed_fields)} field(s)...")
            for f in needed_fields:
                nisp_j_map[f] = get_nisp_fits(f, "J")
                nisp_h_map[f] = get_nisp_fits(f, "H")

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
                        pgal  = row["xgb_pgal"]
                        c_val = float(pgal) if pd.notna(pgal) and np.isfinite(float(pgal)) else None
                        g_str = f"G={row['phot_g_mean_mag']:.2f}" if pd.notna(row['phot_g_mean_mag']) else ""
                        sid = str(row.get('g_source_id', row.get('source_id', '')))

                        # Dynamic cutout size from flux_radius when available
                        fr = row.get("flux_radius", np.nan)
                        if pd.notna(fr) and np.isfinite(float(fr)) and float(fr) > 0:
                            size = float(np.clip(
                                float(fr) * VIS_PIXEL_SCALE * SIZE_SCALE,
                                SIZE_MIN, SIZE_MAX,
                            ))
                        else:
                            size = SIZE_MIN

                        title = (f"{cat_label}  cluster{k}\n"
                                 f"RA={float(row['ra']):.4f}  Dec={float(row['dec']):+.4f}\n"
                                 f"{g_str}  id={sid}")
                        img = make_cutout_image(
                            fits_path, float(row["ra"]), float(row["dec"]),
                            size, title, dpi=args.dpi, c_star=c_val,
                            fits_path_nisp_j=nisp_j_map.get(field) if not args.no_color else None,
                            fits_path_nisp_h=nisp_h_map.get(field) if not args.no_color else None,
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
