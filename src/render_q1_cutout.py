#!/usr/bin/env python3
"""
Euclid Q1 VIS cutout utility — importable module.

Exposes one public function:

    make_q1_cutouts(df, output_dir, ...)

Any script that produces a source list (DataFrame with 'ra' and 'dec' columns)
can call this directly.  Tile FITS files are downloaded on demand and cached to
output/euclid_q1_fits_cache/{field_name}/ — subsequent calls reuse already-cached tiles.

Tile discovery uses the IRSA TAP service to query euclid_q1_mer_catalogue:
  https://irsa.ipac.caltech.edu/TAP/sync?QUERY=SELECT+TOP+1+tileid+FROM+euclid_q1_mer_catalogue+...

FITS files are downloaded directly from IRSA:
  https://irsa.ipac.caltech.edu/ibe/data/euclid/q1/MER/{tile_id}/VIS/{filename}

Can also be run standalone for testing:
  python src/download_q1_utils.py --csv path/to/sources.csv --out path/to/output/
"""

from __future__ import annotations

import argparse
import html.parser
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np
import pandas as pd

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
FITS_CACHE      = PROJECT_ROOT / "output" / "euclid_q1_fits_cache"
ARRAYS_DIR      = PROJECT_ROOT / "output" / "umap" / "arrays"
SEED_MODELS_DIR = PROJECT_ROOT / "output" / "xgboost_q1" / "seed_models"

# ── Companion marker colors (edit here to affect all downstream cutout scripts) ──
GAIA_RING     = "#1e90ff"   # bright blue  — Gaia matched within 1" of WISE (ring)
GAIA_FOV_RING = "#90ee90"   # light green  — Gaia in stamp FOV but outside 1" (ring)
EUCLID_RING   = "red"       # red          — secondary Euclid within 1" (crosshair)
WISE_RING     = "#f032e6"   # magenta      — WISE source centroid (x marker)

IRSA_TAP    = "https://irsa.ipac.caltech.edu/TAP/sync"
IRSA_IBE_DATA = "https://irsa.ipac.caltech.edu/ibe/data/euclid/q1/MER"

VIS_PIXEL_SCALE  = 0.1   # arcsec / pixel (Euclid VIS)
NISP_PIXEL_SCALE = 0.3   # arcsec / pixel (Euclid NISP)


# ── Tile info ─────────────────────────────────────────────────────────────────

class TileInfo(NamedTuple):
    tile_id:   str    # 9-digit string, e.g. "102018211"
    filename:  str    # e.g. "EUC_MER_BGSUB-MOSAIC-VIS_TILE102018211-ADF1FF_...fits"
    url:       str    # full IRSA download URL
    field_name: str   # human-readable field name, e.g. "EDF-Fornax"


# ── Batch tile discovery (one query per field group) ──────────────────────────

def _discover_tiles(ra_min: float, ra_max: float,
                    dec_min: float, dec_max: float,
                    field_name: str,
                    margin: float = 0.3) -> list[dict]:
    """
    Query ESA Euclid TAP for all VIS MER tiles overlapping a bounding box.
    Returns a list of dicts: tile_id, filename, url, corners (4×2 array), bbox.
    One call covers all sources in a field — far fewer round-trips than per-source.
    Falls back to per-source IRSA TAP if astroquery is unavailable.
    """
    polygon = (f"{ra_min-margin},{dec_min-margin}, "
               f"{ra_max+margin},{dec_min-margin}, "
               f"{ra_max+margin},{dec_max+margin}, "
               f"{ra_min-margin},{dec_max+margin}")
    try:
        from astroquery.esa.euclid import Euclid
        job = Euclid.launch_job(
            f"SELECT file_name, tile_index, fov FROM q1.mosaic_product "
            f"WHERE instrument_name='VIS' "
            f"AND INTERSECTS(POLYGON('ICRS', {polygon}), fov)=1"
        )
        t = job.get_results()
        tiles = []
        for row in t:
            vals    = [float(x) for x in str(row["fov"]).strip("()").split(",")]
            corners = np.array(vals).reshape(4, 2)
            tile_id  = str(int(row["tile_index"]))
            filename = str(row["file_name"]).strip()
            url      = f"{IRSA_IBE_DATA}/{tile_id}/VIS/{filename}"
            tiles.append({
                "tile_id":  tile_id,
                "filename": filename,
                "url":      url,
                "field_name": field_name,
                "corners":  corners,
                "ra_min":   corners[:, 0].min(),
                "ra_max":   corners[:, 0].max(),
                "dec_min":  corners[:, 1].min(),
                "dec_max":  corners[:, 1].max(),
            })
        return tiles
    except Exception as exc:
        print(f"    WARNING: ESA TAP batch query failed ({exc}); falling back to per-source IRSA TAP.")
        return []   # caller will fall back to _query_tile_irsa per source


def _point_in_polygon(ra: float, dec: float, corners: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test for a 4-corner tile footprint."""
    n = len(corners)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = corners[i]
        xj, yj = corners[j]
        if ((yi > dec) != (yj > dec)) and (ra < (xj - xi) * (dec - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _assign_to_tiles(ra: float, dec: float,
                     tiles: list[dict], field_name: str) -> TileInfo | None:
    """Assign a single (ra, dec) to its tile using bbox pre-filter + polygon test."""
    for tile in tiles:
        if (tile["ra_min"] <= ra <= tile["ra_max"] and
                tile["dec_min"] <= dec <= tile["dec_max"] and
                _point_in_polygon(ra, dec, tile["corners"])):
            return TileInfo(tile_id=tile["tile_id"], filename=tile["filename"],
                            url=tile["url"], field_name=field_name)
    return None


# ── Per-source IRSA TAP fallback ──────────────────────────────────────────────

# Cache: tile_id -> filename so we only scrape each tile directory once
_tile_filename_cache: dict[str, str | None] = {}


def _vis_filename(tile_id: str, retries: int = 3) -> str | None:
    """Scrape the IRSA VIS directory for a tile to get the BGSUB-MOSAIC-VIS filename."""
    if tile_id in _tile_filename_cache:
        return _tile_filename_cache[tile_id]
    dir_url = f"{IRSA_IBE_DATA}/{tile_id}/VIS/"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(dir_url, headers={"User-Agent": "euclid-q1-cutout"})
            with urllib.request.urlopen(req, timeout=30) as r:
                html = r.read().decode("utf-8", errors="replace")
            m = re.search(r'href="(EUC_MER_BGSUB-MOSAIC-VIS_TILE[^"]+\.fits)"', html)
            filename = m.group(1) if m else None
            _tile_filename_cache[tile_id] = filename
            return filename
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    WARNING: directory listing failed for tile {tile_id}: {exc}")
                _tile_filename_cache[tile_id] = None
                return None


_nisp_filename_cache: dict[tuple[str, str], str | None] = {}


def _nisp_filename(tile_id: str, band: str, retries: int = 3) -> str | None:
    """Scrape the IRSA NISP/ directory for a tile to get the BGSUB-MOSAIC-NIR-{band} filename.

    All NISP bands (J, H, Y) live together under a single NISP/ subdirectory.
    Filename pattern: EUC_MER_BGSUB-MOSAIC-NIR-{band}_TILE...fits
    """
    key = (tile_id, band)
    if key in _nisp_filename_cache:
        return _nisp_filename_cache[key]
    dir_url = f"{IRSA_IBE_DATA}/{tile_id}/NISP/"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(dir_url, headers={"User-Agent": "euclid-q1-cutout"})
            with urllib.request.urlopen(req, timeout=30) as r:
                html_text = r.read().decode("utf-8", errors="replace")
            pattern = rf'href="(EUC_MER_BGSUB-MOSAIC-NIR-{band}_TILE[^"]+\.fits)"'
            m = re.search(pattern, html_text)
            filename = m.group(1) if m else None
            _nisp_filename_cache[key] = filename
            return filename
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                _nisp_filename_cache[key] = None
                return None


def _query_tile_irsa(ra: float, dec: float, field_name: str) -> TileInfo | None:
    """Per-source fallback: IRSA TAP cone search + directory scrape for filename."""
    query = (
        f"SELECT tileid FROM euclid_q1_mer_catalogue "
        f"WHERE CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, 0.02)) = 1"
    )
    url = IRSA_TAP + "?QUERY=" + urllib.parse.quote(query) + "&FORMAT=json"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "euclid-q1-cutout"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            rows = data["VOTABLE"]["RESOURCE_ARRAY"][0]["TABLE"]["DATA"]["TABLEDATA"]
            ids = [r[0] for r in rows if r[0]]
            if not ids:
                return None
            tile_id = max(set(ids), key=ids.count)
            break
        except Exception as exc:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"    WARNING: IRSA TAP failed for RA={ra:.4f} Dec={dec:.4f}: {exc}")
                return None

    filename = _vis_filename(tile_id)
    if filename is None:
        print(f"    WARNING: no BGSUB-MOSAIC-VIS file in tile {tile_id}")
        return None
    return TileInfo(tile_id=tile_id, filename=filename,
                    url=f"{IRSA_IBE_DATA}/{tile_id}/VIS/{filename}",
                    field_name=field_name)


# ── XGBoost P(gal) ────────────────────────────────────────────────────────────

def predict_pgal(X: np.ndarray) -> np.ndarray:
    """Mean P(galaxy) from the 5-seed XGBoost ensemble in seed_models/."""
    import xgboost as xgb
    seed_paths = sorted(SEED_MODELS_DIR.glob("xgb_classifier_q1_seed*.json"))
    if not seed_paths:
        return np.full(len(X), np.nan)
    probs = []
    for p in seed_paths:
        m = xgb.XGBClassifier()
        m.load_model(str(p))
        probs.append(m.predict_proba(X.astype(np.float32))[:, 1])
    return np.mean(probs, axis=0)


# ── LRGB colour composite ─────────────────────────────────────────────────────

def _lrgb_composite(vis, nisp_j, nisp_h, colour_alpha: float = 0.25) -> np.ndarray:
    """Blend VIS (luminance) with NISP J/H (colour) into an RGB image array."""
    from scipy.ndimage import gaussian_filter
    from astropy.visualization import LogStretch, ImageNormalize, ManualInterval

    vis_f = np.where(np.isfinite(vis), vis, 0.0)
    p01   = float(np.nanpercentile(vis_f, 1.0))
    p995  = float(np.nanpercentile(vis_f, 99.5))
    vis_s = np.clip(vis_f - p01 + 1.0, 1e-6, None)
    norm  = ImageNormalize(vis_s, interval=ManualInterval(1.0, max(p995 - p01 + 1.0, 10.0)),
                           stretch=LogStretch())
    lum   = norm(vis_s).filled(0).astype(float)
    gray  = np.stack([lum, lum, lum], axis=-1)

    def _lin(img):
        img = np.where(np.isfinite(img), img, 0.0)
        lo, hi = float(np.nanpercentile(img, 1)), float(np.nanpercentile(img, 99))
        return np.clip((img - lo) / (hi - lo + 1e-30), 0, 1)

    r_raw = gaussian_filter(_lin(nisp_h), sigma=9.0)
    g_raw = gaussian_filter(_lin(nisp_j), sigma=9.0)
    b_raw = gaussian_filter(_lin(vis_f),  sigma=9.0)
    colour = np.stack([r_raw, g_raw, b_raw], axis=-1).clip(0, 1)
    colour_lum = (0.299 * colour[..., 0] + 0.587 * colour[..., 1]
                  + 0.114 * colour[..., 2]) + 1e-10
    colour_matched = (colour * (lum / colour_lum)[..., None]).clip(0, 1)
    return ((1.0 - colour_alpha) * gray + colour_alpha * colour_matched).clip(0, 1).astype(np.float32)


# ── Scale bar ─────────────────────────────────────────────────────────────────

def _add_scale_bar(ax: "plt.Axes", n_pix: int, size_arcsec: float,
                   pixel_scale: float, color: str) -> None:
    """Draw an adaptive scale bar with a stroke-outlined label."""
    candidates = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    bar_as = candidates[0]
    for candidate in candidates:
        if candidate / size_arcsec >= 0.15:
            bar_as = candidate
            break
    bar_px = bar_as / pixel_scale
    x0 = n_pix * 0.05
    y0 = n_pix * 0.08
    ax.plot([x0, x0 + bar_px], [y0, y0], color=color, lw=2.0,
            solid_capstyle="butt")
    label = f'{bar_as:.4g}"' if bar_as < 1 else f'{bar_as:.4g}"'
    txt = ax.text(x0 + bar_px / 2, y0 + n_pix * 0.04, label,
                  color=color, fontsize=9, ha="center", va="bottom")
    txt.set_path_effects([
        patheffects.Stroke(linewidth=2.5, foreground="black"),
        patheffects.Normal(),
    ])


# ── HTTP range-request helpers (avoids downloading full ~1.4 GB tile) ─────────

class _HttpRangeFile:
    """Seekable file-like object backed by HTTP range requests.

    Allows astropy to open a remote FITS and read only the bytes it needs
    (header + cutout pixel rows).  Each 6" stamp fetches ~3–5 MB instead of
    the full 1.4 GB tile.
    """
    def __init__(self, url: str):
        self.url  = url
        self._pos = 0
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "euclid-q1-cutout"})
        with urllib.request.urlopen(req, timeout=30) as r:
            self._size = int(r.headers["Content-Length"])

    def read(self, n: int = -1) -> bytes:
        if n == 0:
            return b""
        end = (self._size - 1) if n < 0 else min(self._pos + n - 1, self._size - 1)
        if self._pos > end:
            return b""
        headers = {"Range": f"bytes={self._pos}-{end}",
                   "User-Agent": "euclid-q1-cutout"}
        req = urllib.request.Request(self.url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        self._pos += len(data)
        return data

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:   self._pos = pos
        elif whence == 1: self._pos += pos
        elif whence == 2: self._pos = self._size + pos
        return self._pos

    def tell(self) -> int:        return self._pos
    def readable(self) -> bool:   return True
    def seekable(self) -> bool:   return True
    def writable(self) -> bool:   return False
    def close(self):              pass
    def __enter__(self):          return self
    def __exit__(self, *_):       pass


def _cutout_data_remote(url: str, ra: float, dec: float,
                        size_arcsec: float,
                        pixel_scale: float = VIS_PIXEL_SCALE) -> np.ndarray:
    """Fetch only the pixel rows needed for a cutout via HTTP range requests.

    Downloads the FITS header (~KB) then ONE range request for the relevant
    rows (~3–5 MB for a 6" stamp), instead of the full ~1.4 GB tile.
    """
    from astropy.io.fits import Header
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS

    fobj = _HttpRangeFile(url)

    # Read FITS header blocks (2880 bytes each) until END card
    header_bytes = b""
    while True:
        block = fobj.read(2880)
        if not block:
            break
        header_bytes += block
        if b"END" + b" " * 77 in block:
            break
    data_start = fobj.tell()

    header  = Header.fromstring(header_bytes.decode("ascii", errors="replace"))
    naxis1  = int(header["NAXIS1"])
    naxis2  = int(header["NAXIS2"])
    bitpix  = int(header["BITPIX"])
    bscale  = float(header.get("BSCALE", 1.0))
    bzero   = float(header.get("BZERO",  0.0))
    bytes_pp = abs(bitpix) // 8

    wcs   = WCS(header)
    coord = SkyCoord(ra=ra, dec=dec, unit="deg")
    xc, yc = wcs.world_to_pixel(coord)
    xc, yc = int(round(float(xc))), int(round(float(yc)))

    n_pix = max(10, int(round(size_arcsec / pixel_scale)))
    half  = n_pix // 2
    x1, x2 = max(0, xc - half), min(naxis1, xc + half)
    y1, y2 = max(0, yc - half), min(naxis2, yc + half)

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(
            f"Cutout out of bounds: pixel ({xc},{yc}) in {naxis1}×{naxis2}")

    row_bytes = naxis1 * bytes_pp
    fobj.seek(data_start + y1 * row_bytes)
    raw = fobj.read((y2 - y1) * row_bytes)

    dtype_map = {16: ">i2", -32: ">f4", -64: ">f8", 8: ">u1", 32: ">i4"}
    dtype = np.dtype(dtype_map.get(bitpix, ">f4"))
    rows  = np.frombuffer(raw, dtype=dtype).reshape(y2 - y1, naxis1)
    patch = rows[:, x1:x2].astype(np.float64)
    if bscale != 1.0 or bzero != 0.0:
        patch = patch * bscale + bzero

    # Always return an n_pix × n_pix square, padding with NaN where the tile
    # boundary was hit (avoids landscape/portrait cutouts near tile edges).
    image = np.full((n_pix, n_pix), np.nan, dtype=np.float32)
    row_off = half - (yc - y1)
    col_off = half - (xc - x1)
    image[row_off:row_off + (y2 - y1), col_off:col_off + (x2 - x1)] = patch
    # WCS for the RETURNED image (source at pixel (half, half)) — so callers can place
    # overlay markers (e.g. deblended MER positions) via world_to_pixel.
    cutout_wcs = wcs.deepcopy()
    cutout_wcs.wcs.crpix[0] += half - xc
    cutout_wcs.wcs.crpix[1] += half - yc
    return image, cutout_wcs


def _cutout_data(src, ra: float, dec: float, size_arcsec: float,
                 pixel_scale: float = VIS_PIXEL_SCALE):
    """Cutout from a remote tile URL (HTTP range reads) OR a local FITS Path (memory-mapped
    Cutout2D). Returns (image, wcs) where wcs.world_to_pixel gives pixel coords IN the
    returned image (for overlay markers). Lets Q1 (IRSA URLs) and ERO (cached full-field
    FITS) share the same renderer, incl. the NISP colour bands."""
    if isinstance(src, str):
        return _cutout_data_remote(src, ra, dec, size_arcsec, pixel_scale=pixel_scale)
    from astropy.coordinates import SkyCoord
    from astropy.io import fits as afits
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    import astropy.units as u
    with afits.open(src, memmap=True) as hdul:
        hdu = next((h for h in hdul if hasattr(h, "data") and isinstance(h.data, np.ndarray)
                    and h.data.ndim == 2), None)
        if hdu is None:
            raise RuntimeError(f"No 2D image HDU in {getattr(src, 'name', src)}")
        cutout = Cutout2D(hdu.data, SkyCoord(ra=ra, dec=dec, unit="deg"),
                          u.Quantity([size_arcsec, size_arcsec], unit=u.arcsec),
                          wcs=WCS(hdu.header), mode="partial", fill_value=np.nan)
    return cutout.data.astype(float), cutout.wcs


# ── Cutout rendering ──────────────────────────────────────────────────────────

def _render_cutout(tile_url_or_path: str | Path, ra: float, dec: float,
                   size_arcsec: float, title: str, dpi: int = 75,
                   c_star: float | None = None,
                   nisp_j_url: str | None = None,
                   nisp_h_url: str | None = None,
                   draw_crosshair: bool = True,
                   companions: list | None = None,
                   crosshair_color: str | None = None,
                   companion_color: str = "#00e5ff") -> np.ndarray:
    """Render a cutout stamp to a numpy array using log stretch.

    Accepts either a URL (uses HTTP range requests, ~3–5 MB per stamp) or a
    local Path (uses memory-mapped FITS).  URL mode is the default and avoids
    downloading the full 1.4 GB tile.

    c_star:      XGBoost P(gal) in [0, 1] — colours the crosshair via coolwarm_r
                 (red = star, blue = galaxy).  White if None.
    nisp_j_url:  Remote URL for the NISP-J tile (optional).
    nisp_h_url:  Remote URL for the NISP-H tile (optional).
                 When both are provided and SNR ≥ 10, an LRGB colour composite
                 is produced instead of grayscale.
    """
    import io
    from astropy.visualization import LogStretch, ImageNormalize, ManualInterval

    image, cut_wcs = _cutout_data(tile_url_or_path, ra, dec, size_arcsec, pixel_scale=VIS_PIXEL_SCALE)

    image = np.asarray(image, dtype=float)
    finite = np.isfinite(image)

    # Reject edge-tile cutouts with >30% NaN
    nan_frac = 1.0 - finite.mean()
    if nan_frac > 0.30:
        raise RuntimeError(f"Edge tile: {nan_frac:.0%} NaN pixels")

    if not finite.any():
        raise RuntimeError("No finite pixels in cutout")

    # Fill NaN/inf with local median so log stretch doesn't produce black patches
    image = np.where(finite, image, float(np.nanmedian(image[finite])))

    # Log stretch: shift p01 → 1.0 so all inputs are > 0
    p01   = float(np.nanpercentile(image, 1.0))
    p995  = float(np.nanpercentile(image, 99.5))
    shift = -p01 + 1.0
    img_s = np.clip(image + shift, 1e-6, None)
    vmin_s = 1.0
    vmax_s = max(p995 + shift, vmin_s * 10)

    # Faint-source SNR check — only attempt colour if source is bright enough
    n_pix  = img_s.shape[0]
    cx_i = cy_i = n_pix // 2
    r = max(1, n_pix // 8)
    central  = img_s[cy_i - r:cy_i + r, cx_i - r:cx_i + r]
    vis_bg   = float(np.nanpercentile(img_s, 10))
    vis_rms  = float(np.nanstd(img_s[img_s < np.nanpercentile(img_s, 50)]) or 1.0)
    vis_peak = float(np.nanmax(central))
    bright_enough = (vis_peak - vis_bg) / vis_rms >= 10.0

    use_color = bright_enough and (nisp_j_url is not None) and (nisp_h_url is not None)
    rgb = None
    if use_color:
        try:
            from scipy.ndimage import zoom as _zoom
            # NISP tiles are 0.3"/px — fetch at native resolution then upsample 3×.
            # _cutout_data accepts a URL (Q1 IRSA tiles) OR a local FITS Path (ERO).
            nisp_j_raw, _ = _cutout_data(nisp_j_url, ra, dec, size_arcsec,
                                         pixel_scale=NISP_PIXEL_SCALE)
            nisp_h_raw, _ = _cutout_data(nisp_h_url, ra, dec, size_arcsec,
                                         pixel_scale=NISP_PIXEL_SCALE)
            scale = NISP_PIXEL_SCALE / VIS_PIXEL_SCALE  # 3.0

            def _match(arr, tgt):
                th, tw = tgt.shape
                a = arr[:th, :tw]
                if a.shape[0] < th or a.shape[1] < tw:
                    pad = np.full(tgt.shape, np.nanmedian(a) if np.isfinite(a).any() else 0.0)
                    pad[:a.shape[0], :a.shape[1]] = a
                    a = pad
                return a

            nisp_j = _match(_zoom(nisp_j_raw, scale, order=3), image)
            nisp_h = _match(_zoom(nisp_h_raw, scale, order=3), image)

            # Check NISP NaN fraction — skip colour if either band is too empty
            if (np.isnan(nisp_j).mean() < 0.50 and np.isnan(nisp_h).mean() < 0.50):
                rgb = _lrgb_composite(image, nisp_j, nisp_h)
            else:
                use_color = False
        except Exception:
            use_color = False

    norm = ImageNormalize(img_s, interval=ManualInterval(vmin_s, vmax_s),
                          stretch=LogStretch())

    # Crosshair colour: explicit override (e.g. red to mark the Euclid position) wins;
    # else coolwarm_r by P(gal) [0,1] (red=star, blue=galaxy); else white.
    if crosshair_color is not None:
        pass
    elif c_star is not None and np.isfinite(c_star):
        crosshair_color = plt.cm.coolwarm_r(float(np.clip(c_star, 0, 1)))
    else:
        crosshair_color = "white"

    fig, ax = plt.subplots(figsize=(4, 4))
    if use_color and rgb is not None:
        ax.imshow(rgb, origin="lower", interpolation="nearest")
    else:
        ax.imshow(img_s, origin="lower", cmap="gray", norm=norm,
                  interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    n   = img_s.shape[0]
    gap = n * 0.10
    cx  = cy = n / 2
    if draw_crosshair:
        kw  = dict(color=crosshair_color, lw=1.0, alpha=0.55, solid_capstyle="butt")
        ax.plot([0,        cx - gap], [cy, cy],      **kw)
        ax.plot([cx + gap, n - 1],   [cy, cy],       **kw)
        ax.plot([cx, cx],  [0,        cy - gap],     **kw)
        ax.plot([cx, cx],  [cy + gap, n - 1],        **kw)
    # Overlay companion detections as small rings. Each companion is either a 3-tuple
    # (ra, dec, label) or a 4-tuple (ra, dec, label, color). The 4-tuple form lets
    # callers assign per-companion colors (e.g. blue for matched Gaia within 1",
    # pale-blue for FOV-only Gaia, orange for secondary Euclid sources). When the
    # 4th element is absent or None the global companion_color is used as fallback.
    if companions:
        import matplotlib.patheffects as _pe
        from astropy.coordinates import SkyCoord
        m = n * 0.03                 # corner margin (pixels)
        used: dict = {}              # (ha, va) -> stack count
        for comp in companions:
            cra, cdec, clabel = comp[0], comp[1], comp[2]
            ccolor = comp[3] if len(comp) > 3 and comp[3] else companion_color
            # 5th element: marker style — "ring" (default), "crosshair", or "x"
            cmarker = comp[4] if len(comp) > 4 and comp[4] else "ring"
            try:
                px, py = cut_wcs.world_to_pixel(SkyCoord(cra, cdec, unit="deg"))
                px, py = float(px), float(py)
            except Exception:
                continue
            if not (0 <= px < n and 0 <= py < n):
                continue
            arm_cross = 0.18 * n / size_arcsec   # arm half-length for + and × markers (arcsec)
            arm_ring  = 0.22 * n / size_arcsec   # radius for ○ ring markers (arcsec)
            mkw = dict(color=ccolor, lw=1.8, alpha=0.6, zorder=6, solid_capstyle="butt")
            if cmarker == "crosshair":
                ax.plot([px - arm_cross, px + arm_cross], [py, py], **mkw)
                ax.plot([px, px], [py - arm_cross, py + arm_cross], **mkw)
            elif cmarker == "x":
                d = arm_cross / np.sqrt(2)
                ax.plot([px - d, px + d], [py - d, py + d], **mkw)
                ax.plot([px + d, px - d], [py - d, py + d], **mkw)
            else:  # "ring"
                ax.add_patch(plt.Circle((px, py), arm_ring, fill=False, ec=ccolor,
                                        lw=1.5, alpha=0.6, zorder=6))
            if not clabel:
                continue
            right = px >= cx                                   # image x quadrant
            upper = py >= cy                                   # origin='lower' → larger y is up
            ha = "right" if right else "left"
            va = "top" if upper else "bottom"
            xt = (n - m) if right else m
            yt = (n - m) if upper else m
            k = used.get((ha, va), 0); used[(ha, va)] = k + 1  # stack same-corner labels
            yt += (-1 if upper else 1) * k * n * 0.10          # grow inward, away from corner
            t = ax.text(xt, yt, str(clabel), color=ccolor, fontsize=11,  # <- label size knob
                        ha=ha, va=va, zorder=7)
            t.set_path_effects([_pe.Stroke(linewidth=2.4, foreground="black"), _pe.Normal()])
    _add_scale_bar(ax, n, size_arcsec, VIS_PIXEL_SCALE, "white")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def _make_contact_sheet(cell_images: list, out_stem: Path,
                        ncols: int = 5, max_rows: int = 6) -> None:
    if not cell_images:
        return
    per_page = ncols * max_rows
    pages = [cell_images[i:i + per_page] for i in range(0, len(cell_images), per_page)]
    for pi, imgs in enumerate(pages, 1):
        out = (Path(str(out_stem) + ".png") if len(pages) == 1
               else Path(str(out_stem) + f"_p{pi}.png"))
        nrows = (len(imgs) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                                 squeeze=False)
        for i, ax in enumerate(axes.flat):
            if i < len(imgs):
                ax.imshow(imgs[i])
            ax.axis("off")

        # Fixed-height header (1.10") for consistent layout across all sheet sizes
        fig_h_in  = 5.0 * nrows
        header_in = 1.10
        top_frac  = 1.0 - header_in / fig_h_in
        _ih       = header_in / fig_h_in

        fig.subplots_adjust(top=top_frac, bottom=0.01, left=0.01, right=0.99,
                            hspace=0.05, wspace=0.05)

        legend_w = 0.28
        legend_h = 0.15 / fig_h_in
        legend_x = 0.07
        legend_y = top_frac + 0.20 / fig_h_in
        sup_y    = legend_y + legend_h * 0.5

        plt.suptitle(f"Page {pi}/{len(pages)}  ({len(imgs)} sources)",
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

        plt.savefig(out, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Contact sheet: {out}")


# ── Public API ────────────────────────────────────────────────────────────────

def make_q1_cutouts(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    field_col: str = "field",         # df column with Q1 field name (e.g. "EDF-Fornax")
    default_field: str = "unknown",   # fallback if field_col missing
    size_arcsec: float = 6.0,
    size_col: str | None = None,      # column with per-source size in arcsec
    size_min: float = 3.0,
    size_max: float = 10.0,
    size_scale: float = 8.0,          # multiplier on flux_radius (px) if size_col set
    c_star_col: str | None = None,    # column with Gaia BP/RP excess for crosshair colour
    title_extra_cols: list[str] | None = None,  # extra columns to include in cutout title
    title_fn=None,                    # callable(row, size_arcsec) -> str overrides default title
    tiles: list[dict] | None = None,  # pre-discovered tile list; skips TAP query if supplied
    contact_sheet_stem: str = "contact_sheet",
    manifest_stem: str = "manifest",
    individuals_dir: Path | None = None,  # if set, save individual PNGs here instead of output_dir
    dpi: int = 75,
    limit: int | None = None,
    save_individuals: bool = False,
) -> pd.DataFrame:
    """
    Make Euclid Q1 VIS cutouts for every source in df.

    Required columns in df:  ra, dec
    Recommended column:      field  (Q1 field name, stored in manifest)
    Optional useful columns: any identifier/classification columns for titles

    Uses HTTP range requests to fetch only the pixel rows needed per stamp
    (~3–5 MB each), so no full tile download is required.
    A manifest CSV and paginated contact sheet are written to output_dir.

    Returns a DataFrame with a 'status' column added.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if limit is not None:
        df = df.head(limit).copy()

    # ── Step 1: resolve tile for every source ─────────────────────────────────
    print(f"  Resolving Q1 tile IDs for {len(df):,} sources...")
    tile_map: dict[int, TileInfo | None] = {}

    if tiles is not None:
        # Pre-discovered tiles supplied — just assign geometrically (no network calls)
        fields = (df[field_col].astype(str) if field_col in df.columns
                  else pd.Series(default_field, index=df.index))
        for idx, row in df.iterrows():
            field_name = fields[idx]
            tile_map[idx] = _assign_to_tiles(
                float(row["ra"]), float(row["dec"]), tiles, field_name)
    else:
        # Batch: one ESA TAP query per field group
        fields = (df[field_col].astype(str) if field_col in df.columns
                  else pd.Series(default_field, index=df.index))
        for field_name in fields.unique():
            mask  = fields == field_name
            group = df[mask]
            ras   = group["ra"].astype(float)
            decs  = group["dec"].astype(float)
            print(f"  Field '{field_name}': {len(group)} sources  "
                  f"(RA {ras.min():.2f}–{ras.max():.2f}, "
                  f"Dec {decs.min():.2f}–{decs.max():.2f})")
            discovered = _discover_tiles(
                ras.min(), ras.max(), decs.min(), decs.max(), field_name)
            if discovered:
                print(f"    → {len(discovered)} tile(s) via ESA TAP")
                for idx, row in group.iterrows():
                    tile_map[idx] = _assign_to_tiles(
                        float(row["ra"]), float(row["dec"]), discovered, field_name)
            else:
                print(f"    → falling back to per-source IRSA TAP...")
                for idx, row in group.iterrows():
                    tile_map[idx] = _query_tile_irsa(
                        float(row["ra"]), float(row["dec"]), field_name)

    unique_tiles = {t for t in tile_map.values() if t is not None}
    print(f"  {len(unique_tiles)} unique tile(s): "
          f"{sorted(t.tile_id for t in unique_tiles)}")

    # ── Step 2: collect tile URLs (no full download — range requests only) ───────
    tile_urls: dict[str, str] = {t.tile_id: t.url for t in unique_tiles}

    # Resolve NISP-J and NISP-H URLs for each unique tile
    nisp_j_urls: dict[str, str | None] = {}
    nisp_h_urls: dict[str, str | None] = {}
    for t in unique_tiles:
        for band, cache in (("J", nisp_j_urls), ("H", nisp_h_urls)):
            fn = _nisp_filename(t.tile_id, band)
            cache[t.tile_id] = (f"{IRSA_IBE_DATA}/{t.tile_id}/NISP/{fn}"
                                 if fn else None)
    n_nisp_j = sum(1 for v in nisp_j_urls.values() if v)
    n_nisp_h = sum(1 for v in nisp_h_urls.values() if v)
    print(f"  NISP-J tiles resolved: {n_nisp_j}/{len(unique_tiles)}  "
          f"NISP-H: {n_nisp_h}/{len(unique_tiles)}")
    print(f"  Using HTTP range requests (~3–5 MB per stamp, no full tile download).")

    # ── Step 3: render cutouts ─────────────────────────────────────────────────
    records     = []
    cell_images = []
    n = len(df)

    for i, (idx, row) in enumerate(df.iterrows()):
        tile = tile_map.get(idx)
        status, message = "ok", ""
        records_cutout_file = None

        if tile is None:
            status, message = "skipped", "tile lookup failed"
        else:
            url = tile_urls.get(tile.tile_id)
            if url is None:
                status, message = "skipped", "tile URL unavailable"
            else:
                # determine cutout size
                if size_col and size_col in row.index and pd.notna(row[size_col]):
                    size = float(np.clip(
                        float(row[size_col]) * VIS_PIXEL_SCALE * size_scale,
                        size_min, size_max,
                    ))
                else:
                    size = size_arcsec

                # build title
                if title_fn is not None:
                    title = title_fn(row, size)
                else:
                    line2 = ""
                    if title_extra_cols:
                        parts = [f"{c}={row[c]:.3g}" for c in title_extra_cols
                                 if c in row.index and pd.notna(row[c])]
                        line2 = "  ".join(parts)
                    title = (f"RA={float(row['ra']):.4f}  Dec={float(row['dec']):+.4f}\n"
                             + (f"{line2}  " if line2 else "")
                             + f"[{size:.1f}\"]")
                try:
                    c_star = (float(row[c_star_col])
                              if c_star_col and c_star_col in row.index
                              and pd.notna(row[c_star_col]) else None)
                    nisp_j = nisp_j_urls.get(tile.tile_id)
                    nisp_h = nisp_h_urls.get(tile.tile_id)
                    img = _render_cutout(url, float(row["ra"]),
                                         float(row["dec"]), size, title,
                                         dpi=dpi, c_star=c_star,
                                         nisp_j_url=nisp_j,
                                         nisp_h_url=nisp_h)
                    cell_images.append(img)
                    if save_individuals:
                        ind_dir = Path(individuals_dir) if individuals_dir else output_dir
                        ind_dir.mkdir(parents=True, exist_ok=True)
                        indiv_path = ind_dir / f"cutout_{i:04d}.png"
                        plt.imsave(str(indiv_path), img)
                        records_cutout_file = str(indiv_path.relative_to(output_dir))
                    else:
                        records_cutout_file = None
                except Exception as exc:
                    status, message = "error", str(exc)
                    records_cutout_file = None

        print(f"  {i+1:>4}/{n}  RA={float(row['ra']):.4f}  "
              f"Dec={float(row['dec']):+.4f}  {status}"
              + (f"  — {message}" if message else ""))
        records.append({**row.to_dict(), "status": status, "message": message,
                        "tile_id": tile.tile_id if tile else None,
                        "cutout_file": records_cutout_file})

    result_df = pd.DataFrame(records)
    manifest_path = output_dir / f"{manifest_stem}.csv"
    result_df.to_csv(manifest_path, index=False)
    _make_contact_sheet(cell_images, output_dir / contact_sheet_stem)

    n_ok  = (result_df["status"] == "ok").sum()
    n_err = (result_df["status"] != "ok").sum()
    print(f"\n  Done: {n_ok} cutouts OK, {n_err} skipped/errors")
    print(f"  Manifest: {manifest_path}")
    return result_df


# ── Standalone CLI ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Make Euclid Q1 VIS cutouts from a source CSV."
    )
    p.add_argument("--csv", required=True, help="Input CSV with ra, dec columns.")
    p.add_argument("--out", required=True, help="Output directory for cutouts.")
    p.add_argument("--size-arcsec", type=float, default=6.0,
                   help="Fixed cutout size in arcsec (default: 6).")
    p.add_argument("--size-col", default=None,
                   help="Column with flux_radius (px) for dynamic sizing.")
    p.add_argument("--limit", type=int, default=None,
                   help="Max sources to process.")
    p.add_argument("--dpi", type=int, default=75)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    df   = pd.read_csv(args.csv)
    print(f"Loaded {len(df):,} sources from {args.csv}")
    make_q1_cutouts(
        df,
        output_dir=Path(args.out),
        size_arcsec=args.size_arcsec,
        size_col=args.size_col,
        limit=args.limit,
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
