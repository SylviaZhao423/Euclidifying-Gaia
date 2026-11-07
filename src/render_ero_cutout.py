#!/usr/bin/env python3
"""
Euclid ERO VIS cutouts from the cached full-field FITS (output/euclid_ero_fits_cache/),
for ERO sources that have NO Q1 MER tile on IRSA.

The Euclid+WISE sample unions Q1 (EDF) and ERO sources. Q1 imaging comes from IRSA
per-tile range reads (render_q1_cutout); ERO imaging is a single full-field VIS mosaic per
target, downloaded by save_ero_fits.py. This helper maps an (ra, dec) to whichever ERO
field footprint covers it and renders the cutout straight from that local FITS via
render_q1_cutout._render_cutout (which memory-maps the FITS and Cutout2D's the region), so
the ERO and Q1 stamps share the exact same renderer. Grayscale VIS (no NISP colour).

Requires the field's VIS FITS in the cache:  python src/save_ero_fits.py --fields "<name>"
"""
from __future__ import annotations

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
FITS_CACHE = PROJECT_DIR / "output" / "euclid_ero_fits_cache"

_FIELDS: list | None = None   # cached [(name, path, wcs, nx, ny), ...]


def _load_fields() -> list:
    """Lazily build the ERO field footprints (WCS + pixel dims) from cached VIS FITS."""
    global _FIELDS
    if _FIELDS is not None:
        return _FIELDS
    from astropy.io import fits
    from astropy.wcs import WCS
    _FIELDS = []
    for p in sorted(FITS_CACHE.glob("Euclid-VIS-ERO-*-Flattened.DR3.fits")):
        try:
            with fits.open(p, memmap=True) as hd:
                hdu = next(h for h in hd
                           if getattr(h, "data", None) is not None and h.data.ndim == 2)
                ny, nx = hdu.data.shape
                wcs = WCS(hdu.header)
        except Exception:
            continue
        _FIELDS.append((p.name, p, wcs, nx, ny))
    return _FIELDS


def euclid_release(row) -> str:
    """'Q1' or 'ERO' for a Euclid+WISE source — from the projected CSV's 'euclid' column
    if present (authoritative, set by extract_qso_euclid_wise_gaia), else inferred from
    whether the position falls in a cached ERO field footprint."""
    e = row.get("euclid") if hasattr(row, "get") else None
    if isinstance(e, str) and e in ("Q1", "ERO"):
        return e
    return "ERO" if ero_field_for(float(row["qso_ra"]), float(row["qso_dec"])) is not None else "Q1"


def ero_field_for(ra: float, dec: float) -> Path | None:
    """Path of the ERO VIS FITS whose footprint covers (ra, dec), or None."""
    for _name, path, wcs, nx, ny in _load_fields():
        try:
            x, y = wcs.world_to_pixel_values(ra, dec)
        except Exception:
            continue
        if 0 <= x < nx and 0 <= y < ny:
            return path
    return None


def render_ero_cutout(ra: float, dec: float, size_arcsec: float,
                      c_star: float | None = None, draw_crosshair: bool = True,
                      companions: list | None = None,
                      crosshair_color: str | None = None, companion_color: str = "#00e5ff"):
    """Rendered ERO stamp (numpy array) via the shared _render_cutout — a VIS+NISP LRGB
    colour composite where the field's NISP-H/J FITS are cached (grayscale VIS otherwise).
    Returns None if the position isn't in any cached ERO field / the cutout fails."""
    vis = ero_field_for(ra, dec)
    if vis is None:
        return None
    # Same field's NISP-H / NISP-J full-field FITS → colour composite (if downloaded).
    nh = vis.with_name(vis.name.replace("Euclid-VIS-", "Euclid-NISP-H-"))
    nj = vis.with_name(vis.name.replace("Euclid-VIS-", "Euclid-NISP-J-"))
    nh = nh if nh.exists() else None
    nj = nj if nj.exists() else None
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from render_q1_cutout import _render_cutout  # noqa: E402
    try:
        return _render_cutout(vis, ra, dec, size_arcsec, "", c_star=c_star,
                              nisp_j_url=nj, nisp_h_url=nh, draw_crosshair=draw_crosshair,
                              companions=companions, crosshair_color=crosshair_color,
                              companion_color=companion_color)
    except Exception:
        return None
