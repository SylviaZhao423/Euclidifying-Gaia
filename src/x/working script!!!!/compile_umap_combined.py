#!/usr/bin/env python3
"""
Combined Sky + 3D UMAP + Sidebar interactive HTML visualization.

Three-panel layout: sky map (WebGL) | 3D UMAP (Plotly) | sidebar (cutouts/sheets)

Dataset dropdown switches both sky dots AND 3D traces simultaneously.
Color dropdown updates sky + 3D scatter + Q1 base simultaneously.

Reads:
  output/umap/arrays/q1_indiv.npz
  output/umap/arrays/q1_allphz.npz
  output/umap/arrays/q1_allphz_desi.npz
  output/umap/arrays/q1_indiv_reducer.joblib
  output/umap/arrays/q1_cluster_labels.npy
  output/umap/arrays/q1_features.csv
  output/umap/arrays/ero_indiv.npz
  output/umap/arrays/ero_indiv_rebalanced.npz
  output/umap/arrays/ero_features.csv
  output/umap/arrays/xgb_gaia_projected.csv
  output/umap/arrays/xgb_gaia_hilat_projected.csv
  output/umap/arrays/ero_desi.npz
  output/umap/arrays/projected/*.csv
  output/umap/arrays/gaia_dr3_aitoff_gal_8k.jpg

Writes:
  output/umap/q1_individual/umap_combined.html

Usage:
  python src/compile_umap_combined.py
  python src/compile_umap_combined.py --reproject
  python src/compile_umap_combined.py --max-points 2000
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import astropy.units as u
import joblib
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "q1_individual"
CUTOUTS_DIR     = PROJECT_DIR / "output" / "euclid_cutouts" / "q1_cutouts" / "phz"
ERO_CUTOUTS_DIR = PROJECT_DIR / "output" / "euclid_cutouts" / "ero_cutouts" / "q1_to_ero"

ERO_GROUPS = [
    "agree_both_star",
    "agree_both_galaxy",
    "disagree_ero_star_q1_gal",
    "disagree_ero_gal_q1_star",
]
ERO_GROUP_LABELS = {
    "agree_both_star":          "Agree: both Star",
    "agree_both_galaxy":        "Agree: both Galaxy",
    "disagree_ero_star_q1_gal": "ERO=Star, Q1=Galaxy",
    "disagree_ero_gal_q1_star": "ERO=Galaxy, Q1=Star",
}

MAG_CUT   = 17

Q1_EUCLID_FEATURES = [
    "ellipticity",
    "flux_vis_psf",
    "fluxerr_vis_psf",
    "fluxsig_vis_psf",
]

ERO_VIS_FEATURES = [
    "e_v_class_star",
    "e_v_ellipticity",
    "e_v_elongation",
    "e_v_flux_radius",
]

PHZ_LABELS = {
     1: ("Star",               "#F45347",  "circle"),
     2: ("Galaxy",             "cyan",     "circle"),
     3: ("Multi (Star ∩ Gal)",   "#FFC107",  "square"),
     4: ("QSO",                "#4CAF50",  "circle"),
     5: ("Multi (Star ∩ QSO)",   "#E53935",  "diamond"),
     6: ("Multi (Gal ∩ QSO)",    "#00ACC1",  "diamond"),
     7: ("Multi (all three)",  "#FF7043",  "cross"),
    -1: ("Rejected",           "#9C27B0",  "circle-open"),
     0: ("Outlier",            "#FF9800",  "circle-open"),
}

PHZ_DIRS = {
     1: "1_star",      2: "2_gal",         3: "3_star_x_gal",
     4: "4_qso",       5: "5_star_x_qso",  6: "6_gal_x_qso",
     7: "7_all_three", -1: "-1_rej",        0: "0_outlier",
}

DESI_CUTOUTS_DIR = PROJECT_DIR / "output" / "euclid_cutouts" / "q1_cutouts" / "desi"

DESI_CATEGORY_DIRS = {
     2: "agree_both_gal",
     1: "disagree_desi_gal_q1_star_1",
     3: "disagree_desi_gal_q1_star_x_gal_3",
     4: "disagree_desi_gal_q1_qso_4",
     5: "disagree_desi_gal_q1_star_x_qso_5",
     6: "disagree_desi_gal_q1_gal_x_qso_6",
     7: "disagree_desi_gal_q1_all_three_7",
    -1: "disagree_desi_gal_q1_rej_n1",
     0: "disagree_desi_gal_q1_outlier_0",
}

DESI_CATEGORY_LABELS = {
     1: ("Star",              "#F45347",  "circle"),
     2: ("Galaxy (agree)",    "cyan",     "circle"),
     3: ("Multi (Star ∩ Gal)",  "#FFC107",  "square"),
     4: ("QSO",               "#4CAF50",  "circle"),
     5: ("Multi (Star ∩ QSO)",  "#E53935",  "diamond"),
     6: ("Multi (Gal ∩ QSO)",   "#00ACC1",  "diamond"),
     7: ("Multi (all three)", "#FF7043",  "cross"),
    -1: ("Rejected",          "#9C27B0",  "circle-open"),
     0: ("Outlier",           "#FF9800",  "circle-open"),
}

CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#F298B6",
    "mediumpurple", "#17becf", "#d62728", "#8c564b",
    "#7f7f7f", "#bcbd22",
]

# ERO overlay: 4 agreement/disagreement categories matching q1_to_ero CSV folders
ERO_GROUP_CONFIGS = {
    "agree_both_star":          ("Star (XGB Star)",      "#F45347"),
    "agree_both_galaxy":        ("Galaxy (XGB Galaxy)",  "cyan"),
    "disagree_ero_star_q1_gal": ("Star (XGB Galaxy)",    "#FFB0C8"),
    "disagree_ero_gal_q1_star": ("Galaxy (XGB Star)",    "#91D3F0"),
}

ERO_DESI_LABELS = {
    2: ("ERO Star",    "#CE93D8"),   # purple — ERO class_star says star, DESI says gal
    3: ("ERO Galaxy",  "#A5D6A7"),   # green  — ERO class_star says gal, DESI says gal
}

ERO_DESI_CUTOUTS_DIR = PROJECT_DIR / "output" / "euclid_cutouts" / "ero_cutouts" / "desi"
ERO_DESI_CATEGORY_DIRS = {
    2: "disagree_desi_gal_ero_star",
    3: "agree_both_gal",
}

Q1_TO_ERO_DIR      = PROJECT_DIR / "output" / "q1_to_ero"
Q1_SEED_MODELS_DIR = PROJECT_DIR / "output" / "xgboost_q1" / "seed_models"
PROJECTED_DIR      = ARRAYS_DIR  / "projected"
GAIA_XGB_CSV       = ARRAYS_DIR  / "xgb_gaia_projected.csv"
GAIA_HILAT_CSV     = ARRAYS_DIR  / "xgb_gaia_hilat_projected.csv"

GAL_THRESH = 0.5   # XGBoost galaxy probability threshold

# ── Sky map constants ─────────────────────────────────────────────────────────
SKY_IMG    = ARRAYS_DIR / "gaia_dr3_aitoff_gal_8k.jpg"
IMG_W, IMG_H = 8000, 4000
IMG_CRPIX = (4000.5, 2000.5)
IMG_CDELT = (-0.045, 0.045)  # deg/pixel (360/8000)

def _hips2fits_url(w, h):
    cdelt = 360.0 / w
    wcs = json.dumps({
        "NAXIS": 2, "NAXIS1": w, "NAXIS2": h,
        "CTYPE1": "GLON-AIT", "CTYPE2": "GLAT-AIT",
        "CRPIX1": w / 2 + 0.5, "CRPIX2": h / 2 + 0.5,
        "CRVAL1": 0.0, "CRVAL2": 0.0,
        "CDELT1": -cdelt, "CDELT2": cdelt,
    })
    return (
        "https://alasky.cds.unistra.fr/hips-image-services/hips2fits?"
        + urllib.parse.urlencode({
            "hips": "CDS/P/DM/flux-color-Rp-G-Bp/I/355/gaiadr3",
            "format": "jpg", "wcs": wcs,
        })
    )

HIPS2FITS_URL = _hips2fits_url(IMG_W, IMG_H)

# ERO field definitions (ra, dec, radius_deg)
_ERO_CSV = PROJECT_DIR / "output" / "ero_fields_info.csv"
if _ERO_CSV.exists():
    _ero_df = pd.read_csv(_ERO_CSV)
    ERO_FIELDS = {
        row["field_name"]: (row["ra_deg"], row["dec_deg"],
                            math.sqrt(row["area_deg2"] / math.pi))
        for _, row in _ero_df.iterrows()
    }
else:
    ERO_FIELDS = {}

Q1_FIELDS = {
    'EDF-North':  (269.733, +66.018, math.sqrt(22.9 / math.pi)),
    'EDF-South':  (61.241,  -48.423, math.sqrt(28.1 / math.pi)),
    'EDF-Fornax': (52.932,  -28.088, math.sqrt(12.1 / math.pi)),
    'LDN 1641':   (85.75,   -8.367,  math.sqrt(0.5  / math.pi)),
}


# ── Sky coordinate helpers ────────────────────────────────────────────────────

def radec_to_galactic(ra, dec):
    """Convert arrays of RA/Dec (deg) to galactic l, b (deg)."""
    sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    g = sc.galactic
    return g.l.deg, g.b.deg


def hammer_aitoff(l_deg, b_deg):
    """Hammer-Aitoff projection: galactic (l, b) in deg → (x, y) in native deg."""
    l = np.deg2rad(np.where(l_deg > 180, l_deg - 360, l_deg))
    b = np.deg2rad(b_deg)
    z = np.sqrt(1 + np.cos(b) * np.cos(l / 2))
    x = -np.rad2deg(2 * np.cos(b) * np.sin(l / 2) / z)
    y = np.rad2deg(np.sin(b) / z)
    return x, y


def aitoff_to_imgpx(ax, ay):
    """Aitoff native coords (deg) → image pixel coords."""
    px = IMG_CRPIX[0] + ax / IMG_CDELT[0]
    py = IMG_CRPIX[1] + ay / IMG_CDELT[1]
    return px, py


def _round_list(arr, d=4):
    return [round(float(v), d) if np.isfinite(v) else None for v in arr]


def _blacken_outside_ellipse(img_path):
    """Set pixels outside the Aitoff ellipse to black, normalise brightness."""
    from PIL import Image
    img = Image.open(img_path)
    arr = np.array(img)
    h, w = arr.shape[:2]
    semi_a = 2 * np.sqrt(2) * 180 / np.pi / abs(IMG_CDELT[0])
    semi_b = np.sqrt(2) * 180 / np.pi / abs(IMG_CDELT[1])
    cy, cx = IMG_CRPIX[1] - 0.5, IMG_CRPIX[0] - 0.5
    yy, xx = np.mgrid[:h, :w]
    outside = ((xx - cx) / semi_a) ** 2 + ((yy - cy) / semi_b) ** 2 > 1
    inside_vals = arr[~outside].astype(np.float32)
    current_median = float(np.median(inside_vals))
    TARGET_MEDIAN = 32.0
    if current_median > TARGET_MEDIAN * 1.1 and current_median < 255:
        gamma = np.log(TARGET_MEDIAN / 255.0) / np.log(current_median / 255.0)
        arr_f = arr.astype(np.float32) / 255.0
        arr_f = np.power(np.clip(arr_f, 1e-6, 1.0), gamma) * 255.0
        arr = arr_f.clip(0, 255).astype(np.uint8)
    arr[outside] = [8, 8, 8]
    img_out = Image.fromarray(arr)
    img_out.save(img_path, quality=92)


def ensure_sky_image():
    """Download Gaia DR3 sky image if not cached."""
    if SKY_IMG.exists():
        return SKY_IMG
    print(f"  Downloading Gaia DR3 sky image...")
    try:
        urllib.request.urlretrieve(HIPS2FITS_URL, SKY_IMG)
        print(f"  -> Downloaded ({SKY_IMG.stat().st_size / 1e6:.1f} MB)")
        _blacken_outside_ellipse(SKY_IMG)
        print(f"  -> {SKY_IMG} ({SKY_IMG.stat().st_size / 1e6:.1f} MB)")
        return SKY_IMG
    except Exception as e:
        print(f"  WARNING: Could not download sky image: {e}")
        return None


def _prepare_fields(fields_dict):
    """Convert field dict to list for JS circle drawing."""
    result = []
    for name, vals in fields_dict.items():
        ra, dec, radius = vals
        result.append({"name": name, "ra": round(float(ra), 4),
                       "dec": round(float(dec), 4),
                       "radius": round(float(radius), 4)})
    return result


def _prepare_sky_data(ra, dec, umap1, umap2, sids, feature_cols=None,
                      legend_colors=None, categories=None, clusters=None):
    """Convert per-dataset arrays to JSON-ready dict with sky coords.

    Parameters
    ----------
    ra, dec : array-like, degrees
    umap1, umap2 : array-like, UMAP coordinates
    sids : array-like of str, source IDs
    feature_cols : dict {name: array} of feature values for color mapping
    legend_colors : list of hex color strings for Legend mode
    categories : list of str, per-point category label for legend filtering
    clusters : array-like of int, per-point cluster assignment
    """
    gl, gb = radec_to_galactic(np.asarray(ra, dtype=float),
                                np.asarray(dec, dtype=float))
    ax, ay = hammer_aitoff(gl, gb)
    ipx, ipy = aitoff_to_imgpx(ax, ay)
    data = {
        "ra":  _round_list(ra, 4),
        "dec": _round_list(dec, 4),
        "gl":  _round_list(gl, 4),
        "gb":  _round_list(gb, 4),
        "ipx": _round_list(ipx, 2),
        "ipy": _round_list(ipy, 2),
        "u1":  _round_list(umap1),
        "u2":  _round_list(umap2),
        "sid": [str(s) for s in sids],
    }
    if feature_cols:
        for name, arr in feature_cols.items():
            data[name] = _round_list(arr)
    if legend_colors is not None:
        data["legend_color"] = list(legend_colors)
    if categories is not None:
        data["cat"] = [str(c) for c in categories]
    if clusters is not None:
        data["cluster"] = [int(c) for c in clusters]
    return data


# ── Data loading ─────────────────────────────────────────────────────────────

def load_allphz_from_npz() -> pd.DataFrame:
    """Load all phz classification sources from q1_allphz.npz."""
    data = np.load(ARRAYS_DIR / "q1_allphz.npz", allow_pickle=True)
    df = pd.DataFrame(data["X_features"], columns=SELECTED_GAIA_PARAMETERS)
    df["source_id"]       = data["source_ids"].astype(str)
    df["ra"]              = data["ra"].astype(float)
    df["dec"]             = data["dec"].astype(float)
    df["phot_g_mean_mag"] = data["phot_g_mean_mag"].astype(float)
    df["phz"]             = data["phz"].astype(int)
    df["phz_pgal"]        = data["phz_gal_prob"].astype(float)
    print(f"  -> {len(df):,} sources (from q1_allphz.npz)")
    return df


def load_desi_from_npz() -> pd.DataFrame:
    """Load DESI-confirmed galaxies from q1_allphz_desi.npz."""
    data = np.load(ARRAYS_DIR / "q1_allphz_desi.npz", allow_pickle=True)
    df = pd.DataFrame(data["X_features"], columns=SELECTED_GAIA_PARAMETERS)
    df["source_id"]       = data["source_ids"].astype(str)
    df["ra"]              = data["ra"].astype(float)
    df["dec"]             = data["dec"].astype(float)
    df["phot_g_mean_mag"] = data["phot_g_mean_mag"].astype(float)
    df["phz"]             = data["phz"].astype(int)
    df["d_desi_z"]        = data["desi_z"].astype(float)
    df["phz_pgal"]        = data["phz_gal_prob"].astype(float)
    print(f"  -> {len(df):,} DESI sources (from q1_allphz_desi.npz)")
    return df


# ── Cluster assignment ────────────────────────────────────────────────────────

def assign_to_clusters(src_emb, q1_emb, q1_labels):
    valid_mask = q1_labels >= 0
    if valid_mask.sum() == 0:
        return np.full(len(src_emb), -1, dtype=int)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(q1_emb[valid_mask])
    _, idx = nn.kneighbors(src_emb)
    idx = idx.ravel()
    return q1_labels[np.where(valid_mask)[0][idx]]


# ── Euclid feature loaders ────────────────────────────────────────────────────

def load_q1_euclid_feats(csv_path: Path, source_ids: np.ndarray) -> tuple[np.ndarray | None, list[str]]:
    """Return (N×M array, ['euclid_q1::name', ...]) aligned to source_ids."""
    if not csv_path.exists():
        print(f"  [Q1 Euclid] {csv_path.name} not found — skipping.")
        return None, []
    df = pd.read_csv(csv_path)
    df["source_id"] = df["source_id"].astype(str)
    sid_to_idx = {sid: i for i, sid in enumerate(df["source_id"])}
    umap_idx   = np.array([sid_to_idx.get(sid, -1) for sid in source_ids])
    cols, names = [], []
    for col in Q1_EUCLID_FEATURES:
        if col not in df.columns:
            continue
        arr   = np.full(len(source_ids), np.nan)
        valid = umap_idx >= 0
        arr[valid] = pd.to_numeric(df[col].values[umap_idx[valid]], errors="coerce")
        if np.isfinite(arr).sum() > 0:
            cols.append(arr)
            names.append(f"euclid_q1::{col}")
            print(f"    euclid_q1::{col:<35} {int(np.isfinite(arr).sum()):>8,} finite values")
    return (np.stack(cols, axis=1) if cols else None), names


def load_ero_feats(csv_path: Path, source_ids: np.ndarray) -> tuple[np.ndarray | None, list[str], np.ndarray | None]:
    """Return (N×M array, ['feat_name', ...], gmag_array) aligned to source_ids."""
    if not csv_path.exists():
        print(f"  [ERO feats] {csv_path.name} not found — skipping.")
        return None, [], None
    df = pd.read_csv(csv_path)
    df["source_id"] = df["source_id"].astype(str)
    sid_to_idx = {sid: i for i, sid in enumerate(df["source_id"])}
    ero_idx    = np.array([sid_to_idx.get(sid, -1) for sid in source_ids])
    gmag = None
    if "phot_g_mean_mag" in df.columns:
        arr_g    = np.full(len(source_ids), np.nan)
        valid    = ero_idx >= 0
        arr_g[valid] = pd.to_numeric(df["phot_g_mean_mag"].values[ero_idx[valid]], errors="coerce")
        gmag = arr_g
    cols, names = [], []
    for col in ERO_VIS_FEATURES:
        if col not in df.columns:
            continue
        arr   = np.full(len(source_ids), np.nan)
        valid = ero_idx >= 0
        arr[valid] = pd.to_numeric(df[col].values[ero_idx[valid]], errors="coerce")
        if np.isfinite(arr).sum() > 0:
            cols.append(arr)
            names.append(col)
    return (np.stack(cols, axis=1) if cols else None), names, gmag


# ── Cutout / contact-sheet caches ────────────────────────────────────────────

def load_cutout_cache(cutouts_dir: Path) -> dict:
    import base64
    from io import BytesIO
    from PIL import Image
    cache = {}
    for manifest_path in sorted(cutouts_dir.rglob("manifest*.csv")):
        cell_dir = manifest_path.parent
        try:
            df = pd.read_csv(manifest_path)
        except Exception:
            continue
        if "cutout_file" not in df.columns:
            continue
        for _, row in df.iterrows():
            if pd.isna(row.get("cutout_file")):
                continue
            png_path = cell_dir / row["cutout_file"]
            if not png_path.exists():
                continue
            try:
                img = Image.open(png_path).convert("RGB")
                buf = BytesIO(); img.save(buf, format="PNG")
                cache[str(row["source_id"])] = base64.b64encode(buf.getvalue()).decode()
            except Exception:
                continue
    print(f"  Loaded {len(cache)} cached cutout thumbnails")
    return cache


def load_cutout_cache_ero(ero_cutouts_dir: Path) -> dict:
    """Load individual ERO cutout thumbnails from {group}/cluster{k}/cutouts/.

    Reads manifest.csv files that have a 'source_id' and 'cutout_file' column,
    matching the format written by download_e_q1_to_ero_cutouts.py.
    Returns {source_id_str: base64_png}.
    """
    import base64
    from io import BytesIO
    from PIL import Image
    cache: dict[str, str] = {}
    for manifest_path in sorted(ero_cutouts_dir.rglob("manifest*.csv")):
        cell_dir = manifest_path.parent
        # Only read cluster-level manifests (skip the top-level group manifest)
        if cell_dir == ero_cutouts_dir or cell_dir.parent == ero_cutouts_dir and \
                not cell_dir.name.startswith("cluster"):
            continue
        try:
            df = pd.read_csv(manifest_path)
        except Exception:
            continue
        if "cutout_file" not in df.columns or "source_id" not in df.columns:
            continue
        for _, row in df.iterrows():
            if pd.isna(row.get("cutout_file")) or not row["cutout_file"]:
                continue
            png_path = cell_dir / row["cutout_file"]
            if not png_path.exists():
                continue
            try:
                img = Image.open(png_path).convert("RGB")
                buf = BytesIO(); img.save(buf, format="PNG")
                cache[str(row["source_id"])] = base64.b64encode(buf.getvalue()).decode()
            except Exception:
                continue
    print(f"  Loaded {len(cache)} ERO individual cutout thumbnail(s)")
    return cache


def _load_all_sheet_pages(cluster_dir: Path) -> list[str]:
    """Load all contact sheet pages from a cluster dir as base64 strings.

    Returns a list of base64-encoded PNGs for contact_sheet.png (or _p1, _p2, ...).
    """
    import base64 as _b64
    from io import BytesIO
    from PIL import Image
    pages: list[str] = []
    # Single-page case
    single = cluster_dir / "contact_sheet.png"
    if single.exists():
        try:
            img = Image.open(single).convert("RGB")
            buf = BytesIO(); img.save(buf, format="PNG")
            pages.append(_b64.b64encode(buf.getvalue()).decode())
        except Exception:
            pass
        return pages  # single page only
    # Multi-page case: _p1, _p2, ...
    for i in range(1, 100):
        p = cluster_dir / f"contact_sheet_p{i}.png"
        if not p.exists():
            break
        try:
            img = Image.open(p).convert("RGB")
            buf = BytesIO(); img.save(buf, format="PNG")
            pages.append(_b64.b64encode(buf.getvalue()).decode())
        except Exception:
            continue
    return pages


def load_contact_sheet_cache_allphz(cutouts_dir: Path) -> tuple[dict, dict, dict]:
    cache: dict[str, dict[str, str]] = {}
    counts: dict[str, dict[str, int]] = {}
    extra: dict[str, dict[str, list[str]]] = {}   # pages beyond the first
    for v, subdir in PHZ_DIRS.items():
        phz_root = cutouts_dir / subdir
        if not phz_root.exists():
            continue
        for cluster_dir in sorted(phz_root.glob("cluster*")):
            k = cluster_dir.name.replace("cluster", "")
            all_pages = _load_all_sheet_pages(cluster_dir)
            if not all_pages:
                continue
            cache.setdefault(str(v), {})[k] = all_pages[0]
            if len(all_pages) > 1:
                extra.setdefault(str(v), {})[k] = all_pages[1:]
            manifest = cluster_dir / "manifest.csv"
            n = 0
            if manifest.exists():
                try:
                    with open(manifest) as f:
                        n = max(0, sum(1 for _ in f) - 1)
                except Exception:
                    n = 0
            counts.setdefault(str(v), {})[k] = n
    print(f"  Loaded {sum(len(d) for d in cache.values())} contact sheet(s)")
    return cache, counts, extra


def load_contact_sheet_cache_desi(desi_dir: Path) -> tuple[dict, dict, dict]:
    cache: dict[str, dict[str, str]] = {}
    counts: dict[str, dict[str, int]] = {}
    extra: dict[str, dict[str, list[str]]] = {}
    for v, subdir in DESI_CATEGORY_DIRS.items():
        cat_root = desi_dir / subdir
        if not cat_root.exists():
            continue
        for cluster_dir in sorted(cat_root.glob("cluster*")):
            k = cluster_dir.name.replace("cluster", "")
            all_pages = _load_all_sheet_pages(cluster_dir)
            if not all_pages:
                continue
            cache.setdefault(str(v), {})[k] = all_pages[0]
            if len(all_pages) > 1:
                extra.setdefault(str(v), {})[k] = all_pages[1:]
            manifest = cluster_dir / "manifest.csv"
            n = 0
            if manifest.exists():
                try:
                    with open(manifest) as f:
                        n = max(0, sum(1 for _ in f) - 1)
                except Exception:
                    n = 0
            counts.setdefault(str(v), {})[k] = n
    print(f"  Loaded {sum(len(d) for d in cache.values())} DESI contact sheet(s)")
    return cache, counts, extra


def load_contact_sheet_cache_ero_desi(ero_desi_dir: Path) -> tuple[dict, dict, dict]:
    """Load per-cluster ERO-DESI contact sheets."""
    cache: dict[str, dict[str, str]] = {}
    counts: dict[str, dict[str, int]] = {}
    extra: dict[str, dict[str, list[str]]] = {}
    for v, subdir in ERO_DESI_CATEGORY_DIRS.items():
        cat_root = ero_desi_dir / subdir
        if not cat_root.exists():
            continue
        for cluster_dir in sorted(cat_root.glob("cluster*")):
            k = cluster_dir.name.replace("cluster", "")
            all_pages = _load_all_sheet_pages(cluster_dir)
            if not all_pages:
                continue
            cache.setdefault(str(v), {})[k] = all_pages[0]
            if len(all_pages) > 1:
                extra.setdefault(str(v), {})[k] = all_pages[1:]
            manifest = cluster_dir / "manifest.csv"
            n = 0
            if manifest.exists():
                try:
                    with open(manifest) as f:
                        n = max(0, sum(1 for _ in f) - 1)
                except Exception:
                    n = 0
            counts.setdefault(str(v), {})[k] = n
    print(f"  Loaded {sum(len(d) for d in cache.values())} ERO-DESI contact sheet(s)")
    return cache, counts, extra


def load_contact_sheet_cache_ero(ero_cutouts_dir: Path) -> tuple[dict, dict, dict]:
    """Load per-cluster ERO contact sheets from {group}/cluster{k}/contact_sheet.png."""
    cache: dict[str, dict[str, str]] = {}
    counts: dict[str, dict[str, int]] = {}
    extra: dict[str, dict[str, list[str]]] = {}
    for group in ERO_GROUPS:
        group_dir = ero_cutouts_dir / group
        if not group_dir.exists():
            continue
        for cluster_dir in sorted(group_dir.glob("cluster*")):
            k = cluster_dir.name.replace("cluster", "")
            all_pages = _load_all_sheet_pages(cluster_dir)
            if not all_pages:
                continue
            cache.setdefault(group, {})[k] = all_pages[0]
            if len(all_pages) > 1:
                extra.setdefault(group, {})[k] = all_pages[1:]
            manifest = cluster_dir / "manifest.csv"
            n = 0
            if manifest.exists():
                try:
                    with open(manifest) as f:
                        n = max(0, sum(1 for _ in f) - 1)
                except Exception:
                    n = 0
            counts.setdefault(group, {})[k] = n
    total = sum(len(d) for d in cache.values())
    print(f"  Loaded {total} ERO contact sheet(s) across {len(cache)} group(s)")
    return cache, counts, extra


def load_ero_group_map() -> dict[str, str]:
    """Return {source_id_str → group_name} from the q1_to_ero CSV files."""
    result: dict[str, str] = {}
    for group in ERO_GROUP_CONFIGS:
        csv_path = Q1_TO_ERO_DIR / f"{group}.csv"
        if not csv_path.exists():
            print(f"  [ERO groups] {csv_path.name} not found — skipping.")
            continue
        df = pd.read_csv(csv_path, usecols=["g_source_id"])
        for sid in df["g_source_id"].astype(str):
            result[sid] = group
    print(f"  ERO group map: {len(result):,} sources across "
          f"{sum(1 for g in ERO_GROUP_CONFIGS if any(v==g for v in result.values()))} groups")
    return result


def load_ero_extra_data_from_npz(npz_path: Path, source_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Load RA, Dec, class_star for ERO sources directly from the npz file."""
    n = len(source_ids)
    if not npz_path.exists():
        return {"ra": np.full(n, np.nan), "dec": np.full(n, np.nan),
                "class_star": np.full(n, np.nan)}
    npz = np.load(npz_path, allow_pickle=True)
    npz_sids = npz["source_ids"].astype(str)
    lookup = {sid: i for i, sid in enumerate(npz_sids)}
    sids = source_ids.astype(str)
    idx = np.array([lookup.get(s, -1) for s in sids])
    valid = idx >= 0
    result: dict[str, np.ndarray] = {}
    for col in ["ra", "dec", "class_star"]:
        arr = np.full(n, np.nan)
        if col in npz:
            arr[valid] = npz[col].astype(float)[idx[valid]]
        result[col] = arr
    print(f"  ERO extra data from {npz_path.name}: {valid.sum():,} / {n:,} matched")
    return result


def _align_features(arr: np.ndarray, src_names: list[str], target_names: list[str]) -> np.ndarray:
    """Reorder/pad feature array columns from src_names order to target_names order.

    Columns present in target_names but missing from src_names are filled with NaN.
    Columns in src_names not in target_names are dropped.
    """
    n = len(arr)
    result = np.full((n, len(target_names)), np.nan)
    for j, name in enumerate(target_names):
        if name in src_names:
            result[:, j] = arr[:, src_names.index(name)]
    return result


def _fill_col(df: pd.DataFrame | None, col: str, sid_map: dict) -> None:
    """Fill missing values in *col* by mapping source_id → sid_map."""
    if df is None or not sid_map:
        return
    mapped = df["source_id"].astype(str).map(sid_map)
    if col not in df.columns:
        df[col] = mapped
    else:
        df.loc[df[col].isna(), col] = mapped[df[col].isna()]


def _load_gaia_features_from_npz(npz_path: Path, target_sids: np.ndarray) -> np.ndarray:
    """Load X_raw (Gaia features) from npz, aligned to target_sids by source_id."""
    npz = np.load(npz_path, allow_pickle=True)
    npz_sids = npz["source_ids"].astype(str)
    X_raw = npz["X_raw"].astype(float)
    sid_to_idx = {s: i for i, s in enumerate(npz_sids)}
    idx = np.array([sid_to_idx.get(str(s), -1) for s in target_sids])
    result = np.full((len(target_sids), X_raw.shape[1]), np.nan)
    valid = idx >= 0
    result[valid] = X_raw[idx[valid]]
    n_matched = int(valid.sum())
    print(f"  Gaia features: {n_matched:,} / {len(target_sids):,} matched from {npz_path.name}")
    return result


def _subsample_keep_cutouts(df, cutout_sids, max_points, seed=42):
    has_cutout = df["source_id"].astype(float).astype(str).isin(cutout_sids)
    df_with    = df[has_cutout]
    df_without = df[~has_cutout]
    n_fill = max(0, max_points - len(df_with))
    if n_fill < len(df_without):
        df_without = df_without.sample(n_fill, random_state=seed)
    result = pd.concat([df_with, df_without], ignore_index=True)
    print(f"  Subsampled to {len(result):,} ({len(df_with):,} with cutouts + {len(df_without):,} without)")
    return result


def _subsample_hybrid(df, phz_col, phz_labels, cutout_sids,
                      max_total=5000, floor=500, seed=42):
    """Hybrid density-preserving subsample for All PHZ overlay.

    Classes with fewer than `floor` sources are kept in full.
    The remaining budget (max_total - small classes) is split proportionally
    among the larger classes, preserving their relative densities.
    """
    parts = []
    small_classes = {}
    large_classes = {}
    for v in phz_labels:
        sub = df[df[phz_col] == v]
        if len(sub) == 0:
            continue
        if len(sub) <= floor:
            small_classes[v] = sub
        else:
            large_classes[v] = sub

    # Keep all small classes
    small_total = sum(len(s) for s in small_classes.values())
    for v, sub in small_classes.items():
        lbl = phz_labels[v]
        print(f"     phz={v} {lbl}: {len(sub):,} (all kept, below floor)")
        parts.append(sub)

    # Distribute remaining budget proportionally among large classes
    budget = max(0, max_total - small_total)
    large_total = sum(len(s) for s in large_classes.values())
    for v, sub in large_classes.items():
        lbl = phz_labels[v]
        if large_total > 0:
            n_alloc = max(1, int(round(budget * len(sub) / large_total)))
        else:
            n_alloc = budget
        n_alloc = min(n_alloc, len(sub))
        if n_alloc < len(sub):
            sub = _subsample_keep_cutouts(sub, cutout_sids, n_alloc, seed)
        else:
            print(f"     phz={v} {lbl}: {len(sub):,} (all kept)")
        parts.append(sub)

    result = pd.concat(parts, ignore_index=True) if parts else df
    print(f"  Total after hybrid subsample: {len(result):,} / {len(df):,}")
    return result


def _random_subsample_idx(n_total: int, max_points: int, seed: int = 42) -> np.ndarray:
    """Return indices for a uniform random subsample (density-preserving)."""
    if n_total <= max_points:
        return np.arange(n_total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, max_points, replace=False))


def _compute_gal_ecl_lat(ra_deg: np.ndarray, dec_deg: np.ndarray):
    """Compute galactic latitude (b) and ecliptic latitude from RA/Dec (ICRS degrees).

    Returns (gal_lat, ecl_lat) in degrees.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    gal_lat = coords.galactic.b.deg
    ecl_lat = coords.barycentricmeanecliptic.lat.deg
    return np.asarray(gal_lat), np.asarray(ecl_lat)


def _filter_gaia_xgb_hilat(df_gxgb, min_abs_gal_lat=10.0, min_abs_ecl_lat=45.0):
    """Filter Gaia XGB DataFrame to high galactic + ecliptic latitude sources.

    Uses b and ecl_lat columns if present, otherwise computes from ra/dec.
    Returns filtered DataFrame.
    """
    if "b" in df_gxgb.columns and "ecl_lat" in df_gxgb.columns:
        gal_lat = df_gxgb["b"].values.astype(float)
        ecl_lat = df_gxgb["ecl_lat"].values.astype(float)
    elif "ra" in df_gxgb.columns and "dec" in df_gxgb.columns:
        gal_lat, ecl_lat = _compute_gal_ecl_lat(
            df_gxgb["ra"].values.astype(float),
            df_gxgb["dec"].values.astype(float))
    else:
        return None
    mask = (np.abs(gal_lat) > min_abs_gal_lat) & (np.abs(ecl_lat) > min_abs_ecl_lat)
    return df_gxgb[mask].reset_index(drop=True)


# ── ERO projection helper ─────────────────────────────────────────────────────

def _project_ero(npz_path: Path, q1_mean, q1_std, reducer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project ERO sources from npz into Q1 UMAP space. Returns (emb, y, source_ids)."""
    npz      = np.load(npz_path, allow_pickle=True)
    X_raw    = npz["X_raw"].astype(float)
    y        = npz["y"]
    sids     = npz["source_ids"].astype(str)
    std_safe = np.where(q1_std > 0, q1_std, 1.0)
    X_scaled = (X_raw - q1_mean) / std_safe
    X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0.0)
    emb      = reducer.transform(X_scaled).astype(np.float64)
    return emb, y, sids


# ── 3D Plotly HTML ────────────────────────────────────────────────────────────

def _build_ero_traces(emb, y, gmag, features, feature_names, prefix, traces,
                      sids=None, sid_to_group=None, label_prefix="",
                      cluster_labels=None, q1_cluster_ids=None,
                      model_pgal=None,
                      ero_ra=None, ero_dec=None, ero_class_star=None,
                      gaia_features=None):
    """Add ERO overlay traces as per-cluster sub-traces + legend-only dummy traces.

    Layout appended to `traces`:
      [0 .. n_dummy-1]  : dummy legend traces (showlegend=True, [None] coords)
      [n_dummy .. end]  : sub-traces per group×cluster (showlegend=False,
                          named ``ero_{prefix}_grp={group} C{k}``)

    Returns (added_groups, n_dummy, n_sub, sub_z_gmag, sub_z_model_pgal, sub_z_feat).
    """
    import plotly.graph_objects as go
    added_groups: list[str] = []

    # Determine which groups have data
    group_masks: dict[str, np.ndarray] = {}
    for group, (base_lbl, color) in ERO_GROUP_CONFIGS.items():
        if sids is not None and sid_to_group:
            m = np.array([sid_to_group.get(sid, None) == group for sid in sids])
        else:
            continue
        if m.any():
            group_masks[group] = m
            added_groups.append(group)

    # ── Phase 1: legend dummy traces ──────────────────────────────────────────
    for group in added_groups:
        base_lbl, color = ERO_GROUP_CONFIGS[group]
        display_name = f"{label_prefix}{base_lbl}" if label_prefix else base_lbl
        traces.append(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="markers",
            marker=dict(size=1.8, color=color, opacity=0.7,
                        line=dict(width=0)),
            name=display_name,
            showlegend=True,
            visible=False,
        ))

    n_dummy = len(added_groups)

    # ── Phase 2: sub-traces per group×cluster ─────────────────────────────────
    c_ids = sorted(q1_cluster_ids) if q1_cluster_ids is not None else []
    n_sub = 0
    sub_z_gmag: list[list] = []
    sub_z_model_pgal: list[list] = []
    sub_z_feat: dict[str, list] = {fn: [] for fn in SELECTED_GAIA_PARAMETERS}

    for group in added_groups:
        m_grp = group_masks[group]
        base_lbl, color = ERO_GROUP_CONFIGS[group]

        for k in c_ids:
            mk = m_grp & (cluster_labels == k) if cluster_labels is not None else m_grp
            if not mk.any():
                continue
            n_pts = int(mk.sum())
            z_ero = gmag[mk] if gmag is not None else np.zeros(n_pts)

            sids_m = sids[mk] if sids is not None else np.array([""] * n_pts)
            ra_m  = ero_ra[mk]  if ero_ra  is not None else np.full(n_pts, np.nan)
            dec_m = ero_dec[mk] if ero_dec is not None else np.full(n_pts, np.nan)
            cs_m  = ero_class_star[mk] if ero_class_star is not None else np.full(n_pts, np.nan)
            pg_m  = model_pgal[mk] if model_pgal is not None else np.full(n_pts, np.nan)
            gm_m  = gmag[mk] if gmag is not None else np.full(n_pts, np.nan)
            # customdata: [sid, ra, dec, gmag, class_star, xgb_pgal, feat1, feat2, ...]
            feat_part: list[list] = []
            if features is not None and feature_names:
                cd_arr = features[mk]
                feat_part = [[None if not np.isfinite(v) else round(float(v), 4) for v in row]
                             for row in cd_arr]
            else:
                feat_part = [[] for _ in range(n_pts)]
            cd = [
                [sids_m[i],
                 round(float(ra_m[i]), 4)  if np.isfinite(ra_m[i])  else None,
                 round(float(dec_m[i]), 4) if np.isfinite(dec_m[i]) else None,
                 round(float(gm_m[i]), 2)  if np.isfinite(gm_m[i])  else None,
                 round(float(cs_m[i]), 3)  if np.isfinite(cs_m[i])  else None,
                 round(float(pg_m[i]), 4)  if np.isfinite(pg_m[i])  else None,
                ] + feat_part[i]
                for i in range(n_pts)
            ]

            traces.append(go.Scatter3d(
                x=emb[mk, 0], y=emb[mk, 1], z=z_ero,
                mode="markers",
                marker=dict(size=1.8, color=color, opacity=0.7,
                            line=dict(width=0)),
                name=f"ero_{prefix}_grp={group} C{k}",
                customdata=cd,
                hovertemplate=f"RA=%{{customdata[1]:.4f}}<br>Dec=%{{customdata[2]:+.4f}}<br>G=%{{customdata[3]:.2f}}<extra>{base_lbl}</extra>",
                showlegend=False,
                visible=False,
            ))
            sub_z_gmag.append(z_ero.tolist())
            if model_pgal is not None:
                sub_z_model_pgal.append(
                    [round(float(v), 4) if np.isfinite(v) else None for v in model_pgal[mk]]
                )
            else:
                sub_z_model_pgal.append([None] * n_pts)
            for fi, fn in enumerate(SELECTED_GAIA_PARAMETERS):
                if gaia_features is not None:
                    vals = gaia_features[mk, fi]
                    sub_z_feat[fn].append(
                        [round(float(v), 4) if np.isfinite(v) else None for v in vals])
                else:
                    sub_z_feat[fn].append([None] * n_pts)
            n_sub += 1

    return added_groups, n_dummy, n_sub, sub_z_gmag, sub_z_model_pgal, sub_z_feat


def plot_umap_3d_plotly(
    q1_emb, q1_labels, q1_gmag,
    df_allphz,
    output_dir: "Path | None" = None,
    cutout_cache=None,
    contact_sheet_cache=None,
    sheet_counts=None,
    sheet_extra=None,             # extra pages beyond first {phz: {cluster: [b64,...]}}
    q1_features=None, feature_names=None,
    df_desi=None,
    desi_contact_sheet_cache=None,
    desi_sheet_counts=None,
    desi_sheet_extra=None,
    # ERO
    ero_regular_emb=None, ero_regular_y=None, ero_regular_gmag=None, ero_regular_features=None,
    ero_rebal_emb=None,   ero_rebal_y=None,   ero_rebal_gmag=None,   ero_rebal_features=None,
    ero_feature_names=None,
    ero_reg_sids=None, ero_rebal_sids=None,
    ero_reg_extra=None, ero_rebal_extra=None,
    sid_to_group=None,
    ero_contact_sheet_cache=None,
    ero_sheet_counts=None,
    ero_sheet_extra=None,
    ero_cutout_cache=None,
    # Gaia XGB
    gaia_xgb_emb=None, gaia_xgb_gal_prob=None,
    gaia_xgb_sids=None, gaia_xgb_gmag=None,
    # Pre-computed range overrides (e.g. ERO features — range from ERO distribution)
    feature_ranges_override=None,
    # Z-axis alternative arrays (P(gal) options)
    q1_model_pgal=None,    # shape (n_q1,)  — Q1 ensemble P(gal)
    q1_phz_pgal=None,      # shape (n_q1,)  — PHZ photometric P(gal), NaN where missing
    ero_reg_pgal=None,     # shape (n_ero_reg,) — Q1 model P(gal) for ERO regular sources
    ero_rebal_pgal=None,   # shape (n_ero_rebal,)
    # ERO-DESI overlay
    ero_desi_emb=None,       # (n, 2)
    ero_desi_cat=None,       # (n,) cat_value array
    ero_desi_model_pgal=None,  # (n,) XGB P(gal)
    ero_desi_gmag=None,      # (n,) G-mag for z-axis
    ero_desi_spectype=None,  # (n,) DESI spectype string
    ero_desi_z=None,         # (n,) DESI redshift
    ero_desi_class_star=None,  # (n,) ERO class_star
    ero_desi_ra=None,          # (n,) RA for sidebar
    ero_desi_dec=None,         # (n,) Dec for sidebar
    ero_desi_contact_sheet_cache=None,
    ero_desi_sheet_counts=None,
    ero_desi_sheet_extra=None,
    ero_desi_sids=None,             # (n,) source_id strings for cutout lookup
    # Gaia XGB extra
    gaia_xgb_ra=None,
    gaia_xgb_dec=None,
    # Gaia XGB high-latitude subset
    gaia_hilat_emb=None, gaia_hilat_gal_prob=None,
    gaia_hilat_sids=None, gaia_hilat_gmag=None,
    gaia_hilat_ra=None, gaia_hilat_dec=None,
    density_hist_data=None,   # pre-built per-category histogram dict from _build_density_hists
    # Gaia features for z-axis (14 cols, aligned to source order)
    ero_reg_gaia_features=None, ero_rebal_gaia_features=None,
    gaia_xgb_gaia_features=None, gaia_hilat_gaia_features=None,
    ero_desi_gaia_features=None,
    # Sky data (combined script additions)
    sky_datasets_json=None,
    sky_img_src=None,
    q1_fields_js=None,
    ero_fields_js=None,
):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  plotly not available — pip install plotly"); return

    cluster_ids = sorted(set(q1_labels[q1_labels >= 0]))

    # Feature ranges (Q1 base: Gaia + Q1 Euclid)
    feat_ranges: dict = {}
    if q1_features is not None and feature_names:
        for i, fname in enumerate(feature_names):
            col    = q1_features[:, i]
            finite = col[np.isfinite(col)]
            feat_ranges[fname] = (
                [round(float(np.percentile(finite, 1)), 4),
                 round(float(np.percentile(finite, 99)), 4)]
                if len(finite) else [0.0, 1.0]
            )
    # Apply overrides (e.g. ERO NN-mapped features use ERO source distribution range)
    if feature_ranges_override:
        feat_ranges.update(feature_ranges_override)

    # ERO feature ranges
    ero_feat_ranges: dict = {}
    if ero_regular_features is not None and ero_feature_names:
        for i, fname in enumerate(ero_feature_names):
            col    = ero_regular_features[:, i]
            finite = col[np.isfinite(col)]
            ero_feat_ranges[fname] = (
                [round(float(np.percentile(finite, 1)), 4),
                 round(float(np.percentile(finite, 99)), 4)]
                if len(finite) else [0.0, 1.0]
            )

    def _feat_cd(mask):
        if q1_features is None: return None
        return [[None if not np.isfinite(v) else round(float(v), 4) for v in row]
                for row in q1_features[mask]]

    traces  = []
    b64_js: dict = {}

    # ── Z-array tracking (per trace, in trace order) ──────────────────────────
    _q1_z_gmag       : list = []
    _q1_z_model_pgal : list = []
    _q1_z_phz_pgal   : list = []
    _phz_sub_z_gmag       : list = []
    _phz_sub_z_phz_pgal   : list = []
    _phz_sub_z_model_pgal : list = []
    _ero_reg_z_gmag      : list = []
    _ero_reg_z_model_pgal: list = []
    _ero_rebal_z_gmag      : list = []
    _ero_rebal_z_model_pgal: list = []
    _gaia_z_gmag      : list = []
    _gaia_z_model_pgal: list = []
    _ero_desi_z_gmag      : list = []
    _ero_desi_z_model_pgal: list = []
    # Per-Gaia-feature z-arrays (dict of lists, one list per feature)
    _q1_z_feat:         dict[str, list] = {fn: [] for fn in SELECTED_GAIA_PARAMETERS}
    _phz_sub_z_feat:    dict[str, list] = {fn: [] for fn in SELECTED_GAIA_PARAMETERS}
    _ero_reg_z_feat:    dict[str, list] = {fn: [] for fn in SELECTED_GAIA_PARAMETERS}
    _ero_rebal_z_feat:  dict[str, list] = {fn: [] for fn in SELECTED_GAIA_PARAMETERS}
    _gaia_z_feat:       dict[str, list] = {fn: [] for fn in SELECTED_GAIA_PARAMETERS}
    _gaia_hilat_z_feat: dict[str, list] = {fn: [] for fn in SELECTED_GAIA_PARAMETERS}
    _ero_desi_z_feat:   dict[str, list] = {fn: [] for fn in SELECTED_GAIA_PARAMETERS}

    def _q1_z_model(mask):
        if q1_model_pgal is not None:
            v = q1_model_pgal[mask]
            return [round(float(x), 4) if np.isfinite(x) else None for x in v]
        return None

    def _q1_z_phz(mask):
        if q1_phz_pgal is not None:
            v = q1_phz_pgal[mask]
            return [round(float(x), 4) if np.isfinite(x) else None for x in v]
        return None

    # ── Q1 base ───────────────────────────────────────────────────────────────
    # Q1 base always sits flat at z=0 so overlays are visible above it.
    _q1_ht = "x: %{x:.4f}<br>y: %{y:.4f}<br>z: %{z:.4f}<extra>%{data.name}</extra>"
    noise_mask = q1_labels == -1
    if noise_mask.any():
        z_noise = np.zeros(noise_mask.sum())
        traces.append(go.Scatter3d(
            x=q1_emb[noise_mask, 0], y=q1_emb[noise_mask, 1], z=z_noise,
            mode="markers", marker=dict(size=1, color="#2a2a2a", opacity=0.5),
            name="noise", customdata=_feat_cd(noise_mask),
            hovertemplate=_q1_ht, showlegend=False,
        ))
        _q1_z_gmag.append(z_noise.tolist())
        _m = _q1_z_model(noise_mask); _q1_z_model_pgal.append(_m if _m is not None else z_noise.tolist())
        _p = _q1_z_phz(noise_mask);   _q1_z_phz_pgal.append(_p if _p is not None else z_noise.tolist())
        for fn in SELECTED_GAIA_PARAMETERS:
            _q1_z_feat[fn].append([1.45] * int(noise_mask.sum()))
    for k in cluster_ids:
        m   = q1_labels == k
        z_k = np.zeros(m.sum())
        traces.append(go.Scatter3d(
            x=q1_emb[m, 0], y=q1_emb[m, 1], z=z_k,
            mode="markers",
            marker=dict(size=1, color=CLUSTER_COLORS[k % len(CLUSTER_COLORS)], opacity=0.15),
            name=f"Cluster {k}", customdata=_feat_cd(m),
            hovertemplate=_q1_ht, showlegend=False,
        ))
        _q1_z_gmag.append(z_k.tolist())
        _m = _q1_z_model(m); _q1_z_model_pgal.append(_m if _m is not None else z_k.tolist())
        _p = _q1_z_phz(m);   _q1_z_phz_pgal.append(_p if _p is not None else z_k.tolist())
        for fn in SELECTED_GAIA_PARAMETERS:
            _q1_z_feat[fn].append([1.45] * int(m.sum()))

    # ── PHZ overlay ───────────────────────────────────────────────────────────
    phz_arr      = df_allphz["phz"].values       if df_allphz is not None else np.array([])
    phz_gmag     = df_allphz["phot_g_mean_mag"].values.astype(float) if df_allphz is not None else np.array([])
    phz_emb      = df_allphz[["umap1","umap2"]].values if df_allphz is not None else np.zeros((0,2))
    _phz_cl      = df_allphz["cluster"].values   if (df_allphz is not None and "cluster" in df_allphz.columns) else None
    _phz_pgal_col   = df_allphz["phz_pgal"].values.astype(float)   if (df_allphz is not None and "phz_pgal"   in df_allphz.columns) else None
    _phz_model_col  = df_allphz["model_pgal"].values.astype(float) if (df_allphz is not None and "model_pgal" in df_allphz.columns) else None

    desi_arr        = df_desi["phz"].values       if df_desi is not None else np.array([])
    desi_gmag       = df_desi["phot_g_mean_mag"].values.astype(float) if df_desi is not None else np.array([])
    desi_emb        = df_desi[["umap1","umap2"]].values if df_desi is not None else np.zeros((0,2))
    _desi_cl        = df_desi["cluster"].values   if (df_desi is not None and "cluster" in df_desi.columns) else None
    _desi_phz_col   = df_desi["phz_pgal"].values.astype(float)   if (df_desi is not None and "phz_pgal"   in df_desi.columns) else None
    _desi_model_col = df_desi["model_pgal"].values.astype(float) if (df_desi is not None and "model_pgal" in df_desi.columns) else None

    for v, (lbl, col, sym) in PHZ_LABELS.items():
        m_v = phz_arr == v
        if not m_v.any(): continue
        mkr = dict(size=1.8, color=col, symbol=sym, opacity=0.7,
                   line=dict(width=0))
        traces.append(go.Scatter3d(
            x=[float("nan")], y=[float("nan")], z=[float("nan")],
            mode="markers", marker=mkr, name=f"phz={v} {lbl}",
            showlegend=True, hoverinfo="skip",
        ))
        sub_clusters = sorted(set(int(c) for c in _phz_cl[m_v])) if _phz_cl is not None else [-1]
        for k in sub_clusters:
            mk = m_v & (_phz_cl == k) if _phz_cl is not None else m_v
            if not mk.any(): continue
            z_v = phz_gmag[mk]; custom = []
            if df_allphz is not None:
                sub     = df_allphz[mk].reset_index(drop=True)
                n_sub   = len(sub)
                sids_a  = sub["source_id"].astype(str).values
                ras_a   = np.round(sub["ra"].values.astype(float), 4)
                decs_a  = np.round(sub["dec"].values.astype(float), 4)
                gmags_a = np.round(sub["phot_g_mean_mag"].values.astype(float), 4)
                lbl_val = f"phz={v} {lbl}"
                feat_arrs = [
                    pd.to_numeric(sub[fn], errors="coerce").values.astype(float)
                    if fn in sub.columns else np.full(n_sub, np.nan)
                    for fn in (feature_names or [])
                ]
                # Per-source phz_pgal and model_pgal for sidebar display
                pp_a = _phz_pgal_col[mk] if _phz_pgal_col is not None else np.full(n_sub, np.nan)
                mp_a = _phz_model_col[mk] if _phz_model_col is not None else np.full(n_sub, np.nan)
                for i in range(n_sub):
                    sid   = sids_a[i]
                    pp_v  = None if not np.isfinite(pp_a[i]) else round(float(pp_a[i]), 4)
                    mp_v  = None if not np.isfinite(mp_a[i]) else round(float(mp_a[i]), 4)
                    feats = [None if not np.isfinite(fa[i]) else round(float(fa[i]), 4)
                             for fa in feat_arrs]
                    # customdata: [sid, ra, dec, gmag, label, phz_pgal, model_pgal, desi_z, feat1, ...]
                    custom.append([sid, float(ras_a[i]), float(decs_a[i]),
                                   float(gmags_a[i]), lbl_val, pp_v, mp_v, None] + feats)
                    if cutout_cache:
                        ck = str(float(sid)) if sid else ""
                        if ck in cutout_cache: b64_js[sid] = cutout_cache[ck]
            traces.append(go.Scatter3d(
                x=phz_emb[mk,0], y=phz_emb[mk,1], z=z_v, mode="markers", marker=mkr,
                name=f"phz={v} C{k}", customdata=custom,
                hovertemplate="RA=%{customdata[1]:.4f}<br>Dec=%{customdata[2]:+.4f}<br>G=%{customdata[3]:.2f}<extra>%{customdata[4]}</extra>",
                showlegend=False,
            ))
            _phz_sub_z_gmag.append(z_v.tolist())
            # phz_pgal: None where missing so out-of-[0,1] fallback never clips
            _phz_sub_z_phz_pgal.append(
                [float(x) if np.isfinite(x) else None for x in _phz_pgal_col[mk]]
                if _phz_pgal_col is not None else [None] * int(mk.sum())
            )
            # model_pgal: XGB P(gal) applied to these Q1 sources
            _phz_sub_z_model_pgal.append(
                [float(x) if np.isfinite(x) else None for x in _phz_model_col[mk]]
                if _phz_model_col is not None else [None] * int(mk.sum())
            )
            # Per-Gaia-feature z-arrays
            for fi, fn in enumerate(SELECTED_GAIA_PARAMETERS):
                if df_allphz is not None and fn in df_allphz.columns:
                    vals = df_allphz[fn].values.astype(float)[mk]
                    _phz_sub_z_feat[fn].append(
                        [round(float(x), 4) if np.isfinite(x) else None for x in vals])
                else:
                    _phz_sub_z_feat[fn].append([None] * int(mk.sum()))

    # ── DESI overlay ──────────────────────────────────────────────────────────
    n_pre_desi = len(traces)
    for v, (lbl, col, sym) in DESI_CATEGORY_LABELS.items():
        m_v = desi_arr == v
        if not m_v.any(): continue
        mkr = dict(size=1.8, color=col, symbol=sym, opacity=0.7,
                   line=dict(width=0))
        traces.append(go.Scatter3d(
            x=[float("nan")], y=[float("nan")], z=[float("nan")],
            mode="markers", marker=mkr, name=f"Q1 phz={v} {lbl}",
            showlegend=True, hoverinfo="skip", visible=False,
        ))
        sub_clusters = sorted(set(int(c) for c in _desi_cl[m_v])) if _desi_cl is not None else [-1]
        for k in sub_clusters:
            mk = m_v & (_desi_cl == k) if _desi_cl is not None else m_v
            if not mk.any(): continue
            z_v = desi_gmag[mk]; custom = []
            if df_desi is not None:
                sub      = df_desi[mk].reset_index(drop=True)
                n_sub    = len(sub)
                sids_a   = sub["source_id"].astype(str).values
                ras_a    = np.round(sub["ra"].values.astype(float), 4)
                decs_a   = np.round(sub["dec"].values.astype(float), 4)
                gmags_a  = np.round(sub["phot_g_mean_mag"].values.astype(float), 4)
                desi_z_a = sub["d_desi_z"].values if "d_desi_z" in sub.columns else np.full(n_sub, np.nan)
                feat_arrs = [
                    pd.to_numeric(sub[fn], errors="coerce").values.astype(float)
                    if fn in sub.columns else np.full(n_sub, np.nan)
                    for fn in (feature_names or [])
                ]
                # Per-source phz_pgal and model_pgal for sidebar display
                dp_a = _desi_phz_col[mk] if _desi_phz_col is not None else np.full(n_sub, np.nan)
                dm_a = _desi_model_col[mk] if _desi_model_col is not None else np.full(n_sub, np.nan)
                for i in range(n_sub):
                    sid   = sids_a[i]
                    z_val = desi_z_a[i]
                    pp_v  = None if not np.isfinite(dp_a[i]) else round(float(dp_a[i]), 4)
                    mp_v  = None if not np.isfinite(dm_a[i]) else round(float(dm_a[i]), 4)
                    feats = [None if not np.isfinite(fa[i]) else round(float(fa[i]), 4)
                             for fa in feat_arrs]
                    # customdata: [sid, ra, dec, gmag, label, phz_pgal, model_pgal, desi_z, feat1, ...]
                    custom.append([sid, float(ras_a[i]), float(decs_a[i]),
                                   float(gmags_a[i]), f"phz={v} {lbl}",
                                   pp_v, mp_v,
                                   None if not np.isfinite(z_val) else round(float(z_val), 4)]
                                  + feats)
                    if cutout_cache:
                        ck = str(float(sid)) if sid else ""
                        if ck in cutout_cache: b64_js[sid] = cutout_cache[ck]
            traces.append(go.Scatter3d(
                x=desi_emb[mk,0], y=desi_emb[mk,1], z=z_v, mode="markers", marker=mkr,
                name=f"phz={v} C{k}", customdata=custom,
                hovertemplate="RA=%{customdata[1]:.4f}<br>Dec=%{customdata[2]:+.4f}<br>G=%{customdata[3]:.2f}<extra>%{customdata[4]}</extra>",
                showlegend=False, visible=False,
            ))
            # DESI traces share the same Z tracking lists as PHZ (same JS helper selects both).
            # DESI = Q1-DESI matched sources → they have PHZ P(gal) and XGB P(gal).
            _phz_sub_z_gmag.append(z_v.tolist())
            _phz_sub_z_phz_pgal.append(
                [float(x) if np.isfinite(x) else None for x in _desi_phz_col[mk]]
                if _desi_phz_col is not None else [None] * int(mk.sum())
            )
            _phz_sub_z_model_pgal.append(
                [float(x) if np.isfinite(x) else None for x in _desi_model_col[mk]]
                if _desi_model_col is not None else [None] * int(mk.sum())
            )
            # Per-Gaia-feature z-arrays (DESI shares PHZ tracking lists)
            for fi, fn in enumerate(SELECTED_GAIA_PARAMETERS):
                if df_desi is not None and fn in df_desi.columns:
                    vals = df_desi[fn].values.astype(float)[mk]
                    _phz_sub_z_feat[fn].append(
                        [round(float(x), 4) if np.isfinite(x) else None for x in vals])
                else:
                    _phz_sub_z_feat[fn].append([None] * int(mk.sum()))

    # ── ERO overlays (regular + rebalanced, both hidden by default) ───────────
    # Each call appends n_dummy legend traces then n_sub data sub-traces.
    _ero_reg_cl = (assign_to_clusters(ero_regular_emb, q1_emb, q1_labels)
                   if ero_regular_emb is not None else None)
    _ero_rebal_cl = (assign_to_clusters(ero_rebal_emb, q1_emb, q1_labels)
                     if ero_rebal_emb is not None else None)
    _q1_cluster_ids = sorted(set(q1_labels[q1_labels >= 0]))

    # ERO extra data (RA, Dec, class_star) passed from caller
    _ero_reg_extra = ero_reg_extra or {}
    _ero_rebal_extra = ero_rebal_extra or {}

    n_pre_ero_regular = len(traces)
    ero_trace_groups: dict[str, str] = {}   # {str(global_trace_idx): group_name}
    n_ero_reg_dummy = 0
    n_ero_reg_sub   = 0
    if ero_regular_emb is not None and ero_regular_y is not None:
        reg_groups, n_ero_reg_dummy, n_ero_reg_sub, _ero_reg_z_gmag, _ero_reg_z_model_pgal, _ero_reg_z_feat = \
            _build_ero_traces(
                ero_regular_emb, ero_regular_y, ero_regular_gmag,
                ero_regular_features, ero_feature_names, "ero", traces,
                sids=ero_reg_sids, sid_to_group=sid_to_group,
                cluster_labels=_ero_reg_cl, q1_cluster_ids=_q1_cluster_ids,
                model_pgal=ero_reg_pgal,
                ero_ra=_ero_reg_extra.get("ra"),
                ero_dec=_ero_reg_extra.get("dec"),
                ero_class_star=_ero_reg_extra.get("class_star"),
                gaia_features=ero_reg_gaia_features,
            )
        for offset, grp in enumerate(reg_groups):
            ero_trace_groups[str(n_pre_ero_regular + offset)] = grp
    n_ero_reg_sub_start = n_pre_ero_regular + n_ero_reg_dummy

    n_pre_ero_rebal = len(traces)
    n_ero_rebal_dummy = 0
    n_ero_rebal_sub   = 0
    if ero_rebal_emb is not None and ero_rebal_y is not None:
        rebal_groups, n_ero_rebal_dummy, n_ero_rebal_sub, _ero_rebal_z_gmag, _ero_rebal_z_model_pgal, _ero_rebal_z_feat = \
            _build_ero_traces(
                ero_rebal_emb, ero_rebal_y, ero_rebal_gmag,
                ero_rebal_features, ero_feature_names, "eror", traces,
                sids=ero_rebal_sids, sid_to_group=sid_to_group,
                label_prefix="",
                cluster_labels=_ero_rebal_cl, q1_cluster_ids=_q1_cluster_ids,
                model_pgal=ero_rebal_pgal,
                ero_ra=_ero_rebal_extra.get("ra"),
                ero_dec=_ero_rebal_extra.get("dec"),
                ero_class_star=_ero_rebal_extra.get("class_star"),
                gaia_features=ero_rebal_gaia_features,
            )
        for offset, grp in enumerate(rebal_groups):
            ero_trace_groups[str(n_pre_ero_rebal + offset)] = grp
    n_ero_rebal_sub_start = n_pre_ero_rebal + n_ero_rebal_dummy

    # ── Gaia XGB overlay (legend + sub-traces per category×cluster) ──────────
    _GAIA_MK_SIZE = 1.8  # match compile_umap_sky.py overlay size
    n_pre_gaia_xgb     = len(traces)
    n_gaia_xgb_legend  = 0
    _gaia_xgb_cl = (assign_to_clusters(gaia_xgb_emb, q1_emb, q1_labels)
                    if gaia_xgb_emb is not None else None)
    _gaia_xgb_cats: list[str] = []  # category labels for legend traces
    _gaia_cat_data: list[tuple] = []  # (mask, short_name, lbl, color)
    if gaia_xgb_emb is not None and gaia_xgb_gal_prob is not None:
        m_gal  = gaia_xgb_gal_prob >= GAL_THRESH
        m_star = ~m_gal
        gaia_pgal = gaia_xgb_gal_prob
        for mask, short, lbl, color in [
            (m_star, "star", f"XGB Star  (P(gal) < {GAL_THRESH})", "#F2BDCE"),
            (m_gal,  "gal",  f"XGB Galaxy  (P(gal) ≥ {GAL_THRESH})",  "#91D3F0"),
        ]:
            if not mask.any():
                continue
            # Legend trace
            traces.append(go.Scatter3d(
                x=[None], y=[None], z=[None], mode="markers",
                marker=dict(size=_GAIA_MK_SIZE, color=color, opacity=0.7,
                            line=dict(width=0)),
                name=lbl, showlegend=True, visible=False,
            ))
            _gaia_xgb_cats.append(short)
            _gaia_cat_data.append((mask, short, lbl, color))
            n_gaia_xgb_legend += 1

    # Sub-traces per category×cluster
    n_pre_gaia_xgb_sub = len(traces)
    n_gaia_xgb_sub     = 0
    if gaia_xgb_emb is not None and gaia_xgb_gal_prob is not None:
        gaia_pgal = gaia_xgb_gal_prob
        for mask, short, lbl, color in _gaia_cat_data:
            for k in _q1_cluster_ids:
                mk = mask & (_gaia_xgb_cl == k) if _gaia_xgb_cl is not None else mask
                if not mk.any():
                    continue
                n_pts   = int(mk.sum())
                sids_m  = gaia_xgb_sids[mk] if gaia_xgb_sids is not None else np.array([""] * n_pts)
                probs_m = gaia_xgb_gal_prob[mk]
                z_g     = gaia_xgb_gmag[mk] if gaia_xgb_gmag is not None else np.zeros(n_pts)
                ra_m    = gaia_xgb_ra[mk]  if gaia_xgb_ra  is not None else np.full(n_pts, np.nan)
                dec_m   = gaia_xgb_dec[mk] if gaia_xgb_dec is not None else np.full(n_pts, np.nan)
                gm_m    = gaia_xgb_gmag[mk] if gaia_xgb_gmag is not None else np.full(n_pts, np.nan)
                # customdata: [sid, gal_prob, ra, dec, gmag]
                cd = [[str(sids_m[i]), round(float(probs_m[i]), 4),
                       round(float(ra_m[i]), 4)  if np.isfinite(ra_m[i])  else None,
                       round(float(dec_m[i]), 4) if np.isfinite(dec_m[i]) else None,
                       round(float(gm_m[i]), 2)  if np.isfinite(gm_m[i])  else None]
                      for i in range(n_pts)]
                traces.append(go.Scatter3d(
                    x=gaia_xgb_emb[mk, 0], y=gaia_xgb_emb[mk, 1], z=z_g,
                    mode="markers",
                    marker=dict(size=_GAIA_MK_SIZE, color=color, opacity=0.7,
                                line=dict(width=0)),
                    name=f"gxgb_{short} C{k}",
                    hovertemplate=f"RA=%{{customdata[2]:.4f}}<br>Dec=%{{customdata[3]:+.4f}}<br>G=%{{customdata[4]:.2f}}<extra>{lbl.split('(')[0].strip()}</extra>",
                    customdata=cd, showlegend=False, visible=False,
                ))
                _gaia_z_gmag.append(z_g.tolist())
                _gaia_z_model_pgal.append([round(float(v), 4) for v in gaia_pgal[mk]])
                for fi, fn in enumerate(SELECTED_GAIA_PARAMETERS):
                    if gaia_xgb_gaia_features is not None:
                        vals = gaia_xgb_gaia_features[mk, fi]
                        _gaia_z_feat[fn].append(
                            [round(float(v), 4) if np.isfinite(v) else None for v in vals])
                    else:
                        _gaia_z_feat[fn].append([None] * n_pts)
                n_gaia_xgb_sub += 1

    # ── Gaia XGB high-lat overlay (legend + sub-traces, same structure) ──────
    n_pre_gaia_hilat     = len(traces)
    n_gaia_hilat_legend  = 0
    _gaia_hilat_cl = (assign_to_clusters(gaia_hilat_emb, q1_emb, q1_labels)
                      if gaia_hilat_emb is not None else None)
    _gaia_hilat_cats: list[str] = []
    _gaia_hilat_cat_data: list[tuple] = []
    _gaia_hilat_z_gmag: list[list] = []
    _gaia_hilat_z_model_pgal: list[list] = []
    if gaia_hilat_emb is not None and gaia_hilat_gal_prob is not None:
        m_gal_hl  = gaia_hilat_gal_prob >= GAL_THRESH
        m_star_hl = ~m_gal_hl
        hilat_pgal = gaia_hilat_gal_prob
        for mask, short, lbl, color in [
            (m_star_hl, "star", f"XGB Star  (P(gal) < {GAL_THRESH})", "#F2BDCE"),
            (m_gal_hl,  "gal",  f"XGB Galaxy  (P(gal) ≥ {GAL_THRESH})",  "#91D3F0"),
        ]:
            if not mask.any():
                continue
            traces.append(go.Scatter3d(
                x=[None], y=[None], z=[None], mode="markers",
                marker=dict(size=_GAIA_MK_SIZE, color=color, opacity=0.7,
                            line=dict(width=0)),
                name=lbl, showlegend=True, visible=False,
            ))
            _gaia_hilat_cats.append(short)
            _gaia_hilat_cat_data.append((mask, short, lbl, color))
            n_gaia_hilat_legend += 1

    # High-lat sub-traces per category×cluster
    n_pre_gaia_hilat_sub = len(traces)
    n_gaia_hilat_sub     = 0
    if gaia_hilat_emb is not None and gaia_hilat_gal_prob is not None:
        hilat_pgal = gaia_hilat_gal_prob
        for mask, short, lbl, color in _gaia_hilat_cat_data:
            for k in _q1_cluster_ids:
                mk = mask & (_gaia_hilat_cl == k) if _gaia_hilat_cl is not None else mask
                if not mk.any():
                    continue
                n_pts   = int(mk.sum())
                sids_m  = gaia_hilat_sids[mk] if gaia_hilat_sids is not None else np.array([""] * n_pts)
                probs_m = gaia_hilat_gal_prob[mk]
                z_g     = gaia_hilat_gmag[mk] if gaia_hilat_gmag is not None else np.zeros(n_pts)
                ra_m    = gaia_hilat_ra[mk]  if gaia_hilat_ra  is not None else np.full(n_pts, np.nan)
                dec_m   = gaia_hilat_dec[mk] if gaia_hilat_dec is not None else np.full(n_pts, np.nan)
                gm_m    = gaia_hilat_gmag[mk] if gaia_hilat_gmag is not None else np.full(n_pts, np.nan)
                cd = [[str(sids_m[i]), round(float(probs_m[i]), 4),
                       round(float(ra_m[i]), 4)  if np.isfinite(ra_m[i])  else None,
                       round(float(dec_m[i]), 4) if np.isfinite(dec_m[i]) else None,
                       round(float(gm_m[i]), 2)  if np.isfinite(gm_m[i])  else None]
                      for i in range(n_pts)]
                traces.append(go.Scatter3d(
                    x=gaia_hilat_emb[mk, 0], y=gaia_hilat_emb[mk, 1], z=z_g,
                    mode="markers",
                    marker=dict(size=_GAIA_MK_SIZE, color=color, opacity=0.7,
                                line=dict(width=0)),
                    name=f"gxhl_{short} C{k}",
                    hovertemplate=f"RA=%{{customdata[2]:.4f}}<br>Dec=%{{customdata[3]:+.4f}}<br>G=%{{customdata[4]:.2f}}<extra>{lbl.split('(')[0].strip()}</extra>",
                    customdata=cd, showlegend=False, visible=False,
                ))
                _gaia_hilat_z_gmag.append(z_g.tolist())
                _gaia_hilat_z_model_pgal.append([round(float(v), 4) for v in hilat_pgal[mk]])
                for fi, fn in enumerate(SELECTED_GAIA_PARAMETERS):
                    if gaia_hilat_gaia_features is not None:
                        vals = gaia_hilat_gaia_features[mk, fi]
                        _gaia_hilat_z_feat[fn].append(
                            [round(float(v), 4) if np.isfinite(v) else None for v in vals])
                    else:
                        _gaia_hilat_z_feat[fn].append([None] * n_pts)
                n_gaia_hilat_sub += 1

    # ── ERO-DESI overlay (sub-traces per cluster, like PHZ) ──────────────────
    n_pre_ero_desi       = len(traces)
    n_ero_desi_legend    = 0   # number of legend traces
    n_ero_desi_sub       = 0   # number of sub-traces (per cat × cluster)
    _ero_desi_trace_cat: list[int] = []   # cat_value for each legend trace
    _ero_desi_cl = (assign_to_clusters(ero_desi_emb, q1_emb, q1_labels)
                    if ero_desi_emb is not None else None)
    if ero_desi_emb is not None and ero_desi_cat is not None:
        for cat_val, (lbl, color) in sorted(ERO_DESI_LABELS.items()):
            m = ero_desi_cat == cat_val
            if not m.any():
                continue
            n_pts = int(m.sum())
            # Legend trace (no data — just shows in legend)
            traces.append(go.Scatter3d(
                x=[None], y=[None], z=[None], mode="markers",
                marker=dict(size=1.8, color=color, opacity=0.7,
                            line=dict(width=0)),
                name=lbl,
                showlegend=True, visible=False,
            ))
            _ero_desi_trace_cat.append(int(cat_val))
            n_ero_desi_legend += 1

    # Sub-traces: one per (cat × cluster), like PHZ sub-traces
    n_pre_ero_desi_sub = len(traces)
    if ero_desi_emb is not None and ero_desi_cat is not None:
        for cat_val, (lbl, color) in sorted(ERO_DESI_LABELS.items()):
            m_cat = ero_desi_cat == cat_val
            if not m_cat.any():
                continue
            for k in sorted(set(q1_labels[q1_labels >= 0])):
                mk = m_cat & (_ero_desi_cl == k) if _ero_desi_cl is not None else m_cat
                if not mk.any():
                    continue
                n_k   = int(mk.sum())
                z_ed  = ero_desi_gmag[mk] if ero_desi_gmag is not None else np.zeros(n_k)
                pgal  = ero_desi_model_pgal[mk] if ero_desi_model_pgal is not None else np.full(n_k, np.nan)
                stype = ero_desi_spectype[mk] if ero_desi_spectype is not None else np.full(n_k, '', dtype=object)
                dz    = ero_desi_z[mk] if ero_desi_z is not None else np.full(n_k, np.nan)
                cs    = ero_desi_class_star[mk] if ero_desi_class_star is not None else np.full(n_k, np.nan)
                sids  = ero_desi_sids[mk] if ero_desi_sids is not None else np.full(n_k, '', dtype=object)
                ra_m  = ero_desi_ra[mk]  if ero_desi_ra  is not None else np.full(n_k, np.nan)
                dec_m = ero_desi_dec[mk] if ero_desi_dec is not None else np.full(n_k, np.nan)
                gm_m  = ero_desi_gmag[mk] if ero_desi_gmag is not None else np.full(n_k, np.nan)
                # customdata: [cat_label, pgal, spectype, desi_z, class_star, source_id, ra, dec, gmag]
                cd    = [[f"cat={cat_val}",
                          round(float(pgal[i]), 4) if np.isfinite(pgal[i]) else None,
                          str(stype[i]),
                          round(float(dz[i]), 4) if np.isfinite(dz[i]) else None,
                          round(float(cs[i]), 3) if np.isfinite(cs[i]) else None,
                          str(sids[i]),
                          round(float(ra_m[i]), 4)  if np.isfinite(ra_m[i])  else None,
                          round(float(dec_m[i]), 4) if np.isfinite(dec_m[i]) else None,
                          round(float(gm_m[i]), 2)  if np.isfinite(gm_m[i])  else None]
                         for i in range(n_k)]
                traces.append(go.Scatter3d(
                    x=ero_desi_emb[mk, 0], y=ero_desi_emb[mk, 1], z=z_ed,
                    mode="markers",
                    marker=dict(size=1.8, color=color, opacity=0.7,
                                line=dict(width=0)),
                    name=f"ed_cat={cat_val} C{k}",
                    customdata=cd, showlegend=False, visible=False,
                    hovertemplate=f"RA=%{{customdata[6]:.4f}}<br>Dec=%{{customdata[7]:+.4f}}<br>G=%{{customdata[8]:.2f}}<extra>{lbl}</extra>",
                ))
                _ero_desi_z_gmag.append(z_ed.tolist())
                _ero_desi_z_model_pgal.append(
                    [round(float(v), 4) if np.isfinite(v) else None for v in pgal]
                )
                for fi, fn in enumerate(SELECTED_GAIA_PARAMETERS):
                    if ero_desi_gaia_features is not None:
                        vals = ero_desi_gaia_features[mk, fi]
                        _ero_desi_z_feat[fn].append(
                            [round(float(v), 4) if np.isfinite(v) else None for v in vals])
                    else:
                        _ero_desi_z_feat[fn].append([None] * n_pts)
                n_ero_desi_sub += 1

    # Compute which Q1 clusters each ERO-DESI cat_value populates
    _ero_desi_cluster_coverage: dict[str, list[int]] = {}
    if ero_desi_emb is not None and ero_desi_cat is not None and _ero_desi_cl is not None:
        _ed_clusters = _ero_desi_cl
        for cat_val in sorted(ERO_DESI_LABELS.keys()):
            m = ero_desi_cat == cat_val
            if m.any():
                _ero_desi_cluster_coverage[str(cat_val)] = sorted(
                    set(int(c) for c in _ed_clusters[m] if c >= 0)
                )

    # ── Axis ranges ───────────────────────────────────────────────────────────
    _pad    = 0.05
    _extra_x = [q1_emb[:, 0]]; _extra_y = [q1_emb[:, 1]]
    for _e in [phz_emb, desi_emb]:
        if len(_e): _extra_x.append(_e[:,0]); _extra_y.append(_e[:,1])
    for _e in [ero_regular_emb, ero_rebal_emb]:
        if _e is not None and len(_e): _extra_x.append(_e[:,0]); _extra_y.append(_e[:,1])
    if gaia_xgb_emb is not None and len(gaia_xgb_emb):
        _extra_x.append(gaia_xgb_emb[:, 0]); _extra_y.append(gaia_xgb_emb[:, 1])
    if ero_desi_emb is not None and len(ero_desi_emb):
        _extra_x.append(ero_desi_emb[:, 0]); _extra_y.append(ero_desi_emb[:, 1])
    _all_x = np.concatenate(_extra_x); _all_y = np.concatenate(_extra_y)
    _xr = [float(_all_x.min()-np.ptp(_all_x)*_pad), float(_all_x.max()+np.ptp(_all_x)*_pad)]
    _yr = [float(_all_y.min()-np.ptp(_all_y)*_pad), float(_all_y.max()+np.ptp(_all_y)*_pad)]
    if q1_gmag is not None and len(phz_gmag):
        _gz   = np.concatenate([q1_gmag[np.isfinite(q1_gmag)], phz_gmag[np.isfinite(phz_gmag)]])
        _zpad = float(np.ptp(_gz)*_pad)
        # Include 0 for the Q1 base plane (always flat at z=0)
        _zr   = [float(_gz.max())+_zpad, min(float(_gz.min())-_zpad, -_zpad)]
    elif q1_gmag is not None:
        _gz   = q1_gmag[np.isfinite(q1_gmag)]
        _zpad = float(np.ptp(_gz)*_pad)
        _zr   = [float(_gz.max())+_zpad, min(float(_gz.min())-_zpad, -_zpad)]
    else:
        _zr = None

    _ax = dict(backgroundcolor="#080808", gridcolor="#1c1c1c", showbackground=True,
               zerolinecolor="#222", tickfont=dict(color="#444",size=9),
               title_font=dict(color="#666",size=10))
    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor="#080808", plot_bgcolor="#080808",
        font=dict(color="#aaa",
                  family='-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif',
                  size=11),
        scene=dict(
            bgcolor="#080808", uirevision="locked",
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(**_ax, title="UMAP 1", autorange=False, range=_xr),
            yaxis=dict(**_ax, title="UMAP 2", autorange=False, range=_yr),
            zaxis=dict(**_ax, title="G mag",
                       autorange=False, range=[-0.05, 1.45]),
        ),
        uirevision="locked",
        legend=dict(bgcolor="rgba(12,12,12,0.9)", bordercolor="#222", borderwidth=1,
                    font=dict(color="#999",size=11), itemsizing="constant",
                    x=1, xanchor="right", y=0.96, yanchor="top"),
        margin=dict(l=0,r=0,b=0,t=0), width=1100, height=750,
    )

    plot_div = fig.to_html(full_html=False, include_plotlyjs="cdn",
                           config=dict(displaylogo=False, displayModeBar="hover",
                                       modeBarButtonsToRemove=["toImage","sendDataToCloud","editInChartStudio"]))

    # JSON blobs
    b64_json                 = json.dumps(b64_js)
    n_cached                 = len(b64_js)
    contact_sheets_json      = json.dumps(contact_sheet_cache or {})
    sheet_counts_json        = json.dumps(sheet_counts or {})
    sheet_extra_json         = json.dumps(sheet_extra or {})
    desi_contact_sheets_json = json.dumps(desi_contact_sheet_cache or {})
    desi_sheet_counts_json   = json.dumps(desi_sheet_counts or {})
    desi_sheet_extra_json    = json.dumps(desi_sheet_extra or {})
    desi_trace_start_json    = json.dumps(n_pre_desi)
    ero_regular_start_json   = json.dumps(n_pre_ero_regular)
    ero_rebal_start_json     = json.dumps(n_pre_ero_rebal)
    ero_reg_n_dummy_json       = json.dumps(n_ero_reg_dummy)
    ero_reg_sub_start_json     = json.dumps(n_ero_reg_sub_start)
    ero_reg_n_sub_json         = json.dumps(n_ero_reg_sub)
    ero_rebal_n_dummy_json     = json.dumps(n_ero_rebal_dummy)
    ero_rebal_sub_start_json   = json.dumps(n_ero_rebal_sub_start)
    ero_rebal_n_sub_json       = json.dumps(n_ero_rebal_sub)
    feature_names_json       = json.dumps(feature_names or [])
    feature_ranges_json      = json.dumps(feat_ranges)
    ero_feature_names_json        = json.dumps(ero_feature_names or [])
    ero_feature_ranges_json       = json.dumps(ero_feat_ranges)
    ero_cutouts_json              = json.dumps(ero_cutout_cache or {})
    ero_contact_sheets_json       = json.dumps(ero_contact_sheet_cache or {})
    ero_sheet_counts_json         = json.dumps(ero_sheet_counts or {})
    ero_sheet_extra_json          = json.dumps(ero_sheet_extra or {})
    ero_group_labels_json         = json.dumps(ERO_GROUP_LABELS)
    ero_trace_groups_json         = json.dumps(ero_trace_groups)
    ero_group_configs_json        = json.dumps(
        {g: {"label": l, "color": c} for g, (l, c) in ERO_GROUP_CONFIGS.items()}
    )
    cluster_colors_json           = json.dumps(CLUSTER_COLORS)
    phz_colors_json               = json.dumps({str(v): c for v, (_, c, _) in PHZ_LABELS.items()})
    phz_names_json                = json.dumps({str(v): lbl for v, (lbl, _, _) in PHZ_LABELS.items()})
    desi_names_json               = json.dumps({str(v): lbl for v, (lbl, _, _) in DESI_CATEGORY_LABELS.items()})
    ero_desi_colors_json          = json.dumps({str(v): c for v, (_, c) in ERO_DESI_LABELS.items()})
    gaia_xgb_start_json           = json.dumps(n_pre_gaia_xgb)
    gaia_xgb_n_legend_json        = json.dumps(n_gaia_xgb_legend)
    gaia_xgb_sub_start_json       = json.dumps(n_pre_gaia_xgb_sub)
    gaia_xgb_n_sub_json           = json.dumps(n_gaia_xgb_sub)
    gaia_xgb_cats_json            = json.dumps(_gaia_xgb_cats)
    gaia_hilat_start_json         = json.dumps(n_pre_gaia_hilat)
    gaia_hilat_n_legend_json      = json.dumps(n_gaia_hilat_legend)
    gaia_hilat_sub_start_json     = json.dumps(n_pre_gaia_hilat_sub)
    gaia_hilat_n_sub_json         = json.dumps(n_gaia_hilat_sub)
    gaia_hilat_cats_json          = json.dumps(_gaia_hilat_cats)
    ero_desi_start_json           = json.dumps(n_pre_ero_desi)
    ero_desi_n_legend_json        = json.dumps(n_ero_desi_legend)
    ero_desi_sub_start_json       = json.dumps(n_pre_ero_desi_sub)
    ero_desi_n_sub_json           = json.dumps(n_ero_desi_sub)
    ero_desi_contact_sheets_json  = json.dumps(ero_desi_contact_sheet_cache or {})
    ero_desi_sheet_counts_json    = json.dumps(ero_desi_sheet_counts or {})
    ero_desi_sheet_extra_json     = json.dumps(ero_desi_sheet_extra or {})
    ero_desi_cluster_coverage_json = json.dumps(_ero_desi_cluster_coverage)
    ero_desi_trace_cat_json        = json.dumps(_ero_desi_trace_cat)
    n_base_traces            = (1 if noise_mask.any() else 0) + len(cluster_ids)
    # ── Normalise G-mag z arrays to [0, 1] so the z-axis stays fixed ────────
    _all_gmag_vals: list[float] = []
    for _zlist in (_phz_sub_z_gmag, _ero_reg_z_gmag, _ero_rebal_z_gmag,
                   _gaia_z_gmag, _gaia_hilat_z_gmag, _ero_desi_z_gmag):
        for _arr in _zlist:
            _all_gmag_vals.extend(v for v in _arr if v is not None)
    if q1_gmag is not None:
        _all_gmag_vals.extend(float(v) for v in q1_gmag[np.isfinite(q1_gmag)])
    if _all_gmag_vals:
        _gmag_lo = min(_all_gmag_vals)
        _gmag_hi = max(_all_gmag_vals)
        _gmag_pct = np.array(_all_gmag_vals)
        _gmag_color_lo = round(float(np.percentile(_gmag_pct, 1)), 4)
        _gmag_color_hi = round(float(np.percentile(_gmag_pct, 99)), 4)
    else:
        _gmag_lo, _gmag_hi = 17.0, 22.0
        _gmag_color_lo, _gmag_color_hi = 17.0, 22.0
    _gmag_span = _gmag_hi - _gmag_lo if _gmag_hi > _gmag_lo else 1.0

    def _norm_gmag_z(arr_list):
        return [[round((v - _gmag_lo) / _gmag_span, 4) if v is not None else None
                 for v in arr] for arr in arr_list]

    _phz_sub_z_gmag   = _norm_gmag_z(_phz_sub_z_gmag)
    _ero_reg_z_gmag   = _norm_gmag_z(_ero_reg_z_gmag)
    _ero_rebal_z_gmag = _norm_gmag_z(_ero_rebal_z_gmag)
    _gaia_z_gmag      = _norm_gmag_z(_gaia_z_gmag)
    _gaia_hilat_z_gmag = _norm_gmag_z(_gaia_hilat_z_gmag)
    _ero_desi_z_gmag  = _norm_gmag_z(_ero_desi_z_gmag)
    # Q1 base always at z=1.45 regardless of z-axis mode
    _q1_z_gmag = [[1.45] * len(arr) for arr in _q1_z_gmag]

    # ── Normalise Gaia feature z arrays to [0, 1] ────────────────────────────
    _overlay_feat_dicts = [_phz_sub_z_feat, _ero_reg_z_feat, _ero_rebal_z_feat,
                           _gaia_z_feat, _gaia_hilat_z_feat, _ero_desi_z_feat]
    _feat_norm: dict[str, list[float]] = {}
    _feat_color_bounds: dict[str, list[float]] = {}  # 1st/99th percentile for color display
    for fi, fn in enumerate(SELECTED_GAIA_PARAMETERS):
        all_vals: list[float] = []
        for zdict in _overlay_feat_dicts:
            for _arr in zdict[fn]:
                all_vals.extend(v for v in _arr if v is not None)
        if q1_features is not None:
            col = q1_features[:, fi]
            all_vals.extend(float(v) for v in col[np.isfinite(col)])
        if all_vals:
            f_lo, f_hi = min(all_vals), max(all_vals)
            # 1st/99th percentile for color mapping (clips outliers)
            _pct_arr = np.array(all_vals)
            _feat_color_bounds[fn] = [
                round(float(np.percentile(_pct_arr, 1)), 4),
                round(float(np.percentile(_pct_arr, 99)), 4),
            ]
        else:
            f_lo, f_hi = 0.0, 1.0
            _feat_color_bounds[fn] = [0.0, 1.0]
        _feat_norm[fn] = [round(f_lo, 4), round(f_hi, 4)]
        f_span = f_hi - f_lo if f_hi > f_lo else 1.0
        for zdict in _overlay_feat_dicts:
            zdict[fn] = [[round((v - f_lo) / f_span, 4) if v is not None else None
                          for v in arr] for arr in zdict[fn]]
        _q1_z_feat[fn] = [[1.45] * len(a) for a in _q1_z_feat[fn]]

    # Z-axis arrays (per-trace, in trace order within each group)
    _q1_z_dict: dict = {"g_mag": _q1_z_gmag, "model_pgal": _q1_z_model_pgal, "phz_pgal": _q1_z_phz_pgal}
    _q1_z_dict.update(_q1_z_feat)
    q1_base_z_json = json.dumps(_q1_z_dict)

    _phz_z_dict: dict = {"g_mag": _phz_sub_z_gmag, "phz_pgal": _phz_sub_z_phz_pgal, "model_pgal": _phz_sub_z_model_pgal}
    _phz_z_dict.update(_phz_sub_z_feat)
    phz_sub_z_json = json.dumps(_phz_z_dict)

    _ero_reg_z_dict: dict = {"g_mag": _ero_reg_z_gmag, "model_pgal": _ero_reg_z_model_pgal}
    _ero_reg_z_dict.update(_ero_reg_z_feat)
    ero_reg_z_json = json.dumps(_ero_reg_z_dict)

    _ero_rebal_z_dict: dict = {"g_mag": _ero_rebal_z_gmag, "model_pgal": _ero_rebal_z_model_pgal}
    _ero_rebal_z_dict.update(_ero_rebal_z_feat)
    ero_rebal_z_json = json.dumps(_ero_rebal_z_dict)

    _gaia_z_dict: dict = {"g_mag": _gaia_z_gmag, "model_pgal": _gaia_z_model_pgal}
    _gaia_z_dict.update(_gaia_z_feat)
    gaia_xgb_z_json = json.dumps(_gaia_z_dict)

    _gaia_hilat_z_dict: dict = {"g_mag": _gaia_hilat_z_gmag, "model_pgal": _gaia_hilat_z_model_pgal}
    _gaia_hilat_z_dict.update(_gaia_hilat_z_feat)
    gaia_hilat_z_json = json.dumps(_gaia_hilat_z_dict)

    _ero_desi_z_dict: dict = {"g_mag": _ero_desi_z_gmag, "model_pgal": _ero_desi_z_model_pgal}
    _ero_desi_z_dict.update(_ero_desi_z_feat)
    ero_desi_z_json = json.dumps(_ero_desi_z_dict)

    gmag_norm_json     = json.dumps([round(_gmag_lo, 2), round(_gmag_hi, 2)])
    feat_norm_json     = json.dumps(_feat_norm)
    # Percentile-based color bounds (1st/99th) for overlay coloring
    _feat_color_bounds["g_mag"] = [_gmag_color_lo, _gmag_color_hi]
    _feat_color_bounds["model_pgal"] = [0.0, 1.0]  # probabilities are already [0,1]
    feat_color_bounds_json = json.dumps(_feat_color_bounds)
    density_hist_json  = json.dumps(density_hist_data or {})

    # Q1 base point UMAP coordinates (stored separately from Plotly traces for
    # reliable density bin lookup — Scatter3d trace .x/.y may not be accessible)
    _q1_base_coords = []
    if noise_mask.any():
        _q1_base_coords.append({
            "x": [round(float(v), 4) for v in q1_emb[noise_mask, 0]],
            "y": [round(float(v), 4) for v in q1_emb[noise_mask, 1]],
        })
    for k in cluster_ids:
        m = q1_labels == k
        _q1_base_coords.append({
            "x": [round(float(v), 4) for v in q1_emb[m, 0]],
            "y": [round(float(v), 4) for v in q1_emb[m, 1]],
        })
    q1_base_coords_json = json.dumps(_q1_base_coords)


    # Z-axis Gaia feature options
    gaia_z_opts = "\n".join(
        f'        <option value="{fn}">{fn}</option>'
        for fn in SELECTED_GAIA_PARAMETERS
    )

    # Colour dropdown HTML with optgroups
    gaia_opts = "\n".join(
        f'        <option value="{fn}">{fn}</option>'
        for fn in SELECTED_GAIA_PARAMETERS
    )
    q1_euclid_optgroup = ""
    if feature_names:
        q1_eu = [fn for fn in feature_names if fn.startswith("euclid_q1::")]
        if q1_eu:
            inner = "\n".join(
                f'        <option value="{fn}">{fn.replace("euclid_q1::","")}</option>'
                for fn in q1_eu
            )
            q1_euclid_optgroup = f'      <optgroup label="— Euclid Q1 features —">\n{inner}\n      </optgroup>'
    ero_mapped_optgroup = ""
    if feature_names:
        ero_mapped = [fn for fn in feature_names if fn.startswith("euclid_ero::")]
        if ero_mapped:
            inner = "\n".join(
                f'        <option value="{fn}">{fn.replace("euclid_ero::", "")}</option>'
                for fn in ero_mapped
            )
            ero_mapped_optgroup = f'      <optgroup id="ero-colour-optgroup" label="Euclid ERO (Q1 base, NN-mapped)">\n{inner}\n      </optgroup>'

    _sky_datasets_js = sky_datasets_json or "{}"
    _sky_img_src = sky_img_src or ""
    _q1_fields_js = q1_fields_js or "[]"
    _ero_fields_js = ero_fields_js or "[]"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Sky + UMAP Explorer</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  .plotly-notifier {{ display: none !important; }}
  body {{
    background: #080808; color: #bbb;
    display: flex; flex-direction: column; height: 100vh; overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  }}
  #main-panels {{ display: flex; flex: 1; min-height: 0; }}
  #sky-section {{
    flex: 0 0 38%; display: flex; flex-direction: column; min-width: 0;
  }}
  #sky-viewport {{
    flex: 1; min-width: 0; position: relative; overflow: hidden;
    background: #080808; cursor: grab;
  }}
  #sky-viewport:active {{ cursor: grabbing; }}
  #sky-gl {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; }}
  #source-canvas {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }}
  #fov-info {{
    position: absolute; top: 4px; left: 5px; z-index: 5;
    color: #ccc; font-size: 9px; font-family: "SF Mono", Menlo, Consolas, monospace;
    white-space: nowrap; pointer-events: none;
    text-shadow: 0 0 4px #000, 0 0 2px #000;
  }}
  #minimap-wrap {{
    position: absolute; bottom: 10px; right: 10px; z-index: 20;
    width: 180px; height: 90px;
    border: 1px solid #333; border-radius: 3px; overflow: hidden;
    box-shadow: 0 2px 10px rgba(0,0,0,0.8); background: #000;
  }}
  #minimap {{ width: 100%; height: 100%; position: relative; }}
  #minimap img {{ width: 100%; height: 100%; object-fit: contain; display: block; opacity: 0.6; }}
  #minimap-canvas {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }}
  #umap-section {{ flex: 1; display: flex; min-width: 0; }}
  #plot-container {{ flex: 1; min-width: 0; overflow: hidden; background: #080808; position: relative; }}
  #plot-container .plotly-graph-div {{ width: 100% !important; height: 100% !important; }}
  #overlay-cbar-side {{
    position:absolute; left:20px; top:50%; transform:translateY(-50%);
    display:none; flex-direction:column; align-items:center;
    height:40%; pointer-events:none; z-index:10;
  }}
  #overlay-cbar-gradient {{
    width:14px; flex:1; border-radius:3px; border:1px solid #2a2a2a;
  }}
  #overlay-cbar-min, #overlay-cbar-max {{
    color:#999; font-size:9px; font-family:"SF Mono",Menlo,Consolas,monospace;
    padding:2px 0; text-shadow:0 0 4px #000, 0 0 2px #000;
  }}
  #cluster-legend {{
    position:absolute; left:8px; top:10px; z-index:10;
    background:rgba(12,12,12,0.85); border:1px solid #222; border-radius:4px;
    padding:6px 10px; pointer-events:auto; user-select:none;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
    font-size:10px; color:#999; line-height:1;
  }}
  #cluster-legend .cl-row {{ display:flex; gap:12px; }}
  #cluster-legend .cl-row + .cl-row {{ margin-top:8px; }}
  #cluster-legend .cl-item {{
    display:flex; align-items:center; gap:3px; cursor:pointer;
    opacity:1; transition: opacity 0.15s;
  }}
  #cluster-legend .cl-item.hidden {{ opacity:0.25; }}
  #cluster-legend .cl-dot {{
    width:7px; height:7px; border-radius:50%; flex-shrink:0;
  }}
  .panel-handle {{
    width: 5px; flex-shrink: 0; cursor: col-resize;
    background: #111; border-left: 1px solid #1a1a1a;
    display: flex; align-items: center; justify-content: center;
    user-select: none; transition: background 0.15s;
  }}
  .panel-handle:hover, .panel-handle.dragging {{ background: #1e2a3a; }}
  .panel-handle::after {{ content: '⋮'; color: #2a2a2a; font-size: 14px; pointer-events: none; }}
  #sidebar {{
    width: 270px; flex-shrink: 0;
    background: #0c0c0c; border-left: 1px solid #181818;
    display: flex; flex-direction: column; overflow: hidden;
  }}
  #sidebar-header {{ padding: 16px 16px 12px; border-bottom: 1px solid #181818; flex-shrink: 0; }}
  #sidebar-eyebrow {{ font-size: 13px; font-weight: 600; color: #888; line-height: 1.3; margin-bottom: 5px; }}
  #sidebar-hint {{ font-size: 11px; color: #444; margin-top: 4px; }}
  #tip-card {{
    margin: 10px 16px 0; padding: 9px 10px 9px 12px;
    background: #111; border: 1px solid #1e2e40; border-radius: 3px;
    font-size: 10.5px; line-height: 1.6; color: #4a7a9b;
    display: flex; gap: 8px; align-items: flex-start; flex-shrink: 0;
  }}
  #tip-card span {{ flex: 1; }}
  #tip-dismiss {{
    cursor: pointer; color: #2a4a5c; font-size: 13px; line-height: 1;
    flex-shrink: 0; padding: 0 2px; background: none; border: none; transition: color 0.15s;
  }}
  #tip-dismiss:hover {{ color: #4a9edd; }}
  #sidebar-body {{ padding: 14px 16px; overflow-y: auto; flex: 1; }}
  #pin-row {{ display: flex; justify-content: flex-end; margin-bottom: 10px; }}
  #pin-btn {{
    font-size: 9px; letter-spacing: 0.1em; text-transform: uppercase;
    padding: 3px 8px; cursor: pointer;
    border: 1px solid #222; border-radius: 2px;
    background: transparent; color: #444; transition: all 0.15s;
  }}
  #pin-btn:hover {{ border-color: #383838; color: #777; }}
  #pin-btn.pinned {{ background: #0e2035; color: #4a9edd; border-color: #1a3d5c; }}
  #cutout-img {{
    width: 100%; display: none; border: 1px solid #181818; border-radius: 2px; margin-bottom: 10px;
  }}
  #cutout-info {{
    font-size: 10.5px; line-height: 1.7; white-space: pre;
    font-family: "SF Mono","Fira Code",Menlo,Consolas,monospace; color: #555;
  }}
  #no-cutout {{ font-size: 11px; color: #2a2a2a; font-style: italic; }}
  #sheet-view {{
    display: none; width: 100%; margin-top: 16px; padding-top: 14px;
    border-top: 1px solid #181818;
  }}
  #sheet-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
  #sheet-title {{ font-size: 9px; letter-spacing: 0.13em; text-transform: uppercase; color: #444; }}
  #sheet-close {{
    font-size: 11px; padding: 1px 6px; cursor: pointer;
    border: 1px solid #222; border-radius: 2px;
    background: transparent; color: #444; transition: all 0.15s; flex-shrink: 0;
  }}
  #sheet-close:hover {{ color: #888; border-color: #444; }}
  #sheet-images .sheet-cluster-label {{
    font-size: 9px; letter-spacing: 0.1em; text-transform: uppercase;
    color: #444; margin-bottom: 4px; margin-top: 8px;
  }}
  #sheet-images img {{
    width: 100%; border: 1px solid #181818; margin-bottom: 4px;
    display: block; border-radius: 2px; cursor: pointer;
  }}
  #sheet-images img:hover {{ border-color: #555; }}
  .ctrl-bar {{
    padding: 7px 16px; border-bottom: 1px solid #181818;
    display: flex; align-items: center; gap: 8px; flex-shrink: 0;
  }}
  .ctrl-label {{ font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase; color: #666; white-space: nowrap; }}
  .ctrl-bar select {{
    flex: 1; background: #111; color: #ccc; border: 1px solid #222;
    border-radius: 3px; padding: 3px 6px; font-size: 11px; cursor: pointer;
  }}
  .ctrl-bar select:focus {{ outline: none; border-color: #333; }}
  .ctrl-bar select option {{ background: #111; color: #ccc; }}
  .ctrl-bar select optgroup {{ background: #111; color: #666; }}
  #stats-panel {{
    padding: 5px 10px; background: #0a0a0a; border-top: 1px solid #181818;
    font-size: 10px; color: #777; flex-shrink: 0; min-height: 26px;
    font-family: "SF Mono", Menlo, Consolas, monospace;
    display: flex; align-items: center; gap: 12px;
    overflow-x: auto; white-space: nowrap;
  }}
  #stats-panel b {{ color: #ccc; }}
  .stat-label {{ color: #666; }}
  .hint {{ color: #555; font-size: 11px; font-style: italic; }}
  #clear-sel-btn {{
    background: #1a1a1a; color: #aaa; border: 1px solid #333; border-radius: 3px;
    padding: 1px 7px; font-size: 10px; cursor: pointer; flex-shrink: 0;
  }}
  #clear-sel-btn:hover {{ background: #282828; color: #ddd; }}
</style>
</head>
<body>
<div id="main-panels">
  <div id="sky-section">
    <div id="sky-viewport">
      <canvas id="sky-gl"></canvas>
      <canvas id="source-canvas"></canvas>
      <span id="fov-info"></span>
      <div id="minimap-wrap">
        <div id="minimap">
          <img src="{_sky_img_src}">
          <canvas id="minimap-canvas"></canvas>
        </div>
      </div>
    </div>
    <div id="stats-panel">
      <span id="stats-default" class="hint"></span>
      <span id="stats-content" style="display:none"></span>
      <button id="clear-sel-btn" style="display:none" onclick="doClearSelection()">Clear</button>
    </div>
  </div>
  <div id="sky-handle" class="panel-handle"></div>
  <div id="umap-section">
    <div id="plot-container">
      {plot_div}
      <div id="overlay-cbar-side">
        <span id="overlay-cbar-max">1.00</span>
        <div id="overlay-cbar-gradient"></div>
        <span id="overlay-cbar-min">0.00</span>
      </div>
      <div id="cluster-legend">
        <div class="cl-row">
          <div class="cl-item" data-cluster="noise" onclick="toggleClusterLegend(this)">
            <span class="cl-dot" style="background:#2a2a2a"></span>noise</div>
          <div class="cl-item" data-cluster="0" onclick="toggleClusterLegend(this)">
            <span class="cl-dot" style="background:#1f77b4"></span>Cluster 0</div>
          <div class="cl-item" data-cluster="1" onclick="toggleClusterLegend(this)">
            <span class="cl-dot" style="background:#ff7f0e"></span>Cluster 1</div>
        </div>
        <div class="cl-row">
          <div class="cl-item" data-cluster="2" onclick="toggleClusterLegend(this)">
            <span class="cl-dot" style="background:#2ca02c"></span>Cluster 2</div>
          <div class="cl-item" data-cluster="3" onclick="toggleClusterLegend(this)">
            <span class="cl-dot" style="background:#F298B6"></span>Cluster 3</div>
          <div class="cl-item" data-cluster="4" onclick="toggleClusterLegend(this)">
            <span class="cl-dot" style="background:mediumpurple"></span>Cluster 4</div>
        </div>
      </div>
    </div>
    <div id="resize-handle" class="panel-handle"></div>
    <div id="sidebar">
      <div id="sidebar-header">
        <div id="sidebar-eyebrow">Sky + UMAP Explorer</div>
        <div id="sidebar-hint">hover a point · double-click to pin</div>
      </div>
    <div class="ctrl-bar">
      <span class="ctrl-label">Dataset</span>
      <select id="overlay-select">
        <option value="gaia_xgb">Gaia Random 50k</option>
        <option value="gaia_hilat">Gaia High-lat 50k</option>
        <option value="allphz">Q1</option>
        <option value="ero">ERO</option>
        <option value="ero_rebalanced">ERO Rebalanced</option>
        <option value="euclid_desi">Euclid x DESI</option>
      </select>
    </div>
    <div class="ctrl-bar">
      <span class="ctrl-label">Color</span>
      <select id="color-mode-select">
        <option value="model_pgal" selected>XGB P(gal)</option>
        <option value="clusters">Legend</option>
        <option value="g_mag">G magnitude</option>
        <option value="umap_density">Source Density</option>
        <optgroup label="— Gaia features —">
{gaia_opts}
        </optgroup>
{ero_mapped_optgroup}
      </select>
    </div>
    <div class="ctrl-bar">
      <span class="ctrl-label">Z axis</span>
      <select id="z-select">
        <option value="g_mag">G magnitude</option>
        <option value="model_pgal">XGB P(gal)</option>
        <option value="phz_pgal">PHZ P(gal)</option>
        <optgroup label="— Gaia features —">
{gaia_z_opts}
        </optgroup>
      </select>
    </div>
    <div id="tip-card">
      <span>Click a legend to remove it.<br>⌘+click a legend to isolate it.<br>Press Esc to restore everything.</span>
      <button id="tip-dismiss">&#x2715;</button>
    </div>
    <div id="sidebar-body">
      <div id="pin-row"><button id="pin-btn">Pin</button></div>
      <img id="cutout-img" src="">
      <div id="cutout-info"></div>
      <div id="no-cutout"></div>
      <div id="sheet-view">
        <div id="sheet-header">
          <span id="sheet-title">Contact sheets</span>
          <button id="sheet-close">&#x2715;</button>
        </div>
        <div id="sheet-images"></div>
      </div>
    </div>
  </div>
  </div>
</div>
<script>
var cutouts                = {b64_json};
var contact_sheets         = {contact_sheets_json};
var sheet_counts           = {sheet_counts_json};
var desi_contact_sheets    = {desi_contact_sheets_json};
var desi_sheet_counts      = {desi_sheet_counts_json};
var desi_sheet_extra       = {desi_sheet_extra_json};
var sheet_extra_pages      = {sheet_extra_json};
var ero_cutouts            = {ero_cutouts_json};
var ero_contact_sheets     = {ero_contact_sheets_json};
var ero_sheet_counts_map   = {ero_sheet_counts_json};
var ero_sheet_extra        = {ero_sheet_extra_json};
var ERO_GROUP_LABELS       = {ero_group_labels_json};
var ERO_TRACE_GROUPS       = {ero_trace_groups_json};
var ERO_GROUP_CONFIGS_JS   = {ero_group_configs_json};
var DESI_TRACE_START       = {desi_trace_start_json};
var ERO_REGULAR_START      = {ero_regular_start_json};
var ERO_REBAL_START        = {ero_rebal_start_json};
/* ERO overlay: legend dummy traces first, then per-group×cluster sub-traces */
var ERO_REG_N_DUMMY          = {ero_reg_n_dummy_json};
var ERO_REG_SUB_START        = {ero_reg_sub_start_json};
var ERO_REG_N_SUB            = {ero_reg_n_sub_json};
var ERO_REBAL_N_DUMMY        = {ero_rebal_n_dummy_json};
var ERO_REBAL_SUB_START      = {ero_rebal_sub_start_json};
var ERO_REBAL_N_SUB          = {ero_rebal_n_sub_json};
var GAIA_XGB_START         = {gaia_xgb_start_json};
var GAIA_XGB_N_LEGEND      = {gaia_xgb_n_legend_json};
var GAIA_XGB_SUB_START     = {gaia_xgb_sub_start_json};
var GAIA_XGB_N_SUB         = {gaia_xgb_n_sub_json};
var GAIA_XGB_CATS          = {gaia_xgb_cats_json};
var GAIA_HILAT_START       = {gaia_hilat_start_json};
var GAIA_HILAT_N_LEGEND    = {gaia_hilat_n_legend_json};
var GAIA_HILAT_SUB_START   = {gaia_hilat_sub_start_json};
var GAIA_HILAT_N_SUB       = {gaia_hilat_n_sub_json};
var GAIA_HILAT_CATS        = {gaia_hilat_cats_json};
var ERO_DESI_START         = {ero_desi_start_json};
var ERO_DESI_N_LEGEND      = {ero_desi_n_legend_json};
var ERO_DESI_SUB_START     = {ero_desi_sub_start_json};
var ERO_DESI_N_SUB         = {ero_desi_n_sub_json};
var ero_desi_contact_sheets = {ero_desi_contact_sheets_json};
var ero_desi_sheet_counts   = {ero_desi_sheet_counts_json};
var ero_desi_sheet_extra    = {ero_desi_sheet_extra_json};
var ERO_DESI_CLUSTER_COVERAGE = {ero_desi_cluster_coverage_json};
var ERO_DESI_TRACE_CAT        = {ero_desi_trace_cat_json};
var PHZ_COLORS                = {phz_colors_json};
var PHZ_NAMES                 = {phz_names_json};
var DESI_NAMES                = {desi_names_json};
var ERO_DESI_COLORS           = {ero_desi_colors_json};
/* Z-axis arrays (per-trace, in trace order within each group) */
var Q1_BASE_Z   = {q1_base_z_json};
var PHZ_SUB_Z   = {phz_sub_z_json};
var ERO_REG_Z   = {ero_reg_z_json};
var ERO_REBAL_Z = {ero_rebal_z_json};
var GAIA_XGB_Z  = {gaia_xgb_z_json};
var GAIA_HILAT_Z = {gaia_hilat_z_json};
var ERO_DESI_Z  = {ero_desi_z_json};
var GMAG_NORM = {gmag_norm_json};  /* [min, max] for G-mag tick labels */
var FEAT_NORM = {feat_norm_json};  /* {{feat_name: [min, max]}} for Gaia feature tick labels */
var FEAT_COLOR_BOUNDS = {feat_color_bounds_json};  /* 1st/99th percentile for color mapping */
// RdYlBu reversed: blue(low) → yellow(mid) → red(high)
var CSCALE_RdYlBu_r = [
  [0,'rgb(49,54,149)'],[0.1,'rgb(69,117,180)'],[0.2,'rgb(116,173,209)'],
  [0.3,'rgb(171,217,233)'],[0.4,'rgb(224,243,248)'],
  [0.5,'rgb(255,255,191)'],
  [0.6,'rgb(254,224,144)'],[0.7,'rgb(253,174,97)'],
  [0.8,'rgb(244,109,67)'],[0.9,'rgb(215,48,39)'],[1,'rgb(165,0,38)']
];
/* RdYlBu (non-reversed): 0=red(star) → 1=blue(galaxy), used for P(gal) */
var CSCALE_RdYlBu = [
  [0,'rgb(165,0,38)'],[0.1,'rgb(215,48,39)'],[0.2,'rgb(244,109,67)'],
  [0.3,'rgb(253,174,97)'],[0.4,'rgb(254,224,144)'],
  [0.5,'rgb(255,255,191)'],
  [0.6,'rgb(224,243,248)'],[0.7,'rgb(171,217,233)'],
  [0.8,'rgb(116,173,209)'],[0.9,'rgb(69,117,180)'],[1,'rgb(49,54,149)']
];

var cluster_sheets = {{}};
Object.keys(contact_sheets).forEach(function(v) {{
  Object.keys(contact_sheets[v]).forEach(function(k) {{
    if (!cluster_sheets[k]) cluster_sheets[k] = {{}};
    cluster_sheets[k][v] = contact_sheets[v][k];
  }});
}});
var desi_cluster_sheets = {{}};
Object.keys(desi_contact_sheets).forEach(function(v) {{
  Object.keys(desi_contact_sheets[v]).forEach(function(k) {{
    if (!desi_cluster_sheets[k]) desi_cluster_sheets[k] = {{}};
    desi_cluster_sheets[k][v] = desi_contact_sheets[v][k];
  }});
}});
var ero_desi_cluster_sheets = {{}};
Object.keys(ero_desi_contact_sheets).forEach(function(v) {{
  Object.keys(ero_desi_contact_sheets[v]).forEach(function(k) {{
    if (!ero_desi_cluster_sheets[k]) ero_desi_cluster_sheets[k] = {{}};
    ero_desi_cluster_sheets[k][v] = ero_desi_contact_sheets[v][k];
  }});
}});

/* Q1 feature names (Gaia + Q1 Euclid, index into Q1 base customdata) */
var featureNames   = {feature_names_json};
var featureRanges  = {feature_ranges_json};
var densityHistData = {density_hist_json};
var q1BaseCoords    = {q1_base_coords_json};

/* ERO feature names (index into ERO trace customdata rows) */
var eroFeatureNames  = {ero_feature_names_json};
var eroFeatureRanges = {ero_feature_ranges_json};

var CLUSTER_COLORS_JS = {cluster_colors_json};
var N_BASE_TRACES     = {n_base_traces};

/* (removed old COLORSCALE_RDBU — using CSCALE_RdYlBu_r everywhere) */
var COLORSCALE_BLUES = [
  [0,'rgb(247,251,255)'],[0.125,'rgb(222,235,247)'],
  [0.25,'rgb(198,219,239)'],[0.375,'rgb(158,202,225)'],
  [0.5,'rgb(107,174,214)'],[0.625,'rgb(66,146,198)'],
  [0.75,'rgb(33,113,181)'],[0.875,'rgb(8,81,156)'],
  [1,'rgb(8,48,107)']
];

/* ══════════════════════════════════════════════════════════════
   SKY MAP DATA + RENDERING
   ══════════════════════════════════════════════════════════════ */
var SKY_DATASETS = {_sky_datasets_js};
var Q1F = {_q1_fields_js};
var EROF = {_ero_fields_js};
var IW = {IMG_W}, IH = {IMG_H};
var PI = Math.PI, DEG = PI / 180;
var CRPIX_X = {IMG_CRPIX[0]}, CRPIX_Y = {IMG_CRPIX[1]};
var CDELT_X = {IMG_CDELT[0]}, CDELT_Y = {IMG_CDELT[1]};

/* Dataset-to-overlay mapping */
var DATASET_TO_OVERLAY = {{
  gaia_random: 'gaia_xgb', gaia_hilat: 'gaia_hilat',
  q1: 'allphz', euclid_desi: 'euclid_desi',
  ero: 'ero', ero_rebal: 'ero_rebalanced'
}};
var OVERLAY_TO_SKY = {{}};
Object.keys(DATASET_TO_OVERLAY).forEach(function(k){{ OVERLAY_TO_SKY[DATASET_TO_OVERLAY[k]] = k; }});

var _firstSkyKey = Object.keys(SKY_DATASETS)[0] || 'gaia_random';
var D = SKY_DATASETS[_firstSkyKey] || {{ra:[],dec:[],gl:[],gb:[],ipx:[],ipy:[],u1:[],u2:[],sid:[]}};
var N = D.ra ? D.ra.length : 0;

/* Sky state */
var selected = new Uint8Array(N);
var hasSelection = false;
var selCircle = null;
var curSkyColor = 'model_pgal';

/* ── Color mapping (RdYlBu_r) ── */
var _cstops = [
  [0,49,54,149],[0.1,69,117,180],[0.2,116,173,209],
  [0.3,171,217,233],[0.4,224,243,248],[0.5,255,255,191],
  [0.6,254,224,144],[0.7,253,174,97],[0.8,244,109,67],
  [0.9,215,48,39],[1,165,0,38]
];
function _interpStops(stops, v, mn, mx) {{
  var t = (v - mn) / (mx - mn); t = Math.max(0, Math.min(1, t));
  var i = 0;
  for (var j = 1; j < stops.length; j++) {{ if (stops[j][0] >= t) {{ i = j - 1; break; }} }}
  var s0 = stops[i], s1 = stops[i + 1] || s0;
  var f = (s1[0] === s0[0]) ? 0 : (t - s0[0]) / (s1[0] - s0[0]);
  var r = Math.round(s0[1] + f * (s1[1] - s0[1]));
  var g = Math.round(s0[2] + f * (s1[2] - s0[2]));
  var b = Math.round(s0[3] + f * (s1[3] - s0[3]));
  return 'rgb(' + r + ',' + g + ',' + b + ')';
}}
function valToRGB(v, mn, mx) {{ return _interpStops(_cstops, v, mn, mx); }}
/* Inverted RdYlBu: 0=red → 1=blue, for P(gal) */
var _cstops_inv = _cstops.slice().reverse().map(function(s,i,a){{return [1-s[0],s[1],s[2],s[3]];}});
function valToRGB_inv(v, mn, mx) {{ return _interpStops(_cstops_inv, v, mn, mx); }}
/* Blues stops for density coloring (matplotlib Blues) */
var _bluesStops = [
  [0,247,251,255],[0.125,222,235,247],[0.25,198,219,239],
  [0.375,158,202,225],[0.5,107,174,214],[0.625,66,146,198],
  [0.75,33,113,181],[0.875,8,81,156],[1,8,48,107]
];
function densityToRGB(v, mn, mx) {{ return _interpStops(_bluesStops, v, mn, mx); }}
function getSkyColorBounds(feat) {{
  /* Use 1st/99th percentile bounds if available (clips outliers) */
  if (FEAT_COLOR_BOUNDS && FEAT_COLOR_BOUNDS[feat]) return FEAT_COLOR_BOUNDS[feat];
  if (!D[feat]) return [0, 1];
  var arr = D[feat], mn = Infinity, mx = -Infinity;
  for (var i = 0; i < N; i++) if (arr[i] !== null) {{
    if (arr[i] < mn) mn = arr[i]; if (arr[i] > mx) mx = arr[i];
  }}
  return [mn, mx];
}}

/* ── Angular distance ── */
function angDist(ra1, dec1, ra2, dec2) {{
  var dd = (dec2 - dec1) * DEG, dr = (ra2 - ra1) * DEG;
  var a = Math.sin(dd / 2) * Math.sin(dd / 2) +
          Math.cos(dec1 * DEG) * Math.cos(dec2 * DEG) * Math.sin(dr / 2) * Math.sin(dr / 2);
  return 2 * Math.asin(Math.sqrt(Math.min(1, a))) / DEG;
}}
function angDistGal(l1, b1, l2, b2) {{
  var db = (b2 - b1) * DEG, dl = (l2 - l1) * DEG;
  var a = Math.sin(db / 2) * Math.sin(db / 2) +
          Math.cos(b1 * DEG) * Math.cos(b2 * DEG) * Math.sin(dl / 2) * Math.sin(dl / 2);
  return 2 * Math.asin(Math.sqrt(Math.min(1, a))) / DEG;
}}

/* ── Coordinate transforms ── */
function galToEq(l, b) {{
  var lr = l * DEG, br = b * DEG;
  var ra_ngp = 192.85948 * DEG, dec_ngp = 27.12825 * DEG, l_ncp = 122.93192 * DEG;
  var sinDec = Math.sin(br)*Math.sin(dec_ngp) + Math.cos(br)*Math.cos(dec_ngp)*Math.sin(lr - l_ncp);
  var dec = Math.asin(Math.max(-1, Math.min(1, sinDec)));
  var cosRA = (Math.cos(br)*Math.cos(lr - l_ncp)) / Math.cos(dec);
  var sinRA = (Math.sin(br)*Math.cos(dec_ngp) - Math.cos(br)*Math.sin(dec_ngp)*Math.sin(lr - l_ncp)) / Math.cos(dec);
  var ra = ra_ngp + Math.atan2(cosRA, sinRA);
  ra = ra / DEG; dec = dec / DEG;
  if (ra < 0) ra += 360; if (ra >= 360) ra -= 360;
  return {{ ra: ra, dec: dec }};
}}
function eqToGal(ra, dec) {{
  var ra_ngp = 192.85948 * DEG, dec_ngp = 27.12825 * DEG, l_ncp = 122.93192 * DEG;
  var rar = ra * DEG, decr = dec * DEG;
  var sinB = Math.sin(decr)*Math.sin(dec_ngp) + Math.cos(decr)*Math.cos(dec_ngp)*Math.cos(rar - ra_ngp);
  var b = Math.asin(Math.max(-1, Math.min(1, sinB)));
  var num = Math.cos(decr)*Math.sin(rar - ra_ngp);
  var den = Math.sin(decr)*Math.cos(dec_ngp) - Math.cos(decr)*Math.sin(dec_ngp)*Math.cos(rar - ra_ngp);
  var l = l_ncp - Math.atan2(num, den);
  l = l / DEG; b = b / DEG;
  if (l < 0) l += 360; if (l >= 360) l -= 360;
  return {{ l: l, b: b }};
}}

/* ══ WebGL sky rendering ══ */
var viewport = document.getElementById('sky-viewport');
var glCanvas = document.getElementById('sky-gl');
var gl = glCanvas ? glCanvas.getContext('webgl', {{antialias:false, depth:false}}) : null;
var centerL = 0, centerB = 0, fov = 2.0, aspect = 1;

var VERT_SRC = 'attribute vec2 a_pos; varying vec2 v_uv; void main(){{ v_uv=a_pos; gl_Position=vec4(a_pos,0,1); }}';
var FRAG_SRC = [
  'precision highp float;',
  'uniform sampler2D u_tex;',
  'uniform float u_cl, u_cb, u_fov, u_asp;',
  'varying vec2 v_uv;',
  'const float PI=3.14159265359;',
  'const float S2=1.41421356237;',
  'void main(){{',
  '  float ht=2.0*tan(u_fov*0.25);',
  '  float gx=v_uv.x*ht*u_asp, gy=v_uv.y*ht;',
  '  float rho=sqrt(gx*gx+gy*gy);',
  '  float sb=sin(u_cb), cb=cos(u_cb), l, b;',
  '  if(rho<1e-6){{ l=u_cl; b=u_cb; }}',
  '  else{{',
  '    float c=2.0*atan(rho*0.5), sc=sin(c), cc=cos(c);',
  '    b=asin(clamp(cc*sb+gy*sc*cb/rho,-1.0,1.0));',
  '    l=u_cl+atan(gx*sc, rho*cb*cc-gy*sb*sc);',
  '  }}',
  '  l=l-floor(l/(2.0*PI))*2.0*PI;',
  '  float lc=l>PI?l-2.0*PI:l;',
  '  float lon=-lc, cosB=cos(b);',
  '  float z=sqrt(1.0+cosB*cos(lon*0.5));',
  '  float ax=2.0*cosB*sin(lon*0.5)/z, ay=sin(b)/z;',
  '  float X=ax*S2, Y=ay*S2;',
  '  if(X*X/16.0+Y*Y/4.0>1.0){{ gl_FragColor=vec4(0.031,0.031,0.031,1); return; }}',
  '  float px={IMG_CRPIX[0]}+(ax*180.0/PI)/({IMG_CDELT[0]});',
  '  float py={IMG_CRPIX[1]}+(ay*180.0/PI)/({IMG_CDELT[1]});',
  '  float u=px/{float(IMG_W)}, v=1.0-py/{float(IMG_H)};',
  '  if(u<0.0||u>1.0||v<0.0||v>1.0){{ gl_FragColor=vec4(0.031,0.031,0.031,1); return; }}',
  '  gl_FragColor=texture2D(u_tex,vec2(u,v));',
  '}}'
].join('\\n');

var skyProg = null, skyTex = null, skyTexReady = false;
if (gl) {{
  function _compileShader(src, type) {{
    var s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s;
  }}
  skyProg = gl.createProgram();
  gl.attachShader(skyProg, _compileShader(VERT_SRC, gl.VERTEX_SHADER));
  gl.attachShader(skyProg, _compileShader(FRAG_SRC, gl.FRAGMENT_SHADER));
  gl.linkProgram(skyProg); gl.useProgram(skyProg);
  var buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
  var aPos = gl.getAttribLocation(skyProg, 'a_pos');
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
  skyTex = gl.createTexture();
  var skyImg = new Image();
  skyImg.onload = function() {{
    gl.bindTexture(gl.TEXTURE_2D, skyTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, skyImg);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    skyTexReady = true;
    renderAll();
  }};
  skyImg.src = '{_sky_img_src}';
}}

function renderSky() {{
  if (!gl || !skyTexReady) return;
  gl.viewport(0, 0, glCanvas.width, glCanvas.height);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(skyProg);
  gl.uniform1f(gl.getUniformLocation(skyProg, 'u_cl'), centerL);
  gl.uniform1f(gl.getUniformLocation(skyProg, 'u_cb'), centerB);
  gl.uniform1f(gl.getUniformLocation(skyProg, 'u_fov'), fov);
  gl.uniform1f(gl.getUniformLocation(skyProg, 'u_asp'), aspect);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, skyTex);
  gl.uniform1i(gl.getUniformLocation(skyProg, 'u_tex'), 0);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}}

/* ── Stereographic projection helpers ── */
function gnomFwd(lr, br) {{
  var dl = lr - centerL;
  if (dl > PI) dl -= 2*PI;
  if (dl < -PI) dl += 2*PI;
  var sb = Math.sin(br), cb = Math.cos(br);
  var scb = Math.sin(centerB), ccb = Math.cos(centerB);
  var cdl = Math.cos(dl);
  var cosc = scb * sb + ccb * cb * cdl;
  if (cosc <= -0.9) return null;
  var k = 2.0 / (1.0 + cosc);
  return {{ gx: k * cb * Math.sin(dl), gy: k * (ccb * sb - scb * cb * cdl) }};
}}
function gnomToScreen(gx, gy) {{
  if (!viewport) return null;
  var rect = viewport.getBoundingClientRect();
  var w = rect.width, h = rect.height;
  var ht = 2 * Math.tan(fov / 4);
  var sx = (gx / (ht * aspect) + 1) * 0.5 * w;
  var sy = (1 - gy / ht) * 0.5 * h;
  return {{ x: sx, y: sy }};
}}
function screenToGal(cx, cy) {{
  if (!viewport) return null;
  var rect = viewport.getBoundingClientRect();
  var w = rect.width, h = rect.height;
  var nx = cx / w * 2 - 1;
  var ny = 1 - cy / h * 2;
  var ht = 2 * Math.tan(fov / 4);
  var gx = nx * ht * aspect;
  var gy = ny * ht;
  var rho = Math.sqrt(gx * gx + gy * gy);
  if (rho < 1e-8) return {{ l: centerL / DEG, b: centerB / DEG }};
  var c = 2 * Math.atan(rho / 2);
  var sc = Math.sin(c), cc = Math.cos(c);
  var sb0 = Math.sin(centerB), cb0 = Math.cos(centerB);
  var b = Math.asin(Math.max(-1, Math.min(1, cc * sb0 + gy * sc * cb0 / rho)));
  var l = centerL + Math.atan2(gx * sc, rho * cb0 * cc - gy * sb0 * sc);
  l = ((l % (2 * PI)) + 2 * PI) % (2 * PI);
  return {{ l: l / DEG, b: b / DEG }};
}}

/* ── Source canvas drawing ── */
var srcCanvas = document.getElementById('source-canvas');
var srcCtx = srcCanvas ? srcCanvas.getContext('2d') : null;

/* ── Sky crosshair for hovered/pinned 3D point ── */
var _crosshairRA = null, _crosshairDec = null, _crosshairPinned = false;

function _drawCrosshair() {{
  if (_crosshairRA === null || _crosshairDec === null) return;
  if (!srcCtx) return;
  var gb = eqToGal(_crosshairRA, _crosshairDec);
  if (!gb) return;
  var gp = gnomFwd(gb.l * DEG, gb.b * DEG);
  if (!gp) return;
  var sp = gnomToScreen(gp.gx, gp.gy);
  if (!sp) return;
  var x = sp.x, y = sp.y;
  var r = 3.5, tick = 3.5, gap = 1.5, cap = 1.5, lw = 1;
  srcCtx.save();
  srcCtx.strokeStyle = '#F472B6';
  srcCtx.lineWidth = lw;
  /* circle */
  srcCtx.beginPath();
  srcCtx.arc(x, y, r, 0, 2 * Math.PI);
  srcCtx.stroke();
  /* center dot */
  srcCtx.fillStyle = '#F472B6';
  srcCtx.beginPath();
  srcCtx.arc(x, y, 1.2, 0, 2 * Math.PI);
  srcCtx.fill();
  /* four tick lines with perpendicular caps */
  var o1 = r + gap, o2 = o1 + tick;
  srcCtx.beginPath();
  /* right */
  srcCtx.moveTo(x + o1, y); srcCtx.lineTo(x + o2, y);
  srcCtx.moveTo(x + o2, y - cap); srcCtx.lineTo(x + o2, y + cap);
  /* left */
  srcCtx.moveTo(x - o1, y); srcCtx.lineTo(x - o2, y);
  srcCtx.moveTo(x - o2, y - cap); srcCtx.lineTo(x - o2, y + cap);
  /* top */
  srcCtx.moveTo(x, y - o1); srcCtx.lineTo(x, y - o2);
  srcCtx.moveTo(x - cap, y - o2); srcCtx.lineTo(x + cap, y - o2);
  /* bottom */
  srcCtx.moveTo(x, y + o1); srcCtx.lineTo(x, y + o2);
  srcCtx.moveTo(x - cap, y + o2); srcCtx.lineTo(x + cap, y + o2);
  srcCtx.stroke();
  srcCtx.restore();
}}

function showSkyCrosshair(ra, dec) {{
  if (_crosshairPinned) return;  /* don't move pinned crosshair */
  _crosshairRA = ra; _crosshairDec = dec;
  drawSources();  /* redraws sources + crosshair on top */
}}

function clearSkyCrosshair() {{
  if (_crosshairPinned) return;  /* don't clear pinned crosshair */
  _crosshairRA = null; _crosshairDec = null;
  drawSources();
}}

function unpinCrosshair() {{
  _crosshairPinned = false;
  _crosshairRA = null; _crosshairDec = null;
  drawSources();
}}

function resizeCanvases() {{
  if (!viewport) return;
  var rect = viewport.getBoundingClientRect();
  var w = rect.width, h = rect.height;
  var dpr = window.devicePixelRatio || 1;
  if (glCanvas) {{
    glCanvas.width = w * dpr; glCanvas.height = h * dpr;
    glCanvas.style.width = w + 'px'; glCanvas.style.height = h + 'px';
  }}
  if (srcCanvas) {{
    srcCanvas.width = w * dpr; srcCanvas.height = h * dpr;
    srcCanvas.style.width = w + 'px'; srcCanvas.style.height = h + 'px';
    srcCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }}
  aspect = w / h;
}}

/* Map 3D color mode values to sky data keys */
var SKY_COLOR_ALIAS = {{
  model_pgal: 'gal_prob',
  g_mag: 'phot_g_mean_mag'
}};
function skyColorKey(mode) {{
  return SKY_COLOR_ALIAS[mode] || mode;
}}

/* Check whether sky point i should be visible given the current
   3D legend isolation / hidden state.  Returns false if hidden. */
function _skyPointVisible(i) {{
  if (!D.cat) return true;  /* no category data → show all */
  var cat = D.cat[i];
  var cl  = (D.cluster && D.cluster[i] !== undefined) ? String(D.cluster[i]) : null;

  /* Cluster-level filtering (applies to all overlay types) */
  if (isolatedCluster !== null) {{
    if (cl !== String(isolatedCluster)) return false;
  }} else if (cl !== null && hiddenClusters[cl]) {{
    return false;
  }}

  /* Category-level filtering — depends on active overlay type */
  var ov = activeOverlay || 'allphz';

  if (ov === 'allphz' || ov === 'desi') {{
    /* PHZ category: cat is phz value string e.g. "1", "-1" */
    if (isolatedPhz !== null) return (cat === isolatedPhz);
    if (hiddenPhz[cat]) return false;

  }} else if (ov === 'gaia_xgb') {{
    /* Gaia XGB: cat is "star"/"gal", map to trace index for hidden/isolated check */
    if (isolatedGaiaXgb !== null) {{
      var _isoIdx = isolatedGaiaXgb - GAIA_XGB_START;
      var _isoCat = (_isoIdx >= 0 && _isoIdx < GAIA_XGB_CATS.length) ? GAIA_XGB_CATS[_isoIdx] : null;
      return (cat === _isoCat);
    }}
    var _catLeg = GAIA_XGB_START + GAIA_XGB_CATS.indexOf(cat);
    if (hiddenGaiaXgb[_catLeg]) return false;

  }} else if (ov === 'gaia_hilat') {{
    if (isolatedGaiaHilat !== null) {{
      var _isoIdx = isolatedGaiaHilat - GAIA_HILAT_START;
      var _isoCat = (_isoIdx >= 0 && _isoIdx < GAIA_HILAT_CATS.length) ? GAIA_HILAT_CATS[_isoIdx] : null;
      return (cat === _isoCat);
    }}
    var _catLeg = GAIA_HILAT_START + GAIA_HILAT_CATS.indexOf(cat);
    if (hiddenGaiaHilat[_catLeg]) return false;

  }} else if (ov === 'ero' || ov === 'ero_rebalanced') {{
    /* ERO: cat is group name string e.g. "agree_star_star" */
    if (isolatedEROGroup !== null) return (cat === isolatedEROGroup);
    if (hiddenEROGroups[cat]) return false;

  }} else if (ov === 'ero_desi') {{
    /* ERO-DESI standalone: cat is "ed_2"/"ed_3", map to trace index */
    var _edCatNum = parseInt(cat.replace('ed_', ''));
    if (isolatedEroDesi !== null) {{
      var _isoIdx = isolatedEroDesi - ERO_DESI_START;
      var _isoCatVal = (_isoIdx >= 0 && _isoIdx < ERO_DESI_TRACE_CAT.length) ? ERO_DESI_TRACE_CAT[_isoIdx] : null;
      return (_edCatNum === _isoCatVal);
    }}
    var _catLeg = ERO_DESI_START + ERO_DESI_TRACE_CAT.indexOf(_edCatNum);
    if (hiddenEroDesi[_catLeg]) return false;

  }} else if (ov === 'euclid_desi') {{
    /* Combined: Q1-DESI cats are phz strings, ERO-DESI cats are "ed_2"/"ed_3" */
    if (cat.indexOf('ed_') === 0) {{
      var _edCatNum = parseInt(cat.replace('ed_', ''));
      if (isolatedEroDesi !== null) {{
        var _isoIdx = isolatedEroDesi - ERO_DESI_START;
        var _isoCatVal = (_isoIdx >= 0 && _isoIdx < ERO_DESI_TRACE_CAT.length) ? ERO_DESI_TRACE_CAT[_isoIdx] : null;
        return (_edCatNum === _isoCatVal);
      }}
      var _catLeg = ERO_DESI_START + ERO_DESI_TRACE_CAT.indexOf(_edCatNum);
      if (hiddenEroDesi[_catLeg]) return false;
    }} else {{
      if (isolatedPhz !== null) return (cat === isolatedPhz);
      if (hiddenPhz[cat]) return false;
    }}
  }}
  return true;
}}

function drawSources() {{
  if (!srcCtx || !viewport) return;
  var rect = viewport.getBoundingClientRect();
  var w = rect.width, h = rect.height;
  srcCtx.clearRect(0, 0, w, h);

  var ht = 2 * Math.tan(fov / 4);
  var htAsp = ht * aspect;
  var PS = Math.max(3, Math.round(Math.min(w, h) / 180));
  var PSs = Math.max(2, PS - 2);
  var hs = Math.floor(PSs / 2);

  var sb0 = Math.sin(centerB), cb0 = Math.cos(centerB);
  var colorMode = skyColorKey(curSkyColor);
  var useLegend = (curSkyColor === 'clusters');
  var useDensity = (curSkyColor === 'umap_density');
  var mn = 0, mx = 1;
  if (!useLegend && !useDensity && D[colorMode]) {{
    var bnds = getSkyColorBounds(colorMode);
    mn = bnds[0]; mx = bnds[1];
  }}
  /* Pre-compute density values for sky dots */
  var _skyDenVmax = 0;
  if (useDensity) {{
    _densityCache = null;
    var _dl = _getDensityLookup();
    _skyDenVmax = _dl ? _dl.vmax : 1;
    if (_skyDenVmax === 0) _skyDenVmax = 1;
  }}

  function toScreen(i) {{
    var lr = D.gl[i] * DEG, br = D.gb[i] * DEG;
    var dl = lr - centerL;
    if (dl > PI) dl -= 2*PI;
    if (dl < -PI) dl += 2*PI;
    var sb = Math.sin(br), cbb = Math.cos(br);
    var cdl = Math.cos(dl);
    var cos_c = sb0*sb + cb0*cbb*cdl;
    if (cos_c <= -0.9) return null;
    var k = 2.0 / (1.0 + cos_c);
    var gx = k * cbb * Math.sin(dl);
    var gy = k * (cb0*sb - sb0*cbb*cdl);
    var sx = (gx / htAsp + 1) * 0.5 * w;
    var sy = (1 - gy / ht) * 0.5 * h;
    if (sx < -20 || sx > w+20 || sy < -20 || sy > h+20) return null;
    return {{x: sx, y: sy}};
  }}

  function getColor(i) {{
    if (useLegend && D.legend_color) return D.legend_color[i] || '#888';
    if (useDensity && D.u1 && D.u2) return densityToRGB(_densityAt(D.u1[i], D.u2[i]), 0, _skyDenVmax);
    if (!useLegend && !useDensity && D[colorMode] && D[colorMode][i] !== null) {{
      if (colorMode === 'model_pgal') return valToRGB_inv(D[colorMode][i], mn, mx);
      return valToRGB(D[colorMode][i], mn, mx);
    }}
    return '#888';
  }}

  if (hasSelection) {{
    srcCtx.globalAlpha = 0.2;
    for (var i = 0; i < N; i++) {{
      if (selected[i] || D.gl[i] === null || !_skyPointVisible(i)) continue;
      srcCtx.fillStyle = getColor(i);
      var s = toScreen(i);
      if (!s) continue;
      srcCtx.fillRect(s.x-hs, s.y-hs, PSs, PSs);
    }}
    srcCtx.globalAlpha = 0.9;
    for (var i = 0; i < N; i++) {{
      if (!selected[i] || D.gl[i] === null || !_skyPointVisible(i)) continue;
      srcCtx.fillStyle = getColor(i);
      var s = toScreen(i);
      if (!s) continue;
      srcCtx.fillRect(s.x-hs, s.y-hs, PSs, PSs);
    }}
  }} else {{
    srcCtx.globalAlpha = 0.75;
    for (var i = 0; i < N; i++) {{
      if (D.gl[i] === null || !_skyPointVisible(i)) continue;
      srcCtx.fillStyle = getColor(i);
      var s = toScreen(i);
      if (!s) continue;
      srcCtx.fillRect(s.x-hs, s.y-hs, PSs, PSs);
    }}
  }}
  srcCtx.globalAlpha = 1;
  drawFieldsGnom(Q1F, '#FF4444', 1.5);
  drawFieldsGnom(EROF, '#ffffff', 1);
  if (selCircle && selCircle.r > 0) drawSelCircle();
  if (_crosshairRA !== null) _drawCrosshair();
}}

function drawFieldsGnom(fields, color, lw) {{
  if (!srcCtx) return;
  var w = viewport.clientWidth, h = viewport.clientHeight;
  var hfov = fov * 0.5, tanH = Math.tan(hfov);
  srcCtx.strokeStyle = color; srcCtx.lineWidth = lw; srcCtx.globalAlpha = 0.5;
  fields.forEach(function(f) {{
    var g = eqToGal(f.ra, f.dec);
    var nPts = 60, pts = [];
    for (var i = 0; i <= nPts; i++) {{
      var th = 2 * PI * i / nPts;
      var dLat = f.radius * Math.sin(th);
      var dLon = f.radius * Math.cos(th) / Math.cos(g.b * DEG);
      var fl = (g.l + dLon), fb = g.b + dLat;
      var p = gnomFwd(fl * DEG, fb * DEG);
      if (p) pts.push(gnomToScreen(p.gx, p.gy));
      else pts.push(null);
    }}
    srcCtx.beginPath();
    var started = false;
    pts.forEach(function(p) {{
      if (!p) {{ started = false; return; }}
      if (!started) {{ srcCtx.moveTo(p.x, p.y); started = true; }}
      else srcCtx.lineTo(p.x, p.y);
    }});
    srcCtx.stroke();
    /* label */
    var lp = gnomFwd(g.l * DEG, (g.b + f.radius + 0.5) * DEG);
    if (lp) {{
      var sp = gnomToScreen(lp.gx, lp.gy);
      if (sp) {{
        srcCtx.fillStyle = color; srcCtx.font = '9px sans-serif';
        srcCtx.textAlign = 'center'; srcCtx.globalAlpha = 0.6;
        srcCtx.fillText(f.name, sp.x, sp.y);
      }}
    }}
  }});
  srcCtx.globalAlpha = 1;
}}

function drawSelCircle() {{
  if (!selCircle || !srcCtx) return;
  var w = viewport.clientWidth, h = viewport.clientHeight;
  var nPts = 80, pts = [];
  for (var i = 0; i <= nPts; i++) {{
    var th = 2 * PI * i / nPts;
    var dLat = selCircle.r * Math.sin(th);
    var dLon = selCircle.r * Math.cos(th) / Math.max(0.01, Math.cos(selCircle.b * DEG));
    var fl = selCircle.l + dLon, fb = selCircle.b + dLat;
    var p = gnomFwd(fl * DEG, fb * DEG);
    if (p) pts.push(gnomToScreen(p.gx, p.gy));
    else pts.push(null);
  }}
  srcCtx.strokeStyle = '#FF4081'; srcCtx.lineWidth = 1.5; srcCtx.globalAlpha = 0.8;
  srcCtx.beginPath();
  var started = false;
  pts.forEach(function(p) {{
    if (!p) {{ started = false; return; }}
    if (!started) {{ srcCtx.moveTo(p.x, p.y); started = true; }}
    else srcCtx.lineTo(p.x, p.y);
  }});
  srcCtx.stroke();
  srcCtx.globalAlpha = 1;
}}

function renderAll() {{
  renderSky();
  drawSources();
  updateMinimap();
  updateFovInfo();
}}

function updateFovInfo() {{
  var el = document.getElementById('fov-info');
  if (el) el.textContent = 'FOV: ' + (fov / DEG).toFixed(1) + '\u00b0  l=' + ((centerL/DEG+360)%360).toFixed(1) + '\u00b0 b=' + (centerB / DEG).toFixed(1) + '\u00b0';
}}

/* ── Minimap ── */
function jsHammerAitoff(lDeg, bDeg) {{
  var l = (lDeg > 180 ? lDeg - 360 : lDeg) * DEG;
  var b = bDeg * DEG;
  var z = Math.sqrt(1 + Math.cos(b) * Math.cos(l / 2));
  return {{ x: -2 * Math.cos(b) * Math.sin(l / 2) / z * (180 / PI),
            y: Math.sin(b) / z * (180 / PI) }};
}}
function lbToImgPx(lDeg, bDeg) {{
  var a = jsHammerAitoff(lDeg, bDeg);
  return {{ px: CRPIX_X + a.x / CDELT_X, py: CRPIX_Y + a.y / CDELT_Y }};
}}
function updateMinimap() {{
  var mmCanvas = document.getElementById('minimap-canvas');
  if (!mmCanvas) return;
  var cw = mmCanvas.clientWidth, ch = mmCanvas.clientHeight;
  mmCanvas.width = cw; mmCanvas.height = ch;
  var ctx = mmCanvas.getContext('2d');
  ctx.clearRect(0, 0, cw, ch);
  var scaleX = cw / IW, scaleY = ch / IH;
  /* FOV polygon */
  var nPts = 40, pts = [];
  for (var i = 0; i < nPts; i++) {{
    var th = 2 * PI * i / nPts;
    var ht = 2 * Math.tan(fov / 4);
    var gx = Math.cos(th) * ht * aspect;
    var gy = Math.sin(th) * ht;
    var rho = Math.sqrt(gx * gx + gy * gy);
    var c = 2 * Math.atan(rho * 0.5);
    var sc = Math.sin(c), cc = Math.cos(c);
    var b = Math.asin(cc * Math.sin(centerB) + gy * sc * Math.cos(centerB) / rho);
    var l = centerL + Math.atan2(gx * sc, rho * Math.cos(centerB) * cc - gy * Math.sin(centerB) * sc);
    var lDeg = l / DEG; if (lDeg < 0) lDeg += 360;
    var ip = lbToImgPx(lDeg, b / DEG);
    pts.push({{ x: ip.px * scaleX, y: ip.py * scaleY }});
  }}
  ctx.fillStyle = 'rgba(200,200,255,0.12)';
  ctx.beginPath();
  pts.forEach(function(p, i) {{ i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y); }});
  ctx.closePath(); ctx.fill();
  /* center crosshair */
  var cp = lbToImgPx(centerL / DEG < 0 ? centerL / DEG + 360 : centerL / DEG, centerB / DEG);
  var cx = cp.px * scaleX, cy = cp.py * scaleY;
  ctx.strokeStyle = '#f44'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(cx - 4, cy); ctx.lineTo(cx + 4, cy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx, cy - 4); ctx.lineTo(cx, cy + 4); ctx.stroke();
}}

/* ── Sky mouse interaction ── */
var _dragging = false, _selDrag = null, _moveStart = null;
if (viewport) {{
  viewport.style.pointerEvents = 'auto';
  srcCanvas.style.pointerEvents = 'auto';
  var _lastMouse = null;
  srcCanvas.addEventListener('mousedown', function(e) {{
    e.preventDefault();
    var rect = viewport.getBoundingClientRect();
    var cx = e.clientX - rect.left, cy = e.clientY - rect.top;
    if (e.metaKey || e.ctrlKey) {{
      /* Cmd+drag: create selection */
      var g = screenToGal(cx, cy);
      if (g) _selDrag = {{ l: g.l, b: g.b, r: 0, startL: g.l, startB: g.b }};
      return;
    }}
    if (e.shiftKey && selCircle && selCircle.r > 0) {{
      /* Shift+drag: move selection */
      var g2 = screenToGal(cx, cy);
      if (g2) _moveStart = {{ l: g2.l, b: g2.b, origL: selCircle.l, origB: selCircle.b }};
      return;
    }}
    _dragging = true;
    _lastMouse = {{ x: e.clientX, y: e.clientY }};
  }});
  window.addEventListener('mousemove', function(e) {{
    if (_selDrag) {{
      var rect = viewport.getBoundingClientRect();
      var g = screenToGal(e.clientX - rect.left, e.clientY - rect.top);
      if (g) {{
        _selDrag.r = angDistGal(_selDrag.startL, _selDrag.startB, g.l, g.b);
        selCircle = {{ l: _selDrag.startL, b: _selDrag.startB, r: _selDrag.r }};
        drawSources();
      }}
      return;
    }}
    if (_moveStart) {{
      var rect = viewport.getBoundingClientRect();
      var g = screenToGal(e.clientX - rect.left, e.clientY - rect.top);
      if (g && selCircle) {{
        selCircle.l = _moveStart.origL + (g.l - _moveStart.l);
        selCircle.b = _moveStart.origB + (g.b - _moveStart.b);
        drawSources();
      }}
      return;
    }}
    if (!_dragging || !_lastMouse) return;
    var dx = e.clientX - _lastMouse.x, dy = e.clientY - _lastMouse.y;
    _lastMouse = {{ x: e.clientX, y: e.clientY }};
    var w = viewport.clientWidth, h = viewport.clientHeight;
    var ht = 2 * Math.tan(fov / 4);
    var dl = -dx / w * 2 * ht * aspect;
    var db = dy / h * 2 * ht;
    centerL += dl; centerB += db;
    centerB = Math.max(-PI * 0.499, Math.min(PI * 0.499, centerB));
    renderAll();
  }});
  window.addEventListener('mouseup', function(e) {{
    if (_selDrag) {{
      if (_selDrag.r > 0.5) {{
        updateSelection();
      }} else {{
        selCircle = null; hasSelection = false;
        selected = new Uint8Array(N);
        clearStats();
      }}
      _selDrag = null;
      drawSources();
      return;
    }}
    if (_moveStart) {{
      _moveStart = null;
      if (selCircle && selCircle.r > 0.5) updateSelection();
      drawSources();
      return;
    }}
    _dragging = false; _lastMouse = null;
  }});
  srcCanvas.addEventListener('wheel', function(e) {{
    e.preventDefault();
    var factor = e.deltaY > 0 ? 1.12 : 1 / 1.12;
    fov = Math.max(0.02, Math.min(3.2, fov * factor));
    renderAll();
  }}, {{ passive: false }});
  srcCanvas.addEventListener('dblclick', function(e) {{
    e.preventDefault();
    centerL = 0; centerB = 0; fov = 2.0;
    selCircle = null; hasSelection = false;
    selected = new Uint8Array(N);
    clearStats();
    renderAll();
    applySkySelectionTo3D();
  }});
}}

/* ── Selection logic ── */
function updateSelection() {{
  if (!selCircle || selCircle.r < 0.5) {{
    hasSelection = false;
    selected = new Uint8Array(N);
    clearStats();
    applySkySelectionTo3D();
    return;
  }}
  hasSelection = true;
  var sl = selCircle.l, sb = selCircle.b, sr = selCircle.r;
  var indices = [];
  for (var i = 0; i < N; i++) {{
    if (D.gl[i] === null) {{ selected[i] = 0; continue; }}
    if (angDistGal(D.gl[i], D.gb[i], sl, sb) <= sr) {{
      selected[i] = 1; indices.push(i);
    }} else {{ selected[i] = 0; }}
  }}
  drawSources();
  updateStats(indices);
  applySkySelectionTo3D();
}}

function doClearSelection() {{
  selCircle = null; hasSelection = false;
  selected = new Uint8Array(N);
  drawSources();
  clearStats();
  applySkySelectionTo3D();
}}

/* ── Stats panel ── */
function updateStats(indices) {{
  var defEl = document.getElementById('stats-default');
  var conEl = document.getElementById('stats-content');
  var clrBtn = document.getElementById('clear-sel-btn');
  if (!conEl) return;
  var n = indices.length;
  if (n === 0) {{ clearStats(); return; }}
  var parts = ['<b>' + n.toLocaleString() + '</b> selected'];
  if (D.gal_prob) {{
    var gals = 0, sumP = 0;
    indices.forEach(function(i) {{
      if (D.gal_prob[i] !== null) {{ sumP += D.gal_prob[i]; if (D.gal_prob[i] >= 0.5) gals++; }}
    }});
    parts.push('<span class="stat-label">gal%:</span> <b>' + (100 * gals / n).toFixed(1) + '%</b>');
    parts.push('<span class="stat-label">mean P:</span> <b>' + (sumP / n).toFixed(3) + '</b>');
  }}
  if (D.phot_g_mean_mag) {{
    var sumG = 0, cnt = 0;
    indices.forEach(function(i) {{ if (D.phot_g_mean_mag[i] !== null) {{ sumG += D.phot_g_mean_mag[i]; cnt++; }} }});
    if (cnt) parts.push('<span class="stat-label">mean G:</span> <b>' + (sumG / cnt).toFixed(2) + '</b>');
  }}
  if (defEl) defEl.style.display = 'none';
  conEl.innerHTML = parts.join(' &nbsp; ');
  conEl.style.display = '';
  if (clrBtn) clrBtn.style.display = '';
}}
function clearStats() {{
  var defEl = document.getElementById('stats-default');
  var conEl = document.getElementById('stats-content');
  var clrBtn = document.getElementById('clear-sel-btn');
  if (defEl) {{
    defEl.textContent = N.toLocaleString() + ' sources \u00b7 Cmd+drag to select \u00b7 Shift+drag to move';
    defEl.style.display = '';
  }}
  if (conEl) conEl.style.display = 'none';
  if (clrBtn) clrBtn.style.display = 'none';
}}

/* Sky selection → 3D highlighting bridge.
   When a sky circle selection is active, dim unselected overlay points in 3D.
   Strategy: match sky selected RA/Dec against 3D overlay trace customdata RA/Dec. */
/* Saved original marker colors per overlay sub-trace for restoring after selection */
var _savedOverlayColors = {{}};

function applySkySelectionTo3D() {{
  if (!plotDiv || !plotDiv.data) return;
  var subIdxs = _currentOverlaySubIndices();
  if (!subIdxs || !subIdxs.length) return;

  if (!hasSelection) {{
    /* Clear selection: restore original colors + sizes */
    var restoreColors = [], restoreSizes = [], restoreOps = [];
    subIdxs.forEach(function(si) {{
      restoreColors.push(_savedOverlayColors[si] || plotDiv.data[si].marker.color);
      restoreSizes.push(1.8);
      restoreOps.push(0.7);
    }});
    Plotly.restyle(plotDiv, {{
      'marker.color': restoreColors,
      'marker.size': restoreSizes,
      'marker.opacity': restoreOps,
    }}, subIdxs);
    return;
  }}

  /* Build a set of selected RA/Dec for fast lookup.
     Use quantized keys to allow for floating-point differences. */
  var selSet = {{}};
  for (var i = 0; i < N; i++) {{
    if (!selected[i]) continue;
    var rk = (D.ra[i] !== null && D.dec[i] !== null)
           ? (Math.round(D.ra[i] * 1000) + '_' + Math.round(D.dec[i] * 1000))
           : null;
    if (rk) selSet[rk] = true;
  }}

  /* For each overlay sub-trace, build per-point color + size arrays.
     Selected = original color + larger, unselected = grey + small.
     RA/Dec positions in customdata vary by overlay type:
       PHZ/DESI sub-traces:  cd[1]=ra, cd[2]=dec
       Gaia XGB/Hilat:       cd[2]=ra, cd[3]=dec
       ERO-DESI:             cd[6]=ra, cd[7]=dec
       ERO:                  cd[1]=ra, cd[2]=dec  */
  /* Save original colors on first selection so subsequent selections
     can always reference the true original (not the grey-mixed array). */
  var newColors = [], newSizes = [], newOps = [];
  subIdxs.forEach(function(si) {{
    var trace = plotDiv.data[si];
    var cd = trace.customdata || [];
    var nm = trace.name || '';
    /* Capture the pristine color BEFORE any selection modifies it */
    if (_savedOverlayColors[si] === undefined) {{
      _savedOverlayColors[si] = trace.marker.color;
    }}
    var origColor = _savedOverlayColors[si];
    /* Determine RA/Dec column indices */
    var ri = 1, di = 2;
    if (/^gxgb_/.test(nm) || /^gxhl_/.test(nm)) {{ ri = 2; di = 3; }}
    else if (/^ed_cat=/.test(nm)) {{ ri = 6; di = 7; }}
    var cols = new Array(cd.length);
    var szs  = new Array(cd.length);
    var isArrayColor = Array.isArray(origColor);
    for (var j = 0; j < cd.length; j++) {{
      var row = cd[j] || [];
      var ra = row[ri], dec = row[di];
      var isSel = false;
      if (ra !== null && ra !== undefined && dec !== null && dec !== undefined) {{
        var rk = Math.round(ra * 1000) + '_' + Math.round(dec * 1000);
        isSel = !!selSet[rk];
      }}
      cols[j] = isSel ? (isArrayColor ? origColor[j] : origColor) : '#222';
      szs[j]  = isSel ? 2.5 : 0.8;
    }}
    newColors.push(cols);
    newSizes.push(szs);
    newOps.push(0.7);
  }});
  Plotly.restyle(plotDiv, {{
    'marker.color': newColors,
    'marker.size': newSizes,
    'marker.opacity': newOps,
    'marker.showscale': [false],
  }}, subIdxs);
}}

/* ══════════════════════════════════════════════════════════════
   3D UMAP STATE + INTERACTION (from compile_umap_3d.py)
   ══════════════════════════════════════════════════════════════ */
var pinned          = false;
var sheetMode       = false;
var sheetPhz        = null;
var sheetCluster    = null;
var isolatedPhz     = null;
var isolatedCluster = null;
var hiddenClusters  = {{}};
var hiddenPhz       = {{}};
var isolatedEROGroup = null;
var hiddenEROGroups  = {{}};
var isolatedGaiaXgb  = null;
var hiddenGaiaXgb    = {{}};
var isolatedGaiaHilat = null;
var hiddenGaiaHilat   = {{}};
var isolatedEroDesi  = null;
var hiddenEroDesi    = {{}};
var activeOverlay   = 'gaia_xgb';

var plotDiv     = document.querySelector('.plotly-graph-div');
var img         = document.getElementById('cutout-img');
var infoDiv     = document.getElementById('cutout-info');
var noCutout    = document.getElementById('no-cutout');
var pinBtn      = document.getElementById('pin-btn');
var sidebar     = document.getElementById('sidebar');
var handle      = document.getElementById('resize-handle');
var sheetView   = document.getElementById('sheet-view');
var sheetTitle  = document.getElementById('sheet-title');
var sheetImages = document.getElementById('sheet-images');
var sheetClose  = document.getElementById('sheet-close');

document.getElementById('tip-dismiss').addEventListener('click', function() {{
  document.getElementById('tip-card').style.display = 'none';
}});
pinBtn.addEventListener('click', function() {{
  pinned = !pinned;
  pinBtn.textContent = pinned ? 'Unpin' : 'Pin';
  pinBtn.classList.toggle('pinned', pinned);
  if (!pinned) unpinCrosshair();
}});

/* ── contact sheet helpers ── */
function clearSheetMode() {{
  sheetMode = false; sheetPhz = null; sheetCluster = null;
  sheetView.style.display = 'none'; sheetImages.innerHTML = '';
}}
function _countTotal(o) {{ return Object.values(o).reduce(function(a,b){{return a+b;}},0); }}

function showContactSheets(v) {{
  if (_isEROOverlay()) {{ showEROSheets(); return; }}
  if (_isEroDesiOverlay()) {{ showEroDesiSheets(_eroDesiIsolatedCat()); return; }}
  var stype = (activeOverlay==='desi') ? 'desi' : 'phz';
  var sheets = (stype==='desi') ? desi_contact_sheets : contact_sheets;
  var counts = (stype==='desi') ? desi_sheet_counts   : sheet_counts;
  var clusters = sheets[v];
  if (!clusters || !Object.keys(clusters).length) {{
    var _pc = PHZ_COLORS[String(v)] || '#aaa';
    var _nameMap = (stype==='desi') ? DESI_NAMES : PHZ_NAMES;
    var _prefix = (stype==='desi') ? 'Q1 ' : '';
    var _lbl = _nameMap[String(v)] || '';
    sheetTitle.innerHTML = '<span style="color:' + _pc + '">' + _prefix + 'phz=' + v + ' ' + _lbl + '</span> — no sheets cached';
    sheetImages.innerHTML = ''; sheetView.style.display = 'block'; return;
  }}
  var keys = Object.keys(clusters).map(Number).sort(function(a,b){{return a-b;}});
  var catCounts = counts[v] || {{}};
  var total = _countTotal(catCounts), html = '';
  var _pc = PHZ_COLORS[String(v)] || '#aaa';
  var _nameMap2 = (stype==='desi') ? DESI_NAMES : PHZ_NAMES;
  var _prefix2 = (stype==='desi') ? 'Q1 ' : '';
  var _lbl2 = _nameMap2[String(v)] || '';
  keys.forEach(function(k) {{
    var n = catCounts[String(k)] !== undefined ? catCounts[String(k)] : '?';
    html += '<div class="sheet-cluster-label">Cluster ' + k + ' (' + n + ' sources)</div>';
    html += '<img data-stype="' + stype + '" data-cat="' + v + '" data-cluster="' + k + '" src="data:image/png;base64,' + clusters[String(k)] + '">';
  }});
  sheetTitle.innerHTML = '<span style="color:' + _pc + '">' + _prefix2 + 'phz=' + v + ' ' + _lbl2 + '</span><br>' + total + ' sources';
  sheetImages.innerHTML = html; sheetView.style.display = 'block';
}}
function showEROSheets() {{
  var groups = Object.keys(ero_contact_sheets);
  if (!groups.length) {{
    sheetTitle.innerHTML = 'ERO — no sheets cached';
    sheetImages.innerHTML = ''; sheetView.style.display = 'block'; return;
  }}
  var html = '', total = 0;
  groups.forEach(function(grp) {{
    var clusters = ero_contact_sheets[grp];
    var counts   = ero_sheet_counts_map[grp] || {{}};
    var config   = ERO_GROUP_CONFIGS_JS[grp] || {{}};
    var label    = config.label || ERO_GROUP_LABELS[grp] || grp;
    var col      = config.color || '#6a9fd8';
    var keys = Object.keys(clusters).map(Number).sort(function(a,b){{return a-b;}});
    if (!keys.length) return;
    html += '<div class="sheet-cluster-label" style="color:' + col + ';margin-top:10px">' + label + '</div>';
    keys.forEach(function(k) {{
      var n = counts[String(k)] !== undefined ? counts[String(k)] : '?';
      if (typeof n === 'number') total += n;
      html += '<div class="sheet-cluster-label">Cluster ' + k + ' (' + n + ' sources)</div>';
      html += '<img data-stype="ero" data-cat="' + grp + '" data-cluster="' + k + '" src="data:image/png;base64,' + clusters[String(k)] + '">';
    }});
  }});
  sheetTitle.innerHTML = 'ERO contact sheets<br>' + total + ' sources';
  sheetImages.innerHTML = html; sheetView.style.display = 'block';
}}
var ERO_DESI_CAT_LABELS = {{'2': 'Disagree: ERO=star, DESI=gal', '3': 'Agree: ERO=gal, DESI=gal'}};
function _eroDesiIsolatedCat() {{
  if (isolatedEroDesi === null) return null;
  var idx = isolatedEroDesi - ERO_DESI_START;
  return (idx >= 0 && idx < ERO_DESI_TRACE_CAT.length) ? String(ERO_DESI_TRACE_CAT[idx]) : null;
}}
function showEroDesiSheets(catFilter) {{
  var cats = catFilter ? [catFilter] : Object.keys(ero_desi_contact_sheets);
  if (!cats.length) {{
    sheetTitle.innerHTML = 'ERO-DESI — no sheets cached';
    sheetImages.innerHTML = ''; sheetView.style.display = 'block'; return;
  }}
  var html = '', total = 0;
  cats.forEach(function(v) {{
    var clusters = ero_desi_contact_sheets[v];
    if (!clusters) return;
    var counts   = ero_desi_sheet_counts[v] || {{}};
    var label    = ERO_DESI_CAT_LABELS[v] || ('cat=' + v);
    var keys = Object.keys(clusters).map(Number).sort(function(a,b){{return a-b;}});
    if (!keys.length) return;
    var col = ERO_DESI_COLORS[v] || '#CE93D8';
    /* Only show category sub-header when showing multiple categories */
    if (!catFilter) html += '<div class="sheet-cluster-label" style="color:' + col + ';margin-top:10px">' + label + '</div>';
    keys.forEach(function(k) {{
      var n = counts[String(k)] !== undefined ? counts[String(k)] : '?';
      if (typeof n === 'number') total += n;
      html += '<div class="sheet-cluster-label">Cluster ' + k + ' (' + n + ' sources)</div>';
      html += '<img data-stype="ero_desi" data-cat="' + v + '" data-cluster="' + k + '" src="data:image/png;base64,' + clusters[String(k)] + '">';
    }});
  }});
  var _edCol = catFilter ? (ERO_DESI_COLORS[catFilter] || '#CE93D8') : '#aaa';
  var titleLabel = catFilter
    ? '<span style="color:' + _edCol + '">' + (ERO_DESI_CAT_LABELS[catFilter] || 'cat=' + catFilter) + '</span>'
    : 'ERO-DESI';
  sheetTitle.innerHTML = titleLabel + '<br>' + total + ' sources';
  sheetImages.innerHTML = html; sheetView.style.display = 'block';
}}
function showClusterSheets(k) {{
  if (_isEroDesiOverlay()) {{
    var catFilter = _eroDesiIsolatedCat();
    var cats = catFilter ? [catFilter] : Object.keys(ero_desi_contact_sheets);
    if (!cats.length) {{
      sheetTitle.innerHTML = 'Cluster ' + k + ' — no ERO-DESI sheets cached';
      sheetImages.innerHTML = ''; sheetView.style.display = 'block'; return;
    }}
    var html = '', total = 0;
    cats.forEach(function(v) {{
      var img = (ero_desi_contact_sheets[v]||{{}})[String(k)];
      if (!img) return;
      var n = ((ero_desi_sheet_counts[v]||{{}})[String(k)]);
      if (typeof n === 'number') total += n; else n = '?';
      var label = ERO_DESI_CAT_LABELS[v] || ('cat=' + v);
      html += '<div class="sheet-cluster-label">' + label + ' (' + n + ' sources)</div>';
      html += '<img data-stype="ero_desi" data-cat="' + v + '" data-cluster="' + k + '" src="data:image/png;base64,' + img + '">';
    }});
    if (!html) {{
      sheetTitle.innerHTML = 'Cluster ' + k + ' — no ERO-DESI sheets cached';
      sheetImages.innerHTML = ''; sheetView.style.display = 'block'; return;
    }}
    var _ck = CLUSTER_COLORS_JS[k % CLUSTER_COLORS_JS.length] || '#333';
    sheetTitle.innerHTML = '<span style="color:' + _ck + '">CLUSTER ' + k + '</span><br>' + total + ' sources';
    sheetImages.innerHTML = html; sheetView.style.display = 'block';
    return;
  }}
  if (_isEROOverlay()) {{
    /* For ERO: show all groups for this cluster */
    var groups = Object.keys(ero_contact_sheets);
    if (!groups.length) {{
      sheetTitle.innerHTML = 'Cluster ' + k + ' — no ERO sheets cached';
      sheetImages.innerHTML = ''; sheetView.style.display = 'block'; return;
    }}
    var html = '', total = 0;
    groups.forEach(function(grp) {{
      var img = (ero_contact_sheets[grp]||{{}})[String(k)];
      if (!img) return;
      var n = ((ero_sheet_counts_map[grp]||{{}})[String(k)]);
      if (typeof n === 'number') total += n; else n = '?';
      var _cfg = ERO_GROUP_CONFIGS_JS[grp] || {{}};
      var label = _cfg.label || ERO_GROUP_LABELS[grp] || grp;
      html += '<div class="sheet-cluster-label">' + label + ' (' + n + ' sources)</div>';
      html += '<img data-stype="ero" data-cat="' + grp + '" data-cluster="' + k + '" src="data:image/png;base64,' + img + '">';
    }});
    if (!html) {{
      sheetTitle.innerHTML = 'Cluster ' + k + ' — no ERO sheets cached';
      sheetImages.innerHTML = ''; sheetView.style.display = 'block'; return;
    }}
    var _ck2 = CLUSTER_COLORS_JS[k % CLUSTER_COLORS_JS.length] || '#333';
    sheetTitle.innerHTML = '<span style="color:' + _ck2 + '">CLUSTER ' + k + '</span><br>' + total + ' sources';
    sheetImages.innerHTML = html; sheetView.style.display = 'block';
    return;
  }}
  var stype = (activeOverlay==='desi') ? 'desi' : 'phz';
  var cl_sheets = (stype==='desi') ? desi_cluster_sheets : cluster_sheets;
  var counts    = (stype==='desi') ? desi_sheet_counts   : sheet_counts;
  var phzs = cl_sheets[String(k)];
  if (!phzs || !Object.keys(phzs).length) {{
    sheetTitle.innerHTML = 'Cluster ' + k + ' — no sheets cached';
    sheetImages.innerHTML = ''; sheetView.style.display = 'block'; return;
  }}
  var keys = Object.keys(phzs).map(Number).sort(function(a,b){{return a-b;}});
  var total = 0, html = '';
  var _nameMap3 = (stype==='desi') ? DESI_NAMES : PHZ_NAMES;
  var _prefix3 = (stype==='desi') ? 'Q1 ' : '';
  keys.forEach(function(v) {{
    var n = (counts[String(v)]||{{}})[String(k)]; n = (n!==undefined)?n:'?';
    if (typeof n==='number') total += n;
    var _lbl3 = _nameMap3[String(v)] || '';
    html += '<div class="sheet-cluster-label">' + _prefix3 + 'phz=' + v + ' ' + _lbl3 + ' (' + n + ' sources)</div>';
    html += '<img data-stype="' + stype + '" data-cat="' + v + '" data-cluster="' + k + '" src="data:image/png;base64,' + phzs[String(v)] + '">';
  }});
  var _ck3 = CLUSTER_COLORS_JS[k % CLUSTER_COLORS_JS.length] || '#333';
  sheetTitle.innerHTML = '<span style="color:' + _ck3 + '">CLUSTER ' + k + '</span><br>' + total + ' sources';
  sheetImages.innerHTML = html; sheetView.style.display = 'block';
}}
sheetClose.addEventListener('click', function() {{ clearSheetMode(); }});
/* Look up extra pages for a contact sheet cell */
var _extraVarNames = {{'phz':'sheet_extra_pages','desi':'desi_sheet_extra','ero':'ero_sheet_extra','ero_desi':'ero_desi_sheet_extra'}};
function _getExtraMap(stype) {{
  if (stype === 'phz') return sheet_extra_pages;
  if (stype === 'desi') return desi_sheet_extra;
  if (stype === 'ero') return ero_sheet_extra;
  if (stype === 'ero_desi') return ero_desi_sheet_extra;
  return {{}};
}}
function _getExtraPages(stype, cat, cluster) {{
  return ((_getExtraMap(stype)[cat] || {{}})[cluster]) || [];
}}
function _openSheetPages(b64, extras) {{
  if (extras.length === 0) {{
    try {{
      var bin = atob(b64), arr = new Uint8Array(bin.length);
      for (var i=0;i<bin.length;i++) arr[i]=bin.charCodeAt(i);
      window.open(URL.createObjectURL(new Blob([arr],{{type:'image/png'}})),'_blank');
    }} catch(_) {{ window.open('data:image/png;base64,'+b64,'_blank'); }}
  }} else {{
    var allB64 = [b64].concat(extras);
    var h = '<!DOCTYPE html><html><head><title>Contact sheets</title>'
      + '<style>body{{background:#111;margin:0;padding:16px;text-align:center}}'
      + 'img{{max-width:100%;margin-bottom:12px;display:block;margin-left:auto;margin-right:auto}}'
      + '.label{{color:#888;font:12px monospace;margin:8px 0 4px}}</style></head><body>';
    for (var p=0; p<allB64.length; p++) {{
      h += '<div class="label">Page ' + (p+1) + ' of ' + allB64.length + '</div>';
      h += '<img src="data:image/png;base64,' + allB64[p] + '">';
    }}
    h += '</body></html>';
    var blob = new Blob([h], {{type:'text/html'}});
    window.open(URL.createObjectURL(blob), '_blank');
  }}
}}
sheetImages.addEventListener('click', function(e) {{
  if (e.target.tagName !== 'IMG') return;
  var img = e.target;
  var stype = img.getAttribute('data-stype');
  var cat = img.getAttribute('data-cat');
  var cluster = img.getAttribute('data-cluster');
  var src = img.src;
  var b64 = src.slice(src.indexOf(',')+1);
  if (!stype || !cat || !cluster) {{
    _openSheetPages(b64, []);
    return;
  }}
  var extras = _getExtraPages(stype, cat, cluster);
  if (extras.length > 0) {{
    _openSheetPages(b64, extras);
    return;
  }}
  /* extras empty — might be lazy-loading; fetch explicitly then retry */
  var varName = _extraVarNames[stype];
  if (!varName) {{ _openSheetPages(b64, []); return; }}
  fetch('data/' + varName + '.json')
    .then(function(r) {{ return r.json(); }})
    .then(function(d) {{
      if (stype === 'phz') sheet_extra_pages = d;
      else if (stype === 'desi') desi_sheet_extra = d;
      else if (stype === 'ero') ero_sheet_extra = d;
      else if (stype === 'ero_desi') ero_desi_sheet_extra = d;
      var ex2 = ((d[cat] || {{}})[cluster]) || [];
      _openSheetPages(b64, ex2);
    }})
    .catch(function() {{ _openSheetPages(b64, []); }});
}});

/* ── trace index caches (computed once — traces never change at runtime) ── */
var _IDX = {{}};  /* populated by _initIndices() */

function _initIndices() {{
  var q1 = [], phzSub = [], allERO = [];
  var eroReg = [], eroRegDummy = [], eroRebal = [], eroRebalDummy = [];
  var gaiaXgb = [], eroDesiLeg = [], eroDesiSub = [];
  plotDiv.data.forEach(function(tr,i) {{
    if (/^Cluster \\d+$/.test(tr.name) || tr.name==='noise') q1.push(i);
    if (/^phz=-?\\d+ C/.test(tr.name)) phzSub.push(i);
  }});
  var i;
  for (i = ERO_REGULAR_START; i < GAIA_XGB_START; i++) allERO.push(i);
  for (i = ERO_REGULAR_START; i < ERO_REGULAR_START + ERO_REG_N_DUMMY; i++) eroRegDummy.push(i);
  for (i = ERO_REG_SUB_START; i < ERO_REG_SUB_START + ERO_REG_N_SUB; i++) eroReg.push(i);
  for (i = ERO_REBAL_START; i < ERO_REBAL_START + ERO_REBAL_N_DUMMY; i++) eroRebalDummy.push(i);
  for (i = ERO_REBAL_SUB_START; i < ERO_REBAL_SUB_START + ERO_REBAL_N_SUB; i++) eroRebal.push(i);
  var gaiaXgbLeg = [], gaiaXgbSub = [];
  for (i = GAIA_XGB_START; i < GAIA_XGB_START + GAIA_XGB_N_LEGEND; i++) gaiaXgbLeg.push(i);
  for (i = GAIA_XGB_SUB_START; i < GAIA_XGB_SUB_START + GAIA_XGB_N_SUB; i++) {{ gaiaXgb.push(i); gaiaXgbSub.push(i); }}
  var gaiaHilatLeg = [], gaiaHilatSub = [], gaiaHilat = [];
  for (i = GAIA_HILAT_START; i < GAIA_HILAT_START + GAIA_HILAT_N_LEGEND; i++) gaiaHilatLeg.push(i);
  for (i = GAIA_HILAT_SUB_START; i < GAIA_HILAT_SUB_START + GAIA_HILAT_N_SUB; i++) {{ gaiaHilat.push(i); gaiaHilatSub.push(i); }}
  for (i = ERO_DESI_START; i < ERO_DESI_START + ERO_DESI_N_LEGEND; i++) eroDesiLeg.push(i);
  for (i = ERO_DESI_SUB_START; i < ERO_DESI_SUB_START + ERO_DESI_N_SUB; i++) eroDesiSub.push(i);

  /* trace map (was buildTraceMap) */
  var baseCluster={{}}, phzLegend={{}}, phzSubMap={{}}, desiLegend={{}}, desiSub={{}};
  plotDiv.data.forEach(function(tr,idx) {{
    var nm=tr.name;
    if (idx >= ERO_REGULAR_START) return;
    var isDesi = (idx>=DESI_TRACE_START && idx<ERO_REGULAR_START);
    var mSub=nm.match(/^phz=(-?\\d+) C(.+)$/);
    var mCl=nm.match(/^Cluster (\\d+)$/);
    var mPhz=nm.match(/(?:^|Q1 )phz=(-?\\d+)/);
    if (mSub) {{
      if (isDesi) {{ if (!desiSub[mSub[1]]) desiSub[mSub[1]]={{}}; desiSub[mSub[1]][mSub[2]]=idx; }}
      else         {{ if (!phzSubMap[mSub[1]])  phzSubMap[mSub[1]]={{}};  phzSubMap[mSub[1]][mSub[2]]=idx; }}
    }} else if (mCl) {{
      baseCluster[mCl[1]]=idx;
    }} else if (mPhz) {{
      if (isDesi) desiLegend[mPhz[1]]=idx; else phzLegend[mPhz[1]]=idx;
    }}
  }});

  /* Gaia XGB sub-trace map: gaiaXgbSubMap[cat][cluster] = traceIdx */
  var gaiaXgbSubMap = {{}};
  gaiaXgbSub.forEach(function(idx) {{
    var nm = plotDiv.data[idx].name;
    var m  = nm.match(/^gxgb_(\\w+) C(\\d+)$/);
    if (m) {{
      if (!gaiaXgbSubMap[m[1]]) gaiaXgbSubMap[m[1]] = {{}};
      gaiaXgbSubMap[m[1]][m[2]] = idx;
    }}
  }});

  /* Gaia high-lat sub-trace map: gaiaHilatSubMap[cat][cluster] = traceIdx */
  var gaiaHilatSubMap = {{}};
  gaiaHilatSub.forEach(function(idx) {{
    var nm = plotDiv.data[idx].name;
    var m  = nm.match(/^gxhl_(\\w+) C(\\d+)$/);
    if (m) {{
      if (!gaiaHilatSubMap[m[1]]) gaiaHilatSubMap[m[1]] = {{}};
      gaiaHilatSubMap[m[1]][m[2]] = idx;
    }}
  }});

  /* ERO sub-trace maps: eroRegSubMap[group][cluster] = traceIdx */
  var eroRegSubMap = {{}};
  eroReg.forEach(function(idx) {{
    var nm = plotDiv.data[idx].name;
    var m  = nm.match(/^ero_ero_grp=(.+) C(\\d+)$/);
    if (m) {{
      if (!eroRegSubMap[m[1]]) eroRegSubMap[m[1]] = {{}};
      eroRegSubMap[m[1]][m[2]] = idx;
    }}
  }});
  var eroRebalSubMap = {{}};
  eroRebal.forEach(function(idx) {{
    var nm = plotDiv.data[idx].name;
    var m  = nm.match(/^ero_eror_grp=(.+) C(\\d+)$/);
    if (m) {{
      if (!eroRebalSubMap[m[1]]) eroRebalSubMap[m[1]] = {{}};
      eroRebalSubMap[m[1]][m[2]] = idx;
    }}
  }});

  /* ERO-DESI sub-trace map: eroDesiSubMap[cat][cluster] = traceIdx */
  var eroDesiSubMap = {{}};
  eroDesiSub.forEach(function(idx) {{
    var nm = plotDiv.data[idx].name;
    var m  = nm.match(/^ed_cat=(-?\\d+) C(\\d+)$/);
    if (m) {{
      if (!eroDesiSubMap[m[1]]) eroDesiSubMap[m[1]] = {{}};
      eroDesiSubMap[m[1]][m[2]] = idx;
    }}
  }});

  _IDX = {{
    q1: q1, phzSub: phzSub, allERO: allERO,
    eroReg: eroReg, eroRegDummy: eroRegDummy, eroRegSubMap: eroRegSubMap,
    eroRebal: eroRebal, eroRebalDummy: eroRebalDummy, eroRebalSubMap: eroRebalSubMap,
    gaiaXgb: gaiaXgb, gaiaXgbLeg: gaiaXgbLeg, gaiaXgbSub: gaiaXgbSub, gaiaXgbSubMap: gaiaXgbSubMap,
    gaiaHilat: gaiaHilat, gaiaHilatLeg: gaiaHilatLeg, gaiaHilatSub: gaiaHilatSub, gaiaHilatSubMap: gaiaHilatSubMap,
    eroDesiLeg: eroDesiLeg, eroDesiSub: eroDesiSub, eroDesiSubMap: eroDesiSubMap,
    baseCluster: baseCluster, phzLegend: phzLegend,
    phzSubMap: phzSubMap, desiLegend: desiLegend, desiSub: desiSub,
  }};
}}

/* convenience accessors (use cached indices) */
function _q1BaseIndices() {{ return _IDX.q1; }}
function _phzSubIndices() {{ return _IDX.phzSub; }}
function _eroTraceIndices(isRebal) {{ return isRebal ? _IDX.eroRebal : _IDX.eroReg; }}
function _eroDummyIndices(isRebal) {{ return isRebal ? _IDX.eroRebalDummy : _IDX.eroRegDummy; }}
function _allEROIndices() {{ return _IDX.allERO; }}
function buildTraceMap() {{
  return {{baseCluster:_IDX.baseCluster, phzLegend:_IDX.phzLegend,
           phzSub:_IDX.phzSubMap, desiLegend:_IDX.desiLegend, desiSub:_IDX.desiSub}};
}}
function _isGaiaXgbOverlay() {{
  return activeOverlay === 'gaia_xgb';
}}
function _isGaiaHilatOverlay() {{
  return activeOverlay === 'gaia_hilat';
}}
function _isEROOverlay() {{
  return (activeOverlay==='ero' || activeOverlay==='ero_rebalanced');
}}
function _isERORebal() {{
  return activeOverlay === 'ero_rebalanced';
}}
function _isEroDesiOverlay() {{
  return activeOverlay === 'ero_desi';
}}
function _isEuclidDesiOverlay() {{
  return activeOverlay === 'euclid_desi';
}}

/* ── Dynamic density helpers ── */
function _currentOverlaySubIndices() {{
  var ov = activeOverlay || 'allphz';
  if (ov==='allphz' || ov==='desi') return _IDX.phzSub;
  if (ov==='ero')             return _IDX.eroReg;
  if (ov==='ero_rebalanced')  return _IDX.eroRebal;
  if (ov==='ero_desi')        return _IDX.eroDesiSub || [];
  if (ov==='euclid_desi')     return (_IDX.phzSub || []).concat(_IDX.eroDesiSub || []);
  if (ov==='gaia_xgb')        return _IDX.gaiaXgbSub || [];
  if (ov==='gaia_hilat')      return _IDX.gaiaHilatSub || [];
  return [];
}}

/* Extract category key from sub-trace name for density histogram lookup */
function _extractDensityCat(name) {{
  var m;
  m = name.match(/^phz=(-?\\d+) C\\d+$/);       if (m) return m[1];
  m = name.match(/^ero_(?:ero|eror)_grp=(.+) C\\d+$/); if (m) return m[1];
  m = name.match(/^gxgb_(\\w+) C\\d+$/);         if (m) return m[1].toLowerCase();
  m = name.match(/^gxhl_(\\w+) C\\d+$/);          if (m) return m[1].toLowerCase();
  m = name.match(/^ed_cat=(\\d+) C\\d+$/);         if (m) return m[1];
  return null;
}}

/* Density lookup helper: given UMAP x,y → log10(count+1) */
var _densityCache = null;
function _getDensityLookup() {{
  if (_densityCache) return _densityCache;
  if (!densityHistData || !densityHistData.bins) return null;
  var bins = densityHistData.bins;
  var nBins = bins.n, xmin = bins.xmin, xmax = bins.xmax, ymin = bins.ymin, ymax = bins.ymax;
  var xStep = (xmax - xmin) / nBins, yStep = (ymax - ymin) / nBins;
  var ov = activeOverlay || 'allphz';
  var histKey = (ov === 'ero_rebalanced') ? 'ero_rebal' : ov;
  var ovHists = (densityHistData.overlays || {{}})[histKey] || {{}};

  var visCats = {{}};
  var subIdxs = _currentOverlaySubIndices();
  subIdxs.forEach(function(si) {{
    var t = plotDiv.data[si];
    if (t.visible !== true) return;
    var cat = _extractDensityCat(t.name);
    if (cat) visCats[cat] = true;
  }});

  var fullHist = new Float32Array(nBins * nBins);
  Object.keys(ovHists).forEach(function(cat) {{
    var h = ovHists[cat]; if (!h) return;
    for (var i = 0; i < h.length && i < fullHist.length; i++) fullHist[i] += h[i];
  }});
  var sumHist = new Float32Array(nBins * nBins);
  var hasCats = false;
  Object.keys(visCats).forEach(function(cat) {{
    var h = ovHists[cat]; if (!h) return;
    hasCats = true;
    for (var i = 0; i < h.length && i < sumHist.length; i++) sumHist[i] += h[i];
  }});
  if (!hasCats) {{ for (var i = 0; i < fullHist.length; i++) sumHist[i] = fullHist[i]; }}

  /* Normalize to this dataset's own density range (not a global max) */
  var fixedVmax = 0;
  for (var _di = 0; _di < sumHist.length; _di++) {{
    var _fv = Math.log10(sumHist[_di] + 1);
    if (_fv > fixedVmax) fixedVmax = _fv;
  }}
  if (fixedVmax === 0) fixedVmax = 1;

  _densityCache = {{
    nBins: nBins, xmin: xmin, xmax: xmax, ymin: ymin, ymax: ymax,
    xStep: xStep, yStep: yStep, sumHist: sumHist, vmax: fixedVmax
  }};
  return _densityCache;
}}
function _densityAt(ux, uy) {{
  var d = _getDensityLookup();
  if (!d) return 0;
  var xi = Math.min(Math.max(Math.floor((ux - d.xmin) / d.xStep), 0), d.nBins - 1);
  var yi = Math.min(Math.max(Math.floor((uy - d.ymin) / d.yStep), 0), d.nBins - 1);
  return Math.log10(d.sumHist[xi * d.nBins + yi] + 1);
}}

function _recomputeDensityColor() {{
  if (_selectedFeature() !== 'umap_density') return;
  if (!densityHistData || !densityHistData.bins) return;
  _densityCache = null;  /* force recompute */
  _savedOverlayColors = {{}};  /* reset for selection restore */
  var dl = _getDensityLookup();
  if (!dl) return;
  var vmin = 0, vmax = dl.vmax;

  /* ── Q1 base traces ── */
  var baseIdxs = _q1BaseIndices();
  if (baseIdxs.length) {{
    baseIdxs.forEach(function(bi, ii) {{
      var coords = q1BaseCoords[ii];
      if (!coords) return;
      var xs = coords.x, ys = coords.y;
      var colors = new Array(xs.length);
      for (var j = 0; j < xs.length; j++) colors[j] = _densityAt(xs[j], ys[j]);
      var mk = plotDiv.data[bi].marker;
      mk.color      = colors;
      mk.colorscale = COLORSCALE_BLUES;
      mk.cmin       = vmin;
      mk.cmax       = vmax;
      mk.opacity    = 0.5;
      mk.showscale  = false;
    }});
  }}

  /* ── Overlay sub-traces: color by density at their UMAP x,y ── */
  var subIdxs = _currentOverlaySubIndices();
  if (subIdxs.length) {{
    subIdxs.forEach(function(si) {{
      var t = plotDiv.data[si];
      var xs = t.x, ys = t.y;
      if (!xs || !xs.length) return;
      var colors = new Array(xs.length);
      for (var j = 0; j < xs.length; j++) colors[j] = _densityAt(xs[j], ys[j]);
      t.marker.color      = colors;
      t.marker.colorscale = COLORSCALE_BLUES;
      t.marker.cmin       = vmin;
      t.marker.cmax       = vmax;
      t.marker.opacity    = 0.7;
      t.marker.showscale  = false;
    }});
  }}

  Plotly.redraw(plotDiv);
  _showOverlayCbar(vmin, vmax, CBAR_CSS_BLUES);
}}

/* ── Q1 base colour helpers ── */
function _selectedFeature() {{ return document.getElementById('color-mode-select').value; }}
function updatePhzHoverTemplate(feature) {{
  var idxs = _phzSubIndices(); if (!idxs.length) return;
  var base = 'RA=%{{customdata[1]:.4f}}<br>Dec=%{{customdata[2]:+.4f}}<br>G=%{{customdata[3]:.2f}}';
  var ext;
  if (feature==='clusters') {{
    ext = '<extra>%{{customdata[4]}}</extra>';
  }} else {{
    /* customdata layout: [sid, ra, dec, gmag, label, phz_pgal, model_pgal, (desi_z?), feat1, ...]
       All PHZ: features start at index 7.  Q1 DESI: features start at index 8 (desi_z at 7).
       Hovertemplate is the same for both; sidebar JS handles the detailed display. */
    var fi = featureNames.indexOf(feature);
    ext = (fi<0) ? '<extra>%{{customdata[4]}}</extra>'
                 : '<extra>%{{customdata[4]}}<br>' + feature + ': %{{customdata['+(8+fi)+']:.4g}}</extra>';
  }}
  Plotly.restyle(plotDiv, {{'hovertemplate': base+ext}}, idxs);
}}
function recolorQ1Base(feature) {{
  var idxs = _q1BaseIndices(); if (!idxs.length) return;
  if (feature==='clusters') {{
    var colors=[], opacities=[], showscales=[];
    idxs.forEach(function(i) {{
      var nm=plotDiv.data[i].name, mCl=nm.match(/^Cluster (\\d+)$/);
      colors.push(mCl ? CLUSTER_COLORS_JS[parseInt(mCl[1])%CLUSTER_COLORS_JS.length] : '#2a2a2a');
      opacities.push(mCl ? 0.15 : 0.5); showscales.push(false);
    }});
    Plotly.restyle(plotDiv, {{'marker.color':colors,'marker.opacity':opacities,'marker.showscale':showscales}}, idxs);
  }} else if (feature==='umap_density') {{
    _recomputeDensityColor();
    return;
  }} else if (feature==='model_pgal' || feature==='phz_pgal' || feature==='g_mag') {{
    /* These use Q1_BASE_Z arrays (not customdata) */
    var zArr = Q1_BASE_Z[feature];
    if (!zArr) return;
    var cmin, cmax;
    if (feature === 'g_mag') {{ cmin = GMAG_NORM[0]; cmax = GMAG_NORM[1]; }}
    else {{ cmin = 0; cmax = 1; }}
    var colorArrays=[], _csA=[], _cmiA=[], _cmaA=[], _opA=[];
    idxs.forEach(function(i,ii) {{
      var vals = zArr[ii] || [];
      if (feature === 'g_mag') {{
        vals = vals.map(function(v) {{
          return (v!==null&&v!==undefined) ? GMAG_NORM[0]+v*(GMAG_NORM[1]-GMAG_NORM[0]) : null;
        }});
      }}
      colorArrays.push(vals);
      var cs = (feature==='model_pgal') ? CSCALE_RdYlBu : CSCALE_RdYlBu_r;
      _csA.push(cs); _cmiA.push(cmin); _cmaA.push(cmax); _opA.push(0.5);
    }});
    Plotly.restyle(plotDiv, {{
      'marker.color': colorArrays, 'marker.colorscale': _csA,
      'marker.cmin': _cmiA, 'marker.cmax': _cmaA, 'marker.opacity': _opA,
      'marker.showscale': [false],
    }}, idxs);
    var cbGrad = (feature==='model_pgal') ? CBAR_CSS_RDYLBU : null;
    _showOverlayCbar(cmin, cmax, cbGrad);
  }} else {{
    var fi=featureNames.indexOf(feature); if (fi<0) return;
    var fcb=FEAT_COLOR_BOUNDS[feature], rng=featureRanges[feature]||[0,1];
    var cmin=fcb?fcb[0]:rng[0], cmax=fcb?fcb[1]:rng[1];
    var colorArrays=[];
    var _csA=[], _cmiA=[], _cmaA=[], _opA=[];
    idxs.forEach(function(i,ii) {{
      var cd=plotDiv.data[i].customdata||[];
      colorArrays.push(cd.map(function(row) {{
        var v=row?row[fi]:null;
        return (v!==null&&v!==undefined&&isFinite(v))?v:null;
      }}));
      _csA.push(CSCALE_RdYlBu_r); _cmiA.push(cmin); _cmaA.push(cmax); _opA.push(0.5);
    }});
    Plotly.restyle(plotDiv, {{
      'marker.color': colorArrays, 'marker.colorscale': _csA,
      'marker.cmin': _cmiA, 'marker.cmax': _cmaA, 'marker.opacity': _opA,
      'marker.showscale': [false],
    }}, idxs);
    _showOverlayCbar(cmin, cmax);
  }}
  updatePhzHoverTemplate(feature);
}}

/* ── ERO colour helpers ── */
function _eroSubGroup(traceIdx) {{
  /* Extract group name from sub-trace name like "ero_ero_grp=agree_star C2" */
  var nm = plotDiv.data[traceIdx].name;
  var m  = nm.match(/^ero_e(?:ro|ror)_grp=(.+) C\\d+$/);
  return m ? m[1] : null;
}}
function restoreERODefaultColors() {{
  if (!_isEROOverlay()) return;
  var idxs = _eroTraceIndices(_isERORebal()); if (!idxs.length) return;
  var colors=[], showscales=[];
  idxs.forEach(function(i) {{
    var grp = _eroSubGroup(i);
    var col = (grp && ERO_GROUP_CONFIGS_JS[grp]) ? ERO_GROUP_CONFIGS_JS[grp].color : '#aaa';
    colors.push(col); showscales.push(false);
  }});
  Plotly.restyle(plotDiv, {{
    'marker.color': colors, 'marker.opacity': [0.7],
    'marker.showscale': showscales,
  }}, idxs);
}}
function recolorEROOverlay(feat) {{
  if (!_isEROOverlay()) return;
  var idxs = _eroTraceIndices(_isERORebal()); if (!idxs.length) return;
  var fi=eroFeatureNames.indexOf(feat); if (fi<0) return;
  var fi_cd = fi + 6;
  var rng=eroFeatureRanges[feat]||[0,1], colorArrays=[], showscales=[];
  idxs.forEach(function(i,ii) {{
    var cd=plotDiv.data[i].customdata||[];
    colorArrays.push(cd.map(function(row) {{
      var v=row?row[fi_cd]:null;
      return (v!==null&&v!==undefined&&isFinite(v))?v:null;
    }}));
  }});
  Plotly.restyle(plotDiv, {{
    'marker.color': colorArrays, 'marker.colorscale': [CSCALE_RdYlBu_r],
    'marker.cmin': [rng[0]], 'marker.cmax': [rng[1]],
    'marker.opacity': [0.5],
    'marker.showscale': [false],
  }}, idxs);
  _showOverlayCbar(rng[0], rng[1]);
}}

/* ── Hovertemplate updaters for other overlays ── */
function _updateOverlayHoverTemplates(feature) {{
  /* ERO sub-traces: features at index 6+ in customdata (raw eroFeatureNames).
     COLOUR dropdown uses combined featureNames; strip "euclid_ero::" prefix to match. */
  var eroIdxs = _eroTraceIndices(_isERORebal());
  if (eroIdxs.length) {{
    var templates = [];
    var eroFeat = feature.replace(/^euclid_ero::/, '');
    var fi = eroFeatureNames.indexOf(eroFeat);
    eroIdxs.forEach(function(i) {{
      /* extract baked-in label from existing hovertemplate */
      var ht = plotDiv.data[i].hovertemplate || '';
      var m = ht.match(/<extra>(.*?)<\\/extra>/);
      var lbl = m ? m[1].replace(/<br>.*/, '') : plotDiv.data[i].name;
      var ext;
      if (feature === 'clusters' || fi < 0) {{
        ext = '<extra>' + lbl + '</extra>';
      }} else {{
        ext = '<extra>' + lbl + '<br>' + eroFeat + ': %{{customdata[' + (6+fi) + ']:.4g}}</extra>';
      }}
      templates.push('RA=%{{customdata[1]:.4f}}<br>Dec=%{{customdata[2]:+.4f}}<br>G=%{{customdata[3]:.2f}}' + ext);
    }});
    Plotly.restyle(plotDiv, {{'hovertemplate': templates}}, eroIdxs);
  }}

  /* Gaia XGB sub-traces: no features, but update to consistent format */
  var gxIdxs = _IDX.gaiaXgbSub || [];
  if (gxIdxs.length) {{
    var templates = [];
    gxIdxs.forEach(function(i) {{
      var ht = plotDiv.data[i].hovertemplate || '';
      var m = ht.match(/<extra>(.*?)<\\/extra>/);
      var lbl = m ? m[1].replace(/<br>.*/, '') : plotDiv.data[i].name;
      templates.push('RA=%{{customdata[2]:.4f}}<br>Dec=%{{customdata[3]:+.4f}}<br>G=%{{customdata[4]:.2f}}<extra>' + lbl + '</extra>');
    }});
    Plotly.restyle(plotDiv, {{'hovertemplate': templates}}, gxIdxs);
  }}

  /* Gaia high-lat sub-traces: same format as Gaia XGB */
  var ghIdxs = _IDX.gaiaHilatSub || [];
  if (ghIdxs.length) {{
    var templates = [];
    ghIdxs.forEach(function(i) {{
      var ht = plotDiv.data[i].hovertemplate || '';
      var m = ht.match(/<extra>(.*?)<\\/extra>/);
      var lbl = m ? m[1].replace(/<br>.*/, '') : plotDiv.data[i].name;
      templates.push('RA=%{{customdata[2]:.4f}}<br>Dec=%{{customdata[3]:+.4f}}<br>G=%{{customdata[4]:.2f}}<extra>' + lbl + '</extra>');
    }});
    Plotly.restyle(plotDiv, {{'hovertemplate': templates}}, ghIdxs);
  }}

  /* ERO DESI sub-traces: no features, consistent format */
  var edIdxs = _IDX.eroDesiSub || [];
  if (edIdxs.length) {{
    var templates = [];
    edIdxs.forEach(function(i) {{
      var ht = plotDiv.data[i].hovertemplate || '';
      var m = ht.match(/<extra>(.*?)<\\/extra>/);
      var lbl = m ? m[1].replace(/<br>.*/, '') : plotDiv.data[i].name;
      templates.push('RA=%{{customdata[6]:.4f}}<br>Dec=%{{customdata[7]:+.4f}}<br>G=%{{customdata[8]:.2f}}<extra>' + lbl + '</extra>');
    }});
    Plotly.restyle(plotDiv, {{'hovertemplate': templates}}, edIdxs);
  }}
}}

/* ── Recolor active overlay traces by feature ── */
function _getOverlayZDict() {{
  var ov = activeOverlay || 'allphz';
  if (ov==='allphz' || ov==='desi') return PHZ_SUB_Z;
  if (ov==='ero')            return ERO_REG_Z;
  if (ov==='ero_rebalanced') return ERO_REBAL_Z;
  if (ov==='gaia_xgb')       return GAIA_XGB_Z;
  if (ov==='gaia_hilat')     return GAIA_HILAT_Z;
  if (ov==='ero_desi')       return ERO_DESI_Z;
  return null;
}}

function recolorOverlayTraces(feature) {{
  if (feature === 'clusters') {{
    restoreOverlayDefaultColors();
    return;
  }}
  if (feature === 'umap_density') {{
    /* Density mode: handled by _recomputeDensityColor which colors
       both Q1 base AND overlay traces */
    return;
  }}
  var ov = activeOverlay || 'allphz';

  /* Map color-dropdown values to z-dict keys */
  var zKey = feature;
  if (feature === 'g_mag') zKey = 'g_mag';
  else if (feature === 'model_pgal') zKey = 'model_pgal';

  if (ov === 'euclid_desi') {{
    /* Combined: recolor DESI sub-traces from PHZ_SUB_Z, ERO-DESI from ERO_DESI_Z */
    _recolorTracesFromZ(_IDX.phzSub, PHZ_SUB_Z, zKey);
    _recolorTracesFromZ(_IDX.eroDesiSub || [], ERO_DESI_Z, zKey);
    return;
  }}

  var subIdxs = _currentOverlaySubIndices();
  var zDict   = _getOverlayZDict();
  if (!subIdxs.length || !zDict) return;
  _recolorTracesFromZ(subIdxs, zDict, zKey);
}}

function _recolorTracesFromZ(idxs, zDict, zKey) {{
  _savedOverlayColors = {{}};  /* reset so next selection captures recolored state */
  if (!idxs.length) return;
  var zArr = zDict[zKey];
  if (!zArr || zArr.length !== idxs.length) return;

  /* Use percentile-based color bounds (1st/99th) from FEAT_COLOR_BOUNDS.
     Z arrays are normalized to [0,1] using FEAT_NORM (min/max); de-normalize
     back to original values, then let Plotly clip to percentile cmin/cmax. */
  var fcb = FEAT_COLOR_BOUNDS[zKey];
  var fnorm = FEAT_NORM[zKey];
  var cmin, cmax;
  if (fcb) {{
    cmin = fcb[0]; cmax = fcb[1];
  }} else if (zKey === 'g_mag') {{
    cmin = GMAG_NORM[0]; cmax = GMAG_NORM[1];
  }} else {{
    cmin = 0; cmax = 1;
  }}

  var colorArrays = [], showscales = [], csArr = [];
  var cmiArr = [], cmaArr = [], opArr = [];
  idxs.forEach(function(tIdx, j) {{
    var vals = zArr[j];
    /* De-normalize from [0,1] back to original value range */
    if (zKey === 'g_mag') {{
      vals = vals.map(function(v) {{
        return (v !== null && v !== undefined) ? GMAG_NORM[0] + v * (GMAG_NORM[1] - GMAG_NORM[0]) : null;
      }});
    }} else if (fnorm) {{
      vals = vals.map(function(v) {{
        return (v !== null && v !== undefined) ? fnorm[0] + v * (fnorm[1] - fnorm[0]) : null;
      }});
    }}
    colorArrays.push(vals);
    var cs = (zKey==='model_pgal') ? CSCALE_RdYlBu : CSCALE_RdYlBu_r;
    csArr.push(cs);
    cmiArr.push(cmin); cmaArr.push(cmax);
    opArr.push(0.7);
  }});
  Plotly.restyle(plotDiv, {{
    'marker.color': colorArrays, 'marker.colorscale': csArr,
    'marker.cmin': cmiArr, 'marker.cmax': cmaArr,
    'marker.opacity': opArr, 'marker.showscale': [false],
  }}, idxs);
  var cbGrad = (zKey==='model_pgal') ? CBAR_CSS_RDYLBU : null;
  _showOverlayCbar(cmin, cmax, cbGrad);
}}

function restoreOverlayDefaultColors() {{
  _hideOverlayCbar();
  _savedOverlayColors = {{}};  /* reset so next selection captures fresh colors */
  var ov = activeOverlay || 'allphz';
  /* Restore ERO overlays */
  if (_isEROOverlay()) {{ restoreERODefaultColors(); return; }}
  /* Restore overlay sub-traces to their original legend color */
  var subIdxs = _currentOverlaySubIndices();
  if (!subIdxs.length) return;
  var colors = [], showscales = [], opArr = [];
  subIdxs.forEach(function(i) {{
    var t = plotDiv.data[i];
    var nm = t.name || '';
    var col = '#aaa';
    /* PHZ/DESI sub-traces: extract phz value from name */
    var mPhz = nm.match(/^phz=(-?\\d+)/);
    if (mPhz) {{
      var v = mPhz[1];
      col = PHZ_COLORS[v] || '#888';
    }}
    /* Gaia XGB sub-traces */
    var mGx = nm.match(/^gxgb_(\\w+)/);
    if (mGx) {{
      col = (mGx[1] === 'gal') ? '#91D3F0' : '#F2BDCE';
    }}
    /* Gaia Hilat sub-traces */
    var mGh = nm.match(/^gxhl_(\\w+)/);
    if (mGh) {{
      col = (mGh[1] === 'gal') ? '#91D3F0' : '#F2BDCE';
    }}
    /* ERO-DESI sub-traces */
    var mEd = nm.match(/^ed_cat=(\\d+)/);
    if (mEd) {{
      col = ERO_DESI_COLORS[mEd[1]] || '#888';
    }}
    colors.push(col); showscales.push(false); opArr.push(0.7);
  }});
  Plotly.restyle(plotDiv, {{
    'marker.color': colors, 'marker.opacity': opArr,
    'marker.showscale': showscales,
  }}, subIdxs);
}}

/* ── CSS overlay colorbar (replaces Plotly's built-in) ── */
var CBAR_CSS_RDYLBU_R = 'linear-gradient(to top,#313695,#74add1,#ffffbf,#f46d43,#a50026)';
var CBAR_CSS_RDYLBU   = 'linear-gradient(to top,#a50026,#f46d43,#ffffbf,#74add1,#313695)';
var CBAR_CSS_BLUES  = 'linear-gradient(to top,rgb(247,251,255),rgb(198,219,239),rgb(107,174,214),rgb(33,113,181),rgb(8,48,107))';
var _cbarFmt = function(v) {{ return Math.abs(v) < 10 ? v.toFixed(2) : v.toFixed(0); }};
function _showOverlayCbar(mn, mx, gradient) {{
  var el = document.getElementById('overlay-cbar-side');
  if (!el) return;
  el.style.display = 'flex';
  document.getElementById('overlay-cbar-gradient').style.background = gradient || CBAR_CSS_RDYLBU_R;
  document.getElementById('overlay-cbar-min').textContent = _cbarFmt(mn);
  document.getElementById('overlay-cbar-max').textContent = _cbarFmt(mx);
}}
function _hideOverlayCbar() {{
  var el = document.getElementById('overlay-cbar-side');
  if (el) el.style.display = 'none';
}}

/* ── Colour dropdown handler — colours Q1 base + overlay + sky ── */
document.getElementById('color-mode-select').addEventListener('change', function() {{
  recolorQ1Base(this.value);
  recolorOverlayTraces(this.value);
  _updateOverlayHoverTemplates(this.value);
  /* Sync sky panel colour mode */
  curSkyColor = this.value;
  drawSources();
}});

/* ── Overlay switching ── */

/* Hide all PHZ + Q1-DESI legend/sub traces in the visibility array */
function _hideAllPhzDesi(vis, map) {{
  Object.keys(map.phzLegend).forEach(function(v){{vis[map.phzLegend[v]]=false;}});
  Object.keys(map.phzSub).forEach(function(v){{
    Object.keys(map.phzSub[v]).forEach(function(k){{vis[map.phzSub[v][k]]=false;}});
  }});
  Object.keys(map.desiLegend).forEach(function(v){{vis[map.desiLegend[v]]=false;}});
  Object.keys(map.desiSub).forEach(function(v){{
    Object.keys(map.desiSub[v]).forEach(function(k){{vis[map.desiSub[v][k]]=false;}});
  }});
}}

/* ── Shared visibility helpers ──────────────────────────────────── */

/* Show/hide a simple list of traces by isolation index / hidden map.
   Used by ERO-DESI and Gaia XGB overlays. */
function _applyOverlayTraces(vis, traceArr, isolatedIdx, hiddenMap) {{
  traceArr.forEach(function(i) {{
    if (isolatedIdx !== null) {{
      vis[i] = (isolatedIdx === i) ? true : 'legendonly';
    }} else {{
      vis[i] = hiddenMap[i] ? 'legendonly' : true;
    }}
  }});
}}

/* Apply Q1 base cluster visibility.
   coverageClusters: array of cluster ids the isolated overlay group populates (or null).
   When coverageClusters is set, only those clusters survive (minus hiddenClusters).
   Otherwise falls through to isolatedCluster / hiddenClusters. */
function _applyClusterVis(vis, map, coverageClusters) {{
  if (coverageClusters) {{
    Object.keys(map.baseCluster).forEach(function(k) {{
      if (coverageClusters.indexOf(parseInt(k)) < 0 || hiddenClusters[k])
        vis[map.baseCluster[k]] = 'legendonly';
    }});
  }} else if (isolatedCluster !== null) {{
    Object.keys(map.baseCluster).forEach(function(k) {{
      if (k !== String(isolatedCluster)) vis[map.baseCluster[k]] = 'legendonly';
    }});
  }} else {{
    Object.keys(hiddenClusters).forEach(function(k) {{
      if (hiddenClusters[k] && map.baseCluster[k] !== undefined)
        vis[map.baseCluster[k]] = 'legendonly';
    }});
  }}
}}

/* ── Main visibility function ────────────────────────────────────── */

function applyVisibility() {{
  var map=buildTraceMap(), n=plotDiv.data.length;
  var vis=new Array(n).fill(true);

  /* Always hide all ERO, Gaia XGB, Gaia high-lat, and ERO-DESI traces first; re-enable active set below */
  _allEROIndices().forEach(function(i){{vis[i]=false;}});
  _IDX.gaiaXgbLeg.forEach(function(i){{vis[i]=false;}});
  _IDX.gaiaXgbSub.forEach(function(i){{vis[i]=false;}});
  _IDX.gaiaHilatLeg.forEach(function(i){{vis[i]=false;}});
  _IDX.gaiaHilatSub.forEach(function(i){{vis[i]=false;}});
  _IDX.eroDesiLeg.forEach(function(i){{vis[i]=false;}});
  _IDX.eroDesiSub.forEach(function(i){{vis[i]=false;}});

  if (_isEuclidDesiOverlay()) {{
    /* Combined mode: show BOTH Q1-DESI + ERO-DESI traces simultaneously.
       Hide PHZ legend traces (keep DESI sub-traces visible). */
    Object.keys(map.phzLegend).forEach(function(v){{vis[map.phzLegend[v]]=false;}});
    Object.keys(map.phzSub).forEach(function(v){{
      Object.keys(map.phzSub[v]).forEach(function(k){{vis[map.phzSub[v][k]]=false;}});
    }});
    /* Show DESI overlay legend + sub-traces */
    Object.keys(map.desiLegend).forEach(function(v){{vis[map.desiLegend[v]]=true;}});
    Object.keys(map.desiSub).forEach(function(v){{
      Object.keys(map.desiSub[v]).forEach(function(k){{
        vis[map.desiSub[v][k]] = hiddenPhz[v] ? 'legendonly' : (hiddenClusters[k] ? 'legendonly' : true);
      }});
    }});
    /* Show ERO-DESI overlay legend + sub-traces */
    _applyOverlayTraces(vis, _IDX.eroDesiLeg, isolatedEroDesi, hiddenEroDesi);
    var _edSubMap = _IDX.eroDesiSubMap;
    Object.keys(_edSubMap).forEach(function(cat) {{
      var catLegIdx = ERO_DESI_START + ERO_DESI_TRACE_CAT.indexOf(parseInt(cat));
      var catHidden = hiddenEroDesi[catLegIdx];
      Object.keys(_edSubMap[cat]).forEach(function(k) {{
        vis[_edSubMap[cat][k]] = (catHidden || hiddenClusters[k]) ? 'legendonly' : true;
      }});
    }});
    _applyClusterVis(vis, map, null);

  }} else if (_isEroDesiOverlay()) {{
    _hideAllPhzDesi(vis, map);
    /* Legend traces: isolate / hide by category */
    _applyOverlayTraces(vis, _IDX.eroDesiLeg, isolatedEroDesi, hiddenEroDesi);
    /* Sub-traces: show/hide per cat×cluster (mirrors PHZ sub-trace logic) */
    var _edSubMap = _IDX.eroDesiSubMap;
    var _edCov = null;

    if (isolatedCluster !== null) {{
      /* Cluster isolated: only show sub-traces for that cluster */
      var ck = String(isolatedCluster);
      Object.keys(_edSubMap).forEach(function(cat) {{
        var catLegIdx = ERO_DESI_START + ERO_DESI_TRACE_CAT.indexOf(parseInt(cat));
        var catHidden = hiddenEroDesi[catLegIdx];
        Object.keys(_edSubMap[cat]).forEach(function(k) {{
          vis[_edSubMap[cat][k]] = (k !== ck || catHidden) ? 'legendonly' : true;
        }});
      }});
      _applyClusterVis(vis, map, null);

    }} else if (isolatedEroDesi !== null) {{
      /* Category isolated: show sub-traces for that cat, respecting hidden clusters */
      var _edTraceIdx = isolatedEroDesi - ERO_DESI_START;
      var _edCat = (_edTraceIdx >= 0 && _edTraceIdx < ERO_DESI_TRACE_CAT.length)
                   ? String(ERO_DESI_TRACE_CAT[_edTraceIdx]) : null;
      _edCov = (_edCat && ERO_DESI_CLUSTER_COVERAGE[_edCat]) || [];
      Object.keys(_edSubMap).forEach(function(cat) {{
        Object.keys(_edSubMap[cat]).forEach(function(k) {{
          vis[_edSubMap[cat][k]] = (cat !== _edCat || hiddenClusters[k]) ? 'legendonly' : true;
        }});
      }});
      _applyClusterVis(vis, map, _edCov);

    }} else {{
      /* Default: show all, respecting hidden categories and hidden clusters */
      Object.keys(_edSubMap).forEach(function(cat) {{
        var catLegIdx = ERO_DESI_START + ERO_DESI_TRACE_CAT.indexOf(parseInt(cat));
        var catHidden = hiddenEroDesi[catLegIdx];
        Object.keys(_edSubMap[cat]).forEach(function(k) {{
          vis[_edSubMap[cat][k]] = (catHidden || hiddenClusters[k]) ? 'legendonly' : true;
        }});
      }});
      _applyClusterVis(vis, map, null);
    }}

  }} else if (_isGaiaXgbOverlay()) {{
    _hideAllPhzDesi(vis, map);
    /* Legend traces */
    _applyOverlayTraces(vis, _IDX.gaiaXgbLeg, isolatedGaiaXgb, hiddenGaiaXgb);
    var _gxSubMap = _IDX.gaiaXgbSubMap;

    if (isolatedCluster !== null) {{
      var ck = String(isolatedCluster);
      Object.keys(_gxSubMap).forEach(function(cat) {{
        var catLegIdx = GAIA_XGB_START + GAIA_XGB_CATS.indexOf(cat);
        var catHidden = hiddenGaiaXgb[catLegIdx];
        Object.keys(_gxSubMap[cat]).forEach(function(k) {{
          vis[_gxSubMap[cat][k]] = (k !== ck || catHidden) ? 'legendonly' : true;
        }});
      }});
      _applyClusterVis(vis, map, null);

    }} else if (isolatedGaiaXgb !== null) {{
      var _gxTraceIdx = isolatedGaiaXgb - GAIA_XGB_START;
      var _gxCat = (_gxTraceIdx >= 0 && _gxTraceIdx < GAIA_XGB_CATS.length)
                   ? GAIA_XGB_CATS[_gxTraceIdx] : null;
      Object.keys(_gxSubMap).forEach(function(cat) {{
        Object.keys(_gxSubMap[cat]).forEach(function(k) {{
          vis[_gxSubMap[cat][k]] = (cat !== _gxCat || hiddenClusters[k]) ? 'legendonly' : true;
        }});
      }});
      _applyClusterVis(vis, map, null);

    }} else {{
      Object.keys(_gxSubMap).forEach(function(cat) {{
        var catLegIdx = GAIA_XGB_START + GAIA_XGB_CATS.indexOf(cat);
        var catHidden = hiddenGaiaXgb[catLegIdx];
        Object.keys(_gxSubMap[cat]).forEach(function(k) {{
          vis[_gxSubMap[cat][k]] = (catHidden || hiddenClusters[k]) ? 'legendonly' : true;
        }});
      }});
      _applyClusterVis(vis, map, null);
    }}

  }} else if (_isGaiaHilatOverlay()) {{
    _hideAllPhzDesi(vis, map);
    _applyOverlayTraces(vis, _IDX.gaiaHilatLeg, isolatedGaiaHilat, hiddenGaiaHilat);
    var _ghSubMap = _IDX.gaiaHilatSubMap;

    if (isolatedCluster !== null) {{
      var ck = String(isolatedCluster);
      Object.keys(_ghSubMap).forEach(function(cat) {{
        var catLegIdx = GAIA_HILAT_START + GAIA_HILAT_CATS.indexOf(cat);
        var catHidden = hiddenGaiaHilat[catLegIdx];
        Object.keys(_ghSubMap[cat]).forEach(function(k) {{
          vis[_ghSubMap[cat][k]] = (k !== ck || catHidden) ? 'legendonly' : true;
        }});
      }});
      _applyClusterVis(vis, map, null);

    }} else if (isolatedGaiaHilat !== null) {{
      var _ghTraceIdx = isolatedGaiaHilat - GAIA_HILAT_START;
      var _ghCat = (_ghTraceIdx >= 0 && _ghTraceIdx < GAIA_HILAT_CATS.length)
                   ? GAIA_HILAT_CATS[_ghTraceIdx] : null;
      Object.keys(_ghSubMap).forEach(function(cat) {{
        Object.keys(_ghSubMap[cat]).forEach(function(k) {{
          vis[_ghSubMap[cat][k]] = (cat !== _ghCat || hiddenClusters[k]) ? 'legendonly' : true;
        }});
      }});
      _applyClusterVis(vis, map, null);

    }} else {{
      Object.keys(_ghSubMap).forEach(function(cat) {{
        var catLegIdx = GAIA_HILAT_START + GAIA_HILAT_CATS.indexOf(cat);
        var catHidden = hiddenGaiaHilat[catLegIdx];
        Object.keys(_ghSubMap[cat]).forEach(function(k) {{
          vis[_ghSubMap[cat][k]] = (catHidden || hiddenClusters[k]) ? 'legendonly' : true;
        }});
      }});
      _applyClusterVis(vis, map, null);
    }}

  }} else if (_isEROOverlay()) {{
    _hideAllPhzDesi(vis, map);
    var _isRebal  = _isERORebal();
    var _eroSubMap = _isRebal ? _IDX.eroRebalSubMap : _IDX.eroRegSubMap;
    /* Show legend dummy traces */
    _eroDummyIndices(_isRebal).forEach(function(i){{vis[i]=true;}});

    if (isolatedCluster !== null) {{
      /* Cluster isolated: show only sub-traces for that cluster */
      var ck = String(isolatedCluster);
      Object.keys(_eroSubMap).forEach(function(grp) {{
        Object.keys(_eroSubMap[grp]).forEach(function(k) {{
          vis[_eroSubMap[grp][k]] = (k !== ck || hiddenEROGroups[grp]) ? 'legendonly' : true;
        }});
      }});
      /* Hide dummy legend for hidden groups */
      _eroDummyIndices(_isRebal).forEach(function(i) {{
        var grp = ERO_TRACE_GROUPS[String(i)];
        if (hiddenEROGroups[grp]) vis[i] = 'legendonly';
      }});
      _applyClusterVis(vis, map, null);

    }} else if (isolatedEROGroup !== null) {{
      /* Group isolated: show sub-traces for that group, respecting hidden clusters */
      Object.keys(_eroSubMap).forEach(function(grp) {{
        Object.keys(_eroSubMap[grp]).forEach(function(k) {{
          vis[_eroSubMap[grp][k]] = (grp !== isolatedEROGroup || hiddenClusters[k]) ? 'legendonly' : true;
        }});
      }});
      /* Hide legend for non-isolated groups */
      _eroDummyIndices(_isRebal).forEach(function(i) {{
        var grp = ERO_TRACE_GROUPS[String(i)];
        if (grp !== isolatedEROGroup) vis[i] = 'legendonly';
      }});
      /* Derive cluster coverage from actual sub-trace data (not contact sheets) */
      var _grpClusters = Object.keys(_eroSubMap[isolatedEROGroup] || {{}});
      _applyClusterVis(vis, map, _grpClusters.map(Number));

    }} else {{
      /* Default: show all, respecting hidden groups and hidden clusters */
      Object.keys(_eroSubMap).forEach(function(grp) {{
        Object.keys(_eroSubMap[grp]).forEach(function(k) {{
          vis[_eroSubMap[grp][k]] = (hiddenEROGroups[grp] || hiddenClusters[k]) ? 'legendonly' : true;
        }});
      }});
      _eroDummyIndices(_isRebal).forEach(function(i) {{
        var grp = ERO_TRACE_GROUPS[String(i)];
        if (hiddenEROGroups[grp]) vis[i] = 'legendonly';
      }});
      _applyClusterVis(vis, map, null);
    }}
  }} else {{
    /* Standard PHZ / DESI switching logic */
    var activeLegend   = (activeOverlay==='desi') ? map.desiLegend : map.phzLegend;
    var activeSub      = (activeOverlay==='desi') ? map.desiSub    : map.phzSub;
    var inactiveLegend = (activeOverlay==='desi') ? map.phzLegend  : map.desiLegend;
    var inactiveSub    = (activeOverlay==='desi') ? map.phzSub     : map.desiSub;
    Object.keys(inactiveLegend).forEach(function(v){{vis[inactiveLegend[v]]=false;}});
    Object.keys(inactiveSub).forEach(function(v){{
      Object.keys(inactiveSub[v]).forEach(function(k){{vis[inactiveSub[v][k]]=false;}});
    }});
    if (isolatedCluster!==null) {{
      var ck=String(isolatedCluster);
      Object.keys(map.baseCluster).forEach(function(k){{if(k!==ck)vis[map.baseCluster[k]]='legendonly';}});
      Object.keys(activeLegend).forEach(function(v){{vis[activeLegend[v]]='legendonly';}});
      Object.keys(activeSub).forEach(function(v){{
        Object.keys(activeSub[v]).forEach(function(k){{if(k!==ck||hiddenPhz[v])vis[activeSub[v][k]]='legendonly';}});
      }});
    }} else if (isolatedPhz!==null) {{
      var cv=String(isolatedPhz);
      Object.keys(activeLegend).forEach(function(v){{if(v!==cv)vis[activeLegend[v]]='legendonly';}});
      Object.keys(map.baseCluster).forEach(function(k){{
        var hasData=activeSub[cv]&&activeSub[cv][k]!==undefined;
        if(!hasData||hiddenClusters[k])vis[map.baseCluster[k]]='legendonly';
      }});
      Object.keys(activeSub).forEach(function(v){{
        Object.keys(activeSub[v]).forEach(function(k){{
          if(v!==cv||hiddenClusters[k])vis[activeSub[v][k]]='legendonly';
        }});
      }});
    }} else {{
      Object.keys(hiddenClusters).forEach(function(k){{
        if(!hiddenClusters[k]) return;
        if(map.baseCluster[k]!==undefined)vis[map.baseCluster[k]]='legendonly';
        Object.keys(activeSub).forEach(function(v){{if(activeSub[v][k]!==undefined)vis[activeSub[v][k]]='legendonly';}});
      }});
      Object.keys(hiddenPhz).forEach(function(v){{
        if(!hiddenPhz[v]) return;
        if(activeLegend[v]!==undefined)vis[activeLegend[v]]='legendonly';
        Object.keys((activeSub[v]||{{}})).forEach(function(k){{vis[activeSub[v][k]]='legendonly';}});
      }});
    }}
  }}

  /* In non-PHZ/DESI overlays, disable Q1 base hover detection so overlay
     points (ERO, Gaia XGB, ERO-DESI) are reachable for hover/click/pin */
  var skipQ1 = (_isEROOverlay() || _isGaiaXgbOverlay() || _isGaiaHilatOverlay() || _isEroDesiOverlay() || _isEuclidDesiOverlay());
  var q1Hi   = _q1BaseIndices();
  var hoverArr = new Array(n);
  for (var _h = 0; _h < n; _h++) hoverArr[_h] = 'all';
  if (skipQ1) {{ q1Hi.forEach(function(i){{ hoverArr[i] = 'skip'; }}); }}

  Plotly.restyle(plotDiv, {{visible: vis, hoverinfo: hoverArr}}).then(function() {{
    _recomputeDensityColor();
  }});
  /* Sync sky map to match the legend hide/isolate state */
  drawSources();
}}

function _updateEROOptgroup() {{
  var sel = document.getElementById('color-mode-select');
  var og  = document.getElementById('ero-colour-optgroup');
  if (!og) return;
  if (_isEROOverlay()) {{
    /* Re-insert into select if it was removed */
    if (og.parentNode !== sel) sel.appendChild(og);
  }} else {{
    /* Physically remove from DOM — display:none is unreliable on <optgroup> */
    if (og.parentNode) og.parentNode.removeChild(og);
    /* If an ERO feature is currently selected, reset to clusters */
    if (sel.value && sel.value.startsWith('euclid_ero::')) {{
      sel.value = 'clusters';
    }}
  }}
}}

/* ── Z-axis ── */
var _currentZ = 'g_mag';

function applyZAxis(opt) {{
  _currentZ = opt;
  var zTitles = {{'g_mag':'G magnitude','model_pgal':'XGB P(gal)','phz_pgal':'PHZ P(gal)'}};
  var zTitle  = zTitles[opt] || (FEAT_NORM[opt] ? opt : 'G magnitude');

  /* Save current camera so the view stays stable across z-axis changes */
  var cam = null;
  if (plotDiv.layout && plotDiv.layout.scene && plotDiv.layout.scene.camera) {{
    cam = JSON.parse(JSON.stringify(plotDiv.layout.scene.camera));
  }}

  var q1Idx = _q1BaseIndices();

  function _q1FlatZ(zVal) {{
    var ref = Q1_BASE_Z['g_mag'] || Q1_BASE_Z['model_pgal'] || Q1_BASE_Z['phz_pgal'];
    if (!ref || q1Idx.length !== ref.length) return null;
    return ref.map(function(a) {{ return a.map(function() {{ return zVal; }}); }});
  }}

  var rIdx = [], rZ = [];

  /* Q1 base always at z=1.45 regardless of z-axis mode */
  var q1Z = _q1FlatZ(1.45);
  if (q1Z && q1Idx.length === q1Z.length) {{
    q1Idx.forEach(function(i,j) {{ rIdx.push(i); rZ.push(q1Z[j]); }});
  }}

  /* PHZ + DESI sub-traces */
  var phzIdx = _phzSubIndices();
  var phzKey = PHZ_SUB_Z[opt] ? opt : 'g_mag';
  if (phzIdx.length && phzIdx.length === PHZ_SUB_Z[phzKey].length) {{
    phzIdx.forEach(function(i,j) {{
      rIdx.push(i);
      rZ.push(PHZ_SUB_Z[phzKey][j]);
    }});
  }}

  /* Active overlay traces */
  if (_isEROOverlay()) {{
    var eroIdx = _eroTraceIndices(_isERORebal());
    var eroZ   = _isERORebal() ? ERO_REBAL_Z : ERO_REG_Z;
    var eroKey = eroZ[opt] ? opt : 'g_mag';
    if (eroIdx.length && eroIdx.length === eroZ[eroKey].length) {{
      eroIdx.forEach(function(i,j) {{
        rIdx.push(i);
        rZ.push(eroZ[eroKey][j]);
      }});
    }}
  }} else if (_isGaiaXgbOverlay()) {{
    var gxIdx = _IDX.gaiaXgbSub;
    var gxKey = GAIA_XGB_Z[opt] ? opt : 'g_mag';
    if (gxIdx.length && gxIdx.length === GAIA_XGB_Z[gxKey].length) {{
      gxIdx.forEach(function(i,j) {{
        rIdx.push(i);
        rZ.push(GAIA_XGB_Z[gxKey][j]);
      }});
    }}
  }} else if (_isGaiaHilatOverlay()) {{
    var ghIdx = _IDX.gaiaHilatSub;
    var ghKey = GAIA_HILAT_Z[opt] ? opt : 'g_mag';
    if (ghIdx.length && ghIdx.length === GAIA_HILAT_Z[ghKey].length) {{
      ghIdx.forEach(function(i,j) {{
        rIdx.push(i);
        rZ.push(GAIA_HILAT_Z[ghKey][j]);
      }});
    }}
  }} else if (_isEroDesiOverlay()) {{
    var edIdx = _IDX.eroDesiSub;
    var edKey = ERO_DESI_Z[opt] ? opt : 'g_mag';
    if (edIdx.length && edIdx.length === ERO_DESI_Z[edKey].length) {{
      edIdx.forEach(function(i,j) {{
        rIdx.push(i);
        rZ.push(ERO_DESI_Z[edKey][j]);
      }});
    }}
  }}

  /* Z axis: always fixed range [-0.05, 1.45].
     Data values occupy [0, 1], Q1 base sits at z=1.45.
     Tick labels map back to original values. */
  var relayoutArgs = {{'scene.zaxis.title.text': zTitle}};
  relayoutArgs['scene.zaxis.range']     = [-0.05, 1.45];
  relayoutArgs['scene.zaxis.autorange'] = false;

  /* Tick labels: map normalised [0,1] back to original values */
  var tpos = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
  var tvals = tpos.slice();
  tvals.push(1.45);
  var ttexts;
  if (opt === 'g_mag') {{
    ttexts = tpos.map(function(t) {{
      return (GMAG_NORM[0] + t * (GMAG_NORM[1] - GMAG_NORM[0])).toFixed(1);
    }});
  }} else if (FEAT_NORM[opt]) {{
    var fnorm = FEAT_NORM[opt];
    ttexts = tpos.map(function(t) {{
      var v = fnorm[0] + t * (fnorm[1] - fnorm[0]);
      return Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(4);
    }});
  }} else {{
    ttexts = tpos.map(function(t) {{ return t.toFixed(1); }});
  }}
  ttexts.push('Q1');
  relayoutArgs['scene.zaxis.tickvals'] = tvals;
  relayoutArgs['scene.zaxis.ticktext'] = ttexts;

  /* Preserve camera + aspect ratio */
  if (cam) relayoutArgs['scene.camera'] = cam;
  relayoutArgs['scene.aspectmode'] = 'manual';
  relayoutArgs['scene.aspectratio'] = plotDiv.layout.scene.aspectratio
    || {{x:1, y:1, z:1}};

  /* Single combined update — one WebGL re-render instead of 3-4 */
  if (rIdx.length > 0) {{
    Plotly.update(plotDiv, {{z: rZ}}, relayoutArgs, rIdx);
  }} else {{
    Plotly.relayout(plotDiv, relayoutArgs);
  }}
}}

document.getElementById('z-select').addEventListener('change', function() {{
  applyZAxis(this.value);
}});

/* Disable Z-axis options that don't apply to the current overlay.
   PHZ P(gal) is a Q1 pipeline output — not available for ERO or Gaia XGB sources. */
function _updateZOptions() {{
  var sel    = document.getElementById('z-select');
  var phzOpt = sel.querySelector('option[value="phz_pgal"]');
  if (!phzOpt) return;
  var noPhz = _isEROOverlay() || _isGaiaXgbOverlay() || _isGaiaHilatOverlay() || _isEroDesiOverlay();
  phzOpt.disabled = noPhz;
  phzOpt.style.color = noPhz ? '#555' : '';
  /* Auto-switch if the user was on PHZ P(gal) and switched to an unsupported overlay */
  if (noPhz && _currentZ === 'phz_pgal') {{
    sel.value = 'g_mag';
    applyZAxis('g_mag');
  }}
}}

function switchOverlay(name) {{
  activeOverlay = name;
  _densityCache = null;  /* invalidate density lookup on overlay switch */
  _savedOverlayColors = {{}};  /* invalidate saved colors for selection restore */
  isolatedCluster=null; isolatedPhz=null; hiddenClusters={{}}; hiddenPhz={{}};
  isolatedEROGroup=null; hiddenEROGroups={{}}; isolatedGaiaXgb=null; hiddenGaiaXgb={{}};
  isolatedGaiaHilat=null; hiddenGaiaHilat={{}};
  isolatedEroDesi=null; hiddenEroDesi={{}};
  clearSheetMode();
  _updateEROOptgroup();
  _updateZOptions();
  applyVisibility();
  recolorQ1Base(_selectedFeature());
  recolorOverlayTraces(_selectedFeature());
  applyZAxis(_currentZ);

  /* ── Sync sky panel ── */
  var skyKey = OVERLAY_TO_SKY[name];
  if (skyKey && SKY_DATASETS[skyKey]) {{
    D = SKY_DATASETS[skyKey];
    N = D.ra ? D.ra.length : 0;
    selected = new Uint8Array(N);
    hasSelection = false; selCircle = null;
    curSkyColor = _selectedFeature();
    resizeCanvases(); renderAll(); clearStats();
  }}
}}

document.getElementById('overlay-select').addEventListener('change', function() {{
  switchOverlay(this.value);
}});

function restoreAll() {{
  isolatedPhz=null; isolatedCluster=null; hiddenClusters={{}}; hiddenPhz={{}};
  isolatedEROGroup=null; hiddenEROGroups={{}}; isolatedGaiaXgb=null; hiddenGaiaXgb={{}};
  isolatedGaiaHilat=null; hiddenGaiaHilat={{}};
  isolatedEroDesi=null; hiddenEroDesi={{}};
  applyVisibility();
}}

/* ── Legend click ── */
function showEROGroupSheets(group) {{
  var clusters = ero_contact_sheets[group];
  var counts   = ero_sheet_counts_map[group] || {{}};
  var config   = ERO_GROUP_CONFIGS_JS[group] || {{}};
  var label    = config.label || ERO_GROUP_LABELS[group] || group;
  var col      = config.color || '#aaa';
  if (!clusters || !Object.keys(clusters).length) {{
    sheetTitle.innerHTML = '<span style="color:' + col + '">' + label + '</span> — no sheets cached';
    sheetImages.innerHTML = ''; sheetView.style.display = 'block'; return;
  }}
  var keys = Object.keys(clusters).map(Number).sort(function(a,b){{return a-b;}});
  var total = 0, html = '';
  keys.forEach(function(k) {{
    var n = counts[String(k)] !== undefined ? counts[String(k)] : '?';
    if (typeof n === 'number') total += n;
    html += '<div class="sheet-cluster-label">Cluster ' + k + ' (' + n + ' sources)</div>';
    html += '<img src="data:image/png;base64,' + clusters[String(k)] + '">';
  }});
  sheetTitle.innerHTML = '<span style="color:' + col + '">' + label + '</span><br>' + total + ' sources';
  sheetImages.innerHTML = html; sheetView.style.display = 'block';
}}

/* ── Custom cluster legend (HTML overlay) ── */
function toggleClusterLegend(el) {{
  var k = el.getAttribute('data-cluster');
  var cmd = window.event && (window.event.metaKey || window.event.ctrlKey);
  if (cmd) {{
    hiddenClusters={{}};hiddenPhz={{}};
    if (k==='noise') {{
      /* noise can't isolate with contact sheets, just toggle */
      isolatedCluster = (isolatedCluster==='noise') ? null : 'noise';
    }} else {{
      if(isolatedCluster===k){{isolatedCluster=null;clearSheetMode();}}
      else{{isolatedCluster=k;isolatedPhz=null;sheetMode=true;sheetCluster=k;showClusterSheets(Number(k));}}
    }}
  }} else {{
    isolatedCluster=null;
    hiddenClusters[k]=!hiddenClusters[k];
  }}
  applyVisibility();
  _syncClusterLegendUI();
}}
function _syncClusterLegendUI() {{
  var items = document.querySelectorAll('#cluster-legend .cl-item');
  items.forEach(function(el) {{
    var k = el.getAttribute('data-cluster');
    if (isolatedCluster !== null) {{
      el.classList.toggle('hidden', k !== isolatedCluster);
    }} else {{
      el.classList.toggle('hidden', !!hiddenClusters[k]);
    }}
  }});
}}

plotDiv.on('plotly_legendclick', function(data) {{
  var traceIdx  = data.expandedIndex;
  var traceName = data.data[traceIdx].name;
  if (/^phz=-?\\d+ C/.test(traceName)) return false;
  if (traceIdx >= ERO_DESI_START && traceIdx < ERO_DESI_START + ERO_DESI_N_LEGEND) {{
    var cmd = data.event && (data.event.metaKey || data.event.ctrlKey);
    if (cmd) {{
      hiddenClusters = {{}}; hiddenEroDesi = {{}};
      if (isolatedEroDesi === traceIdx) {{
        isolatedEroDesi = null; clearSheetMode();
      }} else {{
        isolatedEroDesi = traceIdx;
        sheetMode = true;
        var _cat = _eroDesiIsolatedCat();
        showEroDesiSheets(_cat);
      }}
    }} else {{
      isolatedEroDesi = null;
      hiddenEroDesi[traceIdx] = !hiddenEroDesi[traceIdx];
    }}
    applyVisibility(); return false;
  }}
  if (traceIdx >= GAIA_XGB_START && traceIdx < GAIA_XGB_START + GAIA_XGB_N_LEGEND) {{
    var cmd = data.event && (data.event.metaKey || data.event.ctrlKey);
    if (cmd) {{
      hiddenClusters = {{}}; hiddenGaiaXgb = {{}};
      isolatedGaiaXgb = (isolatedGaiaXgb === traceIdx) ? null : traceIdx;
    }} else {{
      isolatedGaiaXgb = null;
      hiddenGaiaXgb[traceIdx] = !hiddenGaiaXgb[traceIdx];
    }}
    applyVisibility(); return false;
  }}
  if (traceIdx >= GAIA_HILAT_START && traceIdx < GAIA_HILAT_START + GAIA_HILAT_N_LEGEND) {{
    var cmd = data.event && (data.event.metaKey || data.event.ctrlKey);
    if (cmd) {{
      hiddenClusters = {{}}; hiddenGaiaHilat = {{}};
      isolatedGaiaHilat = (isolatedGaiaHilat === traceIdx) ? null : traceIdx;
    }} else {{
      isolatedGaiaHilat = null;
      hiddenGaiaHilat[traceIdx] = !hiddenGaiaHilat[traceIdx];
    }}
    applyVisibility(); return false;
  }}
  if (traceIdx >= ERO_REGULAR_START && traceIdx < GAIA_XGB_START) {{
    var cmd = data.event && (data.event.metaKey || data.event.ctrlKey);
    /* Dummy legend traces come first: [setStart .. setStart+nDummy) */
    var isRebal   = (traceIdx >= ERO_REBAL_START);
    var setStart  = isRebal ? ERO_REBAL_START  : ERO_REGULAR_START;
    var nDummy    = isRebal ? ERO_REBAL_N_DUMMY : ERO_REG_N_DUMMY;
    var isDummy   = (traceIdx >= setStart && traceIdx < setStart + nDummy);
    if (isDummy) {{
      /* Clicking a dummy legend icon — cmd+click isolates, plain click toggles */
      var grp = ERO_TRACE_GROUPS[String(traceIdx)];
      if (cmd) {{
        hiddenClusters = {{}}; hiddenEROGroups = {{}};
        if (isolatedEROGroup === grp) {{ isolatedEROGroup = null; clearSheetMode(); }}
        else {{ isolatedEROGroup = grp; sheetMode = true; if (grp) showEROGroupSheets(grp); else showEROSheets(); }}
        applyVisibility();
      }} else {{
        isolatedEROGroup = null;
        hiddenEROGroups[grp] = !hiddenEROGroups[grp];
        applyVisibility();
      }}
    }} else {{
      /* Click on a data trace (shouldn't normally appear in legend, but just in case) */
      if (cmd) {{
        var grp = ERO_TRACE_GROUPS[String(traceIdx)];
        sheetMode = true;
        if (grp) showEROGroupSheets(grp); else showEROSheets();
      }}
    }}
    return false;
  }}
  var mCl=traceName.match(/^Cluster (\\d+)$/);
  var mPhz=traceName.match(/(?:^|Q1 )phz=(-?\\d+)/);
  var mAct_v = mPhz ? mPhz[1] : null;
  var cmd = data.event && (data.event.metaKey || data.event.ctrlKey);
  if (cmd) {{
    hiddenClusters={{}}; hiddenPhz={{}};
    if (mCl) {{
      var k=mCl[1];
      if(isolatedCluster===k){{isolatedCluster=null;clearSheetMode();}}
      else{{isolatedCluster=k;isolatedPhz=null;sheetMode=true;sheetCluster=k;showClusterSheets(Number(k));}}
    }} else if (mAct_v!==null) {{
      var v=mAct_v;
      if(isolatedPhz===v){{isolatedPhz=null;clearSheetMode();}}
      else{{isolatedPhz=v;isolatedCluster=null;sheetMode=true;sheetPhz=v;showContactSheets(v);}}
    }}
  }} else {{
    if (mCl) {{
      var k=mCl[1];
      if(isolatedPhz!==null){{hiddenClusters[k]=!hiddenClusters[k];}}
      else{{isolatedCluster=null;hiddenClusters[k]=!hiddenClusters[k];}}
    }} else if (mAct_v!==null) {{
      var v=mAct_v; isolatedPhz=null; hiddenPhz[v]=!hiddenPhz[v];
    }}
  }}
  applyVisibility(); return false;
}});
plotDiv.on('plotly_legenddoubleclick', function() {{ return false; }});

document.addEventListener('keydown', function(e) {{
  if (e.key==='Escape') {{
    restoreAll(); clearSheetMode(); _syncClusterLegendUI();
    Plotly.relayout(plotDiv, {{'scene.camera': {{
      eye:{{x:1.25,y:1.25,z:1.25}}, up:{{x:0,y:0,z:1}}, center:{{x:0,y:0,z:0}}
    }}}});
  }}
}});
plotDiv.addEventListener('dblclick', function(e) {{
  var anyHidden=Object.values(hiddenClusters).some(Boolean)||Object.values(hiddenPhz).some(Boolean)||Object.values(hiddenEROGroups).some(Boolean);
  if(isolatedPhz===null&&isolatedCluster===null&&isolatedEROGroup===null&&isolatedGaiaXgb===null&&isolatedGaiaHilat===null&&isolatedEroDesi===null&&!anyHidden) return;
  var node=e.target;
  while(node){{if(node.classList&&node.classList.contains('legend'))return;node=node.parentNode;}}
  restoreAll(); clearSheetMode();
}});

/* ── Hover / click ── */
function showPoint(sid, info) {{
  infoDiv.textContent = info;
  var b64 = cutouts[sid] || ero_cutouts[sid] || null;
  if (b64) {{
    img.src='data:image/png;base64,'+b64; img.style.display='block'; noCutout.textContent='';
  }} else {{
    img.style.display='none'; noCutout.textContent=sid?'(no cutout cached)':'';
  }}
}}
function showPointFromPt(pt) {{
  var nm=pt.data.name;
  var ci=pt.curveNumber;
  var isQ1       = /^Cluster \\d+$/.test(nm) || nm==='noise';
  var isEroDesi  = (ci >= ERO_DESI_START && ci < ERO_DESI_START + ERO_DESI_N_LEGEND) ||
                   (ci >= ERO_DESI_SUB_START && ci < ERO_DESI_SUB_START + ERO_DESI_N_SUB);
  var isGaiaXgb  = (ci >= GAIA_XGB_START && ci < GAIA_XGB_START + GAIA_XGB_N_LEGEND) ||
                   (ci >= GAIA_XGB_SUB_START && ci < GAIA_XGB_SUB_START + GAIA_XGB_N_SUB);
  var isGaiaHilat = (ci >= GAIA_HILAT_START && ci < GAIA_HILAT_START + GAIA_HILAT_N_LEGEND) ||
                    (ci >= GAIA_HILAT_SUB_START && ci < GAIA_HILAT_SUB_START + GAIA_HILAT_N_SUB);
  var isERO      = (ci >= ERO_REGULAR_START && ci < GAIA_XGB_START);
  if (isEroDesi) {{
    var cd = pt.customdata || [];
    var pgal = (cd[1] !== undefined && cd[1] !== null) ? cd[1] : null;
    var stype = cd[2] || '';
    var dz = (cd[3] !== undefined && cd[3] !== null) ? cd[3] : null;
    var cs = (cd[4] !== undefined && cd[4] !== null) ? cd[4] : null;
    var sid = cd[5] ? String(cd[5]) : '';
    var ra  = (cd[6] !== undefined && cd[6] !== null) ? cd[6] : null;
    var dec = (cd[7] !== undefined && cd[7] !== null) ? cd[7] : null;
    var gm  = (cd[8] !== undefined && cd[8] !== null) ? cd[8] : null;
    var _edM = nm.match(/^ed_cat=(\\d+) C(\\d+)$/);
    var _edLabel = nm.replace(/\\s+\\(n=.*\\)/, '');
    if (_edM) {{
      var _edCat = _edM[1];
      var _edNames = {{'2':'Disagree: ERO=star, DESI=gal','3':'Agree: ERO=gal, DESI=gal'}};
      _edLabel = (_edNames[_edCat] || 'ERO DESI cat=' + _edCat) + '  C' + _edM[2];
    }}
    var lines = [_edLabel];
    if (ra !== null && dec !== null)
      lines.push('RA=' + parseFloat(ra).toFixed(4) + '  Dec=' + (dec>=0?'+':'') + parseFloat(dec).toFixed(4));
    if (gm !== null && isFinite(gm)) lines.push('G=' + parseFloat(gm).toFixed(2));
    if (stype) lines.push('DESI: ' + stype);
    if (dz !== null && isFinite(dz)) lines.push('z=' + parseFloat(dz).toFixed(4));
    if (cs !== null && isFinite(cs)) lines.push('class_star=' + parseFloat(cs).toFixed(3));
    if (pgal !== null && isFinite(pgal))
      lines.push('XGB P(gal)=' + parseFloat(pgal).toPrecision(4));
    if (sid) lines.push('id=' + sid);
    showPoint(sid, lines.join('\\n'));
    showSkyCrosshair(ra, dec);
  }} else if (isGaiaXgb) {{
    var cd  = pt.customdata || [];
    var sid  = cd[0] ? String(cd[0]) : '';
    var prob = (cd[1] !== undefined && cd[1] !== null) ? cd[1] : null;
    var ra   = (cd[2] !== undefined && cd[2] !== null) ? cd[2] : null;
    var dec  = (cd[3] !== undefined && cd[3] !== null) ? cd[3] : null;
    var gm   = (cd[4] !== undefined && cd[4] !== null) ? cd[4] : null;
    var _gxM = nm.match(/^gxgb_(star|gal) C(\\d+)$/);
    var _gxLabel = nm.replace(/\\s+\\(n=.*\\)/, '');
    if (_gxM) {{
      _gxLabel = (_gxM[1]==='star' ? 'Gaia XGB Star' : 'Gaia XGB Galaxy') + '  C' + _gxM[2];
    }}
    var lines = [_gxLabel];
    if (ra !== null && dec !== null)
      lines.push('RA=' + parseFloat(ra).toFixed(4) + '  Dec=' + (dec>=0?'+':'') + parseFloat(dec).toFixed(4));
    if (gm !== null && isFinite(gm)) lines.push('G=' + parseFloat(gm).toFixed(2));
    if (prob !== null && prob !== undefined && isFinite(prob))
      lines.push('XGB P(gal)=' + parseFloat(prob).toPrecision(4));
    if (sid) lines.push('id=' + sid);
    showPoint(sid, lines.join('\\n'));
    showSkyCrosshair(ra, dec);
  }} else if (isGaiaHilat) {{
    var cd  = pt.customdata || [];
    var sid  = cd[0] ? String(cd[0]) : '';
    var prob = (cd[1] !== undefined && cd[1] !== null) ? cd[1] : null;
    var ra   = (cd[2] !== undefined && cd[2] !== null) ? cd[2] : null;
    var dec  = (cd[3] !== undefined && cd[3] !== null) ? cd[3] : null;
    var gm   = (cd[4] !== undefined && cd[4] !== null) ? cd[4] : null;
    var _ghM = nm.match(/^gxhl_(star|gal) C(\\d+)$/);
    var _ghLabel = nm.replace(/\\s+\\(n=.*\\)/, '');
    if (_ghM) {{
      _ghLabel = (_ghM[1]==='star' ? 'Gaia XGB HL Star' : 'Gaia XGB HL Galaxy') + '  C' + _ghM[2];
    }}
    var lines = [_ghLabel];
    if (ra !== null && dec !== null)
      lines.push('RA=' + parseFloat(ra).toFixed(4) + '  Dec=' + (dec>=0?'+':'') + parseFloat(dec).toFixed(4));
    if (gm !== null && isFinite(gm)) lines.push('G=' + parseFloat(gm).toFixed(2));
    if (prob !== null && prob !== undefined && isFinite(prob))
      lines.push('XGB P(gal)=' + parseFloat(prob).toPrecision(4));
    if (sid) lines.push('id=' + sid);
    showPoint(sid, lines.join('\\n'));
    showSkyCrosshair(ra, dec);
  }} else if (isQ1) {{
    var feature=_selectedFeature(), lines=[nm];
    if (feature!=='clusters' && pt.customdata) {{
      var fi=featureNames.indexOf(feature);
      var raw=(fi>=0)?pt.customdata[fi]:undefined;
      var val=(raw!==null&&raw!==undefined&&isFinite(raw))?parseFloat(raw).toPrecision(4):'N/A';
      lines.push(feature+': '+val);
    }}
    showPoint('', lines.join('\\n'));
    clearSkyCrosshair();
  }} else if (isERO) {{
    /* customdata: [sid, ra, dec, gmag, class_star, xgb_pgal, feat1, feat2, ...] */
    var cd = pt.customdata || [];
    var sid = cd[0] ? String(cd[0]) : '';
    var ra  = (cd[1] !== undefined && cd[1] !== null) ? cd[1] : null;
    var dec = (cd[2] !== undefined && cd[2] !== null) ? cd[2] : null;
    var gm  = (cd[3] !== undefined && cd[3] !== null) ? cd[3] : null;
    var cs  = (cd[4] !== undefined && cd[4] !== null) ? cd[4] : null;
    var pg  = (cd[5] !== undefined && cd[5] !== null) ? cd[5] : null;
    /* Derive friendly label from sub-trace name */
    var _eroM = nm.match(/^ero_e(?:ro|ror)_grp=(.+) C(\\d+)$/);
    var _eroLabel = nm;
    if (_eroM) {{
      var _cfg = ERO_GROUP_CONFIGS_JS[_eroM[1]];
      _eroLabel = (_cfg ? _cfg.label : _eroM[1]) + '  C' + _eroM[2];
    }}
    var lines=[_eroLabel];
    if (ra !== null && dec !== null)
      lines.push('RA=' + parseFloat(ra).toFixed(4) + '  Dec=' + (dec>=0?'+':'') + parseFloat(dec).toFixed(4));
    if (gm !== null && isFinite(gm)) lines.push('G=' + parseFloat(gm).toFixed(2));
    if (cs !== null && isFinite(cs)) lines.push('class_star=' + parseFloat(cs).toFixed(3));
    if (pg !== null && isFinite(pg)) lines.push('XGB P(gal)=' + parseFloat(pg).toPrecision(4));
    if (sid) lines.push('id='+sid);
    showPoint(sid, lines.join('\\n'));
    showSkyCrosshair(ra, dec);
  }} else {{
    /* All PHZ / Q1 DESI sub-traces:
       customdata: [sid, ra, dec, gmag, label, phz_pgal, model_pgal, desi_z, feat1, ...] */
    if (!pt.customdata) return;
    var cd=pt.customdata;
    var sid=cd[0]?String(cd[0]):'', ra=parseFloat(cd[1]), dec=parseFloat(cd[2]),
        gmag=parseFloat(cd[3]), lbl=cd[4]||nm;
    var ppgal=(cd[5]!==null&&cd[5]!==undefined)?cd[5]:null;
    var mpgal=(cd[6]!==null&&cd[6]!==undefined)?cd[6]:null;
    var dz=(cd[7]!==null&&cd[7]!==undefined)?cd[7]:null;
    var lines=[lbl,
      'RA='+ra.toFixed(4)+'  Dec='+(dec>=0?'+':'')+dec.toFixed(4),
      'G='+gmag.toFixed(2)];
    if (dz!==null&&isFinite(dz)) lines.push('z='+parseFloat(dz).toFixed(4));
    if (ppgal!==null&&isFinite(ppgal))
      lines.push('PHZ P(gal)='+parseFloat(ppgal).toPrecision(4));
    if (mpgal!==null&&isFinite(mpgal))
      lines.push('XGB P(gal)='+parseFloat(mpgal).toPrecision(4));
    if (sid) lines.push('id='+sid);
    showPoint(sid, lines.join('\\n'));
    showSkyCrosshair(ra, dec);
  }}
}}
plotDiv.on('plotly_hover', function(data) {{ if(pinned)return; showPointFromPt(data.points[0]); }});
plotDiv.on('plotly_unhover', function() {{
  if(pinned)return; img.style.display='none'; infoDiv.textContent=''; noCutout.textContent='';
  clearSkyCrosshair();
}});
var _lastClickTime=0, _lastClickPt=null, _dblClickMs=400;
plotDiv.on('plotly_click', function(data) {{
  if(!data.points||!data.points.length)return;
  var now=Date.now(), pt=data.points[0];
  if(now-_lastClickTime<_dblClickMs && _lastClickPt && _lastClickPt.pointIndex===pt.pointIndex && _lastClickPt.curveNumber===pt.curveNumber) {{
    showPointFromPt(pt); pinned=true; _crosshairPinned=true; pinBtn.textContent='Unpin'; pinBtn.classList.add('pinned');
    _lastClickTime=0; _lastClickPt=null;
  }} else {{
    _lastClickTime=now; _lastClickPt=pt;
  }}
}});

/* ── Resize ── */
function resizePlot() {{
  var c=document.getElementById('plot-container');
  Plotly.relayout(plotDiv, {{width:c.clientWidth, height:c.clientHeight}});
}}
function resizeAll() {{
  resizePlot();
  resizeCanvases();
  renderAll();
}}
window.addEventListener('load', resizeAll);
window.addEventListener('resize', resizeAll);
function scaleLegendMarkers() {{
  var s=0.7, lt=plotDiv.querySelectorAll('.legend .traces');
  for (var i=0;i<Math.min(N_BASE_TRACES,lt.length);i++) {{
    var pts=lt[i].querySelector('.legendpoints path'); if(!pts)continue;
    var t=(pts.getAttribute('transform')||'').replace(/\\s*scale\\([^)]*\\)/g,'');
    pts.setAttribute('transform',t+' scale('+s+')');
  }}
}}
plotDiv.on('plotly_afterplot', scaleLegendMarkers);

/* ── Sidebar resize handle ── */
var isResizing=false;
handle.addEventListener('mousedown',function(e){{
  isResizing=true; handle.classList.add('dragging');
  document.body.style.cursor='col-resize'; document.body.style.userSelect='none'; e.preventDefault();
}});
document.addEventListener('mousemove',function(e){{
  if(!isResizing)return;
  var w=Math.max(160,Math.min(700,window.innerWidth-e.clientX-handle.offsetWidth));
  sidebar.style.width=w+'px'; resizePlot(); resizeCanvases(); renderAll();
}});
document.addEventListener('mouseup',function(){{
  if(!isResizing)return;
  isResizing=false; handle.classList.remove('dragging');
  document.body.style.cursor=''; document.body.style.userSelect='';
}});

/* ── Sky / 3D panel resize handle ── */
var skyHandle = document.getElementById('sky-handle');
var skySection = document.getElementById('sky-section');
var isSkyResizing = false;
skyHandle.addEventListener('mousedown', function(e) {{
  isSkyResizing = true; skyHandle.classList.add('dragging');
  document.body.style.cursor = 'col-resize'; document.body.style.userSelect = 'none'; e.preventDefault();
}});
document.addEventListener('mousemove', function(e) {{
  if (!isSkyResizing) return;
  var panelsRect = document.getElementById('main-panels').getBoundingClientRect();
  var skyW = Math.max(200, Math.min(e.clientX - panelsRect.left, panelsRect.width - 500));
  skySection.style.flex = 'none';
  skySection.style.width = skyW + 'px';
  resizePlot(); resizeCanvases(); renderAll();
}});
document.addEventListener('mouseup', function() {{
  if (!isSkyResizing) return;
  isSkyResizing = false; skyHandle.classList.remove('dragging');
  document.body.style.cursor = ''; document.body.style.userSelect = '';
}});

/* ── Initial visibility ── */
_initIndices();
_updateEROOptgroup();
_updateZOptions();
switchOverlay('gaia_xgb');
/* Initial state: base UMAP in cluster colors, overlay+sky in XGB P(gal) */
recolorQ1Base('clusters');

/* ── Sky panel initialization ── */
resizeCanvases();
renderAll();
clearStats();
</script>
</body>
</html>"""

    if output_dir is not None:
        out = output_dir / "umap_combined.html"
        out.write_text(html, encoding="utf-8")
        print(f"  -> {out}  ({n_cached} cutouts embedded)")
    return html



def _density_bin_edges(q1_emb: np.ndarray, n_bins: int = 100):
    """Return fixed bin edges spanning Q1 UMAP range + 0.5 margin."""
    x_edges = np.linspace(q1_emb[:, 0].min() - 0.5, q1_emb[:, 0].max() + 0.5, n_bins + 1)
    y_edges = np.linspace(q1_emb[:, 1].min() - 0.5, q1_emb[:, 1].max() + 0.5, n_bins + 1)
    return x_edges, y_edges


def _build_density_hists(
    q1_emb: np.ndarray,
    df_allphz: pd.DataFrame | None,
    df_desi: pd.DataFrame | None,
    ero_reg_emb: np.ndarray | None, ero_reg_sids: np.ndarray | None,
    ero_rebal_emb: np.ndarray | None, ero_rebal_sids: np.ndarray | None,
    sid_to_group: dict | None,
    gaia_xgb_emb: np.ndarray | None, gaia_xgb_gal_prob: np.ndarray | None,
    gaia_hilat_emb: np.ndarray | None, gaia_hilat_gal_prob: np.ndarray | None,
    ero_desi_emb: np.ndarray | None, ero_desi_cat: np.ndarray | None,
    gal_thresh: float = 0.5,
    n_bins: int = 100,
) -> dict:
    """Build per-category histogram counts for each overlay from full (pre-subsample) data.

    Returns dict ready for JSON: {
        "bins": {"xmin": ..., "xmax": ..., "ymin": ..., "ymax": ..., "n": int},
        "overlays": { overlay_name: { category_key: [flat counts], ... }, ... }
    }
    """
    x_edges, y_edges = _density_bin_edges(q1_emb, n_bins)
    result: dict = {
        "bins": {
            "xmin": round(float(x_edges[0]), 4),
            "xmax": round(float(x_edges[-1]), 4),
            "ymin": round(float(y_edges[0]), 4),
            "ymax": round(float(y_edges[-1]), 4),
            "n": n_bins,
        },
        "overlays": {},
    }

    def _hist(emb_2d):
        H, _, _ = np.histogram2d(emb_2d[:, 0], emb_2d[:, 1], bins=[x_edges, y_edges])
        return [int(v) for v in H.ravel()]

    # Q1 base (self-density, no categories)
    result["overlays"]["q1"] = {"all": _hist(q1_emb)}

    # All PHZ — per phz value
    if df_allphz is not None and "umap1" in df_allphz.columns:
        cats = {}
        for v in sorted(df_allphz["phz"].unique()):
            m = df_allphz["phz"] == v
            if m.sum() == 0:
                continue
            pts = df_allphz.loc[m, ["umap1", "umap2"]].values
            cats[str(int(v))] = _hist(pts)
        result["overlays"]["allphz"] = cats

    # DESI — per phz value
    if df_desi is not None and "umap1" in df_desi.columns:
        cats = {}
        for v in sorted(df_desi["phz"].unique()):
            m = df_desi["phz"] == v
            if m.sum() == 0:
                continue
            pts = df_desi.loc[m, ["umap1", "umap2"]].values
            cats[str(int(v))] = _hist(pts)
        result["overlays"]["desi"] = cats

    # ERO regular — per group
    if ero_reg_emb is not None and ero_reg_sids is not None and sid_to_group:
        cats = {}
        groups = np.array([sid_to_group.get(sid, None) for sid in ero_reg_sids])
        for g in sorted(set(groups) - {None}):
            m = groups == g
            if m.sum() == 0:
                continue
            cats[g] = _hist(ero_reg_emb[m])
        result["overlays"]["ero"] = cats

    # ERO rebalanced — per group
    if ero_rebal_emb is not None and ero_rebal_sids is not None and sid_to_group:
        cats = {}
        groups = np.array([sid_to_group.get(sid, None) for sid in ero_rebal_sids])
        for g in sorted(set(groups) - {None}):
            m = groups == g
            if m.sum() == 0:
                continue
            cats[g] = _hist(ero_rebal_emb[m])
        result["overlays"]["ero_rebal"] = cats

    # Gaia XGB — star / galaxy
    if gaia_xgb_emb is not None and gaia_xgb_gal_prob is not None:
        m_gal = gaia_xgb_gal_prob >= gal_thresh
        result["overlays"]["gaia_xgb"] = {
            "star": _hist(gaia_xgb_emb[~m_gal]),
            "gal":  _hist(gaia_xgb_emb[m_gal]),
        }

    # Gaia XGB high-lat — star / galaxy
    if gaia_hilat_emb is not None and gaia_hilat_gal_prob is not None:
        m_gal = gaia_hilat_gal_prob >= gal_thresh
        result["overlays"]["gaia_hilat"] = {
            "star": _hist(gaia_hilat_emb[~m_gal]),
            "gal":  _hist(gaia_hilat_emb[m_gal]),
        }

    # ERO DESI — per cat value
    if ero_desi_emb is not None and ero_desi_cat is not None:
        cats = {}
        for cv in sorted(set(ero_desi_cat)):
            m = ero_desi_cat == cv
            if m.sum() == 0:
                continue
            cats[str(int(cv))] = _hist(ero_desi_emb[m])
        result["overlays"]["ero_desi"] = cats

    # Euclid x DESI combined — merge DESI (phz cats) + ERO-DESI (cat vals)
    _ed_cats = {}
    if "desi" in result["overlays"]:
        for k, v in result["overlays"]["desi"].items():
            _ed_cats[k] = v
    if "ero_desi" in result["overlays"]:
        for k, v in result["overlays"]["ero_desi"].items():
            _ed_cats[k] = v
    if _ed_cats:
        result["overlays"]["euclid_desi"] = _ed_cats

    n_overlays = len(result["overlays"])
    n_cats = sum(len(v) for v in result["overlays"].values())
    print(f"  Built density histograms: {n_overlays} overlays, {n_cats} categories, {n_bins}×{n_bins} bins")
    return result


def _nn_map_ero_features_onto_q1(
    q1_emb: np.ndarray,
    ero_emb: np.ndarray,
    ero_features: np.ndarray,
    ero_feature_names: list[str],
    q1_features: np.ndarray | None,
    feature_names: list[str],
) -> tuple[np.ndarray | None, list[str], dict]:
    """NN-map ERO feature values onto Q1 base points.

    For each feature column, the NN search is restricted to ERO sources that
    have a finite (non-NaN) value for that feature.  This ensures every Q1
    point receives a valid mapped value even when many ERO sources lack a
    particular measurement (e.g. kron_radius is NULL for point sources).

    The colorscale range for each mapped feature is derived from the original
    ERO source distribution (not the NN-mapped Q1 values), so the colours
    match the static q1_features_ero.png.

    Returns (q1_features, feature_names, ero_feat_ranges) where
    ero_feat_ranges = {'euclid_ero::name': [vmin, vmax], ...}.
    """
    print("  NN-mapping ERO features onto Q1 base points (per-feature valid-only NN)...")
    n_q1    = len(q1_emb)
    n_feats = ero_features.shape[1]
    mapped  = np.full((n_q1, n_feats), np.nan)
    ero_feat_ranges: dict = {}

    for j in range(n_feats):
        valid = np.isfinite(ero_features[:, j])
        if not valid.any():
            print(f"    {ero_feature_names[j]}: all NaN in ERO — skipped")
            continue
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(ero_emb[valid])
        _, idx = nn.kneighbors(q1_emb)
        mapped[:, j] = ero_features[valid][idx.ravel(), j]
        # Range from original ERO distribution (matches static plot)
        src = ero_features[valid, j]
        vmin = round(float(np.percentile(src, 1)),  4)
        vmax = round(float(np.percentile(src, 99)), 4)
        if vmin == vmax:
            vmin, vmax = float(src.min()), float(src.max())
        ero_feat_ranges[f"euclid_ero::{ero_feature_names[j]}"] = [vmin, vmax]
        print(f"    {ero_feature_names[j]}: mapped from {valid.sum():,} valid ERO sources  "
              f"range=[{vmin:.3g}, {vmax:.3g}]")

    new_names = [f"euclid_ero::{fn}" for fn in ero_feature_names]
    if q1_features is not None:
        q1_features = np.concatenate([q1_features, mapped], axis=1)
    else:
        q1_features = mapped
    feature_names = list(feature_names) + new_names
    print(f"  Added {len(new_names)} ERO feature column(s) to Q1 base customdata")
    return q1_features, feature_names, ero_feat_ranges


# ── Z-axis helpers ────────────────────────────────────────────────────────────

def _compute_q1_model_pgal(q1_X_raw: np.ndarray) -> np.ndarray:
    """Run the Q1 XGBoost seed ensemble on Q1 raw features → mean P(galaxy)."""
    import xgboost as xgb
    seed_paths = sorted(Q1_SEED_MODELS_DIR.glob("xgb_classifier_q1_seed*.json"))
    if not seed_paths:
        print(f"  WARNING: no seed models in {Q1_SEED_MODELS_DIR} — model_pgal unavailable")
        return np.full(len(q1_X_raw), np.nan)
    X = q1_X_raw.astype(np.float32).copy()
    for c in range(X.shape[1]):
        bad = ~np.isfinite(X[:, c])
        if bad.any():
            X[bad, c] = np.nanmedian(X[:, c])
    all_probs = []
    for p in seed_paths:
        m = xgb.XGBClassifier()
        m.load_model(str(p))
        all_probs.append(m.predict_proba(X)[:, 1])
    pgal = np.mean(all_probs, axis=0)
    print(f"  Q1 ensemble ({len(seed_paths)} seeds) → P(gal) range "
          f"[{pgal.min():.3f}, {pgal.max():.3f}]")
    return pgal


def _load_phz_gal_prob(source_ids: np.ndarray) -> np.ndarray:
    """Load phz_gal_prob from q1_indiv.npz for Q1 source_ids.  Returns NaN where missing."""
    data = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    if "phz_gal_prob" not in data:
        print("  phz_gal_prob not found in q1_indiv.npz — returning NaN")
        return np.full(len(source_ids), np.nan)
    npz_sids  = data["source_ids"].astype(str)
    npz_pgal  = data["phz_gal_prob"].astype(float)
    lookup    = dict(zip(npz_sids, npz_pgal))
    arr       = np.array([lookup.get(str(s), np.nan) for s in source_ids], dtype=float)
    n_ok      = np.isfinite(arr).sum()
    print(f"  PHZ gal_prob: {n_ok:,} / {len(source_ids):,} sources matched (from q1_indiv.npz)")
    return arr


def _load_q1_z_arrays(q1_sids: np.ndarray, q1_X_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (model_pgal, phz_pgal) arrays aligned to q1_sids."""
    print("  Computing Q1 model P(gal) via ensemble...")
    model_pgal = _compute_q1_model_pgal(q1_X_raw)
    print("  Loading PHZ gal_prob from q1_indiv.npz...")
    phz_pgal = _load_phz_gal_prob(q1_sids)
    return model_pgal, phz_pgal


def _load_ero_pgal(sids: np.ndarray) -> np.ndarray:
    """Load Q1 model P(gal) for ERO sources from the agreement CSVs (xgb_q1_prob)."""
    csv_names = [
        "agree_both_star.csv", "agree_both_galaxy.csv",
        "disagree_ero_star_q1_gal.csv", "disagree_ero_gal_q1_star.csv",
    ]
    dfs = []
    for name in csv_names:
        p = Q1_TO_ERO_DIR / name
        if p.exists():
            dfs.append(pd.read_csv(p, usecols=["g_source_id", "xgb_q1_prob"]))
    if not dfs:
        print("  WARNING: no q1_to_ero CSVs found — ERO model_pgal unavailable")
        return np.full(len(sids), np.nan)
    combined = pd.concat(dfs, ignore_index=True)
    combined["g_source_id"] = combined["g_source_id"].astype(str)
    lookup   = dict(zip(combined["g_source_id"], combined["xgb_q1_prob"]))
    arr      = np.array([lookup.get(sid, np.nan) for sid in sids], dtype=float)
    n_ok     = np.isfinite(arr).sum()
    print(f"  ERO model P(gal): {n_ok:,} / {len(sids):,} sources matched")
    return arr


def _compute_model_pgal_for_df(df: pd.DataFrame) -> np.ndarray:
    """Run Q1 XGBoost seed ensemble on df[SELECTED_GAIA_PARAMETERS] → P(gal).

    Gives 100% coverage for any DataFrame with the 14 Gaia feature columns.
    """
    import xgboost as xgb
    seed_paths = sorted(Q1_SEED_MODELS_DIR.glob("xgb_classifier_q1_seed*.json"))
    if not seed_paths:
        print(f"  WARNING: no seed models in {Q1_SEED_MODELS_DIR}")
        return np.full(len(df), np.nan)
    X = df[SELECTED_GAIA_PARAMETERS].values.astype(np.float32).copy()
    for c in range(X.shape[1]):
        bad = ~np.isfinite(X[:, c])
        if bad.any():
            med = np.nanmedian(X[:, c])
            X[bad, c] = med if np.isfinite(med) else 0.0
    all_probs = []
    for p in seed_paths:
        m = xgb.XGBClassifier()
        m.load_model(str(p))
        all_probs.append(m.predict_proba(X)[:, 1])
    pgal = np.mean(all_probs, axis=0)
    print(f"  XGB ensemble ({len(seed_paths)} seeds) → P(gal) range "
          f"[{pgal.min():.3f}, {pgal.max():.3f}]  ({len(df):,} sources)")
    return pgal


# ── Main ──────────────────────────────────────────────────────────────────────

def build_combined_html(
    reproject: bool = False,
    max_points: int = 50000,
    seed: int = 42,
    write_output: bool = True,
) -> str:
    """Load all data and build the 3D UMAP HTML. Returns the HTML string.
    If write_output=True, also writes umap_3d.html to OUTPUT_DIR."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load Q1 UMAP ───────────────────────────────────────────────────────
    print("[1] Loading Q1 UMAP embedding...")
    q1               = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    q1_emb           = q1["embedding"]
    q1_gmag          = q1["g_mag"].astype(float) if "g_mag" in q1 else None
    q1_features_base = q1["X_raw"].astype(float) if "X_raw" in q1 else None
    q1_sids          = q1["source_ids"].astype(str) if "source_ids" in q1 else None
    print(f"  Q1: {q1_emb.shape[0]:,} points")

    # ── 2. Load cluster labels ────────────────────────────────────────────────
    print("\n[2] Loading cluster labels...")
    labels_path = ARRAYS_DIR / "q1_cluster_labels.npy"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} not found. Run analyze_umap_q1_clusters.py first."
        )
    q1_labels   = np.load(labels_path)
    cluster_ids = sorted(set(q1_labels[q1_labels >= 0]))
    print(f"  Clusters: " + "  ".join(f"C{k}={(q1_labels==k).sum():,}" for k in cluster_ids))

    # ── 3. Load Q1 Euclid features ────────────────────────────────────────────
    print("\n[3] Loading Q1 Euclid features...")
    q1_features_csv = ARRAYS_DIR / "q1_features.csv"
    if q1_sids is not None and q1_features_csv.exists():
        euclid_feats, euclid_names = load_q1_euclid_feats(q1_features_csv, q1_sids)
    else:
        if q1_sids is None:
            print("  source_ids missing in q1_indiv.npz — Q1 Euclid features unavailable.")
        euclid_feats, euclid_names = None, []

    if q1_features_base is not None and euclid_feats is not None:
        q1_features   = np.concatenate([q1_features_base, euclid_feats], axis=1)
        feature_names = list(SELECTED_GAIA_PARAMETERS) + euclid_names
    elif q1_features_base is not None:
        q1_features   = q1_features_base
        feature_names = list(SELECTED_GAIA_PARAMETERS)
    else:
        q1_features   = euclid_feats
        feature_names = euclid_names
    print(f"  Total Q1 feature columns: {len(feature_names)}")

    PROJECTED_DIR.mkdir(parents=True, exist_ok=True)
    allphz_path    = PROJECTED_DIR / "q1_allphz_projected.csv"
    ero_reg_path   = PROJECTED_DIR / "ero_regular_projected.csv"
    ero_rebal_path = PROJECTED_DIR / "ero_rebal_projected.csv"

    ero_features_csv = ARRAYS_DIR / "ero_features.csv"

    # ── Step A: UMAP reducer (lazy load) ─────────────────────────────────────
    q1_mean  = q1["mean"]
    q1_std   = q1["std"]
    std_safe = np.where(q1_std > 0, q1_std, 1.0)
    reducer = None  # loaded on demand

    def _get_reducer():
        nonlocal reducer
        if reducer is None:
            reducer = joblib.load(ARRAYS_DIR / "q1_indiv_reducer.joblib")
        return reducer

    # ── Step B: allphz ────────────────────────────────────────────────────────
    if allphz_path.exists() and not reproject:
        print("\n[4a] Loading saved allphz projected data...")
        df_allphz = pd.read_csv(allphz_path, float_precision="round_trip")
        df_allphz["phz"] = df_allphz["phz"].astype(int)
    else:
        print("\n[4a] Loading + projecting allphz from npz...")
        df_allphz = load_allphz_from_npz()
        X = df_allphz[SELECTED_GAIA_PARAMETERS].values.astype(float)
        X_scaled = np.where(np.isfinite((X - q1_mean) / std_safe),
                            (X - q1_mean) / std_safe, 0.0)
        emb = _get_reducer().transform(X_scaled).astype(np.float64)
        df_allphz = df_allphz.copy()
        df_allphz["umap1"] = emb[:, 0]
        df_allphz["umap2"] = emb[:, 1]
        df_allphz["cluster"] = assign_to_clusters(emb, q1_emb, q1_labels)
        df_allphz.to_csv(allphz_path, index=False, float_format="%.17g")
        print(f"  Saved → {allphz_path}")
    if "cluster" not in df_allphz.columns:
        df_allphz["cluster"] = assign_to_clusters(
            df_allphz[["umap1", "umap2"]].values, q1_emb, q1_labels)
    print(f"  {len(df_allphz):,} sources")

    # ── Step C: DESI ──────────────────────────────────────────────────────────
    df_desi = None
    desi_path = PROJECTED_DIR / "q1_desi_projected.csv"
    if desi_path.exists() and not reproject:
        print("\n[4b] Loading saved DESI data...")
        df_desi = pd.read_csv(desi_path, float_precision="round_trip")
        df_desi["phz"] = df_desi["phz"].astype(int)
    else:
        print("\n[4b] Loading + projecting DESI from npz...")
        try:
            df_desi = load_desi_from_npz()
            X_d = df_desi[SELECTED_GAIA_PARAMETERS].values.astype(float)
            X_d_scaled = np.where(np.isfinite((X_d - q1_mean) / std_safe),
                                  (X_d - q1_mean) / std_safe, 0.0)
            emb_d = _get_reducer().transform(X_d_scaled).astype(np.float64)
            df_desi = df_desi.copy()
            df_desi["umap1"] = emb_d[:, 0]
            df_desi["umap2"] = emb_d[:, 1]
            df_desi["cluster"] = assign_to_clusters(emb_d, q1_emb, q1_labels)
            df_desi.to_csv(desi_path, index=False, float_format="%.17g")
            print(f"  Saved → {desi_path}")
        except Exception as e:
            print(f"  DESI load failed: {e} — DESI overlay disabled.")
    if df_desi is not None and "cluster" not in df_desi.columns:
        df_desi["cluster"] = assign_to_clusters(
            df_desi[["umap1", "umap2"]].values, q1_emb, q1_labels)
    if df_desi is not None:
        print(f"  {len(df_desi):,} DESI sources")

    # ── Step D: ERO regular + rebalanced ──────────────────────────────────────
    print("\n[4c] Loading ERO data...")
    ero_regular_emb = ero_regular_y = ero_regular_gmag = ero_regular_features = None
    ero_rebal_emb   = ero_rebal_y   = ero_rebal_gmag   = ero_rebal_features   = None
    ero_reg_sids = ero_rebal_sids = None
    ero_feature_names = []
    ero_npz_regular = ARRAYS_DIR / "ero_indiv.npz"
    ero_npz_rebal   = ARRAYS_DIR / "ero_indiv_rebalanced.npz"
    _fn_rebal: list[str] = []

    for tag, npz_path, csv_path in [
        ("ERO regular",    ero_npz_regular, ero_reg_path),
        ("ERO rebalanced", ero_npz_rebal,   ero_rebal_path),
    ]:
        emb_ero = y_ero = sids_ero = gmag_ero = feats_ero = None
        fn: list[str] = []

        if csv_path.exists() and not reproject:
            df_ero = pd.read_csv(csv_path, float_precision="round_trip")
            emb_ero  = df_ero[["umap1", "umap2"]].values
            y_ero    = df_ero["y"].values.astype(int)
            sids_ero = df_ero["source_id"].astype(str).values
            gmag_ero = df_ero["phot_g_mean_mag"].values.astype(float) \
                       if "phot_g_mean_mag" in df_ero.columns else None
            print(f"  {tag}: {len(emb_ero):,} sources (from CSV)")
        elif npz_path.exists():
            print(f"  Projecting {tag} into Q1 UMAP space...")
            emb_ero, y_ero, sids_ero = _project_ero(
                npz_path, q1_mean, std_safe, _get_reducer())
            print(f"  {len(emb_ero):,} sources  "
                  f"(STAR={(y_ero==0).sum():,}  GAL={(y_ero==1).sum():,})")
            # Save CSV
            df_ero_save = pd.DataFrame({
                "source_id": sids_ero, "y": y_ero,
                "umap1": emb_ero[:, 0], "umap2": emb_ero[:, 1],
            })
            # gmag will come from load_ero_feats below
            feats_ero, fn, gmag_ero = load_ero_feats(ero_features_csv, sids_ero)
            if gmag_ero is not None:
                df_ero_save["phot_g_mean_mag"] = gmag_ero
            df_ero_save.to_csv(csv_path, index=False, float_format="%.17g")
            print(f"  Saved → {csv_path}")
        else:
            print(f"  {npz_path.name} not found — {tag} overlay disabled.")
            continue

        # Load features (if not already loaded during projection)
        if feats_ero is None:
            feats_ero, fn, gmag_from_feats = load_ero_feats(
                ero_features_csv, sids_ero)
            if gmag_ero is None:
                gmag_ero = gmag_from_feats
        if not ero_feature_names:
            ero_feature_names = fn

        # Assign to the right variables
        if tag == "ERO regular":
            ero_regular_emb, ero_regular_y = emb_ero, y_ero
            ero_regular_gmag, ero_regular_features = gmag_ero, feats_ero
            ero_reg_sids = sids_ero
        else:
            ero_rebal_emb, ero_rebal_y = emb_ero, y_ero
            ero_rebal_gmag, ero_rebal_features = gmag_ero, feats_ero
            ero_rebal_sids = sids_ero
            _fn_rebal = fn

    # Ensure rebalanced feature columns match regular (ero_feature_names) ordering
    if (ero_rebal_features is not None and ero_feature_names
            and _fn_rebal and _fn_rebal != ero_feature_names):
        ero_rebal_features = _align_features(
            ero_rebal_features, _fn_rebal, ero_feature_names)
        print(f"  Aligned rebal features: {_fn_rebal} → {ero_feature_names}")

    # ── Step E: ERO groups + extra data ───────────────────────────────────────
    print("\n[4d] Loading ERO groups + extra data...")
    sid_to_group = load_ero_group_map()
    ero_reg_extra = load_ero_extra_data_from_npz(
        ARRAYS_DIR / "ero_indiv.npz", ero_reg_sids) if ero_reg_sids is not None else {}
    ero_rebal_extra = load_ero_extra_data_from_npz(
        ARRAYS_DIR / "ero_indiv_rebalanced.npz", ero_rebal_sids) if ero_rebal_sids is not None else {}

    # ── Step F: ERO-DESI ──────────────────────────────────────────────────────
    print("\n[4e] Loading ERO-DESI data...")
    ero_desi_emb = ero_desi_cat = ero_desi_model_pgal = ero_desi_gmag = None
    ero_desi_spectype = ero_desi_z = ero_desi_class_star = ero_desi_sids = None
    ero_desi_ra = ero_desi_dec = None
    ero_desi_csv = PROJECTED_DIR / "ero_desi_projected.csv"
    ero_desi_npz = ARRAYS_DIR / "ero_desi.npz"

    if ero_desi_csv.exists() and not reproject:
        df_ed = pd.read_csv(ero_desi_csv, float_precision="round_trip")
        ero_desi_emb        = df_ed[["umap1", "umap2"]].values
        ero_desi_cat        = df_ed["cat_value"].values.astype(int)
        ero_desi_model_pgal = df_ed["model_pgal"].values.astype(float) \
                              if "model_pgal" in df_ed.columns else None
        ero_desi_gmag       = df_ed["phot_g_mean_mag"].values.astype(float) \
                              if "phot_g_mean_mag" in df_ed.columns else None
        ero_desi_spectype   = df_ed["spectype"].values.astype(str) \
                              if "spectype" in df_ed.columns else None
        ero_desi_z          = df_ed["desi_z"].values.astype(float) \
                              if "desi_z" in df_ed.columns else None
        ero_desi_class_star = df_ed["class_star"].values.astype(float) \
                              if "class_star" in df_ed.columns else None
        ero_desi_sids       = df_ed["source_id"].values.astype(str) \
                              if "source_id" in df_ed.columns else None
        ero_desi_ra         = df_ed["ra"].values.astype(float) \
                              if "ra" in df_ed.columns else None
        ero_desi_dec        = df_ed["dec"].values.astype(float) \
                              if "dec" in df_ed.columns else None
        print(f"  {len(df_ed):,} ERO-DESI sources  "
              f"(cat=2: {(ero_desi_cat==2).sum():,},  cat=3: {(ero_desi_cat==3).sum():,})")
    elif ero_desi_npz.exists():
        print("  Projecting ERO-DESI on the fly...")
        ed_npz = np.load(ero_desi_npz, allow_pickle=True)
        ed_X   = ed_npz["X_features"].astype(float)
        ed_cat = ed_npz["cat_value"].astype(int)
        ed_sids = ed_npz["source_ids"].astype(str) \
                  if "source_ids" in ed_npz else None
        ero_desi_gmag = ed_npz["phot_g_mean_mag"].astype(float) \
                        if "phot_g_mean_mag" in ed_npz else None
        ero_desi_spectype = ed_npz["spectype"].astype(str) \
                            if "spectype" in ed_npz else None
        ero_desi_z = ed_npz["desi_z"].astype(float) \
                     if "desi_z" in ed_npz else None
        ero_desi_class_star = ed_npz["class_star"].astype(float) \
                              if "class_star" in ed_npz else None
        ero_desi_ra = ed_npz["ra"].astype(float) \
                      if "ra" in ed_npz else None
        ero_desi_dec = ed_npz["dec"].astype(float) \
                       if "dec" in ed_npz else None
        # Standardise and project
        ed_scaled = (ed_X - q1_mean) / std_safe
        ed_scaled = np.where(np.isfinite(ed_scaled), ed_scaled, 0.0)
        ero_desi_emb = _get_reducer().transform(ed_scaled).astype(np.float64)
        ero_desi_cat = ed_cat
        ero_desi_sids = ed_sids
        # XGB inference for model_pgal
        df_ed_for_xgb = pd.DataFrame(ed_X, columns=SELECTED_GAIA_PARAMETERS)
        ero_desi_model_pgal = _compute_model_pgal_for_df(df_ed_for_xgb)
        # Save CSV for future reuse
        save_dict = {
            "umap1": ero_desi_emb[:, 0], "umap2": ero_desi_emb[:, 1],
            "cat_value": ero_desi_cat, "model_pgal": ero_desi_model_pgal,
        }
        if ed_sids is not None:
            save_dict["source_id"] = ed_sids
        if ero_desi_gmag is not None:
            save_dict["phot_g_mean_mag"] = ero_desi_gmag
        if ero_desi_spectype is not None:
            save_dict["spectype"] = ero_desi_spectype
        if ero_desi_z is not None:
            save_dict["desi_z"] = ero_desi_z
        if ero_desi_class_star is not None:
            save_dict["class_star"] = ero_desi_class_star
        if ero_desi_ra is not None:
            save_dict["ra"] = ero_desi_ra
        if ero_desi_dec is not None:
            save_dict["dec"] = ero_desi_dec
        pd.DataFrame(save_dict).to_csv(ero_desi_csv, index=False, float_format="%.17g")
        print(f"  {len(ed_cat):,} ERO-DESI sources projected  "
              f"(cat=2: {(ed_cat==2).sum():,},  cat=3: {(ed_cat==3).sum():,})")
        print(f"  Saved → {ero_desi_csv}")
    else:
        print("  ero_desi.npz not found — ERO-DESI overlay disabled.")

    # ── Step G: Gaia XGB ──────────────────────────────────────────────────────
    print("\n[4f] Loading Gaia XGB data...")
    gaia_xgb_emb = gaia_xgb_gal_prob = gaia_xgb_sids = gaia_xgb_gmag = None
    gaia_xgb_ra = gaia_xgb_dec = None
    if GAIA_XGB_CSV.exists():
        df_gxgb            = pd.read_csv(GAIA_XGB_CSV)
        gaia_xgb_emb       = df_gxgb[["umap1", "umap2"]].values
        gaia_xgb_gal_prob = df_gxgb["gal_prob"].values.astype(float)
        gaia_xgb_sids      = df_gxgb["source_id"].astype(str).values
        gaia_xgb_gmag      = df_gxgb["phot_g_mean_mag"].values.astype(float) \
                             if "phot_g_mean_mag" in df_gxgb.columns else None
        gaia_xgb_ra        = df_gxgb["ra"].values.astype(float) \
                             if "ra" in df_gxgb.columns else None
        gaia_xgb_dec       = df_gxgb["dec"].values.astype(float) \
                             if "dec" in df_gxgb.columns else None
        print(f"  {len(gaia_xgb_emb):,} Gaia XGB sources  "
              f"(galaxies={(gaia_xgb_gal_prob >= GAL_THRESH).sum():,}  "
              f"stars={(gaia_xgb_gal_prob < GAL_THRESH).sum():,})")
        # Save full data for density histograms before subsampling
        _gaia_xgb_full_emb  = gaia_xgb_emb.copy()
        _gaia_xgb_full_gprob = gaia_xgb_gal_prob.copy()
        # Use all 50k Gaia XGB sources (data is small relative to image blobs)
        _gxgb_cap = 50_000
        if len(gaia_xgb_emb) > _gxgb_cap:
            _gi = _random_subsample_idx(len(gaia_xgb_emb), _gxgb_cap, seed)
            gaia_xgb_emb       = gaia_xgb_emb[_gi]
            gaia_xgb_gal_prob = gaia_xgb_gal_prob[_gi]
            gaia_xgb_sids      = gaia_xgb_sids[_gi]
            if gaia_xgb_gmag is not None: gaia_xgb_gmag = gaia_xgb_gmag[_gi]
            if gaia_xgb_ra   is not None: gaia_xgb_ra   = gaia_xgb_ra[_gi]
            if gaia_xgb_dec  is not None: gaia_xgb_dec  = gaia_xgb_dec[_gi]
            print(f"  Subsampled to {_gxgb_cap:,} (density-preserving)")
    else:
        _gaia_xgb_full_emb = _gaia_xgb_full_gprob = None
        print(f"  {GAIA_XGB_CSV.name} not found — Gaia XGB overlay disabled.")

    # ── Step H: Gaia hilat ────────────────────────────────────────────────────
    gaia_hilat_emb = gaia_hilat_gal_prob = gaia_hilat_sids = gaia_hilat_gmag = None
    gaia_hilat_ra = gaia_hilat_dec = None
    if GAIA_HILAT_CSV.exists():
        df_hilat = pd.read_csv(GAIA_HILAT_CSV)
    elif GAIA_XGB_CSV.exists():
        df_gxgb_full = pd.read_csv(GAIA_XGB_CSV)
        df_hilat = _filter_gaia_xgb_hilat(df_gxgb_full)
    else:
        df_hilat = None
    if df_hilat is not None and len(df_hilat) > 0:
        gaia_hilat_emb       = df_hilat[["umap1", "umap2"]].values
        gaia_hilat_gal_prob = df_hilat["gal_prob"].values.astype(float)
        gaia_hilat_sids      = df_hilat["source_id"].astype(str).values
        gaia_hilat_gmag      = df_hilat["phot_g_mean_mag"].values.astype(float) \
                               if "phot_g_mean_mag" in df_hilat.columns else None
        gaia_hilat_ra        = df_hilat["ra"].values.astype(float) \
                               if "ra" in df_hilat.columns else None
        gaia_hilat_dec       = df_hilat["dec"].values.astype(float) \
                               if "dec" in df_hilat.columns else None
        _gaia_hilat_full_emb   = gaia_hilat_emb.copy()
        _gaia_hilat_full_gprob = gaia_hilat_gal_prob.copy()
        _ghl_cap = 50_000
        if len(gaia_hilat_emb) > _ghl_cap:
            _ghi = _random_subsample_idx(len(gaia_hilat_emb), _ghl_cap, seed)
            gaia_hilat_emb       = gaia_hilat_emb[_ghi]
            gaia_hilat_gal_prob = gaia_hilat_gal_prob[_ghi]
            gaia_hilat_sids      = gaia_hilat_sids[_ghi]
            if gaia_hilat_gmag is not None: gaia_hilat_gmag = gaia_hilat_gmag[_ghi]
            if gaia_hilat_ra   is not None: gaia_hilat_ra   = gaia_hilat_ra[_ghi]
            if gaia_hilat_dec  is not None: gaia_hilat_dec  = gaia_hilat_dec[_ghi]
            print(f"  High-lat: subsampled to {_ghl_cap:,}")
        print(f"  High-lat: {len(gaia_hilat_emb):,} sources  "
              f"(|b|>10, |ecl_lat|>45)  "
              f"(galaxies={(gaia_hilat_gal_prob >= GAL_THRESH).sum():,}  "
              f"stars={(gaia_hilat_gal_prob < GAL_THRESH).sum():,})")
    else:
        _gaia_hilat_full_emb = _gaia_hilat_full_gprob = None
        print("  High-lat: no sources pass latitude cuts or CSV not found.")

    # ── Step I: Cutouts ───────────────────────────────────────────────────────
    print("\n[5a] Loading cutout thumbnails and contact sheets...")
    cutout_cache  = load_cutout_cache(CUTOUTS_DIR)
    contact_sheet_cache, sheet_counts, sheet_extra = load_contact_sheet_cache_allphz(CUTOUTS_DIR)
    desi_contact_sheet_cache, desi_sheet_counts, desi_sheet_extra = load_contact_sheet_cache_desi(DESI_CUTOUTS_DIR)
    ero_desi_cs_cache, ero_desi_cs_counts, ero_desi_sheet_extra = load_contact_sheet_cache_ero_desi(ERO_DESI_CUTOUTS_DIR)
    ero_contact_sheet_cache, ero_sheet_counts_data, ero_sheet_extra = load_contact_sheet_cache_ero(ERO_CUTOUTS_DIR)
    ero_cutout_cache_data = load_cutout_cache_ero(ERO_CUTOUTS_DIR)
    # Also load ERO-DESI individual cutouts (keyed by source_id)
    _ero_desi_cutouts = load_cutout_cache_ero(ERO_DESI_CUTOUTS_DIR)
    if _ero_desi_cutouts:
        ero_cutout_cache_data.update(_ero_desi_cutouts)

    # ── Step J: Save full copies for density hists, then subsample ────────────
    _df_allphz_full = df_allphz.copy() if df_allphz is not None else None
    _df_desi_full   = df_desi.copy()   if df_desi is not None else None
    _ero_reg_full_emb    = ero_regular_emb.copy() if ero_regular_emb is not None else None
    _ero_reg_full_sids   = ero_reg_sids.copy()    if ero_reg_sids is not None else None
    _ero_rebal_full_emb  = ero_rebal_emb.copy()   if ero_rebal_emb is not None else None
    _ero_rebal_full_sids = ero_rebal_sids.copy()   if ero_rebal_sids is not None else None

    # Subsample allphz
    if max_points is not None:
        print(f"\n  Applying --max-points {max_points} (hybrid density-preserving)...")
        cutout_sids = set(cutout_cache.keys())
        df_allphz = _subsample_hybrid(
            df_allphz, "phz", PHZ_LABELS, cutout_sids,
            max_total=max_points, floor=500, seed=seed)

    # Subsample ERO overlays (no cap — show all available sources)
    _ero_cap = 999_999
    if ero_regular_emb is not None and len(ero_regular_emb) > _ero_cap:
        _ei = _random_subsample_idx(len(ero_regular_emb), _ero_cap, seed)
        ero_regular_emb = ero_regular_emb[_ei]
        if ero_regular_y is not None: ero_regular_y = ero_regular_y[_ei]
        if ero_regular_gmag is not None: ero_regular_gmag = ero_regular_gmag[_ei]
        if ero_regular_features is not None: ero_regular_features = ero_regular_features[_ei]
        if ero_reg_sids is not None: ero_reg_sids = ero_reg_sids[_ei]
        if ero_reg_extra:
            ero_reg_extra = {k: v[_ei] for k, v in ero_reg_extra.items()}
        print(f"  ERO regular subsampled to {_ero_cap:,} (density-preserving)")
    if ero_rebal_emb is not None and len(ero_rebal_emb) > _ero_cap:
        _ei = _random_subsample_idx(len(ero_rebal_emb), _ero_cap, seed)
        ero_rebal_emb = ero_rebal_emb[_ei]
        if ero_rebal_y is not None: ero_rebal_y = ero_rebal_y[_ei]
        if ero_rebal_gmag is not None: ero_rebal_gmag = ero_rebal_gmag[_ei]
        if ero_rebal_features is not None: ero_rebal_features = ero_rebal_features[_ei]
        if ero_rebal_sids is not None: ero_rebal_sids = ero_rebal_sids[_ei]
        if ero_rebal_extra:
            ero_rebal_extra = {k: v[_ei] for k, v in ero_rebal_extra.items()}
        print(f"  ERO rebalanced subsampled to {_ero_cap:,} (density-preserving)")

    # ── Step J2: Load Gaia features for z-axis ──────────────────────────────
    print("\n  Loading Gaia features for z-axis overlays...")
    ero_reg_gaia_feats = ero_rebal_gaia_feats = None
    if ero_reg_sids is not None and ero_npz_regular.exists():
        ero_reg_gaia_feats = _load_gaia_features_from_npz(ero_npz_regular, ero_reg_sids)
    if ero_rebal_sids is not None and ero_npz_rebal.exists():
        ero_rebal_gaia_feats = _load_gaia_features_from_npz(ero_npz_rebal, ero_rebal_sids)

    gaia_xgb_gaia_feats = gaia_hilat_gaia_feats = None
    if gaia_xgb_emb is not None:
        _gxgb_feat_cols = [c for c in SELECTED_GAIA_PARAMETERS if c in df_gxgb.columns]
        if len(_gxgb_feat_cols) == len(SELECTED_GAIA_PARAMETERS):
            gaia_xgb_gaia_feats = df_gxgb[list(SELECTED_GAIA_PARAMETERS)].values.astype(float)
            if len(gaia_xgb_gaia_feats) > _gxgb_cap:
                gaia_xgb_gaia_feats = gaia_xgb_gaia_feats[_gi]
            print(f"  Gaia XGB features: {len(gaia_xgb_gaia_feats):,} sources, {len(SELECTED_GAIA_PARAMETERS)} features")
    if gaia_hilat_emb is not None and df_hilat is not None:
        _ghl_feat_cols = [c for c in SELECTED_GAIA_PARAMETERS if c in df_hilat.columns]
        if len(_ghl_feat_cols) == len(SELECTED_GAIA_PARAMETERS):
            gaia_hilat_gaia_feats = df_hilat[list(SELECTED_GAIA_PARAMETERS)].values.astype(float)
            if len(gaia_hilat_gaia_feats) > _ghl_cap:
                gaia_hilat_gaia_feats = gaia_hilat_gaia_feats[_ghi]
            print(f"  Gaia hilat features: {len(gaia_hilat_gaia_feats):,} sources, {len(SELECTED_GAIA_PARAMETERS)} features")

    ero_desi_gaia_feats = None
    if ero_desi_emb is not None and ero_desi_npz.exists():
        _ed_npz = np.load(ero_desi_npz, allow_pickle=True)
        _ed_sids_npz = _ed_npz["source_ids"].astype(str) if "source_ids" in _ed_npz else None
        if _ed_sids_npz is not None and "X_features" in _ed_npz:
            _ed_X = _ed_npz["X_features"].astype(float)
            _ed_sid_map = {s: i for i, s in enumerate(_ed_sids_npz)}
            _ed_idx = np.array([_ed_sid_map.get(str(s), -1) for s in ero_desi_sids])
            ero_desi_gaia_feats = np.full((len(ero_desi_sids), _ed_X.shape[1]), np.nan)
            _ed_valid = _ed_idx >= 0
            ero_desi_gaia_feats[_ed_valid] = _ed_X[_ed_idx[_ed_valid]]
            print(f"  ERO-DESI features: {int(_ed_valid.sum()):,} / {len(ero_desi_sids):,} matched")

    # ── Step K: ERO feature mapping ───────────────────────────────────────────
    _ero_feat_ranges: dict = {}
    if ero_regular_emb is not None and ero_regular_features is not None and ero_feature_names:
        q1_features, feature_names, _ero_feat_ranges = _nn_map_ero_features_onto_q1(
            q1_emb, ero_regular_emb, ero_regular_features,
            ero_feature_names, q1_features, feature_names or [],
        )

    # ── Step L: Q1 Z arrays + XGB P(gal) + ERO P(gal) ────────────────────────
    print("\n[5b] Loading Q1 Z arrays (model P(gal) + PHZ P(gal))...")
    q1_model_pgal = q1_phz_pgal = None
    if q1_sids is not None and q1_features_base is not None:
        q1_model_pgal, q1_phz_pgal = _load_q1_z_arrays(
            q1_sids, q1_features_base)
        # Fill PHZ pgal gaps from Q1 cache
        if q1_phz_pgal is not None:
            sid_to_phz_pgal = dict(zip(q1_sids, q1_phz_pgal))
            _fill_col(df_allphz, "phz_pgal", sid_to_phz_pgal)
            _fill_col(df_desi,   "phz_pgal", sid_to_phz_pgal)

    # XGB inference on all PHZ/DESI sources for 100% model_pgal coverage
    print("\n[5c] Computing XGB P(gal) for all overlay sources...")
    if df_allphz is not None:
        df_allphz["model_pgal"] = _compute_model_pgal_for_df(df_allphz)
    if df_desi is not None:
        df_desi["model_pgal"] = _compute_model_pgal_for_df(df_desi)

    print("\n[5d] Loading ERO model P(gal)...")
    ero_reg_pgal   = _load_ero_pgal(ero_reg_sids)   if ero_reg_sids   is not None else None
    ero_rebal_pgal = _load_ero_pgal(ero_rebal_sids) if ero_rebal_sids is not None else None

    # ── Step M: Density histograms ────────────────────────────────────────────
    print("\n[5e] Building per-category density histograms...")
    _density_hist = _build_density_hists(
        q1_emb, _df_allphz_full, _df_desi_full,
        _ero_reg_full_emb, _ero_reg_full_sids,
        _ero_rebal_full_emb, _ero_rebal_full_sids,
        sid_to_group,
        _gaia_xgb_full_emb, _gaia_xgb_full_gprob,
        _gaia_hilat_full_emb, _gaia_hilat_full_gprob,
        ero_desi_emb=ero_desi_emb, ero_desi_cat=ero_desi_cat,
        gal_thresh=GAL_THRESH,
    )

    # ── Step O: Sky image + sky data per dataset ───────────────────────────────
    print("\n[6] Loading sky image and preparing sky data...")
    sky_path = ensure_sky_image()
    sky_b64 = None
    if sky_path and sky_path.exists():
        sky_b64 = base64.b64encode(sky_path.read_bytes()).decode("ascii")
        print(f"  Sky image: {len(sky_b64) / 1e6:.1f} MB base64")
    else:
        print("  No sky image available — sky panel will show dark background")

    # Helper: build feature_cols dict for a DataFrame with Gaia feature columns
    def _feat_cols_from_df(df):
        cols = {}
        # Galaxy probability: try gal_prob, model_pgal, phz_pgal
        for pgal_col in ("gal_prob", "model_pgal", "phz_pgal"):
            if pgal_col in df.columns:
                cols["gal_prob"] = df[pgal_col].values.astype(float)
                break
        if "phot_g_mean_mag" in df.columns:
            cols["phot_g_mean_mag"] = df["phot_g_mean_mag"].values.astype(float)
        for fn in SELECTED_GAIA_PARAMETERS:
            if fn in df.columns:
                cols[fn] = df[fn].values.astype(float)
        return cols

    sky_datasets = {}

    # [a] Gaia Random 50k
    if GAIA_XGB_CSV.exists():
        df_sky_gaia = pd.read_csv(GAIA_XGB_CSV)
        _g_prob = df_sky_gaia["gal_prob"].values.astype(float)
        legend_c = [("#91D3F0" if p >= GAL_THRESH else "#F2BDCE") for p in _g_prob]
        _gaia_cats = [("gal" if p >= GAL_THRESH else "star") for p in _g_prob]
        _gaia_cl = assign_to_clusters(
            df_sky_gaia[["umap1", "umap2"]].values, q1_emb, q1_labels)
        sky_datasets["gaia_random"] = _prepare_sky_data(
            df_sky_gaia["ra"].values, df_sky_gaia["dec"].values,
            df_sky_gaia["umap1"].values, df_sky_gaia["umap2"].values,
            df_sky_gaia["source_id"].astype(str).values,
            feature_cols=_feat_cols_from_df(df_sky_gaia),
            legend_colors=legend_c,
            categories=_gaia_cats,
            clusters=_gaia_cl,
        )
        print(f"  Gaia Random: {len(df_sky_gaia):,} sources")

    # [b] Gaia High-lat 50k
    if GAIA_HILAT_CSV.exists():
        df_sky_hilat = pd.read_csv(GAIA_HILAT_CSV)
        _g_prob_hl = df_sky_hilat["gal_prob"].values.astype(float)
        legend_c_hl = [("#91D3F0" if p >= GAL_THRESH else "#F2BDCE") for p in _g_prob_hl]
        _hilat_cats = [("gal" if p >= GAL_THRESH else "star") for p in _g_prob_hl]
        _hilat_cl = assign_to_clusters(
            df_sky_hilat[["umap1", "umap2"]].values, q1_emb, q1_labels)
        sky_datasets["gaia_hilat"] = _prepare_sky_data(
            df_sky_hilat["ra"].values, df_sky_hilat["dec"].values,
            df_sky_hilat["umap1"].values, df_sky_hilat["umap2"].values,
            df_sky_hilat["source_id"].astype(str).values,
            feature_cols=_feat_cols_from_df(df_sky_hilat),
            legend_colors=legend_c_hl,
            categories=_hilat_cats,
            clusters=_hilat_cl,
        )
        print(f"  Gaia Hilat: {len(df_sky_hilat):,} sources")

    # [c] Q1 (All PHZ)
    if df_allphz is not None and "ra" in df_allphz.columns:
        _phz_vals = df_allphz["phz"].values.astype(int)
        _phz_color_map = {int(v): c for v, (_, c, _) in PHZ_LABELS.items()}
        legend_c_phz = [_phz_color_map.get(int(v), "#888888") for v in _phz_vals]
        _q1_clusters = df_allphz["cluster"].values.astype(int) if "cluster" in df_allphz.columns else None
        sky_datasets["q1"] = _prepare_sky_data(
            df_allphz["ra"].values, df_allphz["dec"].values,
            df_allphz["umap1"].values, df_allphz["umap2"].values,
            df_allphz["source_id"].astype(str).values,
            feature_cols=_feat_cols_from_df(df_allphz),
            legend_colors=legend_c_phz,
            categories=[str(v) for v in _phz_vals],
            clusters=_q1_clusters,
        )
        print(f"  Q1 (allphz): {len(df_allphz):,} sources")

    # [d] Euclid x DESI (combined Q1-DESI + ERO-DESI)
    _desi_parts = []
    _desi_cat_parts = []
    _desi_cl_parts = []
    if df_desi is not None and "ra" in df_desi.columns:
        _phz_vals_d = df_desi["phz"].values.astype(int)
        _desi_lc = [_phz_color_map.get(int(v), "#888888") for v in _phz_vals_d]
        _desi_cat_parts.extend([str(v) for v in _phz_vals_d])
        _desi_cl_d = df_desi["cluster"].values.astype(int) if "cluster" in df_desi.columns else None
        if _desi_cl_d is not None:
            _desi_cl_parts.extend(_desi_cl_d.tolist())
        _desi_parts.append((df_desi, _desi_lc))
    if ero_desi_ra is not None and ero_desi_emb is not None:
        # Build a mini-DataFrame for ERO-DESI
        _ed_df = pd.DataFrame({
            "ra": ero_desi_ra, "dec": ero_desi_dec,
            "umap1": ero_desi_emb[:, 0], "umap2": ero_desi_emb[:, 1],
            "source_id": ero_desi_sids if ero_desi_sids is not None else [""] * len(ero_desi_ra),
        })
        if ero_desi_gmag is not None:
            _ed_df["phot_g_mean_mag"] = ero_desi_gmag
        if ero_desi_model_pgal is not None:
            _ed_df["gal_prob"] = ero_desi_model_pgal
        _ed_cat_vals = ero_desi_cat
        _ed_color_map = {2: "#9575cd", 3: "#66bb6a"}
        _ed_lc = [_ed_color_map.get(int(c), "#888888") for c in _ed_cat_vals]
        # ERO-DESI categories use "ed_<cat>" prefix to distinguish from PHZ
        _desi_cat_parts.extend([f"ed_{int(c)}" for c in _ed_cat_vals])
        _ed_cl = assign_to_clusters(ero_desi_emb, q1_emb, q1_labels)
        _desi_cl_parts.extend(_ed_cl.tolist())
        _desi_parts.append((_ed_df, _ed_lc))
    if _desi_parts:
        _all_dfs = [p[0] for p in _desi_parts]
        _all_lc = []
        for _, lc in _desi_parts:
            _all_lc.extend(lc)
        _combined = pd.concat(_all_dfs, ignore_index=True)
        sky_datasets["euclid_desi"] = _prepare_sky_data(
            _combined["ra"].values, _combined["dec"].values,
            _combined["umap1"].values, _combined["umap2"].values,
            _combined["source_id"].astype(str).values,
            feature_cols=_feat_cols_from_df(_combined),
            legend_colors=_all_lc,
            categories=_desi_cat_parts if _desi_cat_parts else None,
            clusters=_desi_cl_parts if _desi_cl_parts else None,
        )
        print(f"  Euclid x DESI: {len(_combined):,} sources")

    # [e] ERO regular
    _ero_reg_extra_data = ero_reg_extra or {}
    if ero_regular_emb is not None and "ra" in _ero_reg_extra_data:
        _ero_ra = _ero_reg_extra_data["ra"]
        _ero_dec = _ero_reg_extra_data["dec"]
        _ero_sids = ero_reg_sids if ero_reg_sids is not None else np.array([""] * len(ero_regular_emb))
        # Legend color + category from ERO group
        _ero_group_colors = {g: color for g, (_, color) in ERO_GROUP_CONFIGS.items()}
        _ero_lc = []
        _ero_cats = []
        for sid in _ero_sids:
            grp = sid_to_group.get(str(sid), "") if sid_to_group else ""
            _ero_lc.append(_ero_group_colors.get(grp, "#888888"))
            _ero_cats.append(grp)
        _ero_reg_cl_sky = assign_to_clusters(ero_regular_emb, q1_emb, q1_labels)
        _feat_cols_ero = {}
        if ero_regular_gmag is not None:
            _feat_cols_ero["phot_g_mean_mag"] = ero_regular_gmag
        if ero_reg_pgal is not None:
            _feat_cols_ero["gal_prob"] = ero_reg_pgal
        sky_datasets["ero"] = _prepare_sky_data(
            _ero_ra, _ero_dec,
            ero_regular_emb[:, 0], ero_regular_emb[:, 1],
            _ero_sids,
            feature_cols=_feat_cols_ero,
            legend_colors=_ero_lc,
            categories=_ero_cats,
            clusters=_ero_reg_cl_sky,
        )
        print(f"  ERO: {len(ero_regular_emb):,} sources")

    # [f] ERO Rebalanced
    _ero_rebal_extra_data = ero_rebal_extra or {}
    if ero_rebal_emb is not None and "ra" in _ero_rebal_extra_data:
        _eror_ra = _ero_rebal_extra_data["ra"]
        _eror_dec = _ero_rebal_extra_data["dec"]
        _eror_sids = ero_rebal_sids if ero_rebal_sids is not None else np.array([""] * len(ero_rebal_emb))
        _eror_lc = []
        _eror_cats = []
        for sid in _eror_sids:
            grp = sid_to_group.get(str(sid), "") if sid_to_group else ""
            _eror_lc.append(_ero_group_colors.get(grp, "#888888"))
            _eror_cats.append(grp)
        _ero_rebal_cl_sky = assign_to_clusters(ero_rebal_emb, q1_emb, q1_labels)
        _feat_cols_eror = {}
        if ero_rebal_gmag is not None:
            _feat_cols_eror["phot_g_mean_mag"] = ero_rebal_gmag
        if ero_rebal_pgal is not None:
            _feat_cols_eror["gal_prob"] = ero_rebal_pgal
        sky_datasets["ero_rebal"] = _prepare_sky_data(
            _eror_ra, _eror_dec,
            ero_rebal_emb[:, 0], ero_rebal_emb[:, 1],
            _eror_sids,
            feature_cols=_feat_cols_eror,
            legend_colors=_eror_lc,
            categories=_eror_cats,
            clusters=_ero_rebal_cl_sky,
        )
        print(f"  ERO Rebal: {len(ero_rebal_emb):,} sources")

    sky_datasets_json = json.dumps(sky_datasets)
    q1_fields_js = json.dumps(_prepare_fields(Q1_FIELDS))
    ero_fields_js = json.dumps(_prepare_fields(ERO_FIELDS))
    sky_img_src = f"data:image/jpeg;base64,{sky_b64}" if sky_b64 else ""
    print(f"  Sky datasets: {list(sky_datasets.keys())}")

    # ── Step P: Generate HTML ─────────────────────────────────────────────────
    print("\n[7] Generating combined interactive HTML...")
    html = plot_umap_3d_plotly(
        q1_emb, q1_labels, q1_gmag, df_allphz,
        output_dir=OUTPUT_DIR if write_output else None,
        cutout_cache=cutout_cache,
        contact_sheet_cache=contact_sheet_cache, sheet_counts=sheet_counts,
        sheet_extra=sheet_extra,
        q1_features=q1_features, feature_names=feature_names,
        feature_ranges_override=_ero_feat_ranges,
        df_desi=df_desi,
        desi_contact_sheet_cache=desi_contact_sheet_cache,
        desi_sheet_counts=desi_sheet_counts,
        desi_sheet_extra=desi_sheet_extra,
        ero_contact_sheet_cache=ero_contact_sheet_cache,
        ero_sheet_counts=ero_sheet_counts_data,
        ero_sheet_extra=ero_sheet_extra,
        ero_regular_emb=ero_regular_emb, ero_regular_y=ero_regular_y,
        ero_regular_gmag=ero_regular_gmag, ero_regular_features=ero_regular_features,
        ero_rebal_emb=ero_rebal_emb, ero_rebal_y=ero_rebal_y,
        ero_rebal_gmag=ero_rebal_gmag, ero_rebal_features=ero_rebal_features,
        ero_feature_names=ero_feature_names,
        ero_reg_sids=ero_reg_sids, ero_rebal_sids=ero_rebal_sids,
        ero_reg_extra=ero_reg_extra, ero_rebal_extra=ero_rebal_extra,
        sid_to_group=sid_to_group,
        ero_cutout_cache=ero_cutout_cache_data,
        gaia_xgb_emb=gaia_xgb_emb, gaia_xgb_gal_prob=gaia_xgb_gal_prob,
        gaia_xgb_sids=gaia_xgb_sids, gaia_xgb_gmag=gaia_xgb_gmag,
        q1_model_pgal=q1_model_pgal, q1_phz_pgal=q1_phz_pgal,
        ero_reg_pgal=ero_reg_pgal, ero_rebal_pgal=ero_rebal_pgal,
        ero_desi_emb=ero_desi_emb, ero_desi_cat=ero_desi_cat,
        ero_desi_model_pgal=ero_desi_model_pgal,
        ero_desi_gmag=ero_desi_gmag,
        ero_desi_spectype=ero_desi_spectype,
        ero_desi_z=ero_desi_z,
        ero_desi_class_star=ero_desi_class_star,
        ero_desi_ra=ero_desi_ra, ero_desi_dec=ero_desi_dec,
        ero_desi_contact_sheet_cache=ero_desi_cs_cache,
        ero_desi_sheet_counts=ero_desi_cs_counts,
        ero_desi_sheet_extra=ero_desi_sheet_extra,
        ero_desi_sids=ero_desi_sids,
        gaia_xgb_ra=gaia_xgb_ra, gaia_xgb_dec=gaia_xgb_dec,
        gaia_hilat_emb=gaia_hilat_emb, gaia_hilat_gal_prob=gaia_hilat_gal_prob,
        gaia_hilat_sids=gaia_hilat_sids, gaia_hilat_gmag=gaia_hilat_gmag,
        gaia_hilat_ra=gaia_hilat_ra, gaia_hilat_dec=gaia_hilat_dec,
        density_hist_data=_density_hist,
        ero_reg_gaia_features=ero_reg_gaia_feats,
        ero_rebal_gaia_features=ero_rebal_gaia_feats,
        gaia_xgb_gaia_features=gaia_xgb_gaia_feats,
        gaia_hilat_gaia_features=gaia_hilat_gaia_feats,
        ero_desi_gaia_features=ero_desi_gaia_feats,
        # Sky data
        sky_datasets_json=sky_datasets_json,
        sky_img_src=sky_img_src,
        q1_fields_js=q1_fields_js,
        ero_fields_js=ero_fields_js,
    )

    if write_output:
        print(f"\nDone. Output: {OUTPUT_DIR / 'umap_combined.html'}")
    return html


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combined Sky + 3D UMAP + Sidebar visualization"
    )
    parser.add_argument("--reproject", action="store_true",
                        help="Force re-projection from raw npz files even if projected CSVs exist.")
    parser.add_argument("--max-points", type=int, default=50000,
                        help="Max total overlay points for All PHZ (default: 50000). "
                             "Classes with <500 sources are kept in full; "
                             "remaining budget is split proportionally.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        build_combined_html(
            reproject=args.reproject,
            max_points=args.max_points,
            seed=args.seed,
            write_output=True,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
