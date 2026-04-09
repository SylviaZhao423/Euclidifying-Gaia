#!/usr/bin/env python3
"""
Interactive 3D UMAP HTML — Q1 base + overlays (All PHZ / DESI / ERO / ERO Rebalanced)
+ grouped colour dropdown (Gaia / Euclid Q1 / Euclid ERO).

Overlay dropdown
  All PHZ          — all phz classification sources projected into Q1 UMAP
  DESI             — DESI-confirmed galaxies
  ERO              — ERO sources (ero_indiv.npz) projected into Q1 UMAP
  ERO Rebalanced   — ERO rebalanced (ero_indiv_rebalanced.npz) projected

Colour Q1 dropdown
  clusters         — Q1 cluster colours (default)
  ── Gaia ──       — SELECTED_GAIA_PARAMETERS (colours Q1 base)
  Euclid Q1        — ellipticity, flux_vis_psf, … (colours Q1 base)
  Euclid ERO       — e_v_class_star, e_v_ellipticity, … (colours active ERO overlay)

When an ERO overlay is active:
  • Q1 base is greyed out automatically
  • Selecting an "Euclid ERO" colour option recolours the ERO overlay points
  • Switching back to a non-ERO overlay restores Q1 colouring

Reads:
  output/umap/arrays/q1_indiv.npz
  output/umap/arrays/q1_allphz.npz
  output/umap/arrays/q1_allphz_desi.npz
  output/umap/arrays/q1_indiv_reducer.joblib
  output/umap/arrays/q1_cluster_labels.npy
  output/umap/arrays/q1_features.csv              (from analyze_umap_q1_q1_features.py)
  output/umap/arrays/ero_indiv.npz
  output/umap/arrays/ero_indiv_rebalanced.npz
  output/umap/arrays/ero_features.csv              (from analyze_umap_q1_ero_features.py)
  output/umap/arrays/xgb_gaia_projected.csv        (from analyze_umap_gaia_xgb.py)
  output/umap/arrays/ero_desi.npz
  output/umap/arrays/projected/*.csv               (cached, from previous default-mode run)

Writes:
  output/umap/arrays/projected/q1_allphz_projected.csv
  output/umap/arrays/projected/q1_desi_projected.csv
  output/umap/arrays/projected/ero_regular_projected.csv
  output/umap/arrays/projected/ero_rebal_projected.csv
  output/umap/arrays/projected/ero_desi_projected.csv
  output/umap/q1_individual/q1_3d.html

Usage:
  python src/analyze_umap_q1_3d.py
  python src/analyze_umap_q1_3d.py --3d
  python src/analyze_umap_q1_3d.py --3d --max-points 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
     1: ("Star",               "cyan",     "circle"),
     2: ("Galaxy",             "#F45347",  "circle"),
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

DESI_CUTOUTS_DIR = CUTOUTS_DIR / "desi"

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
     1: ("Star",              "cyan",     "circle"),
     2: ("Galaxy (agree)",    "#F45347",  "circle"),
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
    "agree_both_star":          ("Agree: Q1=star,  ERO=star",   "#F45347"),
    "agree_both_galaxy":        ("Agree: Q1=gal,   ERO=gal",    "cyan"),
    "disagree_ero_star_q1_gal": ("Disagree: Q1=gal,  ERO=star", "#FFB0C8"),
    "disagree_ero_gal_q1_star": ("Disagree: Q1=star, ERO=gal",  "#90CAF9"),
}

ERO_DESI_LABELS = {
    2: ("Disagree: ERO=star, DESI=gal",   "#CE93D8"),   # purple
    3: ("Agree: ERO=gal, DESI=gal",       "#A5D6A7"),   # green
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

STAR_THRESH = 0.5   # XGBoost star probability threshold


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
    emb      = reducer.transform(X_scaled)
    return emb, y, sids


# ── 3D Plotly HTML ────────────────────────────────────────────────────────────

def _build_ero_traces(emb, y, gmag, features, feature_names, prefix, traces,
                      sids=None, sid_to_group=None, label_prefix="",
                      cluster_labels=None, q1_cluster_ids=None,
                      model_pgal=None,
                      ero_ra=None, ero_dec=None, ero_class_star=None):
    """Add ERO overlay traces as per-cluster sub-traces + legend-only dummy traces.

    Layout appended to `traces`:
      [0 .. n_dummy-1]  : dummy legend traces (showlegend=True, [None] coords)
      [n_dummy .. end]  : sub-traces per group×cluster (showlegend=False,
                          named ``ero_{prefix}_grp={group} C{k}``)

    Returns (added_groups, n_dummy, n_sub, sub_z_gmag, sub_z_model_pgal).
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
            marker=dict(size=5, color=color, opacity=0.95,
                        line=dict(width=0.2, color="rgba(255,255,255,0.25)")),
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
                marker=dict(size=5, color=color, opacity=0.95,
                            line=dict(width=0.2, color="rgba(255,255,255,0.25)")),
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
            n_sub += 1

    return added_groups, n_dummy, n_sub, sub_z_gmag, sub_z_model_pgal


def plot_umap_3d_plotly(
    q1_emb, q1_labels, q1_gmag,
    df_allphz,
    output_dir,
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
    gaia_xgb_emb=None, gaia_xgb_star_prob=None,
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
            hovertemplate=_q1_ht,
        ))
        _q1_z_gmag.append(z_noise.tolist())
        _m = _q1_z_model(noise_mask); _q1_z_model_pgal.append(_m if _m is not None else z_noise.tolist())
        _p = _q1_z_phz(noise_mask);   _q1_z_phz_pgal.append(_p if _p is not None else z_noise.tolist())
    for k in cluster_ids:
        m   = q1_labels == k
        z_k = np.zeros(m.sum())
        traces.append(go.Scatter3d(
            x=q1_emb[m, 0], y=q1_emb[m, 1], z=z_k,
            mode="markers",
            marker=dict(size=1, color=CLUSTER_COLORS[k % len(CLUSTER_COLORS)], opacity=0.15),
            name=f"Cluster {k}", customdata=_feat_cd(m),
            hovertemplate=_q1_ht,
        ))
        _q1_z_gmag.append(z_k.tolist())
        _m = _q1_z_model(m); _q1_z_model_pgal.append(_m if _m is not None else z_k.tolist())
        _p = _q1_z_phz(m);   _q1_z_phz_pgal.append(_p if _p is not None else z_k.tolist())

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
        mkr = dict(size=5, color=col, symbol=sym, opacity=0.95,
                   line=dict(width=0.2, color="rgba(255,255,255,0.25)"))
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

    # ── DESI overlay ──────────────────────────────────────────────────────────
    n_pre_desi = len(traces)
    for v, (lbl, col, sym) in DESI_CATEGORY_LABELS.items():
        m_v = desi_arr == v
        if not m_v.any(): continue
        mkr = dict(size=5, color=col, symbol=sym, opacity=0.95,
                   line=dict(width=0.2, color="rgba(255,255,255,0.25)"))
        traces.append(go.Scatter3d(
            x=[float("nan")], y=[float("nan")], z=[float("nan")],
            mode="markers", marker=mkr, name=f"phz={v} {lbl}",
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
        reg_groups, n_ero_reg_dummy, n_ero_reg_sub, _ero_reg_z_gmag, _ero_reg_z_model_pgal = \
            _build_ero_traces(
                ero_regular_emb, ero_regular_y, ero_regular_gmag,
                ero_regular_features, ero_feature_names, "ero", traces,
                sids=ero_reg_sids, sid_to_group=sid_to_group,
                cluster_labels=_ero_reg_cl, q1_cluster_ids=_q1_cluster_ids,
                model_pgal=ero_reg_pgal,
                ero_ra=_ero_reg_extra.get("ra"),
                ero_dec=_ero_reg_extra.get("dec"),
                ero_class_star=_ero_reg_extra.get("class_star"),
            )
        for offset, grp in enumerate(reg_groups):
            ero_trace_groups[str(n_pre_ero_regular + offset)] = grp
    n_ero_reg_sub_start = n_pre_ero_regular + n_ero_reg_dummy

    n_pre_ero_rebal = len(traces)
    n_ero_rebal_dummy = 0
    n_ero_rebal_sub   = 0
    if ero_rebal_emb is not None and ero_rebal_y is not None:
        rebal_groups, n_ero_rebal_dummy, n_ero_rebal_sub, _ero_rebal_z_gmag, _ero_rebal_z_model_pgal = \
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
            )
        for offset, grp in enumerate(rebal_groups):
            ero_trace_groups[str(n_pre_ero_rebal + offset)] = grp
    n_ero_rebal_sub_start = n_pre_ero_rebal + n_ero_rebal_dummy

    # ── Gaia XGB overlay (legend + sub-traces per category×cluster) ──────────
    n_pre_gaia_xgb     = len(traces)
    n_gaia_xgb_legend  = 0
    _gaia_xgb_cl = (assign_to_clusters(gaia_xgb_emb, q1_emb, q1_labels)
                    if gaia_xgb_emb is not None else None)
    _gaia_xgb_cats: list[str] = []  # category labels for legend traces
    _gaia_cat_data: list[tuple] = []  # (mask, short_name, lbl, color)
    if gaia_xgb_emb is not None and gaia_xgb_star_prob is not None:
        m_star = gaia_xgb_star_prob >= STAR_THRESH
        m_gal  = ~m_star
        gaia_pgal = 1.0 - gaia_xgb_star_prob
        for mask, short, lbl, color in [
            (m_star, "star", f"Gaia XGB Stars  (prob ≥ {STAR_THRESH},  n={int(m_star.sum()):,})", "#EF9A9A"),
            (m_gal,  "gal",  f"Gaia XGB Galaxies  (prob < {STAR_THRESH},  n={int(m_gal.sum()):,})",  "#90CAF9"),
        ]:
            if not mask.any():
                continue
            # Legend trace
            traces.append(go.Scatter3d(
                x=[None], y=[None], z=[None], mode="markers",
                marker=dict(size=5, color=color, opacity=0.95,
                            line=dict(width=0.2, color="rgba(255,255,255,0.25)")),
                name=lbl, showlegend=True, visible=False,
            ))
            _gaia_xgb_cats.append(short)
            _gaia_cat_data.append((mask, short, lbl, color))
            n_gaia_xgb_legend += 1

    # Sub-traces per category×cluster
    n_pre_gaia_xgb_sub = len(traces)
    n_gaia_xgb_sub     = 0
    if gaia_xgb_emb is not None and gaia_xgb_star_prob is not None:
        gaia_pgal = 1.0 - gaia_xgb_star_prob
        for mask, short, lbl, color in _gaia_cat_data:
            for k in _q1_cluster_ids:
                mk = mask & (_gaia_xgb_cl == k) if _gaia_xgb_cl is not None else mask
                if not mk.any():
                    continue
                n_pts   = int(mk.sum())
                sids_m  = gaia_xgb_sids[mk] if gaia_xgb_sids is not None else np.array([""] * n_pts)
                probs_m = gaia_xgb_star_prob[mk]
                z_g     = gaia_xgb_gmag[mk] if gaia_xgb_gmag is not None else np.zeros(n_pts)
                ra_m    = gaia_xgb_ra[mk]  if gaia_xgb_ra  is not None else np.full(n_pts, np.nan)
                dec_m   = gaia_xgb_dec[mk] if gaia_xgb_dec is not None else np.full(n_pts, np.nan)
                gm_m    = gaia_xgb_gmag[mk] if gaia_xgb_gmag is not None else np.full(n_pts, np.nan)
                # customdata: [sid, star_prob, ra, dec, gmag]
                cd = [[str(sids_m[i]), round(float(probs_m[i]), 4),
                       round(float(ra_m[i]), 4)  if np.isfinite(ra_m[i])  else None,
                       round(float(dec_m[i]), 4) if np.isfinite(dec_m[i]) else None,
                       round(float(gm_m[i]), 2)  if np.isfinite(gm_m[i])  else None]
                      for i in range(n_pts)]
                traces.append(go.Scatter3d(
                    x=gaia_xgb_emb[mk, 0], y=gaia_xgb_emb[mk, 1], z=z_g,
                    mode="markers",
                    marker=dict(size=5, color=color, opacity=0.95,
                                line=dict(width=0.2, color="rgba(255,255,255,0.25)")),
                    name=f"gxgb_{short} C{k}",
                    hovertemplate=f"RA=%{{customdata[2]:.4f}}<br>Dec=%{{customdata[3]:+.4f}}<br>G=%{{customdata[4]:.2f}}<extra>{lbl.split('(')[0].strip()}</extra>",
                    customdata=cd, showlegend=False, visible=False,
                ))
                _gaia_z_gmag.append(z_g.tolist())
                _gaia_z_model_pgal.append([round(float(v), 4) for v in gaia_pgal[mk]])
                n_gaia_xgb_sub += 1

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
                marker=dict(size=5, color=color, opacity=0.95,
                            line=dict(width=0.2, color="rgba(255,255,255,0.25)")),
                name=f"{lbl}  (n={n_pts:,})",
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
                    marker=dict(size=5, color=color, opacity=0.95,
                                line=dict(width=0.2, color="rgba(255,255,255,0.25)")),
                    name=f"ed_cat={cat_val} C{k}",
                    customdata=cd, showlegend=False, visible=False,
                    hovertemplate=f"RA=%{{customdata[6]:.4f}}<br>Dec=%{{customdata[7]:+.4f}}<br>G=%{{customdata[8]:.2f}}<extra>{lbl}</extra>",
                ))
                _ero_desi_z_gmag.append(z_ed.tolist())
                _ero_desi_z_model_pgal.append(
                    [round(float(v), 4) if np.isfinite(v) else None for v in pgal]
                )
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
                       **dict(autorange=False, range=_zr) if _zr else dict(autorange="reversed")),
        ),
        uirevision="locked",
        legend=dict(bgcolor="rgba(12,12,12,0.9)", bordercolor="#222", borderwidth=1,
                    font=dict(color="#999",size=11), itemsizing="constant",
                    x=1, xanchor="right", y=0.98, yanchor="top"),
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
    ero_desi_colors_json          = json.dumps({str(v): c for v, (_, c) in ERO_DESI_LABELS.items()})
    gaia_xgb_start_json           = json.dumps(n_pre_gaia_xgb)
    gaia_xgb_n_legend_json        = json.dumps(n_gaia_xgb_legend)
    gaia_xgb_sub_start_json       = json.dumps(n_pre_gaia_xgb_sub)
    gaia_xgb_n_sub_json           = json.dumps(n_gaia_xgb_sub)
    gaia_xgb_cats_json            = json.dumps(_gaia_xgb_cats)
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
    # Z-axis arrays (per-trace, in trace order within each group)
    q1_base_z_json     = json.dumps({
        # null when q1_gmag unavailable — JS detects this and uses flat z=0 + autorange.
        "g_mag":      _q1_z_gmag if q1_gmag is not None else None,
        "model_pgal": _q1_z_model_pgal,
        "phz_pgal":   _q1_z_phz_pgal,
    })
    phz_sub_z_json     = json.dumps({"g_mag":       _phz_sub_z_gmag,
                                     "phz_pgal":    _phz_sub_z_phz_pgal,
                                     "model_pgal":  _phz_sub_z_model_pgal})
    ero_reg_z_json     = json.dumps({"g_mag":      _ero_reg_z_gmag,
                                     "model_pgal": _ero_reg_z_model_pgal})
    ero_rebal_z_json   = json.dumps({"g_mag":      _ero_rebal_z_gmag,
                                     "model_pgal": _ero_rebal_z_model_pgal})
    gaia_xgb_z_json    = json.dumps({"g_mag":      _gaia_z_gmag,
                                     "model_pgal": _gaia_z_model_pgal})
    ero_desi_z_json    = json.dumps({"g_mag":      _ero_desi_z_gmag,
                                     "model_pgal": _ero_desi_z_model_pgal})
    gmag_z_range_json  = json.dumps(_zr if _zr else [25, 17])

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
            q1_euclid_optgroup = f'      <optgroup label="Euclid Q1">\n{inner}\n      </optgroup>'
    ero_mapped_optgroup = ""
    if feature_names:
        ero_mapped = [fn for fn in feature_names if fn.startswith("euclid_ero::")]
        if ero_mapped:
            inner = "\n".join(
                f'        <option value="{fn}">{fn.replace("euclid_ero::", "")}</option>'
                for fn in ero_mapped
            )
            ero_mapped_optgroup = f'      <optgroup id="ero-colour-optgroup" label="Euclid ERO (Q1 base, NN-mapped)">\n{inner}\n      </optgroup>'

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Euclid Q1 — UMAP 3D</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  .plotly-notifier {{ display: none !important; }}
  body {{
    background: #080808; color: #bbb;
    display: flex; height: 100vh; overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  }}
  #plot-container {{ flex: 1; min-width: 0; overflow: hidden; background: #080808; }}
  #plot-container .plotly-graph-div {{ width: 100% !important; height: 100% !important; }}
  #resize-handle {{
    width: 5px; flex-shrink: 0; cursor: col-resize;
    background: #111; border-left: 1px solid #1a1a1a;
    display: flex; align-items: center; justify-content: center;
    user-select: none; transition: background 0.15s;
  }}
  #resize-handle:hover, #resize-handle.dragging {{ background: #1e2a3a; }}
  #resize-handle::after {{ content: '⋮'; color: #2a2a2a; font-size: 14px; pointer-events: none; }}
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
  #color-mode-bar {{
    padding: 7px 16px; border-bottom: 1px solid #181818;
    display: flex; align-items: center; gap: 8px; flex-shrink: 0;
  }}
  #color-mode-label {{ font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase; color: #666; white-space: nowrap; }}
  #color-mode-select {{
    flex: 1; background: #111; color: #ccc;
    border: 1px solid #222; border-radius: 3px; font-size: 11px; padding: 3px 6px; cursor: pointer;
  }}
  #color-mode-select:focus {{ outline: none; border-color: #333; }}
  #color-mode-select option {{ background: #111; color: #ccc; }}
  #color-mode-select optgroup {{ background: #111; color: #666; }}
  #overlay-bar {{
    padding: 7px 16px; border-bottom: 1px solid #181818;
    display: flex; align-items: center; gap: 8px; flex-shrink: 0;
  }}
  #overlay-label {{ font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase; color: #666; white-space: nowrap; }}
  #overlay-select, #z-select {{
    flex: 1; background: #111; color: #ccc; border: 1px solid #222;
    border-radius: 3px; padding: 3px 6px; font-size: 11px; cursor: pointer;
  }}
  #overlay-select:focus, #z-select:focus {{ outline: none; border-color: #333; }}
  #overlay-select option, #z-select option {{ background: #111; }}
</style>
</head>
<body>
<div id="plot-container">{plot_div}</div>
<div id="resize-handle"></div>
<div id="sidebar">
  <div id="sidebar-header">
    <div id="sidebar-eyebrow">Euclid Q1 · UMAP</div>
    <div id="sidebar-hint">hover a point · click to pin</div>
  </div>
  <div id="overlay-bar">
    <span id="overlay-label">Overlay</span>
    <select id="overlay-select">
      <option value="allphz">All PHZ</option>
      <option value="desi">Q1 DESI</option>
      <option value="ero_desi">ERO DESI</option>
      <option value="ero">ERO</option>
      <option value="ero_rebalanced">ERO Rebalanced</option>
      <option value="gaia_xgb">Gaia XGB</option>
    </select>
  </div>
  <div id="color-mode-bar">
    <span id="color-mode-label">Z axis</span>
    <select id="z-select">
      <option value="g_mag">G magnitude</option>
      <option value="phz_pgal">PHZ P(gal)</option>
      <option value="model_pgal">XGB P(gal)</option>
    </select>
  </div>
  <div id="color-mode-bar">
    <span id="color-mode-label">Colour</span>
    <select id="color-mode-select">
      <option value="clusters">clusters</option>
      <optgroup label="── Gaia ──">
{gaia_opts}
      </optgroup>
{q1_euclid_optgroup}
{ero_mapped_optgroup}
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
var ERO_DESI_COLORS           = {ero_desi_colors_json};
/* Z-axis arrays (per-trace, in trace order within each group) */
var Q1_BASE_Z   = {q1_base_z_json};
var PHZ_SUB_Z   = {phz_sub_z_json};
var ERO_REG_Z   = {ero_reg_z_json};
var ERO_REBAL_Z = {ero_rebal_z_json};
var GAIA_XGB_Z  = {gaia_xgb_z_json};
var ERO_DESI_Z  = {ero_desi_z_json};
var GMAG_Z_RANGE = {gmag_z_range_json};

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

/* ERO feature names (index into ERO trace customdata rows) */
var eroFeatureNames  = {ero_feature_names_json};
var eroFeatureRanges = {ero_feature_ranges_json};

var CLUSTER_COLORS_JS = {cluster_colors_json};
var N_BASE_TRACES     = {n_base_traces};

var COLORSCALE_RDBU = [
  [0,'rgb(5,48,97)'],[0.25,'rgb(33,102,172)'],
  [0.5,'rgb(247,247,247)'],
  [0.75,'rgb(214,96,77)'],[1,'rgb(103,0,31)']
];

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
var isolatedEroDesi  = null;
var hiddenEroDesi    = {{}};
var activeOverlay   = 'allphz';

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
    sheetTitle.innerHTML = '<span style="color:' + _pc + '">phz=' + v + '</span> — no sheets cached';
    sheetImages.innerHTML = ''; sheetView.style.display = 'block'; return;
  }}
  var keys = Object.keys(clusters).map(Number).sort(function(a,b){{return a-b;}});
  var catCounts = counts[v] || {{}};
  var total = _countTotal(catCounts), html = '';
  var _pc = PHZ_COLORS[String(v)] || '#aaa';
  keys.forEach(function(k) {{
    var n = catCounts[String(k)] !== undefined ? catCounts[String(k)] : '?';
    html += '<div class="sheet-cluster-label">Cluster ' + k + ' (' + n + ' sources)</div>';
    html += '<img data-stype="' + stype + '" data-cat="' + v + '" data-cluster="' + k + '" src="data:image/png;base64,' + clusters[String(k)] + '">';
  }});
  sheetTitle.innerHTML = '<span style="color:' + _pc + '">phz=' + v + '</span><br>' + total + ' sources';
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
  keys.forEach(function(v) {{
    var n = (counts[String(v)]||{{}})[String(k)]; n = (n!==undefined)?n:'?';
    if (typeof n==='number') total += n;
    html += '<div class="sheet-cluster-label">phz=' + v + ' (' + n + ' sources)</div>';
    html += '<img data-stype="' + stype + '" data-cat="' + v + '" data-cluster="' + k + '" src="data:image/png;base64,' + phzs[String(v)] + '">';
  }});
  var _ck3 = CLUSTER_COLORS_JS[k % CLUSTER_COLORS_JS.length] || '#333';
  sheetTitle.innerHTML = '<span style="color:' + _ck3 + '">CLUSTER ' + k + '</span><br>' + total + ' sources';
  sheetImages.innerHTML = html; sheetView.style.display = 'block';
}}
sheetClose.addEventListener('click', function() {{ clearSheetMode(); }});
/* Look up extra pages for a contact sheet cell */
function _getExtraPages(stype, cat, cluster) {{
  var map = {{}};
  if (stype === 'phz') map = sheet_extra_pages;
  else if (stype === 'desi') map = desi_sheet_extra;
  else if (stype === 'ero') map = ero_sheet_extra;
  else if (stype === 'ero_desi') map = ero_desi_sheet_extra;
  return ((map[cat] || {{}})[cluster]) || [];
}}
sheetImages.addEventListener('click', function(e) {{
  if (e.target.tagName !== 'IMG') return;
  var img = e.target;
  var stype = img.getAttribute('data-stype');
  var cat = img.getAttribute('data-cat');
  var cluster = img.getAttribute('data-cluster');
  var src = img.src;
  var b64 = src.slice(src.indexOf(',')+1);
  var extras = (stype && cat && cluster) ? _getExtraPages(stype, cat, cluster) : [];
  if (extras.length === 0) {{
    /* Single page — open image directly */
    try {{
      var bin = atob(b64), arr = new Uint8Array(bin.length);
      for (var i=0;i<bin.length;i++) arr[i]=bin.charCodeAt(i);
      window.open(URL.createObjectURL(new Blob([arr],{{type:'image/png'}})),'_blank');
    }} catch(_) {{ window.open(src,'_blank'); }}
  }} else {{
    /* Multi-page — open HTML with all pages */
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
    var mPhz=nm.match(/^phz=(-?\\d+)/);
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
function _isEROOverlay() {{
  return (activeOverlay==='ero' || activeOverlay==='ero_rebalanced');
}}
function _isERORebal() {{
  return activeOverlay === 'ero_rebalanced';
}}
function _isEroDesiOverlay() {{
  return activeOverlay === 'ero_desi';
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
  }} else {{
    var fi=featureNames.indexOf(feature); if (fi<0) return;
    var rng=featureRanges[feature]||[0,1], colorArrays=[], showscales=[];
    idxs.forEach(function(i,ii) {{
      var cd=plotDiv.data[i].customdata||[];
      colorArrays.push(cd.map(function(row) {{
        var v=row?row[fi]:null;
        return (v!==null&&v!==undefined&&isFinite(v))?v:null;
      }}));
      showscales.push(ii===0);
    }});
    Plotly.restyle(plotDiv, {{
      'marker.color': colorArrays, 'marker.colorscale': [COLORSCALE_RDBU],
      'marker.cmin': [rng[0]], 'marker.cmax': [rng[1]], 'marker.opacity': [0.5],
      'marker.showscale': showscales,
      'marker.colorbar.x': 0.02, 'marker.colorbar.xanchor': 'left',
      'marker.colorbar.len': 0.4, 'marker.colorbar.thickness': 10,
      'marker.colorbar.tickfont': {{'color':'#555','size':8}},
    }}, idxs);
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
    'marker.color': colors, 'marker.opacity': [0.95],
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
    showscales.push(ii===0);
  }});
  Plotly.restyle(plotDiv, {{
    'marker.color': colorArrays, 'marker.colorscale': [COLORSCALE_RDBU],
    'marker.cmin': [rng[0]], 'marker.cmax': [rng[1]],
    'marker.opacity': [0.5],
    'marker.showscale': showscales,
    'marker.colorbar.x': 0.02, 'marker.colorbar.xanchor': 'left',
    'marker.colorbar.len': 0.4, 'marker.colorbar.thickness': 10,
    'marker.colorbar.tickfont': {{'color':'#555','size':8}},
  }}, idxs);
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

/* ── Colour dropdown handler — always colours Q1 base ── */
document.getElementById('color-mode-select').addEventListener('change', function() {{
  restoreERODefaultColors();
  recolorQ1Base(this.value);
  _updateOverlayHoverTemplates(this.value);
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

  /* Always hide all ERO, Gaia XGB, and ERO-DESI traces first; re-enable active set below */
  _allEROIndices().forEach(function(i){{vis[i]=false;}});
  _IDX.gaiaXgbLeg.forEach(function(i){{vis[i]=false;}});
  _IDX.gaiaXgbSub.forEach(function(i){{vis[i]=false;}});
  _IDX.eroDesiLeg.forEach(function(i){{vis[i]=false;}});
  _IDX.eroDesiSub.forEach(function(i){{vis[i]=false;}});

  if (_isEroDesiOverlay()) {{
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
  var skipQ1 = (_isEROOverlay() || _isGaiaXgbOverlay() || _isEroDesiOverlay());
  var q1Hi   = _q1BaseIndices();
  var hoverArr = new Array(n);
  for (var _h = 0; _h < n; _h++) hoverArr[_h] = 'all';
  if (skipQ1) {{ q1Hi.forEach(function(i){{ hoverArr[i] = 'skip'; }}); }}

  Plotly.restyle(plotDiv, {{visible: vis, hoverinfo: hoverArr}});

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
  var zTitle  = zTitles[opt] || 'G magnitude';

  /* Save current camera so the view stays stable across z-axis changes */
  var cam = null;
  if (plotDiv.layout && plotDiv.layout.scene && plotDiv.layout.scene.camera) {{
    cam = JSON.parse(JSON.stringify(plotDiv.layout.scene.camera));
  }}

  var q1HasGmag = (Q1_BASE_Z['g_mag'] !== null);
  var q1Idx     = _q1BaseIndices();
  var isProb    = (opt !== 'g_mag');

  function _q1FlatZ(zVal) {{
    var ref = Q1_BASE_Z['g_mag'] || Q1_BASE_Z['model_pgal'] || Q1_BASE_Z['phz_pgal'];
    if (!ref || q1Idx.length !== ref.length) return null;
    return ref.map(function(a) {{ return a.map(function() {{ return zVal; }}); }});
  }}

  var rIdx = [], rZ = [];

  /* Q1 base z-values:
     - G-mag mode: actual G-mag z if available, otherwise flat at z=0
     - Probability mode: flat sentinel at z=1.45 (above [0,1] data range) */
  if (isProb) {{
    var q1Z = _q1FlatZ(1.45);
    if (q1Z && q1Idx.length === q1Z.length) {{
      q1Idx.forEach(function(i,j) {{ rIdx.push(i); rZ.push(q1Z[j]); }});
    }}
  }} else {{
    /* G-mag mode: use actual G-mag z if available, otherwise flat at z=0 */
    var q1Src = Q1_BASE_Z['g_mag'];
    if (q1Src && q1Idx.length === q1Src.length) {{
      q1Idx.forEach(function(i,j) {{ rIdx.push(i); rZ.push(q1Src[j]); }});
    }} else {{
      var q1Z = _q1FlatZ(0);
      if (q1Z && q1Idx.length === q1Z.length) {{
        q1Idx.forEach(function(i,j) {{ rIdx.push(i); rZ.push(q1Z[j]); }});
      }}
    }}
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

  /* Z axis range:
     - Probability mode: [-0.05, 1.45], non-reversed (0 at bottom, 1 at top, base at 1.45)
     - G-mag + data:     GMAG_Z_RANGE (reversed)
     - G-mag, no data:   autorange reversed (flat at z=0) */
  var relayoutArgs = {{'scene.zaxis.title.text': zTitle}};
  if (isProb) {{
    relayoutArgs['scene.zaxis.range']     = [-0.05, 1.45];
    relayoutArgs['scene.zaxis.autorange'] = false;
  }} else if (q1HasGmag) {{
    relayoutArgs['scene.zaxis.range']     = GMAG_Z_RANGE;
    relayoutArgs['scene.zaxis.autorange'] = false;
  }} else {{
    relayoutArgs['scene.zaxis.autorange'] = 'reversed';
  }}

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
  var noPhz = _isEROOverlay() || _isGaiaXgbOverlay() || _isEroDesiOverlay();
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
  isolatedCluster=null; isolatedPhz=null; hiddenClusters={{}}; hiddenPhz={{}};
  isolatedEROGroup=null; hiddenEROGroups={{}}; isolatedGaiaXgb=null; hiddenGaiaXgb={{}};
  isolatedEroDesi=null; hiddenEroDesi={{}};
  clearSheetMode();
  _updateEROOptgroup();
  _updateZOptions();
  applyVisibility();
  restoreERODefaultColors();
  recolorQ1Base(_selectedFeature());
  applyZAxis(_currentZ);
}}

document.getElementById('overlay-select').addEventListener('change', function() {{
  switchOverlay(this.value);
}});

function restoreAll() {{
  isolatedPhz=null; isolatedCluster=null; hiddenClusters={{}}; hiddenPhz={{}};
  isolatedEROGroup=null; hiddenEROGroups={{}}; isolatedGaiaXgb=null; hiddenGaiaXgb={{}};
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
  var mPhz=traceName.match(/^phz=(-?\\d+)/);
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
    restoreAll(); clearSheetMode();
    Plotly.relayout(plotDiv, {{'scene.camera': {{
      eye:{{x:1.25,y:1.25,z:1.25}}, up:{{x:0,y:0,z:1}}, center:{{x:0,y:0,z:0}}
    }}}});
  }}
}});
plotDiv.addEventListener('dblclick', function(e) {{
  var anyHidden=Object.values(hiddenClusters).some(Boolean)||Object.values(hiddenPhz).some(Boolean)||Object.values(hiddenEROGroups).some(Boolean);
  if(isolatedPhz===null&&isolatedCluster===null&&isolatedEROGroup===null&&isolatedGaiaXgb===null&&isolatedEroDesi===null&&!anyHidden) return;
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
      lines.push('XGB P(gal)=' + (1.0 - parseFloat(prob)).toPrecision(4));
    if (sid) lines.push('id=' + sid);
    showPoint(sid, lines.join('\\n'));
  }} else if (isQ1) {{
    var feature=_selectedFeature(), lines=[nm];
    if (feature!=='clusters' && pt.customdata) {{
      var fi=featureNames.indexOf(feature);
      var raw=(fi>=0)?pt.customdata[fi]:undefined;
      var val=(raw!==null&&raw!==undefined&&isFinite(raw))?parseFloat(raw).toPrecision(4):'N/A';
      lines.push(feature+': '+val);
    }}
    showPoint('', lines.join('\\n'));
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
  }}
}}
plotDiv.on('plotly_hover', function(data) {{ if(pinned)return; showPointFromPt(data.points[0]); }});
plotDiv.on('plotly_unhover', function() {{
  if(pinned)return; img.style.display='none'; infoDiv.textContent=''; noCutout.textContent='';
}});
plotDiv.on('plotly_click', function(data) {{
  if(!data.points||!data.points.length)return;
  showPointFromPt(data.points[0]); pinned=true; pinBtn.textContent='Unpin'; pinBtn.classList.add('pinned');
}});

/* ── Resize ── */
function resizePlot() {{
  var c=document.getElementById('plot-container');
  Plotly.relayout(plotDiv, {{width:c.clientWidth, height:c.clientHeight}});
}}
window.addEventListener('load', resizePlot);
window.addEventListener('resize', resizePlot);
function scaleLegendMarkers() {{
  var s=0.7, lt=plotDiv.querySelectorAll('.legend .traces');
  for (var i=0;i<Math.min(N_BASE_TRACES,lt.length);i++) {{
    var pts=lt[i].querySelector('.legendpoints path'); if(!pts)continue;
    var t=(pts.getAttribute('transform')||'').replace(/\\s*scale\\([^)]*\\)/g,'');
    pts.setAttribute('transform',t+' scale('+s+')');
  }}
}}
plotDiv.on('plotly_afterplot', scaleLegendMarkers);

var isResizing=false;
handle.addEventListener('mousedown',function(e){{
  isResizing=true; handle.classList.add('dragging');
  document.body.style.cursor='col-resize'; document.body.style.userSelect='none'; e.preventDefault();
}});
document.addEventListener('mousemove',function(e){{
  if(!isResizing)return;
  var w=Math.max(160,Math.min(700,window.innerWidth-e.clientX-handle.offsetWidth));
  sidebar.style.width=w+'px'; resizePlot();
}});
document.addEventListener('mouseup',function(){{
  if(!isResizing)return;
  isResizing=false; handle.classList.remove('dragging');
  document.body.style.cursor=''; document.body.style.userSelect='';
}});

/* ── Initial visibility ── */
_initIndices();
_updateEROOptgroup();
_updateZOptions();
applyVisibility();
recolorQ1Base('clusters');
applyZAxis(_currentZ);
</script>
</body>
</html>"""

    out = output_dir / "q1_3d.html"
    out.write_text(html, encoding="utf-8")
    print(f"  -> {out}  ({n_cached} cutouts embedded)")


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
        vmin = round(float(np.percentile(src, 2)),  4)
        vmax = round(float(np.percentile(src, 98)), 4)
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

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive 3D UMAP with Euclid feature coloring and ERO overlays"
    )
    parser.add_argument("--3d", dest="make_3d", action="store_true",
                        help="Regenerate HTML from saved CSVs (no DB query).")
    parser.add_argument("--max-points", type=int, default=500,
                        help="Max points per phz class (default: 500).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
        print(f"ERROR: {labels_path} not found. Run compute_q1_clusters.py first.")
        return 1
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

    if args.make_3d:
        # ── --3d mode ─────────────────────────────────────────────────────────
        if not allphz_path.exists():
            print(f"ERROR: {allphz_path} not found. Run without --3d first.")
            return 1

        print("\n[4] Loading saved allphz projected data...")
        df_allphz = pd.read_csv(allphz_path)
        df_allphz["phz"] = df_allphz["phz"].astype(int)
        if "cluster" not in df_allphz.columns:
            df_allphz["cluster"] = assign_to_clusters(
                df_allphz[["umap1","umap2"]].values, q1_emb, q1_labels)
        print(f"  {len(df_allphz):,} sources")

        df_desi = None
        desi_path = PROJECTED_DIR / "q1_desi_projected.csv"
        if desi_path.exists():
            print("\n[4b] Loading saved DESI projected data...")
            df_desi = pd.read_csv(desi_path)
            df_desi["phz"] = df_desi["phz"].astype(int)
            if "cluster" not in df_desi.columns:
                df_desi["cluster"] = assign_to_clusters(
                    df_desi[["umap1","umap2"]].values, q1_emb, q1_labels)
            print(f"  {len(df_desi):,} DESI sources")

        print("\n[4c] Loading ERO projected data...")
        ero_regular_emb = ero_regular_y = ero_regular_gmag = ero_regular_features = None
        ero_rebal_emb   = ero_rebal_y   = ero_rebal_gmag   = ero_rebal_features   = None
        ero_reg_sids = ero_rebal_sids = None
        ero_feature_names = []

        if ero_reg_path.exists():
            df_ero_reg      = pd.read_csv(ero_reg_path)
            ero_regular_emb = df_ero_reg[["umap1","umap2"]].values
            ero_regular_y   = df_ero_reg["y"].values.astype(int)
            ero_reg_sids    = df_ero_reg["source_id"].astype(str).values
            ero_regular_gmag = df_ero_reg["phot_g_mean_mag"].values.astype(float) \
                               if "phot_g_mean_mag" in df_ero_reg.columns else None
            print(f"  ERO regular: {len(ero_regular_emb):,} sources")
            ero_regular_features, ero_feature_names, _ = load_ero_feats(ero_features_csv, ero_reg_sids)
        else:
            print("  No ero_regular_projected.csv — run without --3d first.")

        if ero_rebal_path.exists():
            df_ero_rebal   = pd.read_csv(ero_rebal_path)
            ero_rebal_emb  = df_ero_rebal[["umap1","umap2"]].values
            ero_rebal_y    = df_ero_rebal["y"].values.astype(int)
            ero_rebal_sids = df_ero_rebal["source_id"].astype(str).values
            ero_rebal_gmag = df_ero_rebal["phot_g_mean_mag"].values.astype(float) \
                             if "phot_g_mean_mag" in df_ero_rebal.columns else None
            print(f"  ERO rebalanced: {len(ero_rebal_emb):,} sources")
            ero_rebal_features, fn2, _ = load_ero_feats(ero_features_csv, ero_rebal_sids)
            if not ero_feature_names:
                ero_feature_names = fn2
            elif ero_rebal_features is not None and fn2 != ero_feature_names:
                ero_rebal_features = _align_features(ero_rebal_features, fn2, ero_feature_names)
        else:
            print("  No ero_rebal_projected.csv — run without --3d first.")

        print("\n[4d] Loading ERO group assignments + extra data...")
        sid_to_group = load_ero_group_map()
        ero_reg_extra = load_ero_extra_data_from_npz(
            ARRAYS_DIR / "ero_indiv.npz", ero_reg_sids) if ero_reg_sids is not None else {}
        ero_rebal_extra = load_ero_extra_data_from_npz(
            ARRAYS_DIR / "ero_indiv_rebalanced.npz", ero_rebal_sids) if ero_rebal_sids is not None else {}

        print("\n[4e] Loading Gaia XGB projected data...")
        gaia_xgb_emb = gaia_xgb_star_prob = gaia_xgb_sids = gaia_xgb_gmag = None
        gaia_xgb_ra = gaia_xgb_dec = None
        if GAIA_XGB_CSV.exists():
            df_gxgb          = pd.read_csv(GAIA_XGB_CSV)
            gaia_xgb_emb     = df_gxgb[["umap1", "umap2"]].values
            gaia_xgb_star_prob = df_gxgb["star_prob"].values.astype(float)
            gaia_xgb_sids    = df_gxgb["source_id"].astype(str).values
            gaia_xgb_gmag    = df_gxgb["phot_g_mean_mag"].values.astype(float) \
                               if "phot_g_mean_mag" in df_gxgb.columns else None
            gaia_xgb_ra      = df_gxgb["ra"].values.astype(float) \
                               if "ra" in df_gxgb.columns else None
            gaia_xgb_dec     = df_gxgb["dec"].values.astype(float) \
                               if "dec" in df_gxgb.columns else None
            print(f"  {len(gaia_xgb_emb):,} Gaia XGB sources  "
                  f"(stars={(gaia_xgb_star_prob >= STAR_THRESH).sum():,}  "
                  f"galaxies={(gaia_xgb_star_prob < STAR_THRESH).sum():,})")
        else:
            print(f"  {GAIA_XGB_CSV.name} not found — Gaia XGB overlay disabled.")

        print("\n[5] Loading cutout thumbnails and contact sheets...")
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

        if args.max_points is not None:
            print(f"\n  Applying --max-points {args.max_points}...")
            cutout_sids = set(cutout_cache.keys()); parts = []
            for v in PHZ_LABELS:
                sub = df_allphz[df_allphz["phz"]==v]
                if not len(sub): continue
                if len(sub) > args.max_points:
                    print(f"  phz={v}:")
                    sub = _subsample_keep_cutouts(sub, cutout_sids, args.max_points, args.seed)
                parts.append(sub)
            df_allphz = pd.concat(parts, ignore_index=True) if parts else df_allphz

        _ero_feat_ranges: dict = {}
        if ero_regular_emb is not None and ero_regular_features is not None and ero_feature_names:
            q1_features, feature_names, _ero_feat_ranges = _nn_map_ero_features_onto_q1(
                q1_emb, ero_regular_emb, ero_regular_features,
                ero_feature_names, q1_features, feature_names or [],
            )

        print("\n[5e] Loading Q1 Z arrays (model P(gal) + PHZ P(gal))...")
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
        print("\n[5e2] Computing XGB P(gal) for all overlay sources...")
        if df_allphz is not None:
            df_allphz["model_pgal"] = _compute_model_pgal_for_df(df_allphz)
        if df_desi is not None:
            df_desi["model_pgal"] = _compute_model_pgal_for_df(df_desi)

        print("\n[5f] Loading ERO model P(gal)...")
        ero_reg_pgal   = _load_ero_pgal(ero_reg_sids)   if ero_reg_sids   is not None else None
        ero_rebal_pgal = _load_ero_pgal(ero_rebal_sids) if ero_rebal_sids is not None else None

        # ── ERO-DESI overlay ──────────────────────────────────────────────────
        print("\n[5g] Loading ERO-DESI data...")
        ero_desi_emb = ero_desi_cat = ero_desi_model_pgal = ero_desi_gmag = None
        ero_desi_spectype = ero_desi_z = ero_desi_class_star = ero_desi_sids = None
        ero_desi_ra = ero_desi_dec = None
        ero_desi_csv = PROJECTED_DIR / "ero_desi_projected.csv"
        ero_desi_npz = ARRAYS_DIR / "ero_desi.npz"
        if ero_desi_csv.exists():
            df_ed = pd.read_csv(ero_desi_csv)
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
            # Project on the fly — load reducer + mean/std
            print("  Projecting ERO-DESI on the fly...")
            _q1_mean = q1["mean"]; _q1_std = q1["std"]
            _std_safe = np.where(_q1_std > 0, _q1_std, 1.0)
            _reducer = joblib.load(ARRAYS_DIR / "q1_indiv_reducer.joblib")
            ed_npz = np.load(ero_desi_npz, allow_pickle=True)
            ed_X   = ed_npz["X_features"].astype(float)
            ero_desi_cat = ed_npz["cat_value"].astype(int)
            ero_desi_gmag = ed_npz["phot_g_mean_mag"].astype(float) \
                            if "phot_g_mean_mag" in ed_npz else None
            ero_desi_spectype = ed_npz["spectype"].astype(str) \
                                if "spectype" in ed_npz else None
            ero_desi_z = ed_npz["desi_z"].astype(float) \
                         if "desi_z" in ed_npz else None
            ero_desi_class_star = ed_npz["class_star"].astype(float) \
                                  if "class_star" in ed_npz else None
            ero_desi_sids = ed_npz["source_ids"].astype(str) \
                            if "source_ids" in ed_npz else None
            ero_desi_ra = ed_npz["ra"].astype(float) \
                          if "ra" in ed_npz else None
            ero_desi_dec = ed_npz["dec"].astype(float) \
                           if "dec" in ed_npz else None
            ed_scaled = (ed_X - _q1_mean) / _std_safe
            ed_scaled = np.where(np.isfinite(ed_scaled), ed_scaled, 0.0)
            ero_desi_emb = _reducer.transform(ed_scaled)
            df_ed_xgb = pd.DataFrame(ed_X, columns=SELECTED_GAIA_PARAMETERS)
            ero_desi_model_pgal = _compute_model_pgal_for_df(df_ed_xgb)
            # Save for next --3d run
            save_dict = {
                "umap1": ero_desi_emb[:, 0], "umap2": ero_desi_emb[:, 1],
                "cat_value": ero_desi_cat, "model_pgal": ero_desi_model_pgal,
            }
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
            pd.DataFrame(save_dict).to_csv(ero_desi_csv, index=False)
            print(f"  {len(ero_desi_cat):,} ERO-DESI sources projected → {ero_desi_csv}")
        else:
            print("  ero_desi.npz not found — ERO-DESI overlay disabled.")

        print("\n[6] Generating 3D HTML...")
        plot_umap_3d_plotly(
            q1_emb, q1_labels, q1_gmag, df_allphz, OUTPUT_DIR,
            cutout_cache=cutout_cache,
            contact_sheet_cache=contact_sheet_cache, sheet_counts=sheet_counts,
            sheet_extra=sheet_extra,
            q1_features=q1_features, feature_names=feature_names,
            feature_ranges_override=_ero_feat_ranges,
            df_desi=df_desi,
            desi_contact_sheet_cache=desi_contact_sheet_cache,
            desi_sheet_counts=desi_sheet_counts,
            desi_sheet_extra=desi_sheet_extra,
            ero_regular_emb=ero_regular_emb, ero_regular_y=ero_regular_y,
            ero_regular_gmag=ero_regular_gmag, ero_regular_features=ero_regular_features,
            ero_rebal_emb=ero_rebal_emb, ero_rebal_y=ero_rebal_y,
            ero_rebal_gmag=ero_rebal_gmag, ero_rebal_features=ero_rebal_features,
            ero_feature_names=ero_feature_names,
            ero_reg_sids=ero_reg_sids, ero_rebal_sids=ero_rebal_sids,
            ero_reg_extra=ero_reg_extra, ero_rebal_extra=ero_rebal_extra,
            sid_to_group=sid_to_group,
            ero_contact_sheet_cache=ero_contact_sheet_cache,
            ero_sheet_counts=ero_sheet_counts_data,
            ero_sheet_extra=ero_sheet_extra,
            ero_cutout_cache=ero_cutout_cache_data,
            gaia_xgb_emb=gaia_xgb_emb, gaia_xgb_star_prob=gaia_xgb_star_prob,
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
        )
        return 0

    # ── Default mode: load npz + project + save ────────────────────────────────
    q1_mean  = q1["mean"]
    q1_std   = q1["std"]
    reducer  = joblib.load(ARRAYS_DIR / "q1_indiv_reducer.joblib")
    std_safe = np.where(q1_std > 0, q1_std, 1.0)

    print("\n[4] Loading all phz sources from npz...")
    df_allphz = load_allphz_from_npz()

    print("\n[4b] Loading DESI sources from npz...")
    df_desi = None
    try:
        df_desi = load_desi_from_npz()
    except Exception as e:
        print(f"  DESI load failed: {e} — DESI overlay disabled.")

    print("\n[5] Projecting allphz into UMAP space...")
    X = df_allphz[SELECTED_GAIA_PARAMETERS].values.astype(float)
    X_scaled  = np.where(np.isfinite((X - q1_mean)/std_safe), (X - q1_mean)/std_safe, 0.0)
    emb       = reducer.transform(X_scaled)
    df_allphz = df_allphz.copy()
    df_allphz["umap1"] = emb[:, 0]; df_allphz["umap2"] = emb[:, 1]
    df_allphz["cluster"] = assign_to_clusters(emb, q1_emb, q1_labels)
    df_allphz.to_csv(allphz_path, index=False)
    print(f"  Saved → {allphz_path}")

    if df_desi is not None:
        print("\n[5b] Projecting DESI sources...")
        X_d        = df_desi[SELECTED_GAIA_PARAMETERS].values.astype(float)
        X_d_scaled = np.where(np.isfinite((X_d-q1_mean)/std_safe), (X_d-q1_mean)/std_safe, 0.0)
        emb_d      = reducer.transform(X_d_scaled)
        df_desi    = df_desi.copy()
        df_desi["umap1"] = emb_d[:, 0]; df_desi["umap2"] = emb_d[:, 1]
        df_desi["cluster"] = assign_to_clusters(emb_d, q1_emb, q1_labels)
        desi_path = PROJECTED_DIR / "q1_desi_projected.csv"
        df_desi.to_csv(desi_path, index=False)
        print(f"  Saved → {desi_path}")

    # ── ERO projection ────────────────────────────────────────────────────────
    ero_regular_emb = ero_regular_y = ero_regular_gmag = ero_regular_features = None
    ero_rebal_emb   = ero_rebal_y   = ero_rebal_gmag   = ero_rebal_features   = None
    ero_reg_sids = ero_rebal_sids = None
    ero_feature_names = []
    _fn_rebal: list[str] = []

    ero_npz_regular = ARRAYS_DIR / "ero_indiv.npz"
    ero_npz_rebal   = ARRAYS_DIR / "ero_indiv_rebalanced.npz"

    for tag, npz_path, out_path in [
        ("ERO regular",     ero_npz_regular, ero_reg_path),
        ("ERO rebalanced",  ero_npz_rebal,   ero_rebal_path),
    ]:
        if not npz_path.exists():
            print(f"\n  {npz_path.name} not found — {tag} overlay disabled.")
            continue
        print(f"\n[5c/{tag}] Projecting {tag} into Q1 UMAP space...")
        emb_ero, y_ero, sids_ero = _project_ero(npz_path, q1_mean, std_safe, reducer)
        print(f"  {len(emb_ero):,} sources  (STAR={(y_ero==0).sum():,}  GAL={(y_ero==1).sum():,})")

        print(f"  Loading ERO VIS features...")
        feats_ero, fn, gmag_ero = load_ero_feats(ero_features_csv, sids_ero)
        if not ero_feature_names:
            ero_feature_names = fn

        df_ero_save = pd.DataFrame({
            "source_id": sids_ero, "y": y_ero,
            "umap1": emb_ero[:,0], "umap2": emb_ero[:,1],
        })
        if gmag_ero is not None:
            df_ero_save["phot_g_mean_mag"] = gmag_ero
        df_ero_save.to_csv(out_path, index=False)
        print(f"  Saved → {out_path}")

        if tag == "ERO regular":
            ero_regular_emb, ero_regular_y, ero_regular_gmag, ero_regular_features = \
                emb_ero, y_ero, gmag_ero, feats_ero
            ero_reg_sids = sids_ero
        else:
            ero_rebal_emb, ero_rebal_y, ero_rebal_gmag, ero_rebal_features = \
                emb_ero, y_ero, gmag_ero, feats_ero
            ero_rebal_sids = sids_ero
            _fn_rebal = fn

    # Ensure rebalanced feature columns match regular (ero_feature_names) ordering
    if (ero_rebal_features is not None and ero_feature_names
            and _fn_rebal != ero_feature_names):
        ero_rebal_features = _align_features(ero_rebal_features, _fn_rebal, ero_feature_names)
        print(f"  Aligned rebal features: {_fn_rebal} → {ero_feature_names}")

    print("\n[5d] Loading ERO group assignments + extra data...")
    sid_to_group = load_ero_group_map()
    ero_reg_extra = load_ero_extra_data_from_npz(
        ARRAYS_DIR / "ero_indiv.npz", ero_reg_sids) if ero_reg_sids is not None else {}
    ero_rebal_extra = load_ero_extra_data_from_npz(
        ARRAYS_DIR / "ero_indiv_rebalanced.npz", ero_rebal_sids) if ero_rebal_sids is not None else {}

    print("\n[6] Loading cutout thumbnails and contact sheets...")
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

    if args.max_points is not None:
        print(f"\n  Applying --max-points {args.max_points}...")
        cutout_sids = set(cutout_cache.keys()); parts = []
        for v in PHZ_LABELS:
            sub = df_allphz[df_allphz["phz"]==v]
            if not len(sub): continue
            if len(sub) > args.max_points:
                print(f"  phz={v}:")
                sub = _subsample_keep_cutouts(sub, cutout_sids, args.max_points, args.seed)
            parts.append(sub)
        df_allphz = pd.concat(parts, ignore_index=True) if parts else df_allphz

    print("\n[6b] Loading Gaia XGB projected data...")
    gaia_xgb_emb = gaia_xgb_star_prob = gaia_xgb_sids = gaia_xgb_gmag = None
    gaia_xgb_ra = gaia_xgb_dec = None
    if GAIA_XGB_CSV.exists():
        df_gxgb            = pd.read_csv(GAIA_XGB_CSV)
        gaia_xgb_emb       = df_gxgb[["umap1", "umap2"]].values
        gaia_xgb_star_prob = df_gxgb["star_prob"].values.astype(float)
        gaia_xgb_sids      = df_gxgb["source_id"].astype(str).values
        gaia_xgb_gmag      = df_gxgb["phot_g_mean_mag"].values.astype(float) \
                             if "phot_g_mean_mag" in df_gxgb.columns else None
        gaia_xgb_ra        = df_gxgb["ra"].values.astype(float) \
                             if "ra" in df_gxgb.columns else None
        gaia_xgb_dec       = df_gxgb["dec"].values.astype(float) \
                             if "dec" in df_gxgb.columns else None
        print(f"  {len(gaia_xgb_emb):,} Gaia XGB sources  "
              f"(stars={(gaia_xgb_star_prob >= STAR_THRESH).sum():,}  "
              f"galaxies={(gaia_xgb_star_prob < STAR_THRESH).sum():,})")
    else:
        print(f"  {GAIA_XGB_CSV.name} not found — Gaia XGB overlay disabled.")

    _ero_feat_ranges: dict = {}
    if ero_regular_emb is not None and ero_regular_features is not None and ero_feature_names:
        q1_features, feature_names, _ero_feat_ranges = _nn_map_ero_features_onto_q1(
            q1_emb, ero_regular_emb, ero_regular_features,
            ero_feature_names, q1_features, feature_names or [],
        )

    print("\n[6c] Computing Q1 Z arrays (model P(gal) + PHZ P(gal))...")
    q1_model_pgal = q1_phz_pgal = None
    if q1_sids is not None and q1_features_base is not None:
        q1_model_pgal, q1_phz_pgal = _load_q1_z_arrays(q1_sids, q1_features_base)
        # Fill PHZ pgal gaps from Q1 cache
        if q1_phz_pgal is not None:
            sid_to_phz_pgal = dict(zip(q1_sids, q1_phz_pgal))
            _fill_col(df_allphz, "phz_pgal", sid_to_phz_pgal)
            _fill_col(df_desi,   "phz_pgal", sid_to_phz_pgal)

    # XGB inference on all PHZ/DESI sources for 100% model_pgal coverage
    print("\n[6c2] Computing XGB P(gal) for all overlay sources...")
    if df_allphz is not None:
        df_allphz["model_pgal"] = _compute_model_pgal_for_df(df_allphz)
    if df_desi is not None:
        df_desi["model_pgal"] = _compute_model_pgal_for_df(df_desi)

    print("\n[6d] Loading ERO model P(gal)...")
    ero_reg_pgal   = _load_ero_pgal(ero_reg_sids)   if ero_reg_sids   is not None else None
    ero_rebal_pgal = _load_ero_pgal(ero_rebal_sids) if ero_rebal_sids is not None else None

    # ── ERO-DESI overlay ─────────────────────────────────────────────────────
    print("\n[6e] Projecting ERO-DESI sources...")
    ero_desi_emb = ero_desi_cat = ero_desi_model_pgal = ero_desi_gmag = None
    ero_desi_spectype = ero_desi_z = ero_desi_class_star = ero_desi_sids = None
    ero_desi_ra = ero_desi_dec = None
    ero_desi_npz_path = ARRAYS_DIR / "ero_desi.npz"
    ero_desi_csv_path = PROJECTED_DIR / "ero_desi_projected.csv"
    if ero_desi_npz_path.exists():
        ed_npz = np.load(ero_desi_npz_path, allow_pickle=True)
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
        ero_desi_emb = reducer.transform(ed_scaled)
        ero_desi_cat = ed_cat
        ero_desi_sids = ed_sids
        # XGB inference for model_pgal
        df_ed_for_xgb = pd.DataFrame(ed_X, columns=SELECTED_GAIA_PARAMETERS)
        ero_desi_model_pgal = _compute_model_pgal_for_df(df_ed_for_xgb)
        # Save to CSV for --3d reuse
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
        pd.DataFrame(save_dict).to_csv(ero_desi_csv_path, index=False)
        print(f"  {len(ed_cat):,} ERO-DESI sources projected  "
              f"(cat=2: {(ed_cat==2).sum():,},  cat=3: {(ed_cat==3).sum():,})")
        print(f"  Saved → {ero_desi_csv_path}")
    else:
        print("  ero_desi.npz not found — ERO-DESI overlay disabled.")

    print("\n[7] Generating interactive 3D HTML...")
    plot_umap_3d_plotly(
        q1_emb, q1_labels, q1_gmag, df_allphz, OUTPUT_DIR,
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
        gaia_xgb_emb=gaia_xgb_emb, gaia_xgb_star_prob=gaia_xgb_star_prob,
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
    )

    print(f"\nDone. Output: {OUTPUT_DIR / 'q1_3d.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
