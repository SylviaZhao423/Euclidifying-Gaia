#!/usr/bin/env python3
"""Legacy/DECaLS cutouts of the four inter-cluster BRIDGES in the DESI+SDSS QSO×Gaia
sample (C0–C1, C1–C3, C2–C3, C2–C4), in the standard baked-title / P(gal)-border
contact-sheet format.

Bridge membership is computed exactly as in the umap_qso_overlay_desisdss.png stream
panel — outside_mask (sparse Q1 reference) + stream_labels (nearest named corridor),
both from the shared umap_streams.py (the single bridge definition), so the labels stay
in sync with the overlay.

Outputs (output/qso_cutouts/bridges/):
    {bridge}/cutout_NNNN.png        per-bridge individual Legacy stamps (P(gal)-sorted)
    contact_sheet_bridge{a}-{b}.png one sheet per bridge (a,b = endpoint clusters)
    manifest.csv

Usage:
    conda run -n base python3 src/download_ds_g_bridges_cutouts.py --n 40
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_qso_ds_g_cutouts import (load_projected, MERGED_CSV, fetch_ls_jpg,   # noqa: E402
                                       render_to_array, _place_block, _finish_sheet)
from analyze_umap_qso import ARRAYS_DIR  # noqa: E402
from umap_streams import STREAM_PAIRS, outside_mask, stream_labels  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_DIR / "output" / "qso_cutouts" / "ds_g_bridges"


def load_desisdss() -> pd.DataFrame:
    # the combined DESI+SDSS projection written by analyze_umap_qso.py --catalog both
    csv = MERGED_CSV
    if not csv.exists():
        raise SystemExit(f"{csv.name} not found — run analyze_umap_qso.py (default / --catalog both) first.")
    return load_projected(csv)


def assign_bridges(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each QSO with its bridge index (0..len(STREAM_PAIRS)-1), or -1 (other) /
    -2 (core), exactly as the overlay stream panel does."""
    q1 = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    emb = q1["embedding"]
    labels = np.load(ARRAYS_DIR / "q1_cluster_labels.npy")
    ids = sorted(int(c) for c in set(labels[labels >= 0]))
    centroids = {c: (emb[labels == c, 0].mean(), emb[labels == c, 1].mean()) for c in ids}
    pts = df[["umap1", "umap2"]].to_numpy()
    outside = outside_mask(pts, emb)
    df = df.copy()
    df["bridge"] = stream_labels(pts, outside, centroids)
    return df


def sample_bridge(sub: pd.DataFrame, n: int, gmax: float, rng) -> pd.DataFrame:
    """Up to n members, drawn from the brighter half (G<gmax where possible) at random
    for a representative, well-imaged sheet (same policy as the per-cluster script)."""
    bright = sub[sub["phot_g_mean_mag"] < gmax]
    pool = bright if len(bright) >= n else sub
    take = min(n, len(pool))
    return pool.iloc[rng.choice(len(pool), take, replace=False)]


def bridge_contact_sheet(name: str, items, out: Path) -> None:
    """One-bridge sheet: stamps sorted by descending P(gal), P(gal)-coloured border.
    Same single-block construction as the per-cluster contact sheets."""
    items = sorted(items, key=lambda t: -t[1])
    if not items:
        return
    ncols = 8
    nrows = math.ceil(len(items) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    _place_block(axes, items, 0, ncols)
    _finish_sheet(fig, out, f"{name} bridge  ({len(items)} QSOs, Legacy/DECaLS, "
                  f"sorted by P(gal))", None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40, help="max cutouts per bridge (default 40)")
    ap.add_argument("--size", type=float, default=12.0, help="cutout FOV arcsec (default 12)")
    ap.add_argument("--gmax", type=float, default=20.8, help="prefer sources brighter than this G")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = assign_bridges(load_desisdss())
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    manifest = []

    for k, (name, _verts, _col) in enumerate(STREAM_PAIRS):
        sub = df[df["bridge"] == k]
        safe = name.replace("–", "-")                       # en-dash → ascii for folder names
        print(f"=== {name} bridge: {len(sub):,} members ===")
        if not len(sub):
            print("  (none)"); continue
        rows = sample_bridge(sub, args.n, args.gmax, rng).sort_values(
            "model_pgal", ascending=False).reset_index(drop=True)
        bdir = OUT_ROOT / safe
        bdir.mkdir(parents=True, exist_ok=True)
        for old in bdir.glob("*.png"):                      # clear stale cutouts
            old.unlink()
        items = []
        print(f"  {len(rows)} cutouts...")
        for i, (_, row) in enumerate(rows.iterrows()):
            sid = int(row["sid_str"])                       # exact int64 id (row is float-upcast)
            img = fetch_ls_jpg(float(row["qso_ra"]), float(row["qso_dec"]), args.size)
            arr = render_to_array(row, img, args.size, f"{name} bridge")   # baked title, no border
            fname = f"cutout_{i:04d}.png"                   # numbered by P(gal)
            plt.imsave(bdir / fname, np.clip(arr, 0, 1))
            items.append((arr, float(row["model_pgal"])))
            manifest.append({
                "cutout_file": f"{safe}/{fname}", "bridge": name, "gaia_source_id": sid,
                "qso_id": int(row["qso_id"]), "qso_catalog": row.get("qso_catalog"),
                "qso_ra": row["qso_ra"], "qso_dec": row["qso_dec"], "qso_z": row["qso_z"],
                "phot_g_mean_mag": row["phot_g_mean_mag"], "model_pgal": row["model_pgal"],
                "cluster": int(row["cluster"]), "umap1": row["umap1"], "umap2": row["umap2"],
                "has_image": img is not None,
            })
        a, b = _verts[0], _verts[-1]          # endpoint cluster ids → sheet name
        bridge_contact_sheet(name, items, OUT_ROOT / f"contact_sheet_bridge{a}-{b}.png")

    man = pd.DataFrame(manifest)
    man.to_csv(OUT_ROOT / "manifest.csv", index=False)
    print(f"\n  -> {len(STREAM_PAIRS)} per-bridge sheets (contact_sheet_bridge*.png) in {OUT_ROOT}")
    print(f"  -> {OUT_ROOT / 'manifest.csv'}  ({len(man)} cutouts, "
          f"{int(man['has_image'].sum())} with imagery)")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
