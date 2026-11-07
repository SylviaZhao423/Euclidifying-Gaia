#!/usr/bin/env python3
"""Legacy/DECaLS cutouts of the inter-cluster BRIDGES in the NSA-v1 nucleus × Gaia
sample (C0–C1, C0–C2, C1–C3, C2–C3, C2–C4), in the standard baked-title /
P(gal)-border contact-sheet format — the NSA analogue of
download_ds_g_bridges_cutouts.py.

Bridge membership is computed exactly as in the umap_nsa_overlay_v1.png stream panel —
outside_mask (sparse Q1 reference) + stream_labels (nearest named corridor), both from
the shared umap_streams.py (the single bridge definition), so the labels stay in sync
with the overlay and the QSO bridges.

Reads the projected NSA nuclei written by analyze_umap_nsa.py:
    output/umap/nsa/arrays/nsa_gaia_matches.csv   (v1)

Outputs (output/nsa_cutouts/v1_g_bridges/):
    {bridge}/cutout_NNNN.png        per-bridge individual Legacy stamps (P(gal)-sorted)
    contact_sheet_bridge{a}-{b}.png one sheet per bridge (a,b = endpoint clusters)
    manifest.csv

Usage:
    conda run -n base python3 src/download_nsa_v1_g_bridges_cutouts.py --n 40
"""
from __future__ import annotations

import argparse
import io
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_qso_ds_g_cutouts import fetch_ls_jpg, pgal_color, _place_block, _finish_sheet  # noqa: E402
from analyze_umap_qso import ARRAYS_DIR  # noqa: E402
from umap_streams import STREAM_PAIRS, outside_mask, stream_labels  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
NSA_CSV = PROJECT_DIR / "output" / "umap" / "nsa" / "arrays" / "nsa_gaia_matches.csv"
OUT_ROOT = PROJECT_DIR / "output" / "nsa_cutouts" / "v1_g_bridges"


def load_nuclei() -> pd.DataFrame:
    """The projected NSA nuclei (only the has_gaia rows carry a umap position)."""
    if not NSA_CSV.exists():
        raise SystemExit(f"{NSA_CSV} not found — run analyze_umap_nsa.py (v1) first.")
    # gaia_source_id has NULLs for unmatched galaxies (LEFT JOIN) → read as string to
    # avoid the float64 precision loss on 19-digit ids; nsaid is the primary NSA key.
    df = pd.read_csv(NSA_CSV, dtype={"gaia_source_id": "str"})
    det = df.dropna(subset=["umap1", "umap2"]).reset_index(drop=True)
    print(f"  {len(det):,} projected NSA nuclei (of {len(df):,} galaxies)")
    return det


def assign_bridges(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each nucleus with its bridge index (0..len(STREAM_PAIRS)-1), or -1 (other) /
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
    for a representative, well-imaged sheet (same policy as the QSO bridge script)."""
    bright = sub[sub["phot_g_mean_mag"] < gmax]
    pool = bright if len(bright) >= n else sub
    take = min(n, len(pool))
    return pool.iloc[rng.choice(len(pool), take, replace=False)]


def render_to_array(row: pd.Series, img, size_arcsec: float, name: str, dpi: int = 100) -> np.ndarray:
    """Bake one NSA-nucleus stamp (title + Legacy image + scale bar, no border) into an
    array so the sheet tiles them tightly; border is added by _place_block from P(gal)."""
    half = size_arcsec / 2.0
    fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=dpi)
    if img is not None:
        ax.imshow(np.asarray(img), extent=[half, -half, -half, half], origin="upper")
    else:
        ax.set_facecolor("#111111")
        ax.text(0, 0, "no\nimage", color="w", ha="center", va="center", fontsize=7)
    ax.plot([-half + 1.3, -half + 0.3], [-half + 0.7, -half + 0.7], "-", color="w", lw=2.0)
    ax.text(-half + 0.8, -half + 1.05, '1"', color="w", fontsize=7, ha="center", va="bottom")
    cl = int(row["cluster"])
    aps = int(row["astrometric_params_solved"]) if pd.notna(row["astrometric_params_solved"]) else 0
    apsl = {3: "2-param", 31: "5-param", 95: "6-param"}.get(aps, str(aps))
    g, pg, lm = row.get("phot_g_mean_mag", np.nan), row.get("model_pgal", np.nan), row.get("gal_log_mass", np.nan)
    gt = f"{g:.2f}" if pd.notna(g) else "?"
    pt = f"{pg:.2f}" if pd.notna(pg) else "?"
    lmt = f"{lm:.2f}" if pd.notna(lm) else "?"
    title = (f"NSA nucleus  {name}  Cluster {cl}  ({apsl})\n"
             f"RA={float(row['gal_ra']):.4f}  Dec={float(row['gal_dec']):+.4f}  P(gal)={pt}\n"
             f"G={gt}  logM={lmt}  id={row['gaia_source_id']}")
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


def bridge_contact_sheet(name: str, items, out: Path) -> None:
    """One-bridge sheet: stamps sorted by descending P(gal), P(gal)-coloured border."""
    items = sorted(items, key=lambda t: -t[1])
    if not items:
        return
    ncols = 8
    nrows = math.ceil(len(items) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    _place_block(axes, items, 0, ncols)
    _finish_sheet(fig, out, f"{name} bridge  ({len(items)} NSA nuclei, Legacy/DECaLS, "
                  f"sorted by P(gal))", None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40, help="max cutouts per bridge (default 40)")
    ap.add_argument("--size", type=float, default=12.0, help="cutout FOV arcsec (default 12)")
    ap.add_argument("--gmax", type=float, default=20.8, help="prefer sources brighter than this G")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = assign_bridges(load_nuclei())
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
            img = fetch_ls_jpg(float(row["gal_ra"]), float(row["gal_dec"]), args.size)
            arr = render_to_array(row, img, args.size, name)      # baked title, no border
            fname = f"cutout_{i:04d}.png"                          # numbered by P(gal)
            plt.imsave(bdir / fname, np.clip(arr, 0, 1))
            items.append((arr, float(row["model_pgal"])))
            manifest.append({
                "cutout_file": f"{safe}/{fname}", "bridge": name,
                "nsaid": int(row["nsaid"]), "nsa_catalog": row.get("nsa_catalog"),
                "gaia_source_id": row.get("gaia_source_id"),
                "gal_ra": row["gal_ra"], "gal_dec": row["gal_dec"],
                "gal_log_mass": row.get("gal_log_mass"),
                "phot_g_mean_mag": row["phot_g_mean_mag"], "model_pgal": row["model_pgal"],
                "cluster": int(row["cluster"]), "umap1": row["umap1"], "umap2": row["umap2"],
                "has_image": img is not None,
            })
        a, b = _verts[0], _verts[-1]          # endpoint cluster ids → sheet name
        bridge_contact_sheet(name, items, OUT_ROOT / f"contact_sheet_bridge{a}-{b}.png")

    man = pd.DataFrame(manifest)
    man.to_csv(OUT_ROOT / "manifest.csv", index=False)
    print(f"\n  -> per-bridge sheets (contact_sheet_bridge*.png) in {OUT_ROOT}")
    print(f"  -> {OUT_ROOT / 'manifest.csv'}  ({len(man)} cutouts, "
          f"{int(man['has_image'].sum()) if len(man) else 0} with imagery)")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
