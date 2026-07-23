#!/usr/bin/env python3
"""Euclid VIS(+NISP) cutouts of the inter-cluster BRIDGES in the Euclid+WISE ×Gaia sample
(C0–C1, C0–C2, C1–C3, C2–C3, C2–C4) — the Euclid counterpart of
download_qso_ds_g_bridges_cutouts.py (which does the same for the DESI+SDSS Legacy
sample). Same baked-title / P(gal)-border contact-sheet format, one sheet per bridge.

Which sample?  The inter-cluster streams are populated by the FULL all-WISE-matched
sample (--catalog euclidwise_all, the 439k crossmatch_euclid_wise_gaia — 762 outside the
Q1 cores), NOT by the W1-W2>=0.8 quasar cut (--catalog euclidwise, 6,265 — only 1 bridge
member; the AGN-cut sources sit firmly inside cores). So the default is euclidwise_all.

Bridge membership is computed exactly as in the umap_qso_overlay_*.png stream panel —
outside_mask (sparse Q1 reference) + stream_labels (nearest named corridor), both from the
shared umap_streams.py (the single bridge definition), so the labels stay in sync with the
overlay and the DESI+SDSS / NSA bridge cutouts.

Reads the projected sample written by analyze_umap_qso.py --catalog <catalog>:
    output/umap/qso/arrays/qso_gaia_matches_{euclidwise_all,euclidwise}.csv

Outputs (output/qso_cutouts/{euclid_wise_all,euclid_wise}_g_bridges/):
    {bridge}/cutout_NNNN.png        per-bridge individual Euclid stamps (P(gal)-sorted)
    contact_sheet_bridge{a}-{b}.png one sheet per bridge (a,b = endpoint clusters)
    manifest.csv

Euclid Q1 tiles: IRSA range reads (cached). ERO: local FITS. No Legacy/WSDB needed.

Usage:
    conda run -n base python3 src/download_qso_euclid_wise_g_bridges_cutouts.py --n 40 --size 6
    ... --catalog euclidwise           # the W1-W2>=0.8 QSO cut instead (~1 member)
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
from download_qso_euclid_wise_g_cutouts import (fetch_euclid, pgal_color,        # noqa: E402
                                                CLUSTER_COLOURS, APS_LABEL)
from euclid_ero_cutout import euclid_release  # noqa: E402
from analyze_umap_qso import ARRAYS_DIR       # noqa: E402
from umap_streams import STREAM_PAIRS, outside_mask, stream_labels  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
ARR_DIR = PROJECT_DIR / "output" / "umap" / "qso" / "arrays"
# catalog -> (projected CSV, output-folder stem). euclidwise_all (439k, no colour cut) is
# where the streams actually live; euclidwise (6,265, W1-W2>=0.8) has ~1 bridge member.
CATALOGS = {
    "euclidwise_all": (ARR_DIR / "qso_gaia_matches_euclidwise_all.csv", "euclid_wise_all_g_bridges"),
    "euclidwise":     (ARR_DIR / "qso_gaia_matches_euclidwise.csv",     "euclid_wise_g_bridges"),
}


def load_sample(csv: Path) -> pd.DataFrame:
    """The projected Euclid+WISE sample (only the has-umap rows carry a position)."""
    if not csv.exists():
        raise SystemExit(f"{csv.name} not found — run analyze_umap_qso.py for that catalog first.")
    df = pd.read_csv(csv, dtype={"gaia_source_id": "int64"}).dropna(subset=["umap1", "umap2"])
    df["sid_str"] = df["gaia_source_id"].astype("int64").astype(str)
    return df.reset_index(drop=True)


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
    for a representative, well-imaged sheet (same policy as the QSO/NSA bridge scripts)."""
    bright = sub[sub["phot_g_mean_mag"] < gmax]
    pool = bright if len(bright) >= n else sub
    take = min(n, len(pool))
    return pool.iloc[rng.choice(len(pool), take, replace=False)]


def render_to_array(row: pd.Series, arr, name: str, dpi: int = 100) -> np.ndarray:
    """Bake one Euclid bridge stamp (black title + image, no border) into an array so the
    sheet tiles them tightly; title carries the bridge name + endpoint clusters. Mirrors
    the per-cluster Euclid renderer but with a bridge-labelled title."""
    aps = int(row["astrometric_params_solved"]) if pd.notna(row["astrometric_params_solved"]) else 0
    apsl = APS_LABEL.get(aps, str(aps))
    g = row.get("phot_g_mean_mag", np.nan)
    pg = row.get("model_pgal", np.nan)
    gt = f"{g:.2f}" if pd.notna(g) else "?"
    pt = f"{pg:.2f}" if pd.notna(pg) else "?"
    title = (f"QSO Euclid {euclid_release(row)} + WISE  {name} bridge  ({apsl})\n"
             f"RA={float(row['qso_ra']):.4f}  Dec={float(row['qso_dec']):+.4f}  P(gal)={pt}\n"
             f"G={gt}  Cluster {int(row['cluster'])}  id={row['sid_str']}")
    fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=dpi)
    if arr is not None:
        ax.imshow(np.clip(arr, 0, 1))
    else:
        ax.set_facecolor("#111")
        ax.text(0.5, 0.5, "no Euclid\ntile", color="w", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8, color="black", pad=10)
    for sp in ax.spines.values():
        sp.set_visible(False)                 # border is added by the contact sheet
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def bridge_contact_sheet(name: str, items: list, out: Path) -> None:
    """One-bridge Euclid sheet: stamps sorted by descending P(gal); outer box coloured by
    P(gal); fixed-height header (so single-row sheets don't overlap the title). Mirrors the
    per-cluster Euclid contact sheet but titled by bridge (bridge spans two clusters → black
    suptitle). `items` = list of (baked_array, pgal)."""
    items = sorted(items, key=lambda t: -t[1])
    n = len(items)
    if n == 0:
        return
    ncols = 8
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (arr, pg) in enumerate(items):
        ax = axes[i // ncols][i % ncols]
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(arr)
        for sp in ax.spines.values():
            sp.set_edgecolor(pgal_color(pg)); sp.set_linewidth(2.6)
    fig_h_in = fig.get_figheight()
    top_frac = 1.0 - 1.0 / fig_h_in
    fig.subplots_adjust(top=top_frac, bottom=0.01, left=0.02, right=0.99, hspace=0.05, wspace=0.05)
    cbar_ax = fig.add_axes([0.06, top_frac + 0.30 / fig_h_in, 0.20, 0.10 / fig_h_in])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=plt.Normalize(0, 1)),
                        cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 0.5, 1]); cbar.set_ticklabels(["star", "0.5", "galaxy"])
    cbar.ax.tick_params(labelsize=7, pad=1)
    cbar_ax.set_title("border = XGBoost P(gal)", fontsize=8, pad=3, loc="left")
    fig.suptitle(f"Euclid+WISE — {name} bridge — Euclid VIS+NISP "
                 f"({n} sources, sorted by P(gal))", fontsize=13, fontweight="bold",
                 x=0.5, y=top_frac + 0.72 / fig_h_in, color="k")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", choices=list(CATALOGS), default="euclidwise_all",
                    help="which projection (default euclidwise_all — where the streams live)")
    ap.add_argument("--n", type=int, default=40, help="max cutouts per bridge (default 40)")
    ap.add_argument("--size", type=float, default=6.0, help="cutout FOV arcsec (default 6)")
    ap.add_argument("--gmax", type=float, default=21.5, help="prefer sources brighter than this G")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    csv, stem = CATALOGS[args.catalog]
    out_root = PROJECT_DIR / "output" / "qso_cutouts" / stem
    print(f"catalog: {args.catalog}  ({csv.name})  ->  {out_root}")
    df = assign_bridges(load_sample(csv))
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    manifest = []

    for k, (name, verts, _col) in enumerate(STREAM_PAIRS):
        sub = df[df["bridge"] == k]
        safe = name.replace("–", "-")                       # en-dash → ascii for folder names
        print(f"=== {name} bridge: {len(sub):,} members ===")
        if not len(sub):
            print("  (none)"); continue
        rows = sample_bridge(sub, args.n, args.gmax, rng).sort_values(
            "model_pgal", ascending=False).reset_index(drop=True)
        bdir = out_root / safe
        bdir.mkdir(parents=True, exist_ok=True)
        for old in bdir.glob("*.png"):                      # clear stale cutouts
            old.unlink()
        items = []
        n_ok = 0
        print(f"  {len(rows)} Euclid cutouts...")
        for i, (_, row) in enumerate(rows.iterrows()):
            arr = fetch_euclid(row, args.size)              # Q1 IRSA tile, else ERO local FITS
            n_ok += arr is not None
            baked = render_to_array(row, arr, name)         # bridge-labelled title, no border
            fname = f"cutout_{i:04d}.png"                   # numbered by P(gal)
            plt.imsave(bdir / fname, np.clip(baked, 0, 1))
            items.append((baked, float(row["model_pgal"])))
            manifest.append({
                "cutout_file": f"{safe}/{fname}", "bridge": name,
                "gaia_source_id": row["sid_str"], "euclid": euclid_release(row),
                "aps": int(row["astrometric_params_solved"]),
                "qso_ra": row["qso_ra"], "qso_dec": row["qso_dec"],
                "phot_g_mean_mag": row["phot_g_mean_mag"], "model_pgal": row["model_pgal"],
                "cluster": int(row["cluster"]), "umap1": row["umap1"], "umap2": row["umap2"],
                "euclid_ok": arr is not None,
            })
        a, b = verts[0], verts[-1]            # endpoint cluster ids → sheet name
        bridge_contact_sheet(name, items, out_root / f"contact_sheet_bridge{a}-{b}.png")
        print(f"    {name}: {len(rows)} cutouts ({n_ok} Euclid ok)")

    man = pd.DataFrame(manifest)
    man.to_csv(out_root / "manifest.csv", index=False)
    print(f"\n  -> per-bridge sheets (contact_sheet_bridge*.png) in {out_root}")
    print(f"  -> {out_root / 'manifest.csv'}  ({len(man)} cutouts, "
          f"{int(man['euclid_ok'].sum()) if len(man) else 0} Euclid ok)")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
