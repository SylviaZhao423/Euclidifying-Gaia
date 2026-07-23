#!/usr/bin/env python3
"""Per-cluster Legacy/DECaLS cutouts for the merged DESI+SDSS QSO web viewer.

SDSS and DESI are already merged into one projection (qso_gaia_matches_desisdss.csv),
so we sample up to N sources per UMAP cluster from the COMBINED sample and save one set
of per-cluster folders/sheets directly under desi+sdss/ (each stamp's title notes its
catalogue). Cutouts numbered by descending XGBoost P(gal), like the blend/Euclid folders:

  output/qso_cutouts/desi+sdss/cluster{c}/cutout_NNNN.png
  output/qso_cutouts/desi+sdss/contact_sheet_cluster{c}.png
  output/qso_cutouts/desi+sdss/manifest.csv

Only ~N/cluster get cutouts (we cannot download for a million QSOs). Sampling prefers
brighter sources (better imaging) but is randomised so the sheet is representative.

Usage:
    conda run -n base python3 src/download_ds_g_cutouts.py            # 50/cluster
    ... --n 40 --gmax 20.8 --size 12
"""
from __future__ import annotations

import argparse
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_qso_ds_g_blend_cutouts import fetch_ls_jpg, CLUSTER_COLOURS  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
QSO_DIR = PROJECT_DIR / "output" / "umap" / "qso"
OUT_ROOT = PROJECT_DIR / "output" / "qso_cutouts" / "desi+sdss"

# merged DESI+SDSS projection, written by analyze_umap_qso.py (default / --catalog both)
MERGED_CSV = QSO_DIR / "arrays" / "qso_gaia_matches_desisdss.csv"


def _cat_tag(qso_catalog) -> str:
    """Short catalogue tag for a stamp title (SDSS_DR16Q → SDSS, DESI_DR1Q → DESI)."""
    s = str(qso_catalog).upper()
    return "SDSS" if "SDSS" in s else ("DESI" if "DESI" in s else s)


def load_projected(csv: Path) -> pd.DataFrame:
    cols = ["qso_catalog", "qso_id", "qso_ra", "qso_dec", "qso_z", "gaia_source_id",
            "astrometric_params_solved", "phot_g_mean_mag", "umap1", "umap2",
            "cluster", "model_pgal"]
    df = pd.read_csv(csv, usecols=lambda c: c in cols,
                     dtype={"gaia_source_id": "int64", "qso_id": "int64"})
    # String id column: df.iterrows() upcasts an all-numeric row to float64, which
    # truncates 19-digit int64 Gaia source_ids (>2^53). Always read the id from
    # this string column, never int(row["gaia_source_id"]).
    df["sid_str"] = df["gaia_source_id"].astype("int64").astype(str)
    return df


def sample_per_cluster(df: pd.DataFrame, n: int, gmax: float, seed: int) -> pd.DataFrame:
    """Up to n sources per cluster, drawn from the brighter half (G<gmax where
    possible) at random for a representative, well-imaged sheet."""
    rng = np.random.default_rng(seed)
    picks = []
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        bright = sub[sub["phot_g_mean_mag"] < gmax]
        pool = bright if len(bright) >= n else sub
        take = min(n, len(pool))
        picks.append(pool.iloc[rng.choice(len(pool), take, replace=False)])
    return pd.concat(picks, ignore_index=True)


def pgal_color(p):
    """Outer-border colour by XGB P(gal) (RdYlBu_r: red=galaxy, blue=star)."""
    return plt.cm.RdYlBu_r(float(np.clip(p, 0, 1))) if np.isfinite(p) else "0.5"


def render_stamp(ax, row: pd.Series, img, size_arcsec: float, tag: str,
                 show_border: bool = True) -> None:
    """QSO stamp: Legacy image + scale bar + black title; outer box coloured by XGB
    P(gal). Same construction as the blend / Euclid stamps."""
    half = size_arcsec / 2.0
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
    g = row.get("phot_g_mean_mag", np.nan)
    z = row.get("qso_z", np.nan)
    pg = row.get("model_pgal", np.nan)
    sid = row["sid_str"] if "sid_str" in row else str(int(row["gaia_source_id"]))
    gt = f"{g:.2f}" if pd.notna(g) else "?"
    zt = f"{z:.2f}" if pd.notna(z) else "?"
    pt = f"{pg:.2f}" if pd.notna(pg) else "?"
    title = (f"QSO {tag}  Cluster {cl}  ({apsl})\n"
             f"RA={float(row['qso_ra']):.4f}  Dec={float(row['qso_dec']):+.4f}  P(gal)={pt}\n"
             f"G={gt}  z={zt}  id={sid}")
    ax.set_title(title, fontsize=8, color="black", pad=10)
    ax.set_xlim(half, -half); ax.set_ylim(-half, half)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        if show_border:
            sp.set_edgecolor(pgal_color(pg)); sp.set_linewidth(2.6)
        else:
            sp.set_visible(False)


def render_to_array(row: pd.Series, img, size_arcsec: float, tag: str, dpi: int = 100) -> np.ndarray:
    """Bake one stamp (title + image + scale bar, no border) into an array so the
    contact sheet can tile them tightly without title overlap (as blend / Euclid)."""
    fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=dpi)
    render_stamp(ax, row, img, size_arcsec, tag, show_border=False)
    fig.tight_layout()
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def _place_block(axes, items, r0: int, ncols: int, ylabel: str | None = None) -> int:
    """Draw pre-rendered stamps (list of (array, pgal)) into `axes` starting at grid
    row r0; P(gal)-coloured border; optional bold ylabel on the block's first stamp.
    Returns the number of grid rows the block consumed."""
    for k, (arr, pg) in enumerate(items):
        ax = axes[r0 + k // ncols][k % ncols]
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(arr)
        for sp in ax.spines.values():
            sp.set_edgecolor(pgal_color(pg)); sp.set_linewidth(2.6)
        if k == 0 and ylabel:
            ax.set_ylabel(ylabel, fontsize=15, fontweight="bold", labelpad=8)
    return math.ceil(len(items) / ncols) if items else 0


def _finish_sheet(fig, out: Path, title: str, cluster: int) -> None:
    """Shared colour-bar key + cluster-coloured suptitle + save, for both the single
    and combined sheets (identical construction to qso_euclid_contact_sheet.png)."""
    # Fixed-height header (inches) so the colour-bar + suptitle never collide with the
    # stamps — critical for a single-row sheet, where a fractional top would overlap.
    fig_h_in = fig.get_figheight()
    header_in = 1.0
    top_frac = 1.0 - header_in / fig_h_in
    fig.subplots_adjust(top=top_frac, bottom=0.01, left=0.05, right=0.99, hspace=0.05, wspace=0.05)
    cbar_ax = fig.add_axes([0.08, top_frac + 0.30 / fig_h_in, 0.20, 0.10 / fig_h_in])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=plt.Normalize(0, 1)),
                        cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 0.5, 1]); cbar.set_ticklabels(["star", "0.5", "galaxy"])
    cbar.ax.tick_params(labelsize=7, pad=1)
    cbar_ax.set_title("border = XGBoost P(gal)", fontsize=8, pad=3, loc="left")
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.5, y=top_frac + 0.72 / fig_h_in,
                 color=CLUSTER_COLOURS.get(cluster, "k"))
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def contact_sheet(items, cluster: int, out: Path) -> None:
    """Per-cluster DESI+SDSS sheet: stamps sorted by descending P(gal); outer box
    coloured by P(gal). `items` = list of (image_array, pgal)."""
    items = sorted(items, key=lambda t: -t[1])
    n = len(items)
    if n == 0:
        return
    ncols = 8
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    _place_block(axes, items, 0, ncols)
    _finish_sheet(fig, out, f"DESI+SDSS — Cluster {cluster}  ({n} QSOs, Legacy/DECaLS, "
                  f"sorted by P(gal))", cluster)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=None,
                    help="override projected CSV path (default: qso_gaia_matches_desisdss.csv)")
    ap.add_argument("--n", type=int, default=50, help="max cutouts per cluster (default 50)")
    ap.add_argument("--size", type=float, default=12.0, help="cutout FOV arcsec (default 12)")
    ap.add_argument("--gmax", type=float, default=20.8, help="prefer sources brighter than this G")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=12,
                    help="parallel Legacy fetches per cluster (default 12)")
    args = ap.parse_args()

    csv = args.csv or MERGED_CSV
    if not csv.exists():
        print(f"ERROR: {csv} not found — run analyze_umap_qso.py (default / --catalog both) first.")
        return 1

    df = load_projected(csv)
    print(f"{len(df):,} DESI+SDSS QSOs; sampling up to {args.n}/cluster (G<{args.gmax})...")
    samp = sample_per_cluster(df, args.n, args.gmax, args.seed)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = []
    for c in sorted(samp["cluster"].unique()):
        # within a cluster, order (and number) the cutouts by descending XGB P(gal)
        rows = samp[samp["cluster"] == c].sort_values(
            "model_pgal", ascending=False).reset_index(drop=True)
        cdir = OUT_ROOT / f"cluster{c}"; cdir.mkdir(parents=True, exist_ok=True)
        for old in cdir.glob("*.png"):     # clear stale cutouts before renumbering
            old.unlink()
        # Legacy fetch is I/O-bound → fetch the whole cluster CONCURRENTLY (huge speedup
        # over serial when the sample is uncached), then render serially (matplotlib is
        # not thread-safe).
        print(f"  cluster {c}: fetching {len(rows)} Legacy cutouts ({args.workers} parallel)...")
        row_list = [row for _, row in rows.iterrows()]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            imgs = list(ex.map(
                lambda row: fetch_ls_jpg(float(row["qso_ra"]), float(row["qso_dec"]), args.size),
                row_list))
        items = []
        for i, (row, img) in enumerate(zip(row_list, imgs)):
            arr = render_to_array(row, img, args.size, _cat_tag(row["qso_catalog"]))  # per-source tag
            fname = f"cutout_{i:04d}.png"      # numbered by P(gal), like blend/euclid
            plt.imsave(cdir / fname, np.clip(arr, 0, 1))
            items.append((arr, float(row["model_pgal"])))
            manifest.append({
                "cutout_file": f"cluster{c}/{fname}", "cluster": c,
                "qso_catalog": row["qso_catalog"], "gaia_source_id": int(row["sid_str"]),
                "qso_id": int(row["qso_id"]), "qso_ra": row["qso_ra"], "qso_dec": row["qso_dec"],
                "qso_z": row["qso_z"], "phot_g_mean_mag": row["phot_g_mean_mag"],
                "model_pgal": row["model_pgal"], "umap1": row["umap1"], "umap2": row["umap2"],
                "has_image": img is not None,
            })
        print(f"    cluster {c}: {sum(im is not None for im in imgs)}/{len(rows)} imaged")
        contact_sheet(items, c, OUT_ROOT / f"contact_sheet_cluster{c}.png")

    man = pd.DataFrame(manifest)
    man.to_csv(OUT_ROOT / "manifest.csv", index=False)
    print(f"  -> {OUT_ROOT/'manifest.csv'}  ({len(man)} cutouts, "
          f"{int(man['has_image'].sum())} with imagery)")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
