#!/usr/bin/env python3
"""Per-cluster Euclid cutouts for the Euclid+WISE quasars.

These sources come from the Euclid MER catalogue, so every one has Euclid Q1
imaging. Per UMAP cluster we save numbered Euclid VIS+NISP stamps and a contact
sheet, ordered by descending XGB P(gal), with the outer box coloured by P(gal)
(red=galaxy, blue=star) and no crosshair (scale bar only):

  output/qso_cutouts/euclid_wise/
    cluster{c}/cutout_0000.png ...            individual Euclid stamps (P(gal)-sorted)
    contact_sheet_cluster{c}.png              per-cluster Euclid contact sheet
    manifest.csv

Euclid tiles: per-source IRSA IBE range reads (cached).

Usage:
    conda run -n base python3 src/download_euclid_wise_cutouts.py --n 40 --size 6
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
from save_q1_utils import (_discover_tiles, _assign_to_tiles, _render_cutout,  # noqa: E402
                           _nisp_filename, IRSA_IBE_DATA)
from euclid_ero_cutout import render_ero_cutout, euclid_release  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV = PROJECT_DIR / "output" / "umap" / "qso" / "arrays" / "qso_gaia_matches_euclidwise.csv"
OUT_DIR = PROJECT_DIR / "output" / "qso_cutouts" / "euclid_wise"
CLUSTER_COLOURS = {0: "#e6194b", 1: "#4363d8", 2: "#f58231", 3: "#3cb44b", 4: "#911eb4"}
APS_LABEL = {3: "2-param", 31: "5-param", 95: "6-param"}
_TILE_CACHE: dict = {}


def pgal_color(p):
    """Outer-border colour by XGB P(gal) (RdYlBu_r: red=galaxy, blue=star)."""
    return plt.cm.RdYlBu_r(float(np.clip(p, 0, 1))) if np.isfinite(p) else "0.5"


def load_sample(n: int, gmax: float, seed: int) -> pd.DataFrame:
    df = pd.read_csv(CSV, dtype={"gaia_source_id": "int64"},
                     usecols=["gaia_source_id", "qso_ra", "qso_dec", "cluster",
                              "astrometric_params_solved", "model_pgal", "phot_g_mean_mag"])
    df["sid_str"] = df["gaia_source_id"].astype("int64").astype(str)
    rng = np.random.default_rng(seed)
    picks = []
    for c in sorted(df["cluster"].dropna().unique()):
        sub = df[df["cluster"] == c]
        bright = sub[sub["phot_g_mean_mag"] < gmax]
        pool = bright if len(bright) >= n else sub
        take = min(n, len(pool))
        picks.append(pool.iloc[rng.choice(len(pool), take, replace=False)])
    # group by cluster; within a cluster order by descending XGB P(gal)
    return pd.concat(picks, ignore_index=True).sort_values(
        ["cluster", "model_pgal"], ascending=[True, False]).reset_index(drop=True)


def _tiles_near(ra: float, dec: float) -> list:
    """Per-source Euclid tile discovery (the sample spans the whole Q1 footprint,
    so one global bbox finds nothing). Cached by 0.1-deg cell."""
    key = (round(ra, 1), round(dec, 1))
    if key not in _TILE_CACHE:
        m = 0.05
        _TILE_CACHE[key] = _discover_tiles(ra - m, ra + m, dec - m, dec + m, "euclidwise")
    return _TILE_CACHE[key]


import signal
CACHE_DIR = PROJECT_DIR / "output" / "qso_cutouts" / ".euclid_stamp_cache"
FETCH_TIMEOUT = 60   # seconds per source — IRSA range reads occasionally stall forever


class _FetchTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _FetchTimeout()


def fetch_euclid(row: pd.Series, size: float) -> np.ndarray | None:
    """Rendered Euclid stamp (raw array). Cached to disk so reruns don't re-fetch;
    a hard SIGALRM timeout skips any source whose IRSA range read stalls."""
    sid = row["sid_str"]
    cache = CACHE_DIR / f"{sid}_{size:.0f}.png"
    if cache.exists():
        return plt.imread(cache)
    ra, dec = float(row["qso_ra"]), float(row["qso_dec"])
    ti = _assign_to_tiles(ra, dec, _tiles_near(ra, dec), "euclidwise")
    if ti is None:
        # No Q1 MER tile → ERO source: render from the cached ERO VIS full-field FITS.
        arr = render_ero_cutout(ra, dec, size, c_star=float(row["model_pgal"]),
                                draw_crosshair=False)
        if arr is not None:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            plt.imsave(cache, np.clip(arr, 0, 1))
        return arr
    njf, nhf = _nisp_filename(ti.tile_id, "J"), _nisp_filename(ti.tile_id, "H")
    nju = f"{IRSA_IBE_DATA}/{ti.tile_id}/NISP/{njf}" if njf else None
    nhu = f"{IRSA_IBE_DATA}/{ti.tile_id}/NISP/{nhf}" if nhf else None
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(FETCH_TIMEOUT)
    try:
        arr = _render_cutout(ti.url, ra, dec, size, "", c_star=float(row["model_pgal"]),
                             nisp_j_url=nju, nisp_h_url=nhu, draw_crosshair=False)
    except _FetchTimeout:
        print(f"    euclid {sid}: TIMEOUT ({FETCH_TIMEOUT}s) — skipping"); return None
    except Exception as exc:
        print(f"    euclid {sid}: FAILED ({exc})"); return None
    finally:
        signal.alarm(0)
    if arr is not None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        plt.imsave(cache, np.clip(arr, 0, 1))
    return arr


def render_to_array(row: pd.Series, arr, dpi: int = 100) -> np.ndarray:
    """Bake one Euclid stamp (black title + image, no border) into an array so the
    contact sheet can tile them tightly — identical construction to the blend /
    Legacy renderers (square figure + tight_layout, title pad=10)."""
    aps = int(row["astrometric_params_solved"]) if pd.notna(row["astrometric_params_solved"]) else 0
    apsl = APS_LABEL.get(aps, str(aps))
    g = row.get("phot_g_mean_mag", np.nan)
    pg = row.get("model_pgal", np.nan)
    gt = f"{g:.2f}" if pd.notna(g) else "?"
    pt = f"{pg:.2f}" if pd.notna(pg) else "?"
    title = (f"QSO Euclid {euclid_release(row)} + WISE  Cluster {int(row['cluster'])}  ({apsl})\n"
             f"RA={float(row['qso_ra']):.4f}  Dec={float(row['qso_dec']):+.4f}  P(gal)={pt}\n"
             f"G={gt}  id={row['sid_str']}")
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
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def contact_sheet(items: list, out: Path, cluster: int):
    """Per-cluster Euclid+WISE sheet: pre-rendered stamps sorted by descending P(gal);
    outer box coloured by P(gal). Same construction as qso_euclid_contact_sheet.png.
    `items` = list of (baked_array, pgal)."""
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
    # Fixed-height header (inches) so the colour-bar + suptitle never collide with the
    # stamps — critical for a single-row sheet, where a fractional top would overlap.
    fig_h_in = fig.get_figheight()
    header_in = 1.0
    top_frac = 1.0 - header_in / fig_h_in
    fig.subplots_adjust(top=top_frac, bottom=0.01, left=0.02, right=0.99, hspace=0.05, wspace=0.05)
    cbar_ax = fig.add_axes([0.06, top_frac + 0.30 / fig_h_in, 0.20, 0.10 / fig_h_in])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=plt.Normalize(0, 1)),
                        cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 0.5, 1]); cbar.set_ticklabels(["star", "0.5", "galaxy"])
    cbar.ax.tick_params(labelsize=7, pad=1)
    cbar_ax.set_title("border = XGBoost P(gal)", fontsize=8, pad=3, loc="left")
    fig.suptitle(f"Euclid+WISE quasars — Cluster {cluster} — Euclid VIS+NISP "
                 f"({n} sources, sorted by P(gal))", fontsize=13, fontweight="bold",
                 x=0.5, y=top_frac + 0.72 / fig_h_in, color=CLUSTER_COLOURS.get(cluster, "k"))
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40, help="sources per cluster (default 40)")
    ap.add_argument("--size", type=float, default=6.0, help="cutout FOV arcsec (default 6)")
    ap.add_argument("--gmax", type=float, default=21.5, help="prefer sources brighter than this G")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not CSV.exists():
        print(f"{CSV} not found — run analyze_umap_qso.py --catalog euclidwise first."); return 1

    samp = load_sample(args.n, args.gmax, args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for c in sorted(samp["cluster"].unique()):
        rows = samp[samp["cluster"] == c].reset_index(drop=True)
        cint = int(c)
        cdir = OUT_DIR / f"cluster{cint}"; cdir.mkdir(parents=True, exist_ok=True)
        for old in cdir.glob("*.png"):     # clear any stale cutouts before renumbering
            old.unlink()
        print(f"cluster {cint}: {len(rows)} Euclid cutouts...")
        items = []
        ne = 0
        for i, (_, row) in enumerate(rows.iterrows()):
            arr = fetch_euclid(row, args.size)
            ne += arr is not None
            baked = render_to_array(row, arr)              # title baked in, no border
            fn = f"cutout_{i:04d}.png"                      # numbered by P(gal), like the others
            plt.imsave(cdir / fn, np.clip(baked, 0, 1))
            items.append((baked, float(row["model_pgal"])))
            manifest.append({"cluster": cint, "cutout_file": f"cluster{cint}/{fn}",
                             "gaia_source_id": row["sid_str"], "aps": int(row["astrometric_params_solved"]),
                             "qso_ra": row["qso_ra"], "qso_dec": row["qso_dec"],
                             "model_pgal": row["model_pgal"], "euclid_ok": arr is not None})
        contact_sheet(items, OUT_DIR / f"contact_sheet_cluster{cint}.png", cint)
        print(f"  cluster{cint}/cutout_NNNN.png + contact sheet  ({ne}/{len(items)} Euclid ok)")
    pd.DataFrame(manifest).to_csv(OUT_DIR / "manifest.csv", index=False)
    print(f"  -> {OUT_DIR/'manifest.csv'}  ({len(manifest)} sources)")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
