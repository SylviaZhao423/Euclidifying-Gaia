#!/usr/bin/env python3
"""Redshift-binned Euclid cutouts for the spectroscopic (DESI DR1 + SDSS DR16) QSOs
that fall in the Euclid footprint.

Companion to download_qso_euclid_wise_g_cutouts.py, but for the SPEC sample: instead
of the WISE-selected candidates (organised by UMAP cluster / XGB P(gal)), these are
the ~2,933 DESI/SDSS QSOs with a spectroscopic redshift AND Euclid imaging — the same
sample as the colour(z) plot (analyze_qso_euclid_color_z.py). Having a real z, we
organise by REDSHIFT BIN instead of cluster and colour the outer border by z, so you
can see how a QSO's Euclid appearance/colour changes across cosmic time.

  output/qso_cutouts/euclid_ds/
    z0-1/cutout_0000.png ...            individual Euclid VIS+NISP stamps (z-sorted)
    contact_sheet_z0-1.png             per-z-bin contact sheet (border = redshift)
    manifest.csv

Sources (WSDB, needs VPN + PGUSER):
    {PGUSER}.crossmatch_qso_desi_dr1_euclid   (2,915; 2,843 Q1 + 72 ERO)
    {PGUSER}.crossmatch_qso_sdss_dr16_euclid  (18)
Cutouts are centred on the Euclid MER centroid (e_ra/e_dec):
    Q1  → render_q1_cutout: IRSA VIS MER mosaic tiles via HTTP range reads (cached)
    ERO → render_ero_cutout: cached ERO VIS Flattened FITS
The Euclid centroid is marked with a RED crosshair and every Gaia DR3 source in the
FOV with a small BLUE ring (same marking as the euclid_wise cutouts). Rendering itself
needs only internet (IRSA/ESA CDN); the Gaia rings need WSDB.

Usage:
    PGUSER=siwei_zhao_2025 python src/download_qso_euclid_ds_cutouts.py --n 40 --size 6
    PGUSER=siwei_zhao_2025 python src/download_qso_euclid_ds_cutouts.py --all       # all 2,933
    PGUSER=siwei_zhao_2025 python src/download_qso_euclid_ds_cutouts.py --no-gaia   # skip rings
"""
from __future__ import annotations

import argparse
import math
import os
import signal
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_q1_cutout import (_discover_tiles, _assign_to_tiles, _render_cutout,  # noqa: E402
                           _nisp_filename, IRSA_IBE_DATA)
from render_ero_cutout import render_ero_cutout                                # noqa: E402
from builder_qso_gaia_crossmatch import WSDB_HOST, WSDB_DB                     # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR   = PROJECT_DIR / "output" / "qso_cutouts" / "euclid_ds"
CACHE_DIR = PROJECT_DIR / "output" / "qso_cutouts" / ".euclid_stamp_cache"
GAIA_RING = "#1e90ff"          # blue rings = Gaia positions; Euclid centre = red cross
WSDB_USER = "siwei_zhao_2025"

# Spec-QSO × Euclid tables (same builder → same columns). cat char prefixes the cutout id.
TABLES = {"d": "crossmatch_qso_desi_dr1_euclid", "s": "crossmatch_qso_sdss_dr16_euclid"}

# Redshift bins → sub-directory / sheet tag (upper edge exclusive; last is open-ended).
Z_BINS = [(0.0, 1.0, "z0-1"), (1.0, 2.0, "z1-2"), (2.0, 3.0, "z2-3"), (3.0, np.inf, "z3+")]
Z_NORM = plt.Normalize(0, 4)   # border colour scale (viridis over z∈[0,4])

FETCH_TIMEOUT = 60             # seconds per source — IRSA range reads occasionally stall
_TILE_CACHE: dict = {}


def z_color(z):
    """Outer-border colour by redshift (viridis over [0, 4])."""
    return plt.cm.viridis(Z_NORM(float(z))) if np.isfinite(z) else "0.5"


def _zbin_tag(z: float) -> str | None:
    for lo, hi, tag in Z_BINS:
        if lo <= z < hi:
            return tag
    return None


# ------------------------------------------------------------------------------ data
def load_sample(n: int, take_all: bool, seed: int) -> pd.DataFrame:
    """Pull both spec-QSO × Euclid tables, assign a z-bin, and (unless --all) sample n
    per bin. Sorted by z within each bin. Needs WSDB (VPN + PGUSER)."""
    import psycopg2
    user = os.getenv("PGUSER", WSDB_USER)
    frames = []
    for cat0, table in TABLES.items():
        sql = f"""
            SELECT qso_catalog, qso_id, qso_z, qso_ra, qso_dec,
                   e_ra, e_dec, euclid_source
            FROM {user}.{table}
            WHERE qso_z IS NOT NULL AND qso_z > 0
              AND e_ra IS NOT NULL AND e_dec IS NOT NULL
        """
        try:
            conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                                    connect_timeout=30, keepalives=1, keepalives_idle=30,
                                    keepalives_interval=10, keepalives_count=5)
            df = pd.read_sql_query(sql, conn); conn.close()
        except Exception as exc:
            print(f"  skipped {table} ({exc})."); continue
        df["cat0"] = cat0
        frames.append(df)
        print(f"  {table}: {len(df):,} rows")
    if not frames:
        raise SystemExit("No spec-QSO × Euclid tables found (VPN? PGUSER? tables built?).")
    df = pd.concat(frames, ignore_index=True)
    df["cid"] = df["cat0"] + df["qso_id"].astype("int64").astype(str)
    df["zbin"] = df["qso_z"].map(_zbin_tag)
    df = df[df["zbin"].notna()].copy()

    rng = np.random.default_rng(seed)
    picks = []
    for _, _, tag in Z_BINS:
        sub = df[df["zbin"] == tag]
        if not len(sub):
            continue
        take = len(sub) if take_all else min(n, len(sub))
        picks.append(sub.iloc[rng.choice(len(sub), take, replace=False)])
    out = pd.concat(picks, ignore_index=True).sort_values(["zbin", "qso_z"]).reset_index(drop=True)
    print(f"  selected {len(out):,} QSOs "
          f"({'all' if take_all else f'≤{n}/bin'}): "
          + ", ".join(f"{t}={int((out['zbin']==t).sum())}" for _, _, t in Z_BINS))
    return out


# --------------------------------------------------------------------------- imaging
def _tiles_near(ra: float, dec: float) -> list:
    """Per-source Euclid Q1 tile discovery (sample spans the whole footprint). Cached by
    0.1-deg cell."""
    key = (round(ra, 1), round(dec, 1))
    if key not in _TILE_CACHE:
        m = 0.05
        _TILE_CACHE[key] = _discover_tiles(ra - m, ra + m, dec - m, dec + m, "euclid_ds")
    return _TILE_CACHE[key]


class _FetchTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _FetchTimeout()


def fetch_gaia_positions(df: pd.DataFrame, size: float) -> dict:
    """{cid: [(ra, dec, None), ...]} — every Gaia DR3 source within the cutout FOV of the
    Euclid centroid, to ring in blue. Needs WSDB; {} on failure → render without rings."""
    import psycopg2
    r_deg = (size / 2.0) / 3600.0
    out: dict = {}
    user = os.environ.get("PGUSER", WSDB_USER)
    try:
        conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                                connect_timeout=30, keepalives=1, keepalives_idle=30,
                                keepalives_interval=10, keepalives_count=5)
        cur = conn.cursor(); cur.execute("SET statement_timeout = 120000")
        for _, row in df.iterrows():
            ra, dec = float(row["e_ra"]), float(row["e_dec"])
            cur.execute("SELECT ra, dec FROM gaia_dr3.gaia_source "
                        "WHERE q3c_join(%s, %s, ra, dec, %s)", (ra, dec, r_deg))
            out[row["cid"]] = [(float(a), float(b), None) for a, b in cur.fetchall()]
        cur.close(); conn.close()
        got = sum(len(v) for v in out.values())
        print(f"  Gaia sources within {size/2:.1f}\" from WSDB: {got} across {len(out)} cutouts")
    except Exception as exc:
        print(f"  WSDB Gaia-position query failed ({exc}); rendering without rings. VPN up?")
        return {}
    return out


def fetch_euclid(row: pd.Series, size: float, gaia: list | None = None) -> np.ndarray | None:
    """Rendered Euclid stamp (raw array), centred on the Euclid centroid (e_ra/e_dec).
    Cached to disk; a hard SIGALRM timeout skips any stalled IRSA read. With `gaia`, the
    centroid gets a RED crosshair and each Gaia source a small BLUE ring ('_g' cache key)."""
    cid = row["cid"]
    mark = gaia is not None
    cache = CACHE_DIR / f"{cid}_{size:.0f}{'_g' if mark else ''}.png"
    if cache.exists():
        return plt.imread(cache)
    ra, dec = float(row["e_ra"]), float(row["e_dec"])
    ch = "red" if mark else None
    if str(row["euclid_source"]) == "ERO":
        arr = render_ero_cutout(ra, dec, size, draw_crosshair=mark, companions=gaia,
                                crosshair_color=ch, companion_color=GAIA_RING)
        if arr is not None:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            plt.imsave(cache, np.clip(arr, 0, 1))
        return arr
    ti = _assign_to_tiles(ra, dec, _tiles_near(ra, dec), "euclid_ds")
    if ti is None:
        return None                       # tagged Q1 but no VIS tile found
    njf, nhf = _nisp_filename(ti.tile_id, "J"), _nisp_filename(ti.tile_id, "H")
    nju = f"{IRSA_IBE_DATA}/{ti.tile_id}/NISP/{njf}" if njf else None
    nhu = f"{IRSA_IBE_DATA}/{ti.tile_id}/NISP/{nhf}" if nhf else None
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(FETCH_TIMEOUT)
    try:
        arr = _render_cutout(ti.url, ra, dec, size, "", nisp_j_url=nju, nisp_h_url=nhu,
                             draw_crosshair=mark, companions=gaia,
                             crosshair_color=ch, companion_color=GAIA_RING)
    except _FetchTimeout:
        print(f"    euclid {cid}: TIMEOUT ({FETCH_TIMEOUT}s) — skipping"); return None
    except Exception as exc:
        print(f"    euclid {cid}: FAILED ({exc})"); return None
    finally:
        signal.alarm(0)
    if arr is not None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        plt.imsave(cache, np.clip(arr, 0, 1))
    return arr


def render_to_array(row: pd.Series, arr, dpi: int = 100) -> np.ndarray:
    """Bake one Euclid stamp (black title + image, no border) into an array for the sheet."""
    cat = "DESI" if row["cat0"] == "d" else "SDSS"
    title = (f"QSO Euclid {row['euclid_source']} ({cat})  z={float(row['qso_z']):.3f}\n"
             f"RA={float(row['e_ra']):.4f}  Dec={float(row['e_dec']):+.4f}\n"
             f"id={row['qso_id']}")
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
        sp.set_visible(False)
    fig.tight_layout()
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return plt.imread(buf)


def contact_sheet(items: list, out: Path, tag: str):
    """Per-z-bin sheet: stamps sorted by z, outer box coloured by redshift.
    `items` = list of (baked_array, z)."""
    items = sorted(items, key=lambda t: t[1])
    n = len(items)
    if n == 0:
        return
    ncols = 8
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (arr, z) in enumerate(items):
        ax = axes[i // ncols][i % ncols]
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(arr)
        for sp in ax.spines.values():
            sp.set_edgecolor(z_color(z)); sp.set_linewidth(2.6)
    fig_h_in = fig.get_figheight()
    header_in = 1.0
    top_frac = 1.0 - header_in / fig_h_in
    fig.subplots_adjust(top=top_frac, bottom=0.01, left=0.02, right=0.99, hspace=0.05, wspace=0.05)
    cbar_ax = fig.add_axes([0.06, top_frac + 0.30 / fig_h_in, 0.20, 0.10 / fig_h_in])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=Z_NORM),
                        cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 1, 2, 3, 4]); cbar.ax.tick_params(labelsize=7, pad=1)
    cbar_ax.set_title("border = spectroscopic z", fontsize=8, pad=3, loc="left")
    fig.suptitle(f"Spectroscopic QSOs (DESI+SDSS) — {tag} — Euclid VIS+NISP "
                 f"({n} sources, sorted by z)", fontsize=13, fontweight="bold",
                 x=0.5, y=top_frac + 0.72 / fig_h_in)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40, help="sources per z-bin (default 40)")
    ap.add_argument("--all", action="store_true", help="render every source (ignores --n)")
    ap.add_argument("--size", type=float, default=6.0, help="cutout FOV arcsec (default 6)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-gaia", action="store_true",
                    help="skip the WSDB Gaia-position query (no red crosshair / blue rings)")
    args = ap.parse_args()

    samp = load_sample(args.n, args.all, args.seed)
    gaia = {} if args.no_gaia else fetch_gaia_positions(samp, args.size)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for _, _, tag in Z_BINS:
        rows = samp[samp["zbin"] == tag].reset_index(drop=True)
        if not len(rows):
            continue
        zdir = OUT_DIR / tag; zdir.mkdir(parents=True, exist_ok=True)
        for old in zdir.glob("*.png"):
            old.unlink()
        print(f"{tag}: {len(rows)} Euclid cutouts...")
        items = []
        ne = 0
        for i, (_, row) in enumerate(rows.iterrows()):
            arr = fetch_euclid(row, args.size, gaia=None if args.no_gaia else gaia.get(row["cid"], []))
            ne += arr is not None
            baked = render_to_array(row, arr)
            fn = f"cutout_{i:04d}.png"
            plt.imsave(zdir / fn, np.clip(baked, 0, 1))
            items.append((baked, float(row["qso_z"])))
            manifest.append({"zbin": tag, "cutout_file": f"{tag}/{fn}",
                             "qso_catalog": row["qso_catalog"], "qso_id": row["qso_id"],
                             "qso_z": row["qso_z"], "e_ra": row["e_ra"], "e_dec": row["e_dec"],
                             "euclid_source": row["euclid_source"],
                             "n_gaia_in_fov": len(gaia.get(row["cid"], [])),
                             "euclid_ok": arr is not None})
        contact_sheet(items, OUT_DIR / f"contact_sheet_{tag}.png", tag)
        print(f"  {tag}/cutout_NNNN.png + contact sheet  ({ne}/{len(items)} Euclid ok)")
    pd.DataFrame(manifest).to_csv(OUT_DIR / "manifest.csv", index=False)
    print(f"  -> {OUT_DIR/'manifest.csv'}  ({len(manifest)} sources)")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
