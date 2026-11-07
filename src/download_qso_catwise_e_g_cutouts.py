#!/usr/bin/env python3
"""Per-cluster Euclid cutouts for the Euclid+WISE quasars.

These sources come from the Euclid MER catalogue, so every one has Euclid Q1
imaging. Per UMAP cluster we save numbered Euclid VIS+NISP stamps and a contact
sheet, ordered by descending XGB P(gal), with the outer box coloured by P(gal)
(red=galaxy, blue=star).

Companion rings on each stamp (colour-coded):
  Red crosshair   — Euclid primary centroid (cutout centre)
  Orange rings    — Secondary Euclid sources within 1" (n_euclid > 1)
  Magenta ring    — WISE source position (the catalog selector)
  Bright blue     — Gaia sources within 1" of the WISE source (matched in crossmatch)
  Pale blue       — Gaia sources in stamp FOV but outside 1" (unmatched neighbours)

Blend flags in the title when multiple sources share the same match:
  nG=N  — N Gaia within 1" of this WISE source
  nE=N  — N Euclid within 1" of this WISE source
  ★     — mutual blend (n_euclid > 1 AND n_qso > 1, or same for Gaia)

Blend info is pulled once at startup from:
  {user}.crossmatch_qso_catwise_euclid_gaia  (combined table: all blend arrays + WISE photometry)
linked to the flat CSV via g_source_id.

Outputs:
  output/qso_cutouts/catwise_e_g/
    cluster{c}/cutout_0000.png ...         individual Euclid stamps (P(gal)-sorted)
    contact_sheet_cluster{c}.png           per-cluster Euclid contact sheet
    contact_sheet_cluster{c}_p{n}.png      (split into pages of 80 when a cluster has >80)
    manifest.csv                           includes blend columns

Usage:
    PGUSER=siwei_zhao_2025 conda run -n base python3 src/download_qso_catwise_e_g_cutouts.py
    PGUSER=siwei_zhao_2025 ... --n 40 --size 6
    ... --no-gaia      # skip WSDB entirely (no rings at all)
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_q1_cutout import (_discover_tiles, _assign_to_tiles, _render_cutout,  # noqa: E402
                              _nisp_filename, IRSA_IBE_DATA,
                              GAIA_RING, GAIA_FOV_RING, EUCLID_RING, WISE_RING)
from render_ero_cutout import render_ero_cutout, euclid_release  # noqa: E402
from builder_qso_gaia_crossmatch import WSDB_HOST, WSDB_DB       # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV = PROJECT_DIR / "output" / "umap" / "qso" / "arrays" / "catwise_euclid_gaia.csv"
OUT_DIR = PROJECT_DIR / "output" / "qso_cutouts" / "catwise_e_g"

CLUSTER_COLOURS = {0: "#e6194b", 1: "#4363d8", 2: "#f58231", 3: "#3cb44b", 4: "#911eb4"}
APS_LABEL = {3: "2-param", 31: "5-param", 95: "6-param"}
_TILE_CACHE: dict = {}


def pgal_color(p):
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
    return pd.concat(picks, ignore_index=True).sort_values(
        ["cluster", "model_pgal"], ascending=[True, False]).reset_index(drop=True)


# ── Blend info from crossmatch tables ─────────────────────────────────────────

def fetch_blend_info(samp: pd.DataFrame, user: str) -> dict:
    """Query crossmatch_qso_catwise_euclid_gaia for all sources in samp.
    Returns {sid_str: row_dict} with all blend columns + WISE photometry.
    {} on failure (VPN down etc.) — cutout still renders without colour-coded rings."""
    import psycopg2
    gids = samp["gaia_source_id"].astype("int64").tolist()
    ids_sql = ",".join(str(g) for g in gids)
    try:
        conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                                connect_timeout=30, keepalives=1, keepalives_idle=30,
                                keepalives_interval=10, keepalives_count=5)
        conn.cursor().execute("SET statement_timeout = 60000")
        df = pd.read_sql_query(f"""
            SELECT g_source_id,
                   qso_id                        AS wise_id,
                   qso_ra,
                   qso_dec,
                   gaia_sep_arcsec,
                   n_gaia_within_1arcsec,
                   all_gaia_ids_within_1arcsec,
                   all_gaia_seps_within_1arcsec,
                   gaia_is_mutual_blend           AS gaia_is_blend,
                   e_id,
                   e_ra,
                   e_dec,
                   euclid_sep_arcsec,
                   n_euclid_within_1arcsec,
                   all_euclid_ids_within_1arcsec,
                   all_euclid_seps_within_1arcsec,
                   euclid_is_mutual_blend         AS euclid_is_blend,
                   euclid_source,
                   w1mpro,
                   w2mpro
            FROM {user}.crossmatch_qso_catwise_euclid_gaia
            WHERE g_source_id IN ({ids_sql})
        """, conn)
        conn.close()
        print(f"  blend info: {len(df):,} sources from crossmatch_qso_catwise_euclid_gaia")
        return {str(int(r["g_source_id"])): r.to_dict() for _, r in df.iterrows()}
    except Exception as exc:
        print(f"  blend info fetch failed ({exc}) — rings will use single colour")
        return {}


def fetch_euclid_secondary_positions(samp: pd.DataFrame, blend_info: dict,
                                     user: str) -> dict:
    """Fetch RA/Dec for secondary Euclid sources (all_euclid_ids_within_1arcsec[1:])
    for sources with n_euclid > 1. Returns {sid_str: [(ra, dec, None, EUCLID_RING), ...]}.

    Routing uses euclid_source ("Q1" or "ERO") from blend_info — NOT ID magnitude.
    Q1 object_id values are signed bigints that can be negative (overflow), so the
    old >10^14 check silently misrouted all Q1 secondaries to ERO, finding nothing.
    Q1 IDs are globally unique — batch-fetched. ERO v_objid values are also
    globally unique — batch-fetched in the same way."""
    import psycopg2

    # Collect secondary IDs, split by Q1 vs ERO using euclid_source tag
    q1_secondary:  dict[int, list[str]] = {}  # eid → [sid_str, ...]
    ero_secondary: dict[int, list[str]] = {}  # eid → [sid_str, ...]

    for _, row in samp.iterrows():
        sid = row["sid_str"]
        b = blend_info.get(sid)
        if b is None:
            continue
        all_ids = b.get("all_euclid_ids_within_1arcsec") or []
        if len(all_ids) <= 1:
            continue
        primary_eid = b.get("e_id")
        is_q1 = b.get("euclid_source") == "Q1"
        for eid in all_ids:
            if eid == primary_eid:
                continue
            if is_q1:
                q1_secondary.setdefault(int(eid), []).append(sid)
            else:
                ero_secondary.setdefault(int(eid), []).append(sid)

    if not q1_secondary and not ero_secondary:
        return {}

    out: dict[str, list] = {}

    try:
        conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                                connect_timeout=30, keepalives=1, keepalives_idle=30,
                                keepalives_interval=10, keepalives_count=5)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 60000")

        # Q1: batch fetch (globally unique IDs)
        if q1_secondary:
            cur.execute("SELECT object_id, right_ascension, declination "
                        "FROM euclid_q1.mer_final_cat WHERE object_id = ANY(%s)",
                        (list(q1_secondary.keys()),))
            for eid, ra, dec in cur.fetchall():
                for sid in q1_secondary.get(int(eid), []):
                    out.setdefault(sid, []).append((float(ra), float(dec), None, EUCLID_RING))

        # ERO: batch fetch by v_objid (globally unique — no proximity filter needed)
        if ero_secondary:
            cur.execute(
                "SELECT v_objid, v_alpha_j2000, v_delta_j2000 "
                "FROM euclid_ero.merge_cat "
                "WHERE v_objid = ANY(%s)",
                (list(ero_secondary.keys()),))
            for eid, ra, dec in cur.fetchall():
                for sid in ero_secondary.get(int(eid), []):
                    out.setdefault(sid, []).append((float(ra), float(dec), None, EUCLID_RING))

        cur.close(); conn.close()
    except Exception as exc:
        print(f"  secondary Euclid position fetch failed ({exc})")
        return {}

    n_total = sum(len(v) for v in out.values())
    print(f"  secondary Euclid sources: {n_total} rings across {len(out)} stamps")
    return out


# ── Gaia positions in FOV ──────────────────────────────────────────────────────

def _tiles_near(ra: float, dec: float) -> list:
    key = (round(ra, 1), round(dec, 1))
    if key not in _TILE_CACHE:
        m = 0.05
        _TILE_CACHE[key] = _discover_tiles(ra - m, ra + m, dec - m, dec + m, "euclidwise")
    return _TILE_CACHE[key]


import signal
CACHE_DIR = PROJECT_DIR / "output" / "qso_cutouts" / ".euclid_stamp_cache"
FETCH_TIMEOUT = 60


class _FetchTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _FetchTimeout()


def fetch_gaia_positions(df: pd.DataFrame, size: float, blend_info: dict) -> dict:
    """{sid_str: [(ra, dec, None, color), ...]} — every Gaia DR3 source within the
    stamp FOV. Color = GAIA_RING if the source_id is in all_gaia_ids_within_1arcsec
    (matched within 1" of the WISE source), else GAIA_FOV_RING (in-stamp neighbour).
    Falls back to all rings in GAIA_RING if blend_info is empty."""
    import psycopg2
    r_deg = (size / 2.0) / 3600.0
    out: dict = {}
    user = os.environ.get("PGUSER", "siwei_zhao_2025")
    try:
        conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                                connect_timeout=30, keepalives=1, keepalives_idle=30,
                                keepalives_interval=10, keepalives_count=5)
        cur = conn.cursor(); cur.execute("SET statement_timeout = 120000")
        for _, row in df.iterrows():
            sid = row["sid_str"]
            ra, dec = float(row["qso_ra"]), float(row["qso_dec"])
            cur.execute("SELECT source_id, ra, dec FROM gaia_dr3.gaia_source "
                        "WHERE q3c_join(%s, %s, ra, dec, %s)", (ra, dec, r_deg))
            b = blend_info.get(sid, {})
            matched_ids = set(b.get("all_gaia_ids_within_1arcsec") or [])
            rings = []
            for gid, gra, gdec in cur.fetchall():
                color = GAIA_RING if (not matched_ids or int(gid) in matched_ids) else GAIA_FOV_RING
                rings.append((float(gra), float(gdec), None, color))
            out[sid] = rings
        cur.close(); conn.close()
        got = sum(len(v) for v in out.values())
        print(f"  Gaia sources in {size/2:.1f}\" FOV: {got} across {len(out)} stamps")
    except Exception as exc:
        print(f"  WSDB Gaia query failed ({exc}); rendering without rings. VPN up?")
        return {}
    return out


# ── Euclid tile fetch ──────────────────────────────────────────────────────────

def fetch_euclid(row: pd.Series, size: float,
                 companions: list | None = None,
                 center_ra: float | None = None,
                 center_dec: float | None = None) -> np.ndarray | None:
    """Rendered Euclid stamp. Cached to disk; hard SIGALRM timeout skips stalled reads.
    companions = 4-tuple list (ra, dec, label, color) for ring overlays.
    center_ra/center_dec override the cutout centre (default: qso_ra/qso_dec)."""
    sid = row["sid_str"]
    mark = companions is not None
    ec = center_ra is not None
    cache = CACHE_DIR / f"{sid}_{size:.0f}{'_ec' if ec else ''}{'_g' if mark else ''}.png"
    if cache.exists():
        return plt.imread(cache)
    ra  = center_ra  if center_ra  is not None else float(row["qso_ra"])
    dec = center_dec if center_dec is not None else float(row["qso_dec"])
    ch = "red" if mark else None
    ti = _assign_to_tiles(ra, dec, _tiles_near(ra, dec), "euclidwise")
    if ti is None:
        arr = render_ero_cutout(ra, dec, size, c_star=float(row["model_pgal"]),
                                draw_crosshair=mark, companions=companions,
                                crosshair_color=ch)
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
                             nisp_j_url=nju, nisp_h_url=nhu, draw_crosshair=mark,
                             companions=companions, crosshair_color=ch)
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


# ── Title + stamp rendering ────────────────────────────────────────────────────

def render_to_array(row: pd.Series, arr, blend: dict, dpi: int = 100,
                    cluster_label: str = None) -> np.ndarray:
    """Bake one Euclid stamp (title + image, no border) into an array. `cluster_label`
    overrides the default 'Cluster {row.cluster}' text in the title (e.g. 'Q1_Cluster 1'
    or 'QSO_Cluster 3' depending on which UMAP the caller assigned the source with)."""
    import io
    aps  = int(row["astrometric_params_solved"]) if pd.notna(row.get("astrometric_params_solved")) else 0
    apsl = APS_LABEL.get(aps, str(aps))
    g    = row.get("phot_g_mean_mag", np.nan)
    pg   = row.get("model_pgal", np.nan)
    gt   = f"{g:.2f}" if pd.notna(g) else "?"
    pt   = f"{pg:.2f}" if pd.notna(pg) else "?"

    # Blend flags for title
    n_gaia   = int(blend.get("n_gaia_within_1arcsec") or 1)
    n_euclid = int(blend.get("n_euclid_within_1arcsec") or 1)
    g_blend  = bool(blend.get("gaia_is_blend"))
    e_blend  = bool(blend.get("euclid_is_blend"))
    blend_parts = []
    if n_gaia   > 1: blend_parts.append(f"nG={n_gaia}")
    if n_euclid > 1: blend_parts.append(f"nE={n_euclid}")
    if g_blend or e_blend: blend_parts.append("★blend")
    blend_str = f"  [{', '.join(blend_parts)}]" if blend_parts else ""

    # WISE photometry — from the blend table, else fall back to the row's CatWISE cols (flat
    # CSV carries w1/w2/qso_id for every pick, so sources absent from the catwise crossmatch
    # table still get their W1-W2 / WISE id).
    def _finite(v):
        try:
            return v is not None and np.isfinite(float(v))
        except (TypeError, ValueError):
            return False
    w1 = blend.get("w1mpro"); w2 = blend.get("w2mpro")
    if not (_finite(w1) and _finite(w2)):
        w1, w2 = row.get("w1"), row.get("w2")
    w_str = (f"  W1-W2={float(w1)-float(w2):.2f}" if (_finite(w1) and _finite(w2)) else "")
    wise_id = blend.get("wise_id") or row.get("qso_id", "") or ""

    e_ra  = blend.get("e_ra")  or row.get("qso_ra")
    e_dec = blend.get("e_dec") or row.get("qso_dec")
    cl = cluster_label if cluster_label is not None else f"Cluster {int(row['cluster'])}"
    title = (f"QSO Euclid {euclid_release(row)} + WISE  {cl}  ({apsl}){blend_str}\n"
             f"E_RA={float(e_ra):.4f}  E_Dec={float(e_dec):+.4f}  P(gal)={pt}{w_str}\n"
             f"G={gt}  id={row['sid_str']}"
             + (f"  WISE={wise_id}" if wise_id else ""))

    fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=dpi)
    if arr is not None:
        ax.imshow(np.clip(arr, 0, 1))
    else:
        ax.set_facecolor("#111")
        ax.text(0.5, 0.5, "no Euclid\ntile", color="w", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=7, color="black", pad=10)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


# ── Contact sheet ──────────────────────────────────────────────────────────────

PER_SHEET = 80   # max stamps per contact sheet (8 cols x 10 rows); >this is paginated


def contact_sheet(items: list, out: Path, cluster: int, per_sheet: int = PER_SHEET):
    """Per-cluster Euclid+WISE sheet(s), sorted by P(gal). Paginated: when a cluster has
    more than `per_sheet` stamps they are split across contact_sheet_cluster{c}_p1.png,
    _p2.png, ... (single page keeps the plain contact_sheet_cluster{c}.png name).
    Mirrors render_q1_cutout._make_contact_sheet. `items` = list of (baked_array, pgal)."""
    items = sorted(items, key=lambda t: -t[1])
    n = len(items)
    if n == 0:
        return
    pages = [items[i:i + per_sheet] for i in range(0, n, per_sheet)]
    for pi, page in enumerate(pages, 1):
        page_out = out if len(pages) == 1 else out.with_name(f"{out.stem}_p{pi}{out.suffix}")
        _render_sheet(page, page_out, cluster, n, pi, len(pages))


def _render_sheet(items: list, out: Path, cluster: int, n_total: int, page: int, npages: int):
    """Render one contact-sheet page (`items` already sorted, at most PER_SHEET of them)."""
    ncols = 8
    nrows = math.ceil(len(items) / ncols)
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
    header_in = 1.0
    top_frac = 1.0 - header_in / fig_h_in
    fig.subplots_adjust(top=top_frac, bottom=0.01, left=0.02, right=0.99, hspace=0.05, wspace=0.05)
    cbar_ax = fig.add_axes([0.06, top_frac + 0.30 / fig_h_in, 0.20, 0.10 / fig_h_in])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=plt.Normalize(0, 1)),
                        cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 0.5, 1]); cbar.set_ticklabels(["star", "0.5", "galaxy"])
    cbar.ax.tick_params(labelsize=7, pad=1)
    cbar_ax.set_title("border = XGBoost P(gal)", fontsize=8, pad=3, loc="left")

    # Legend for ring colours
    legend_lines = [
        plt.Line2D([0], [0], color=WISE_RING,     lw=0, marker="x", ms=9, mew=1.5, label="WISE centroid"),
        plt.Line2D([0], [0], color=EUCLID_RING,   lw=0, marker="+", ms=9, mew=1.5, label="Secondary Euclid (<1\")"),
        plt.Line2D([0], [0], color=GAIA_RING,     lw=0, marker="o", ms=8, markerfacecolor="none", mew=1.2, label="Gaia matched (<1\")"),
        plt.Line2D([0], [0], color=GAIA_FOV_RING, lw=0, marker="o", ms=8, markerfacecolor="none", mew=1.2, label="Gaia in FOV (>1\")"),
    ]
    fig.legend(handles=legend_lines, loc="upper right",
               bbox_to_anchor=(0.99, top_frac + 0.90 / fig_h_in),
               fontsize=7, framealpha=0.9, ncol=4)

    page_str = "" if npages == 1 else f", page {page}/{npages}"
    fig.suptitle(f"Euclid+WISE quasars — Cluster {cluster} — Euclid VIS+NISP "
                 f"({n_total} sources total{page_str}, sorted by P(gal))", fontsize=13, fontweight="bold",
                 x=0.5, y=top_frac + 0.72 / fig_h_in, color=CLUSTER_COLOURS.get(cluster, "k"))
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  contact sheet: {out}  ({len(items)} stamps)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n",      type=int,   default=40,   help="sources per cluster (default 40)")
    ap.add_argument("--size",   type=float, default=6.0,  help="cutout FOV arcsec (default 6)")
    ap.add_argument("--gmax",   type=float, default=21.5, help="prefer G < gmax sources")
    ap.add_argument("--seed",   type=int,   default=42)
    ap.add_argument("--no-gaia", action="store_true",
                    help="skip all WSDB queries (no rings, no blend info)")
    args = ap.parse_args()
    if not CSV.exists():
        print(f"{CSV} not found — run analyze_umap_qso.py --catalog euclidwise first.")
        return 1

    user = os.environ.get("PGUSER", "siwei_zhao_2025")
    samp = load_sample(args.n, args.gmax, args.seed)

    blend_info     = {}
    gaia           = {}
    euclid_sec     = {}
    if not args.no_gaia:
        blend_info = fetch_blend_info(samp, user)
        gaia       = fetch_gaia_positions(samp, args.size, blend_info)
        euclid_sec = fetch_euclid_secondary_positions(samp, blend_info, user)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for c in sorted(samp["cluster"].unique()):
        rows = samp[samp["cluster"] == c].reset_index(drop=True)
        cint = int(c)
        cdir = OUT_DIR / f"cluster{cint}"; cdir.mkdir(parents=True, exist_ok=True)
        for old in cdir.glob("*.png"):
            old.unlink()
        print(f"cluster {cint}: {len(rows)} Euclid cutouts...")
        items = []
        ne = 0
        for i, (_, row) in enumerate(rows.iterrows()):
            sid = row["sid_str"]
            b   = blend_info.get(sid, {})

            # Euclid centroid = cutout centre; WISE + Gaia + secondary Euclid as companions
            e_ra  = b.get("e_ra")  or float(row["qso_ra"])
            e_dec = b.get("e_dec") or float(row["qso_dec"])
            companions = None
            if not args.no_gaia:
                # secondary Euclid: red crosshair; WISE: magenta x; Gaia: blue rings
                sec_euclid = [(ra, dec, lbl, EUCLID_RING, "crosshair")
                              for ra, dec, lbl, *_ in euclid_sec.get(sid, [])]
                companions = (list(gaia.get(sid, []))
                              + sec_euclid
                              + [(float(row["qso_ra"]), float(row["qso_dec"]), None, WISE_RING, "x")])

            arr   = fetch_euclid(row, args.size, companions=companions,
                                 center_ra=e_ra, center_dec=e_dec)
            ne   += arr is not None
            baked = render_to_array(row, arr, b)
            fn    = f"cutout_{i:04d}.png"
            plt.imsave(cdir / fn, np.clip(baked, 0, 1))
            items.append((baked, float(row["model_pgal"])))

            # Manifest row — blend columns as pipe-separated strings where array
            def _arr_str(v):
                return "|".join(str(x) for x in (v or [])) if v else ""

            manifest.append({
                "cluster":         cint,
                "cutout_file":     f"cluster{cint}/{fn}",
                "gaia_source_id":  sid,
                "aps":             int(row.get("astrometric_params_solved", 0) or 0),
                "qso_ra":          row["qso_ra"],
                "qso_dec":         row["qso_dec"],
                "model_pgal":      row["model_pgal"],
                "euclid_ok":       arr is not None,
                "wise_id":         b.get("wise_id", ""),
                "w1mpro":          b.get("w1mpro", ""),
                "w2mpro":          b.get("w2mpro", ""),
                "gaia_sep_arcsec": b.get("gaia_sep_arcsec", ""),
                "n_gaia_within_1arcsec":   b.get("n_gaia_within_1arcsec", ""),
                "all_gaia_ids":    _arr_str(b.get("all_gaia_ids_within_1arcsec")),
                "all_gaia_seps":   _arr_str(b.get("all_gaia_seps_within_1arcsec")),
                "gaia_is_blend":   b.get("gaia_is_blend", ""),
                "euclid_sep_arcsec": b.get("euclid_sep_arcsec", ""),
                "n_euclid_within_1arcsec": b.get("n_euclid_within_1arcsec", ""),
                "all_euclid_ids":  _arr_str(b.get("all_euclid_ids_within_1arcsec")),
                "all_euclid_seps": _arr_str(b.get("all_euclid_seps_within_1arcsec")),
                "euclid_is_blend": b.get("euclid_is_blend", ""),
                "n_gaia_in_fov":   len(gaia.get(sid, [])),
                "n_euclid_secondary": len(euclid_sec.get(sid, [])),
            })

        contact_sheet(items, OUT_DIR / f"contact_sheet_cluster{cint}.png", cint)
        print(f"  cluster{cint}: {ne}/{len(items)} Euclid ok")

    pd.DataFrame(manifest).to_csv(OUT_DIR / "manifest.csv", index=False)
    print(f"  -> {OUT_DIR / 'manifest.csv'}  ({len(manifest)} sources)")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
