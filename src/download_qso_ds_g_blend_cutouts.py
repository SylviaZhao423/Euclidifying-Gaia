#!/usr/bin/env python3
"""Cutouts for the DESI+SDSS QSO "blend" candidates (n_gaia_within_1arcsec > 1).

Purpose: sanity-check the blend count. A QSO is flagged as a blend when >1 Gaia
source sits within 1" of it. Those extra detections are EITHER a genuine second
object OR a Gaia PSF-split artifact (one source deblended into several spurious
detections, typically < 0.3" apart). To tell them apart we need imaging with the
Gaia detection positions overlaid:

    * two detections on two visibly distinct objects  -> real blend
    * two detections on one unresolved blob           -> Gaia split artifact

None of the blends fall in the small Euclid footprint (all 13 Euclid-imaged QSOs
are n_gaia = 1), so we use the DESI Legacy Survey cutout service, which covers
the whole SDSS footprint. Each Gaia source within 1" is drawn as a labelled
circle; the QSO catalogue position is a crosshair.

Usage:
    conda run -n base python3 src/download_ds_g_blend_cutouts.py            # Legacy/DECaLS sheet
    ... --survey both                                                     # paired Legacy-over-SDSS sheet
    ... --survey sdss                                                     # SDSS only
    ... --html                                                            # + interactive hover-cutout UMAP
    ... --only-html                                                       # just the interactive UMAP
    ... --limit 40                                                        # top-40 by companion sep
    ... --size 14                                                         # tune FOV
    ... --rebuild-companions                                              # re-query WSDB (needs VPN)

WSDB auth via ~/.pgpass  (PGUSER=siwei_zhao_2025). Never pass password=.
"""
from __future__ import annotations

import argparse
import io
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import requests
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
MATCHES_CSV = PROJECT_DIR / "output" / "umap" / "qso" / "arrays" / "qso_gaia_matches_desisdss.csv"
OUT_DIR = PROJECT_DIR / "output" / "qso_cutouts" / "ds_g_blend"
MANIFEST = OUT_DIR / "manifest.csv"   # single comprehensive file (companions folded in; WSDB cache)
# Shared, hidden Legacy/SDSS download cache (used by blend + web + euclid_wise cutouts);
# kept out of the deliverable folders so it doesn't clutter them.
IMG_CACHE = PROJECT_DIR / "output" / "qso_cutouts" / ".ls_jpg_cache"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB = "wsdb"
GAIA_TABLE = "gaia_dr3.gaia_source"
MATCH_RADIUS_ARCSEC = 1.0

CLUSTER_COLOURS = {0: "#e6194b", 1: "#4363d8", 2: "#f58231", 3: "#3cb44b", 4: "#911eb4"}
APS_LABEL = {3: "2-param", 31: "5-param", 95: "6-param"}
LS_CUTOUT = "https://www.legacysurvey.org/viewer/cutout.jpg"
SDSS_CUTOUT = "https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg"


# --------------------------------------------------------------------------- #
#  data                                                                        #
# --------------------------------------------------------------------------- #
def load_blends() -> pd.DataFrame:
    """The blend candidates from the SDSS projection, minus the qso_id=-1
    sentinel row (a bad catalogue entry that q3c-matches a dense field)."""
    df = pd.read_csv(MATCHES_CSV, dtype={"qso_id": "int64", "gaia_source_id": "int64"})
    b = df[df["n_gaia_within_1arcsec"] > 1].copy()
    n0 = len(b)
    b = b[b["qso_id"] != -1]
    if len(b) < n0:
        print(f"  dropped {n0 - len(b)} sentinel row(s) (qso_id=-1, dense-field artifact)")
    return b.reset_index(drop=True)


def fetch_companions(blends: pd.DataFrame) -> pd.DataFrame:
    """Every Gaia source within 1" of each blend QSO (positions + G mag + sep),
    via a temp-table q3c join. Cached to COMPANIONS_CSV."""
    import psycopg2

    user = os.environ.get("PGUSER", "siwei_zhao_2025")
    r_deg = MATCH_RADIUS_ARCSEC / 3600.0
    rows = blends[["qso_id", "qso_ra", "qso_dec"]].drop_duplicates("qso_id")
    print(f"  querying Gaia neighbours for {len(rows)} blend QSOs (VPN required)...")
    conn = None
    try:
        conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user,
                                connect_timeout=30, keepalives=1, keepalives_idle=30,
                                keepalives_interval=10, keepalives_count=5)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 0")
        cur.execute("CREATE TEMP TABLE _blend_pos (qso_id bigint, ra double precision, dec double precision)")
        from psycopg2.extras import execute_values
        execute_values(cur,
                       "INSERT INTO _blend_pos (qso_id, ra, dec) VALUES %s",
                       list(rows.itertuples(index=False, name=None)))
        cur.execute("ANALYZE _blend_pos")          # else the planner seq-scans 1.8B gaia rows
        cur.execute(f"""
            SELECT b.qso_id, g.source_id, g.ra, g.dec, g.phot_g_mean_mag,
                   g.astrometric_params_solved,
                   q3c_dist(b.ra, b.dec, g.ra, g.dec) * 3600.0 AS sep_arcsec
            FROM _blend_pos b, {GAIA_TABLE} g
            WHERE q3c_join(b.ra, b.dec, g.ra, g.dec, {r_deg})
            ORDER BY b.qso_id, sep_arcsec
        """)
        comp = pd.DataFrame(cur.fetchall(),
                            columns=["qso_id", "source_id", "ra", "dec",
                                     "phot_g_mean_mag", "astrometric_params_solved", "sep_arcsec"])
        cur.close()
    finally:
        if conn is not None:
            conn.close()
    print(f"  {len(comp)} Gaia detections from WSDB")
    return comp


# --------------------------------------------------------------------------- #
#  imaging                                                                     #
# --------------------------------------------------------------------------- #
def _download_jpg(url: str, params: dict, fn: Path, retries: int = 3) -> Image.Image | None:
    """Fetch a JPEG cutout to disk cache and return it (both surveys serve
    N-up/E-left RGB JPEGs, so the Gaia overlays line up either way). Retries transient
    network failures (slow/flaky Legacy) a few times; a valid non-image response
    (out-of-coverage) returns None immediately (no point retrying that)."""
    fn.parent.mkdir(parents=True, exist_ok=True)
    if not fn.exists():
        for attempt in range(retries):
            try:
                r = requests.get(url, params=params, timeout=30)
                if r.status_code == 200 and r.content.startswith(b"\xff\xd8"):
                    fn.write_bytes(r.content)
                    break
                return None                      # served a response, but no image (coverage)
            except requests.RequestException:
                if attempt == retries - 1:
                    return None                  # transient failure persisted
    try:
        return Image.open(fn).convert("RGB")
    except Exception:
        return None


def fetch_ls_jpg(ra: float, dec: float, size_arcsec: float, layer: str = "ls-dr10",
                 out_px: int = 256) -> Image.Image | None:
    """DESI Legacy Survey (DECaLS) RGB cutout. pixscale set so the image is
    ~out_px on a side at the requested FOV."""
    fn = IMG_CACHE / "legacy" / f"{layer}_{ra:.5f}_{dec:+.5f}_{size_arcsec:.1f}.jpg"
    return _download_jpg(LS_CUTOUT, dict(ra=f"{ra:.6f}", dec=f"{dec:.6f}", layer=layer,
                                         pixscale=f"{size_arcsec / out_px:.4f}",
                                         size=out_px), fn)


def fetch_sdss_jpg(ra: float, dec: float, size_arcsec: float, layer: str = "dr16",
                   out_px: int = 256) -> Image.Image | None:
    """SDSS SkyServer (2.5m) RGB cutout of the same sky position — a different
    telescope from Legacy/DECaLS, for side-by-side comparison. scale is
    arcsec/pixel."""
    fn = IMG_CACHE / "sdss" / f"{layer}_{ra:.5f}_{dec:+.5f}_{size_arcsec:.1f}.jpg"
    return _download_jpg(SDSS_CUTOUT, dict(ra=f"{ra:.6f}", dec=f"{dec:.6f}",
                                           scale=f"{size_arcsec / out_px:.4f}",
                                           width=out_px, height=out_px), fn)


SURVEYS = {
    "legacy": (fetch_ls_jpg, "Legacy Survey / DECaLS  [ls-dr10]"),
    "sdss":   (fetch_sdss_jpg, "SDSS  [dr16]"),
}


# --------------------------------------------------------------------------- #
#  rendering                                                                   #
# --------------------------------------------------------------------------- #
def _offsets_arcsec(ra, dec, ra0, dec0):
    """East-left tangent-plane offsets (arcsec) of (ra,dec) from centre."""
    dra = (ra - ra0) * math.cos(math.radians(dec0)) * 3600.0
    ddec = (dec - dec0) * 3600.0
    return dra, ddec


def pgal_color(p):
    """Outer-border colour by XGB P(gal) (RdYlBu_r: red=galaxy, blue=star)."""
    return plt.cm.RdYlBu_r(float(np.clip(p, 0, 1))) if pd.notna(p) and np.isfinite(p) else "0.5"


def fold_companions(blends: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    """Per-blend manifest: blend columns + each blend's Gaia detections folded into
    ';'-joined columns (comp_source_ids/ras/decs/gmags/seps), plus comp_sep (the 2nd
    detection's separation). One row per cutout."""
    folded = ["comp_source_ids", "comp_ras", "comp_decs", "comp_gmags", "comp_seps", "comp_sep"]
    blends = blends.drop(columns=[c for c in folded if c in blends.columns], errors="ignore")
    g = comp.sort_values(["qso_id", "sep_arcsec"])
    agg = g.groupby("qso_id").agg(
        comp_source_ids=("source_id", lambda s: ";".join(s.astype("int64").astype(str))),
        comp_ras=("ra", lambda s: ";".join(f"{v:.6f}" for v in s)),
        comp_decs=("dec", lambda s: ";".join(f"{v:+.6f}" for v in s)),
        comp_gmags=("phot_g_mean_mag", lambda s: ";".join("" if pd.isna(v) else f"{v:.3f}" for v in s)),
        comp_seps=("sep_arcsec", lambda s: ";".join(f"{v:.4f}" for v in s)),
        comp_sep=("sep_arcsec", lambda s: s.iloc[1] if len(s) > 1 else np.nan),
    ).reset_index()
    return blends.merge(agg, on="qso_id", how="left")


def explode_companions(man: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the per-detection comp DataFrame from a folded manifest (for rendering)."""
    rows = []
    for _, r in man.iterrows():
        sids = str(r["comp_source_ids"]).split(";")
        ras, decs = str(r["comp_ras"]).split(";"), str(r["comp_decs"]).split(";")
        gms, seps = str(r["comp_gmags"]).split(";"), str(r["comp_seps"]).split(";")
        for i in range(len(sids)):
            rows.append({"qso_id": int(r["qso_id"]), "source_id": int(sids[i]),
                         "ra": float(ras[i]), "dec": float(decs[i]),
                         "phot_g_mean_mag": float(gms[i]) if gms[i] else np.nan,
                         "sep_arcsec": float(seps[i])})
    return pd.DataFrame(rows)


def render_blend_stamp(ax, row: pd.Series, comp: pd.DataFrame, img: Image.Image | None,
                       size_arcsec: float, show_title: bool = True,
                       show_scalebar: bool = True, show_border: bool = True) -> None:
    """One blend: survey image, Gaia detections as thin circles, QSO catalogue
    position as a small gapped crosshair (open centre, so it does not cover the
    source), corner labels, scale bar."""
    half = size_arcsec / 2.0
    ext = [half, -half, -half, half]           # east (RA+) to the left
    if img is not None:
        ax.imshow(np.asarray(img), extent=ext, origin="upper")
    else:
        ax.set_facecolor("#111111")
        ax.text(0, 0, "no\nimage", color="w", ha="center", va="center", fontsize=7)

    ra0, dec0 = float(row["qso_ra"]), float(row["qso_dec"])
    # gapped crosshair on the QSO position: four short ticks, open centre
    gap, tick = 0.28, 0.62
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        ax.plot([dx * gap, dx * tick], [dy * gap, dy * tick], "-",
                color="#ff3b3b", lw=0.7, zorder=6)

    # Gaia detections: thin cyan circle each; "G=mag / source_id" label parked at
    # the stamp corner matching the detection's own quadrant, clear of the source.
    # (x-axis inverted: east/RA+ is to the left, i.e. visual-left = dra > 0.)
    g = comp[comp["qso_id"] == row["qso_id"]].sort_values("sep_arcsec")
    # String id column: iterrows() upcasts an all-numeric row to float64, truncating
    # 19-digit int64 source_ids (>2^53). Label from this string, not int(s["source_id"]).
    g = g.assign(_sid=g["source_id"].astype("int64").astype(str))
    inset = 0.35
    used: dict = {}
    for _, s in g.iterrows():
        dra, ddec = _offsets_arcsec(float(s["ra"]), float(s["dec"]), ra0, dec0)
        ax.add_patch(plt.Circle((dra, ddec), 0.18, fill=False, ec="#00e5ff",
                                lw=0.7, zorder=5))
        cx = (half - inset) if dra >= 0 else (-half + inset)
        ha = "left" if dra >= 0 else "right"
        cy0 = (half - inset) if ddec >= 0 else (-half + inset)
        va = "top" if ddec >= 0 else "bottom"
        k = used.get((ha, va), 0); used[(ha, va)] = k + 1          # stack if same corner
        cy = cy0 - k * 1.7 if va == "top" else cy0 + k * 1.7
        gm = s["phot_g_mean_mag"]
        lbl = (f"G={gm:.1f}\n" if pd.notna(gm) else "G=?\n") + s["_sid"]
        t = ax.text(cx, cy, lbl, color="#00e5ff", fontsize=8.0, ha=ha, va=va,
                    zorder=7, linespacing=0.9)
        t.set_path_effects([pe.Stroke(linewidth=1.6, foreground="black"), pe.Normal()])

    if show_scalebar:   # scale bar (1"), bottom-right, clear of corner labels
        ax.plot([-half + 1.3, -half + 0.3], [-half + 0.7, -half + 0.7], "-", color="w", lw=2.0)
        ax.text(-half + 0.8, -half + 1.05, '1"', color="w", fontsize=7, ha="center", va="bottom")

    cl = int(row["cluster"])
    pgal = row.get("model_pgal", float("nan"))
    if show_title:
        seps = g["sep_arcsec"].values
        comp_sep = seps[1] if len(seps) > 1 else float("nan")   # companion separation
        aps = int(row["astrometric_params_solved"])
        apsl = {3: "2-param", 31: "5-param", 95: "6-param"}.get(aps, str(aps))
        pgtxt = f"{pgal:.2f}" if pd.notna(pgal) else "?"
        title = (f"QSO Blend  Cluster {cl}  ({apsl})\n"
                 f"RA={float(row['qso_ra']):.4f}  Dec={float(row['qso_dec']):+.4f}  P(gal)={pgtxt}\n"
                 f"Sep={comp_sep:.2f}\"  qso_id={int(row['qso_id'])}")
        ax.set_title(title, fontsize=8, color="black", pad=10)  # black + clear gap above the image
    ax.set_xlim(half, -half); ax.set_ylim(-half, half)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        if show_border:
            sp.set_edgecolor(pgal_color(pgal)); sp.set_linewidth(2.4)
        else:
            sp.set_visible(False)


def _sorted_blends(blends: pd.DataFrame, comp: pd.DataFrame,
                   limit: int | None) -> pd.DataFrame:
    """Blends with a `comp_sep` column, sorted largest-separation first (most
    likely real blends on top), optionally truncated to the top-N."""
    csep = (comp.sort_values("sep_arcsec").groupby("qso_id")["sep_arcsec"]
            .apply(lambda s: s.iloc[1] if len(s) > 1 else np.nan))
    out = blends.assign(comp_sep=blends["qso_id"].map(csep)).sort_values(
        "comp_sep", ascending=False, na_position="last").reset_index(drop=True)
    return out.head(limit) if limit else out


def _suptitle(fig, blends, n, extra):
    n_sub03 = int((blends["comp_sep"] < 0.3).sum())
    n_over05 = int((blends["comp_sep"] > 0.5).sum())
    fig.suptitle(
        f"DESI+SDSS QSO blend candidates (>1 Gaia within 1\")  —  {n} shown, "
        f"sorted by companion sep   ·   {extra}\n"
        f"red + = QSO position   cyan circle = Gaia detection   "
        f"{n_sub03} have companion <0.3\" (likely Gaia split) · {n_over05} > 0.5\"",
        fontsize=11, y=0.997)


def _order_blends(blends: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    """Contact-sheet / cutout order: by UMAP cluster, then descending XGB P(gal)."""
    out = blends.sort_values(["cluster", "model_pgal"],
                             ascending=[True, False]).reset_index(drop=True)
    return out.head(limit) if limit else out


def _pgal_cbar(fig, fig_h):
    """Small horizontal 'border = XGB P(gal)' key in the fixed top band."""
    cax = fig.add_axes([0.06, 1 - 0.42 / fig_h, 0.20, 0.14 / fig_h])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=plt.Normalize(0, 1)),
                      cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["star", "0.5", "galaxy"])
    cb.ax.tick_params(labelsize=6, pad=1)
    cax.set_title("border = XGB P(gal)", fontsize=7, loc="left", pad=2)


def render_blend_to_array(row: pd.Series, comp: pd.DataFrame, img: Image.Image | None,
                          size_arcsec: float, dpi: int = 100) -> np.ndarray:
    """Render one blend stamp — title, crosshair, Gaia circles/labels, scale bar all
    BAKED into a numpy image (no border) — so the contact sheet can tile them tightly
    without matplotlib titles overlapping (exactly how the Euclid sheet is built)."""
    fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=dpi)   # square, like the Euclid renderer
    render_blend_stamp(ax, row, comp, img, size_arcsec, show_title=True,
                       show_scalebar=True, show_border=False)
    fig.tight_layout()                                    # default pad → small title→image gap (matches Euclid)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def contact_sheet(stamps: dict, survey_lbl: str, out_dir: Path, suffix: str = "") -> None:
    """PER-CLUSTER blend sheets: one contact_sheet_cluster{c}{suffix}.png per UMAP cluster,
    stamps sorted by descending XGB P(gal), outer box coloured by P(gal), cluster-coloured
    suptitle. `stamps` = {qso_id: (image_array, aps, pgal, cluster)}."""
    if not stamps:
        print("  no stamps to sheet."); return
    by_cl: dict = {}
    for (img, _ap, pg, cl) in stamps.values():
        by_cl.setdefault(cl, []).append((img, pg))
    out_dir.mkdir(parents=True, exist_ok=True)
    for cl in sorted(by_cl):
        items = sorted(by_cl[cl], key=lambda t: -t[1])
        n = len(items)
        ncols = 8
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for i, (img, pg) in enumerate(items):
            ax = axes[i // ncols][i % ncols]
            ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(img)
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
        fig.suptitle(f"DESI+SDSS QSO blend candidates — Cluster {cl} — {survey_lbl}  "
                     f"({n}, >1 Gaia within 1\", sorted by P(gal))", fontsize=13,
                     fontweight="bold", x=0.5, y=top_frac + 0.72 / fig_h_in,
                     color=CLUSTER_COLOURS.get(cl, "k"))
        out = out_dir / f"contact_sheet_cluster{cl}{suffix}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out}")


def paired_contact_sheet(blends: pd.DataFrame, comp: pd.DataFrame,
                         size_arcsec: float, limit: int | None) -> None:
    """Side-by-side comparison sheet: each QSO gets a stacked pair of stamps —
    Legacy/DECaLS on top, SDSS directly below (same object, same overlays) — so
    the two surveys can be compared cutout-by-cutout, like the upper/lower panel
    layout used for the Q1-feature UMAPs."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    blends = _sorted_blends(blends, comp, limit)
    n = len(blends)
    ncols = 8
    nblocks = math.ceil(n / ncols)
    print(f"  [paired] rendering {n} QSOs as Legacy/SDSS pairs "
          f"({nblocks} blocks x {ncols}), FOV={size_arcsec}\"...")

    fig = plt.figure(figsize=(ncols * 1.7, nblocks * 4.0))
    outer = GridSpec(nblocks, 1, figure=fig, hspace=0.42,
                     top=0.985, bottom=0.006, left=0.045, right=0.997)
    for b in range(nblocks):
        inner = GridSpecFromSubplotSpec(2, ncols, subplot_spec=outer[b],
                                        hspace=0.22, wspace=0.06)
        for c in range(ncols):
            i = b * ncols + c
            axt = fig.add_subplot(inner[0, c])
            axb = fig.add_subplot(inner[1, c])
            if i >= n:
                axt.axis("off"); axb.axis("off"); continue
            row = blends.iloc[i]
            ra, dec = float(row["qso_ra"]), float(row["qso_dec"])
            render_blend_stamp(axt, row, comp, fetch_ls_jpg(ra, dec, size_arcsec),
                               size_arcsec, show_title=True, show_scalebar=False)
            render_blend_stamp(axb, row, comp, fetch_sdss_jpg(ra, dec, size_arcsec),
                               size_arcsec, show_title=False, show_scalebar=True)
            if c == 0:
                axt.set_ylabel("Legacy", fontsize=7)
                axb.set_ylabel("SDSS", fontsize=7)
        if (b + 1) % 4 == 0:
            print(f"    block {b + 1}/{nblocks}")

    _suptitle(fig, blends, n, "each QSO: upper = Legacy/DECaLS, lower = SDSS")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "qso_blend_contact_sheet_paired.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def sep_histogram(blends: pd.DataFrame, comp: pd.DataFrame) -> None:
    """Distribution of companion separations — the quantitative version of the
    real-blend-vs-split question."""
    csep = (comp.sort_values("sep_arcsec").groupby("qso_id")["sep_arcsec"]
            .apply(lambda s: s.iloc[1] if len(s) > 1 else np.nan)).dropna()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(csep, bins=np.arange(0, 1.01, 0.05), color="#4363d8", edgecolor="k")
    ax.axvline(0.3, color="#e6194b", ls="--", lw=1.5, label='0.3" (Gaia-split scale)')
    ax.axvline(0.5, color="#f58231", ls="--", lw=1.5, label='0.5"')
    n_sub03 = int((csep < 0.3).sum())
    ax.set_xlabel('companion separation (")'); ax.set_ylabel("number of blend QSOs")
    ax.set_title(f"Blend companion separations  ({len(csep)} QSOs; "
                 f"{n_sub03} = {100*n_sub03/len(csep):.0f}% below 0.3\")")
    ax.legend()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "qso_blend_sep_hist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# --------------------------------------------------------------------------- #
#  interactive UMAP with hover cutouts                                         #
# --------------------------------------------------------------------------- #
def _stamp_b64(row: pd.Series, comp: pd.DataFrame, img: Image.Image | None,
               size_arcsec: float, px: int = 230) -> str:
    """Render a single full-format blend stamp (title, gapped crosshair, Gaia
    circles/labels, scale bar) and return it as a base64 PNG string."""
    import base64
    fig, ax = plt.subplots(figsize=(px / 100, px / 100 * 1.15), dpi=100)
    render_blend_stamp(ax, row, comp, img, size_arcsec, show_title=True, show_scalebar=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>QSO blends on the Q1 UMAP</title>
<style>
  html, body {{ margin:0; height:100%; font-family:-apple-system,Segoe UI,Arial,sans-serif; background:#fafafa; }}
  #wrap {{ display:flex; height:100vh; width:100vw; }}
  #plotcol {{ flex:1 1 0; min-width:0; height:100%; }}
  #plotcol .plotly-graph-div {{ width:100% !important; height:100% !important; }}
  #side {{ flex:0 0 300px; padding:12px; border-left:1px solid #ddd; overflow:auto;
           display:flex; flex-direction:column; align-items:center; gap:6px; background:#fff; }}
  #side h3 {{ margin:4px 0; font-size:14px; }}
  #side .lab {{ font-size:12px; font-weight:bold; color:#333; align-self:flex-start; }}
  #side img {{ width:260px; border:1px solid #ccc; background:#000; }}
  #hint {{ color:#888; font-size:12px; margin-top:30px; text-align:center; }}
</style></head><body>
<div id="wrap">
  <div id="plotcol">{plot_div}</div>
  <div id="side">
    <h3>hover a red star</h3>
    <div class="lab">Legacy / DECaLS</div><img id="legImg" alt="">
    <div class="lab">SDSS</div><img id="sdssImg" alt="">
    <div id="hint">Move the cursor over any of the 154 blend<br>stars to see its two cutouts.</div>
  </div>
</div>
<script>
  const LEG = {leg_json};
  const SDSS = {sdss_json};
  function wireHover() {{
    const gd = document.getElementById('umap');
    if (!gd || !gd.on) {{ return setTimeout(wireHover, 200); }}   // wait for Plotly init
    const show = function(e) {{
      if (!e.points || !e.points.length) return;
      const p = e.points[0];
      if (p.customdata === undefined || p.customdata === null) return;
      const i = p.customdata;
      document.getElementById('legImg').src = 'data:image/png;base64,' + LEG[i];
      document.getElementById('sdssImg').src = 'data:image/png;base64,' + SDSS[i];
      document.getElementById('hint').style.display = 'none';
    }};
    gd.on('plotly_hover', show);
    gd.on('plotly_click', show);   // click also works (handy on trackpads)
  }}
  wireHover();
</script>
</body></html>"""


def build_umap_html(blends: pd.DataFrame, comp: pd.DataFrame, size_arcsec: float) -> None:
    """Interactive HTML: the 154 blends on the Q1 UMAP; hovering a star shows its
    Legacy and SDSS cutouts (full stamp format) in a side panel."""
    import json
    import plotly.graph_objects as go

    blends = _sorted_blends(blends, comp, None)
    if "umap1" not in blends.columns:
        print("  [html] blends lack umap1/umap2 — skipping"); return
    n = len(blends)

    arrays = PROJECT_DIR / "output" / "umap" / "arrays"
    proj_csv = PROJECT_DIR / "output" / "umap" / "qso" / "arrays" / "qso_gaia_matches_desisdss.csv"
    q1 = np.load(arrays / "q1_indiv.npz", allow_pickle=True)
    q1_emb, y = q1["embedding"], q1["y"]
    labels = np.load(arrays / "q1_cluster_labels.npy")
    cluster_ids = sorted(set(labels[labels >= 0]))
    centroids = {c: (q1_emb[labels == c, 0].mean(), q1_emb[labels == c, 1].mean()) for c in cluster_ids}
    galpct = {c: 100 * (y[labels == c] == 1).sum() / (labels == c).sum() for c in cluster_ids}
    allq = pd.read_csv(proj_csv, usecols=["umap1", "umap2"]).to_numpy()

    rng = np.random.default_rng(0)
    def _ds(a, k):
        return a if len(a) <= k else a[rng.choice(len(a), k, replace=False)]
    q1_bg = _ds(q1_emb, 30000)
    qso_bg = _ds(allq, 25000)

    print(f"  [html] rendering {n}x2 hover stamps (Legacy + SDSS)...")
    leg_b64, sdss_b64 = [], []
    for i, (_, row) in enumerate(blends.iterrows()):
        ra, dec = float(row["qso_ra"]), float(row["qso_dec"])
        leg_b64.append(_stamp_b64(row, comp, fetch_ls_jpg(ra, dec, size_arcsec), size_arcsec))
        sdss_b64.append(_stamp_b64(row, comp, fetch_sdss_jpg(ra, dec, size_arcsec), size_arcsec))
        if (i + 1) % 40 == 0:
            print(f"    {i + 1}/{n}")

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=q1_bg[:, 0], y=q1_bg[:, 1], mode="markers",
                               marker=dict(size=2, color="#dddddd"), hoverinfo="skip",
                               name="Q1 (30k of 50k)"))
    fig.add_trace(go.Scattergl(x=qso_bg[:, 0], y=qso_bg[:, 1], mode="markers",
                               marker=dict(size=2, color="#9ecae1", opacity=0.35),
                               hoverinfo="skip", name="all SDSS QSOs (25k)"))
    hov = [f"C{int(r.cluster)}  {APS_LABEL.get(int(r.astrometric_params_solved), '?')}"
           f"  comp sep={r.comp_sep:.2f}\"  id={int(r.qso_id)}" for r in blends.itertuples()]
    fig.add_trace(go.Scattergl(
        x=blends["umap1"], y=blends["umap2"], mode="markers",
        marker=dict(symbol="star", size=12, color="#e6194b",
                    line=dict(color="black", width=0.6)),
        customdata=list(range(n)), text=hov, hovertemplate="%{text}<extra></extra>",
        name=f"blends (>1 Gaia within 1\") — {n}"))
    for c in cluster_ids:
        cx, cy = centroids[c]
        fig.add_annotation(x=cx, y=cy, text=f"<b>C{c}</b><br>{galpct[c]:.0f}%gal",
                           showarrow=False, font=dict(size=12, color=CLUSTER_COLOURS[c]),
                           bgcolor="rgba(255,255,255,0.6)")
    fig.update_layout(
        title="QSO blend candidates on the Q1 UMAP — hover a star for its Legacy + SDSS cutouts",
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white", hovermode="closest")

    plot_div = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="umap")
    html = _HTML_TEMPLATE.format(plot_div=plot_div,
                                 leg_json=json.dumps(leg_b64),
                                 sdss_json=json.dumps(sdss_b64))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "qso_blend_umap.html"
    out.write_text(html, encoding="utf-8")
    print(f"  -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def render_all(order: pd.DataFrame, comp: pd.DataFrame, size_arcsec: float,
               survey: str, save_cutouts: bool) -> tuple[dict, list]:
    """Render every blend (in `order`) to a baked stamp array; optionally save the
    numbered individual cutouts. Returns ({qso_id: (array, aps, pgal, cluster)}, files)."""
    fetch_fn, _ = SURVEYS[survey]
    stamps, files = {}, []
    if save_cutouts:                       # per-cluster folders (cluster{c}/), like euclid_wise/
        for cl in sorted(order["cluster"].unique()):
            cdir = OUT_DIR / f"cluster{int(cl)}"; cdir.mkdir(parents=True, exist_ok=True)
            for old in cdir.glob("*.png"):
                old.unlink()
    per_cl = {}
    for _, row in order.iterrows():
        cl = int(row["cluster"])
        img = fetch_fn(float(row["qso_ra"]), float(row["qso_dec"]), size_arcsec)
        arr = render_blend_to_array(row, comp, img, size_arcsec)
        stamps[int(row["qso_id"])] = (arr, int(row["astrometric_params_solved"]),
                                      float(row["model_pgal"]), cl)
        if save_cutouts:
            j = per_cl.get(cl, 0); per_cl[cl] = j + 1        # number within cluster, by P(gal)
            rel = f"cluster{cl}/cutout_{j:04d}.png"
            plt.imsave(OUT_DIR / rel, np.clip(arr, 0, 1))
            files.append(rel)
    if save_cutouts:
        print(f"  -> {len(files)} individual Legacy cutouts in {OUT_DIR}/cluster*/")
    return stamps, files


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=float, default=12.0, help="cutout FOV in arcsec (default 12)")
    ap.add_argument("--survey", choices=["legacy", "sdss", "both"], default="legacy",
                    help="imaging for the contact sheet: 'legacy' (DECaLS) or 'sdss' "
                         "= one standalone sheet; 'both' = both standalone sheets PLUS "
                         "a paired sheet stacking Legacy over SDSS per QSO (default legacy)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only render the top-N blends by companion separation")
    ap.add_argument("--rebuild-companions", action="store_true",
                    help="re-query Gaia neighbours from WSDB (needs VPN), ignoring the manifest cache")
    ap.add_argument("--companions-only", action="store_true",
                    help="ONLY fetch the blends' Gaia neighbours from WSDB and write the "
                         "manifest cache, then exit (no Legacy imagery). Run this with the "
                         "VPN ON; then re-run with the VPN OFF to render from Legacy.")
    ap.add_argument("--html", action="store_true",
                    help="also build an interactive UMAP HTML where hovering a "
                         "blend star shows its Legacy + SDSS cutouts")
    ap.add_argument("--only-html", action="store_true",
                    help="build only the interactive HTML (skip the static sheets)")
    args = ap.parse_args()

    # --companions-only: WSDB-only pass — fetch neighbours, write the manifest cache, exit.
    # (Legacy and WSDB need opposite VPN states, so this decouples them: run companions-only
    #  VPN-ON, then re-run VPN-OFF to render from the cache.)
    if args.companions_only:
        blends = load_blends()
        print(f"blend candidates: {len(blends)}  — fetching Gaia neighbours from WSDB...")
        comp = fetch_companions(blends)
        man = fold_companions(blends, comp)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        man.to_csv(MANIFEST, index=False)
        print(f"  -> companion cache written: {MANIFEST}  ({len(man)} blends, "
              f"{len(comp)} Gaia detections)\n  Now re-run VPN-OFF (no flag) to render Legacy cutouts.")
        return 0

    # manifest.csv doubles as the WSDB cache: read it (companions folded in) if present
    if MANIFEST.exists() and not args.rebuild_companions:
        blends = pd.read_csv(MANIFEST, dtype={"qso_id": "int64", "gaia_source_id": "int64"})
        comp = explode_companions(blends)
        print(f"  manifest cache: {MANIFEST.name} ({len(blends)} blends, {len(comp)} Gaia detections)")
    else:
        blends = load_blends()
        print(f"blend candidates: {len(blends)}")
        comp = fetch_companions(blends)

    if not args.only_html:
        order = _order_blends(blends, args.limit)
        # render each blend once (title baked in); save numbered individuals + tile the sheet
        print(f"  rendering {len(order)} blend stamps (Legacy)...")
        stamps, files = render_all(order, comp, args.size, "legacy", save_cutouts=True)
        contact_sheet(stamps, SURVEYS["legacy"][1], OUT_DIR)   # per-cluster sheets
        order = order.copy(); order["cutout_file"] = files
        # ONE comprehensive manifest: cutout_file + blend metadata + folded companions
        man = fold_companions(order, comp)
        keep = ["cutout_file", "qso_catalog", "qso_id", "gaia_source_id", "qso_ra", "qso_dec",
                "qso_z", "phot_g_mean_mag", "astrometric_params_solved", "model_pgal", "cluster",
                "n_gaia_within_1arcsec", "comp_sep", "comp_source_ids", "comp_ras", "comp_decs",
                "comp_gmags", "comp_seps"]
        man[[c for c in keep if c in man.columns]].to_csv(MANIFEST, index=False)
        print(f"  -> comprehensive manifest.csv ({len(man)} blends)")
        # SDSS comparison sheet only on request (--survey sdss|both)
        if args.survey in ("sdss", "both"):
            sd_stamps, _ = render_all(order, comp, args.size, "sdss", save_cutouts=False)
            contact_sheet(sd_stamps, SURVEYS["sdss"][1], OUT_DIR, suffix="_sdss")
    if args.html or args.only_html:
        build_umap_html(blends, comp, args.size)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
