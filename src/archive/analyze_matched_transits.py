#!/usr/bin/env python3
"""
Combined Gaia transit-feature analysis for Euclid Q1 + ERO fields.

Runs all four analyses in sequence, sharing one Gaia field query and one
scanning-law map build:

  1. Mollweide sky map of astrometric_matched_transits_ratio
     + per-field cutout panels  (was analyze_match_ratio.py)

  2. Per-field cutout panels colored by astrometric_matched_transits
     and matched_transits  (was analyze_matched_transits_cutouts.py)

  3. Per-field cutout panels colored by scan-normalised transit features
     (was analyze_transit_efficiency_cutouts.py)

  4. Distribution histograms of matched_transits / N_fov_transits across
     Q1/ERO and STAR/GALAXY slices  (was analyze_dist_matched_transits_over_n_fov.py)

Outputs:
  output/match_ratio_fields.png
  output/match_ratio_field_cutouts.png
  output/astrometric_matched_transits_field_cutouts.png
  output/matched_transits_field_cutouts.png
  output/astrometric_matched_transits_over_n_fov_field_cutouts.png
  output/matched_transits_over_n_fov_field_cutouts.png
  output/dist_matched_transits_over_n_fov.png

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_transit_features.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import healpy as hp
import matplotlib.colors as _mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse as _Ell

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = PROJECT_DIR / "output" / "matched_transits"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
MAG_CUT   = 17
SCAN_NSIDE = 64
MAX_PLOT_POINTS_PER_FIELD = 20_000

Q1_COLOR     = "#BA68C8"
ERO_COLOR    = "#66BB6A"
STAR_COLOR   = "coral"
GALAXY_COLOR = "steelblue"

ERO_FIELDS = {
    "Horsehead":   (85.150,  -2.613, 0.58),
    "Messier 78":  (86.690,  -0.015, 0.60),
    "Taurus":      (64.983, +28.023, 0.61),
    "Barnard 30":  (82.880, +12.316, 0.60),
    "NGC 6254":   (254.303,  -4.100, 0.60),
    "NGC 6397":   (265.174, -53.658, 0.61),
    "NGC 6822":   (296.236, -14.788, 0.60),
    "Holmberg II":(124.790, +70.706, 0.60),
    "IC 10":        (5.063, +59.288, 0.62),
    "IC 342":      (56.730, +68.084, 0.59),
    "NGC 2403":   (114.211, +65.586, 0.60),
    "NGC 6744":   (287.430, -63.842, 0.60),
    "Fornax":      (54.017, -35.267, 0.57),
    "Dorado":      (64.014, -55.780, 0.60),
    "Perseus":     (49.638, +41.651, 0.70),
    "Abell 2390": (328.397, +17.709, 0.75),
    "Abell 2764":   (5.713, -49.249, 0.75),
}

Q1_FIELDS = {
    "EDF-North":  (269.733, +66.018, 22.9),
    "EDF-South":   (61.241, -48.423, 28.1),
    "EDF-Fornax":  (52.932, -28.088, 12.1),
    "LDN 1641":    (85.750,  -8.367,  0.5),
}

ALL_FIELDS = {
    **{f"Q1:{k}": v for k, v in Q1_FIELDS.items()},
    **{f"ERO:{k}": v for k, v in ERO_FIELDS.items()},
}


# ── Shared: scanning-law map ──────────────────────────────────────────────────

def build_scanning_law_map(nside: int = SCAN_NSIDE) -> np.ndarray:
    """Build a full-sky Gaia DR3 nominal FoV transit-count HEALPix map."""
    from gaiaunlimited.scanninglaw import GaiaScanningLaw
    from tqdm import tqdm

    print(f"  Building Gaia DR3 scanning-law map (nside={nside})...")
    sl = GaiaScanningLaw("dr3_nominal")

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra_deg  = np.degrees(phi)
    dec_deg = 90.0 - np.degrees(theta)

    def get_totaln(ra: float, dec: float) -> float:
        return sum(sl.query(ra, dec, count_only=True))

    result = [get_totaln(ra, dec) for ra, dec in tqdm(zip(ra_deg, dec_deg), total=npix)]
    scan_map = np.asarray(result, dtype=float)
    scan_map[~np.isfinite(scan_map)] = hp.UNSEEN
    return scan_map


def add_scan_normalized_columns(df: pd.DataFrame, scan_map: np.ndarray,
                                nside: int = SCAN_NSIDE) -> pd.DataFrame:
    """Attach n_fov_transits and derive scan-normalised transit columns."""
    theta = np.radians(90.0 - df["dec"].to_numpy())
    phi   = np.radians(df["ra"].to_numpy())
    pix   = hp.ang2pix(nside, theta, phi)
    n_fov = scan_map[pix].astype(float)

    out = df.copy()
    out["n_fov_transits"] = n_fov
    valid = np.isfinite(n_fov) & (n_fov > 0)
    out = out.loc[valid].copy()
    out["matched_transits_over_n_fov"] = (
        out["matched_transits"].astype(float) / out["n_fov_transits"])
    if "astrometric_matched_transits" in out.columns:
        out["astrometric_matched_transits_over_n_fov"] = (
            out["astrometric_matched_transits"].astype(float) / out["n_fov_transits"])
    return out


# ── Shared: Gaia field query (scripts 1, 2, 3) ───────────────────────────────

def query_gaia_in_all_fields(username: str) -> pd.DataFrame:
    """One query for all Gaia columns needed by sections 1–3."""
    import psycopg2

    conn   = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    frames = []
    print("  Querying Gaia DR3 in all Q1 + ERO field footprints...")
    for field_name, (field_ra, field_dec, area) in ALL_FIELDS.items():
        radius = np.sqrt(area / np.pi)
        query  = f"""
            SELECT
                ra, dec,
                astrometric_matched_transits,
                matched_transits,
                astrometric_matched_transits::float
                    / NULLIF(matched_transits, 0) AS match_ratio
            FROM gaia_dr3.gaia_source
            WHERE q3c_radial_query(ra, dec, {field_ra}, {field_dec}, {radius})
              AND astrometric_matched_transits IS NOT NULL
              AND matched_transits IS NOT NULL
              AND matched_transits > 0
        """
        print(f"    {field_name} (r={radius:.3f}°)...", end="", flush=True)
        df_f = pd.read_sql_query(query, conn)
        df_f["field"] = field_name
        print(f" {len(df_f):,} sources")
        frames.append(df_f)

    conn.close()
    df = pd.concat(frames, ignore_index=True)
    print(f"  Total: {len(df):,} Gaia sources")
    return df


# ── Shared: per-field cutout plot ─────────────────────────────────────────────

def plot_field_cutouts(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title_label: str,
    cbar_label: str,
    norm,
):
    """Plot one local RA/Dec panel per Q1/ERO field, colored by value_col."""
    field_names = list(ALL_FIELDS.keys())
    n_cols = 4
    n_rows = int(np.ceil(len(field_names) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.1 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    fig.subplots_adjust(left=0.05, right=0.90, top=0.94, bottom=0.05,
                        wspace=0.28, hspace=0.36)

    last_sc = None
    for ax, field_name in zip(axes, field_names):
        ra0, dec0, area = ALL_FIELDS[field_name]
        radius = np.sqrt(area / np.pi)
        fd = df[df["field"] == field_name].copy()

        if len(fd) > MAX_PLOT_POINTS_PER_FIELD:
            fd_plot = fd.sample(MAX_PLOT_POINTS_PER_FIELD, random_state=0)
        else:
            fd_plot = fd

        if len(fd) == 0:
            ax.text(0.5, 0.5, "No Gaia\nsources", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title(field_name, fontsize=10, fontweight="bold")
        else:
            last_sc = ax.scatter(
                fd_plot["ra"], fd_plot["dec"],
                c=fd_plot[value_col],
                cmap="viridis", norm=norm,
                s=3 if field_name.startswith("Q1:") else 8,
                alpha=0.85, linewidths=0, rasterized=True,
            )
            title_name = field_name.replace("Q1:", "").replace("ERO:", "")
            ax.set_title(
                f"{title_name}\nn={len(fd):,}, median={fd[value_col].median():.3f}",
                fontsize=10, fontweight="bold",
            )

        ax.set_xlim(ra0 + radius * 1.15, ra0 - radius * 1.15)
        ax.set_ylim(dec0 - radius * 1.15, dec0 + radius * 1.15)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("RA [deg]", fontsize=9)
        ax.set_ylabel("Dec [deg]", fontsize=9)

        prefix = field_name.split(":", 1)[0]
        spine_color = "red" if prefix == "Q1" else "white"
        for spine in ax.spines.values():
            spine.set_color(spine_color)
            spine.set_linewidth(2.0 if prefix == "Q1" else 1.4)

    for ax in axes[len(field_names):]:
        ax.axis("off")

    if last_sc is not None:
        cax  = fig.add_axes([0.92, 0.17, 0.02, 0.68])
        cbar = fig.colorbar(last_sc, cax=cax)
        cbar.set_label(cbar_label, fontsize=12)

    fig.suptitle(
        f"{title_label} in Euclid field cutouts\n"
        "Q1 fields outlined in red, ERO fields outlined in white",
        fontsize=16, fontweight="bold", y=0.995,
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Section 1: Mollweide match-ratio sky map ──────────────────────────────────

def build_match_ratio_map(df: pd.DataFrame, nside: int = 64):
    theta = np.radians(90.0 - df["dec"].to_numpy())
    phi   = np.radians(df["ra"].to_numpy())
    pix   = hp.ang2pix(nside, theta, phi, nest=False)

    npix      = hp.nside2npix(nside)
    ratio_map = np.full(npix, hp.UNSEEN, dtype=float)
    ratio     = df["match_ratio"].to_numpy()
    order     = np.argsort(pix)
    pix_s, ratio_s = pix[order], ratio[order]

    unique_pix, starts = np.unique(pix_s, return_index=True)
    ends = np.r_[starts[1:], len(pix_s)]
    for p, lo, hi in zip(unique_pix, starts, ends):
        ratio_map[p] = np.nanmedian(ratio_s[lo:hi])
    return ratio_map


def _plot_field_circle(ra_center, dec_center, area_deg2, color, *, n_pts=72):
    radius  = np.sqrt(area_deg2 / np.pi)
    theta   = np.linspace(0, 2 * np.pi, n_pts)
    cos_dec = max(np.cos(np.radians(dec_center)), 0.01)
    ra_pts  = ra_center + radius * np.cos(theta) / cos_dec
    dec_pts = np.clip(dec_center + radius * np.sin(theta), -89.9, 89.9)
    hp.projplot(ra_pts, dec_pts, lonlat=True, coord="CG",
                color=color, linewidth=1.5, zorder=10)
    hp.projscatter(ra_center, dec_center, lonlat=True, coord="CG",
                   color=color, s=3, marker=".", zorder=10)


def plot_match_ratio_mollweide(ratio_map: np.ndarray, output_path: Path):
    fig  = plt.figure()
    cmap = cm.get_cmap("magma").copy()
    cmap.set_under("0.55")
    cmap.set_bad("0.55")
    hp.mollview(ratio_map, coord="CG", title="",
                unit="Median astrometric_matched_transits_ratio",
                min=0.2, max=1.0, cmap=cmap,
                badcolor="0.55", bgcolor="0.55", hold=True, fig=fig)

    fig.set_size_inches(16, 6)
    fig.patch.set_facecolor("white")
    map_ax = fig.axes[0]
    map_ax.set_position([0.02, 0.12, 0.96, 0.82])
    map_ax.set_aspect("auto")
    map_ax.set_facecolor("white")

    cbar_ax = fig.axes[-1]
    cbar_ax.set_position([0.25, 0.06, 0.50, 0.03])

    healpy_patches = list(map_ax.patches)
    healpy_lines   = list(map_ax.lines)
    grey_ell = _Ell((0.5, 0.5), width=1.0, height=1.0,
                    facecolor="0.55", edgecolor="none",
                    zorder=-1, transform=map_ax.transAxes)
    map_ax.add_patch(grey_ell)
    clip_ell = _Ell((0.5, 0.5), width=1.0, height=1.0,
                    transform=map_ax.transAxes)
    for _img in map_ax.get_images():
        _img.set_clip_path(clip_ell)
    for _line in healpy_lines:
        rgba = _mc.to_rgba(_line.get_color())
        if rgba[0] < 0.15 and rgba[1] < 0.15 and rgba[2] < 0.15:
            _line.set_visible(False)
        else:
            _line.set_clip_path(clip_ell)
            _line.set_clip_on(True)
    for _patch in healpy_patches:
        _patch.set_visible(False)

    ero_offsets = {
        "Horsehead":   (0, -4, "left"),   "Messier 78": (0, +4, "center"),
        "Taurus":      (0, +4, "center"), "Barnard 30": (0, +4, "center"),
        "NGC 6254":    (0, +4, "center"), "NGC 6397":   (0, -5, "center"),
        "NGC 6822":    (0, +4, "center"), "Holmberg II":(+5,  0, "left"),
        "IC 10":       (0, -5, "center"), "IC 342":     (0, -5, "center"),
        "NGC 2403":    (-5, 0, "right"),  "NGC 6744":   (0, +4, "center"),
        "Fornax":      (+3, 0, "left"),   "Dorado":     (+4, 0, "left"),
        "Perseus":     (0, -5, "center"), "Abell 2390": (0, +4, "center"),
        "Abell 2764":  (0, -4, "center"),
    }
    q1_offsets = {
        "EDF-North":  (-8, 0, "left"),  "EDF-South":  (-8, 0, "right"),
        "EDF-Fornax": (-6, 0, "left"),  "LDN 1641":   (0, -8, "center"),
    }

    for name, (ra, dec, area) in ERO_FIELDS.items():
        _plot_field_circle(ra, dec, area, "white")
        dra, ddec, ha = ero_offsets[name]
        hp.projtext(ra + dra, dec + ddec, name, lonlat=True, coord="CG",
                    color="white", fontsize=5, ha=ha, zorder=11)
    for name, (ra, dec, area) in Q1_FIELDS.items():
        _plot_field_circle(ra, dec, area, "red")
        dra, ddec, ha = q1_offsets[name]
        hp.projtext(ra + dra, dec + ddec, name, lonlat=True, coord="CG",
                    color="#FF6666", fontsize=6, fontweight="bold", ha=ha, zorder=11)

    plt.legend(handles=[
        Line2D([0], [0], color="red",   linewidth=2, label="Q1 fields"),
        Line2D([0], [0], color="white", linewidth=2, label="ERO fields"),
    ], loc="upper left", fontsize=11, framealpha=0.9,
       edgecolor="white", labelcolor="white", facecolor="#333333")
    plt.title("Gaia astrometric_matched_transits_ratio with Euclid Q1 & ERO Fields",
              fontsize=14, fontweight="bold", y=1.01)
    plt.savefig(output_path, dpi=150, facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_match_ratio_field_cutouts(df: pd.DataFrame, output_path: Path):
    """Per-field cutout panels colored by match_ratio (magma, linear scale)."""
    field_names = list(ALL_FIELDS.keys())
    n_cols = 4
    n_rows = int(np.ceil(len(field_names) / n_cols))

    finite = df["match_ratio"].to_numpy()
    vmin   = np.nanpercentile(finite, 5)
    vmax   = np.nanpercentile(finite, 95)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.1 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    fig.subplots_adjust(left=0.05, right=0.90, top=0.96, bottom=0.05,
                        wspace=0.28, hspace=0.36)

    last_sc = None
    for ax, field_name in zip(axes, field_names):
        ra0, dec0, area = ALL_FIELDS[field_name]
        radius = np.sqrt(area / np.pi)
        fd = df[df["field"] == field_name].copy()

        if len(fd) == 0:
            ax.text(0.5, 0.5, "No Gaia\nsources", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title(field_name, fontsize=10, fontweight="bold")
        else:
            last_sc = ax.scatter(
                fd["ra"], fd["dec"],
                c=fd["match_ratio"],
                cmap="magma", vmin=vmin, vmax=vmax,
                s=8 if field_name.startswith("Q1:") else 10,
                alpha=0.85, linewidths=0, rasterized=True,
            )
            title_name = field_name.replace("Q1:", "").replace("ERO:", "")
            ax.set_title(
                f"{title_name}\nn={len(fd):,}, median={fd['match_ratio'].median():.3f}",
                fontsize=10, fontweight="bold",
            )

        ax.set_xlim(ra0 + radius * 1.15, ra0 - radius * 1.15)
        ax.set_ylim(dec0 - radius * 1.15, dec0 + radius * 1.15)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("RA [deg]", fontsize=9)
        ax.set_ylabel("Dec [deg]", fontsize=9)

        prefix = field_name.split(":", 1)[0]
        spine_color = "red" if prefix == "Q1" else "white"
        for spine in ax.spines.values():
            spine.set_color(spine_color)
            spine.set_linewidth(2.0 if prefix == "Q1" else 1.4)

    for ax in axes[len(field_names):]:
        ax.axis("off")

    if last_sc is not None:
        cax  = fig.add_axes([0.92, 0.17, 0.02, 0.68])
        cbar = fig.colorbar(last_sc, cax=cax)
        cbar.set_label("astrometric_matched_transits_ratio", fontsize=12)

    fig.suptitle("Gaia astrometric_matched_transits_ratio in Euclid field cutouts",
                 fontsize=16, fontweight="bold", y=0.995)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def print_field_statistics(df: pd.DataFrame, ratio_map: np.ndarray, nside: int):
    def map_value(ra, dec):
        pix = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra), nest=False)
        return ratio_map[pix]

    print("\n" + "=" * 72)
    print("ASTROMETRIC_MATCHED_TRANSITS_RATIO AT EUCLID FIELD POSITIONS")
    print("=" * 72)
    print(f"\n{'Field':<15} {'RA':>7} {'Dec':>7} {'Area':>6} {'Map med':>8}")
    print("-" * 52)
    print("\nQ1 Fields:")
    for name, (ra, dec, area) in Q1_FIELDS.items():
        v = map_value(ra, dec)
        print(f"  {name:<15} {ra:>7.1f} {dec:>+7.1f} {area:>5.1f}° "
              f"{'n/a':>8}" if not np.isfinite(v) else
              f"  {name:<15} {ra:>7.1f} {dec:>+7.1f} {area:>5.1f}° {v:>8.3f}")
    print("\nERO Fields:")
    for name, (ra, dec, area) in ERO_FIELDS.items():
        v = map_value(ra, dec)
        print(f"  {name:<15} {ra:>7.1f} {dec:>+7.1f} {area:>5.2f}° "
              f"{'n/a':>8}" if not np.isfinite(v) else
              f"  {name:<15} {ra:>7.1f} {dec:>+7.1f} {area:>5.2f}° {v:>8.3f}")
    print(
        f"\nSource-level: median={df['match_ratio'].median():.3f}, "
        f"5th–95th={df['match_ratio'].quantile(0.05):.3f}–"
        f"{df['match_ratio'].quantile(0.95):.3f}"
    )


# ── Section 4: distribution histograms ───────────────────────────────────────

def query_crossmatch_datasets(username: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Query Q1 and ERO crossmatch tables for the distribution plot."""
    import psycopg2

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)

    q1_query = f"""
        SELECT
            g_ra AS ra, g_dec AS dec,
            g_matched_transits AS matched_transits,
            CASE
                WHEN p_phz_classification = 1 THEN 'STAR'
                WHEN p_phz_classification = 2 THEN 'GALAXY'
            END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE p_phz_classification IN (1, 2)
          AND g_phot_g_mean_mag >= {MAG_CUT}
          AND g_matched_transits IS NOT NULL
          AND g_matched_transits > 0
          AND g_ra IS NOT NULL AND g_dec IS NOT NULL
    """
    ero_query = f"""
        SELECT
            g_ra AS ra, g_dec AS dec,
            g_matched_transits AS matched_transits,
            CASE
                WHEN e_v_class_star >= 0.65 THEN 'STAR'
                WHEN e_v_class_star <  0.65 THEN 'GALAXY'
            END AS label
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE e_v_class_star IS NOT NULL
          AND g_phot_g_mean_mag >= {MAG_CUT}
          AND g_matched_transits IS NOT NULL
          AND g_matched_transits > 0
          AND g_ra IS NOT NULL AND g_dec IS NOT NULL
    """

    print("  Querying Q1 crossmatch...")
    q1_df  = pd.read_sql_query(q1_query,  conn)
    print(f"  Q1:  {len(q1_df):,}  STAR={(q1_df['label']=='STAR').sum():,}  "
          f"GALAXY={(q1_df['label']=='GALAXY').sum():,}")
    print("  Querying ERO crossmatch...")
    ero_df = pd.read_sql_query(ero_query, conn)
    print(f"  ERO: {len(ero_df):,}  STAR={(ero_df['label']=='STAR').sum():,}  "
          f"GALAXY={(ero_df['label']=='GALAXY').sum():,}")
    conn.close()

    for df in (q1_df, ero_df):
        df["matched_transits"] = pd.to_numeric(df["matched_transits"], errors="coerce")
    return q1_df, ero_df


COL = "matched_transits_over_n_fov"


def _hist_panel(ax, data_a, data_b, label_a, label_b, color_a, color_b, title):
    data_a, data_b = data_a.dropna(), data_b.dropna()
    all_data = pd.concat([data_a, data_b])
    lo = max(0.0, float(all_data.quantile(0.01)))
    hi = float(all_data.quantile(0.99))
    if not np.isfinite(hi) or lo == hi:
        hi = lo + 0.1
    bins = np.linspace(lo, hi, 50)
    if len(data_a) > 0:
        ax.hist(data_a, bins=bins, alpha=0.5, density=True,
                color=color_a, label=f"{label_a} (n={len(data_a):,})")
    if len(data_b) > 0:
        ax.hist(data_b, bins=bins, alpha=0.5, density=True,
                color=color_b, label=f"{label_b} (n={len(data_b):,})")
    ax.set_xlabel(COL, fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")


def plot_dist_matched_transits_over_n_fov(
    q1_df: pd.DataFrame, ero_df: pd.DataFrame, output_path: Path
):
    combined_df = pd.concat([q1_df, ero_df], ignore_index=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    _hist_panel(axes[0, 0], q1_df[COL],  ero_df[COL],
                "Q1", "ERO", Q1_COLOR, ERO_COLOR, "Q1 vs ERO — All sources")
    _hist_panel(axes[0, 1],
                q1_df.loc[q1_df["label"] == "GALAXY", COL],
                ero_df.loc[ero_df["label"] == "GALAXY", COL],
                "Q1", "ERO", Q1_COLOR, ERO_COLOR, "Q1 vs ERO — Galaxies only")
    _hist_panel(axes[0, 2],
                q1_df.loc[q1_df["label"] == "STAR", COL],
                ero_df.loc[ero_df["label"] == "STAR", COL],
                "Q1", "ERO", Q1_COLOR, ERO_COLOR, "Q1 vs ERO — Stars only")

    for ax, df, title in [
        (axes[1, 0], combined_df, "STAR vs GALAXY — Q1+ERO combined"),
        (axes[1, 1], ero_df,      "STAR vs GALAXY — ERO only"),
        (axes[1, 2], q1_df,       "STAR vs GALAXY — Q1 only"),
    ]:
        _hist_panel(ax,
                    df.loc[df["label"] == "GALAXY", COL],
                    df.loc[df["label"] == "STAR",   COL],
                    "GALAXY", "STAR", GALAXY_COLOR, STAR_COLOR, title)

    fig.suptitle(
        f"matched_transits / N_fov_transits distributions  (G >= {MAG_CUT})\n"
        "Row 1: Q1 vs ERO  |  Row 2: STAR vs GALAXY",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Gaia transit-feature analysis — Euclid Q1 + ERO fields")
    print("=" * 72)

    # ── Shared: scanning-law map ───────────────────────────────────────────
    print("\n[Step 1/4] Building scanning-law map ...")
    scan_map = build_scanning_law_map(nside=SCAN_NSIDE)

    # ── Shared: one Gaia field query ───────────────────────────────────────
    print("\n[Step 2/4] Querying Gaia sources in all fields ...")
    gaia_df = query_gaia_in_all_fields(username)

    # ── Section 1: match-ratio Mollweide + cutouts ─────────────────────────
    print("\n[Step 3/4] Plotting match-ratio maps ...")
    ratio_map = build_match_ratio_map(gaia_df, nside=SCAN_NSIDE)
    print_field_statistics(gaia_df, ratio_map, nside=SCAN_NSIDE)
    plot_match_ratio_mollweide(ratio_map, OUTPUT_DIR / "match_ratio_fields.png")
    plot_match_ratio_field_cutouts(gaia_df, OUTPUT_DIR / "match_ratio_field_cutouts.png")

    # ── Section 2: raw transit-count cutouts (LogNorm) ─────────────────────
    print("\n  Plotting raw matched-transit cutouts ...")
    for col, title, cbar in [
        ("astrometric_matched_transits",
         "Gaia astrometric_matched_transits",
         "astrometric_matched_transits"),
        ("matched_transits",
         "Gaia matched_transits",
         "matched_transits"),
    ]:
        vals = gaia_df[col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        norm = LogNorm(vmin=max(1, np.nanpercentile(vals, 5)),
                       vmax=np.nanpercentile(vals, 95))
        plot_field_cutouts(
            gaia_df, col,
            OUTPUT_DIR / f"{col}_field_cutouts.png",
            title, cbar, norm,
        )

    # ── Section 3: scan-normalised cutouts (linear) ────────────────────────
    print("\n  Plotting scan-normalised transit cutouts ...")
    gaia_norm_df = add_scan_normalized_columns(gaia_df, scan_map, nside=SCAN_NSIDE)
    for col, title, cbar in [
        ("astrometric_matched_transits_over_n_fov",
         "Gaia astrometric_matched_transits / N_fov_transits",
         "astrometric_matched_transits / N_fov_transits"),
        ("matched_transits_over_n_fov",
         "Gaia matched_transits / N_fov_transits",
         "matched_transits / N_fov_transits"),
    ]:
        vals = gaia_norm_df[col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        norm = Normalize(vmin=0.0, vmax=max(1.0, float(np.nanpercentile(vals, 98))))
        plot_field_cutouts(
            gaia_norm_df, col,
            OUTPUT_DIR / f"{col}_field_cutouts.png",
            title, cbar, norm,
        )

    # ── Section 4: distribution histograms ────────────────────────────────
    print("\n[Step 4/4] Distribution of matched_transits / N_fov_transits ...")
    q1_df, ero_df = query_crossmatch_datasets(username)
    q1_df  = add_scan_normalized_columns(q1_df,  scan_map, nside=SCAN_NSIDE)
    ero_df = add_scan_normalized_columns(ero_df, scan_map, nside=SCAN_NSIDE)
    plot_dist_matched_transits_over_n_fov(
        q1_df, ero_df,
        OUTPUT_DIR / "dist_matched_transits_over_n_fov.png",
    )

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
