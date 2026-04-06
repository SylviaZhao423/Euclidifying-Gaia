#!/usr/bin/env python3
"""
Overlay Q1 and ERO field positions on a sky map of Gaia
astrometric_matched_transits_ratio.

The Mollweide background is a sparse HEALPix map built only from Gaia sources
queried inside the Q1 + ERO field footprints: for each populated nside pixel
we store the median of

    astrometric_matched_transits / matched_transits

over Gaia DR3 sources in that pixel. Field cutout panels use the same direct
Gaia-in-field query.

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_match_ratio.py
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
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse as _Ell

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB = "wsdb"

# Euclid field centers (J2000) from Table 1
ERO_FIELDS = {
    "Horsehead": (85.150, -2.613, 0.58),
    "Messier 78": (86.690, -0.015, 0.60),
    "Taurus": (64.983, +28.023, 0.61),
    "Barnard 30": (82.880, +12.316, 0.60),
    "NGC 6254": (254.303, -4.100, 0.60),
    "NGC 6397": (265.174, -53.658, 0.61),
    "NGC 6822": (296.236, -14.788, 0.60),
    "Holmberg II": (124.790, +70.706, 0.60),
    "IC 10": (5.063, +59.288, 0.62),
    "IC 342": (56.730, +68.084, 0.59),
    "NGC 2403": (114.211, +65.586, 0.60),
    "NGC 6744": (287.430, -63.842, 0.60),
    "Fornax": (54.017, -35.267, 0.57),
    "Dorado": (64.014, -55.780, 0.60),
    "Perseus": (49.638, +41.651, 0.70),
    "Abell 2390": (328.397, +17.709, 0.75),
    "Abell 2764": (5.713, -49.249, 0.75),
}

Q1_FIELDS = {
    "EDF-North": (269.733, +66.018, 22.9),
    "EDF-South": (61.241, -48.423, 28.1),
    "EDF-Fornax": (52.932, -28.088, 12.1),
    "LDN 1641": (85.75, -8.367, 0.5),
}

ALL_FIELDS = {
    **{f"Q1:{k}": v for k, v in Q1_FIELDS.items()},
    **{f"ERO:{k}": v for k, v in ERO_FIELDS.items()},
}


def build_match_ratio_map(df: pd.DataFrame, nside: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Build a sparse HEALPix median match-ratio map from already-queried field Gaia sources."""
    theta = np.radians(90.0 - df["dec"].to_numpy())
    phi = np.radians(df["ra"].to_numpy())
    pix = hp.ang2pix(nside, theta, phi, nest=False)

    npix = hp.nside2npix(nside)
    ratio_map = np.full(npix, hp.UNSEEN, dtype=float)
    counts_map = np.zeros(npix, dtype=int)

    ratio = df["match_ratio"].to_numpy()
    order = np.argsort(pix)
    pix_sorted = pix[order]
    ratio_sorted = ratio[order]

    unique_pix, starts = np.unique(pix_sorted, return_index=True)
    ends = np.r_[starts[1:], len(pix_sorted)]
    for p, lo, hi in zip(unique_pix, starts, ends):
        vals = ratio_sorted[lo:hi]
        ratio_map[p] = np.nanmedian(vals)
        counts_map[p] = hi - lo

    return ratio_map, counts_map


def query_gaia_in_all_fields(username: str) -> pd.DataFrame:
    """Query Gaia sources directly within each Q1/ERO field footprint."""
    import psycopg2

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    frames = []
    print("  Querying Gaia DR3 directly in all Q1 + ERO field footprints...")
    for field_name, (field_ra, field_dec, area) in ALL_FIELDS.items():
        radius = np.sqrt(area / np.pi)
        query = f"""
            SELECT
                ra,
                dec,
                astrometric_matched_transits::float / NULLIF(matched_transits, 0) AS match_ratio
            FROM gaia_dr3.gaia_source
            WHERE q3c_radial_query(ra, dec, {field_ra}, {field_dec}, {radius})
              AND astrometric_matched_transits IS NOT NULL
              AND matched_transits IS NOT NULL
              AND matched_transits > 0
        """
        print(f"    {field_name} (r={radius:.3f}°)...", end="", flush=True)
        df_f = pd.read_sql_query(query, conn)
        df_f["field"] = field_name
        print(f" {len(df_f):,} Gaia sources")
        frames.append(df_f)

    conn.close()
    df = pd.concat(frames, ignore_index=True)
    print(f"  Retrieved {len(df):,} Gaia sources across all field cutouts")
    return df


def plot_field_circle(ra_center, dec_center, area_deg2, color, *, n_pts=72):
    """Draw a field circle on the current healpy Mollweide axes."""
    radius = np.sqrt(area_deg2 / np.pi)
    theta = np.linspace(0, 2 * np.pi, n_pts)
    cos_dec = max(np.cos(np.radians(dec_center)), 0.01)
    ra_pts = ra_center + radius * np.cos(theta) / cos_dec
    dec_pts = np.clip(dec_center + radius * np.sin(theta), -89.9, 89.9)
    hp.projplot(ra_pts, dec_pts, lonlat=True, coord="CG",
                color=color, linewidth=1.5, zorder=10)
    hp.projscatter(ra_center, dec_center, lonlat=True, coord="CG",
                   color=color, s=3, marker=".", zorder=10)


def plot_match_ratio_with_fields(ratio_map: np.ndarray, counts_map: np.ndarray, output_path: Path):
    """Plot full-sky Mollweide map with Q1 and ERO fields (layout matches scanning_law)."""
    fig = plt.figure()
    cmap = cm.get_cmap("magma").copy()
    cmap.set_under("0.55")   # values < min appear grey (matches unsampled background)
    cmap.set_bad("0.55")     # UNSEEN / NaN pixels appear grey
    plot_map = ratio_map.copy()
    # coord='CG': celestial HEALPix map, displayed in Galactic (same as analyze_scanning_law.py)
    hp.mollview(
        plot_map,
        coord="CG",
        title="",
        unit="Median astrometric_matched_transits_ratio",
        min=0.2,
        max=1.0,
        cmap=cmap,
        badcolor="0.55",
        bgcolor="0.55",
        hold=True,
        fig=fig,
    )

    fig.set_size_inches(16, 6)
    fig.patch.set_facecolor("white")
    map_ax = fig.axes[0]
    map_ax.set_position([0.02, 0.12, 0.96, 0.82])
    map_ax.set_aspect("auto")
    map_ax.set_facecolor("white")  # white for corners outside the ellipse

    cbar_ax = fig.axes[-1]
    cbar_ax.set_position([0.25, 0.06, 0.50, 0.03])

    # healpy ignores bgcolor/badcolor in this version: bad pixels are transparent
    # (axes facecolor shows through) and corners are hardcoded opaque black.
    # Fix: grey ellipse behind the image (shows through transparent bad pixels),
    # clip the image + graticule lines to the ellipse to hide black corners and
    # prevent lines bleeding into the title, and hide healpy's boundary patch.
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

    # Hide black lines (projection boundary) and clip grey ones (graticule)
    for _line in healpy_lines:
        rgba = _mc.to_rgba(_line.get_color())
        if rgba[0] < 0.15 and rgba[1] < 0.15 and rgba[2] < 0.15:
            _line.set_visible(False)   # boundary line — hide it
        else:
            _line.set_clip_path(clip_ell)
            _line.set_clip_on(True)

    # Hide healpy's projection boundary patches
    for _patch in healpy_patches:
        _patch.set_visible(False)

    ero_offsets = {
        "Horsehead": (0, -4, "left"),
        "Messier 78": (0, +4, "center"),
        "Taurus": (0, +4, "center"),
        "Barnard 30": (0, +4, "center"),
        "NGC 6254": (0, +4, "center"),
        "NGC 6397": (0, -5, "center"),
        "NGC 6822": (0, +4, "center"),
        "Holmberg II": (+5, 0, "left"),
        "IC 10": (0, -5, "center"),
        "IC 342": (0, -5, "center"),
        "NGC 2403": (-5, 0, "right"),
        "NGC 6744": (0, +4, "center"),
        "Fornax": (+3, 0, "left"),
        "Dorado": (+4, 0, "left"),
        "Perseus": (0, -5, "center"),
        "Abell 2390": (0, +4, "center"),
        "Abell 2764": (0, -4, "center"),
    }

    q1_offsets = {
        "EDF-North": (-8, 0, "left"),
        "EDF-South": (-8, 0, "right"),
        "EDF-Fornax": (-6, 0, "left"),
        "LDN 1641": (0, -8, "center"),
    }

    for name, (ra, dec, area) in ERO_FIELDS.items():
        plot_field_circle(ra, dec, area, "white")
        dra, ddec, ha = ero_offsets[name]
        hp.projtext(ra + dra, dec + ddec, name, lonlat=True, coord="CG",
                    color="white", fontsize=5, ha=ha, zorder=11)

    for name, (ra, dec, area) in Q1_FIELDS.items():
        plot_field_circle(ra, dec, area, "red")
        dra, ddec, ha = q1_offsets[name]
        hp.projtext(ra + dra, dec + ddec, name, lonlat=True, coord="CG",
                    color="#FF6666", fontsize=6, fontweight="bold", ha=ha, zorder=11)

    legend_elements = [
        Line2D([0], [0], color="red", linewidth=2, label="Q1 fields"),
        Line2D([0], [0], color="white", linewidth=2, label="ERO fields"),
    ]
    plt.legend(handles=legend_elements, loc="upper left", fontsize=11,
               framealpha=0.9, edgecolor="white",
               labelcolor="white", facecolor="#333333")

    n_filled = int(np.count_nonzero((np.isfinite(ratio_map)) & (~np.isclose(ratio_map, hp.UNSEEN))))
    title = (
        "Gaia astrometric_matched_transits_ratio with Euclid Q1 & ERO Fields"
    )
    plt.title(title, fontsize=14, fontweight="bold", y=1.01)
    plt.savefig(output_path, dpi=150, facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_match_ratio_field_cutouts(df: pd.DataFrame, output_path: Path):
    """Plot one local RA/Dec panel per Q1/ERO field, colored by Gaia match ratio."""
    field_names = list(ALL_FIELDS.keys())
    n_cols = 4
    n_rows = int(np.ceil(len(field_names) / n_cols))

    finite = df["match_ratio"].to_numpy()
    vmin = np.nanpercentile(finite, 5)
    vmax = np.nanpercentile(finite, 95)

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
                cmap="magma",
                vmin=vmin, vmax=vmax,
                s=8 if field_name.startswith("Q1:") else 10,
                alpha=0.85, linewidths=0, rasterized=True,
            )
            title_name = field_name.replace("Q1:", "").replace("ERO:", "")
            ax.set_title(
                f"{title_name}\n"
                f"n={len(fd):,}, median={fd['match_ratio'].median():.3f}",
                fontsize=10, fontweight="bold"
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
        spine_width = 2.0 if prefix == "Q1" else 1.4
        for spine in ax.spines.values():
            spine.set_color(spine_color)
            spine.set_linewidth(spine_width)

    for ax in axes[len(field_names):]:
        ax.axis("off")

    if last_sc is not None:
        cax = fig.add_axes([0.92, 0.17, 0.02, 0.68])
        cbar = fig.colorbar(last_sc, cax=cax)
        cbar.set_label("astrometric_matched_transits_ratio", fontsize=12)

    fig.suptitle(
        "Gaia astrometric_matched_transits_ratio in Euclid field cutouts",
        fontsize=16, fontweight="bold", y=0.995,
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def print_field_statistics(df: pd.DataFrame, ratio_map: np.ndarray, nside: int):
    """Print field-center and source-level ratio summaries."""
    print("\n" + "=" * 72)
    print("ASTROMETRIC_MATCHED_TRANSITS_RATIO AT EUCLID FIELD POSITIONS")
    print("=" * 72)

    print(f"\n{'Field':<15} {'RA':>7} {'Dec':>7} {'Area':>6} {'Map med':>8}")
    print("-" * 52)

    def map_value(ra, dec):
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        pix = hp.ang2pix(nside, theta, phi, nest=False)
        return ratio_map[pix]

    print("\nQ1 Fields:")
    for name, (ra, dec, area) in Q1_FIELDS.items():
        val = map_value(ra, dec)
        val_str = f"{val:.3f}" if np.isfinite(val) else "n/a"
        print(f"  {name:<15} {ra:>7.1f} {dec:>+7.1f} {area:>5.1f}° {val_str:>8}")

    print("\nERO Fields:")
    for name, (ra, dec, area) in ERO_FIELDS.items():
        val = map_value(ra, dec)
        val_str = f"{val:.3f}" if np.isfinite(val) else "n/a"
        print(f"  {name:<15} {ra:>7.1f} {dec:>+7.1f} {area:>5.2f}° {val_str:>8}")

    print(
        "\nSource-level summary: "
        f"median={df['match_ratio'].median():.3f}, "
        f"5th–95th percentile="
        f"{df['match_ratio'].quantile(0.05):.3f}–{df['match_ratio'].quantile(0.95):.3f}"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER environment variable not set.")
        print("Set it with: export PGUSER=your_username")
        return 1

    print("=" * 72)
    print("Gaia astrometric_matched_transits_ratio: Q1 vs ERO field coverage")
    print(f"Q1 fields: {len(Q1_FIELDS)}, ERO fields: {len(ERO_FIELDS)}")
    print("=" * 72)

    print("\nField cutouts: querying Gaia in each Euclid disk...")
    df = query_gaia_in_all_fields(username)

    nside = 64
    print(f"\nBuilding sparse HEALPix median map (nside={nside}) from field Gaia sources...")
    ratio_map, counts_map = build_match_ratio_map(df, nside=nside)

    print_field_statistics(df, ratio_map, nside)

    print("\nGenerating plot...")
    plot_match_ratio_with_fields(
        ratio_map,
        counts_map,
        OUTPUT_DIR / "match_ratio_fields.png",
    )
    plot_match_ratio_field_cutouts(
        df,
        OUTPUT_DIR / "match_ratio_field_cutouts.png",
    )

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
