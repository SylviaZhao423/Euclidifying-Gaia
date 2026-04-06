#!/usr/bin/env python3
"""
Plot Euclid field cutouts colored by Gaia transit-count features.

This mirrors the field-cutout layout used in analyze_match_ratio.py, but
produces two separate figures:

  1. astrometric_matched_transits_field_cutouts.png
  2. matched_transits_field_cutouts.png

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_matched_transits_cutouts.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB = "wsdb"

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
                astrometric_matched_transits,
                matched_transits
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


def plot_field_cutouts(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title_label: str,
    cbar_label: str,
):
    """Plot one local RA/Dec panel per field, colored by `value_col`."""
    field_names = list(ALL_FIELDS.keys())
    n_cols = 4
    n_rows = int(np.ceil(len(field_names) / n_cols))

    finite = df[value_col].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    norm = LogNorm(
        vmin=max(1, np.nanpercentile(finite, 5)),
        vmax=np.nanpercentile(finite, 95),
    )

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.1 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    fig.subplots_adjust(left=0.05, right=0.90, top=0.94, bottom=0.05,
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
                c=fd[value_col],
                cmap="viridis",
                norm=norm,
                s=8 if field_name.startswith("Q1:") else 10,
                alpha=0.85, linewidths=0, rasterized=True,
            )
            title_name = field_name.replace("Q1:", "").replace("ERO:", "")
            ax.set_title(
                f"{title_name}\n"
                f"n={len(fd):,}, median={fd[value_col].median():.1f}",
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
        cbar.set_label(cbar_label, fontsize=12)

    fig.suptitle(
        f"{title_label} in Euclid field cutouts\n"
        "Q1 fields outlined in red, ERO fields outlined in white",
        fontsize=16, fontweight="bold", y=0.995,
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER environment variable not set.")
        print("Set it with: export PGUSER=your_username")
        return 1

    print("=" * 72)
    print("Gaia matched-transit features: Q1 vs ERO field cutouts")
    print(f"Q1 fields: {len(Q1_FIELDS)}, ERO fields: {len(ERO_FIELDS)}")
    print("=" * 72)

    df = query_gaia_in_all_fields(username)

    print("\nGenerating plots...")
    plot_field_cutouts(
        df,
        "astrometric_matched_transits",
        OUTPUT_DIR / "astrometric_matched_transits_field_cutouts.png",
        "Gaia astrometric_matched_transits",
        "astrometric_matched_transits",
    )
    plot_field_cutouts(
        df,
        "matched_transits",
        OUTPUT_DIR / "matched_transits_field_cutouts.png",
        "Gaia matched_transits",
        "matched_transits",
    )

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
