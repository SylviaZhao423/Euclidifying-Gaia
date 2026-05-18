#!/usr/bin/env python3
"""
3 × 2 contact sheet of 6 hand-picked sources from
output/euclid_cutouts/ero_cutouts/q1_to_ero/agree_both_galaxy/cluster0.

Mirrors the styling of download_e_q1_to_ero_cutouts.py::make_contact_sheet
(colorbar legend top-left, page-style title) but for a fixed 3×2 set.

Reads:
  output/euclid_cutouts/ero_cutouts/q1_to_ero/agree_both_galaxy/cluster0/
      manifest.csv   (source_id → cutout_file)
      cutouts/cutout_NNNN.png

Writes:
  docs/slides/figs/contact_sheet_agree_both_galaxy_cluster0_pick6.png

Usage:
  python src/slides/make_contact_sheet_agree_both_galaxy_cluster0.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
GROUP_DIR   = (PROJECT_DIR / "output" / "euclid_cutouts" / "ero_cutouts"
               / "q1_to_ero" / "agree_both_galaxy" / "cluster0")
OUTPUT_DIR  = PROJECT_DIR / "docs" / "slides" / "figs"

# Hand-picked source IDs in the order requested
PICKED_IDS = [
    4973890304241456384,   # RA=5.7828  Dec=-49.5893  G=21.59
    4974122713514264064,   # RA=5.9906  Dec=-49.1782  G=21.34
    4973936518090372096,   # RA=6.3753  Dec=-49.1867  G=21.87
    6438595648215417856,   # RA=287.2596 Dec=-63.6429 G=20.85
    1098502108599980672,   # RA=124.1363 Dec=+70.8737 G=21.51
    493637702483818368,    # RA=56.5740  Dec=+68.4112 G=21.84
]
NCOLS, NROWS = 3, 2


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = GROUP_DIR / "manifest.csv"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found.")
        return 1

    df = pd.read_csv(manifest_path)
    df["g_source_id"] = df["g_source_id"].astype("int64")
    by_id = df.set_index("g_source_id")

    cell_paths = []
    for sid in PICKED_IDS:
        if sid not in by_id.index:
            print(f"  WARNING: source_id {sid} not in manifest.")
            continue
        rel = by_id.loc[sid, "cutout_file"]
        if isinstance(rel, pd.Series):
            rel = rel.iloc[0]
        p = GROUP_DIR / rel
        if not p.exists():
            print(f"  WARNING: {p} not found.")
            continue
        cell_paths.append(p)
        print(f"  {sid} -> {rel}")

    if len(cell_paths) != NCOLS * NROWS:
        print(f"ERROR: expected {NCOLS * NROWS} cutouts, got {len(cell_paths)}.")
        return 1

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(5 * NCOLS, 5 * NROWS),
                             squeeze=False)
    for ax, p in zip(axes.flat, cell_paths):
        ax.imshow(mpimg.imread(str(p)))
        ax.axis("off")

    # Match the layout of make_contact_sheet in download_e_q1_to_ero_cutouts.py
    fig_h_in  = 5.0 * NROWS
    header_in = 1.10
    top_frac  = 1.0 - header_in / fig_h_in

    fig.subplots_adjust(top=top_frac, bottom=0.01, left=0.01, right=0.99,
                        hspace=0.05, wspace=0.05)

    legend_w = 0.28
    legend_h = 0.15 / fig_h_in
    legend_x = 0.07
    legend_y = top_frac + 0.20 / fig_h_in
    sup_y    = legend_y + legend_h * 0.5

    plt.suptitle(f"agree_both_galaxy / cluster0  ({len(cell_paths)} sources)",
                 fontsize=10, y=sup_y)

    cbar_ax = fig.add_axes([legend_x, legend_y, legend_w, legend_h])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap="coolwarm_r",
                              norm=plt.Normalize(vmin=0, vmax=1)),
        cax=cbar_ax, orientation="horizontal",
    )
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["star", "0.5", "galaxy"])
    cbar.ax.tick_params(labelsize=7, pad=1)
    cbar_ax.set_title("XGBoost P(gal) — crosshair colour",
                       fontsize=7, pad=3, loc="left")

    out_path = OUTPUT_DIR / "contact_sheet_agree_both_galaxy_cluster0_pick6.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
