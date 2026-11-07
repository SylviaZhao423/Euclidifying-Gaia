#!/usr/bin/env python3
"""
Euclid cutouts for the 12 double-matched QSO pairs (extract_double_matched_qso_pairs.py):
spectroscopic QSO pairs whose two members collapse onto ONE Euclid source (and one Gaia
source). One stamp per pair, centred on the shared Euclid source, so you can eyeball whether
Euclid resolves two objects (genuine tight pair) or one (duplicate spectroscopic entry).

Each stamp is a VIS(+NISP LRGB when bright) Q1 cutout with:
  * white centre crosshair  — the shared Euclid source (e_ra, e_dec)
  * cyan crosshair          — QSO member A  (brighter component; labelled with its spec-z)
  * orange crosshair        — QSO member B  (labelled with its spec-z)
  * blue ring               — the shared Gaia source
Title carries pair separation, Δz, both redshifts and G. Laid out in one contact sheet.

Reads:
  output/qso_pairs/double_matched_qso_pairs.csv   (from extract_double_matched_qso_pairs.py)
Writes:
  output/qso_pairs/double_matched_cutouts.png

Usage:
  python src/download_double_matched_cutouts.py                 # 5" FOV
  python src/download_double_matched_cutouts.py --size 8
"""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_q1_cutout import (_assign_to_tiles, _render_cutout, _nisp_filename,   # noqa: E402
                              IRSA_IBE_DATA, GAIA_RING)
from download_qso_catwise_e_g_cutouts import _tiles_near, FETCH_TIMEOUT, _on_alarm  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
PAIRS_CSV = PROJECT_DIR / "output" / "qso_pairs" / "double_matched_qso_pairs.csv"
OUT = PROJECT_DIR / "output" / "qso_pairs" / "double_matched_cutouts.png"
A_COLOR = "#00e5ff"    # cyan  — member A
B_COLOR = "#ffa500"    # orange — member B


def render_pair(row, size: float):
    """Q1 Euclid stamp centred on the shared Euclid source, with A/B QSO + Gaia overlaid."""
    ra, dec = float(row["e_ra"]), float(row["e_dec"])
    comps = [
        (float(row["ra_a"]), float(row["dec_a"]), f"A z={row['z_a']:.2f}", A_COLOR, "crosshair"),
        (float(row["ra_b"]), float(row["dec_b"]), f"B z={row['z_b']:.2f}", B_COLOR, "crosshair"),
        (float(row["g_ra"]), float(row["g_dec"]), "Gaia", GAIA_RING, "ring"),
    ]
    ti = _assign_to_tiles(ra, dec, _tiles_near(ra, dec), "euclidwise")
    if ti is None:
        return None
    njf, nhf = _nisp_filename(ti.tile_id, "J"), _nisp_filename(ti.tile_id, "H")
    nju = f"{IRSA_IBE_DATA}/{ti.tile_id}/NISP/{njf}" if njf else None
    nhu = f"{IRSA_IBE_DATA}/{ti.tile_id}/NISP/{nhf}" if nhf else None
    signal.signal(signal.SIGALRM, _on_alarm); signal.alarm(FETCH_TIMEOUT)
    try:
        return _render_cutout(ti.url, ra, dec, size, "", draw_crosshair=True,
                              nisp_j_url=nju, nisp_h_url=nhu, companions=comps,
                              crosshair_color="white")
    except Exception as exc:
        print(f"    pair e_id={row['e_id']} @ ({ra:.5f},{dec:.5f}) failed: {exc}")
        return None
    finally:
        signal.alarm(0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=float, default=5.0, help="cutout FOV arcsec (default 5).")
    args = ap.parse_args()
    if not PAIRS_CSV.exists():
        print(f"  {PAIRS_CSV} missing — run extract_double_matched_qso_pairs.py first."); return 1
    pairs = pd.read_csv(PAIRS_CSV).sort_values("pair_sep_arcsec").reset_index(drop=True)
    print(f"  rendering {len(pairs)} double-matched pairs ({args.size}\" FOV) ...")

    ncols = 4
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(4.0 * ncols, 4.6 * nrows))
    for ax in axes.ravel():
        ax.axis("off")
    for i, (_, row) in enumerate(pairs.iterrows()):
        ax = axes[i // ncols][i % ncols]
        arr = render_pair(row, args.size)
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        if arr is not None:
            ax.imshow(np.clip(arr, 0, 1))
        else:
            ax.text(0.5, 0.5, "no Euclid tile", ha="center", va="center", color="w")
            ax.set_facecolor("k")
        ax.set_title(f"sep={row['pair_sep_arcsec']:.2f}\"  Δz={row['dz']:+.3f}\n"
                     f"z={row['z_a']:.2f}/{row['z_b']:.2f}  G={row['G_a']:.1f}",
                     fontsize=9)
        print(f"    e_id={int(row['e_id'])}  sep={row['pair_sep_arcsec']:.2f}\"  "
              f"dz={row['dz']:+.3f}  euclid_ok={arr is not None}")
    fig.suptitle("Double-matched QSO pairs — Euclid stamp centred on the shared Euclid source\n"
                 "cyan + = QSO A, orange + = QSO B, blue ○ = shared Gaia, white + = Euclid centre",
                 fontsize=13, fontweight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
