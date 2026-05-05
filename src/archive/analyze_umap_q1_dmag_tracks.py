#!/usr/bin/env python3
"""
Project Δm-track sources onto the Q1 UMAP embedding.

Queries crossmatch_q1_mer_phz_gaia for sources in three Δm tracks:
  - Upper track:  Δm > +1   (Euclid fainter than Gaia)
  - Main track:   -1 ≤ Δm ≤ +1
  - Lower track:  Δm < -2   (Euclid brighter than Gaia)

Matches their g_source_id to the UMAP embedding and overlays them
on the base Q1 UMAP coloured by Δm track membership.

Reads:
  output/umap/arrays/q1_indiv.npz

Writes:
  output/umap/q1_individual/dmag_tracks_in_q1_space.png

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_umap_q1_dmag_tracks.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "q1_individual"
OUTPUT_FILE = OUTPUT_DIR / "dmag_tracks_in_q1_space.png"

WSDB_HOST    = "wsdb.ast.cam.ac.uk"
WSDB_DB      = "wsdb"
GAIA_AB_CORR = 0.11

# Track definitions
UPPER_LO  =  1.0
LOWER_HI  = -2.0

TRACK_COLORS = {
    "upper": "#e6194b",   # red
    "main":  "#aaaaaa",   # grey
    "lower": "#4363d8",   # blue
}


def query_dmag_source_ids(username):
    """Fetch g_source_id and Δm for all Q1×Gaia matches."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()
    print("Querying Δm and source IDs from crossmatch_q1_mer_phz_gaia...")
    cursor.execute(f"""
        SELECT
            g_source_id,
            m_ie_mag - (g_phot_g_mean_mag + {GAIA_AB_CORR}) AS delta_m
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE m_ie_mag IS NOT NULL
          AND g_phot_g_mean_mag IS NOT NULL
          AND g_phot_g_mean_mag BETWEEN 0 AND 30
          AND m_vis_det = 1
    """)
    rows = cursor.fetchall()
    cursor.close(); conn.close()

    sids = np.array([str(r[0]) for r in rows])
    dm   = np.array([r[1] for r in rows], dtype=float)
    print(f"  Retrieved {len(dm):,} sources")
    return sids, dm


def main():
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set"); return 1

    # ── Load UMAP embedding ───────────────────────────────────────────────────
    print("Loading UMAP embedding...")
    q1 = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    umap_sids      = q1["source_ids"].astype(str)
    embedding      = q1["embedding"]
    y              = q1["y"]           # 0=star, 1=galaxy
    phz_gal_prob   = q1["phz_gal_prob"].astype(float)
    print(f"  UMAP: {len(umap_sids):,} sources")

    # ── Query Δm tracks ───────────────────────────────────────────────────────
    db_sids, dm = query_dmag_source_ids(username)
    finite = np.isfinite(dm)
    db_sids, dm = db_sids[finite], dm[finite]

    upper_sids = set(db_sids[dm >  UPPER_LO])
    lower_sids = set(db_sids[dm <  LOWER_HI])
    print(f"  Upper track (Δm > +{UPPER_LO}): {len(upper_sids):,} sources")
    print(f"  Lower track (Δm < {LOWER_HI}):  {len(lower_sids):,} sources")

    # ── Match to UMAP ─────────────────────────────────────────────────────────
    upper_mask = np.isin(umap_sids, list(upper_sids))
    lower_mask = np.isin(umap_sids, list(lower_sids))
    base_mask  = ~upper_mask & ~lower_mask

    print(f"  UMAP upper track hits: {upper_mask.sum():,}")
    print(f"  UMAP lower track hits: {lower_mask.sum():,}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                             gridspec_kw={"wspace": 0.25})
    fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.08)

    titles = [
        f"Upper track  (Δm > +{UPPER_LO},  n_UMAP={upper_mask.sum():,})",
        f"Lower track  (Δm < {LOWER_HI},  n_UMAP={lower_mask.sum():,})",
        "Both tracks overlaid",
    ]

    for col, (highlight_mask, title, color) in enumerate([
        (upper_mask, titles[0], TRACK_COLORS["upper"]),
        (lower_mask, titles[1], TRACK_COLORS["lower"]),
        (None,       titles[2], None),
    ]):
        ax = axes[col]

        # Base: all UMAP sources in grey
        ax.scatter(embedding[base_mask, 0], embedding[base_mask, 1],
                   s=0.3, c=TRACK_COLORS["main"], alpha=0.15, rasterized=True)

        if col < 2:
            ax.scatter(embedding[highlight_mask, 0], embedding[highlight_mask, 1],
                       s=2.5, c=color, alpha=0.7, rasterized=True,
                       label=f"n={highlight_mask.sum():,}")
        else:
            # Both overlaid
            ax.scatter(embedding[upper_mask, 0], embedding[upper_mask, 1],
                       s=2.5, c=TRACK_COLORS["upper"], alpha=0.7, rasterized=True,
                       label=f"Upper (n={upper_mask.sum():,})")
            ax.scatter(embedding[lower_mask, 0], embedding[lower_mask, 1],
                       s=2.5, c=TRACK_COLORS["lower"], alpha=0.7, rasterized=True,
                       label=f"Lower (n={lower_mask.sum():,})")

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=10)
        ax.set_ylabel("UMAP 2", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, markerscale=4, loc="upper right")
        ax.grid(True, alpha=0.15)

    fig.suptitle(
        "Q1–Gaia Δm Track Sources in Q1 UMAP Space\n"
        f"Upper: Δm > +{UPPER_LO} (Euclid fainter)   |   "
        f"Lower: Δm < {LOWER_HI} (Euclid brighter)",
        fontsize=12, fontweight="bold",
    )

    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
