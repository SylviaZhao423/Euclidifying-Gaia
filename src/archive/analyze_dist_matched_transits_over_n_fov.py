#!/usr/bin/env python3
"""
Distribution of matched_transits / N_fov_transits across all 6 comparison
slices in a single 6-panel figure (2 rows × 3 cols).

Row 1: Q1 vs ERO  —  All sources | Galaxies only | Stars only
Row 2: STAR vs GALAXY  —  Q1+ERO combined | ERO only | Q1 only

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_dist_matched_transits_over_n_fov.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB = "wsdb"
MAG_CUT = 17
SCAN_NSIDE = 64

COL = "matched_transits_over_n_fov"

Q1_COLOR = "#BA68C8"
ERO_COLOR = "#66BB6A"
STAR_COLOR = "coral"
GALAXY_COLOR = "steelblue"


def build_scanning_law_map(nside: int = SCAN_NSIDE) -> np.ndarray:
    """Build full-sky Gaia DR3 nominal FoV transit-count map."""
    from gaiaunlimited.scanninglaw import GaiaScanningLaw
    from tqdm import tqdm

    print("  Loading Gaia DR3 scanning law...")
    sl = GaiaScanningLaw("dr3_nominal")

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra_deg = np.degrees(phi)
    dec_deg = 90.0 - np.degrees(theta)

    print(f"  Querying scan counts for {npix} HEALPix pixels (nside={nside})...")

    def get_totaln(ra: float, dec: float) -> float:
        return sum(sl.query(ra, dec, count_only=True))

    result = [get_totaln(ra, dec) for ra, dec in tqdm(zip(ra_deg, dec_deg), total=npix)]
    return np.asarray(result, dtype=float)


def add_scan_normalized_column(df: pd.DataFrame, scan_map: np.ndarray, nside: int = SCAN_NSIDE) -> pd.DataFrame:
    """Attach local N_fov_transits and matched_transits / N_fov_transits."""
    theta = np.radians(90.0 - df["dec"].to_numpy())
    phi = np.radians(df["ra"].to_numpy())
    pix = hp.ang2pix(nside, theta, phi)
    n_fov = scan_map[pix].astype(float)

    out = df.copy()
    out["n_fov_transits"] = n_fov
    valid = np.isfinite(n_fov) & (n_fov > 0)
    out = out.loc[valid].copy()
    out[COL] = out["matched_transits"].astype(float) / out["n_fov_transits"]
    return out


def query_dataset(username: str, source: str) -> pd.DataFrame:
    """Return DataFrame with [ra, dec, matched_transits, label] for q1 or ero."""
    import psycopg2

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)

    if source == "q1":
        query = f"""
            SELECT
                g_ra AS ra,
                g_dec AS dec,
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
              AND g_ra IS NOT NULL
              AND g_dec IS NOT NULL
        """
    else:
        query = f"""
            SELECT
                g_ra AS ra,
                g_dec AS dec,
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
              AND g_ra IS NOT NULL
              AND g_dec IS NOT NULL
        """

    print(f"  Querying {source.upper()}...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    df["matched_transits"] = pd.to_numeric(df["matched_transits"], errors="coerce")
    print(
        f"  {source.upper()}: {len(df):,}  "
        f"STAR={(df['label'] == 'STAR').sum():,}  "
        f"GALAXY={(df['label'] == 'GALAXY').sum():,}"
    )
    return df


def _hist_panel(ax, data_a, data_b, label_a, label_b, color_a, color_b, title):
    """Draw overlapping density histograms on ax."""
    data_a = data_a.dropna()
    data_b = data_b.dropna()

    all_data = pd.concat([data_a, data_b])
    lo = max(0.0, float(all_data.quantile(0.01)))
    hi = float(all_data.quantile(0.99))
    if not np.isfinite(hi) or lo == hi:
        hi = lo + 0.1
    bins = np.linspace(lo, hi, 50)

    if len(data_a) > 0:
        ax.hist(
            data_a,
            bins=bins,
            alpha=0.5,
            density=True,
            color=color_a,
            label=f"{label_a} (n={len(data_a):,})",
        )
    if len(data_b) > 0:
        ax.hist(
            data_b,
            bins=bins,
            alpha=0.5,
            density=True,
            color=color_b,
            label=f"{label_b} (n={len(data_b):,})",
        )

    ax.set_xlabel(COL, fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")


def main() -> int:
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Distribution of {COL}  (G >= {MAG_CUT})")
    print("=" * 72)

    scan_map = build_scanning_law_map(nside=SCAN_NSIDE)

    q1_df = add_scan_normalized_column(query_dataset(username, "q1"), scan_map, nside=SCAN_NSIDE)
    ero_df = add_scan_normalized_column(query_dataset(username, "ero"), scan_map, nside=SCAN_NSIDE)
    combined_df = pd.concat([q1_df, ero_df], ignore_index=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    _hist_panel(
        axes[0, 0],
        q1_df[COL], ero_df[COL],
        "Q1", "ERO", Q1_COLOR, ERO_COLOR,
        "Q1 vs ERO — All sources",
    )
    _hist_panel(
        axes[0, 1],
        q1_df.loc[q1_df["label"] == "GALAXY", COL],
        ero_df.loc[ero_df["label"] == "GALAXY", COL],
        "Q1", "ERO", Q1_COLOR, ERO_COLOR,
        "Q1 vs ERO — Galaxies only",
    )
    _hist_panel(
        axes[0, 2],
        q1_df.loc[q1_df["label"] == "STAR", COL],
        ero_df.loc[ero_df["label"] == "STAR", COL],
        "Q1", "ERO", Q1_COLOR, ERO_COLOR,
        "Q1 vs ERO — Stars only",
    )

    for ax, df, title in [
        (axes[1, 0], combined_df, "STAR vs GALAXY — Q1+ERO combined"),
        (axes[1, 1], ero_df, "STAR vs GALAXY — ERO only"),
        (axes[1, 2], q1_df, "STAR vs GALAXY — Q1 only"),
    ]:
        _hist_panel(
            ax,
            df.loc[df["label"] == "GALAXY", COL],
            df.loc[df["label"] == "STAR", COL],
            "GALAXY", "STAR", GALAXY_COLOR, STAR_COLOR,
            title,
        )

    fig.suptitle(
        f"matched_transits / N_fov_transits distributions  (G >= {MAG_CUT})\n"
        "Row 1: Q1 vs ERO  |  Row 2: STAR vs GALAXY",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out = OUTPUT_DIR / "matched_transits_over_n_fov_dist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
