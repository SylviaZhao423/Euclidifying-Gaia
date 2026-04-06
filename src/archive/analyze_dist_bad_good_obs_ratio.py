#!/usr/bin/env python3
"""
Distribution of astrometric_n_bad_good_obs_al across all 6 comparison slices
in a single 6-panel figure (2 rows × 3 cols).

Row 1: Q1 vs ERO  —  All sources | Galaxies only | Stars only
Row 2: STAR vs GALAXY  —  Q1+ERO combined | ERO only | Q1 only

Reads the pre-computed derived column g_astrometric_n_bad_good_obs_al
(added by add_derived_columns.py).

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_dist_bad_good_obs_ratio.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = PROJECT_DIR / "output" 

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
MAG_CUT   = 17

COL       = "astrometric_n_bad_good_obs_al"   # local name (no g_ prefix)
DB_COL    = f"g_{COL}"                         # DB column name

Q1_COLOR     = "#BA68C8"   # pink
ERO_COLOR    = "#66BB6A"   # green
STAR_COLOR   = "coral"
GALAXY_COLOR = "steelblue"


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query_dataset(username: str, source: str) -> pd.DataFrame:
    """Return DataFrame with columns [COL, label] for q1 or ero."""
    import psycopg2

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)

    if source == "q1":
        query = f"""
            SELECT
                {DB_COL} AS {COL},
                CASE
                    WHEN p_phz_classification = 1 THEN 'STAR'
                    WHEN p_phz_classification = 2 THEN 'GALAXY'
                END AS label
            FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
            WHERE p_phz_classification IN (1, 2)
              AND g_phot_g_mean_mag >= {MAG_CUT}
              AND {DB_COL} IS NOT NULL
        """
    else:
        query = f"""
            SELECT
                {DB_COL} AS {COL},
                CASE
                    WHEN e_v_class_star >= 0.65 THEN 'STAR'
                    WHEN e_v_class_star <  0.65 THEN 'GALAXY'
                END AS label
            FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
            WHERE e_v_class_star IS NOT NULL
              AND g_phot_g_mean_mag >= {MAG_CUT}
              AND {DB_COL} IS NOT NULL
        """

    print(f"  Querying {source.upper()}...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    df[COL] = pd.to_numeric(df[COL], errors="coerce")
    print(f"  {source.upper()}: {len(df):,}  "
          f"STAR={( df['label']=='STAR').sum():,}  "
          f"GALAXY={(df['label']=='GALAXY').sum():,}")
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _hist_panel(ax, data_a, data_b, label_a, label_b, color_a, color_b, title):
    """Draw overlapping density histograms on ax."""
    data_a = data_a.dropna()
    data_b = data_b.dropna()

    all_data = pd.concat([data_a, data_b])
    lo = all_data.quantile(0.01)
    hi = all_data.quantile(0.99)
    if lo == hi:
        lo -= 0.5; hi += 0.5
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Distribution of {COL}  (G ≥ {MAG_CUT})")
    print("=" * 70)

    q1_df  = query_dataset(username, "q1")
    ero_df = query_dataset(username, "ero")

    combined_df = pd.concat([q1_df, ero_df], ignore_index=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    # ── Row 0: Q1 vs ERO ──────────────────────────────────────────────────
    _hist_panel(axes[0, 0],
                q1_df[COL], ero_df[COL],
                "Q1", "ERO", Q1_COLOR, ERO_COLOR,
                "Q1 vs ERO — All sources")

    _hist_panel(axes[0, 1],
                q1_df.loc[q1_df["label"] == "GALAXY", COL],
                ero_df.loc[ero_df["label"] == "GALAXY", COL],
                "Q1", "ERO", Q1_COLOR, ERO_COLOR,
                "Q1 vs ERO — Galaxies only")

    _hist_panel(axes[0, 2],
                q1_df.loc[q1_df["label"] == "STAR", COL],
                ero_df.loc[ero_df["label"] == "STAR", COL],
                "Q1", "ERO", Q1_COLOR, ERO_COLOR,
                "Q1 vs ERO — Stars only")

    # ── Row 1: STAR vs GALAXY ─────────────────────────────────────────────
    for ax, df, title in [
        (axes[1, 0], combined_df, "STAR vs GALAXY — Q1+ERO combined"),
        (axes[1, 1], ero_df,      "STAR vs GALAXY — ERO only"),
        (axes[1, 2], q1_df,       "STAR vs GALAXY — Q1 only"),
    ]:
        _hist_panel(ax,
                    df.loc[df["label"] == "GALAXY", COL],
                    df.loc[df["label"] == "STAR",   COL],
                    "GALAXY", "STAR", GALAXY_COLOR, STAR_COLOR,
                    title)

    fig.suptitle(
        f"astrometric_n_bad_good_obs_al distributions  (G ≥ {MAG_CUT})\n"
        "Row 1: Q1 vs ERO  |  Row 2: STAR vs GALAXY",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out = OUTPUT_DIR / "bad_good_obs_ratio_dist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
