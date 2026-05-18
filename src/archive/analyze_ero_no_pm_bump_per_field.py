#!/usr/bin/env python3
"""
Diagnostic: where do ERO no-PM (Gaia 2-par) sources live, and which ERO
fields produce the bump at sep ~0.13″?

Splits the no-PM sample by closest ERO field and overlays per-field
histograms of the raw separation. Expected outcome: the bump is dominated
by NGC 6397 (densest field, many faint cluster members below Gaia's PM
detection limit, all displaced by the cluster's bulk PM).

Reads (WSDB):
  siwei_zhao_2025.crossmatch_ero_gaia_0_3
    columns: e_v_alpha_j2000, e_v_delta_j2000,
             g_ra, g_dec, g_pmra, g_pmdec, g_astrometric_params_solved

Writes:
  output/ero_no_pm_bump_per_field.png

Usage:
  PGUSER=siwei_zhao_2025 python src/slides/analyze_ero_no_pm_bump_per_field.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_DIR / "output" / "ero_no_pm_bump_per_field.png"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

# ERO field centres from src/slides/analyze_dr3_scanning_law.py
ERO_FIELDS = {
    "Horsehead":    ( 85.150,  -2.613),
    "Messier 78":   ( 86.690,  -0.015),
    "Taurus":       ( 64.983, +28.023),
    "Barnard 30":   ( 82.880, +12.316),
    "NGC 6254":     (254.303,  -4.100),
    "NGC 6397":     (265.174, -53.658),
    "NGC 6822":     (296.236, -14.788),
    "Holmberg II":  (124.790, +70.706),
    "IC 10":        (  5.063, +59.288),
    "IC 342":       ( 56.730, +68.084),
    "NGC 2403":     (114.211, +65.586),
    "NGC 6744":     (287.430, -63.842),
    "Fornax":       ( 54.017, -35.267),
    "Dorado":       ( 64.014, -55.780),
    "Perseus":      ( 49.638, +41.651),
    "Abell 2390":   (328.397, +17.709),
    "Abell 2764":   (  5.713, -49.249),
}
FIELD_RADIUS_DEG = 0.6   # ERO fields are ~0.5–0.75 deg² → ~0.5 deg radius


def query_data(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cur = conn.cursor()
    cur.execute("""
        SELECT e_v_alpha_j2000, e_v_delta_j2000,
               g_ra, g_dec, g_pmra, g_pmdec,
               g_astrometric_params_solved AS params_solved
        FROM siwei_zhao_2025.crossmatch_ero_gaia_0_3
        WHERE g_astrometric_params_solved = 3
    """)
    rows = cur.fetchall()
    cur.close(); conn.close()
    print(f"  Loaded {len(rows):,} ERO no-PM (2-par) sources")
    return rows


def assign_field(g_ra, g_dec):
    """Assign each source to its nearest ERO field (or 'other' if outside)."""
    field_names = list(ERO_FIELDS.keys())
    field_ra  = np.array([ERO_FIELDS[f][0] for f in field_names])
    field_dec = np.array([ERO_FIELDS[f][1] for f in field_names])

    cos_dec = np.cos(np.radians(g_dec))
    # Sky distance to every field for every source
    dra  = (g_ra[:, None]  - field_ra[None, :]) * cos_dec[:, None]
    ddec =  g_dec[:, None] - field_dec[None, :]
    dist = np.sqrt(dra ** 2 + ddec ** 2)

    nearest_idx = np.argmin(dist, axis=1)
    nearest_d   = dist[np.arange(len(g_ra)), nearest_idx]

    assigned = np.array(field_names, dtype=object)[nearest_idx]
    assigned[nearest_d > FIELD_RADIUS_DEG] = "other"
    return assigned


def main() -> int:
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set"); return 1

    print("Querying ERO 2-par (no-PM) sources...")
    rows = query_data(username)

    e_ra  = np.array([r[0] for r in rows], dtype=float)
    e_dec = np.array([r[1] for r in rows], dtype=float)
    g_ra  = np.array([r[2] for r in rows], dtype=float)
    g_dec = np.array([r[3] for r in rows], dtype=float)

    cos_dec = np.cos(np.radians(g_dec))
    sep_arcsec = np.sqrt(((e_ra - g_ra) * cos_dec) ** 2 +
                         (e_dec - g_dec) ** 2) * 3600

    print("Assigning each no-PM source to its nearest ERO field...")
    field = assign_field(g_ra, g_dec)

    # Counts per field, sorted by total
    from collections import Counter
    field_counts = Counter(field.tolist())
    sorted_fields = sorted(field_counts.items(), key=lambda kv: -kv[1])

    print()
    print(f"  {'Field':<14}  {'n':>7}  {'frac':>6}  "
          f"{'med sep':>8}  {'n in [0.10, 0.15]':>17}")
    print(f"  {'-'*14}  {'-'*7}  {'-'*6}  {'-'*8}  {'-'*17}")
    n_total = len(field)
    for fname, n in sorted_fields:
        m = field == fname
        bump_n = ((sep_arcsec[m] >= 0.10) & (sep_arcsec[m] <= 0.15)).sum()
        print(f"  {fname:<14}  {n:>7,}  {100*n/n_total:>5.1f}%  "
              f"{np.median(sep_arcsec[m]):>7.4f}″  {bump_n:>17,}")

    # ── Plot: overlay top 6 fields + 'other' ────────────────────────────────
    TOP_N = 6
    top_fields = [fname for fname, _ in sorted_fields[:TOP_N]]
    other_mask = ~np.isin(field, top_fields + ["other"])
    n_other = int(other_mask.sum() + (field == "other").sum())

    x_limit  = 0.6
    bin_edges = np.linspace(0, x_limit, 101)

    fig, ax = plt.subplots(figsize=(10, 5.0))

    # Background: the entire no-PM sample
    ax.hist(sep_arcsec[sep_arcsec <= x_limit], bins=bin_edges, density=False,
            color="#CCCCCC", alpha=0.45, edgecolor="none", zorder=1,
            label=f"All no-PM (n={len(sep_arcsec):,})")

    # Highlight top fields + remainder
    cmap = plt.get_cmap("tab10")
    for i, fname in enumerate(top_fields):
        m = (field == fname) & (sep_arcsec <= x_limit)
        ax.hist(sep_arcsec[m], bins=bin_edges, density=False,
                histtype="step", color=cmap(i), linewidth=1.8, zorder=3 + i,
                label=f"{fname}  (n={int(m.sum()):,})")

    rest_mask = (other_mask | (field == "other")) & (sep_arcsec <= x_limit)
    if rest_mask.any():
        ax.hist(sep_arcsec[rest_mask], bins=bin_edges, density=False,
                histtype="step", color="black", linewidth=1.2,
                linestyle="--", zorder=2,
                label=f"other ({len(ERO_FIELDS) - TOP_N} fields + outside, "
                      f"n={int(rest_mask.sum()):,})")

    ax.set_xlim(0, x_limit)
    ax.set_xlabel("Angular Separation (arcsec)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Counts per bin", fontsize=14, fontweight="bold", labelpad=15)
    ax.set_title("ERO no-PM (2-par Gaia) sources by field",
                 fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=9, loc="upper right", ncol=1, framealpha=0.85)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
