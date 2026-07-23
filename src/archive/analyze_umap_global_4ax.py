#!/usr/bin/env python3
"""
Overlay 4 axes on the original 14-feature Q1 UMAP (all 50k sources).
Each panel is coloured by one axis; cluster centroids are annotated.

  AEN sig                 — astrometric_excess_noise_sig  (X_raw[:, 3])
  BPRP excess             — corrected_bp_rp_excess        (X_raw[:, 11])
  variability             — sqrt(phot_g_n_obs)/phot_g_mean_flux_over_error
                            (q1_addvar.npz column 14)
  astrometric_params_solved — discrete integer from Gaia DR3 (3 / 31 / 95)
                              fetched from WSDB

Reads:
  output/umap/arrays/q1_indiv.npz          — original 14-feat embedding + source_ids
  output/umap/arrays/q1_cluster_labels.npy — cluster assignments
  output/umap/q1_addvar/q1_addvar.npz      — X_raw col 14 = variability

Writes:
  output/umap/q1_clusters_3ax/umap_global_4ax.png

Usage:
  python src/analyze_umap_global_4ax.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

ARRAYS_DIR = PROJECT_DIR / "output" / "umap" / "arrays"
ADDVAR_NPZ = PROJECT_DIR / "output" / "umap" / "q1_addvar" / "q1_addvar.npz"
OUTPUT_DIR = PROJECT_DIR / "output" / "umap" / "q1_clusters_3ax"

from selected_parameters_list import SELECTED_GAIA_PARAMETERS  # noqa: E402

AEN_IDX    = SELECTED_GAIA_PARAMETERS.index("astrometric_excess_noise_sig")
BPRPEF_IDX = SELECTED_GAIA_PARAMETERS.index("corrected_bp_rp_excess")

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
WSDB_USER = "siwei_zhao_2025"

CLUSTER_COLOURS = {
    0: "#e6194b",
    1: "#4363d8",
    2: "#f58231",
    3: "#3cb44b",
    4: "#911eb4",
}


def fetch_astrometric_params_solved(source_ids: np.ndarray) -> np.ndarray:
    """Fetch astrometric_params_solved for each source_id from WSDB.

    Returns an int array aligned with source_ids; NaN-filled (as float) where missing.
    """
    import psycopg2

    print(f"  Connecting to WSDB ({WSDB_HOST})...")
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=WSDB_USER)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("""
        CREATE TEMP TABLE _umap_sids (source_id BIGINT PRIMARY KEY)
    """)

    sids_int = [int(s) for s in source_ids]
    chunk = 5000
    for start in range(0, len(sids_int), chunk):
        batch = sids_int[start:start + chunk]
        args = ",".join(f"({sid})" for sid in batch)
        cur.execute(f"INSERT INTO _umap_sids VALUES {args} ON CONFLICT DO NOTHING")
        if (start // chunk) % 10 == 0:
            print(f"    inserted {min(start + chunk, len(sids_int)):,}/{len(sids_int):,}")

    print("  Querying astrometric_params_solved...")
    cur.execute("""
        SELECT s.source_id, g.astrometric_params_solved
        FROM _umap_sids s
        JOIN gaia_dr3.gaia_source g ON g.source_id = s.source_id
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    print(f"  Retrieved {len(rows):,} rows")

    sid_to_val: dict[int, int] = {int(r[0]): int(r[1]) for r in rows if r[1] is not None}
    result = np.array(
        [float(sid_to_val.get(int(s), np.nan)) for s in source_ids],
        dtype=float,
    )
    n_found = np.isfinite(result).sum()
    unique_vals = sorted({v for v in sid_to_val.values()})
    print(f"  Found: {n_found:,}/{len(source_ids):,}  unique values: {unique_vals}")
    return result


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    print("Loading arrays...")
    data   = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    labels = np.load(ARRAYS_DIR / "q1_cluster_labels.npy")
    addvar = np.load(ADDVAR_NPZ, allow_pickle=True)

    embedding  = data["embedding"]
    X_raw      = data["X_raw"]
    y          = data["y"]
    source_ids = data["source_ids"].astype(str)

    variability = addvar["X_raw"][:, 14]

    # ── Fetch astrometric_params_solved ───────────────────────────────────
    aps = fetch_astrometric_params_solved(source_ids)

    # ── Cluster metadata ──────────────────────────────────────────────────
    cluster_ids = sorted(set(labels[labels >= 0]))
    centroids = {
        cid: (embedding[labels == cid, 0].mean(),
              embedding[labels == cid, 1].mean())
        for cid in cluster_ids
    }
    cluster_galpct = {
        cid: 100 * (y[labels == cid] == 1).sum() / (labels == cid).sum()
        for cid in cluster_ids
    }

    # ── Axis specs ────────────────────────────────────────────────────────
    xlim = (embedding[:, 0].min() - 0.8, embedding[:, 0].max() + 0.8)
    ylim = (embedding[:, 1].min() - 0.8, embedding[:, 1].max() + 0.8)

    # Discrete colormap for astrometric_params_solved (values 3, 31, 95)
    unique_aps = sorted(v for v in np.unique(aps[np.isfinite(aps)]))
    n_levels   = len(unique_aps)
    aps_cmap   = plt.cm.get_cmap("viridis", n_levels)
    aps_norm   = mcolors.BoundaryNorm(
        boundaries=[unique_aps[0] - 1] + [u + (unique_aps[i+1] - u) / 2
                                           for i, u in enumerate(unique_aps[:-1])] + [unique_aps[-1] + 1],
        ncolors=n_levels,
    )

    axes_spec = [
        dict(name="AEN sig",    vals=X_raw[:, AEN_IDX],    cmap="RdYlBu_r", norm=None, discrete=False),
        dict(name="BPRP excess", vals=X_raw[:, BPRPEF_IDX], cmap="RdYlBu_r", norm=None, discrete=False),
        dict(name="variability", vals=variability,           cmap="RdYlBu_r", norm=None, discrete=False),
        dict(name="astrometric_params_solved", vals=aps,     cmap=aps_cmap,   norm=aps_norm, discrete=True),
    ]

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    fig.subplots_adjust(wspace=0.08, left=0.03, right=0.97, top=0.88, bottom=0.04)

    for ax, spec in zip(axes, axes_spec):
        v    = spec["vals"]
        cmap = spec["cmap"]
        norm = spec["norm"]

        if spec["discrete"]:
            finite = v[np.isfinite(v)]
            sc = ax.scatter(
                embedding[:, 0], embedding[:, 1],
                c=v, cmap=cmap, norm=norm,
                s=2, alpha=0.7, linewidths=0, rasterized=True,
            )
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.set_ticks(unique_aps)
            cb.set_ticklabels([str(int(u)) for u in unique_aps])
            cb.ax.tick_params(labelsize=8)
        else:
            lo, hi = np.nanpercentile(v, 1), np.nanpercentile(v, 99)
            vc = np.clip(v, lo, hi)
            sc = ax.scatter(
                embedding[:, 0], embedding[:, 1],
                c=vc, cmap=cmap, vmin=lo, vmax=hi,
                s=2, alpha=0.55, linewidths=0, rasterized=True,
            )
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=8)

        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(spec["name"], fontsize=12, fontweight="bold", pad=6)
        ax.grid(alpha=0.15)

        for cid in cluster_ids:
            cx, cy = centroids[cid]
            gp = cluster_galpct[cid]
            txt = ax.text(
                cx, cy, f"C{cid}\n{gp:.0f}%gal",
                fontsize=8, fontweight="bold", ha="center", va="center",
                color=CLUSTER_COLOURS[cid],
            )
            txt.set_path_effects([
                pe.Stroke(linewidth=2, foreground="white"),
                pe.Normal(),
            ])

    fig.suptitle(
        "Q1 UMAP (original 14-feature embedding) coloured by animation sort axes + astrometric solution\n"
        "Cluster centroids labelled with galaxy fraction",
        fontsize=11, fontweight="bold",
    )

    out = OUTPUT_DIR / "umap_global_4ax.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
