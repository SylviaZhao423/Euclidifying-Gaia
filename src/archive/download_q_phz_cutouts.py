#!/usr/bin/env python3
"""
Download Euclid Q1 VIS cutouts for all phz classes, stratified by UMAP cluster.

Handles phz = 1 (Star), 2 (Galaxy), 3 (Multi Star∩Gal), 4 (QSO),
         5 (Multi Star∩QSO), 6 (Multi Gal∩QSO), 7 (Multi all three),
        -1 (Rejected), 0 (Outlier).

Crosshair coloured by Gaia corrected BP/RP excess (coolwarm: blue=star, red=galaxy).

Reads:
  output/umap/arrays/q1_allphz.npz             (all PHZ classes with Gaia features)
  output/umap/arrays/q1_indiv.npz              (Q1 UMAP embedding + normalisation)
  output/umap/arrays/q1_indiv_reducer.joblib   (fitted UMAP reducer)
  output/umap/arrays/q1_cluster_labels.npy     (HDBSCAN cluster labels)

Output layout
-------------
output/euclid_cutouts/q1_cutouts/phz/
  1_star/        (phz=1)
  2_gal/         (phz=2)
  3_star_x_gal/  (phz=3)
  4_qso/         (phz=4)
  5_star_x_qso/  (phz=5)
  6_gal_x_qso/   (phz=6)
  7_all_three/ (phz=7)
  -1_rej/        (phz=-1)
  0_outlier/     (phz=0)
    cluster{k}/
      contact_sheet.png
      manifest.csv
      cutouts/
        cutout_NNNN.png

Usage
-----
  python src/download_q_q1_cutouts.py

  # Download only stars and rejected
  python src/download_q_q1_cutouts.py --phz 1,-1

  # Download only QSO classes
  python src/download_q_q1_cutouts.py --phz 4,5,6

  # Download galaxies with 20 per cluster
  python src/download_q_q1_cutouts.py --phz 2 --per-cell 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS
from save_q1_utils import make_q1_cutouts, _discover_tiles

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
CUTOUTS_DIR = PROJECT_DIR / "output" / "euclid_cutouts" / "q1_cutouts" / "phz"

PHZ_LABELS = {
     1: "Star",
     2: "Galaxy",
     3: "Multi (Star∩Gal)",
     4: "QSO",
     5: "Multi (Star∩QSO)",
     6: "Multi (Gal∩QSO)",
     7: "Multi (all three)",
    -1: "Rejected",
     0: "Outlier",
}

PHZ_DIRS = {
     1: "1_star",
     2: "2_gal",
     3: "3_star_x_gal",
     4: "4_qso",
     5: "5_star_x_qso",
     6: "6_gal_x_qso",
     7: "7_all_three",
    -1: "-1_rej",
     0: "0_outlier",
}

DEFAULT_PHZ = [1, 2, 3, 4, 5, 6, 7, -1, 0]


# ── Load from npz ────────────────────────────────────────────────────────────

def load_phz_sources(phz_values: list[int]) -> pd.DataFrame:
    """Load selected phz classes from q1_allphz.npz (no DB needed)."""
    npz_path = ARRAYS_DIR / "q1_allphz.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found. Run query_umap_array.py first.")
    data = np.load(npz_path, allow_pickle=True)

    phz_arr = data["phz"].astype(int)
    mask = np.isin(phz_arr, phz_values)

    df = pd.DataFrame(data["X_features"][mask], columns=SELECTED_GAIA_PARAMETERS)
    df["source_id"]       = data["source_ids"][mask]
    df["ra"]              = data["ra"].astype(float)[mask]
    df["dec"]             = data["dec"].astype(float)[mask]
    df["phot_g_mean_mag"] = data["phot_g_mean_mag"].astype(float)[mask]
    df["phz"]             = phz_arr[mask]

    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    phz_str = ", ".join(str(v) for v in phz_values)
    print(f"  Loaded phz IN ({phz_str}) from {npz_path.name}")
    print(f"  -> {len(df):,} total sources")
    for v in phz_values:
        lbl = PHZ_LABELS[v]
        n = (df["phz"] == v).sum()
        if n:
            print(f"     phz={v}  {lbl}: {n:,}")
    return df


# ── Cluster assignment ────────────────────────────────────────────────────────

def assign_to_clusters(src_emb: np.ndarray,
                       q1_emb: np.ndarray,
                       q1_labels: np.ndarray) -> np.ndarray:
    valid_mask = q1_labels >= 0
    if valid_mask.sum() == 0:
        return np.full(len(src_emb), -1, dtype=int)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(q1_emb[valid_mask])
    _, idx = nn.kneighbors(src_emb)
    idx = idx.ravel()
    valid_indices = np.where(valid_mask)[0]
    return q1_labels[valid_indices[idx]]


# ── Title factory ─────────────────────────────────────────────────────────────

def _make_title(phz_val: int, lbl: str, k: int):
    def _title(row, size, _phz=phz_val, _lbl=lbl, _k=k):
        sid = str(row.get('source_id', ''))
        return (f"phz={_phz} {_lbl}  Cluster {_k}\n"
                f"RA={float(row['ra']):.4f}  Dec={float(row['dec']):+.4f}\n"
                f"G={float(row['phot_g_mean_mag']):.2f}  id={sid}")
    return _title


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Q1 VIS cutouts for all phz classes, stratified by cluster"
    )
    parser.add_argument("--phz", type=str, default=None,
                        help="Comma-separated phz values to download, e.g. --phz 1,2,-1. "
                             "Default: all of 1,2,3,4,5,6,7,-1,0.")
    parser.add_argument("--per-cell", type=int, default=30,
                        help="Max sources per (phz × cluster) cell (default: 30).")
    parser.add_argument("--size-arcsec", type=float, default=6.0,
                        help="VIS cutout size in arcsec (default: 6.0).")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--dpi",        type=int,   default=90)
    parser.add_argument("--no-cutouts", action="store_true",
                        help="Skip cutout downloads; just show cluster assignment stats.")
    args = parser.parse_args()

    # Parse --phz argument (handle negatives like -1)
    if args.phz is not None:
        try:
            phz_values = [int(x.strip()) for x in args.phz.split(",")]
        except ValueError:
            print("ERROR: --phz must be a comma-separated list of integers, e.g. --phz 1,2,-1")
            return 1
        unknown = [v for v in phz_values if v not in PHZ_LABELS]
        if unknown:
            print(f"ERROR: Unknown phz values: {unknown}. "
                  f"Supported: {sorted(PHZ_LABELS.keys())}")
            return 1
    else:
        phz_values = DEFAULT_PHZ

    print("=" * 60)
    phz_desc = ", ".join(f"phz={v} ({PHZ_LABELS[v]})" for v in phz_values)
    print(f"Downloading cutouts for: {phz_desc}")
    print("=" * 60)

    # ── 1. Load Q1 UMAP ───────────────────────────────────────────────────────
    print("\n[1] Loading Q1 UMAP embedding...")
    q1 = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    q1_emb  = q1["embedding"]
    q1_mean = q1["mean"]
    q1_std  = q1["std"]
    reducer = joblib.load(ARRAYS_DIR / "q1_indiv_reducer.joblib")
    print(f"  Q1 embedding: {q1_emb.shape[0]:,} points")

    # ── 2. Load cluster labels ─────────────────────────────────────────────────
    print("\n[2] Loading cluster labels...")
    labels_path = ARRAYS_DIR / "q1_cluster_labels.npy"
    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found.")
        print("  Run first:  python src/compute_q1_clusters.py")
        return 1
    q1_labels = np.load(labels_path)
    print(f"  Loaded from: {labels_path}")
    cluster_ids = sorted(set(q1_labels[q1_labels >= 0]))
    print(f"  Cluster sizes: " +
          "  ".join(f"C{k}={(q1_labels==k).sum():,}" for k in cluster_ids))

    # ── 3. Load selected phz classes from npz ────────────────────────────────
    print("\n[3] Loading sources from q1_allphz.npz...")
    df_all = load_phz_sources(phz_values)

    # ── 4. Project into UMAP space ────────────────────────────────────────────
    print("\n[4] Projecting into UMAP space...")
    X = df_all[SELECTED_GAIA_PARAMETERS].values.astype(float)
    std_safe = np.where(q1_std > 0, q1_std, 1.0)
    X_scaled = (X - q1_mean) / std_safe
    X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0.0)
    emb = reducer.transform(X_scaled)
    print(f"  Projected {len(emb):,} sources")

    df_all = df_all.copy()
    df_all["umap1"]   = emb[:, 0]
    df_all["umap2"]   = emb[:, 1]

    # ── 5. Assign to nearest Q1 cluster ───────────────────────────────────────
    print("\n[5] Assigning to nearest cluster...")
    src_cluster = assign_to_clusters(emb, q1_emb, q1_labels)
    df_all["cluster"] = src_cluster

    print("\n  Cluster assignment summary:")
    for v in phz_values:
        lbl = PHZ_LABELS[v]
        sub = df_all[df_all["phz"] == v]
        if len(sub) == 0:
            continue
        print(f"  phz={v} {lbl}:")
        for k in cluster_ids:
            n = (sub["cluster"] == k).sum()
            if n:
                print(f"    C{k}: {n:,}")
        n_noise = (sub["cluster"] == -1).sum()
        if n_noise:
            print(f"    unassigned: {n_noise:,}")

    if args.no_cutouts:
        print("\n[6] --no-cutouts set, skipping downloads.")
        return 0

    # ── 6. Discover tiles once for all sources ────────────────────────────────
    # Use EDF-North field bounds (not data range, which spans multiple fields)
    print("\n[6] Discovering tiles (one TAP query for EDF-North)...")
    tiles = _discover_tiles(260.0, 278.0, 62.0, 70.0,
                            field_name="EDF-North")
    if tiles:
        print(f"  {len(tiles)} tile(s) found — reusing for all clusters.")
    else:
        print("  ESA TAP failed; each cluster will fall back to per-source IRSA TAP.")
        tiles = None

    # ── 7. Download cutouts per (phz × cluster) cell ──────────────────────────
    print("\n[7] Downloading cutouts per (phz × cluster) cell...")
    rng = np.random.default_rng(args.seed)

    for v in phz_values:
        lbl        = PHZ_LABELS[v]
        subdir     = PHZ_DIRS[v]
        phz_dir    = CUTOUTS_DIR / subdir
        df_phz     = df_all[df_all["phz"] == v]
        per_cell   = args.per_cell
        if len(df_phz) == 0:
            print(f"\n  phz={v} {lbl}: no sources, skipping.")
            continue

        print(f"\n  ── phz={v} {lbl} ({len(df_phz):,} sources) ──")
        phz_dir.mkdir(parents=True, exist_ok=True)

        for k in cluster_ids:
            cell_df = df_phz[df_phz["cluster"] == k]
            if len(cell_df) == 0:
                continue

            if len(cell_df) > per_cell:
                cell_df = cell_df.sample(per_cell,
                                         random_state=int(rng.integers(1e6)))
            cell_df = cell_df.reset_index(drop=True)

            cluster_dir = phz_dir / f"cluster{k}"
            print(f"\n    [cluster{k}]  {len(cell_df)} sources")

            make_q1_cutouts(
                cell_df,
                output_dir=cluster_dir,
                size_arcsec=args.size_arcsec,
                c_star_col="corrected_bp_rp_excess",
                title_fn=_make_title(v, lbl, k),
                tiles=tiles,
                contact_sheet_stem="contact_sheet",
                manifest_stem="manifest",
                individuals_dir=cluster_dir / "cutouts",
                dpi=args.dpi,
                save_individuals=True,
            )

    print("\n" + "=" * 60)
    print(f"Done. Outputs in: {CUTOUTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
