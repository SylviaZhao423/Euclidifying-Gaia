#!/usr/bin/env python3
"""
Download Euclid Q1 VIS cutouts for DESI-confirmed galaxies cross-matched with Q1,
categorised by whether Q1 PHZ agrees or disagrees with the DESI galaxy label.

All sources are DESI-confirmed galaxies loaded from q1_allphz_desi.npz.
Crosshair coloured by Gaia corrected BP/RP excess (coolwarm: blue=star, red=galaxy).

Reads:
  output/umap/arrays/q1_allphz_desi.npz        (all PHZ×DESI with Gaia features)
  output/umap/arrays/q1_indiv.npz              (Q1 UMAP embedding + normalisation)
  output/umap/arrays/q1_indiv_reducer.joblib   (fitted UMAP reducer)
  output/umap/arrays/q1_cluster_labels.npy     (HDBSCAN cluster labels)

Categories are defined by the Q1 PHZ classification (9-class scheme):

  agree_both_gal                — phz=2   Q1 also calls it Galaxy
  disagree_desi_gal_q1_star_1   — phz=1   Q1 calls it Star
  disagree_desi_gal_q1_star_x_gal_3 — phz=3 Q1 calls it Multi (Star∩Gal)
  disagree_desi_gal_q1_qso_4    — phz=4   Q1 calls it QSO
  disagree_desi_gal_q1_star_x_qso_5 — phz=5
  disagree_desi_gal_q1_gal_x_qso_6  — phz=6
  disagree_desi_gal_q1_rej       — phz=-1  Q1 Rejected
  disagree_desi_gal_q1_outlier_0 — phz=0  Q1 Outlier

Output layout
-------------
output/euclid_cutouts/q1_cutouts/desi/
  {category}/
    cluster{k}/
      contact_sheet.png
      manifest.csv
      cutouts/
        cutout_NNNN.png

Usage
-----
  python src/download_q_desi_cutouts.py

  # Only the agree + star disagreement
  python src/download_q_desi_cutouts.py --phz 2,1

  # Custom per-cluster limit
  python src/download_q_desi_cutouts.py --per-cell 20
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
from render_q1_cutout import make_q1_cutouts, _discover_tiles

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
CUTOUTS_DIR = PROJECT_DIR / "output" / "euclid_cutouts" / "q1_cutouts" / "desi"

# Q1 PHZ codes (9-class scheme from crossmatch_q1_mer_phz_vis_gaia)
# phz=2 is the "agree" case; all others are disagreements with DESI galaxy label.
CATEGORY_DIRS = {
     2: "agree_both_gal",
     1: "disagree_desi_gal_q1_star_1",
     3: "disagree_desi_gal_q1_star_x_gal_3",
     4: "disagree_desi_gal_q1_qso_4",
     5: "disagree_desi_gal_q1_star_x_qso_5",
     6: "disagree_desi_gal_q1_gal_x_qso_6",
     7: "disagree_desi_gal_q1_all_three_7",
    -1: "disagree_desi_gal_q1_rej_n1",
     0: "disagree_desi_gal_q1_outlier_0",
}

PHZ_LABELS = {
     2: "Galaxy (agree)",
     1: "Star",
     3: "Multi (Star∩Gal)",
     4: "QSO",
     5: "Multi (Star∩QSO)",
     6: "Multi (Gal∩QSO)",
     7: "Multi (all three)",
    -1: "Rejected",
     0: "Outlier",
}

DEFAULT_PHZ = list(CATEGORY_DIRS.keys())


# ── Load from npz ────────────────────────────────────────────────────────────

def load_desi_sources(phz_values: list[int]) -> pd.DataFrame:
    """Load DESI-confirmed galaxies from q1_allphz_desi.npz (no DB needed)."""
    npz_path = ARRAYS_DIR / "q1_allphz_desi.npz"
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
    df["d_desi_z"]        = data["desi_z"].astype(float)[mask]
    df["d_desi_spectype"] = data["spectype"][mask]

    # XGBoost P(gal) from the Q1 seed ensemble
    from render_q1_cutout import predict_pgal
    X = data["X_features"][mask].astype(float)
    df["xgb_pgal"] = predict_pgal(X)

    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    phz_str = ", ".join(str(v) for v in phz_values)
    print(f"  Loaded phz IN ({phz_str}) from {npz_path.name}")
    print(f"  -> {len(df):,} total sources")
    for v in phz_values:
        lbl = PHZ_LABELS.get(v, str(v))
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
        description="Download Q1 VIS cutouts for DESI-confirmed galaxies, by Q1 PHZ category"
    )
    parser.add_argument("--phz", type=str, default=None,
                        help="Comma-separated Q1 phz values to include, e.g. --phz 2,1,4. "
                             "Default: all categories.")
    parser.add_argument("--per-cell", type=int, default=30,
                        help="Max sources per (category × cluster) cell (default: 30).")
    parser.add_argument("--size-arcsec", type=float, default=6.0,
                        help="VIS cutout size in arcsec (default: 6.0).")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--dpi",        type=int, default=90)
    parser.add_argument("--no-cutouts", action="store_true",
                        help="Skip cutout downloads; just show cluster assignment stats.")
    args = parser.parse_args()

    if args.phz is not None:
        try:
            phz_values = [int(x.strip()) for x in args.phz.split(",")]
        except ValueError:
            print("ERROR: --phz must be a comma-separated list of integers, e.g. --phz 2,1,-1")
            return 1
        unknown = [v for v in phz_values if v not in CATEGORY_DIRS]
        if unknown:
            print(f"ERROR: Unknown phz values: {unknown}. "
                  f"Supported: {sorted(CATEGORY_DIRS.keys())}")
            return 1
    else:
        phz_values = DEFAULT_PHZ

    print("=" * 60)
    print("DESI-confirmed galaxy cutouts — Q1 PHZ category breakdown")
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
        print("  Run first:  python src/analyze_umap_q1_clusters.py")
        return 1
    q1_labels = np.load(labels_path)
    cluster_ids = sorted(set(q1_labels[q1_labels >= 0]))
    print(f"  Loaded from: {labels_path}")
    print(f"  Cluster sizes: " +
          "  ".join(f"C{k}={(q1_labels==k).sum():,}" for k in cluster_ids))

    # ── 3. Load DESI sources from npz ────────────────────────────────────────
    print("\n[3] Loading DESI-confirmed galaxies from q1_allphz_desi.npz...")
    df_all = load_desi_sources(phz_values)

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
    df_all["cluster"] = assign_to_clusters(emb, q1_emb, q1_labels)

    print("\n  Cluster assignment summary:")
    for v in phz_values:
        lbl = PHZ_LABELS.get(v, str(v))
        sub = df_all[df_all["phz"] == v]
        if len(sub) == 0:
            continue
        print(f"  phz={v} {lbl}:")
        for k in cluster_ids:
            n = (sub["cluster"] == k).sum()
            if n:
                print(f"    C{k}: {n:,}")

    if args.no_cutouts:
        print("\n[6] --no-cutouts set, skipping downloads.")
        return 0

    # ── 6. Discover tiles once for all sources ────────────────────────────────
    # Use EDF-North field bounds (not data range, which may span multiple fields)
    print("\n[6] Discovering tiles (one TAP query for EDF-North)...")
    tiles = _discover_tiles(260.0, 278.0, 62.0, 70.0,
                            field_name="EDF-North")
    if tiles:
        print(f"  {len(tiles)} tile(s) found — reusing for all clusters.")
    else:
        print("  ESA TAP failed; each cluster will fall back to per-source IRSA TAP.")
        tiles = None

    # ── 7. Download cutouts per (category × cluster) ──────────────────────────
    print("\n[7] Downloading cutouts per (category × cluster) cell...")
    rng = np.random.default_rng(args.seed)

    for v in phz_values:
        lbl       = PHZ_LABELS.get(v, str(v))
        cat_dir   = CUTOUTS_DIR / CATEGORY_DIRS[v]
        df_cat    = df_all[df_all["phz"] == v]
        if len(df_cat) == 0:
            print(f"\n  phz={v} {lbl}: no sources, skipping.")
            continue

        print(f"\n  ── phz={v} {lbl} → {CATEGORY_DIRS[v]} ({len(df_cat):,} sources) ──")
        cat_dir.mkdir(parents=True, exist_ok=True)

        for k in cluster_ids:
            cell_df = df_cat[df_cat["cluster"] == k]
            if len(cell_df) == 0:
                continue

            if len(cell_df) > args.per_cell:
                cell_df = cell_df.sample(args.per_cell,
                                         random_state=int(rng.integers(1e6)))
            cell_df = cell_df.reset_index(drop=True)

            cluster_dir = cat_dir / f"cluster{k}"
            print(f"\n    [cluster{k}]  {len(cell_df)} sources")

            make_q1_cutouts(
                cell_df,
                output_dir=cluster_dir,
                size_arcsec=args.size_arcsec,
                c_star_col="xgb_pgal",
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
