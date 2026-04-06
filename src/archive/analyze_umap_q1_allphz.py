#!/usr/bin/env python3
"""
Q1 all-PHZ UMAP — fit UMAP on all PHZ classification categories.

Unlike analyze_umap_q1.py (which fits on phz=1/2 only), this script
includes ALL PHZ classes so the UMAP can discover structure beyond
the star/galaxy boundary (blends, QSOs, uncertain, unclassified).

Each class is capped at max_per_class sources so rare classes (QSOs)
are not drowned out by the 181k stars.

Writes:
  output/umap/q1_allphz/q1_allphz_umap.png

Usage:
  PGUSER=siwei_zhao_2025 python src/analyze_umap_q1_allphz.py
  PGUSER=siwei_zhao_2025 python src/analyze_umap_q1_allphz.py --test-mode
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap" / "q1_allphz"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
MAG_CUT   = 17

# DB phz_classification value → (internal key, label, color)
# QSO codes 4/5/6 are merged
PHZ_MAP = {
     1: (0, "STAR",           "#1976D2"),   # blue
     2: (1, "GALAXY",         "#E53935"),   # red
    -1: (2, "Unclassified",   "#9C27B0"),   # purple
     0: (3, "Uncertain",      "#FF9800"),   # orange
     3: (4, "Star-gal blend", "#FFC107"),   # amber
    99: (5, "QSO (4/5/6)",    "#4CAF50"),   # green  (merged sentinel)
}
# DB values that map to QSO sentinel
QSO_DB_VALS = {4, 5, 6}


def query_all_phz(username: str, max_per_class: int = 10_000,
                  test_mode: bool = False) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[int, int]]]:
    """Query all PHZ classes from DB, cap each at max_per_class.

    Returns:
      X           — (N × n_features)
      y           — (N,) internal keys 0–5
      class_counts — dict: internal_key → (total_before_cap, used_after_cap)
    """
    import psycopg2

    gaia_cols = ", ".join([f"g_{col} AS {col}" for col in SELECTED_GAIA_PARAMETERS])
    limit = "LIMIT 5000" if test_mode else ""
    query = f"""
        SELECT {gaia_cols}, p_phz_classification
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE g_phot_g_mean_mag >= {MAG_CUT}
          AND p_phz_classification IS NOT NULL
        ORDER BY g_source_id
        {limit}
    """
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    df = pd.read_sql_query(query, conn)
    conn.close()

    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace("NOT_AVAILABLE", np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Merge QSO codes into sentinel 99
    df["phz"] = df["p_phz_classification"].apply(
        lambda v: 99 if int(v) in QSO_DB_VALS else int(v)
    )

    rng = np.random.RandomState(42)
    parts = []
    class_counts: dict[int, tuple[int, int]] = {}
    for db_val, (key, label, _) in PHZ_MAP.items():
        mask = df["phz"] == db_val
        sub  = df[mask]
        n    = len(sub)
        if n == 0:
            continue
        if n > max_per_class:
            sub = sub.sample(max_per_class, random_state=42)
        parts.append((sub, key, label, n))
        class_counts[key] = (n, len(sub))
        print(f"  phz={db_val:>3}  {label:<20} total={n:>7,}  used={len(sub):>6,}")

    combined = pd.concat([p[0] for p in parts], ignore_index=True)
    # shuffle so UMAP doesn't see classes in blocks
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    X = combined[SELECTED_GAIA_PARAMETERS].values.astype(float)
    y_keys = np.array([
        PHZ_MAP[v][0] for v in combined["phz"].values
    ])

    # Impute NaN with column median
    for c in range(X.shape[1]):
        bad = np.isnan(X[:, c])
        if bad.any():
            X[bad, c] = np.nanmedian(X[:, c])

    return X, y_keys, class_counts


def fit_umap(X: np.ndarray, random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardise X and fit UMAP. Returns embedding, mean, std."""
    mean_ = np.nanmean(X, axis=0)
    std_  = np.nanstd(X, axis=0)
    X_scaled = (X - mean_) / np.where(std_ > 0, std_, 1.0)

    print(f"  Fitting UMAP on {len(X):,} sources ({X.shape[1]} features)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                        random_state=random_state, n_jobs=1)
    embedding = reducer.fit_transform(X_scaled)
    return embedding, mean_, std_


def plot_allphz_umap(embedding: np.ndarray, y: np.ndarray, output_dir: Path,
                     class_counts: dict[int, tuple[int, int]] | None = None,
                     max_points_per_group: int = 15_000,
                     filename: str = "q1_allphz_umap.png") -> None:
    """2-row × 7-col plot.

    Row 1: scatter — col 0 = all classes, cols 1–6 = one class each.
    Row 2: 2D density histogram weighted by true counts before capping,
           so relative densities reflect the real source population.
    """
    rng = np.random.RandomState(0)

    x_mid = (embedding[:, 0].min() + embedding[:, 0].max()) / 2
    y_mid = (embedding[:, 1].min() + embedding[:, 1].max()) / 2
    half  = max(embedding[:, 0].max() - embedding[:, 0].min(),
                embedding[:, 1].max() - embedding[:, 1].min()) / 2 * 1.03
    x_lo, x_hi = x_mid - half, x_mid + half
    y_lo, y_hi = y_mid - half, y_mid + half

    def _sub(emb):
        if len(emb) > max_points_per_group:
            idx = rng.choice(len(emb), max_points_per_group, replace=False)
            return emb[idx]
        return emb

    def _style(ax, title):
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, loc="upper right", markerscale=1.5)

    # Order of panels: col 0 = all, then one per class in PHZ_MAP order
    class_order = list(PHZ_MAP.items())  # (db_val, (key, label, color))

    # Weight per point: true_total / used_in_umap (restores pre-cap density)
    weights: dict[int, float] = {}
    if class_counts:
        for key, (total, used) in class_counts.items():
            weights[key] = total / used if used > 0 else 1.0
    else:
        for _, (key, _, _) in class_order:
            weights[key] = 1.0

    from matplotlib.colors import LogNorm
    n_bins   = 200
    x_edges  = np.linspace(x_lo, x_hi, n_bins + 1)
    y_edges  = np.linspace(y_lo, y_hi, n_bins + 1)
    cmap_dens = plt.cm.Blues.copy()
    cmap_dens.set_bad(alpha=0)

    fig, axes = plt.subplots(2, 7, figsize=(42, 12), constrained_layout=True)

    # ── Row 1 col 0: all classes ──────────────────────────────────────────────
    ax = axes[0, 0]
    for db_val, (key, label, color) in class_order:
        mask = y == key
        if not mask.any():
            continue
        e = _sub(embedding[mask])
        s     = 12 if mask.sum() < 1000 else 5
        alpha = 0.8 if mask.sum() < 1000 else 0.3
        ax.scatter(e[:, 0], e[:, 1], c=color, s=s, alpha=alpha,
                   linewidths=0, label=f"{label} (n={mask.sum():,})")
    _style(ax, "All PHZ classes")

    # ── Row 1 cols 1–6: one class per panel ───────────────────────────────────
    bg = _sub(embedding)
    for ci, (db_val, (key, label, color)) in enumerate(class_order, start=1):
        ax = axes[0, ci]
        mask = y == key
        ax.scatter(bg[:, 0], bg[:, 1], c="lightgrey", s=3, alpha=0.15,
                   linewidths=0, zorder=1)
        if mask.any():
            e = _sub(embedding[mask])
            s     = 12 if mask.sum() < 1000 else 6
            alpha = 0.8 if mask.sum() < 1000 else 0.5
            ax.scatter(e[:, 0], e[:, 1], c=color, s=s, alpha=alpha,
                       linewidths=0, zorder=3,
                       label=f"{label} (n={mask.sum():,})")
        _style(ax, label)

    # ── Row 2 col 0: combined weighted density ────────────────────────────────
    ax = axes[1, 0]
    H_all = np.zeros((n_bins, n_bins))
    for db_val, (key, label, color) in class_order:
        mask = y == key
        if not mask.any():
            continue
        w = weights.get(key, 1.0)
        H, _, _ = np.histogram2d(embedding[mask, 0], embedding[mask, 1],
                                  bins=[x_edges, y_edges])
        H_all += H * w
    H_all = np.where(H_all > 0, H_all, np.nan)
    sc = ax.pcolormesh(x_edges, y_edges, H_all.T,
                       cmap=cmap_dens, norm=LogNorm(), rasterized=True)
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("UMAP 1", fontsize=9); ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_title("All — source density\n(weighted to pre-cap counts)", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax, label="Count (log)", pad=0.02, fraction=0.046)

    # ── Row 2 cols 1–6: per-class weighted density ────────────────────────────
    for ci, (db_val, (key, label, color)) in enumerate(class_order, start=1):
        ax = axes[1, ci]
        mask = y == key
        if mask.any():
            w = weights.get(key, 1.0)
            total = class_counts[key][0] if class_counts and key in class_counts else mask.sum()
            H, _, _ = np.histogram2d(embedding[mask, 0], embedding[mask, 1],
                                      bins=[x_edges, y_edges])
            H_w = np.where(H > 0, H * w, np.nan)
            sc = ax.pcolormesh(x_edges, y_edges, H_w.T,
                               cmap=cmap_dens, norm=LogNorm(), rasterized=True)
            fig.colorbar(sc, ax=ax, label="Count (log)", pad=0.02, fraction=0.046)
            ax.set_title(f"{label}\n(true N={total:,})", fontsize=10, fontweight="bold")
        else:
            ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("UMAP 1", fontsize=9); ax.set_ylabel("UMAP 2", fontsize=9)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.25)

    fig.suptitle(
        f"Q1 UMAP — all PHZ classes  (G ≥ {MAG_CUT},  {len(SELECTED_GAIA_PARAMETERS)} features)\n"
        f"Row 1: scatter (capped to equalise density) | Row 2: 2D density weighted to true counts",
        fontsize=12, fontweight="bold",
    )
    out = output_dir / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {out}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--max-per-class", type=int, default=10_000,
                        help="Max sources per PHZ class for UMAP fitting (default 10000)")
    args = parser.parse_args()

    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set")
        print("  Usage: PGUSER=siwei_zhao_2025 python src/analyze_umap_q1_allphz.py")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Q1 all-PHZ UMAP")
    print("=" * 60)

    print(f"\n[1] Querying all PHZ classes (max {args.max_per_class:,} per class)...")
    X, y, class_counts = query_all_phz(username, max_per_class=args.max_per_class,
                                        test_mode=args.test_mode)
    print(f"  Total for UMAP: {len(X):,} sources")

    print("\n[2] Fitting UMAP...")
    embedding, mean_, std_ = fit_umap(X)

    print("\n[3] Plotting...")
    plot_allphz_umap(embedding, y, OUTPUT_DIR, class_counts=class_counts)

    print(f"\nAll outputs saved under: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
