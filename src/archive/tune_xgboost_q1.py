#!/usr/bin/env python3
"""
Hyperparameter tuning for XGBoost Q1 star/galaxy classifier.

Uses scikit-learn GridSearchCV with 5-fold stratified CV to search over
max_depth, n_estimators, and learning_rate. Produces a heatmap of
ROC AUC across the grid and prints the best parameters.

Data: loads Q1 training data from WSDB (same query as train_xgboost_q1.py).

Outputs saved to output/xgboost_q1/:
  hyperparam_grid_results.csv   — full grid results
  hyperparam_heatmap.png        — ROC AUC heatmaps

Usage:
    PGUSER=siwei_zhao_2025 python src/tune_xgboost_q1.py
    python src/tune_xgboost_q1.py --from-cache   # use cached q1_indiv.npz
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR  = PROJECT_DIR / "output" / "xgboost_q1"
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"

# Same constants as train_xgboost_q1.py
WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
CLASS_MAPPING = {"STAR": 0, "GALAXY": 1}
MAG_CUT       = 17
MAG_CUT_FAINT = 99

# ── Parameter grid ──────────────────────────────────────────────
PARAM_GRID = {
    "max_depth":      [4, 6, 8, 10],
    "n_estimators":   [100, 200, 500],
    "learning_rate":  [0.01, 0.05, 0.1, 0.2],
}

# Fixed params (not tuned)
FIXED_PARAMS = {
    "objective":        "binary:logistic",
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "tree_method":      "hist",
    "eval_metric":      "logloss",
    "random_state":     42,
}


# ── Data loading ────────────────────────────────────────────────

def load_from_db(username: str):
    """Query Q1 data from WSDB (same as train_xgboost_q1.py)."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    feat_names = list(SELECTED_GAIA_PARAMETERS)
    gaia_cols = ", ".join([f"g_{col} AS {col}" for col in feat_names])
    query = f"""
        SELECT
            {gaia_cols},
            CASE
                WHEN p_phz_classification = 1 THEN 'STAR'
                WHEN p_phz_classification = 2 THEN 'GALAXY'
            END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep
        WHERE p_phz_classification IN (1, 2)
          AND g_phot_g_mean_mag >= {MAG_CUT}
          AND g_phot_g_mean_mag <= {MAG_CUT_FAINT}
        ORDER BY g_random_index
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    for col in feat_names:
        df[col] = pd.to_numeric(df[col].replace("NOT_AVAILABLE", np.nan),
                                errors="coerce")
    X = df[feat_names].values.astype(float)
    y = df["label"].map(CLASS_MAPPING).values
    return X, y, feat_names


def load_from_cache():
    """Load from q1_indiv.npz (saved during UMAP pipeline)."""
    npz_path = ARRAYS_DIR / "q1_indiv.npz"
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found.")
        print("Run with PGUSER=... to query from database instead.")
        sys.exit(1)

    q1 = np.load(npz_path, allow_pickle=True)
    X = q1["X_raw"]
    y = q1["y"]
    n_feat = X.shape[1]
    feat_names = list(SELECTED_GAIA_PARAMETERS)

    if n_feat != len(feat_names):
        print(f"  Note: npz has {n_feat} features, current selected list has "
              f"{len(feat_names)}.")
        print(f"  Using npz features (names not stored, using generic labels).")
        feat_names = [f"f{i}" for i in range(n_feat)]

    return X, y, feat_names


# ── Plotting ────────────────────────────────────────────────────

def plot_heatmaps(results_df: pd.DataFrame, output_dir: Path):
    """2×2 heatmaps: ROC AUC for each learning_rate, as depth vs n_estimators."""
    lrs = sorted(results_df["param_learning_rate"].unique())
    n_lr = len(lrs)
    fig, axes = plt.subplots(1, n_lr, figsize=(5 * n_lr, 4.5))
    if n_lr == 1:
        axes = [axes]

    depths = sorted(results_df["param_max_depth"].unique())
    n_ests = sorted(results_df["param_n_estimators"].unique())

    vmin = results_df["mean_test_score"].min()
    vmax = results_df["mean_test_score"].max()

    for ax, lr in zip(axes, lrs):
        sub = results_df[results_df["param_learning_rate"] == lr]
        grid = np.zeros((len(depths), len(n_ests)))
        for i, d in enumerate(depths):
            for j, n in enumerate(n_ests):
                row = sub[(sub["param_max_depth"] == d) &
                          (sub["param_n_estimators"] == n)]
                if len(row) > 0:
                    grid[i, j] = row["mean_test_score"].values[0]

        im = ax.imshow(grid, cmap="RdYlGn", aspect="auto",
                       vmin=vmin, vmax=vmax, origin="lower")

        # Annotate cells
        for i in range(len(depths)):
            for j in range(len(n_ests)):
                val = grid[i, j]
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if val < (vmin + vmax) / 2 else "black")

        ax.set_xticks(range(len(n_ests)))
        ax.set_xticklabels(n_ests)
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels(depths)
        ax.set_xlabel("n_estimators", fontsize=11)
        ax.set_ylabel("max_depth", fontsize=11)
        ax.set_title(f"learning_rate = {lr}", fontsize=11, fontweight="bold")

    fig.suptitle("XGBoost Hyperparameter Grid Search — ROC AUC (5-fold CV)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.colorbar(im, ax=axes, label="Mean ROC AUC", shrink=0.8)

    out = output_dir / "hyperparam_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")


def plot_depth_sensitivity(results_df: pd.DataFrame, output_dir: Path):
    """Line plot: ROC AUC vs max_depth for each (lr, n_estimators) combo."""
    fig, ax = plt.subplots(figsize=(8, 5))

    lrs = sorted(results_df["param_learning_rate"].unique())
    n_ests = sorted(results_df["param_n_estimators"].unique())
    depths = sorted(results_df["param_max_depth"].unique())

    colors = plt.cm.Set2(np.linspace(0, 1, len(lrs) * len(n_ests)))
    ci = 0
    for lr in lrs:
        for ne in n_ests:
            sub = results_df[(results_df["param_learning_rate"] == lr) &
                             (results_df["param_n_estimators"] == ne)]
            sub = sub.sort_values("param_max_depth")
            ax.errorbar(sub["param_max_depth"], sub["mean_test_score"],
                        yerr=sub["std_test_score"],
                        marker="o", markersize=4, lw=1.2, capsize=3,
                        color=colors[ci],
                        label=f"lr={lr}, n={ne}")
            ci += 1

    ax.set_xlabel("max_depth", fontsize=11, fontweight="bold")
    ax.set_ylabel("ROC AUC (5-fold CV)", fontsize=11, fontweight="bold")
    ax.set_title("Sensitivity to max_depth", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, ncol=3, loc="lower right")
    ax.grid(alpha=0.3)

    out = output_dir / "hyperparam_sensitivity.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")


# ── Main ────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for XGBoost Q1 classifier"
    )
    parser.add_argument("--from-cache", action="store_true",
                        help="Load data from q1_indiv.npz instead of WSDB")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ────────────────────────────────────────────
    print("[1] Loading Q1 data...")
    if args.from_cache:
        X, y, feat_names = load_from_cache()
    else:
        import os
        username = os.environ.get("PGUSER")
        if not username:
            print("  No PGUSER set, falling back to --from-cache")
            X, y, feat_names = load_from_cache()
        else:
            X, y, feat_names = load_from_db(username)

    n_star   = (y == 0).sum()
    n_galaxy = (y == 1).sum()
    print(f"  {len(X):,} sources  STAR: {n_star:,} ({100*n_star/len(X):.1f}%)  "
          f"GALAXY: {n_galaxy:,} ({100*n_galaxy/len(X):.1f}%)")
    print(f"  Features: {len(feat_names)}")

    # ── 2. Grid search ──────────────────────────────────────────
    n_combos = 1
    for v in PARAM_GRID.values():
        n_combos *= len(v)
    print(f"\n[2] Running GridSearchCV: {n_combos} combos × 5 folds = "
          f"{n_combos * 5} fits...")

    estimator = xgb.XGBClassifier(**FIXED_PARAMS, verbosity=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator, PARAM_GRID,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    grid.fit(X, y)

    # ── 3. Results ──────────────────────────────────────────────
    print(f"\n[3] Results:")
    print(f"  Best ROC AUC: {grid.best_score_:.5f}")
    print(f"  Best params:  {grid.best_params_}")

    results_df = pd.DataFrame(grid.cv_results_)
    # Keep useful columns
    keep_cols = [c for c in results_df.columns
                 if c.startswith("param_") or c in
                 ("mean_test_score", "std_test_score", "rank_test_score",
                  "mean_train_score", "std_train_score", "mean_fit_time")]
    results_df = results_df[keep_cols].sort_values("rank_test_score")

    csv_path = OUTPUT_DIR / "hyperparam_grid_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  -> Saved: {csv_path}")

    # Print top 10
    print(f"\n  Top 10 configurations:")
    print(f"  {'Rank':>4}  {'max_depth':>9}  {'n_est':>5}  {'lr':>5}  "
          f"{'AUC_mean':>9}  {'AUC_std':>8}  {'train_AUC':>9}")
    print("  " + "-" * 65)
    for _, row in results_df.head(10).iterrows():
        print(f"  {int(row['rank_test_score']):>4}  "
              f"{int(row['param_max_depth']):>9}  "
              f"{int(row['param_n_estimators']):>5}  "
              f"{row['param_learning_rate']:>5.2f}  "
              f"{row['mean_test_score']:>9.5f}  "
              f"{row['std_test_score']:>8.5f}  "
              f"{row['mean_train_score']:>9.5f}")

    # ── 4. Compare with current params ──────────────────────────
    current = {"max_depth": 8, "n_estimators": 200, "learning_rate": 0.1}
    current_row = results_df[
        (results_df["param_max_depth"] == current["max_depth"]) &
        (results_df["param_n_estimators"] == current["n_estimators"]) &
        (results_df["param_learning_rate"] == current["learning_rate"])
    ]
    if len(current_row) > 0:
        cr = current_row.iloc[0]
        print(f"\n  Current params (depth={current['max_depth']}, "
              f"n_est={current['n_estimators']}, lr={current['learning_rate']}):")
        print(f"    ROC AUC = {cr['mean_test_score']:.5f} ± {cr['std_test_score']:.5f}"
              f"  (rank {int(cr['rank_test_score'])}/{len(results_df)})")
        delta = grid.best_score_ - cr["mean_test_score"]
        print(f"    Best vs current: {'+' if delta >= 0 else ''}{delta:.5f}")

    # ── 5. Plots ────────────────────────────────────────────────
    print(f"\n[4] Plotting...")
    plot_heatmaps(results_df, OUTPUT_DIR)
    plot_depth_sensitivity(results_df, OUTPUT_DIR)

    # ── 6. Save best params ─────────────────────────────────────
    best = {**FIXED_PARAMS, **grid.best_params_}
    best_path = OUTPUT_DIR / "best_hyperparams.json"
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"  -> Saved: {best_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
