#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for XGBoost Q1 classifier — Galaxy F1,
with early_stopping_rounds as a tuned parameter.

Uses a train/val split within each CV fold to enable early stopping.

Outputs saved to output/xgboost_q1_tuning_optuna_es/:
  optuna_results.csv            — all trial results
  optuna_best_params.json       — best params + all metrics
  optuna_history.png            — optimisation history
  optuna_importance.png         — hyperparameter importance

Usage:
    PGUSER=siwei_zhao_2025 python src/tune_xgboost_q1_optuna_es.py
    python src/tune_xgboost_q1_optuna_es.py --from-cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score)

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR  = PROJECT_DIR / "output" / "xgboost_q1_tuning_optuna_es"
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
CLASS_MAPPING = {"STAR": 0, "GALAXY": 1}
MAG_CUT       = 17
MAG_CUT_FAINT = 99

N_TRIALS = 100


# ── Data loading ────────────────────────────────────────────────

def load_from_db(username: str):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    feat_names = list(SELECTED_GAIA_PARAMETERS)
    gaia_cols = ", ".join([f"g_{col} AS {col}" for col in feat_names])
    query = f"""
        SELECT {gaia_cols},
            CASE WHEN p_phz_classification = 1 THEN 'STAR'
                 WHEN p_phz_classification = 2 THEN 'GALAXY' END AS label
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep
        WHERE p_phz_classification IN (1, 2)
          AND g_phot_g_mean_mag >= {MAG_CUT}
          AND g_phot_g_mean_mag <= {MAG_CUT_FAINT}
        ORDER BY g_random_index
    """
    print("  Querying WSDB...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    for col in feat_names:
        df[col] = pd.to_numeric(df[col].replace("NOT_AVAILABLE", np.nan),
                                errors="coerce")
    return df[feat_names].values.astype(float), df["label"].map(CLASS_MAPPING).values


def load_from_cache():
    npz_path = ARRAYS_DIR / "q1_indiv.npz"
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found."); sys.exit(1)
    q1 = np.load(npz_path, allow_pickle=True)
    return q1["X_raw"], q1["y"]


# ── CV with early stopping ─────────────────────────────────────

def cv_with_early_stopping(params, early_stopping_rounds, X, y, cv):
    """Manual 5-fold CV with early stopping using a val split within each fold."""
    from sklearn.model_selection import train_test_split

    f1_scores = []
    for train_idx, test_idx in cv.split(X, y):
        X_fold_train_full, y_fold_train_full = X[train_idx], y[train_idx]
        X_fold_test, y_fold_test = X[test_idx], y[test_idx]

        # Split train fold into train + val for early stopping (85/15)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_fold_train_full, y_fold_train_full,
            test_size=0.15, random_state=42, stratify=y_fold_train_full)

        model = xgb.XGBClassifier(**params, random_state=42)
        model.set_params(early_stopping_rounds=early_stopping_rounds)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict(X_fold_test)
        f1 = f1_score(y_fold_test, y_pred, pos_label=1, zero_division=0)
        f1_scores.append(f1)

    return np.mean(f1_scores)


# ── Evaluate all metrics ───────────────────────────────────────

def evaluate_all_metrics(params, early_stopping_rounds, X, y, cv):
    """Full evaluation with all per-class metrics."""
    from sklearn.model_selection import train_test_split

    all_y_true, all_y_pred, all_y_proba = [], [], []
    best_iterations = []

    for train_idx, test_idx in cv.split(X, y):
        X_fold_train_full, y_fold_train_full = X[train_idx], y[train_idx]
        X_fold_test, y_fold_test = X[test_idx], y[test_idx]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_fold_train_full, y_fold_train_full,
            test_size=0.15, random_state=42, stratify=y_fold_train_full)

        model = xgb.XGBClassifier(**params, random_state=42)
        model.set_params(early_stopping_rounds=early_stopping_rounds)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        best_iterations.append(model.best_iteration)
        all_y_true.extend(y_fold_test)
        all_y_pred.extend(model.predict(X_fold_test))
        all_y_proba.extend(model.predict_proba(X_fold_test)[:, 1])

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    y_proba = np.array(all_y_proba)

    return {
        "roc_auc":           roc_auc_score(y_true, y_proba),
        "galaxy_precision":  precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "galaxy_recall":     recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "galaxy_f1":         f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "star_precision":    precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "star_recall":       recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "star_f1":           f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "mean_best_iteration": float(np.mean(best_iterations)),
    }


# ── Main ────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    args = parser.parse_args()

    try:
        import optuna
    except ImportError:
        print("ERROR: pip install optuna"); return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ────────────────────────────────────────────
    print("[1] Loading data...")
    if args.from_cache:
        X, y = load_from_cache()
    else:
        username = os.environ.get("PGUSER")
        if not username:
            print("  No PGUSER, falling back to --from-cache")
            X, y = load_from_cache()
        else:
            X, y = load_from_db(username)

    print(f"  {len(X):,} sources  STAR: {(y==0).sum():,}  GALAXY: {(y==1).sum():,}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── 2. Define objective ─────────────────────────────────────
    def objective(trial):
        params = {
            "objective":        "binary:logistic",
            "tree_method":      "hist",
            "eval_metric":      "logloss",
            "verbosity":        0,
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "n_estimators":     trial.suggest_int("n_estimators", 50, 800),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        es_rounds = trial.suggest_int("early_stopping_rounds", 5, 50)

        return cv_with_early_stopping(params, es_rounds, X, y, cv)

    # ── 3. Run Optuna ───────────────────────────────────────────
    print(f"\n[2] Running Optuna: {args.n_trials} trials, 5-fold CV, "
          f"optimising Galaxy F1 (with early stopping as hyperparameter)...")
    print(f"  Searching: max_depth, n_estimators, learning_rate, subsample, "
          f"colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda, "
          f"early_stopping_rounds")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # ── 4. Results ──────────────────────────────────────────────
    bp = study.best_params
    es_rounds = bp.pop("early_stopping_rounds")

    print(f"\n[3] Results:")
    print(f"  Best Galaxy F1 (CV): {study.best_value:.5f}")
    print(f"  Best params:")
    for k, v in bp.items():
        if isinstance(v, float):
            print(f"    {k:<20} {v:.6f}")
        else:
            print(f"    {k:<20} {v}")
    print(f"    {'early_stopping':<20} {es_rounds}")

    # ── 5. Evaluate best with all metrics ───────────────────────
    print(f"\n[4] Evaluating best config with all metrics...")
    xgb_params = {
        "objective": "binary:logistic", "tree_method": "hist",
        "eval_metric": "logloss", "verbosity": 0,
        **{k: v for k, v in bp.items()},
    }
    all_m = evaluate_all_metrics(xgb_params, es_rounds, X, y, cv)

    print(f"\n  {'Metric':<22} {'Optuna+ES':>11} {'Original':>11}")
    print("  " + "-" * 48)
    orig_metrics = {
        "roc_auc": 0.85548, "galaxy_precision": 0.86909,
        "galaxy_recall": 0.53050, "galaxy_f1": 0.65883,
        "star_precision": 0.85458, "star_recall": 0.97185,
        "star_f1": 0.90945,
    }
    for name, val in all_m.items():
        if name == "mean_best_iteration":
            print(f"  {'Avg best iteration':<22} {val:>11.1f}")
            continue
        orig_val = orig_metrics.get(name, float("nan"))
        delta = val - orig_val
        label = name.replace("_", " ").title()
        print(f"  {label:<22} {val:>11.5f} {orig_val:>11.5f}  "
              f"({'+'if delta>=0 else ''}{delta:.5f})")

    # Compare with previous Optuna (no ES)
    print(f"\n  Galaxy F1 comparison:")
    print(f"    Optuna + ES:      {study.best_value:.5f}  (es={es_rounds})")
    print(f"    Optuna (no ES):   0.66314")
    print(f"    Grid search best: 0.66069")
    print(f"    Original:         0.65883")

    # ── 6. Save ─────────────────────────────────────────────────
    trials_df = study.trials_dataframe()
    trials_df.to_csv(OUTPUT_DIR / "optuna_results.csv", index=False)

    result = {
        "params": {k: (float(v) if isinstance(v, float) else int(v) if isinstance(v, int) else v)
                   for k, v in bp.items()},
        "early_stopping_rounds": es_rounds,
        "best_galaxy_f1_cv": float(study.best_value),
        "all_metrics": {k: float(v) for k, v in all_m.items()},
        "n_trials": args.n_trials,
        "n_sources": len(X),
    }
    with open(OUTPUT_DIR / "optuna_best_params.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  -> {OUTPUT_DIR / 'optuna_results.csv'}")
    print(f"  -> {OUTPUT_DIR / 'optuna_best_params.json'}")

    # ── 7. Plots ────────────────────────────────────────────────
    print(f"\n[5] Plotting...")

    # History
    fig, ax = plt.subplots(figsize=(10, 5))
    trials = trials_df.sort_values("number")
    ax.scatter(trials["number"], trials["value"], s=15, alpha=0.6,
               color="steelblue", label="Trial Galaxy F1")
    running_best = trials["value"].cummax()
    ax.plot(trials["number"], running_best, color="red", lw=2,
            label=f"Best so far ({running_best.iloc[-1]:.5f})")
    ax.axhline(0.66314, color="green", ls="--", lw=1,
               label="Optuna no-ES best (0.6631)")
    ax.axhline(0.65883, color="orange", ls="--", lw=1,
               label="Original (0.6588)")
    ax.set_xlabel("Trial number", fontsize=11, fontweight="bold")
    ax.set_ylabel("Galaxy F1 (5-fold CV)", fontsize=11, fontweight="bold")
    ax.set_title(f"Optuna + early stopping — Galaxy F1 ({args.n_trials} trials)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    out = OUTPUT_DIR / "optuna_history.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")

    # Importance
    try:
        importances = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(8, 5))
        names = list(importances.keys())
        vals = list(importances.values())
        ax.barh(range(len(names)), vals, color="steelblue", edgecolor="white")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Importance", fontsize=11, fontweight="bold")
        ax.set_title("Hyperparameter importance (Galaxy F1 + early stopping)",
                     fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3, axis="x")
        out = OUTPUT_DIR / "optuna_importance.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  -> {out}")
    except Exception as e:
        print(f"  (importance plot skipped: {e})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
