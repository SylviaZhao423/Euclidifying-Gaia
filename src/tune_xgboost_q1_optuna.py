#!/usr/bin/env python3
"""
Bayesian hyperparameter tuning for XGBoost Q1 classifier using Optuna.

Runs separate Optuna studies optimising for:
  - ROC AUC
  - Galaxy F1
  - Galaxy Precision
  - Galaxy Recall

Each study searches over 9 parameters using the TPE sampler.

Outputs saved to output/xgboost_q1_tuning/:
  optuna_results_{metric}.csv       — per-metric trial results
  optuna_best_params.json           — best params for all metrics
  optuna_summary.png                — comparison heatmap
  optuna_history.png                — optimisation history (all metrics)
  optuna_importance.png             — hyperparameter importance

Usage:
    PGUSER=siwei_zhao_2025 python src/tune_xgboost_q1_optuna.py
    python src/tune_xgboost_q1_optuna.py --from-cache
    python src/tune_xgboost_q1_optuna.py --plot-only   # replot from cached results
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
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, make_scorer)

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR  = PROJECT_DIR / "output" / "xgboost_q1_tuning_optuna"
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
CLASS_MAPPING = {"STAR": 0, "GALAXY": 1}
MAG_CUT       = 17
MAG_CUT_FAINT = 99

N_TRIALS = 100

# Metrics to optimise
OPTUNA_TARGETS = {
    "roc_auc":          ("ROC AUC",          "roc_auc"),
    "galaxy_f1":        ("Galaxy F1",        make_scorer(f1_score, pos_label=1, zero_division=0)),
    "galaxy_precision": ("Galaxy Precision", make_scorer(precision_score, pos_label=1, zero_division=0)),
    "galaxy_recall":    ("Galaxy Recall",    make_scorer(recall_score, pos_label=1, zero_division=0)),
}

# All metrics to evaluate on the best config
EVAL_METRICS = [
    ("ROC AUC",          "roc_auc"),
    ("Galaxy Precision", "galaxy_precision"),
    ("Galaxy Recall",    "galaxy_recall"),
    ("Galaxy F1",        "galaxy_f1"),
    ("Star Precision",   "star_precision"),
    ("Star Recall",      "star_recall"),
    ("Star F1",          "star_f1"),
]


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


# ── Evaluate best config with all metrics ───────────────────────

def evaluate_all_metrics(params, X, y, cv):
    """Run cross-val predict and compute all metrics."""
    model = xgb.XGBClassifier(**params, objective="binary:logistic",
                               tree_method="hist", eval_metric="logloss",
                               verbosity=0, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    y_proba = cross_val_predict(model, X, y, cv=cv, n_jobs=-1,
                                 method="predict_proba")[:, 1]
    return {
        "roc_auc":           roc_auc_score(y, y_proba),
        "galaxy_precision":  precision_score(y, y_pred, pos_label=1, zero_division=0),
        "galaxy_recall":     recall_score(y, y_pred, pos_label=1, zero_division=0),
        "galaxy_f1":         f1_score(y, y_pred, pos_label=1, zero_division=0),
        "star_precision":    precision_score(y, y_pred, pos_label=0, zero_division=0),
        "star_recall":       recall_score(y, y_pred, pos_label=0, zero_division=0),
        "star_f1":           f1_score(y, y_pred, pos_label=0, zero_division=0),
    }


# ── Plotting ────────────────────────────────────────────────────

def plot_history(all_studies, output_dir):
    """Optimisation history for all metrics on one plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (target_key, (label, _)) in zip(axes, OPTUNA_TARGETS.items()):
        csv_path = output_dir / f"optuna_results_{target_key}.csv"
        if not csv_path.exists():
            continue
        trials = pd.read_csv(csv_path).sort_values("number")
        ax.scatter(trials["number"], trials["value"], s=12, alpha=0.5,
                   color="steelblue", label="Trial score")
        running_best = trials["value"].cummax()
        ax.plot(trials["number"], running_best, color="red", lw=2,
                label=f"Best so far ({running_best.iloc[-1]:.5f})")
        ax.set_xlabel("Trial", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"Optimising: {label}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Optuna optimisation history — per metric",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = output_dir / "optuna_history.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


def plot_summary_heatmap(results, output_dir):
    """Heatmap comparing Optuna best per metric + original."""
    metric_labels = [m[0].replace("_", "\n") for m in EVAL_METRICS]
    metric_keys = [m[1] for m in EVAL_METRICS]

    config_labels = []
    values_list = []

    for name, data in results.items():
        config_labels.append(name)
        values_list.append([data["all_metrics"].get(k, 0) for k in metric_keys])

    values = np.array(values_list)
    n_configs, n_metrics = values.shape

    col_min = values.min(axis=0)
    col_max = values.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    normed = (values - col_min) / col_range

    fig, ax = plt.subplots(figsize=(12, 0.8 * n_configs + 2))
    im = ax.imshow(normed, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    for j in range(n_metrics):
        best_i = np.argmax(values[:, j])
        for i in range(n_configs):
            val = values[i, j]
            is_best = (i == best_i)
            color = "black" if normed[i, j] > 0.4 else "white"
            text = f"{val:.4f}"
            if is_best:
                text += " \u2605"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    fontweight="bold" if is_best else "normal", color=color)

    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels([m[0].replace("_", "\n") for m in EVAL_METRICS],
                       fontsize=10, fontweight="bold", rotation=30, ha="right")
    ax.set_yticks(range(n_configs))
    ax.set_yticklabels(config_labels, fontsize=9)

    for i in range(n_configs + 1):
        ax.axhline(i - 0.5, color="white", lw=1.5)
    for j in range(n_metrics + 1):
        ax.axvline(j - 0.5, color="white", lw=1.5)

    ax.set_title("Optuna tuning — best config per optimisation target\n"
                 "(\u2605 = best in column, colour normalised per column)",
                 fontsize=12, fontweight="bold", pad=15)
    fig.colorbar(im, ax=ax, label="Relative score (per column)",
                 shrink=0.8, pad=0.02)

    out = output_dir / "optuna_summary.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


# ── Main ────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--plot-only", action="store_true",
                        help="Replot from cached JSON results")
    args = parser.parse_args()

    try:
        import optuna
    except ImportError:
        print("ERROR: pip install optuna"); return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Plot-only mode ──────────────────────────────────────────
    if args.plot_only:
        results_path = OUTPUT_DIR / "optuna_best_params.json"
        if not results_path.exists():
            print("ERROR: no cached results. Run without --plot-only first.")
            return 1
        with open(results_path) as f:
            results = json.load(f)
        print("[1] Replotting from cached results...")
        plot_history(None, OUTPUT_DIR)
        plot_summary_heatmap(results, OUTPUT_DIR)
        return 0

    # ── Load data ───────────────────────────────────────────────
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

    # ── Run Optuna for each metric ──────────────────────────────
    results = {}

    # Original baseline
    print(f"\n[2] Evaluating original config (d=8, n=200, lr=0.1)...")
    orig_metrics = evaluate_all_metrics(
        {"max_depth": 8, "n_estimators": 200, "learning_rate": 0.1,
         "subsample": 0.8, "colsample_bytree": 0.8}, X, y, cv)
    results["Original\n(d=8, n=200, lr=0.1)"] = {
        "params": {"max_depth": 8, "n_estimators": 200, "learning_rate": 0.1},
        "all_metrics": orig_metrics,
    }

    for target_key, (label, scorer) in OPTUNA_TARGETS.items():
        print(f"\n[3] Optuna: optimising {label} ({args.n_trials} trials)...")

        def make_objective(scoring):
            def objective(trial):
                params = {
                    "objective": "binary:logistic", "tree_method": "hist",
                    "eval_metric": "logloss", "verbosity": 0,
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
                model = xgb.XGBClassifier(**params, random_state=42)
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring,
                                         n_jobs=-1)
                return scores.mean()
            return objective

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(make_objective(scorer), n_trials=args.n_trials,
                       show_progress_bar=True)

        print(f"  Best {label}: {study.best_value:.5f}")
        bp = study.best_params
        print(f"  Params: d={bp['max_depth']}, n={bp['n_estimators']}, "
              f"lr={bp['learning_rate']:.4f}")

        # Save trial CSV
        trials_df = study.trials_dataframe()
        trials_df.to_csv(OUTPUT_DIR / f"optuna_results_{target_key}.csv",
                         index=False)

        # Evaluate best with all metrics
        print(f"  Evaluating with all metrics...")
        all_m = evaluate_all_metrics(bp, X, y, cv)

        config_label = (f"Best {label}\n"
                        f"(d={bp['max_depth']}, n={bp['n_estimators']}, "
                        f"lr={bp['learning_rate']:.4f})")
        results[config_label] = {
            "params": {k: (float(v) if isinstance(v, (int, float)) else v)
                       for k, v in bp.items()},
            "target_metric": target_key,
            "target_score": float(study.best_value),
            "all_metrics": {k: float(v) for k, v in all_m.items()},
        }

    # ── Print comparison ────────────────────────────────────────
    print(f"\n[4] Comparison across all Optuna runs:")
    header_metrics = [m[1] for m in EVAL_METRICS]
    header_labels = ["AUC", "Gal_P", "Gal_R", "Gal_F1", "Sta_P", "Sta_R", "Sta_F1"]
    print(f"  {'Config':<35} " + " ".join(f"{h:>7}" for h in header_labels))
    print("  " + "-" * 90)
    for name, data in results.items():
        short = name.split("\n")[0]
        vals = [data["all_metrics"].get(k, 0) for k in header_metrics]
        print(f"  {short:<35} " + " ".join(f"{v:>7.4f}" for v in vals))

    # ── Save results ────────────────────────────────────────────
    out_json = OUTPUT_DIR / "optuna_best_params.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  -> {out_json}")

    # ── Plots ───────────────────────────────────────────────────
    print(f"\n[5] Plotting...")
    plot_history(None, OUTPUT_DIR)
    plot_summary_heatmap(results, OUTPUT_DIR)


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
