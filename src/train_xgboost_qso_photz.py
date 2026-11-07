#!/usr/bin/env python3
"""
STEP 2 of the QSO-companion colour pipeline: an XGBoost photometric-redshift regressor
that predicts a QSO's redshift from Euclid aperture photometry.

Step 1 (analyze_qso_color_vs_specz.py) showed the Euclid colour(z) locus is narrow, so
colour -> z is learnable. We train on DESI DR1 spectroscopic QSOs matched to Euclid Q1,
the 12 within-aperture adjacent COLOURS (I-Y, Y-J, J-H at each of 1/2/3/4 FWHM) as input
features — deliberately colours, not magnitudes, so the model has no absolute-brightness
handle and cannot lean on the flux-limited brightness<->z (Malmquist) prior of the DESI
training set. That keeps the photo-z transferable to differently-selected samples (CatWISE).
See experiment_photz_features.py / _disentangle.py for the colours-vs-mags comparison.
It MUST be Euclid-only: the Step-3 target (WISE+Euclid QSO-pair candidates) has only
Euclid photometry available.

Honest evaluation
-----------------
The sample is small (~2,800), so metrics come from 5-fold OUT-OF-FOLD cross-validation,
averaged over 10 project seeds: every QSO gets a prediction from models that never saw it.
  Δz = (z_pred - z_spec) / (1 + z_spec)
  bias        = median(Δz)
  σ_NMAD      = 1.48 × median(|Δz - median Δz|)     (robust scatter)
  outlier frac= fraction with |Δz| > 0.15
Reliable to z~3 (dense training), extrapolation beyond (few anchors z>3.5).

Data source
-----------
DESI DR1 QSOs matched to Euclid Q1 via crossmatch_qso_desi_dr1_euclid (euclid_source='Q1').
Aperture fluxes pulled fresh from euclid_q1.mer_final_cat by e_id; converted to AB mag with
ZPC=1.05 for VIS, 1.0 for NISP (project convention, see crossmatch_q1_mer_phz.py).

Features (12): IE_YE_a YE_JE_a JE_HE_a for a in 1/2/3/4 FWHM  (adjacent within-aperture colours)

Reads (WSDB; Astronomy/Cambridge VPN; auth via ~/.pgpass):
  {PGUSER}.crossmatch_qso_desi_dr1_euclid
  euclid_q1.mer_final_cat
Writes (the z-phot scatter/residual figures are made separately by analyze_qso_photz_vs_specz.py):
  output/xgboost_qso_photz/models/xgb_photz_seed{seed}.json  (10 full-sample models for Step 3)
  output/xgboost_qso_photz/models/features.json
  output/xgboost_qso_photz/feature_importance.png (+ .csv)  (feature-importance stability)
  output/xgboost_qso_photz/oof.csv
  output/xgboost_qso_photz/metrics.json

Usage:
  PGUSER=siwei_zhao_2025 python src/train_xgboost_qso_photz.py          # all 16 features
  PGUSER=siwei_zhao_2025 python src/train_xgboost_qso_photz.py --log1pz  # predict log(1+z) internally
  python src/analyze_qso_photz_vs_specz.py                            # then make the plots
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR  = PROJECT_DIR / "output" / "xgboost_qso_photz"
MODELS_DIR  = OUTPUT_DIR / "models"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_qso_color_vs_specz import (  # noqa: E402
    WSDB_HOST, WSDB_DB, WSDB_USER, _flux_to_mag, ZPC_VIS,
)

SEEDS    = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]
APERS    = [1, 2, 3, 4]
BANDS    = [("IE", "vis"), ("YE", "y"), ("JE", "j"), ("HE", "h")]
MER_TABLE = "euclid_q1.mer_final_cat"

# 12 features: within-aperture adjacent colours (I-Y, Y-J, J-H) × aperture 1/2/3/4.
# Colours (not mags) -> no absolute-brightness / Malmquist handle -> transferable photo-z.
ADJ_COLOURS  = [("IE", "YE"), ("YE", "JE"), ("JE", "HE")]
ALL_FEATURES = [f"{x}_{y}_{a}" for a in APERS for x, y in ADJ_COLOURS]

# kept for backward compat (imported by train_qso_euclid_photz_morph.py etc.)
COLOR_FEATURES = ALL_FEATURES
MAG_ANCHOR     = None


def add_colours(df):
    """Add the 12 within-aperture adjacent colours to a frame that already has the
    {band}_{aper} magnitudes (IE_1..HE_4). Shared by training and the apply scripts so the
    exact same colour definition is used everywhere."""
    for a in APERS:
        for x, y in ADJ_COLOURS:
            df[f"{x}_{y}_{a}"] = df[f"{x}_{a}"] - df[f"{y}_{a}"]
    return df

N_FOLDS     = 5
OUTLIER_DZ  = 0.15

XGB_PARAMS = dict(
    objective="reg:squarederror",
    max_depth=6,
    learning_rate=0.05,
    n_estimators=600,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    tree_method="hist",
    eval_metric="rmse",
    early_stopping_rounds=30,
)


# ----------------------------------------------------------------------- data
def build_training_data(user: str) -> pd.DataFrame:
    """Pull DESI×Euclid Q1 sources, join to mer_final_cat for all 4×4 aperture mags."""
    import psycopg2
    conn = psycopg2.connect(
        host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30,
        keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5,
    )
    xmatch = f"{user}.crossmatch_qso_desi_dr1_euclid"
    print(f"  reading {xmatch} (Q1, z>0) ...")
    ids = pd.read_sql_query(
        f"""SELECT e_id::bigint AS object_id, qso_id, qso_z
            FROM {xmatch}
            WHERE euclid_source = 'Q1' AND qso_z IS NOT NULL AND qso_z > 0
              AND e_id IS NOT NULL""", conn,
    ).drop_duplicates("object_id")
    print(f"  {len(ids):,} unique Q1 QSO Euclid ids")

    # build SELECT: flux + fluxerr for every band × aperture
    fcols, ecols = [], []
    for _, s in BANDS:
        for a in APERS:
            fcols.append(f"flux_{s}_{a}fwhm_aper")
            ecols.append(f"fluxerr_{s}_{a}fwhm_aper")
    sel = ", ".join(["object_id"] + fcols + ecols)
    print(f"  querying {MER_TABLE} for all 4×4 aperture fluxes ...")
    mer = pd.read_sql_query(
        f"SELECT {sel} FROM {MER_TABLE} WHERE object_id = ANY(%(ids)s)",
        conn, params={"ids": ids["object_id"].tolist()},
    )
    conn.close()

    df = ids.merge(mer, on="object_id", how="inner")
    print(f"  matched {len(df):,} in mer_final_cat")

    # flux → AB mag for every band × aperture
    for band, s in BANDS:
        zpc = ZPC_VIS if s == "vis" else 1.0
        for a in APERS:
            df[f"{band}_{a}"] = _flux_to_mag(
                df[f"flux_{s}_{a}fwhm_aper"], df[f"fluxerr_{s}_{a}fwhm_aper"], zpc
            )

    df = add_colours(df)
    keep = ["qso_id", "qso_z"] + ALL_FEATURES
    return df[keep]


# --------------------------------------------------------------------- metrics
def photz_metrics(zs: np.ndarray, zp: np.ndarray) -> dict:
    dz   = (zp - zs) / (1.0 + zs)
    bias = float(np.median(dz))
    sig  = float(1.48 * np.median(np.abs(dz - bias)))
    out  = float(np.mean(np.abs(dz) > OUTLIER_DZ))
    rmse = float(np.sqrt(np.mean((zp - zs) ** 2)))
    return {"N": int(len(zs)), "bias": bias, "sigma_nmad": sig,
            "outlier_frac": out, "rmse": rmse}


# ----------------------------------------------------------------------- train
def _fit_one(X_tr, y_tr, seed):
    Xa, Xv, ya, yv = train_test_split(X_tr, y_tr, test_size=0.15, random_state=seed)
    m = xgb.XGBRegressor(random_state=seed, **XGB_PARAMS)
    m.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)
    return m


def oof_predict(X, y, log1pz: bool, zs: np.ndarray):
    """5-fold OOF predictions averaged over all seeds. Returns z_pred (linear z).

    Folds are stratified by redshift quantile (N_FOLDS equal-count z bins) so
    each fold's training and eval sets cover the full z distribution — important
    at z>2.5 where training sources are sparse.
    """
    z_strata = pd.qcut(zs, q=N_FOLDS, labels=False, duplicates="drop")
    n   = len(y)
    oof = np.zeros((len(SEEDS), n))
    for si, seed in enumerate(SEEDS):
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr, te in kf.split(X, z_strata):
            m    = _fit_one(X[tr], y[tr], seed)
            pred = m.predict(X[te])
            oof[si, te] = np.expm1(pred) if log1pz else pred
        print(f"  seed {seed}: OOF done")
    return oof.mean(axis=0)


def fit_final_models(X, y, log1pz: bool, models_dir: Path, features: list):
    models_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        m = _fit_one(X, y, seed)
        m.save_model(models_dir / f"xgb_photz_seed{seed}.json")
    (models_dir / "features.json").write_text(json.dumps(
        {"features": features, "target": "log1p_z" if log1pz else "z"}, indent=2))
    print(f"  -> {len(SEEDS)} models in {models_dir}")


def feature_importance_plot(models_dir: Path, features: list, out: Path, met=None, met3=None):
    """Ranked gain feature-importance STABILITY across the 10 seeds — same format as
    train_xgboost_q1.py's q1_feature_importance.png: bars coloured by CV = std/mean
    (green = stable, red = unstable), ±1σ error bars, per-bar value+CV labels, and an
    optional photo-z metrics box. Writes a companion CSV too."""
    imps = np.zeros((len(SEEDS), len(features)))
    for si, seed in enumerate(SEEDS):
        m = xgb.XGBRegressor(); m.load_model(models_dir / f"xgb_photz_seed{seed}.json")
        score = m.get_booster().get_score(importance_type="gain")
        for fi in range(len(features)):
            imps[si, fi] = score.get(f"f{fi}", 0.0)
    imps = imps / (imps.sum(axis=1, keepdims=True) + 1e-12)   # per-seed fraction of total gain
    mu, std = imps.mean(0), imps.std(0)
    cv = std / (mu + 1e-9)
    order = np.argsort(mu)[::-1]
    feats = [features[i] for i in order]
    mu_s, std_s, cv_s = mu[order], std[order], cv[order]
    n = len(feats)

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.38)))
    norm = plt.Normalize(vmin=0, vmax=0.5)
    colors = plt.cm.RdYlGn_r(norm(cv_s))
    bars = ax.barh(range(n), mu_s, xerr=std_s, color=colors, edgecolor="white", capsize=4,
                   error_kw={"elinewidth": 1.2, "ecolor": "black"})
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"#{i+1}  {f}" for i, f in enumerate(feats)], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance  (mean ± 1σ across seeds)", fontsize=13, fontweight="bold")
    xmax = max(0.5, float((mu_s + std_s).max()) * 1.2)
    ax.set_xlim(0, xmax); ax.set_xticks(np.arange(0, xmax + 1e-6, 0.1)); ax.grid(alpha=0.3, axis="x")
    for bar, mv, sv, cvv in zip(bars, mu_s, std_s, cv_s):
        ax.text(mv + sv + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{mv:.4f}  CV={cvv:.2f}", va="center", fontsize=8)
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=norm); sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("CV = std / mean", fontsize=9)
    cbar.ax.text(0.4, 1.04, "unstable", transform=cbar.ax.transAxes, fontsize=8, va="bottom", ha="left", color="darkred")
    cbar.ax.text(0.4, -0.05, "stable", transform=cbar.ax.transAxes, fontsize=8, va="top", ha="left", color="darkgreen")
    if met is not None:
        lines = ["photo-z (OOF) metrics:",
                 f"{'sigma_NMAD':<13}: {met['sigma_nmad']:.3f}",
                 f"{'outliers':<13}: {100*met['outlier_frac']:.1f}%",
                 f"{'bias':<13}: {met['bias']:+.3f}"]
        if met3 is not None:
            lines.append(f"{'sigma (z<=3)':<13}: {met3['sigma_nmad']:.3f}")
        ax.text(0.6, 0.05, "\n".join(lines), transform=ax.transAxes, fontsize=9.5, family="monospace",
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#bbbbbb", alpha=0.95))
    ax.tick_params(axis="x", labelsize=12)
    plt.suptitle(f"Euclid photo-z Feature Importance Stability  ({len(SEEDS)} seeds, {n} features)",
                 fontsize=14, fontweight="bold", y=0.96)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")
    pd.DataFrame({"feature": feats, "importance_mean": mu_s, "importance_std": std_s,
                  "importance_cv": cv_s, "rank": range(1, n + 1)}).to_csv(out.with_suffix(".csv"), index=False)
    print(f"  -> {out.with_suffix('.csv')}")


# ------------------------------------------------------------------- pipeline
def train_pipeline(df: pd.DataFrame, features: list, output_dir: Path,
                   models_dir: Path, log1pz: bool, title_note: str = ""):
    """Clean → OOF CV → metrics → save oof.csv + metrics.json + 10 seed models.
    Plotting is decoupled — run analyze_qso_photz_vs_specz.py on output_dir afterwards.
    Exported and reused by train_qso_euclid_photz_morph.py."""
    df = df.dropna(subset=features + ["qso_z"])
    df = df[df["qso_z"] > 0].reset_index(drop=True)
    X  = df[features].to_numpy(dtype=float)
    zs = df["qso_z"].to_numpy(dtype=float)
    y  = np.log1p(zs) if log1pz else zs
    print(f"  {len(df):,} QSOs, {len(features)} features, target={'log(1+z)' if log1pz else 'z'}")

    print("Out-of-fold cross-validation (z-stratified folds) ...")
    zp   = oof_predict(X, y, log1pz, zs)
    met  = photz_metrics(zs, zp)
    met3 = photz_metrics(zs[zs <= 3], zp[zs <= 3])
    print(f"  all z : N={met['N']:,}  bias={met['bias']:+.3f}  "
          f"σ_NMAD={met['sigma_nmad']:.3f}  outliers={100*met['outlier_frac']:.1f}%")
    print(f"  z<=3  : N={met3['N']:,}  σ_NMAD={met3['sigma_nmad']:.3f}  "
          f"outliers={100*met3['outlier_frac']:.1f}%")

    output_dir.mkdir(parents=True, exist_ok=True)
    df.assign(z_phot_oof=zp).to_csv(output_dir / "oof.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(
        {"features": features, "target": "log1p_z" if log1pz else "z",
         "all_z": met, "z_le_3": met3}, indent=2))
    print("Training final full-sample models ...")
    fit_final_models(X, y, log1pz, models_dir, features)
    feature_importance_plot(models_dir, features, output_dir / "feature_importance.png", met, met3)
    print(f"  z-phot plots: python src/analyze_qso_photz_vs_specz.py --dir {output_dir}"
          f"{(' --title-note ' + title_note) if title_note else ''}")


# ----------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log1pz", action="store_true",
                    help="predict log(1+z) internally (compresses the high-z tail)")
    args = ap.parse_args()

    user = os.getenv("PGUSER", WSDB_USER)
    df   = build_training_data(user)
    train_pipeline(df, ALL_FEATURES, OUTPUT_DIR, MODELS_DIR, args.log1pz)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
