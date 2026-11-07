#!/usr/bin/env python3
"""
STEP 2 of the QSO-companion colour pipeline: an XGBoost photometric-redshift regressor
that predicts a QSO's redshift from its EUCLID COLOURS alone.

Step 1 (analyze_qso_desi_q1_color_z.py) showed the Euclid colour(z) locus is narrow, so
colour -> z is learnable. We train on the ~2,915 DESI/SDSS spectroscopic QSOs that have
Euclid photometry, using ONLY Euclid colours as input (I_E-Y_E, Y_E-J_E, J_E-H_E). It
MUST be Euclid-only: the target application (Step 3) is the WISE+Euclid QSO-pair
candidates whose members are blended in WISE, so only Euclid colours are available.

Honest evaluation
-----------------
The sample is small, so metrics come from 5-fold OUT-OF-FOLD cross-validation, averaged
over the 10 project seeds: every QSO gets a prediction from models that never saw it.
We report the standard photo-z metrics:
  Δz = (z_pred - z_spec) / (1 + z_spec)
  bias        = median(Δz)
  σ_NMAD      = 1.48 × median(|Δz - median Δz|)     (robust scatter)
  outlier frac= fraction with |Δz| > 0.15
Reliable to z~3 (dense training), extrapolation beyond (few anchors z>3.5) — we also
report metrics restricted to z<=3.

This is a REGRESSOR (XGBRegressor.predict); the predict_proba rule is for classifiers.

Reads (directly from WSDB; needs the VPN + PGUSER):
  {PGUSER}.crossmatch_qso_desi_dr1_euclid + crossmatch_qso_sdss_dr16_euclid
  (via build_dataframe() in analyze_qso_desi_q1_color_z.py — identical flux→colour logic)
Writes:
  output/qso_zphot/models/xgb_zphot_seed{seed}.json  (10 full-sample models for Step 3)
  output/qso_zphot/qso_euclid_zphot_oof.csv          (out-of-fold z_pred per QSO)
  output/qso_zphot/qso_euclid_zphot_metrics.json
  output/qso_zphot/qso_euclid_zphot.png              (z_pred vs z_spec + residuals)

Usage:
  PGUSER=siwei_zhao_2025 python src/train_qso_euclid_zphot.py   # pull from WSDB, colours only, predict z
  python src/train_qso_euclid_zphot.py --add-mag               # + H_E magnitude as a flux anchor
  python src/train_qso_euclid_zphot.py --log1pz                # predict log(1+z) internally
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
from sklearn.model_selection import KFold, train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "output" / "qso_zphot"
MODELS_DIR = OUTPUT_DIR / "models"

# Pull the training sample straight from the two spec-QSO × Euclid WSDB tables, reusing the
# identical flux→mag→colour logic of Step 1 (so features match exactly).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_qso_desi_q1_color_z import build_dataframe, WSDB_USER  # noqa: E402

SEEDS = [42, 123, 456, 789, 2025, 314, 999, 1337, 7777, 31415]
COLOR_FEATURES = ["IE_YE", "YE_JE", "JE_HE"]   # the 3 independent Euclid colours
MAG_ANCHOR     = "HE"                          # optional flux anchor (--add-mag)
N_FOLDS = 5

XGB_PARAMS = dict(
    objective="reg:squarederror",
    max_depth=6,               # shallower than the classifier (small sample)
    learning_rate=0.05,
    n_estimators=600,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    tree_method="hist",
    eval_metric="rmse",
    early_stopping_rounds=30,
)

OUTLIER_DZ = 0.15              # |Δz| threshold for the catastrophic-outlier fraction


# ------------------------------------------------------------------------- metrics
def zphot_metrics(zs: np.ndarray, zp: np.ndarray) -> dict:
    dz = (zp - zs) / (1.0 + zs)
    bias = float(np.median(dz))
    sig  = float(1.48 * np.median(np.abs(dz - bias)))
    out  = float(np.mean(np.abs(dz) > OUTLIER_DZ))
    rmse = float(np.sqrt(np.mean((zp - zs) ** 2)))
    return {"N": int(len(zs)), "bias": bias, "sigma_nmad": sig,
            "outlier_frac": out, "rmse": rmse}


# --------------------------------------------------------------------------- train
def _fit_one(X_tr, y_tr, seed):
    """One XGBRegressor with an internal val split for early stopping."""
    Xa, Xv, ya, yv = train_test_split(X_tr, y_tr, test_size=0.15, random_state=seed)
    m = xgb.XGBRegressor(random_state=seed, **XGB_PARAMS)
    m.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)
    return m


def oof_predict(X, y, log1pz: bool):
    """5-fold out-of-fold predictions, averaged over all seeds. Returns z_pred (in z),
    even when the model is trained on log(1+z)."""
    n = len(y)
    oof = np.zeros((len(SEEDS), n))
    for si, seed in enumerate(SEEDS):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr, te in kf.split(X):
            m = _fit_one(X[tr], y[tr], seed)
            pred = m.predict(X[te])
            oof[si, te] = np.expm1(pred) if log1pz else pred
        print(f"  seed {seed}: OOF fold set done")
    return oof.mean(axis=0)


def fit_final_models(X, y, log1pz: bool, models_dir: Path, features):
    """Train the 10 full-sample models used to predict z for Step-3 candidates."""
    models_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        m = _fit_one(X, y, seed)
        m.save_model(models_dir / f"xgb_zphot_seed{seed}.json")
    (models_dir / "features.json").write_text(json.dumps(
        {"features": features, "target": "log1p_z" if log1pz else "z"}, indent=2))
    print(f"  -> {len(SEEDS)} models in {models_dir}")


FEATURE_LABELS = {"IE_YE": "$I_E-Y_E$", "YE_JE": "$Y_E-J_E$", "JE_HE": "$J_E-H_E$",
                  "HE": "$H_E$", "point_like_prob": "point_like_prob"}


def feature_importance_plot(models_dir: Path, features, out: Path):
    """Horizontal bar of XGBoost GAIN importance, averaged over the 10 seed models
    (% of total gain, ± seed-to-seed sd)."""
    imps = np.zeros((len(SEEDS), len(features)))
    for si, seed in enumerate(SEEDS):
        m = xgb.XGBRegressor(); m.load_model(models_dir / f"xgb_zphot_seed{seed}.json")
        score = m.get_booster().get_score(importance_type="gain")   # {'f0':gain, ...}
        for fi in range(len(features)):
            imps[si, fi] = score.get(f"f{fi}", 0.0)
    tot = imps.sum(axis=1, keepdims=True); tot[tot == 0] = 1.0
    pct = 100 * imps / tot                                          # per-seed % of total gain
    mean, std = pct.mean(0), pct.std(0)
    order = np.argsort(mean)
    labels = [FEATURE_LABELS.get(features[i], features[i]) for i in order]
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(features) + 1.6))
    ax.barh(labels, mean[order], xerr=std[order], color="#4363d8", edgecolor="k", capsize=3)
    for yi, v in enumerate(mean[order]):
        ax.text(v + 0.8, yi, f"{v:.0f}%", va="center", fontsize=9)
    ax.set_xlim(0, min(100, mean.max() + std.max() + 8))
    ax.set_xlabel("mean gain importance  (% of total, 10-seed mean ± sd)", fontsize=10)
    ax.set_title(f"XGBoost feature importance — {len(features)} features", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------- plot
def make_plot(zs, zp, met, met3, out: Path, features):
    dz = (zp - zs) / (1.0 + zs)
    zmax = 4.2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 6.2),
                                 gridspec_kw={"width_ratios": [1, 1]})
    # (1) z_pred vs z_spec
    a1.hexbin(zs, zp, gridsize=55, bins="log", cmap="viridis",
              extent=(0, zmax, 0, zmax), mincnt=1)
    a1.plot([0, zmax], [0, zmax], "k-", lw=1)
    zl = np.linspace(0, zmax, 100)
    a1.plot(zl, zl + OUTLIER_DZ * (1 + zl), "r--", lw=0.9, label=f"|Δz|={OUTLIER_DZ}")
    a1.plot(zl, zl - OUTLIER_DZ * (1 + zl), "r--", lw=0.9)
    a1.set_xlim(0, zmax); a1.set_ylim(0, zmax)
    a1.set_xlabel("spectroscopic $z$", fontsize=11)
    a1.set_ylabel("photometric $z$ (OOF, Euclid colours)", fontsize=11)
    a1.set_title("Euclid-colour photo-z — out-of-fold", fontsize=13, fontweight="bold")
    box = (f"all z (N={met['N']:,}):\n bias={met['bias']:+.3f}\n"
           f" σ_NMAD={met['sigma_nmad']:.3f}\n outliers={100*met['outlier_frac']:.1f}%\n"
           f"z≤3 (N={met3['N']:,}):\n σ_NMAD={met3['sigma_nmad']:.3f}\n"
           f" outliers={100*met3['outlier_frac']:.1f}%")
    a1.text(0.03, 0.97, box, transform=a1.transAxes, fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.92))
    a1.legend(loc="lower right", fontsize=9)
    # (2) residuals
    a2.hexbin(zs, dz, gridsize=55, bins="log", cmap="magma",
              extent=(0, zmax, -1, 1), mincnt=1)
    a2.axhline(0, color="k", lw=1)
    a2.axhline(OUTLIER_DZ, color="r", ls="--", lw=0.9)
    a2.axhline(-OUTLIER_DZ, color="r", ls="--", lw=0.9)
    # running median residual
    edges = np.arange(0, zmax + 0.2, 0.2)
    mid, med = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (zs >= lo) & (zs < hi)
        if m.sum() >= 15:
            mid.append(0.5 * (lo + hi)); med.append(np.median(dz[m]))
    a2.plot(mid, med, "c-", lw=1.8, label="median Δz")
    a2.set_xlim(0, zmax); a2.set_ylim(-1, 1)
    a2.set_xlabel("spectroscopic $z$", fontsize=11)
    a2.set_ylabel(r"$\Delta z = (z_{\rm phot}-z_{\rm spec})/(1+z_{\rm spec})$", fontsize=11)
    a2.set_title("residuals vs redshift", fontsize=13, fontweight="bold")
    a2.legend(loc="upper right", fontsize=9)
    fig.suptitle(f"XGBoost Euclid photo-z for spectroscopic QSOs "
                 f"({', '.join(features)})", fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=132, bbox_inches="tight"); plt.close(fig)   # <2000px wide (viewer limit)
    print(f"  -> {out}")


# --------------------------------------------------------------------------- pipeline
def train_pipeline(df, features, output_dir: Path, models_dir: Path, log1pz: bool):
    """Shared: clean -> OOF CV -> metrics -> z-phot plot -> save 10 models -> importance.
    Reused by the colour-only model (this file) and the +point_like_prob model
    (train_qso_euclid_zphot_morph.py)."""
    df = df.dropna(subset=features + ["qso_z"])
    df = df[df["qso_z"] > 0].reset_index(drop=True)
    X = df[features].to_numpy(dtype=float)
    zs = df["qso_z"].to_numpy(dtype=float)
    y = np.log1p(zs) if log1pz else zs
    print(f"  {len(df):,} QSOs, features={features}, target={'log(1+z)' if log1pz else 'z'}")

    print("Out-of-fold cross-validation ...")
    zp = oof_predict(X, y, log1pz)
    met  = zphot_metrics(zs, zp)
    met3 = zphot_metrics(zs[zs <= 3], zp[zs <= 3])
    print(f"  all z : N={met['N']:,}  bias={met['bias']:+.3f}  σ_NMAD={met['sigma_nmad']:.3f}  "
          f"outliers={100*met['outlier_frac']:.1f}%")
    print(f"  z<=3  : N={met3['N']:,}  σ_NMAD={met3['sigma_nmad']:.3f}  "
          f"outliers={100*met3['outlier_frac']:.1f}%")

    output_dir.mkdir(parents=True, exist_ok=True)
    df.assign(z_phot_oof=zp).to_csv(output_dir / "qso_euclid_zphot_oof.csv", index=False)
    (output_dir / "qso_euclid_zphot_metrics.json").write_text(json.dumps(
        {"features": features, "target": "log1p_z" if log1pz else "z",
         "all_z": met, "z_le_3": met3}, indent=2))
    make_plot(zs, zp, met, met3, output_dir / "qso_euclid_zphot.png", features)

    print("Training final full-sample models ...")
    fit_final_models(X, y, log1pz, models_dir, features)
    feature_importance_plot(models_dir, features, output_dir / "qso_euclid_zphot_importance.png")


# ---------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--add-mag", action="store_true",
                    help=f"add {MAG_ANCHOR} magnitude as a flux anchor (default: colours only)")
    ap.add_argument("--log1pz", action="store_true",
                    help="predict log(1+z) internally (compresses the high-z tail)")
    args = ap.parse_args()
    features = COLOR_FEATURES + ([MAG_ANCHOR] if args.add_mag else [])

    # pull directly from the two spec-QSO × Euclid tables (fast, ~0.3 s; no CSV cache)
    df = build_dataframe(os.getenv("PGUSER", WSDB_USER))
    train_pipeline(df, features, OUTPUT_DIR, MODELS_DIR, args.log1pz)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
