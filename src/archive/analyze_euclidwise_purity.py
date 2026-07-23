#!/usr/bin/env python3
"""WISE-vs-PHZ purity cross-check for the Euclid+WISE quasar sample.

The W1-W2>=0.8 mid-IR cut selects AGN, but can admit contaminants. Euclid's own
photo-z classifier gives an independent QSO probability (phz_qso_prob). This asks:
do the sources that land in C3 (the stellar-body cluster, mostly 5-param) also have
high Euclid PHZ quasar probability — i.e. are they real AGN that are merely
astrometrically star-like (hosted/obscured) — or low, i.e. WISE contaminants?

Reads:
  output/umap/qso/arrays/qso_gaia_matches_euclidwise.csv          (cluster, aps, model_pgal)
  WSDB {PGUSER}.crossmatch_euclid_wise_qso_gaia            (phz_qso_prob, point_like_prob, w1_w2)

Writes:
  output/umap/qso/euclidwise_phz_purity.png
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
QSO_DIR = PROJECT_DIR / "output" / "umap" / "qso"
WSDB_HOST, WSDB_DB = "wsdb.ast.cam.ac.uk", "wsdb"
CLUSTER_COLOURS = {0: "#e6194b", 1: "#4363d8", 2: "#f58231", 3: "#3cb44b", 4: "#911eb4"}
APS_LABEL = {3: "2-param", 31: "5-param", 95: "6-param"}


def fetch_phz(user: str) -> pd.DataFrame:
    import psycopg2
    sql = (f"SELECT g_source_id AS gaia_source_id, phz_qso_prob, point_like_prob, "
           f"w1, w2, w1_w2 FROM {user}.crossmatch_euclid_wise_qso_gaia")
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df


def main() -> int:
    user = os.getenv("PGUSER", "siwei_zhao_2025")
    proj = pd.read_csv(QSO_DIR / "arrays" / "qso_gaia_matches_euclidwise.csv",
                       usecols=["gaia_source_id", "cluster", "astrometric_params_solved", "model_pgal"],
                       dtype={"gaia_source_id": "int64"})
    phz = fetch_phz(user)
    phz["gaia_source_id"] = phz["gaia_source_id"].astype("int64")
    df = proj.merge(phz, on="gaia_source_id", how="left")
    print(f"{len(df):,} Euclid+WISE sources; phz_qso_prob present for "
          f"{int(df['phz_qso_prob'].notna().sum()):,}")

    HI = 0.5  # PHZ QSO-probability threshold for "PHZ agrees it's a QSO"
    print("\n=== PHZ quasar probability by UMAP cluster ===")
    print(f"{'cluster':<8}{'N':>7}{'med phz_qso':>13}{'frac phz>0.5':>14}{'med ptlike':>12}")
    for c in sorted(df["cluster"].dropna().unique()):
        s = df[df["cluster"] == c]
        print(f"C{int(c):<7}{len(s):>7}{s['phz_qso_prob'].median():>13.2f}"
              f"{100*(s['phz_qso_prob'] >= HI).mean():>13.0f}%{s['point_like_prob'].median():>12.2f}")

    print("\n=== by astrometric_params_solved ===")
    for aps in sorted(df["astrometric_params_solved"].dropna().unique()):
        s = df[df["astrometric_params_solved"] == aps]
        print(f"{APS_LABEL.get(int(aps), aps):<10} N={len(s):>5}  med phz_qso={s['phz_qso_prob'].median():.2f}"
              f"  frac phz>0.5={100*(s['phz_qso_prob'] >= HI).mean():.0f}%")

    # focused: the C3 5-param population (the 'is it real AGN?' question)
    c3_5p = df[(df["cluster"] == 3) & (df["astrometric_params_solved"] == 31)]
    c1_2p = df[(df["cluster"] == 1) & (df["astrometric_params_solved"] == 3)]
    print(f"\nC3 & 5-param (N={len(c3_5p)}): median phz_qso_prob={c3_5p['phz_qso_prob'].median():.2f}, "
          f"{100*(c3_5p['phz_qso_prob'] >= HI).mean():.0f}% have phz_qso_prob>0.5")
    print(f"C1 & 2-param (N={len(c1_2p)}): median phz_qso_prob={c1_2p['phz_qso_prob'].median():.2f}, "
          f"{100*(c1_2p['phz_qso_prob'] >= HI).mean():.0f}% have phz_qso_prob>0.5")

    # ── figure: phz_qso_prob distributions ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    bins = np.linspace(0, 1, 26)
    ax = axes[0]
    for c in [0, 1, 2, 3, 4]:
        s = df[df["cluster"] == c]["phz_qso_prob"].dropna()
        if len(s):
            ax.hist(s, bins=bins, histtype="step", lw=2, color=CLUSTER_COLOURS[c],
                    label=f"C{c} (N={len(s)})", density=True)
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel("Euclid PHZ quasar probability"); ax.set_ylabel("normalised count")
    ax.set_title("PHZ QSO prob by UMAP cluster"); ax.legend(fontsize=8)

    ax = axes[1]
    for aps in [3, 31, 95]:
        s = df[df["astrometric_params_solved"] == aps]["phz_qso_prob"].dropna()
        if len(s):
            ax.hist(s, bins=bins, histtype="step", lw=2, label=f"{APS_LABEL[aps]} (N={len(s)})", density=True)
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel("Euclid PHZ quasar probability"); ax.set_title("PHZ QSO prob by astrometry")
    ax.legend(fontsize=8)

    ax = axes[2]
    sc = ax.scatter(df["w1_w2"], df["phz_qso_prob"], s=6, c=df["cluster"].map(CLUSTER_COLOURS),
                    alpha=0.4, linewidths=0)
    ax.axhline(0.5, color="k", ls="--", lw=1); ax.axvline(0.8, color="0.5", ls=":", lw=1)
    ax.set_xlabel("W1 - W2 (Vega)"); ax.set_ylabel("Euclid PHZ quasar probability")
    ax.set_title("WISE colour vs PHZ QSO prob (coloured by cluster)")
    fig.suptitle(f"Euclid+WISE quasar purity: mid-IR (W1-W2>=0.8) vs Euclid PHZ  "
                 f"[{len(df):,} sources]", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = QSO_DIR / "euclidwise_phz_purity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
