#!/usr/bin/env python3
"""
Cross-match the known dual/projected/contaminant quasar-pair reference list to Gaia DR3
and compute our XGB extragalactic probability p(gal) on the fly — to test whether known
PHYSICAL dual quasars present as high-p(gal) Gaia sources (i.e. whether a Gaia p(gal)
self-pairing finder could recover them), and how they separate from the negatives.

Reference list (built by fetch_known_dual_quasars.py + Deng-IV tex parse):
  output/qso_pairs/known_dual_quasars_deng2026IV.csv        (17 confirmed physical duals; positives)
  output/qso_pairs/known_projected_quasars_deng2026IV.csv   (projected pairs; negatives, diff z)
  output/qso_pairs/known_dual_quasars.csv                   (VODKA + Hwang; dual/lensed + quasar+star)

For each component: q3c-match to gaia_dr3.gaia_source (nearest within 1.5"), pull the 14
SELECTED_GAIA_PARAMETERS via deploy's _feature_select_expr, and run the Q1 XGB ensemble
(load_ensemble/predict_ensemble) -> p(gal) mean/std. Identical machinery to
deploy_q1_models_on_gaia.py, so consistent with the all-Gaia predictions.

Writes:  output/qso_pairs/known_pairs_gaia_pgal.csv   (+ prints validation summary)

Usage:   PGUSER=siwei_zhao_2025 python src/analyze_known_dual_quasars_gaia.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS          # noqa: E402
from deploy_q1_models_on_gaia import (_feature_select_expr, load_ensemble,  # noqa: E402
                                      predict_ensemble, Q1_MODEL_DIR)
from analyze_qso_color_vs_specz import WSDB_HOST, WSDB_DB, WSDB_USER    # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
QP_DIR = PROJECT_DIR / "output" / "qso_pairs"
OUT_CSV = QP_DIR / "known_pairs_gaia_pgal.csv"
MATCH_R = 1.5   # arcsec


def load_reference() -> pd.DataFrame:
    """Combine the reference CSVs into one components frame: pair_id, member, ra, dec, z, label."""
    frames = []
    dual = pd.read_csv(QP_DIR / "known_dual_quasars_deng2026IV.csv")
    dual = dual.assign(pair_id=dual.system_id, label="dual")
    frames.append(dual[["pair_id", "member", "ra", "dec", "z", "label"]])

    pq = QP_DIR / "known_projected_quasars_deng2026IV.csv"
    if pq.exists():
        p = pd.read_csv(pq); p = p.assign(pair_id=p.system_id, label="projected")
        frames.append(p[["pair_id", "member", "ra", "dec", "z", "label"]])

    vod = pd.read_csv(QP_DIR / "known_dual_quasars.csv")
    vod = vod[vod.tier == "resolved"].copy()           # the VODKA HST-classified systems
    cls = vod["class"].fillna("")
    vod["label"] = np.where(cls.str.contains("star"), "quasar+star",
                    np.where(cls.str.contains("lens"), "dual_or_lens",
                    np.where(cls.str.contains("dual"), "dual_or_lens", "other")))
    vod = vod.assign(pair_id=vod["name"], member="")
    frames.append(vod[["pair_id", "member", "ra", "dec", "z", "label"]])

    df = pd.concat(frames, ignore_index=True)
    df = df[df.ra.notna() & df.dec.notna()].reset_index(drop=True)
    df["cid"] = df["pair_id"].astype(str) + "|" + df["member"].fillna("").astype(str)
    return df


def match_and_predict(df: pd.DataFrame, user: str) -> pd.DataFrame:
    import psycopg2
    from psycopg2.extras import execute_values
    feat_sql = ",\n           ".join(_feature_select_expr(c) for c in SELECTED_GAIA_PARAMETERS)
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30,
                            keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 600000")
    # a real (temp) driver table lets the planner use gaia_source's q3c index (a VALUES
    # CTE does not — it forces a full-table scan). Same pattern as the crossmatch builders.
    cur.execute("CREATE TEMP TABLE seed (cid text, sra double precision, sdec double precision)")
    execute_values(cur, "INSERT INTO seed (cid, sra, sdec) VALUES %s",
                   [(r.cid, float(r.ra), float(r.dec)) for r in df.itertuples()])
    cur.execute("ANALYZE seed")
    sql = f"""
      SELECT DISTINCT ON (seed.cid)
             seed.cid, g.source_id,
             q3c_dist(seed.sra, seed.sdec, g.ra, g.dec)*3600.0 AS gaia_sep_arcsec,
             g.phot_g_mean_mag,
             {feat_sql}
      FROM seed, gaia_dr3.gaia_source g
      WHERE q3c_join(seed.sra, seed.sdec, g.ra, g.dec, {MATCH_R}/3600.0)
      ORDER BY seed.cid, q3c_dist(seed.sra, seed.sdec, g.ra, g.dec)
    """
    print(f"  q3c-matching {len(df)} components to gaia_dr3.gaia_source (temp-table driver) ...")
    g = pd.read_sql_query(sql, conn); conn.close()
    print(f"  {len(g)} Gaia matches within {MATCH_R}\"")

    for c in SELECTED_GAIA_PARAMETERS:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    models = load_ensemble(Q1_MODEL_DIR, "Q1")
    X = g[list(SELECTED_GAIA_PARAMETERS)].to_numpy(np.float32)
    g["pgal"], g["pgal_std"] = predict_ensemble(models, X)
    keep = (["cid", "source_id", "gaia_sep_arcsec", "phot_g_mean_mag", "pgal", "pgal_std"]
            + list(SELECTED_GAIA_PARAMETERS))          # carry features for the UMAP projection
    return df.merge(g[keep], on="cid", how="left")


def main() -> int:
    user = os.getenv("PGUSER", WSDB_USER)
    df = load_reference()
    print(f"  reference: {len(df)} components")
    print(df["label"].value_counts().to_string())
    m = match_and_predict(df, user)

    # project the Gaia-matched components onto the Q1 14-feature UMAP + assign cluster
    umap_ok = False
    try:
        from analyze_umap_qso_catwise_euclid_gaia import project_and_assign
        mm = m[m.source_id.notna()].copy()
        proj = project_and_assign(mm)
        m = m.merge(proj[["cid", "umap1", "umap2", "cluster"]], on="cid", how="left")
        umap_ok = True
    except Exception as exc:
        print(f"  UMAP projection failed ({type(exc).__name__}: {exc})")

    QP_DIR.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT_CSV, index=False)

    print("\n================ Gaia match + p(gal) by label ================")
    for lab, sub in m.groupby("label"):
        mm = sub[sub.source_id.notna()]
        pg = mm["pgal"].dropna()
        print(f"  {lab:12s}  N={len(sub):3d}  Gaia-matched={len(mm):3d} ({100*len(mm)/len(sub):3.0f}%)"
              f"  median p(gal)={pg.median():.2f}  frac>0.5={ (pg>0.5).mean():.2f}" if len(pg) else
              f"  {lab:12s}  N={len(sub):3d}  Gaia-matched={len(mm):3d}")

    # per-system recovery for the confirmed physical duals (both members high-p(gal))
    dual = m[m.label == "dual"]
    rec = dual.groupby("pair_id").agg(
        n=("member", "size"),
        n_gaia=("source_id", lambda s: s.notna().sum()),
        n_pgal=("pgal", lambda s: (s > 0.5).sum()))
    both = (rec.n_pgal == 2).sum()
    print(f"\n  Confirmed physical duals: {len(rec)} systems")
    print(f"    both members Gaia-matched : {(rec.n_gaia==2).sum()}/{len(rec)}")
    print(f"    both members p(gal)>0.5   : {both}/{len(rec)}   (finder would recover these as a high-p(gal) pair)")

    if umap_ok:
        print("\n================ UMAP cluster by label ================")
        ct = m[m.cluster.notna()].pivot_table(index="label", columns="cluster",
                                              values="cid", aggfunc="count", fill_value=0)
        print(ct.to_string())
        make_umap_plot(m)
    print(f"\n  -> {OUT_CSV}")
    return 0


def make_umap_plot(m: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    q1 = np.load(PROJECT_DIR / "output" / "umap" / "arrays" / "q1_indiv.npz", allow_pickle=True)
    emb = q1["embedding"]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(emb[::40, 0], emb[::40, 1], s=2, c="0.85", lw=0, rasterized=True, label="Q1 Gaia (background)")
    colors = {"dual": "#e6194B", "projected": "#3cb44b", "quasar+star": "#4363d8",
              "dual_or_lens": "#f58231", "other": "#000000"}
    sub = m[m.umap1.notna()]
    for lab, s in sub.groupby("label"):
        ax.scatter(s.umap1, s.umap2, s=45, color=colors.get(lab, "0.4"),
                   edgecolor="k", lw=0.4, label=f"{lab} ({len(s)})")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title("Known QSO-pair members on the Q1 Gaia 14-feature UMAP")
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    out = QP_DIR / "known_pairs_umap.png"
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
