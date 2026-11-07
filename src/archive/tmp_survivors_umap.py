#!/usr/bin/env python3
"""TEMPORARY — project the "survivor" picks onto the Q1 UMAP.

Survivors = placed Q1 hand-picks that (a) have Gaia PM significance <= 10 (or no PM at
all) AND (b) sit INSIDE the candidate 16-84% band in ALL THREE Euclid colours
(I_E-Y_E, Y_E-J_E, J_E-H_E) at their XGB photo-z — i.e. the 7 sources most consistent
with the QSO colour(z) locus (see the filter in tmp_picked_color_vs_z).

Their Gaia source_ids are projected onto the UMAP with the SAME machinery as
tmp_picked_umap.py: 14 derived Gaia features pulled fresh -> standardise with the Q1
train mean/std -> q1_indiv_reducer.joblib.transform -> nearest-Q1 cluster label.

Output: output/survivors_qso_umap.png
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_picked_qso_contact_sheet import PICKED, CLUSTER_COLOURS
from tmp_picked_umap import fetch_features, _load_reducer, ARRAYS_DIR, CSV
from tmp_picked_color_vs_z import predict_picked, POP_CACHE, ZMAX_PLOT
from analyze_qso_color_vs_specz import COLORS, _running_stats, WSDB_USER
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

OUT = Path(__file__).resolve().parents[1] / "output" / "survivors_qso_umap.png"


def survivor_qsos(user):
    """WISE qso_ids of the placed Q1 picks with pm_sig<=10 (or NULL) AND inside all
    three colour 16-84% bands at their z_xgb."""
    picked, _ = predict_picked(user)
    picked = picked[picked["z_xgb"].notna()].copy()
    pop = pd.read_csv(POP_CACHE)
    zp = pop["z_xgb"].to_numpy(float)

    def in_band(zs, cs, col):
        c = pop[col].to_numpy(float)
        m = np.isfinite(c) & np.isfinite(zp) & (zp <= ZMAX_PLOT)
        mid, _, lo, hi = _running_stats(zp[m], c[m])
        return (cs >= np.interp(zs, mid, lo)) & (cs <= np.interp(zs, mid, hi))

    inb = np.ones(len(picked), bool)
    for _, b, r in COLORS:
        col = f"{b}_{r}"
        inb &= in_band(picked["z_xgb"].to_numpy(float), picked[col].to_numpy(float), col)
    pm = picked["pm_sig"]
    keep = (pm.isna() | (pm <= 10)) & inb
    surv = picked[keep]
    print(f"  survivors: {len(surv)} qso_ids "
          f"({int(pm[keep].isna().sum())} with no PM, {int((pm[keep] <= 10).sum())} pm<=10)")
    return surv["qso_id"].tolist()


def main() -> int:
    user = os.environ.get("PGUSER", WSDB_USER)

    surv_q = set(survivor_qsos(user))
    flat = pd.read_csv(CSV, dtype={"gaia_source_id": "int64"})
    q_of = dict(zip(flat["gaia_source_id"], flat["qso_id"]))
    # Gaia ids among the hand-picks whose qso_id is a survivor (per gaia id)
    gids = [g for g in PICKED if q_of.get(g) in surv_q]
    wise_of = {g: q_of.get(g, "") for g in gids}
    print(f"  projecting {len(gids)} Gaia source_ids for {len(surv_q)} survivor qso_ids")

    ids = pd.DataFrame({"gaia_source_id": gids})
    feats = fetch_features(ids["gaia_source_id"].tolist(), user)
    feats = feats.rename(columns={"source_id": "gaia_source_id"})
    ids = ids.merge(feats, on="gaia_source_id", how="left")

    # standardise with Q1 train mean/std, then reducer.transform (== project_and_assign)
    q1 = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    labels = np.load(ARRAYS_DIR / "q1_cluster_labels.npy")
    q1_emb = q1["embedding"]
    mean, std = q1["mean"], q1["std"]
    std_safe = np.where(std > 0, std, 1.0)
    X = ids[list(SELECTED_GAIA_PARAMETERS)].apply(pd.to_numeric, errors="coerce").values.astype(float)
    Xs = np.where(np.isfinite((X - mean) / std_safe), (X - mean) / std_safe, 0.0)
    emb = _load_reducer().transform(Xs).astype(float)
    ids["umap1"], ids["umap2"] = emb[:, 0], emb[:, 1]

    valid = labels >= 0
    nn = NearestNeighbors(n_neighbors=1).fit(q1_emb[valid])
    _, idx = nn.kneighbors(emb)
    ids["cluster"] = labels[np.where(valid)[0][idx.ravel()]]
    ids["wise"] = ids["gaia_source_id"].map(wise_of)

    cluster_ids = sorted(set(labels[labels >= 0]))
    centroids = {c: (q1_emb[labels == c, 0].mean(), q1_emb[labels == c, 1].mean())
                 for c in cluster_ids}
    xlim = (q1_emb[:, 0].min() - 0.8, q1_emb[:, 0].max() + 0.8)
    ylim = (q1_emb[:, 1].min() - 0.8, q1_emb[:, 1].max() + 0.8)

    # ── plot ──
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.scatter(q1_emb[:, 0], q1_emb[:, 1], s=2, c="#e2e2e2", alpha=0.6,
               linewidths=0, rasterized=True, zorder=1)
    for c in cluster_ids:
        s = ids[ids.cluster == c]
        if len(s):
            ax.scatter(s["umap1"], s["umap2"], s=240, marker="*", c=CLUSTER_COLOURS[c],
                       edgecolors="k", linewidths=1.0, zorder=6)
    for _, r in ids.iterrows():
        t = ax.text(r["umap1"] + 0.12, r["umap2"] + 0.12, str(r["wise"]), fontsize=7,
                    color="0.15", zorder=7, ha="left", va="bottom")
        t.set_path_effects([pe.Stroke(linewidth=1.3, foreground="white"), pe.Normal()])
    for c in cluster_ids:
        cx, cy = centroids[c]
        t = ax.text(cx, cy, f"C{c}", fontsize=13, fontweight="bold",
                    ha="center", va="center", color=CLUSTER_COLOURS[c], zorder=8)
        t.set_path_effects([pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()])

    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_xticks([]); ax.set_yticks([])
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=CLUSTER_COLOURS[c],
                      markeredgecolor="k", markersize=6, label=f"Cluster {c}")
               for c in cluster_ids]
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.92)
    ax.set_title(f"{len(gids)} survivor QSO picks projected onto the Q1 UMAP\n"
                 f"(pm_sig ≤ 10 & inside all three colour 16–84% bands · 14 features → reducer.transform)",
                 fontsize=13, fontweight="bold")

    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n  survivor projections (WISE  ->  cluster):")
    for _, r in ids.sort_values("cluster").iterrows():
        print(f"    {r['wise']:<22} C{int(r['cluster'])}  umap=({r['umap1']:.2f}, {r['umap2']:.2f})")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
