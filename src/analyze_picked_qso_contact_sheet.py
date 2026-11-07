#!/usr/bin/env python3
"""TEMPORARY — hand-picked QSO Euclid+WISE cutouts → one multi-cluster contact sheet.

Sources are the ones Sylvia manually screenshotted from
output/qso_cutouts/catwise_e_g (+ the multi_euclid blend folder). Their exact
gaia_source_ids were recovered from the manifests / flat UMAP table.

Re-renders every stamp with the same style as download_qso_catwise_e_g_cutouts.py
(Euclid centroid crosshair, Gaia blue rings, secondary-Euclid red crosshairs,
WISE magenta x), then lays them out into contact sheets: rows grouped by UMAP
cluster, sorted within cluster by descending XGBoost P(gal), wrapping at NCOLS,
with a rotated cluster label on the left and a P(gal) colour-bar legend on top.

This file is the single source of truth for the hand-picked list (PICKED). It also
projects the picks onto the NEW combined QSO×Gaia UMAP locally (14 features straight
from the flat CSV -> saved reducer), so no import from analyze_picked_qso_umap.py is
needed (that module imports PICKED from here — the projection is duplicated to break
the loop).

Outputs (output/picked_qso/cutouts/):
  picked_qso_contact_sheet_cluster{c}.png  ONE sheet per NEW combined QSO×Gaia UMAP cluster
                                           (0,1,2,3,...), like the per-cluster download sheets
  og_picked_qso_contact_sheet.png          all clusters in one sheet, rows = original Q1 UMAP cluster
"""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

import download_qso_catwise_e_g_cutouts as dl
from download_qso_catwise_e_g_cutouts import (
    fetch_blend_info, fetch_gaia_positions, fetch_euclid_secondary_positions,
    fetch_euclid, render_to_array, pgal_color, EUCLID_RING, WISE_RING, PROJECT_DIR,
)
from render_q1_cutout import GAIA_RING, GAIA_FOV_RING
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

CLUSTER_COLOURS = {0: "#e6194b", 1: "#4363d8", 2: "#f58231", 3: "#3cb44b", 4: "#911eb4"}
NCOLS = 6
SIZE  = 6.0

# NEW combined QSO×Gaia UMAP (the one behind combined_qso_gaia_clusters.png). We project the
# picks onto it here — the 14 features are already carried in the flat CSV, so no DB query and,
# crucially, no import from analyze_picked_qso_umap.py (which imports PICKED from THIS file);
# keeping the projection local keeps this script the single source of truth for the pick list.
NEW_UMAP_NPZ = PROJECT_DIR / "output" / "qso_umap" / "base" / "arrays" / "combined_qso_gaia.npz"
NEW_UMAP_RED = PROJECT_DIR / "output" / "qso_umap" / "base" / "arrays" / "combined_qso_gaia_reducer.joblib"

# Authoritative gaia_source_ids of the 28 hand-picked sources (recovered + verified).
PICKED = [
    6438777823548176128, 4182448842295887616, 5057350452333817344,
    4830604663329901568, 4831228326941346432, 4833094777992052224,   # C0
    5084879612154785408, 1441009154702250880, 2259779798497927296,
    4182247700384158848, 5084667509488668800, 4830456499841387136,
    4830298238887168640, 5081268364996187520, 4832096288290543104,
    4182516707071325824,                                             # C1
    4831443586405716864, 1634684137909491200, 164408736752500992,
    428261363861755520,                                             # C2
    1441080176280946304, 1633244361791693056, 4829663202202378880,
    2257449035710066688, 4182286939207868032, 2257600940112711808,
    239508267147980160,                                             # C3
    5921742131381273216,                                             # C4
]

# All sources from output/qso_cutouts/catwise_e_g_blend/multi_gaia (dedup handles overlaps)
MULTI_GAIA = [
    4182450736372664320, 5057350452333817344, 1634449293392314624,
    3219151075810716800, 4830604663329901568, 4974123297627607936,
    2161082515187886208, 4182448842295887616, 4778637895508914048,
    4829578127489714304,
]

# Second batch of hand-picked cutouts (30 screenshots, 2026-07-31). Every gaia_source_id
# was transcribed from the tile "id=" field and POSITION-VERIFIED against gaia_dr3.gaia_source
# (all within <1.1" of the WISE coord; 3 ids fixed a transcription typo). dedup drops overlaps.
NEW_PICKS = [
    5084798037839071104, 4831228326941346432, 4182450736372664320,   # C0 / blend
    5056647688309940992, 4830357749953156864, 1779205324949283840,   # C1
    4828842932168558464, 4788719180185622400, 4831889472730872320,   # C1
    1634386513855872512, 1440790970362220928, 5921381152269576704,   # C1
    2161082515187886208, 3340984072339594112,                        # C1
    4831026218663275520, 5057454218743874304, 4831443586405716864,   # C2
    4182516707071325824, 3219223299980727168, 4182450491559190272,   # C2 / ERO
    2257716633650152704, 5921753165167168896, 428297441598454528,    # C2 / ERO
    164408736752500992,  4831169670570860288, 3219147845996327168,   # C2 / ERO
    428261363861755520,  5921747667588276096, 428233085802923136,    # C2 / ERO
    1634684137909491200,                                             # C2
]

# Third batch (11 screenshots, 2026-07-31) — all position-verified against gaia_dr3
# (within <1.3"); 3 are dups of earlier batches (handled by dedup).
NEW_PICKS_2 = [
    3219582629829500672, 428233085802923136,  6438513253560492288,   # C2/C3 ERO
    4974173080592248064, 4182285397312686720, 5921380636870880768,   # C3 ERO
    2257763328534730112, 5921742131381273216, 5921744437772396288,   # C4
    4829578127489714304, 1439903282227125504,                        # C3
]
# Fourth batch (3 screenshots, 2026-07-31) — all Cluster 3, Q1+WISE; position-verified
# against gaia_dr3.gaia_source (ids 1 & 3 within 0.1"; id 2 offset because its tile is
# centred on the Euclid coord, but G matched the tile exactly).
NEW_PICKS_3 = [
    2258146847636249472, 4830217910113107328, 4837412869392745216,   # C3
]
# Fifth batch (7 screenshots, 2026-08-02) — all Cluster 3 Q1+WISE; position-verified
# against gaia_dr3.gaia_source (<1.5"). 5 of the 7 were dups of earlier batches; only
# these two are genuinely new.
NEW_PICKS_4 = [
    1634357204998629248, 2161399616917323904,                        # C3
]
# Sixth batch (23 screenshots, 2026-08-02) — 12 were dups of earlier batches; these 11 are
# new, all position-verified against gaia_dr3.gaia_source (<1.5"). All C3 except the last (C4).
NEW_PICKS_5 = [
    5060421727612552192, 4828795103412490880, 4831373526899346688,   # C3
    4830236636170244480, 4829486219486241152, 4788982959896256896,   # C3
    4830334625849146496, 4831376962873209472, 1440690773070473984,   # C3
    2257853346751659776,                                             # C3
    1439921256665537280,                                             # C4
]
# Seventh batch (19 screenshots, 2026-08-03) — 10 were dups of earlier batches; these 9 are
# new, all position-verified against gaia_dr3.gaia_source (<0.2"). Q1 cluster labels below.
NEW_PICKS_6 = [
    4828898732383206144, 4830580160539896576, 4831474407090925440,   # C3  Q1
    4832548423792490112, 1779215220553631104,                        # C2  (Q1, ERO)
    1633614381817769216, 4782810267259320832, 1634283297200931968,   # C1  Q1
    2257487729072199936,                                             # C1  Q1
]
# Eighth batch (29 screenshots, 2026-08-04) — all Q1 Cluster 1; 12 were dups of earlier batches,
# these 17 are new, all position-verified against gaia_dr3.gaia_source (<1.3", most <0.25").
NEW_PICKS_7 = [
    4831497702993838208, 2161097564752837632, 4831389714631272192,   # C1
    4837362399231894016, 4789114519039299072, 5081508264689256064,   # C1
    2161467309897221248, 5056956788516657664, 4829675576003215360,   # C1
    4830092973809558656, 4830241236081143552, 2257436219527618688,   # C1
    4836992890310388224, 4830394721032009600, 1633153171046749568,   # C1
    4829463542057972992, 1634387205346060032,                        # C1
]
# Ninth batch (16 screenshots, 2026-08-04) — all Q1 Cluster 1 (2 ERO); 1 was a dup, these 15
# are new, all position-verified against gaia_dr3.gaia_source (<1.2", most <0.2").
NEW_PICKS_8 = [
    4789090673381469696, 5057332653991018752, 1633525939848315904,   # C1
    4832622469029004544, 2161141476498562816, 2257675779921251584,   # C1
    1634569410742089856, 2258003949780827264, 4788736428774100992,   # C1
    4829798579571475328, 2161377901564466816, 4830073697996578304,   # C1
    5921781546301474816, 2257748618269002240, 1773190893265500672,   # C1 (2 ERO)
]
# Tenth batch (21 screenshots, 2026-08-04) — all Q1 Cluster 1 (2 ERO); 5 were dups, these 9
# are new, all position-verified against gaia_dr3.gaia_source (<0.2").
NEW_PICKS_9 = [
    1633150353548333440, 4829968247959853824, 1439825972814811904,   # C1
    2258156610097530624, 4830008796745697536, 4831599132940501120,   # C1
    5921747701962678528, 5056273274536520832, 4182519627649295104,   # C1 (2 ERO)
]
# Eleventh batch (34 screenshots, 2026-08-04) — all Q1 Cluster 3 (4 ERO); 3 were dups, these 31
# are new, all position-verified against gaia_dr3.gaia_source (<0.25"; one OCR id auto-corrected).
NEW_PICKS_10 = [
    2258302256733125248, 5081520668554820096, 4831928677191761792,   # C3
    5081497372652200704, 5060794260194102656, 5084658885195862528,   # C3
    4829745803013102848, 5084862397926956288, 4788825076900398720,   # C3
    4831094285304869376, 1440800655513876096, 4837275292999765120,   # C3
    5060332014337260800, 1095800501155222528, 5057111579137948416,   # C3
    4830250513209515520, 4365614079180846464, 4837420703412901504,   # C3
    5084697230663895296, 4829905919393908736, 4829489032688425856,   # C3
    4831380948601894272, 2161162259844307968, 5081574304106782720,   # C3
    4830589300230345216, 4788883862115630208, 4182243405416775296,   # C3
    4860363422275862016, 4830611015584887936, 4830564114541936384,   # C3
    4829428593908103168,                                             # C3
]
# Twelfth batch (14 screenshots, 2026-08-04) — all Q1/ERO Cluster 3 (1 ERO); tile #12 was
# submitted twice, so 13 unique, all new and position-verified against gaia_dr3.gaia_source (<0.2").
NEW_PICKS_11 = [
    1779023866876825088, 1440879270594404608, 1634489799228479616,   # C3 (1 ERO)
    5056919443776166016, 4831101122893652992, 4838001730885201792,   # C3
    2257628668421699712, 1440781414059523584, 1634526357990302336,   # C3
    2257889080878722560, 4830207151220405504, 4830017760342061440,   # C3
    4831804814631210880,                                             # C3
]
# Thirteenth batch (24 screenshots, 2026-08-04) — all Q1/ERO Cluster 3 (2 ERO); 1 was a dup
# (already in MULTI_GAIA), these 23 are new, all position-verified against gaia_dr3.gaia_source (<0.45").
NEW_PICKS_12 = [
    2257546037548578048, 4830786903085952384, 2257382206018473088,   # C3
    4829645305074598016, 2258126369232468480, 5056985723711327744,   # C3
    4789000311564942720, 4829997522456034944, 4828819567546361856,   # C3
    2161022518789160832, 4365640403041389440, 2160992728895886336,   # C3 (1 ERO)
    4837070788131799040, 2161601308582447360, 5060359300266023680,   # C3
    5057278365603113984, 2258403514881622912, 428977382156390784,    # C3 (1 ERO)
    2161117351667197440, 1633132211606296320, 2257712579201034240,   # C3
    1633188600234177024, 2161699100692606464,                        # C3
]
# Fourteenth batch (4 screenshots, 2026-08-04) — all Q1/ERO Cluster 3 (1 ERO); all 4 new,
# position-verified against gaia_dr3.gaia_source (<0.2").
NEW_PICKS_13 = [
    4832926930670483456, 239375363679653632, 4832942392552942080,   # C3 (1 ERO)
    1633192375508356736,                                            # C3
]
# Sources removed on review (2026-07-31): rejected on inspection of their cutouts.
REMOVE = {
    6438777823548176128,   # J190655.20-634046.6
    5057350452333817344,   # J032836.50-284155.9
    4830604663329901568,   # J040356.55-491207.7
    4974123297627607936,   # J002424.85-490636.7
    4833094777992052224,   # J034853.20-482339.9
    3219147845996327168,   # J054619.06+000329.5
    3219223299980727168,   # J054631.05+002533.7
}
PICKED = [g for g in dict.fromkeys(
              PICKED + MULTI_GAIA + NEW_PICKS + NEW_PICKS_2 + NEW_PICKS_3 + NEW_PICKS_4
              + NEW_PICKS_5 + NEW_PICKS_6 + NEW_PICKS_7 + NEW_PICKS_8 + NEW_PICKS_9
              + NEW_PICKS_10 + NEW_PICKS_11 + NEW_PICKS_12 + NEW_PICKS_13)
          if g not in REMOVE]  # deduped union, minus removals

CSV    = PROJECT_DIR / "output" / "umap" / "qso" / "arrays" / "catwise_euclid_gaia.csv"
CS_DIR = PROJECT_DIR / "output" / "picked_qso" / "cutouts"                        # all contact sheets
OUT    = CS_DIR / "picked_qso_contact_sheet.png"            # new UMAP clusters (prefix for per-cluster)
OUT_Q1 = CS_DIR / "og_picked_qso_contact_sheet.png"         # original Q1 clusters (single sheet)


def load_picked() -> pd.DataFrame:
    flat = pd.read_csv(CSV, dtype={"gaia_source_id": "int64"})
    sub = flat[flat["gaia_source_id"].isin(PICKED)].copy()
    sub = sub.rename(columns={"qso_ra": "qso_ra", "qso_dec": "qso_dec"})
    sub["sid_str"] = sub["gaia_source_id"].astype("int64").astype(str)
    missing = set(PICKED) - set(sub["gaia_source_id"])
    if missing:
        print(f"  WARNING: {len(missing)} picked ids not in flat CSV: {missing}")
    print(f"  loaded {len(sub)} / {len(PICKED)} picked sources")
    return sub


def fetch_picked_stamps(user: str, samp=None) -> dict:
    """Fetch each pick's Euclid tile + companion markers ONCE (the network-heavy part) and
    return {sid_str: (row, arr, blend, model_pgal, q1cluster, euclid_ok)} WITHOUT baking the
    title — so callers can bake with whatever cluster label they want (Q1 vs new UMAP) reusing
    the same fetched tile."""
    if samp is None:
        samp = load_picked()
    blend_info = fetch_blend_info(samp, user)
    gaia       = fetch_gaia_positions(samp, SIZE, blend_info)
    euclid_sec = fetch_euclid_secondary_positions(samp, blend_info, user)
    out: dict = {}
    for _, row in samp.iterrows():
        sid = row["sid_str"]
        b   = blend_info.get(sid, {})
        e_ra  = b.get("e_ra")  or float(row["qso_ra"])
        e_dec = b.get("e_dec") or float(row["qso_dec"])
        sec_euclid = [(ra, dec, lbl, EUCLID_RING, "crosshair")
                      for ra, dec, lbl, *_ in euclid_sec.get(sid, [])]
        companions = (list(gaia.get(sid, []))
                      + sec_euclid
                      + [(float(row["qso_ra"]), float(row["qso_dec"]), None, WISE_RING, "x")])
        arr   = fetch_euclid(row, SIZE, companions=companions,
                             center_ra=e_ra, center_dec=e_dec)
        c = int(row["cluster"])
        out[sid] = (row, arr, b, float(row["model_pgal"]), c, arr is not None)
        print(f"    C{c}  pgal={float(row['model_pgal']):.2f}  {b.get('wise_id','')}  euclid_ok={arr is not None}")
    return out


def render_picked_arrays(user: str, samp=None) -> dict:
    """Render EVERY pick's Euclid+WISE stamp fresh (same pipeline as the contact sheet) and
    return {sid_str: (baked_array, model_pgal, new_cluster, euclid_ok)}. Used by
    compile_picked_qso_umap_web.py; the title carries the NEW combined QSO×Gaia UMAP cluster
    label ('QSO_Cluster {nc}') to match picked_qso_contact_sheet.png."""
    if samp is None:
        samp = load_picked()
    new_of = assign_new_clusters(samp)                       # {sid: new QSO-UMAP cluster}
    out = {}
    for sid, (row, arr, b, pgal, c, ok) in fetch_picked_stamps(user, samp).items():
        nc = new_of.get(sid, c)
        out[sid] = (render_to_array(row, arr, b, cluster_label=f"QSO_Cluster {nc}"), pgal, nc, ok)
    return out


def _load_reducer(path):
    """Load a saved UMAP reducer, patching pynndescent attrs missing across versions."""
    import pynndescent
    for attr, default in [("quantization", None), ("_quantized_data", None),
                          ("parallel_batch_queries", False)]:
        if not hasattr(pynndescent.NNDescent, attr):
            setattr(pynndescent.NNDescent, attr, default)
    return joblib.load(path)


def assign_new_clusters(samp: pd.DataFrame) -> dict:
    """Project the picks onto the NEW combined QSO×Gaia UMAP (n_all) and return
    {sid_str: cluster} via nearest-neighbour on the build embedding — exactly the
    projection analyze_picked_qso_umap.py does, kept LOCAL here (14 features straight
    from the flat CSV + the build's train mean/std + saved reducer) so this file stays
    the single source of truth for the pick list with no circular import."""
    from sklearn.neighbors import NearestNeighbors
    d = np.load(NEW_UMAP_NPZ, allow_pickle=True)
    emb, labels = d["embedding"], d["cluster"]
    mean, std = d["mean"], d["std"]
    std_safe = np.where(std > 0, std, 1.0)
    X  = samp[list(SELECTED_GAIA_PARAMETERS)].apply(pd.to_numeric, errors="coerce").values.astype(float)
    Xs = np.where(np.isfinite((X - mean) / std_safe), (X - mean) / std_safe, 0.0)
    pemb = _load_reducer(NEW_UMAP_RED).transform(Xs).astype(float)
    valid = labels >= 0
    _, idx = NearestNeighbors(n_neighbors=1).fit(emb[valid]).kneighbors(pemb)
    newc = labels[np.where(valid)[0][idx.ravel()]]
    return dict(zip(samp["sid_str"].values, [int(c) for c in newc]))


def render_contact_sheet(baked_by_cluster: dict, out, subtitle: str) -> None:
    """Lay out {cluster: [(baked_array, pgal), ...]} into one contact sheet: rows grouped by
    cluster, sorted within cluster by descending P(gal), wrapping at NCOLS, with rotated cluster
    labels on the left and a P(gal) colour-bar + marker legend on top."""
    # Sort within cluster by descending P(gal); rows-per-cluster = ceil(n / NCOLS)
    clusters = sorted(baked_by_cluster)
    for c in clusters:
        baked_by_cluster[c].sort(key=lambda t: -t[1])
    rows_per = {c: int(np.ceil(len(baked_by_cluster[c]) / NCOLS)) for c in clusters}
    n_total  = sum(len(v) for v in baked_by_cluster.values())

    # ---- explicit inch-based layout so cluster groups get real gaps between them ----
    from matplotlib.lines import Line2D
    S      = 3.0    # stamp cell size (in)
    HGAP   = 0.10   # gap between columns
    VGAP   = 0.10   # gap between rows within one cluster
    CGAP   = 0.60   # BIG gap between cluster groups
    L_IN   = 0.85   # left margin for cluster labels
    R_IN   = 0.10
    HEADER = 1.85   # top space: title on line 1, colour-bar + marker legend on line 2
    BOTTOM = 0.20

    content_h = (sum(rows_per[c] * S + (rows_per[c] - 1) * VGAP for c in clusters)
                 + CGAP * (len(clusters) - 1))
    fig_w = L_IN + NCOLS * S + (NCOLS - 1) * HGAP + R_IN
    fig_h = HEADER + content_h + BOTTOM
    fig   = plt.figure(figsize=(fig_w, fig_h))

    top_edge = fig_h - HEADER          # top of content region (in, from bottom)
    off      = 0.0                     # inches consumed from the top, per cluster
    label_y, gap_centres = {}, []
    for i, c in enumerate(clusters):
        cluster_top = top_edge - off
        label_y[c]  = (cluster_top - S / 2) / fig_h      # centre of this cluster's TOP row
        for k, (arr, pg) in enumerate(baked_by_cluster[c]):
            r, col   = divmod(k, NCOLS)
            cell_top = cluster_top - r * (S + VGAP)
            left     = (L_IN + col * (S + HGAP)) / fig_w
            ax = fig.add_axes([left, (cell_top - S) / fig_h, S / fig_w, S / fig_h])
            ax.imshow(np.clip(arr, 0, 1)); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(pgal_color(pg)); sp.set_linewidth(2.8)
        off += rows_per[c] * S + (rows_per[c] - 1) * VGAP
        if i < len(clusters) - 1:
            gap_centres.append((top_edge - off - CGAP / 2) / fig_h)
        off += CGAP

    # Cluster labels aligned with the TOP row of each group
    for c in clusters:
        fig.text(0.013, label_y[c], f"Cluster {c}", rotation=90, ha="center", va="center",
                 fontsize=16, fontweight="bold", color=CLUSTER_COLOURS[c])

    # Dashed separators centred in each (now wide) inter-cluster gap
    for y_sep in gap_centres:
        fig.add_artist(Line2D([0.028, 0.997], [y_sep, y_sep], transform=fig.transFigure,
                              color="0.5", lw=1.1, ls="--"))

    # P(gal) colour-bar legend (top-left)
    cbar_ax = fig.add_axes([0.06, (fig_h - 1.30) / fig_h, 0.17, 0.14 / fig_h])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=plt.Normalize(0, 1)),
                        cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 0.5, 1]); cbar.set_ticklabels(["star", "0.5", "galaxy"])
    cbar.ax.tick_params(labelsize=8, pad=1)
    cbar_ax.set_title("border = XGBoost P(gal)", fontsize=9, pad=3, loc="left")

    # Marker legend (same as the per-cluster download sheets)
    legend_lines = [
        Line2D([0], [0], color=WISE_RING,     lw=0, marker="x", ms=10, mew=1.6, label="WISE centroid"),
        Line2D([0], [0], color=EUCLID_RING,   lw=0, marker="+", ms=10, mew=1.6, label="Secondary Euclid (<1\")"),
        Line2D([0], [0], color=GAIA_RING,     lw=0, marker="o", ms=9, markerfacecolor="none", mew=1.3, label="Gaia matched (<1\")"),
        Line2D([0], [0], color=GAIA_FOV_RING, lw=0, marker="o", ms=9, markerfacecolor="none", mew=1.3, label="Gaia in FOV (>1\")"),
    ]
    fig.legend(handles=legend_lines, loc="upper right",
               bbox_to_anchor=(0.995, 1 - 0.95 / fig_h), fontsize=10,
               framealpha=0.9, ncol=4)

    fig.suptitle(f"Hand-picked QSO Euclid ⨯ CatWISE ⨯ Gaia cutouts ({n_total}) — {subtitle}",
                 fontsize=17, fontweight="bold", x=0.52, y=1 - 0.34 / fig_h)

    out = out if hasattr(out, "parent") else PROJECT_DIR / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> {out}  ({n_total} stamps)")


def render_cluster_sheets(baked_by_cluster: dict, out_prefix, subtitle: str) -> None:
    """Write ONE contact sheet PER cluster: {out_prefix}_cluster{c}.png (same spirit as the
    per-cluster download sheets in download_qso_catwise_e_g_cutouts.py). Each is a NCOLS-wide grid
    of that cluster's stamps sorted by descending P(gal), with the P(gal) colour-bar + marker
    legend on top and a cluster-coloured title."""
    from matplotlib.lines import Line2D
    out_prefix = out_prefix if hasattr(out_prefix, "parent") else PROJECT_DIR / out_prefix
    S = 3.0; HGAP = 0.10; VGAP = 0.10; L_IN = 0.15; R_IN = 0.10
    HEADER = 1.85; BOTTOM = 0.20
    for c in sorted(baked_by_cluster):
        items = sorted(baked_by_cluster[c], key=lambda t: -t[1])
        n = len(items)
        if n == 0:
            continue
        nrows = int(np.ceil(n / NCOLS))
        content_h = nrows * S + (nrows - 1) * VGAP
        fig_w = L_IN + NCOLS * S + (NCOLS - 1) * HGAP + R_IN
        fig_h = HEADER + content_h + BOTTOM
        fig   = plt.figure(figsize=(fig_w, fig_h))
        top_edge = fig_h - HEADER
        for k, (arr, pg) in enumerate(items):
            r, col   = divmod(k, NCOLS)
            cell_top = top_edge - r * (S + VGAP)
            left     = (L_IN + col * (S + HGAP)) / fig_w
            ax = fig.add_axes([left, (cell_top - S) / fig_h, S / fig_w, S / fig_h])
            ax.imshow(np.clip(arr, 0, 1)); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(pgal_color(pg)); sp.set_linewidth(2.8)

        # P(gal) colour-bar legend (top-left)
        cbar_ax = fig.add_axes([0.06, (fig_h - 1.30) / fig_h, 0.17, 0.14 / fig_h])
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=plt.Normalize(0, 1)),
                            cax=cbar_ax, orientation="horizontal")
        cbar.set_ticks([0, 0.5, 1]); cbar.set_ticklabels(["star", "0.5", "galaxy"])
        cbar.ax.tick_params(labelsize=8, pad=1)
        cbar_ax.set_title("border = XGBoost P(gal)", fontsize=9, pad=3, loc="left")

        legend_lines = [
            Line2D([0], [0], color=WISE_RING,     lw=0, marker="x", ms=10, mew=1.6, label="WISE centroid"),
            Line2D([0], [0], color=EUCLID_RING,   lw=0, marker="+", ms=10, mew=1.6, label="Secondary Euclid (<1\")"),
            Line2D([0], [0], color=GAIA_RING,     lw=0, marker="o", ms=9, markerfacecolor="none", mew=1.3, label="Gaia matched (<1\")"),
            Line2D([0], [0], color=GAIA_FOV_RING, lw=0, marker="o", ms=9, markerfacecolor="none", mew=1.3, label="Gaia in FOV (>1\")"),
        ]
        fig.legend(handles=legend_lines, loc="upper right",
                   bbox_to_anchor=(0.995, 1 - 0.95 / fig_h), fontsize=10, framealpha=0.9, ncol=4)
        fig.suptitle(f"Hand-picked QSO Euclid ⨯ CatWISE ⨯ Gaia — Cluster {c} ({n}) — {subtitle}",
                     fontsize=17, fontweight="bold", color=CLUSTER_COLOURS.get(c, "k"),
                     x=0.52, y=1 - 0.34 / fig_h)

        out = out_prefix.parent / f"{out_prefix.name}_cluster{c}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"  -> {out}  ({n} stamps)")


def main() -> int:
    user = os.environ.get("PGUSER", "siwei_zhao_2025")
    samp = load_picked()

    # Fetch each Euclid tile once (network); bake per-sheet with the sheet's own cluster label
    stamps = fetch_picked_stamps(user, samp)                 # {sid: (row, arr, blend, pgal, q1c, ok)}
    new_of = assign_new_clusters(samp)                       # {sid: new-UMAP cluster} via projection

    # Sheet 1 — grouped/labelled by the ORIGINAL Q1 UMAP cluster (carried in the flat CSV)
    q1_by: dict[int, list] = {}
    for sid, (row, arr, b, pgal, c, _ok) in stamps.items():
        baked = render_to_array(row, arr, b, cluster_label=f"Q1_Cluster {c}")
        q1_by.setdefault(c, []).append((baked, pgal))
    render_contact_sheet(q1_by, OUT_Q1, "rows = original Q1 UMAP cluster, sorted by P(gal)")

    # Sheet 2 — grouped/labelled by the NEW combined QSO×Gaia UMAP cluster (projected here)
    new_by: dict[int, list] = {}
    for sid, (row, arr, b, pgal, _c, _ok) in stamps.items():
        nc = new_of[sid]
        baked = render_to_array(row, arr, b, cluster_label=f"QSO_Cluster {nc}")
        new_by.setdefault(nc, []).append((baked, pgal))
    # one contact sheet PER new QSO×Gaia UMAP cluster, in the cutouts/ subfolder:
    #   output/picked_qso/cutouts/picked_qso_contact_sheet_cluster{c}.png
    render_cluster_sheets(new_by, OUT.with_suffix(""), "sorted by P(gal)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
