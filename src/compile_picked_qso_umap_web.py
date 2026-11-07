#!/usr/bin/env python3
"""
Interactive 3D web explorer for the hand-picked QSOs on the NEW QSO×Gaia UMAP
(output/qso_umap/ds_qso/n_all combined build) — same spirit as compile_umap_combined.py
but focused on the picks.

Layout (single self-contained HTML): LEFT = 3D UMAP (Plotly Scatter3d) with the full QSO×Gaia
population as a faint flat background cloud (z=0) and the HAND-PICKED QSOs projected as the
elevated "middle" layer (z = a chosen metric); RIGHT = a sidebar that, on hover/click, shows
that pick's Euclid+WISE cutout, its metadata, a link + embedded panel to the CDS Portal for the
source's coordinates, and a gallery of the analyze_picked_qso_umap.py summary plots.

Interactions:
  * hover a pick  -> its individual cutout + metadata in the sidebar
  * double-click  -> pin that pick (hover stops updating); button to unpin
  * "CDS Portal"  -> opens https://cdsportal.cds.unistra.fr/?target=<ra> <dec> for the pinned
                     source (new tab, always works) + a best-effort inline iframe (CDS may block
                     framing via X-Frame-Options, in which case use the link)
  * Z axis select -> P(gal) / XGB photo-z / colour(z) distance / G magnitude (picks re-elevate)
  * "Info plots"  -> opens a standalone gallery of the summary PNGs in a NEW browser tab (contact sheet, spec-z/colour dist/XGB dz/
                     Q1-ERO/density/params_solved, UMAP projection, feature grid + distributions,
                     clusters)

Reads (WSDB; VPN; ~/.pgpass) + local:
  output/qso_umap/base/arrays/combined_qso_gaia.{npz,_reducer.joblib}
  output/umap/qso/arrays/catwise_euclid_gaia.csv       (pick metadata: RA/Dec/WISE/P(gal)/G/aps)
  output/qso_cutouts/catwise_e_g/manifest.csv + PNGs   (per-pick cutout thumbnails)
  output/picked_qso/*.png                              (summary plots to embed)
Writes:
  output/picked_qso/picked_qso_umap_web.html

Usage:
  PGUSER=siwei_zhao_2025 python src/compile_picked_qso_umap_web.py
  PGUSER=siwei_zhao_2025 python src/compile_picked_qso_umap_web.py --sample combined --n-bg 45000
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Plotly 6 emits binary typed-arrays for numpy inputs, which break Scatter3d in this setup.
# Force plain JSON lists at trace-creation time (same patch as compile_umap_combined.py).
import plotly.graph_objects as _go
_orig_s3 = _go.Scatter3d.__init__


def _to_list(v):
    return v.tolist() if isinstance(v, np.ndarray) else v


def _patched(self, **kw):
    for k in ("x", "y", "z"):
        if k in kw:
            kw[k] = _to_list(kw[k])
    _orig_s3(self, **kw)


_go.Scatter3d.__init__ = _patched

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_picked_qso_contact_sheet import PICKED                     # noqa: E402
from analyze_picked_qso_umap import fetch_features, _load_reducer        # noqa: E402
from selected_parameters_list import SELECTED_GAIA_PARAMETERS            # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
DS_DIR      = PROJECT_DIR / "output" / "qso_umap" / "ds_qso"
BASE_ARRAYS = PROJECT_DIR / "output" / "qso_umap" / "base" / "arrays"  # curated combined-only bundle
FLAT_CSV    = PROJECT_DIR / "output" / "umap" / "qso" / "arrays" / "catwise_euclid_gaia.csv"
PILE_DIR    = PROJECT_DIR / "output" / "qso_cutouts" / "catwise_e_g"
PLOTS_DIR   = PROJECT_DIR / "output" / "picked_qso"
OUT_HTML    = PLOTS_DIR / "picked_qso_umap_web.html"

CLUSTER_COLOURS = {0: "#e6194b", 1: "#4363d8", 2: "#f58231", 3: "#3cb44b",
                   4: "#911eb4", -1: "#888888"}

# summary plots to embed in the "Info plots" gallery (title -> filename), in display order
SUMMARY_PLOTS = [
    ("contact sheet — rows = NEW combined QSO×Gaia UMAP cluster", "picked_qso_contact_sheet.png"),
    ("contact sheet — rows = original Q1 UMAP cluster", "picked_qso_contact_sheet_og.png"),
    ("HDBSCAN clusters", "qso_gaia_clusters.png"),
    ("14 Gaia features (population)", "qso_gaia_features.png"),
    ("14 Gaia features + picks over-plotted", "qso_gaia_features_picked.png"),
    ("14 Gaia-feature distributions (population by HDBSCAN cluster)", "qso_gaia_features_hist.png"),
    ("14 Gaia-feature distributions (population + picks)", "qso_gaia_features_hist_picked.png"),
    ("colour(z) / photo-z / Q1-ERO / density / params_solved", "qso_other_features.png"),
]


def _b64(path: Path) -> str | None:
    try:
        return base64.b64encode(path.read_bytes()).decode()
    except Exception:
        return None


def _arr_to_b64(arr, maxw: int = 420) -> str:
    """Encode a float RGB(A) stamp array ([0,1]) to a base64 PNG, downscaled to <=maxw wide."""
    import io
    from PIL import Image
    a = (np.clip(np.asarray(arr, dtype=float), 0, 1) * 255).astype("uint8")
    mode = "RGBA" if (a.ndim == 3 and a.shape[2] == 4) else None
    im = Image.fromarray(a, mode) if mode else Image.fromarray(a)
    if im.width > maxw:
        im = im.resize((maxw, int(im.height * maxw / im.width)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def render_pick_cutouts(user: str, samp=None) -> dict:
    """{gaia_source_id_str: base64 png} by rendering EVERY pick fresh — the exact stamps from
    analyze_picked_qso_contact_sheet.py (not the stale catwise_e_g pile, which only had 60/78)."""
    from analyze_picked_qso_contact_sheet import render_picked_arrays
    arrays = render_picked_arrays(user, samp)      # {sid_str: (baked, pgal, cluster, euclid_ok)}
    cache = {sid: _arr_to_b64(baked) for sid, (baked, _pg, _c, _ok) in arrays.items()}
    print(f"  rendered {len(cache)} pick cutouts fresh (all picks)")
    return cache


# colour(z) river-deviation cache (full Q1 candidate colour(z), from the archived
# analyze_picked_color_vs_z.py pipeline)
POP_CACHE = PROJECT_DIR / "output" / "qso_color_vs_z" / "arrays" / "catwise_euclid_gaia_xgbz.csv"


def predict_picked(user):
    """z_xgb + 2FWHM Euclid colours for the picked Q1 sources (XGB photo-z ensemble on
    mer_final_cat apertures). Folded in from the archived analyze_picked_color_vs_z.py — this web
    compiler is the only remaining consumer. Returns (df, idmap)."""
    import json
    import psycopg2
    import xgboost as xgb
    from analyze_qso_color_vs_specz import COLORS, ZPC_VIS, _flux_to_mag, WSDB_HOST, WSDB_DB
    from train_xgboost_qso_photz import add_colours
    xgb_dir = PROJECT_DIR / "output" / "xgboost_qso_photz"
    models_dir = xgb_dir / "models"
    band_db = {"IE": "vis", "YE": "y", "JE": "j", "HE": "h"}
    catwise_eg = "crossmatch_qso_catwise_euclid_gaia"; mer_table = "euclid_q1.mer_final_cat"
    feats  = json.loads((models_dir / "features.json").read_text())["features"]
    target = json.loads((xgb_dir / "metrics.json").read_text())["target"]

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30)
    conn.cursor().execute("SET statement_timeout = 300000")
    idmap = pd.read_sql_query(
        f"""SELECT g_source_id, e_id AS object_id, qso_id, euclid_source,
                   g_pm_significance AS pm_sig
            FROM {user}.{catwise_eg}
            WHERE g_source_id = ANY(%(g)s) AND e_id IS NOT NULL""",
        conn, params={"g": [int(x) for x in PICKED]})
    q1 = idmap[idmap["euclid_source"] == "Q1"].drop_duplicates("object_id")
    print(f"  picked: {len(idmap)} matched  ({(idmap.euclid_source=='Q1').sum()} Q1, "
          f"{(idmap.euclid_source=='ERO').sum()} ERO)")

    fcols = [f"flux_{s}_{a}fwhm_aper" for s in band_db.values() for a in (1, 2, 3, 4)]
    ecols = [f"fluxerr_{s}_{a}fwhm_aper" for s in band_db.values() for a in (1, 2, 3, 4)]
    mer = pd.read_sql_query(
        f"SELECT object_id, {', '.join(fcols + ecols)} FROM {mer_table} "
        f"WHERE object_id = ANY(%(b)s)", conn, params={"b": q1["object_id"].tolist()})
    conn.close()

    df = q1.merge(mer, on="object_id", how="inner").drop_duplicates("object_id").reset_index(drop=True)
    for band, s in band_db.items():
        zpc = ZPC_VIS if s == "vis" else 1.0
        for a in (1, 2, 3, 4):
            df[f"{band}_{a}"] = _flux_to_mag(df[f"flux_{s}_{a}fwhm_aper"],
                                             df[f"fluxerr_{s}_{a}fwhm_aper"], zpc)
    df = add_colours(df)
    ok = df.dropna(subset=feats).reset_index(drop=True)
    X = ok[feats].to_numpy(float)
    preds = []
    for p in sorted(models_dir.glob("xgb_photz_seed*.json")):
        m = xgb.XGBRegressor(); m.load_model(str(p))
        pr = m.predict(X)
        preds.append(np.expm1(pr) if target == "log1p_z" else pr)
    ok["z_xgb"] = np.mean(preds, axis=0)
    for _, b, r in COLORS:
        ok[f"{b}_{r}"] = ok[f"{b}_2"] - ok[f"{r}_2"]
    print(f"  {len(ok)}/{len(q1)} Q1 picks have all 16 bands -> z_xgb assigned")
    return ok[["object_id", "qso_id", "z_xgb", "pm_sig"] + [f"{b}_{r}" for _, b, r in COLORS]], idmap


def river_deviation(user):
    """Per picked Q1 source: RMS colour distance from the CANDIDATE median colour(z) locus,
    evaluated at the source's own XGB photo-z. Returns {qso_id(WISE) -> deviation in mag}.
    Folded in from analyze_picked_qso_umap.py (its picked-projection plots were retired)."""
    from analyze_qso_color_vs_specz import COLORS, _running_stats, ZMAX_PLOT
    picked, _ = predict_picked(user)
    pop = pd.read_csv(POP_CACHE); zp = pop["z_xgb"].to_numpy(float)
    med = {}
    for _, b, r in COLORS:
        col = f"{b}_{r}"; c = pop[col].to_numpy(float)
        m = np.isfinite(c) & np.isfinite(zp) & (zp <= ZMAX_PLOT)
        mid, mmed, _, _ = _running_stats(zp[m], c[m])
        med[col] = (np.asarray(mid), np.asarray(mmed))
    dev_of = {}
    for _, s in picked.iterrows():
        z = s["z_xgb"]
        if not np.isfinite(z):
            continue
        sq, nok = 0.0, 0
        for _, b, r in COLORS:
            col = f"{b}_{r}"; cval = s[col]
            if not np.isfinite(cval):
                continue
            sq += (cval - np.interp(z, med[col][0], med[col][1])) ** 2; nok += 1
        if nok:
            dev_of[str(s["qso_id"])] = float(np.sqrt(sq / nok))
    print(f"  colour(z) river deviation computed for {len(dev_of)} picked Q1 sources")
    return dev_of


def project_picks(sample: str, user: str):
    """Project the PICKED gaia sources onto the {sample} n_all UMAP; return (bg dict, picks df)."""
    # combined build lives in the curated base/ bundle; desi/sdss stay under ds_qso/n_all
    adir = BASE_ARRAYS if sample == "combined" else DS_DIR / "n_all" / "arrays"
    npz = adir / f"{sample}_qso_gaia.npz"
    red = adir / f"{sample}_qso_gaia_reducer.joblib"
    if not (npz.exists() and red.exists()):
        raise SystemExit(f"  missing n_all arrays for {sample} — build with query_qso_umap_ds.py "
                         f"then query_qso_umap_base.py")
    d = np.load(npz, allow_pickle=True)
    emb = d["embedding"]; labels = d["cluster"]; mean, std = d["mean"], d["std"]
    std_safe = np.where(std > 0, std, 1.0)

    # pick metadata from the flat catwise_euclid_gaia table
    flat = pd.read_csv(FLAT_CSV, dtype={"gaia_source_id": "int64"})
    picks = flat[flat["gaia_source_id"].isin([int(g) for g in PICKED])].copy()
    picks = picks.drop_duplicates("gaia_source_id").reset_index(drop=True)
    print(f"  {len(picks)} picks found in the flat metadata table")

    # the flat catwise_euclid_gaia table already carries the 14 features (bare names) ->
    # standardise with the build's train mean/std -> reducer.transform
    X = picks[list(SELECTED_GAIA_PARAMETERS)].apply(pd.to_numeric, errors="coerce").values.astype(float)
    Xs = np.where(np.isfinite((X - mean) / std_safe), (X - mean) / std_safe, 0.0)
    pemb = _load_reducer(red).transform(Xs).astype(float)
    picks["umap1"], picks["umap2"] = pemb[:, 0], pemb[:, 1]
    valid = labels >= 0
    _, idx = NearestNeighbors(n_neighbors=1).fit(emb[valid]).kneighbors(pemb)
    picks["ncluster"] = labels[np.where(valid)[0][idx.ravel()]]

    # optional: XGB photo-z + colour(z) river deviation per pick (keyed by WISE qso_id)
    picks["z_xgb"] = np.nan; picks["dev"] = np.nan
    try:
        pk, _ = predict_picked(user)
        zx = dict(zip(pk["qso_id"].astype(str), pk["z_xgb"]))
        picks["z_xgb"] = picks["qso_id"].astype(str).map(zx)
    except Exception as exc:
        print(f"  (skipping XGB photo-z per pick: {exc})")
    try:
        dev = river_deviation(user)
        picks["dev"] = picks["qso_id"].astype(str).map(dev)
    except Exception as exc:
        print(f"  (skipping colour(z) deviation per pick: {exc})")

    bg = {"u1": emb[:, 0], "u2": emb[:, 1], "cl": labels}
    return bg, picks, len(emb)


def build_figure(bg, picks, n_bg):
    import plotly.graph_objects as go
    rng = np.random.RandomState(42)
    n = len(bg["u1"])
    sel = np.arange(n) if n <= n_bg else rng.choice(n, n_bg, replace=False)
    bx, by, bcl = bg["u1"][sel], bg["u2"][sel], bg["cl"][sel]

    traces = []
    # faint flat background cloud, per cluster (z=0) — toggleable in the legend
    for c in sorted(set(int(x) for x in bcl)):
        m = bcl == c
        traces.append(go.Scatter3d(
            x=bx[m], y=by[m], z=np.zeros(int(m.sum())), mode="markers",
            marker=dict(size=1.4, color=CLUSTER_COLOURS.get(c, "#888"), opacity=0.16,
                        line=dict(width=0)),
            name=f"C{c} · all QSOs", hoverinfo="skip", showlegend=True))

    # picks — elevated layer, coloured by cluster, one trace (per-point colour)
    pgal = pd.to_numeric(picks["model_pgal"], errors="coerce").to_numpy(float)
    colors = [CLUSTER_COLOURS.get(int(c), "#fff") for c in picks["ncluster"]]
    cd = []
    for _, r in picks.iterrows():
        cd.append([
            str(int(r["gaia_source_id"])),
            str(r.get("qso_id", "")),
            None if pd.isna(r["qso_ra"]) else round(float(r["qso_ra"]), 5),
            None if pd.isna(r["qso_dec"]) else round(float(r["qso_dec"]), 5),
            None if pd.isna(r.get("phot_g_mean_mag")) else round(float(r["phot_g_mean_mag"]), 2),
            None if pd.isna(r.get("model_pgal")) else round(float(r["model_pgal"]), 3),
            int(r["ncluster"]),
            None if pd.isna(r.get("z_xgb")) else round(float(r["z_xgb"]), 3),
            None if pd.isna(r.get("dev")) else round(float(r["dev"]), 3),
            None if pd.isna(r.get("astrometric_params_solved")) else int(r["astrometric_params_solved"]),
        ])
    z0 = np.nan_to_num(pd.to_numeric(picks["model_pgal"], errors="coerce").to_numpy(float), nan=0.0)
    traces.append(go.Scatter3d(
        x=picks["umap1"].to_numpy(float), y=picks["umap2"].to_numpy(float), z=z0,
        mode="markers",
        marker=dict(size=5, color=colors, opacity=0.95, line=dict(width=0.5, color="#111")),
        name=f"picked QSOs ({len(picks)})", customdata=cd,
        hovertemplate="%{customdata[1]}<br>RA=%{customdata[2]}  Dec=%{customdata[3]}<extra>C%{customdata[6]}</extra>",
        showlegend=True))
    pick_trace_index = len(traces) - 1

    # highlight ring — empty until a contact-sheet cutout is clicked, then moved onto that pick
    traces.append(go.Scatter3d(
        x=[], y=[], z=[], mode="markers",
        marker=dict(size=18, symbol="circle-open", color="#ff2fd6", line=dict(width=4, color="#ff2fd6")),
        name="highlight", hoverinfo="skip", showlegend=False))
    hl_trace_index = len(traces) - 1

    xr = [float(bg["u1"].min()) - 0.8, float(bg["u1"].max()) + 0.8]
    yr = [float(bg["u2"].min()) - 0.8, float(bg["u2"].max()) + 0.8]
    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor="#080808", plot_bgcolor="#080808",
        font=dict(color="#aaa", size=11),
        scene=dict(
            bgcolor="#080808", aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.6),
            domain=dict(x=[0.18, 1.0], y=[0.0, 1.0]),   # shift the 3D scene right so the cube centres
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9), up=dict(x=0, y=0, z=1)),
            xaxis=dict(title="UMAP 1", range=xr, autorange=False, showspikes=False,
                       showbackground=True, backgroundcolor="#0a0a0a", gridcolor="#1d1d1d",
                       zerolinecolor="#2a2a2a", color="#8a8a8a"),
            yaxis=dict(title="UMAP 2", range=yr, autorange=False, showspikes=False,
                       showbackground=True, backgroundcolor="#0a0a0a", gridcolor="#1d1d1d",
                       zerolinecolor="#2a2a2a", color="#8a8a8a"),
            zaxis=dict(title="P(gal)", range=[-0.02, 1.05], autorange=False, showspikes=False,
                       showbackground=True, backgroundcolor="#0a0a0a", gridcolor="#1d1d1d",
                       zerolinecolor="#2a2a2a", color="#8a8a8a")),
        legend=dict(bgcolor="rgba(12,12,12,0.85)", bordercolor="#222", borderwidth=1,
                    itemsizing="constant"),
        margin=dict(l=0, r=0, b=0, t=0), height=780)

    # per-pick z arrays for the Z-axis selector (index = pick order == last trace)
    def _col(name, default=0.0):
        return [default if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
                for v in picks[name]]
    z_arrays = {
        "pgal":  _col("model_pgal"),
        "z_xgb": _col("z_xgb"),
        "dev":   _col("dev"),
        "gmag":  _col("phot_g_mean_mag"),
    }
    z_meta = {  # (axis title, [zmin, zmax])
        "pgal":  ("P(gal)", [-0.02, 1.05]),
        "z_xgb": ("XGB photo-z", [0, 3.6]),
        "dev":   ("colour(z) deviation (mag)", [0, 0.6]),
        "gmag":  ("G magnitude", [16, 22]),
    }

    # per-pick record for the clickable contact grid + highlight (sorted by cluster, P(gal))
    picks_js = []
    for i, (_, r) in enumerate(picks.iterrows()):
        picks_js.append({
            "sid":  str(int(r["gaia_source_id"])),
            "wise": str(r.get("qso_id", "")),
            "cl":   int(r["ncluster"]),
            "pgal": None if pd.isna(r.get("model_pgal")) else round(float(r["model_pgal"]), 3),
            "u1":   float(r["umap1"]), "u2": float(r["umap2"]),
            "ra":   None if pd.isna(r["qso_ra"]) else round(float(r["qso_ra"]), 5),
            "dec":  None if pd.isna(r["qso_dec"]) else round(float(r["qso_dec"]), 5),
            "g":    None if pd.isna(r.get("phot_g_mean_mag")) else round(float(r["phot_g_mean_mag"]), 2),
            "xgbz": None if pd.isna(r.get("z_xgb")) else round(float(r["z_xgb"]), 3),
            "dev":  None if pd.isna(r.get("dev")) else round(float(r["dev"]), 3),
            "aps":  None if pd.isna(r.get("astrometric_params_solved")) else int(r["astrometric_params_solved"]),
            "z":    {"pgal": z_arrays["pgal"][i], "z_xgb": z_arrays["z_xgb"][i],
                     "dev": z_arrays["dev"][i], "gmag": z_arrays["gmag"][i]},
        })
    return fig, z_arrays, z_meta, pick_trace_index, hl_trace_index, picks_js


HTML_TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Picked QSOs — 3D UMAP explorer</title>
<style>
 html,body{margin:0;height:100%;background:#080808;color:#cfcfcf;
   font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;}
 #wrap{position:absolute;inset:0;}
 #plot{position:absolute;left:0;right:520px;top:0;bottom:0;}
 #plot .plotly-graph-div{width:100%!important;height:100%!important;}
 #side{position:absolute;right:0;top:0;bottom:0;width:520px;background:#0c0c0c;
   border-left:1px solid #191919;display:flex;flex-direction:column;overflow-y:auto;z-index:5;}
 #side h1{font-size:14px;margin:12px 12px 2px;color:#eee;font-weight:600;}
 #hint{font-size:11px;color:#777;margin:0 12px 8px;}
 .ctrl{display:flex;align-items:center;gap:8px;padding:6px 12px;border-top:1px solid #161616;}
 .ctrl label{font-size:11px;color:#999;min-width:52px;}
 select,button{background:#151515;color:#ddd;border:1px solid #2a2a2a;border-radius:5px;
   padding:4px 7px;font-size:12px;cursor:pointer;}
 button:hover{background:#222;}
 #body{padding:10px 12px;}
 #detail{display:none;}
 #back{margin-bottom:8px;}
 #cut{width:100%;border-radius:6px;display:none;background:#000;}
 #nocut{font-size:11px;color:#666;font-style:italic;}
 #info{font-size:12px;line-height:1.5;white-space:pre-wrap;margin-top:8px;color:#cfcfcf;}
 #cds-row{margin-top:10px;display:none;}
 #cds-link{display:inline-block;background:#1f6feb;color:#fff;padding:5px 10px;border-radius:5px;
   text-decoration:none;font-size:12px;}
 #cds-link:hover{background:#388bfd;}
 #cds-frame{width:100%;height:320px;border:1px solid #222;border-radius:6px;margin-top:8px;
   background:#111;display:none;}
 #cds-note{font-size:10px;color:#666;margin-top:4px;}
 #grid{padding:2px 8px 10px;display:grid;grid-template-columns:repeat(5,1fr);gap:4px;align-items:start;}
 #grid .ghint{grid-column:1/-1;font-size:11px;color:#777;margin:2px 4px 6px;}
 .clhdr{grid-column:1/-1;font-size:12px;font-weight:600;margin:10px 4px 3px;color:#ddd;
   border-top:1px solid #191919;padding-top:7px;}
 .cell{cursor:pointer;text-align:center;border:2px solid transparent;border-radius:5px;}
 .cell:hover{border-color:#5a7cff;}
 .cell.sel{border-color:#ff2fd6;}
 .cell img{width:100%;border-radius:3px;display:block;}
 .cell .cap{font-size:8px;color:#aaa;line-height:1.15;margin-top:1px;word-break:break-all;}
 .cell .ph{width:100%;aspect-ratio:1/1;background:#151515;border-radius:3px;font-size:9px;color:#666;
   display:flex;align-items:center;justify-content:center;}
 .legend{font-size:11px;padding:8px 12px;border-top:1px solid #161616;}
 .lg{display:inline-flex;align-items:center;gap:5px;margin:2px 8px 2px 0;}
 .sw{width:10px;height:10px;border-radius:2px;display:inline-block;}
 #modal{position:fixed;inset:0;background:rgba(0,0,0,0.92);display:none;z-index:99;
   overflow-y:auto;padding:24px;}
 #modal .close{position:fixed;top:14px;right:22px;font-size:26px;color:#ccc;cursor:pointer;}
 #modal h2{font-size:13px;color:#9cf;margin:18px 0 6px;}
 #modal img{max-width:100%;border:1px solid #222;border-radius:6px;background:#000;}
</style></head><body>
<div id="wrap">
  <div id="plot">__PLOT_DIV__</div>
  <div id="side">
    <h1>Picked QSOs · 3D UMAP</h1>
    <div id="hint">hover a pick → cutout · click a cutout below → highlight it in 3D</div>
    <div class="ctrl"><label>Z axis</label>
      <select id="zsel">
        <option value="pgal" selected>XGB P(gal)</option>
        <option value="z_xgb">XGB photo-z</option>
        <option value="dev">colour(z) deviation</option>
      </select></div>
    <div class="ctrl"><label>Panels</label>
      <button id="pin">Pin</button>
      <button id="plotsbtn">Info plots</button></div>
    <div id="body">
      <div id="detail">
        <button id="back">← back to contact sheet</button>
        <img id="cut" src=""><div id="nocut"></div>
        <div id="info"></div>
        <div id="cds-row">
          <a id="cds-link" href="#" target="_blank" rel="noopener">Open in CDS Portal ↗</a>
          <div id="cds-note">CDS may block inline framing; use the link if the panel is blank.</div>
          <iframe id="cds-frame"></iframe>
        </div>
      </div>
      <div id="grid"></div>
    </div>
    <div class="legend" id="legend"></div>
  </div>
</div>
<script>
var CUTOUTS = __CUTOUTS_JSON__;
var ZARR    = __ZARR_JSON__;
var ZMETA   = __ZMETA_JSON__;
var PICK_TRACE = __PICK_TRACE__;
var HL_TRACE   = __HL_TRACE__;
var CLCOL   = __CLCOL_JSON__;
var PICKS   = __PICKS_JSON__;
var PMAP    = {}; PICKS.forEach(function(p){PMAP[p.sid]=p;});
var plotDiv = document.querySelector('.plotly-graph-div');
var detail=document.getElementById('detail'), grid=document.getElementById('grid'),
    img=document.getElementById('cut'), nocut=document.getElementById('nocut'),
    info=document.getElementById('info'), cdsRow=document.getElementById('cds-row'),
    cdsLink=document.getElementById('cds-link'), cdsFrame=document.getElementById('cds-frame');
var pinned=false, pinBtn=document.getElementById('pin');
var curZ='pgal', curSid=null;
var apsL={3:'2-param (no PM)',31:'5-param',95:'6-param'};

function cdsURL(ra,dec){return 'https://cdsportal.cds.unistra.fr/?target='+
    encodeURIComponent(Number(ra).toFixed(5)+' '+(dec>=0?'+':'')+Number(dec).toFixed(5));}

// render the detail card from a normalised object {sid,wise,ra,dec,g,pgal,cl,xgbz,dev,aps}
function showDetail(o, doFrame){
  detail.style.display='block'; grid.style.display='none';
  var b64=CUTOUTS[o.sid]||null;
  if(b64){img.src='data:image/png;base64,'+b64;img.style.display='block';nocut.textContent='';}
  else{img.style.display='none';nocut.textContent='(no cutout cached for this source)';}
  info.innerHTML='<b>'+(o.wise||o.sid)+'</b>\n'+
    'gaia '+o.sid+'\ncluster C'+o.cl+
    '\nRA, Dec = '+(o.ra!=null?o.ra:'—')+', '+(o.dec!=null?o.dec:'—')+
    '\nG = '+(o.g!=null?o.g:'—')+'   P(gal) = '+(o.pgal!=null?o.pgal:'—')+
    '\nXGB photo-z = '+(o.xgbz!=null?o.xgbz:'—')+
    '\ncolour(z) dev = '+(o.dev!=null?o.dev:'—')+' mag'+
    '\nastrom = '+(o.aps!=null?(apsL[o.aps]||o.aps):'—');
  if(o.ra!=null && o.dec!=null){cdsRow.style.display='block';cdsLink.href=cdsURL(o.ra,o.dec);
    if(doFrame){cdsFrame.style.display='block';cdsFrame.src=cdsURL(o.ra,o.dec);}}
  else {cdsRow.style.display='none';}
}
function showGrid(){detail.style.display='none';grid.style.display='grid';}  // 'grid' not 'block' — keep the 5-col layout
function cdToObj(cd){return {sid:String(cd[0]),wise:cd[1],ra:cd[2],dec:cd[3],g:cd[4],
    pgal:cd[5],cl:cd[6],xgbz:cd[7],dev:cd[8],aps:cd[9]};}

function highlight(p){
  var z=(p.z&&p.z[curZ]!=null)?p.z[curZ]:0;
  Plotly.restyle(plotDiv,{x:[[p.u1]],y:[[p.u2]],z:[[z]]},[HL_TRACE]);
}
function clearHighlight(){Plotly.restyle(plotDiv,{x:[[]],y:[[]],z:[[]]},[HL_TRACE]);}

// click a contact-sheet cutout -> highlight in 3D + pin its detail
function selectPick(sid){
  var p=PMAP[sid]; if(!p)return;
  curSid=sid; pinned=true; pinBtn.textContent='Unpin';
  markSelected(sid);
  showDetail(p,true); highlight(p);
}
function markSelected(sid){
  var cells=grid.getElementsByClassName('cell');
  for(var i=0;i<cells.length;i++){cells[i].classList.toggle('sel',cells[i].dataset.sid===sid);}
}

// build the clickable contact sheet (grouped by cluster, sorted by P(gal) desc)
function buildGrid(){
  var byCl={}; PICKS.forEach(function(p){(byCl[p.cl]=byCl[p.cl]||[]).push(p);});
  var html='<div class="ghint">contact sheet — click a cutout to highlight it in the 3D plot</div>';
  grid.innerHTML=html;
  Object.keys(byCl).map(Number).sort(function(a,b){return a-b;}).forEach(function(c){
    var arr=byCl[c].slice().sort(function(a,b){return (b.pgal||0)-(a.pgal||0);});
    var hdr=document.createElement('div'); hdr.className='clhdr';
    hdr.style.color=CLCOL[c]||'#ddd';
    hdr.textContent='C'+c+'  ('+arr.length+' picks)'; grid.appendChild(hdr);
    arr.forEach(function(p){
      var cell=document.createElement('div'); cell.className='cell'; cell.dataset.sid=p.sid;
      var b64=CUTOUTS[p.sid];
      var thumb=b64?('<img src="data:image/png;base64,'+b64+'">'):'<div class="ph">no cutout</div>';
      cell.innerHTML=thumb+'<div class="cap">'+(p.wise||p.sid)+'<br>P(gal)='+(p.pgal!=null?p.pgal:'—')+'</div>';
      cell.addEventListener('click',function(){selectPick(p.sid);});
      grid.appendChild(cell);
    });
  });
}
buildGrid();

// hover a 3D pick -> transient detail (no pin)
plotDiv.on('plotly_hover',function(d){if(pinned)return;var pt=d.points[0];
  if(pt.customdata)showDetail(cdToObj(pt.customdata),false);});
plotDiv.on('plotly_unhover',function(){if(pinned)return;showGrid();});
// double-click a 3D pick -> pin + CDS iframe + highlight
var _lt=0,_lp=null;
plotDiv.on('plotly_click',function(d){
  if(!d.points||!d.points.length)return;var now=Date.now(),pt=d.points[0];
  if(now-_lt<400&&_lp&&_lp.pointIndex===pt.pointIndex&&_lp.curveNumber===pt.curveNumber){
    if(pt.customdata){var o=cdToObj(pt.customdata);curSid=o.sid;pinned=true;pinBtn.textContent='Unpin';
      showDetail(o,true);if(PMAP[o.sid])highlight(PMAP[o.sid]);markSelected(o.sid);}
    _lt=0;_lp=null;
  } else {_lt=now;_lp=pt;}
});
pinBtn.onclick=function(){pinned=!pinned;pinBtn.textContent=pinned?'Unpin':'Pin';
  if(!pinned){cdsFrame.style.display='none';cdsFrame.src='';curSid=null;clearHighlight();markSelected(null);showGrid();}};
document.getElementById('back').onclick=function(){pinned=false;pinBtn.textContent='Pin';
  cdsFrame.style.display='none';cdsFrame.src='';curSid=null;clearHighlight();markSelected(null);showGrid();};

// Z-axis selector — re-elevate the picks layer (and the highlight, if one is active)
document.getElementById('zsel').onchange=function(){
  curZ=this.value; var z=ZARR[curZ], meta=ZMETA[curZ];
  Plotly.restyle(plotDiv,{z:[z]},[PICK_TRACE]);
  Plotly.relayout(plotDiv,{'scene.zaxis.title':meta[0],'scene.zaxis.range':meta[1]});
  if(curSid&&PMAP[curSid])highlight(PMAP[curSid]);
};

// Info plots -> open the embedded gallery in a NEW browser tab (self-contained, single file:
// the gallery HTML is embedded here and served to the new tab via an in-memory Blob URL)
var INFO_HTML = __INFO_HTML_JSON__;
document.getElementById('plotsbtn').onclick=function(){
  var u=URL.createObjectURL(new Blob([INFO_HTML],{type:'text/html'}));
  window.open(u,'_blank');
};

// cluster legend
var lg=document.getElementById('legend'),html='<b>clusters</b><br>';
for(var c in CLCOL){html+='<span class="lg"><span class="sw" style="background:'+CLCOL[c]+'"></span>C'+c+'</span>';}
lg.innerHTML=html;

// make sure Plotly fills the (absolutely-positioned) plot pane, not the whole window
function _resize(){try{Plotly.Plots.resize(plotDiv);}catch(e){}}
window.addEventListener('resize',_resize);
setTimeout(_resize,150); setTimeout(_resize,600);
</script></body></html>"""


INFO_TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Picked QSOs — info plots</title>
<style>
 body{margin:0;background:#080808;color:#cfcfcf;padding:20px 24px;
   font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;}
 h1{font-size:16px;color:#eee;margin:0 0 4px;}
 .sub{font-size:12px;color:#777;margin-bottom:14px;}
 h2{font-size:13px;color:#9cf;margin:22px 0 6px;}
 img{max-width:100%;border:1px solid #222;border-radius:6px;background:#000;}
</style></head><body>
<h1>Picked QSOs — summary / info plots</h1>
<div class="sub">companion to picked_qso_umap_web.html</div>
__PLOTS_HTML__
</body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", choices=["combined", "desi", "sdss"], default="combined")
    ap.add_argument("--n-bg", type=int, default=45000,
                    help="background points to render (subsample of the full UMAP; browser perf).")
    args = ap.parse_args()
    user = os.environ.get("PGUSER", "siwei_zhao_2025")

    print("[1] projecting picks onto the", args.sample, "n_all UMAP ...")
    bg, picks, n_full = project_picks(args.sample, user)
    print("[2] rendering pick cutouts fresh (same pipeline as the contact sheet) ...")
    cutouts = render_pick_cutouts(user)
    print("[3] building 3D figure ...")
    fig, z_arrays, z_meta, pick_trace, hl_trace, picks_js = build_figure(bg, picks, args.n_bg)
    plot_div = fig.to_html(full_html=False, include_plotlyjs="cdn",
                           config=dict(displaylogo=False, responsive=True))

    print("[4] embedding summary plots ...")
    plots_html = ""
    for title, fn in SUMMARY_PLOTS:
        b = _b64(PLOTS_DIR / fn)
        if b:
            plots_html += f'<h2>{title}</h2><img src="data:image/png;base64,{b}">\n'
    if not plots_html:
        plots_html = "<p style='color:#888'>No summary plots found in output/picked_qso/.</p>"

    # info-plots gallery — embedded IN this file (opened in a new tab via a Blob URL) so the
    # whole thing stays a single self-contained HTML that can be shared by download.
    info_html = INFO_TEMPLATE.replace("__PLOTS_HTML__", plots_html)

    clcol = {int(c): CLUSTER_COLOURS.get(int(c), "#888")
             for c in sorted(set(int(x) for x in picks["ncluster"]))}
    html = (HTML_TEMPLATE
            .replace("__PLOT_DIV__", plot_div)
            .replace("__CUTOUTS_JSON__", json.dumps(cutouts))
            .replace("__ZARR_JSON__", json.dumps(z_arrays))
            .replace("__ZMETA_JSON__", json.dumps(z_meta))
            .replace("__PICK_TRACE__", str(pick_trace))
            .replace("__HL_TRACE__", str(hl_trace))
            .replace("__PICKS_JSON__", json.dumps(picks_js))
            .replace("__CLCOL_JSON__", json.dumps(clcol))
            .replace("__INFO_HTML_JSON__", json.dumps(info_html)))
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    mb = OUT_HTML.stat().st_size / 1e6
    print(f"[5] {len(picks)} picks · {min(len(bg['u1']), args.n_bg):,}/{n_full:,} bg pts · "
          f"{len(cutouts)} cutouts")
    print(f"  -> {OUT_HTML}  ({mb:.1f} MB, single self-contained file)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
