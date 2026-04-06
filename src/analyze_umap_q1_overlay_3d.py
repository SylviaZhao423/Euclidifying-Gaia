#!/usr/bin/env python3
"""
Combined interactive 3D UMAP HTML with switchable QSO / Rejected overlay.

Produces:
  q1_overlay_3d.html  — interactive 3D: UMAP x/y + G magnitude as z
                        Q1 base always shown; overlay dropdown switches between
                        QSO variants (phz=4/5/6/7) and Rejected (phz=-1).

Reads:
  output/umap/arrays/q1_indiv.npz
  output/umap/arrays/q1_indiv_reducer.joblib
  output/umap/arrays/q1_cluster_labels.npy

Writes:
  output/umap/q1_individual/q1_overlay_3d.html
  output/umap/arrays/q1_qso_projected.csv
  output/umap/arrays/q1_rej_projected.csv

Usage:
  # Full run: DB query + project + save CSVs + generate HTML
  PGUSER=siwei_zhao_2025 python src/analyze_umap_q1_overlay_3d.py
  PGUSER=siwei_zhao_2025 python src/analyze_umap_q1_overlay_3d.py --max-points 5000

  # Fast re-render: skip DB query, regenerate HTML from saved CSVs
  python src/analyze_umap_q1_overlay_3d.py --3d
  python src/analyze_umap_q1_overlay_3d.py --3d --max-points 5000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR     = Path(__file__).resolve().parents[1]
ARRAYS_DIR      = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR      = PROJECT_DIR / "output" / "umap" / "q1_individual"
CUTOUTS_DIR_QSO = PROJECT_DIR / "output" / "euclid_cutouts" / "q1_cutouts" / "qso"
CUTOUTS_DIR_REJ = PROJECT_DIR / "output" / "euclid_cutouts" / "q1_cutouts" / "rej"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
MAG_CUT   = 17

QSO_LABELS = {
    4: ("QSO",                "cyan"),
    5: ("Multi (Star ∩ QSO)", "yellow"),
    6: ("Multi (Gal ∩ QSO)",  "#F45347"),
    7: ("Outlier∩QSO",        "orange"),
}

# Plotly 3D symbol equivalents
PHZ_PLOTLY_SYMBOLS = {4: "circle", 5: "square", 6: "diamond", 7: "cross"}

CLUSTER_COLORS = [
    "#1f77b4",       # 0 — blue
    "#ff7f0e",       # 1 — orange
    "#2ca02c",       # 2 — green
    "#F298B6",       # 3 — pink
    "mediumpurple",  # 4 — purple
    "#17becf",       # 5 — teal
    "#d62728",       # 6 — red
    "#8c564b",       # 7 — brown
    "#7f7f7f",       # 8 — grey
    "#bcbd22",       # 9 — olive
]


# ── DB queries ────────────────────────────────────────────────────────────────

def query_qso_variants(username: str) -> pd.DataFrame:
    import psycopg2
    gaia_cols = ", ".join([f"g_{col} AS {col}" for col in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT
            g_source_id AS source_id,
            m_right_ascension AS ra,
            m_declination     AS dec,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            p_phz_classification AS phz,
            {gaia_cols}
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE p_phz_classification IN (4, 5, 6, 7)
          AND g_phot_g_mean_mag >= {MAG_CUT}
          AND m_right_ascension IS NOT NULL
          AND m_declination IS NOT NULL
        ORDER BY g_source_id
    """
    print("  Querying QSO variants (phz=4/5/6/7)...")
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    df = pd.read_sql_query(query, conn)
    conn.close()

    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace("NOT_AVAILABLE", np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["phz"] = df["phz"].astype(int)
    print(f"  -> {len(df):,} QSO-variant sources")
    for v, (lbl, _) in QSO_LABELS.items():
        n = (df["phz"] == v).sum()
        if n:
            print(f"     phz={v}  {lbl}: {n:,}")
    return df


def query_rejected(username: str) -> pd.DataFrame:
    """Query phz=-1 sources with RA/Dec and Gaia features."""
    import psycopg2
    gaia_cols = ", ".join([f"g_{col} AS {col}" for col in SELECTED_GAIA_PARAMETERS])
    query = f"""
        SELECT
            g_source_id AS source_id,
            m_right_ascension AS ra,
            m_declination     AS dec,
            g_phot_g_mean_mag AS phot_g_mean_mag,
            {gaia_cols}
        FROM siwei_zhao_2025.crossmatch_q1_mer_phz_gaia
        WHERE p_phz_classification = -1
          AND g_phot_g_mean_mag >= {MAG_CUT}
          AND m_right_ascension IS NOT NULL
          AND m_declination IS NOT NULL
        ORDER BY g_source_id
    """
    print("  Querying Rejected (phz=-1) sources...")
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    df = pd.read_sql_query(query, conn)
    conn.close()

    df[SELECTED_GAIA_PARAMETERS] = df[SELECTED_GAIA_PARAMETERS].replace("NOT_AVAILABLE", np.nan)
    for col in SELECTED_GAIA_PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"  -> {len(df):,} rejected sources")
    return df


# ── Cluster assignment ────────────────────────────────────────────────────────

def assign_to_clusters(src_emb: np.ndarray,
                       q1_emb: np.ndarray,
                       q1_labels: np.ndarray) -> np.ndarray:
    valid_mask = q1_labels >= 0
    if valid_mask.sum() == 0:
        return np.full(len(src_emb), -1, dtype=int)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(q1_emb[valid_mask])
    _, idx = nn.kneighbors(src_emb)
    idx = idx.ravel()
    valid_indices = np.where(valid_mask)[0]
    return q1_labels[valid_indices[idx]]


# ── Cutout caches (QSO) ───────────────────────────────────────────────────────

def load_cutout_cache_qso(cutouts_dir: Path) -> dict:
    """Scan cluster*/manifest_phz*.csv and return {source_id_str: base64_png_str}."""
    import base64
    from io import BytesIO
    from PIL import Image

    cache = {}
    for manifest_path in sorted(cutouts_dir.glob("*/manifest_phz*.csv")):
        cell_dir = manifest_path.parent
        try:
            df = pd.read_csv(manifest_path)
        except Exception:
            continue
        if "cutout_file" not in df.columns:
            continue
        for _, row in df.iterrows():
            if pd.isna(row.get("cutout_file")):
                continue
            png_path = cell_dir / row["cutout_file"]
            if not png_path.exists():
                continue
            try:
                img = Image.open(png_path).convert("RGB")
                buf = BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                sid = str(row["source_id"])
                cache[sid] = b64
            except Exception:
                continue
    print(f"  [QSO] Loaded {len(cache)} cached cutout thumbnails")
    return cache


def load_contact_sheet_cache_qso(cutouts_dir: Path) -> tuple[dict, dict]:
    """Scan cluster*/contact_sheet_phz*.png.

    Returns
    -------
    cache  : {phz_str: {cluster_str: base64_str}}
    counts : {phz_str: {cluster_str: int}}  — rows in the matching manifest
    """
    import base64
    from io import BytesIO
    from PIL import Image

    cache:  dict[str, dict[str, str]] = {}
    counts: dict[str, dict[str, int]] = {}
    for cluster_dir in sorted(cutouts_dir.glob("cluster*")):
        k = cluster_dir.name.replace("cluster", "")
        for sheet_path in sorted(cluster_dir.glob("contact_sheet_phz*.png")):
            v = sheet_path.stem.replace("contact_sheet_phz", "")
            try:
                img = Image.open(sheet_path).convert("RGB")
                buf = BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                cache.setdefault(v, {})[k] = b64
            except Exception:
                continue
            manifest = cluster_dir / f"manifest_phz{v}.csv"
            if manifest.exists():
                try:
                    with open(manifest) as f:
                        n = sum(1 for _ in f) - 1
                except Exception:
                    n = 0
            else:
                n = 0
            counts.setdefault(v, {})[k] = max(n, 0)
    n_sheets = sum(len(d) for d in cache.values())
    print(f"  [QSO] Loaded {n_sheets} contact sheet(s) for phz values: {sorted(cache.keys())}")
    return cache, counts


# ── Cutout caches (REJ) ───────────────────────────────────────────────────────

def load_cutout_cache_rej(cutouts_dir: Path) -> dict:
    """Scan cluster*/manifest.csv and return {source_id_str: base64_png_str}."""
    import base64
    from io import BytesIO
    from PIL import Image

    cache = {}
    for manifest_path in sorted(cutouts_dir.glob("*/manifest.csv")):
        cell_dir = manifest_path.parent
        try:
            df = pd.read_csv(manifest_path)
        except Exception:
            continue
        if "cutout_file" not in df.columns:
            continue
        for _, row in df.iterrows():
            if pd.isna(row.get("cutout_file")):
                continue
            png_path = cell_dir / row["cutout_file"]
            if not png_path.exists():
                continue
            try:
                img = Image.open(png_path).convert("RGB")
                buf = BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                cache[str(row["source_id"])] = b64
            except Exception:
                continue
    print(f"  [REJ] Loaded {len(cache)} cached cutout thumbnails")
    return cache


def load_contact_sheet_cache_rej(cutouts_dir: Path) -> tuple[dict, dict]:
    """Scan cluster*/contact_sheet.png.

    Returns
    -------
    cache  : {cluster_str: base64_str}
    counts : {cluster_str: int}  — rows in the matching manifest.csv
    """
    import base64
    from io import BytesIO
    from PIL import Image

    cache:  dict[str, str] = {}
    counts: dict[str, int] = {}
    for cluster_dir in sorted(cutouts_dir.glob("cluster*")):
        k = cluster_dir.name.replace("cluster", "")
        sheet_path = cluster_dir / "contact_sheet.png"
        if sheet_path.exists():
            try:
                img = Image.open(sheet_path).convert("RGB")
                buf = BytesIO()
                img.save(buf, format="PNG")
                cache[k] = base64.b64encode(buf.getvalue()).decode()
            except Exception:
                pass
        manifest = cluster_dir / "manifest.csv"
        if manifest.exists():
            try:
                with open(manifest) as f:
                    n = sum(1 for _ in f) - 1
            except Exception:
                n = 0
        else:
            n = 0
        counts[k] = max(n, 0)
    print(f"  [REJ] Loaded {len(cache)} contact sheet(s) for clusters: {sorted(cache.keys())}")
    return cache, counts


# ── Subsampling ───────────────────────────────────────────────────────────────

def _subsample_keep_cutouts(df: pd.DataFrame, cutout_sids: set,
                            max_points: int, seed: int = 42) -> pd.DataFrame:
    """Keep all rows with cutouts; fill remaining quota randomly from the rest."""
    has_cutout = df["source_id"].astype(str).isin(cutout_sids)
    df_with    = df[has_cutout]
    df_without = df[~has_cutout]
    n_fill = max(0, max_points - len(df_with))
    if n_fill < len(df_without):
        df_without = df_without.sample(n_fill, random_state=seed)
    result = pd.concat([df_with, df_without], ignore_index=True)
    print(f"  Subsampled to {len(result):,}  "
          f"({len(df_with):,} with cutouts + {len(df_without):,} without, "
          f"from {len(df):,} total)")
    return result


# ── 3D Plotly HTML ────────────────────────────────────────────────────────────

def plot_umap_3d_plotly(
    q1_emb, q1_labels, q1_gmag,
    qso_emb, qso_phz, qso_gmag,
    rej_emb, rej_cluster, rej_gmag,
    output_dir,
    cutout_cache_qso=None, df_qso=None,
    cutout_cache_rej=None, df_rej=None,
    qso_contact_sheet_cache=None, qso_sheet_counts=None,
    rej_contact_sheet_cache=None, rej_sheet_counts=None,
    q1_features=None, feature_names=None,
):
    """Interactive 3D: UMAP x/y + G mag as z.  Switchable QSO / Rejected overlay."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  plotly not available — skipping 3D plot. Install with: pip install plotly")
        return

    cluster_ids = sorted(set(q1_labels[q1_labels >= 0]))

    # ── Feature-coloring support ───────────────────────────────────────────────
    feat_ranges: dict = {}
    if q1_features is not None and feature_names is not None:
        for i, fname in enumerate(feature_names):
            col = q1_features[:, i]
            finite = col[np.isfinite(col)]
            if len(finite) > 0:
                feat_ranges[fname] = [
                    round(float(np.percentile(finite, 1)), 4),
                    round(float(np.percentile(finite, 99)), 4),
                ]
            else:
                feat_ranges[fname] = [0.0, 1.0]

    def _feat_cd(mask):
        """Feature customdata rows for the given mask; None if no features loaded."""
        if q1_features is None:
            return None
        return [[None if not np.isfinite(v) else round(float(v), 4) for v in row]
                for row in q1_features[mask]]

    traces = []

    # ── Q1 base background ────────────────────────────────────────────────────
    noise_mask = q1_labels == -1
    if noise_mask.any():
        z_noise = q1_gmag[noise_mask] if q1_gmag is not None else np.zeros(noise_mask.sum())
        traces.append(go.Scatter3d(
            x=q1_emb[noise_mask, 0], y=q1_emb[noise_mask, 1], z=z_noise,
            mode="markers",
            marker=dict(size=1, color="#2a2a2a", opacity=0.5),
            name="noise",
            customdata=_feat_cd(noise_mask),
        ))
    for k in cluster_ids:
        m = q1_labels == k
        z_k = q1_gmag[m] if q1_gmag is not None else np.zeros(m.sum())
        traces.append(go.Scatter3d(
            x=q1_emb[m, 0], y=q1_emb[m, 1], z=z_k,
            mode="markers",
            marker=dict(size=1, color=CLUSTER_COLORS[k % len(CLUSTER_COLORS)], opacity=0.15),
            name=f"Q1 Cluster {k}",
            customdata=_feat_cd(m),
        ))

    # ── QSO overlay traces ────────────────────────────────────────────────────
    b64_js: dict = {}
    _qso_cl = df_qso["cluster"].values if (df_qso is not None and "cluster" in df_qso.columns) else None

    for v, (lbl, col) in QSO_LABELS.items():
        m_v = qso_phz == v
        if not m_v.any():
            continue

        mkr = dict(size=5, color=col, symbol=PHZ_PLOTLY_SYMBOLS.get(v, "circle"),
                   opacity=0.95, line=dict(width=0.3, color="rgba(255,255,255,0.25)"))

        # Legend-entry trace (NaN phantom, showlegend=True)
        traces.append(go.Scatter3d(
            x=[float("nan")], y=[float("nan")], z=[float("nan")],
            mode="markers",
            marker=mkr,
            name=f"phz={v} {lbl}",
            showlegend=True,
            hoverinfo="skip",
        ))

        # Data sub-traces split by cluster (showlegend=False)
        sub_clusters = sorted(set(int(c) for c in _qso_cl[m_v])) if _qso_cl is not None else [-1]
        for k in sub_clusters:
            mk = m_v & (_qso_cl == k) if _qso_cl is not None else m_v
            if not mk.any():
                continue
            z_q = qso_gmag[mk] if qso_gmag is not None else np.zeros(mk.sum())
            custom = []
            if df_qso is not None:
                sub_df = df_qso[mk].reset_index(drop=True)
                for _, row in sub_df.iterrows():
                    sid   = str(row.get("source_id", ""))
                    ra    = float(row["ra"])
                    dec   = float(row["dec"])
                    gmag  = float(row["phot_g_mean_mag"])
                    label = f"phz={v} {lbl}"
                    feats: list = []
                    for fn in (feature_names or []):
                        try:
                            fv = float(row.get(fn, float("nan")))
                            feats.append(None if not np.isfinite(fv) else round(fv, 4))
                        except (ValueError, TypeError):
                            feats.append(None)
                    custom.append([sid, round(ra, 4), round(dec, 4),
                                   round(gmag, 4), label] + feats)
                    if cutout_cache_qso and sid in cutout_cache_qso:
                        b64_js[sid] = cutout_cache_qso[sid]
            else:
                _empty = ["", 0.0, 0.0, 0.0, ""] + [None] * len(feature_names or [])
                custom = [_empty for _ in range(mk.sum())]
            traces.append(go.Scatter3d(
                x=qso_emb[mk, 0], y=qso_emb[mk, 1], z=z_q,
                mode="markers",
                marker=mkr,
                name=f"phz={v} C{k}",
                customdata=custom,
                hovertemplate="RA=%{customdata[1]:.4f}<br>Dec=%{customdata[2]:+.4f}<br>G=%{customdata[3]:.2f}<extra>%{customdata[4]}</extra>",
                showlegend=False,
            ))

    # ── REJ overlay traces ────────────────────────────────────────────────────
    for k in cluster_ids:
        m = rej_cluster == k
        if not m.any():
            continue
        z_r = rej_gmag[m] if rej_gmag is not None else np.zeros(m.sum())
        label = f"Rejected C{k}"
        custom = []
        if df_rej is not None:
            sub = df_rej[m].reset_index(drop=True)
            for _, row in sub.iterrows():
                sid  = str(row.get("source_id", ""))
                ra   = float(row["ra"])
                dec  = float(row["dec"])
                gmag = float(row["phot_g_mean_mag"])
                feats: list = []
                for fn in (feature_names or []):
                    try:
                        fv = float(row.get(fn, float("nan")))
                        feats.append(None if not np.isfinite(fv) else round(fv, 4))
                    except (ValueError, TypeError):
                        feats.append(None)
                custom.append([sid, round(ra, 4), round(dec, 4), round(gmag, 4), label] + feats)
                if cutout_cache_rej and sid in cutout_cache_rej:
                    b64_js[sid] = cutout_cache_rej[sid]
        else:
            _empty = ["", 0.0, 0.0, 0.0, label] + [None] * len(feature_names or [])
            custom = [_empty for _ in range(m.sum())]

        traces.append(go.Scatter3d(
            x=rej_emb[m, 0], y=rej_emb[m, 1], z=z_r,
            mode="markers",
            marker=dict(
                size=4,
                color=CLUSTER_COLORS[k % len(CLUSTER_COLORS)],
                symbol="diamond", opacity=0.92,
                line=dict(width=0.3, color="rgba(255,255,255,0.2)"),
            ),
            name=label,
            customdata=custom,
            hovertemplate="RA=%{customdata[1]:.4f}<br>Dec=%{customdata[2]:+.4f}<br>G=%{customdata[3]:.2f}<extra>%{customdata[4]}</extra>",
            showlegend=True,
        ))

    # Unassigned rejected
    unassigned = rej_cluster == -1
    if unassigned.any():
        z_u = rej_gmag[unassigned] if rej_gmag is not None else np.zeros(unassigned.sum())
        label_u = "Rejected (unassigned)"
        custom_u = []
        if df_rej is not None:
            sub = df_rej[unassigned].reset_index(drop=True)
            for _, row in sub.iterrows():
                sid  = str(row.get("source_id", ""))
                ra   = float(row["ra"])
                dec  = float(row["dec"])
                gmag = float(row["phot_g_mean_mag"])
                feats: list = []
                for fn in (feature_names or []):
                    try:
                        fv = float(row.get(fn, float("nan")))
                        feats.append(None if not np.isfinite(fv) else round(fv, 4))
                    except (ValueError, TypeError):
                        feats.append(None)
                custom_u.append([sid, round(ra, 4), round(dec, 4), round(gmag, 4), label_u] + feats)
                if cutout_cache_rej and sid in cutout_cache_rej:
                    b64_js[sid] = cutout_cache_rej[sid]
        traces.append(go.Scatter3d(
            x=rej_emb[unassigned, 0], y=rej_emb[unassigned, 1], z=z_u,
            mode="markers",
            marker=dict(size=3, color="#666", symbol="x", opacity=0.5,
                        line=dict(width=1, color="#444")),
            name=label_u,
            customdata=custom_u,
            hovertemplate="RA=%{customdata[1]:.4f}<br>Dec=%{customdata[2]:+.4f}<br>G=%{customdata[3]:.2f}<extra>%{customdata[4]}</extra>",
            showlegend=True,
        ))

    # ── Fixed axis ranges ──────────────────────────────────────────────────────
    _pad = 0.05
    _all_x = np.concatenate([q1_emb[:, 0], qso_emb[:, 0], rej_emb[:, 0]])
    _all_y = np.concatenate([q1_emb[:, 1], qso_emb[:, 1], rej_emb[:, 1]])
    _xr = [float(_all_x.min() - (np.ptp(_all_x) * _pad)),
           float(_all_x.max() + (np.ptp(_all_x) * _pad))]
    _yr = [float(_all_y.min() - (np.ptp(_all_y) * _pad)),
           float(_all_y.max() + (np.ptp(_all_y) * _pad))]
    if q1_gmag is not None and qso_gmag is not None and rej_gmag is not None:
        _gz = np.concatenate([q1_gmag[np.isfinite(q1_gmag)],
                              qso_gmag[np.isfinite(qso_gmag)],
                              rej_gmag[np.isfinite(rej_gmag)]])
        _zpad = float(np.ptp(_gz) * _pad)
        _zr = [float(_gz.max()) + _zpad, float(_gz.min()) - _zpad]  # reversed: faint→bright
    else:
        _zr = None

    _ax = dict(
        backgroundcolor="#080808",
        gridcolor="#1c1c1c",
        showbackground=True,
        zerolinecolor="#222",
        tickfont=dict(color="#444", size=9),
        title_font=dict(color="#666", size=10),
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor="#080808",
        plot_bgcolor="#080808",
        font=dict(color="#aaa",
                  family='-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif',
                  size=11),
        scene=dict(
            bgcolor="#080808",
            uirevision="locked",
            xaxis=dict(**_ax, title="UMAP 1", autorange=False, range=_xr),
            yaxis=dict(**_ax, title="UMAP 2", autorange=False, range=_yr),
            zaxis=dict(**_ax, title="G mag",
                       **dict(autorange=False, range=_zr) if _zr else dict(autorange="reversed")),
        ),
        uirevision="locked",
        legend=dict(
            bgcolor="rgba(12,12,12,0.9)",
            bordercolor="#222",
            borderwidth=1,
            font=dict(color="#999", size=11),
            itemsizing="constant",
            x=1, xanchor="right",
            y=0.98, yanchor="top",
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        width=1100, height=750,
    )

    plot_div = fig.to_html(full_html=False, include_plotlyjs="cdn",
                           config=dict(displaylogo=False, displayModeBar="hover",
                                       modeBarButtonsToRemove=["toImage", "sendDataToCloud",
                                                               "editInChartStudio"]))

    # Merge cutout caches: QSO first, then REJ (REJ may overwrite if same sid, but sids are distinct)
    merged_cutouts = {}
    if cutout_cache_qso:
        merged_cutouts.update(cutout_cache_qso)
    if cutout_cache_rej:
        merged_cutouts.update(cutout_cache_rej)
    # Only embed the sids actually referenced in traces
    merged_cutouts = {k: v for k, v in merged_cutouts.items() if k in b64_js}

    merged_cutouts_json = json.dumps(merged_cutouts)
    n_cached            = len(merged_cutouts)
    qso_cs_json         = json.dumps(qso_contact_sheet_cache or {})
    qso_counts_json     = json.dumps(qso_sheet_counts or {})
    rej_cs_json         = json.dumps(rej_contact_sheet_cache or {})
    rej_counts_json     = json.dumps(rej_sheet_counts or {})
    feature_names_json  = json.dumps(feature_names or [])
    feature_ranges_json = json.dumps(feat_ranges)
    cluster_colors_json = json.dumps(CLUSTER_COLORS)
    feature_options_html = "\n".join(
        f'      <option value="{fn}">{fn}</option>'
        for fn in (feature_names or [])
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Euclid Q1 — Overlay UMAP</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  .plotly-notifier {{ display: none !important; }}
  body {{
    background: #080808; color: #bbb;
    display: flex; height: 100vh; overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  }}
  #plot-container {{ flex: 1; min-width: 0; overflow: hidden; background: #080808; }}
  #plot-container .plotly-graph-div {{ width: 100% !important; height: 100% !important; }}

  #resize-handle {{
    width: 5px; flex-shrink: 0; cursor: col-resize;
    background: #111; border-left: 1px solid #1a1a1a;
    display: flex; align-items: center; justify-content: center;
    user-select: none; transition: background 0.15s;
  }}
  #resize-handle:hover, #resize-handle.dragging {{ background: #1e2a3a; }}
  #resize-handle::after {{
    content: '⋮'; color: #2a2a2a; font-size: 14px; pointer-events: none;
  }}

  #sidebar {{
    width: 270px; flex-shrink: 0;
    background: #0c0c0c; border-left: 1px solid #181818;
    display: flex; flex-direction: column; overflow: hidden;
  }}
  #sidebar-header {{
    padding: 16px 16px 12px; border-bottom: 1px solid #181818; flex-shrink: 0;
  }}
  #sidebar-eyebrow {{
    font-size: 13px; font-weight: 600; color: #888; line-height: 1.3;
    margin-bottom: 5px;
  }}
  #sidebar-hint {{
    font-size: 11px; color: #444; margin-top: 4px;
  }}
  #tip-card {{
    margin: 10px 16px 0; padding: 9px 10px 9px 12px;
    background: #111; border: 1px solid #1e2e40; border-radius: 3px;
    font-size: 10.5px; line-height: 1.6; color: #4a7a9b;
    display: flex; gap: 8px; align-items: flex-start; flex-shrink: 0;
  }}
  #tip-card span {{ flex: 1; }}
  #tip-dismiss {{
    cursor: pointer; color: #2a4a5c; font-size: 13px; line-height: 1;
    flex-shrink: 0; padding: 0 2px; background: none; border: none;
    transition: color 0.15s;
  }}
  #tip-dismiss:hover {{ color: #4a9edd; }}

  #sidebar-body {{
    padding: 14px 16px; overflow-y: auto; flex: 1;
  }}

  #pin-row {{
    display: flex; justify-content: flex-end; margin-bottom: 10px;
  }}
  #pin-btn {{
    font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
    padding: 3px 8px; cursor: pointer;
    border: 1px solid #222; border-radius: 2px;
    background: transparent; color: #444;
    transition: all 0.15s;
  }}
  #pin-btn:hover {{ border-color: #383838; color: #777; }}
  #pin-btn.pinned {{ background: #0e2035; color: #4a9edd; border-color: #1a3d5c; }}

  #cutout-img {{
    width: 100%; display: none;
    border: 1px solid #181818; border-radius: 2px;
    margin-bottom: 10px;
  }}
  #cutout-info {{
    font-size: 10.5px; line-height: 1.7; white-space: pre;
    font-family: "SF Mono", "Fira Code", Menlo, Consolas, monospace;
    color: #555;
  }}
  #no-cutout {{ font-size: 11px; color: #2a2a2a; font-style: italic; }}

  #sheet-view {{
    display: none; width: 100%;
    margin-top: 16px; padding-top: 14px;
    border-top: 1px solid #181818;
  }}
  #sheet-header {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 10px;
  }}
  #sheet-title {{
    font-size: 9px; letter-spacing: 0.13em; text-transform: uppercase; color: #444;
  }}
  #sheet-close {{
    font-size: 11px; padding: 1px 6px; cursor: pointer;
    border: 1px solid #222; border-radius: 2px;
    background: transparent; color: #444;
    transition: all 0.15s; flex-shrink: 0;
  }}
  #sheet-close:hover {{ color: #888; border-color: #444; }}
  #sheet-images .sheet-cluster-label {{
    font-size: 9px; letter-spacing: 0.1em; text-transform: uppercase;
    color: #444; margin-bottom: 4px; margin-top: 8px;
  }}
  #sheet-images img {{
    width: 100%; border: 1px solid #181818; margin-bottom: 4px;
    display: block; border-radius: 2px;
  }}
  #color-mode-bar {{
    padding: 7px 16px; border-bottom: 1px solid #181818;
    display: flex; align-items: center; gap: 8px; flex-shrink: 0;
  }}
  #color-mode-label {{
    font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase;
    color: #666; white-space: nowrap;
  }}
  #color-mode-select {{
    flex: 1; background: #0e0e0e; color: #666;
    border: 1px solid #222; border-radius: 2px;
    font-size: 10px; padding: 3px 4px; cursor: pointer;
    font-family: "SF Mono", "Fira Code", Menlo, Consolas, monospace;
    max-width: 100%;
  }}
  #color-mode-select:focus {{ outline: none; border-color: #333; }}
  #color-mode-select option {{ background: #111; }}
  #overlay-bar {{
    padding: 7px 16px; border-bottom: 1px solid #181818;
    display: flex; align-items: center; gap: 8px; flex-shrink: 0;
  }}
  #overlay-label {{
    font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase;
    color: #666; white-space: nowrap;
  }}
  #overlay-select {{
    flex: 1; background: #0e0e0e; color: #666;
    border: 1px solid #222; border-radius: 2px;
    font-size: 10px; padding: 3px 4px; cursor: pointer;
    font-family: "SF Mono", "Fira Code", Menlo, Consolas, monospace;
    max-width: 100%;
  }}
  #overlay-select:focus {{ outline: none; border-color: #333; }}
  #overlay-select option {{ background: #111; }}
</style>
</head>
<body>
<div id="plot-container">{plot_div}</div>
<div id="resize-handle"></div>
<div id="sidebar">
  <div id="sidebar-header">
    <div id="sidebar-eyebrow">Euclid Q1 · UMAP</div>
    <div id="sidebar-hint">hover a point · click to pin</div>
  </div>
  <div id="overlay-bar">
    <span id="overlay-label">Overlay</span>
    <select id="overlay-select">
      <option value="qso">QSO variants</option>
      <option value="rej">Rejected</option>
    </select>
  </div>
  <div id="color-mode-bar">
    <span id="color-mode-label">Colour Q1</span>
    <select id="color-mode-select">
      <option value="clusters">clusters</option>
{feature_options_html}
    </select>
  </div>
  <div id="tip-card">
    <span>Click a legend to remove it.<br>⌘+click a legend to isolate it.<br>Press Esc to restore everything.</span>
    <button id="tip-dismiss">&#x2715;</button>
  </div>
  <div id="sidebar-body">
    <div id="pin-row">
      <button id="pin-btn">Pin</button>
    </div>
    <img id="cutout-img" src="">
    <div id="cutout-info"></div>
    <div id="no-cutout"></div>
    <div id="sheet-view">
      <div id="sheet-header">
        <span id="sheet-title">Contact sheets</span>
        <button id="sheet-close">&#x2715;</button>
      </div>
      <div id="sheet-images"></div>
    </div>
  </div>
</div>
<script>
var cutouts = {merged_cutouts_json};
var qso_contact_sheets = {qso_cs_json};
var qso_sheet_counts = {qso_counts_json};
var rej_contact_sheets = {rej_cs_json};
var rej_sheet_counts = {rej_counts_json};
/* cluster_sheets: invert qso_contact_sheets → {{cluster: {{phz: base64}}}} */
var cluster_sheets = {{}};
Object.keys(qso_contact_sheets).forEach(function(v) {{
  Object.keys(qso_contact_sheets[v]).forEach(function(k) {{
    if (!cluster_sheets[k]) cluster_sheets[k] = {{}};
    cluster_sheets[k][v] = qso_contact_sheets[v][k];
  }});
}});
/* feature coloring */
var featureNames  = {feature_names_json};
var featureRanges = {feature_ranges_json};
var CLUSTER_COLORS_JS = {cluster_colors_json};
var COLORSCALE_RDBU = [
  [0,    'rgb(5,48,97)'],   [0.25, 'rgb(33,102,172)'],
  [0.5,  'rgb(247,247,247)'],
  [0.75, 'rgb(214,96,77)'], [1,    'rgb(103,0,31)']
];
var activeOverlay = 'qso';
var pinned = false;
var sheetMode = false;
var sheetPhz = null;
var sheetCluster = null;
var isolatedPhz = null;
var isolatedCluster = null;
var hiddenClusters = {{}};
var hiddenPhz = {{}};
var plotDiv = document.querySelector('.plotly-graph-div');
var img = document.getElementById('cutout-img');
var infoDiv = document.getElementById('cutout-info');
var noCutout = document.getElementById('no-cutout');
var pinBtn = document.getElementById('pin-btn');
var sidebar = document.getElementById('sidebar');
var handle = document.getElementById('resize-handle');
var sheetView = document.getElementById('sheet-view');
var sheetTitle = document.getElementById('sheet-title');
var sheetImages = document.getElementById('sheet-images');
var sheetClose = document.getElementById('sheet-close');

document.getElementById('tip-dismiss').addEventListener('click', function() {{
  var card = document.getElementById('tip-card');
  card.style.display = 'none';
}});

/* ── pin button ── */
pinBtn.addEventListener('click', function() {{
  pinned = !pinned;
  pinBtn.textContent = pinned ? 'Unpin' : 'Pin';
  pinBtn.classList.toggle('pinned', pinned);
}});

/* ── contact sheet helpers ── */
function clearSheetMode() {{
  sheetMode = false; sheetPhz = null; sheetCluster = null;
  sheetView.style.display = 'none';
  sheetImages.innerHTML = '';
}}

function _countTotal(countsObj) {{
  /* sum all integer values in a {{key: int}} object */
  return Object.values(countsObj).reduce(function(a,b){{return a+b;}}, 0);
}}

function showContactSheets(v) {{
  /* QSO phz isolation → show all clusters for this phz */
  var clusters = qso_contact_sheets[v];
  if (!clusters || Object.keys(clusters).length === 0) {{
    sheetTitle.innerHTML = 'phz=' + v + ' — no sheets cached';
    sheetImages.innerHTML = '';
    sheetView.style.display = 'block';
    return;
  }}
  var keys = Object.keys(clusters).map(Number).sort(function(a,b){{return a-b;}});
  var phzCounts = qso_sheet_counts[v] || {{}};
  var total = _countTotal(phzCounts);
  var html = '';
  keys.forEach(function(k) {{
    var n = phzCounts[String(k)] !== undefined ? phzCounts[String(k)] : '?';
    html += '<div class="sheet-cluster-label">Cluster ' + k + ' (' + n + ' sources)</div>';
    html += '<img src="data:image/png;base64,' + clusters[String(k)] + '">';
  }});
  sheetTitle.innerHTML = 'phz=' + v + ' contact sheets<br>' + total + ' sources';
  sheetImages.innerHTML = html;
  sheetView.style.display = 'block';
}}

function showClusterSheets(k) {{
  /* QSO cluster isolation → show all phz for this cluster */
  var phzs = cluster_sheets[String(k)];
  if (!phzs || Object.keys(phzs).length === 0) {{
    sheetTitle.innerHTML = 'Cluster ' + k + ' — no sheets cached';
    sheetImages.innerHTML = '';
    sheetView.style.display = 'block';
    return;
  }}
  var keys = Object.keys(phzs).map(Number).sort(function(a,b){{return a-b;}});
  var total = 0;
  var html = '';
  keys.forEach(function(v) {{
    var n = (qso_sheet_counts[String(v)] || {{}})[String(k)];
    n = (n !== undefined) ? n : '?';
    if (typeof n === 'number') total += n;
    html += '<div class="sheet-cluster-label">phz=' + v + ' (' + n + ' sources)</div>';
    html += '<img src="data:image/png;base64,' + phzs[String(v)] + '">';
  }});
  sheetTitle.innerHTML = 'Cluster ' + k + ' contact sheets<br>' + total + ' sources';
  sheetImages.innerHTML = html;
  sheetView.style.display = 'block';
}}

function showRejContactSheet(k) {{
  /* REJ cluster isolation → single contact sheet */
  var b64 = rej_contact_sheets[String(k)];
  var n = rej_sheet_counts[String(k)] !== undefined ? rej_sheet_counts[String(k)] : '?';
  if (!b64) {{
    sheetTitle.innerHTML = 'Cluster ' + k + ' — no sheet cached';
    sheetImages.innerHTML = '';
    sheetView.style.display = 'block';
    return;
  }}
  sheetTitle.innerHTML = 'Cluster ' + k + ' contact sheet<br>' + n + ' sources';
  sheetImages.innerHTML = '<img src="data:image/png;base64,' + b64 + '">';
  sheetView.style.display = 'block';
}}

sheetClose.addEventListener('click', function() {{ clearSheetMode(); }});

/* ── overlay switching ── */
function switchOverlay(name) {{
  activeOverlay = name;
  isolatedCluster = null; isolatedPhz = null;
  hiddenClusters = {{}}; hiddenPhz = {{}};
  clearSheetMode();
  /* heading removed; overlay selection visible in the dropdown */
  applyVisibility();
}}
document.getElementById('overlay-select').addEventListener('change', function() {{
  switchOverlay(this.value);
}});

/* ── feature coloring ── */
function _q1BaseIndices() {{
  var out = [];
  plotDiv.data.forEach(function(tr, i) {{
    if (/^Q1 Cluster \\d+$/.test(tr.name) || tr.name === 'noise') out.push(i);
  }});
  return out;
}}

function _phzSubIndices() {{
  var out = [];
  plotDiv.data.forEach(function(tr, i) {{
    if (/^phz=\\d+ C/.test(tr.name)) out.push(i);
  }});
  return out;
}}

function _rejIndices() {{
  var out = [];
  plotDiv.data.forEach(function(tr, i) {{
    if (/^Rejected C\\d+$/.test(tr.name) || tr.name === 'Rejected (unassigned)') out.push(i);
  }});
  return out;
}}

/* Update PHZ tooltip: RA/Dec/G + phz label (+ feature if active). */
function updatePhzHoverTemplate(feature) {{
  var idxs = _phzSubIndices();
  if (!idxs.length) return;
  var base = 'RA=%{{customdata[1]:.4f}}<br>Dec=%{{customdata[2]:+.4f}}<br>G=%{{customdata[3]:.2f}}';
  var ext;
  if (feature === 'clusters') {{
    ext = '<extra>%{{customdata[4]}}</extra>';
  }} else {{
    var fi = featureNames.indexOf(feature);
    if (fi < 0) {{ ext = '<extra>%{{customdata[4]}}</extra>'; }}
    else {{ ext = '<extra>%{{customdata[4]}}<br>' + feature + ': %{{customdata[' + (5 + fi) + ']:.4g}}</extra>'; }}
  }}
  Plotly.restyle(plotDiv, {{'hovertemplate': base + ext}}, idxs);
}}

/* Update REJ tooltip: RA/Dec/G + cluster label (+ feature if active). */
function updateRejHoverTemplate(feature) {{
  var idxs = _rejIndices();
  if (!idxs.length) return;
  var base = 'RA=%{{customdata[1]:.4f}}<br>Dec=%{{customdata[2]:+.4f}}<br>G=%{{customdata[3]:.2f}}';
  var ext;
  if (feature === 'clusters') {{
    ext = '<extra>%{{customdata[4]}}</extra>';
  }} else {{
    var fi = featureNames.indexOf(feature);
    if (fi < 0) {{ ext = '<extra>%{{customdata[4]}}</extra>'; }}
    else {{ ext = '<extra>%{{customdata[4]}}<br>' + feature + ': %{{customdata[' + (5 + fi) + ']:.4g}}</extra>'; }}
  }}
  Plotly.restyle(plotDiv, {{'hovertemplate': base + ext}}, idxs);
}}

function recolorQ1Base(feature) {{
  var idxs = _q1BaseIndices();
  if (!idxs.length) return;
  if (feature === 'clusters') {{
    var colors = [], opacities = [], showscales = [];
    idxs.forEach(function(i) {{
      var nm = plotDiv.data[i].name;
      var mCl = nm.match(/^Q1 Cluster (\\d+)$/);
      colors.push(mCl ? CLUSTER_COLORS_JS[parseInt(mCl[1]) % CLUSTER_COLORS_JS.length] : '#2a2a2a');
      opacities.push(mCl ? 0.15 : 0.5);
      showscales.push(false);
    }});
    Plotly.restyle(plotDiv, {{'marker.color': colors, 'marker.opacity': opacities,
                              'marker.showscale': showscales}}, idxs);
  }} else {{
    var fi = featureNames.indexOf(feature);
    if (fi < 0) return;
    var rng = featureRanges[feature] || [0, 1];
    var colorArrays = [], showscales = [];
    idxs.forEach(function(i, ii) {{
      var cd = plotDiv.data[i].customdata || [];
      colorArrays.push(cd.map(function(row) {{
        var v = row ? row[fi] : null;
        return (v !== null && v !== undefined && isFinite(v)) ? v : null;
      }}));
      showscales.push(ii === 0);
    }});
    Plotly.restyle(plotDiv, {{
      'marker.color':              colorArrays,
      'marker.colorscale':         [COLORSCALE_RDBU],
      'marker.cmin':               [rng[0]],
      'marker.cmax':               [rng[1]],
      'marker.opacity':            [0.5],
      'marker.showscale':          showscales,
      'marker.colorbar.x':         0.02,
      'marker.colorbar.xanchor':   'left',
      'marker.colorbar.len':       0.4,
      'marker.colorbar.thickness': 10,
      'marker.colorbar.tickfont':  {{'color': '#555', 'size': 8}},
    }}, idxs);
  }}
  updatePhzHoverTemplate(feature);
  updateRejHoverTemplate(feature);
}}

document.getElementById('color-mode-select').addEventListener('change', function() {{
  recolorQ1Base(this.value);
}});

/* ── trace map & visibility ── */
function buildTraceMap() {{
  var baseCluster = {{}}, phzLegend = {{}}, phzSub = {{}}, rejTrace = {{}};
  plotDiv.data.forEach(function(tr, i) {{
    var nm = tr.name;
    var mSub = nm.match(/^phz=(\\d+) C(.+)$/);
    var mCl  = nm.match(/^Q1 Cluster (\\d+)$/);
    var mPhz = nm.match(/^phz=(\\d+)/);
    var mRej = nm.match(/^Rejected C(\\d+)$/);
    if (mSub) {{
      if (!phzSub[mSub[1]]) phzSub[mSub[1]] = {{}};
      phzSub[mSub[1]][mSub[2]] = i;
    }} else if (mCl) {{
      baseCluster[mCl[1]] = i;
    }} else if (mPhz) {{
      phzLegend[mPhz[1]] = i;
    }} else if (mRej) {{
      rejTrace[mRej[1]] = i;
    }}
  }});
  return {{baseCluster: baseCluster, phzLegend: phzLegend, phzSub: phzSub, rejTrace: rejTrace}};
}}

function applyVisibility() {{
  var map = buildTraceMap();
  var n = plotDiv.data.length;
  var vis = new Array(n).fill(true);

  /* First: completely hide all traces belonging to the inactive overlay */
  if (activeOverlay === 'qso') {{
    /* hide all REJ traces — false removes them from legend entirely */
    Object.keys(map.rejTrace).forEach(function(k) {{ vis[map.rejTrace[k]] = false; }});
    plotDiv.data.forEach(function(tr, i) {{
      if (tr.name === 'Rejected (unassigned)') vis[i] = false;
    }});
  }} else {{
    /* hide all QSO traces — false removes them from legend entirely */
    Object.keys(map.phzLegend).forEach(function(v) {{ vis[map.phzLegend[v]] = false; }});
    Object.keys(map.phzSub).forEach(function(v) {{
      Object.keys(map.phzSub[v]).forEach(function(k) {{ vis[map.phzSub[v][k]] = false; }});
    }});
  }}

  /* Then: apply isolation/hidden logic for the active overlay */
  if (activeOverlay === 'qso') {{
    if (isolatedCluster !== null) {{
      var ck = String(isolatedCluster);
      Object.keys(map.baseCluster).forEach(function(k) {{
        if (k !== ck) vis[map.baseCluster[k]] = 'legendonly';
      }});
      Object.keys(map.phzLegend).forEach(function(v) {{ vis[map.phzLegend[v]] = 'legendonly'; }});
      Object.keys(map.phzSub).forEach(function(v) {{
        Object.keys(map.phzSub[v]).forEach(function(k) {{
          if (k !== ck || hiddenPhz[v]) vis[map.phzSub[v][k]] = 'legendonly';
        }});
      }});
    }} else if (isolatedPhz !== null) {{
      var cv = String(isolatedPhz);
      Object.keys(map.phzLegend).forEach(function(v) {{
        if (v !== cv) vis[map.phzLegend[v]] = 'legendonly';
      }});
      Object.keys(map.baseCluster).forEach(function(k) {{
        var hasPhz = map.phzSub[cv] && map.phzSub[cv][k] !== undefined;
        if (!hasPhz || hiddenClusters[k]) vis[map.baseCluster[k]] = 'legendonly';
      }});
      Object.keys(map.phzSub).forEach(function(v) {{
        Object.keys(map.phzSub[v]).forEach(function(k) {{
          if (v !== cv || hiddenClusters[k]) vis[map.phzSub[v][k]] = 'legendonly';
        }});
      }});
    }} else {{
      Object.keys(hiddenClusters).forEach(function(k) {{
        if (!hiddenClusters[k]) return;
        if (map.baseCluster[k] !== undefined) vis[map.baseCluster[k]] = 'legendonly';
        Object.keys(map.phzSub).forEach(function(v) {{
          if (map.phzSub[v][k] !== undefined) vis[map.phzSub[v][k]] = 'legendonly';
        }});
      }});
      Object.keys(hiddenPhz).forEach(function(v) {{
        if (!hiddenPhz[v]) return;
        if (map.phzLegend[v] !== undefined) vis[map.phzLegend[v]] = 'legendonly';
        Object.keys((map.phzSub[v] || {{}})).forEach(function(k) {{
          vis[map.phzSub[v][k]] = 'legendonly';
        }});
      }});
    }}
  }} else {{
    /* activeOverlay === 'rej' */
    if (isolatedCluster !== null) {{
      var ck = String(isolatedCluster);
      Object.keys(map.baseCluster).forEach(function(k) {{
        if (k !== ck) vis[map.baseCluster[k]] = 'legendonly';
      }});
      Object.keys(map.rejTrace).forEach(function(k) {{
        if (k !== ck) vis[map.rejTrace[k]] = 'legendonly';
      }});
    }} else {{
      Object.keys(hiddenClusters).forEach(function(k) {{
        if (!hiddenClusters[k]) return;
        if (map.baseCluster[k] !== undefined) vis[map.baseCluster[k]] = 'legendonly';
        if (map.rejTrace[k]    !== undefined) vis[map.rejTrace[k]]    = 'legendonly';
      }});
    }}
  }}

  Plotly.restyle(plotDiv, {{visible: vis}});
}}

function restoreAll() {{
  isolatedPhz = null; isolatedCluster = null;
  hiddenClusters = {{}}; hiddenPhz = {{}};
  applyVisibility();
}}

/* Single click: toggle hide/show
   Cmd+click:    isolate (or un-isolate if already isolated) */
plotDiv.on('plotly_legendclick', function(data) {{
  var traceName = data.data[data.expandedIndex].name;
  var cmd = data.event && (data.event.metaKey || data.event.ctrlKey);

  if (activeOverlay === 'qso') {{
    if (/^phz=\\d+ C/.test(traceName)) return false;  /* ignore sub-traces */
    var mCl  = traceName.match(/^Q1 Cluster (\\d+)$/);
    var mPhz = traceName.match(/^phz=(\\d+)/);
    if (!mCl && !mPhz) return false;
    if (cmd) {{
      hiddenClusters = {{}}; hiddenPhz = {{}};
      if (mCl) {{
        var k = mCl[1];
        if (isolatedCluster === k) {{ isolatedCluster = null; clearSheetMode(); }}
        else {{ isolatedCluster = k; isolatedPhz = null; sheetMode = true; sheetCluster = k; showClusterSheets(Number(k)); }}
      }} else if (mPhz) {{
        var v = mPhz[1];
        if (isolatedPhz === v) {{ isolatedPhz = null; clearSheetMode(); }}
        else {{ isolatedPhz = v; isolatedCluster = null; sheetMode = true; sheetPhz = v; showContactSheets(v); }}
      }}
    }} else {{
      if (mCl) {{
        var k = mCl[1];
        if (isolatedPhz !== null) {{
          hiddenClusters[k] = !hiddenClusters[k];
        }} else {{
          isolatedCluster = null;
          hiddenClusters[k] = !hiddenClusters[k];
        }}
      }} else if (mPhz) {{
        var v = mPhz[1];
        isolatedPhz = null;
        hiddenPhz[v] = !hiddenPhz[v];
      }}
    }}
  }} else {{
    /* activeOverlay === 'rej' */
    var mCl  = traceName.match(/^Q1 Cluster (\\d+)$/);
    var mRej = traceName.match(/^Rejected C(\\d+)$/);
    if (!mCl && !mRej) return false;  /* noise / unassigned — ignore */
    var k = mCl ? mCl[1] : mRej[1];
    if (cmd) {{
      hiddenClusters = {{}};
      if (isolatedCluster === k) {{ isolatedCluster = null; clearSheetMode(); }}
      else {{ isolatedCluster = k; sheetMode = true; sheetCluster = k; showRejContactSheet(Number(k)); }}
    }} else {{
      isolatedCluster = null;
      hiddenClusters[k] = !hiddenClusters[k];
    }}
  }}

  applyVisibility();
  return false;
}});

plotDiv.on('plotly_legenddoubleclick', function(data) {{
  return false;  /* suppress Plotly default; all logic is in legendclick */
}});

document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') {{
    restoreAll(); clearSheetMode();
    Plotly.relayout(plotDiv, {{'scene.camera': {{
      eye: {{x: 1.25, y: 1.25, z: 1.25}},
      up:  {{x: 0,    y: 0,    z: 1}},
      center: {{x: 0, y: 0,    z: 0}}
    }}}});
  }}
}});

plotDiv.addEventListener('dblclick', function(e) {{
  var anyHidden = Object.values(hiddenClusters).some(Boolean) ||
                  Object.values(hiddenPhz).some(Boolean);
  if (isolatedPhz === null && isolatedCluster === null && !anyHidden) return;
  var node = e.target;
  while (node) {{
    if (node.classList && node.classList.contains('legend')) return;
    node = node.parentNode;
  }}
  restoreAll(); clearSheetMode();
}});

/* ── hover events ── */
function showPoint(sid, info) {{
  infoDiv.textContent = info;
  if (cutouts[sid]) {{
    img.src = 'data:image/png;base64,' + cutouts[sid];
    img.style.display = 'block';
    noCutout.textContent = '';
  }} else {{
    img.style.display = 'none';
    noCutout.textContent = sid ? '(no cutout cached)' : '';
  }}
}}

function _selectedFeature() {{ return document.getElementById('color-mode-select').value; }}

/* Build sidebar content from a hovered/clicked Plotly point.
   Q1 base traces   → show cluster name + selected feature value (no cutout).
   PHZ sub-traces   → show phz label + RA/Dec/G + selected feature value + cutout.
   REJ traces       → show cluster label + RA/Dec/G + selected feature value + cutout. */
function showPointFromPt(pt) {{
  var nm = pt.data.name;
  var isQ1 = /^Q1 Cluster \\d+$/.test(nm) || nm === 'noise';
  if (isQ1) {{
    var feature = _selectedFeature();
    var lines = [nm];
    if (feature !== 'clusters' && pt.customdata) {{
      var fi = featureNames.indexOf(feature);
      var raw = (fi >= 0) ? pt.customdata[fi] : undefined;
      var val = (raw !== null && raw !== undefined && isFinite(raw))
                ? parseFloat(raw).toPrecision(4) : 'N/A';
      lines.push(feature + ': ' + val);
    }}
    showPoint('', lines.join('\\n'));
  }} else {{
    if (!pt.customdata) return;
    var cd   = pt.customdata;
    var sid  = cd[0] || '';
    var ra   = parseFloat(cd[1]);
    var dec  = parseFloat(cd[2]);
    var gmag = parseFloat(cd[3]);
    var lbl  = cd[4] || nm;
    var lines = [
      lbl,
      'RA=' + ra.toFixed(4) + '  Dec=' + (dec >= 0 ? '+' : '') + dec.toFixed(4),
      'G=' + gmag.toFixed(2)
    ];
    var feature = _selectedFeature();
    if (feature !== 'clusters') {{
      var fi   = featureNames.indexOf(feature);
      var fval = (fi >= 0) ? cd[5 + fi] : undefined;
      if (fval !== null && fval !== undefined && isFinite(fval)) {{
        lines.push(feature + ': ' + parseFloat(fval).toPrecision(4));
      }}
    }}
    showPoint(sid, lines.join('\\n'));
  }}
}}

plotDiv.on('plotly_hover', function(data) {{
  if (pinned) return;
  showPointFromPt(data.points[0]);
}});

plotDiv.on('plotly_unhover', function() {{
  if (pinned) return;
  img.style.display = 'none';
  infoDiv.textContent = '';
  noCutout.textContent = '';
}});

plotDiv.on('plotly_click', function(data) {{
  if (!data.points || !data.points.length) return;
  showPointFromPt(data.points[0]);
  pinned = true;
  pinBtn.textContent = 'Unpin';
  pinBtn.classList.add('pinned');
}});

/* ── fit plot to container ── */
function resizePlot() {{
  var container = document.getElementById('plot-container');
  Plotly.relayout(plotDiv, {{
    width:  container.clientWidth,
    height: container.clientHeight,
  }});
}}
window.addEventListener('load',   resizePlot);
window.addEventListener('resize', resizePlot);

/* ── drag-to-resize ── */
var isResizing = false;
handle.addEventListener('mousedown', function(e) {{
  isResizing = true;
  handle.classList.add('dragging');
  document.body.style.cursor = 'col-resize';
  document.body.style.userSelect = 'none';
  e.preventDefault();
}});
document.addEventListener('mousemove', function(e) {{
  if (!isResizing) return;
  var newWidth = window.innerWidth - e.clientX - handle.offsetWidth;
  newWidth = Math.max(160, Math.min(700, newWidth));
  sidebar.style.width = newWidth + 'px';
  resizePlot();
}});
document.addEventListener('mouseup', function() {{
  if (!isResizing) return;
  isResizing = false;
  handle.classList.remove('dragging');
  document.body.style.cursor = '';
  document.body.style.userSelect = '';
}});

/* Initialise: hide REJ traces on load (default overlay is QSO) */
applyVisibility();

/* Permanently suppress Rejected C* legend entries — Q1 Cluster entries already
   show the same cluster colours, so the Rejected labels are redundant. */
(function() {{
  var rejIdxs = [];
  plotDiv.data.forEach(function(tr, i) {{
    if (/^Rejected C\\d+$/.test(tr.name) || tr.name === 'Rejected (unassigned)') {{
      rejIdxs.push(i);
    }}
  }});
  if (rejIdxs.length) Plotly.restyle(plotDiv, {{showlegend: false}}, rejIdxs);
}})();

</script>
</body>
</html>"""

    out = output_dir / "q1_overlay_3d.html"
    out.write_text(html, encoding="utf-8")
    print(f"  -> {out}  ({n_cached} cutouts embedded)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combined interactive 3D UMAP with switchable QSO / Rejected overlay"
    )
    parser.add_argument("--3d", dest="make_3d", action="store_true",
                        help="Generate the 3D HTML from saved CSVs (no DB query). "
                             "Run default mode first to save projected data.")
    parser.add_argument("--max-points", type=int, default=500,
                        help="Max points per overlay (applied independently to QSO and REJ). "
                             "Sources with cached cutouts are always kept; remainder is "
                             "random-sampled. Default: no limit.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subsampling (default: 42).")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for any raster outputs (default: 150).")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Q1 UMAP (always needed) ──────────────────────────────────────────
    print("[1] Loading Q1 UMAP embedding...")
    q1 = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    q1_emb      = q1["embedding"]
    q1_gmag     = q1["g_mag"].astype(float) if "g_mag" in q1 else None
    q1_features = q1["X_raw"].astype(float) if "X_raw" in q1 else None
    print(f"  Q1 embedding: {q1_emb.shape[0]:,} points"
          + ("" if q1_gmag is not None
             else "  (no G mag in npz — z-axis unavailable for Q1 base)"))
    if q1_features is not None:
        print(f"  Q1 features:  {q1_features.shape[1]} columns (feature colouring enabled)")

    # ── Load cluster labels (always needed) ───────────────────────────────────
    print("\n[2] Loading cluster labels...")
    labels_path = ARRAYS_DIR / "q1_cluster_labels.npy"
    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found.")
        print("  Run first:  python src/compute_q1_clusters.py")
        return 1
    q1_labels = np.load(labels_path)
    cluster_ids = sorted(set(q1_labels[q1_labels >= 0]))
    print(f"  Loaded from: {labels_path}")
    print(f"  Cluster sizes: " +
          "  ".join(f"C{k}={(q1_labels==k).sum():,}" for k in cluster_ids))

    # ── --3d mode: load saved CSVs, generate HTML ─────────────────────────────
    if args.make_3d:
        qso_path = ARRAYS_DIR / "q1_qso_projected.csv"
        rej_path = ARRAYS_DIR / "q1_rej_projected.csv"
        for p in (qso_path, rej_path):
            if not p.exists():
                print(f"ERROR: {p} not found. Run without --3d first.")
                return 1

        print("\n[3] Loading saved QSO and Rejected data...")
        df_qso  = pd.read_csv(qso_path)
        df_rej  = pd.read_csv(rej_path)
        qso_emb = df_qso[["umap1", "umap2"]].values
        rej_emb = df_rej[["umap1", "umap2"]].values
        print(f"  {len(df_qso):,} QSO sources loaded")
        print(f"  {len(df_rej):,} Rejected sources loaded")

        print("\n[4] Loading cached cutout thumbnails and contact sheets...")
        cutout_cache_qso = load_cutout_cache_qso(CUTOUTS_DIR_QSO)
        qso_cs, qso_counts = load_contact_sheet_cache_qso(CUTOUTS_DIR_QSO)
        cutout_cache_rej = load_cutout_cache_rej(CUTOUTS_DIR_REJ)
        rej_cs, rej_counts = load_contact_sheet_cache_rej(CUTOUTS_DIR_REJ)

        if args.max_points is not None:
            if len(df_qso) > args.max_points:
                print(f"\n  Applying --max-points {args.max_points} to QSO...")
                df_qso  = _subsample_keep_cutouts(
                    df_qso, set(cutout_cache_qso.keys()), args.max_points, args.seed)
                qso_emb = df_qso[["umap1", "umap2"]].values
            if len(df_rej) > args.max_points:
                print(f"\n  Applying --max-points {args.max_points} to REJ...")
                df_rej  = _subsample_keep_cutouts(
                    df_rej, set(cutout_cache_rej.keys()), args.max_points, args.seed)
                rej_emb = df_rej[["umap1", "umap2"]].values

        print("\n[5] Generating combined 3D HTML...")
        qso_gmag = df_qso["phot_g_mean_mag"].values.astype(float)
        rej_gmag = df_rej["phot_g_mean_mag"].values.astype(float)
        plot_umap_3d_plotly(
            q1_emb, q1_labels, q1_gmag,
            qso_emb, df_qso["phz"].values, qso_gmag,
            rej_emb, df_rej["cluster"].values, rej_gmag,
            OUTPUT_DIR,
            cutout_cache_qso=cutout_cache_qso, df_qso=df_qso,
            cutout_cache_rej=cutout_cache_rej, df_rej=df_rej,
            qso_contact_sheet_cache=qso_cs, qso_sheet_counts=qso_counts,
            rej_contact_sheet_cache=rej_cs, rej_sheet_counts=rej_counts,
            q1_features=q1_features,
            feature_names=SELECTED_GAIA_PARAMETERS,
        )
        return 0

    # ── Default mode: query DB + project + save CSVs ──────────────────────────
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER environment variable not set.")
        return 1

    q1_mean = q1["mean"]
    q1_std  = q1["std"]
    reducer = joblib.load(ARRAYS_DIR / "q1_indiv_reducer.joblib")
    std_safe = np.where(q1_std > 0, q1_std, 1.0)

    print("\n[3] Querying QSO variants and Rejected sources from DB...")
    df_qso = query_qso_variants(username)
    df_rej = query_rejected(username)

    print("\n[4] Projecting QSO sources into UMAP space...")
    X_qso = df_qso[SELECTED_GAIA_PARAMETERS].values.astype(float)
    X_qso_scaled = np.where(np.isfinite((X_qso - q1_mean) / std_safe),
                            (X_qso - q1_mean) / std_safe, 0.0)
    qso_emb = reducer.transform(X_qso_scaled)
    print(f"  Projected {len(qso_emb):,} QSO sources")
    qso_cluster = assign_to_clusters(qso_emb, q1_emb, q1_labels)
    df_qso = df_qso.copy()
    df_qso["umap1"]   = qso_emb[:, 0]
    df_qso["umap2"]   = qso_emb[:, 1]
    df_qso["cluster"] = qso_cluster

    print("\n  QSO cluster assignment:")
    for k in cluster_ids:
        n = (qso_cluster == k).sum()
        if n:
            print(f"    C{k}: {n:,}")
    n_noise = (qso_cluster == -1).sum()
    if n_noise:
        print(f"    unassigned: {n_noise:,}")

    qso_path = ARRAYS_DIR / "q1_qso_projected.csv"
    df_qso.to_csv(qso_path, index=False)
    print(f"\n  QSO projected data saved → {qso_path}")

    print("\n[5] Projecting Rejected sources into UMAP space...")
    X_rej = df_rej[SELECTED_GAIA_PARAMETERS].values.astype(float)
    X_rej_scaled = np.where(np.isfinite((X_rej - q1_mean) / std_safe),
                            (X_rej - q1_mean) / std_safe, 0.0)
    rej_emb = reducer.transform(X_rej_scaled)
    print(f"  Projected {len(rej_emb):,} Rejected sources")
    rej_cluster = assign_to_clusters(rej_emb, q1_emb, q1_labels)
    df_rej = df_rej.copy()
    df_rej["umap1"]   = rej_emb[:, 0]
    df_rej["umap2"]   = rej_emb[:, 1]
    df_rej["cluster"] = rej_cluster

    print("\n  Rejected cluster assignment:")
    for k in cluster_ids:
        n = (rej_cluster == k).sum()
        if n:
            print(f"    C{k}: {n:,}")
    n_noise = (rej_cluster == -1).sum()
    if n_noise:
        print(f"    unassigned: {n_noise:,}")

    rej_path = ARRAYS_DIR / "q1_rej_projected.csv"
    df_rej.to_csv(rej_path, index=False)
    print(f"\n  Rejected projected data saved → {rej_path}")

    print("\n[6] Loading cached cutout thumbnails and contact sheets...")
    cutout_cache_qso = load_cutout_cache_qso(CUTOUTS_DIR_QSO)
    qso_cs, qso_counts = load_contact_sheet_cache_qso(CUTOUTS_DIR_QSO)
    cutout_cache_rej = load_cutout_cache_rej(CUTOUTS_DIR_REJ)
    rej_cs, rej_counts = load_contact_sheet_cache_rej(CUTOUTS_DIR_REJ)

    if args.max_points is not None:
        if len(df_qso) > args.max_points:
            print(f"\n  Applying --max-points {args.max_points} to QSO...")
            df_qso  = _subsample_keep_cutouts(
                df_qso, set(cutout_cache_qso.keys()), args.max_points, args.seed)
            qso_emb = df_qso[["umap1", "umap2"]].values
        if len(df_rej) > args.max_points:
            print(f"\n  Applying --max-points {args.max_points} to REJ...")
            df_rej  = _subsample_keep_cutouts(
                df_rej, set(cutout_cache_rej.keys()), args.max_points, args.seed)
            rej_emb = df_rej[["umap1", "umap2"]].values

    print("\n[7] Generating combined 3D HTML...")
    qso_gmag = df_qso["phot_g_mean_mag"].values.astype(float)
    rej_gmag = df_rej["phot_g_mean_mag"].values.astype(float)
    plot_umap_3d_plotly(
        q1_emb, q1_labels, q1_gmag,
        qso_emb, df_qso["phz"].values, qso_gmag,
        rej_emb, df_rej["cluster"].values, rej_gmag,
        OUTPUT_DIR,
        cutout_cache_qso=cutout_cache_qso, df_qso=df_qso,
        cutout_cache_rej=cutout_cache_rej, df_rej=df_rej,
        qso_contact_sheet_cache=qso_cs, qso_sheet_counts=qso_counts,
        rej_contact_sheet_cache=rej_cs, rej_sheet_counts=rej_counts,
        q1_features=q1_features,
        feature_names=SELECTED_GAIA_PARAMETERS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
