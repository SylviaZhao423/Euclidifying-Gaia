#!/usr/bin/env python3
"""
UMAP cluster plots for Q1 QSO-variant sources.

Produces:
  q1_qso.png    — 2D: Q1 coloured by cluster + QSO overlaid
  q1_qso_3d.html — interactive 3D: UMAP x/y + G magnitude as z

Reads:
  output/umap/arrays/q1_indiv.npz
  output/umap/arrays/q1_indiv_reducer.joblib
  output/umap/arrays/q1_cluster_labels.npy
  output/umap/arrays/q1_allphz.npz

Writes:
  output/umap/q1_individual/q1_qso.png
  output/umap/q1_individual/q1_qso_3d.html
  output/umap/arrays/q1_qso_projected.csv

Usage:
  # Full run: load npz + project + save CSV + 2D PNG + 3D HTML
  python src/analyze_umap_q1_qso_3d.py
  python src/analyze_umap_q1_qso_3d.py --max-points 5000

  # Fast re-render: skip projection, regenerate 3D HTML from saved CSV
  python src/analyze_umap_q1_qso_3d.py --3d
  python src/analyze_umap_q1_qso_3d.py --3d --max-points 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

PROJECT_DIR  = Path(__file__).resolve().parents[1]
ARRAYS_DIR   = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR   = PROJECT_DIR / "output" / "umap" / "q1_individual"
CUTOUTS_DIR  = PROJECT_DIR / "output" / "euclid_cutouts" / "q1_cutouts" / "qso"

MAG_CUT   = 17

QSO_LABELS = {
    4: ("QSO",              "cyan"),
    5: ("Multi (Star ∩ QSO)", "yellow"),
    6: ("Multi (Gal ∩ QSO)",  "#F45347"),
    7: ("Multi (all three)", "orange"),
}

PHZ_MARKERS = {4: "o", 5: "s", 6: "D", 7: "+"}
# Plotly 3D symbol equivalents (3D has no triangle/pentagon)
PHZ_PLOTLY_SYMBOLS = {4: "circle", 5: "square", 6: "diamond", 7: "cross"}

# Cluster colours — used by both the 2D and 3D plots.
# Edit this list to change cluster colours everywhere.
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


def load_qso_from_npz() -> pd.DataFrame:
    """Load QSO-variant sources (phz=4/5/6/7) from q1_allphz.npz."""
    data = np.load(ARRAYS_DIR / "q1_allphz.npz", allow_pickle=True)
    df = pd.DataFrame(data["X_features"], columns=SELECTED_GAIA_PARAMETERS)
    df["source_id"]       = data["source_ids"].astype(str)
    df["ra"]              = data["ra"].astype(float)
    df["dec"]             = data["dec"].astype(float)
    df["phot_g_mean_mag"] = data["phot_g_mean_mag"].astype(float)
    df["phz"]             = data["phz"].astype(int)
    df = df[df["phz"].isin([4, 5, 6, 7])].reset_index(drop=True)
    print(f"  -> {len(df):,} QSO-variant sources (from q1_allphz.npz)")
    for v, (lbl, _) in QSO_LABELS.items():
        n = (df["phz"] == v).sum()
        if n:
            print(f"     phz={v}  {lbl}: {n:,}")
    return df



def assign_to_clusters(qso_emb: np.ndarray,
                       q1_emb: np.ndarray,
                       q1_labels: np.ndarray) -> np.ndarray:
    valid_mask = q1_labels >= 0
    if valid_mask.sum() == 0:
        return np.full(len(qso_emb), -1, dtype=int)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(q1_emb[valid_mask])
    _, idx = nn.kneighbors(qso_emb)
    idx = idx.ravel()
    valid_indices = np.where(valid_mask)[0]
    return q1_labels[valid_indices[idx]]



def plot_q1_clusters(q1_emb, q1_labels, qso_emb, qso_phz,
                       qso_cluster, output_dir):
    """2D: Q1 coloured by cluster + QSO variants overlaid."""
    import matplotlib.lines as mlines
    cluster_ids = sorted(set(q1_labels[q1_labels >= 0]))

    # ── Shared style constants ─────────────────────────────────────────────────
    # All phz scatter calls share these kwargs (only c= and marker= differ per call).
    PHZ_EDGE       = "k"
    PHZ_EDGE_WIDTH = 0.3          # used in both scatter (linewidths) and legend (markeredgewidth)
    PHZ_SIZE_PT    = 18           # scatter s=
    PHZ_SIZE_LEG   = 5            # legend markersize

    BG_KW  = dict(c="#E0E0E0", s=2, alpha=0.4, lw=0)   # noise / Q1-base background
    CL_KW  = dict(s=2, alpha=0.4, lw=0)                  # cluster scatter (add c= per call)
    PHZ_KW = dict(s=PHZ_SIZE_PT, edgecolors=PHZ_EDGE, linewidths=PHZ_EDGE_WIDTH, zorder=5)

    def _lh(face, edge="none", marker="o", size=PHZ_SIZE_LEG, label=""):
        """Create a legend Line2D handle using the same edge style as scatter."""
        return mlines.Line2D([], [], markerfacecolor=face,
                             markeredgecolor=edge, markeredgewidth=PHZ_EDGE_WIDTH,
                             marker=marker, markersize=size,
                             linestyle="None", label=label)

    fig, axes = plt.subplots(1, 2, figsize=(23, 9))

    # ── Left: Q1 base coloured by cluster ─────────────────────────────────────
    ax = axes[0]
    noise_mask = q1_labels == -1
    if noise_mask.any():
        ax.scatter(q1_emb[noise_mask, 0], q1_emb[noise_mask, 1], **BG_KW)
    for k in cluster_ids:
        m = q1_labels == k
        ax.scatter(q1_emb[m, 0], q1_emb[m, 1],
                   c=CLUSTER_COLORS[k % len(CLUSTER_COLORS)], **CL_KW)
    for v, (lbl, col) in QSO_LABELS.items():
        m = qso_phz == v
        if m.any():
            ax.scatter(qso_emb[m, 0], qso_emb[m, 1],
                       c=col, marker=PHZ_MARKERS.get(v, "o"), alpha=1, **PHZ_KW)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title("Q1 UMAP — clusters + QSO variants", fontweight="bold")

    # Manual 3-col legend (row-first visual order via transpose trick)
    leg_h, leg_l = [], []
    if noise_mask.any():
        leg_h.append(_lh("#E0E0E0", size=3.5)); leg_l.append("noise")
    for k in cluster_ids:
        leg_h.append(_lh(CLUSTER_COLORS[k % len(CLUSTER_COLORS)], size=3.5))
        leg_l.append(f"Cluster {k}")
    for v, (lbl, col) in QSO_LABELS.items():
        m_v = qso_phz == v
        if m_v.any():
            leg_h.append(_lh(col, edge=PHZ_EDGE, marker=PHZ_MARKERS.get(v, "o")))
            leg_l.append(f"phz={v} {lbl} (n={m_v.sum():,})")
    ncols = 3
    nrows = (len(leg_h) + ncols - 1) // ncols
    pad   = nrows * ncols - len(leg_h)
    ph = leg_h + [mlines.Line2D([], [], linestyle="None")] * pad
    pl = leg_l + [""] * pad
    row_h = [ph[ci * nrows + ri] for ri in range(nrows) for ci in range(ncols)]
    row_l = [pl[ci * nrows + ri] for ri in range(nrows) for ci in range(ncols)]
    ax.legend(row_h, row_l, fontsize=8, loc="lower left", ncol=ncols)
    ax.grid(alpha=0.2)

    # ── Right: QSO coloured by assigned cluster, shape encodes phz ────────────
    ax = axes[1]
    ax.scatter(q1_emb[:, 0], q1_emb[:, 1], **BG_KW)
    unassigned_color = CLUSTER_COLORS[len(cluster_ids) % len(CLUSTER_COLORS)]
    for k in cluster_ids:
        col_k = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        for v in QSO_LABELS:
            m = (qso_cluster == k) & (qso_phz == v)
            if m.any():
                ax.scatter(qso_emb[m, 0], qso_emb[m, 1],
                           c=col_k, marker=PHZ_MARKERS.get(v, "o"), alpha=1, **PHZ_KW)
    unassigned = qso_cluster == -1
    if unassigned.any():
        ax.scatter(qso_emb[unassigned, 0], qso_emb[unassigned, 1],
                   c=unassigned_color, s=PHZ_SIZE_PT, marker="x", zorder=5)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title("QSO variants — cluster (colour) × phz (shape)", fontweight="bold")

    # Combined 2-col legend: col 1 = cluster colors, col 2 = phz shapes
    cluster_handles = [_lh("#E0E0E0", size=3.5, label="Q1 base")]
    cluster_handles += [_lh(CLUSTER_COLORS[k % len(CLUSTER_COLORS)], edge=PHZ_EDGE, size=5, label=f"Cluster {k}")
                        for k in cluster_ids]
    if unassigned.any():
        cluster_handles.append(_lh(unassigned_color, edge=PHZ_EDGE, marker="x", size=5, label="unassigned"))
    phz_handles = [_lh("none", edge="k", marker=PHZ_MARKERS.get(v, "o"),
                        label=f"phz={v} {QSO_LABELS[v][0]}")
                   for v in QSO_LABELS if (qso_phz == v).any()]
    _blank = mlines.Line2D([], [], linestyle="None", label="")
    _nrows = max(len(cluster_handles), len(phz_handles))
    _col1 = cluster_handles + [_blank] * (_nrows - len(cluster_handles))
    _col2 = phz_handles    + [_blank] * (_nrows - len(phz_handles))
    ax.legend(handles=_col1 + _col2, fontsize=8, loc="lower left", ncol=2)
    ax.grid(alpha=0.2)

    fig.subplots_adjust(wspace=0.15)
    out = output_dir / "q1_qso.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")


def load_cutout_cache(cutouts_dir: Path) -> dict:
    """Scan cluster*/manifest_phz*.csv files and return {source_id_str: base64_png_str}.

    Only sources with a saved cutout_file entry are included.
    Images are embedded at full resolution.
    """
    import base64
    from io import BytesIO
    import matplotlib.image as mpimg
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
    print(f"  Loaded {len(cache)} cached cutout thumbnails")
    return cache


def load_contact_sheet_cache(cutouts_dir: Path) -> tuple[dict, dict]:
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
            # Count rows in the corresponding manifest (subtract 1 for header)
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
    print(f"  Loaded {n_sheets} contact sheet(s) for phz values: {sorted(cache.keys())}")
    return cache, counts


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


def plot_umap_3d_plotly(q1_emb, q1_labels, q1_gmag,
                        qso_emb, qso_phz, qso_gmag,
                        output_dir, cutout_cache=None, df_qso=None,
                        contact_sheet_cache=None, sheet_counts=None,
                        q1_features=None, feature_names=None):
    """Interactive 3D: UMAP x/y + G magnitude as z."""
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

    b64_js = {}
    # Cluster assignments for QSO points (used to split phz sub-traces by cluster)
    _qso_cl = df_qso["cluster"].values if (df_qso is not None and "cluster" in df_qso.columns) else None

    for v, (lbl, col) in QSO_LABELS.items():
        m_v = qso_phz == v
        if not m_v.any():
            continue

        mkr = dict(size=5, color=col, symbol=PHZ_PLOTLY_SYMBOLS.get(v, "circle"),
                   opacity=0.95, line=dict(width=0.3, color="rgba(255,255,255,0.25)"))

        # Legend-entry trace: single NaN point (invisible) so Plotly shows the legend entry
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
                    # feature values (indices 5…5+n_feat-1 in customdata)
                    feats: list = []
                    for fn in (feature_names or []):
                        try:
                            fv = float(row.get(fn, float("nan")))
                            feats.append(None if not np.isfinite(fv) else round(fv, 4))
                        except (ValueError, TypeError):
                            feats.append(None)
                    custom.append([sid, round(ra, 4), round(dec, 4),
                                   round(gmag, 4), label] + feats)
                    if cutout_cache and sid in cutout_cache:
                        b64_js[sid] = cutout_cache[sid]
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

    # ── Fixed axis ranges (computed from all data so hiding traces never rescales) ──
    _pad = 0.05
    _all_x = np.concatenate([q1_emb[:, 0], qso_emb[:, 0]])
    _all_y = np.concatenate([q1_emb[:, 1], qso_emb[:, 1]])
    _xr = [float(_all_x.min() - (np.ptp(_all_x) * _pad)),
           float(_all_x.max() + (np.ptp(_all_x) * _pad))]
    _yr = [float(_all_y.min() - (np.ptp(_all_y) * _pad)),
           float(_all_y.max() + (np.ptp(_all_y) * _pad))]
    if q1_gmag is not None and qso_gmag is not None:
        _gz = np.concatenate([q1_gmag[np.isfinite(q1_gmag)], qso_gmag[np.isfinite(qso_gmag)]])
        _zpad = float(np.ptp(_gz) * _pad)
        _zr = [float(_gz.max()) + _zpad, float(_gz.min()) - _zpad]   # reversed: faint→bright
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
    b64_json = json.dumps(b64_js)
    n_cached = len(b64_js)
    contact_sheets_json = json.dumps(contact_sheet_cache or {})
    sheet_counts_json   = json.dumps(sheet_counts or {})
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
<title>Euclid Q1 — QSO UMAP</title>
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
    font-size: 9px; letter-spacing: 0.15em; text-transform: uppercase;
    color: #444; margin-bottom: 5px;
  }}
  #sidebar-heading {{
    font-size: 13px; font-weight: 600; color: #888; line-height: 1.3;
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
    font-size: 9px; letter-spacing: 0.1em; text-transform: uppercase;
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
    color: #444; white-space: nowrap;
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
</style>
</head>
<body>
<div id="plot-container">{plot_div}</div>
<div id="resize-handle"></div>
<div id="sidebar">
  <div id="sidebar-header">
    <div id="sidebar-eyebrow">Euclid Q1 · UMAP</div>
    <div id="sidebar-heading">QSO Variants</div>
    <div id="sidebar-hint">hover a point · click to pin</div>
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
var cutouts = {b64_json};
var contact_sheets = {contact_sheets_json};
var sheet_counts = {sheet_counts_json};
/* cluster_sheets: invert contact_sheets → {{cluster: {{phz: base64}}}} */
var cluster_sheets = {{}};
Object.keys(contact_sheets).forEach(function(v) {{
  Object.keys(contact_sheets[v]).forEach(function(k) {{
    if (!cluster_sheets[k]) cluster_sheets[k] = {{}};
    cluster_sheets[k][v] = contact_sheets[v][k];
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
var pinned = false;
var sheetMode = false;
var sheetPhz = null;
var sheetCluster = null;
var isolatedPhz = null;
var isolatedCluster = null;
var hiddenClusters = {{}};
var hiddenPhz = {{}};
var _clickTimer = null;
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

/* ── feature coloring ── */
function _q1BaseIndices() {{
  var out = [];
  plotDiv.data.forEach(function(tr, i) {{
    if (/^Q1 Cluster \d+$/.test(tr.name) || tr.name === 'noise') out.push(i);
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

/* Update PHZ tooltip: RA/Dec/G on the left, phz label (+ feature if active) on the right.
   Called whenever the colour mode changes so the extra panel stays in sync. */
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

function recolorQ1Base(feature) {{
  var idxs = _q1BaseIndices();
  if (!idxs.length) return;
  if (feature === 'clusters') {{
    var colors = [], opacities = [], showscales = [];
    idxs.forEach(function(i) {{
      var nm = plotDiv.data[i].name;
      var mCl = nm.match(/^Q1 Cluster (\d+)$/);
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
}}

document.getElementById('color-mode-select').addEventListener('change', function() {{
  recolorQ1Base(this.value);
}});

function _countTotal(countsObj) {{
  /* sum all integer values in a {{key: int}} object */
  return Object.values(countsObj).reduce(function(a,b){{return a+b;}}, 0);
}}

function showContactSheets(v) {{
  /* phz isolation → show all clusters for this phz */
  var clusters = contact_sheets[v];
  if (!clusters || Object.keys(clusters).length === 0) {{
    sheetTitle.innerHTML = 'phz=' + v + ' — no sheets cached';
    sheetImages.innerHTML = '';
    sheetView.style.display = 'block';
    return;
  }}
  var keys = Object.keys(clusters).map(Number).sort(function(a,b){{return a-b;}});
  var phzCounts = sheet_counts[v] || {{}};
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
  /* cluster isolation → show all phz for this cluster */
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
    var n = (sheet_counts[String(v)] || {{}})[String(k)];
    n = (n !== undefined) ? n : '?';
    if (typeof n === 'number') total += n;
    html += '<div class="sheet-cluster-label">phz=' + v + ' (' + n + ' sources)</div>';
    html += '<img src="data:image/png;base64,' + phzs[String(v)] + '">';
  }});
  sheetTitle.innerHTML = 'Cluster ' + k + ' contact sheets<br>' + total + ' sources';
  sheetImages.innerHTML = html;
  sheetView.style.display = 'block';
}}

sheetClose.addEventListener('click', function() {{ clearSheetMode(); }});

/* ── trace map & visibility ── */
function buildTraceMap() {{
  var baseCluster = {{}}, phzLegend = {{}}, phzSub = {{}};
  plotDiv.data.forEach(function(tr, i) {{
    var nm = tr.name;
    var mSub = nm.match(/^phz=(\\d+) C(.+)$/);
    var mCl  = nm.match(/^Q1 Cluster (\\d+)$/);
    var mPhz = nm.match(/^phz=(\\d+)/);
    if (mSub) {{
      if (!phzSub[mSub[1]]) phzSub[mSub[1]] = {{}};
      phzSub[mSub[1]][mSub[2]] = i;
    }} else if (mCl) {{
      baseCluster[mCl[1]] = i;
    }} else if (mPhz) {{
      phzLegend[mPhz[1]] = i;
    }}
  }});
  return {{baseCluster: baseCluster, phzLegend: phzLegend, phzSub: phzSub}};
}}

function applyVisibility() {{
  var map = buildTraceMap();
  var n = plotDiv.data.length;
  var vis = new Array(n).fill(true);

  if (isolatedCluster !== null) {{
    /* Cluster isolated: hide other clusters; phz legend always hidden;
       within the isolated cluster, also respect hiddenPhz. */
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
    /* Phz isolated: hide other phz; within isolated phz, respect hiddenClusters. */
    var cv = String(isolatedPhz);
    Object.keys(map.phzLegend).forEach(function(v) {{
      if (v !== cv) vis[map.phzLegend[v]] = 'legendonly';
    }});
    Object.keys(map.baseCluster).forEach(function(k) {{
      /* hide clusters that have no data for this phz, or are explicitly hidden */
      var hasPhz = map.phzSub[cv] && map.phzSub[cv][k] !== undefined;
      if (!hasPhz || hiddenClusters[k]) vis[map.baseCluster[k]] = 'legendonly';
    }});
    Object.keys(map.phzSub).forEach(function(v) {{
      Object.keys(map.phzSub[v]).forEach(function(k) {{
        if (v !== cv || hiddenClusters[k]) vis[map.phzSub[v][k]] = 'legendonly';
      }});
    }});

  }} else {{
    /* Normal mode: apply hiddenClusters and hiddenPhz independently. */
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
  Plotly.restyle(plotDiv, {{visible: vis}});
}}

function restoreAll() {{
  isolatedPhz = null; isolatedCluster = null;
  hiddenClusters = {{}}; hiddenPhz = {{}};
  applyVisibility();
}}

/* Single click: toggle hide/show
   Cmd+click:    isolate (or un-isolate if already isolated)
   No debounce needed — Cmd distinguishes the two actions instantly. */
plotDiv.on('plotly_legendclick', function(data) {{
  var traceName = data.data[data.expandedIndex].name;
  if (/^phz=\\d+ C/.test(traceName)) return false;  /* ignore sub-traces */
  var mCl  = traceName.match(/^Q1 Cluster (\\d+)$/);
  var mPhz = traceName.match(/^phz=(\\d+)/);
  var cmd = data.event && (data.event.metaKey || data.event.ctrlKey);
  if (cmd) {{
    /* Cmd+click → isolate / un-isolate */
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
    /* Plain click → toggle.
       If a cluster is isolated and user clicks a phz entry, keep the isolation
       and just toggle hiddenPhz within it (and vice versa). Only clear the
       isolation when clicking the same type as what is isolated. */
    if (mCl) {{
      var k = mCl[1];
      if (isolatedPhz !== null) {{
        /* Within phz isolation: toggle this cluster (base + its phz sub-trace)
           without clearing the phz isolation. */
        hiddenClusters[k] = !hiddenClusters[k];
      }} else {{
        isolatedCluster = null; /* exit cluster isolation — now in normal toggle mode */
        hiddenClusters[k] = !hiddenClusters[k];
      }}
    }} else if (mPhz) {{
      var v = mPhz[1];
      isolatedPhz = null;    /* clicking phz clears phz isolation */
      /* keep isolatedCluster — user is filtering phz inside an isolated cluster */
      hiddenPhz[v] = !hiddenPhz[v];
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
   Q1 base traces  → show cluster name + selected feature value (no cutout).
   PHZ sub-traces  → show phz label + RA/Dec/G + selected feature value + cutout. */
function showPointFromPt(pt) {{
  var nm = pt.data.name;
  var isQ1 = /^Q1 Cluster \d+$/.test(nm) || nm === 'noise';
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
    var phzLines = [
      lbl,
      'RA=' + ra.toFixed(4) + '  Dec=' + (dec >= 0 ? '+' : '') + dec.toFixed(4),
      'G=' + gmag.toFixed(2)
    ];
    var feature = _selectedFeature();
    if (feature !== 'clusters') {{
      var fi   = featureNames.indexOf(feature);
      var fval = (fi >= 0) ? cd[5 + fi] : undefined;
      if (fval !== null && fval !== undefined && isFinite(fval)) {{
        phzLines.push(feature + ': ' + parseFloat(fval).toPrecision(4));
      }}
    }}
    showPoint(sid, phzLines.join('\\n'));
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

/* clicking a point pins it */
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
  /* sidebar width = distance from mouse to right edge of viewport */
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

</script>
</body>
</html>"""

    out = output_dir / "q1_qso_3d.html"
    out.write_text(html, encoding="utf-8")
    print(f"  -> {out}  ({n_cached} cutouts embedded)")



def main() -> int:
    parser = argparse.ArgumentParser(description="UMAP cluster plots for Q1 QSO variants")
    parser.add_argument("--3d", dest="make_3d", action="store_true",
                        help="Generate the interactive 3D HTML using saved QSO data "
                             "(no DB query). Run the default script first to save QSO data.")
    parser.add_argument("--max-points", type=int, default=None,
                        help="Max QSO sources in 3D plot. Sources with cached cutouts "
                             "are always kept; remainder is random-sampled. "
                             "Default: no limit (all sources plotted).")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Q1 UMAP (always needed) ──────────────────────────────────────────
    print("[1] Loading Q1 UMAP embedding...")
    q1 = np.load(ARRAYS_DIR / "q1_indiv.npz", allow_pickle=True)
    q1_emb      = q1["embedding"]
    q1_gmag     = q1["g_mag"].astype(float) if "g_mag" in q1 else None
    q1_features = q1["X_raw"].astype(float) if "X_raw" in q1 else None
    print(f"  Q1 embedding: {q1_emb.shape[0]:,} points"
          + ("" if q1_gmag is not None else "  (no G mag in npz — z-axis unavailable for Q1 base)"))
    if q1_features is not None:
        print(f"  Q1 features:  {q1_features.shape[1]} columns (feature colouring enabled)")

    # ── --3d mode: load saved data, generate 3D HTML, done ────────────────────
    if args.make_3d:
        qso_path = ARRAYS_DIR / "q1_qso_projected.csv"
        labels_path = ARRAYS_DIR / "q1_cluster_labels.npy"
        if not qso_path.exists():
            print(f"ERROR: {qso_path} not found. Run without --3d first.")
            return 1
        if not labels_path.exists():
            print(f"ERROR: {labels_path} not found. Run without --3d first.")
            return 1
        print("\n[2] Loading saved cluster labels and QSO data...")
        q1_labels = np.load(labels_path)
        df_qso    = pd.read_csv(qso_path)
        qso_emb   = df_qso[["umap1", "umap2"]].values
        print(f"  {len(df_qso):,} QSO sources loaded")
        print("\n[3] Loading cached cutout thumbnails and contact sheets...")
        cutout_cache        = load_cutout_cache(CUTOUTS_DIR)
        contact_sheet_cache, sheet_counts = load_contact_sheet_cache(CUTOUTS_DIR)
        if args.max_points is not None and len(df_qso) > args.max_points:
            print(f"\n  Applying --max-points {args.max_points}...")
            df_qso = _subsample_keep_cutouts(
                df_qso, set(cutout_cache.keys()), args.max_points)
            qso_emb = df_qso[["umap1", "umap2"]].values
        print("\n[4] Saving interactive 3D plot...")
        qso_gmag = df_qso["phot_g_mean_mag"].values.astype(float)
        plot_umap_3d_plotly(q1_emb, q1_labels, q1_gmag,
                            qso_emb, df_qso["phz"].values, qso_gmag,
                            OUTPUT_DIR,
                            cutout_cache=cutout_cache, df_qso=df_qso,
                            contact_sheet_cache=contact_sheet_cache,
                            sheet_counts=sheet_counts,
                            q1_features=q1_features,
                            feature_names=SELECTED_GAIA_PARAMETERS)
        return 0

    # ── Default mode: full run (load npz + 2D plot + save data) ────────────────
    q1_mean = q1["mean"]
    q1_std  = q1["std"]
    reducer = joblib.load(ARRAYS_DIR / "q1_indiv_reducer.joblib")

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
          "  ".join(f"C{k}={( q1_labels==k).sum():,}" for k in cluster_ids))

    print("\n[3] Loading QSO variants from npz...")
    df_qso = load_qso_from_npz()

    print("\n[4] Projecting QSO sources into UMAP space...")
    X = df_qso[SELECTED_GAIA_PARAMETERS].values.astype(float)
    std_safe = np.where(q1_std > 0, q1_std, 1.0)
    X_scaled = (X - q1_mean) / std_safe
    X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0.0)
    qso_emb  = reducer.transform(X_scaled)
    print(f"  Projected {len(qso_emb):,} QSO sources")

    qso_cluster = assign_to_clusters(qso_emb, q1_emb, q1_labels)
    df_qso = df_qso.copy()
    df_qso["umap1"]   = qso_emb[:, 0]
    df_qso["umap2"]   = qso_emb[:, 1]
    df_qso["cluster"] = qso_cluster

    qso_path = ARRAYS_DIR / "q1_qso_projected.csv"
    df_qso.to_csv(qso_path, index=False)
    print(f"  QSO projected data saved → {qso_path}")

    print("\n[5] Saving 2D cluster plot...")
    plot_q1_clusters(q1_emb, q1_labels, qso_emb,
                     df_qso["phz"].values, qso_cluster, OUTPUT_DIR)

    print("\n[6] Loading cached cutout thumbnails and contact sheets...")
    cutout_cache        = load_cutout_cache(CUTOUTS_DIR)
    contact_sheet_cache, sheet_counts = load_contact_sheet_cache(CUTOUTS_DIR)
    if args.max_points is not None and len(df_qso) > args.max_points:
        print(f"\n  Applying --max-points {args.max_points}...")
        df_qso  = _subsample_keep_cutouts(
            df_qso, set(cutout_cache.keys()), args.max_points)
        qso_emb = df_qso[["umap1", "umap2"]].values

    print("\n[7] Saving interactive 3D plot...")
    qso_gmag = df_qso["phot_g_mean_mag"].values.astype(float)
    plot_umap_3d_plotly(q1_emb, q1_labels, q1_gmag,
                        qso_emb, df_qso["phz"].values, qso_gmag,
                        OUTPUT_DIR,
                        cutout_cache=cutout_cache, df_qso=df_qso,
                        contact_sheet_cache=contact_sheet_cache,
                        sheet_counts=sheet_counts,
                        q1_features=q1_features,
                        feature_names=SELECTED_GAIA_PARAMETERS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
