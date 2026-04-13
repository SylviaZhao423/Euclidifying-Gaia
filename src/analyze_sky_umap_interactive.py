#!/usr/bin/env python3
"""
Interactive sky map + linked 3D UMAP visualization.

Uses a static Gaia DR3 sky image (Hammer-Aitoff, galactic coords) with
CSS transform-based pan/zoom for instant, lag-free interaction.
Data points are drawn on a canvas that transforms with the image.

Usage:
    python src/analyze_sky_umap_interactive.py
"""

from __future__ import annotations

import argparse
import base64
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARRAYS_DIR  = PROJECT_DIR / "output" / "umap" / "arrays"
OUTPUT_DIR  = PROJECT_DIR / "output" / "umap"

RANDOM_CSV = ARRAYS_DIR / "xgb_gaia_projected.csv"
HILAT_CSV  = ARRAYS_DIR / "xgb_gaia_hilat_projected.csv"
SKY_IMG    = ARRAYS_DIR / "gaia_dr3_aitoff_gal_8k.jpg"

# Image WCS: GLON-AIT / GLAT-AIT, 8000×4000
IMG_W, IMG_H = 8000, 4000
IMG_CRPIX = (4000.5, 2000.5)
IMG_CDELT = (-0.045, 0.045)  # deg/pixel  (360/8000)

# hips2fits API for downloading the Gaia DR3 sky image
def _hips2fits_url(w, h):
    import json as _json
    cdelt = 360.0 / w
    wcs = _json.dumps({
        "NAXIS": 2, "NAXIS1": w, "NAXIS2": h,
        "CTYPE1": "GLON-AIT", "CTYPE2": "GLAT-AIT",
        "CRPIX1": w / 2 + 0.5, "CRPIX2": h / 2 + 0.5,
        "CRVAL1": 0.0, "CRVAL2": 0.0,
        "CDELT1": -cdelt, "CDELT2": cdelt,
    })
    return (
        "https://alasky.cds.unistra.fr/hips-image-services/hips2fits?"
        + urllib.parse.urlencode({
            "hips": "CDS/P/DM/flux-color-Rp-G-Bp/I/355/gaiadr3",
            "format": "jpg",
            "wcs": wcs,
        })
    )

HIPS2FITS_URL = _hips2fits_url(IMG_W, IMG_H)

# Load ERO fields from CSV (actual ra, dec, area_deg2 → radius = sqrt(area/pi))
import math as _math
import pandas as _fields_pd
_ERO_CSV = Path(__file__).parent.parent / "output" / "ero_fields_info.csv"
_ero_df = _fields_pd.read_csv(_ERO_CSV)
ERO_FIELDS = {
    row["field_name"]: (row["ra_deg"], row["dec_deg"],
                        _math.sqrt(row["area_deg2"] / _math.pi))
    for _, row in _ero_df.iterrows()
}
# Q1 field areas from analyze_dr3_scanning_law.py: radius = sqrt(area/pi)
Q1_FIELDS = {
    'EDF-North':  (269.733, +66.018, _math.sqrt(22.9 / _math.pi)),   # ~2.70°
    'EDF-South':  (61.241,  -48.423, _math.sqrt(28.1 / _math.pi)),   # ~2.99°
    'EDF-Fornax': (52.932,  -28.088, _math.sqrt(12.1 / _math.pi)),   # ~1.96°
    'LDN 1641':   (85.75,   -8.367,  _math.sqrt(0.5  / _math.pi)),   # ~0.40°
}

# Image WCS: GLON-AIT / GLAT-AIT, 6000×3000

# ── Coordinate helpers ──────────────────────────────────────────────────────

def radec_to_galactic(ra, dec):
    sc = SkyCoord(ra=np.asarray(ra) * u.deg, dec=np.asarray(dec) * u.deg, frame="icrs")
    g = sc.galactic
    return g.l.deg, g.b.deg


def hammer_aitoff(l_deg, b_deg):
    """Galactic (l,b) → Aitoff (ax, ay). ax ∈ [-2√2, 2√2], ay ∈ [-√2, √2]."""
    l_c = np.where(np.asarray(l_deg) > 180, np.asarray(l_deg) - 360, l_deg)
    lon = np.radians(-np.asarray(l_c, dtype=float))
    lat = np.radians(np.asarray(b_deg, dtype=float))
    z = np.sqrt(1 + np.cos(lat) * np.cos(lon / 2))
    return (2 * np.cos(lat) * np.sin(lon / 2) / z,
            np.sin(lat) / z)


def aitoff_to_imgpx(ax, ay):
    """Aitoff (ax,ay) → image pixel coords matching the hips2fits WCS."""
    # The hips2fits image uses GLON-AIT with CDELT1=-0.12 (lon increases left)
    # Aitoff x = 2*cos(b)*sin(-l/2)/z  (we already negate l in hammer_aitoff)
    # The WCS maps: pixel_x = CRPIX1 + ax_wcs / CDELT1
    # where ax_wcs is the WCS Aitoff x in degrees = ax * (180/π) * scale
    # For standard Aitoff: x_rad = ax (our value), x_deg = ax * 180/π
    # But WCS Aitoff uses x_deg directly, so:
    # px = CRPIX1 + (ax * 180/π) / CDELT1
    ax_deg = np.asarray(ax) * 180.0 / np.pi
    ay_deg = np.asarray(ay) * 180.0 / np.pi
    px = IMG_CRPIX[0] + ax_deg / IMG_CDELT[0]
    py = IMG_CRPIX[1] + ay_deg / IMG_CDELT[1]
    return px, py


# ── Data helpers ────────────────────────────────────────────────────────────

def _round_list(arr, d=4):
    return [round(float(v), d) if np.isfinite(v) else None for v in arr]


def _prepare_data(df, color_opts):
    """Convert a DataFrame into JSON-ready dict with image pixel coords."""
    ra, dec = df["ra"].values, df["dec"].values
    gl, gb = radec_to_galactic(ra, dec)
    ax, ay = hammer_aitoff(gl, gb)
    ipx, ipy = aitoff_to_imgpx(ax, ay)

    data = {
        "ra": _round_list(ra, 4),
        "dec": _round_list(dec, 4),
        "gl": _round_list(gl, 4),    # galactic l (deg)
        "gb": _round_list(gb, 4),    # galactic b (deg)
        "ipx": _round_list(ipx, 2),  # image pixel x
        "ipy": _round_list(ipy, 2),  # image pixel y
        "u1": _round_list(df["umap1"].values),
        "u2": _round_list(df["umap2"].values),
        "sid": df["source_id"].astype(str).tolist(),
    }
    for c in color_opts:
        data[c] = _round_list(df[c].values)
    return data


def _prepare_fields(fields_dict):
    """Convert field dict to list with ra, dec, radius_deg for JS circle drawing."""
    result = []
    for name, vals in fields_dict.items():
        ra, dec, radius = vals
        result.append({"name": name, "ra": round(float(ra), 4),
                       "dec": round(float(dec), 4),
                       "radius": round(float(radius), 4)})
    return result


# ── Sky image ───────────────────────────────────────────────────────────────

def _blacken_outside_ellipse(img_path):
    """Set pixels outside the Aitoff ellipse to black."""
    from PIL import Image
    img = Image.open(img_path)
    arr = np.array(img)
    h, w = arr.shape[:2]
    # Aitoff ellipse semi-axes in pixels from center
    # In WCS native coords: x_max = 2√2 * (180/π) deg, y_max = √2 * (180/π) deg
    # In pixels: semi_a = x_max / |CDELT1|, semi_b = y_max / |CDELT2|
    semi_a = 2 * np.sqrt(2) * 180 / np.pi / abs(IMG_CDELT[0])
    semi_b = np.sqrt(2) * 180 / np.pi / abs(IMG_CDELT[1])
    cy, cx = IMG_CRPIX[1] - 0.5, IMG_CRPIX[0] - 0.5  # 0-indexed
    yy, xx = np.mgrid[:h, :w]
    outside = ((xx - cx) / semi_a) ** 2 + ((yy - cy) / semi_b) ** 2 > 1
    arr[outside] = [8, 8, 8]  # match body background #080808
    img_out = Image.fromarray(arr)
    img_out.save(img_path, quality=92)


def ensure_sky_image():
    """Download Gaia DR3 sky image if not cached, blacken outside ellipse."""
    if SKY_IMG.exists():
        return SKY_IMG
    print(f"  Downloading Gaia DR3 sky image...")
    try:
        urllib.request.urlretrieve(HIPS2FITS_URL, SKY_IMG)
        print(f"  -> Downloaded ({SKY_IMG.stat().st_size / 1e6:.1f} MB)")
        print(f"  Masking background to black...")
        _blacken_outside_ellipse(SKY_IMG)
        print(f"  -> {SKY_IMG} ({SKY_IMG.stat().st_size / 1e6:.1f} MB)")
        return SKY_IMG
    except Exception as e:
        print(f"  WARNING: Could not download sky image: {e}")
        return None


# ── HTML generation ─────────────────────────────────────────────────────────

Q1_EMB_FILE = ARRAYS_DIR / "q1_indiv.npz"
Q1_LABELS_FILE = ARRAYS_DIR / "q1_cluster_labels.npy"
CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#F298B6", "mediumpurple",
    "#17becf", "#d62728", "#8c564b", "#7f7f7f", "#bcbd22",
]


def _prepare_q1_base():
    """Load Q1 UMAP embedding + cluster labels + Gaia features for background."""
    if not Q1_EMB_FILE.exists():
        return None
    d = np.load(Q1_EMB_FILE, allow_pickle=True)
    emb = d["embedding"]  # (50000, 2)
    labels = np.load(Q1_LABELS_FILE) if Q1_LABELS_FILE.exists() else d["y"]
    X_raw = d["X_raw"].astype(float) if "X_raw" in d else None
    g_mag = d["g_mag"].astype(float) if "g_mag" in d else None

    from selected_parameters_list import SELECTED_GAIA_PARAMETERS
    feat_names = list(SELECTED_GAIA_PARAMETERS)

    # Build per-cluster dicts with x, y, and Gaia feature arrays
    clusters = {}
    for k in sorted(set(labels)):
        m = labels == k
        entry = {
            "x": _round_list(emb[m, 0], 2),
            "y": _round_list(emb[m, 1], 2),
        }
        if X_raw is not None:
            for fi, fn in enumerate(feat_names):
                entry[fn] = _round_list(X_raw[m, fi])
        if g_mag is not None:
            entry["phot_g_mean_mag"] = _round_list(g_mag[m])
        clusters[int(k)] = entry

    # Pre-compute feature ranges using 1st/99th percentiles (matching q1_3d.html)
    feat_ranges = {}
    if X_raw is not None:
        for fi, fn in enumerate(feat_names):
            col = X_raw[:, fi]
            finite = col[np.isfinite(col)]
            if len(finite):
                feat_ranges[fn] = [
                    round(float(np.percentile(finite, 1)), 4),
                    round(float(np.percentile(finite, 99)), 4),
                ]
            else:
                feat_ranges[fn] = [0.0, 1.0]
        if g_mag is not None:
            finite = g_mag[np.isfinite(g_mag)]
            feat_ranges["phot_g_mean_mag"] = [
                round(float(np.percentile(finite, 1)), 4),
                round(float(np.percentile(finite, 99)), 4),
            ] if len(finite) else [0.0, 1.0]

    # Return clusters + feature names + ranges for dropdown
    return {"clusters": clusters, "feature_names": feat_names, "feat_ranges": feat_ranges}


def build_html(df, hilat_df=None, sky_b64=None, q1_base=None):
    color_opts = {"gal_prob": "P(gal)", "phot_g_mean_mag": "G mag"}
    # Add all Gaia features using their raw names
    from selected_parameters_list import SELECTED_GAIA_PARAMETERS
    for fn in SELECTED_GAIA_PARAMETERS:
        if fn not in color_opts:
            color_opts[fn] = fn
    color_opts = {k: v for k, v in color_opts.items() if k in df.columns}

    data = _prepare_data(df, color_opts)
    hilat_data_json = "null"
    if hilat_df is not None:
        hilat_data_json = json.dumps(_prepare_data(hilat_df, color_opts))

    n = len(df)
    q1_fields_js = json.dumps(_prepare_fields(Q1_FIELDS))
    ero_fields_js = json.dumps(_prepare_fields(ERO_FIELDS))

    sky_img_src = ""
    if sky_b64:
        sky_img_src = f"data:image/jpeg;base64,{sky_b64}"

    q1_base_json = json.dumps(q1_base["clusters"]) if q1_base else "null"
    q1_base_feats = json.dumps(q1_base["feature_names"]) if q1_base else "[]"
    q1_feat_ranges_json = json.dumps(q1_base.get("feat_ranges", {})) if q1_base else "{}"
    # Build Q1 base color options: clusters + Gaia features (raw names)
    q1_color_opts = {"clusters": "clusters"}
    if q1_base and "feature_names" in q1_base:
        for fn in q1_base["feature_names"]:
            q1_color_opts[fn] = fn
        if "phot_g_mean_mag" not in q1_base["feature_names"]:
            q1_color_opts["phot_g_mean_mag"] = "phot_g_mean_mag"

    # WCS constants for the shader
    crpx0, crpx1 = IMG_CRPIX
    cd0, cd1 = IMG_CDELT

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sky + UMAP Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  background:#080808; color:#bbb;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
  display:flex; flex-direction:column; height:100vh; overflow:hidden;
}}
#controls {{
  padding:4px 16px; background:#0c0c0c; border-bottom:1px solid #181818;
  display:flex; flex-direction:column; gap:3px; font-size:12px; flex-shrink:0;
}}
#controls-row1 {{ display:flex; align-items:flex-start; gap:10px; }}
.cbar-group {{ display:flex; flex-direction:column; gap:2px; }}
.cbar-top {{ display:flex; align-items:center; gap:4px; }}
#overlay-cbar-wrap, #base-cbar-wrap {{ display:flex; align-items:center; gap:4px; padding-left:2px; }}
#overlay-cbar-gradient, #base-cbar-gradient {{ width:130px; height:8px; border-radius:2px; border:1px solid #2a2a2a; }}
#overlay-cbar-min, #overlay-cbar-max,
#base-cbar-min, #base-cbar-max {{ color:#555; font-size:9px; font-family:"SF Mono",Menlo,Consolas,monospace; }}
#controls label {{ color:#777; font-size:11px; }}
.controls-right {{ margin-left:auto; display:flex; gap:10px; align-items:flex-start; padding-top:3px; }}
#controls select, #controls input[type=range] {{
  background:#111; color:#ccc; border:1px solid #333;
  border-radius:3px; padding:3px 6px; font-size:11px; cursor:pointer;
}}
#controls button {{
  background:#1a1a1a; color:#aaa; border:1px solid #333;
  border-radius:3px; padding:3px 10px; font-size:11px; cursor:pointer;
}}
#controls button:hover {{ background:#282828; color:#ddd; }}
h1 {{ font-size:13px; font-weight:600; color:#ddd; white-space:nowrap; margin-top:2px; }}
#panels {{ display:flex; flex:1; min-height:0; }}
#sky-viewport {{
  flex:1.2; min-width:0; position:relative; overflow:hidden;
  background:#080808; cursor:grab;
}}
#sky-viewport:active {{ cursor:grabbing; }}
#sky-gl {{
  position:absolute; top:0; left:0; width:100%; height:100%;
}}
#source-canvas {{
  position:absolute; top:0; left:0; width:100%; height:100%;
  pointer-events:none;
}}
#umap-container {{ flex:1; min-width:0; }}
#stats-panel {{
  padding:8px 16px; background:#0c0c0c; border-top:1px solid #181818;
  font-size:12px; color:#888; flex-shrink:0; min-height:40px;
  font-family:"SF Mono",Menlo,Consolas,monospace;
  display:flex; align-items:center; gap:16px;
  overflow-x:auto; white-space:nowrap;
}}
#stats-panel b {{ color:#ccc; }}
.stat-label {{ color:#666; }}
.hint {{ color:#555; font-size:11px; font-style:italic; }}
#fov-info {{ color:#666; font-size:10px; margin-left:4px; }}
/* Minimap */
#minimap {{
  position:absolute; bottom:10px; right:10px; width:180px; height:90px;
  border:1px solid #333; border-radius:3px; overflow:hidden; z-index:20;
  box-shadow:0 2px 10px rgba(0,0,0,0.8); background:#000;
}}
#minimap img {{
  width:100%; height:100%; object-fit:contain; display:block;
  opacity:0.6;
}}
#minimap-canvas {{
  position:absolute; top:0; left:0; width:100%; height:100%;
  pointer-events:none;
}}
</style>
</head>
<body>
<div id="controls">
  <div id="controls-row1">
    <h1>Sky + UMAP Explorer</h1>
    <button id="clear-sel-btn" style="padding:2px 8px;cursor:pointer;font-size:11px;">Clear Selection</button>
    <span id="fov-info"></span>
    <span class="controls-right">
      <span class="cbar-group">
        <span class="cbar-top"><label>DATASET</label>
        <select id="dataset-select">
          <option value="random">Random 50k</option>
          <option value="hilat">High-lat 50k</option>
        </select></span>
      </span>
      <span class="cbar-group">
        <span class="cbar-top"><label>Z AXIS</label><select id="z-select"></select></span>
      </span>
      <span class="cbar-group">
        <span class="cbar-top"><label>BASE</label><select id="base-color-select"></select></span>
        <div id="base-cbar-wrap">
          <span id="base-cbar-min">0.00</span>
          <div id="base-cbar-gradient"></div>
          <span id="base-cbar-max">1.00</span>
        </div>
      </span>
      <span class="cbar-group">
        <span class="cbar-top"><label>OVERLAY</label><select id="color-select"></select></span>
        <div id="overlay-cbar-wrap">
          <span id="overlay-cbar-min">0.00</span>
          <div id="overlay-cbar-gradient"></div>
          <span id="overlay-cbar-max">1.00</span>
        </div>
      </span>
    </span>
  </div>
</div>

<div id="panels">
  <div id="sky-viewport">
    <canvas id="sky-gl"></canvas>
    <canvas id="source-canvas"></canvas>
    <div id="minimap">
      <img src="{sky_img_src}">
      <canvas id="minimap-canvas"></canvas>
    </div>
  </div>
  <div id="umap-container"></div>
</div>
<div id="stats-panel">
  <span class="hint">{n:,} sources &middot; drag to pan, scroll to zoom, Cmd+drag to select, Shift+drag to move selection</span>
</div>

<script>
// ══════════════════════════════════════════════════════════════
// DATA
// ══════════════════════════════════════════════════════════════
var DATASETS = {{
  random: {json.dumps(data)},
  hilat: {hilat_data_json}
}};
var D = DATASETS.random;
var COPTS = {json.dumps(color_opts)};
var Q1F = {q1_fields_js};
var EROF = {ero_fields_js};
var N = D.ra.length;
var IW = {IMG_W}, IH = {IMG_H};
var PI = Math.PI, DEG = PI / 180;
var Q1_BASE = {q1_base_json};
var Q1_BASE_FEATS = {q1_base_feats};
var Q1_FEAT_RANGES = {q1_feat_ranges_json};
var Q1_COLOR_OPTS = {json.dumps(q1_color_opts)};
var CLUSTER_COLORS_JS = {json.dumps(CLUSTER_COLORS)};

// WCS constants
var CRPIX_X = {crpx0}, CRPIX_Y = {crpx1};
var CDELT_X = {cd0}, CDELT_Y = {cd1};

// ── Dropdowns ──
var datasetSel = document.getElementById('dataset-select');
var colorSel = document.getElementById('color-select');
var zSel = document.getElementById('z-select');
var baseColorSel = document.getElementById('base-color-select');
Object.keys(COPTS).forEach(function(k) {{
  var o1 = document.createElement('option'); o1.value = k; o1.textContent = COPTS[k];
  var o2 = o1.cloneNode(true);
  colorSel.appendChild(o1); zSel.appendChild(o2);
}});
Object.keys(Q1_COLOR_OPTS).forEach(function(k) {{
  var o = document.createElement('option'); o.value = k; o.textContent = Q1_COLOR_OPTS[k];
  baseColorSel.appendChild(o);
}});
if (!DATASETS.hilat) {{
  datasetSel.querySelector('option[value="hilat"]').disabled = true;
}}

// ── Color mapping (RdYlBu_r) ──
var curColor = 'gal_prob';
var curBaseColor = 'clusters';
function valToRGB(v, mn, mx) {{
  var t = (v - mn) / (mx - mn); t = Math.max(0, Math.min(1, t));
  var r, g, b;
  if (t < 0.5) {{
    var s = t * 2;
    r = Math.round(49 + s * 206); g = Math.round(54 + s * 200); b = Math.round(149 - s * 100);
  }} else {{
    var s = (t - 0.5) * 2;
    r = Math.round(255 - s * 40); g = Math.round(254 - s * 189); b = Math.round(49 + s * 5);
  }}
  return 'rgb(' + r + ',' + g + ',' + b + ')';
}}
function getColorBounds() {{
  var arr = D[curColor], mn = Infinity, mx = -Infinity;
  for (var i = 0; i < N; i++) if (arr[i] !== null) {{
    if (arr[i] < mn) mn = arr[i]; if (arr[i] > mx) mx = arr[i];
  }}
  return [mn, mx];
}}

// ── Angular distance (degrees) ──
function angDist(ra1, dec1, ra2, dec2) {{
  var dd = (dec2 - dec1) * DEG, dr = (ra2 - ra1) * DEG;
  var a = Math.sin(dd / 2) * Math.sin(dd / 2) +
          Math.cos(dec1 * DEG) * Math.cos(dec2 * DEG) * Math.sin(dr / 2) * Math.sin(dr / 2);
  return 2 * Math.asin(Math.sqrt(Math.min(1, a))) / DEG;
}}

function angDistGal(l1, b1, l2, b2) {{
  var db = (b2 - b1) * DEG, dl = (l2 - l1) * DEG;
  var a = Math.sin(db / 2) * Math.sin(db / 2) +
          Math.cos(b1 * DEG) * Math.cos(b2 * DEG) * Math.sin(dl / 2) * Math.sin(dl / 2);
  return 2 * Math.asin(Math.sqrt(Math.min(1, a))) / DEG;
}}

// ── Selection state ──
var selected = new Uint8Array(N);
var hasSelection = false;
var selCircle = null;  // galactic l, b, r in degrees

// ══════════════════════════════════════════════════════════════
// WEBGL SKY RENDERING (stereographic reprojection of Aitoff texture)
// ══════════════════════════════════════════════════════════════
var viewport = document.getElementById('sky-viewport');
var glCanvas = document.getElementById('sky-gl');
var gl = glCanvas.getContext('webgl', {{antialias:false, depth:false}});

// View state
var centerL = 0;     // galactic longitude (radians)
var centerB = 0;     // galactic latitude (radians)
var fov = 2.0;       // vertical field of view (radians), ~115°
var aspect = 1;

// Shaders (stereographic projection — handles wide FOV without distortion)
var VERT_SRC = 'attribute vec2 a_pos; varying vec2 v_uv; void main(){{ v_uv=a_pos; gl_Position=vec4(a_pos,0,1); }}';
var FRAG_SRC = [
  'precision highp float;',
  'uniform sampler2D u_tex;',
  'uniform float u_cl, u_cb, u_fov, u_asp;',
  'varying vec2 v_uv;',
  'const float PI=3.14159265359;',
  'const float S2=1.41421356237;',
  'void main(){{',
  '  float ht=2.0*tan(u_fov*0.25);',
  '  float gx=v_uv.x*ht*u_asp, gy=v_uv.y*ht;',
  '  float rho=sqrt(gx*gx+gy*gy);',
  '  float sb=sin(u_cb), cb=cos(u_cb), l, b;',
  '  if(rho<1e-6){{ l=u_cl; b=u_cb; }}',
  '  else{{',
  '    float c=2.0*atan(rho*0.5), sc=sin(c), cc=cos(c);',
  '    b=asin(clamp(cc*sb+gy*sc*cb/rho,-1.0,1.0));',
  '    l=u_cl+atan(gx*sc, rho*cb*cc-gy*sb*sc);',
  '  }}',
  '  l=l-floor(l/(2.0*PI))*2.0*PI;',
  '  float lc=l>PI?l-2.0*PI:l;',
  '  float lon=-lc, cosB=cos(b);',
  '  float z=sqrt(1.0+cosB*cos(lon*0.5));',
  '  float ax=2.0*cosB*sin(lon*0.5)/z, ay=sin(b)/z;',
  '  float X=ax*S2, Y=ay*S2;',
  '  if(X*X/16.0+Y*Y/4.0>1.0){{ gl_FragColor=vec4(0.031,0.031,0.031,1); return; }}',
  '  float px={crpx0}+(ax*180.0/PI)/({cd0});',
  '  float py={crpx1}+(ay*180.0/PI)/({cd1});',
  '  float u=px/{float(IMG_W)}, v=1.0-py/{float(IMG_H)};',
  '  if(u<0.0||u>1.0||v<0.0||v>1.0){{ gl_FragColor=vec4(0.031,0.031,0.031,1); return; }}',
  '  gl_FragColor=texture2D(u_tex,vec2(u,v));',
  '}}'
].join('\\n');

function mkShader(type, src) {{
  var s = gl.createShader(type);
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {{
    console.error('Shader error:', gl.getShaderInfoLog(s)); return null;
  }}
  return s;
}}

var vs = mkShader(gl.VERTEX_SHADER, VERT_SRC);
var fs = mkShader(gl.FRAGMENT_SHADER, FRAG_SRC);
var prog = gl.createProgram();
gl.attachShader(prog, vs); gl.attachShader(prog, fs);
gl.linkProgram(prog); gl.useProgram(prog);

// Fullscreen quad
var buf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, buf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
var aPos = gl.getAttribLocation(prog, 'a_pos');
gl.enableVertexAttribArray(aPos);
gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

// Uniform locations
var uCl = gl.getUniformLocation(prog, 'u_cl');
var uCb = gl.getUniformLocation(prog, 'u_cb');
var uFov = gl.getUniformLocation(prog, 'u_fov');
var uAsp = gl.getUniformLocation(prog, 'u_asp');
var uTex = gl.getUniformLocation(prog, 'u_tex');

// Load sky texture
var skyTex = gl.createTexture();
var skyReady = false;
var skyImg = new Image();
skyImg.onload = function() {{
  gl.bindTexture(gl.TEXTURE_2D, skyTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, skyImg);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  skyReady = true;
  renderAll();
}};
skyImg.src = '{sky_img_src}';

function renderSky() {{
  if (!skyReady) return;
  gl.viewport(0, 0, glCanvas.width, glCanvas.height);
  gl.useProgram(prog);
  gl.uniform1f(uCl, centerL);
  gl.uniform1f(uCb, centerB);
  gl.uniform1f(uFov, fov);
  gl.uniform1f(uAsp, aspect);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, skyTex);
  gl.uniform1i(uTex, 0);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}}

function renderAll() {{
  renderSky();
  drawSources();
  updateMinimap();
  updateFovInfo();
}}

function updateFovInfo() {{
  document.getElementById('fov-info').textContent =
    'FOV: ' + (fov / DEG).toFixed(1) + '°  l=' + ((centerL/DEG+360)%360).toFixed(1) + '° b=' + (centerB/DEG).toFixed(1) + '°';
}}

// ── Canvas sizing ──
var srcCanvas = document.getElementById('source-canvas');
var srcCtx = srcCanvas.getContext('2d');

function resizeCanvases() {{
  var rect = viewport.getBoundingClientRect();
  var w = rect.width, h = rect.height;
  var dpr = window.devicePixelRatio || 1;
  glCanvas.width = w * dpr;
  glCanvas.height = h * dpr;
  srcCanvas.width = w * dpr;
  srcCanvas.height = h * dpr;
  srcCanvas.style.width = w + 'px';
  srcCanvas.style.height = h + 'px';
  srcCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  aspect = w / h;
}}

// ══════════════════════════════════════════════════════════════
// GNOMONIC PROJECTION HELPERS
// ══════════════════════════════════════════════════════════════

// Forward stereographic: (l,b) in radians → (sx,sy) or null if behind
function gnomFwd(lr, br) {{
  var dl = lr - centerL;
  if (dl > PI) dl -= 2*PI;
  if (dl < -PI) dl += 2*PI;
  var sb = Math.sin(br), cb = Math.cos(br);
  var sb0 = Math.sin(centerB), cb0 = Math.cos(centerB);
  var cdl = Math.cos(dl);
  var cos_c = sb0*sb + cb0*cb*cdl;
  if (cos_c <= -0.9) return null;  // nearly antipodal
  var k = 2.0 / (1.0 + cos_c);
  return {{
    x: k * cb * Math.sin(dl),
    y: k * (cb0*sb - sb0*cb*cdl)
  }};
}}

// Inverse stereographic: (gx,gy) → {{l,b}} in radians
function gnomInv(gx, gy) {{
  var rho = Math.sqrt(gx*gx + gy*gy);
  if (rho < 1e-8) return {{l: centerL, b: centerB}};
  var c = 2 * Math.atan(rho / 2);
  var sc = Math.sin(c), cc = Math.cos(c);
  var sb0 = Math.sin(centerB), cb0 = Math.cos(centerB);
  var b = Math.asin(Math.max(-1, Math.min(1, cc*sb0 + gy*sc*cb0/rho)));
  var l = centerL + Math.atan2(gx*sc, rho*cb0*cc - gy*sb0*sc);
  l = ((l % (2*PI)) + 2*PI) % (2*PI);
  return {{l: l, b: b}};
}}

// Stereographic to screen pixel (for source canvas, which has CSS dimensions w×h)
function gnomToScreen(gx, gy) {{
  var rect = viewport.getBoundingClientRect();
  var w = rect.width, h = rect.height;
  var ht = 2 * Math.tan(fov / 4);
  var sx = (gx / (ht * aspect) + 1) * 0.5 * w;
  var sy = (1 - gy / ht) * 0.5 * h;
  return {{x: sx, y: sy}};
}}

// ══════════════════════════════════════════════════════════════
// MOUSE INTERACTION
// ══════════════════════════════════════════════════════════════
var _drag = null;
var _selDrag = null;   // Shift+drag selection state
var _moveDrag = null;  // Cmd+drag move state

function screenToGal(clientX, clientY) {{
  var rect = viewport.getBoundingClientRect();
  var nx = (clientX - rect.left) / rect.width * 2 - 1;
  var ny = 1 - (clientY - rect.top) / rect.height * 2;
  var ht = 2 * Math.tan(fov / 4);
  var gx = nx * ht * aspect;
  var gy = ny * ht;
  return gnomInv(gx, gy);  // returns l, b in radians
}}

viewport.addEventListener('mousedown', function(e) {{
  if (e.button !== 0) return;
  if (e.metaKey) {{
    // Cmd+drag: new selection circle
    var lb = screenToGal(e.clientX, e.clientY);
    if (lb) {{
      _selDrag = {{l: lb.l / DEG, b: lb.b / DEG, r: 0}};
      selCircle = {{l: _selDrag.l, b: _selDrag.b, r: 0}};
    }}
  }} else if (e.shiftKey && selCircle && selCircle.r > 0) {{
    // Shift+drag: move existing selection circle
    var lb = screenToGal(e.clientX, e.clientY);
    if (lb) {{
      _moveDrag = {{
        startL: lb.l / DEG, startB: lb.b / DEG,
        origL: selCircle.l, origB: selCircle.b
      }};
    }}
  }} else {{
    _drag = {{sx: e.clientX, sy: e.clientY, cl: centerL, cb: centerB, moved: false}};
  }}
}});
window.addEventListener('mousemove', function(e) {{
  if (_selDrag) {{
    var lb = screenToGal(e.clientX, e.clientY);
    if (lb) {{
      var r = angDistGal(_selDrag.l, _selDrag.b, lb.l / DEG, lb.b / DEG);
      _selDrag.r = r;
      selCircle = {{l: _selDrag.l, b: _selDrag.b, r: r}};
      drawSources();
    }}
    return;
  }}
  if (_moveDrag) {{
    var lb = screenToGal(e.clientX, e.clientY);
    if (lb) {{
      var dl = lb.l / DEG - _moveDrag.startL;
      var db = lb.b / DEG - _moveDrag.startB;
      selCircle.l = _moveDrag.origL + dl;
      selCircle.b = Math.max(-90, Math.min(90, _moveDrag.origB + db));
      drawSources();
    }}
    return;
  }}
  if (!_drag) return;
  var dx = e.clientX - _drag.sx, dy = e.clientY - _drag.sy;
  if (Math.abs(dx) > 3 || Math.abs(dy) > 3) _drag.moved = true;
  var rect = viewport.getBoundingClientRect();
  var dL = -dx / rect.width * fov * aspect;
  var dB = dy / rect.height * fov;
  centerL = _drag.cl + dL;
  centerB = Math.max(-PI/2 + 0.01, Math.min(PI/2 - 0.01, _drag.cb + dB));
  centerL = ((centerL % (2*PI)) + 2*PI) % (2*PI);
  renderAll();
}});
window.addEventListener('mouseup', function(e) {{
  if (_selDrag) {{
    if (_selDrag.r > 0.5) {{
      updateSelection();
    }} else {{
      selCircle = null;
      drawSources();
    }}
    _selDrag = null;
    return;
  }}
  if (_moveDrag) {{
    _moveDrag = null;
    updateSelection();
    return;
  }}
  if (!_drag) return;
  _drag = null;
}});

// Modifier key cursor hints
window.addEventListener('keydown', function(e) {{
  if (e.key === 'Shift') viewport.style.cursor = 'crosshair';
  if (e.key === 'Meta' && selCircle) viewport.style.cursor = 'move';
}});
window.addEventListener('keyup', function(e) {{
  if (e.key === 'Shift' || e.key === 'Meta') viewport.style.cursor = '';
}});

// Scroll to zoom (change FOV)
viewport.addEventListener('wheel', function(e) {{
  e.preventDefault();
  var factor = e.deltaY > 0 ? 1.12 : 1/1.12;
  fov = Math.max(0.02, Math.min(3.2, fov * factor));
  renderAll();
}}, {{passive: false}});

// Double-click to reset
viewport.addEventListener('dblclick', function(e) {{
  e.preventDefault();
  centerL = 0; centerB = 0; fov = 2.0;
  renderAll();
}});

// ══════════════════════════════════════════════════════════════
// DATA POINTS ON 2D CANVAS (stereographic projection)
// ══════════════════════════════════════════════════════════════

function drawSources() {{
  var rect = viewport.getBoundingClientRect();
  var w = rect.width, h = rect.height;
  srcCtx.clearRect(0, 0, w, h);

  var ht = 2 * Math.tan(fov / 4);
  var cb = getColorBounds();
  var mn = cb[0], mx = cb[1];
  var PS = Math.max(3, Math.round(Math.min(w, h) / 180));
  var PSs = Math.max(2, PS - 2);
  var PSb = PS + 3;

  var sb0 = Math.sin(centerB), cb0 = Math.cos(centerB);
  var htAsp = ht * aspect;

  // Inline stereographic + screen projection for speed
  function toScreen(i) {{
    var lr = D.gl[i] * DEG, br = D.gb[i] * DEG;
    var dl = lr - centerL;
    if (dl > PI) dl -= 2*PI;
    if (dl < -PI) dl += 2*PI;
    var sb = Math.sin(br), cbb = Math.cos(br);
    var cdl = Math.cos(dl);
    var cos_c = sb0*sb + cb0*cbb*cdl;
    if (cos_c <= -0.9) return null;
    var k = 2.0 / (1.0 + cos_c);
    var gx = k * cbb * Math.sin(dl);
    var gy = k * (cb0*sb - sb0*cbb*cdl);
    var sx = (gx / htAsp + 1) * 0.5 * w;
    var sy = (1 - gy / ht) * 0.5 * h;
    if (sx < -20 || sx > w+20 || sy < -20 || sy > h+20) return null;
    return {{x: sx, y: sy}};
  }}

  var hs = Math.floor(PSs/2);
  // Draw unselected first (dimmer), then selected on top
  if (hasSelection) {{
    srcCtx.globalAlpha = 0.2;
    for (var i = 0; i < N; i++) {{
      if (selected[i]) continue;
      srcCtx.fillStyle = valToRGB(D[curColor][i] || 0, mn, mx);
      var s = toScreen(i);
      if (!s) continue;
      srcCtx.fillRect(s.x-hs, s.y-hs, PSs, PSs);
    }}
    srcCtx.globalAlpha = 0.9;
    for (var i = 0; i < N; i++) {{
      if (!selected[i]) continue;
      srcCtx.fillStyle = valToRGB(D[curColor][i] || 0, mn, mx);
      var s = toScreen(i);
      if (!s) continue;
      srcCtx.fillRect(s.x-hs, s.y-hs, PSs, PSs);
    }}
  }} else {{
    srcCtx.globalAlpha = 0.75;
    for (var i = 0; i < N; i++) {{
      srcCtx.fillStyle = valToRGB(D[curColor][i] || 0, mn, mx);
      var s = toScreen(i);
      if (!s) continue;
      srcCtx.fillRect(s.x-hs, s.y-hs, PSs, PSs);
    }}
  }}
  srcCtx.globalAlpha = 1;

  // Field outlines
  drawFieldsGnom(Q1F, '#FF4444', 1.5);
  drawFieldsGnom(EROF, '#ffffff', 1);

  // Selection circle (drawn as polyline for correct projection)
  if (selCircle && selCircle.r > 0) {{
    srcCtx.beginPath();
    var nSeg = 64;
    var cL = selCircle.l * DEG, cB = selCircle.b * DEG;
    var rRad = selCircle.r * DEG;
    var started = false;
    for (var seg = 0; seg <= nSeg; seg++) {{
      var a = seg / nSeg * 2 * PI;
      // Point on circle at angular distance r from center
      var bP = Math.asin(Math.max(-1, Math.min(1,
        Math.sin(cB) * Math.cos(rRad) + Math.cos(cB) * Math.sin(rRad) * Math.cos(a))));
      var dL = Math.atan2(
        Math.sin(a) * Math.sin(rRad) * Math.cos(cB),
        Math.cos(rRad) - Math.sin(cB) * Math.sin(bP));
      var lP = cL + dL;
      var gP = gnomFwd(lP, bP);
      if (!gP) {{ started = false; continue; }}
      var sP = gnomToScreen(gP.x, gP.y);
      if (sP.x < -50 || sP.x > w+50 || sP.y < -50 || sP.y > h+50) {{ started = false; continue; }}
      if (!started) {{ srcCtx.moveTo(sP.x, sP.y); started = true; }}
      else srcCtx.lineTo(sP.x, sP.y);
    }}
    srcCtx.strokeStyle = '#FF4081';
    srcCtx.lineWidth = 2;
    srcCtx.stroke();
  }}
}}

function drawFieldsGnom(fields, color, lineWidth) {{
  var nSeg = 64;
  var rect = srcCanvas.getBoundingClientRect();
  var w = rect.width, h = rect.height;
  srcCtx.strokeStyle = color;
  srcCtx.lineWidth = lineWidth || 1.5;
  srcCtx.fillStyle = color;
  srcCtx.font = '11px sans-serif';
  for (var i = 0; i < fields.length; i++) {{
    var f = fields[i];
    var lb = eqToGal(f.ra, f.dec);
    var cL = lb.l * DEG, cB = lb.b * DEG;
    var rRad = f.radius * DEG;
    // Draw circle outline in gnomonic projection
    // Collect all valid projected points first, then draw as a closed path
    var pts = [];
    for (var seg = 0; seg < nSeg; seg++) {{
      var a = seg / nSeg * 2 * PI;
      var bP = Math.asin(Math.max(-1, Math.min(1,
        Math.sin(cB)*Math.cos(rRad) + Math.cos(cB)*Math.sin(rRad)*Math.cos(a))));
      var dL = Math.atan2(
        Math.sin(a)*Math.sin(rRad)*Math.cos(cB),
        Math.cos(rRad) - Math.sin(cB)*Math.sin(bP));
      var lP = cL + dL;
      var g = gnomFwd(lP, bP);
      if (!g) {{ pts.push(null); continue; }}
      var s = gnomToScreen(g.x, g.y);
      if (s.x < -500 || s.x > w+500 || s.y < -500 || s.y > h+500) {{ pts.push(null); continue; }}
      pts.push(s);
    }}
    // Draw contiguous segments; close if all points valid (no wrap)
    srcCtx.beginPath();
    var started = false;
    var firstVisible = null;
    var allValid = pts.every(function(p){{ return p !== null; }});
    for (var pi2 = 0; pi2 < pts.length; pi2++) {{
      var p = pts[pi2];
      if (!p) {{ started = false; continue; }}
      if (!firstVisible) firstVisible = p;
      if (!started) {{ srcCtx.moveTo(p.x, p.y); started = true; }}
      else srcCtx.lineTo(p.x, p.y);
    }}
    if (allValid && firstVisible) srcCtx.closePath();
    srcCtx.stroke();
    // Label: place at top of circle
    var labelPt = null;
    var topA = PI / 2; // north
    var bL = Math.asin(Math.max(-1, Math.min(1,
      Math.sin(cB)*Math.cos(rRad) + Math.cos(cB)*Math.sin(rRad)*Math.cos(topA))));
    var dLL = Math.atan2(
      Math.sin(topA)*Math.sin(rRad)*Math.cos(cB),
      Math.cos(rRad) - Math.sin(cB)*Math.sin(bL));
    var gL = gnomFwd(cL + dLL, bL);
    if (gL) {{
      var sL = gnomToScreen(gL.x, gL.y);
      if (sL.x > -200 && sL.x < w+200 && sL.y > -200 && sL.y < h+200) labelPt = sL;
    }}
    if (!labelPt && firstVisible) labelPt = firstVisible;
    if (labelPt) srcCtx.fillText(f.name, labelPt.x + 3, labelPt.y - 3);
  }}
}}

// ══════════════════════════════════════════════════════════════
// MINIMAP: gnomonic viewport → Aitoff polygon (naturally curved!)
// ══════════════════════════════════════════════════════════════
var minimapEl = document.getElementById('minimap');
var miniCvs = document.getElementById('minimap-canvas');
var miniCtx = miniCvs.getContext('2d');

function jsHammerAitoff(l_deg, b_deg) {{
  var lc = l_deg > 180 ? l_deg - 360 : l_deg;
  var lon = -lc * DEG, lat = b_deg * DEG;
  var z = Math.sqrt(1 + Math.cos(lat) * Math.cos(lon/2));
  return {{
    ax: 2 * Math.cos(lat) * Math.sin(lon/2) / z,
    ay: Math.sin(lat) / z
  }};
}}

function lbToImgPx(l_deg, b_deg) {{
  var ab = jsHammerAitoff(l_deg, b_deg);
  return [
    CRPIX_X + (ab.ax * 180 / PI) / CDELT_X,
    CRPIX_Y + (ab.ay * 180 / PI) / CDELT_Y
  ];
}}

function updateMinimap() {{
  var mw = minimapEl.clientWidth, mh = minimapEl.clientHeight;
  var dpr = window.devicePixelRatio || 1;
  miniCvs.width = mw * dpr; miniCvs.height = mh * dpr;
  miniCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  miniCtx.clearRect(0, 0, mw, mh);

  var ht = 2 * Math.tan(fov / 4);
  var steps = 24;
  var mpts = [];

  // Sample viewport edges in stereographic space, then project to Aitoff minimap
  // Straight lines in tangent plane → curved lines in Aitoff = the ESA-style polygon!
  var edges = [
    [-ht*aspect, ht, ht*aspect, ht],       // top
    [ht*aspect, ht, ht*aspect, -ht],        // right
    [ht*aspect, -ht, -ht*aspect, -ht],      // bottom
    [-ht*aspect, -ht, -ht*aspect, ht]       // left
  ];

  for (var e = 0; e < 4; e++) {{
    var gx0=edges[e][0], gy0=edges[e][1], gx1=edges[e][2], gy1=edges[e][3];
    for (var i = 0; i <= steps; i++) {{
      var t = i / steps;
      var gx = gx0 + t * (gx1 - gx0);
      var gy = gy0 + t * (gy1 - gy0);
      var lb = gnomInv(gx, gy);
      var ip = lbToImgPx(lb.l / DEG, lb.b / DEG);
      mpts.push([ip[0] / IW * mw, ip[1] / IH * mh]);
    }}
  }}

  if (mpts.length > 2) {{
    miniCtx.beginPath();
    miniCtx.moveTo(mpts[0][0], mpts[0][1]);
    for (var i = 1; i < mpts.length; i++) {{
      miniCtx.lineTo(mpts[i][0], mpts[i][1]);
    }}
    miniCtx.closePath();
    miniCtx.fillStyle = 'rgba(200,200,255,0.12)';
    miniCtx.fill();
    miniCtx.strokeStyle = 'rgba(220,220,220,0.9)';
    miniCtx.lineWidth = 1.5;
    miniCtx.stroke();
  }}

  // Draw center crosshair
  var cip = lbToImgPx(centerL / DEG, centerB / DEG);
  var cx = cip[0] / IW * mw, cy = cip[1] / IH * mh;
  miniCtx.strokeStyle = 'rgba(255,100,100,0.7)';
  miniCtx.lineWidth = 1;
  miniCtx.beginPath();
  miniCtx.moveTo(cx-5, cy); miniCtx.lineTo(cx+5, cy);
  miniCtx.moveTo(cx, cy-5); miniCtx.lineTo(cx, cy+5);
  miniCtx.stroke();
}}

// (Sky click handler removed — selection is now Shift+drag)

// ══════════════════════════════════════════════════════════════
// COORDINATE TRANSFORMS
// ══════════════════════════════════════════════════════════════
function galToEq(l, b) {{
  var lr = l * DEG, br = b * DEG;
  var ra_ngp = 192.85948 * DEG, dec_ngp = 27.12825 * DEG, l_ncp = 122.93192 * DEG;
  var sinDec = Math.sin(br)*Math.sin(dec_ngp) + Math.cos(br)*Math.cos(dec_ngp)*Math.sin(lr - l_ncp);
  var dec = Math.asin(Math.max(-1, Math.min(1, sinDec)));
  var cosRA = (Math.cos(br)*Math.cos(lr - l_ncp)) / Math.cos(dec);
  var sinRA = (Math.sin(br)*Math.cos(dec_ngp) - Math.cos(br)*Math.sin(dec_ngp)*Math.sin(lr - l_ncp)) / Math.cos(dec);
  var ra = ra_ngp + Math.atan2(cosRA, sinRA);
  ra = ra / DEG; dec = dec / DEG;
  if (ra < 0) ra += 360; if (ra >= 360) ra -= 360;
  return {{ra: ra, dec: dec}};
}}

function eqToGal(ra, dec) {{
  var ra_ngp = 192.85948 * DEG, dec_ngp = 27.12825 * DEG, l_ncp = 122.93192 * DEG;
  var rar = ra * DEG, decr = dec * DEG;
  var sinB = Math.sin(decr)*Math.sin(dec_ngp) + Math.cos(decr)*Math.cos(dec_ngp)*Math.cos(rar - ra_ngp);
  var b = Math.asin(Math.max(-1, Math.min(1, sinB)));
  var num = Math.sin(decr)*Math.cos(dec_ngp) - Math.cos(decr)*Math.sin(dec_ngp)*Math.cos(rar - ra_ngp);
  var den = Math.cos(decr)*Math.sin(rar - ra_ngp);
  var l = l_ncp - Math.atan2(num, den);
  l = l / DEG; b = b / DEG;
  if (l < 0) l += 360; if (l >= 360) l -= 360;
  return {{l: l, b: b}};
}}

// ══════════════════════════════════════════════════════════════
// SELECTION LOGIC
// ══════════════════════════════════════════════════════════════
function updateSelection() {{
  if (!selCircle) {{
    hasSelection = false; selected.fill(0);
    drawSources(); initUMAP(); clearStats();
    return;
  }}
  hasSelection = true;
  var sel = [];
  for (var i = 0; i < N; i++) {{
    if (angDistGal(D.gl[i], D.gb[i], selCircle.l, selCircle.b) <= selCircle.r) {{
      selected[i] = 1; sel.push(i);
    }} else selected[i] = 0;
  }}
  drawSources(); updateUMAPSizes();
  if (sel.length > 0) updateStats(sel); else clearStats();
}}

function switchDataset(key) {{
  if (!DATASETS[key]) return;
  D = DATASETS[key]; N = D.ra.length;
  selected = new Uint8Array(N);
  hasSelection = false; selCircle = null;
  renderAll();
  if (_umapReady) initUMAP();
  clearStats();
}}

// Clear Selection button
var clearBtn = document.getElementById('clear-sel-btn');
function doClearSelection() {{ selCircle = null; updateSelection(); }}
clearBtn.addEventListener('click', doClearSelection);

// ══════════════════════════════════════════════════════════════
// PLOTLY 3D UMAP
// ══════════════════════════════════════════════════════════════
var umapDiv = document.getElementById('umap-container');
var curZ = 'gal_prob';
var _umapReady = false;

function getZArr(col) {{ return D[col].map(function(v) {{ return v !== null ? v : 0; }}); }}
function getCBounds(col) {{
  var arr = D[col], mn = Infinity, mx = -Infinity;
  for (var i = 0; i < N; i++) if (arr[i] !== null) {{
    if (arr[i] < mn) mn = arr[i]; if (arr[i] > mx) mx = arr[i];
  }}
  return [mn, mx];
}}
var CSCALES = {{
  gal_prob: 'RdYlBu_r', phot_g_mean_mag: 'Viridis_r',
  bp_g: 'Plasma', g_rp: 'Plasma',
}};
// CSS gradients for the custom horizontal colorbar (left=low, right=high)
var CBAR_CSS = {{
  gal_prob:        'linear-gradient(to right,#4575b4,#ffffbf,#d73027)',
  phot_g_mean_mag: 'linear-gradient(to right,#fde725,#21908c,#440154)',
  bp_g:            'linear-gradient(to right,#0d0887,#cc4778,#f0f921)',
  g_rp:            'linear-gradient(to right,#0d0887,#cc4778,#f0f921)',
}};
var CBAR_CSS_DEFAULT = 'linear-gradient(to right,#4575b4,#ffffbf,#d73027)';
var _fmt = function(v) {{ return Math.abs(v) < 10 ? v.toFixed(2) : v.toFixed(0); }};
function updateOverlayCbar() {{
  var cb2 = getCBounds(curColor);
  var grad = CBAR_CSS[curColor] || CBAR_CSS_DEFAULT;
  document.getElementById('overlay-cbar-gradient').style.background = grad;
  document.getElementById('overlay-cbar-min').textContent = _fmt(cb2[0]);
  document.getElementById('overlay-cbar-max').textContent = _fmt(cb2[1]);
}}
// BASE colorscale is always COLORSCALE_RDBU (RdBu), but only meaningful when not 'clusters'
var BASE_CBAR_CSS = 'linear-gradient(to right,rgb(5,48,97),rgb(247,247,247),rgb(103,0,31))';
function updateBaseCbar() {{
  var isCluster = curBaseColor === 'clusters';
  var wrap = document.getElementById('base-cbar-wrap');
  if (isCluster) {{ wrap.style.visibility = 'hidden'; return; }}
  wrap.style.visibility = 'visible';
  var fb = q1BaseFeatureBounds(curBaseColor);
  document.getElementById('base-cbar-gradient').style.background = BASE_CBAR_CSS;
  document.getElementById('base-cbar-min').textContent = _fmt(fb[0]);
  document.getElementById('base-cbar-max').textContent = _fmt(fb[1]);
}}
var COLORSCALE_RDBU = [
  [0,'rgb(5,48,97)'],[0.25,'rgb(33,102,172)'],
  [0.5,'rgb(247,247,247)'],
  [0.75,'rgb(214,96,77)'],[1,'rgb(103,0,31)']
];
var umapLayout = {{
  paper_bgcolor: '#080808',
  scene: {{
    xaxis: {{title:'UMAP 1',color:'#555',gridcolor:'#1a1a1a',backgroundcolor:'#080808',showbackground:true}},
    yaxis: {{title:'UMAP 2',color:'#555',gridcolor:'#1a1a1a',backgroundcolor:'#080808',showbackground:true}},
    zaxis: {{title:'P(gal)',color:'#555',gridcolor:'#1a1a1a',backgroundcolor:'#080808',showbackground:true}},
    bgcolor:'#080808',
  }},
  title: {{text:'3D UMAP', font:{{size:13, color:'#888'}}}},
  margin: {{l:0, r:0, t:35, b:0}},
}};

function q1BaseZVal() {{
  // Put Q1 base above the data range for the current z-axis
  var arr = getZArr(curZ);
  var mx = -Infinity;
  for (var i = 0; i < arr.length; i++) {{ if (arr[i] > mx) mx = arr[i]; }}
  return mx + (mx - 0) * 0.15 + 0.1;  // 15% above max
}}

function q1BaseFeatureBounds(feat) {{
  // Use pre-computed 1st/99th percentile ranges (matching q1_3d.html)
  if (Q1_FEAT_RANGES && Q1_FEAT_RANGES[feat]) return Q1_FEAT_RANGES[feat];
  // Fallback: compute from data
  var mn = Infinity, mx = -Infinity;
  var keys = Object.keys(Q1_BASE);
  for (var i = 0; i < keys.length; i++) {{
    var arr = Q1_BASE[keys[i]][feat];
    if (!arr) continue;
    for (var j = 0; j < arr.length; j++) {{
      if (arr[j] !== null && isFinite(arr[j])) {{
        if (arr[j] < mn) mn = arr[j]; if (arr[j] > mx) mx = arr[j];
      }}
    }}
  }}
  return [mn, mx];
}}

function q1BaseTraces() {{
  var traces = [];
  if (!Q1_BASE) return traces;
  var zTop = q1BaseZVal();
  var isCluster = curBaseColor === 'clusters';
  var fb = !isCluster ? q1BaseFeatureBounds(curBaseColor) : null;
  var keys = Object.keys(Q1_BASE).sort(function(a,b){{ return parseInt(a)-parseInt(b); }});

  if (isCluster) {{
    // One trace per cluster with fixed cluster color
    for (var i = 0; i < keys.length; i++) {{
      var k = parseInt(keys[i]);
      var cl = Q1_BASE[keys[i]];
      var isNoise = k === -1;
      var z0 = new Array(cl.x.length); for(var j=0;j<z0.length;j++) z0[j]=zTop;
      traces.push({{
        type:'scatter3d', mode:'markers',
        x:cl.x, y:cl.y, z:z0,
        marker:{{ size:1, color: isNoise ? '#2a2a2a' : CLUSTER_COLORS_JS[k % CLUSTER_COLORS_JS.length],
          opacity: isNoise ? 0.3 : 0.12 }},
        hoverinfo:'skip', showlegend:false,
        name: isNoise ? 'Q1 noise' : 'Q1 cluster ' + k,
      }});
    }}
  }} else {{
    // Single merged trace colored by feature (RdBu colorscale, null=grey)
    var allX=[], allY=[], allZ=[], allC=[];
    for (var i = 0; i < keys.length; i++) {{
      var cl = Q1_BASE[keys[i]];
      var feat = cl[curBaseColor];
      if (!feat) continue;
      for (var j = 0; j < cl.x.length; j++) {{
        allX.push(cl.x[j]); allY.push(cl.y[j]); allZ.push(zTop);
        var v = feat[j];
        allC.push((v !== null && v !== undefined && isFinite(v)) ? v : null);
      }}
    }}
    traces.push({{
      type:'scatter3d', mode:'markers',
      x:allX, y:allY, z:allZ,
      marker:{{ size:1, color:allC, colorscale:COLORSCALE_RDBU,
        cmin:fb[0], cmax:fb[1], opacity:0.5,
        showscale:false }},
      hoverinfo:'skip', showlegend:false,
      name: 'Q1 base',
    }});
  }}
  return traces;
}}

function initUMAP() {{
  var cb2 = getCBounds(curColor);
  var traces = q1BaseTraces();
  traces.push({{
    type:'scatter3d', mode:'markers',
    x:D.u1, y:D.u2, z:getZArr(curZ),
    marker:{{ size:1.8, opacity:0.7,
      color:D[curColor], colorscale:CSCALES[curColor]||'Viridis',
      cmin:cb2[0], cmax:cb2[1],
      showscale:false,
    }},
    hoverinfo:'text',
    text:D.sid.map(function(s,i){{
      return 'P(gal)='+(D.gal_prob[i]!==null?D.gal_prob[i].toFixed(3):'?')+'\\nid='+s;
    }}),
    showlegend:false,
  }});
  Plotly.newPlot(umapDiv, traces, umapLayout, {{displaylogo:false}});
  _umapReady = true;
}}

function updateUMAPSizes() {{
  if (!_umapReady) return;
  var cb2 = getCBounds(curColor);
  var traces = q1BaseTraces();
  var zArr = getZArr(curZ);
  if (hasSelection) {{
    // Unselected: grey, small
    var ux=[], uy=[], uz=[], ut=[];
    // Selected: colored, larger
    var sx=[], sy=[], sz=[], sc=[], st=[];
    for (var i=0;i<N;i++) {{
      if (selected[i]) {{
        sx.push(D.u1[i]); sy.push(D.u2[i]); sz.push(zArr[i]);
        sc.push(D[curColor][i]);
        st.push('P(gal)='+(D.gal_prob[i]!==null?D.gal_prob[i].toFixed(3):'?')+'\\nid='+D.sid[i]);
      }} else {{
        ux.push(D.u1[i]); uy.push(D.u2[i]); uz.push(zArr[i]);
        ut.push('id='+D.sid[i]);
      }}
    }}
    // Unselected trace (grey)
    traces.push({{
      type:'scatter3d', mode:'markers',
      x:ux, y:uy, z:uz,
      marker:{{ size:1, opacity:0.3, color:'#888888' }},
      hoverinfo:'text', text:ut, showlegend:false,
    }});
    // Selected trace (colored)
    traces.push({{
      type:'scatter3d', mode:'markers',
      x:sx, y:sy, z:sz,
      marker:{{ size:1.8, opacity:0.7,
        color:sc, colorscale:CSCALES[curColor]||'Viridis',
        cmin:cb2[0], cmax:cb2[1],
        showscale:false,
      }},
      hoverinfo:'text', text:st, showlegend:false,
    }});
  }} else {{
    traces.push({{
      type:'scatter3d', mode:'markers',
      x:D.u1, y:D.u2, z:zArr,
      marker:{{ size:1.8, opacity:0.7,
        color:D[curColor], colorscale:CSCALES[curColor]||'Viridis',
        cmin:cb2[0], cmax:cb2[1],
        showscale:false,
      }},
      hoverinfo:'text',
      text:D.sid.map(function(s,i){{
        return 'P(gal)='+(D.gal_prob[i]!==null?D.gal_prob[i].toFixed(3):'?')+'\\nid='+s;
      }}),
      showlegend:false,
    }});
  }}
  Plotly.react(umapDiv, traces, umapLayout, {{displaylogo:false}});
}}

initUMAP();
updateOverlayCbar();
updateBaseCbar();

// The data trace is the LAST trace (after Q1 base traces)
function dataTraceIdx() {{ return umapDiv.data ? umapDiv.data.length - 1 : 0; }}

zSel.addEventListener('change', function() {{
  curZ = this.value;
  // Update data trace z
  Plotly.restyle(umapDiv, {{z:[getZArr(curZ)]}}, [dataTraceIdx()]);
  // Update Q1 base traces z to stay above the new data range
  var zTop = q1BaseZVal();
  var nBase = dataTraceIdx();  // number of base traces
  for (var ti = 0; ti < nBase; ti++) {{
    var n = umapDiv.data[ti].z.length;
    var newZ = new Array(n); for(var j=0;j<n;j++) newZ[j]=zTop;
    Plotly.restyle(umapDiv, {{z:[newZ]}}, [ti]);
  }}
  Plotly.relayout(umapDiv, {{'scene.zaxis.title':COPTS[curZ]||curZ}});
}});
colorSel.addEventListener('change', function() {{
  curColor = this.value;
  drawSources();
  var cb2 = getCBounds(curColor);
  Plotly.restyle(umapDiv, {{
    'marker.color':[D[curColor]], 'marker.colorscale':CSCALES[curColor]||'Viridis',
    'marker.cmin':cb2[0], 'marker.cmax':cb2[1],
  }}, [dataTraceIdx()]);
  updateOverlayCbar();
}});
datasetSel.addEventListener('change', function() {{ switchDataset(this.value); updateOverlayCbar(); updateBaseCbar(); }});
baseColorSel.addEventListener('change', function() {{
  curBaseColor = this.value;
  initUMAP();
  if (hasSelection) updateUMAPSizes();
  updateBaseCbar();
}});

// ── Stats ──
function updateStats(indices) {{
  var nSel=indices.length, probs=[], mags=[], ras=[], decs=[];
  for (var i=0;i<nSel;i++) {{
    var idx=indices[i];
    if (D.gal_prob[idx]!==null) probs.push(D.gal_prob[idx]);
    if (D.phot_g_mean_mag&&D.phot_g_mean_mag[idx]!==null) mags.push(D.phot_g_mean_mag[idx]);
    ras.push(D.ra[idx]); decs.push(D.dec[idx]);
  }}
  if (!probs.length){{clearStats();return;}}
  var gf=probs.filter(function(p){{return p>=0.5;}}).length/probs.length;
  var mp=probs.reduce(function(a,b){{return a+b;}},0)/probs.length;
  var sp=probs.slice().sort(function(a,b){{return a-b;}}); var med=sp[Math.floor(sp.length/2)];
  var mm=mags.length?(mags.reduce(function(a,b){{return a+b;}},0)/mags.length):NaN;
  var p=['<b>'+nSel+'</b> <span class="stat-label">sources</span>',
    '<span class="stat-label">Gal frac:</span> <b>'+(gf*100).toFixed(1)+'%</b>',
    '<span class="stat-label">Mean P(gal):</span> <b>'+mp.toFixed(3)+'</b>',
    '<span class="stat-label">Median P(gal):</span> <b>'+med.toFixed(3)+'</b>'];
  if (!isNaN(mm)) p.push('<span class="stat-label">Mean G:</span> <b>'+mm.toFixed(2)+'</b>');
  p.push('<span class="stat-label">RA:</span> <b>['+Math.min.apply(null,ras).toFixed(1)+', '+Math.max.apply(null,ras).toFixed(1)+']</b>');
  p.push('<span class="stat-label">Dec:</span> <b>['+Math.min.apply(null,decs).toFixed(1)+', '+Math.max.apply(null,decs).toFixed(1)+']</b>');
  document.getElementById('stats-panel').innerHTML=p.join(' &nbsp;&middot;&nbsp; ');
}}
function clearStats() {{
  document.getElementById('stats-panel').innerHTML='<span class="hint">{n:,} sources &middot; drag to pan, scroll to zoom, Cmd+drag to select, Shift+drag to move selection</span>';
}}

// ── Init ──
window.addEventListener('resize', function() {{
  resizeCanvases();
  renderAll();
  Plotly.Plots.resize(umapDiv);
}});

resizeCanvases();
renderAll();
</script>
</body>
</html>"""
    return html


# ── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    if not RANDOM_CSV.exists():
        print(f"ERROR: {RANDOM_CSV} not found.")
        return 1

    print(f"[1] Loading {RANDOM_CSV.name} ...")
    df = pd.read_csv(RANDOM_CSV)
    print(f"  {len(df):,} random sources loaded")

    hilat_df = None
    if HILAT_CSV.exists():
        print(f"  Loading {HILAT_CSV.name} ...")
        hilat_df = pd.read_csv(HILAT_CSV)
        print(f"  {len(hilat_df):,} high-lat sources loaded")
    else:
        print(f"  {HILAT_CSV.name} not found — hilat option will be disabled")

    print("[2] Ensuring Gaia sky image ...")
    sky_path = ensure_sky_image()
    sky_b64 = None
    if sky_path and sky_path.exists():
        sky_b64 = base64.b64encode(sky_path.read_bytes()).decode("ascii")
        print(f"  Embedded as base64 ({len(sky_b64) / 1e6:.1f} MB)")
    else:
        print("  No sky image — will show dark background")

    print("[3] Loading Q1 base UMAP ...")
    q1_base = _prepare_q1_base()
    if q1_base:
        clusters = q1_base["clusters"]
        total = sum(len(v["x"]) for v in clusters.values())
        print(f"  {total:,} Q1 sources in {len(clusters)} clusters, {len(q1_base['feature_names'])} Gaia features")
    else:
        print("  Q1 base data not found — UMAP will show data only")

    print("[4] Building HTML ...")
    html = build_html(df, hilat_df=hilat_df, sky_b64=sky_b64, q1_base=q1_base)

    out_path = OUTPUT_DIR / "sky_umap_interactive.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    sz_mb = out_path.stat().st_size / 1e6
    print(f"  -> {out_path}  ({sz_mb:.1f} MB)")
    print(f"\n  open {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
