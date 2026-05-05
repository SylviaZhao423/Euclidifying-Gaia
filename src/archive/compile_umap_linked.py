#!/usr/bin/env python3
"""
compile_umap_linked.py
======================
Build a combined Sky + 3D UMAP viewer from the outputs of:
    compile_umap_sky.py   →  umap_sky.html
    compile_umap_3d.py    →  umap_3d.html

Sidebar controls:
  DATASET  →  switches sky source layer + 3D overlay simultaneously
  COLOR    →  updates sky dot colour + Q1 base + 3D scatter simultaneously
  Z AXIS   →  controls 3D z-axis only (unchanged from umap_3d behaviour)

Usage:
    python src/compile_umap_sky.py
    python src/compile_umap_3d.py --max-points 50000
    python src/compile_umap_linked.py
    python src/export_web.py          # (optional) split for web serving
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).resolve().parent
ROOT       = HERE.parent
ARRAYS_DIR = ROOT / "output" / "umap" / "arrays"
OUTPUT_DIR = ROOT / "output" / "umap" / "q1_individual"
SKY_HTML   = OUTPUT_DIR / "umap_sky.html"
UMAP_3D    = OUTPUT_DIR / "umap_3d.html"
OUT_FILE   = OUTPUT_DIR / "umap_linked.html"

# Random subsample cap for sky overlay datasets
MAX_SKY = 50_000

# Gaia feature column names (must match what umap_sky.html embeds)
try:
    sys.path.insert(0, str(HERE))
    from selected_parameters_list import SELECTED_GAIA_PARAMETERS  # type: ignore
except ImportError:
    SELECTED_GAIA_PARAMETERS = [
        "parallax_over_error", "astrometric_gof_al", "astrometric_chi2_al",
        "astrometric_excess_noise_sig", "ipd_gof_harmonic_amplitude", "ruwe",
        "phot_g_mean_flux_over_error", "phot_bp_mean_flux_over_error",
        "phot_rp_mean_flux_over_error", "bp_g", "g_rp",
        "corrected_bp_rp_excess", "pm_significance", "position_error",
    ]


# ── HTML extraction helpers ───────────────────────────────────────────────────

def _extract_css(html: str) -> str:
    """Extract content of the first <style>…</style> block.
    Uses index (not rindex) for </style> to avoid matching </style> literals
    embedded inside JavaScript template strings later in the file.
    """
    s = html.index("<style>") + len("<style>")
    e = html.index("</style>", s)   # first closing tag after the opening
    return html[s:e]

def _extract_body(html: str) -> str:
    """Return body content using rindex for </body> to avoid JS string matches."""
    s = html.index("<body>") + len("<body>")
    e = html.rindex("</body>")
    return html[s:e]

def _extract_plain_script(html: str) -> str:
    """Content of the last bare <script> (no attributes) in html."""
    matches = list(re.finditer(r"<script>", html))
    if not matches:
        raise ValueError("No plain <script> found")
    last = matches[-1]
    end  = html.index("</script>", last.end())
    return html[last.end() : end]

def _div_outer(html: str, div_id: str) -> str:
    """Full outer HTML of the first <div id='div_id'>."""
    pat = re.compile(r'<div\s[^>]*\bid=["\']' + re.escape(div_id) + r'["\'][^>]*>', re.I)
    m = pat.search(html)
    if not m:
        raise ValueError(f"<div id={div_id!r}> not found")
    depth, pos = 1, m.end()
    while depth and pos < len(html):
        no = html.find("<div",  pos)
        nc = html.find("</div>", pos)
        if nc == -1: break
        if no != -1 and no < nc:
            depth += 1; pos = no + 4
        else:
            depth -= 1; pos = nc + 6
    return html[m.start() : pos]


# ── Sky overlay data loading ──────────────────────────────────────────────────

def _rng_idx(n: int, cap: int, seed: int = 42) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, cap, replace=False))

def _to_list(arr, decimals=4):
    """Round numpy array and return as Python list, replacing NaN/inf with None."""
    out = []
    for v in arr:
        if np.isfinite(v):
            out.append(round(float(v), decimals))
        else:
            out.append(None)
    return out

def _sky_dataset_from_arrays(ra, dec, g_mag=None, gal_prob=None, features: dict | None = None,
                              cap: int = MAX_SKY, seed: int = 42) -> dict:
    """Build a sky dataset dict matching the DATASETS.random structure."""
    n = len(ra)
    idx = _rng_idx(n, cap, seed)
    ra  = ra[idx];  dec = dec[idx]
    g_mag     = g_mag[idx]     if g_mag     is not None else np.full(len(idx), np.nan)
    gal_prob  = gal_prob[idx]  if gal_prob  is not None else np.full(len(idx), np.nan)

    # Galactic coordinates (approx)
    ra_r  = np.deg2rad(ra)
    dec_r = np.deg2rad(dec)
    l_r   = np.arctan2(
        np.sin(ra_r - np.deg2rad(282.86)) * np.cos(dec_r),
        np.cos(dec_r) * np.sin(np.deg2rad(62.87)) * np.cos(dec_r) -
        np.sin(dec_r) * np.cos(np.deg2rad(62.87))
    ) + np.deg2rad(32.93)
    b_r = np.arcsin(
        np.sin(dec_r) * np.sin(np.deg2rad(62.87)) +
        np.cos(dec_r) * np.cos(np.deg2rad(62.87)) * np.cos(ra_r - np.deg2rad(282.86))
    )
    gl = np.rad2deg(l_r) % 360
    gb = np.rad2deg(b_r)

    ds: dict = {
        "ra":  _to_list(ra,  4),
        "dec": _to_list(dec, 4),
        "gl":  _to_list(gl,  4),
        "gb":  _to_list(gb,  4),
        "phot_g_mean_mag": _to_list(g_mag,    3),
        "gal_prob":        _to_list(gal_prob,  4),
    }
    if features:
        for fname, arr in features.items():
            ds[fname] = _to_list(arr[idx], 4)
    return ds


def load_overlay_sky_datasets() -> dict[str, dict]:
    """Load sky-renderable datasets for each 3D overlay."""
    print("  Loading overlay sky datasets ...")
    result: dict[str, dict] = {}

    # ── Q1 / allphz ──────────────────────────────────────────────────────────
    npz = np.load(ARRAYS_DIR / "q1_allphz.npz", allow_pickle=True)
    result["allphz"] = _sky_dataset_from_arrays(
        ra=npz["ra"].astype(float), dec=npz["dec"].astype(float),
        g_mag=npz["phot_g_mean_mag"].astype(float) if "phot_g_mean_mag" in npz else None,
        gal_prob=npz["phz_gal_prob"].astype(float) if "phz_gal_prob" in npz else None,
    )
    print(f"    allphz: {len(result['allphz']['ra']):,} sky sources")

    # ── Q1 × DESI ────────────────────────────────────────────────────────────
    npz_desi = np.load(ARRAYS_DIR / "q1_allphz_desi.npz", allow_pickle=True)
    # ── ERO × DESI ───────────────────────────────────────────────────────────
    npz_ero_desi = np.load(ARRAYS_DIR / "ero_desi.npz", allow_pickle=True)
    # Merge both for the "Euclid × DESI" sky layer
    ra_ed  = np.concatenate([npz_desi["ra"].astype(float), npz_ero_desi["ra"].astype(float)])
    dec_ed = np.concatenate([npz_desi["dec"].astype(float), npz_ero_desi["dec"].astype(float)])
    gm_ed  = np.concatenate([
        npz_desi["phot_g_mean_mag"].astype(float) if "phot_g_mean_mag" in npz_desi else np.full(len(npz_desi["ra"]), np.nan),
        npz_ero_desi["phot_g_mean_mag"].astype(float) if "phot_g_mean_mag" in npz_ero_desi else np.full(len(npz_ero_desi["ra"]), np.nan),
    ])
    result["euclid_x_desi"] = _sky_dataset_from_arrays(ra=ra_ed, dec=dec_ed, g_mag=gm_ed)
    print(f"    euclid_x_desi: {len(result['euclid_x_desi']['ra']):,} sky sources")

    # ── ERO ──────────────────────────────────────────────────────────────────
    npz_ero = np.load(ARRAYS_DIR / "ero_indiv.npz", allow_pickle=True)
    result["ero"] = _sky_dataset_from_arrays(
        ra=npz_ero["ra"].astype(float), dec=npz_ero["dec"].astype(float),
        g_mag=npz_ero["g_mag"].astype(float) if "g_mag" in npz_ero else None,
    )
    print(f"    ero: {len(result['ero']['ra']):,} sky sources")

    # ── ERO Rebalanced ───────────────────────────────────────────────────────
    npz_eror = np.load(ARRAYS_DIR / "ero_indiv_rebalanced.npz", allow_pickle=True)
    result["ero_rebalanced"] = _sky_dataset_from_arrays(
        ra=npz_eror["ra"].astype(float), dec=npz_eror["dec"].astype(float),
        g_mag=npz_eror["g_mag"].astype(float) if "g_mag" in npz_eror else None,
    )
    print(f"    ero_rebalanced: {len(result['ero_rebalanced']['ra']):,} sky sources")

    return result


# ── CSS layout override ───────────────────────────────────────────────────────

_LAYOUT_CSS = """\
/* ══ Combined layout — must come last to override body rules ══ */
* { margin:0; padding:0; box-sizing:border-box; }
body {
  background:#080808; color:#bbb;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
  display:flex; flex-direction:row;
  height:100vh; overflow:hidden;
}
#sky-section {
  flex:0 0 45%; display:flex; flex-direction:column;
  min-width:200px; overflow:hidden;
}
#sky-viewport {
  flex:1; min-height:0; position:relative; overflow:hidden;
  background:#080808; cursor:grab;
}
#sky-viewport:active { cursor:grabbing; }
#sky-divider {
  width:4px; flex-shrink:0; cursor:col-resize;
  background:#111; border-left:1px solid #1a1a1a; border-right:1px solid #1a1a1a;
  user-select:none; transition:background 0.15s;
}
#sky-divider:hover, #sky-divider.dragging { background:#1e2a3a; }
#umap-section { flex:1; display:flex; min-width:300px; overflow:hidden; }
"""

# Extra CSS for the new sidebar controls (DATASET / COLOR rows)
_CTRL_CSS = """\
/* ── Unified sidebar control bars ── */
.sb-bar {
  padding:6px 14px; border-bottom:1px solid #181818;
  display:flex; align-items:center; gap:8px; flex-shrink:0;
}
.sb-label {
  font-size:9px; letter-spacing:0.12em; text-transform:uppercase;
  color:#666; white-space:nowrap; min-width:48px;
}
.sb-select {
  flex:1; background:#111; color:#ccc; border:1px solid #222;
  border-radius:3px; padding:3px 6px; font-size:11px; cursor:pointer;
}
.sb-select:focus { outline:none; border-color:#333; }
.sb-select option, .sb-select optgroup { background:#111; color:#ccc; }
"""


# ── Sidebar HTML injection ────────────────────────────────────────────────────

def _build_gaia_optgroup_html() -> str:
    opts = "\n".join(
        f'        <option value="{fn}">{fn}</option>'
        for fn in SELECTED_GAIA_PARAMETERS
    )
    return f'      <optgroup label="— Gaia features —">\n{opts}\n      </optgroup>'

def _make_sidebar_controls_html() -> str:
    gaia_opts = _build_gaia_optgroup_html()
    return f"""\
  <!-- DATASET -->
  <div class="sb-bar" id="linked-dataset-bar">
    <span class="sb-label">DATASET</span>
    <select class="sb-select" id="linked-dataset-select">
      <option value="gaia_xgb"       selected>Gaia Random 50k</option>
      <option value="gaia_hilat"             >Gaia High-lat 50k</option>
      <option value="allphz"                 >Q1 (All PHZ)</option>
      <option value="euclid_x_desi"          >Euclid × DESI</option>
      <option value="ero"                    >ERO</option>
      <option value="ero_rebalanced"         >ERO Rebalanced</option>
    </select>
  </div>
  <!-- COLOR -->
  <div class="sb-bar" id="linked-color-bar">
    <span class="sb-label">COLOR</span>
    <select class="sb-select" id="linked-color-select">
      <option value="default"  selected>— Legend colors —</option>
      <option value="model_pgal"       >XGB P(gal)</option>
      <option value="g_mag"            >G mag</option>
      <option value="umap_density"     >Density</option>
{gaia_opts}
    </select>
  </div>"""


def _patch_3d_sidebar(sidebar_html: str, new_controls: str) -> str:
    """Replace the existing OVERLAY bar (and remove Q1 BASE bar) with new controls."""
    # Insert new controls right after #sidebar-header
    header_end = sidebar_html.index("</div>", sidebar_html.index('id="sidebar-header"')) + 6
    result = sidebar_html[:header_end] + "\n" + new_controls + "\n" + sidebar_html[header_end:]

    # Remove the old OVERLAY bar (now handled by linked-dataset-select)
    result = re.sub(
        r'\s*<div id="overlay-bar">.*?</div>\s*',
        "\n", result, flags=re.DOTALL, count=1
    )
    # Remove the Q1 BASE color-mode-bar (now handled by linked-color-select)
    result = re.sub(
        r'\s*<div id="color-mode-bar">\s*<span[^>]*>Q1 BASE</span>.*?</div>\s*',
        "\n", result, flags=re.DOTALL, count=1
    )
    return result


# ── Sky JS patches ────────────────────────────────────────────────────────────

def _patch_sky_js(js: str) -> str:
    # Null-guard 2D umapDiv references (the old 2D UMAP div doesn't exist)
    for old, new in [
        ("Plotly.newPlot(umapDiv,",  "umapDiv && Plotly.newPlot(umapDiv,"),
        ("Plotly.react(umapDiv,",    "umapDiv && Plotly.react(umapDiv,"),
        ("Plotly.relayout(umapDiv,", "umapDiv && Plotly.relayout(umapDiv,"),
        ("Plotly.restyle(umapDiv,",  "umapDiv && Plotly.restyle(umapDiv,"),
        ("Plotly.Plots.resize(umapDiv);",
         "if(typeof plotDiv!=='undefined'&&plotDiv)Plotly.Plots.resize(plotDiv);"),
    ]:
        js = js.replace(old, new)
    js = js.replace("  if (_umapReady) initUMAP();\n", "")

    # Use the pre-reserved WebGL context (created before Plotly) if available
    js = js.replace(
        "var gl = glCanvas.getContext('webgl',",
        "var gl = (window._skyGlReserved || glCanvas.getContext('webgl',"
    )
    # Close the extra parenthesis if replacement happened
    js = js.replace(
        "window._skyGlReserved || glCanvas.getContext('webgl', {antialias:false, depth:false})",
        "window._skyGlReserved || glCanvas.getContext('webgl', {antialias:false, depth:false}))"
    )

    # ── Guard event listeners for removed top-bar control elements ──────────
    # The sky HTML's top controls bar (#controls) is NOT included in the linked
    # layout. Any direct getElementById().addEventListener() on those elements
    # would throw TypeError and stop all subsequent sky JS execution.
    # Wrap each with a null-safe element fallback.
    _removed_ids = [
        'dataset-select', 'z-select', 'color-select',
        'overlay-select', 'overlay-cbar-gradient',
        'overlay-cbar-min', 'overlay-cbar-max', 'overlay-cbar-side',
    ]
    for eid in _removed_ids:
        old_ref = f"document.getElementById('{eid}')"
        new_ref = f"(document.getElementById('{eid}')||{{addEventListener:function(){{}}}})"
        js = js.replace(old_ref, new_ref)

    # ── Patch renderSky() to rebind vertex buffer before drawing ────────────
    # When a canvas is resized (canvas.width = n), Safari/WebGL resets vertex
    # attribute state. We must rebind buf/aPos before gl.drawArrays() each frame.
    js = js.replace(
        "  gl.useProgram(prog);\n  gl.uniform1f(uCl",
        "  gl.useProgram(prog);\n"
        "  if(typeof buf!=='undefined'){gl.bindBuffer(gl.ARRAY_BUFFER,buf);}\n"
        "  if(typeof aPos!=='undefined'){gl.enableVertexAttribArray(aPos);"
        "gl.vertexAttribPointer(aPos,2,gl.FLOAT,false,0,0);}\n"
        "  gl.uniform1f(uCl"
    )

    # Override color-bar update: elements removed from top bar
    js += "\nfunction updateOverlayCbar(){} // no cbar in linked layout\n"
    # Stub (will be overridden by linking JS)
    js += "\nfunction updateUMAPSizes(){} // overridden by linking JS\n"
    return js


# ── Linking JS ────────────────────────────────────────────────────────────────

def _build_linking_js(overlay_sky_json: str) -> str:
    return f"""\
// ══════════════════════════════════════════════════════════════════════════════
// LINKING JS  —  unified DATASET / COLOR / Z AXIS wiring + sky↔3D sync
// ══════════════════════════════════════════════════════════════════════════════

/* Sky overlay datasets (ra/dec/g_mag per overlay source) */
var OVERLAY_SKY = {overlay_sky_json};

/* Map linked-dataset-select value → sky DATASET key and 3D overlay key */
var _DS_TO_SKY  = {{
  gaia_xgb:      'random',
  gaia_hilat:    'hilat',
  allphz:        'allphz',
  euclid_x_desi: 'euclid_x_desi',
  ero:           'ero',
  ero_rebalanced:'ero_rebalanced',
}};
var _DS_TO_3D = {{
  gaia_xgb:      'gaia_xgb',
  gaia_hilat:    'gaia_hilat',
  allphz:        'allphz',
  euclid_x_desi: '_euclid_x_desi',   // special: show both desi + ero_desi
  ero:           'ero',
  ero_rebalanced:'ero_rebalanced',
}};

/* Map linked-color-select value → sky curColor key */
var _COLOR_TO_SKY = {{
  default:     'gal_prob',
  model_pgal:  'gal_prob',
  g_mag:       'phot_g_mean_mag',
  umap_density: null,
}};

/* ── Dataset switch ───────────────────────────────────────────────────── */
function _switchDataset(dsVal) {{
  // 1. Switch sky layer
  var skyKey = _DS_TO_SKY[dsVal];
  if (skyKey === 'random' || skyKey === 'hilat') {{
    if (typeof DATASETS !== 'undefined' && DATASETS[skyKey]) {{
      D = DATASETS[skyKey];
      N = D.ra.length;
    }}
  }} else if (OVERLAY_SKY[skyKey]) {{
    D = OVERLAY_SKY[skyKey];
    N = D.ra.length;
  }}
  selected = new Uint8Array(N);
  hasSelection = false; selCircle = null;
  if (typeof drawSources === 'function') drawSources();
  if (typeof clearStats  === 'function') clearStats();

  // 2. Switch 3D overlay
  var d3Val = _DS_TO_3D[dsVal];
  if (d3Val === '_euclid_x_desi') {{
    // Show both Q1×DESI and ERO×DESI simultaneously
    activeOverlay = 'ero_desi';   // primary for JS state
    if (typeof _showEuclidXDesi === 'function') _showEuclidXDesi();
    else if (typeof switchOverlay === 'function') switchOverlay('ero_desi');
  }} else {{
    if (typeof switchOverlay === 'function') switchOverlay(d3Val);
  }}
}}

/* ── Color switch ────────────────────────────────────────────────────── */
function _switchColor(colorVal) {{
  // 1. Sky
  if (colorVal === 'default') {{
    if (typeof _skyClusterMode !== 'undefined') _skyClusterMode = true;
    if (typeof drawSources === 'function') drawSources();
  }} else if (colorVal === 'umap_density') {{
    if (typeof _skyClusterMode !== 'undefined') _skyClusterMode = false;
    // Sky doesn't change for density (no 2D density map) — leave as-is
  }} else {{
    if (typeof _skyClusterMode !== 'undefined') _skyClusterMode = false;
    var skyKey = _COLOR_TO_SKY[colorVal] || colorVal;
    if (typeof curColor !== 'undefined') curColor = skyKey;
    if (typeof drawSources === 'function') drawSources();
  }}

  // 2. Q1 base (3D flat layer)
  var q1ColorKey = (colorVal === 'default') ? 'clusters' : colorVal;
  if (typeof recolorQ1Base === 'function') recolorQ1Base(q1ColorKey);

  // 3. 3D overlay scatter
  if (colorVal === 'umap_density') {{
    if (typeof _recolorOverlayByDensity === 'function') _recolorOverlayByDensity();
  }} else if (colorVal === 'default') {{
    if (typeof restoreERODefaultColors === 'function') restoreERODefaultColors();
    // Re-apply legend colours to overlay
    if (typeof applyVisibility === 'function') applyVisibility();
  }} else {{
    if (typeof _recolorActiveOverlay === 'function') _recolorActiveOverlay(colorVal);
  }}

  // 4. Sync the hidden 3D color-mode-select so existing handlers still work
  var cmEl = document.getElementById('color-mode-select');
  if (cmEl) {{ cmEl.value = q1ColorKey; }}
}}

/* ── Event listeners ──────────────────────────────────────────────────── */
(function() {{
  var dsEl    = document.getElementById('linked-dataset-select');
  var colorEl = document.getElementById('linked-color-select');

  if (dsEl)    dsEl.addEventListener('change',    function() {{ _switchDataset(this.value); }});
  if (colorEl) colorEl.addEventListener('change', function() {{ _switchColor(this.value); }});
}})();

/* ── override _selectedFeature so density / recolor reads from linked-color ── */
function _selectedFeature() {{
  var el = document.getElementById('linked-color-select');
  if (el) {{
    var v = el.value;
    return (v === 'default') ? 'clusters' : v;
  }}
  var old = document.getElementById('color-mode-select');
  return old ? old.value : 'clusters';
}}

/* ── Sky/UMAP panel divider drag ─────────────────────────────────────── */
(function() {{
  var handle = document.getElementById('sky-divider');
  var skyEl  = document.getElementById('sky-section');
  if (!handle || !skyEl) return;
  var _drag = null;
  handle.addEventListener('mousedown', function(e) {{
    _drag = {{x0: e.clientX, w0: skyEl.getBoundingClientRect().width}};
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    handle.classList.add('dragging');
    e.preventDefault();
  }});
  window.addEventListener('mousemove', function(e) {{
    if (!_drag) return;
    var w = Math.max(200, Math.min(window.innerWidth - 300, _drag.w0 + e.clientX - _drag.x0));
    skyEl.style.flex = '0 0 ' + w + 'px';
    if (typeof resizeCanvases === 'function') {{ resizeCanvases(); renderAll(); }}
    if (typeof plotDiv !== 'undefined' && plotDiv) Plotly.Plots.resize(plotDiv);
  }});
  window.addEventListener('mouseup', function() {{
    if (!_drag) return;
    _drag = null;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    handle.classList.remove('dragging');
  }});
}})();

/* ── override updateUMAPSizes (sky selection → dim 3D) ──────────────── */
function updateUMAPSizes() {{
  if (typeof plotDiv === 'undefined' || !plotDiv || !plotDiv.data) return;
  if (!hasSelection || !selCircle) {{
    // Clear: restore default opacities
    var all = [];
    for (var i = 0; i < plotDiv.data.length; i++) {{
      if (plotDiv.data[i].visible === true) all.push(i);
    }}
    if (all.length) Plotly.restyle(plotDiv, {{'marker.opacity': null, 'marker.size': null}}, all);
    return;
  }}
  var selL = selCircle.l, selB = selCircle.b, selR = selCircle.r;
  var updateIdx = [], ops = [], szs = [];
  plotDiv.data.forEach(function(tr, ti) {{
    if (tr.visible !== true || !tr.x || !tr.x.length) return;
    var defOp = (tr.marker && tr.marker.opacity != null) ? tr.marker.opacity : 0.7;
    var defSz = (tr.marker && tr.marker.size != null) ? tr.marker.size : 3;
    var tOps = [], tSzs = [];
    for (var j = 0; j < tr.x.length; j++) {{
      var cd = tr.customdata && tr.customdata[j];
      var ra = cd && cd[1] != null ? parseFloat(cd[1]) : null;
      var dec = cd && cd[2] != null ? parseFloat(cd[2]) : null;
      var inside = false;
      if (ra != null && dec != null && typeof eqToGal === 'function') {{
        var g = eqToGal(ra, dec);
        inside = angDistGal(g.l, g.b, selL, selB) <= selR;
      }} else {{
        inside = true;
      }}
      tOps.push(inside ? defOp : 0.05);
      tSzs.push(inside ? defSz : Math.max(1, defSz * 0.5));
    }}
    updateIdx.push(ti); ops.push(tOps); szs.push(tSzs);
  }});
  if (updateIdx.length)
    Plotly.restyle(plotDiv, {{'marker.opacity': ops, 'marker.size': szs}}, updateIdx);
}}

/* ── 3D hover → sky crosshair ────────────────────────────────────────── */
function _install3DHover() {{
  if (typeof plotDiv === 'undefined' || !plotDiv) return;
  plotDiv.on('plotly_hover', function(data) {{
    if (!data.points || !data.points[0]) return;
    var pt = data.points[0], cd = pt.customdata, nm = (pt.data && pt.data.name) || '';
    var ra, dec;
    if (/^gxgb_|^gxhl_/.test(nm))              {{ ra=cd&&cd[2]; dec=cd&&cd[3]; }}
    else if (/^ed_cat=/.test(nm))               {{ ra=cd&&cd[6]; dec=cd&&cd[7]; }}
    else if (/^(Cluster \\d+|noise)$/.test(nm)) {{ if(typeof clearHoverMarker==='function')clearHoverMarker(); return; }}
    else                                        {{ ra=cd&&cd[1]; dec=cd&&cd[2]; }}
    if (ra==null) {{ if(typeof clearHoverMarker==='function')clearHoverMarker(); return; }}
    if (typeof eqToGal==='function' && typeof drawHoverMarker==='function') {{
      var g = eqToGal(parseFloat(ra), parseFloat(dec));
      drawHoverMarker(g.l, g.b);
    }}
  }});
  plotDiv.on('plotly_unhover', function() {{
    if(typeof clearHoverMarker==='function') clearHoverMarker();
  }});
}}

/* ── Resize helper ───────────────────────────────────────────────────── */
function _doResize() {{
  if (typeof resizeCanvases === 'function') {{ resizeCanvases(); }}
  if (typeof renderAll     === 'function') {{ renderAll(); }}
  if (typeof resizePlot    === 'function') {{ resizePlot(); }}
  else if (typeof plotDiv  !== 'undefined' && plotDiv) {{ Plotly.Plots.resize(plotDiv); }}
}}

/* ── Hook plotly_afterplot to resize once Plotly finishes rendering ── */
(function() {{
  if (typeof plotDiv === 'undefined' || !plotDiv || typeof plotDiv.on !== 'function') return;
  var _fired = false;
  plotDiv.on('plotly_afterplot', function _onFirstRender() {{
    if (_fired) return;
    _fired = true;
    _doResize();
    _install3DHover();
  }});
}})();

/* ── Sky-specific: poll until skyReady becomes true, then render ─────── */
/* Handles cases where skyImg.onload fired but WebGL state was stale.     */
(function() {{
  var _attempts = 0;
  var _skyInterval = setInterval(function() {{
    _attempts++;
    if (_attempts > 60) {{ clearInterval(_skyInterval); return; }} // give up after 60s

    /* If sky is ready, force a render and stop */
    if (typeof skyReady !== 'undefined' && skyReady) {{
      if (typeof resizeCanvases === 'function') resizeCanvases();
      if (typeof renderAll      === 'function') renderAll();
      /* Keep running a few more times to ensure it sticks */
      if (_attempts > 3) clearInterval(_skyInterval);
      return;
    }}

    /* If skyImg exists but hasn't loaded yet, nudge it */
    if (typeof skyImg !== 'undefined' && skyImg && !skyImg.complete) {{
      return; // still loading, wait
    }}

    /* Image claims to be complete but skyReady is still false:
       re-trigger the onload by resetting src (forces browser to re-fire) */
    if (typeof skyImg !== 'undefined' && skyImg && skyImg.complete && _attempts % 3 === 0) {{
      var _src = skyImg.src;
      if (_src && _src !== '') {{
        skyImg.src = '';
        skyImg.src = _src;
      }}
    }}
  }}, 1000);
}})();

/* ── WebGL context loss recovery ─────────────────────────────────────── */
(function() {{
  var skyCanvas = document.getElementById('sky-gl');
  if (!skyCanvas) return;
  skyCanvas.addEventListener('webglcontextlost', function(e) {{
    e.preventDefault(); // allow restoration
  }});
  skyCanvas.addEventListener('webglcontextrestored', function() {{
    /* Re-trigger image load to re-upload texture */
    if (typeof skyImg !== 'undefined' && skyImg) {{
      if (typeof skyReady !== 'undefined') skyReady = false;
      var _src = skyImg.src;
      skyImg.src = '';
      skyImg.src = _src || 'data/sky_image.jpg';
    }}
  }});
}})();

/* ── CSS sky background fallback ────────────────────────────────────── */
/* The WebGL canvas is transparent when no pixels are written.
   Setting the CSS background on #sky-viewport shows through the canvas,
   giving guaranteed sky visibility regardless of WebGL state.
   When WebGL renders correctly, its opaque pixels cover this background. */
(function() {{
  var vp = document.getElementById('sky-viewport');
  if (!vp) return;
  vp.style.backgroundImage = "url('data/sky_image.jpg')";
  vp.style.backgroundSize = "cover";
  vp.style.backgroundPosition = "center center";
  vp.style.backgroundRepeat = "no-repeat";
}})();

/* ── Safe sky texture upload (handles MAX_TEXTURE_SIZE limits) ───────── */
/* The sky image is 8000×4000. If the GPU's MAX_TEXTURE_SIZE < 8000,
   gl.texImage2D silently fails and the sky stays black.
   This function scales the image into a 2D canvas first if needed.       */
function _uploadSkyTexture(imgSrc) {{
  if (typeof gl === 'undefined' || !gl || gl.isContextLost()) return;
  if (typeof skyTex === 'undefined') return;
  var maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE) || 4096;
  var img = new Image();
  img.onload = function() {{
    var src = img;
    /* Scale down if image exceeds max texture size */
    if (img.naturalWidth > maxTex || img.naturalHeight > maxTex) {{
      var scale = maxTex / Math.max(img.naturalWidth, img.naturalHeight);
      var c = document.createElement('canvas');
      c.width  = Math.floor(img.naturalWidth  * scale);
      c.height = Math.floor(img.naturalHeight * scale);
      c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
      src = c;
    }}
    if (gl.isContextLost()) return;
    gl.bindTexture(gl.TEXTURE_2D, skyTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    if (typeof skyReady !== 'undefined') skyReady = true;
    if (typeof resizeCanvases === 'function') resizeCanvases();
    if (typeof renderAll      === 'function') renderAll();
  }};
  img.src = imgSrc;
}}

/* Run safe upload 1s after load to give Plotly time to initialise */
setTimeout(function() {{
  _uploadSkyTexture('data/sky_image.jpg');
}}, 1000);

/* ── Fallback: trigger on window.load ────────────────────────────────── */
window.addEventListener('load', function() {{
  setTimeout(_doResize, 200);
  setTimeout(function() {{ _uploadSkyTexture('data/sky_image.jpg'); }}, 500);
}});

/* ── Fallback timeout: fire after 3s in case Plotly is very slow ─────── */
setTimeout(function() {{
  if (typeof plotDiv === 'undefined' || !plotDiv) {{
    var gd = document.querySelector('#plot-container .js-plotly-plot');
    if (gd) plotDiv = gd;
  }}
  _doResize();
  _install3DHover();
}}, 3000);

window.addEventListener('resize', _doResize);
"""


# ── Main assembly ─────────────────────────────────────────────────────────────

def main() -> int:
    for f in (SKY_HTML, UMAP_3D):
        if not f.exists():
            print(f"ERROR: {f} not found — run compile_umap_sky.py and compile_umap_3d.py first.")
            return 1

    # ── Load overlay sky datasets ──────────────────────────────────────────
    overlay_sky = load_overlay_sky_datasets()
    overlay_sky_json = json.dumps(overlay_sky)

    # ── Read source HTMLs ──────────────────────────────────────────────────
    print(f"Reading {SKY_HTML.name}  ({SKY_HTML.stat().st_size/1e6:.0f} MB) ...")
    sky_html = SKY_HTML.read_text(encoding="utf-8")

    print(f"Reading {UMAP_3D.name}   ({UMAP_3D.stat().st_size/1e6:.0f} MB) ...")
    d3_html  = UMAP_3D.read_text(encoding="utf-8")

    # ── Extract sky pieces ─────────────────────────────────────────────────
    print("Extracting sky pieces ...")
    sky_css  = _extract_css(sky_html)
    sky_body = _extract_body(sky_html)
    sky_js   = _extract_plain_script(sky_body)
    sky_vp   = _div_outer(sky_body, "sky-viewport")   # WebGL canvas + minimap
    sky_stat = _div_outer(sky_body, "stats-panel")    # bottom stats bar

    # Reserve the sky's WebGL context BEFORE Plotly.newPlot() runs.
    # Plotly.newPlot() (in the umap-section inline scripts) synchronously creates
    # a three.js WebGLRenderer — taking one WebGL context. On some Safari versions
    # this exhausts the context budget. By reserving the sky context first (via an
    # inline script that runs during HTML parsing, before the Plotly scripts),
    # we guarantee the sky gets its context regardless of Plotly's allocation.
    sky_early_ctx = """\
<script>
/* Reserve sky WebGL context before Plotly takes GPU resources */
(function(){
  var c = document.getElementById('sky-gl');
  if (c) window._skyGlReserved = c.getContext('webgl', {antialias:false, depth:false});
})();
</script>"""

    sky_panel_html = (
        '<div id="sky-section">\n'
        + sky_vp + "\n"
        + sky_stat + "\n"
        + "</div>\n"
        + sky_early_ctx  # placed AFTER sky-section, BEFORE umap-section (Plotly)
    )

    # ── Extract 3D pieces ──────────────────────────────────────────────────
    print("Extracting 3D pieces ...")
    d3_css   = _extract_css(d3_html)
    d3_body  = _extract_body(d3_html)
    d3_js    = _extract_plain_script(d3_body)

    # 3D non-script HTML (everything before the last plain <script>)
    plain_scripts  = list(re.finditer(r"<script>", d3_body))
    d3_non_script  = d3_body[: plain_scripts[-1].start()].strip()

    # Patch the sidebar: add DATASET/COLOR, remove old OVERLAY/Q1-BASE controls
    new_controls   = _make_sidebar_controls_html()
    d3_non_script  = _patch_3d_sidebar(d3_non_script, new_controls)

    umap_section_html = (
        '<div id="umap-section">\n'
        + d3_non_script + "\n"
        + "</div>"
    )

    # ── Patch sky JS ───────────────────────────────────────────────────────
    sky_js = _patch_sky_js(sky_js)

    # ── Patch 3D JS: null-guard getElementById calls on removed elements ───
    # We removed #overlay-bar (overlay-select) and Q1-BASE bar (color-mode-select)
    # from the sidebar. Direct .addEventListener/.value calls on these null
    # elements crash the 3D JS, preventing _initIndices() + applyVisibility().
    _d3_removed = ['overlay-select', 'color-mode-select']
    for eid in _d3_removed:
        # Guard .addEventListener(... calls
        old_ae = f"document.getElementById('{eid}').addEventListener("
        new_ae = f"(document.getElementById('{eid}')||{{addEventListener:function(){{}}}}).addEventListener("
        d3_js = d3_js.replace(old_ae, new_ae)
        # Guard direct .value reads
        old_val = f"document.getElementById('{eid}').value"
        new_val = f"((document.getElementById('{eid}')||{{}}).value||'clusters')"
        d3_js = d3_js.replace(old_val, new_val)

    # ── Build linking JS ──────────────────────────────────────────────────
    linking_js = _build_linking_js(overlay_sky_json)

    # ── Assemble ──────────────────────────────────────────────────────────
    print("Assembling combined HTML ...")
    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sky + 3D UMAP Explorer</title>
<style>
/* ── Sky styles ── */
{sky_css}
/* ── 3D styles ── */
{d3_css}
/* ── Sidebar control bar styles ── */
{_CTRL_CSS}
/* ── Combined layout (overrides body rules above) ── */
{_LAYOUT_CSS}
</style>
</head>
<body>
{sky_panel_html}
<div id="sky-divider"></div>
{umap_section_html}
<script>
/* ════ Sky JS ════ */
{sky_js}
/* ════ 3D JS ════ */
{d3_js}
/* ════ Linking JS ════ */
{linking_js}
</script>
</body>
</html>"""

    # ── Write ──────────────────────────────────────────────────────────────
    print(f"Writing {OUT_FILE.name} ...")
    OUT_FILE.write_text(html, encoding="utf-8")
    sz = OUT_FILE.stat().st_size / 1e6
    print(f"\n  → {OUT_FILE}  ({sz:.0f} MB)")
    print(f"  open {OUT_FILE}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
