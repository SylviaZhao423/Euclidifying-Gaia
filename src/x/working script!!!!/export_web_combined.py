#!/usr/bin/env python3
"""
Post-process umap_combined.html: externalize embedded image/sky data for web deployment.

Reads the monolithic umap_combined.html (~1 GB, mostly base64 cutout/contact-sheet
images + sky data) and produces a deployable directory:

    output/umap/q1_individual/web_combined/
        index.html          (~17 MB — plot data + JS code)
        data/
            cutouts.json
            contact_sheets.json
            sky_datasets.json
            sky_image.jpg
            ...

Image data and sky coordinates are fetched on demand via relative URLs, so the
directory can be served from any static host (GitHub Pages, Netlify, etc.).

Usage:
    python src/export_web_combined.py
    # then:  cd output/umap/q1_individual/web_combined && python -m http.server 8100
"""

from __future__ import annotations

import base64
import json
import re
import sys
import time
from pathlib import Path

INPUT  = Path(__file__).resolve().parents[1] / "output" / "umap" / "q1_individual" / "umap_combined.html"
OUTDIR = Path(__file__).resolve().parents[1] / "output" / "umap" / "q1_individual" / "web_combined"

# Loaded at startup (contact sheets for transposition + cutouts for hover).
# Plot initialises only after all of these arrive.
EAGER_VARS = [
    "contact_sheets",
    "desi_contact_sheets",
    "ero_contact_sheets",
    "ero_desi_contact_sheets",
    "cutouts",
    "ero_cutouts",
]

# Extra contact-sheet pages — large, rarely needed (only when clicking
# through multi-page sheets).  Loaded lazily on first access.
LAZY_VARS = [
    "sheet_extra_pages",
    "desi_sheet_extra",
    "ero_sheet_extra",
    "ero_desi_sheet_extra",
]

ALL_VARS = EAGER_VARS + LAZY_VARS


# ── helpers ──────────────────────────────────────────────────────────────────

def _find_var_span(html: str, varname: str) -> tuple[int, int, str]:
    """Find  var <varname> = <JSON>;  and return (start, end, json_body).

    Uses json.JSONDecoder.raw_decode to correctly handle nested braces.
    """
    marker = f"var {varname}"
    pos = html.find(marker)
    if pos == -1:
        for sp in ["  ", "\t"]:
            alt = f"var{sp}{varname}"
            pos = html.find(alt)
            if pos != -1:
                break
    if pos == -1:
        raise ValueError(f"Variable '{varname}' not found in HTML")

    eq_pos = html.index("=", pos)
    brace_pos = html.index("{", eq_pos)

    decoder = json.JSONDecoder()
    _, end_offset = decoder.raw_decode(html, brace_pos)
    json_body = html[brace_pos:end_offset]

    semi_pos = html.index(";", end_offset)
    return pos, semi_pos + 1, json_body


def _lazy_proxy_stub(varname: str) -> str:
    """JS code: Proxy that fetches on first access + marks loading state."""
    return (
        f"var {varname} = new Proxy({{}}, {{get:function(t,k){{\n"
        f"  if(typeof k==='symbol') return undefined;\n"
        f"  if(!t._f){{t._f=1;\n"
        f"    if(typeof _loadingExtras!=='undefined') _loadingExtras.add('{varname}');\n"
        f"    fetch('data/{varname}.json')\n"
        f"      .then(function(r){{return r.json();}})\n"
        f"      .then(function(d){{{varname}=d;\n"
        f"        if(typeof _loadingExtras!=='undefined') _loadingExtras.delete('{varname}');\n"
        f"      }});}}\n"
        f"  return t[k];\n"
        f"}}}});"
    )


def _extract_sky_image(html: str, data_dir: Path) -> str:
    """Extract base64 sky image from HTML, save as JPEG, replace inline refs.

    The sky image appears in two places:
      1. skyImg.src = 'data:image/jpeg;base64,...'  (WebGL texture)
      2. <img src="data:image/jpeg;base64,...">       (minimap thumbnail)

    Returns the modified HTML.
    """
    # Find the base64 data in the skyImg.src assignment
    pattern = r"skyImg\.src\s*=\s*'(data:image/jpeg;base64,([A-Za-z0-9+/=\s]+))'"
    match = re.search(pattern, html)
    if not match:
        print("  WARNING: sky image base64 not found in HTML — skipping")
        return html

    full_data_uri = match.group(1)
    b64_data = match.group(2).replace("\n", "").replace(" ", "")

    # Decode and save as JPEG
    img_bytes = base64.b64decode(b64_data)
    sky_path = data_dir / "sky_image.jpg"
    sky_path.write_bytes(img_bytes)
    print(f"  Sky image: {len(img_bytes)/1e6:.1f} MB → {sky_path.name}")

    # Replace skyImg.src with relative URL
    html = html.replace(
        f"skyImg.src = '{full_data_uri}'",
        "skyImg.src = 'data/sky_image.jpg'"
    )

    # Replace minimap <img src="data:image/jpeg;base64,...">
    minimap_pattern = r'<img src="(data:image/jpeg;base64,[A-Za-z0-9+/=\s]+)">'
    html = re.sub(minimap_pattern, '<img src="data/sky_image.jpg">', html)

    return html


def _extract_sky_datasets(html: str, data_dir: Path) -> tuple[str, int, int]:
    """Extract SKY_DATASETS JSON, save to file, replace with fetch stub.

    Returns (modified_html, start_pos, end_pos) of the replacement.
    """
    start, end, json_body = _find_var_span(html, "SKY_DATASETS")
    out_path = data_dir / "sky_datasets.json"
    out_path.write_text(json_body)
    sz_mb = len(json_body) / 1e6
    print(f"  SKY_DATASETS: {sz_mb:.1f} MB → {out_path.name}")
    return html, start, end


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    if not INPUT.exists():
        print(f"ERROR: {INPUT} not found.  Run compile_umap_combined.py first.")
        return 1

    print(f"Reading {INPUT.name} ...")
    t0 = time.time()
    html = INPUT.read_text()
    print(f"  {len(html)/1e6:.1f} MB read in {time.time()-t0:.1f}s")

    data_dir = OUTDIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Extract sky image to separate JPEG ────────────────────────────
    print("\n[1] Extracting sky image ...")
    html = _extract_sky_image(html, data_dir)

    # ── 2. Extract image variables to JSON ───────────────────────────────
    print("\n[2] Extracting image variables ...")
    replacements: list[tuple[int, int, str]] = []

    for varname in ALL_VARS:
        try:
            start, end, json_body = _find_var_span(html, varname)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  WARNING: {varname}: {e} — skipped")
            continue

        out_path = data_dir / f"{varname}.json"
        out_path.write_text(json_body)
        sz_mb = len(json_body) / 1e6
        print(f"  {varname}: {sz_mb:.1f} MB → {out_path.name}")

        if varname in EAGER_VARS:
            replacements.append((start, end, f"var {varname} = {{}};"))
        else:
            replacements.append((start, end, _lazy_proxy_stub(varname)))

    # ── 3. Extract SKY_DATASETS to JSON ──────────────────────────────────
    print("\n[3] Extracting SKY_DATASETS ...")
    html, sky_start, sky_end = _extract_sky_datasets(html, data_dir)

    # Replace SKY_DATASETS with empty obj; will be loaded by eager fetch
    replacements.append((sky_start, sky_end, "var SKY_DATASETS = {};"))

    # ── 4. Replace transposition block with eager-fetch loader ───────────
    print("\n[4] Injecting eager-fetch loader ...")

    trans_start_marker = "var cluster_sheets = {};"
    trans_end_marker   = "/* Q1 feature names"
    ts = html.find(trans_start_marker)
    te = html.find(trans_end_marker)
    if ts == -1 or te == -1:
        print("  WARNING: transposition block not found — skipping eager fetch")
    else:
        # The eager-fetch block loads image data AND sky datasets
        eager_block = """var cluster_sheets = {}, desi_cluster_sheets = {}, ero_desi_cluster_sheets = {};
var _loadingExtras = new Set();

/* Non-blocking sidebar loading indicator */
var _loadedCount = 0, _totalEager = 8;
function _updateLoadHint(name) {
  _loadedCount++;
  var el = document.getElementById('sidebar-hint');
  if (!el) return;
  if (_loadedCount < _totalEager) {
    el.innerHTML = 'Loading data... <span style="color:#666">' + _loadedCount + '/' + _totalEager + '</span>';
  } else {
    el.textContent = 'hover a point \\u00b7 double-click to pin';
  }
}
function _jfetch(name) {
  return fetch('data/' + name + '.json')
    .then(function(r){return r.json();})
    .then(function(d){ _updateLoadHint(name); return d; });
}

/* Show loading state immediately */
setTimeout(function() {
  var el = document.getElementById('sidebar-hint');
  if (el && _loadedCount < _totalEager)
    el.innerHTML = 'Loading data... <span style="color:#666">0/' + _totalEager + '</span>';
}, 0);

var _dataReady = Promise.all([
  _jfetch('contact_sheets'),
  _jfetch('desi_contact_sheets'),
  _jfetch('ero_contact_sheets'),
  _jfetch('ero_desi_contact_sheets'),
  _jfetch('cutouts'),
  _jfetch('ero_cutouts'),
  _jfetch('sky_datasets'),
  new Promise(function(resolve) { _updateLoadHint('sky_image'); resolve(); }),
]).then(function(arr) {
  contact_sheets = arr[0];
  desi_contact_sheets = arr[1];
  ero_contact_sheets = arr[2];
  ero_desi_contact_sheets = arr[3];
  cutouts = arr[4];
  ero_cutouts = arr[5];
  /* Restore sky datasets and re-init sky panel */
  SKY_DATASETS = arr[6];
  var _firstKey = Object.keys(SKY_DATASETS)[0] || 'gaia_random';
  D = SKY_DATASETS[_firstKey] || {ra:[],dec:[],gl:[],gb:[],ipx:[],ipy:[],u1:[],u2:[],sid:[]};
  N = D.ra ? D.ra.length : 0;
  selected = new Uint8Array(N);
  hasSelection = false; selCircle = null;
  resizeCanvases(); renderAll(); clearStats();
  /* Transpose contact sheets */
  Object.keys(contact_sheets).forEach(function(v) {
    Object.keys(contact_sheets[v]).forEach(function(k) {
      if (!cluster_sheets[k]) cluster_sheets[k] = {};
      cluster_sheets[k][v] = contact_sheets[v][k];
    });
  });
  Object.keys(desi_contact_sheets).forEach(function(v) {
    Object.keys(desi_contact_sheets[v]).forEach(function(k) {
      if (!desi_cluster_sheets[k]) desi_cluster_sheets[k] = {};
      desi_cluster_sheets[k][v] = desi_contact_sheets[v][k];
    });
  });
  Object.keys(ero_desi_contact_sheets).forEach(function(v) {
    Object.keys(ero_desi_contact_sheets[v]).forEach(function(k) {
      if (!ero_desi_cluster_sheets[k]) ero_desi_cluster_sheets[k] = {};
      ero_desi_cluster_sheets[k][v] = ero_desi_contact_sheets[v][k];
    });
  });
});

"""
        replacements.append((ts, te, eager_block))

    # ── 5. Apply all replacements (reverse order to preserve offsets) ─────
    print("\n[5] Applying replacements ...")
    replacements.sort(key=lambda r: r[0], reverse=True)
    for start, end, new_text in replacements:
        html = html[:start] + new_text + html[end:]

    # ── 6. Write index.html ──────────────────────────────────────────────
    out_html = OUTDIR / "index.html"
    out_html.write_text(html)
    print(f"\n[6] Output written to {OUTDIR}/")

    # ── 7. Report sizes ──────────────────────────────────────────────────
    print("\n  File sizes:")
    total = 0
    for f in sorted(OUTDIR.rglob("*")):
        if f.is_file():
            sz = f.stat().st_size
            total += sz
            print(f"    {f.relative_to(OUTDIR)}: {sz/1e6:.1f} MB")
    print(f"    ─────────────────────────────")
    print(f"    total: {total/1e6:.1f} MB")
    print(f"\n  To test locally:")
    print(f"    cd {OUTDIR} && python -m http.server 8100")
    print(f"    # then open http://localhost:8100")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
