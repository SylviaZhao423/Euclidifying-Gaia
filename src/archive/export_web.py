#!/usr/bin/env python3
"""
Post-process umap_linked.html: externalize embedded image data for web deployment.

Reads the monolithic umap_linked.html (~1079 MB) and produces a deployable directory:

    output/umap/q1_individual/web/
        index.html          (~60 MB — sky JS + plot data + linking JS)
        data/
            sky_image.jpg           (Gaia DR3 sky background, ~7 MB)
            contact_sheets.json
            desi_contact_sheets.json
            ero_contact_sheets.json
            ero_desi_contact_sheets.json
            cutouts.json
            ero_cutouts.json
            sheet_extra_pages.json  (lazy)
            desi_sheet_extra.json   (lazy)
            ero_sheet_extra.json    (lazy)
            ero_desi_sheet_extra.json (lazy)

Image data is fetched on demand via relative URLs, so the directory can be
served from any static host (GitHub Pages, Netlify, Cloudflare Pages, etc.).

Usage:
    python src/export_web.py
    # then:  cd output/umap/q1_individual/web && python -m http.server 8000
"""

from __future__ import annotations

import base64
import json
import re
import sys
import time
from pathlib import Path

INPUT  = Path(__file__).resolve().parents[1] / "output" / "umap" / "q1_individual" / "umap.html"
OUTDIR = Path(__file__).resolve().parents[1] / "output" / "umap" / "q1_individual" / "web"

# Loaded eagerly (contact sheets for transposition + cutouts for hover).
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

    eq_pos    = html.index("=", pos)
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


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    if not INPUT.exists():
        print(f"ERROR: {INPUT} not found.")
        print("       Run  python src/compile_umap_linked.py  first.")
        return 1

    print(f"Reading {INPUT.name} ...")
    t0 = time.time()
    html = INPUT.read_text(encoding="utf-8")
    print(f"  {len(html)/1e6:.1f} MB read in {time.time()-t0:.1f}s")

    data_dir = OUTDIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    replacements: list[tuple[int, int, str]] = []   # (start, end, new_text)

    # ── 1. Externalize Gaia sky image ────────────────────────────────────────
    print("\n[1] Extracting Gaia sky image ...")
    sky_pattern = re.compile(r"skyImg\.src\s*=\s*'data:image/jpeg;base64,([A-Za-z0-9+/=]+)';")
    sky_match = sky_pattern.search(html)
    if sky_match:
        b64_data = sky_match.group(1)
        jpg_path = data_dir / "sky_image.jpg"
        jpg_path.write_bytes(base64.b64decode(b64_data))
        sz_mb = jpg_path.stat().st_size / 1e6
        print(f"  sky_image.jpg: {sz_mb:.1f} MB → {jpg_path.name}")
        # Replace the data URI assignment with a relative URL
        replacements.append((
            sky_match.start(),
            sky_match.end(),
            "skyImg.src = 'data/sky_image.jpg';",
        ))
    else:
        print("  WARNING: skyImg.src not found — sky image stays inline")

    # ── 2. Extract each 3D image variable and write to JSON ──────────────────
    print("\n[2] Extracting 3D image variables ...")
    for varname in ALL_VARS:
        try:
            start, end, json_body = _find_var_span(html, varname)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  WARNING: {varname}: {e} — skipped")
            continue

        out_path = data_dir / f"{varname}.json"
        out_path.write_text(json_body, encoding="utf-8")
        sz_mb = len(json_body) / 1e6
        print(f"  {varname}: {sz_mb:.1f} MB → {out_path.name}")

        if varname in EAGER_VARS:
            replacements.append((start, end, f"var {varname} = {{}};"))
        else:
            replacements.append((start, end, _lazy_proxy_stub(varname)))

    # ── 3. Replace the startup transposition block with eager fetch ───────────
    print("\n[3] Injecting eager-fetch loader ...")

    trans_start_marker = "var cluster_sheets = {};"
    trans_end_marker   = "/* Q1 feature names"
    ts = html.find(trans_start_marker)
    te = html.find(trans_end_marker)
    if ts == -1 or te == -1:
        print("  WARNING: transposition block not found — skipping eager fetch")
    else:
        eager_block = """var cluster_sheets = {}, desi_cluster_sheets = {}, ero_desi_cluster_sheets = {};
var _loadingExtras = new Set();

/* Non-blocking sidebar loading indicator */
var _loadedCount = 0, _totalEager = 6;
function _updateLoadHint(name) {
  _loadedCount++;
  var el = document.getElementById('sidebar-hint');
  if (!el) return;
  if (_loadedCount < _totalEager) {
    el.innerHTML = 'Loading image data... <span style="color:#666">' + _loadedCount + '/' + _totalEager + '</span>';
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
    el.innerHTML = 'Loading image data... <span style="color:#666">0/' + _totalEager + '</span>';
}, 0);

var _dataReady = Promise.all([
  _jfetch('contact_sheets'),
  _jfetch('desi_contact_sheets'),
  _jfetch('ero_contact_sheets'),
  _jfetch('ero_desi_contact_sheets'),
  _jfetch('cutouts'),
  _jfetch('ero_cutouts'),
]).then(function(arr) {
  contact_sheets = arr[0];
  desi_contact_sheets = arr[1];
  ero_contact_sheets = arr[2];
  ero_desi_contact_sheets = arr[3];
  cutouts = arr[4];
  ero_cutouts = arr[5];
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
        print("  OK — eager-fetch block injected")

    # ── 4. Apply all replacements (reverse order to preserve offsets) ─────────
    print("\n[4] Applying replacements ...")
    replacements.sort(key=lambda r: r[0], reverse=True)
    for start, end, new_text in replacements:
        html = html[:start] + new_text + html[end:]

    # ── 5. Write index.html ───────────────────────────────────────────────────
    out_html = OUTDIR / "index.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"\n[5] Output written to {OUTDIR}/")

    # ── 6. Report sizes ───────────────────────────────────────────────────────
    print("\n  File sizes:")
    total = 0
    for f in sorted(OUTDIR.rglob("*")):
        if f.is_file():
            sz = f.stat().st_size
            total += sz
            print(f"    {f.relative_to(OUTDIR)}: {sz/1e6:.1f} MB")
    print(f"    {'─'*33}")
    print(f"    total: {total/1e6:.1f} MB")
    print(f"\n  To test locally:")
    print(f"    cd {OUTDIR} && python -m http.server 8000")
    print(f"    # then open http://localhost:8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
