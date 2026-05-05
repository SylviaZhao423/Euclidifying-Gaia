#!/usr/bin/env python3
"""
Post-process umap_3d.html: externalize embedded image data for web deployment.

Reads the monolithic umap_3d.html (~430 MB, mostly base64 cutout/contact-sheet
images) and produces a deployable directory:

    output/umap/q1_individual/web_3d/
        index.html          (~17 MB — plot data + JS code)
        data/
            cutouts.json
            contact_sheets.json
            ...

Image data is fetched on demand via relative URLs, so the directory can be
served from any static host (GitHub Pages, Netlify, Cloudflare Pages, etc.).

Usage:
    python src/export_web_3d.py
    # then:  cd output/umap/q1_individual/web_3d && python -m http.server 8100
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

INPUT  = Path(__file__).resolve().parents[1] / "output" / "umap" / "q1_individual" / "umap_3d.html"
OUTDIR = Path(__file__).resolve().parents[1] / "output" / "umap" / "q1_individual" / "web_3d"

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
    # There could be whitespace variants; search for the exact marker
    pos = html.find(marker)
    if pos == -1:
        # Try with extra spaces
        for sp in ["  ", "\t"]:
            alt = f"var{sp}{varname}"
            pos = html.find(alt)
            if pos != -1:
                break
    if pos == -1:
        raise ValueError(f"Variable '{varname}' not found in HTML")

    # Advance past the '=' to the opening brace
    eq_pos = html.index("=", pos)
    brace_pos = html.index("{", eq_pos)

    decoder = json.JSONDecoder()
    _, end_offset = decoder.raw_decode(html, brace_pos)
    json_body = html[brace_pos:end_offset]

    # The full declaration ends at the next ';'
    semi_pos = html.index(";", end_offset)
    return pos, semi_pos + 1, json_body


def _lazy_proxy_stub(varname: str) -> str:
    """JS code: Proxy that fetches on first access + marks loading state."""
    # _loadingExtras is a Set managed by the eager-fetch block (injected later).
    # The sidebar code checks it to show "Loading extra pages..." when needed.
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
        print(f"ERROR: {INPUT} not found.  Run analyze_3dumap_q1.py --3d first.")
        return 1

    print(f"Reading {INPUT.name} ...")
    t0 = time.time()
    html = INPUT.read_text()
    print(f"  {len(html)/1e6:.1f} MB read in {time.time()-t0:.1f}s")

    data_dir = OUTDIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Extract each image variable and write to JSON ─────────────────
    print("\n[1] Extracting image variables ...")
    replacements: list[tuple[int, int, str]] = []   # (start, end, new_text)

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
            # Will be loaded by the eager-fetch block; declare as empty for now
            replacements.append((start, end, f"var {varname} = {{}};"))
        else:
            replacements.append((start, end, _lazy_proxy_stub(varname)))

    # ── 2. Replace the startup transposition block with eager fetch ───────
    print("\n[2] Injecting eager-fetch loader ...")

    # Find the transposition block:  var cluster_sheets = {}; ... });
    # It starts at "var cluster_sheets = {};" and ends after the
    # ero_desi_cluster_sheets forEach block (before "/* Q1 feature names")
    trans_start_marker = "var cluster_sheets = {};"
    trans_end_marker   = "/* Q1 feature names"
    ts = html.find(trans_start_marker)
    te = html.find(trans_end_marker)
    if ts == -1 or te == -1:
        print("  WARNING: transposition block not found — skipping eager fetch")
    else:
        # The eager-fetch block replaces everything from cluster_sheets decl
        # through the end of the ero_desi forEach (up to trans_end_marker)
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

    # ── 3. Init code runs immediately (no wrapping needed) ─────────────
    #    Plot init (_initIndices, applyVisibility, recolorQ1Base) doesn't
    #    depend on image data.  Contact sheet transposition and cutout
    #    lookups happen in _dataReady.then() above.
    print("[3] Init code runs immediately (no _dataReady wrapper needed).")

    # ── 4. Apply all replacements (reverse order to preserve offsets) ─────
    print("\n[4] Applying replacements ...")
    replacements.sort(key=lambda r: r[0], reverse=True)
    for start, end, new_text in replacements:
        html = html[:start] + new_text + html[end:]

    # ── 5. Write index.html ──────────────────────────────────────────────
    out_html = OUTDIR / "index.html"
    out_html.write_text(html)
    print(f"\n[5] Output written to {OUTDIR}/")

    # ── 6. Report sizes ──────────────────────────────────────────────────
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
