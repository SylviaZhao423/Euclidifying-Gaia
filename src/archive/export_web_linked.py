#!/usr/bin/env python3
"""
export_web_linked.py
====================
Post-process umap_linked.html: externalize embedded image data for web serving.

Reads:
    output/umap/q1_individual/umap_linked.html   (~2 GB standalone)

Produces:
    output/umap/q1_individual/web_linked/
        index.html          (~130 MB  — all JS/data except images)
        data/
            sky_image.jpg                  (~10 MB, Gaia DR3 background)
            contact_sheets.json            (Q1 PHZ contact sheets)
            desi_contact_sheets.json       (Q1×DESI contact sheets)
            ero_contact_sheets.json        (ERO contact sheets)
            ero_desi_contact_sheets.json   (ERO×DESI contact sheets)
            cutouts.json                   (Q1 cutout thumbnails)
            ero_cutouts.json               (ERO cutout thumbnails)
            sheet_extra_pages.json         (lazy — extra contact-sheet pages)
            desi_sheet_extra.json          (lazy)
            ero_sheet_extra.json           (lazy)
            ero_desi_sheet_extra.json      (lazy)

Usage:
    python src/compile_umap_linked.py        # build umap_linked.html first
    python src/export_web_linked.py          # split for serving
    cd output/umap/q1_individual/web_linked && python -m http.server 8000
"""
from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path

INPUT  = Path(__file__).resolve().parents[1] / "output" / "umap" / "q1_individual" / "umap_linked.html"
OUTDIR = Path(__file__).resolve().parents[1] / "output" / "umap" / "q1_individual" / "web_linked"

# ── Variables to externalize ──────────────────────────────────────────────────

# Loaded eagerly on page load (needed before user can hover/click anything)
EAGER_VARS = [
    "contact_sheets",
    "desi_contact_sheets",
    "ero_contact_sheets",
    "ero_desi_contact_sheets",
    "cutouts",
    "ero_cutouts",
]

# Loaded lazily on first access (only when user pages through contact sheets)
LAZY_VARS = [
    "sheet_extra_pages",
    "desi_sheet_extra",
    "ero_sheet_extra",
    "ero_desi_sheet_extra",
]

ALL_VARS = EAGER_VARS + LAZY_VARS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_all_var_spans(html: str, varname: str) -> list[tuple[int, int, str]]:
    """Find ALL occurrences of  var <varname> = <JSON>;  in html.

    The combined HTML has each large variable twice (once from the sky JS,
    once from the 3D JS). We replace every occurrence.
    Returns list of (start, end, json_body) sorted by start position.
    """
    decoder = json.JSONDecoder()
    results = []
    search_from = 0
    while True:
        pos = html.find(f"var {varname}", search_from)
        if pos == -1:
            break
        try:
            eq_pos    = html.index("=", pos)
            brace_pos = html.index("{", eq_pos)
            _, end_offset = decoder.raw_decode(html, brace_pos)
            json_body = html[brace_pos:end_offset]
            semi_pos  = html.index(";", end_offset)
            results.append((pos, semi_pos + 1, json_body))
            search_from = semi_pos + 1
        except (ValueError, json.JSONDecodeError):
            search_from = pos + len(varname)
    if not results:
        raise ValueError(f"Variable '{varname}' not found in HTML")
    return results


def _lazy_proxy_stub(varname: str) -> str:
    """JS stub: Proxy that fetches the JSON file on first property access."""
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    if not INPUT.exists():
        print(f"ERROR: {INPUT} not found.")
        print("       Run  python src/compile_umap_linked.py  first.")
        return 1

    print(f"Reading {INPUT.name} ...")
    t0   = time.time()
    html = INPUT.read_text(encoding="utf-8")
    print(f"  {len(html)/1e6:.1f} MB read in {time.time()-t0:.1f}s")

    data_dir = OUTDIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    replacements: list[tuple[int, int, str]] = []

    # ── 1. Externalize sky image ──────────────────────────────────────────────
    print("\n[1] Extracting Gaia sky image ...")
    sky_pat = re.compile(r"skyImg\.src\s*=\s*'data:image/jpeg;base64,([A-Za-z0-9+/=]+)';")
    sky_m   = sky_pat.search(html)
    if sky_m:
        jpg_path = data_dir / "sky_image.jpg"
        jpg_path.write_bytes(base64.b64decode(sky_m.group(1)))
        print(f"  sky_image.jpg: {jpg_path.stat().st_size/1e6:.1f} MB")
        replacements.append((sky_m.start(), sky_m.end(), "skyImg.src = 'data/sky_image.jpg';"))
    else:
        print("  WARNING: skyImg.src not found — sky image stays inline")

    # ── 2. Externalize image variables (replace ALL occurrences) ─────────────
    print("\n[2] Extracting image variables ...")
    for varname in ALL_VARS:
        try:
            spans = _find_all_var_spans(html, varname)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  WARNING: {varname}: {e} — skipped")
            continue

        # Write the JSON from the first (largest/canonical) occurrence
        _, _, json_body = spans[0]
        out_path = data_dir / f"{varname}.json"
        out_path.write_text(json_body, encoding="utf-8")
        n = len(spans)
        print(f"  {varname}: {len(json_body)/1e6:.1f} MB → {out_path.name}"
              + (f"  ({n} occurrences replaced)" if n > 1 else ""))

        # Replace every occurrence: first → fetch stub, extras → empty {}
        if varname in EAGER_VARS:
            stub = f"var {varname} = {{}};"
        else:
            stub = _lazy_proxy_stub(varname)

        for i, (start, end, _) in enumerate(spans):
            replacements.append((start, end, stub if i == 0 else f"var {varname} = {{}};"))

    # ── 3. Replace cluster-sheet transposition block with eager fetch ─────────
    # Find the LAST occurrence (in the 3D JS, not the sky JS copy)
    print("\n[3] Injecting eager-fetch loader ...")
    ts = html.rfind("var cluster_sheets = {};")
    te = html.rfind("/* Q1 feature names")
    if ts == -1 or te == -1:
        print("  WARNING: transposition block not found — skipping")
    else:
        eager_block = """\
var cluster_sheets = {}, desi_cluster_sheets = {}, ero_desi_cluster_sheets = {};
var _loadingExtras = new Set();

var _loadedCount = 0, _totalEager = 6;
function _updateLoadHint(name) {
  _loadedCount++;
  var el = document.getElementById('sidebar-hint');
  if (!el) return;
  if (_loadedCount < _totalEager) {
    el.innerHTML = 'Loading image data\u2026 <span style="color:#666">' + _loadedCount + '/' + _totalEager + '</span>';
  } else {
    el.textContent = 'hover a point \u00b7 double-click to pin';
  }
}
function _jfetch(name) {
  return fetch('data/' + name + '.json')
    .then(function(r){return r.json();})
    .then(function(d){ _updateLoadHint(name); return d; });
}
setTimeout(function() {
  var el = document.getElementById('sidebar-hint');
  if (el && _loadedCount < _totalEager)
    el.innerHTML = 'Loading image data\u2026 <span style="color:#666">0/' + _totalEager + '</span>';
}, 0);

var _dataReady = Promise.all([
  _jfetch('contact_sheets'),
  _jfetch('desi_contact_sheets'),
  _jfetch('ero_contact_sheets'),
  _jfetch('ero_desi_contact_sheets'),
  _jfetch('cutouts'),
  _jfetch('ero_cutouts'),
]).then(function(arr) {
  contact_sheets          = arr[0];
  desi_contact_sheets     = arr[1];
  ero_contact_sheets      = arr[2];
  ero_desi_contact_sheets = arr[3];
  cutouts                 = arr[4];
  ero_cutouts             = arr[5];
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

    # ── 4. Apply replacements (reverse order preserves offsets) ──────────────
    print("\n[4] Applying replacements ...")
    replacements.sort(key=lambda r: r[0], reverse=True)
    for start, end, new_text in replacements:
        html = html[:start] + new_text + html[end:]

    # ── 5. Write index.html ───────────────────────────────────────────────────
    out_html = OUTDIR / "index.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"\n[5] Written to {OUTDIR}/")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print("\n  File sizes:")
    total = 0
    for f in sorted(OUTDIR.rglob("*")):
        if f.is_file():
            sz = f.stat().st_size
            total += sz
            print(f"    {f.relative_to(OUTDIR)}: {sz/1e6:.1f} MB")
    print(f"    {'─'*35}")
    print(f"    total: {total/1e6:.1f} MB")
    print(f"\n  To serve locally:")
    print(f"    cd {OUTDIR} && python -m http.server 8000")
    print(f"    open http://localhost:8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
