#!/usr/bin/env python3
"""Assemble the COMBINED-only QSO×Gaia UMAP "base" bundle from the full n_all build.

query_qso_umap_ds.py builds all three samples (desi / sdss / combined) under
output/qso_umap/ds_qso/n_all/. Downstream only needs the COMBINED base, so this script copies just
the combined reducer + npz into a clean, self-contained folder (output/qso_umap/base/arrays/),
leaving the large per-sample ds_qso tree untouched. No DB, no UMAP fit — it only reuses files.

The CatWISE-space projection (arrays/catwise_gaia_projected.npz + qso_space_catwise_*.png) is
written straight into this base/ by analyze_qso_umap_space_catwise.py, so it is NOT copied here.

Gathers (from output/qso_umap/ds_qso/n{N}/arrays/):
  combined_qso_gaia.npz            -> base/arrays/combined_qso_gaia.npz
  combined_qso_gaia_reducer.joblib -> base/arrays/combined_qso_gaia_reducer.joblib

Usage:
  python src/query_qso_umap_base.py                 # copy into output/qso_umap/base/arrays/
  python src/query_qso_umap_base.py --symlink       # symlink instead of copy (saves ~0.8 GB)
  python src/query_qso_umap_base.py --force         # overwrite existing base/arrays files
  python src/query_qso_umap_base.py --n-sample all  # source build size (default all -> n_all)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
QSO_UMAP = PROJECT_DIR / "output" / "qso_umap"
BASE_DIR = QSO_UMAP / "base"

# combined-only artifacts to pull from the build's arrays/ folder
ARRAY_FILES = ["combined_qso_gaia.npz", "combined_qso_gaia_reducer.joblib"]


def n_suffix(n):
    return "n_all" if str(n).lower() == "all" else f"n{int(n)}"


def _size(p: Path) -> str:
    b = (sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir()
         else p.stat().st_size)
    return f"{b / 1e9:.2f} GB" if b >= 1e9 else f"{b / 1e6:.0f} MB"


def _place(src: Path, dst: Path, symlink: bool) -> None:
    """Copy (or symlink) src -> dst, replacing dst if it already exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.is_dir():
        shutil.rmtree(dst)
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)
    print(f"  {'linked ' if symlink else 'copied '} {dst.relative_to(PROJECT_DIR)}  ({_size(src)})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sample", default="all",
                    help="source build size under ds_qso/ (default all -> n_all).")
    ap.add_argument("--symlink", action="store_true",
                    help="symlink the artifacts instead of copying (saves disk).")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing base/arrays files.")
    args = ap.parse_args()

    arrays = QSO_UMAP / "ds_qso" / n_suffix(args.n_sample) / "arrays"
    missing = [arrays / f for f in ARRAY_FILES if not (arrays / f).exists()]
    if missing:
        print("  missing combined base file(s) — build them first (query_qso_umap_ds.py "
              f"--n-sample {args.n_sample}):")
        for m in missing:
            print(f"    {m}")
        return 1
    dst_arrays = BASE_DIR / "arrays"
    if not args.force and any((dst_arrays / f).exists() for f in ARRAY_FILES):
        print(f"  {dst_arrays.relative_to(PROJECT_DIR)} already has the combined base — "
              f"use --force to overwrite.")
        return 1

    print(f"[base] copying combined-only UMAP base from {arrays.relative_to(PROJECT_DIR)} "
          f"-> {dst_arrays.relative_to(PROJECT_DIR)}")
    for f in ARRAY_FILES:
        _place(arrays / f, dst_arrays / f, args.symlink)
    print(f"\n  base total: {_size(BASE_DIR)}")
    print("  (run analyze_qso_umap_space_catwise.py to add the CatWISE projection into base/)")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
