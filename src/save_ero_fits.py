#!/usr/bin/env python3
"""
Download Euclid ERO VIS Flattened FITS files for all 17 ERO fields.

Each file is extracted directly from the public CDN tar archive using HTTP
range requests (no need to download the full ~2.8 GB tar), decompressed
once, and saved to the local cache directory.

CDN pattern:
  https://cdn.euclid.esac.esa.int/Stack/Euclid-VIS-Stack-ERO-{TARGET}.DR3.tar

Cache directory:
  output/euclid_ero_fits_cache/

Already-downloaded fields are skipped automatically.

Usage:
  python src/save_ero_fits.py
  python src/save_ero_fits.py --fields "NGC 6397" "Perseus"
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FITS_CACHE   = PROJECT_ROOT / "output" / "euclid_ero_fits_cache"

CDN_BASE = "https://cdn.euclid.esac.esa.int/Stack"

FIELD_TO_TARGET = {
    "Horsehead":   "ERO-Horsehead",
    "Messier 78":  "ERO-Messier78",
    "Taurus":      "ERO-Taurus",
    "Barnard 30":  "ERO-Barnard30",
    "NGC 6254":    "ERO-NGC6254",
    "NGC 6397":    "ERO-NGC6397",
    "NGC 6822":    "ERO-NGC6822",
    "Holmberg II": "ERO-HolmbergII",
    "IC 10":       "ERO-IC10",
    "IC 342":      "ERO-IC342",
    "NGC 2403":    "ERO-NGC2403",
    "NGC 6744":    "ERO-NGC6744",
    "Fornax":      "ERO-Fornax",
    "Dorado":      "ERO-Dorado",
    "Perseus":     "ERO-Perseus",
    "Abell 2390":  "ERO-Abell2390",
    "Abell 2764":  "ERO-Abell2764",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Euclid ERO VIS FITS files.")
    p.add_argument("--fields", nargs="+", metavar="FIELD",
                   help="Download only these named fields (default: all 17).")
    return p.parse_args()


def http_range(url: str, start: int, end: int) -> bytes:
    headers = {"Range": f"bytes={start}-{end}", "User-Agent": "euclid-fits-downloader"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def stream_download(url: str, dest: Path, start: int, size: int,
                    chunk: int = 4 << 20) -> None:
    headers = {"Range": f"bytes={start}-{start + size - 1}",
               "User-Agent": "euclid-fits-downloader"}
    req = urllib.request.Request(url, headers=headers)
    downloaded = 0
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        while True:
            block = r.read(chunk)
            if not block:
                break
            f.write(block)
            downloaded += len(block)
            print(f"    {downloaded/1e6:,.0f} / {size/1e6:,.0f} MB  "
                  f"({100*downloaded/size:.1f}%)", end="\r", flush=True)
    print()
    if downloaded != size:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Incomplete download: got {downloaded:,} / {size:,} bytes. "
            "Deleted partial file — re-run to retry."
        )


def read_tar_first_entry(tar_url: str) -> tuple[str, int, int]:
    header_bytes = http_range(tar_url, 0, 2047)
    offset = 0
    while offset + 512 <= len(header_bytes):
        block = header_bytes[offset:offset + 512]
        name     = block[0:100].rstrip(b"\x00").decode("utf-8", errors="replace")
        size_oct = block[124:136].rstrip(b"\x00 ").decode("ascii", errors="replace")
        size     = int(size_oct, 8) if size_oct else 0
        typeflag = block[156:157]
        if typeflag not in (b"5",) and size > 0:
            return name, offset + 512, size
        padded = ((size + 511) // 512) * 512
        offset += 512 + padded
    raise RuntimeError(f"No regular file entry found in tar: {tar_url}")


def download_field(field: str) -> Path | None:
    target    = FIELD_TO_TARGET[field]
    tar_url   = f"{CDN_BASE}/Euclid-VIS-Stack-{target}.DR3.tar"
    gz_path   = FITS_CACHE / f"Euclid-VIS-{target}-Flattened.DR3.fits.gz"
    fits_path = FITS_CACHE / f"Euclid-VIS-{target}-Flattened.DR3.fits"

    if fits_path.exists():
        print(f"  [cached] {fits_path.name}  ({fits_path.stat().st_size/1e9:.1f} GB)")
        return fits_path

    print(f"  Fetching tar header: {tar_url}")
    fname, data_offset, data_size = read_tar_first_entry(tar_url)
    print(f"    -> {fname}  ({data_size/1e9:.2f} GB compressed)")

    gz_complete = gz_path.exists() and gz_path.stat().st_size == data_size
    if not gz_complete:
        if gz_path.exists():
            print(f"  Partial download ({gz_path.stat().st_size/1e6:.0f} / "
                  f"{data_size/1e6:.0f} MB) — re-downloading")
        print(f"  Downloading -> {gz_path.name}")
        stream_download(tar_url, gz_path, data_offset, data_size)

    import gzip, shutil
    print(f"  Decompressing -> {fits_path.name}  (one-time, ~5-10 min) ...")
    with gzip.open(gz_path, "rb") as f_in, open(fits_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=32 << 20)
    print(f"  Decompressed: {fits_path.name}  ({fits_path.stat().st_size/1e9:.1f} GB)")

    gz_path.unlink()
    print(f"  Removed .gz: {gz_path.name}")

    return fits_path


def main() -> int:
    args = parse_args()
    FITS_CACHE.mkdir(parents=True, exist_ok=True)

    fields = args.fields if args.fields else list(FIELD_TO_TARGET.keys())
    unknown = [f for f in fields if f not in FIELD_TO_TARGET]
    if unknown:
        print(f"ERROR: unknown field names: {unknown}")
        print(f"Valid fields: {list(FIELD_TO_TARGET.keys())}")
        return 1

    print("=" * 70)
    print(f"Euclid ERO VIS FITS downloader  ({len(fields)} fields)")
    print(f"Cache: {FITS_CACHE}")
    print("=" * 70)

    ok, skipped, failed = 0, 0, 0
    for field in fields:
        print(f"\n[{field}]")
        try:
            path = download_field(field)
            if path is not None:
                ok += 1
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed += 1

    print(f"\nDone.  downloaded={ok}  cached/skipped={skipped}  failed={failed}")
    print(f"Cache: {FITS_CACHE}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
