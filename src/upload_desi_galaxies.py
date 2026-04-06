#!/usr/bin/env python3
"""
Upload DESI-confirmed galaxy catalogues (ERO + Q1) to WSDB.

Reads two FITS binary table files and uploads them to WSDB using
PostgreSQL COPY (same fast-bulk approach as upload_q1_phz_all.py).

Input files:
  data/with_desi/desi_euclid_ero_galaxies.fits   (1,844 rows, 75 columns)
  data/with_desi/desi_euclid_q1_galaxies.fits   (44,667 rows, 73 columns)

Creates tables:
  {username}.desi_ero_galaxies
  {username}.desi_q1_galaxies

Usage:
  PGUSER=siwei_zhao_2025 python src/upload_desi_galaxies.py
"""

import io
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR    = PROJECT_DIR / "data" / "with_desi"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"

FITS_FILES = {
    "desi_ero_galaxies": DATA_DIR / "desi_euclid_ero_galaxies.fits",
    "desi_q1_galaxies":  DATA_DIR / "desi_euclid_q1_galaxies.fits",
}


def fits_to_dataframe(fits_path: Path) -> pd.DataFrame:
    """Read a FITS binary table into a pandas DataFrame."""
    from astropy.io import fits as afits
    with afits.open(fits_path, memmap=False) as hdul:
        # Find the first binary table HDU
        for hdu in hdul:
            if hdu.data is not None and hasattr(hdu, "columns"):
                table = hdu
                break
        else:
            raise RuntimeError(f"No binary table HDU found in {fits_path.name}")

        df = table.data.copy()

    # Convert astropy table to pandas, handling byte strings and multidim columns
    records = {}
    for col in df.dtype.names:
        arr = df[col]
        # Byte strings → regular strings
        if arr.dtype.kind in ("S", "U") or arr.dtype.kind == "O":
            arr = np.array([v.decode("utf-8").strip() if isinstance(v, bytes) else str(v)
                            for v in arr])
        # Skip multidimensional columns (rare in these catalogues)
        if arr.ndim > 1:
            print(f"  Skipping multidim column: {col} {arr.shape}")
            continue
        records[col] = arr

    return pd.DataFrame(records)


def pandas_dtype_to_pg(dtype) -> str:
    """Map a pandas/numpy dtype to a PostgreSQL type."""
    kind = np.dtype(dtype).kind
    itemsize = np.dtype(dtype).itemsize
    if kind == "i":
        return "BIGINT" if itemsize >= 8 else "INTEGER"
    if kind == "u":
        return "BIGINT"
    if kind == "f":
        return "REAL" if itemsize <= 4 else "DOUBLE PRECISION"
    if kind == "b":
        return "BOOLEAN"
    return "TEXT"


def upload_dataframe(df: pd.DataFrame, table_name: str, conn) -> None:
    """Upload a DataFrame to wsdb via PostgreSQL COPY."""
    cursor = conn.cursor()

    # Build CREATE TABLE
    col_defs = [f'"{col}" {pandas_dtype_to_pg(df[col].dtype)}' for col in df.columns]
    cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
    cursor.execute(f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(col_defs) + "\n);")
    conn.commit()
    print(f"  Created table: {table_name}  ({len(df.columns)} columns)")

    # COPY from in-memory CSV
    buf = io.StringIO()
    df.to_csv(buf, index=False, sep="\t", na_rep="\\N")
    buf.seek(0)
    col_list = ", ".join([f'"{c}"' for c in df.columns])
    cursor.copy_expert(
        f"COPY {table_name} ({col_list}) FROM STDIN "
        f"WITH (FORMAT csv, DELIMITER E'\\t', NULL '\\N', HEADER)",
        buf,
    )
    conn.commit()

    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    n = cursor.fetchone()[0]
    print(f"  Uploaded {n:,} rows  →  {table_name}")
    cursor.close()


def main():
    username = os.environ.get("PGUSER")
    if not username:
        print("ERROR: set PGUSER"); sys.exit(1)

    import psycopg2
    print("=" * 65)
    print("Uploading DESI galaxy catalogues to WSDB")
    print(f"  Host : {WSDB_HOST}  DB : {WSDB_DB}  User : {username}")
    print("=" * 65)

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)

    for table_suffix, fits_path in FITS_FILES.items():
        table_name = f"{username}.{table_suffix}"
        print(f"\n[{table_suffix}]")
        print(f"  File : {fits_path.name}  ({fits_path.stat().st_size / 1e6:.1f} MB)")

        print("  Reading FITS binary table...")
        df = fits_to_dataframe(fits_path)
        print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")

        upload_dataframe(df, table_name, conn)

    conn.close()

    print("\n" + "=" * 65)
    print("Done. Tables created:")
    for table_suffix in FITS_FILES:
        print(f"  {username}.{table_suffix}")
    print("\nNext step: create crossmatch tables joining these with")
    print("  crossmatch_ero_gaia_0_3  (via position, q3c)")
    print("  crossmatch_q1_mer_phz_gaia  (via euclid_gaia_id = g_source_id)")
    print("=" * 65)


if __name__ == "__main__":
    main()
