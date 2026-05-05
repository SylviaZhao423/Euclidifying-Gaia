#!/usr/bin/env python3
"""
Analyze coverage of original Gaia DR3 columns in the crossmatch tables.

Queries the actual crossmatch table (not gaia_dr3.gaia_source) and checks
every original g_* column (excluding derived features like corrected_bp_rp_excess,
pm_significance, position_error) for all forms of missing data:
  - SQL NULL  (psycopg2 returns None)
  - Text sentinels: 'NOT_AVAILABLE', 'null', 'nan', 'N/A', '', etc.
  - Non-finite floats: NaN, Inf, -Inf (after numeric conversion)

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_feature_coverage.py
    PGUSER=siwei_zhao_2025 python src/analyze_feature_coverage.py --table ero
    PGUSER=siwei_zhao_2025 python src/analyze_feature_coverage.py --sample-size 50000 --all
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "output"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB = "wsdb"

# Derived columns added to the crossmatch table (not original Gaia DR3).
# These are pre-computed via add_derived_columns.py and should be excluded.
DERIVED_COLUMNS = {
    "g_corrected_bp_rp_excess",
    "g_pm_significance",
    "g_position_error",
    "g_astrometric_n_bad_good_obs_al",
    "g_astrometric_matched_transits_ratio",
}

# Text values that should be treated as missing
MISSING_SENTINELS = {
    "NOT_AVAILABLE", "not_available",
    "null", "NULL", "Null",
    "nan", "NaN", "NAN",
    "N/A", "n/a", "NA", "na",
    "None", "none",
    "", " ",
}

# Tables to check
TABLES = {
    "q1": {
        "schema": "siwei_zhao_2025",
        "table": "crossmatch_q1_mer_phz_vis_gaia_sep",
        "label": "Q1 crossmatch (PHZ star/galaxy, G>=17)",
        "gaia_prefix": "g_",
        "filter": "p_phz_classification IN (1, 2) AND g_phot_g_mean_mag >= 17",
    },
    "ero": {
        "schema": "siwei_zhao_2025",
        "table": "crossmatch_q1_mer_phz_vis_gaia",
        "label": "ERO crossmatch (G>=17)",
        "gaia_prefix": "g_",
        "filter": "g_phot_g_mean_mag >= 17",
    },
}


def discover_gaia_columns(cursor, schema: str, table: str, prefix: str) -> list[dict]:
    """Discover all Gaia columns (g_*) in the crossmatch table via information_schema."""
    query = f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = '{schema}'
          AND table_name = '{table}'
          AND column_name LIKE '{prefix}%'
        ORDER BY ordinal_position
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    # Exclude derived columns — keep only original Gaia DR3 features
    return [{"name": r[0], "type": r[1]} for r in rows
            if r[0] not in DERIVED_COLUMNS]


def query_all_gaia_columns(username: str, table_key: str, sample_size: int) -> tuple[pd.DataFrame, list[dict]]:
    """Query ALL g_* columns from crossmatch table, returning raw values + column info."""
    import psycopg2

    tbl = TABLES[table_key]
    schema = tbl["schema"]
    table = tbl["table"]
    prefix = tbl["gaia_prefix"]
    filt = tbl["filter"]

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cursor = conn.cursor()

    # Step 1: discover all g_* columns
    print(f"  Discovering columns in {schema}.{table} ...")
    col_info = discover_gaia_columns(cursor, schema, table, prefix)
    print(f"  Found {len(col_info)} original Gaia columns (excluding derived)")

    # Step 2: query them all
    col_names = [c["name"] for c in col_info]
    col_list = ", ".join(col_names)

    query = f"""
        SELECT {col_list}
        FROM {schema}.{table}
        WHERE {filt}
        ORDER BY random()
        LIMIT {sample_size}
    """

    print(f"  Querying {sample_size:,} rows (filter: {filt}) ...")
    df = pd.read_sql_query(query, conn)
    cursor.close()
    conn.close()
    print(f"  Retrieved {len(df):,} rows x {len(df.columns)} columns")
    return df, col_info


def analyze_coverage(df: pd.DataFrame, col_info: list[dict], prefix: str) -> pd.DataFrame:
    """Analyze missing-value coverage for every column.

    Checks for:
      1. Python None / NaN (from SQL NULL)
      2. Text sentinels ('NOT_AVAILABLE', 'null', etc.)
      3. Non-finite values after numeric conversion (Inf, -Inf)
    """
    n = len(df)
    results = []

    for ci in col_info:
        col_name = ci["name"]
        col_type = ci["type"]
        # Display name: strip g_ prefix
        display = col_name[len(prefix):] if col_name.startswith(prefix) else col_name

        if col_name not in df.columns:
            results.append({
                "column": display, "db_column": col_name, "db_type": col_type,
                "total": n, "sql_null": n, "text_sentinel": 0, "non_finite": 0,
                "total_missing": n, "valid": 0, "pct_valid": 0.0,
                "sentinel_values": "COLUMN NOT FOUND",
            })
            continue

        raw = df[col_name]

        # 1. SQL NULL -> Python None/NaN
        sql_null = int(raw.isna().sum())

        # 2. Text sentinels (check non-null values as strings)
        non_null = raw.dropna()
        str_vals = non_null.astype(str).str.strip()
        sentinel_mask = str_vals.isin(MISSING_SENTINELS)
        text_sentinel = int(sentinel_mask.sum())
        found_sentinels = sorted(set(str_vals[sentinel_mask].unique()))

        # 3. Non-finite after numeric conversion
        cleaned = raw.copy()
        cleaned = cleaned.replace(list(MISSING_SENTINELS), np.nan)
        numeric = pd.to_numeric(cleaned, errors="coerce")
        non_finite = int(np.isinf(numeric).sum()) if numeric.dtype != object else 0

        total_missing = sql_null + text_sentinel + non_finite
        valid = n - total_missing
        pct_valid = round((valid / n) * 100, 2) if n > 0 else 0.0

        results.append({
            "column": display,
            "db_column": col_name,
            "db_type": col_type,
            "total": n,
            "sql_null": sql_null,
            "text_sentinel": text_sentinel,
            "non_finite": non_finite,
            "total_missing": total_missing,
            "valid": valid,
            "pct_valid": pct_valid,
            "sentinel_values": ", ".join(found_sentinels) if found_sentinels else "",
        })

    return pd.DataFrame(results)


def print_report(stats: pd.DataFrame, label: str):
    """Print formatted coverage report."""
    n = stats["total"].iloc[0] if len(stats) > 0 else 0
    total_cols = len(stats)

    print(f"\n{'=' * 120}")
    print(f"Feature Coverage: {label}  (N = {n:,} sources, {total_cols} Gaia columns)")
    print(f"{'=' * 120}")
    print(f"  {'Column':<40} {'Type':<12} {'Valid':>8} {'NULL':>8} {'Sentinel':>9} {'Inf':>5} {'% Valid':>8}  {'Sentinels'}")
    print(f"  {'-' * 116}")

    for _, row in stats.iterrows():
        pct = row["pct_valid"]
        if pct >= 99.9:
            flag = " "
        elif pct >= 95:
            flag = "~"
        elif pct >= 50:
            flag = "!"
        else:
            flag = "X"

        sentinel_info = f"  [{row['sentinel_values']}]" if row["sentinel_values"] else ""

        print(f"{flag} {row['column']:<40} {row['db_type']:<12} {row['valid']:>8,} {row['sql_null']:>8,} "
              f"{row['text_sentinel']:>9,} {row['non_finite']:>5,} {pct:>7.2f}%{sentinel_info}")

    print(f"{'=' * 120}")

    # Summary
    full = (stats["pct_valid"] >= 99.9).sum()
    high = ((stats["pct_valid"] >= 95) & (stats["pct_valid"] < 99.9)).sum()
    moderate = ((stats["pct_valid"] >= 50) & (stats["pct_valid"] < 95)).sum()
    sparse = (stats["pct_valid"] < 50).sum()
    has_sentinels = (stats["text_sentinel"] > 0).sum()

    print(f"\n  Summary ({total_cols} columns):")
    print(f"    Full coverage (>=99.9%):  {full}")
    print(f"    High coverage (95-99.9%): {high}")
    print(f"    Moderate (50-95%):        {moderate}")
    print(f"    Sparse (<50%):            {sparse}")
    print(f"    Columns with text sentinels: {has_sentinels}")
    if has_sentinels > 0:
        sentinel_cols = stats[stats["text_sentinel"] > 0]
        for _, row in sentinel_cols.iterrows():
            print(f"      {row['column']}: {row['text_sentinel']:,} rows  [{row['sentinel_values']}]")

    # Flag columns that have ONLY sentinel-based missing (no SQL NULL)
    # — these would be missed by the old coverage script
    sentinel_only = stats[(stats["text_sentinel"] > 0) & (stats["sql_null"] == 0)]
    if len(sentinel_only) > 0:
        print(f"\n  WARNING: {len(sentinel_only)} columns have missing values ONLY as text sentinels")
        print(f"           (would appear 100% populated in NULL-only checks!):")
        for _, row in sentinel_only.iterrows():
            print(f"      {row['column']}: {row['text_sentinel']:,} sentinel rows  [{row['sentinel_values']}]")


def plot_coverage(stats: pd.DataFrame, label: str, output_path: Path):
    """Horizontal bar chart of per-column coverage, color-coded by missing type."""
    n_cols = len(stats)
    fig_height = max(6, n_cols * 0.22)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    n = stats["total"].iloc[0]
    y = np.arange(n_cols)

    valid_pct = stats["valid"].values / n * 100
    null_pct = stats["sql_null"].values / n * 100
    sentinel_pct = stats["text_sentinel"].values / n * 100
    inf_pct = stats["non_finite"].values / n * 100

    ax.barh(y, valid_pct, color="#4CAF50", label="Valid", height=0.7)
    ax.barh(y, null_pct, left=valid_pct, color="#F44336", label="SQL NULL", height=0.7)
    ax.barh(y, sentinel_pct, left=valid_pct + null_pct, color="#FF9800",
            label="Text sentinel", height=0.7)
    ax.barh(y, inf_pct, left=valid_pct + null_pct + sentinel_pct, color="#9C27B0",
            label="Non-finite", height=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(stats["column"].values, fontsize=7)
    ax.set_xlabel("Percentage", fontsize=11, fontweight="bold")
    ax.set_title(f"Gaia Column Coverage in Crossmatch Table\n{label}  (N = {n:,})",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze coverage of ALL Gaia columns in crossmatch tables"
    )
    parser.add_argument("--table", choices=list(TABLES.keys()), default="q1",
                        help="Which crossmatch table to check (default: q1)")
    parser.add_argument("--sample-size", type=int, default=20000,
                        help="Number of sources to sample (default: 20000)")
    parser.add_argument("--all", action="store_true",
                        help="Check all tables")
    args = parser.parse_args()

    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER environment variable not set!")
        print("Set with: export PGUSER=your_username")
        return 1

    tables_to_check = list(TABLES.keys()) if args.all else [args.table]

    print("=" * 70)
    print("Gaia Column Coverage Analysis (Crossmatch Tables)")
    print("=" * 70)
    print(f"  User: {username}")
    print(f"  Sample size: {args.sample_size:,}")
    print(f"  Missing sentinels: {sorted(s for s in MISSING_SENTINELS if s.strip())}")

    for tkey in tables_to_check:
        tbl = TABLES[tkey]
        print(f"\n{'─' * 70}")
        print(f"Table: {tbl['label']}")
        print(f"  {tbl['schema']}.{tbl['table']}")
        print(f"{'─' * 70}")

        df, col_info = query_all_gaia_columns(username, tkey, args.sample_size)
        stats = analyze_coverage(df, col_info, tbl["gaia_prefix"])
        print_report(stats, tbl["label"])

        # Save CSV
        csv_path = OUTPUT_DIR / f"feature_coverage_{tkey}.csv"
        stats.to_csv(csv_path, index=False)
        print(f"  CSV saved: {csv_path}")

        # Save plot
        plot_path = OUTPUT_DIR / f"feature_coverage_{tkey}.png"
        plot_coverage(stats, tbl["label"], plot_path)

    print(f"\n{'=' * 70}")
    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
