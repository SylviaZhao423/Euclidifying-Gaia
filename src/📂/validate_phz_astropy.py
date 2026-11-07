ƒ#!/usr/bin/env python3
"""
Validate that the astroquery.esa.euclid TAP pipeline reproduces the phz tables we originally
built from manually-downloaded VOTables (upload_q1_phz_all.py / upload_q1_phz_withgaia.py).
ƒ
Confirmed source queries (row counts match the originals exactly):
  q1_phz_all      <- SELECT * FROM catalogue.phz_classification              (29,643,313 / 15 cols)
  q1_phz_withgaia <- catalogue.phz_classification p
                     JOIN catalogue.mer_catalogue m ON m.object_id=p.object_id
                     WHERE m.gaia_id IS NOT NULL   (p.* + m.gaia_id)         (348,059 / 16 cols)

What this does:
  --withgaia : recreate q1_phz_withgaia_aspy fully via TAP (keyset-paginated join + COPY),
               then compare row-by-row (join on object_id) to the original q1_phz_withgaia.
  --sample-all N : pull N random q1_phz_all object_ids from catalogue.phz_classification via
               TAP and compare values to the WSDB q1_phz_all (avoids re-pulling all 29.6M).
Default: both. Floats compared with np.isclose; ints/object_id exact.

Usage: PGUSER=siwei_zhao_2025 python src/📂/validate_phz_tap.py
"""
import argparse
import io
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
WSDB = dict(host="wsdb.ast.cam.ac.uk", database="wsdb")

WG_COLS = ["basic_download_data_oid", "missing_band", "qso_threshold", "to_be_published",
           "vis_det", "object_id", "configuration", "phz_star_prob", "phz_gal_prob",
           "phz_qso_prob", "phz_glob_cl_prob", "phz_classification", "star_threshold",
           "galaxy_threshold", "globc_threshold", "gaia_id"]
ALL_COLS = [c for c in WG_COLS if c != "gaia_id"]   # q1_phz_all = catalogue.phz_classification


def _connect(user):
    import psycopg2
    return psycopg2.connect(user=user, connect_timeout=30, keepalives=1, keepalives_idle=30,
                            keepalives_interval=10, keepalives_count=5, **WSDB)


def _compare(a: pd.DataFrame, b: pd.DataFrame, key="object_id", label=""):
    """Row-by-row compare two frames with the same columns, joined on key."""
    cols = [c for c in a.columns if c in b.columns]
    a = a[cols].sort_values(key).reset_index(drop=True)
    b = b[cols].sort_values(key).reset_index(drop=True)
    print(f"  [{label}] N_orig={len(a):,}  N_aspy={len(b):,}")
    if len(a) != len(b):
        print(f"    ROW COUNT DIFFERS")
    m = a.merge(b, on=key, suffixes=("_o", "_a"), how="inner")
    print(f"    matched on {key}: {len(m):,}")
    ok = True
    for c in cols:
        if c == key:
            continue
        o, n = m[f"{c}_o"], m[f"{c}_a"]
        if pd.api.types.is_float_dtype(o) or pd.api.types.is_float_dtype(n):
            same = np.isclose(o.astype(float), n.astype(float), rtol=1e-6, atol=1e-9, equal_nan=True)
        else:
            same = (o.fillna(-999999) == n.fillna(-999999))
        nbad = int((~same).sum())
        if nbad:
            ok = False
            print(f"    MISMATCH {c}: {nbad:,} rows differ  (e.g. {o[~same].iloc[0]} vs {n[~same].iloc[0]})")
    print(f"    -> {'IDENTICAL' if ok and len(a)==len(b) else 'DIFFERENCES FOUND'}")
    return ok


def recreate_withgaia(user):
    from astroquery.esa.euclid import Euclid
    conn = _connect(user); cur = conn.cursor()
    full = f"{user}.q1_phz_withgaia_aspy"
    cur.execute(f"DROP TABLE IF EXISTS {full} CASCADE;")
    cur.execute(f"CREATE TABLE {full} (LIKE {user}.q1_phz_withgaia INCLUDING DEFAULTS);")
    conn.commit()
    sel = ", ".join(("m.gaia_id" if c == "gaia_id" else f"p.{c}") for c in WG_COLS)
    last = None; done = 0; t0 = time.time()
    while True:
        where = f"AND p.object_id > {int(last)} " if last is not None else ""
        q = (f"SELECT TOP 200000 {sel} FROM catalogue.phz_classification p "
             f"JOIN catalogue.mer_catalogue m ON m.object_id=p.object_id "
             f"WHERE m.gaia_id IS NOT NULL {where}ORDER BY p.object_id")
        r = Euclid.launch_job_async(q).get_results().to_pandas()
        if len(r) == 0:
            break
        r = r[WG_COLS].replace([np.inf, -np.inf], np.nan)
        buf = io.StringIO(); r.to_csv(buf, index=False, header=False, sep="\t", na_rep="\\N"); buf.seek(0)
        cur.copy_expert(f"COPY {full} ({','.join(WG_COLS)}) FROM STDIN "
                        f"WITH (FORMAT csv, DELIMITER E'\t', NULL '\\N')", buf)
        conn.commit(); done += len(r); last = int(r["object_id"].max())
        print(f"  +{len(r):,} -> {done:,}  ({time.time()-t0:.0f}s)", flush=True)
        if len(r) < 200000:
            break
    cur.close(); conn.close()
    print(f"  built {full}: {done:,} rows")


def recreate_all(user):
    """Full q1_phz_all_aspy from catalogue.phz_classification via TAP (keyset, RESUMABLE)."""
    from astroquery.esa.euclid import Euclid
    conn = _connect(user); cur = conn.cursor()
    full = f"{user}.q1_phz_all_aspy"
    cur.execute(f"CREATE TABLE IF NOT EXISTS {full} (LIKE {user}.q1_phz_all INCLUDING DEFAULTS);")
    conn.commit()
    cur.execute(f"SELECT count(*), max(object_id) FROM {full}")
    done, last = cur.fetchone()
    print(f"  resume: {done:,} rows already loaded" + (f", last={last}" if last is not None else ""))
    sel = ", ".join(ALL_COLS); t0 = time.time()
    while True:
        where = f"WHERE object_id > {int(last)} " if last is not None else ""
        q = f"SELECT TOP 200000 {sel} FROM catalogue.phz_classification {where}ORDER BY object_id"
        r = Euclid.launch_job_async(q).get_results().to_pandas()
        if len(r) == 0:
            break
        r = r[ALL_COLS].replace([np.inf, -np.inf], np.nan)
        buf = io.StringIO(); r.to_csv(buf, index=False, header=False, sep="\t", na_rep="\\N"); buf.seek(0)
        cur.copy_expert(f"COPY {full} ({sel}) FROM STDIN WITH (FORMAT csv, DELIMITER E'\t', NULL '\\N')", buf)
        conn.commit(); done += len(r); last = int(r["object_id"].max())
        print(f"  +{len(r):,} -> {done:,}  ({done/max(1e-9,time.time()-t0):,.0f} r/s)", flush=True)
        if len(r) < 200000:
            break
    cur.close(); conn.close()
    print(f"  built {full}: {done:,} rows")


def compare_all_full(user):
    """Full value-to-value (in-DB, via EXCEPT with missing_band '' <-> NULL normalised) +
    row-count check of q1_phz_all vs q1_phz_all_aspy. If _aspy is still loading, compares only
    the object_id range loaded so far (keyset load is ascending, so that range is complete)."""
    conn = _connect(user); cur = conn.cursor()
    o, a = f"{user}.q1_phz_all", f"{user}.q1_phz_all_aspy"
    cur.execute(f"SELECT count(*) FROM {o}"); no = cur.fetchone()[0]
    cur.execute(f"SELECT count(*) FROM {a}"); na = cur.fetchone()[0]
    cur.execute(f"SELECT max(object_id) FROM {a}"); hi = cur.fetchone()[0]
    partial = na < no
    rng = f"WHERE object_id <= {int(hi)}" if partial else ""
    cur.execute(f"SELECT count(*) FROM {o} {rng}"); no_r = cur.fetchone()[0]
    tag = f"(PARTIAL — object_id<={hi})" if partial else "(FULL)"
    print(f"  rows {tag}: q1_phz_all[in range]={no_r:,}  q1_phz_all_aspy={na:,}  "
          f"-> {'MATCH' if no_r == na else 'MISMATCH'}")
    if not partial:
        print(f"  full-table rows: q1_phz_all={no:,}  q1_phz_all_aspy={na:,}  "
              f"-> {'MATCH' if no == na else 'MISMATCH'}")
    sel = ", ".join((f"NULLIF({c},'') AS {c}" if c == "missing_band" else c) for c in ALL_COLS)
    ox, ax = f"(SELECT {sel} FROM {o} {rng})", f"(SELECT {sel} FROM {a} {rng})"
    cur.execute(f"SELECT count(*) FROM ({ox} EXCEPT {ax}) t"); d1 = cur.fetchone()[0]
    cur.execute(f"SELECT count(*) FROM ({ax} EXCEPT {ox}) t"); d2 = cur.fetchone()[0]
    print(f"  value-to-value (EXCEPT, missing_band ''<->NULL normalised):")
    print(f"    rows in q1_phz_all not in _aspy: {d1:,}")
    print(f"    rows in _aspy not in q1_phz_all: {d2:,}")
    print(f"    -> {'IDENTICAL' if d1 == 0 and d2 == 0 and no_r == na else 'DIFFERENCES FOUND'}"
          + ("  [partial so far]" if partial else ""))
    conn.close()


def compare_withgaia(user):
    conn = _connect(user)
    a = pd.read_sql_query(f"SELECT {','.join(WG_COLS)} FROM {user}.q1_phz_withgaia", conn)
    b = pd.read_sql_query(f"SELECT {','.join(WG_COLS)} FROM {user}.q1_phz_withgaia_aspy", conn)
    conn.close()
    _compare(a, b, label="q1_phz_withgaia vs _aspy")


def sample_check_all(user, n):
    from astroquery.esa.euclid import Euclid
    conn = _connect(user)
    a = pd.read_sql_query(f"SELECT * FROM {user}.q1_phz_all ORDER BY random() LIMIT {n}", conn)
    conn.close()
    ids = a["object_id"].astype("int64").tolist()
    frames = []
    for i in range(0, len(ids), 500):
        inlist = ",".join(map(str, ids[i:i + 500]))
        frames.append(Euclid.launch_job(
            f"SELECT * FROM catalogue.phz_classification WHERE object_id IN ({inlist})"
        ).get_results().to_pandas())
    b = pd.concat(frames, ignore_index=True)
    _compare(a, b, label=f"q1_phz_all random {n:,} vs phz_classification")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--withgaia", action="store_true", help="recreate + compare q1_phz_withgaia only")
    ap.add_argument("--sample-all", type=int, default=0, help="sample-check N rows of q1_phz_all only")
    ap.add_argument("--build-all", action="store_true",
                    help="build the FULL q1_phz_all_aspy (29.6M) via TAP, then compare row counts")
    ap.add_argument("--compare-all", action="store_true",
                    help="value-to-value + row-count compare q1_phz_all vs _aspy (partial-aware)")
    args = ap.parse_args()
    user = os.getenv("PGUSER")
    if not user:
        print("ERROR: set PGUSER"); return 1

    if args.compare_all:
        print("== q1_phz_all : value-to-value + row-count compare ==")
        compare_all_full(user); print("Done."); return 0

    if args.build_all:
        print("== q1_phz_all : build FULL _aspy via TAP, then full value+row compare ==")
        recreate_all(user)
        compare_all_full(user)
        print("Done."); return 0

    both = not (args.withgaia or args.sample_all)
    if args.withgaia or both:
        print("== q1_phz_withgaia : recreate via TAP + compare ==")
        recreate_withgaia(user); compare_withgaia(user)
    if args.sample_all or both:
        n = args.sample_all or 20000
        print(f"\n== q1_phz_all : sample-check {n:,} rows ==")
        sample_check_all(user, n)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
