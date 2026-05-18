#!/usr/bin/env python3
"""
Deploy Q1-only and Q1+ERO XGBoost ensembles onto Gaia DR3 and save the
predictions as a WSDB table owned by the user.

Source pool
-----------
{username}.gaia_source_hilat + gaia_source_lolat — pre-built tables
covering all of Gaia DR3 (~1.65B rows) with the 14 SELECTED_GAIA_PARAMETERS
already as direct columns (no SQL recomputation). Together these are
all of Gaia DR3 (the split is purely a performance partition; querying
gaia_dr3.gaia_source directly with derived-column SQL is ~40× slower).
The two tables are processed sequentially so each batch query uses its
own random_index index cleanly.

Filter (matches training conditions)
------------------------------------
  - phot_g_mean_mag IS NOT NULL
  - phot_g_mean_mag >= 17                 (MAG_CUT — out-of-distribution below)

Models
------
  Q1-only ensemble : output/xgboost_q1/seed_models/*.json                              (10 seeds)
  Q1+ERO max-ratio : output/xgboost_mix_q1_ero/ratio_sg8.63_all_ero/seed_models/*.json (10 seeds)

Each ensemble's `predict_proba(X)[:, 1]` outputs are averaged across
seeds → `*_pgal_mean`; std across seeds → `*_pgal_std` (uncertainty).

NaN feature values are passed through to XGBoost, which handles them
natively (the same way they were handled in training — no imputation).

Output table
------------
{username}.gaia_q1_pgal_predictions
  source_id        BIGINT PRIMARY KEY
  random_index     BIGINT
  ra, dec          DOUBLE PRECISION
  phot_g_mean_mag  REAL
  q1_pgal_mean     REAL    (Q1-only 10-seed ensemble mean)
  q1_pgal_std      REAL    (Q1-only 10-seed ensemble std)
  mix_pgal_mean    REAL    (Q1+ERO 10-seed ensemble mean, max ratio)
  mix_pgal_std     REAL    (Q1+ERO 10-seed ensemble std)

Streaming
---------
Processes Gaia in `random_index` ranges of size --batch-size (default
5,000,000). Resumable via --start-ri / --end-ri.

Usage
-----
    # quick verification (one batch ≈ 5M rows):
    PGUSER=siwei_zhao_2025 python src/deploy_q1_models_on_gaia.py --test-mode

    # full deployment (will produce ~1.5B rows, ~3-4 hr):
    PGUSER=siwei_zhao_2025 python src/deploy_q1_models_on_gaia.py

    # resumable: e.g. resume from where a prior run left off
    PGUSER=siwei_zhao_2025 python src/deploy_q1_models_on_gaia.py --start-ri 800000000

    # smaller batches if memory pressure
    PGUSER=siwei_zhao_2025 python src/deploy_q1_models_on_gaia.py --batch-size 1000000
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from selected_parameters_list import SELECTED_GAIA_PARAMETERS

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'
MAG_CUT   = 17

PROJECT_DIR    = Path(__file__).resolve().parents[1]
Q1_MODEL_DIR   = PROJECT_DIR / 'output' / 'xgboost_q1' / 'seed_models'
MIX_MODEL_DIR  = PROJECT_DIR / 'output' / 'xgboost_mix_q1_ero' / 'ratio_sg8.63_all_ero' / 'seed_models'

DEFAULT_BATCH  = 5_000_000
# Gaia DR3 random_index covers ~[0, 1.81e9]; round up generously.
DEFAULT_END_RI = 2_000_000_000

OUTPUT_TABLE_BASENAME = 'gaia_q1_pgal_predictions'


def load_ensemble(model_dir: Path, label: str) -> list[xgb.Booster]:
    """Load all seed boosters from a directory."""
    paths = sorted(model_dir.glob('xgb_classifier_q1_seed*.json'))
    if not paths:
        raise FileNotFoundError(f'No models found in {model_dir}')
    print(f'  [{label}] loading {len(paths):>2} seed models...')
    boosters = []
    for p in paths:
        b = xgb.Booster()
        b.load_model(str(p))
        boosters.append(b)
    return boosters


def predict_ensemble(boosters: list[xgb.Booster],
                     X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) of P(galaxy) across the seed ensemble."""
    dmat = xgb.DMatrix(X)
    probs = np.stack([b.predict(dmat) for b in boosters], axis=0)
    return probs.mean(axis=0), probs.std(axis=0)


def _feature_select_expr(col: str) -> str:
    """Return the SQL expression to compute one feature against gaia_dr3.gaia_source.

    Raw columns map directly. Derived columns are inlined using the same formulas
    as add_derived_columns.py (Riello+2021 Table 2 for corrected_bp_rp_excess).
    """
    if col == 'corrected_bp_rp_excess':
        return (
            "(phot_bp_rp_excess_factor - "
            "(CASE "
            "WHEN bp_rp < 0.5 THEN 1.154360 + 0.033772 * bp_rp + 0.032277 * bp_rp * bp_rp "
            "WHEN bp_rp < 4.0 THEN 1.162004 + 0.011464 * bp_rp + 0.049255 * bp_rp * bp_rp "
            "- 0.005879 * bp_rp * bp_rp * bp_rp "
            "ELSE 1.057572 + 0.140537 * bp_rp END)) AS corrected_bp_rp_excess"
        )
    if col == 'pm_significance':
        return (
            "SQRT((pmra * pmra * pmdec_error * pmdec_error "
            "- 2.0 * pmra_pmdec_corr * pmra_error * pmdec_error * pmra * pmdec "
            "+ pmdec * pmdec * pmra_error * pmra_error) "
            "/ (pmra_error * pmra_error * pmdec_error * pmdec_error "
            "* (1.0 - pmra_pmdec_corr * pmra_pmdec_corr))) AS pm_significance"
        )
    if col == 'position_error':
        return (
            "SQRT(ra_error * ra_error * dec_error * dec_error "
            "* (1.0 - ra_dec_corr * ra_dec_corr)) AS position_error"
        )
    return col


def fetch_batch_dataframe(cur, table: str, ri_lo: int, ri_hi: int) -> pd.DataFrame:
    """Fetch one random_index range directly from gaia_dr3.gaia_source.

    The 3 derived features (corrected_bp_rp_excess, pm_significance,
    position_error) are computed inline in SQL since gaia_dr3.gaia_source
    doesn't pre-materialise them.
    """
    feature_select = ',\n               '.join(
        _feature_select_expr(c) for c in SELECTED_GAIA_PARAMETERS
    )
    cur.execute(f"""
        SELECT source_id, random_index, ra, dec, phot_g_mean_mag,
               {feature_select}
        FROM {table}
        WHERE random_index >= {ri_lo}
          AND random_index <  {ri_hi}
          AND phot_g_mean_mag IS NOT NULL
          AND phot_g_mean_mag >= {MAG_CUT}
    """)
    rows = cur.fetchall()
    cols = ['source_id', 'random_index', 'ra', 'dec', 'phot_g_mean_mag'] + \
           list(SELECTED_GAIA_PARAMETERS)
    df = pd.DataFrame(rows, columns=cols)
    for c in SELECTED_GAIA_PARAMETERS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def copy_predictions(cur, table_name: str, df: pd.DataFrame,
                     q1_mean: np.ndarray, q1_std:  np.ndarray,
                     mix_mean: np.ndarray, mix_std: np.ndarray) -> None:
    """COPY a batch of predictions into the output table."""
    buf = io.StringIO()
    sids = df['source_id'].astype(np.int64).values
    ris  = df['random_index'].astype(np.int64).values
    ras  = df['ra'].astype(float).values
    decs = df['dec'].astype(float).values
    gmag = df['phot_g_mean_mag'].astype(float).values
    for i in range(len(df)):
        buf.write(
            f"{int(sids[i])}\t{int(ris[i])}\t"
            f"{ras[i]:.10f}\t{decs[i]:.10f}\t{gmag[i]:.6f}\t"
            f"{q1_mean[i]:.6f}\t{q1_std[i]:.6f}\t"
            f"{mix_mean[i]:.6f}\t{mix_std[i]:.6f}\n"
        )
    buf.seek(0)
    cur.copy_from(buf, table_name.split('.', 1)[1], sep='\t',
                  columns=('source_id', 'random_index', 'ra', 'dec',
                           'phot_g_mean_mag',
                           'q1_pgal_mean', 'q1_pgal_std',
                           'mix_pgal_mean', 'mix_pgal_std'))


def main() -> int:
    # Force line-buffered stdout so progress prints flush in real time
    # when redirected/piped (default is full-buffered for non-tty).
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--test-mode', action='store_true',
                        help='Process only the first batch (≈5M rows) for verification.')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH,
                        help=f'random_index range per batch (default {DEFAULT_BATCH:,})')
    parser.add_argument('--start-ri', type=int, default=0,
                        help='Start random_index (default 0; use to resume)')
    parser.add_argument('--end-ri', type=int, default=DEFAULT_END_RI,
                        help=f'End random_index (default {DEFAULT_END_RI:,})')
    parser.add_argument('--drop-existing', action='store_true',
                        help='DROP the output table before writing (default: append)')
    parser.add_argument('--output-suffix', default='',
                        help='Append to output table name (e.g. "_test")')
    args = parser.parse_args()

    username = os.getenv('PGUSER')
    if not username:
        print('ERROR: PGUSER not set'); return 1

    output_table = f'{username}.{OUTPUT_TABLE_BASENAME}{args.output_suffix}'
    if args.test_mode and not args.output_suffix:
        output_table += '_test'

    print('=' * 70)
    print('Deploy Q1 + Q1+ERO ensembles onto Gaia DR3')
    print(f'  Source     : gaia_dr3.gaia_source  (direct, no latitude split)')
    print(f'  Filter     : phot_g_mean_mag >= {MAG_CUT}')
    print(f'  Output     : {output_table}')
    print(f'  Batch size : {args.batch_size:,}  (random_index range)')
    print(f'  RI range   : [{args.start_ri:,}, {args.end_ri:,})')
    print(f'  Test mode  : {args.test_mode}')
    print('=' * 70)

    # ── Load model ensembles ─────────────────────────────────────────────────
    print('\n[1] Loading model ensembles...')
    q1_models  = load_ensemble(Q1_MODEL_DIR,  'Q1-only')
    mix_models = load_ensemble(MIX_MODEL_DIR, 'Q1+ERO max ratio')

    # ── Connect + create table ───────────────────────────────────────────────
    print(f'\n[2] Connecting to WSDB...')
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cur = conn.cursor()

    if args.drop_existing:
        cur.execute(f'DROP TABLE IF EXISTS {output_table};')
        print(f'  DROPped existing {output_table}')

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {output_table} (
            source_id        BIGINT PRIMARY KEY,
            random_index     BIGINT,
            ra               DOUBLE PRECISION,
            dec              DOUBLE PRECISION,
            phot_g_mean_mag  REAL,
            q1_pgal_mean     REAL,
            q1_pgal_std      REAL,
            mix_pgal_mean    REAL,
            mix_pgal_std     REAL
        );
    """)
    conn.commit()
    print(f'  Output table ready: {output_table}')

    # ── Stream + predict + COPY ──────────────────────────────────────────────
    # Query gaia_dr3.gaia_source directly using random_index range scans.
    print(f'\n[3] Streaming gaia_dr3.gaia_source, predicting, COPYing...')
    n_batches = 0
    n_total   = 0
    t_start   = time.time()
    source_table = 'gaia_dr3.gaia_source'
    ri_lo = args.start_ri
    while ri_lo < args.end_ri:
        ri_hi = min(ri_lo + args.batch_size, args.end_ri)
        batch_t0 = time.time()
        df = fetch_batch_dataframe(cur, source_table, ri_lo, ri_hi)
        n_rows = len(df)
        t_query = time.time() - batch_t0
        if n_rows == 0:
            print(f'  ri[{ri_lo:>11,}, {ri_hi:>11,})  '
                  f'empty (q={t_query:.1f}s) — skipped')
            ri_lo = ri_hi
            continue

        X = df[SELECTED_GAIA_PARAMETERS].values.astype(np.float32)
        t_pred0 = time.time()
        q1_mean, q1_std = predict_ensemble(q1_models, X)
        mix_mean, mix_std = predict_ensemble(mix_models, X)
        t_pred = time.time() - t_pred0

        t_copy0 = time.time()
        copy_predictions(cur, output_table, df,
                         q1_mean, q1_std, mix_mean, mix_std)
        conn.commit()
        t_copy = time.time() - t_copy0

        elapsed = time.time() - batch_t0
        n_batches += 1
        n_total   += n_rows
        print(f'  ri[{ri_lo:>11,}, {ri_hi:>11,})  '
              f'rows={n_rows:>9,}  '
              f'(q={t_query:6.1f}s  pred={t_pred:5.1f}s  '
              f'copy={t_copy:5.1f}s  total={elapsed:6.1f}s)  '
              f'cum={n_total:>12,}  wall={(time.time()-t_start)/60:.1f}min')

        if args.test_mode:
            print('  TEST MODE — stopping after one batch.')
            break

        ri_lo = ri_hi

    # ── Index + ANALYZE ──────────────────────────────────────────────────────
    if not args.test_mode and n_total > 0:
        print(f'\n[4] Adding q3c spatial index + ANALYZE...')
        try:
            cur.execute(f'CREATE INDEX IF NOT EXISTS '
                        f'{OUTPUT_TABLE_BASENAME}{args.output_suffix}_q3c_idx '
                        f'ON {output_table} (q3c_ang2ipix(ra, dec));')
            cur.execute(f'CLUSTER {output_table} USING '
                        f'{OUTPUT_TABLE_BASENAME}{args.output_suffix}_q3c_idx;')
            conn.commit()
            print('  ✓ q3c spatial index built + clustered')
        except Exception as e:
            conn.rollback()
            print(f'  q3c index step failed (non-fatal): {e}')
        cur.execute(f'ANALYZE {output_table};')
        conn.commit()

    cur.execute(f'SELECT COUNT(*) FROM {output_table};')
    final_count = cur.fetchone()[0]

    cur.close(); conn.close()

    elapsed_total = time.time() - t_start
    print(f'\n' + '=' * 70)
    print(f'Done.  {n_batches} batch(es), {n_total:,} rows processed in '
          f'{elapsed_total/60:.1f} min')
    print(f'Output table {output_table}: {final_count:,} total rows')
    print('=' * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
