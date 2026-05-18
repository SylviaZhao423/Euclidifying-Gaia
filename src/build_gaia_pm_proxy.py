#!/usr/bin/env python3
"""
Build {username}.gaia_pm_proxy — neighbour-PM proxy for Gaia 2-par sources
near Euclid (Q1 + ERO) cross-match candidates.

The median-PM aggregation is done in Python with ``scipy.spatial.cKDTree``,
processing each Euclid field separately to keep the q3c index lookups
small. (A pure-SQL version exists in ``archive/build_gaia_pm_proxy_sql.py``
that ran ~100× slower for the same numerical result.)

The candidate-2-par footprint covers ~21 separated Euclid fields
spanning the full sky, so a single bounding box over all candidates is
unworkable (it pulls hundreds of millions of rows). Instead we process
each known Q1 / ERO field separately: for each field we use
``q3c_radial_query`` (which hits the q3c index) to pull only the 5/6-par
Gaia neighbours within (field_radius + 30") of the field centre, then
build a cKDTree and query the candidate medians locally.

Pipeline
--------
  1. Pull distinct 2-par g_source_ids near Euclid (from
     crossmatch_q1_mer_phz_vis_gaia and crossmatch_ero_gaia_1_0_uncor —
     both PM-correction-agnostic, no chicken-and-egg with this script's
     consumers).
  2. For each predefined Q1 / ERO field, pull all 5/6-par Gaia
     (source_id, ra, dec, pmra, pmdec) within (field_radius + 30").
  3. Build a cKDTree over the 5/6-par positions for that field, and
     query the candidates that fall within (field_radius + 30") of the
     centre. Take per-axis median PM, keep if ≥ 5 neighbours.
  4. Accumulate results across fields. COPY back to PostgreSQL as
     {username}.gaia_pm_proxy (g_source_id, proxy_pmra, proxy_pmdec,
     n_neighbours).

Usage:
    PGUSER=siwei_zhao_2025 python src/build_gaia_pm_proxy.py
"""

import io
import os
import sys
import time

import numpy as np
import psycopg2
from scipy.spatial import cKDTree

WSDB_HOST = 'wsdb.ast.cam.ac.uk'
WSDB_DB   = 'wsdb'

NEIGHBOUR_RADIUS_ARCSEC = 30.0
MIN_NEIGHBOURS          = 5
NEIGHBOUR_MARGIN_DEG    = NEIGHBOUR_RADIUS_ARCSEC / 3600.0

# Fields covering all candidate 2-par sources.
# (RA, Dec, field_radius_deg) — radius generous to cover the field
# AND a 30" margin for cKDTree queries. Q1 fields use ~3° (EDF-N is
# 22.9 deg² ≈ 2.7° radius, plus margin); LDN 1641 is small. ERO fields
# are small pointings, ~0.6° comfortably covers each + margin.
Q1_FIELDS = [
    ("EDF-N",     269.733, +66.018, 4.0),
    ("EDF-S",      61.241, -48.423, 4.5),
    ("EDF-F",      52.932, -28.088, 3.5),
    ("LDN 1641",   85.75,   -8.367, 1.5),
]
ERO_FIELDS = [
    ("Horsehead",     85.150,  -2.613, 1.0),
    ("Messier 78",    86.690,  -0.015, 1.0),
    ("Taurus",        64.983, +28.023, 1.0),
    ("Barnard 30",    82.880, +12.316, 1.0),
    ("NGC 6254",     254.303,  -4.100, 1.0),
    ("NGC 6397",     265.174, -53.658, 1.0),
    ("NGC 6822",     296.236, -14.788, 1.0),
    ("Holmberg II",  124.790, +70.706, 1.0),
    ("IC 10",          5.063, +59.288, 1.0),
    ("IC 342",        56.730, +68.084, 1.0),
    ("NGC 2403",     114.211, +65.586, 1.0),
    ("NGC 6744",     287.430, -63.842, 1.0),
    ("Fornax",        54.017, -35.267, 1.0),
    ("Dorado",        64.014, -55.780, 1.0),
    ("Perseus",       49.638, +41.651, 1.2),
    ("Abell 2390",   328.397, +17.709, 1.2),
    ("Abell 2764",     5.713, -49.249, 1.2),
]
ALL_FIELDS = [(n, r, d, rad + 0.05) for n, r, d, rad in Q1_FIELDS + ERO_FIELDS]


def _sky_to_xyz(ra_deg, dec_deg):
    ra  = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    cd  = np.cos(dec)
    return np.column_stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)])


def _great_circle_arcsec(ra1, dec1, ra2, dec2):
    """Single-pair haversine distance in arcsec — used for cand-to-field assignment."""
    ra1, dec1 = np.radians(ra1), np.radians(dec1)
    ra2, dec2 = np.radians(ra2), np.radians(dec2)
    dra, ddec = ra2 - ra1, dec2 - dec1
    a = np.sin(ddec/2)**2 + np.cos(dec1)*np.cos(dec2)*np.sin(dra/2)**2
    return 2 * np.arcsin(np.sqrt(a)) * 180.0/np.pi * 3600.0


def main():
    username = os.getenv('PGUSER')
    if not username:
        print("ERROR: PGUSER not set"); return 1

    proxy_table = f'{username}.gaia_pm_proxy'

    print('=' * 70)
    print(f'Building {proxy_table}  (Python-native, per-field)')
    print(f'  Neighbour radius      : {NEIGHBOUR_RADIUS_ARCSEC}"')
    print(f'  Min 5/6-par neighbours: {MIN_NEIGHBOURS}')
    print(f'  Fields                : {len(ALL_FIELDS)} (Q1 + ERO)')
    print('=' * 70)

    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cur  = conn.cursor()

    # ── [1] pull candidates ────────────────────────────────────────────────
    print('\n[1] Pulling candidate 2-par sources from Q1 + ERO crossmatches...')
    t0 = time.time()
    cur.execute(f"""
        SELECT g_source_id, g_ra, g_dec FROM (
            SELECT g_source_id, g_ra, g_dec
              FROM {username}.crossmatch_q1_mer_phz_vis_gaia
              WHERE g_pmra IS NULL AND g_pmdec IS NULL
            UNION
            SELECT g_source_id, g_ra, g_dec
              FROM {username}.crossmatch_ero_gaia_1_0_uncor
              WHERE g_pmra IS NULL AND g_pmdec IS NULL
        ) AS s
    """)
    rows = cur.fetchall()
    cand_id  = np.array([r[0] for r in rows], dtype=np.int64)
    cand_ra  = np.array([r[1] for r in rows], dtype=float)
    cand_dec = np.array([r[2] for r in rows], dtype=float)
    print(f'  → {len(cand_id):,} candidate 2-par sources  ({time.time()-t0:.1f} s)')

    # Per-source result accumulators
    proxy_pmra   = np.full(len(cand_id), np.nan)
    proxy_pmdec  = np.full(len(cand_id), np.nan)
    n_neighbours = np.zeros(len(cand_id), dtype=int)

    # ── [2-3] per-field neighbour pull + median ───────────────────────────
    print(f'\n[2-3] Processing {len(ALL_FIELDS)} fields...')
    total_t0 = time.time()
    total_neighbours_pulled = 0
    total_candidates_processed = 0

    for fi, (name, ra_c, dec_c, rad_deg) in enumerate(ALL_FIELDS, 1):
        # candidates inside this field's effective radius (+ margin)
        in_field = (_great_circle_arcsec(cand_ra, cand_dec, ra_c, dec_c)
                    <= rad_deg * 3600.0)
        n_cand_field = int(in_field.sum())
        if n_cand_field == 0:
            print(f'  [{fi:2d}/{len(ALL_FIELDS)}] {name:<14}  '
                  f'(no candidates) — skipped')
            continue

        # pull 5/6-par neighbours within (field_radius + 30") of centre via q3c
        pull_radius_deg = rad_deg + NEIGHBOUR_MARGIN_DEG
        t_pull = time.time()
        cur.execute(f"""
            SELECT source_id, ra, dec, pmra, pmdec
            FROM gaia_dr3.gaia_source
            WHERE astrometric_params_solved IN (31, 95)
              AND q3c_radial_query(ra, dec, %s, %s, %s)
        """, (ra_c, dec_c, pull_radius_deg))
        npairs = cur.fetchall()
        if not npairs:
            print(f'  [{fi:2d}/{len(ALL_FIELDS)}] {name:<14}  '
                  f'cand={n_cand_field:>6,}  '
                  f'no 5/6-par neighbours found — skipped')
            continue

        n5p_ra   = np.array([r[1] for r in npairs], dtype=float)
        n5p_dec  = np.array([r[2] for r in npairs], dtype=float)
        n5p_pmra = np.array([r[3] for r in npairs], dtype=float)
        n5p_pmd  = np.array([r[4] for r in npairs], dtype=float)
        total_neighbours_pulled += len(n5p_ra)

        # cKDTree on the unit sphere
        tree_xyz = _sky_to_xyz(n5p_ra, n5p_dec)
        tree = cKDTree(tree_xyz)
        chord_radius = 2.0 * np.sin(np.radians(NEIGHBOUR_RADIUS_ARCSEC / 3600.0) / 2.0)

        cand_xyz = _sky_to_xyz(cand_ra[in_field], cand_dec[in_field])
        idx_per_cand = tree.query_ball_point(cand_xyz, r=chord_radius, workers=-1)

        # write into accumulators
        local_indices = np.where(in_field)[0]
        for li, neighbours in enumerate(idx_per_cand):
            gi = local_indices[li]
            if len(neighbours) > n_neighbours[gi]:
                # this field gave more neighbours — overwrite the proxy
                # (in practice, candidates rarely fall in two field-radius
                # windows simultaneously)
                n_neighbours[gi] = len(neighbours)
                if len(neighbours) >= MIN_NEIGHBOURS:
                    proxy_pmra[gi]  = np.median(n5p_pmra[neighbours])
                    proxy_pmdec[gi] = np.median(n5p_pmd[neighbours])

        total_candidates_processed += n_cand_field
        print(f'  [{fi:2d}/{len(ALL_FIELDS)}] {name:<14}  '
              f'cand={n_cand_field:>6,}  5/6-par={len(n5p_ra):>7,}  '
              f'({time.time()-t_pull:.1f} s)')

    print(f'\n  Total per-field processing: {time.time()-total_t0:.1f} s')
    print(f'  Total 5/6-par neighbours pulled (across fields): '
          f'{total_neighbours_pulled:,}')

    keep = np.isfinite(proxy_pmra)
    print(f'\n  Sources with valid proxy PM (≥{MIN_NEIGHBOURS} neighbours): '
          f'{int(keep.sum()):,} / {len(cand_id):,}  '
          f'({100*keep.mean():.1f}%)')

    # ── [4] COPY back to Postgres ─────────────────────────────────────────
    print(f'\n[4] Writing {proxy_table} via COPY...')
    t0 = time.time()
    cur.execute(f"DROP TABLE IF EXISTS {proxy_table};")
    cur.execute(f"""
        CREATE TABLE {proxy_table} (
            g_source_id  bigint PRIMARY KEY,
            proxy_pmra   real,
            proxy_pmdec  real,
            n_neighbours int
        );
    """)
    conn.commit()

    buf = io.StringIO()
    for sid, pra, pdec, nn in zip(cand_id[keep], proxy_pmra[keep],
                                   proxy_pmdec[keep], n_neighbours[keep]):
        buf.write(f"{int(sid)}\t{float(pra):.8f}\t{float(pdec):.8f}\t{int(nn)}\n")
    buf.seek(0)
    cur.copy_from(buf, proxy_table.split('.', 1)[1],
                  sep='\t', columns=('g_source_id', 'proxy_pmra',
                                      'proxy_pmdec', 'n_neighbours'))
    conn.commit()
    cur.execute(f"ANALYZE {proxy_table};")
    cur.execute(f"SELECT COUNT(*) FROM {proxy_table};")
    n_proxy = cur.fetchone()[0]
    print(f'  → {n_proxy:,} rows written  ({time.time()-t0:.1f} s)')

    cur.close(); conn.close()
    print('\nDone.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
