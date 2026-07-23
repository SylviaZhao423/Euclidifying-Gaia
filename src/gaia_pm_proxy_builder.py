#!/usr/bin/env python3
"""Shared neighbour-PM proxy engine (library, not a standalone step).

Given a list of 2-parameter Gaia candidate sources and a set of Euclid fields,
compute — per field, in Python with scipy.cKDTree — the median proper motion of
the 5/6-parameter Gaia neighbours within 30" (kept only when ≥ 5 neighbours) and
write it to a proxy table:  (g_source_id, proxy_pmra, proxy_pmdec, n_neighbours).

Used by the two per-catalogue proxy builders:
    crossmatch_ero_gaia_1_0_pm_proxy.py          → gaia_pm_proxy_ero
    crossmatch_q1_mer_phz_vis_gaia_pm_proxy.py   → gaia_pm_proxy_q1

(These replace the old combined build_gaia_pm_proxy.py; the ERO and Q1 candidate
sets are disjoint on-sky, so the two proxy tables partition the old gaia_pm_proxy.)
"""
from __future__ import annotations

import io
import time

import numpy as np
from scipy.spatial import cKDTree

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB = "wsdb"

NEIGHBOUR_RADIUS_ARCSEC = 30.0
MIN_NEIGHBOURS = 5
NEIGHBOUR_MARGIN_DEG = NEIGHBOUR_RADIUS_ARCSEC / 3600.0
FIELD_MARGIN_DEG = 0.05          # small buffer on the field radius for candidate assignment

# (name, RA, Dec, field_radius_deg) — see build_gaia_pm_proxy.py history for provenance.
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
Q1_FIELDS = [
    ("EDF-N",     269.733, +66.018, 4.0),
    ("EDF-S",      61.241, -48.423, 4.5),
    ("EDF-F",      52.932, -28.088, 3.5),
    ("LDN 1641",   85.75,   -8.367, 1.5),
]


def _sky_to_xyz(ra_deg, dec_deg):
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    cd = np.cos(dec)
    return np.column_stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)])


def _great_circle_arcsec(ra1, dec1, ra2, dec2):
    ra1, dec1 = np.radians(ra1), np.radians(dec1)
    ra2, dec2 = np.radians(ra2), np.radians(dec2)
    dra, ddec = ra2 - ra1, dec2 - dec1
    a = np.sin(ddec / 2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra / 2)**2
    return 2 * np.arcsin(np.sqrt(a)) * 180.0 / np.pi * 3600.0


def build_proxy(conn, cand_id, cand_ra, cand_dec, fields, out_table) -> int:
    """Compute the neighbour-PM proxy for `cand_*` over `fields` and write it to
    `out_table` (fully-qualified). Returns the number of proxy rows written.

    `fields` = [(name, ra_deg, dec_deg, field_radius_deg), ...] (base radii; a
    FIELD_MARGIN_DEG buffer is added internally, then +30" for the neighbour pull)."""
    cand_id = np.asarray(cand_id, dtype=np.int64)
    cand_ra = np.asarray(cand_ra, dtype=float)
    cand_dec = np.asarray(cand_dec, dtype=float)
    proxy_pmra = np.full(len(cand_id), np.nan)
    proxy_pmdec = np.full(len(cand_id), np.nan)
    n_neighbours = np.zeros(len(cand_id), dtype=int)

    cur = conn.cursor()
    print(f"\n[proxy] {len(cand_id):,} candidates over {len(fields)} fields...")
    total_t0 = time.time()
    chord_radius = 2.0 * np.sin(np.radians(NEIGHBOUR_RADIUS_ARCSEC / 3600.0) / 2.0)

    for fi, (name, ra_c, dec_c, base_rad) in enumerate(fields, 1):
        rad_deg = base_rad + FIELD_MARGIN_DEG
        in_field = _great_circle_arcsec(cand_ra, cand_dec, ra_c, dec_c) <= rad_deg * 3600.0
        n_cand = int(in_field.sum())
        if n_cand == 0:
            print(f"  [{fi:2d}/{len(fields)}] {name:<14} (no candidates) — skipped")
            continue

        cur.execute("""
            SELECT source_id, ra, dec, pmra, pmdec
            FROM gaia_dr3.gaia_source
            WHERE astrometric_params_solved IN (31, 95)
              AND q3c_radial_query(ra, dec, %s, %s, %s)
        """, (ra_c, dec_c, rad_deg + NEIGHBOUR_MARGIN_DEG))
        npairs = cur.fetchall()
        if not npairs:
            print(f"  [{fi:2d}/{len(fields)}] {name:<14} cand={n_cand:>6,}  "
                  f"no 5/6-par neighbours — skipped")
            continue
        n5p_ra = np.array([r[1] for r in npairs], dtype=float)
        n5p_dec = np.array([r[2] for r in npairs], dtype=float)
        n5p_pmra = np.array([r[3] for r in npairs], dtype=float)
        n5p_pmd = np.array([r[4] for r in npairs], dtype=float)

        tree = cKDTree(_sky_to_xyz(n5p_ra, n5p_dec))
        idx_per_cand = tree.query_ball_point(_sky_to_xyz(cand_ra[in_field], cand_dec[in_field]),
                                             r=chord_radius, workers=-1)
        local = np.where(in_field)[0]
        for li, nb in enumerate(idx_per_cand):
            gi = local[li]
            if len(nb) > n_neighbours[gi]:     # a candidate rarely falls in two fields
                n_neighbours[gi] = len(nb)
                if len(nb) >= MIN_NEIGHBOURS:
                    proxy_pmra[gi] = np.median(n5p_pmra[nb])
                    proxy_pmdec[gi] = np.median(n5p_pmd[nb])
        print(f"  [{fi:2d}/{len(fields)}] {name:<14} cand={n_cand:>6,}  "
              f"5/6-par={len(n5p_ra):>7,}")

    keep = np.isfinite(proxy_pmra)
    print(f"\n[proxy] valid (≥{MIN_NEIGHBOURS} neighbours): {int(keep.sum()):,} / "
          f"{len(cand_id):,} ({100*keep.mean():.1f}%)  [{time.time()-total_t0:.0f}s]")

    cur.execute(f"DROP TABLE IF EXISTS {out_table};")
    cur.execute(f"""
        CREATE TABLE {out_table} (
            g_source_id  bigint PRIMARY KEY,
            proxy_pmra   real,
            proxy_pmdec  real,
            n_neighbours int
        );""")
    conn.commit()
    buf = io.StringIO()
    for sid, pra, pdec, nn in zip(cand_id[keep], proxy_pmra[keep],
                                  proxy_pmdec[keep], n_neighbours[keep]):
        buf.write(f"{int(sid)}\t{float(pra):.8f}\t{float(pdec):.8f}\t{int(nn)}\n")
    buf.seek(0)
    cur.copy_from(buf, out_table.split(".", 1)[1], sep="\t",
                  columns=("g_source_id", "proxy_pmra", "proxy_pmdec", "n_neighbours"))
    conn.commit()
    cur.execute(f"ANALYZE {out_table};")
    cur.execute(f"SELECT COUNT(*) FROM {out_table};")
    n = cur.fetchone()[0]
    cur.close()
    print(f"[proxy] → {out_table}  ({n:,} rows)")
    return n
