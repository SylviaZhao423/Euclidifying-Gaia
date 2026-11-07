#!/usr/bin/env python3
"""
Diagnostic: show the individual Euclid MEMBERS of a blended system side by side.

For each given primary (a Gaia source_id in crossmatch_qso_catwise_euclid_gaia), look up
its `all_euclid_ids_within_1arcsec`, resolve every member's RA/Dec
(euclid_q1.mer_final_cat for Q1, euclid_ero.merge_cat for ERO), and render one Euclid
cutout CENTRED ON EACH MEMBER — so the two (or more) components of a single system sit
side by side. The other member(s) + the WISE position are overlaid as rings, so you can
see the companion from each centring (unlike the main script, which makes one stamp per
primary row).

Row = system (primary), Col = member (sorted by separation from the primary; col 0 = the
primary itself). Titles give the Euclid id and its separation from the primary.

Reads (WSDB; VPN; ~/.pgpass):
  {PGUSER}.crossmatch_qso_catwise_euclid_gaia   (members + positions per system)
  euclid_q1.mer_final_cat / euclid_ero.merge_cat (member RA/Dec)
Writes:
  output/qso_pairs/pair_members_<...>.png

Usage:
  PGUSER=siwei_zhao_2025 python src/inspect_euclid_pair_members.py     # the two default systems
  PGUSER=siwei_zhao_2025 python src/inspect_euclid_pair_members.py --gaia-ids 3219147845996327168,3219223299980727168 --size 8
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_q1_cutout import (_assign_to_tiles, _render_cutout, _nisp_filename,  # noqa: E402
                              IRSA_IBE_DATA, EUCLID_RING, WISE_RING)
from render_ero_cutout import render_ero_cutout                                  # noqa: E402
from builder_qso_gaia_crossmatch import WSDB_HOST, WSDB_DB                       # noqa: E402
from download_qso_catwise_e_g_cutouts import _tiles_near, FETCH_TIMEOUT, _on_alarm  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_DIR / "output" / "qso_pairs"
PILE_DIR = PROJECT_DIR / "output" / "qso_cutouts" / "catwise_e_g"   # the main cutout pile
CATWISE_TABLE = "crossmatch_qso_catwise_euclid_gaia"
# the two systems from the ERO cutouts in question
DEFAULT_GIDS = [3219147845996327168, 3219223299980727168]


def fetch_systems(gids: list[int], user: str) -> list[dict]:
    """One dict per primary: e_id, e_ra/e_dec, euclid_source, member ids + seps, WISE pos,
    and each member's resolved (ra, dec)."""
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=user, connect_timeout=30,
                            keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5)
    cur = conn.cursor(); cur.execute("SET statement_timeout = 60000")
    cur.execute(f"""
        SELECT g_source_id, e_id, e_ra, e_dec, euclid_source,
               all_euclid_ids_within_1arcsec, all_euclid_seps_within_1arcsec,
               qso_ra, qso_dec
        FROM {user}.{CATWISE_TABLE}
        WHERE g_source_id = ANY(%s)""", (gids,))
    cols = [d[0] for d in cur.description]
    systems = [dict(zip(cols, r)) for r in cur.fetchall()]

    # collect member ids per catalog and resolve positions
    q1_ids, ero_ids = set(), set()
    for s in systems:
        ids = list(s["all_euclid_ids_within_1arcsec"] or [s["e_id"]])
        s["member_ids"] = [int(x) for x in ids]
        (q1_ids if s["euclid_source"] == "Q1" else ero_ids).update(s["member_ids"])
    pos = {}
    if q1_ids:
        cur.execute("SELECT object_id, right_ascension, declination "
                    "FROM euclid_q1.mer_final_cat WHERE object_id = ANY(%s)", (list(q1_ids),))
        pos.update({int(i): (float(a), float(d)) for i, a, d in cur.fetchall()})
    if ero_ids:
        cur.execute("SELECT v_objid, v_alpha_j2000, v_delta_j2000 "
                    "FROM euclid_ero.merge_cat WHERE v_objid = ANY(%s)", (list(ero_ids),))
        pos.update({int(i): (float(a), float(d)) for i, a, d in cur.fetchall()})
    cur.close(); conn.close()

    for s in systems:
        seps = list(s["all_euclid_seps_within_1arcsec"] or [0.0])
        # primary always shown, from its own e_ra/e_dec (don't rely on the id lookup)
        members = [(int(s["e_id"]), float(s["e_ra"]), float(s["e_dec"]), 0.0)]
        seen = {int(s["e_id"])}
        for j, mid in enumerate(s["member_ids"]):
            if mid in seen or mid not in pos:
                continue
            sep = seps[j] if j < len(seps) else float("nan")
            members.append((mid, pos[mid][0], pos[mid][1], sep)); seen.add(mid)
        members.sort(key=lambda t: t[3])   # by separation (primary ~0 first)
        s["members"] = members
    return systems


def _pile_map() -> dict:
    """{gaia_source_id: Path} for each source's existing cutout in the catwise_e_g pile."""
    mf = PILE_DIR / "manifest.csv"
    if not mf.exists():
        return {}
    m = pd.read_csv(mf, usecols=["gaia_source_id", "cutout_file"])
    return {int(r.gaia_source_id): PILE_DIR / r.cutout_file for r in m.itertuples()}


def render_member(ra: float, dec: float, size: float, source: str, companions: list):
    """Render a single Euclid stamp centred on (ra, dec), with `companions` overlaid."""
    if source == "ERO":
        return render_ero_cutout(ra, dec, size, draw_crosshair=True,
                                 companions=companions, crosshair_color="red")
    ti = _assign_to_tiles(ra, dec, _tiles_near(ra, dec), "euclidwise")
    if ti is None:
        return None
    njf, nhf = _nisp_filename(ti.tile_id, "J"), _nisp_filename(ti.tile_id, "H")
    nju = f"{IRSA_IBE_DATA}/{ti.tile_id}/NISP/{njf}" if njf else None
    nhu = f"{IRSA_IBE_DATA}/{ti.tile_id}/NISP/{nhf}" if nhf else None
    signal.signal(signal.SIGALRM, _on_alarm); signal.alarm(FETCH_TIMEOUT)
    try:
        return _render_cutout(ti.url, ra, dec, size, "", draw_crosshair=True,
                              nisp_j_url=nju, nisp_h_url=nhu,
                              companions=companions, crosshair_color="red")
    except Exception as exc:
        print(f"    member @ ({ra:.5f},{dec:.5f}) failed: {exc}"); return None
    finally:
        signal.alarm(0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gaia-ids", default=",".join(map(str, DEFAULT_GIDS)),
                    help="comma-separated Gaia source_ids of the primaries to inspect.")
    ap.add_argument("--size", type=float, default=8.0, help="cutout FOV arcsec (default 8).")
    ap.add_argument("--no-pile", action="store_true",
                    help="don't prepend each system's existing catwise_e_g pile cutout as column 0.")
    args = ap.parse_args()
    gids = [int(x) for x in args.gaia_ids.split(",")]
    user = os.environ.get("PGUSER", "siwei_zhao_2025")

    systems = fetch_systems(gids, user)
    if not systems:
        print("  no matching systems in the crossmatch."); return 1
    for s in systems:
        print(f"  gaia {s['g_source_id']}  {s['euclid_source']}  e_id={s['e_id']}  "
              f"members={[m[0] for m in s['members']]}  seps={[round(m[3],2) for m in s['members']]}")

    with_pile = not args.no_pile
    pile = _pile_map() if with_pile else {}
    off = 1 if with_pile else 0
    nmem = max(len(s["members"]) for s in systems)
    ncols = nmem + off
    fig, axes = plt.subplots(len(systems), ncols, squeeze=False,
                             figsize=(4.2 * ncols, 4.6 * len(systems)))
    for i, s in enumerate(systems):
        for ax in axes[i]:
            ax.axis("off")
        # col 0: the source's existing stamp from the catwise_e_g pile (title baked in)
        if with_pile:
            ax = axes[i][0]
            pp = pile.get(int(s["g_source_id"]))
            if pp and pp.exists():
                ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
                ax.imshow(plt.imread(str(pp)))
                ax.set_title(f"in pile: {pp.parent.name}/{pp.name}", fontsize=8)
            else:
                ax.text(0.5, 0.5, "not in pile", ha="center", va="center", color="0.5")
        mem = s["members"]
        for j, (mid, ra, dec, sep) in enumerate(mem):
            ax = axes[i][off + j]
            others = [(r2, d2, None, EUCLID_RING, "crosshair") for k, (m2, r2, d2, _) in enumerate(mem) if k != j]
            others.append((float(s["qso_ra"]), float(s["qso_dec"]), None, WISE_RING, "x"))
            arr = render_member(ra, dec, args.size, s["euclid_source"], others)
            ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
            if arr is not None:
                ax.imshow(np.clip(arr, 0, 1))
            else:
                ax.text(0.5, 0.5, "no Euclid tile", ha="center", va="center", color="w")
                ax.set_facecolor("k")
            tag = "primary" if j == 0 else f"companion {j}"
            ax.set_title(f"{tag}: e_id={mid}\nsep={sep:.2f}\"  ({ra:.5f}, {dec:+.5f})", fontsize=8)
        axes[i][0].set_ylabel(f"gaia {s['g_source_id']}\n{s['euclid_source']}",
                              fontsize=9, rotation=0, ha="right", va="center")
    fig.suptitle("Euclid members per system — col 0 = existing pile cutout; other cols centred on each member "
                 "(red + = other member, magenta ✕ = WISE)", fontsize=12, fontweight="bold")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"pair_members_{'_'.join(str(g) for g in gids)}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
