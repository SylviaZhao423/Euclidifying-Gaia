#!/usr/bin/env python3
"""
Sankey diagram showing how ERO training data is refined through successive filters.

Pipeline:
    ERO merge_cat
      → × Gaia at 0.3″ (PM-corrected AND raw — intersection canonical)
      → Gaia G ≥ 17
      → Point-like (class_star ≥ 0.65) / Extended (class_star < 0.65)

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_ero_filter_sankey.py

Output:
    output/ero_filter_sankey.html
    output/ero_filter_sankey.png
"""

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_HTML = PROJECT_DIR / "output" / "ero_filter_sankey.html"
OUTPUT_PNG  = PROJECT_DIR / "output" / "ero_filter_sankey.png"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"


def query_counts(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cur  = conn.cursor()

    def n(table, where=""):
        sql = f"SELECT COUNT(*) FROM {table}"
        if where:
            sql += f" WHERE {where}"
        cur.execute(sql)
        return cur.fetchone()[0]

    print("Querying ERO filter-stage counts...")

    c = {}

    c["merge"] = n("euclid_ero.merge_cat")
    print(f"  ERO merge_cat (total):      {c['merge']:>10,}")

    # Canonical _0_3 = intersection (PM-corrected sep ≤ 0.3 AND raw sep ≤ 0.3)
    c["gaia"] = n("siwei_zhao_2025.crossmatch_ero_gaia_0_3",
                  "e_v_class_star IS NOT NULL")
    print(f"  After × Gaia 0.3\" (PM+raw): {c['gaia']:>10,}")

    c["gmag"] = n("siwei_zhao_2025.crossmatch_ero_gaia_0_3",
                  "e_v_class_star IS NOT NULL AND g_phot_g_mean_mag >= 17")
    print(f"  After G ≥ 17 (training):    {c['gmag']:>10,}")

    c["point"] = n("siwei_zhao_2025.crossmatch_ero_gaia_0_3",
                   "e_v_class_star IS NOT NULL AND g_phot_g_mean_mag >= 17 "
                   "AND e_v_class_star >= 0.65")
    c["ext"] = n("siwei_zhao_2025.crossmatch_ero_gaia_0_3",
                 "e_v_class_star IS NOT NULL AND g_phot_g_mean_mag >= 17 "
                 "AND e_v_class_star < 0.65")
    print(f"    Point-like (≥0.65):       {c['point']:>10,}")
    print(f"    Extended   (<0.65):       {c['ext']:>10,}")

    cur.close()
    conn.close()
    return c


def _scale(v):
    """Log2-compress values for display so large upstream drops don't dominate."""
    import math
    return max(1, int(math.log2(v + 1) * 120))


def build_sankey(c):
    import plotly.graph_objects as go

    PASS_COLOR  = "#4a86c8"
    POINT_COLOR = "#e07b54"
    EXT_COLOR   = "#5b9bd5"
    DROP_COLOR  = "#bbbbbb"
    DROP_LINK   = "rgba(180,180,180,0.35)"
    PASS_LINK   = "rgba(74,134,200,0.42)"
    POINT_LINK  = "rgba(224,123,84,0.45)"
    EXT_LINK    = "rgba(91,155,213,0.45)"

    # Node indices:
    # 0–2  main pipeline
    # 3–4  terminal: Point-like, Extended
    # 5–6  drop buckets (pipeline order: No Gaia match, G<17)

    labels = [
        # 0–2 pipeline
        f"ERO MER<br>{c['merge']:,}",
        f"× Gaia 0.3\"<br>(PM + raw)<br>{c['gaia']:,}",
        f"Training set<br>G ≥ 17<br>{c['gmag']:,}",
        # 3–4 terminals
        f"Point-like<br>≥ 0.65<br>{c['point']:,}",
        f"Extended<br>< 0.65<br>{c['ext']:,}",
        # 5–6 drops (in pipeline order, stacked below terminals)
        f"No Gaia match<br>−{c['merge']-c['gaia']:,}",
        f"G < 17<br>−{c['gaia']-c['gmag']:,}",
    ]

    node_colors = [
        PASS_COLOR, PASS_COLOR, PASS_COLOR,
        POINT_COLOR, EXT_COLOR,
        DROP_COLOR, DROP_COLOR,
    ]

    # Pipeline rises bottom-left → top-right; drops stacked on right below terminals.
    # Plotly Sankey: y=0 top, y=1 bottom.
    node_x = [
        0.04, 0.45, 0.74,   # pipeline
        0.95, 0.95,          # terminals
        0.95, 0.95,          # drops
    ]
    node_y = [
        0.80, 0.45, 0.22,   # pipeline
        0.05, 0.17,          # Point-like top, Extended just below
        0.65, 0.45,          # No Gaia match (earliest drop, lowest), G<17 above it
    ]

    links = [
        # pass-through
        (0, 1, _scale(c["gaia"]),  PASS_LINK),
        (1, 2, _scale(c["gmag"]),  PASS_LINK),
        # training set → Point-like / Extended (proportional split)
        (2, 3, int(_scale(c["gmag"]) * c["point"] / c["gmag"]), POINT_LINK),
        (2, 4, int(_scale(c["gmag"]) * c["ext"]   / c["gmag"]), EXT_LINK),
        # drops
        (0, 5, _scale(c["merge"] - c["gaia"]),  DROP_LINK),
        (1, 6, _scale(c["gaia"]  - c["gmag"]),  DROP_LINK),
    ]

    src, tgt, val, col = zip(*links)

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18,
            thickness=22,
            line=dict(color="white", width=0.5),
            label=labels,
            color=node_colors,
            x=node_x,
            y=node_y,
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=list(src),
            target=list(tgt),
            value=list(val),
            color=list(col),
        ),
    ))

    fig.update_layout(
        # title=dict(
        #     text=(
        #         "ERO Training Data Filter Pipeline<br>"
        #         "<sup>Euclid ERO × Gaia DR3 — source counts at each selection stage  |  band widths are log-scaled</sup>"
        #     ),
        #     font=dict(size=32, weight="bold"),
        #     x=0.5,
        #     xanchor="center",
        # ),
        font=dict(size=26, family="Arial", weight="bold"),
        width=1600,
        height=800,
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=20, b=90),
    )

    return fig


def main():
    username = os.getenv("PGUSER")
    if not username:
        print("ERROR: PGUSER not set"); return 1

    c = query_counts(username)
    print("\nBuilding Sankey diagram...")
    fig = build_sankey(c)

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(OUTPUT_HTML),
        include_plotlyjs="cdn",
        config={
            "toImageButtonOptions": {
                "format": "png",
                "filename": "ero_filter_sankey",
                "width": 820,
                "height": 400,
                "scale": 2,
            }
        },
    )
    print(f"Saved: {OUTPUT_HTML}")
    fig.write_image(str(OUTPUT_PNG), scale=2)
    print(f"Saved: {OUTPUT_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
