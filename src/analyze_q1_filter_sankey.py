#!/usr/bin/env python3
"""
Sankey diagram showing how Q1 training data is refined through successive filters.

Pipeline:
    Q1 × Gaia crossmatch
      → m_vis_det = 1
      → angular sep ≤ 0.3″
      → PHZ classification available
      → PHZ ∈ {Star (1), Galaxy (2)}
      → Gaia G ≥ 17
      → 50k UMAP subsample

Usage:
    PGUSER=siwei_zhao_2025 python src/analyze_q1_filter_sankey.py

Output:
    output/q1_filter_sankey.html
"""

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_HTML = PROJECT_DIR / "output" / "q1_filter_sankey.html"
OUTPUT_PNG  = PROJECT_DIR / "output" / "q1_filter_sankey.png"

WSDB_HOST = "wsdb.ast.cam.ac.uk"
WSDB_DB   = "wsdb"
UMAP_SAMPLE = 50_000


def query_counts(username):
    import psycopg2
    conn = psycopg2.connect(host=WSDB_HOST, database=WSDB_DB, user=username)
    cur  = conn.cursor()

    def n_table(table, where=""):
        sql = f"SELECT COUNT(*) FROM {table}"
        if where:
            sql += f" WHERE {where}"
        cur.execute(sql)
        return cur.fetchone()[0]

    def n_sep(where=""):
        return n_table("siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep", where)

    def add(cond, prev=""):
        return (prev + " AND " + cond) if prev else cond

    print("Querying filter-stage counts (this may take a minute)...")

    c = {}
    c["mer"]          = n_table("euclid_q1.mer_final_cat")
    print(f"  MER full catalog:        {c['mer']:>10,}")

    c["mer_phz"]      = n_table("siwei_zhao_2025.crossmatch_q1_mer_phz")
    print(f"  After × PHZ:             {c['mer_phz']:>10,}")

    c["vis_phz"]      = n_table("siwei_zhao_2025.crossmatch_q1_mer_phz_vis")
    print(f"  After vis_det=1 (PHZ):   {c['vis_phz']:>10,}")

    c["total"]        = n_table("siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia")
    print(f"  After × Gaia:            {c['total']:>10,}")

    c["sep"]          = n_table("siwei_zhao_2025.crossmatch_q1_mer_phz_vis_gaia_sep")
    print(f"  After sep ≤ 0.3\":        {c['sep']:>10,}")

    c["phz_notnull"]  = n_sep("p_phz_classification IS NOT NULL")
    print(f"  After PHZ not null:      {c['phz_notnull']:>10,}")

    c["phz_12"]       = n_sep("p_phz_classification IN (1, 2)")
    print(f"  After PHZ in {{1,2}}:      {c['phz_12']:>10,}")

    # breakdown of dropped PHZ classes
    c["phz_star"]  = n_sep("p_phz_classification = 1")
    c["phz_gal"]   = n_sep("p_phz_classification = 2")

    w_gmag = "p_phz_classification IN (1, 2) AND g_phot_g_mean_mag >= 17"
    c["gmag"]         = n_sep(w_gmag)
    print(f"  After G ≥ 17:            {c['gmag']:>10,}")

    c["gmag_star"] = n_sep("p_phz_classification = 1 AND g_phot_g_mean_mag >= 17")
    c["gmag_gal"]  = n_sep("p_phz_classification = 2 AND g_phot_g_mean_mag >= 17")

    cur.close()
    conn.close()
    return c


def _scale(v):
    """Log2-compress values for display so large upstream drops don't dominate."""
    import math
    return max(1, int(math.log2(v + 1) * 120))


def build_sankey(c):
    import plotly.graph_objects as go

    PASS_COLOR = "#4a86c8"
    STAR_COLOR = "#e07b54"
    GAL_COLOR  = "#5b9bd5"
    DROP_COLOR = "#bbbbbb"
    DROP_LINK  = "rgba(180,180,180,0.35)"
    PASS_LINK  = "rgba(74,134,200,0.42)"
    STAR_LINK  = "rgba(224,123,84,0.45)"
    GAL_LINK   = "rgba(91,155,213,0.45)"

    # ── Node indices ──────────────────────────────────────────────────────────
    # 0–6  main pipeline (no intermediate S/G split)
    # 7–8  terminal: Stars, Galaxies
    # 9–14 drop buckets (ordered by pipeline stage)

    labels = [
        # 0–6 pipeline (vis_det now before × Gaia)
        f"Q1 MER<br>{c['mer']:,}",
        f"× PHZ<br>{c['mer_phz']:,}",
        f"VIS detected<br>{c['vis_phz']:,}",
        f"× Gaia<br>{c['total']:,}",
        f"Sep ≤ 0.3\"<br>{c['sep']:,}",
        f"PHZ {{1,2}}<br>{c['phz_12']:,}",
        f"Training set<br>G ≥ 17<br>{c['gmag']:,}",
        # 7–8 terminal
        f"Stars<br>{c['gmag_star']:,}",
        f"Galaxies<br>{c['gmag_gal']:,}",
        # 9–14 drops (in pipeline order)
        f"No PHZ<br>−{c['mer']-c['mer_phz']:,}",
        f"No VIS det<br>−{c['mer_phz']-c['vis_phz']:,}",
        f"No Gaia<br>−{c['vis_phz']-c['total']:,}",
        f"Sep > 0.3\"<br>−{c['total']-c['sep']:,}",
        f"Other PHZ<br>−{c['sep']-c['phz_12']:,}",
        f"G < 17<br>−{c['phz_12']-c['gmag']:,}",
    ]

    node_colors = [
        PASS_COLOR, PASS_COLOR, PASS_COLOR, PASS_COLOR,
        PASS_COLOR, PASS_COLOR, PASS_COLOR,
        STAR_COLOR, GAL_COLOR,
        DROP_COLOR, DROP_COLOR, DROP_COLOR,
        DROP_COLOR, DROP_COLOR, DROP_COLOR,
    ]

    # ── Links ─────────────────────────────────────────────────────────────────
    links = [
        # pass-through (vis_det before × Gaia)
        (0, 1, _scale(c["mer_phz"]),   PASS_LINK),
        (1, 2, _scale(c["vis_phz"]),   PASS_LINK),
        (2, 3, _scale(c["total"]),     PASS_LINK),
        (3, 4, _scale(c["sep"]),       PASS_LINK),
        (4, 5, _scale(c["phz_12"]),    PASS_LINK),
        (5, 6, _scale(c["gmag"]),      PASS_LINK),
        # training set → Stars / Galaxies (proportional split)
        (6, 7, int(_scale(c["gmag"]) * c["gmag_star"] / c["gmag"]), STAR_LINK),
        (6, 8, int(_scale(c["gmag"]) * c["gmag_gal"]  / c["gmag"]), GAL_LINK),
        # drops
        (0, 9,  _scale(c["mer"]      - c["mer_phz"]),  DROP_LINK),
        (1, 10, _scale(c["mer_phz"]  - c["vis_phz"]),  DROP_LINK),
        (2, 11, _scale(c["vis_phz"]  - c["total"]),    DROP_LINK),
        (3, 12, _scale(c["total"]    - c["sep"]),      DROP_LINK),
        (4, 13, _scale(c["sep"]      - c["phz_12"]),   DROP_LINK),
        (5, 14, _scale(c["phz_12"]   - c["gmag"]),     DROP_LINK),
    ]

    src, tgt, val, col = zip(*links)

    # Initial node positions matching desired layout (pipeline bottom-left → top-right,
    # drops stacked on far right). snap arrangement keeps these draggable.
    # x,y in [0,1]; y=1 is top.
    node_x = [
        # 0–6 main pipeline
        0.02, 0.14, 0.27, 0.40, 0.51, 0.63, 0.76,
        # 7–8 terminals
        0.98, 0.98,
        # 9–14 drops (far right, stacked top→bottom in pipeline order)
        0.98, 0.98, 0.98, 0.98, 0.98, 0.98,
    ]
    node_y = [
        # 0–6 main pipeline (y=0 top, y=1 bottom; pipeline rises left→right so y decreases)
        0.90, 0.82, 0.70, 0.57, 0.47, 0.37, 0.24,
        # 7–8 terminals: Stars at very top, Galaxies just below
        0.05, 0.17,
        # 9–14 drops (No PHZ at bottom → G<17 nearest training set at top)
        0.90, 0.78, 0.67, 0.57, 0.44, 0.33,
    ]

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
        #         "Q1 Training Data Filter Pipeline<br>"
        #         "<sup>Euclid Q1 × Gaia DR3 — source counts at each selection stage  |  band widths are log-scaled</sup>"
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
                "filename": "q1_filter_sankey",
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
