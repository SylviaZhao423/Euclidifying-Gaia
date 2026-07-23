#!/usr/bin/env python3
"""
Inter-cluster "bridges" / streams on the Q1 14-feature UMAP — the single source of
truth shared by every script that draws or samples them:
    analyze_umap_qso.py            (umap_qso_overlay_*.png streams panel)
    analyze_umap_nsa.py            (umap_nsa_overlay_*.png streams panel)
    download_ds_g_bridges_cutouts.py  (per-bridge cutout sheets)

To adjust a bridge, edit STREAM_PAIRS here and every consumer updates in lock-step.

A point is "outside" the dense cluster cores when the distance to its k-th nearest Q1
reference point exceeds OUTSIDE_KNN_THR (sparse = a bridge / outskirt). Each outside
point is then relabelled to the nearest named polyline corridor in STREAM_PAIRS, or
-1 "other" if farther than STREAM_CAP from all of them.
"""
from __future__ import annotations

import numpy as np

# "Outside" points: those that land where the Q1 REFERENCE embedding is SPARSE — i.e.
# off the dense cluster cores, in the inter-cluster bridges / streams. Detected purely
# from the reference geometry (no hand-drawn corridors): the distance to the k-th
# nearest Q1 point is tiny on the dense manifold and jumps in the sparse bridges, so
# points above a small threshold are "not in the background Q1 UMAP".
OUTSIDE_KNN_K = 8
OUTSIDE_KNN_THR = 0.4          # UMAP units; kth-NN dist above this => outside a core


def outside_mask(pts: np.ndarray, ref_emb: np.ndarray) -> np.ndarray:
    """True where a point sits in a sparse region of the Q1 reference (a stream/outskirt),
    measured by distance to its k-th nearest Q1 reference point."""
    from scipy.spatial import cKDTree
    dk, _ = cKDTree(ref_emb).query(pts, k=OUTSIDE_KNN_K)
    return dk[:, -1] > OUTSIDE_KNN_THR


# Named inter-cluster streams (bridges). Each OUTSIDE point is relabelled to the nearest
# of these polyline corridors (traced along the visible density ridge), or -1 "other" if
# farther than STREAM_CAP from all. Each entry: (name, vertices, colour); a vertex is an
# int (that cluster's centroid) or an explicit (x, y) waypoint.
STREAM_PAIRS = [
    ("C0–C1", [0, 1],                                        "#f032e6"),  # magenta: C0→C1 diagonal
    ("C0–C2", [0, (2.0, 2.8), 2],                           "#469990"),  # teal: C0→C2 (lower the waypoint y to drop the band)
    ("C1–C3", [1, (4.0, 13.4), (6.5, 13.9), (9.4, 14.0), 3], "#00e5ff"),  # cyan: arc over the top into C3
    ("C2–C3", [2, (5.9, 10.8), (7.3, 11.2), 3],              "#ffe119"),   # gold: plume rising out of C2
    ("C2–C4", [2, (6.8, 3.4), (7.0, 1.0), 4],              "#00b050"),    # green: bridge down to C4
]
STREAM_CAP = 2.0   # UMAP units; outside point farther than this from every corridor => "other"


def _poly_dist(op: np.ndarray, verts: list) -> np.ndarray:
    """Perpendicular distance from each point to a polyline (min over its segments)."""
    best = np.full(len(op), np.inf)
    for i in range(len(verts) - 1):
        a, b = verts[i], verts[i + 1]
        v = b - a
        L2 = v @ v
        t = np.clip(((op - a) @ v) / (L2 if L2 else 1.0), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(op - (a + np.outer(t, v)), axis=1))
    return best


def stream_labels(pts: np.ndarray, outside: np.ndarray, centroids: dict) -> np.ndarray:
    """Relabel OUTSIDE points into named streams: index into STREAM_PAIRS for the nearest
    polyline corridor, -1 = "other" (too far), -2 = non-outside (core) point."""
    lab = np.full(len(pts), -2, dtype=int)
    op = pts[outside]
    if not len(op):
        return lab

    def _resolve(verts):
        return [np.asarray(centroids[v], float) if isinstance(v, (int, np.integer))
                else np.asarray(v, float) for v in verts]

    D = np.stack([_poly_dist(op, _resolve(verts)) for _n, verts, _c in STREAM_PAIRS], axis=1)
    asg = np.argmin(D, axis=1)
    mind = D[np.arange(len(op)), asg]
    lab[outside] = np.where(mind <= STREAM_CAP, asg, -1)
    return lab
