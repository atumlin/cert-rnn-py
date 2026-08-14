"""2-D zonotope geometry: vertex enumeration and polygon helpers.

Promoted from slides/make_reach_visuals.py with degenerate-case handling and
property tests (tests/soundness/test_vertex_enum.py, soundness.md T-2.3).
Used by Phase 0 measurement scripts; will back the ZRLT Tier 1 candidate set.
"""

from __future__ import annotations

import numpy as np


def zono2d_vertices(c, G, angle_tol: float = 1e-9) -> np.ndarray:
    """Vertices of the 2-D zonotope {c + G.T @ alpha : alpha in [-1,1]^p}.

    c: (2,) center. G: (p, 2) generator rows. Returns (m, 2) vertices in
    counter-clockwise order, m <= 2p.

    Construction: drop zero generators; canonicalize each generator into
    the upper half-plane (angle in [0, pi)) — sign flips leave the zonotope
    unchanged; merge (exactly sum) generators whose angles agree within
    angle_tol — exact for parallel generators; sort by angle; walk from the
    bottom vertex c - sum(g) adding 2*g per step (traces the right boundary
    upward, CCW), then mirror the chain through c for the left boundary.

    Degenerate cases: p = 0 or all-zero -> 1 vertex (the point c); all
    generators parallel -> 2 vertices (a segment).
    """
    c = np.asarray(c, dtype=np.float64).reshape(2)
    G = np.asarray(G, dtype=np.float64).reshape(-1, 2)
    if G.shape[0] > 0:
        G = G[np.hypot(G[:, 0], G[:, 1]) > 0.0]
    if G.shape[0] == 0:
        return c[None, :].copy()

    flip = (G[:, 1] < 0) | ((G[:, 1] == 0) & (G[:, 0] < 0))
    G = np.where(flip[:, None], -G, G)
    ang = np.arctan2(G[:, 1], G[:, 0])  # in [0, pi)
    order = np.argsort(ang, kind="stable")
    G = G[order]
    ang = ang[order]

    merged = []
    start = 0
    for i in range(1, G.shape[0] + 1):
        if i == G.shape[0] or ang[i] - ang[start] > angle_tol:
            merged.append(G[start:i].sum(axis=0))
            start = i
    Gm = np.asarray(merged)

    v0 = c - Gm.sum(axis=0)
    chain = v0[None, :] + 2.0 * np.vstack(
        [np.zeros((1, 2)), np.cumsum(Gm, axis=0)]
    )  # (m+1, 2): bottom vertex ... top vertex c + sum(g)
    if chain.shape[0] > 2:
        # Continuing CCW from the top vertex, the left boundary subtracts
        # the generators in the SAME ascending-angle order, so its
        # vertices are the mirrors of the interior chain in FORWARD order:
        # 2c - chain[j] = (c + sum g) - 2 * sum_{i<=j} g_i.
        other = 2.0 * c - chain[1:-1]
        return np.vstack([chain, other])
    return chain  # segment: 2 vertices


def polygon_area(verts: np.ndarray) -> float:
    """Shoelace area of a CCW polygon, (m, 2)."""
    if verts.shape[0] < 3:
        return 0.0
    x, y = verts[:, 0], verts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def points_in_convex_polygon(pts: np.ndarray, verts: np.ndarray,
                             tol: float = 1e-9) -> np.ndarray:
    """Boolean mask: which pts (n, 2) lie inside the CCW convex polygon
    (m, 2), with `tol` slack scaled by edge length."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    m = verts.shape[0]
    if m == 1:
        return (np.abs(pts - verts[0]).max(axis=1) <= tol)
    if m == 2:
        # Degenerate polygon = segment. The generic edge loop below only
        # constrains perpendicular distance (both "edges" are the same
        # line), so points beyond the endpoints would pass. Constrain the
        # along-segment projection too.
        a, b = verts[0], verts[1]
        e = b - a
        L2 = float(e @ e)
        if L2 == 0.0:
            return (np.abs(pts - a).max(axis=1) <= tol)
        rel = pts - a
        perp = np.abs(e[0] * rel[:, 1] - e[1] * rel[:, 0]) / np.sqrt(L2)
        s = (rel @ e) / L2
        return (perp <= tol) & (s >= -tol) & (s <= 1 + tol)
    inside = np.ones(pts.shape[0], dtype=bool)
    span = float(np.max(np.abs(verts))) + 1.0
    for i in range(m):
        a = verts[i]
        b = verts[(i + 1) % m]
        e = b - a
        elen = float(np.hypot(*e))
        if elen == 0.0:
            continue
        # cross = elen * signed_distance; the slack must be a DISTANCE
        # (scaled by edge length), never an absolute cross-product bound —
        # for near-zero cap edges of sliver polygons an absolute bound is
        # vacuous and admits points arbitrarily far past the caps.
        cross = e[0] * (pts[:, 1] - a[1]) - e[1] * (pts[:, 0] - a[0])
        inside &= cross >= -tol * span * elen
    return inside
