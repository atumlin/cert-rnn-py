"""T-2.3: 2-D zonotope vertex enumeration correctness (soundness.md).

Checks, for random generator sets including degenerate ones:
  (a) every enumerated vertex lies in Z, AND the polygon reaches the
      support of Z in every tested direction (two-sided: no vertex outside,
      no vertex missing);
  (b) the polygon is convex (CCW) and centrally symmetric about the center;
  (c) shoelace area equals the independent closed-form zonotope area
      (instrument.zono2d_area, itself validated against a naive double sum
      and an exact box case), and sampled eps-space points fall inside;
  (d) vertex count <= 2p.
"""

from __future__ import annotations

import numpy as np
import pytest

from cert_rnn.geometry import (
    points_in_convex_polygon,
    polygon_area,
    zono2d_vertices,
)
from cert_rnn.instrument import zono2d_area

SEED = 20260814


def _support(G, d):
    return float(np.sum(np.abs(G @ d)))


def _random_gen_sets(rng):
    for p in (0, 1, 2, 3, 8, 50):
        for rep in range(10):
            G = rng.normal(0, 1, (p, 2))
            if p >= 3 and rep % 3 == 0:
                G[1] = -1.7 * G[0]          # antiparallel duplicate direction
                G[2] = 0.0                   # zero generator
            if p >= 2 and rep % 4 == 0:
                G[:] = G[0] * rng.uniform(-2, 2, (p, 1))  # all parallel
            yield p, G


@pytest.mark.soundness
def test_vertex_enumeration_properties():
    print(f"\n[seed={SEED}] vertex enumeration T-2.3")
    rng = np.random.default_rng(SEED)
    for p, G in _random_gen_sets(rng):
        c = rng.normal(0, 2, 2)
        verts = zono2d_vertices(c, G)
        m = verts.shape[0]

        # (d) count
        assert m <= max(1, 2 * p), (p, m)

        # (a) two-sided support check: max_v <d, v-c> must equal the
        # zonotope support sum_i |<d, g_i>| in every direction.
        dirs = rng.normal(0, 1, (64, 2))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        if p > 0:
            nrm = np.hypot(G[:, 0], G[:, 1])
            nz = G[nrm > 0]
            if nz.shape[0]:
                edge_normals = np.stack([-nz[:, 1], nz[:, 0]], axis=1)
                edge_normals /= np.linalg.norm(edge_normals, axis=1,
                                               keepdims=True)
                dirs = np.vstack([dirs, edge_normals])
        for d in dirs:
            sup = _support(G, d)
            reach = float(np.max((verts - c) @ d))
            scale = max(1.0, sup)
            assert reach <= sup + 1e-9 * scale, "vertex outside Z"
            assert reach >= sup - 1e-9 * scale, "missing vertex (support not attained)"

        # (b) convex CCW + central symmetry
        if m >= 3:
            e = np.roll(verts, -1, axis=0) - verts
            cross = e[:, 0] * np.roll(e, -1, axis=0)[:, 1] \
                - e[:, 1] * np.roll(e, -1, axis=0)[:, 0]
            assert np.all(cross >= -1e-9 * np.max(np.abs(verts) + 1)), \
                "non-convex or wrong orientation"
        mirrored = 2.0 * c - verts
        a_sorted = verts[np.lexsort(verts.T)]
        b_sorted = mirrored[np.lexsort(mirrored.T)]
        np.testing.assert_allclose(a_sorted, b_sorted, atol=1e-9)

        # (c) area agreement + eps-sample containment
        area_poly = polygon_area(verts)
        area_closed = zono2d_area(G[:, 0], G[:, 1]) if p > 0 else 0.0
        assert area_poly == pytest.approx(area_closed, rel=1e-9, abs=1e-12)
        if p > 0:
            alphas = rng.uniform(-1, 1, (200, p))
            corners = rng.choice([-1.0, 1.0], size=(64, p))
            pts = np.vstack([alphas, corners]) @ G + c
            assert points_in_convex_polygon(pts, verts, tol=1e-9).all()


@pytest.mark.soundness
def test_segment_polygon_rejects_points_beyond_endpoints():
    """Regression: for a degenerate (2-vertex) polygon, points ON the
    carrier line but BEYOND the segment's ends must be rejected. The
    generic edge test only constrains perpendicular distance, which
    silently inflated Measurement 1's gap_Z on perfectly-correlated
    operand pairs."""
    verts = np.array([[0.0, 0.0], [2.0, 2.0]])
    pts = np.array([
        [1.0, 1.0],      # on segment -> inside
        [0.0, 0.0],      # endpoint -> inside
        [3.0, 3.0],      # on line, beyond end -> OUTSIDE
        [-1.0, -1.0],    # on line, before start -> OUTSIDE
        [1.0, 1.5],      # off line -> OUTSIDE
    ])
    got = points_in_convex_polygon(pts, verts, tol=1e-9)
    assert got.tolist() == [True, True, False, False, False]
