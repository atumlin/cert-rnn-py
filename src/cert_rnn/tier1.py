"""ZRLT Tier 1: bilinear residual offsets over the joint zonotope.

Cert-RNN's bilinear transformers compute (C1, C2) = exact min/max of the
residual g = f - Ax - By over the operand BOX. The two operands share
noise symbols, so their joint reachable set is a 2-D zonotope Z strictly
inside the box; the box corners where the extrema of g typically sit are
usually unreachable (proposal §2.2, §4).

Construction used here (see docs/phase0_findings.md §H for why not the
literal vertex+edge-stationary swap): cover Z with axis-aligned grid cells
and evaluate the EXISTING closed-form box enumeration on every admitted
cell with the SAME (A, B):

    C1 = min over cells of C1_box(cell),   C2 = max over cells of C2_box(cell)

Soundness: the admitted cells' union contains Z (cell/zonotope
intersection is decided by the separating-axis theorem, exact for two
convex polygons, with an outward tolerance), and every reachable (x, y)
lies in Z, so g at every reachable point lies within some cell's exact
[C1, C2]. Tightness: the union is inside the box, so C1 >= C1_box and
C2 <= C2_box always (T-2.2 is a theorem, not a hope). Nothing new is
enumerated: the per-cell extrema come from the same code the R-tests
audit, so F-1/F-2/F-3 are inherited unchanged and no new candidate
machinery is introduced (ledger row S16).

Degenerate operands (a point in x or y, or no shared structure) fall back
to the box result.
"""

from __future__ import annotations

import numpy as np

# Grid resolution per axis. Cells intersecting Z are kept; for a thin
# diagonal Z that is O(n) cells, for a fat Z up to n^2 (where the box was
# nearly exact anyway). Chosen from the recovery-vs-n measurement in
# research/phase1_tier1_recovery.py.
GRID_N = 16
# Outward slack on the SAT test, relative to the cell/zonotope scale.
SAT_TOL = 1e-12
# Guard 1 (pre-construction): skip coordinates whose joint zonotope already
# fills this fraction of its bounding box (closed-form area ratio, O(p log p),
# computed BEFORE any cell is built): the box is nearly exact there and the
# grid cover would recover little. Falling back to the box result is sound;
# this only trades tightness for time.
HEADROOM_SKIP = 0.9
# Guard 2 (post-admission fraction) is DISABLED (1.0): measured on ieee9-S
# it cost exactly one search granule (1.2e-4) of certified radius at 0.9.
# Mechanism: with the tilt fixed, the min/max over ALL cells equals the box
# bound (R-1 equality), so Tier 1's entire recovery comes from the EXCLUDED
# cells — which are exactly the box-corner cells where the residual extrema
# sit. Admission fraction is therefore a poor proxy for "nothing to
# recover"; even 5% exclusion can carry most of the gain. The area-ratio
# guard above covers the provably-nothing case at zero tightness cost.
ADMIT_SKIP = 1.0
# SAT separating directions are deduplicated by angle within this tolerance.
# Dropping a near-duplicate direction can only ADMIT more cells (sound); the
# support sums along the kept directions run over ALL original generators,
# so no outward slack is involved. Collapses ~1000 projected generators to
# a few dozen distinct directions in the multi-frame regime.
DIR_TOL = 1e-3
# Optional instrumentation: when a dict is assigned here, admitted_cells
# accumulates {"dirs_raw": .., "dirs_kept": .., "cells_admitted": ..,
# "cells_total": .., "skip_area": .., "skip_admit": .., "coords": ..}.
STATS = None


def _area_ratio(gx, gy, wx, wy):
    """area(Z) / (wx * wy) in closed form (see instrument.zono2d_area)."""
    flip = (gy < 0) | ((gy == 0) & (gx < 0))
    a = np.where(flip, -gx, gx); b = np.where(flip, -gy, gy)
    order = np.argsort(np.arctan2(b, a))
    a = a[order]; b = b[order]
    Sa = np.cumsum(a) - a; Sb = np.cumsum(b) - b
    return 4.0 * float(np.sum(Sa * b - Sb * a)) / (wx * wy)


def admitted_cells(cx, cy, gx, gy, lx, ux, ly, uy, n=GRID_N):
    """Axis-aligned n x n grid over the box; return the cells whose
    closed rectangle intersects the 2-D zonotope Z = (cx,cy) + sum g_i a_i.

    Returns (clx, cux, cly, cuy) arrays of the admitted cells (m,), or
    None if the box is degenerate (zero width in x or y).
    """
    wx = ux - lx
    wy = uy - ly
    if not (wx > 0 and wy > 0):
        return None
    G = np.stack([gx, gy], axis=1)
    nrm = np.hypot(gx, gy)
    G = G[nrm > 0]
    if G.shape[0] == 0:
        return None
    # SAT directions: rectangle normals e1, e2 are satisfied by construction
    # for grid cells inside the box; test the zonotope's edge normals,
    # i.e. the perpendiculars of its generators (deduplicated by angle).
    perp = np.stack([-G[:, 1], G[:, 0]], axis=1)
    perp /= np.linalg.norm(perp, axis=1, keepdims=True)
    # canonicalize to angle in [0, pi) and drop near-duplicates within
    # DIR_TOL. Dropping a separating direction can only admit MORE cells
    # (sound); the supports along kept directions still sum over ALL
    # original generators (exact), so no outward rounding is needed here.
    flip = (perp[:, 1] < 0) | ((perp[:, 1] == 0) & (perp[:, 0] < 0))
    perp = np.where(flip[:, None], -perp, perp)
    ang = np.arctan2(perp[:, 1], perp[:, 0])
    order = np.argsort(ang)
    perp = perp[order]; ang = ang[order]
    keep = np.ones(perp.shape[0], dtype=bool)
    last = ang[0]
    for i in range(1, ang.shape[0]):
        if ang[i] - last > DIR_TOL:
            last = ang[i]
        else:
            keep[i] = False
    D = perp[keep]                                   # (m, 2)
    if STATS is not None:
        STATS["dirs_raw"] = STATS.get("dirs_raw", 0) + int(perp.shape[0])
        STATS["dirs_kept"] = STATS.get("dirs_kept", 0) + int(D.shape[0])
    # Z's extent along each direction: center projection +- support
    zc = D @ np.array([cx, cy])                       # (m,)
    zr = np.abs(D @ G.T).sum(axis=1)                  # (m,)
    # grid cells
    xe = lx + (np.arange(n + 1) / n) * wx
    ye = ly + (np.arange(n + 1) / n) * wy
    xe[-1] = ux; ye[-1] = uy
    ccx = 0.5 * (xe[:-1] + xe[1:]); ccy = 0.5 * (ye[:-1] + ye[1:])
    hwx = 0.5 * wx / n; hwy = 0.5 * wy / n
    CX, CY = np.meshgrid(ccx, ccy, indexing="ij")     # (n, n)
    cc = np.stack([CX.ravel(), CY.ravel()], axis=1)   # (n^2, 2)
    proj = cc @ D.T                                   # (n^2, m)
    crad = np.abs(D[:, 0]) * hwx + np.abs(D[:, 1]) * hwy   # (m,)
    tol = SAT_TOL * max(1.0, abs(cx) + abs(cy) + wx + wy)
    # intervals [proj - crad, proj + crad] and [zc - zr, zc + zr] overlap?
    sep = (proj + crad[None, :] < zc[None, :] - zr[None, :] - tol) | \
          (proj - crad[None, :] > zc[None, :] + zr[None, :] + tol)
    admitted = ~sep.any(axis=1)                       # (n^2,)
    idx = np.flatnonzero(admitted)
    if STATS is not None:
        STATS["cells_admitted"] = STATS.get("cells_admitted", 0) + int(idx.size)
        STATS["cells_total"] = STATS.get("cells_total", 0) + int(n * n)
    if idx.size == 0:
        # cannot happen for a valid Z (its center is in the box), but be
        # safe: fall back to the whole box
        return None
    if idx.size >= ADMIT_SKIP * n * n:
        # Guard 2: Z nearly fills the box; per-cell enumeration on ~n^2
        # cells would recover almost nothing. Box fallback (sound).
        if STATS is not None:
            STATS["skip_admit"] = STATS.get("skip_admit", 0) + 1
        return None
    ix = idx // n; iy = idx % n
    return xe[ix], xe[ix + 1], ye[iy], ye[iy + 1]


def c1c2_over_zono(kind, A, B, cx, cy, Vx, Vy, lx, ux, ly, uy,
                   C1_box, C2_box, n=GRID_N):
    """Tier 1 (C1, C2) for K coordinates.

    kind: "sigtanh" | "sigid". A, B, cx, cy, lx, ux, ly, uy, C1_box,
    C2_box: (K,). Vx, Vy: (K, P) aligned generator rows. Coordinates with
    degenerate boxes or a zero-width box result keep the box values.
    Returns (C1, C2) each (K,), with C1 >= C1_box, C2 <= C2_box.
    """
    from cert_rnn.transformers import _c1c2_sigid_batch, _c1c2_sigtanh_batch

    K = A.shape[0]
    C1 = C1_box.copy()
    C2 = C2_box.copy()
    all_cells = []
    owners = []
    wx = ux - lx; wy = uy - ly
    for k in range(K):
        if STATS is not None:
            STATS["coords"] = STATS.get("coords", 0) + 1
        if C2_box[k] - C1_box[k] <= 0.0 or not (wx[k] > 0 and wy[k] > 0):
            continue
        if _area_ratio(Vx[k], Vy[k], wx[k], wy[k]) > HEADROOM_SKIP:
            if STATS is not None:
                STATS["skip_area"] = STATS.get("skip_area", 0) + 1
            continue
        cells = admitted_cells(cx[k], cy[k], Vx[k], Vy[k],
                               lx[k], ux[k], ly[k], uy[k], n)
        if cells is None:
            continue
        m = cells[0].shape[0]
        if m >= n * n:
            continue   # every cell admitted: identical to the box result
        all_cells.append(cells)
        owners.append(np.full(m, k))
    if not all_cells:
        return C1, C2
    clx = np.concatenate([c[0] for c in all_cells])
    cux = np.concatenate([c[1] for c in all_cells])
    cly = np.concatenate([c[2] for c in all_cells])
    cuy = np.concatenate([c[3] for c in all_cells])
    own = np.concatenate(owners)
    fn = _c1c2_sigtanh_batch if kind == "sigtanh" else _c1c2_sigid_batch
    c1, c2 = fn(A[own], B[own], clx, cux, cly, cuy)
    # per-owner min / max
    c1_min = np.full(K, np.inf); c2_max = np.full(K, -np.inf)
    np.minimum.at(c1_min, own, c1)
    np.maximum.at(c2_max, own, c2)
    touched = np.isfinite(c1_min)
    # Raw cell extrema, deliberately NOT clamped to the box values: the
    # ordering C1 >= C1_box, C2 <= C2_box holds by construction, and
    # tests/soundness/test_tier1.py asserts it so that a violation beyond
    # fp surfaces as a finding instead of being clamped away.
    C1[touched] = c1_min[touched]
    C2[touched] = c2_max[touched]
    return C1, C2
