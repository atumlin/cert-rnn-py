"""Cert-RNN sound zonotope abstract transformers.

tanh, sigmoid, sigma(x)*tanh(y), x*sigma(y), per Du et al., CCS 2021
(§4.2.2, Theorems 4.2/4.3, Appendices A-C). Ported from
code/nnv/engine/nn/cert_rnn/CertRNN.m in the sibling MATLAB repo.

Plane fit (A, B) is corner-fit, applied uniformly across sign regions
(empirically beats the 7-candidate sweep on LSTM workloads, which
includes the per-case Table 8/9 formulas). The (C1, C2) error spread
is the EXACT min/max of the residual g(x, y) = f(x, y) - A x - B y
over the input box, found by enumerating corners, edge stationary
points, and (for sigma*tanh) interior critical points via the quartic
    p^4 - (2+B) p^3 + (1+2B) p^2 - B p - A^2 = 0,   p = sigma(x)
solved with numpy.roots. Table 9's printed sigmoid-identity formulas
contain transcription errors; the sigid plane here is derived from
the Appendix C.1/C.2 proofs directly.

Each transformer returns a Zono with K new fresh predicates (one per
output dimension) allocated via the module-level PredAllocator.
"""

from __future__ import annotations

import numpy as np

from cert_rnn.zono import (
    PredAllocator,
    Zono,
    align_pred_space,
    get_default_allocator,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------- unary transformers ----------


def tanh_zono(z: Zono, allocator: PredAllocator | None = None) -> Zono:
    """Sound elementwise tanh transformer (§4.2.2, parallel-tangent planes)."""
    alloc = allocator if allocator is not None else get_default_allocator()
    K = z.dim
    lb, ub = z.get_ranges()
    a = np.zeros(K)
    C1 = np.zeros(K)
    C2 = np.zeros(K)

    A_FLOOR = 1e-12
    LEN_FLOOR = 1e-12

    for k in range(K):
        lk, uk = lb[k], ub[k]
        if uk - lk < LEN_FLOOR:
            ak = 1.0 - np.tanh(lk) ** 2
            a[k] = ak
            Ck = np.tanh(lk) - ak * lk
            C1[k] = Ck
            C2[k] = Ck
            continue
        ak = (np.tanh(uk) - np.tanh(lk)) / (uk - lk)
        a[k] = ak
        if ak < A_FLOOR or ak >= 1.0 - A_FLOOR:
            x_star, x_star2 = lk, uk
        else:
            s = np.sqrt(1.0 - ak)
            x_prime = -np.arctanh(s)
            x_prime2 = np.arctanh(s)
            x_star = max(x_prime, lk)
            x_star2 = min(x_prime2, uk)
        C1[k] = np.tanh(x_star) - ak * x_star
        C2[k] = np.tanh(x_star2) - ak * x_star2

    new_c = a * z.c + 0.5 * (C1 + C2)
    scaled_V = a[:, None] * z.V if z.n_pred > 0 else np.zeros((K, 0))
    fresh_V = np.diag(0.5 * (C2 - C1))
    new_V = np.hstack([scaled_V, fresh_V])
    fresh_ids = alloc.next_n(K)
    return Zono(new_c, new_V, z.pred_ids + fresh_ids)


def sigmoid_zono(z: Zono, allocator: PredAllocator | None = None) -> Zono:
    """Sound elementwise sigmoid transformer (§4.2.2)."""
    alloc = allocator if allocator is not None else get_default_allocator()
    K = z.dim
    lb, ub = z.get_ranges()
    a = np.zeros(K)
    C1 = np.zeros(K)
    C2 = np.zeros(K)

    A_FLOOR = 1e-15
    A_CEIL = 0.25 - 1e-15
    LEN_FLOOR = 1e-12

    for k in range(K):
        lk, uk = lb[k], ub[k]
        if uk - lk < LEN_FLOOR:
            sk = _sigmoid(lk)
            ak = sk * (1.0 - sk)
            a[k] = ak
            Ck = sk - ak * lk
            C1[k] = Ck
            C2[k] = Ck
            continue
        ak = (_sigmoid(uk) - _sigmoid(lk)) / (uk - lk)
        a[k] = ak
        if ak < A_FLOOR or ak >= A_CEIL:
            x_star, x_star2 = lk, uk
        else:
            s = np.sqrt(1.0 - 4.0 * ak)
            x_prime = -2.0 * np.arctanh(s)
            x_prime2 = 2.0 * np.arctanh(s)
            x_star = max(x_prime, lk)
            x_star2 = min(x_prime2, uk)
        C1[k] = _sigmoid(x_star) - ak * x_star
        C2[k] = _sigmoid(x_star2) - ak * x_star2

    new_c = a * z.c + 0.5 * (C1 + C2)
    scaled_V = a[:, None] * z.V if z.n_pred > 0 else np.zeros((K, 0))
    fresh_V = np.diag(0.5 * (C2 - C1))
    new_V = np.hstack([scaled_V, fresh_V])
    fresh_ids = alloc.next_n(K)
    return Zono(new_c, new_V, z.pred_ids + fresh_ids)


# ---------- 1D plane helpers ----------


def _sigmoid_plane_1d(lx, ux):
    sl, su = _sigmoid(lx), _sigmoid(ux)
    if ux - lx < 1e-12:
        a = sl * (1 - sl)
        C = sl - a * lx
        return a, C, C
    a = (su - sl) / (ux - lx)
    if a < 1e-15 or a >= 0.25 - 1e-15:
        x_star, x_star2 = lx, ux
    else:
        s = np.sqrt(1 - 4 * a)
        x_prime = -2 * np.arctanh(s)
        x_prime2 = 2 * np.arctanh(s)
        x_star = max(x_prime, lx)
        x_star2 = min(x_prime2, ux)
    C1 = _sigmoid(x_star) - a * x_star
    C2 = _sigmoid(x_star2) - a * x_star2
    return a, C1, C2


def _tanh_plane_1d(ly, uy):
    if uy - ly < 1e-12:
        a = 1 - np.tanh(ly) ** 2
        C = np.tanh(ly) - a * ly
        return a, C, C
    a = (np.tanh(uy) - np.tanh(ly)) / (uy - ly)
    if a < 1e-15 or a >= 1 - 1e-15:
        y_star, y_star2 = ly, uy
    else:
        s = np.sqrt(1 - a)
        y_prime = -np.arctanh(s)
        y_prime2 = np.arctanh(s)
        y_star = max(y_prime, ly)
        y_star2 = min(y_prime2, uy)
    C1 = np.tanh(y_star) - a * y_star
    C2 = np.tanh(y_star2) - a * y_star2
    return a, C1, C2


# ---------- bilinear: sigma(x) * tanh(y) ----------


def _c1c2_sigtanh(A, B, lx, ux, ly, uy):
    """Exact min/max of g(x, y) = sigma(x) tanh(y) - A x - B y over the box."""

    def g(x, y):
        return _sigmoid(x) * np.tanh(y) - A * x - B * y

    cands = [g(lx, ly), g(lx, uy), g(ux, ly), g(ux, uy)]

    # vertical edges: sigma(x_e) tanh'(y) = B  =>  tanh(y)^2 = 1 - B/sigma(x_e)
    for x_e in (lx, ux):
        s_xe = _sigmoid(x_e)
        if s_xe > 1e-15:
            ratio = B / s_xe
            if 1e-12 < ratio < 1 - 1e-12:
                t = np.sqrt(1 - ratio)
                for tv in (-t, t):
                    if abs(tv) < 1:
                        y_crit = np.arctanh(tv)
                        if ly <= y_crit <= uy:
                            cands.append(g(x_e, y_crit))

    # horizontal edges: sigma'(x) tanh(y_e) = A  =>  p^2 - p + A/tanh(y_e) = 0
    for y_e in (ly, uy):
        ty = np.tanh(y_e)
        if abs(ty) > 1e-15:
            ratio = A / ty
            if 1e-12 < ratio < 0.25 - 1e-12:
                disc = 1 - 4 * ratio
                if disc > 0:
                    s = np.sqrt(disc)
                    for p in ((1 - s) / 2, (1 + s) / 2):
                        if 1e-12 < p < 1 - 1e-12:
                            x_crit = np.log(p / (1 - p))
                            if lx <= x_crit <= ux:
                                cands.append(g(x_crit, y_e))

    # interior critical points: quartic in p = sigma(x)
    coefs = [1.0, -(2.0 + B), (1.0 + 2.0 * B), -B, -(A ** 2)]
    roots = np.roots(coefs)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    for p in real_roots:
        if 1e-12 < p < 1 - 1e-12:
            x_crit = np.log(p / (1 - p))
            if x_crit < lx - 1e-12 or x_crit > ux + 1e-12:
                continue
            ratio_y = A / (p * (1 - p))
            if abs(ratio_y) >= 1 - 1e-12:
                continue
            y_crit = np.arctanh(ratio_y)
            if y_crit < ly - 1e-12 or y_crit > uy + 1e-12:
                continue
            cands.append(g(x_crit, y_crit))

    return float(min(cands)), float(max(cands))


def _sigtanh_plane(lx, ux, ly, uy):
    """Per-element plane (A, B, C1, C2) for f(x, y) = sigma(x) tanh(y)."""
    sl, su = _sigmoid(lx), _sigmoid(ux)
    tly, tuy = np.tanh(ly), np.tanh(uy)
    wx, wy = ux - lx, uy - ly

    if wx < 1e-12 and wy < 1e-12:
        f = sl * tly
        return 0.0, 0.0, f, f
    if wy < 1e-12:
        ty = tly
        if abs(ty) < 1e-15:
            return 0.0, 0.0, 0.0, 0.0
        a_sig, c1s, c2s = _sigmoid_plane_1d(lx, ux)
        A = ty * a_sig
        B = 0.0
        if ty > 0:
            C1 = ty * c1s
            C2 = ty * c2s
        else:
            C1 = ty * c2s
            C2 = ty * c1s
        return A, B, C1, C2
    if wx < 1e-12:
        sx = sl
        if abs(sx) < 1e-15:
            return 0.0, 0.0, 0.0, 0.0
        b_th, c1t, c2t = _tanh_plane_1d(ly, uy)
        A = 0.0
        B = sx * b_th
        if sx > 0:
            C1 = sx * c1t
            C2 = sx * c2t
        else:
            C1 = sx * c2t
            C2 = sx * c1t
        return A, B, C1, C2

    A = (su - sl) * (tly + tuy) / (2 * wx)
    B = (sl + su) * (tuy - tly) / (2 * wy)
    C1, C2 = _c1c2_sigtanh(A, B, lx, ux, ly, uy)
    return A, B, C1, C2


def bilinear_sigmoid_tanh(
    z_x: Zono, z_y: Zono, allocator: PredAllocator | None = None
) -> Zono:
    """Sound bilinear transformer for elementwise f(x, y) = sigma(x) * tanh(y).

    Theorem 4.2 (Du et al., §4.3.1). Pre-aligns z_x and z_y to a shared
    predicate space (Minkowski: shared pred_ids overlap, unshared get
    disjoint columns), then per-element corner-fit plane + exact
    (C1, C2) via _c1c2_sigtanh. K fresh predicates added.
    """
    if z_x.dim != z_y.dim:
        raise ValueError(
            f"bilinear_sigmoid_tanh: dim mismatch {z_x.dim} vs {z_y.dim}"
        )
    alloc = allocator if allocator is not None else get_default_allocator()
    K = z_x.dim
    shared_ids, (V_x, V_y) = align_pred_space(z_x, z_y)
    lb_x, ub_x = z_x.get_ranges()
    lb_y, ub_y = z_y.get_ranges()
    A = np.zeros(K)
    B = np.zeros(K)
    C1 = np.zeros(K)
    C2 = np.zeros(K)
    for k in range(K):
        A[k], B[k], C1[k], C2[k] = _sigtanh_plane(
            lb_x[k], ub_x[k], lb_y[k], ub_y[k]
        )
    new_c = A * z_x.c + B * z_y.c + 0.5 * (C1 + C2)
    scaled_V = A[:, None] * V_x + B[:, None] * V_y
    fresh_V = np.diag(0.5 * (C2 - C1))
    new_V = np.hstack([scaled_V, fresh_V])
    fresh_ids = alloc.next_n(K)
    return Zono(new_c, new_V, shared_ids + fresh_ids)


# ---------- bilinear: x * sigma(y) ----------


def _c1c2_sigid(A, B, lx, ux, ly, uy):
    """Exact min/max of g(x, y) = x sigma(y) - A x - B y over the box.

    Hessian det = -sigma'(y)^2 <= 0, so interior has only saddles.
    Extrema lie on the boundary:
      - 4 corners
      - vertical edges x = const != 0: stationary in y where
        x * sigma'(y) = B; solve p^2 - p + B/x = 0, p = sigma(y).
      - horizontal edges y = const: g linear in x, no interior critical pt.
    """

    def g(x, y):
        return x * _sigmoid(y) - A * x - B * y

    cands = [g(lx, ly), g(lx, uy), g(ux, ly), g(ux, uy)]
    for x_e in (lx, ux):
        if abs(x_e) < 1e-15:
            continue
        ratio = B / x_e
        if 0 < ratio < 0.25 - 1e-12:
            disc = 1 - 4 * ratio
            s = np.sqrt(disc)
            for p in ((1 - s) / 2, (1 + s) / 2):
                if 0 < p < 1:
                    y_crit = np.log(p / (1 - p))
                    if ly <= y_crit <= uy:
                        cands.append(g(x_e, y_crit))
    return float(min(cands)), float(max(cands))


def _sigid_plane(lx, ux, ly, uy):
    """Per-element plane (A, B, C1, C2) for f(x, y) = x * sigma(y)."""
    sly, suy = _sigmoid(ly), _sigmoid(uy)
    wx, wy = ux - lx, uy - ly

    if wy < 1e-12:
        # y is a point: f = x * sigma(ly) is exact affine in x. Zero error.
        return sly, 0.0, 0.0, 0.0
    if wx < 1e-12:
        # x is a point: f(x, y) = lx * sigma(y), 1D in y.
        A = sly
        B = lx * (suy - sly) / wy

        def g(y):
            return lx * _sigmoid(y) - A * lx - B * y

        cands = [g(ly), g(uy)]
        if abs(lx) > 1e-15:
            ratio = B / lx
            if 0 < ratio < 0.25 - 1e-12:
                disc = 1 - 4 * ratio
                s = np.sqrt(disc)
                for p in ((1 - s) / 2, (1 + s) / 2):
                    if 0 < p < 1:
                        y_crit = np.log(p / (1 - p))
                        if ly <= y_crit <= uy:
                            cands.append(g(y_crit))
        return A, B, float(min(cands)), float(max(cands))

    A = (sly + suy) / 2
    B = (lx + ux) * (suy - sly) / (2 * wy)
    C1, C2 = _c1c2_sigid(A, B, lx, ux, ly, uy)
    return A, B, C1, C2


def bilinear_sigmoid_identity(
    z_x: Zono, z_y: Zono, allocator: PredAllocator | None = None
) -> Zono:
    """Sound bilinear transformer for f(x, y) = x * sigma(y).

    Theorem 4.3 (Du et al., §4.3.2). Plane derived from Appendix C
    proofs (Table 9's printed formulas have transcription errors).
    K fresh predicates added.
    """
    if z_x.dim != z_y.dim:
        raise ValueError(
            f"bilinear_sigmoid_identity: dim mismatch {z_x.dim} vs {z_y.dim}"
        )
    alloc = allocator if allocator is not None else get_default_allocator()
    K = z_x.dim
    shared_ids, (V_x, V_y) = align_pred_space(z_x, z_y)
    lb_x, ub_x = z_x.get_ranges()
    lb_y, ub_y = z_y.get_ranges()
    A = np.zeros(K)
    B = np.zeros(K)
    C1 = np.zeros(K)
    C2 = np.zeros(K)
    for k in range(K):
        A[k], B[k], C1[k], C2[k] = _sigid_plane(
            lb_x[k], ub_x[k], lb_y[k], ub_y[k]
        )
    new_c = A * z_x.c + B * z_y.c + 0.5 * (C1 + C2)
    scaled_V = A[:, None] * V_x + B[:, None] * V_y
    fresh_V = np.diag(0.5 * (C2 - C1))
    new_V = np.hstack([scaled_V, fresh_V])
    fresh_ids = alloc.next_n(K)
    return Zono(new_c, new_V, shared_ids + fresh_ids)


# ---------- regression baseline: deliberately unsound ----------


def hadamard_affine_only(z_x: Zono, z_y: Zono) -> Zono:
    """DELIBERATELY UNSOUND affine-only elementwise product baseline.

    Linearizes f(x, y) = x * y around the centers
        f ~= c_x * y + x * c_y - c_x * c_y
    and DROPS the bilinear cross term entirely (no IBP error added).
    Mirrors NNV's Star.HadamardProduct path, the broken transformer
    that the Cert-RNN bilinears replace. Kept here as a regression
    target: tests/test_transformers.py asserts that the (1+alpha)^2
    witness lies outside this baseline's output (true ub=4, baseline
    ub=3).
    """
    if z_x.dim != z_y.dim:
        raise ValueError(
            f"hadamard_affine_only: dim mismatch {z_x.dim} vs {z_y.dim}"
        )
    shared_ids, (V_x, V_y) = align_pred_space(z_x, z_y)
    cx, cy = z_x.c, z_y.c
    new_c = cx * cy
    new_V = cy[:, None] * V_x + cx[:, None] * V_y
    return Zono(new_c, new_V, shared_ids)
