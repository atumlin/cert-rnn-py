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
    # Vectorised 1D plane: produces a, C1, C2 in a single pass via batched
    # numpy ops. The crossover with the scalar loop is around K=4-8; the
    # 1D batch is cheap (no quartic, no candidate stack), so we just use
    # it unconditionally.
    a, C1, C2 = _tanh_plane_1d_batch(lb, ub)
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
    a, C1, C2 = _sigmoid_plane_1d_batch(lb, ub)
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


# ---------- batched 1D helpers ----------


def _sigmoid_plane_1d_batch(lx: np.ndarray, ux: np.ndarray):
    """Batched 1D sigmoid plane. Returns (a, C1, C2) each (K,) arrays."""
    sl = _sigmoid(lx)
    su = _sigmoid(ux)
    w = ux - lx
    point = w < 1e-12
    w_safe = np.where(point, 1.0, w)
    a = np.where(point, sl * (1 - sl), (su - sl) / w_safe)
    # Tangent points where a' = a; clamp into the box.
    A_FLOOR = 1e-15
    A_CEIL = 0.25 - 1e-15
    clamp = (a < A_FLOOR) | (a >= A_CEIL) | point
    # s = sqrt(1 - 4a) where a is in (0, 1/4). For clamped cells we'll override.
    a_safe = np.clip(a, A_FLOOR, A_CEIL - 1e-30)
    s = np.sqrt(np.clip(1 - 4 * a_safe, 0.0, None))
    s_safe = np.clip(s, 0.0, 1 - 1e-15)
    x_prime = -2 * np.arctanh(s_safe)
    x_prime2 = 2 * np.arctanh(s_safe)
    x_star = np.where(clamp, lx, np.maximum(x_prime, lx))
    x_star2 = np.where(clamp, ux, np.minimum(x_prime2, ux))
    # For point case the choice doesn't matter; C will collapse to the point value.
    C1 = _sigmoid(x_star) - a * x_star
    C2 = _sigmoid(x_star2) - a * x_star2
    # Point case: C1 = C2 = sl - a*lx.
    C_pt = sl - a * lx
    C1 = np.where(point, C_pt, C1)
    C2 = np.where(point, C_pt, C2)
    return a, C1, C2


def _tanh_plane_1d_batch(ly: np.ndarray, uy: np.ndarray):
    """Batched 1D tanh plane. Returns (a, C1, C2) each (K,) arrays."""
    tly = np.tanh(ly)
    tuy = np.tanh(uy)
    w = uy - ly
    point = w < 1e-12
    w_safe = np.where(point, 1.0, w)
    a = np.where(point, 1 - tly ** 2, (tuy - tly) / w_safe)
    A_FLOOR = 1e-15
    A_CEIL = 1 - 1e-15
    clamp = (a < A_FLOOR) | (a >= A_CEIL) | point
    a_safe = np.clip(a, A_FLOOR, A_CEIL - 1e-30)
    s = np.sqrt(np.clip(1 - a_safe, 0.0, None))
    s_safe = np.clip(s, 0.0, 1 - 1e-15)
    y_prime = -np.arctanh(s_safe)
    y_prime2 = np.arctanh(s_safe)
    y_star = np.where(clamp, ly, np.maximum(y_prime, ly))
    y_star2 = np.where(clamp, uy, np.minimum(y_prime2, uy))
    C1 = np.tanh(y_star) - a * y_star
    C2 = np.tanh(y_star2) - a * y_star2
    C_pt = tly - a * ly
    C1 = np.where(point, C_pt, C1)
    C2 = np.where(point, C_pt, C2)
    return a, C1, C2


# ---------- batched bilinear residual min/max ----------


def _c1c2_sigtanh_batch(A: np.ndarray, B: np.ndarray,
                        lx: np.ndarray, ux: np.ndarray,
                        ly: np.ndarray, uy: np.ndarray):
    """Batched exact min/max of g(x, y) = sigma(x) tanh(y) - A x - B y.

    A, B, lx, ux, ly, uy: (K,) arrays. Returns C1, C2 each (K,) arrays.
    Searches 4 corners + 4 vertical-edge stationary candidates + 4
    horizontal-edge stationary candidates + 4 interior quartic roots,
    all batched. Invalid candidates use NaN so np.nanmin/nanmax skip them.
    """
    K = A.shape[0]

    def g(x, y):
        return _sigmoid(x) * np.tanh(y) - A * x - B * y

    NAN = np.nan
    cands = [g(lx, ly), g(lx, uy), g(ux, ly), g(ux, uy)]

    # Vertical-edge stationary: at x = x_e, sigma(x_e) tanh'(y) = B
    #   => tanh(y_crit)^2 = 1 - B/sigma(x_e).
    def _vert_edge(x_e):
        sig_xe = _sigmoid(x_e)
        sig_ok = sig_xe > 1e-15
        ratio = np.where(sig_ok, B / np.where(sig_ok, sig_xe, 1.0), NAN)
        valid = (ratio > 1e-12) & (ratio < 1 - 1e-12)
        t2 = np.where(valid, 1.0 - ratio, NAN)
        t = np.sqrt(np.where(valid, np.maximum(t2, 0.0), 0.0))
        tv_unit = valid & (t < 1.0)
        t_safe = np.clip(t, 0.0, 1.0 - 1e-15)
        y_pos = np.where(tv_unit, np.arctanh(t_safe), NAN)
        y_neg = -y_pos
        in_pos = tv_unit & (y_pos >= ly) & (y_pos <= uy)
        in_neg = tv_unit & (y_neg >= ly) & (y_neg <= uy)
        return (
            np.where(in_pos, g(x_e, np.where(in_pos, y_pos, ly)), NAN),
            np.where(in_neg, g(x_e, np.where(in_neg, y_neg, ly)), NAN),
        )

    for c1, c2 in (_vert_edge(lx), _vert_edge(ux)):
        cands.append(c1); cands.append(c2)

    # Horizontal-edge stationary: at y = y_e, sigma'(x) tanh(y_e) = A
    #   => p^2 - p + A/tanh(y_e) = 0 with p = sigma(x_crit).
    def _horiz_edge(y_e):
        ty = np.tanh(y_e)
        ty_ok = np.abs(ty) > 1e-15
        ratio = np.where(ty_ok, A / np.where(ty_ok, ty, 1.0), NAN)
        valid = (ratio > 1e-12) & (ratio < 0.25 - 1e-12)
        disc = np.where(valid, 1 - 4 * ratio, NAN)
        valid = valid & (disc > 0)
        s = np.sqrt(np.where(valid, np.maximum(disc, 0.0), 0.0))
        p1 = (1 - s) / 2
        p2 = (1 + s) / 2
        v1 = valid & (p1 > 1e-12) & (p1 < 1 - 1e-12)
        v2 = valid & (p2 > 1e-12) & (p2 < 1 - 1e-12)
        p1s = np.clip(p1, 1e-15, 1 - 1e-15)
        p2s = np.clip(p2, 1e-15, 1 - 1e-15)
        x1 = np.where(v1, np.log(p1s / (1 - p1s)), NAN)
        x2 = np.where(v2, np.log(p2s / (1 - p2s)), NAN)
        in1 = v1 & (x1 >= lx) & (x1 <= ux)
        in2 = v2 & (x2 >= lx) & (x2 <= ux)
        return (
            np.where(in1, g(np.where(in1, x1, lx), y_e), NAN),
            np.where(in2, g(np.where(in2, x2, lx), y_e), NAN),
        )

    for c1, c2 in (_horiz_edge(ly), _horiz_edge(uy)):
        cands.append(c1); cands.append(c2)

    # Interior critical points: quartic in p = sigma(x):
    #   p^4 - (2+B) p^3 + (1+2B) p^2 - B p - A^2 = 0
    # Batched via np.linalg.eigvals on (K, 4, 4) companion matrices.
    companion = np.zeros((K, 4, 4))
    companion[:, 1, 0] = 1.0
    companion[:, 2, 1] = 1.0
    companion[:, 3, 2] = 1.0
    # Monic poly p^4 + c3 p^3 + c2 p^2 + c1 p + c0; companion last col is [-c0,-c1,-c2,-c3].
    # Our poly: c3 = -(2+B), c2 = 1+2B, c1 = -B, c0 = -A^2.
    companion[:, 0, 3] = A * A
    companion[:, 1, 3] = B
    companion[:, 2, 3] = -(1 + 2 * B)
    companion[:, 3, 3] = 2 + B
    eigvals = np.linalg.eigvals(companion)  # (K, 4) complex

    for j in range(4):
        ev = eigvals[:, j]
        real_mask = np.abs(ev.imag) < 1e-10
        p = np.where(real_mask, ev.real, NAN)
        p_unit = (p > 1e-12) & (p < 1 - 1e-12)
        p_safe = np.clip(p, 1e-15, 1 - 1e-15)
        x_crit = np.where(p_unit, np.log(p_safe / (1 - p_safe)), NAN)
        x_in = p_unit & (x_crit >= lx - 1e-12) & (x_crit <= ux + 1e-12)
        denom = np.maximum(p_safe * (1 - p_safe), 1e-30)
        ratio_y = np.where(p_unit, A / denom, NAN)
        y_unit = x_in & (np.abs(ratio_y) < 1 - 1e-12)
        ratio_y_safe = np.clip(ratio_y, -1 + 1e-15, 1 - 1e-15)
        y_crit = np.where(y_unit, np.arctanh(ratio_y_safe), NAN)
        in_box = y_unit & (y_crit >= ly - 1e-12) & (y_crit <= uy + 1e-12)
        cands.append(np.where(in_box,
                              g(np.where(in_box, x_crit, lx),
                                np.where(in_box, y_crit, ly)),
                              NAN))

    stacked = np.stack(cands, axis=-1)
    C1 = np.nanmin(stacked, axis=-1)
    C2 = np.nanmax(stacked, axis=-1)
    return C1, C2


def _sigtanh_plane_batch(lx: np.ndarray, ux: np.ndarray,
                         ly: np.ndarray, uy: np.ndarray):
    """Batched per-element plane (A, B, C1, C2) for f(x, y) = sigma(x) tanh(y).
    All inputs (K,); all outputs (K,). Degenerate cases (wx<eps or wy<eps)
    are handled by overlays from the 1D plane helpers."""
    sl = _sigmoid(lx); su = _sigmoid(ux)
    tly = np.tanh(ly); tuy = np.tanh(uy)
    wx = ux - lx; wy = uy - ly
    LEN = 1e-12; SIG = 1e-15

    x_point = wx < LEN
    y_point = wy < LEN

    wx_safe = np.where(x_point, 1.0, wx)
    wy_safe = np.where(y_point, 1.0, wy)

    A = (su - sl) * (tly + tuy) / (2 * wx_safe)
    B = (sl + su) * (tuy - tly) / (2 * wy_safe)
    C1, C2 = _c1c2_sigtanh_batch(A, B, lx, ux, ly, uy)

    # Degenerate y (y is a point): f = ty * sigma(x); 1D plane in x.
    only_y = y_point & ~x_point
    if only_y.any():
        a_s, c1s, c2s = _sigmoid_plane_1d_batch(lx[only_y], ux[only_y])
        ty = tly[only_y]
        small = np.abs(ty) < SIG
        A_sub = np.where(small, 0.0, ty * a_s)
        B_sub = np.zeros_like(ty)
        pos = ty > 0
        C1_sub = np.where(small, 0.0, np.where(pos, ty * c1s, ty * c2s))
        C2_sub = np.where(small, 0.0, np.where(pos, ty * c2s, ty * c1s))
        A[only_y] = A_sub; B[only_y] = B_sub
        C1[only_y] = C1_sub; C2[only_y] = C2_sub

    # Degenerate x (x is a point): f = sl * tanh(y); 1D plane in y.
    only_x = x_point & ~y_point
    if only_x.any():
        b_t, c1t, c2t = _tanh_plane_1d_batch(ly[only_x], uy[only_x])
        sx = sl[only_x]
        small = np.abs(sx) < SIG
        A_sub = np.zeros_like(sx)
        B_sub = np.where(small, 0.0, sx * b_t)
        pos = sx > 0
        C1_sub = np.where(small, 0.0, np.where(pos, sx * c1t, sx * c2t))
        C2_sub = np.where(small, 0.0, np.where(pos, sx * c2t, sx * c1t))
        A[only_x] = A_sub; B[only_x] = B_sub
        C1[only_x] = C1_sub; C2[only_x] = C2_sub

    # Both points: constant.
    both = x_point & y_point
    if both.any():
        fval = sl[both] * tly[both]
        A[both] = 0.0; B[both] = 0.0
        C1[both] = fval; C2[both] = fval

    return A, B, C1, C2


def _c1c2_sigid_batch(A: np.ndarray, B: np.ndarray,
                      lx: np.ndarray, ux: np.ndarray,
                      ly: np.ndarray, uy: np.ndarray):
    """Batched exact min/max of g(x, y) = x sigma(y) - A x - B y."""
    K = A.shape[0]

    def g(x, y):
        return x * _sigmoid(y) - A * x - B * y

    NAN = np.nan
    cands = [g(lx, ly), g(lx, uy), g(ux, ly), g(ux, uy)]

    # Vertical-edge stationary (x = x_e, x_e != 0): x sigma'(y) = B
    #   => p^2 - p + B/x_e = 0, p = sigma(y_crit).
    def _vert_edge(x_e):
        x_ok = np.abs(x_e) > 1e-15
        ratio = np.where(x_ok, B / np.where(x_ok, x_e, 1.0), NAN)
        valid = x_ok & (ratio > 0) & (ratio < 0.25 - 1e-12)
        disc = np.where(valid, 1 - 4 * ratio, NAN)
        valid = valid & (disc > 0)
        s = np.sqrt(np.where(valid, np.maximum(disc, 0.0), 0.0))
        p1 = (1 - s) / 2
        p2 = (1 + s) / 2
        v1 = valid & (p1 > 0) & (p1 < 1)
        v2 = valid & (p2 > 0) & (p2 < 1)
        p1s = np.clip(p1, 1e-15, 1 - 1e-15)
        p2s = np.clip(p2, 1e-15, 1 - 1e-15)
        y1 = np.where(v1, np.log(p1s / (1 - p1s)), NAN)
        y2 = np.where(v2, np.log(p2s / (1 - p2s)), NAN)
        in1 = v1 & (y1 >= ly) & (y1 <= uy)
        in2 = v2 & (y2 >= ly) & (y2 <= uy)
        return (
            np.where(in1, g(x_e, np.where(in1, y1, ly)), NAN),
            np.where(in2, g(x_e, np.where(in2, y2, ly)), NAN),
        )

    for c1, c2 in (_vert_edge(lx), _vert_edge(ux)):
        cands.append(c1); cands.append(c2)

    stacked = np.stack(cands, axis=-1)
    C1 = np.nanmin(stacked, axis=-1)
    C2 = np.nanmax(stacked, axis=-1)
    return C1, C2


def _sigid_plane_batch(lx: np.ndarray, ux: np.ndarray,
                       ly: np.ndarray, uy: np.ndarray):
    """Batched per-element plane (A, B, C1, C2) for f(x, y) = x * sigma(y)."""
    sly = _sigmoid(ly); suy = _sigmoid(uy)
    wx = ux - lx; wy = uy - ly
    LEN = 1e-12

    y_point = wy < LEN
    x_point = wx < LEN

    wy_safe = np.where(y_point, 1.0, wy)

    A = (sly + suy) / 2
    B = (lx + ux) * (suy - sly) / (2 * wy_safe)
    C1, C2 = _c1c2_sigid_batch(A, B, lx, ux, ly, uy)

    # Degenerate y (y is a point): f = x * sigma(ly) is affine in x. Zero error.
    if y_point.any():
        A[y_point] = sly[y_point]
        B[y_point] = 0.0
        C1[y_point] = 0.0
        C2[y_point] = 0.0

    # Degenerate x (x is a point, y not): 1D in y.
    only_x = x_point & ~y_point
    if only_x.any():
        lx_sub = lx[only_x]; ly_sub = ly[only_x]; uy_sub = uy[only_x]
        sly_sub = sly[only_x]; suy_sub = suy[only_x]
        wy_sub = uy_sub - ly_sub
        A_sub = sly_sub
        B_sub = lx_sub * (suy_sub - sly_sub) / wy_sub

        def g_1d(y):
            return lx_sub * _sigmoid(y) - A_sub * lx_sub - B_sub * y

        # Two corners + up to 2 interior critical points
        cands = [g_1d(ly_sub), g_1d(uy_sub)]
        lx_ok = np.abs(lx_sub) > 1e-15
        ratio = np.where(lx_ok, B_sub / np.where(lx_ok, lx_sub, 1.0), np.nan)
        valid = lx_ok & (ratio > 0) & (ratio < 0.25 - 1e-12)
        disc = np.where(valid, 1 - 4 * ratio, np.nan)
        valid = valid & (disc > 0)
        s = np.sqrt(np.where(valid, np.maximum(disc, 0.0), 0.0))
        p1 = (1 - s) / 2; p2 = (1 + s) / 2
        v1 = valid & (p1 > 0) & (p1 < 1)
        v2 = valid & (p2 > 0) & (p2 < 1)
        p1s = np.clip(p1, 1e-15, 1 - 1e-15)
        p2s = np.clip(p2, 1e-15, 1 - 1e-15)
        y1 = np.where(v1, np.log(p1s / (1 - p1s)), np.nan)
        y2 = np.where(v2, np.log(p2s / (1 - p2s)), np.nan)
        in1 = v1 & (y1 >= ly_sub) & (y1 <= uy_sub)
        in2 = v2 & (y2 >= ly_sub) & (y2 <= uy_sub)
        cands.append(np.where(in1, g_1d(np.where(in1, y1, ly_sub)), np.nan))
        cands.append(np.where(in2, g_1d(np.where(in2, y2, ly_sub)), np.nan))
        stacked_sub = np.stack(cands, axis=-1)
        A[only_x] = A_sub
        B[only_x] = B_sub
        C1[only_x] = np.nanmin(stacked_sub, axis=-1)
        C2[only_x] = np.nanmax(stacked_sub, axis=-1)

    return A, B, C1, C2


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
    # Batched vectorisation across K has ~250 us fixed overhead; for K < 8
    # the scalar loop wins. Crossover measured empirically.
    if K < 8:
        A = np.empty(K); B = np.empty(K); C1 = np.empty(K); C2 = np.empty(K)
        for k in range(K):
            A[k], B[k], C1[k], C2[k] = _sigtanh_plane(
                lb_x[k], ub_x[k], lb_y[k], ub_y[k]
            )
    else:
        A, B, C1, C2 = _sigtanh_plane_batch(lb_x, ub_x, lb_y, ub_y)
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
    if K < 8:
        A = np.empty(K); B = np.empty(K); C1 = np.empty(K); C2 = np.empty(K)
        for k in range(K):
            A[k], B[k], C1[k], C2[k] = _sigid_plane(
                lb_x[k], ub_x[k], lb_y[k], ub_y[k]
            )
    else:
        A, B, C1, C2 = _sigid_plane_batch(lb_x, ub_x, lb_y, ub_y)
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
