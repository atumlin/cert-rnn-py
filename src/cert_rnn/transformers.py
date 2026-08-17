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


# ---------- bilinear mode (baseline box vs ZRLT Tier 1) ----------

_BILINEAR_MODE = "box"   # "box": Cert-RNN baseline; "zono": Tier 1

# ---------- sigma*tanh tilt selection ----------
# "cornerfit": Table 8 Case-1 tilt applied universally (the shipped default,
#              also the MATLAB reference's choice);
# "table8":    faithful per-case Table 8 tilts (cert_rnn.table8), keeping
#              the tightest listed sub-solution where several are listed;
# "best":      tightest of table8 and cornerfit per element.
# All three use the exact certified offsets, so all are sound; they differ
# only in tilt. TABLE8_CASE_COUNTS accumulates per-case coverage when set
# to a dict.
_TILT_MODE = "cornerfit"
TABLE8_CASE_COUNTS = None


def set_tilt_mode(mode: str) -> None:
    global _TILT_MODE
    if mode not in ("cornerfit", "table8", "best"):
        raise ValueError(mode)
    _TILT_MODE = mode


def get_tilt_mode() -> str:
    return _TILT_MODE


class tilt_mode:
    def __init__(self, mode):
        self.mode = mode

    def __enter__(self):
        self.prev = get_tilt_mode(); set_tilt_mode(self.mode); return self

    def __exit__(self, *exc):
        set_tilt_mode(self.prev); return False


def set_bilinear_mode(mode: str) -> None:
    """Select the bilinear residual-offset method globally.
    "box"  -- exact residual extrema over the operand box (baseline).
    "zono" -- ZRLT Tier 1: extrema over a grid cover of the joint 2-D
              zonotope (cert_rnn.tier1); never looser than "box"."""
    global _BILINEAR_MODE
    if mode not in ("box", "zono"):
        raise ValueError(f"unknown bilinear mode {mode!r}")
    _BILINEAR_MODE = mode


def get_bilinear_mode() -> str:
    return _BILINEAR_MODE


class bilinear_mode:
    """Context manager: `with bilinear_mode("zono"): ...`."""

    def __init__(self, mode: str):
        self.mode = mode

    def __enter__(self):
        self.prev = get_bilinear_mode()
        set_bilinear_mode(self.mode)
        return self

    def __exit__(self, *exc):
        set_bilinear_mode(self.prev)
        return False


def _fresh_block(C1: np.ndarray, C2: np.ndarray, alloc: PredAllocator):
    """Fresh-generator block for a transformer output, with zero-width
    pruning: elements whose exact residual spread is zero (point inputs,
    degenerate planes) get NO fresh generator instead of a zero column.

    Arithmetic-neutral: dropping all-zero columns changes no downstream
    sum, range, or bound -- it only stops point-frames from inflating the
    predicate count (in single_frame mode most steps are points, so this
    collapses P and every later alignment/matmul with it).

    Returns (fresh_V (K, K'), fresh_ids tuple of length K')."""
    K = C1.shape[0]
    width = 0.5 * (C2 - C1)
    nz = np.flatnonzero(width != 0.0)
    if nz.size == K:
        return np.diag(width), alloc.next_n(K)
    fresh_V = np.zeros((K, nz.size))
    fresh_V[nz, np.arange(nz.size)] = width[nz]
    return fresh_V, alloc.next_n(int(nz.size))


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
    fresh_V, fresh_ids = _fresh_block(C1, C2, alloc)
    new_V = np.hstack([scaled_V, fresh_V])
    return Zono(new_c, new_V, z.pred_ids + fresh_ids)


def sigmoid_zono(z: Zono, allocator: PredAllocator | None = None) -> Zono:
    """Sound elementwise sigmoid transformer (§4.2.2)."""
    alloc = allocator if allocator is not None else get_default_allocator()
    K = z.dim
    lb, ub = z.get_ranges()
    a, C1, C2 = _sigmoid_plane_1d_batch(lb, ub)
    new_c = a * z.c + 0.5 * (C1 + C2)
    scaled_V = a[:, None] * z.V if z.n_pred > 0 else np.zeros((K, 0))
    fresh_V, fresh_ids = _fresh_block(C1, C2, alloc)
    new_V = np.hstack([scaled_V, fresh_V])
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
    """Batched min/max of g(x, y) = sigma(x) tanh(y) - A x - B y over the box.

    A, B, lx, ux, ly, uy: (K,) arrays. Returns C1, C2 each (K,) arrays.
    Candidates: 4 corners (point values, rounded outward by eta) plus
    CERTIFIED interval enclosures of g at every edge-stationary point and
    every interior critical point (cert_rnn.certified: Smith root
    inclusion + interval admissibility + outward interval evaluation).
    Invalid candidates are NaN so np.nanmin/nanmax skip them.
    """
    from cert_rnn.certified import (
        eta_outward,
        sigtanh_hedge_candidates,
        sigtanh_interior_candidates,
        sigtanh_vedge_candidates,
    )

    def g(x, y):
        return _sigmoid(x) * np.tanh(y) - A * x - B * y

    eta = eta_outward(A, B, lx, ux, ly, uy)
    corners = np.stack([g(lx, ly), g(lx, uy), g(ux, ly), g(ux, uy)], axis=1)
    los = [corners - eta[:, None]]
    his = [corners + eta[:, None]]
    for x_e in (lx, ux):
        lo, hi = sigtanh_vedge_candidates(A, B, ly, uy, x_e)
        los.append(lo); his.append(hi)
    for y_e in (ly, uy):
        lo, hi = sigtanh_hedge_candidates(A, B, lx, ux, y_e)
        los.append(lo); his.append(hi)
    lo, hi = sigtanh_interior_candidates(A, B, lx, ux, ly, uy)
    los.append(lo); his.append(hi)
    C1 = np.nanmin(np.concatenate(los, axis=1), axis=1)
    C2 = np.nanmax(np.concatenate(his, axis=1), axis=1)
    return C1, C2


def _apply_table8_tilts(A, B, C1, C2, lx, ux, ly, uy, nondeg):
    """Replace the corner-fit plane by the Table 8 per-case tilt (mode
    "table8": tightest listed sub-solution; mode "best": tightest of those
    and corner-fit). Offsets are always the exact certified extrema for the
    chosen tilt, so every option is sound. Non-degenerate elements only."""
    from cert_rnn.table8 import table8_tilts

    A = A.copy(); B = B.copy(); C1 = C1.copy(); C2 = C2.copy()
    idx = np.flatnonzero(nondeg)
    if idx.size == 0:
        return A, B, C1, C2
    cand_A = []; cand_B = []; owner = []; tags = []; cases = []
    for k in idx:
        case, tilts = table8_tilts(float(lx[k]), float(ux[k]), float(ly[k]), float(uy[k]))
        cases.append(case)
        for tag, a, b in tilts:
            cand_A.append(a); cand_B.append(b); owner.append(k); tags.append(tag)
    if TABLE8_CASE_COUNTS is not None:
        for c in cases:
            TABLE8_CASE_COUNTS[c] = TABLE8_CASE_COUNTS.get(c, 0) + 1
    cand_A = np.asarray(cand_A); cand_B = np.asarray(cand_B); owner = np.asarray(owner)
    c1, c2 = _c1c2_sigtanh_batch(cand_A, cand_B, lx[owner], ux[owner], ly[owner], uy[owner])
    gap = c2 - c1
    for k in idx:
        m = owner == k
        j = np.flatnonzero(m)[np.argmin(gap[m])]
        best_gap = gap[j]
        if _TILT_MODE == "table8" or best_gap < (C2[k] - C1[k]):
            A[k] = cand_A[j]; B[k] = cand_B[j]; C1[k] = c1[j]; C2[k] = c2[j]
            if TABLE8_CASE_COUNTS is not None:
                TABLE8_CASE_COUNTS["win:" + tags[j]] = TABLE8_CASE_COUNTS.get("win:" + tags[j], 0) + 1
        elif TABLE8_CASE_COUNTS is not None:
            TABLE8_CASE_COUNTS["win:cornerfit"] = TABLE8_CASE_COUNTS.get("win:cornerfit", 0) + 1
    return A, B, C1, C2


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

    if _TILT_MODE != "cornerfit":
        A, B, C1, C2 = _apply_table8_tilts(A, B, C1, C2, lx, ux, ly, uy,
                                           ~(x_point | y_point))

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
        # F-2: y is treated as the point ly but truly spans wy < 1e-12;
        # |d f / d y| = sigma(x) tanh'(y) <= 1, so widen outward by wy.
        C1_sub = C1_sub - wy[only_y]; C2_sub = C2_sub + wy[only_y]
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
        # F-2: |d f / d x| = sigma'(x) tanh(y) <= 1/4 over the true wx.
        C1_sub = C1_sub - 0.25 * wx[only_x]; C2_sub = C2_sub + 0.25 * wx[only_x]
        A[only_x] = A_sub; B[only_x] = B_sub
        C1[only_x] = C1_sub; C2[only_x] = C2_sub

    # Both points: constant (F-2: widen by the collapsed extents).
    both = x_point & y_point
    if both.any():
        fval = sl[both] * tly[both]
        slack = 0.25 * wx[both] + wy[both]
        A[both] = 0.0; B[both] = 0.0
        C1[both] = fval - slack; C2[both] = fval + slack

    return A, B, C1, C2


def _c1c2_sigid_batch(A: np.ndarray, B: np.ndarray,
                      lx: np.ndarray, ux: np.ndarray,
                      ly: np.ndarray, uy: np.ndarray):
    """Batched min/max of g(x, y) = x sigma(y) - A x - B y over the box.
    Corners (outward by eta) + certified vertical-edge stationary intervals
    (the interior has only saddles; horizontal edges are linear in x)."""
    from cert_rnn.certified import eta_outward, sigid_vedge_candidates

    def g(x, y):
        return x * _sigmoid(y) - A * x - B * y

    eta = eta_outward(A, B, lx, ux, ly, uy)
    corners = np.stack([g(lx, ly), g(lx, uy), g(ux, ly), g(ux, uy)], axis=1)
    los = [corners - eta[:, None]]
    his = [corners + eta[:, None]]
    for x_e in (lx, ux):
        lo, hi = sigid_vedge_candidates(A, B, ly, uy, x_e)
        los.append(lo); his.append(hi)
    C1 = np.nanmin(np.concatenate(los, axis=1), axis=1)
    C2 = np.nanmax(np.concatenate(his, axis=1), axis=1)
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

    # Degenerate y (y is a point): f = x * sigma(ly) is affine in x.
    # F-2: over the true wy < 1e-12, |d f / d y| = |x| sigma'(y) <= |x|/4.
    if y_point.any():
        slack = 0.25 * np.maximum(np.abs(lx[y_point]), np.abs(ux[y_point])) * wy[y_point]
        A[y_point] = sly[y_point]
        B[y_point] = 0.0
        C1[y_point] = -slack
        C2[y_point] = slack

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
        # F-2: |d f / d x| = sigma(y) <= 1 over the true wx < 1e-12.
        C1[only_x] = np.nanmin(stacked_sub, axis=-1) - wx[only_x]
        C2[only_x] = np.nanmax(stacked_sub, axis=-1) + wx[only_x]

    return A, B, C1, C2


# ---------- bilinear: sigma(x) * tanh(y) ----------


def _c1c2_sigtanh(A, B, lx, ux, ly, uy):
    """Min/max of g(x, y) = sigma(x) tanh(y) - A x - B y over the box
    (scalar transcription). Corners as Python floats with outward eta;
    stationary-point candidates via the shared certified interval
    routines (cert_rnn.certified) on 1-element arrays — the interval
    logic is deliberately implemented once."""
    from cert_rnn.certified import (
        eta_outward,
        sigtanh_hedge_candidates,
        sigtanh_interior_candidates,
        sigtanh_vedge_candidates,
    )

    def g(x, y):
        return _sigmoid(x) * np.tanh(y) - A * x - B * y

    a1 = np.array([A]); b1 = np.array([B])
    lx1 = np.array([lx]); ux1 = np.array([ux]); ly1 = np.array([ly]); uy1 = np.array([uy])
    eta = float(eta_outward(a1, b1, lx1, ux1, ly1, uy1)[0])
    corners = [g(lx, ly), g(lx, uy), g(ux, ly), g(ux, uy)]
    los = [c - eta for c in corners]
    his = [c + eta for c in corners]
    for x_e in (lx1, ux1):
        lo, hi = sigtanh_vedge_candidates(a1, b1, ly1, uy1, x_e)
        los += [v for v in lo[0] if not np.isnan(v)]
        his += [v for v in hi[0] if not np.isnan(v)]
    for y_e in (ly1, uy1):
        lo, hi = sigtanh_hedge_candidates(a1, b1, lx1, ux1, y_e)
        los += [v for v in lo[0] if not np.isnan(v)]
        his += [v for v in hi[0] if not np.isnan(v)]
    lo, hi = sigtanh_interior_candidates(a1, b1, lx1, ux1, ly1, uy1)
    los += [v for v in lo[0] if not np.isnan(v)]
    his += [v for v in hi[0] if not np.isnan(v)]
    return float(min(los)), float(max(his))


def _sigtanh_plane(lx, ux, ly, uy):
    """Per-element plane (A, B, C1, C2) for f(x, y) = sigma(x) tanh(y)."""
    sl, su = _sigmoid(lx), _sigmoid(ux)
    tly, tuy = np.tanh(ly), np.tanh(uy)
    wx, wy = ux - lx, uy - ly

    # F-2: sub-1e-12 widths are treated as points; widen outward by the
    # Lipschitz bound over the collapsed extent (|df/dx| <= 1/4, |df/dy| <= 1).
    if wx < 1e-12 and wy < 1e-12:
        f = sl * tly
        slack = 0.25 * wx + wy
        return 0.0, 0.0, f - slack, f + slack
    if wy < 1e-12:
        ty = tly
        if abs(ty) < 1e-15:
            return 0.0, 0.0, -wy, wy
        a_sig, c1s, c2s = _sigmoid_plane_1d(lx, ux)
        A = ty * a_sig
        B = 0.0
        if ty > 0:
            C1 = ty * c1s
            C2 = ty * c2s
        else:
            C1 = ty * c2s
            C2 = ty * c1s
        return A, B, C1 - wy, C2 + wy
    if wx < 1e-12:
        sx = sl
        if abs(sx) < 1e-15:
            return 0.0, 0.0, -0.25 * wx, 0.25 * wx
        b_th, c1t, c2t = _tanh_plane_1d(ly, uy)
        A = 0.0
        B = sx * b_th
        if sx > 0:
            C1 = sx * c1t
            C2 = sx * c2t
        else:
            C1 = sx * c2t
            C2 = sx * c1t
        return A, B, C1 - 0.25 * wx, C2 + 0.25 * wx

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
    # Since the certified candidate machinery (cert_rnn.certified) is
    # vectorised, the batched plane is at least as fast as the scalar loop
    # at every K (measured: equal at K=1, 4x faster at K=4). The scalar
    # transcription is kept for the differential canary
    # (tests/soundness/test_scalar_batch_diff.py).
    A, B, C1, C2 = _sigtanh_plane_batch(lb_x, ub_x, lb_y, ub_y)
    if _BILINEAR_MODE == "zono":
        from cert_rnn.tier1 import c1c2_over_zono
        C1, C2 = c1c2_over_zono("sigtanh", A, B, z_x.c, z_y.c, V_x, V_y,
                                lb_x, ub_x, lb_y, ub_y, C1, C2)
    new_c = A * z_x.c + B * z_y.c + 0.5 * (C1 + C2)
    scaled_V = A[:, None] * V_x + B[:, None] * V_y
    fresh_V, fresh_ids = _fresh_block(C1, C2, alloc)
    new_V = np.hstack([scaled_V, fresh_V])
    return Zono(new_c, new_V, shared_ids + fresh_ids)


# ---------- bilinear: x * sigma(y) ----------


def _c1c2_sigid(A, B, lx, ux, ly, uy):
    """Min/max of g(x, y) = x sigma(y) - A x - B y over the box (scalar
    transcription).

    Hessian det = -sigma'(y)^2 <= 0, so interior has only saddles.
    Extrema lie on the boundary: 4 corners (outward by eta) and the
    vertical-edge stationary points x_e sigma'(y) = B (certified
    intervals via cert_rnn.certified); horizontal edges are linear in x.
    """
    from cert_rnn.certified import eta_outward, sigid_vedge_candidates

    def g(x, y):
        return x * _sigmoid(y) - A * x - B * y

    a1 = np.array([A]); b1 = np.array([B])
    lx1 = np.array([lx]); ux1 = np.array([ux]); ly1 = np.array([ly]); uy1 = np.array([uy])
    eta = float(eta_outward(a1, b1, lx1, ux1, ly1, uy1)[0])
    corners = [g(lx, ly), g(lx, uy), g(ux, ly), g(ux, uy)]
    los = [c - eta for c in corners]
    his = [c + eta for c in corners]
    for x_e in (lx1, ux1):
        lo, hi = sigid_vedge_candidates(a1, b1, ly1, uy1, x_e)
        los += [v for v in lo[0] if not np.isnan(v)]
        his += [v for v in hi[0] if not np.isnan(v)]
    return float(min(los)), float(max(his))


def _sigid_plane(lx, ux, ly, uy):
    """Per-element plane (A, B, C1, C2) for f(x, y) = x * sigma(y)."""
    sly, suy = _sigmoid(ly), _sigmoid(uy)
    wx, wy = ux - lx, uy - ly

    if wy < 1e-12:
        # y is a point: f = x * sigma(ly) is affine in x. F-2: widen by
        # the Lipschitz bound |x| sigma'(y) <= |x|/4 over the true wy.
        slack = 0.25 * max(abs(lx), abs(ux)) * wy
        return sly, 0.0, -slack, slack
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
        # F-2: |df/dx| = sigma(y) <= 1 over the true wx < 1e-12.
        return A, B, float(min(cands)) - wx, float(max(cands)) + wx

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
    A, B, C1, C2 = _sigid_plane_batch(lb_x, ub_x, lb_y, ub_y)   # see sigtanh note
    if _BILINEAR_MODE == "zono":
        from cert_rnn.tier1 import c1c2_over_zono
        C1, C2 = c1c2_over_zono("sigid", A, B, z_x.c, z_y.c, V_x, V_y,
                                lb_x, ub_x, lb_y, ub_y, C1, C2)
    new_c = A * z_x.c + B * z_y.c + 0.5 * (C1 + C2)
    scaled_V = A[:, None] * V_x + B[:, None] * V_y
    fresh_V, fresh_ids = _fresh_block(C1, C2, alloc)
    new_V = np.hstack([scaled_V, fresh_V])
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
