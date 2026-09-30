"""Certified stationary-point candidates for the bilinear residual extrema.

Replaces the float64 point-estimate + tolerance-gate logic that produced
findings F-1 and F-3 (docs/phase0_findings.md §G) with interval reasoning:

* Polynomial roots (interior quartic in p = sigma(x); edge quadratics) are
  located by companion eigenvalues, Newton-polished, and then enclosed by
  Smith's inclusion disks [Smith 1970; Braess & Hadeler 1973]: for a monic
  degree-n polynomial q and DISTINCT estimates z_1..z_n, all roots lie in
  the union of disks D_i = {z : |z - z_i| <= n |q(z_i)| / prod_{j!=i}
  |z_i - z_j|}. |q(z_i)| is bounded above by its float64 value plus a
  rigorous rounding term, so the disks are certified. A disk that does not
  reach the real axis contains no real root; the others give real
  intervals J that cover every real root — no |imag| gate, no p-range gate.

* Admissibility (root in (0,1), x = logit(p) in the box, |tanh y*| < 1,
  y* in the box) is decided on the intervals: a candidate is admitted iff
  SOME p~ in J satisfies every condition, so float error near a boundary
  can no longer drop a legitimate critical point.

* tanh(y*) is constrained by BOTH stationarity equations and their
  intervals are intersected: from A, tanh y* = A/(p(1-p)) (ill-conditioned
  when p -> 0 or 1 — the F-1 mechanism), from B, tanh^2 y* = 1 - B/p
  (well conditioned there). The true (x*, y*) lies in the resulting
  rectangle X x Y, and g is evaluated OUTWARD over it with interval
  arithmetic (each transcendental widened by two ulps), giving [lo, hi]
  candidate values instead of a single point value.

The enclosures are exact up to the assumption that libm's exp/tanh/log/
atanh are within 2 ulps, which the 2-ulp widening covers. Interval widths
are ~1e-15 in well-conditioned cases and grow only where the roots are
genuinely ill-conditioned (there the old code was wrong by up to 1e-5;
here the enclosure is merely wider, typically <= 1e-7).

All functions are vectorised over K elements; the scalar transformer paths
call them with K = 1.
"""

from __future__ import annotations

import numpy as np

_U = np.finfo(np.float64).eps / 2.0   # unit roundoff 1.1e-16
_INF = np.inf


# --------------------------------------------------------------------------- #
# interval primitives: every interval is a pair of float64 arrays (lo, hi)
# --------------------------------------------------------------------------- #
def _out(lo, hi):
    """Widen outward by two ulps (covers <=2-ulp libm error)."""
    lo = np.nextafter(np.nextafter(lo, -_INF), -_INF)
    hi = np.nextafter(np.nextafter(hi, _INF), _INF)
    return lo, hi


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def iv_sigmoid(lo, hi):
    return _out(_sigmoid(lo), _sigmoid(hi))


def iv_tanh(lo, hi):
    return _out(np.tanh(lo), np.tanh(hi))


def iv_logit(lo, hi):
    """logit over [lo, hi] subset of [0, 1]; endpoints at 0/1 map to -+inf."""
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(lo <= 0.0, -_INF, np.log(lo / (1.0 - lo)))
        b = np.where(hi >= 1.0, _INF, np.log(hi / (1.0 - hi)))
    return _out(a, b)


def iv_arctanh(lo, hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(lo <= -1.0, -_INF, np.arctanh(np.clip(lo, -1.0, 1.0)))
        b = np.where(hi >= 1.0, _INF, np.arctanh(np.clip(hi, -1.0, 1.0)))
    return _out(a, b)


def iv_scale(c, lo, hi):
    """c * [lo, hi] for scalar-per-element c (array broadcast)."""
    a = c * lo; b = c * hi
    return _out(np.minimum(a, b), np.maximum(a, b))


def iv_mul(alo, ahi, blo, bhi):
    p = np.stack([alo * blo, alo * bhi, ahi * blo, ahi * bhi])
    return _out(np.min(p, axis=0), np.max(p, axis=0))


def iv_sub(alo, ahi, blo, bhi):
    return _out(alo - bhi, ahi - blo)


def iv_intersect(alo, ahi, blo, bhi):
    lo = np.maximum(alo, blo); hi = np.minimum(ahi, bhi)
    empty = ~(lo <= hi)
    return lo, hi, empty


def iv_p1mp(lo, hi):
    """p(1-p) over [lo, hi] subset of [0, 1] (concave, max at 1/2)."""
    f_lo = lo * (1.0 - lo); f_hi = hi * (1.0 - hi)
    mn = np.minimum(f_lo, f_hi)
    mx = np.where((lo <= 0.5) & (hi >= 0.5), 0.25, np.maximum(f_lo, f_hi))
    olo, ohi = _out(np.maximum(mn, 0.0), mx)
    return np.maximum(olo, 0.0), ohi     # never negative: p(1-p) >= 0 on [0,1]


def iv_div_pos(a, dlo, dhi):
    """a / [dlo, dhi] for a denominator interval inside [0, inf): a is a
    per-element scalar. When dlo == 0 the quotient is unbounded on the side
    of sign(a); a == 0 gives [0, 0]."""
    with np.errstate(divide="ignore", invalid="ignore"):
        q_hi_end = np.where(dhi > 0, a / dhi, np.nan)      # closest to 0
        q_lo_end = np.where(dlo > 0, a / dlo, np.nan)      # farthest from 0
    # +-inf only where the quotient is unbounded (dlo == 0, a != 0); built
    # by masks so 0 * inf is never formed (same values as the unmasked form,
    # whose nan entries at a == 0 were overwritten by 0 anyway).
    a_b, dlo_b = np.broadcast_arrays(a, dlo)
    unbounded = ~(dlo_b > 0)
    far = np.where(unbounded, 0.0, q_lo_end)
    inf_side = unbounded & (a_b != 0.0)
    far[inf_side] = np.sign(a_b[inf_side]) * _INF
    far = np.where(a_b == 0.0, 0.0, far)
    near = np.where(dhi > 0, q_hi_end, far)
    lo = np.minimum(near, far); hi = np.maximum(near, far)
    return _out(lo, hi)


def iv_g_sigtanh(xlo, xhi, ylo, yhi, A, B):
    """[lo, hi] of g = sigma(x) tanh(y) - A x - B y over the rectangle."""
    slo, shi = iv_sigmoid(xlo, xhi)
    tlo, thi = iv_tanh(ylo, yhi)
    plo, phi = iv_mul(slo, shi, tlo, thi)
    axlo, axhi = iv_scale(A, xlo, xhi)
    bylo, byhi = iv_scale(B, ylo, yhi)
    lo, hi = iv_sub(plo, phi, axlo, axhi)
    return iv_sub(lo, hi, bylo, byhi)


def iv_g_sigid(xlo, xhi, ylo, yhi, A, B):
    """[lo, hi] of g = x sigma(y) - A x - B y over the rectangle."""
    slo, shi = iv_sigmoid(ylo, yhi)
    plo, phi = iv_mul(xlo, xhi, slo, shi)
    axlo, axhi = iv_scale(A, xlo, xhi)
    bylo, byhi = iv_scale(B, ylo, yhi)
    lo, hi = iv_sub(plo, phi, axlo, axhi)
    return iv_sub(lo, hi, bylo, byhi)


# --------------------------------------------------------------------------- #
# certified polynomial roots (Smith inclusion disks)
# --------------------------------------------------------------------------- #
def _poly_eval(coefs, z):
    """Horner for monic polynomial with coefficient arrays coefs[k] of shape
    (K,) (highest degree first, leading 1 omitted); z: (K, n) complex.
    Returns (value, rigorous |error| bound of the float64 evaluation
    including one-rounding coefficient error)."""
    n = len(coefs)
    val = np.ones_like(z)
    mag = np.ones(z.shape)          # sum |c_k| |z|^k, for the error bound
    az = np.abs(z)
    for c in coefs:
        val = val * z + c[:, None]
        mag = mag * az + np.abs(c)[:, None]
    # gamma_{2n} Horner bound + one rounding per coefficient (~ u each):
    # generous constant 4n u covers both.
    err = (4.0 * n * _U) * mag
    return val, err


def _poly_deriv_eval(coefs, z):
    n = len(coefs)
    d = np.full_like(z, n)          # derivative of z^n
    for k, c in enumerate(coefs[:-1]):
        d = d * z + (n - 1 - k) * c[:, None]
    return d


def _smith_disks(coefs, z):
    """Smith inclusion radii for one configuration of DISTINCT estimates
    z (K, n): R_i = n |q(z_i)| / prod_{j!=i} |z_i - z_j| with |q(z_i)|
    bounded above rigorously. Coincident estimates are nudged apart (any
    distinct points are admissible; poor ones just give larger disks)."""
    K, n = z.shape
    z = z.copy()
    for i in range(n):
        for j in range(i):
            same = np.abs(z[:, i] - z[:, j]) < 1e-13 * np.maximum(1.0, np.abs(z[:, i]))
            z[:, i] = np.where(same, z[:, i] + 1e-13 * (i + 1), z[:, i])
    q, qerr = _poly_eval(coefs, z)
    qbound = np.abs(q) + qerr
    prod = np.ones(z.shape)
    for i in range(n):
        for j in range(n):
            if i != j:
                prod[:, i] *= np.abs(z[:, i] - z[:, j])
    with np.errstate(divide="ignore", invalid="ignore"):
        R = n * qbound / prod
    R = np.where(np.isfinite(R), R * (1.0 + 1e-12) + 1e-300, _INF)
    return z, R


def _polish(coefs, z, n_polish):
    """Guarded Newton polish: a step is taken only if the residual is
    above its rounding floor, the step is small, and it reduces |q|. At an
    exact double root q and q' are both at rounding level and an unguarded
    step (q/q' ~ 1) can jump to a different root — which then collapses
    three estimates together and destroys the Smith radii."""
    for _ in range(n_polish):
        q, qerr = _poly_eval(coefs, z)
        dq = _poly_deriv_eval(coefs, z)
        step = np.where(dq != 0, q / np.where(dq != 0, dq, 1.0), 0.0)
        ok = (np.abs(q) > 2.0 * qerr) & (np.abs(step) <= 1e-2 * np.maximum(1.0, np.abs(z)))
        z_new = z - np.where(ok, step, 0.0)
        q_new, _ = _poly_eval(coefs, z_new)
        accept = ok & (np.abs(q_new) < np.abs(q))
        z = np.where(accept, z_new, z)
    return z


def smith_real_intervals(coefs, z, n_polish: int = 3):
    """coefs: list of (K,) arrays c_{n-1}..c_0 of the monic polynomial
    z^n + c_{n-1} z^{n-1} + ... + c_0; z: (K, n) complex root estimates.
    Returns (lo, hi, has_real): (K, n) arrays; for each estimate the real
    projection of its Smith disk, has_real False where the disk misses the
    real axis (no real root there).

    Cluster handling: for a (near-)double root, two nearby estimates give
    individual Smith radii ~ floor/separation, which explodes as the
    estimates coincide. Smith's theorem holds for ANY distinct estimates,
    so for every close pair we also evaluate the configuration with the
    pair re-placed at zbar +- rho, rho = sqrt(|q(zbar)| / prod_others),
    whose radii are ~ sqrt(floor) instead, and keep whichever
    configuration is tighter on that pair — both are valid covers of all
    roots; choosing the tighter one afterwards is sound."""
    K, n = z.shape
    z = _polish(coefs, z.astype(np.complex128).copy(), n_polish)
    zX, RX = _smith_disks(coefs, z)
    R = RX.copy(); zc = zX.copy()
    for i in range(n):
        for j in range(i + 1, n):
            zbar = 0.5 * (zX[:, i] + zX[:, j])
            close = np.abs(zX[:, i] - zX[:, j]) < 1e-3 * np.maximum(1.0, np.abs(zbar))
            if not close.any():
                continue
            prod_others = np.ones(K)
            for k in range(n):
                if k != i and k != j:
                    prod_others *= np.abs(zbar - zX[:, k])
            qb_bar, qerr_bar = _poly_eval(coefs, zbar[:, None])
            qbound_bar = (np.abs(qb_bar) + qerr_bar)[:, 0]
            with np.errstate(divide="ignore", invalid="ignore"):
                rho = np.sqrt(qbound_bar / np.maximum(prod_others, 1e-300))
            rho = np.maximum(rho, 1e-14 * np.maximum(1.0, np.abs(zbar)))
            zY = zX.copy()
            zY[:, i] = zbar + rho; zY[:, j] = zbar - rho
            zY2, RY = _smith_disks(coefs, zY)
            better = close & (np.maximum(RY[:, i], RY[:, j]) <
                              np.maximum(R[:, i], R[:, j]))
            if better.any():
                R = np.where(better[:, None], RY, R)
                zc = np.where(better[:, None], zY2, zc)
    re = zc.real; im = np.abs(zc.imag)
    has_real = im <= R
    lo = re - R; hi = re + R
    return lo, hi, has_real


# --------------------------------------------------------------------------- #
# sigma(x) tanh(y): interior critical points
# --------------------------------------------------------------------------- #
def sigtanh_interior_candidates(A, B, lx, ux, ly, uy):
    """Certified [lo, hi] value enclosures of g = sigma(x)tanh(y) - Ax - By
    at every interior critical point inside the box. Returns (lo, hi) of
    shape (K, 4) with NaN where a root cluster yields no admissible
    critical point.

    Critical points satisfy, with p = sigma(x):
        p^4 - (2+B) p^3 + (1+2B) p^2 - B p - A^2 = 0,
        tanh y = A / (p(1-p)),   p (1 - tanh^2 y) = B.
    """
    K = A.shape[0]
    c3 = -(2.0 + B); c2 = 1.0 + 2.0 * B; c1 = -B; c0 = -(A * A)
    companion = np.zeros((K, 4, 4))
    companion[:, 1, 0] = 1.0; companion[:, 2, 1] = 1.0; companion[:, 3, 2] = 1.0
    companion[:, 0, 3] = -c0; companion[:, 1, 3] = -c1
    companion[:, 2, 3] = -c2; companion[:, 3, 3] = -c3
    z = np.linalg.eigvals(companion)                       # (K, 4)
    plo, phi, has_real = smith_real_intervals([c3, c2, c1, c0], z)
    return _sigtanh_interior_from_p_intervals(A, B, lx, ux, ly, uy,
                                              plo, phi, has_real)


def _sigtanh_interior_from_p_intervals(A, B, lx, ux, ly, uy, plo, phi, has_real):
    K, n = plo.shape
    A_ = A[:, None]; B_ = B[:, None]
    lx_ = lx[:, None]; ux_ = ux[:, None]; ly_ = ly[:, None]; uy_ = uy[:, None]
    # p in (0, 1)
    plo = np.maximum(plo, 0.0); phi = np.minimum(phi, 1.0)
    ok = has_real & (plo < phi) & (phi > 0.0) & (plo < 1.0)
    plo = np.where(ok, plo, 0.25); phi = np.where(ok, phi, 0.75)   # dummies
    # x = logit(p) must meet the box
    xlo, xhi = iv_logit(plo, phi)
    xlo, xhi, empty = iv_intersect(xlo, xhi, lx_, ux_)
    ok &= ~empty
    xlo = np.where(ok, xlo, lx_); xhi = np.where(ok, xhi, ux_)
    # tanh y* from the A-equation over the p-interval
    qlo, qhi = iv_p1mp(plo, phi)                          # p(1-p) >= 0
    talo, tahi = iv_div_pos(np.broadcast_to(A_, plo.shape), qlo, qhi)
    # tanh^2 y* = 1 - B/p from the B-equation (well conditioned as p -> 1)
    # B/p over J (p > 0 after clipping; plo may be 0 -> unbounded side)
    bplo, bphi = iv_div_pos(np.broadcast_to(B_, plo.shape), plo, phi)
    slo, shi = _out(1.0 - bphi, 1.0 - bplo)
    shi = np.minimum(shi, 1.0)
    has_s = shi >= 0.0
    ok &= has_s
    slo = np.clip(slo, 0.0, 1.0); shi = np.clip(shi, 0.0, 1.0)
    tmag_lo = np.sqrt(slo); tmag_hi = np.sqrt(shi)
    tmag_lo, tmag_hi = _out(np.maximum(tmag_lo, 0.0), tmag_hi)
    lo_out = np.full((K, n, 2), np.nan); hi_out = np.full((K, n, 2), np.nan)
    for br, (blo, bhi) in enumerate(((-tmag_hi, -tmag_lo), (tmag_lo, tmag_hi))):
        tlo, thi, empty_t = iv_intersect(talo, tahi, blo, bhi)
        # |tanh y*| < 1 strictly for a finite y*: intersect with (-1, 1)
        tlo = np.maximum(tlo, -1.0); thi = np.minimum(thi, 1.0)
        good = ok & ~empty_t & (tlo <= thi) & ~((tlo >= 1.0) | (thi <= -1.0))
        tlo = np.where(good, tlo, 0.0); thi = np.where(good, thi, 0.0)
        ylo, yhi = iv_arctanh(tlo, thi)
        ylo, yhi, empty_y = iv_intersect(ylo, yhi, ly_, uy_)
        good &= ~empty_y
        ylo = np.where(good, ylo, ly_); yhi = np.where(good, yhi, uy_)
        glo, ghi = iv_g_sigtanh(xlo, xhi, ylo, yhi, A_, B_)
        lo_out[:, :, br] = np.where(good, glo, np.nan)
        hi_out[:, :, br] = np.where(good, ghi, np.nan)
    return lo_out.reshape(K, 2 * n), hi_out.reshape(K, 2 * n)


# --------------------------------------------------------------------------- #
# quadratic p^2 - p + c = 0 with certified real intervals
# --------------------------------------------------------------------------- #
def quad_p_intervals(c):
    """Certified real-root intervals of p^2 - p + c = 0, c: (K,).
    Returns (lo, hi, has_real) each (K, 2). disc = 1 - 4c is exact by
    Sterbenz when 4c in [0.5, 2] (where cancellation could matter) and has
    the correct sign elsewhere; the roots' only error is sqrt rounding,
    bounded outward by 2 ulps."""
    disc = 1.0 - 4.0 * c
    has_real = disc >= 0.0
    s = np.sqrt(np.maximum(disc, 0.0))
    s_lo, s_hi = _out(s, s)
    s_lo = np.maximum(s_lo, 0.0)
    r1lo, r1hi = _out((1.0 - s_hi) / 2.0, (1.0 - s_lo) / 2.0)
    r2lo, r2hi = _out((1.0 + s_lo) / 2.0, (1.0 + s_hi) / 2.0)
    lo = np.stack([r1lo, r2lo], axis=1); hi = np.stack([r1hi, r2hi], axis=1)
    # near disc ~ 0 the two intervals overlap around 1/2 (correct: double root)
    return lo, hi, np.stack([has_real, has_real], axis=1)


def sigtanh_hedge_candidates(A, B, lx, ux, y_e):
    """Horizontal edge y = y_e: sigma'(x) tanh(y_e) = A, i.e.
    p^2 - p + A/tanh(y_e) = 0. Returns (lo, hi) (K, 2) enclosures of g at
    the admissible stationary points, NaN where none."""
    K = A.shape[0]
    ty = np.tanh(y_e)
    ty_ok = np.abs(ty) > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(ty_ok, A / np.where(ty_ok, ty, 1.0), np.nan)
    c_lo, c_hi = _out(c, c)
    # c uncertain by 2 ulps: enclose roots for both ends and take the hull
    lo1, hi1, hr1 = quad_p_intervals(np.where(ty_ok, c_lo, 0.0))
    lo2, hi2, hr2 = quad_p_intervals(np.where(ty_ok, c_hi, 0.0))
    plo = np.minimum(lo1, lo2); phi = np.maximum(hi1, hi2)
    has = (hr1 | hr2) & ty_ok[:, None]
    plo = np.maximum(plo, 0.0); phi = np.minimum(phi, 1.0)
    ok = has & (plo < phi)
    plo = np.where(ok, plo, 0.25); phi = np.where(ok, phi, 0.75)
    xlo, xhi = iv_logit(plo, phi)
    xlo, xhi, empty = iv_intersect(xlo, xhi, lx[:, None], ux[:, None])
    ok &= ~empty
    xlo = np.where(ok, xlo, lx[:, None]); xhi = np.where(ok, xhi, ux[:, None])
    ye = np.broadcast_to(y_e[:, None], (K, 2))
    glo, ghi = iv_g_sigtanh(xlo, xhi, ye, ye, A[:, None], B[:, None])
    return np.where(ok, glo, np.nan), np.where(ok, ghi, np.nan)


def sigtanh_vedge_candidates(A, B, ly, uy, x_e):
    """Vertical edge x = x_e: sigma(x_e) tanh'(y) = B, i.e.
    tanh^2 y = 1 - B/sigma(x_e). Returns (lo, hi) (K, 2) for the +- branches."""
    K = A.shape[0]
    sx = _sigmoid(x_e)
    sx_lo, sx_hi = _out(sx, sx)
    sx_lo = np.maximum(sx_lo, 0.0)
    rlo, rhi = iv_div_pos(B, sx_lo, sx_hi)         # B / sigma(x_e)
    slo, shi = _out(1.0 - rhi, 1.0 - rlo)          # s = tanh^2 y
    has = shi >= 0.0
    slo = np.clip(slo, 0.0, 1.0); shi = np.clip(shi, 0.0, 1.0)
    tlo, thi = _out(np.sqrt(slo), np.sqrt(shi))
    tlo = np.maximum(tlo, 0.0)
    out_lo = np.full((K, 2), np.nan); out_hi = np.full((K, 2), np.nan)
    for br, (blo, bhi) in enumerate(((-thi, -tlo), (tlo, thi))):
        good = has & ~((blo >= 1.0) | (bhi <= -1.0))
        b_lo = np.where(good, np.maximum(blo, -1.0), 0.0)
        b_hi = np.where(good, np.minimum(bhi, 1.0), 0.0)
        ylo, yhi = iv_arctanh(b_lo, b_hi)
        ylo, yhi, empty = iv_intersect(ylo, yhi, ly, uy)
        good &= ~empty
        ylo = np.where(good, ylo, ly); yhi = np.where(good, yhi, uy)
        glo, ghi = iv_g_sigtanh(x_e, x_e, ylo, yhi, A, B)
        out_lo[:, br] = np.where(good, glo, np.nan)
        out_hi[:, br] = np.where(good, ghi, np.nan)
    return out_lo, out_hi


def sigid_vedge_candidates(A, B, ly, uy, x_e):
    """x sigma(y): vertical edge x = x_e != 0: x_e sigma'(y) = B, i.e.
    p^2 - p + B/x_e = 0 with p = sigma(y). Returns (lo, hi) (K, 2)."""
    K = A.shape[0]
    x_ok = x_e != 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(x_ok, B / np.where(x_ok, x_e, 1.0), np.nan)
    c_lo, c_hi = _out(c, c)
    lo1, hi1, hr1 = quad_p_intervals(np.where(x_ok, c_lo, 0.0))
    lo2, hi2, hr2 = quad_p_intervals(np.where(x_ok, c_hi, 0.0))
    plo = np.minimum(lo1, lo2); phi = np.maximum(hi1, hi2)
    has = (hr1 | hr2) & x_ok[:, None]
    plo = np.maximum(plo, 0.0); phi = np.minimum(phi, 1.0)
    ok = has & (plo < phi)
    plo = np.where(ok, plo, 0.25); phi = np.where(ok, phi, 0.75)
    ylo, yhi = iv_logit(plo, phi)
    ylo, yhi, empty = iv_intersect(ylo, yhi, ly[:, None], uy[:, None])
    ok &= ~empty
    ylo = np.where(ok, ylo, ly[:, None]); yhi = np.where(ok, yhi, uy[:, None])
    xe = np.broadcast_to(x_e[:, None], (K, 2))
    glo, ghi = iv_g_sigid(xe, xe, ylo, yhi, A[:, None], B[:, None])
    return np.where(ok, glo, np.nan), np.where(ok, ghi, np.nan)


# --------------------------------------------------------------------------- #
# outward rounding of the final offsets (ledger S14)
# --------------------------------------------------------------------------- #
def eta_outward(A, B, lx, ux, ly, uy):
    """Magnitude-relative outward rounding for point-evaluated candidates
    (corners): a few ulps of the largest term of g."""
    mx = np.maximum(np.abs(lx), np.abs(ux)); my = np.maximum(np.abs(ly), np.abs(uy))
    return 2e-15 * (1.0 + np.abs(A) * mx + np.abs(B) * my)
