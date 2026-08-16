"""Independent high-precision reference for the bilinear residual extrema.

Deliberately slow, deliberately independent: written directly from the
stationarity mathematics using mpmath at 50 digits, importing NOTHING from
cert_rnn.certified or cert_rnn.transformers. This is the nightly canary
that replaced the retired scalar-transcription R-1 run — genuinely
independent arithmetic rather than a second transcription of shared code.

For g = sigma(x) tanh(y) - A x - B y over [lx,ux] x [ly,uy], the extrema
are attained at: the 4 corners; vertical-edge stationary points
(tanh^2 y = 1 - B / sigma(x_e)); horizontal-edge stationary points
(p^2 - p + A / tanh(y_e) = 0, p = sigma(x)); interior critical points
(quartic p^4 - (2+B) p^3 + (1+2B) p^2 - B p - A^2 = 0 with
tanh y = A / (p (1-p)) — kept when |.| < 1 and (x, y) in the box).
For g = x sigma(y) - A x - B y: corners + vertical-edge stationary points
(p^2 - p + B / x_e = 0, p = sigma(y)); the interior has only saddles.
"""

from __future__ import annotations

from mpmath import atanh, exp, im, mp, mpf, polyroots, re, sqrt, tanh

mp.dps = 50


def _sig(x):
    return 1 / (1 + exp(-x))


def _g_sigtanh(x, y, A, B):
    return _sig(x) * tanh(y) - A * x - B * y


def _g_sigid(x, y, A, B):
    return x * _sig(y) - A * x - B * y


def _quad_roots(c):
    """Real roots of p^2 - p + c = 0 in (0, 1)."""
    disc = 1 - 4 * c
    if disc < 0:
        return []
    s = sqrt(disc)
    return [p for p in ((1 - s) / 2, (1 + s) / 2) if 0 < p < 1]


def true_minmax_sigtanh(A, B, lx, ux, ly, uy):
    """50-digit (min, max) of g over the box."""
    A = mpf(repr(A)); B = mpf(repr(B))
    lx = mpf(repr(lx)); ux = mpf(repr(ux))
    ly = mpf(repr(ly)); uy = mpf(repr(uy))
    cands = [_g_sigtanh(x, y, A, B)
             for x in (lx, ux) for y in (ly, uy)]
    # vertical edges: sigma(x_e) (1 - tanh^2 y) = B
    for x_e in (lx, ux):
        s = _sig(x_e)
        if s > 0:
            t2 = 1 - B / s
            if 0 <= t2 < 1:
                t = sqrt(t2)
                for tv in (t, -t):
                    y = atanh(tv)
                    if ly <= y <= uy:
                        cands.append(_g_sigtanh(x_e, y, A, B))
    # horizontal edges: p(1-p) tanh(y_e) = A
    for y_e in (ly, uy):
        ty = tanh(y_e)
        if ty != 0:
            for p in _quad_roots(A / ty):
                x = mp.log(p / (1 - p))
                if lx <= x <= ux:
                    cands.append(_g_sigtanh(x, y_e, A, B))
    # interior: quartic in p
    for r in polyroots([mpf(1), -(2 + B), 1 + 2 * B, -B, -(A ** 2)],
                       maxsteps=300, extraprec=200):
        if abs(im(r)) > mpf("1e-35"):
            continue
        p = re(r)
        if not (0 < p < 1):
            continue
        q = p * (1 - p)
        if q == 0:
            continue
        u = A / q
        if abs(u) >= 1:
            continue
        x = mp.log(p / (1 - p))
        y = atanh(u)
        if lx <= x <= ux and ly <= y <= uy:
            cands.append(_g_sigtanh(x, y, A, B))
    return min(cands), max(cands)


def true_minmax_sigid(A, B, lx, ux, ly, uy):
    """50-digit (min, max) of x sigma(y) - A x - B y over the box."""
    A = mpf(repr(A)); B = mpf(repr(B))
    lx = mpf(repr(lx)); ux = mpf(repr(ux))
    ly = mpf(repr(ly)); uy = mpf(repr(uy))
    cands = [_g_sigid(x, y, A, B)
             for x in (lx, ux) for y in (ly, uy)]
    for x_e in (lx, ux):
        if x_e != 0:
            for p in _quad_roots(B / x_e):
                y = mp.log(p / (1 - p))
                if ly <= y <= uy:
                    cands.append(_g_sigid(x_e, y, A, B))
    return min(cands), max(cands)
