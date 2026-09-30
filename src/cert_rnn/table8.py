"""Faithful Table 8 (Du et al., CCS 2021, Appendix B) tilt selection for
the sigma(x)*tanh(y) transformer.

Every Table 8 case prescribes a plane TILT (A, B) and states where the
residual g = f - Ax - By attains its extrema for that tilt (its C1/C2
formulas). The shipped transformer computes the exact residual extrema for
ANY tilt (cert_rnn.certified), so the faithful baseline is: Table 8's
per-case tilt + exact offsets. The paper's offset-location claims are
checked separately as an oracle (tests/soundness/test_table8.py) — they
are not needed for soundness and one of them (case 2's C1/C2 labelling)
is inconsistent with its own proof.

Transcription (page 16 of the CCS'21 PDF, verified visually at 300 dpi):
  1  lx >= 0, ly >= 0            : corner-fit  A = (s_u - s_l)(t_l + t_u)/(2 wx),
                                                B = (s_u + s_l)(t_u - t_l)/(2 wy)
  2  ux <= 0, ly >= 0            : FIVE listed sub-solutions (B.2.1-B.2.5):
       (1) A = (f(ux,uy) - f(lx,uy))/wx,  B = df/dy(lx,uy)
       (2) A = df/dx(ux,ly),  B = (f(x',uy) - f(ux,ly))/wy if A > df/dx(lx,uy)
                                 else (f(lx,uy) - f(ux,ly))/wy,
           x' = tangent point of slope-A line to z = sigma(x) tanh(uy), x' < ux
       (3) B = (f(ux,uy) - f(ux,ly))/wy,
           A = (B(ly - y') + f(ux,y') - f(lx,ly))/wx      if B >  df/dy(lx,ly)
               (B(y'' - y') + f(ux,y') - f(lx,y''))/wx    if B <= df/dy(lx,ly)
           y'  = tangent point of slope-B line to z = sigma(ux) tanh(y)
           y'' = tangent point of slope-B line to z = sigma(lx) tanh(y)
       (4) as (3) but B = (f(ux,uy)-f(ux,ly)+f(lx,uy)-f(lx,ly))/(2 wy)
       (5) corner-fit tilt (identical to case 1's formula)
     No selection rule is stated; this module returns ALL five and the
     transformer keeps the tightest (smallest exact C2 - C1). Since (5) is
     corner-fit, case 2 is never worse than corner-fit by construction.
  3  ux <= 0, uy <= 0            : "symmetric to case 2": y-reflection.
     f(x,-y) = -f(x,y) exactly, so apply case 2 to (lx,ux,-uy,-ly) and
     map (A,B) -> (-A, B).
  4  lx >= 0, uy <= 0            : "symmetric to case 1": y-reflection of
     corner-fit = corner-fit on the original box (identity).
  5  lx >= 0, ly < 0 < uy        : A = 0,
     B = min{(f(ux,uy)-f(lx,ly))/wy, (f(lx,uy)-f(ux,ly))/wy}
  6  ux <= 0, ly < 0 < uy        : "symmetric to case 5". The x-reflection
     f(-x,y) = tanh(y) - f(x,y) is NOT an exact symmetry, so "the same way"
     is read as the same construction: A = 0, same B formula.
  7  ly >= 0, lx < 0 < ux        : A = (f(ux,ly) - f(lx,ly))/wx,
                                   B = (f(lx,uy) - f(lx,ly))/wy
  8  uy <= 0, lx < 0 < ux        : y-reflection of case 7 (exact).
  9  lx < 0 < ux, ly < 0 < uy    : "same method as cases 5 and 6":
                                   A = 0, same B formula.
Boundary ties (lx == 0 etc.) fall to the first matching case in the order
1..9 above; the paper's conditions overlap on the boundaries.
"""

from __future__ import annotations

import numpy as np

_S = lambda x: 1.0 / (1.0 + np.exp(-x))   # noqa: E731


def _f(x, y):
    return _S(x) * np.tanh(y)


def _fx(x, y):
    s = _S(x)
    return s * (1 - s) * np.tanh(y)


def _fy(x, y):
    t = np.tanh(y)
    return _S(x) * (1 - t * t)


def _cornerfit(lx, ux, ly, uy):
    sl, su = _S(lx), _S(ux); tl, tu = np.tanh(ly), np.tanh(uy)
    return ((su - sl) * (tl + tu) / (2 * (ux - lx)),
            (sl + su) * (tu - tl) / (2 * (uy - ly)))


def _tangent_x_on_top(A, uy, lx, ux):
    """x' with sigma'(x') tanh(uy) = A, i.e. p(1-p) = A/tanh(uy); the root
    with x' < ux (proof B.2.2). None if no real tangency."""
    ty = np.tanh(uy)
    if ty == 0:
        return None
    c = A / ty
    if not (0 < c <= 0.25):
        return None
    s = np.sqrt(1 - 4 * c)
    cands = [np.log(p / (1 - p)) for p in ((1 - s) / 2, (1 + s) / 2) if 0 < p < 1]
    cands = [x for x in cands if x < ux]
    return max(cands) if cands else None


def _tangent_y(B, x_e, ly, uy):
    """y with sigma(x_e) tanh'(y) = B, i.e. tanh^2 y = 1 - B/sigma(x_e);
    picks the in-box root (prefers the one inside [ly, uy]); None if none."""
    s = _S(x_e)
    if s <= 0:
        return None
    r = 1 - B / s
    if not (0 <= r < 1):
        return None
    t = np.sqrt(r)
    cands = [np.arctanh(t), -np.arctanh(t)]
    inside = [y for y in cands if ly <= y <= uy]
    if inside:
        return inside[0]
    return None


def case_id(lx, ux, ly, uy) -> int:
    if lx >= 0 and ly >= 0:
        return 1
    if ux <= 0 and ly >= 0:
        return 2
    if ux <= 0 and uy <= 0:
        return 3
    if lx >= 0 and uy <= 0:
        return 4
    if lx >= 0 and ly < 0 < uy:
        return 5
    if ux <= 0 and ly < 0 < uy:
        return 6
    if ly >= 0 and lx < 0 < ux:
        return 7
    if uy <= 0 and lx < 0 < ux:
        return 8
    return 9


def _case2_tilts(lx, ux, ly, uy):
    """The five listed sub-solutions of case 2 (ux <= 0, ly >= 0)."""
    wx, wy = ux - lx, uy - ly
    out = []
    # (1)
    out.append(("2.1", (_f(ux, uy) - _f(lx, uy)) / wx, _fy(lx, uy)))
    # (2)
    A = _fx(ux, ly)
    if A > _fx(lx, uy):
        xp = _tangent_x_on_top(A, uy, lx, ux)
        if xp is not None:
            out.append(("2.2", A, (_f(xp, uy) - _f(ux, ly)) / wy))
    else:
        out.append(("2.2", A, (_f(lx, uy) - _f(ux, ly)) / wy))
    # (3) and (4)
    for tag, B in (("2.3", (_f(ux, uy) - _f(ux, ly)) / wy),
                   ("2.4", (_f(ux, uy) - _f(ux, ly) + _f(lx, uy) - _f(lx, ly)) / (2 * wy))):
        yp = _tangent_y(B, ux, ly, uy)
        if yp is None:
            continue
        if B > _fy(lx, ly):
            A = (B * (ly - yp) + _f(ux, yp) - _f(lx, ly)) / wx
        else:
            ypp = _tangent_y(B, lx, ly, uy)
            if ypp is None:
                continue
            A = (B * (ypp - yp) + _f(ux, yp) - _f(lx, ypp)) / wx
        out.append((tag, A, B))
    # (5)
    A, B = _cornerfit(lx, ux, ly, uy)
    out.append(("2.5", A, B))
    return out


def table8_tilts(lx, ux, ly, uy):
    """All Table 8 candidate tilts for a non-degenerate box.
    Returns (case, [(tag, A, B), ...])."""
    c = case_id(lx, ux, ly, uy)
    wx, wy = ux - lx, uy - ly
    if c in (1, 4):
        A, B = _cornerfit(lx, ux, ly, uy)
        return c, [(str(c), A, B)]
    if c == 2:
        return c, _case2_tilts(lx, ux, ly, uy)
    if c == 3:
        # y-reflection: solve case 2 on (lx, ux, -uy, -ly), map A -> -A
        sub = _case2_tilts(lx, ux, -uy, -ly)
        return c, [("3(" + t + ")", -A, B) for t, A, B in sub]
    if c in (5, 6, 9):
        B = min((_f(ux, uy) - _f(lx, ly)) / wy, (_f(lx, uy) - _f(ux, ly)) / wy)
        return c, [(str(c), 0.0, B)]
    if c == 7:
        return c, [("7", (_f(ux, ly) - _f(lx, ly)) / wx, (_f(lx, uy) - _f(lx, ly)) / wy)]
    # c == 8: y-reflection of case 7
    A7 = (_f(ux, -uy) - _f(lx, -uy)) / wx
    B7 = (_f(lx, -ly) - _f(lx, -uy)) / wy
    return c, [("8", -A7, B7)]
