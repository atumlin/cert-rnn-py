"""R-tests: is the (C1, C2) candidate enumeration complete? (soundness.md,
revised: R-1/R-2/R-3 replace T-0.1/0.2/0.3/0.5; T-0.4 kept.)

All tests run BOTH transcriptions (scalar `_c1c2_*` and batched
`_c1c2_*_batch`) and BOTH bilinears. Box strata: the nine sign strata,
saturation/tight/wide/near-degenerate width regimes, and — FIRST — the
adversarial near-degenerate-quartic stratum. Violations are never pooled:
they are binned by the quartic conditioning number 1/(p(1-p)) of the
worst-conditioned interior root, and reported as a magnitude distribution
per bin (relative to max(1, |C|)).

The tests fail on any violation beyond fp noise; the failure message IS the
report. The `nightly` variants scale the budget ~50x.
"""

from __future__ import annotations

import numpy as np
import pytest

from cert_rnn.transformers import (
    _c1c2_sigid,
    _c1c2_sigid_batch,
    _c1c2_sigtanh,
    _c1c2_sigtanh_batch,
    _sigid_plane,
    _sigmoid,
    _sigtanh_plane,
)

from tests.soundness.boxes import (
    near_degenerate_quartic_boxes,
    quartic_conditioning,
    stratified_boxes,
)

SEED = 20260814
FP_TOL = 1e-13         # relative; violations below this are fp noise
# R-1 looseness side: certified interval candidates on a sub-box's new
# edges may be wider than the parent's slack (see _run_r1). Budget for the
# child-looser-than-parent excess; larger = enclosure blow-up.
CHILD_SLACK_TOL = 1e-6
RETIRE_KNOWN = True    # F-1/F-2/F-3 fixed: known signatures no longer xfail
COND_BINS = (0, 10, 100, 1e3, 1e4, 1e5, np.inf)

# ---------- shared helpers ----------


def _g(kind, x, y, A, B):
    if kind == "sigtanh":
        return _sigmoid(x) * np.tanh(y) - A * x - B * y
    return x * _sigmoid(y) - A * x - B * y


def _tilt(kind, lx, ux, ly, uy):
    plane = _sigtanh_plane if kind == "sigtanh" else _sigid_plane
    A = np.empty_like(lx); B = np.empty_like(lx)
    for i in range(lx.shape[0]):
        A[i], B[i], _, _ = plane(lx[i], ux[i], ly[i], uy[i])
    return A, B


def _c1c2(kind, path, A, B, lx, ux, ly, uy):
    if path == "batch":
        fn = _c1c2_sigtanh_batch if kind == "sigtanh" else _c1c2_sigid_batch
        return fn(A, B, lx, ux, ly, uy)
    fn = _c1c2_sigtanh if kind == "sigtanh" else _c1c2_sigid
    C1 = np.empty_like(A); C2 = np.empty_like(A)
    for i in range(A.shape[0]):
        C1[i], C2[i] = fn(A[i], B[i], lx[i], ux[i], ly[i], uy[i])
    return C1, C2


def _all_boxes(rng, n_per, n_adv):
    """Ordered dict: adversarial stratum FIRST, then the standard strata."""
    out = {"near_degenerate_quartic": near_degenerate_quartic_boxes(rng, n_adv)}
    out.update(stratified_boxes(rng, n_per))
    return out


def _cond_bin(c):
    if not np.isfinite(c):
        return "no-interior-root"
    for lo, hi in zip(COND_BINS[:-1], COND_BINS[1:]):
        if lo <= c < hi:
            return f"cond[{lo:g},{hi:g})"
    return "?"


def is_f1_signature(A, B, lx, ux, ly, uy):
    """Finding F-1 (docs/phase0_findings.md): the sigma*tanh interior
    critical point is DROPPED or MISPLACED by float64 root recovery when
    the quartic has a near-double root (p -> 0 or 1). Decide with a
    50-digit recomputation: is there a true interior critical point in the
    box whose float64 counterpart is rejected by a gate (|imag|, p-range,
    |A/(p(1-p))| >= 1, in-box) or evaluated more than 1e-13 away? Cost is
    per-violation only.
    """
    from mpmath import atanh, im, log, mp, mpf, polyroots, re

    mp.dps = 50
    try:
        roots = polyroots([mpf(1), -(2 + mpf(B)), 1 + 2 * mpf(B), -mpf(B),
                           -mpf(A) ** 2], maxsteps=200, extraprec=100)
    except Exception:
        return False
    # float64 pipeline's accepted critical points (mirrors transformers.py)
    f64 = np.roots([1.0, -(2.0 + B), 1.0 + 2.0 * B, -B, -(A ** 2)])
    accepted = []
    for r in f64:
        if abs(r.imag) >= 1e-10:
            continue
        p = r.real
        if not (1e-12 < p < 1 - 1e-12):
            continue
        ry = A / (p * (1 - p))
        if abs(ry) >= 1 - 1e-12:
            continue
        x = np.log(p / (1 - p)); y = np.arctanh(ry)
        if lx - 1e-12 <= x <= ux + 1e-12 and ly - 1e-12 <= y <= uy + 1e-12:
            accepted.append((x, y))
    for r in roots:
        if abs(im(r)) > mpf("1e-30"):
            continue
        p = re(r)
        if not (0 < p < 1):
            continue
        ry = mpf(A) / (p * (1 - p))
        if abs(ry) >= 1:
            continue
        x = float(log(p / (1 - p))); y = float(atanh(ry))
        if not (lx <= x <= ux and ly <= y <= uy):
            continue
        # a true in-box critical point exists; did float64 keep it closely?
        if not any(abs(ax - x) < 1e-6 * max(1, abs(x)) and
                   abs(ay - y) < 1e-6 * max(1, abs(y)) for ax, ay in accepted):
            return True
    return False


DEEP_SAT_X = 27.5   # sigma(x) < 1e-12 for x < -27.6; the 1e-12 gates bite


def is_f3_signature(rel_abs, lx, ux):
    """Finding F-3: the `ratio > 1e-12` / `p > 1e-12` / `p < 1-1e-12`
    gates on edge/interior candidates reject legitimate stationary points
    when sigma(x) or 1-sigma(x) < 1e-12, i.e. |x| > ~27.6. Absolute
    magnitude is bounded by the residual variation in that regime
    (~1e-11)."""
    return rel_abs <= 1e-11 and (lx < -DEEP_SAT_X or ux > DEEP_SAT_X)


class ViolationLedger:
    """Collect (stratum, cond_bin, rel_magnitude, detail) and format.
    Each violation carries a `known` label: "" (new), "F-1", "F-2", "F-3"."""

    def __init__(self, title):
        self.title = title
        self.rows = []
        self.n_checked = 0
        self.known = []   # parallel to rows
        self.child_slack = []   # R-1 looseness-side excesses (informational)

    def add(self, stratum, cond, rel, detail, f1=False, known=""):
        self.rows.append((stratum, _cond_bin(cond), float(rel), detail))
        self.known.append("F-1" if f1 else known)

    @property
    def f1_flags(self):
        return [k != "" for k in self.known]

    def finish(self):
        """Raise per soundness policy: nothing -> pass; only known-signature
        violations -> xfail with the full magnitude report; anything else
        -> fail loudly."""
        if not self.rows:
            return
        rep = self.report()
        # F-1/F-2/F-3 were FIXED (cert_rnn.certified + F-2 slack); their
        # signatures are still labelled in the report for diagnosis but no
        # longer excuse a failure. RETIRE_KNOWN=False restores the xfail
        # behaviour for archaeology only.
        if not RETIRE_KNOWN and all(k != "" for k in self.known):
            pytest.xfail("known signatures (retired mode off):\n" + rep)
        pytest.fail(f"{len(self.rows)} violation(s):\n" + rep)

    def report(self):
        n_f1 = sum(self.f1_flags)
        lines = [f"{self.title}: {len(self.rows)} violations > {FP_TOL:g} rel "
                 f"in {self.n_checked} checks ({n_f1} F-1 signature, "
                 f"{len(self.rows) - n_f1} other)"]
        if self.child_slack:
            cs = np.array(self.child_slack)
            lines.append(f"  [info] child-looser-than-parent (certified enclosure "
                         f"slack on sub-box edges): n={cs.size} max={cs.max():.3e} "
                         f"med={np.median(cs):.3e}")
        bins = {}
        for (s, b, r, _), f in zip(self.rows, self.f1_flags):
            bins.setdefault((s, b), []).append((r, f))
        for (s, b), rf in sorted(bins.items(), key=lambda kv: -max(x[0] for x in kv[1])):
            rs = np.array([x[0] for x in rf]); nf = sum(x[1] for x in rf)
            lines.append(f"  [{s} | {b}] n={rs.size} (F-1: {nf}) max={rs.max():.3e} "
                         f"med={np.median(rs):.3e} min={rs.min():.3e}")
        for (s, b, r, d), k in sorted(zip(self.rows, self.known),
                                      key=lambda x: -x[0][2])[:6]:
            lines.append(f"    {s} {b} rel={r:.3e} {k or 'OTHER'} {d}")
        return "\n".join(lines)


def _run_r1(kind, path, boxes, depth):
    """R-1: with the parent's (A,B), the min/max over the 4 sub-boxes must
    EQUAL the parent's (two-sided). Recurse `depth` levels."""
    led = ViolationLedger(f"R-1 subdivision {kind}/{path}")
    for stratum, (lx, ux, ly, uy) in boxes.items():
        A, B = _tilt(kind, lx, ux, ly, uy)
        _, cond = quartic_conditioning(A, B) if kind == "sigtanh" \
            else (None, np.full(lx.shape, np.nan))
        # level list of boxes: start with parents
        cur = [(lx, ux, ly, uy)]
        for _lvl in range(depth):
            nxt = []
            for (plx, pux, ply, puy) in cur:
                C1p, C2p = _c1c2(kind, path, A, B, plx, pux, ply, puy)
                mx = 0.5 * (plx + pux); my = 0.5 * (ply + puy)
                subs = [(plx, mx, ply, my), (mx, pux, ply, my),
                        (plx, mx, my, puy), (mx, pux, my, puy)]
                C1s = []; C2s = []
                for s in subs:
                    c1, c2 = _c1c2(kind, path, A, B, *s)
                    C1s.append(c1); C2s.append(c2)
                    nxt.append(s)
                C1s = np.min(np.stack(C1s), axis=0)
                C2s = np.max(np.stack(C2s), axis=0)
                led.n_checked += lx.shape[0]
                sc1 = np.maximum(1.0, np.abs(C1p))
                sc2 = np.maximum(1.0, np.abs(C2p))
                # Soundness side: child below parent min (or above parent
                # max) => the parent MISSED an extremum. Hard, tol FP_TOL.
                # Looseness side: child above parent min (or below parent
                # max). With exact point candidates this was an equality;
                # with the certified interval candidates (cert_rnn.certified)
                # a sub-box's NEW interior edge can carry an ill-conditioned
                # stationary candidate whose sound enclosure is wider than
                # the parent's slack, so the child may legitimately be
                # LOOSER by up to the enclosure width. Budget CHILD_SLACK_TOL;
                # anything beyond it is an enclosure blow-up bug.
                for arr_signed, name in ((C1s - C1p, "C1"), (C2p - C2s, "C2")):
                    sc = sc1 if name == "C1" else sc2
                    rel = arr_signed / sc          # < 0: parent missed; > 0: child looser
                    for i in np.flatnonzero(rel < -FP_TOL):
                        f1 = (kind == "sigtanh" and
                              is_f1_signature(A[i], B[i], plx[i], pux[i], ply[i], puy[i]))
                        absv = -rel[i] * sc[i]
                        known = "F-3" if (not f1 and is_f3_signature(absv, plx[i], pux[i])) else ""
                        led.add(stratum, cond[i], -rel[i],
                                f"{name} parent-missed box=({plx[i]:.6g},{pux[i]:.6g},"
                                f"{ply[i]:.6g},{puy[i]:.6g}) A={A[i]:.6g} B={B[i]:.6g}",
                                f1=f1, known=known)
                    for i in np.flatnonzero(rel > CHILD_SLACK_TOL):
                        led.add(stratum, cond[i], rel[i],
                                f"{name} child-looser-than-budget box=({plx[i]:.6g},"
                                f"{pux[i]:.6g},{ply[i]:.6g},{puy[i]:.6g})")
                    led.child_slack.extend(rel[rel > FP_TOL].tolist())
            cur = nxt
    return led


def _run_r3(kind, path, boxes, n_grid, rng, n_refine=200):
    """R-3: dense grid + local refinement oracle. Any grid/refined point
    outside [C1, C2] is a direct counterexample; magnitude recorded."""
    led = ViolationLedger(f"R-3 grid oracle {kind}/{path}")
    s = np.linspace(0.0, 1.0, n_grid)
    for stratum, (lx, ux, ly, uy) in boxes.items():
        A, B = _tilt(kind, lx, ux, ly, uy)
        _, cond = quartic_conditioning(A, B) if kind == "sigtanh" \
            else (None, np.full(lx.shape, np.nan))
        C1, C2 = _c1c2(kind, path, A, B, lx, ux, ly, uy)
        for i in range(lx.shape[0]):
            xs = lx[i] + s * (ux[i] - lx[i])
            ys = ly[i] + s * (uy[i] - ly[i])
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            Xf = X.ravel(); Yf = Y.ravel()
            Gf = _g(kind, Xf, Yf, A[i], B[i])
            extra = [Gf]
            # local refinement: random jitter around the best grid points
            hx = (ux[i] - lx[i]) / (n_grid - 1)
            hy = (uy[i] - ly[i]) / (n_grid - 1)
            for sign in (-1.0, 1.0):
                flat = np.argsort(sign * Gf)[-3:]
                bx = Xf[flat]; by = Yf[flat]
                jx = np.clip(bx[:, None] + rng.uniform(-hx, hx, (3, n_refine)),
                             lx[i], ux[i])
                jy = np.clip(by[:, None] + rng.uniform(-hy, hy, (3, n_refine)),
                             ly[i], uy[i])
                extra.append(_g(kind, jx, jy, A[i], B[i]).ravel())
            G = np.concatenate(extra)
            gmin = float(np.min(G)); gmax = float(np.max(G))
            led.n_checked += 1
            r1 = (C1[i] - gmin) / max(1.0, abs(C1[i]))
            r2 = (gmax - C2[i]) / max(1.0, abs(C2[i]))
            det = (f"box=({lx[i]:.6g},{ux[i]:.6g},{ly[i]:.6g},{uy[i]:.6g}) "
                   f"A={A[i]:.6g} B={B[i]:.6g}")
            if r1 > FP_TOL or r2 > FP_TOL:
                f1 = (kind == "sigtanh" and
                      is_f1_signature(A[i], B[i], lx[i], ux[i], ly[i], uy[i]))
            if r1 > FP_TOL:
                k3 = "F-3" if (not f1 and is_f3_signature(C1[i] - gmin, lx[i], ux[i])) else ""
                led.add(stratum, cond[i], r1, f"C1 above true min by {C1[i]-gmin:.3e} " + det, f1=f1, known=k3)
            if r2 > FP_TOL:
                k3 = "F-3" if (not f1 and is_f3_signature(gmax - C2[i], lx[i], ux[i])) else ""
                led.add(stratum, cond[i], r2, f"C2 below true max by {gmax-C2[i]:.3e} " + det, f1=f1, known=k3)
    return led


# ---------- tests ----------

FAST = dict(n_per=40, n_adv=120, r1_depth=3, r3_grid=101)
NIGHTLY = dict(n_per=1500, n_adv=6000, r1_depth=4, r3_grid=301)


def _params(nightly):
    return NIGHTLY if nightly else FAST


@pytest.mark.soundness
@pytest.mark.parametrize("kind", ["sigtanh", "sigid"])
@pytest.mark.parametrize("path", ["scalar", "batch"])
def test_r1_subdivision(kind, path):
    rng = np.random.default_rng(SEED)
    P = _params(False)
    boxes = _all_boxes(rng, P["n_per"], P["n_adv"])
    led = _run_r1(kind, path, boxes, P["r1_depth"])
    print(f"\n[seed={SEED}] " + led.report())
    led.finish()


@pytest.mark.soundness
@pytest.mark.parametrize("kind", ["sigtanh", "sigid"])
@pytest.mark.parametrize("path", ["scalar", "batch"])
def test_r3_grid_oracle(kind, path):
    rng = np.random.default_rng(SEED)
    P = _params(False)
    boxes = _all_boxes(rng, P["n_per"], P["n_adv"])
    led = _run_r3(kind, path, boxes, P["r3_grid"], rng)
    print(f"\n[seed={SEED}] " + led.report())
    led.finish()


@pytest.mark.soundness
@pytest.mark.parametrize("path", ["scalar", "batch"])
def test_r2_odd_symmetry_sigtanh(path):
    """R-2: tanh odd => f(x,-y) = -f(x,y). For the y-reflected box
    (lx,ux,-uy,-ly) the tilt must satisfy A'=-A, B'=B (derived
    independently from the corner-fit formula) and C1'=-C2, C2'=-C1."""
    rng = np.random.default_rng(SEED)
    boxes = _all_boxes(rng, 60, 200)
    led = ViolationLedger(f"R-2 odd symmetry sigtanh/{path}")
    for stratum, (lx, ux, ly, uy) in boxes.items():
        A, B = _tilt("sigtanh", lx, ux, ly, uy)
        Ar, Br = _tilt("sigtanh", lx, ux, -uy, -ly)
        # independent derivation of the reflected tilt from the corner-fit
        # formula (non-degenerate widths only; the code's point-width
        # branch uses a different, 1-D formula which the A'=-A / B'=B
        # assertions below still cover)
        nd = ((ux - lx) >= 1e-12) & ((uy - ly) >= 1e-12)
        sl = _sigmoid(lx); su = _sigmoid(ux)
        with np.errstate(divide="ignore", invalid="ignore"):
            A_exp = (su - sl) * (np.tanh(-uy) + np.tanh(-ly)) / (2 * (ux - lx))
            B_exp = (sl + su) * (np.tanh(-ly) - np.tanh(-uy)) / (2 * (uy - ly))
        np.testing.assert_allclose(Ar[nd], A_exp[nd], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(Br[nd], B_exp[nd], rtol=1e-12, atol=1e-15)
        # atol 1e-12: the point-width branch evaluates tanh at the OTHER
        # endpoint after reflection (ly vs -uy of a ~1e-13-wide interval)
        np.testing.assert_allclose(Ar, -A, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(Br, B, rtol=1e-12, atol=1e-12)
        C1, C2 = _c1c2("sigtanh", path, A, B, lx, ux, ly, uy)
        C1r, C2r = _c1c2("sigtanh", path, Ar, Br, lx, ux, -uy, -ly)
        _, cond = quartic_conditioning(A, B)
        led.n_checked += lx.shape[0]
        for i in range(lx.shape[0]):
            r = max(abs(C1r[i] + C2[i]) / max(1, abs(C2[i])),
                    abs(C2r[i] + C1[i]) / max(1, abs(C1[i])))
            if r > 1e-12:
                led.add(stratum, cond[i], r,
                        f"box=({lx[i]:.6g},{ux[i]:.6g},{ly[i]:.6g},{uy[i]:.6g}) "
                        f"C=({C1[i]:.12g},{C2[i]:.12g}) reflected=({C1r[i]:.12g},{C2r[i]:.12g})")
    print(f"\n[seed={SEED}] " + led.report())
    led.finish()


@pytest.mark.soundness
@pytest.mark.parametrize("kind", ["sigtanh", "sigid"])
def test_t04_degenerate_boxes(kind):
    """T-0.4: zero width, endpoint exactly 0, widths straddling the 1e-12
    point gate, saturation, very wide, mixed. Checks: (a) scalar==batch,
    (b) grid enclosure, (c) finite outputs. Uses the full plane functions
    (which own the degenerate dispatch)."""
    from cert_rnn.transformers import _sigid_plane_batch, _sigtanh_plane_batch
    plane = _sigtanh_plane if kind == "sigtanh" else _sigid_plane
    pbatch = _sigtanh_plane_batch if kind == "sigtanh" else _sigid_plane_batch
    rng = np.random.default_rng(SEED)
    cases = []
    for c in (-3.0, -0.5, 0.0, 0.5, 3.0, 12.0, -12.0):
        for w in (0.0, 1e-13, 5e-13, 2e-12, 1e-11, 1e-9, 1e-6, 1.0, 40.0):
            cases.append((c - w / 2, c + w / 2))
    cases += [(0.0, 1.0), (-1.0, 0.0), (0.0, 0.0), (0.0, 30.0), (-30.0, 0.0),
              (-50.0, 50.0), (10.0, 35.0), (-35.0, -10.0)]
    xi = rng.integers(0, len(cases), 400)
    yi = rng.integers(0, len(cases), 400)
    lx = np.array([cases[i][0] for i in xi]); ux = np.array([cases[i][1] for i in xi])
    ly = np.array([cases[i][0] for i in yi]); uy = np.array([cases[i][1] for i in yi])
    A = np.empty(400); B = np.empty(400); C1 = np.empty(400); C2 = np.empty(400)
    for i in range(400):
        A[i], B[i], C1[i], C2[i] = plane(lx[i], ux[i], ly[i], uy[i])
    Ab, Bb, C1b, C2b = pbatch(lx, ux, ly, uy)
    led = ViolationLedger(f"T-0.4 degenerate boxes {kind}")
    for name, sc, bt in (("A", A, Ab), ("B", B, Bb), ("C1", C1, C1b), ("C2", C2, C2b)):
        assert np.all(np.isfinite(sc)) and np.all(np.isfinite(bt)), name
        rel = np.abs(sc - bt) / np.maximum(1.0, np.maximum(np.abs(sc), np.abs(bt)))
        for i in np.flatnonzero(rel > 1e-11):
            f1 = kind == "sigtanh" and is_f1_signature(A[i], B[i], lx[i], ux[i], ly[i], uy[i])
            led.add("degenerate", np.nan, rel[i],
                    f"scalar/batch {name} disagree box=({lx[i]},{ux[i]},{ly[i]},{uy[i]}) "
                    f"scalar={sc[i]!r} batch={bt[i]!r}", f1=f1)
    # enclosure on a grid (incl. exact endpoints)
    s_ = np.linspace(0, 1, 61)
    for i in range(400):
        xs = lx[i] + s_ * (ux[i] - lx[i]); ys = ly[i] + s_ * (uy[i] - ly[i])
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        G = _g(kind, X, Y, A[i], B[i])
        r = max((C1[i] - G.min()) / max(1, abs(C1[i])),
                (G.max() - C2[i]) / max(1, abs(C2[i])))
        led.n_checked += 1
        if r > FP_TOL:
            wx = ux[i] - lx[i]; wy = uy[i] - ly[i]
            # F-2: a nonzero width below the 1e-12 point gate is treated
            # as a point, dropping a real error of at most
            # max|x| * 0.25 * width (sigid) / 0.25 * width (sigtanh); no
            # outward rounding absorbs it. Distinct from F-1.
            f2 = (0 < wx < 1e-12) or (0 < wy < 1e-12)
            f1 = kind == "sigtanh" and is_f1_signature(A[i], B[i], lx[i], ux[i], ly[i], uy[i])
            led.add("degenerate", np.nan, r,
                    f"enclosure box=({lx[i]},{ux[i]},{ly[i]},{uy[i]})",
                    f1=f1, known=("F-2" if (f2 and not f1) else ""))
    print(f"\n[seed={SEED}] " + led.report())
    led.finish()


# ---------- nightly-scale ----------

@pytest.mark.nightly
@pytest.mark.parametrize("kind", ["sigtanh", "sigid"])
@pytest.mark.parametrize("path", ["scalar", "batch"])
def test_r1_subdivision_nightly(kind, path):
    rng = np.random.default_rng(SEED + 1)
    P = _params(True)
    boxes = _all_boxes(rng, P["n_per"], P["n_adv"])
    led = _run_r1(kind, path, boxes, P["r1_depth"])
    print(f"\n[seed={SEED+1}] " + led.report())
    led.finish()


@pytest.mark.nightly
@pytest.mark.parametrize("kind", ["sigtanh", "sigid"])
@pytest.mark.parametrize("path", ["scalar", "batch"])
def test_r3_grid_oracle_nightly(kind, path):
    rng = np.random.default_rng(SEED + 1)
    P = _params(True)
    boxes = _all_boxes(rng, P["n_per"], P["n_adv"])
    led = _run_r3(kind, path, boxes, P["r3_grid"], rng)
    print(f"\n[seed={SEED+1}] " + led.report())
    led.finish()


# ---------- mutation canary (soundness.md §3): the suite must be able to fail ----------

@pytest.mark.soundness
def test_mutation_dropped_interior_candidates_is_caught(monkeypatch):
    """Deliberate defect: drop every interior critical-point candidate of
    sigma*tanh. R-3 on the near-degenerate/wide-saturated strata MUST
    report violations; if it does not, the harness is not measuring what
    it claims."""
    import cert_rnn.certified as cert

    def no_interior(A, B, lx, ux, ly, uy):
        K = A.shape[0]
        return np.full((K, 8), np.nan), np.full((K, 8), np.nan)

    monkeypatch.setattr(cert, "sigtanh_interior_candidates", no_interior)
    rng = np.random.default_rng(SEED)
    boxes = _all_boxes(rng, 30, 60)
    led = _run_r3("sigtanh", "batch", boxes, 61, rng)
    print(f"\n[seed={SEED}] mutation canary: {len(led.rows)} violations")
    assert led.rows, "mutation (dropped interior candidates) SURVIVED R-3"
    assert max(r[2] for r in led.rows) > 1e-6, "mutation caught only at fp scale"
