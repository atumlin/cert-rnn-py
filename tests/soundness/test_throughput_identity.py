"""Phase 1 throughput (2a parallel T loop, 2b k-ary search) must be a pure
reordering of identical arithmetic: certified radii come out BIT-IDENTICAL
to the sequential Algorithm-1 implementation. Exact equality, not approx.

Also checks: k-ary with 1 probe/round IS bisection; on synthetic monotone
oracles k-ary and bisection return the same grid point for random
thresholds; and the parallel path's predicate ids are offset but the
resulting zonotopes are identical (allocator-order invariance).
"""

from __future__ import annotations

import numpy as np
import pytest

from cert_rnn.verify import (
    _search_frames,
    bisect_epsilon,
    certify_radius_spec_c,
    kary_epsilon,
    lstm_ae_reach,
    spec_c_score_ub,
)
from cert_rnn.zono import reset_pred_allocator

SEED = 20260815


@pytest.mark.soundness
def test_kary_equals_bisection_on_monotone_oracles():
    rng = np.random.default_rng(SEED)
    for _ in range(300):
        thr = rng.uniform(-0.1, 1.1)
        oracle = lambda e, thr=thr: e <= thr
        b = bisect_epsilon(oracle, 0.5, 12)
        for probes in (1, 3, 7, 15, 31, 127):
            k, rounds = kary_epsilon(lambda es, o=oracle: [o(e) for e in es],
                                     0.5, 12, probes)
            assert k == b, (thr, probes, k, b)
    # sequential depth arithmetic
    assert kary_epsilon(lambda es: [True] * len(es), 0.5, 12, 15)[1] == 4
    assert kary_epsilon(lambda es: [True] * len(es), 0.5, 12, 1)[1] == 13
    assert kary_epsilon(lambda es: [True] * len(es), 0.5, 12, 31)[1] == 3


def _small_ae(rng, D=3, H=4, L=1):
    def layer(i):
        return {"W_in": rng.normal(0, 0.6, (4 * H, i)),
                "W_rec": rng.normal(0, 0.6, (4 * H, H)),
                "b": rng.normal(0, 0.1, 4 * H)}
    enc = {"D": D, "H": H, "L": L, "layers": [layer(D)]}
    dec = {"D": H, "H": H, "L": L, "layers": [layer(H)]}
    head = {"W": rng.normal(0, 0.6, (D, H)), "b": rng.normal(0, 0.1, D)}
    return enc, dec, head


@pytest.mark.soundness
@pytest.mark.parametrize("threat", ["single_frame", "multi_frame"])
def test_radii_bit_identical_random_ae(threat):
    """Fast tier: random small AE, T=6. Sequential Algorithm 1 vs
    (parallel bisect), (kary sequential), (kary parallel)."""
    rng = np.random.default_rng(SEED)
    enc, dec, head = _small_ae(rng)
    x = rng.uniform(0, 1, (6, 3))
    # tau: certify something nontrivial
    from cert_rnn.analysis import reconstruction_score
    tau = 3.0 * reconstruction_score(enc, dec, head, x)
    reset_pred_allocator(0)
    r0, pf0 = certify_radius_spec_c(enc, dec, head, x, tau, 0.5, 12, threat)
    for search, n_workers in (("bisect", 4), ("kary", 1), ("kary", 4)):
        reset_pred_allocator(12345)   # different offset on purpose
        r, pf = certify_radius_spec_c(enc, dec, head, x, tau, 0.5, 12, threat,
                                      search=search, probes=15, n_workers=n_workers)
        assert r == r0, (search, n_workers, r, r0)
        if pf0 is not None:
            assert np.array_equal(pf, pf0), (search, n_workers, pf, pf0)


@pytest.mark.soundness
def test_allocator_offset_invariance():
    """The absolute predicate-id offset must not change any zonotope: same
    reach from allocator start 0 vs 10^6 gives identical (c, V) and the same
    RELATIVE id order."""
    rng = np.random.default_rng(SEED)
    enc, dec, head = _small_ae(rng)
    x = rng.uniform(0, 1, (5, 3))
    reset_pred_allocator(0)
    a_xh, a_x = lstm_ae_reach(enc, dec, head, x, 0.05, "multi_frame", None)
    reset_pred_allocator(10 ** 6)
    b_xh, b_x = lstm_ae_reach(enc, dec, head, x, 0.05, "multi_frame", None)
    for za, zb in zip(a_xh + a_x, b_xh + b_x):
        assert np.array_equal(za.c, zb.c) and np.array_equal(za.V, zb.V)
        ra = np.argsort(np.argsort(za.pred_ids)); rb = np.argsort(np.argsort(zb.pred_ids))
        assert np.array_equal(ra, rb)
    assert spec_c_score_ub(a_xh, a_x) == spec_c_score_ub(b_xh, b_x)


@pytest.mark.nightly
@pytest.mark.parametrize("subject", ["ieee9-S", "synth-H16"])
@pytest.mark.parametrize("threat", ["single_frame", "multi_frame"])
def test_radii_bit_identical_benchmarks(subject, threat):
    """Full benchmarks, all frames, 12-round resolution: exact equality of
    the certified radius and every per-frame radius across the sequential
    baseline, parallel bisection, and k-ary (15 probes) parallel."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from research.phase0_instrumentation import subjects
    case = dict(subjects())[subject]
    reset_pred_allocator(0)
    r0, pf0 = certify_radius_spec_c(case.encoder, case.decoder, case.head,
                                    case.anchor, case.tau, 0.5, 12, threat)
    for search, n_workers in (("bisect", 24), ("kary", 24)):
        reset_pred_allocator(777)
        r, pf = certify_radius_spec_c(case.encoder, case.decoder, case.head,
                                      case.anchor, case.tau, 0.5, 12, threat,
                                      search=search, probes=15, n_workers=n_workers)
        assert r == r0, (search, r, r0)
        if pf0 is not None:
            assert np.array_equal(pf, pf0), (search, np.flatnonzero(pf != pf0))
