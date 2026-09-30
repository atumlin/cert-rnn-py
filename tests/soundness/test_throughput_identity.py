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
@pytest.mark.parametrize("threat", ["single_frame", "multi_frame"])
@pytest.mark.parametrize("score_bound", ["componentwise", "joint"])
@pytest.mark.parametrize("zono_last", [None, 3])
def test_radii_bit_identical_tier1(threat, score_bound, zono_last):
    """Tier-1 (zono) gates, pure and with the round cutover, both score
    bounds: sequential Algorithm 1 vs free-running parallel bisect, kary
    sequential, kary parallel -- exact equality. Under the cutover, bisect
    (13 rounds) and kary (4 rounds) put DIFFERENT probes in zono mode, so
    there each search is compared only with its own serial run."""
    from cert_rnn.analysis import reconstruction_score
    from cert_rnn.transformers import bilinear_mode
    rng = np.random.default_rng(SEED + 1)
    enc, dec, head = _small_ae(rng)
    x = rng.uniform(0, 1, (6, 3))
    tau = 3.0 * reconstruction_score(enc, dec, head, x)
    kw = dict(score_bound=score_bound, zono_last_rounds=zono_last)
    if zono_last is None:
        groups = [[("bisect", 1), ("bisect", 4), ("kary", 1), ("kary", 4)]]
    else:
        groups = [[("bisect", 1), ("bisect", 4)], [("kary", 1), ("kary", 4)]]
    with bilinear_mode("zono"):
        for group in groups:
            ref = None
            for search, n_workers in group:
                reset_pred_allocator(0 if ref is None else 4242)
                r, pf = certify_radius_spec_c(enc, dec, head, x, tau, 0.5, 12, threat,
                                              search=search, probes=15,
                                              n_workers=n_workers, **kw)
                if ref is None:
                    ref = (r, pf)
                    continue
                assert r == ref[0], (search, n_workers, r, ref[0])
                if pf is not None:
                    assert np.array_equal(pf, ref[1]), (search, n_workers, pf, ref[1])


@pytest.mark.soundness
def test_joint_score_bound_never_looser_in_search():
    """The joint score bound only ever certifies MORE (never looser)."""
    from cert_rnn.analysis import reconstruction_score
    rng = np.random.default_rng(SEED + 2)
    enc, dec, head = _small_ae(rng)
    x = rng.uniform(0, 1, (6, 3))
    tau = 3.0 * reconstruction_score(enc, dec, head, x)
    rc, _ = certify_radius_spec_c(enc, dec, head, x, tau, 0.5, 12, "multi_frame")
    rj, _ = certify_radius_spec_c(enc, dec, head, x, tau, 0.5, 12, "multi_frame",
                                  score_bound="joint")
    assert rj >= rc, (rj, rc)


@pytest.mark.soundness
@pytest.mark.parametrize("threat", ["single_frame", "multi_frame"])
def test_tier1_threads_bit_identical(threat):
    """Threads over coordinates inside the Tier-1 gate step change nothing:
    identical reach zonotopes for THREADS = 1 and 4."""
    from cert_rnn import tier1
    from cert_rnn.transformers import bilinear_mode
    rng = np.random.default_rng(SEED + 3)
    enc, dec, head = _small_ae(rng, D=4, H=8)
    x = rng.uniform(0, 1, (6, 4))
    t = 0 if threat == "single_frame" else None
    out = []
    try:
        for th in (1, 4):
            tier1.THREADS = th
            reset_pred_allocator(0)
            with bilinear_mode("zono"):
                out.append(lstm_ae_reach(enc, dec, head, x, 0.05, threat, t))
    finally:
        tier1.THREADS = 1
    for za, zb in zip(out[0][0] + out[0][1], out[1][0] + out[1][1]):
        assert np.array_equal(za.c, zb.c) and np.array_equal(za.V, zb.V)
        assert za.pred_ids == zb.pred_ids


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
