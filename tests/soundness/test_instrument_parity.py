"""Instrumented step must be bit-identical to the engine, and the
closed-form zonotope-area formula must match the naive O(p^2) double sum.

The instrumented reach re-executes lstm_step's body; if the two bodies ever
drift (a refactor touches one and not the other), this test fails rather
than the instrumentation silently measuring a different computation.
"""

from __future__ import annotations

import numpy as np
import pytest

from cert_rnn.instrument import instrumented_lstm_ae_reach, zono2d_area
from cert_rnn.verify import lstm_ae_reach
from cert_rnn.zono import reset_pred_allocator

SEED = 20260814


def _random_ae(rng, D=3, H=5, L=2):
    def layer(in_dim):
        return {
            "W_in": rng.normal(0, 0.5, (4 * H, in_dim)),
            "W_rec": rng.normal(0, 0.5, (4 * H, H)),
            "b": rng.normal(0, 0.1, 4 * H),
        }
    encoder = {"D": D, "H": H, "L": L,
               "layers": [layer(D if i == 0 else H) for i in range(L)]}
    decoder = {"D": H, "H": H, "L": L,
               "layers": [layer(H) for i in range(L)]}
    head = {"W": rng.normal(0, 0.5, (D, H)), "b": rng.normal(0, 0.1, D)}
    return encoder, decoder, head


@pytest.mark.soundness
@pytest.mark.parametrize("threat_model,t_pert", [("multi_frame", None),
                                                 ("single_frame", 0)])
def test_instrumented_reach_bit_identical(threat_model, t_pert):
    print(f"\n[seed={SEED}] instrument parity: {threat_model}")
    rng = np.random.default_rng(SEED)
    encoder, decoder, head = _random_ae(rng)
    x = rng.uniform(0, 1, (4, 3))

    reset_pred_allocator(0)
    ref_xh, ref_x = lstm_ae_reach(encoder, decoder, head, x, 0.05,
                                  threat_model, t_pert)
    reset_pred_allocator(0)
    ins_xh, ins_x, rec = instrumented_lstm_ae_reach(
        encoder, decoder, head, x, 0.05, threat_model, t_pert)

    for ref_seq, ins_seq in ((ref_xh, ins_xh), (ref_x, ins_x)):
        assert len(ref_seq) == len(ins_seq)
        for zr, zi in zip(ref_seq, ins_seq):
            assert zr.pred_ids == zi.pred_ids
            assert np.array_equal(zr.c, zi.c)
            assert np.array_equal(zr.V, zi.V)

    # sanity on the recording itself: T steps x L layers x 3 gates, both phases
    assert len(rec.gates) == 2 * 4 * 2 * 3
    assert len(rec.z4) == 2 * 4 * 2


def _naive_area(a, b):
    total = 0.0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            total += abs(a[i] * b[j] - a[j] * b[i])
    return 4.0 * total


@pytest.mark.soundness
def test_zono2d_area_matches_naive_sum():
    rng = np.random.default_rng(SEED)
    for p in (0, 1, 2, 3, 8, 50):
        for _ in range(20):
            a = rng.normal(0, 1, p)
            b = rng.normal(0, 1, p)
            # include degenerate directions
            if p >= 3:
                b[0] = 0.0
                a[1], b[1] = -a[0], 0.0   # antiparallel to g0
                a[2], b[2] = 0.0, 0.0     # zero generator
            fast = zono2d_area(a, b)
            slow = _naive_area(a, b)
            assert fast == pytest.approx(slow, rel=1e-12, abs=1e-15)


@pytest.mark.soundness
def test_zono2d_area_box_case():
    # axis-aligned box: area must be exactly w_x * w_y
    a = np.array([0.5, 0.0])
    b = np.array([0.0, 1.5])
    assert zono2d_area(a, b) == pytest.approx((2 * 0.5) * (2 * 1.5))
