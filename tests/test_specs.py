"""Phase 3: the Spec abstraction and the unified certify() entry point.

The unified path (model.certify / cert_rnn.certify with a Spec) must be
numerically identical to the dedicated certify_radius_* functions, and the
generic ThresholdSpec must behave monotonically (a looser box certifies at
least as far as a tighter one).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

import cert_rnn as cr
from cert_rnn import (
    CertResult,
    LSTMAutoencoder,
    MarginSpec,
    RNNModel,
    ReconErrorSpec,
    ThresholdSpec,
    certify,
)
from cert_rnn.verify import certify_radius_spec_a, certify_radius_spec_c
from cert_rnn.zono import Zono, reset_pred_allocator


@pytest.fixture(autouse=True)
def _isolate_preds():
    reset_pred_allocator()
    yield


def test_specs_exported_top_level():
    for n in ("Spec", "MarginSpec", "ThresholdSpec", "ReconErrorSpec",
              "CertResult", "certify"):
        assert n in cr.__all__ and hasattr(cr, n)


# ---------- MarginSpec == certify_radius_spec_a ----------

def test_margin_certify_matches_spec_a():
    torch.manual_seed(5)
    cell = nn.LSTMCell(4, 6).double()
    fc = nn.Linear(6, 3).double()
    model = RNNModel.from_torch(cell, fc)
    x = np.random.default_rng(0).standard_normal((5, 4))

    reset_pred_allocator()
    res = model.certify(x, MarginSpec(true_class=0), n_iters=6)
    reset_pred_allocator()
    r_ref, pf_ref = certify_radius_spec_a(model.as_dict(), x, 0, n_iters=6)

    assert isinstance(res, CertResult)
    assert res.radius == r_ref
    assert np.array_equal(res.per_frame, pf_ref)
    assert res.certified == (r_ref > 0)


# ---------- ReconErrorSpec == certify_radius_spec_c ----------

def test_recon_certify_matches_spec_c():
    torch.manual_seed(9)
    enc = nn.LSTMCell(3, 4).double()
    dec = nn.LSTMCell(4, 4).double()
    head = nn.Linear(4, 3).double()
    ae = LSTMAutoencoder.from_torch(enc, dec, head)
    x = np.random.default_rng(3).standard_normal((5, 3))
    tau = 5.0

    reset_pred_allocator()
    res = certify(ae, x, ReconErrorSpec(tau), n_iters=6)
    reset_pred_allocator()
    r_ref, pf_ref = certify_radius_spec_c(ae.encoder, ae.decoder, ae.head, x, tau, n_iters=6)

    assert res.radius == r_ref
    assert np.array_equal(res.per_frame, pf_ref)


# ---------- ThresholdSpec ----------

def test_threshold_spec_holds_logic():
    # z spans [c-|v|, c+|v|] per element.
    z = Zono(np.array([0.0, 1.0]), np.array([[0.5], [0.25]]), (0,))
    # box [-1, 2] contains [-0.5,0.5] and [0.75,1.25]
    assert ThresholdSpec(upper=2.0, lower=-1.0).holds(z)
    # upper 1.0 violated by element 1 (ub=1.25)
    assert not ThresholdSpec(upper=1.0).holds(z)
    # restrict to element 0 only -> upper 1.0 ok
    assert ThresholdSpec(upper=1.0, indices=[0]).holds(z)


def test_threshold_certify_monotonic():
    torch.manual_seed(7)
    cell = nn.LSTMCell(3, 5).double()   # no head -> reach_output is hidden zono
    model = RNNModel.from_torch(cell)
    x = np.random.default_rng(1).standard_normal((4, 3))

    reset_pred_allocator()
    tight = certify(model, x, ThresholdSpec(upper=0.5, lower=-0.5), n_iters=8)
    reset_pred_allocator()
    loose = certify(model, x, ThresholdSpec(upper=2.0, lower=-2.0), n_iters=8)
    # A looser output box can only certify a >= radius.
    assert loose.radius >= tight.radius


# ---------- error handling ----------

def test_certify_rejects_non_wrapper():
    with pytest.raises(TypeError, match="model wrapper"):
        certify({"not": "a model"}, np.zeros((3, 2)), MarginSpec(0))


def test_custom_spec_duck_typed():
    """Any object with holds(output) works -- here, 'class 0 logit > 0'."""
    torch.manual_seed(11)
    cell = nn.LSTMCell(3, 4).double()
    fc = nn.Linear(4, 2).double()
    model = RNNModel.from_torch(cell, fc)
    x = np.random.default_rng(2).standard_normal((4, 3))

    class PositiveLogit:
        def holds(self, z_logits):
            lb, _ = z_logits.get_ranges()
            return bool(lb[0] > 0)

    res = certify(model, x, PositiveLogit(), n_iters=5)
    assert isinstance(res, CertResult)
