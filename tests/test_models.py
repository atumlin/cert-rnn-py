"""Phase 2 wrappers: RNNModel / LSTMAutoencoder and the AE extractor.

These check that the typed wrappers (cert_rnn.models) are exact, thin
handles over the dict-based engine: construction matches from_torch,
verification methods match the verify.* functions, the AE extractor
accepts both monolithic and cell-list stacks, and shape validation
fires on mismatches.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

import cert_rnn as cr
from cert_rnn import LSTMAutoencoder, RNNModel
from cert_rnn.from_torch import lstm_ae_to_model_dicts, lstm_to_model_dict
from cert_rnn.verify import certify_radius_spec_a, lstm_ae_reach, spec_c_score_ub
from cert_rnn.zono import reset_pred_allocator


@pytest.fixture(autouse=True)
def _isolate_preds():
    reset_pred_allocator()
    yield


# ---------- top-level API surface ----------

def test_new_symbols_are_top_level():
    for name in ("RNNModel", "LSTMAutoencoder", "lstm_ae_to_model_dicts"):
        assert name in cr.__all__
        assert hasattr(cr, name)


# ---------- RNNModel ----------

def test_rnnmodel_from_torch_matches_dict():
    torch.manual_seed(1)
    rec = nn.LSTM(input_size=3, hidden_size=5, num_layers=2).double()
    fc = nn.Linear(5, 4).double()
    model = RNNModel.from_torch(rec, fc)
    ref = lstm_to_model_dict(rec, fc)
    assert model.D == 3 and model.H == 5 and model.L == 2 and model.has_head
    assert model.type == "lstm"
    for i in range(model.L):
        for k in ("W_in", "W_rec", "b"):
            assert np.array_equal(model.spec["layers"][i][k], ref["layers"][i][k])
    assert np.array_equal(model.spec["head"]["W"], ref["head"]["W"])


def test_rnnmodel_certify_radius_matches_function():
    torch.manual_seed(2)
    cell = nn.LSTMCell(4, 6).double()
    fc = nn.Linear(6, 3).double()
    model = RNNModel.from_torch(cell, fc)
    x = np.random.default_rng(0).standard_normal((5, 4))

    reset_pred_allocator()
    r_method, pf_method = model.certify_radius(x, true_class=0, n_iters=6)
    reset_pred_allocator()
    r_func, pf_func = certify_radius_spec_a(model.as_dict(), x, 0, n_iters=6)

    assert r_method == r_func
    assert np.array_equal(pf_method, pf_func)


def test_rnnmodel_vanilla_rnn_verify_unsupported():
    torch.manual_seed(3)
    rec = nn.RNN(input_size=2, hidden_size=3, nonlinearity="tanh").double()
    model = RNNModel.from_torch(rec)
    assert model.type == "vanilla_rnn"
    x = np.zeros((4, 2))
    with pytest.raises(NotImplementedError):
        model.certify_radius(x, true_class=0)


# ---------- AE extractor ----------

def _build_ae(D=3, H=4, seed=7):
    torch.manual_seed(seed)
    enc = nn.LSTMCell(D, H).double()
    dec = nn.LSTMCell(H, H).double()   # decoder reads the latent (dim H)
    head = nn.Linear(H, D).double()
    return enc, dec, head


def test_ae_extractor_cell_list_equals_single_cell():
    enc, dec, head = _build_ae()
    d_single = lstm_ae_to_model_dicts(enc, dec, head)
    d_list = lstm_ae_to_model_dicts([enc], [dec], head)
    for side in ("encoder", "decoder"):
        assert d_single[side]["L"] == d_list[side]["L"] == 1
        assert np.array_equal(
            d_single[side]["layers"][0]["W_in"], d_list[side]["layers"][0]["W_in"]
        )


def test_ae_extractor_rejects_bad_shapes():
    enc, dec, head = _build_ae(D=3, H=4)
    # head out_features must equal D
    bad_head = nn.Linear(4, 5).double()
    with pytest.raises(ValueError, match="out_features"):
        lstm_ae_to_model_dicts(enc, dec, bad_head)
    # decoder input size must equal encoder H
    bad_dec = nn.LSTMCell(2, 4).double()
    with pytest.raises(ValueError, match="decoder input size"):
        lstm_ae_to_model_dicts(enc, bad_dec, head)


def test_ae_extractor_rejects_wrong_type():
    enc, dec, _ = _build_ae()
    with pytest.raises(TypeError, match="head must be nn.Linear"):
        lstm_ae_to_model_dicts(enc, dec, nn.ReLU())


# ---------- LSTMAutoencoder soundness/exactness at eps=0 ----------

def _concrete_ae_score(enc, dec, head, x):
    """Plain PyTorch AE forward; returns mean squared recon error."""
    T, D = x.shape
    H = enc.hidden_size
    xt = torch.tensor(x, dtype=torch.float64)
    h = torch.zeros(1, H, dtype=torch.float64)
    c = torch.zeros(1, H, dtype=torch.float64)
    for t in range(T):
        h, c = enc(xt[t : t + 1], (h, c))
    latent = h
    hd = torch.zeros(1, H, dtype=torch.float64)
    cd = torch.zeros(1, H, dtype=torch.float64)
    sq = 0.0
    for t in range(T):
        hd, cd = dec(latent, (hd, cd))
        x_hat = head(hd)[0]
        sq += float(((x_hat - xt[t]) ** 2).sum())
    return sq / (T * D)


def test_lstm_ae_score_ub_exact_at_eps_zero():
    enc, dec, head = _build_ae(D=3, H=4, seed=11)
    ae = LSTMAutoencoder.from_torch(enc, dec, head)
    x = np.random.default_rng(1).standard_normal((6, 3))

    reset_pred_allocator()
    ub = ae.score_ub(x, eps=0.0, t_pert=0)
    concrete = _concrete_ae_score(enc, dec, head, x)
    # eps=0 -> point zonotopes propagate exactly; the sound upper bound
    # collapses to the concrete score.
    assert ub == pytest.approx(concrete, abs=1e-9)


def test_lstm_ae_certifies_matches_reach_path():
    enc, dec, head = _build_ae(D=3, H=4, seed=13)
    ae = LSTMAutoencoder.from_torch(enc, dec, head)
    x = np.random.default_rng(2).standard_normal((5, 3))
    tau = 10.0

    reset_pred_allocator()
    method = ae.certifies(x, eps=0.05, tau=tau, t_pert=0)
    reset_pred_allocator()
    zxh, zx = lstm_ae_reach(ae.encoder, ae.decoder, ae.head, x, 0.05, "single_frame", 0)
    func = spec_c_score_ub(zxh, zx) <= tau
    assert method == func
