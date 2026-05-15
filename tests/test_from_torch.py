"""PyTorch interop: extraction shapes + end-to-end parity with PyTorch forward.

Strategy: build a PyTorch nn.LSTM/nn.LSTMCell/nn.RNN with seeded weights,
cast to float64 to remove dtype drift, run the PyTorch forward, then
extract a Cert-RNN model dict and run lstm_step_stack/rnn_step with
Zono.point inputs at each timestep. Point inputs go through every
transformer exactly (no abstraction error), so equality must hold to
float64 precision.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from cert_rnn import Zono
from cert_rnn.from_torch import lstm_to_model_dict, rnn_to_model_dict
from cert_rnn.lstm import lstm_state_init, lstm_step_stack
from cert_rnn.rnn import rnn_step

ATOL = 1e-10


def _cert_lstm_forward(model_dict, x_seq):
    """Run lstm_step_stack with Zono.point inputs; return (h_T_top, c_T_top) centers."""
    H = model_dict["H"]
    L = model_dict["L"]
    z_h, z_c = lstm_state_init(H, L)
    for t in range(x_seq.shape[0]):
        z_x = Zono.point(x_seq[t])
        z_h, z_c = lstm_step_stack(z_x, z_h, z_c, model_dict["layers"])
    return z_h[-1].c, z_c[-1].c


def _cert_rnn_forward(model_dict, x_seq):
    H = model_dict["H"]
    L = model_dict["L"]
    z_h_layers = [Zono.point(np.zeros(H)) for _ in range(L)]
    for t in range(x_seq.shape[0]):
        inp = Zono.point(x_seq[t])
        for i in range(L):
            lyr = model_dict["layers"][i]
            z_h_layers[i] = rnn_step(inp, z_h_layers[i], lyr["W_in"], lyr["W_rec"], lyr["b"])
            inp = z_h_layers[i]
    return z_h_layers[-1].c


# ---------- shape / contents extraction ----------

def test_lstm_single_layer_extraction():
    torch.manual_seed(0)
    rec = nn.LSTM(input_size=3, hidden_size=4, num_layers=1, batch_first=True).double()
    fc = nn.Linear(4, 2).double()
    d = lstm_to_model_dict(rec, fc)
    assert d["type"] == "lstm"
    assert d["gate_order"] == "ifgo"
    assert d["D"] == 3 and d["H"] == 4 and d["L"] == 1
    assert len(d["layers"]) == 1
    assert d["layers"][0]["W_in"].shape == (16, 3)
    assert d["layers"][0]["W_rec"].shape == (16, 4)
    assert d["layers"][0]["b"].shape == (16,)
    assert d["head"]["W"].shape == (2, 4)
    assert d["head"]["b"].shape == (2,)


def test_lstm_multi_layer_extraction():
    torch.manual_seed(1)
    rec = nn.LSTM(input_size=3, hidden_size=4, num_layers=2, batch_first=True).double()
    d = lstm_to_model_dict(rec)
    assert d["L"] == 2
    assert d["layers"][0]["W_in"].shape == (16, 3)
    assert d["layers"][1]["W_in"].shape == (16, 4)  # second layer reads H-sized hidden
    assert "head" not in d


def test_lstm_cell_extraction():
    torch.manual_seed(2)
    rec = nn.LSTMCell(input_size=3, hidden_size=4).double()
    d = lstm_to_model_dict(rec)
    assert d["D"] == 3 and d["H"] == 4 and d["L"] == 1
    assert d["layers"][0]["W_in"].shape == (16, 3)


def test_rnn_extraction():
    torch.manual_seed(3)
    rec = nn.RNN(input_size=3, hidden_size=4, num_layers=2, nonlinearity="tanh").double()
    d = rnn_to_model_dict(rec)
    assert d["type"] == "vanilla_rnn"
    assert d["nonlinearity"] == "tanh"
    assert d["H"] == 4 and d["L"] == 2
    assert d["layers"][0]["W_in"].shape == (4, 3)
    assert d["layers"][1]["W_in"].shape == (4, 4)


def test_lstm_no_bias():
    rec = nn.LSTM(input_size=3, hidden_size=4, bias=False).double()
    d = lstm_to_model_dict(rec)
    np.testing.assert_allclose(d["layers"][0]["b"], np.zeros(16))


# ---------- rejected configurations ----------

def test_reject_bidirectional_lstm():
    rec = nn.LSTM(input_size=3, hidden_size=4, bidirectional=True).double()
    with pytest.raises(ValueError, match="bidirectional"):
        lstm_to_model_dict(rec)


def test_reject_dropout_lstm():
    rec = nn.LSTM(input_size=3, hidden_size=4, num_layers=2, dropout=0.5).double()
    with pytest.raises(ValueError, match="dropout"):
        lstm_to_model_dict(rec)


def test_reject_relu_rnn():
    rec = nn.RNN(input_size=3, hidden_size=4, nonlinearity="relu").double()
    with pytest.raises(ValueError, match="tanh"):
        rnn_to_model_dict(rec)


def test_reject_wrong_module_type_lstm():
    with pytest.raises(TypeError, match="nn.LSTM"):
        lstm_to_model_dict(nn.Linear(4, 4))


def test_reject_wrong_module_type_rnn():
    with pytest.raises(TypeError, match="nn.RNN"):
        rnn_to_model_dict(nn.Linear(4, 4))


def test_reject_non_linear_fc():
    rec = nn.LSTM(3, 4).double()
    with pytest.raises(TypeError, match="nn.Linear"):
        lstm_to_model_dict(rec, fc=nn.Tanh())


# ---------- end-to-end parity vs PyTorch forward ----------

@pytest.mark.parametrize("D,H,T,L", [(3, 4, 5, 1), (2, 3, 4, 2), (5, 8, 1, 1)])
def test_parity_lstm_vs_pytorch(D, H, T, L):
    torch.manual_seed(42 + D + H + T + L)
    rec = nn.LSTM(input_size=D, hidden_size=H, num_layers=L, batch_first=True).double()
    rec.eval()
    x_torch = torch.randn(1, T, D, dtype=torch.float64)
    with torch.no_grad():
        # PyTorch returns (h_n, c_n) of shape (num_layers, batch, H)
        _, (h_n, c_n) = rec(x_torch)
    h_top = h_n[-1, 0].numpy()
    c_top = c_n[-1, 0].numpy()

    model_dict = lstm_to_model_dict(rec)
    x_seq = x_torch[0].numpy()
    h_cert, c_cert = _cert_lstm_forward(model_dict, x_seq)

    np.testing.assert_allclose(h_cert, h_top, atol=ATOL)
    np.testing.assert_allclose(c_cert, c_top, atol=ATOL)


def test_parity_lstm_cell_vs_pytorch():
    torch.manual_seed(7)
    D, H, T = 3, 4, 5
    cell = nn.LSTMCell(D, H).double().eval()
    x_torch = torch.randn(T, D, dtype=torch.float64)
    h = torch.zeros(1, H, dtype=torch.float64)
    c = torch.zeros(1, H, dtype=torch.float64)
    with torch.no_grad():
        for t in range(T):
            h, c = cell(x_torch[t].unsqueeze(0), (h, c))
    h_pt = h[0].numpy()
    c_pt = c[0].numpy()

    model_dict = lstm_to_model_dict(cell)
    h_cert, c_cert = _cert_lstm_forward(model_dict, x_torch.numpy())

    np.testing.assert_allclose(h_cert, h_pt, atol=ATOL)
    np.testing.assert_allclose(c_cert, c_pt, atol=ATOL)


@pytest.mark.parametrize("D,H,T,L", [(3, 4, 5, 1), (2, 3, 4, 2)])
def test_parity_rnn_vs_pytorch(D, H, T, L):
    torch.manual_seed(11 + D + H + T + L)
    rec = nn.RNN(input_size=D, hidden_size=H, num_layers=L, nonlinearity="tanh",
                  batch_first=True).double().eval()
    x_torch = torch.randn(1, T, D, dtype=torch.float64)
    with torch.no_grad():
        _, h_n = rec(x_torch)
    h_top = h_n[-1, 0].numpy()

    model_dict = rnn_to_model_dict(rec)
    h_cert = _cert_rnn_forward(model_dict, x_torch[0].numpy())
    np.testing.assert_allclose(h_cert, h_top, atol=ATOL)


def test_parity_lstm_with_head():
    """Classifier head: cert engine output through head matches PyTorch."""
    torch.manual_seed(99)
    D, H, T, C = 3, 4, 5, 2
    rec = nn.LSTM(D, H, num_layers=1, batch_first=True).double().eval()
    fc = nn.Linear(H, C).double().eval()
    x_torch = torch.randn(1, T, D, dtype=torch.float64)
    with torch.no_grad():
        out, _ = rec(x_torch)
        logits_pt = fc(out[:, -1, :])[0].numpy()

    model_dict = lstm_to_model_dict(rec, fc)
    h_cert, _ = _cert_lstm_forward(model_dict, x_torch[0].numpy())
    logits_cert = model_dict["head"]["W"] @ h_cert + model_dict["head"]["b"]
    np.testing.assert_allclose(logits_cert, logits_pt, atol=ATOL)
