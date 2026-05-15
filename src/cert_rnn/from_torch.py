"""PyTorch interop: extract Cert-RNN model dicts from nn.LSTM/nn.LSTMCell/nn.RNN.

PyTorch gate order for nn.LSTM weight_ih_l*/weight_hh_l* is [i, f, g, o]
-- the same order the Cert-RNN engine expects (cert_rnn.lstm), so no
row permutation is needed. PyTorch carries two biases (bias_ih, bias_hh)
which the runtime sums; we fold them into one b to match lstm_step's
single-bias signature.

Rejected configurations:
    bidirectional         (no merge semantics in Cert-RNN)
    dropout > 0           (eval-mode no-op, but flagged to avoid surprises)
    nn.RNN nonlinearity != 'tanh'   (ReLU-RNN needs a sound ReLU zono
                                     transformer, outside paper scope)

Returned model dict (consumed by cert_rnn.lstm.lstm_step_stack):
    {
        "type":    "lstm" | "vanilla_rnn",
        "D":       int,
        "H":       int,
        "L":       int,
        "layers":  [
            {"W_in": ndarray (4H or H, D|H),
             "W_rec": ndarray (4H or H, H),
             "b":    ndarray (4H or H,)},
            ...
        ],
        "head":    {"W": ndarray (C, H), "b": ndarray (C,)},   # iff fc passed
        "gate_order":   "ifgo",                                # LSTM only
        "nonlinearity": "tanh",                                # RNN only
    }
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float64)


def _fc_dict(fc: nn.Linear | None) -> dict | None:
    if fc is None:
        return None
    if not isinstance(fc, nn.Linear):
        raise TypeError(f"fc must be nn.Linear, got {type(fc).__name__}")
    W = _to_numpy(fc.weight)
    if fc.bias is not None:
        b = _to_numpy(fc.bias).reshape(-1)
    else:
        b = np.zeros(W.shape[0], dtype=np.float64)
    return {"W": W, "b": b}


def lstm_to_model_dict(
    rec: "nn.LSTM | nn.LSTMCell",
    fc: nn.Linear | None = None,
) -> dict:
    """Extract a Cert-RNN model dict from a PyTorch nn.LSTM or nn.LSTMCell."""
    if isinstance(rec, nn.LSTMCell):
        D = rec.input_size
        H = rec.hidden_size
        W_in = _to_numpy(rec.weight_ih)
        W_rec = _to_numpy(rec.weight_hh)
        if rec.bias:
            b = _to_numpy(rec.bias_ih) + _to_numpy(rec.bias_hh)
        else:
            b = np.zeros(4 * H, dtype=np.float64)
        L = 1
        layers = [{"W_in": W_in, "W_rec": W_rec, "b": b}]
    elif isinstance(rec, nn.LSTM):
        if rec.bidirectional:
            raise ValueError("bidirectional LSTM is not supported")
        if rec.dropout != 0:
            raise ValueError(
                f"dropout={rec.dropout} is a no-op in eval but rejected to avoid "
                "surprises; rebuild the module with dropout=0"
            )
        D = rec.input_size
        H = rec.hidden_size
        L = rec.num_layers
        layers = []
        for i in range(L):
            in_size = D if i == 0 else H
            W_in = _to_numpy(getattr(rec, f"weight_ih_l{i}"))
            W_rec = _to_numpy(getattr(rec, f"weight_hh_l{i}"))
            if rec.bias:
                b = _to_numpy(getattr(rec, f"bias_ih_l{i}")) + _to_numpy(
                    getattr(rec, f"bias_hh_l{i}")
                )
            else:
                b = np.zeros(4 * H, dtype=np.float64)
            if W_in.shape != (4 * H, in_size):
                raise RuntimeError(
                    f"layer {i}: unexpected W_in shape {W_in.shape}, "
                    f"expected {(4 * H, in_size)}"
                )
            if W_rec.shape != (4 * H, H):
                raise RuntimeError(
                    f"layer {i}: unexpected W_rec shape {W_rec.shape}, "
                    f"expected {(4 * H, H)}"
                )
            layers.append({"W_in": W_in, "W_rec": W_rec, "b": b})
    else:
        raise TypeError(
            f"rec must be nn.LSTM or nn.LSTMCell, got {type(rec).__name__}"
        )

    out = {
        "type": "lstm",
        "gate_order": "ifgo",
        "D": int(D),
        "H": int(H),
        "L": int(L),
        "layers": layers,
    }
    head = _fc_dict(fc)
    if head is not None:
        out["head"] = head
    return out


def rnn_to_model_dict(rec: nn.RNN, fc: nn.Linear | None = None) -> dict:
    """Extract a Cert-RNN model dict from a PyTorch nn.RNN (tanh only)."""
    if not isinstance(rec, nn.RNN):
        raise TypeError(f"rec must be nn.RNN, got {type(rec).__name__}")
    if rec.nonlinearity != "tanh":
        raise ValueError(
            f"vanilla RNN must use nonlinearity='tanh' (got '{rec.nonlinearity}'); "
            "ReLU-RNN needs a sound ReLU zono transformer, outside Cert-RNN paper scope"
        )
    if rec.bidirectional:
        raise ValueError("bidirectional RNN is not supported")
    if rec.dropout != 0:
        raise ValueError(
            f"dropout={rec.dropout} not supported; rebuild with dropout=0"
        )
    D = rec.input_size
    H = rec.hidden_size
    L = rec.num_layers
    layers = []
    for i in range(L):
        in_size = D if i == 0 else H
        W_in = _to_numpy(getattr(rec, f"weight_ih_l{i}"))
        W_rec = _to_numpy(getattr(rec, f"weight_hh_l{i}"))
        if rec.bias:
            b = _to_numpy(getattr(rec, f"bias_ih_l{i}")) + _to_numpy(
                getattr(rec, f"bias_hh_l{i}")
            )
        else:
            b = np.zeros(H, dtype=np.float64)
        if W_in.shape != (H, in_size):
            raise RuntimeError(
                f"layer {i}: unexpected W_in shape {W_in.shape}, "
                f"expected {(H, in_size)}"
            )
        if W_rec.shape != (H, H):
            raise RuntimeError(
                f"layer {i}: unexpected W_rec shape {W_rec.shape}, "
                f"expected {(H, H)}"
            )
        layers.append({"W_in": W_in, "W_rec": W_rec, "b": b})

    out = {
        "type": "vanilla_rnn",
        "nonlinearity": "tanh",
        "D": int(D),
        "H": int(H),
        "L": int(L),
        "layers": layers,
    }
    head = _fc_dict(fc)
    if head is not None:
        out["head"] = head
    return out
