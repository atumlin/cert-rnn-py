"""Load an LSTM-AE IEEE-9 checkpoint into Cert-RNN-ready dicts.

Each size (S, M, L, D) ships as:
  data/lstm_ae_ieee9_<size>.pt    PyTorch state dict (enc_cells, dec_cells, head)
  data/lstm_ae_ieee9_<size>.json  metadata (H, T, n_features, tau, anchor_index, anchor_score, ...)
  data/anchor_<size>.npy           anchor sequence (T, D) extracted from training

Returns encoder, decoder, head model dicts that the cert_rnn engine
consumes directly (cert_rnn.lstm.lstm_step_stack), plus the anchor and
spec parameters (tau, anchor_index, anchor_score).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cert_rnn.from_torch import lstm_to_model_dict

DATA_DIR = Path(__file__).parent / "data"
VALID_SIZES = ("S", "M", "L", "D")


def _cell_from_state(sd: dict, prefix: str, in_size: int, H: int) -> nn.LSTMCell:
    cell = nn.LSTMCell(in_size, H).double()
    with torch.no_grad():
        cell.weight_ih.copy_(sd[f"{prefix}.weight_ih"].double())
        cell.weight_hh.copy_(sd[f"{prefix}.weight_hh"].double())
        cell.bias_ih.copy_(sd[f"{prefix}.bias_ih"].double())
        cell.bias_hh.copy_(sd[f"{prefix}.bias_hh"].double())
    return cell


def _head_from_state(sd: dict, H: int, D: int) -> dict:
    W = sd["head.weight"].detach().cpu().numpy().astype(np.float64)
    b = sd["head.bias"].detach().cpu().numpy().astype(np.float64)
    if W.shape != (D, H):
        raise RuntimeError(f"head.weight shape {W.shape}, expected {(D, H)}")
    return {"W": W, "b": b}


def load_lstm_ae(size: str, data_dir: Path | None = None) -> dict:
    """Load one size of the LSTM-AE checkpoint.

    Returns:
        {
            "size":         str,
            "encoder":      cert_rnn model dict,
            "decoder":      cert_rnn model dict,
            "head":         {"W": (D, H), "b": (D,)},
            "anchor":       (T, D) float64,
            "tau":          float,
            "anchor_index": int,
            "anchor_score": float,
            "H": int, "T": int, "D": int,
        }
    """
    if size not in VALID_SIZES:
        raise ValueError(f"size {size!r} not in {VALID_SIZES}")
    root = data_dir if data_dir is not None else DATA_DIR
    meta = json.loads((root / f"lstm_ae_ieee9_{size}.json").read_text())
    sd = torch.load(root / f"lstm_ae_ieee9_{size}.pt", map_location="cpu", weights_only=True)
    H = int(meta["hidden"])
    T = int(meta["T"])
    D = int(meta["n_features"])
    n_enc = int(meta["n_enc_layers"])
    n_dec = int(meta["n_dec_layers"])

    def stack_dict(prefix: str, n_layers: int, first_in_size: int) -> dict:
        layers = []
        for i in range(n_layers):
            in_size = first_in_size if i == 0 else H
            W_in = sd[f"{prefix}.{i}.weight_ih"].detach().cpu().numpy().astype(np.float64)
            W_rec = sd[f"{prefix}.{i}.weight_hh"].detach().cpu().numpy().astype(np.float64)
            b_ih = sd[f"{prefix}.{i}.bias_ih"].detach().cpu().numpy().astype(np.float64)
            b_hh = sd[f"{prefix}.{i}.bias_hh"].detach().cpu().numpy().astype(np.float64)
            if W_in.shape != (4 * H, in_size):
                raise RuntimeError(
                    f"{prefix}.{i}.weight_ih shape {W_in.shape}, expected {(4 * H, in_size)}"
                )
            layers.append({"W_in": W_in, "W_rec": W_rec, "b": b_ih + b_hh})
        return {
            "type": "lstm", "gate_order": "ifgo",
            "D": first_in_size, "H": H, "L": n_layers, "layers": layers,
        }

    encoder = stack_dict("enc_cells", n_enc, D)
    decoder = stack_dict("dec_cells", n_dec, H)
    head = _head_from_state(sd, H, D)

    anchor = np.load(root / f"anchor_{size}.npy").astype(np.float64)
    if anchor.shape != (T, D):
        raise RuntimeError(
            f"anchor shape {anchor.shape}, expected ({T}, {D}) per {size} metadata"
        )

    return {
        "size": size,
        "encoder": encoder,
        "decoder": decoder,
        "head": head,
        "anchor": anchor,
        "tau": float(meta["tau"]),
        "anchor_index": int(meta["anchor_index"]),
        "anchor_score": float(meta["anchor_score"]),
        "H": H,
        "T": T,
        "D": D,
    }
