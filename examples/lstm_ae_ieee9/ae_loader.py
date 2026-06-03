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

from cert_rnn import LSTMAutoencoder

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


def _head_from_state(sd: dict, H: int, D: int) -> nn.Linear:
    head = nn.Linear(H, D).double()
    with torch.no_grad():
        head.weight.copy_(sd["head.weight"].double())
        head.bias.copy_(sd["head.bias"].double())
    return head


def load_lstm_ae(size: str, data_dir: Path | None = None) -> dict:
    """Load one size of the LSTM-AE checkpoint.

    Returns:
        {
            "size":         str,
            "model":        cert_rnn.LSTMAutoencoder,
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

    # Reconstruct the PyTorch cells/head from the flat state dict and let
    # the library extract the Cert-RNN dicts -- the encoder/decoder are
    # ModuleList stacks of LSTMCells (cell i feeds cell i+1).
    enc_cells = [
        _cell_from_state(sd, f"enc_cells.{i}", D if i == 0 else H, H)
        for i in range(n_enc)
    ]
    dec_cells = [
        _cell_from_state(sd, f"dec_cells.{i}", H, H) for i in range(n_dec)
    ]
    ae = LSTMAutoencoder.from_torch(enc_cells, dec_cells, _head_from_state(sd, H, D))
    encoder, decoder, head = ae.encoder, ae.decoder, ae.head

    anchor = np.load(root / f"anchor_{size}.npy").astype(np.float64)
    if anchor.shape != (T, D):
        raise RuntimeError(
            f"anchor shape {anchor.shape}, expected ({T}, {D}) per {size} metadata"
        )

    return {
        "size": size,
        "model": ae,
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
