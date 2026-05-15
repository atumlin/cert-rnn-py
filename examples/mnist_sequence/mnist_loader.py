"""Load an MNIST-sequence LSTM checkpoint exported from PyTorch via
cert_rnn_export.py (sibling MATLAB repo).

The .mat bundles:
  - LSTM weights (flat W_in/W_rec/b for single-layer, or *_layers cell
    arrays for stacked)
  - Linear classifier head (W_fc, b_fc)
  - X_test: (N, 1) cell of (T=28, D=28) MNIST-sequence test samples
  - Y_test: (N,) integer labels
  - python_predictions: (N, num_classes) reference logits from PyTorch
    forward at eps=0 -- used for the cert-vs-PyTorch parity check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

DATA_DIR = Path(__file__).parent / "data"

CONFIG_FILES = {
    "LSTM-1-32": "lstm_1_32_mnist.mat",
    "LSTM-2-32": "lstm_2_32_mnist.mat",
    "LSTM-2-64": "lstm_2_64_mnist.mat",
    "LSTM-4-32": "lstm_4_32_mnist.mat",
    "LSTM-7-32": "lstm_7_32_mnist.mat",
}


def load_mnist_lstm(config: str, data_dir: Path | None = None) -> dict:
    """Load one MNIST LSTM config and return a Cert-RNN-ready bundle.

    Returns:
        {
            "config":             str,
            "model":              cert_rnn model dict (type, D, H, L, layers, head),
            "X_test":             list[np.ndarray] of (T, D),
            "Y_test":             (N,) int labels,
            "python_predictions": (N, num_classes) PyTorch reference logits,
            "T": int (always 28 for MNIST-sequence),
            "num_classes": int,
        }
    """
    if config not in CONFIG_FILES:
        raise ValueError(f"unknown config {config!r}; valid: {list(CONFIG_FILES)}")
    root = data_dir if data_dir is not None else DATA_DIR
    d = scipy.io.loadmat(str(root / CONFIG_FILES[config]))

    D = int(d["D"].item())
    H = int(d["H"].item())
    L = int(d["num_layers"].item())
    C = int(d["num_classes"].item())

    layers: list[dict] = []
    if L == 1 and "W_in" in d:
        layers.append({
            "W_in": d["W_in"].astype(np.float64),
            "W_rec": d["W_rec"].astype(np.float64),
            "b": d["b"].astype(np.float64).reshape(-1),
        })
    else:
        # Multi-layer: W_in_layers is an (L, 1) MATLAB cell array.
        for i in range(L):
            layers.append({
                "W_in": d["W_in_layers"][i, 0].astype(np.float64),
                "W_rec": d["W_rec_layers"][i, 0].astype(np.float64),
                "b": d["b_layers"][i, 0].astype(np.float64).reshape(-1),
            })

    head = {
        "W": d["W_fc"].astype(np.float64),
        "b": d["b_fc"].astype(np.float64).reshape(-1),
    }

    X_cells = d["X_test"]
    X_test = [X_cells[i, 0].astype(np.float64) for i in range(X_cells.shape[0])]
    Y_test = d["Y_test"].flatten().astype(np.int64)
    python_predictions = d["python_predictions"].astype(np.float64)

    T = X_test[0].shape[0]
    return {
        "config": config,
        "model": {
            "type": "lstm",
            "gate_order": "ifgo",
            "D": D, "H": H, "L": L,
            "num_classes": C,
            "layers": layers,
            "head": head,
        },
        "X_test": X_test,
        "Y_test": Y_test,
        "python_predictions": python_predictions,
        "T": T,
        "num_classes": C,
    }
