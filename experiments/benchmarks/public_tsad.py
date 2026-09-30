"""Benchmark: public time-series anomaly-detection datasets (suite E1/E3
cross-domain generality).

Selected for public direct download (no request forms), multivariate
where possible, and community standing:

  smd     Server Machine Dataset (OmniAnomaly, NetManAIOps) -- 38-dim
          server telemetry, labeled anomalies. Clone
          https://github.com/NetManAIOps/OmniAnomaly and point
          --data-root at its ServerMachineDataset/ directory.
  morris  Mississippi State/ORNL power-system attack dataset (Adhikari,
          Pan, Morris et al.) -- 4 PMUs, 128 features, natural-fault vs
          attack labels. Download from
          https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets
  skab    SKAB (waico/SKAB, GPL-3.0) -- 8-dim water-circulation testbed.

SWaT/WADI/EPIC (iTrust) require a request form (~3 working days) and are
NOT auto-downloaded; the loaders accept their CSVs once obtained.

Each loader: z-normalize per channel on the training split, slide
non-overlapping windows of length T, train the shared LSTM-AE
architecture (synthetic.train_lstm_ae), calibrate tau = 1.1 x p99 of
training scores, and emit normal + anomalous Cases. Models/windows are
cached under benchmarks/generated/ keyed by (dataset, entity, T, H, L,
seed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.common import GENERATED_DIR, count_params
from experiments.verifiers.base import Case
from experiments.benchmarks.synthetic import train_lstm_ae

DATA_ROOT = Path(__file__).parent / "data"

_DOWNLOAD_HELP = {
    "smd": ("SMD not found. Clone https://github.com/NetManAIOps/OmniAnomaly "
            "and copy/symlink its ServerMachineDataset/ to "
            "experiments/benchmarks/data/smd/ (expects train/, test/, "
            "test_label/ with machine-*.txt CSVs)."),
    "skab": ("SKAB not found. Clone https://github.com/waico/SKAB and "
             "copy/symlink its data/ to experiments/benchmarks/data/skab/."),
    "morris": ("Morris power-attack dataset not found. Download the "
               "multiclass PMU CSVs from https://sites.google.com/a/uah.edu/"
               "tommy-morris-uah/ics-data-sets into "
               "experiments/benchmarks/data/morris/."),
}


def _require(dataset: str) -> Path:
    root = DATA_ROOT / dataset
    if not root.exists():
        raise FileNotFoundError(_DOWNLOAD_HELP[dataset])
    return root


def _window(x: np.ndarray, T: int) -> np.ndarray:
    n = (x.shape[0] // T) * T
    return x[:n].reshape(-1, T, x.shape[1])


def load_smd_entity(entity: str = "machine-1-1", T: int = 30):
    """Returns (train_windows, test_windows, test_window_labels)."""
    root = _require("smd")
    tr = np.genfromtxt(root / "train" / f"{entity}.txt", delimiter=",")
    te = np.genfromtxt(root / "test" / f"{entity}.txt", delimiter=",")
    lab = np.genfromtxt(root / "test_label" / f"{entity}.txt", delimiter=",")
    mu, sd = tr.mean(0), tr.std(0) + 1e-8
    trw = _window((tr - mu) / sd, T)
    tew = _window((te - mu) / sd, T)
    labw = _window(lab.reshape(-1, 1), T).squeeze(-1).max(axis=1)  # any-step
    return trw, tew, labw


def get_cases(dataset: str = "smd", entity: str = "machine-1-1", T: int = 30,
              H: int = 16, L: int = 1, seed: int = 0, epochs: int = 20,
              n_normal: int = 5, n_anom: int = 5) -> list[Case]:
    """Train-or-load an LSTM-AE on one dataset entity; emit Cases for
    n_normal correctly-negative and n_anom correctly-positive windows
    (only windows the clean model classifies correctly are certifiable
    subjects -- same convention as certified-accuracy on classifiers)."""
    if dataset != "smd":
        raise NotImplementedError(
            f"loader for {dataset!r} not wired yet (see module docstring; "
            "smd is the reference implementation)")
    import torch
    from cert_rnn import LSTMAutoencoder

    key = f"{dataset}-{entity}-T{T}-H{H}-L{L}-s{seed}"
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    pt, js = GENERATED_DIR / f"{key}.pt", GENERATED_DIR / f"{key}.json"

    trw, tew, labw = load_smd_entity(entity, T)
    if not (pt.exists() and js.exists()):
        enc, dec, head, scores = train_lstm_ae(trw, H, L, seed=seed,
                                               epochs=epochs)
        tau = float(np.quantile(scores, 0.99) * 1.1)
        sd = {}
        for i, cell in enumerate(enc):
            for k, v in cell.state_dict().items():
                sd[f"enc_cells.{i}.{k}"] = v
        for i, cell in enumerate(dec):
            for k, v in cell.state_dict().items():
                sd[f"dec_cells.{i}.{k}"] = v
        for k, v in head.state_dict().items():
            sd[f"head.{k}"] = v
        torch.save(sd, pt)
        js.write_text(json.dumps({"tau": tau, "H": H, "T": T, "L": L,
                                  "D": int(trw.shape[2])}))
    meta = json.loads(js.read_text())
    sd = torch.load(pt, map_location="cpu", weights_only=True)

    import torch.nn as nn
    D = meta["D"]
    enc_cells = []
    for i in range(L):
        cell = nn.LSTMCell(D if i == 0 else H, H).double()
        cell.load_state_dict({k.split(".", 2)[2]: v for k, v in sd.items()
                              if k.startswith(f"enc_cells.{i}.")})
        enc_cells.append(cell)
    dec_cells = []
    for i in range(L):
        cell = nn.LSTMCell(H, H).double()
        cell.load_state_dict({k.split(".", 2)[2]: v for k, v in sd.items()
                              if k.startswith(f"dec_cells.{i}.")})
        dec_cells.append(cell)
    head = nn.Linear(H, D).double()
    head.load_state_dict({k.split(".", 1)[1]: v for k, v in sd.items()
                          if k.startswith("head.")})
    ae = LSTMAutoencoder.from_torch(enc_cells, dec_cells, head)
    tau = meta["tau"]

    scores = np.array([float(ae.score(w.astype(np.float64))) for w in tew])
    normal_idx = np.where((labw == 0) & (scores <= tau))[0][:n_normal]
    anom_idx = np.where((labw == 1) & (scores > tau))[0][:n_anom]

    def mk(i, tag):
        return Case(
            benchmark_id=key, anchor_id=f"{tag}#{int(i)}",
            encoder=ae.encoder, decoder=ae.decoder, head=ae.head,
            anchor=tew[i].astype(np.float64), tau=tau,
            meta={"H": H, "T": T, "D": D, "L_enc": L, "L_dec": L,
                  "params": count_params(ae.encoder, ae.decoder, ae.head),
                  "anchor_score": float(scores[i]),
                  "label": tag, "domain": f"{dataset}/{entity}"},
        )

    return ([mk(i, "normal") for i in normal_idx]
            + [mk(i, "anom") for i in anom_idx])
