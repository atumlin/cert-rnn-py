"""Benchmark: deterministic synthetic scaling family (suite E2 + E3).

Purpose: the IEEE-9 checkpoints pin four points in (H, T, D, L) space;
scalability claims need a lattice. This module trains small LSTM-AEs of
any requested shape on a synthetic quasi-periodic multivariate process
(power-system-flavoured: coupled sinusoids + harmonics + noise), caches
the checkpoint + anchor + tau under benchmarks/generated/, and returns
Cases. Everything is seeded: same (H, T, D, L, seed) -> byte-identical
benchmark on every machine.

Anomalies (suite E3): injected additive spikes / level shifts / channel
dropouts on held-out windows, calibrated so the clean model flags them
(score > tau) with margin comparable to the normal-window margin.

This is a benchmark GENERATOR by design -- the paper ships it as an
artifact so any (H, T, D, L) point a reviewer asks for is reproducible
on demand rather than frozen into a fixed tarball."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.common import GENERATED_DIR, count_params
from experiments.verifiers.base import Case


@dataclass(frozen=True)
class SynthSpec:
    H: int = 16
    T: int = 30
    D: int = 9
    L: int = 1           # enc layers == dec layers
    seed: int = 0
    n_train: int = 512   # training windows
    epochs: int = 60

    @property
    def key(self) -> str:
        return f"synth-H{self.H}-T{self.T}-D{self.D}-L{self.L}-s{self.seed}"


def _process_params(spec: SynthSpec, rng: np.random.Generator) -> tuple:
    """Draw the process ONCE per benchmark point: channel mixing matrix +
    oscillator frequencies. Train and held-out windows must come from the
    same process (only phases and noise vary per window)."""
    K = max(3, spec.D // 3)
    mix = rng.normal(0, 1, (spec.D, K)) / np.sqrt(K)
    freqs = rng.uniform(0.5, 2.0, K)
    return mix, freqs


def _gen_process(spec: SynthSpec, n_windows: int, rng: np.random.Generator,
                 params: tuple) -> np.ndarray:
    """(n, T, D) quasi-periodic multivariate windows in [0, 1]-ish range.

    Channels are fixed random mixtures of K shared latent oscillators
    (base + harmonic) plus per-channel noise -- shared structure is what
    gives an autoencoder something to compress, mirroring how bus
    measurements co-move through the grid's physics."""
    mix, freqs = params
    K = freqs.shape[0]
    phases = rng.uniform(0, 2 * np.pi, (n_windows, K))
    t = np.linspace(0, 4 * np.pi, spec.T)
    lat = np.sin(freqs[None, :, None] * t[None, None, :] + phases[:, :, None])
    lat += 0.3 * np.sin(2 * freqs[None, :, None] * t[None, None, :]
                        + phases[:, :, None])
    x = np.einsum("dk,nkt->ntd", mix, lat)
    x += 0.02 * rng.normal(0, 1, x.shape)
    return 0.5 + 0.25 * x  # roughly [0, 1], like the normalized IEEE-9 data


def _inject_anomaly(x: np.ndarray, kind: str, rng: np.random.Generator
                    ) -> np.ndarray:
    """Additive anomaly on one window (T, D). Magnitudes are sized so a
    reconstruction score averaged over all T*D components moves well
    past a p99-calibrated tau (a single-point single-channel blip would
    vanish in the mean -- realistic grid events hit several correlated
    channels for several steps)."""
    y = x.copy()
    T, D = y.shape
    n_ch = max(2, D // 3)
    chans = rng.choice(D, size=n_ch, replace=False)
    t0 = rng.integers(T // 4, 3 * T // 4)
    if kind == "spike":
        dur = max(2, T // 10)
        y[t0:t0 + dur, chans] += rng.choice([-1, 1]) * rng.uniform(0.6, 1.0)
    elif kind == "step":
        y[t0:, chans] += rng.choice([-1, 1]) * rng.uniform(0.4, 0.6)
    elif kind == "dropout":
        y[t0:t0 + max(3, T // 3), chans] = 0.0
    else:
        raise ValueError(kind)
    return y


def train_lstm_ae(x_train: np.ndarray, H: int, L: int, seed: int = 0,
                  epochs: int = 60):
    """Train enc/dec LSTMCell stacks + linear head (same architecture
    family as the IEEE-9 checkpoints) on (n, T, D) windows. Returns
    (enc_cells, dec_cells, head, train_scores). Shared by the synthetic
    family and the public-dataset benchmarks (public_tsad.py)."""
    import torch
    import torch.nn as nn

    D = x_train.shape[2]
    torch.manual_seed(seed)
    dt = torch.float64
    enc = nn.ModuleList([nn.LSTMCell(D if i == 0 else H, H)
                         for i in range(L)]).to(dt)
    dec = nn.ModuleList([nn.LSTMCell(H, H) for _ in range(L)]).to(dt)
    head = nn.Linear(H, D).to(dt)
    params = (list(enc.parameters()) + list(dec.parameters())
              + list(head.parameters()))
    opt = torch.optim.Adam(params, lr=1e-2)

    def forward(xb: "torch.Tensor") -> "torch.Tensor":
        B, T, _D = xb.shape
        h = [xb.new_zeros(B, H) for _ in range(L)]
        c = [xb.new_zeros(B, H) for _ in range(L)]
        for t in range(T):
            inp = xb[:, t]
            for i in range(L):
                h[i], c[i] = enc[i](inp, (h[i], c[i]))
                inp = h[i]
        lat = h[-1]
        hd = [xb.new_zeros(B, H) for _ in range(L)]
        cd = [xb.new_zeros(B, H) for _ in range(L)]
        outs = []
        for _t in range(T):
            inp = lat
            for i in range(L):
                hd[i], cd[i] = dec[i](inp, (hd[i], cd[i]))
                inp = hd[i]
            outs.append(head(hd[-1]))
        return torch.stack(outs, dim=1)

    xb_all = torch.as_tensor(x_train, dtype=dt)
    for _ep in range(epochs):
        perm = torch.randperm(xb_all.shape[0])
        for i in range(0, xb_all.shape[0], 64):
            xb = xb_all[perm[i:i + 64]]
            loss = (forward(xb) - xb).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        scores = (forward(xb_all) - xb_all).pow(2).flatten(1).mean(1).numpy()
    return enc, dec, head, scores


def build(spec: SynthSpec, force: bool = False) -> dict:
    """Train-or-load one synthetic benchmark point. Returns the cached
    payload {state_dict path, meta}; use get_cases() for Case objects."""
    import torch

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    pt = GENERATED_DIR / f"{spec.key}.pt"
    js = GENERATED_DIR / f"{spec.key}.json"
    if pt.exists() and js.exists() and not force:
        return json.loads(js.read_text())

    rng = np.random.default_rng(spec.seed)
    params = _process_params(spec, rng)
    x_train = _gen_process(spec, spec.n_train, rng, params)
    x_norm = _gen_process(spec, 32, rng, params)  # held-out normal windows
    kinds = ["spike", "step", "dropout"]
    x_anom = np.stack([
        _inject_anomaly(x_norm[i % 32], kinds[i % 3], rng) for i in range(16)
    ])

    enc, dec, head, train_scores = train_lstm_ae(
        x_train, spec.H, spec.L, seed=spec.seed, epochs=spec.epochs)
    tau = float(np.quantile(train_scores, 0.99) * 1.1)

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
    np.save(GENERATED_DIR / f"{spec.key}_normal.npy", x_norm)
    np.save(GENERATED_DIR / f"{spec.key}_anom.npy", x_anom)
    meta = {
        "key": spec.key, "H": spec.H, "T": spec.T, "D": spec.D, "L": spec.L,
        "seed": spec.seed, "tau": tau,
        "train_score_p50": float(np.median(train_scores)),
        "train_score_p99": float(np.quantile(train_scores, 0.99)),
        "epochs": spec.epochs, "n_train": spec.n_train,
    }
    js.write_text(json.dumps(meta, indent=2))
    return meta


def get_cases(spec: SynthSpec, anchors: str = "normal", n_anchors: int = 3
              ) -> list[Case]:
    """Cases for one benchmark point. anchors: "normal" (false-alarm
    property, score<=tau) or "anom" (masking property, score>=tau)."""
    import torch

    from cert_rnn import LSTMAutoencoder

    meta = build(spec)
    sd = torch.load(GENERATED_DIR / f"{spec.key}.pt", map_location="cpu",
                    weights_only=True)
    import torch.nn as nn
    enc_cells, dec_cells = [], []
    for i in range(spec.L):
        cell = nn.LSTMCell(spec.D if i == 0 else spec.H, spec.H).double()
        cell.load_state_dict({k.split(".", 2)[2]: v for k, v in sd.items()
                              if k.startswith(f"enc_cells.{i}.")})
        enc_cells.append(cell)
    for i in range(spec.L):
        cell = nn.LSTMCell(spec.H, spec.H).double()
        cell.load_state_dict({k.split(".", 2)[2]: v for k, v in sd.items()
                              if k.startswith(f"dec_cells.{i}.")})
        dec_cells.append(cell)
    head = nn.Linear(spec.H, spec.D).double()
    head.load_state_dict({k.split(".", 1)[1]: v for k, v in sd.items()
                          if k.startswith("head.")})
    ae = LSTMAutoencoder.from_torch(enc_cells, dec_cells, head)

    windows = np.load(GENERATED_DIR / f"{spec.key}_{anchors}.npy")
    # Certifiable subjects only: a normal anchor must sit below tau, an
    # anomalous one above (otherwise the property is vacuous at eps=0).
    scores = np.array([float(ae.score(w.astype(np.float64)))
                       for w in windows])
    keep = scores <= meta["tau"] if anchors == "normal" else scores > meta["tau"]
    if keep.sum() == 0:
        raise RuntimeError(
            f"{spec.key}: no {anchors} window on the correct side of "
            f"tau={meta['tau']:.4g} (scores {scores.min():.4g}.."
            f"{scores.max():.4g}); the model is too weak a detector -- "
            "increase epochs/n_train in SynthSpec")
    windows = windows[keep]
    cases = []
    for i in range(min(n_anchors, windows.shape[0])):
        anchor = windows[i].astype(np.float64)
        cases.append(Case(
            benchmark_id=spec.key,
            anchor_id=f"{anchors}#{i}",
            encoder=ae.encoder, decoder=ae.decoder, head=ae.head,
            anchor=anchor, tau=meta["tau"],
            meta={"H": spec.H, "T": spec.T, "D": spec.D,
                  "L_enc": spec.L, "L_dec": spec.L,
                  "params": count_params(ae.encoder, ae.decoder, ae.head),
                  "anchor_score": float(ae.score(anchor)),
                  "domain": "synthetic quasi-periodic"},
        ))
    return cases
