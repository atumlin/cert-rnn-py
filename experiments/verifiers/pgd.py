"""Adapter: PGD falsification (empirical UPPER bound on the radius).

Not a verifier -- the complement. At each eps it runs multi-restart
projected gradient ascent on the concrete score (descent for the masking
property) and reports whether a concrete violation exists. The bisected
result upper-bounds the true robustness radius, so the gap
[certified lower bound, PGD upper bound] measures each sound tool's
looseness (suite E4)."""

from __future__ import annotations

import numpy as np
import torch

from experiments.verifiers.base import AttackVerifier, Case
from experiments.verifiers.torch_ae import TorchLSTMAE


class PGDAttack(AttackVerifier):
    name = "pgd"

    def __init__(self, n_restarts: int = 10, n_steps: int = 60,
                 direction: str = "up", seed: int = 0):
        # direction "up": violate score<=tau (false-alarm property);
        # "down": violate score>=tau (anomaly-masking property).
        self.n_restarts = n_restarts
        self.n_steps = n_steps
        self.direction = direction
        self.seed = seed
        self._ae_cache: tuple[int, TorchLSTMAE] | None = None

    def _ae(self, case: Case) -> TorchLSTMAE:
        if self._ae_cache is None or self._ae_cache[0] != id(case):
            self._ae_cache = (id(case), TorchLSTMAE(case.encoder, case.decoder,
                                                    case.head))
        return self._ae_cache[1]

    def violates(self, case: Case, eps: float, threat_model: str,
                 t_pert: int | None) -> bool:
        ae = self._ae(case)
        torch.manual_seed(self.seed)
        x0 = torch.as_tensor(case.anchor, dtype=torch.float64)
        T, D = x0.shape
        B = self.n_restarts

        if threat_model == "single_frame":
            mask = torch.zeros(T, D, dtype=torch.float64)
            mask[t_pert] = 1.0
        else:
            mask = torch.ones(T, D, dtype=torch.float64)

        delta = (torch.rand(B, T, D, dtype=torch.float64) * 2 - 1) * eps * mask
        delta[0] = 0.0  # one restart from the anchor itself
        delta.requires_grad_(True)
        step = eps / 5.0
        sign = 1.0 if self.direction == "up" else -1.0

        for _ in range(self.n_steps):
            scores = ae.score(x0.unsqueeze(0) + delta)
            loss = (sign * scores).sum()
            (grad,) = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta += step * torch.sign(grad)  # ascent on loss
                delta.clamp_(-eps, eps)
                delta *= mask
            delta.requires_grad_(True)

        with torch.no_grad():
            final = ae.score(x0.unsqueeze(0) + delta)
        if self.direction == "up":
            return bool((final > case.tau).any())
        return bool((final < case.tau).any())
