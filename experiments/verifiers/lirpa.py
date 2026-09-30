"""Adapter: auto_LiRPA (Xu et al., NeurIPS 2020) on the unrolled AE.

Represents the linear-relaxation family (IBP / CROWN) that dominates
feedforward verification (alpha-beta-CROWN, VNN-COMP). The AE is traced
as an explicit-op unrolled graph (torch_ae.py); auto_LiRPA computes
interval or backward-mode linear bounds on every residual component
AE(x') - x', and the Spec-C score bound is assembled from the
componentwise |residual| maxima exactly as spec_c_score_ub does -- so
the comparison isolates the abstract DOMAIN, holding the (quadratic)
score bounding step fixed across tools.

method: "IBP" | "backward" (CROWN) | "CROWN-optimized" (alpha-CROWN).
Construction of the BoundedModule is per (case, t_pert) and cached; its
cost is included in `seconds` (first call) like any tool's setup."""

from __future__ import annotations

import numpy as np
import torch

from experiments.verifiers.base import BoundVerifier, Case
from experiments.verifiers.torch_ae import (
    MultiFrameResidualAE,
    SingleFrameResidualAE,
    TorchLSTMAE,
)


class LirpaBound(BoundVerifier):
    def __init__(self, method: str = "IBP", masking: bool = False):
        self.method = method
        self.masking = masking
        self.name = f"lirpa-{method.lower()}" + ("-masking" if masking else "")
        self._cache: dict = {}

    def _bounded(self, case: Case, threat_model: str, t_pert: int | None):
        from auto_LiRPA import BoundedModule

        key = (id(case), threat_model, t_pert)
        if key not in self._cache:
            # float32: auto_LiRPA's backward (CROWN) path assumes fp32;
            # a fp32 bound is a caveat, not a soundness proof to 1e-7 --
            # recorded in tool_meta by the suite layer.
            ae = TorchLSTMAE(case.encoder, case.decoder, case.head,
                             dtype=torch.float32)
            if threat_model == "single_frame":
                mod = SingleFrameResidualAE(ae, case.anchor, t_pert,
                                            dtype=torch.float32)
                dummy = torch.as_tensor(case.anchor[t_pert],
                                        dtype=torch.float32).unsqueeze(0)
            else:
                mod = MultiFrameResidualAE(ae)
                dummy = torch.as_tensor(case.anchor,
                                        dtype=torch.float32).unsqueeze(0)
            self._cache.clear()  # one live module; graphs are per-frame
            self._cache[key] = (BoundedModule(mod, dummy), dummy)
        return self._cache[key]

    def _residual_bounds(self, case: Case, eps: float, threat_model: str,
                         t_pert: int | None):
        from auto_LiRPA import BoundedTensor, PerturbationLpNorm

        bounded, dummy = self._bounded(case, threat_model, t_pert)
        ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
        x = BoundedTensor(dummy, ptb)
        lb, ub = bounded.compute_bounds(x=(x,), method=self.method)
        return (lb.detach().numpy().ravel(), ub.detach().numpy().ravel())

    def holds(self, case: Case, eps: float, threat_model: str,
              t_pert: int | None) -> bool:
        lb, ub = self._residual_bounds(case, eps, threat_model, t_pert)
        N = lb.size
        if not self.masking:
            comp_max = np.maximum(np.abs(lb), np.abs(ub))
            return float(np.sum(comp_max ** 2)) / N <= case.tau
        comp_min = np.maximum(0.0, np.maximum(lb, -ub))
        return float(np.sum(comp_min ** 2)) / N >= case.tau
