"""Adapter: this repo's Cert-RNN zonotope engine (the tool under study).

Also exports the Spec-C score LOWER bound over the perturbation set
(needed by suite E3's anomaly-masking property), derived from the same
reach set: min over the ball of a sum of squares is soundly lower-
bounded by the sum of per-component interval minima."""

from __future__ import annotations

import numpy as np

from cert_rnn.verify import lstm_ae_reach, spec_c_score_ub
from cert_rnn.zono import zono_sub

from experiments.verifiers.base import BoundVerifier, Case


class CertRnnZono(BoundVerifier):
    name = "certrnn-zono"

    def holds(self, case: Case, eps: float, threat_model: str,
              t_pert: int | None) -> bool:
        z_xh, z_x = lstm_ae_reach(case.encoder, case.decoder, case.head,
                                  case.anchor, eps, threat_model, t_pert)
        return spec_c_score_ub(z_xh, z_x) <= case.tau


class CertRnnZonoMasking(BoundVerifier):
    """Dual property for anomalous anchors: certify that NO perturbation
    in the eps-ball can push the score BELOW tau (an attacker cannot mask
    the anomaly). Sound iff score_lb(ball) >= tau."""

    name = "certrnn-zono-masking"

    def holds(self, case: Case, eps: float, threat_model: str,
              t_pert: int | None) -> bool:
        z_xh, z_x = lstm_ae_reach(case.encoder, case.decoder, case.head,
                                  case.anchor, eps, threat_model, t_pert)
        return spec_c_score_lb(z_xh, z_x) >= case.tau


def spec_c_score_lb(z_x_hat_seq, z_x_seq) -> float:
    """Sound LOWER bound on score(x') = ||AE(x') - x'||_2^2 / N over the
    perturbation set: per component d, |diff_d| >= max(0, lb, -ub) on its
    interval [lb, ub]; and min of a sum >= sum of mins."""
    T = len(z_x_hat_seq)
    if T == 0:
        return 0.0
    D = z_x_hat_seq[0].dim
    N = T * D
    total = 0.0
    for t in range(T):
        z_diff = zono_sub(z_x_hat_seq[t], z_x_seq[t])
        radius = np.sum(np.abs(z_diff.V), axis=1)
        lb = z_diff.c - radius
        ub = z_diff.c + radius
        comp_min = np.maximum(0.0, np.maximum(lb, -ub))
        total += float(np.sum(comp_min ** 2))
    return total / N
