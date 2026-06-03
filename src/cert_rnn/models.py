"""Typed model wrappers over the Cert-RNN model dicts.

These are thin, ergonomic handles around the plain dicts the engine
consumes (see cert_rnn.from_torch / cert_rnn.verify). They add:

  - discoverable constructors (`from_torch`) instead of bare dict keys,
  - shape validation at construction,
  - verification as methods (`certify_radius`, `certifies`, ...), so a
    user never has to thread `encoder, decoder, head` positionally.

The wrappers carry the underlying dict(s) verbatim and expose them
(`as_dict()` / `.encoder` etc.), so anything that already takes a dict
keeps working -- the wrappers are additive, not a replacement.

    >>> from cert_rnn import RNNModel
    >>> model = RNNModel.from_torch(my_lstm, my_fc)
    >>> radius, per_frame = model.certify_radius(x_seq, true_class=3)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cert_rnn.from_torch import (
    lstm_ae_to_model_dicts,
    lstm_to_model_dict,
    rnn_to_model_dict,
)
from cert_rnn.verify import (
    ThreatModel,
    certify_radius_spec_a,
    certify_radius_spec_c,
    lstm_ae_reach,
    lstm_reach,
    spec_a_margin,
    spec_c_holds,
    spec_c_score_ub,
)


@dataclass(frozen=True)
class RNNModel:
    """A single recurrent classifier (stacked LSTM, optional linear head).

    Wraps the model dict consumed by cert_rnn.verify. The high-level
    verification methods support LSTM models; vanilla-RNN dicts can be
    constructed (for inspection / custom reach) but have no shipped spec.
    """

    spec: dict

    @classmethod
    def from_torch(cls, rec, fc=None) -> "RNNModel":
        """Build from a PyTorch nn.LSTM / nn.LSTMCell / nn.RNN, plus an
        optional nn.Linear classifier head."""
        import torch.nn as nn

        if isinstance(rec, nn.RNN):
            return cls(rnn_to_model_dict(rec, fc))
        return cls(lstm_to_model_dict(rec, fc))

    @property
    def type(self) -> str:
        return self.spec["type"]

    @property
    def D(self) -> int:
        return self.spec["D"]

    @property
    def H(self) -> int:
        return self.spec["H"]

    @property
    def L(self) -> int:
        return self.spec["L"]

    @property
    def has_head(self) -> bool:
        return "head" in self.spec

    def as_dict(self) -> dict:
        return self.spec

    def _require_lstm(self, what: str) -> None:
        if self.spec["type"] != "lstm":
            raise NotImplementedError(
                f"{what} is only implemented for LSTM models; "
                f"this model is {self.spec['type']!r}"
            )

    def reach(
        self,
        x_seq: np.ndarray,
        eps: float,
        threat_model: ThreatModel = "single_frame",
        t_pert: int | None = None,
    ):
        """Per-timestep top-layer hidden zonotope list over the eps-ball."""
        self._require_lstm("reach")
        return lstm_reach(self.spec, x_seq, eps, threat_model, t_pert)

    def certifies_margin(
        self,
        x_seq: np.ndarray,
        eps: float,
        true_class: int,
        threat_model: ThreatModel = "single_frame",
        t_pert: int | None = None,
    ) -> bool:
        """True iff logit[true_class] provably dominates over the eps-ball."""
        self._require_lstm("certifies_margin")
        return spec_a_margin(self.spec, x_seq, eps, true_class, threat_model, t_pert)

    def certify_radius(
        self,
        x_seq: np.ndarray,
        true_class: int,
        eps_init: float = 0.5,
        n_iters: int = 12,
        threat_model: ThreatModel = "single_frame",
    ) -> tuple[float, np.ndarray | None]:
        """Algorithm 1 certified radius for the classifier-margin spec."""
        self._require_lstm("certify_radius")
        return certify_radius_spec_a(
            self.spec, x_seq, true_class, eps_init, n_iters, threat_model
        )


@dataclass(frozen=True)
class LSTMAutoencoder:
    """An LSTM autoencoder (encoder + decoder + per-step linear head) for
    the Spec-C reconstruction-error property."""

    encoder: dict
    decoder: dict
    head: dict

    @classmethod
    def from_torch(cls, encoder, decoder, head) -> "LSTMAutoencoder":
        """Build from PyTorch modules. encoder/decoder may each be an
        nn.LSTM, nn.LSTMCell, or a sequence of nn.LSTMCell; head is an
        nn.Linear back to input space."""
        d = lstm_ae_to_model_dicts(encoder, decoder, head)
        return cls(d["encoder"], d["decoder"], d["head"])

    @property
    def H(self) -> int:
        return self.encoder["H"]

    @property
    def D(self) -> int:
        return self.encoder["D"]

    def reach(
        self,
        x_anchor: np.ndarray,
        eps: float,
        threat_model: ThreatModel = "single_frame",
        t_pert: int | None = None,
    ):
        """Forward the autoencoder; returns (z_x_hat_seq, z_x_seq)."""
        return lstm_ae_reach(
            self.encoder, self.decoder, self.head, x_anchor, eps, threat_model, t_pert
        )

    def score_ub(
        self,
        x_anchor: np.ndarray,
        eps: float,
        threat_model: ThreatModel = "single_frame",
        t_pert: int | None = None,
    ) -> float:
        """Sound upper bound on the reconstruction score over the eps-ball."""
        z_xh, z_x = self.reach(x_anchor, eps, threat_model, t_pert)
        return spec_c_score_ub(z_xh, z_x)

    def certifies(
        self,
        x_anchor: np.ndarray,
        eps: float,
        tau: float,
        threat_model: ThreatModel = "single_frame",
        t_pert: int | None = None,
    ) -> bool:
        """True iff the sound score upper bound is <= tau over the eps-ball."""
        return spec_c_holds(
            self.encoder, self.decoder, self.head, x_anchor, eps, tau,
            threat_model, t_pert,
        )

    def certify_radius(
        self,
        x_anchor: np.ndarray,
        tau: float,
        eps_init: float = 0.5,
        n_iters: int = 12,
        threat_model: ThreatModel = "single_frame",
    ) -> tuple[float, np.ndarray | None]:
        """Algorithm 1 certified radius for the reconstruction-error spec."""
        return certify_radius_spec_c(
            self.encoder, self.decoder, self.head, x_anchor, tau,
            eps_init, n_iters, threat_model,
        )
