"""cert_rnn: zonotope abstract-interpretation transformers for RNN/LSTM robustness.

The top level re-exports the full user-facing API so the common path is a
single import:

    from cert_rnn import lstm_to_model_dict, certify_radius_spec_a

Layers, lowest to highest:
  - zonotope primitives        (Zono, zono_add, ...)
  - abstract transformers      (tanh_zono, bilinear_sigmoid_tanh, ...)
  - per-step / multi-step cells (lstm_step, lstm_step_stack, rnn_step)
  - PyTorch interop            (lstm_to_model_dict, rnn_to_model_dict)
  - verification               (spec_*, certify_radius_*, bisect_epsilon)
"""

from cert_rnn.audit import lp_feasible
from cert_rnn.from_torch import lstm_to_model_dict, rnn_to_model_dict
from cert_rnn.lstm import lstm_state_init, lstm_step, lstm_step_stack
from cert_rnn.rnn import rnn_step
from cert_rnn.transformers import (
    bilinear_sigmoid_identity,
    bilinear_sigmoid_tanh,
    sigmoid_zono,
    tanh_zono,
)
from cert_rnn.verify import (
    ThreatModel,
    bisect_epsilon,
    certify_radius_spec_a,
    certify_radius_spec_c,
    lstm_ae_reach,
    lstm_reach,
    spec_a_margin,
    spec_c_holds,
    spec_c_score_ub,
)
from cert_rnn.zono import (
    PredAllocator,
    Zono,
    align_pred_space,
    get_default_allocator,
    reset_pred_allocator,
    zono_add,
    zono_sub,
)

__all__ = [
    # zonotope primitives
    "PredAllocator",
    "Zono",
    "align_pred_space",
    "get_default_allocator",
    "reset_pred_allocator",
    "zono_add",
    "zono_sub",
    # transformers
    "bilinear_sigmoid_identity",
    "bilinear_sigmoid_tanh",
    "sigmoid_zono",
    "tanh_zono",
    # cells
    "lstm_state_init",
    "lstm_step",
    "lstm_step_stack",
    "rnn_step",
    # PyTorch interop
    "lstm_to_model_dict",
    "rnn_to_model_dict",
    # verification
    "ThreatModel",
    "bisect_epsilon",
    "certify_radius_spec_a",
    "certify_radius_spec_c",
    "lstm_ae_reach",
    "lstm_reach",
    "spec_a_margin",
    "spec_c_holds",
    "spec_c_score_ub",
    # audit
    "lp_feasible",
]
__version__ = "0.0.1"
