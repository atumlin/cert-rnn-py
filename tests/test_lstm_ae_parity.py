"""Parity check: per-frame certified eps (size S) must match the MATLAB
reference (matlab_results/certrnn_lstm_ae_S.mat).

S is the smallest LSTM-AE (H=4) so the full T=30 sweep takes ~15s in
pure Python. M is run on a small frame subset; L is excluded (~70 min
in pure Python — covered by the example script when opted in).

The comparison is byte-tight: both engines run the same Algorithm 1
trajectory on the same checkpoint, so eps values must agree modulo
last-bit float drift in spec_c_score_ub.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io

EXAMPLES = Path(__file__).parent.parent / "examples" / "lstm_ae_ieee9"
sys.path.insert(0, str(EXAMPLES))

# noqa: E402
from ae_loader import load_lstm_ae  # type: ignore[import]

from cert_rnn.verify import bisect_epsilon, spec_c_holds


ATOL = 1e-6   # bisect step at iter 13 is 0.5^13 ~ 1.2e-4; float drift << that


def _bisect_frame(m: dict, t_pert: int, eps_init: float = 0.5, n_iters: int = 12) -> float:
    return bisect_epsilon(
        lambda eps, _t=t_pert: spec_c_holds(
            m["encoder"], m["decoder"], m["head"], m["anchor"], eps, m["tau"],
            threat_model="single_frame", t_pert=_t,
        ),
        eps_init=eps_init,
        n_iters=n_iters,
    )


def _matlab_eps_per_frame(size: str) -> np.ndarray:
    mat = scipy.io.loadmat(str(EXAMPLES / "matlab_results" / f"certrnn_lstm_ae_{size}.mat"))
    return mat["eps_per_frame"].flatten().astype(np.float64)


@pytest.mark.parametrize("t_pert", [0, 5, 10, 15, 20, 25, 29])
def test_lstm_ae_S_per_frame_parity(t_pert):
    """Spot-check a sweep of frames on size S against MATLAB."""
    m = load_lstm_ae("S")
    eps_py = _bisect_frame(m, t_pert)
    eps_matlab = _matlab_eps_per_frame("S")[t_pert]
    assert abs(eps_py - eps_matlab) <= ATOL, (
        f"frame {t_pert}: py={eps_py:.6e}, matlab={eps_matlab:.6e}, "
        f"diff={abs(eps_py - eps_matlab):.3e}"
    )


def test_lstm_ae_S_certified_radius_parity():
    """Full T=30 sweep on size S; certified radius (min over frames) must
    match MATLAB exactly."""
    m = load_lstm_ae("S")
    eps_py = np.array([_bisect_frame(m, t) for t in range(m["T"])])
    eps_matlab = _matlab_eps_per_frame("S")
    max_diff = float(np.max(np.abs(eps_py - eps_matlab)))
    assert max_diff <= ATOL, (
        f"max |eps_py - eps_matlab| = {max_diff:.3e} > atol={ATOL}; "
        f"per-frame diffs:\n{eps_py - eps_matlab}"
    )
    py_cert = float(eps_py.min())
    matlab_cert = float(eps_matlab.min())
    assert abs(py_cert - matlab_cert) <= ATOL, (
        f"cert_radius: py={py_cert:.6e}, matlab={matlab_cert:.6e}"
    )


def test_lstm_ae_M_frame_zero_parity():
    """Single-frame parity on size M (H=16). One frame to keep test fast."""
    m = load_lstm_ae("M")
    eps_py = _bisect_frame(m, t_pert=0)
    eps_matlab = _matlab_eps_per_frame("M")[0]
    assert abs(eps_py - eps_matlab) <= ATOL, (
        f"M frame 0: py={eps_py:.6e}, matlab={eps_matlab:.6e}"
    )
