"""MNIST-sequence parity: same checkpoint (LSTM-1-32), same Algorithm-1
trajectory must produce per-sample certified radii that match MATLAB's
saved values to bisection precision.

The MATLAB fixture matlab_results/cert_radius_mnist_results.mat was
produced by cert_radius_mnist.m on the LSTM-1-32 checkpoint over 30
test samples. Python runs the same Algorithm 1 (single-frame, min over
T=28 frames) and asserts the same per-sample numbers.

Scope: 2 samples by default (~2-3 minutes pure-Python). The headline
mean across all 30 samples (0.01766) is a sanity check, not a unit
test: it requires the full run, which we don't gate on.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import scipy.io

EXAMPLES = Path(__file__).parent.parent / "examples" / "mnist_sequence"
sys.path.insert(0, str(EXAMPLES))

# noqa: E402
from mnist_loader import load_mnist_lstm  # type: ignore[import]
from demo_multilayer import (  # type: ignore[import]
    cert_radius_singleframe,
    parity_at_eps_zero,
    perstep_widths,
)


ATOL = 1e-4   # bisection step at iter 13 = 0.5^13 ~ 1.2e-4


@pytest.fixture(scope="module")
def lstm_1_32():
    return load_mnist_lstm("LSTM-1-32")


@pytest.fixture(scope="module")
def matlab_radii():
    d = scipy.io.loadmat(
        str(EXAMPLES / "matlab_results" / "cert_radius_mnist_results.mat")
    )
    return d["radii"].flatten().astype(np.float64)


def test_eps_zero_parity_matches_matlab(lstm_1_32):
    """parity_err in MATLAB is ~1.93e-6 (float32 cast drift in PyTorch
    forward); Python must agree within float64 precision of that."""
    err = parity_at_eps_zero(lstm_1_32, n_parity=10)
    # MATLAB demo records parity_err = 1.93e-6 for LSTM-1-32.
    assert err < 5e-6, f"parity_err = {err:.2e} (matlab ref ~1.93e-6)"


def test_perstep_widths_single_layer_matches_matlab():
    """LSTM-1-32 has no stacked-layer predicate-aliasing path, so Python
    and MATLAB widths must match to float noise."""
    bundle = load_mnist_lstm("LSTM-1-32")
    widths_py = perstep_widths(bundle["model"], bundle["X_test"][0], eps=0.005, t_pert=0)
    mat = scipy.io.loadmat(str(EXAMPLES / "matlab_results" / "demo_multilayer_results.mat"))
    widths_mat = mat["bound_widths"][0, 0].flatten().astype(np.float64)
    assert float(np.max(np.abs(widths_py - widths_mat))) < 1e-10


@pytest.mark.parametrize(
    "config_idx, config",
    list(enumerate(["LSTM-2-32", "LSTM-2-64", "LSTM-4-32", "LSTM-7-32"], start=1)),
)
def test_perstep_widths_multi_layer_python_at_least_as_loose(config_idx, config):
    """For stacked LSTMs, Python's Minkowski-padded predicate alignment
    produces bounds at least as loose as MATLAB's positional padding,
    AND they must be sound. MATLAB's bounds are tighter because its
    positional padding aliases the previous-step layer-i+1 fresh preds
    with the current-step layer-i fresh preds at the same column index,
    even though those alphas are independent. The aliasing introduces a
    false correlation that can make MATLAB's bound unsound in adversarial
    cases (random-sample red-teams won't catch it). Python's pred_ids
    keep them disjoint -> correctly looser bounds.

    This test asserts Python >= MATLAB everywhere, and validates Python
    soundness with concrete sampling at frame 0 perturbation.
    """
    bundle = load_mnist_lstm(config)
    x0 = bundle["X_test"][0]
    eps = 0.005
    widths_py = perstep_widths(bundle["model"], x0, eps=eps, t_pert=0)
    mat = scipy.io.loadmat(str(EXAMPLES / "matlab_results" / "demo_multilayer_results.mat"))
    widths_mat = mat["bound_widths"][config_idx, 0].flatten().astype(np.float64)

    # Python at least as loose as MATLAB.
    gap = widths_py - widths_mat
    assert float(gap.min()) >= -1e-10, (
        f"{config}: Python narrower than MATLAB by {-float(gap.min()):.3e} at some frame "
        "-- this would be a regression vs the known-soundness-suite Minkowski semantics"
    )

    # Python sound on this anchor.
    def sigm(z): return 1.0 / (1.0 + np.exp(-z))
    def fwd(x):
        H, L = bundle["model"]["H"], bundle["model"]["L"]
        h = [np.zeros(H) for _ in range(L)]
        c = [np.zeros(H) for _ in range(L)]
        h_top_seq = []
        for t in range(x.shape[0]):
            inp = x[t]
            for i in range(L):
                lyr = bundle["model"]["layers"][i]
                pre = lyr["W_in"] @ inp + lyr["W_rec"] @ h[i] + lyr["b"]
                ii, ff, gg, oo = sigm(pre[:H]), sigm(pre[H:2*H]), np.tanh(pre[2*H:3*H]), sigm(pre[3*H:4*H])
                c[i] = ff * c[i] + ii * gg
                h[i] = oo * np.tanh(c[i])
                inp = h[i]
            h_top_seq.append(h[-1].copy())
        return np.array(h_top_seq)

    D = bundle["model"]["D"]
    rng = np.random.default_rng(123)
    nominal = fwd(x0)
    max_dev = np.zeros(28)
    for _ in range(200):
        x = x0.copy()
        x[0] = x0[0] + eps * (2 * rng.random(D) - 1)
        max_dev = np.maximum(max_dev, np.max(np.abs(fwd(x) - nominal), axis=1))

    # widths >= 2*max_dev means concrete reachable range fits inside cert bound.
    py_violation = (2 * max_dev) - widths_py
    assert float(py_violation.max()) <= 1e-9, (
        f"{config}: Python cert width below concrete deviation by "
        f"{float(py_violation.max()):.3e} -- Python unsound on this anchor"
    )


@pytest.mark.parametrize("sample_idx", [0, 1])
def test_lstm_1_32_per_sample_parity(lstm_1_32, matlab_radii, sample_idx):
    """Per-sample certified radius must match MATLAB within bisection step."""
    model = lstm_1_32["model"]
    x = lstm_1_32["X_test"][sample_idx]
    y = int(lstm_1_32["Y_test"][sample_idx])
    t0 = time.perf_counter()
    eps_py = cert_radius_singleframe(model, x, y)
    dt = time.perf_counter() - t0
    eps_matlab = float(matlab_radii[sample_idx])
    print(
        f"\n  sample {sample_idx}: py={eps_py:.6f} matlab={eps_matlab:.6f}  "
        f"diff={eps_py - eps_matlab:+.3e}  ({dt:.1f}s)"
    )
    assert abs(eps_py - eps_matlab) <= ATOL, (
        f"sample {sample_idx}: py={eps_py:.6e}, matlab={eps_matlab:.6e}"
    )
