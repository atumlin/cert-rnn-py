"""Per-transformer soundness audits.

For each transformer:
  - Fixed cases (mirrors MATLAB test_audit_strict.m).
  - Random fuzz across (c, V).
For every sample, the LP-feasibility check (cert_rnn.audit.lp_feasible)
must report the concrete output point inside the abstract zonotope.

The hadamard_affine_only baseline is asserted UNSOUND on the (1+alpha)^2
witness; it mirrors NNV's Star.HadamardProduct path, the broken
transformer that the Cert-RNN bilinears replace.
"""

import numpy as np
import pytest

from cert_rnn import Zono
from cert_rnn.audit import lp_feasible
from cert_rnn.transformers import (
    bilinear_sigmoid_identity,
    bilinear_sigmoid_tanh,
    hadamard_affine_only,
    sigmoid_zono,
    tanh_zono,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _audit_unary(transformer, true_fn, z_in, n_samples, seed):
    z_out = transformer(z_in)
    rng = np.random.default_rng(seed)
    alphas = 2 * rng.random((z_in.n_pred, n_samples)) - 1
    n_viol = 0
    for s in range(n_samples):
        x = z_in.c + z_in.V @ alphas[:, s]
        y_true = true_fn(x)
        if not lp_feasible(z_out, y_true):
            n_viol += 1
    return n_viol


def _audit_binary(transformer, true_fn, z_x, z_y, n_samples, seed):
    """Both zonos must share predicates here (same pred_ids and same V cols).
    The concrete alpha vector is then shared too, so the concrete (x, y)
    are correlated -- exactly what the bilinear transformer must over-bound.
    """
    z_out = transformer(z_x, z_y)
    rng = np.random.default_rng(seed)
    alphas = 2 * rng.random((z_x.n_pred, n_samples)) - 1
    n_viol = 0
    for s in range(n_samples):
        x = z_x.c + z_x.V @ alphas[:, s]
        y = z_y.c + z_y.V @ alphas[:, s]
        y_true = true_fn(x, y)
        if not lp_feasible(z_out, y_true):
            n_viol += 1
    return n_viol


# ---------- tanh ----------

@pytest.mark.parametrize(
    "label,c,r",
    [
        ("pos", 1.0, 0.5),
        ("pos_wide", 2.5, 1.5),
        ("neg", -1.0, 0.5),
        ("mixed", 0.0, 1.0),
        ("wide", 0.0, 3.0),
        ("saturated", 4.5, 1.5),
        ("tight", 0.15, 0.05),
    ],
)
def test_tanh_zono_fixed_cases(label, c, r):
    z = Zono(np.array([c]), np.array([[r]]), (0,))
    assert _audit_unary(tanh_zono, np.tanh, z, 200, seed=42) == 0


def test_tanh_zono_2d_shared_preds():
    z = Zono(
        np.array([0.5, -0.3]),
        np.array([[0.4, 0.2, 0.1], [0.1, 0.3, 0.2]]),
        (0, 1, 2),
    )
    assert _audit_unary(tanh_zono, np.tanh, z, 200, seed=7) == 0


def test_tanh_zono_random_fuzz():
    rng = np.random.default_rng(2026)
    for j in range(50):
        c = 2 * rng.standard_normal(3)
        n_g = int(rng.integers(2, 6))
        V = 0.5 * rng.standard_normal((3, n_g))
        z = Zono(c, V, tuple(range(n_g)))
        assert _audit_unary(tanh_zono, np.tanh, z, 50, seed=j) == 0


# ---------- sigmoid ----------

@pytest.mark.parametrize(
    "label,c,r",
    [
        ("pos", 1.0, 0.5),
        ("pos_wide", 2.5, 1.5),
        ("neg", -1.0, 0.5),
        ("mixed", 0.0, 1.0),
        ("wide", 0.0, 3.0),
        ("saturated", 4.5, 1.5),
        ("tight", 0.15, 0.05),
    ],
)
def test_sigmoid_zono_fixed_cases(label, c, r):
    z = Zono(np.array([c]), np.array([[r]]), (0,))
    assert _audit_unary(sigmoid_zono, _sigmoid, z, 200, seed=42) == 0


def test_sigmoid_zono_random_fuzz():
    rng = np.random.default_rng(2027)
    for j in range(50):
        c = 2 * rng.standard_normal(3)
        n_g = int(rng.integers(2, 6))
        V = 0.5 * rng.standard_normal((3, n_g))
        z = Zono(c, V, tuple(range(n_g)))
        assert _audit_unary(sigmoid_zono, _sigmoid, z, 50, seed=j) == 0


# ---------- bilinear sigmoid * identity (x * sigma(y)) ----------

def _shared_pair(c_x, r_x, c_y, r_y, pred_ids=(0,)):
    z_x = Zono(np.array([c_x]), np.array([[r_x]]), pred_ids)
    z_y = Zono(np.array([c_y]), np.array([[r_y]]), pred_ids)
    return z_x, z_y


@pytest.mark.parametrize(
    "label,cx,rx,cy,ry",
    [
        ("C1", 1.0, 1.0, 0.0, 1.0),
        ("C2", -1.0, 1.0, 0.0, 1.0),
        ("C3_sym", 0.0, 1.0, 0.0, 1.0),
        ("C3_asym", -0.5, 1.5, 0.0, 1.0),
    ],
)
def test_sigid_fixed_cases(label, cx, rx, cy, ry):
    z_x, z_y = _shared_pair(cx, rx, cy, ry)
    fn = lambda x, y: x * _sigmoid(y)
    assert _audit_binary(bilinear_sigmoid_identity, fn, z_x, z_y, 200, seed=300) == 0


def test_sigid_random_fuzz():
    rng = np.random.default_rng(2028)
    fn = lambda x, y: x * _sigmoid(y)
    for j in range(30):
        K = int(rng.integers(1, 5))
        n_g = int(rng.integers(2, 5))
        c_x = rng.standard_normal(K)
        c_y = rng.standard_normal(K)
        V_x = 0.4 * rng.standard_normal((K, n_g))
        V_y = 0.4 * rng.standard_normal((K, n_g))
        ids = tuple(range(n_g))
        z_x = Zono(c_x, V_x, ids)
        z_y = Zono(c_y, V_y, ids)
        assert _audit_binary(bilinear_sigmoid_identity, fn, z_x, z_y, 50, seed=j) == 0


# ---------- bilinear sigmoid * tanh ----------

@pytest.mark.parametrize(
    "label,cx,rx,cy,ry",
    [
        ("pos", 1.0, 1.0, 1.0, 1.0),
        ("neg", -1.0, 1.0, -1.0, 1.0),
        ("mixed", 0.0, 1.0, 0.0, 1.0),
        ("wide", 0.0, 3.0, 0.0, 3.0),
        ("asym", -0.5, 1.5, 0.5, 1.5),
    ],
)
def test_sigtanh_fixed_cases(label, cx, rx, cy, ry):
    z_x, z_y = _shared_pair(cx, rx, cy, ry)
    fn = lambda x, y: _sigmoid(x) * np.tanh(y)
    assert _audit_binary(bilinear_sigmoid_tanh, fn, z_x, z_y, 200, seed=400) == 0


def test_sigtanh_random_fuzz():
    rng = np.random.default_rng(2029)
    fn = lambda x, y: _sigmoid(x) * np.tanh(y)
    for j in range(30):
        K = int(rng.integers(1, 5))
        n_g = int(rng.integers(2, 5))
        c_x = rng.standard_normal(K)
        c_y = rng.standard_normal(K)
        V_x = 0.4 * rng.standard_normal((K, n_g))
        V_y = 0.4 * rng.standard_normal((K, n_g))
        ids = tuple(range(n_g))
        z_x = Zono(c_x, V_x, ids)
        z_y = Zono(c_y, V_y, ids)
        assert _audit_binary(bilinear_sigmoid_tanh, fn, z_x, z_y, 50, seed=j) == 0


# ---------- bilinears with disjoint pred_ids (Minkowski) ----------

def test_sigtanh_disjoint_preds_is_sound():
    """z_x and z_y with disjoint pred_ids -> independent (x, y) at runtime;
    the bilinear must still bound the product correctly."""
    z_x = Zono(np.array([0.5]), np.array([[1.0]]), (0,))
    z_y = Zono(np.array([-0.5]), np.array([[1.0]]), (1,))
    z_out = bilinear_sigmoid_tanh(z_x, z_y)
    rng = np.random.default_rng(99)
    n_viol = 0
    for _ in range(300):
        ax = 2 * rng.random() - 1
        ay = 2 * rng.random() - 1
        x = 0.5 + ax
        y = -0.5 + ay
        y_true = _sigmoid(x) * np.tanh(y)
        if not lp_feasible(z_out, np.array([y_true])):
            n_viol += 1
    assert n_viol == 0


# ---------- regression: hadamard affine-only baseline is UNSOUND ----------

def test_hadamard_affine_only_is_unsound():
    """The naive Hadamard baseline (drops the bilinear cross term)
    under-bounds (1+alpha)^2 -- mirrors NNV's Star.HadamardProduct bug.
    This test documents why the Cert-RNN bilinears are required."""
    z = Zono(np.array([1.0]), np.array([[1.0]]), (0,))  # 1 + alpha
    z_out = hadamard_affine_only(z, z)
    lb, ub = z_out.get_ranges()
    # True (1+alpha)^2 ranges over [0, 4]; baseline misses 4.
    assert ub < 4 - 1e-6, (
        f"baseline ub={ub} unexpectedly >= 4; the bug it documents is gone?"
    )
