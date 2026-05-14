"""LP-feasibility membership test for zonotopes.

Replaces NNV's lpsolver with scipy.optimize.linprog (HiGHS). Used by
the soundness test suite to certify whether sampled concrete points
lie inside an abstract transformer's output zonotope.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from cert_rnn.zono import Zono


def lp_feasible(z: Zono, y_target, tol: float = 1e-7) -> bool:
    """Is y_target in z?

    Returns True iff there exists alpha in [-1, 1]^p with
        z.V @ alpha = y_target - z.c.
    Solved via scipy.optimize.linprog (method='highs'). This is the
    rigorous (sufficient + necessary) containment test; bounding-box
    checks are necessary-only.
    """
    p = z.n_pred
    y = np.asarray(y_target, dtype=np.float64).reshape(-1)
    if y.shape[0] != z.dim:
        raise ValueError(f"lp_feasible: y_target len {y.shape[0]} != z.dim {z.dim}")
    rhs = y - z.c
    if p == 0:
        return float(np.max(np.abs(rhs))) <= tol
    res = linprog(
        c=np.zeros(p),
        A_eq=z.V,
        b_eq=rhs,
        bounds=[(-1.0, 1.0)] * p,
        method="highs",
    )
    return bool(res.success)
