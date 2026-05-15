"""Pytest fixtures shared across the soundness suite.

The Zono pred-id allocator is module-level. Tests that hand-assign
small integer pred_ids would clash with fresh allocations if the
allocator were also at 0. Reset to a high start before every test so
the (test-picked, allocator-issued) namespaces stay disjoint.

Thread limits: cert_rnn does many small numpy ops in per-element loops
(plane computations include np.roots, np.tanh, np.log on scalars).
With default BLAS threading these spawn many threads per call; on a
busy machine the resulting oversubscription dominated runtime
(~28x CPU/wall seen on the MNIST test). Pin BLAS to 1 thread for
consistent timing. Env vars are setdefault so user-provided overrides
still win.
"""

import os

# MUST be set before numpy/scipy/torch import. Pytest loads conftest.py
# before any test module, so this runs first.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Belt-and-suspenders: if a pytest plugin happened to import numpy/scipy
# before this file ran, the env vars above are too late for them.
# threadpool_limits applies to already-loaded BLAS pools at runtime.
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
except Exception:  # threadpoolctl optional, missing or detection bug
    pass

import pytest

from cert_rnn import reset_pred_allocator


@pytest.fixture(autouse=True)
def isolated_pred_allocator():
    reset_pred_allocator(start=10_000)
    yield
