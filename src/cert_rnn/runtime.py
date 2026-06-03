"""Runtime helpers: BLAS thread pinning.

cert_rnn does many small numpy ops; default multi-threaded BLAS adds
contention that hurts more than it helps and burns CPU on busy machines.
Pinning to a single thread is the right default for verification runs.

Env vars (OMP/MKL/OPENBLAS/NUMEXPR) only take effect when read at library
import time, so for a script the most reliable pattern is still to set
them before importing numpy. `limit_blas_threads()` additionally clamps
already-imported BLAS pools live via threadpoolctl when it is installed.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

_BLAS_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def pin_blas_threads(n: int = 1) -> None:
    """Set the BLAS thread-count env vars to `n`. Full effect requires
    calling this before numpy is imported."""
    for var in _BLAS_ENV_VARS:
        os.environ[var] = str(n)


@contextmanager
def limit_blas_threads(n: int = 1):
    """Context manager that limits BLAS threads to `n` for the duration.

    Sets the env vars and, if threadpoolctl is installed, clamps live
    thread pools (works even after numpy import). Without threadpoolctl it
    degrades to setting the env vars only.
    """
    pin_blas_threads(n)
    try:
        import threadpoolctl
    except ImportError:
        yield None
        return
    with threadpoolctl.threadpool_limits(limits=n):
        yield n
