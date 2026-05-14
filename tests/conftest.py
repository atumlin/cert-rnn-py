"""Pytest fixtures shared across the soundness suite.

The Zono pred-id allocator is module-level. Tests that hand-assign
small integer pred_ids would clash with fresh allocations if the
allocator were also at 0. Reset to a high start before every test so
the (test-picked, allocator-issued) namespaces stay disjoint.
"""

import pytest

from cert_rnn import reset_pred_allocator


@pytest.fixture(autouse=True)
def isolated_pred_allocator():
    reset_pred_allocator(start=10_000)
    yield
