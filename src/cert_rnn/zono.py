"""Zonotope datatype with Minkowski-correct predicate tracking.

A Zono represents the set
    Z = { c + V @ alpha : alpha in [-1, 1]^p, indexed by pred_ids }
where c is a (K,) center, V is a (K, p) generator matrix, and pred_ids
is a tuple of length p naming each generator's alpha-variable.

Two zonotopes share a predicate iff they reference the same pred_id.
When combining (zono_add, bilinear transformers, etc.), shared preds
collapse to one column and unshared preds get disjoint columns. This
Minkowski-style embedding is what makes the LSTM step sound for
fresh-per-timestep perturbation; the MATLAB reference's positional
zero-padding aliased fresh predicates with state predicates and was
unsound under that threat model.

Fresh predicates from abstract transformers are allocated via a
module-level PredAllocator monotonic counter; reset via
reset_pred_allocator() for test isolation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


class PredAllocator:
    """Monotonic integer allocator for fresh predicate ids."""

    def __init__(self, start: int = 0):
        self._it = itertools.count(start)

    def next_one(self) -> int:
        return next(self._it)

    def next_n(self, n: int) -> tuple:
        return tuple(next(self._it) for _ in range(n))


_DEFAULT_ALLOC = PredAllocator(start=0)


def get_default_allocator() -> PredAllocator:
    return _DEFAULT_ALLOC


def reset_pred_allocator(start: int = 0) -> None:
    global _DEFAULT_ALLOC
    _DEFAULT_ALLOC = PredAllocator(start)


@dataclass(frozen=True)
class Zono:
    """Zonotope { c + V @ alpha : alpha in [-1, 1]^p }, preds named by pred_ids."""

    c: np.ndarray   # (K,) float64
    V: np.ndarray   # (K, p) float64; p may be 0
    pred_ids: tuple  # length p; each element a unique hashable id (typ. int)

    def __post_init__(self) -> None:
        c = np.ascontiguousarray(self.c, dtype=np.float64).reshape(-1)
        K = c.shape[0]
        V = np.ascontiguousarray(self.V, dtype=np.float64)
        if V.ndim == 1:
            V = V.reshape(K, -1)
        if V.shape[0] != K:
            raise ValueError(f"Zono: V row count {V.shape[0]} != len(c) {K}")
        pred_ids = tuple(self.pred_ids)
        if len(pred_ids) != V.shape[1]:
            raise ValueError(
                f"Zono: len(pred_ids) {len(pred_ids)} != V cols {V.shape[1]}"
            )
        if len(set(pred_ids)) != len(pred_ids):
            raise ValueError("Zono: pred_ids must be unique within a single zono")
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "V", V)
        object.__setattr__(self, "pred_ids", pred_ids)

    # --- factories ---

    @classmethod
    def point(cls, c) -> "Zono":
        c = np.asarray(c, dtype=np.float64).reshape(-1)
        return cls(c, np.zeros((c.shape[0], 0), dtype=np.float64), ())

    @classmethod
    def from_box(cls, c, radius, allocator: PredAllocator | None = None) -> "Zono":
        """L_inf ball: c + diag(radius) @ alpha, fresh alpha per element."""
        c = np.asarray(c, dtype=np.float64).reshape(-1)
        K = c.shape[0]
        if np.isscalar(radius):
            r = np.full(K, float(radius))
        else:
            r = np.asarray(radius, dtype=np.float64).reshape(-1)
            if r.shape[0] != K:
                raise ValueError("from_box: radius length must equal len(c)")
        V = np.diag(r)
        alloc = allocator if allocator is not None else get_default_allocator()
        ids = alloc.next_n(K)
        return cls(c, V, ids)

    # --- properties ---

    @property
    def dim(self) -> int:
        return self.c.shape[0]

    @property
    def n_pred(self) -> int:
        return self.V.shape[1]

    # --- queries ---

    def get_ranges(self) -> tuple[np.ndarray, np.ndarray]:
        radius = np.sum(np.abs(self.V), axis=1)
        return self.c - radius, self.c + radius

    # --- maps ---

    def affine_map(self, W, b=None) -> "Zono":
        """Return W @ self + b. Preserves pred_ids."""
        W = np.asarray(W, dtype=np.float64)
        new_c = W @ self.c
        if b is not None:
            new_c = new_c + np.asarray(b, dtype=np.float64).reshape(-1)
        new_V = W @ self.V
        return Zono(new_c, new_V, self.pred_ids)

    def slice_rows(self, start: int, end: int) -> "Zono":
        """Half-open row slice [start:end]. Preserves pred_ids."""
        return Zono(self.c[start:end], self.V[start:end, :], self.pred_ids)


def align_pred_space(*zonos: Zono) -> tuple[tuple, list[np.ndarray]]:
    """Embed each zono's V into a shared predicate space.

    Shared pred_ids (appearing in multiple inputs) collapse to one column;
    unshared pred_ids get disjoint columns. Order: pred_ids from zonos[0]
    first in their original order, then any new ones from zonos[1], etc.

    Returns:
        (shared_ids, [V_emb_0, V_emb_1, ...])
        shared_ids: tuple of pred ids, length P = total unique pred count.
        V_emb_i: ndarray (zonos[i].dim, P) with z_i.V columns placed in the
            slots assigned to z_i's pred_ids; zero elsewhere.
    """
    seen: dict = {}
    order: list = []
    for z in zonos:
        for pid in z.pred_ids:
            if pid not in seen:
                seen[pid] = len(order)
                order.append(pid)
    P = len(order)
    V_list = []
    for z in zonos:
        V_emb = np.zeros((z.dim, P), dtype=np.float64)
        if z.n_pred > 0:
            col_map = np.array([seen[pid] for pid in z.pred_ids], dtype=int)
            V_emb[:, col_map] = z.V
        V_list.append(V_emb)
    return tuple(order), V_list


def zono_add(z1: Zono, z2: Zono) -> Zono:
    """Sum z1 + z2. Shared pred_ids share an alpha (Minkowski-correct)."""
    if z1.dim != z2.dim:
        raise ValueError(f"zono_add: dim mismatch {z1.dim} vs {z2.dim}")
    ids, (V1, V2) = align_pred_space(z1, z2)
    return Zono(z1.c + z2.c, V1 + V2, ids)


def zono_sub(z1: Zono, z2: Zono) -> Zono:
    """Difference z1 - z2 (Minkowski). Shared pred_ids share an alpha."""
    if z1.dim != z2.dim:
        raise ValueError(f"zono_sub: dim mismatch {z1.dim} vs {z2.dim}")
    ids, (V1, V2) = align_pred_space(z1, z2)
    return Zono(z1.c - z2.c, V1 - V2, ids)
