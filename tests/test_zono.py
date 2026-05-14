"""Phase-1 tests for the Zono datatype and Minkowski alignment."""

import numpy as np
import pytest

from cert_rnn import Zono, align_pred_space, zono_add, zono_sub


def test_zono_construct_and_range():
    z = Zono(np.array([0.5, -0.3]), np.array([[0.4, 0.2], [0.1, 0.3]]), (0, 1))
    lb, ub = z.get_ranges()
    np.testing.assert_allclose(lb, [-0.1, -0.7])
    np.testing.assert_allclose(ub, [1.1, 0.1])
    assert z.dim == 2
    assert z.n_pred == 2


def test_zono_point_has_zero_radius():
    z = Zono.point(np.array([1.0, 2.0]))
    lb, ub = z.get_ranges()
    np.testing.assert_allclose(lb, [1.0, 2.0])
    np.testing.assert_allclose(ub, [1.0, 2.0])
    assert z.n_pred == 0


def test_zono_pred_ids_must_be_unique():
    with pytest.raises(ValueError):
        Zono(np.zeros(2), np.eye(2), (5, 5))


def test_zono_pred_ids_length_must_match_V_cols():
    with pytest.raises(ValueError):
        Zono(np.zeros(2), np.eye(2), (5,))


def test_zono_affine_map_preserves_pred_ids():
    z = Zono(np.array([1.0, 2.0]), np.array([[1.0], [0.5]]), (7,))
    z2 = z.affine_map(np.array([[2.0, 0.0], [0.0, 3.0]]), np.array([1.0, 1.0]))
    np.testing.assert_allclose(z2.c, [3.0, 7.0])
    np.testing.assert_allclose(z2.V, [[2.0], [1.5]])
    assert z2.pred_ids == (7,)


def test_align_pred_space_disjoint():
    z1 = Zono(np.array([0.0]), np.array([[1.0]]), (10,))
    z2 = Zono(np.array([0.0]), np.array([[1.0]]), (20,))
    ids, (V1, V2) = align_pred_space(z1, z2)
    assert ids == (10, 20)
    np.testing.assert_allclose(V1, [[1.0, 0.0]])
    np.testing.assert_allclose(V2, [[0.0, 1.0]])


def test_align_pred_space_shared():
    z1 = Zono(np.array([0.0]), np.array([[1.0, 2.0]]), (10, 20))
    z2 = Zono(np.array([0.0]), np.array([[3.0, 4.0]]), (20, 30))
    ids, (V1, V2) = align_pred_space(z1, z2)
    assert ids == (10, 20, 30)
    np.testing.assert_allclose(V1, [[1.0, 2.0, 0.0]])
    np.testing.assert_allclose(V2, [[0.0, 3.0, 4.0]])


def test_zono_add_shared_pred_collapses():
    # (1 + alpha_0) + (1 + alpha_0) = 2 + 2 alpha_0, range [0, 4]
    z1 = Zono(np.array([1.0]), np.array([[1.0]]), (0,))
    z2 = Zono(np.array([1.0]), np.array([[1.0]]), (0,))
    z = zono_add(z1, z2)
    lb, ub = z.get_ranges()
    np.testing.assert_allclose([lb[0], ub[0]], [0.0, 4.0])
    assert z.n_pred == 1


def test_zono_add_disjoint_pred_concats():
    # (1 + alpha_0) + (1 + alpha_1) over independent alphas; range [0, 4]
    z1 = Zono(np.array([1.0]), np.array([[1.0]]), (0,))
    z2 = Zono(np.array([1.0]), np.array([[1.0]]), (1,))
    z = zono_add(z1, z2)
    lb, ub = z.get_ranges()
    np.testing.assert_allclose([lb[0], ub[0]], [0.0, 4.0])
    assert z.n_pred == 2


def test_zono_sub_shared_pred_cancels():
    # (1 + alpha) - (1 + alpha) = 0 exactly
    z1 = Zono(np.array([1.0]), np.array([[1.0]]), (0,))
    z2 = Zono(np.array([1.0]), np.array([[1.0]]), (0,))
    z = zono_sub(z1, z2)
    lb, ub = z.get_ranges()
    np.testing.assert_allclose([lb[0], ub[0]], [0.0, 0.0])


def test_zono_sub_disjoint_pred_does_not_cancel():
    # (1 + alpha) - (1 + beta), independent alphas; range [-2, 2]
    z1 = Zono(np.array([1.0]), np.array([[1.0]]), (0,))
    z2 = Zono(np.array([1.0]), np.array([[1.0]]), (1,))
    z = zono_sub(z1, z2)
    lb, ub = z.get_ranges()
    np.testing.assert_allclose([lb[0], ub[0]], [-2.0, 2.0])


def test_from_box_allocates_fresh_preds():
    z = Zono.from_box(np.array([0.0, 0.0]), 0.5)
    assert z.dim == 2
    assert z.n_pred == 2
    lb, ub = z.get_ranges()
    np.testing.assert_allclose(lb, [-0.5, -0.5])
    np.testing.assert_allclose(ub, [0.5, 0.5])
