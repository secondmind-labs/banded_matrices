#
# Copyright (c) 2021 The banded_matrices Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import numpy as np
import pytest
import tensorflow as tf

from banded_matrices.banded import chol_solve_band_mat
from tests.utils.banded_matrices_utils import (
    constant_op,
    extract_band_from_matrix,
    generate_band_mat,
    to_dense,
)


@pytest.mark.parametrize("n", [12, 15, 20])
@pytest.mark.parametrize("vector_count", [1, 3, 10])
@pytest.mark.parametrize("left_bandwidth", [0, 1, 5])
def test_forward_chol_solve_band_mat(n, left_bandwidth, vector_count):
    """
    Test of the forward evaluation of the ``chol_solve_band_mat``.
    """
    np.random.seed(41234679)

    with tf.compat.v1.Session(graph=tf.Graph()):
        # construct lower banded matrix and vector
        banded_lower = generate_band_mat(n, left_bandwidth, 0)
        vector = np.random.rand(n, vector_count)
        dense_lower = to_dense(banded_lower, left_bandwidth, 0)

        cst_banded_lower = constant_op(banded_lower)
        cst_dense_lower = constant_op(dense_lower)
        cst_vector = constant_op(vector)

        # banded chol solve op
        chol_solve_op = chol_solve_band_mat(cst_banded_lower, cst_vector)
        chol_solve = chol_solve_op.eval()

        # tensorflow solve op
        chol_solve_tf_op = tf.linalg.cholesky_solve(cst_dense_lower, cst_vector)
        chol_solve_tf = chol_solve_tf_op.eval()

        # compare
        norm = np.sqrt(np.sum(chol_solve**2))
        np.testing.assert_almost_equal(
            actual=chol_solve / norm, desired=chol_solve_tf / norm, decimal=12
        )


@pytest.mark.parametrize("n", [12, 15, 20])
@pytest.mark.parametrize("vector_count", [1, 3, 10])
@pytest.mark.parametrize("left_bandwidth", [0, 1, 5])
def test_chol_solve_mat_rev_mode_gradient_against_tf_chol_solve(
    n, left_bandwidth, vector_count
):
    """
    Test of the ``chol_solve_mat`` gradients against those of
    tf.linalg.cholesky_solve.
    """
    np.random.seed(4123469)

    with tf.compat.v1.Session(graph=tf.Graph()):
        # construct lower banded matrix and vector
        banded_lower = generate_band_mat(n, left_bandwidth, 0)
        vector = np.random.rand(n, vector_count)
        dense_lower = to_dense(banded_lower, left_bandwidth, 0)

        cst_banded_lower = constant_op(banded_lower)
        cst_dense_lower = constant_op(dense_lower)
        cst_vector = constant_op(vector)

        # banded chol solve op
        chol_solve_op = chol_solve_band_mat(cst_banded_lower, cst_vector)
        grad_chol_solve_op = tf.gradients(ys=chol_solve_op, xs=[cst_banded_lower, cst_vector])
        grad_chol_solve_left = grad_chol_solve_op[0].eval()
        grad_chol_solve_right = grad_chol_solve_op[1].eval()

        # tensorflow solve op
        chol_solve_tf_op = tf.linalg.cholesky_solve(cst_dense_lower, cst_vector)
        grad_chol_solve_tf_op = tf.gradients(
            ys=chol_solve_tf_op, xs=[cst_dense_lower, cst_vector]
        )

        # evaluate gradients
        grad_chol_solve_tf_left = extract_band_from_matrix(
            left_bandwidth, 0, grad_chol_solve_tf_op[0].eval()
        )
        grad_chol_solve_tf_right = grad_chol_solve_tf_op[1].eval()

        # compare
        norm = np.sqrt(np.sum(grad_chol_solve_left**2))
        np.testing.assert_almost_equal(
            actual=grad_chol_solve_left / norm,
            desired=grad_chol_solve_tf_left / norm,
            decimal=12,
        )
        norm = np.sqrt(np.sum(grad_chol_solve_right**2))
        np.testing.assert_almost_equal(
            actual=grad_chol_solve_right / norm,
            desired=grad_chol_solve_tf_right / norm,
            decimal=12,
        )
