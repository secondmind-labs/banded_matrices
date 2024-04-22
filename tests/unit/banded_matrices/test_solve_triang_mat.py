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

from banded_matrices.banded import solve_triang_mat
from tests.utils.banded_matrices_utils import (
    compute_gradient_error,
    constant_op,
    extract_construct_banded_matrix,
    generate_band_mat,
    to_dense,
)


def make_argument(dense_matrix: np.ndarray, transpose: bool):
    if transpose:
        return dense_matrix.transpose()
    else:
        return dense_matrix


@pytest.mark.parametrize("n", [12, 15, 20])
@pytest.mark.parametrize("vector_count", [1, 3, 17])
@pytest.mark.parametrize("left_bandwidth", [0, 1, 5])
@pytest.mark.parametrize("transpose_left", [False, True])
def test_forward_solve_triang_mat_against_numpy_solve(
    n, left_bandwidth, transpose_left, vector_count
):
    """
    Test of the forward evaluation of the ``solve_triang_mat``.
    TODO in the future: multivec solve
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

        # banded solve op
        solve_op = solve_triang_mat(cst_banded_lower, cst_vector, transpose_left)
        solve = solve_op.eval()

        # tensorflow solve op
        solve_tf_op = tf.linalg.triangular_solve(
            matrix=cst_dense_lower, rhs=cst_vector, adjoint=transpose_left
        )
        solve_tf = solve_tf_op.eval()

        # compare
        error = np.fabs(solve - solve_tf).max()
        print(error)
        # 10 to 14 decimals is typical, but 8 is occasionally needed:
        np.testing.assert_almost_equal(actual=solve, desired=solve_tf, decimal=8)


@pytest.mark.parametrize("n", [10, 15, 20])
@pytest.mark.parametrize("vector_count", [1, 3, 17])
@pytest.mark.parametrize("left_bandwidth", [0, 3, 5])
@pytest.mark.parametrize("transpose_left", [False, True])
def test_solve_triang_mat_rev_mode_gradient_against_tf_triangular_solve(
    n, left_bandwidth, transpose_left, vector_count
):
    """
    Test of the ``solve_triang_mat`` gradients against those of
    tf.linalg.triangular_solve.
    """
    np.random.seed(4123469)
    with tf.compat.v1.Session(graph=tf.Graph()) as session:
        banded_lower = generate_band_mat(n, left_bandwidth, 0)
        dense_lower = to_dense(banded_lower, left_bandwidth, 0)
        vector = np.random.rand(n, vector_count)

        cst_banded_lower = constant_op(banded_lower)
        cst_vector = constant_op(vector)
        cst_dense_lower = constant_op(dense_lower)

        # Solve operator as we calculate it and in dense form:
        solve_op = solve_triang_mat(
            cst_banded_lower, cst_vector, transpose_left=transpose_left
        )
        solve_tf_op = tf.linalg.triangular_solve(
            matrix=cst_dense_lower, rhs=cst_vector, adjoint=transpose_left
        )

        # Gradients:
        [grad_band_op, grad_vector_op] = tf.gradients(
            ys=solve_op, xs=[cst_banded_lower, cst_vector], grad_ys=np.ones((n, vector_count))
        )
        [grad_dense_tf_op, grad_vector_tf_op] = tf.gradients(
            ys=solve_tf_op,
            xs=[cst_dense_lower, cst_vector],
            grad_ys=np.ones((n, vector_count)),
        )

        # Errors, when converted to dense with 0s out of band:
        grad_left = to_dense(session.run(grad_band_op), left_bandwidth, 0)
        grad_right = session.run(grad_vector_op)
        grad_left_tf = extract_construct_banded_matrix(
            left_bandwidth, 0, session.run(grad_dense_tf_op)
        )
        grad_right_tf = session.run(grad_vector_tf_op)

        grad_err_left = np.fabs(grad_left - grad_left_tf).max()
        grad_err_right = np.fabs(grad_right - grad_right_tf).max()

        print("Solve gradient errors w.r.t. TF dense: ", grad_err_left, grad_err_right)
        assert grad_err_left < 1e-10
        assert grad_err_right < 1e-10


@pytest.mark.parametrize("n", [12, 15])
@pytest.mark.parametrize("vector_count", [1, 3, 17])
@pytest.mark.parametrize("left_bandwidth", [0, 1, 3, 5])
def test_solve_triang_mat_jacobians_using_finite_differencing(n, left_bandwidth, vector_count):
    """
    Finite difference testing for ``solve_triang_mat``.
    The tolerance is unfortunately high on these tests.
    """
    np.random.seed(41234679)
    with tf.compat.v1.Session(graph=tf.Graph()):
        banded_lower = generate_band_mat(n, left_bandwidth, 0)
        vector = np.random.rand(n, vector_count)

        cst_banded_lower = constant_op(banded_lower)
        cst_vector = constant_op(vector)

        result_op = solve_triang_mat(cst_banded_lower, cst_vector)

        # Error for dy/dx1
        grad_err_1 = compute_gradient_error(cst_banded_lower, result_op, delta=1e-7)

        # Error for dy/dx2
        grad_err_2 = compute_gradient_error(cst_vector, result_op, delta=1e-6)

        print("Gradients finite diff errors", grad_err_1, grad_err_2)
        assert grad_err_1 < 3e-3
        assert grad_err_2 < 1e-5
