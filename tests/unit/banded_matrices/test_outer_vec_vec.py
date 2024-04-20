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

from banded_matrices.banded import outer_mat_mat, outer_vec_vec, square_mat
from tests.utils.banded_matrices_utils import (
    compute_gradient_error,
    constant_op,
    extract_band_from_matrix,
    extract_construct_banded_matrix,
)


@pytest.mark.parametrize("n ,l_out", [(5, 2), (10, 9), (6, 0)])
def test_outer_vec_vec(n, l_out):
    """
    Test forward, with same term on left and right arguments.
    """
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        np.random.seed(10)
        v = np.random.rand(n, 1)

        # compute with our implementation
        S = outer_vec_vec(v, v, l_out)
        result = sess.run(S)

        # compute with np
        S_np = extract_band_from_matrix(l_out, 0, v @ v.T)

        # compare
        np.testing.assert_almost_equal(result, S_np)


@pytest.mark.parametrize("num_vectors", [1, 3, 5])
@pytest.mark.parametrize("n, l_out", [(5, 2), (10, 9), (6, 0)])
def test_outer_mat_mat(n, l_out, num_vectors):
    """
    Test forward, with same term on left and right arguments.
    """
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        np.random.seed(10)
        v = np.random.rand(n, num_vectors)

        # compute with our implementation
        S = outer_mat_mat(v, v, l_out)
        result = sess.run(S)

        # compute with np
        S_np = extract_band_from_matrix(l_out, 0, v @ v.T)

        # compare
        np.testing.assert_almost_equal(result, S_np)


@pytest.mark.parametrize("num_vectors", [1, 3, 5])
@pytest.mark.parametrize("n, l_out", [(5, 2), (10, 9), (6, 0)])
def test_square_mat(n, l_out, num_vectors):
    """
    Test the square_mat shortcut.
    """
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        np.random.seed(10)
        v = np.random.rand(n, num_vectors)

        # compute with our implementation
        S = square_mat(v, l_out)
        result = sess.run(S)

        # compute with np
        S_np = extract_band_from_matrix(l_out, 0, v @ v.T)

        # compare
        np.testing.assert_almost_equal(result, S_np)


@pytest.mark.parametrize("n ,l_out, r_out", [(5, 2, 1), (10, 7, 1), (6, 2, 2)])
def test_outer_vec_vec_general(n, l_out, r_out):
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        np.random.seed(10435)
        l = np.random.rand(n, 1)
        r = np.random.rand(n, 1)

        # compute with our implementation
        S = outer_vec_vec(l, r, l_out, r_out)
        result = sess.run(S)

        S_np = extract_band_from_matrix(l_out, r_out, l.reshape((n, 1)) @ r.reshape((1, n)))

        # compare
        np.testing.assert_almost_equal(result, S_np)


@pytest.mark.parametrize("n ,l_out", [(5, 2), (10, 9), (6, 0), (11, 4)])
def test_gradient_outer(n, l_out):
    """
    Finite-difference checks, with same term on left and right arguments.
    """
    np.random.seed(1234567)
    with tf.compat.v1.Session(graph=tf.Graph()):
        banded1 = np.random.rand(n, 1)

        cst_op1 = constant_op(banded1)
        result = outer_vec_vec(cst_op1, cst_op1, l_out)

        # Error for dy/dx1
        grad_err_1 = compute_gradient_error(cst_op1, result)

        print("gradient errors: ", grad_err_1)
        assert grad_err_1 < 1e-8


@pytest.mark.parametrize("n, l_out, u_out", [(5, 2, 1), (10, 9, 0), (6, 0, 0), (11, 4, 1)])
def test_gradient_outer_vec_vec_general(n, l_out, u_out):
    np.random.seed(1234567)
    with tf.compat.v1.Session(graph=tf.Graph()):
        banded1 = np.random.rand(n, 1)
        banded2 = np.random.rand(n, 1)

        cst_op1 = constant_op(banded1)
        cst_op2 = constant_op(banded2)
        result = outer_vec_vec(cst_op1, cst_op2, l_out, u_out)

        grad_err_1 = compute_gradient_error(cst_op1, result)
        grad_err_2 = compute_gradient_error(cst_op2, result)

        print("gradient errors: ", grad_err_1, grad_err_2)
        assert grad_err_1 < 1e-8
        assert grad_err_2 < 1e-8


@pytest.mark.parametrize(
    "n, count_vectors, l_out, u_out",
    [(5, 3, 2, 1), (10, 2, 9, 0), (6, 2, 0, 0), (11, 2, 4, 1)],
)
def test_gradient_outer_mat_mat_general(n, count_vectors, l_out, u_out):
    np.random.seed(1234567)
    with tf.compat.v1.Session(graph=tf.Graph()):
        banded1 = np.random.rand(n, count_vectors)
        banded2 = np.random.rand(n, count_vectors)

        cst_op1 = constant_op(banded1)
        cst_op2 = constant_op(banded2)
        result = outer_mat_mat(cst_op1, cst_op2, l_out, u_out)

        grad_err_1 = compute_gradient_error(cst_op1, result)
        grad_err_2 = compute_gradient_error(cst_op2, result)

        print("gradient errors: ", grad_err_1, grad_err_2)
        assert grad_err_1 < 1e-8
        assert grad_err_2 < 1e-8


@pytest.mark.parametrize("n, count_vectors, l_out", [(5, 1, 2), (10, 2, 9), (6, 2, 0)])
def test_gradient_square_mat_against_tf(n, count_vectors, l_out):
    np.random.seed(1234567)
    with tf.compat.v1.Session(graph=tf.Graph()) as session:
        banded = np.random.rand(n, count_vectors)

        cst_op = constant_op(banded)
        result = square_mat(cst_op, l_out)

        square_tf_op = tf.matmul(cst_op, cst_op, transpose_b=True)

        grad_err = compute_gradient_error(cst_op, result)
        print("FD gradient error", grad_err)

        # gradients ops
        # This should be done consistently with square_band:
        bar_square_dense = extract_construct_banded_matrix(l_out, l_out, np.ones((n, n)))
        bar_square_band = extract_band_from_matrix(l_out, 0, bar_square_dense)
        bar_square_band[1:, :] *= 2.0  # double the non diag entries

        grad_square_op = tf.gradients(ys=result, xs=cst_op, grad_ys=bar_square_band)[0]

        grad_square_tf_op = tf.gradients(ys=square_tf_op, xs=cst_op, grad_ys=bar_square_dense)[
            0
        ]

        grad_square = session.run(grad_square_op)
        grad_square_tf = session.run(grad_square_tf_op)

        np.testing.assert_almost_equal(actual=grad_square, desired=grad_square_tf, decimal=10)
