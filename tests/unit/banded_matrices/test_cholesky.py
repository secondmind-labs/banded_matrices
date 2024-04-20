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
import sys

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.errors import InternalError

from banded_matrices.banded import cholesky_band
from tests.utils.banded_matrices_utils import (
    constant_op,
    construct_banded_matrix_from_band,
    extract_band_from_matrix,
    extract_construct_banded_matrix,
    generate_banded_positive_definite_matrix,
    to_dense,
)


def banded_matrix(m, k):
    n = m.shape[0]
    assert n == m.shape[1]
    a = np.zeros((k, n))
    for i in range(k):
        a[i, : n - i] = np.diagonal(m, offset=-i)
    return a


# Implementation cholesky gradients for dense
# symmetric matrices.
def ref_grad_cholesky_full(L_dense, barL_dense, k):
    n = L_dense.shape[0]
    Abb = np.zeros((n, n))
    L = L_dense.copy()
    Lb = barL_dense.copy()
    for i in range(n - 1, -1, -1):
        j_stop = max(i - k, -1)
        for j in range(i, j_stop, -1):
            if j == i:
                Abb[i, i] = 1 / 2 * Lb[i, i] / L[i, i]
            else:
                Abb[i, j] = Lb[i, j] / L[j, j]
                Lb[j, j] -= Lb[i, j] * L[i, j] / L[j, j]
            a_ij = Abb[i, j]
            for l in range(j - 1, j_stop, -1):
                Lb[i, l] -= a_ij * L[j, l]
                Lb[j, l] -= a_ij * L[i, l]
    return Abb


# Implementation cholesky gradients for banded lower
# triangular symmetric matrices.
def ref_grad_cholesky_band(L_dense, barL_dense, k):
    n, _ = L_dense.shape
    Abb_band = np.zeros((k, n))
    L_band = banded_matrix(L_dense.copy(), k)
    Lb_band = banded_matrix(barL_dense.copy(), k)
    for i in range(n - 1, -1, -1):
        s = min(i + 1, k)
        for j in range(s):
            p = i - j
            if j == 0:
                Abb_band[0, p] = 0.5 * Lb_band[0, p] / L_band[0, p]
            else:
                Abb_band[j, p] = Lb_band[j, p] / L_band[0, p]
                Lb_band[0, p] -= Lb_band[j, p] * L_band[j, p] / L_band[0, p]
            a_jp = Abb_band[j, p]
            for l in range(1, s - j):
                pl = p - l
                jl = j + l
                Lb_band[jl, pl] -= a_jp * L_band[l, pl]
                Lb_band[l, pl] -= a_jp * L_band[jl, pl]
    return Abb_band


@pytest.mark.parametrize("n", [12, 17, 21])
@pytest.mark.parametrize("lower_bandwidth", [0, 1, 2, 3, 4, 5])
def test_forward_cholesky(lower_bandwidth, n):
    np.random.seed(4123469)

    Q_band = generate_banded_positive_definite_matrix(n, lower_bandwidth)
    Q_dense_lower = to_dense(Q_band, lower_bandwidth, 0)
    Q_dense = np.maximum(Q_dense_lower, Q_dense_lower.T)

    with tf.compat.v1.Session(graph=tf.Graph()) as session:
        cst_Q_band = tf.constant(Q_band)

        # Banded result, converted to dense for comparison
        cholQ_band_op = cholesky_band(cst_Q_band)
        cholQ_band = session.run(cholQ_band_op)
        cholQ_dense = to_dense(cholQ_band, lower_bandwidth, 0)

        # This checks that the operator does not mutate its input.
        # We expect strict equality here:
        input_eval = session.run(cst_Q_band)
        assert np.array_equal(input_eval, Q_band)

        # The Ls might not be uniquely determine, compare the resulting Q:
        Q_dense_rec = cholQ_dense @ cholQ_dense.T

        error = np.fabs(Q_dense_rec - Q_dense).max()
        print("Error", error)
        assert error < 1e-10


def test_forward_cholesky_without_result_check():
    # The idea is to set the should_check_result flag to False,
    # and use the smallest float in Python as threshold to
    # observe the desirable behaviour of no exception.
    # This test should pass without any exception since
    # Cholesky result numerical stability check is disabled.
    n = 12  # Dimension of the matrix.
    lower_bandwidth = 3  # Bandwidth.
    np.random.seed(4123469)
    with tf.compat.v1.Session(graph=tf.Graph()) as session:
        Q_band = generate_banded_positive_definite_matrix(n, lower_bandwidth)
        cst_Q_band = tf.constant(Q_band)

        # Banded result, converted to dense for comparison
        cholQ_band_op = cholesky_band(
            cst_Q_band,
            should_check_result=False,
            absolute_tolerance=sys.float_info.min,
            relative_tolerance=sys.float_info.min,
        )
        session.run(cholQ_band_op)


@pytest.mark.skip("Test currently fails: to fix")
def test_forward_cholesky_with_poorly_conditioned_banded_matrix():
    # The idea is to generate a pooly conditioned banded matrix,
    # and observe the result instability check to fail.
    n = 5  # Dimension of the matrix.
    lower_bandwidth = 0  # Bandwidth.
    np.random.seed(4123469)
    with pytest.raises(InternalError) as exp:
        with tf.compat.v1.Session(graph=tf.Graph()) as session:
            Q_band = generate_banded_positive_definite_matrix(n, lower_bandwidth)
            Q_band[0, 0] = 1e-10  # Make the matrix poorly conditioned.
            # For debugging.
            # dense = to_dense(Q_band, lower_bandwidth, lower_bandwidth)
            cst_Q_band = tf.constant(Q_band)

            # Banded result, converted to dense for comparison
            cholQ_band_op = cholesky_band(
                cst_Q_band,
                should_check_result=True,
                absolute_tolerance=sys.float_info.min,
                relative_tolerance=sys.float_info.min,
            )
            session.run(cholQ_band_op)
    assert exp.typename == "InternalError"
    assert exp.value.message.find("Banded Cholesky decomposition failed") == 0


@pytest.mark.skip("See PTKB-7813")
def test_forward_cholesky_with_result_check():
    # The idea is to set the should_check_result flag to True,
    # and use the smallest float in Python as threshold to
    # observe the desirable behaviour of an InternalError exception
    # is thrown.
    n = 12  # Dimension of the matrix.
    lower_bandwidth = 3  # Bandwidth.
    np.random.seed(4123469)
    with pytest.raises(InternalError) as exp:
        with tf.compat.v1.Session(graph=tf.Graph()) as session:
            Q_band = generate_banded_positive_definite_matrix(n, lower_bandwidth)
            cst_Q_band = tf.constant(Q_band)

            # Banded result, converted to dense for comparison
            cholQ_band_op = cholesky_band(
                cst_Q_band,
                should_check_result=True,
                absolute_tolerance=sys.float_info.min,
                relative_tolerance=sys.float_info.min,
            )
            session.run(cholQ_band_op)
    assert exp.typename == "InternalError"
    assert exp.value.message.find("Banded Cholesky decomposition failed") == 0


@pytest.mark.parametrize("lower_bandwidth", [0, 1, 4])
@pytest.mark.parametrize("n", [4, 8, 10])
def test_cholesky_gradient_against_tf_cholesky_gradient(lower_bandwidth, n):
    """
    Comparing reverse mode differentiation gradients of our banded op
    to a tensorflow dense counterpart
    """
    np.random.seed(641269)

    Q_band_lower = generate_banded_positive_definite_matrix(n, lower_bandwidth)
    Q_dense_lower = to_dense(Q_band_lower, lower_bandwidth, 0)

    grad_ys_band = np.ones_like(Q_band_lower)
    grad_ys_dense = to_dense(grad_ys_band, lower_bandwidth, 0)

    with tf.compat.v1.Session(graph=tf.Graph()):
        # forward operators
        cst_Q_band_lower = tf.constant(Q_band_lower)
        cholQ_band_op = cholesky_band(cst_Q_band_lower)
        cst_Q_dense_lower = tf.constant(Q_dense_lower)
        cst_Q_dense = (
            cst_Q_dense_lower
            + tf.transpose(a=cst_Q_dense_lower)
            - tf.linalg.tensor_diag(tf.linalg.tensor_diag_part(cst_Q_dense_lower))
        )
        cholQ_dense_op = tf.linalg.cholesky(cst_Q_dense)

        # Our gradient
        grad_L_band_op = tf.gradients(
            ys=cholQ_band_op, xs=cst_Q_band_lower, grad_ys=grad_ys_band
        )[0]
        grad_L_dense = to_dense(grad_L_band_op.eval(), lower_bandwidth, 0)

        # The TF gradient
        grad_L_dense_tf_op = tf.gradients(
            ys=cholQ_dense_op, xs=cst_Q_dense_lower, grad_ys=grad_ys_dense
        )[0]
        grad_L_dense_tf = extract_construct_banded_matrix(
            lower_bandwidth, 0, grad_L_dense_tf_op.eval()
        )

        # Comparing reverse mode gradients
        np.testing.assert_almost_equal(grad_L_dense_tf, grad_L_dense, decimal=8)


@pytest.mark.parametrize("lower_bandwidth", [0, 1, 4])
@pytest.mark.parametrize("n", [4, 8, 10])
def test_proto_cholesky_gradient(lower_bandwidth, n):
    """
    Comparing reverse mode differentiation gradients of our prototypes for
    banded precisions to a tensorflow dense counterpart
    """
    np.random.seed(641269)

    Q_band = generate_banded_positive_definite_matrix(n, lower_bandwidth)
    Q_dense_lower = to_dense(Q_band, lower_bandwidth, 0)
    Q_dense = np.maximum(Q_dense_lower, Q_dense_lower.T)

    grad_ys_band = np.ones_like(Q_band)
    grad_ys_dense = to_dense(grad_ys_band, lower_bandwidth, 0)

    with tf.compat.v1.Session(graph=tf.Graph()):
        # TF forward operator
        cst_Q_dense = constant_op(Q_dense)
        cholQ_dense_op = tf.linalg.cholesky(cst_Q_dense)

        # TF gradient
        [grad_L_dense_tf_op] = tf.gradients(
            ys=cholQ_dense_op, xs=cst_Q_dense, grad_ys=grad_ys_dense.copy()
        )

        # grad_L_dense_tf_op is a symmetric, not lower_triangular-banded matrix.
        # Gradients are therefore propagated evenly across the lower and upper
        # parts. When comparing to a lower-band representation we need to
        # multiply by 2 the part below diagonal
        symmetric_grad_L_dense_tf = grad_L_dense_tf_op.eval()
        np.testing.assert_almost_equal(symmetric_grad_L_dense_tf, symmetric_grad_L_dense_tf.T)

        crop = extract_band_from_matrix(lower_bandwidth, 0, symmetric_grad_L_dense_tf)
        crop[1:, :] *= 2
        grad_L_dense_tf = construct_banded_matrix_from_band(lower_bandwidth, 0, crop)

        # Reference (prototype) gradients
        cholQ_dense = cholQ_dense_op.eval()

        grad_L_dense_ref = ref_grad_cholesky_full(
            cholQ_dense, grad_ys_dense.copy(), lower_bandwidth + 1
        )
        grad_L_band_ref = construct_banded_matrix_from_band(
            lower_bandwidth,
            0,
            ref_grad_cholesky_band(cholQ_dense, grad_ys_dense.copy(), lower_bandwidth + 1),
        )

        # compare rev mode ref_dense vs ref_banded
        np.testing.assert_almost_equal(grad_L_dense_ref, grad_L_band_ref, decimal=8)

        # compare rev mode ref_dense vs tf_dense
        np.testing.assert_almost_equal(grad_L_dense_ref, grad_L_dense_tf, decimal=8)
