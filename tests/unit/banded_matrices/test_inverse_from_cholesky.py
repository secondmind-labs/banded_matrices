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
from copy import copy

import numpy as np
import pytest
import tensorflow as tf

from banded_matrices.banded import _grad_inverse_from_cholesky_band, inverse_from_cholesky_band
from tests.utils.banded_matrices_utils import (
    constant_op,
    construct_banded_matrix_from_band,
    extract_construct_banded_matrix,
    generate_band_mat,
    to_dense,
)


@pytest.mark.parametrize("n", [12, 21])
@pytest.mark.parametrize("lower_bandwidth", [0, 4])
@pytest.mark.parametrize("result_lower_bandwidth", [0, 1, 4, 6])
def test_forward_inverse_from_cholesky(lower_bandwidth, result_lower_bandwidth, n):
    """
    Testing computation of band of (LL^T)^-1 from L
    """
    np.random.seed(4123469)

    # Generate a lower band with positive diagonal
    # NOTE the lower precision here if we don't tweak the generation (+1)
    band = generate_band_mat(n, lower_bandwidth, 0) + 2
    band[0, :] = np.abs(band[0, :])

    # Compute the Q that the band is a Cholesky of:
    dense_L = to_dense(band, lower_bandwidth, 0)

    with tf.compat.v1.Session(graph=tf.Graph()) as session:
        cst_band = constant_op(band)

        # Banded result, converted to dense for comparison
        inverse_op = inverse_from_cholesky_band(cst_band, result_lower_bandwidth)
        inverse = session.run(inverse_op)
        inverse_dense = to_dense(inverse, result_lower_bandwidth, 0)

        # Dense TF version
        Q_dense = dense_L @ dense_L.T
        inverse_dense_tf2_op = tf.linalg.inv(Q_dense)
        inverse_dense_tf_op = tf.linalg.cholesky_solve(dense_L, np.identity(n))
        inverse_dense_tf = extract_construct_banded_matrix(
            result_lower_bandwidth, 0, session.run(inverse_dense_tf_op)
        )
        inverse_dense_tf2 = extract_construct_banded_matrix(
            result_lower_bandwidth, 0, session.run(inverse_dense_tf2_op)
        )

        error = np.fabs(inverse_dense_tf - inverse_dense).max()
        error2 = np.fabs(inverse_dense_tf2 - inverse_dense).max()

        assert error < 1e-11
        assert error2 < 1e-11


def gradient_reference_code(L, n, k, result_lower_bandwidth, bS, S):
    """
    Reference code for the gradient.
    This has been generated from Tangent and simplified/optimized by hand.
    Note that the matrices are here dense, just 0 out of band.
    """
    assert bS.shape == (n, n), "bad backprop shape for S"
    vec = np.diag(L)
    U = (L / vec).T

    bU = np.zeros_like(U)
    bvec_inv_2 = np.zeros(n)

    # Beginning of backward pass
    for j in range(n):
        for i in range(max(0, j - result_lower_bandwidth), j + 1):
            if i == j:
                bvec_inv_2[i] += bS[i, i]

            # Grad of: S[j, i] = S[i, j]
            tmp = copy(bS[j, i])
            bS[j, i] = 0.0
            bS[i, j] += tmp

            # Grad of: S[i, j] = -np.sum(U[i, i+1:i+k] * S[i+1:i+k, j])
            bU[i, i + 1 : i + k] -= S[i + 1 : i + k, j] * bS[i, j]
            bS[i + 1 : i + k, j] -= U[i, i + 1 : i + k] * bS[i, j]
            bS[i, j] = 0.0

    # Grad of: U = np.transpose(L * vec_inv)
    bL = bU.T / vec

    # Grad of: vec_inv_2 = 1.0 / vec ** 2
    bvec = -2.0 * bvec_inv_2 / vec ** 3

    # Grad of: vec_inv = 1.0 / vec
    bvec -= np.sum(bU.T * L, 0) / (vec ** 2)

    # Grad of: vec = diag(L)
    bL += np.diag(bvec)

    return bL


def gradient_reference_code_short(L, n, k, bS, S):
    """
    Reference code for the gradient.
    This has been generated from Tangent and simplified/optimized by hand.
    Note that the matrices are here dense, just 0 out of band.
    """
    assert bS.shape == (n, n), "bad backprop shape for S"
    vec = np.diag(L)
    U = (L / vec).T
    bU = np.zeros_like(U)
    bL = np.zeros_like(L)

    for j in range(n):
        for i in range(max(0, j - k + 1), j + 1):
            if i != j:
                bS[i, j] += bS[j, i]
            bS[i + 1 : i + k, j] -= U[i, i + 1 : i + k] * bS[i, j]
            bU[i, i + 1 : i + k] -= S[i + 1 : i + k, j] * bS[i, j]

    bL += bU.T / vec + (
        np.diag(-2.0 * np.diag(bS) / vec ** 3 - np.sum(bU.T * L, 0) / (vec ** 2))
    )

    return bL


@pytest.mark.parametrize("n", [12, 17])
@pytest.mark.parametrize("result_lower_bandwidth", [4, 6])
@pytest.mark.parametrize("lower_bandwidth", [0, 4])
def test_gradient_against_reference_python_code(n, lower_bandwidth, result_lower_bandwidth):
    np.random.seed(279)
    with tf.compat.v1.Session(graph=tf.Graph()) as session:

        # The L Cholesky matrix, input of the op in forward mode
        k = lower_bandwidth + 1
        L_band = generate_band_mat(n, lower_bandwidth, 0)
        L_band[0, :] = np.abs(L_band[0, :])
        L_dense = to_dense(L_band, lower_bandwidth, 0)
        # Gradients of output, assumed to be 1 everywhere
        grad_ys = np.ones((result_lower_bandwidth + 1, n))
        grad_ys_dense = to_dense(grad_ys, result_lower_bandwidth, 0)
        grad_ys_dense += grad_ys_dense.T - np.diag(np.diag(grad_ys_dense))

        # This is to take into account implicit symmetry
        grad_ys[1:, :] *= 2.0

        # Our implementation of the gradient:
        cst_k_band = constant_op(L_band)
        inverse_op = inverse_from_cholesky_band(cst_k_band, result_lower_bandwidth)
        grad_L_op = _grad_inverse_from_cholesky_band(inverse_op.op, grad_ys)
        grad_L = to_dense(session.run(grad_L_op), lower_bandwidth, 0)

        S_non_sym = to_dense(session.run(inverse_op), result_lower_bandwidth, 0)
        S_symmetrised = S_non_sym + S_non_sym.T - np.diag(np.diag(S_non_sym))

        # The reference:
        grad_L_ref = gradient_reference_code(
            L_dense,
            n,
            k,
            result_lower_bandwidth=result_lower_bandwidth,
            bS=grad_ys_dense,
            S=S_symmetrised,
        )

        # NOTE: with a debug build this passes up to 8 decimals.
        # In Release build this passes up to 5 decimals only.
        print("Gradient error ", np.fabs(grad_L - grad_L_ref).max())
        np.testing.assert_almost_equal(actual=grad_L, desired=grad_L_ref, decimal=7)


# @pytest.mark.skip(reason="Fixing Before merging")
@pytest.mark.parametrize("n", [17, 19])
@pytest.mark.parametrize("l", [0, 1, 4])
@pytest.mark.parametrize("result_lower_bandwidth", [0, 2, 4, 6])
def test_gradients_inverse_from_cholesky_against_tf_cholesky_solve(
    n, l, result_lower_bandwidth
):
    """
    Comparing reverse mode gradient of operator L -> band[ inv(LL^T) ]
    for our banded operator and the dense version using tf_cholesky_solve
    """
    with tf.compat.v1.Session(graph=tf.Graph()):
        np.random.seed(279)
        # create data : L lower banded and bar{S} symmetric banded
        L_band = np.random.randint(1, 4, size=(l + 1, n)).astype(float)
        L = construct_banded_matrix_from_band(l, 0, L_band)
        L_tf = tf.constant(L, dtype=tf.float64)
        # run forward L -> b[ inv(LL.T) ]
        S_tf = tf.linalg.cholesky_solve(L_tf, np.identity(n))
        # Constructing df/dS
        cst_band = constant_op(L_band)
        grad_ys_band = np.random.rand(result_lower_bandwidth + 1, n)
        # bar[S] is explicitely symmetrised for the band version
        grad_ys = to_dense(grad_ys_band, result_lower_bandwidth, 0)
        grad_ys += grad_ys.T - np.diag(np.diag(grad_ys))
        # This is to take into account implicit symmetry
        grad_ys_band[1:, :] *= 2.0
        inverse_op = inverse_from_cholesky_band(cst_band, result_lower_bandwidth)
        [grad_L_op] = tf.gradients(ys=inverse_op, xs=[cst_band], grad_ys=grad_ys_band)
        grad_L = to_dense(grad_L_op.eval(), l, 0)
        # run gradient
        grad_L_tf_op = tf.gradients(ys=S_tf, xs=L_tf, grad_ys=grad_ys.copy())
        grad_L_tf = extract_construct_banded_matrix(l, 0, grad_L_tf_op[0].eval())
        np.testing.assert_almost_equal(grad_L, grad_L_tf, decimal=9)
