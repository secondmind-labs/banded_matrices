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
from scipy.linalg import cho_solve

from banded_matrices.banded import reverse_inverse_from_cholesky_band
from tests.utils.banded_matrices_utils import (
    construct_banded_matrix_from_band,
    extract_band_from_matrix,
    extract_construct_banded_matrix,
)

_ARBITRARY_BUT_CONSTANT_SEED = 434939


def generate_cholesky_factor(n=100, k=20):
    """
    Generate a toy banded Cholesky factor
    """

    L_band = np.random.uniform(low=0.1, high=1.0, size=(k, n))
    L_band[0, :] = np.abs(L_band[0, :])
    L_dense = construct_banded_matrix_from_band(k - 1, 0, L_band)
    return L_dense


#######################
# Sparse inverse subset


def sparse_inverse_subset(L, k):
    """
    Given a lower-triangular banded matrix L of size n and bandwidth k, compute
    the _band_ of the inverse of the product LLT.

    returns band(inv(L @ L.T))

    :param L: banded Cholesky factor (lower triangular)
    :param k: lower bandwidth of the Cholesky factor + 1 for the main diagonal.
    :return: S, the banded subset inverse of LL^T
    """
    n = L.shape[1]
    # Compute the U and D in Q = LDU
    d = np.diag(L)  # diagonal of the D matrix
    Lbar = L @ np.diag(1 / d)
    U = Lbar.T
    # Compute the sparse inverse subset S
    S = np.zeros((n, n))
    for j in range(n - 1, -1, -1):
        for i in range(j, max(j - k, -1), -1):
            S[i, j] = -np.sum(U[i, i + 1 : i + k].T * S[i + 1 : i + k, j])
            S[j, i] = S[i, j]
            if i == j:
                S[i, i] += 1 / d[i] ** 2
    return S


#######################
# Cholesky from sparse inverse


def reverse_inverse_from_cholesky_band_proto(S, l):
    """
    S -> L
    :param S: sparse subset inverse of banded matrix L
    :param l: number of subdiagonals in S
    :return: Ls: reconstructed cholesky decomposition
    """
    # forward pass
    k = l + 1  # bandwidth
    n = S.shape[1]
    # construct vector e = [1, 0, ..., 0]
    V = np.zeros_like(S)
    e = np.zeros((k))
    e[0] = 1
    for i in range(n):
        chol_S = np.linalg.cholesky(S[i : i + k, i : i + k])
        V[i : i + k, i] = cho_solve((chol_S, True), e[: n - i])
    Ls = V / np.sqrt(np.diag(V)[None, :])

    return Ls


def rev_mode_reverse_inverse_from_cholesky_band_proto(bL, S, l):
    """
    bL -> bS
    :param bL: Sensitivities of cholesky
    :param S: sparse subset inverse of banded matrix L
    :param l: number of subdiagonals in S
    :return: bS: Sensitivities of subset inverse
    """
    # forward pass
    k = l + 1  # bandwidth
    n = S.shape[1]
    # construct vector e = [1, 0, ..., 0]
    V = np.zeros_like(S)
    e = np.zeros((k))
    e[0] = 1
    for i in range(n):
        chol_S = np.linalg.cholesky(S[i : i + k, i : i + k])
        V[i : i + k, i] = cho_solve((chol_S, True), e[: n - i])
    Ls = V / np.sqrt(np.diag(V)[None, :])

    # backward pass
    bS = np.zeros_like(bL)
    for i in range(n):
        bLi = bL[i : i + k, i]
        chol_S = np.linalg.cholesky(S[i : i + k, i : i + k])
        Hi = np.eye(min(n - i, k))
        Hi[:, 0] -= Ls[i : i + k, i] / (2.0 * np.sqrt(V[i, i]))
        Hi /= np.sqrt(V[i, i])

        tmp = (bLi.T @ Hi).T
        tmp2 = cho_solve((chol_S, True), tmp)

        bSi = -V[i : i + k, i : i + 1] @ tmp2[None]
        bS[i : i + k, i : i + k] += 0.5 * (bSi + bSi.T)
    return bS


@pytest.mark.parametrize("n", [12, 21])
@pytest.mark.parametrize("lower_bandwidth", [0, 4])
def test_forward_reverse_inverse_from_cholesky_band(n, lower_bandwidth):
    """
    Testing C++ implementation of
    inverse of inverse from cholesky
    against a Python prototype
    """
    np.random.seed(_ARBITRARY_BUT_CONSTANT_SEED)

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        k = lower_bandwidth + 1

        L_dense = generate_cholesky_factor(n, k)

        # forward
        S_dense = sparse_inverse_subset(L_dense, k)
        S_band = extract_band_from_matrix(lower_bandwidth, 0, S_dense)

        # backward proto
        Ls_dense = reverse_inverse_from_cholesky_band_proto(S_dense, lower_bandwidth)
        Ls_band = extract_band_from_matrix(lower_bandwidth, 0, Ls_dense)

        # backward op
        Ls2_band = sess.run(reverse_inverse_from_cholesky_band(S_band, k))

        np.testing.assert_array_almost_equal(Ls_band, Ls2_band, decimal=10)


@pytest.mark.parametrize("n", [12, 21])
@pytest.mark.parametrize("lower_bandwidth", [0, 4])
def test_rev_mod_reverse_inverse_from_cholesky_band(n, lower_bandwidth):
    """
    Testing C++ implementation of the reverse mode derivatives of
    inverse of inverse from cholesky
    against a Python prototype
    """
    np.random.seed(_ARBITRARY_BUT_CONSTANT_SEED)

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        k = lower_bandwidth + 1

        L_dense = generate_cholesky_factor(n, k)
        S_dense = sparse_inverse_subset(L_dense, k)

        bL_dense = np.random.randn(n, n)
        bL_dense = extract_construct_banded_matrix(lower_bandwidth, 0, bL_dense)
        bL_band = extract_band_from_matrix(lower_bandwidth, 0, bL_dense)

        S_band = extract_band_from_matrix(lower_bandwidth, 0, S_dense)
        S_band_tf = tf.convert_to_tensor(value=S_band)
        L_band_tf = reverse_inverse_from_cholesky_band(S_band_tf, k)
        bS_band_tf = tf.gradients(
            ys=L_band_tf, xs=S_band_tf, grad_ys=tf.convert_to_tensor(value=bL_band)
        )[0]

        bS_dense = rev_mode_reverse_inverse_from_cholesky_band_proto(
            bL_dense, S_dense, lower_bandwidth
        )
        bS_band = extract_band_from_matrix(lower_bandwidth, 0, bS_dense)
        bS2_band = sess.run(bS_band_tf)

        np.testing.assert_array_almost_equal(bS_band, bS2_band, decimal=10)
