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

from banded_matrices.banded import cholesky_band, inverse_from_cholesky_band
from tests.utils.banded_matrices_utils import to_dense


def gauss_kl(q_mu, q_sqrt):
    """
    simplified KL from GPflow (takes cholesky of covariance as argument)
    """
    alpha = q_mu  # M x B
    Lq = tf.linalg.band_part(q_sqrt, -1, 0)  # force lower triangle # B x M x M
    Lq_diag = tf.linalg.diag_part(Lq)  # M x B
    # Mahalanobis term: μqᵀ Σp⁻¹ μq
    mahalanobis = tf.reduce_sum(input_tensor=tf.square(alpha))
    # Constant term: - B * M
    constant = -tf.size(input=q_mu, out_type=tf.float64)
    # Log-determinant of the covariance of q(x):
    logdet_qcov = tf.reduce_sum(input_tensor=tf.math.log(tf.square(Lq_diag)))
    # Trace term: tr(Σp⁻¹ Σq)
    trace = tf.reduce_sum(input_tensor=tf.square(Lq))
    twoKL = mahalanobis + constant - logdet_qcov + trace
    return 0.5 * twoKL


def gauss_kl_white_from_chol_prec(mu, L_band):
    """
    KL ( N(mu, Q=LL^T)|| N(0,1))
    KL =  1/2*{ log[ |Q1| ]  - d + Tr[Q1^-1] +  m1^T m1 }
    """
    n = L_band.shape[1]
    # log det term
    log_det = -tf.reduce_sum(input_tensor=tf.math.log(tf.square(L_band[0, :])))
    # mahalanobis
    mahalanobis = tf.reduce_sum(input_tensor=tf.square(mu))
    # trace term
    trace = tf.reduce_sum(input_tensor=inverse_from_cholesky_band(L_band)[0, :])
    # constant
    constant = -tf.cast(n, dtype=tf.float64)
    twoKL = mahalanobis + constant + trace - log_det
    return 0.5 * twoKL


def gauss_kl_white_from_chol_prec_dense(mu, L):
    """
    KL ( N(mu, Q=LL^T)|| N(0,1))
    KL =  1/2*{ log[ |Q1| ]  - d + Tr[Q1^-1] +  m1^T m1 }
    """
    n = L.shape[1]
    # log det term
    log_det = -tf.reduce_sum(input_tensor=tf.math.log(tf.square(tf.linalg.diag_part(L))))
    # mahalanobis
    mahalanobis = tf.reduce_sum(input_tensor=tf.square(mu))
    # trace term
    trace = tf.linalg.trace(tf.linalg.cholesky_solve(L, np.eye(n)))
    # constant
    constant = -tf.cast(n, dtype=tf.float64)
    twoKL = mahalanobis + constant + trace - log_det
    return 0.5 * twoKL


def gauss_kl_white_from_prec(mu, Q_band):
    L_band = cholesky_band(Q_band)
    return gauss_kl_white_from_chol_prec(mu, L_band)


def gauss_kl_white_from_prec_dense(mu, Q):
    L = tf.linalg.cholesky(Q)
    return gauss_kl_white_from_chol_prec_dense(mu, L)


@pytest.mark.parametrize("n", [10, 15])
@pytest.mark.parametrize("l", [0, 3])
def test_kl(n, l):
    """
    Compares kl using banded ops to full counterpart
    """
    with tf.compat.v1.Session(graph=tf.Graph()):
        np.random.seed(0)

        # generate random cholesky matrix and vector
        L_band = np.random.rand(l + 1, n) + 1
        L_band[0, :] = np.abs(L_band[0, :])
        L_dense = to_dense(L_band, l, 0)
        Ls_dense = np.linalg.inv(L_dense)
        mu = np.random.rand(
            n,
        )

        # compute KL divergences
        kl = gauss_kl_white_from_chol_prec(mu, L_band).eval()
        kl_dense = gauss_kl_white_from_chol_prec_dense(mu, L_dense).eval()
        kl_cov = gauss_kl(mu, Ls_dense).eval()

        # compare
        np.testing.assert_almost_equal(kl, kl_dense, decimal=8)
        print("Error |kl-kl_dense|:", np.fabs(kl - kl_dense).max())
        np.testing.assert_almost_equal(kl, kl_dense, decimal=8)
        print("Error |kl-kl_dense_cov|:", np.fabs(kl - kl_cov).max())
