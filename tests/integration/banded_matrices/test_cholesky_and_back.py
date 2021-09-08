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

from banded_matrices import banded
from tests.utils.banded_matrices_utils import generate_band_mat


def banded_matrix(m, k):
    n = m.shape[0]
    assert n == m.shape[1]
    a = np.zeros((k, n))
    for i in range(k):
        a[i, : n - i] = np.diagonal(m, offset=-i)
    return a


@pytest.mark.parametrize("n", [12, 17, 21])
@pytest.mark.parametrize("lower_bandwidth", [0, 1, 3, 4, 5])
def test_cholesky_and_back(lower_bandwidth, n):
    np.random.seed(41239)

    # Generate a lower band with positive diagonal
    L_init = generate_band_mat(n, lower_bandwidth, 0) + 1
    L_init[0, :] = np.abs(L_init[0, :])

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        Q_init = banded.product_band_band(
            L_init,
            L_init,
            left_lower_bandwidth=lower_bandwidth,
            left_upper_bandwidth=0,
            right_lower_bandwidth=lower_bandwidth,
            right_upper_bandwidth=0,
            result_lower_bandwidth=lower_bandwidth,
            result_upper_bandwidth=0,
            transpose_right=True,
        )

        L = banded.cholesky_band(Q_init)
        Q = banded.square_band(L, lower_bandwidth=lower_bandwidth, upper_bandwidth=0)

        print(sess.run(Q))
        print(sess.run(Q_init))

        grad_ys = generate_band_mat(n, lower_bandwidth, 0)
        grad = tf.gradients(ys=Q, xs=Q_init, grad_ys=grad_ys)
        g = sess.run(grad)[0]

        print(g)
        print(grad_ys)

    np.testing.assert_almost_equal(g, grad_ys, decimal=8)
