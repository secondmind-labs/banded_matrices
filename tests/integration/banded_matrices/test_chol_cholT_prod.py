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

from banded_matrices.banded import product_band_band, product_band_mat
from tests.utils.banded_matrices_utils import (
    construct_banded_matrix_from_band,
    extract_band_from_matrix,
)


@pytest.mark.parametrize("n", [10, 15])
@pytest.mark.parametrize("l", [0, 1, 5])
def test_product_chol_cholT(n, l):
    with tf.compat.v1.Session(graph=tf.Graph()):
        L_band = np.random.randn(l + 1, n)
        L_band[0, :] = np.abs(L_band[0, :])

        L_dense = construct_banded_matrix_from_band(l, 0, L_band)
        Q_dense = L_dense @ L_dense.T

        Q_band_op = product_band_band(
            L_band,
            L_band,
            transpose_right=True,
            left_lower_bandwidth=l,
            left_upper_bandwidth=0,
            right_lower_bandwidth=l,
            right_upper_bandwidth=0,
            result_lower_bandwidth=l,
            result_upper_bandwidth=0,
        )
        Q_band_from_dense = extract_band_from_matrix(l, 0, Q_dense)

        # now do product with vector
        m = np.random.rand(n, 1)
        m_op = tf.constant(m)
        v_band_op = product_band_mat(
            Q_band_op,
            m_op,
            left_lower_bandwidth=l,
            left_upper_bandwidth=0,
            symmetrise_left=True,
        )

        v_dense = Q_dense @ m
        Q_band = Q_band_op.eval()
        v_band = v_band_op.eval()

        print("Integration test")
        np.testing.assert_almost_equal(Q_band, Q_band_from_dense)
        np.testing.assert_almost_equal(v_band, v_dense)
