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

from banded_matrices.banded import product_band_mat
from tests.utils.banded_matrices_utils import (
    compute_gradient_error,
    constant_op,
    construct_extract_banded_matrix,
    extract_band_from_matrix,
    generate_band_mat,
    to_dense,
)


@pytest.mark.parametrize("dim", [16])
@pytest.mark.parametrize("vector_count", [1, 4])
@pytest.mark.parametrize("band", [(2, 0), (0, 4), (0, 0), (3, 3), (1, 5), (7, 0)])
@pytest.mark.parametrize("flags", [(False, False), (True, False), (False, True)])
def test_matrix_vector_product(dim, band, flags, vector_count):

    with tf.compat.v1.Session(graph=tf.Graph()) as session:
        lower_bandwidth, upper_bandwidth = band
        transpose, symmetrise = flags

        if symmetrise and upper_bandwidth != 0:
            # Skip this combination - symmetric should currently be lower-diag
            return

        banded_matrix = generate_band_mat(dim, lower_bandwidth, upper_bandwidth)
        vector = np.random.rand(dim, vector_count)

        dense_matrix = to_dense(banded_matrix, lower_bandwidth, upper_bandwidth)

        left = dense_matrix
        if symmetrise:
            left += dense_matrix.T - np.diag(np.diag(dense_matrix))

        m = constant_op(banded_matrix)
        v = constant_op(vector)

        product_op = product_band_mat(
            m, v, lower_bandwidth, upper_bandwidth, transpose, symmetrise
        )
        product_tf_op = tf.matmul(left, v, transpose_a=transpose)

        product = session.run(product_op)
        product_tf = session.run(product_tf_op)

        np.testing.assert_almost_equal(actual=product, desired=product_tf, decimal=2)


@pytest.mark.parametrize("dim", [10, 20])
@pytest.mark.parametrize("transpose_left", (False, True))
@pytest.mark.parametrize("vector_count", [1, 3])
@pytest.mark.parametrize("band", [(3, 0), (0, 3), (0, 0), (3, 3)])
def test_jacobian_product_band_mat(dim, band, vector_count, transpose_left):
    """
    Gradients are only valid for an operator that has all Boolean flags False.
    """
    with tf.compat.v1.Session(graph=tf.Graph()):

        lower_bandwidth, upper_bandwidth = band
        banded_matrix = generate_band_mat(dim, lower_bandwidth, upper_bandwidth)
        vector = np.random.rand(dim, vector_count)

        m = constant_op(banded_matrix)
        v = constant_op(vector)
        product_op = product_band_mat(
            m, v, lower_bandwidth, upper_bandwidth, transpose_left=transpose_left
        )

        # Error for dp/m
        jac_err_m = compute_gradient_error(m, product_op)

        # Error for dp/m
        jac_err_v = compute_gradient_error(v, product_op)

        print("gradient errors: ", jac_err_m, jac_err_v)
        assert jac_err_m < 1e-10
        assert jac_err_v < 1e-10


@pytest.mark.parametrize("dim", [10, 20])
@pytest.mark.parametrize("vector_count", [1, 17])
@pytest.mark.parametrize("transpose_left", (False, True))
@pytest.mark.parametrize("band", [(3, 0), (0, 3), (0, 0), (3, 3)])
def test_rev_mode_gradients_product_band_mat(dim, band, vector_count, transpose_left):
    """
    Testing reverse mode gradients of product_band_mat against tf.matmul
    """
    with tf.compat.v1.Session(graph=tf.Graph()):

        lower_bandwidth, upper_bandwidth = band
        banded_matrix = generate_band_mat(dim, lower_bandwidth, upper_bandwidth)
        vector = np.random.rand(dim, vector_count)
        grad_ys = np.ones((dim, vector_count))

        m_dense = constant_op(to_dense(banded_matrix, lower_bandwidth, upper_bandwidth))
        m_band = constant_op(banded_matrix)
        v = constant_op(vector)

        product_op = product_band_mat(
            m_band, v, lower_bandwidth, upper_bandwidth, transpose_left=transpose_left
        )
        product_tf_op = tf.matmul(m_dense, v, transpose_a=transpose_left)

        # Gradients banded
        [grad_m_op, grad_v_op] = tf.gradients(ys=product_op, xs=[m_band, v], grad_ys=grad_ys)
        grad_m = construct_extract_banded_matrix(
            lower_bandwidth, upper_bandwidth, grad_m_op.eval()
        )
        grad_v = grad_v_op.eval()

        # Gradients dense (tf)
        [grad_tf_m_op, grad_tf_v_op] = tf.gradients(
            ys=product_tf_op, xs=[m_dense, v], grad_ys=grad_ys
        )
        grad_tf_m = extract_band_from_matrix(
            lower_bandwidth, upper_bandwidth, grad_tf_m_op.eval()
        )
        grad_tf_v = grad_tf_v_op.eval()

        # Error checks
        grad_err_m = np.fabs(grad_m - grad_tf_m).max()
        grad_err_v = np.fabs(grad_v - grad_tf_v).max()

        print("product_band_mat gradient errors w.r.t. TF dense: ", grad_err_m, grad_err_v)

        assert grad_err_m < 1e-10
        assert grad_err_v < 1e-10
