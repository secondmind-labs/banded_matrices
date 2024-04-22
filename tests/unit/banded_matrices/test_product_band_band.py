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

from banded_matrices.banded import product_band_band
from tests.utils.banded_matrices_utils import (
    compute_gradient_error,
    constant_op,
    construct_banded_matrix_from_band,
    extract_band_from_matrix,
    extract_construct_banded_matrix,
    generate_band_mat,
    to_dense,
)

SOME_SHAPES = [(2, 0), (1, 5), (3, 3), (0, 0)]


@pytest.mark.parametrize("bandwidth1", SOME_SHAPES)
@pytest.mark.parametrize("bandwidth2", SOME_SHAPES)
@pytest.mark.parametrize("out_bandwidth", SOME_SHAPES)
@pytest.mark.parametrize("tr1", [True, False])
@pytest.mark.parametrize("tr2", [True, False])
@pytest.mark.parametrize("sym1", [True, False])
@pytest.mark.parametrize("sym2", [True, False])
@pytest.mark.parametrize("n", [15])
def test_forward_product_band_band(
    bandwidth1, bandwidth2, out_bandwidth, tr1, tr2, sym1, sym2, n
):
    """
    For forward mode all combinations of transposition/symmetrization should be correct.
    """

    def make_product_argument(dense_matrix: np.ndarray, transpose, symmetric):
        if transpose:
            return dense_matrix.transpose()
        elif symmetric:
            return dense_matrix + dense_matrix.T - np.diag(np.diag(dense_matrix))
        else:
            return dense_matrix

    l1, u1 = bandwidth1
    l2, u2 = bandwidth2
    lout, uout = out_bandwidth

    if (u2 > 0 and sym2) or (u1 > 0 and sym1) or (tr2 and sym2) or (tr1 and sym1):
        return

    with tf.compat.v1.Session(graph=tf.Graph()) as session:
        print("Evaluating ", (l1, u1), (l2, u2), (tr1, sym1, tr2, sym2))

        banded1 = generate_band_mat(n, l1, u1)
        banded2 = generate_band_mat(n, l2, u2)

        dense1 = to_dense(banded1, l1, u1)
        dense2 = to_dense(banded2, l2, u2)

        cst_op1 = constant_op(banded1)
        cst_op2 = constant_op(banded2)

        product = product_band_band(
            cst_op1,
            cst_op2,
            left_lower_bandwidth=l1,
            left_upper_bandwidth=u1,
            right_lower_bandwidth=l2,
            right_upper_bandwidth=u2,
            result_lower_bandwidth=lout,
            result_upper_bandwidth=uout,
            transpose_left=tr1,
            transpose_right=tr2,
            symmetrise_left=sym1,
            symmetrise_right=sym2,
        )

        banded_prod = session.run(product)
        print(banded_prod.shape)
        calculated_prod = to_dense(banded_prod, lout, uout)
        checked_product = make_product_argument(dense1, tr1, sym1).dot(
            make_product_argument(dense2, tr2, sym2)
        )
        checked_product = extract_construct_banded_matrix(lout, uout, checked_product)

        np.testing.assert_almost_equal(
            actual=calculated_prod, desired=checked_product, decimal=2
        )
        print("foward evaluation OK\n")


@pytest.mark.parametrize("bandwidth1", SOME_SHAPES)
@pytest.mark.parametrize("bandwidth2", SOME_SHAPES)
@pytest.mark.parametrize("n", [15])
def test_jacobian_product_band_band(bandwidth1, bandwidth2, n):
    """
    Gradients are only valid for an operator that has all Boolean flags False.
    """
    l1, u1 = bandwidth1
    l2, u2 = bandwidth2

    with tf.compat.v1.Session(graph=tf.Graph()):
        print("\nChecking jacobian for ", (l1, u1), (l2, u2))

        banded1 = generate_band_mat(n, l1, u1)
        banded2 = generate_band_mat(n, l2, u2)

        cst_banded1 = constant_op(banded1)
        cst_banded2 = constant_op(banded2)

        product = product_band_band(
            cst_banded1,
            cst_banded2,
            left_lower_bandwidth=l1,
            left_upper_bandwidth=u1,
            right_lower_bandwidth=l2,
            right_upper_bandwidth=u2,
        )

        # Error for dp/dx1
        jac_err_1 = compute_gradient_error(cst_banded1, product)

        # Error for dp/dx2
        jac_err_2 = compute_gradient_error(cst_banded2, product)

        print("gradient errors: ", jac_err_1, jac_err_2)
        assert jac_err_1 < 1e-8
        assert jac_err_2 < 1e-8


@pytest.mark.parametrize("bandwidth1", SOME_SHAPES)
@pytest.mark.parametrize("bandwidth2", SOME_SHAPES)
@pytest.mark.parametrize("n", [15])
def test_algebra_reverse_mode_gradient_product_band_band(bandwidth1, bandwidth2, n):
    """
    Testing reverse mode gradients of product_band_band against algebra
    """
    l1, u1 = bandwidth1
    l2, u2 = bandwidth2

    with tf.compat.v1.Session(graph=tf.Graph()):
        print("\nChecking gradients for ", (l1, u1), (l2, u2))

        banded1 = generate_band_mat(n, l1, u1)
        banded2 = generate_band_mat(n, l2, u2)

        dense1 = to_dense(banded1, l1, u1)
        dense2 = to_dense(banded2, l2, u2)

        cst_op1 = constant_op(banded1)
        cst_op2 = constant_op(banded2)

        product = product_band_band(
            cst_op1,
            cst_op2,
            left_lower_bandwidth=l1,
            left_upper_bandwidth=u1,
            right_lower_bandwidth=l2,
            right_upper_bandwidth=u2,
        )

        banded_bar_P = np.ones((l1 + l2 + u1 + u2 + 1, n))
        bar_P = construct_banded_matrix_from_band(l1 + l2, u1 + u2, banded_bar_P)

        # reverse mode for left argument
        banded_bar_B1_np = extract_band_from_matrix(l1, u1, np.dot(bar_P, dense2.T))
        grad_1 = tf.gradients(ys=product, xs=cst_op1, grad_ys=banded_bar_P)[0].eval()

        # reverse mode for right argument
        banded_bar_B2_np = extract_band_from_matrix(l2, u2, np.dot(dense1.T, bar_P))
        grad_2 = tf.gradients(ys=product, xs=cst_op2, grad_ys=banded_bar_P)[0].eval()

        np.testing.assert_almost_equal(grad_1, banded_bar_B1_np)
        np.testing.assert_almost_equal(grad_2, banded_bar_B2_np)


@pytest.mark.parametrize("bandwidth1", SOME_SHAPES)
@pytest.mark.parametrize("bandwidth2", SOME_SHAPES)
@pytest.mark.parametrize("tr1", [False, True])
@pytest.mark.parametrize("tr2", [False, True])
@pytest.mark.parametrize("out_bandwidth", SOME_SHAPES)
@pytest.mark.parametrize("n", [15])
def test_reverse_mode_gradient_product_band_band_against_tf(
    bandwidth1, bandwidth2, n, tr1, tr2, out_bandwidth
):
    """
    Testing reverse mode gradients of product_band_band against tf matmul
    """
    l1, u1 = bandwidth1
    l2, u2 = bandwidth2
    lout, uout = out_bandwidth

    with tf.compat.v1.Session(graph=tf.Graph()):
        print("\nChecking gradients for ", (l1, u1), (l2, u2), (lout, uout), (tr1, tr2))

        banded1 = generate_band_mat(n, l1, u1)
        banded2 = generate_band_mat(n, l2, u2)

        dense1 = to_dense(banded1, l1, u1)
        dense2 = to_dense(banded2, l2, u2)

        cst_banded1 = constant_op(banded1)
        cst_banded2 = constant_op(banded2)

        cst_dense1 = constant_op(dense1)
        cst_dense2 = constant_op(dense2)

        product = product_band_band(
            cst_banded1,
            cst_banded2,
            left_lower_bandwidth=l1,
            left_upper_bandwidth=u1,
            right_lower_bandwidth=l2,
            right_upper_bandwidth=u2,
            transpose_left=tr1,
            transpose_right=tr2,
            result_lower_bandwidth=lout,
            result_upper_bandwidth=uout,
        )

        product_tf = tf.matmul(cst_dense1, cst_dense2, transpose_a=tr1, transpose_b=tr2)

        # compute reverse mode gradients
        banded_bar_P = np.ones((lout + 1 + uout, n))
        bar_P = construct_banded_matrix_from_band(lout, uout, banded_bar_P)

        # reverse mode for left argument
        grad_1 = tf.gradients(ys=product, xs=cst_banded1, grad_ys=banded_bar_P)[0].eval()
        grad_tf_1 = extract_band_from_matrix(
            l1, u1, tf.gradients(ys=product_tf, xs=cst_dense1, grad_ys=bar_P)[0].eval()
        )

        # reverse mode for right argument
        grad_2 = tf.gradients(ys=product, xs=cst_banded2, grad_ys=banded_bar_P)[0].eval()
        grad_tf_2 = extract_band_from_matrix(
            l2, u2, tf.gradients(ys=product_tf, xs=cst_dense2, grad_ys=bar_P)[0].eval()
        )

        # compare
        np.testing.assert_almost_equal(grad_1, grad_tf_1)
        np.testing.assert_almost_equal(grad_2, grad_tf_2)


@pytest.mark.parametrize("tr1", [False, True])
@pytest.mark.parametrize("tr2", [False, True])
def test_gradient_of_square(tr1, tr2):
    """
    Test product where the same term is passed left and right.
    Here the banded matrix for the resulting product is not truncated.
    """
    n = 10
    l, u = 2, 3

    with tf.compat.v1.Session(graph=tf.Graph()):
        banded = generate_band_mat(n, l, u)
        dense = to_dense(banded, l, u)

        cst_banded = constant_op(banded)
        cst_dense = constant_op(dense)

        product = product_band_band(
            cst_banded,
            cst_banded,
            left_lower_bandwidth=l,
            left_upper_bandwidth=u,
            right_lower_bandwidth=l,
            right_upper_bandwidth=u,
            transpose_left=tr1,
            transpose_right=tr2,
        )

        lout = product.op.get_attr("result_lower_bandwidth")
        uout = product.op.get_attr("result_upper_bandwidth")

        product_tf = tf.matmul(cst_dense, cst_dense, transpose_a=tr1, transpose_b=tr2)

        # compute reverse mode gradients
        banded_bar_P = np.ones((lout + 1 + uout, n))
        bar_P = construct_banded_matrix_from_band(lout, uout, banded_bar_P)
        banded_bar_P = extract_band_from_matrix(lout, uout, bar_P)

        grad = tf.gradients(ys=product, xs=cst_banded, grad_ys=banded_bar_P)[0].eval()

        grad_tf = extract_band_from_matrix(
            l, u, tf.gradients(ys=product_tf, xs=cst_dense, grad_ys=bar_P)[0].eval()
        )

        # compare
        np.testing.assert_almost_equal(grad, grad_tf)


@pytest.mark.parametrize("lout", [0, 1, 2])
@pytest.mark.parametrize("uout", [0, 1, 2, 3])
def test_gradient_of_L_Lt(lout, uout):
    """
    Test product for symmetric matrices of the form L * L^T.
    Here we truncate the result to arbitrary sub-bands of the result,
    without consideration of the symmetry of the result.
    The gradient is always consistent with what we'd have with a dense
    representation with 0s out of the band.
    """
    n = 10
    l, u = 1, 3

    with tf.compat.v1.Session(graph=tf.Graph()):
        banded = generate_band_mat(n, l, u)
        dense = to_dense(banded, l, u)

        cst_banded = constant_op(banded)
        cst_dense = constant_op(dense)

        product = product_band_band(
            cst_banded,
            cst_banded,
            transpose_right=True,
            left_lower_bandwidth=l,
            left_upper_bandwidth=u,
            right_lower_bandwidth=l,
            right_upper_bandwidth=u,
            result_lower_bandwidth=lout,
            result_upper_bandwidth=uout,
        )

        # compute reverse mode gradients
        banded_bar_P = np.ones((lout + 1 + uout, n))
        bar_P = construct_banded_matrix_from_band(lout, uout, banded_bar_P)
        banded_bar_P = extract_band_from_matrix(lout, uout, bar_P)

        product_tf = tf.matmul(cst_dense, cst_dense, transpose_b=True)

        grad = tf.gradients(ys=product, xs=cst_banded, grad_ys=banded_bar_P)[0].eval()

        grad_tf = extract_band_from_matrix(
            l, u, tf.gradients(ys=product_tf, xs=cst_dense, grad_ys=bar_P)[0].eval()
        )

        # compare
        np.testing.assert_almost_equal(grad, grad_tf)
