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

from banded_matrices.banded import transpose_band
from tests.utils.banded_matrices_utils import (
    constant_op,
    construct_banded_matrix_from_band,
    extract_band_from_matrix,
    generate_band_mat,
    to_dense,
)

BANDWIDTHS = [(0, 3), (3, 0), (0, 0), (2, 3), (3, 2), (2, 1)]


@pytest.mark.parametrize("bands", BANDWIDTHS)
def test_transpose1(bands):
    """
    Test the forward evaluation of transpose on
    banded lower, upper, diagonal, upper and lower
    """
    n = 10
    l, u = bands

    with tf.compat.v1.Session(graph=tf.Graph()) as session:
        # generate band of matrices and dense representation
        banded = generate_band_mat(n, l, u)
        dense = to_dense(banded, l, u)

        # evaluate band transpose
        cst_op1 = constant_op(banded)
        cst_op1T = transpose_band(cst_op1, l, u)
        bandedT = session.run(cst_op1T)

        # compare
        actual = to_dense(bandedT, u, l)
        np.testing.assert_almost_equal(actual=actual, desired=dense.T, decimal=10)


@pytest.mark.parametrize("bands", BANDWIDTHS)
def test_transpose2(bands):
    """
    Test the gradient of the banded transpose operator.
    """
    n = 10
    l, u = bands

    with tf.compat.v1.Session(graph=tf.Graph()) as session:
        # generate band of matrices and dense representation
        banded = generate_band_mat(n, l, u)
        dense = to_dense(banded, l, u)

        # evaluate band transpose
        cst_op1 = constant_op(banded)
        cst_op1T = transpose_band(cst_op1, l, u)

        # evaluate dense transpose
        cst_dense_op1 = constant_op(dense)
        cst_dense_t = tf.transpose(a=cst_dense_op1)

        # Gradients
        grad_ys = np.random.rand(l + 1 + u, n)
        dense_grad_ys = construct_banded_matrix_from_band(u, l, grad_ys)

        banded_grad = tf.gradients(ys=cst_op1T, xs=cst_op1, grad_ys=grad_ys)
        dense_grad = tf.gradients(ys=cst_dense_t, xs=cst_dense_op1, grad_ys=dense_grad_ys)

        # compare
        actual = to_dense(session.run(banded_grad)[0], l, u)
        desired = session.run(dense_grad)[0]

        np.testing.assert_almost_equal(actual=actual, desired=desired, decimal=10)


@pytest.mark.parametrize("bands", BANDWIDTHS)
def test_transpose_twice(bands):
    """
    Transposing twices should give identity including for gradients.
    """
    n = 6
    l, u = bands

    with tf.compat.v1.Session(graph=tf.Graph()) as session:
        # generate band of matrices and dense representation
        banded = generate_band_mat(n, l, u)

        # evaluate band transpose
        cst_banded = constant_op(banded)
        cst_double_transpose = transpose_band(transpose_band(cst_banded, l, u), u, l)

        # Forward evaluation should give cst_banded
        np.testing.assert_almost_equal(
            actual=session.run(cst_double_transpose), desired=banded, decimal=10
        )

        # Gradient evalyation should give grad_ys
        grad_ys = extract_band_from_matrix(l, u, np.ones((n, n)))
        banded_grad = tf.gradients(ys=cst_double_transpose, xs=cst_banded, grad_ys=grad_ys)
        np.testing.assert_almost_equal(
            actual=session.run(banded_grad)[0], desired=grad_ys, decimal=10
        )
