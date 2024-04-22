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

from banded_matrices.banded import square_band
from tests.utils.banded_matrices_utils import (
    extract_band_from_matrix,
    extract_construct_banded_matrix,
    generate_band_mat,
    to_dense,
)

SOME_SHAPES = [(2, 0), (3, 0), (1, 5), (0, 0), (0, 4)]


@pytest.mark.parametrize("bands", SOME_SHAPES)
@pytest.mark.parametrize("n", [8])
def test_forward_square_band(bands, n):
    for l1, u1 in [bands, reversed(bands)]:

        with tf.compat.v1.Session(graph=tf.Graph()) as session:

            banded1 = generate_band_mat(n, l1, u1)

            dense1 = to_dense(banded1, l1, u1)

            dense1_op = tf.constant(dense1)
            banded_op = tf.constant(banded1)

            square_op = square_band(banded_op, lower_bandwidth=l1, upper_bandwidth=u1)

            square_tf_op = tf.matmul(dense1_op, dense1_op, transpose_b=True)

            square = session.run(square_op)
            square_tf = extract_band_from_matrix(l1 + u1, 0, session.run(square_tf_op))

            np.testing.assert_almost_equal(actual=square, desired=square_tf, decimal=10)
            print("foward evaluation OK\n")


@pytest.mark.parametrize("bands", SOME_SHAPES)
@pytest.mark.parametrize("n", [8])
def test_gradient_square_band_against_tf(bands, n):
    for l1, u1 in [bands, reversed(bands)]:

        with tf.compat.v1.Session(graph=tf.Graph()) as session:

            banded1 = np.random.randint(1, 4, (l1 + u1 + 1, n)).astype(float)
            dense1 = to_dense(banded1, l1, u1)

            dense1_op = tf.constant(dense1)
            banded_op = tf.constant(banded1)

            # forward ops
            square_op = square_band(banded_op, lower_bandwidth=l1, upper_bandwidth=u1)

            square_tf_op = tf.matmul(dense1_op, dense1_op, transpose_b=True)

            # gradients ops
            bar_square_dense = extract_construct_banded_matrix(
                l1 + u1, l1 + u1, np.ones((n, n))
            )
            bar_square_band = extract_band_from_matrix(l1 + u1, 0, bar_square_dense)
            bar_square_band[1:, :] *= 2.0  # double the non diag entries

            grad_square_op = tf.gradients(ys=square_op, xs=banded_op, grad_ys=bar_square_band)[
                0
            ]

            grad_square_tf_op = tf.gradients(
                ys=square_tf_op, xs=dense1_op, grad_ys=bar_square_dense
            )[0]

            grad_square = session.run(grad_square_op)
            grad_square_tf = extract_band_from_matrix(l1, u1, session.run(grad_square_tf_op))

            np.testing.assert_almost_equal(
                actual=grad_square, desired=grad_square_tf, decimal=10
            )
            print("gradient evaluation OK\n")
