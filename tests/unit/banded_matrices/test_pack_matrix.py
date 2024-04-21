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

from banded_matrices.banded import pack_dense_matrix_to_banded, unpack_banded_matrix_to_dense
from tests.utils.banded_matrices_utils import (
    constant_op,
    construct_banded_matrix_from_band,
    extract_band_from_matrix,
    generate_band_mat,
    to_dense,
)

DIMENSION = 15
BANDWIDTHS = [(3, 2), (1, 5), (0, 0)]


@pytest.mark.parametrize("bandwidth", BANDWIDTHS)
def test_pack_unpack_operations(bandwidth):
    with tf.compat.v1.Session(graph=tf.Graph()) as session:

        lower_bandwidth, upper_bandwidth = bandwidth

        banded = generate_band_mat(DIMENSION, lower_bandwidth, upper_bandwidth)
        dense = to_dense(banded, lower_bandwidth, upper_bandwidth)

        banded_from_dense = session.run(
            pack_dense_matrix_to_banded(dense, lower_bandwidth, upper_bandwidth)
        )

        dense_from_banded = session.run(
            unpack_banded_matrix_to_dense(banded, lower_bandwidth, upper_bandwidth)
        )

        assert np.equal(dense_from_banded, dense).all()
        assert np.equal(banded_from_dense, banded).all()


@pytest.mark.parametrize("bandwidth", BANDWIDTHS)
def test_pack_unpack_gradients(bandwidth):
    lower_bandwidth, upper_bandwidth = bandwidth
    width = lower_bandwidth + 1 + upper_bandwidth

    banded = generate_band_mat(DIMENSION, lower_bandwidth, upper_bandwidth)
    dense = to_dense(banded, lower_bandwidth, upper_bandwidth)

    with tf.compat.v1.Session(graph=tf.Graph()):

        banded_op = constant_op(banded)
        dense_op = constant_op(dense)

        banded_from_banded = pack_dense_matrix_to_banded(
            unpack_banded_matrix_to_dense(banded_op, lower_bandwidth, upper_bandwidth),
            lower_bandwidth,
            upper_bandwidth,
        )

        dense_from_dense = unpack_banded_matrix_to_dense(
            pack_dense_matrix_to_banded(dense_op, lower_bandwidth, upper_bandwidth),
            lower_bandwidth,
            upper_bandwidth,
        )

        # Sanity check that forward composition is identity
        assert np.equal(banded_from_banded.eval(), banded).all()
        assert np.equal(dense_from_dense.eval(), dense).all()

        # Check that gradients are identity
        grad_ys = np.ones((width, DIMENSION))
        dense_grad_ys = construct_banded_matrix_from_band(
            lower_bandwidth, upper_bandwidth, grad_ys
        )
        banded_grad_ys = extract_band_from_matrix(
            lower_bandwidth, upper_bandwidth, dense_grad_ys
        )

        grad_banded = tf.gradients(
            ys=banded_from_banded, xs=[banded_op], grad_ys=banded_grad_ys
        )[0]

        grad_dense = tf.gradients(ys=dense_from_dense, xs=[dense_op], grad_ys=dense_grad_ys)[0]

        assert np.equal(grad_banded.eval(), banded_grad_ys).all()
        assert np.equal(grad_dense.eval(), dense_grad_ys).all()
