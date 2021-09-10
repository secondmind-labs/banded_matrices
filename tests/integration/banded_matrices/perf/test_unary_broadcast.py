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
import time

import numpy as np
import pytest
import tensorflow as tf

import banded_matrices.banded as bd
from tests.utils.banded_matrices_utils import generate_banded_tensor

# We stack some matrices and solve each of them against the same right-hand side:
count_stacked = 100
dimension = 500
lower_band, upper_band = 3, 0

left = np.tril(np.random.rand(count_stacked, dimension, dimension))
right = np.random.rand(dimension, 3)
unary = np.stack(
    [
        generate_banded_tensor(
            (count_stacked, lower_band, upper_band, dimension), ensure_positive_definite=True
        )
        for _ in range(2)
    ]
)


def BAND_TO_BLOCK(band):
    return bd.band_to_block(band, block_size=lower_band + 1)


def BLOCK_TO_BAND(band):
    return bd.block_to_band(band, block_size=lower_band + 1)


def CHOLESKY(band):
    return bd.cholesky_band(band, should_check_result=False)


def PACK_DENSE_TO_BAND(dense):
    return bd.pack_dense_matrix_to_banded(dense, lower_band, upper_band)


def UNPACK_BAND_TO_DENSE(band):
    return bd.unpack_banded_matrix_to_dense(band, lower_band, upper_band)


def HALVE_BAND(band):
    return bd.symmetrise_band(band, lower_band)


def SYMMETRISE(band):
    return bd.symmetrise_band(band, lower_band)


def SQUARE_BAND(band):
    return bd.square_band(band, lower_band, upper_band)


def SQUARE_MAT(dense):
    return bd.square_mat(dense, lower_band)


def TRANSPOSE(band):
    return bd.transpose_band(band, lower_band, upper_band)


def INVERSE_CHOLESKY(band):
    return bd.inverse_from_cholesky_band(band)


def REVERSE_INVERSE_CHOLESKY(band):
    return bd.reverse_inverse_from_cholesky_band(band, bandwidth=lower_band + 1)


UNARY_BAND_OPS = [
    TRANSPOSE,
    CHOLESKY,
    BLOCK_TO_BAND,
    BAND_TO_BLOCK,
    SYMMETRISE,
    HALVE_BAND,
    UNPACK_BAND_TO_DENSE,
    INVERSE_CHOLESKY,
    REVERSE_INVERSE_CHOLESKY,
    SQUARE_BAND,
]
UNARY_DENSE_OPS = [PACK_DENSE_TO_BAND]
UNARY_OPS = UNARY_BAND_OPS + UNARY_DENSE_OPS
NO_GRADS = [SYMMETRISE, HALVE_BAND]


def _to_dense(x):
    x_shape = tf.shape(x)
    new_shape = tf.concat([x_shape[:-2], x_shape[-1:], x_shape[-1:]], axis=0)
    return tf.ones(new_shape, dtype=tf.float64)


def broadcast_unary_using_map_fn(func, do_compile, data=None):
    """
    Approach 1: broadcasting directly using map_fn:
    """
    u = tf.constant(unary) if data is None else tf.constant(data)
    if func in UNARY_DENSE_OPS:
        u = _to_dense(u)
    f = func if not do_compile else tf.function(func)

    if func in NO_GRADS:
        result_tensor = tf.map_fn(f, u)
        grad = tf.zeros(1)
    else:
        with tf.GradientTape() as tape:
            tape.watch(u)
            result_tensor = tf.map_fn(f, u)
        grad = tape.gradient(result_tensor, u)
    return [result_tensor, grad]


def broadcast_unary_using_py_broadcast(func, do_compile, data=None):
    """
    Approach 2: broadcasting directly using previous python broadcast
    """
    u = tf.constant(unary) if data is None else tf.constant(data)
    if func in UNARY_DENSE_OPS:
        u = _to_dense(u)
    func_wrapped_py = bd.broadcast_unary_operator(func)
    f = func_wrapped_py if not do_compile else tf.function(func_wrapped_py)

    if func in NO_GRADS:
        result_tensor = f(u)
        grad = tf.zeros(1)
    else:
        with tf.GradientTape() as tape:
            tape.watch(u)
            result_tensor = f(u)
        grad = tape.gradient(result_tensor, u)
    return [result_tensor, grad]


def broadcast_unary_using_native(func, do_compile, data=None):
    """
    Approach 3: broadcasting directly in C++:
    """
    u = tf.constant(unary) if data is None else tf.constant(data)
    if func in UNARY_DENSE_OPS:
        u = _to_dense(u)
    f = func if not do_compile else tf.function(func)

    if func in NO_GRADS:
        result_tensor = f(u)
        grad = tf.zeros(1)
    else:
        with tf.GradientTape() as tape:
            tape.watch(u)
            result_tensor = f(u)
        grad = tape.gradient(result_tensor, u)
    return [result_tensor, grad]


@pytest.mark.parametrize("do_compile", [False, True])
@pytest.mark.parametrize("func", UNARY_OPS)
def test_compare_results(func, do_compile):
    """
    A comparison between several ways to apply the same operation on a stack of banded matrices.
    """

    start = time.time()
    result1, grad1 = broadcast_unary_using_map_fn(func, do_compile)
    print("Time for map_fn version ", time.time() - start)

    start = time.time()
    result2, grad2 = broadcast_unary_using_native(func, do_compile)
    print("Time for native version ", time.time() - start)

    start = time.time()
    result3, grad3 = broadcast_unary_using_py_broadcast(func, do_compile)
    print("Time for py version ", time.time() - start)

    assert np.all(np.equal(result1, result2))
    assert np.all(np.equal(result2, result3))
    assert np.all(np.equal(grad1, grad2))
    assert np.all(np.equal(grad2, grad3))
