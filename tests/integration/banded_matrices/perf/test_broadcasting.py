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
import tensorflow as tf

# We stack some matrices and solve each of them against the same right-hand side:
count_stacked = 100
dimension = 500

left = np.tril(np.random.rand(count_stacked, dimension, dimension))
right = np.random.rand(dimension, 3)


def broadcast_using_map_fn():
    """
    Approach 1: broadcasting directly using map_fn:
    """
    with tf.compat.v1.Session(graph=tf.Graph()) as session:
        l, r = tf.constant(left), tf.constant(right)

        result_tensor = tf.map_fn(lambda l2d: tf.linalg.solve(l2d, r), l)
        return session.run(result_tensor)


def broadcast_using_slice():
    """
    Approach 2: separate the stacked matrices, solve each of them and stack the result:
    """
    with tf.compat.v1.Session(graph=tf.Graph()) as session:
        l = tf.constant(left)
        result = tf.stack(
            [
                tf.linalg.solve(tf.reshape(bit, (500, 500)), right)
                for bit in tf.split(l, count_stacked, axis=0)
            ]
        )

        return session.run(result)


def test_broadcasting_does_the_same_thing():
    """
    A comparison between several ways to apply the same operation on a stack of banded matrices.
    """
    start = time.time()
    result1 = broadcast_using_map_fn()
    print("Time for map_fn version ", time.time() - start)

    start = time.time()
    result2 = broadcast_using_slice()
    print("Time for slicing version ", time.time() - start)

    assert np.all(np.equal(result1, result2))


def test_broadcasting_performance_map_fn(benchmark):
    """
    A comparison between several ways to apply the same operation on a stack of banded matrices.
    """
    benchmark.pedantic(broadcast_using_map_fn)


def test_broadcasting_performance_slice(benchmark):
    """
    A comparison between several ways to apply the same operation on a stack of banded matrices.
    """
    benchmark.pedantic(broadcast_using_slice)
