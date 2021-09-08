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
import tensorflow as tf

from banded_matrices.banded import solve_triang_mat
from tests.utils.banded_matrices_utils import generate_banded_tensor


def test_indexed_slices():
    """
    This tests an anomaly in gradient registration where we get objects of type IndexedSlices
    instead of Tensor as the `grad` argument. Unfortunately while IndexedSlices are "_TensorLike"
    they do not at all implement all the Tensor API, causing some occasional and surprising
    issues.

    To fix this every registered gradient needs to convert its `grad` argument to a Tensor in the
    rare event where we receive an IndexedSlices. This test checks that, and would fail,
    specifically, if, the conversion to tensor of `grad` parameters isn't done.
    """
    with tf.compat.v1.Session(graph=tf.Graph()) as session:
        # We do a solve between arbitrarily-sized matrices:
        cst_banded1 = tf.constant(generate_banded_tensor((1, 0, 15)))
        cst_banded2 = tf.constant(np.random.rand(15, 1))
        raw_op = solve_triang_mat(cst_banded1, cst_banded2)

        # The only case we managed to repro is when we do a tf.gather with a selector variable
        # that's initialized in a very specific way, leading selector.shape is None.
        # This specific way is commoly used in GPflow for DataHolder, therefore it is important to
        # support this kind of variable initialization:
        selected_indices = np.array([[1, 3, 7], [4, 6, 8]], dtype=np.int32)
        initializer = tf.compat.v1.placeholder(tf.int32, shape=None, name="initializer")
        selector = tf.compat.v1.get_variable(
            "selector", initializer=initializer, validate_shape=False, trainable=False
        )

        sliced_op = tf.gather(raw_op, selector, axis=0)
        session.run(selector.initializer, feed_dict={initializer: selected_indices})

        # Only in the gradient computation of the banded operator, here solve_triang_mat,
        # do we obtain an object of type tf.IndexedSlices
        solve_result = session.run(sliced_op)
        banded_bar_P = np.ones(solve_result.shape)

        # That goal is not here to validate the gradient for this operator, just to make sure
        # it does not anymore raise the Exception that was happening before the fixes in banded.py:
        session.run(tf.gradients(ys=sliced_op, xs=cst_banded1, grad_ys=banded_bar_P))
