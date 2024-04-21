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

from banded_matrices.banded import product_band_band
from tests.utils.banded_matrices_utils import Timer, constant_op, generate_band_mat

# On Jenkins we just use a small size by default to check this test just runs OK
# When running by hand, just change the Boolean flag below:
RUN_FULL_SIZE = False

if RUN_FULL_SIZE:
    l, u = 50, 50
    n = 100000
    np.random.seed(279)
else:
    l, u = 3, 3
    n = 10
    np.random.seed(279)


def test_perf_product():
    """
    Perf for a simple and common operator - product;
    This is one example where accelerating inner products (e.g. using SSE)
    could make a difference.
    """
    with tf.compat.v1.Session(graph=tf.Graph()) as session:

        banded1 = generate_band_mat(n, l, u)
        banded2 = generate_band_mat(n, l, u)

        cst_op1 = constant_op(banded1)
        cst_op2 = constant_op(banded2)

        product = product_band_band(
            cst_op1,
            cst_op2,
            left_lower_bandwidth=l,
            left_upper_bandwidth=u,
            right_lower_bandwidth=l,
            right_upper_bandwidth=u,
            result_lower_bandwidth=l,
            result_upper_bandwidth=u,
        )

        with Timer() as timer:
            session.run(product)

        print(
            "Time for a product between ({}, {}) matrices: "
            "{}s".format(n, l + u + 1, timer.elapsed_time)
        )
