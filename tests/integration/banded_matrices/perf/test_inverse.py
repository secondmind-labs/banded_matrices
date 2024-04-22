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

from banded_matrices.banded import _grad_inverse_from_cholesky_band, inverse_from_cholesky_band
from tests.utils.banded_matrices_utils import Timer, constant_op, generate_band_mat

# On Jenkins we just use a small size by default to check this test just runs OK
# When running by hand, just change the Boolean flag below:
RUN_FULL_SIZE = False

if RUN_FULL_SIZE:
    l, u = 50, 50
    n = 20000
    np.random.seed(279)
else:
    l, u = 3, 3
    n = 10
    np.random.seed(279)


def test_perf_inv_from_chol():
    """
    Perf test for an expensive operator, inverse from Cholesky.
    This is really meant to be run by hand when we want a quick perf comparison,
    but will be run by the tests to make sure it does not break.
    """
    # The L Cholesky matrix, input of the op in forward mode
    L_band = generate_band_mat(n, l, 0)
    L_band[0, :] = np.abs(L_band[0, :])

    # Gradients of output, assumed to be 1 everywhere
    grad_ys = np.ones_like(L_band)

    with tf.compat.v1.Session(graph=tf.Graph()) as session:
        # Our implementation of the gradient:
        cst_k_band = constant_op(L_band)
        inverse_op = inverse_from_cholesky_band(cst_k_band)
        grad_L_op = _grad_inverse_from_cholesky_band(inverse_op.op, grad_ys)

        with Timer() as timer:
            session.run(grad_L_op)

    print(
        "Time for a inverse from Cholesky between ({}, {}) matrices: "
        "{}s".format(n, l + u + 1, timer.elapsed_time)
    )
