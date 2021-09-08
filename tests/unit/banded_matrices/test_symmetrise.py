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

from banded_matrices.banded import halve_band, symmetrise_band
from tests.utils.banded_matrices_utils import (
    construct_banded_matrix_from_band,
    construct_extract_banded_matrix,
    extract_band_from_matrix,
)


@pytest.mark.parametrize("n", [10, 20])
@pytest.mark.parametrize("l", [2, 3])
def test_symmetrise_forward(n, l):
    with tf.compat.v1.Session(graph=tf.Graph()):
        # construct lower triangular
        A_lower_part_band = construct_extract_banded_matrix(l, 0, np.random.rand(l + 1, n))
        A_lower_part_dense = construct_banded_matrix_from_band(l, 0, A_lower_part_band)

        # symmetrise
        A_sym_dense = (
            A_lower_part_dense + A_lower_part_dense.T - np.diag(np.diag(A_lower_part_dense))
        )

        A_sym_band_ref = extract_band_from_matrix(l, l, A_sym_dense)

        # symmetrise from band using op
        A_sym_band_op = symmetrise_band(A_lower_part_band, l)
        A_sym_band = A_sym_band_op.eval()

        np.testing.assert_almost_equal(A_sym_band, A_sym_band_ref)


@pytest.mark.parametrize("n", [10, 20])
@pytest.mark.parametrize("l", [2, 3])
def test_halve_forward(n, l):
    with tf.compat.v1.Session(graph=tf.Graph()):
        # construct lower triangular
        A_band = np.random.rand(l + 1, n)
        A_lower_part_band_ref = construct_extract_banded_matrix(l, 0, A_band)
        A_lower_part_dense_ref = construct_banded_matrix_from_band(l, 0, A_lower_part_band_ref)

        # symmetrise
        A_sym_dense = (
            A_lower_part_dense_ref
            + A_lower_part_dense_ref.T
            - np.diag(np.diag(A_lower_part_dense_ref))
        )
        A_sym_band = extract_band_from_matrix(l, l, A_sym_dense)

        # halve using op
        A_lower_part_band_op = halve_band(A_sym_band, l)
        A_lower_part_band = A_lower_part_band_op.eval()

        np.testing.assert_almost_equal(A_lower_part_band, A_lower_part_band_ref)
