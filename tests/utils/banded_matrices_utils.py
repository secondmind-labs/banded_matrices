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
import timeit

import numpy as np
import tensorflow as tf


class Timer:
    """
    A context manager that times what is running within its context.
    """

    def __init__(self):
        self.elapsed_time = None
        self.start_time = None

    def __enter__(self):
        self.elapsed_time = None
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.elapsed_time = timeit.default_timer() - self.start_time


def constant_op(t: np.ndarray) -> tf.Tensor:
    """
    Wrapper around tensorflow.python.framework which confuses pylint/mypy.
    """
    return tf.constant(t)


def generate_band_mat(n, l: int, u: int) -> np.ndarray:
    """
    Constructs the band  as a ( l + u + 1 x n ) array
    """
    return construct_extract_banded_matrix(l, u, np.random.rand(l + u + 1, n))


def to_dense(band: np.ndarray, l: int, u: int) -> np.ndarray:
    """
    Constructs the full ( n x n ) matrix from the band
    """
    return construct_banded_matrix_from_band(l, u, band)


def extract_band(dense_matrix: np.ndarray, l: int, u: int) -> np.ndarray:
    """
    Extract the band of a full matrix into a rectangular array
    """
    return extract_band_from_matrix(l, u, dense_matrix)


def gen_dense_banded_lower_triangular(n: int, k: int) -> np.ndarray:
    """
    Generates a lower triangular banded matrix with k diagonals
    """
    assert k <= n
    return to_dense(generate_band_mat(n, k - 1, 0), k - 1, 0)


def compute_gradient_error(
    input_tensor: tf.Tensor, output_tensor: tf.Tensor, delta: float = 1e-3
) -> float:
    """
    Compute the finite differencing error for d(output)/d(input).
    For TensorFlow < 1.7 we need some care about the shape.
    """
    return tf.compat.v1.test.compute_gradient_error(
        input_tensor,
        [int(d) for d in input_tensor.shape],
        output_tensor,
        [int(d) for d in output_tensor.shape],
        delta=delta,
    )


def generate_banded_positive_definite_matrix(
    dimension: int, lower_bandwidth: int
) -> np.ndarray:
    """
    Generate a banded matrix that is constructed as LL^T for an underlying banded matrix L.
    We don't return L since usually we are not able to recover exactly that decomposition.

    NOTE: Only the lower half of the symmetric resulting matrix is returned;
    so the resulting matrix has shape (lower_bandwidth + 1, dimension).
    """
    # Generate a lower band with positive diagonal
    L = generate_band_mat(dimension, lower_bandwidth, 0) + 1
    L[0, :] = np.abs(L[0, :])
    L_dense = to_dense(L, lower_bandwidth, 0)

    # Compute the Q that L is a Cholesky of, and make it banded with the same bandwidth:
    Q = extract_band(L_dense @ L_dense.T, lower_bandwidth, 0)
    return Q


def generate_banded_tensor(shape_with_bands, ensure_positive_definite=False) -> np.ndarray:
    """
    Generalization of `generate_band_mat` to tensor dimensions possibly higher than 2;
    such tensors "stack-up" banded matrices.

    In `shape_with_bands` elements at position -3 and -2 represent the lower and upper bands,
    whereas the actual tensor shape needs a width which is their sum + 1.
    """
    assert len(shape_with_bands) > 2

    lower_band, upper_band, dimension = shape_with_bands[-3:]
    width = lower_band + 1 + upper_band
    shape = shape_with_bands[:-3] + (width, dimension)

    assert not ensure_positive_definite or upper_band == 0

    if len(shape) == 2:
        return (
            generate_band_mat(dimension, lower_band, upper_band)
            if not ensure_positive_definite
            else generate_banded_positive_definite_matrix(dimension, lower_band)
        )

    return np.stack(
        [
            generate_banded_tensor(shape_with_bands[1:], ensure_positive_definite)
            for _ in range(shape_with_bands[0])
        ]
    )


def to_dense_tensor(matrix: np.ndarray, lower_band: int, upper_band: int) -> np.ndarray:
    """
    Generalization of `to_dense` to tensor dimensions possibly higher than 2;
    such tensors "stack-up" banded matrices.
    """
    assert len(matrix.shape) >= 2
    width, dimension = matrix.shape[-2:]
    assert width == lower_band + 1 + upper_band

    if len(matrix.shape) == 2:
        return to_dense(matrix, lower_band, upper_band)

    dense_shape = matrix.shape[:-2] + (dimension, dimension)

    return np.stack(
        [to_dense_tensor(matrix[d], lower_band, upper_band) for d in range(dense_shape[0])]
    )


def construct_banded_matrix_from_band(
    num_lower_diagonals: int, num_upper_diagonals: int, rect_mat: np.ndarray
) -> np.ndarray:
    """
    Constructs a square banded matrix from a representation of the band.

    :param num_lower_diagonals: aka ``l``
    :param num_upper_diagonals: aka ``u``
    :param rect_mat: Matrix of shape (num_diagonals, size) where size is the size
        of the corresponding square banded matrix.
    """
    assert num_lower_diagonals >= 0
    assert num_upper_diagonals >= 0
    assert len(rect_mat.shape) == 2
    num_diagonals = num_lower_diagonals + 1 + num_upper_diagonals
    assert rect_mat.shape[0] == num_diagonals

    size = rect_mat.shape[1]
    full_matrix = np.zeros((size, size))

    for i in range(-num_upper_diagonals, 1 + num_lower_diagonals):
        row = num_upper_diagonals + i
        for j in range(max(0, -i), max(0, size + min(0, -i))):
            full_matrix[j + i, j] = rect_mat[row, j]

    return full_matrix


def extract_band_from_matrix(
    num_lower_diagonals: int, num_upper_diagonals: int, full_matrix: np.ndarray
) -> np.ndarray:
    """
    Extracts a representation of the band from a square banded matrix.

    :param num_lower_diagonals: aka ``l``
    :param num_upper_diagonals: aka ``u``
    :param full_matrix: Square banded matrix.
    """
    assert num_lower_diagonals >= 0
    assert num_upper_diagonals >= 0
    assert len(full_matrix.shape) == 2
    assert full_matrix.shape[0] == full_matrix.shape[1]

    size = full_matrix.shape[0]
    num_diagonals = num_lower_diagonals + 1 + num_upper_diagonals
    rect_mat = np.empty((num_diagonals, size))

    for i in range(-num_upper_diagonals, num_lower_diagonals + 1):
        row = num_upper_diagonals + i
        for j in range(size):
            rect_mat[row, j] = full_matrix[j + i, j] if 0 <= j + i < size else 0.0

    return rect_mat


def extract_construct_banded_matrix(
    num_lower_diagonals: int, num_upper_diagonals: int, full_matrix: np.ndarray
) -> np.ndarray:
    extracted = extract_band_from_matrix(
        num_lower_diagonals=num_lower_diagonals,
        num_upper_diagonals=num_upper_diagonals,
        full_matrix=full_matrix,
    )
    return construct_banded_matrix_from_band(
        num_lower_diagonals=num_lower_diagonals,
        num_upper_diagonals=num_upper_diagonals,
        rect_mat=extracted,
    )


def construct_extract_banded_matrix(
    num_lower_diagonals: int, num_upper_diagonals: int, rect_mat: np.ndarray
) -> np.ndarray:
    constructed = construct_banded_matrix_from_band(
        num_lower_diagonals=num_lower_diagonals,
        num_upper_diagonals=num_upper_diagonals,
        rect_mat=rect_mat,
    )
    return extract_band_from_matrix(
        num_lower_diagonals=num_lower_diagonals,
        num_upper_diagonals=num_upper_diagonals,
        full_matrix=constructed,
    )
