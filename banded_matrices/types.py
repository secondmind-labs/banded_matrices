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
"""
Some type declarations used by the banded matrices module.
"""
from typing import Union

import numpy as np
import tensorflow as tf

# MATRIX Types:

# All our matrices are of type tf.Tensor, and it should always be possible to pass
# numpy arrays instead. These type aliases are just here to further document
# the assumptions on the matrix:

# A tf.Tensor that represents a Banded matrix. Here the (dense) tensor should be
# of dimension KxN where K is the bandwidth of the represented NxN matrix.
BandedMatrixTensor = Union[tf.Tensor, np.ndarray]

# A ``BandedMatrixTensor`` where the matrix is, additionally, assumed to be
# lower-triangular or upper-triangular.
TriangularBandedMatrixTensor = Union[tf.Tensor, np.ndarray]

# A ``BandedMatrixTensor`` where the matrix is, additionally, assumed to be
# lower-triangular.
LowerTriangularBandedMatrixTensor = Union[tf.Tensor, np.ndarray]

# A tf.Tensor that represents a non-banded matrix. Typically this will be a NxC
# matrix that aggregates C vectors of the considered dimension N.
# The case Nx1 is frequent, corresponding to a vector of size N.
DenseMatrixTensor = Union[tf.Tensor, np.ndarray]

# A special case of DenseMatrixTensor where the shape is Nx1, representing
# a single vector.
VectorTensor = Union[tf.Tensor, np.ndarray]
