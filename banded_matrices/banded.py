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
Main module for banded TensorFlow operators.

MATRIX REPRESENTATION:

Banded matrices are here represented using a dense Tensor that only keeps the
diagonals in the band, i.e. a NxN matrix with bandwidth K is stored as a
KxN dense Tensor. (The original dense matrix is always assumed to be square.)

The element at position (row, col) in the original, dense matrix, is found at
position (row - col + U, col) in the banded storage, where U is the upper
bandwidth of the matrix. (Note that K = L + 1 + U where L is the lower
bandwidth, and K also accounts for the diagonal.)
See [this Band Matrix drawing](https://en.wikipedia.org/wiki/Band_matrix).

Accessing elements outside of the band is invalid. The unused values in the
top-left and bottom-right corners of the banded storage should by convention be
set to 0.

Some operators additionally assume that the matrix is lower- or upper-triangular
(U == 0, or L == 0). This happens in particular when representing symmetric
matrices, for which we only store the lower-triangular half.

OPERATOR INTERFACE:

Many of the operators have an interface with bool or int parameters in addition
to the Tensor arguments themselves:

- When a Tensor is lower-triangular, we know all its bandwidth characteristics
  purely from its shape KxN: the Tensor has lower-bandwidth K-1
  and upper-bandwidth 0.

  When a Tensor represents an arbitrary banded matrix, we need to explicit pass
  lower_bandwidth and upper_bandwidth integer parameters to the operator, using
  TensorFlow's attribute mechanism.

- Some operators are compiled in multiple forms that give special treatment
  to some of their parameters. This is triggered by some Boolean flags:

  * A 'transpose' Boolean flag indicates that the operator should transpose
    one of its parameters. The operator will use (without explicitly
    constructing it) A^T instead of the banded matrix A that is actually passed.

  * A 'symmetrise' Boolean flag indicates that a parameter is
    lower-triangular matrix actually representing a symmetric matrix. The
    operator will use (without explicitly constructing it) the matrix
    A + A^T - diag(A) instead of the matrix A that is actually passed.

  Usually each int/Boolean argument has a longer name such as 'transpose_left'
  which indicates which of the Tensor arguments it refers to (the 'left'
  argument of, say, a product).

  The rationale for these flags is that many gradients need products, or
  other operations, with transposes and occasional symmetrizations. It's often
  simpler, and more efficient, to generate a variant of C++ code that deals with
  the arg directly rather than making repeated use of the transpose_band
  operator, and adding a symmetrise_band operator.

BROADCASTING:

Some operators support a simple form of
[broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html):

For most unary operators (with a single argument), when a tensor of rank > 2 is used,
its two trailing dimensions correspond to banded matrices, i.e. the input is seen as a "stack" of
banded matrices, the operator applies to each of the stacked inputs and what is returned is a
stack of the results.

For some binary operators (with two arguments), broadcasting support is as follows:
|------------------------|------------------------|------------------------------------------------|
| LHS (left-hand side)   | RHS (right-hand side)  | RESULT / COMMENT                               |
|------------------------|------------------------|------------------------------------------------|
| K1 x N                 | K2 x N                 | Normal, non-broadcasting operator call         |
|------------------------|------------------------|------------------------------------------------|
| L x K1 x N             | K2 x N                 | Stacked results of applying the op to each of  |
|                        |                        | the stacked LHS with the single RHS            |
|------------------------|------------------------|------------------------------------------------|
| L1 x ... x Lp x K1 x N | K2 x N                 | Note that arbitrary levels of stacking are     |
|                        |                        | allowed                                        |
|------------------------|------------------------|------------------------------------------------|
| K1 x N                 | L x K2 x N             | Stack of results of applying the op to the     |
|                        |                        | single LHS with each of the RHS                |
|------------------------|------------------------|------------------------------------------------|
| K1 x N                 | L1 x ... x Lp x K2 x N | Arbitrary levels of stacking are also allowed  |
|                        |                        | on the RHS.                                    |
|------------------------|------------------------|------------------------------------------------|
| L x K1 x N             | L x K2 x N             | Stacked result of applying each of op to each  |
|                        |                        | LHS with each matching RHS                     |
|------------------------|------------------------|------------------------------------------------|
| L1 x ... x Lp x K1 x N | L1 x ... x Lp x K2 x N | Several matching levels of stacking are        |
|                        |                        | allowed                                        |
|------------------------|------------------------|------------------------------------------------|
We here assumed a right-hand side that is itself a banded matrix - there are some variants where
the behaviour is naturally generalised from above.

NOTE that there are some forms of broadcasting supported by e.g. numpy that are NOT SUPPORTED
in our current implementation - these are the versions where a 1 in one part of a high-dimensional
shape is expanded to match whatever is on the other side:
|------------------------|------------------------|------------------------------------------------|
| LHS (left-hand side)   | RHS (right-hand side)  | RESULT / COMMENT                               |
|------------------------|------------------------|------------------------------------------------|
| (1 or :) x K1 x N      | L x K2 x N             | NOT SUPPORTED                                  |
|------------------------|------------------------|------------------------------------------------|

NOTE:

For some reason for some operators e.g. Transpose we can't access the
operator if it is simply named "TransposeOp", we need a more complex name with
TransposeBand. For this reason for Cholesky, Transpose, Inverse the operator
class is suffixed with "Band" while files, functions and tests have no suffix.
"""

from functools import wraps
from inspect import signature
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import (  # pylint: disable=locally-disabled, no-name-in-module
    ops,
)

from banded_matrices.library import banded_ops
from banded_matrices.types import (
    BandedMatrixTensor,
    DenseMatrixTensor,
    LowerTriangularBandedMatrixTensor,
    TriangularBandedMatrixTensor,
    VectorTensor,
)

# BROADCASTING


def broadcast_unary_operator(operator):
    """
    Decorator that allows a unary operator - with a single argument that is assumed to be a 2D
    matrix - to broadcast, in the following sense:

    If the first argument of the operator has rank higher than 2, then we see it as a stack
    of matrices and apply the operation on each of the stacked matrices.
    """
    args = list(signature(operator).parameters.keys())
    arg0_name = args[0]

    @wraps(operator)
    def wrapped_operator(*args, **kwargs):
        """
        Calls to the wrapped function will enter here, allowing logic for broadcasting.
        """
        # Get the operator's left and right arguments, by position or name:
        if len(args) >= 1:
            left = args[0]
            args = args[1:]
        else:
            left = kwargs.pop(arg0_name)

        # Try to detect misuse early with messages as clear as possible:
        left_dim = len(left.shape)

        if not isinstance(left, (tf.Tensor, np.ndarray)):
            raise AssertionError("First argument should be a TF tensor or NP array.")

        if left_dim < 2:
            raise AssertionError("First argument should be a matrix.")

        # Broadcasting logic:
        def recurse(input_submatrix):
            """
            Function called on each input sub-matrix by map_fn.
            """
            return wrapped_operator(input_submatrix, *args, **kwargs)

        if left_dim > 2:
            return tf.map_fn(recurse, left)

        else:
            return operator(left, *args, **kwargs)

    return wrapped_operator


def broadcast_binary_operator(operator):
    """
    Decorator that allows a binary operator - with two arguments that are assumed to be 2D
    matrices - to broadcast, in the following sense:

    If the first or second argument of the operator has rank higher than 2, then we see it
    as a stack of matrices and apply the operation on each of the stacked matrices.
    """
    args = list(signature(operator).parameters.keys())
    arg0_name = args[0]
    arg1_name = args[1]

    @wraps(operator)
    def wrapped_operator(*args, **kwargs):
        """
        Calls to the wrapped function will enter here, allowing logic for broadcasting.
        """
        # Get the operator's left and right arguments, by position or name:
        if len(args) >= 1:
            left = args[0]
            args = args[1:]
        else:
            left = kwargs.pop(arg0_name)

        if len(args) >= 1:
            right = args[0]
            args = args[1:]
        else:
            right = kwargs.pop(arg1_name)

        # Try to detect misuse early with messages as clear as possible:
        left_shape = left.shape
        right_shape = right.shape
        left_dim = len(left_shape)
        right_dim = len(right_shape)

        if not isinstance(left, (tf.Tensor, np.ndarray)):
            raise AssertionError("First argument should be a TF tensor or NP array.")

        if not isinstance(right, (tf.Tensor, np.ndarray)):
            raise AssertionError("Second argument should be a TF tensor or NP array.")

        if left_dim < 2:
            raise AssertionError("First argument should be a matrix.")

        if right_dim < 2:
            raise AssertionError("Second argument should be a matrix.")

        # Broadcasting logic:
        def recurse_jointly(submatrices):
            """
            Function called on each input sub-matrix by map_fn
            when both left- and right- hand-sides stack several matrices.
            """
            return wrapped_operator(submatrices[0], submatrices[1], *args, **kwargs)

        def recurse_left(left_submatrix):
            """
            Function called on each input sub-matrix by map_fn
            when the left-hand side stacks several matrices.
            """
            return wrapped_operator(left_submatrix, right, *args, **kwargs)

        def recurse_right(right_submatrix):
            """
            Function called on each input sub-matrix by map_fn
            when the right-hand-side stacks several matrices.
            """
            return wrapped_operator(left, right_submatrix, *args, **kwargs)

        if left_dim > 2 and right_dim > 2:
            # We need to broadcast over the trailing dimensions.
            # We'll expand the shapes of both left and right to match
            # and then flatten onto the first dimension.
            # then vectorize_map over the flattened dimension,
            # and reshape back to the expected size.
            left_bcast_shape = left_shape[:-2]
            right_bcast_shape = right_shape[:-2]
            if right_dim > left_dim:
                left_bcast_shape = tf.concat(
                    [tf.ones(right_dim - left_dim, dtype=tf.int32), left_bcast_shape], axis=0
                )
            if left_dim > right_dim:
                right_bcast_shape = tf.concat(
                    [tf.ones(left_dim - right_dim, dtype=tf.int32), right_bcast_shape], axis=0
                )
            shapes_match = tf.logical_or(
                left_bcast_shape == right_bcast_shape,
                tf.logical_or(right_bcast_shape == 1, left_bcast_shape == 1),
            )
            tf.debugging.assert_equal(
                shapes_match, True, message="Can't broadcast these shapes"
            )

            common_bcast_shape = tf.math.maximum(left_bcast_shape, right_bcast_shape)
            common_bcast_size = tf.math.reduce_prod(common_bcast_shape)
            left_expanded = tf.broadcast_to(
                left, tf.concat([common_bcast_shape, left_shape[-2:]], axis=0)
            )
            right_expanded = tf.broadcast_to(
                right, tf.concat([common_bcast_shape, right_shape[-2:]], axis=0)
            )

            left_flat = tf.reshape(
                left_expanded, tf.concat([[common_bcast_size], left_shape[-2:]], axis=0)
            )
            right_flat = tf.reshape(
                right_expanded, tf.concat([[common_bcast_size], right_shape[-2:]], axis=0)
            )

            # Apply the op pairwise.
            # Note that map_fn needs a dtype in this case to indicate that we want a
            # single value, not a tuple of floats:
            flat_result = tf.map_fn(recurse_jointly, (left_flat, right_flat), dtype=left.dtype)

            return tf.reshape(
                flat_result,
                tf.concat([common_bcast_shape, tf.shape(flat_result)[-2:]], axis=0),
            )

        elif left_dim > 2:
            # Left stacks a number of matrices;
            # Apply the op to each of them, each time with the same right-hand side matrix:
            return tf.map_fn(recurse_left, left)

        elif right_dim > 2:
            # Right stacks a number of matrices;
            # Apply the op to each of them, each time with the same left-hand side matrix:
            return tf.map_fn(recurse_right, right)

        else:
            return operator(left, right, *args, **kwargs)

    return wrapped_operator


def register_gradient(op_type: str):
    """
    Use this decorator for gradient registration.

    It provides a workaround for an anomaly in gradient registration where we get objects of type
    IndexedSlices instead of Tensor as the grad argument. This happens in very rare cases using
    for instance tf.gather with indices that are unknown at graph construction.

    Unfortunately while IndexedSlices are "_TensorLike", they do not at all implement all the
    Tensor API, causing some occasional and surprising issues.

    :param op_type: The string type of an operation. This corresponds to the
        `OpDef.name` field for the proto that defines the operation.
        This string is passed to the @ops.RegisterGradient decorator.
    """

    def registered_gradient_code(gradient_op):
        # pylint: disable=missing-docstring
        @ops.RegisterGradient(op_type)
        @wraps(gradient_op)
        def wrapped_operator(*args, **kwargs):
            op, grad = args

            # We used to check that the type of op was a tf.Operation here, but this doesn't
            # work in eager mode (in eager mode op is something that pretends to be a tf.Operation).

            if isinstance(grad, tf.IndexedSlices):
                grad = tf.convert_to_tensor(value=grad)

            if not isinstance(grad, (tf.Tensor, np.ndarray)):
                raise ValueError(
                    "Unexpected type of `grad` parameter for registered "
                    "gradient: {0}. Expected a TensorFlow tensor or NumPy array.".format(
                        type(grad)
                    )
                )

            return gradient_op(op, grad, **kwargs)

        return wrapped_operator

    return registered_gradient_code


# TODO (@Eric): Remove when we work out the performance hit here.
@broadcast_unary_operator
def unpack_banded_matrix_to_dense(
    matrix: BandedMatrixTensor, lower_bandwidth: int, upper_bandwidth: int
) -> tf.Tensor:
    """
    TensorFlow operator that converts a banded matrix to a dense one;
    mostly useful for debugging purposes.
    """
    return banded_ops.unpack_banded_matrix_to_dense(matrix, lower_bandwidth, upper_bandwidth)


def pack_dense_matrix_to_banded(
    dense_matrix: tf.Tensor, lower_bandwidth: int, upper_bandwidth: int
) -> BandedMatrixTensor:
    """
    TensorFlow operator that converts a dense matrix to a banded one;
    mostly useful for debugging purposes.
    """
    return banded_ops.pack_dense_matrix_to_banded(
        dense_matrix, lower_bandwidth, upper_bandwidth
    )


@register_gradient("PackDenseMatrixToBanded")
def _grad_dense_to_band(op: tf.Operation, grad: BandedMatrixTensor):
    """
    Gradient associated to the ``dense_to_band`` operator.
    """
    return unpack_banded_matrix_to_dense(
        grad,
        lower_bandwidth=op.get_attr("lower_bandwidth"),
        upper_bandwidth=op.get_attr("upper_bandwidth"),
    )


@register_gradient("UnpackBandedMatrixToDense")
def _grad_band_to_dense(op: tf.Operation, grad: tf.Tensor):
    """
    Gradient associated to the ``band_to_dense`` operator.
    """
    return pack_dense_matrix_to_banded(
        grad,
        lower_bandwidth=op.get_attr("lower_bandwidth"),
        upper_bandwidth=op.get_attr("upper_bandwidth"),
    )


def _get_effective_bandwidth(
    lo: int, hi: int, transpose: bool, symmetrise: bool
) -> Tuple[int, int]:
    """
    Given that a matrix has bandwidth (lo, hi) but may be treated
    transposed or symmetrised, return the dimension of the matrix effectively used.
    """
    if transpose:
        return hi, lo
    elif symmetrise:
        return max(lo, hi), max(lo, hi)
    else:
        return lo, hi


def _check_symmetrise_flags(symmetrise: bool, transpose: bool, upper_bandwidth: int):
    """
    Check that the symmetrise flag does not clash with other flags.
    """
    if symmetrise:
        if transpose:
            raise RuntimeError("Having a term both transposed and symmetrised is not allowed")
        if upper_bandwidth > 0:
            raise RuntimeError("Symmetrization assumes lower-triangular matrices")


def transpose_band(
    matrix: BandedMatrixTensor, input_lower_bandwidth: int, input_upper_bandwidth: int
) -> BandedMatrixTensor:
    """
    TensorFlow operator for transposing a banded matrix.
    """
    return banded_ops.transpose_band(matrix, input_lower_bandwidth, input_upper_bandwidth)


@register_gradient("TransposeBand")
def _grad_transpose_band(op: tf.Operation, grad: BandedMatrixTensor) -> BandedMatrixTensor:
    """
    Gradient associated with the ``transpose_band`` operator.
    """
    return transpose_band(
        grad, op.get_attr("input_upper_bandwidth"), op.get_attr("input_lower_bandwidth")
    )


def cholesky_band(
    matrix: LowerTriangularBandedMatrixTensor,
    should_check_result: bool = True,
    relative_tolerance: float = 1e-05,
    absolute_tolerance: float = 1e-08,
) -> LowerTriangularBandedMatrixTensor:
    """
    TensorFlow operator for the Cholesky decomposition of a banded matrix.
    :param matrix: the input matrix that needs to be decomposed.
        It must be a lower-triangular half of a symmetric banded matrix.
    :param should_check_result: Whether to check if the Cholesky decomposition
        results in a lower triangular matrix L that can reconstruct
        the original input. That is, LLᵀ = matrix.
        This check will compare all entries in LLᵀ to corresponding entries in
        the input matrix to see if they are close enough.
        To decide if two matrix entries are close enough, use the same semantics as
        [numpy.allclose](https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html).
        numpy.allclose uses the following predicate to
        decide if a new value is close enough to an actual value,
        where || stands for the absolute function:

        |new - actual| <= absolute_tolerance + relative_tolerance * |actual|

        When the predicate evaluates to True, new and actual are considered
        close enough, otherwise, not close enough.

    :param relative_tolerance: See above link for detail.
    :param absolute_tolerance: See above link for detail.
    :return: Lower-triangular banded matrix L from Cholesky decomposition.
    """
    return banded_ops.cholesky_band(
        matrix, should_check_result, relative_tolerance, absolute_tolerance
    )


@register_gradient("CholeskyBand")
def _grad_cholesky(
    op: tf.Operation, grad: LowerTriangularBandedMatrixTensor
) -> LowerTriangularBandedMatrixTensor:
    """
    Gradient associated to the ``cholesky_band`` operator.
    """
    return banded_ops.cholesky_band_grad(grad, op.outputs[0])


@broadcast_binary_operator
def product_band_mat(
    banded_matrix: BandedMatrixTensor,
    vectors: DenseMatrixTensor,
    left_lower_bandwidth: int,
    left_upper_bandwidth: int,
    transpose_left: bool = False,
    symmetrise_left: bool = False,
) -> DenseMatrixTensor:
    """
    Product of a banded matrix by a vector, or group of vectors put together
    into a non-banded matrix.
    """
    _check_symmetrise_flags(symmetrise_left, transpose_left, left_upper_bandwidth)

    return banded_ops.product_band_mat(
        banded_matrix,
        vectors,
        left_lower_bandwidth,
        left_upper_bandwidth,
        transpose_left,
        symmetrise_left,
    )


@register_gradient("ProductBandMat")
def _grad_product_band_mat(op: tf.Operation, grad: DenseMatrixTensor) -> List[tf.Tensor]:
    """
    Gradients associated to the ``product_band_mat`` operator.
    """
    if op.get_attr("symmetrise_left"):
        raise ValueError("Gradient not supported for symmetric arguments")

    left = op.inputs[0]  # type: tf.Tensor
    right = op.inputs[1]  # type: tf.Tensor

    if op.get_attr("transpose_left"):
        left_grad = outer_mat_mat(
            right,
            grad,
            result_lower_bandwidth=op.get_attr("left_lower_bandwidth"),
            result_upper_bandwidth=op.get_attr("left_upper_bandwidth"),
        )
    else:
        left_grad = outer_mat_mat(
            grad,
            right,
            result_lower_bandwidth=op.get_attr("left_lower_bandwidth"),
            result_upper_bandwidth=op.get_attr("left_upper_bandwidth"),
        )

    right_grad = product_band_mat(
        left,
        grad,
        transpose_left=not op.get_attr("transpose_left"),
        left_lower_bandwidth=op.get_attr("left_lower_bandwidth"),
        left_upper_bandwidth=op.get_attr("left_upper_bandwidth"),
    )

    return [left_grad, right_grad]


def outer_vec_vec(
    left: VectorTensor,
    right: VectorTensor,
    result_lower_bandwidth: int,
    result_upper_bandwidth: int = 0,
) -> BandedMatrixTensor:
    """
    TensorFlow operator for the outer product of two vectors, m.v^T.

    In case where the same term is passed left and right, i.e. m.m^T, the
    result is symmetric and we'll typically want result_upper_bandwidth=0.
    """
    return banded_ops.outer_vec_vec(
        left, right, result_lower_bandwidth, result_upper_bandwidth
    )


@broadcast_binary_operator
def outer_mat_mat(
    left: DenseMatrixTensor,
    right: DenseMatrixTensor,
    result_lower_bandwidth: int,
    result_upper_bandwidth: int = 0,
) -> BandedMatrixTensor:
    """
    TensorFlow operator for a product M.N^T between two non-banded matrices.
    Usually Both M and N are very "thin" matrices of shape (N, k) with k << N,
    and we are interested only in a band of the result.

    NOTE we don't have a gradient here except for the case corresponding to
    ``outer_vec_vec``.
    """
    if left.shape[1] == 1 and right.shape[1] == 1:
        return outer_vec_vec(left, right, result_lower_bandwidth, result_upper_bandwidth)
    else:
        return banded_ops.outer_mat_mat(
            left, right, result_lower_bandwidth, result_upper_bandwidth
        )


def _grad_outer(op: tf.Operation, grad: BandedMatrixTensor) -> List[tf.Tensor]:
    """
    Utility for gradients of outer products.
    """
    left = op.inputs[0]  # type: tf.Tensor
    right = op.inputs[1]  # type: tf.Tensor

    grad_left = product_band_mat(
        grad,
        right,
        left_lower_bandwidth=op.get_attr("result_lower_bandwidth"),
        left_upper_bandwidth=op.get_attr("result_upper_bandwidth"),
    )

    grad_right = product_band_mat(
        grad,
        left,
        transpose_left=True,
        left_lower_bandwidth=op.get_attr("result_lower_bandwidth"),
        left_upper_bandwidth=op.get_attr("result_upper_bandwidth"),
    )

    return [grad_left, grad_right]


@register_gradient("OuterVecVec")
def _grad_outer_vec_vec(op: tf.Operation, grad: BandedMatrixTensor) -> List[tf.Tensor]:
    """
    Gradient associated to the ``outer_vec_vec`` operator.
    """
    return _grad_outer(op, grad)


@register_gradient("OuterMatMat")
def _grad_outer_mat_mat(op: tf.Operation, grad: BandedMatrixTensor) -> List[tf.Tensor]:
    """
    Gradient associated to the ``outer_mat_mat`` operator.
    """
    return _grad_outer(op, grad)


def _get_full_product_dimension(
    left_lower_bandwidth: int,
    left_upper_bandwidth: int,
    right_lower_bandwidth: int,
    right_upper_bandwidth: int,
    transpose_left: bool,
    transpose_right: bool,
    symmetrise_left: bool,
    symmetrise_right: bool,
) -> Tuple[int, int]:
    """
    Get the default dimensions of a product between two banded matrices,
    without cropping.
    """
    _check_symmetrise_flags(symmetrise_left, transpose_left, left_upper_bandwidth)
    _check_symmetrise_flags(symmetrise_right, transpose_right, right_upper_bandwidth)

    # We occasionally get some Dimensions from TensorFlow requiring a cast
    # to int to avoid some issues
    if not all(
        isinstance(x, int)
        for x in [
            left_lower_bandwidth,
            left_upper_bandwidth,
            right_lower_bandwidth,
            right_upper_bandwidth,
        ]
    ):
        raise ValueError("Conversions missing")

    left_lo, left_hi = _get_effective_bandwidth(
        left_lower_bandwidth, left_upper_bandwidth, transpose_left, symmetrise_left
    )

    right_lo, right_hi = _get_effective_bandwidth(
        right_lower_bandwidth, right_upper_bandwidth, transpose_right, symmetrise_right
    )

    return left_lo + right_lo, left_hi + right_hi


@broadcast_binary_operator
def product_band_band(
    left: BandedMatrixTensor,
    right: BandedMatrixTensor,
    left_lower_bandwidth: int,
    left_upper_bandwidth: int,
    right_lower_bandwidth: int,
    right_upper_bandwidth: int,
    result_lower_bandwidth: int = None,
    result_upper_bandwidth: int = None,
    transpose_left: bool = False,
    transpose_right: bool = False,
    symmetrise_left: bool = False,
    symmetrise_right: bool = False,
) -> BandedMatrixTensor:
    """
    TensorFlow operator for the product of two banded matrices.

    Either left- or right- hand side matrices can be transposed or `symmetrized`
    (i.e. only the lower-triangular band of a symmetric matrix is effectively stored).

    Lower and upper bandwidths can optionally be provided in cases where we are only interested in
    part of the result's band.
    """
    _check_symmetrise_flags(symmetrise_left, transpose_left, left_upper_bandwidth)
    _check_symmetrise_flags(symmetrise_right, transpose_right, right_upper_bandwidth)

    if (result_lower_bandwidth is None) != (result_upper_bandwidth is None):
        raise RuntimeError(
            "Specify both, or none, of the lower and upper bandwidths for result"
        )

    if result_lower_bandwidth is None:
        result_lower_bandwidth, result_upper_bandwidth = _get_full_product_dimension(
            left_lower_bandwidth,
            left_upper_bandwidth,
            right_lower_bandwidth,
            right_upper_bandwidth,
            transpose_left,
            transpose_right,
            symmetrise_left,
            symmetrise_right,
        )

    return banded_ops.product_band_band(
        left,
        right,
        left_lower_bandwidth,
        left_upper_bandwidth,
        right_lower_bandwidth,
        right_upper_bandwidth,
        result_lower_bandwidth,
        result_upper_bandwidth,
        transpose_left,
        transpose_right,
        symmetrise_left,
        symmetrise_right,
    )


@register_gradient("ProductBandBand")
def _grad_product_band_band(
    op: tf.Operation, grad: BandedMatrixTensor
) -> List[BandedMatrixTensor]:
    """
    Gradients associated to the ``product_band_band`` operator.
    """
    left = op.inputs[0]  # type: tf.Tensor
    right = op.inputs[1]  # type: tf.Tensor

    transpose_left = op.get_attr("transpose_left")
    transpose_right = op.get_attr("transpose_right")

    # Mapping from each tensor to is pair of widths:
    bandwidth = {
        left: (op.get_attr("left_lower_bandwidth"), op.get_attr("left_upper_bandwidth")),
        right: (op.get_attr("right_lower_bandwidth"), op.get_attr("right_upper_bandwidth")),
        grad: (op.get_attr("result_lower_bandwidth"), op.get_attr("result_upper_bandwidth")),
    }

    def product(
        lhs: tf.Tensor,
        transpose_left: bool,
        rhs: tf.Tensor,
        transpose_right: bool,
        result: tf.Tensor,
    ) -> tf.Tensor:
        """
        Make a banded matrix products of two of the three terms,
        where the target should be shaped as the third term.
        """
        left_lower_bandwidth, left_upper_bandwidth = bandwidth[lhs]
        right_lower_bandwidth, right_upper_bandwidth = bandwidth[rhs]
        result_lower_bandwidth, result_upper_bandwidth = bandwidth[result]

        return product_band_band(
            lhs,
            rhs,
            transpose_left=transpose_left,
            transpose_right=transpose_right,
            left_lower_bandwidth=left_lower_bandwidth,
            left_upper_bandwidth=left_upper_bandwidth,
            right_lower_bandwidth=right_lower_bandwidth,
            right_upper_bandwidth=right_upper_bandwidth,
            result_lower_bandwidth=result_lower_bandwidth,
            result_upper_bandwidth=result_upper_bandwidth,
        )

    left_grad = (
        product(right, transpose_right, grad, True, left)
        if transpose_left
        else product(grad, False, right, not transpose_right, left)
    )

    right_grad = (
        product(grad, True, left, transpose_left, right)
        if transpose_right
        else product(left, not transpose_left, grad, False, right)
    )

    return [left_grad, right_grad]


@broadcast_binary_operator
def solve_triang_band(
    left: TriangularBandedMatrixTensor,
    right: BandedMatrixTensor,
    right_lower_bandwidth: int,
    right_upper_bandwidth: int,
    result_lower_bandwidth: int,
    result_upper_bandwidth: int,
    transpose_left=False,
    transpose_right=False,
    left_is_lower_triangular=True,
) -> BandedMatrixTensor:
    """
    TensorFlow operator for a solve operation with banded matrices.
    Tensor ``left`` must be lower-triangular or upper-triangular.
    This computes L^-1 R where:
    - L is either left or its transpose
    - R is either right or its transpose.

    In general, L^-1 * R will be dense, however we'll only compute the desired
    band of the result. In practice this requires computing a slightly larger
    band internally, and then cropping.
    """
    if left_is_lower_triangular:
        left_lower_bandwidth = int(left.shape[0]) - 1
        left_upper_bandwidth = 0
    else:
        left_lower_bandwidth = 0
        left_upper_bandwidth = int(left.shape[0]) - 1

    # If a user wants left or right to be transposed we explicitly use
    # ``transpose_band`` to allow differentiability:
    if transpose_left:
        left = transpose_band(left, left_lower_bandwidth, left_upper_bandwidth)
        left_lower_bandwidth, left_upper_bandwidth = (
            left_upper_bandwidth,
            left_lower_bandwidth,
        )

    if transpose_right:
        right = transpose_band(right, right_lower_bandwidth, right_upper_bandwidth)
        right_lower_bandwidth, right_upper_bandwidth = (
            right_upper_bandwidth,
            right_lower_bandwidth,
        )

    # Note that we call the version without transposition here:
    return _solve_triang_band(
        left,
        right,
        left_lower_bandwidth,
        left_upper_bandwidth,
        right_lower_bandwidth,
        right_upper_bandwidth,
        result_lower_bandwidth,
        result_upper_bandwidth,
    )


def _solve_triang_band(
    left: TriangularBandedMatrixTensor,
    right: BandedMatrixTensor,
    left_lower_bandwidth: int,
    left_upper_bandwidth: int,
    right_lower_bandwidth: int,
    right_upper_bandwidth: int,
    result_lower_bandwidth: int,
    result_upper_bandwidth: int,
    transpose_left=False,
    transpose_right=False,
) -> BandedMatrixTensor:
    """
    A version of solve that is non-differentiable in general,
    except when we leave the transpose_left and transpose_right parameters
    to False.
    This is only used in the internal implementation of some gradients,
    to use implicit transposition rather than augmenting the graph with
    extra copies of the input or output tensors.
    """
    if left_lower_bandwidth != 0 and left_upper_bandwidth != 0:
        raise RuntimeError("Left matrix of solve should be lower or upper triangular")

    return banded_ops.solve_triang_band(
        left,
        right,
        left_lower_bandwidth,
        left_upper_bandwidth,
        right_lower_bandwidth,
        right_upper_bandwidth,
        result_lower_bandwidth,
        result_upper_bandwidth,
        transpose_left,
        transpose_right,
    )


@register_gradient("SolveTriangBand")
def _grad_solve_triang_band(
    op: tf.Operation, grad: TriangularBandedMatrixTensor
) -> List[tf.Tensor]:
    """
    Gradients associated to the ``solve_triang_band`` operator.
    """
    L = op.inputs[0]  # type: tf.Tensor
    B = op.inputs[1]  # type: tf.Tensor

    left_lower_bandwidth = op.get_attr("left_lower_bandwidth")
    left_upper_bandwidth = op.get_attr("left_upper_bandwidth")

    right_lower_bandwidth = op.get_attr("right_lower_bandwidth")
    right_upper_bandwidth = op.get_attr("right_upper_bandwidth")

    result_lower_bandwidth = op.get_attr("result_lower_bandwidth")
    result_upper_bandwidth = op.get_attr("result_upper_bandwidth")

    # Gradients are not supported for transposed arguments,
    # transpositions should be done using the ``banded_transpose`` operator,
    # as done when calling the publicly exposed ``solve_triang_band`` function.
    assert not op.get_attr("transpose_left")
    assert not op.get_attr("transpose_right")

    assert left_lower_bandwidth + 1 + left_upper_bandwidth == L.shape[0]
    assert left_lower_bandwidth == 0 or left_upper_bandwidth == 0

    # L^-t grad
    right_grad = _solve_triang_band(
        L,
        grad,
        transpose_left=True,
        left_lower_bandwidth=left_lower_bandwidth,
        left_upper_bandwidth=left_upper_bandwidth,
        right_lower_bandwidth=result_lower_bandwidth,
        right_upper_bandwidth=result_upper_bandwidth,
        result_lower_bandwidth=right_lower_bandwidth,
        result_upper_bandwidth=right_upper_bandwidth,
    )

    # The first (inner) solve of the gradient's left-term is essentially the
    # right_grad. However, care is needed when the desired result bandwidth
    # is larger than the right bandwidth - we then need a larger solve:
    # TODO(optim) In these cases we should avoid 2 Solves and use an
    # TODO(optim) extract_band operator. Or is this a rare case not worth?
    if (
        result_lower_bandwidth > right_lower_bandwidth
        or result_upper_bandwidth > right_upper_bandwidth
    ):
        # Note the extended result bands here:
        inner_solve_lower_bandwith = max(right_lower_bandwidth, result_lower_bandwidth)
        inner_solve_upper_bandwith = max(right_upper_bandwidth, result_upper_bandwidth)
        inner_solve = _solve_triang_band(
            L,
            grad,
            transpose_left=True,
            left_lower_bandwidth=left_lower_bandwidth,
            left_upper_bandwidth=left_upper_bandwidth,
            right_lower_bandwidth=result_lower_bandwidth,
            right_upper_bandwidth=result_upper_bandwidth,
            result_lower_bandwidth=inner_solve_lower_bandwith,
            result_upper_bandwidth=inner_solve_upper_bandwith,
        )
    else:
        inner_solve = right_grad
        inner_solve_lower_bandwith = right_lower_bandwidth
        inner_solve_upper_bandwith = right_upper_bandwidth

    # B right_grad^T
    # We only need the upper part of the product:
    P_lower_bandwidth = (
        0
        if left_upper_bandwidth == 0
        else (right_lower_bandwidth + inner_solve_upper_bandwith)
    )
    P_upper_bandwidth = (
        0
        if left_lower_bandwidth == 0
        else (right_upper_bandwidth + inner_solve_lower_bandwith)
    )
    P = product_band_band(
        B,
        inner_solve,
        transpose_right=True,
        left_lower_bandwidth=right_lower_bandwidth,
        left_upper_bandwidth=right_upper_bandwidth,
        right_lower_bandwidth=inner_solve_lower_bandwith,
        right_upper_bandwidth=inner_solve_upper_bandwith,
        result_lower_bandwidth=P_lower_bandwidth,
        result_upper_bandwidth=P_upper_bandwidth,
    )

    # L^-1 P
    almost_left_grad = _solve_triang_band(
        L,
        P,
        left_lower_bandwidth=left_lower_bandwidth,
        left_upper_bandwidth=left_upper_bandwidth,
        right_lower_bandwidth=P_lower_bandwidth,
        right_upper_bandwidth=P_upper_bandwidth,
        # We only need the solve on the band of the lower-triangular left arg:
        result_lower_bandwidth=left_upper_bandwidth,
        result_upper_bandwidth=left_lower_bandwidth,
    )

    # We just need to take the transpose of the negated result,
    # which we do naively:
    left_grad = transpose_band(
        tf.negative(almost_left_grad), left_upper_bandwidth, left_lower_bandwidth
    )

    return [left_grad, right_grad]


@broadcast_binary_operator
def solve_triang_mat(
    left: LowerTriangularBandedMatrixTensor, right: DenseMatrixTensor, transpose_left=False
) -> DenseMatrixTensor:
    """
    TensorFlow operator for a solve operation, i.e.  L^-1 * R,
    where L is either left or its transpose.

    Left is a banded matrix;
    Right is a non-banded matrix representing a single vectors to solve.
    """
    return banded_ops.solve_triang_mat(left, right, transpose_left)


@register_gradient("SolveTriangMat")
def _grad_solve_triang_mat(op: tf.Operation, grad: DenseMatrixTensor) -> List[tf.Tensor]:
    """
    Gradients associated to the ``solve_triang_mat`` operator.
    """
    L = op.inputs[0]  # type: tf.Tensor
    transpose_left = op.get_attr("transpose_left")

    # Left is lower-triangular
    left_lower_bandwidth = int(L.shape[0]) - 1

    # L^-t grad (or L^-1 grad)
    right_grad = solve_triang_mat(L, grad, transpose_left=not transpose_left)

    # L^-1 v (or L^-T v)
    solve_left = op.outputs[0]  # forward solve_triang_mat(L, v)

    # B right_grad^T (or B right_grad)
    if not transpose_left:
        left_grad = tf.negative(
            outer_mat_mat(
                right_grad,
                solve_left,
                result_lower_bandwidth=left_lower_bandwidth,
                result_upper_bandwidth=0,
            )
        )
    else:
        left_grad = tf.negative(
            outer_mat_mat(
                solve_left,
                right_grad,
                result_lower_bandwidth=left_lower_bandwidth,
                result_upper_bandwidth=0,
            )
        )

    return [left_grad, right_grad]


def inverse_from_cholesky_band(
    lower_band: LowerTriangularBandedMatrixTensor, result_lower_bandwidth: Optional[int] = None
) -> LowerTriangularBandedMatrixTensor:
    """
    Given a lower-banded matrix L that is assumed to be the Cholesky
    decomposition of a (symmetric, Positive Definite) matrix Q = LL^T,
    Compute the inverse of Q.
    Only the lower band of this symmetric matrix is returned.
    """
    input_lower_bandwidth = lower_band.shape[-2] - 1
    if result_lower_bandwidth is None:
        result_lower_bandwidth = input_lower_bandwidth

    # The C++ operator assumes for simplicity a desired result bandwidth at least equal to the
    # input's as this is anyway needed for the computation. If needed however we can truncate the
    # result:
    if result_lower_bandwidth < input_lower_bandwidth:
        result = banded_ops.inverse_from_cholesky_band(lower_band, input_lower_bandwidth)
        return result[..., : result_lower_bandwidth + 1, :]
    else:
        return banded_ops.inverse_from_cholesky_band(lower_band, result_lower_bandwidth)


@register_gradient("InverseFromCholeskyBand")
def _grad_inverse_from_cholesky_band(
    op: tf.Operation, grad: LowerTriangularBandedMatrixTensor
) -> LowerTriangularBandedMatrixTensor:
    """
    Gradients associated with the ``inverse_from_cholesky_band`` operator.
    """
    # Note that op is here the forward OP, ``InverseFromCholeskyBand``:
    L = op.inputs[0]
    S = op.outputs[0]
    return banded_ops.gradient_of_inverse_from_cholesky_band(L, S, grad)


# Conversion between banded and block-banded representations:
#
# The two operators below allow to convert banded matrices to and from
# a block representation.
#
# Initial dense representation of a banded matrix:
# _________________
# |\  |   |   |   |
# | A |B.T|   |   |
# |__\|___|___|___|
# |   |\  |   |   |
# | B | C |D.T|   |
# |___|__\|___|___|
# |   |   |\  |   |
# |   | D | E |F.T|
# |___|___|__\|___|
# |   |   |   |\  |
# |   |   | F | G |
# |___|___|___|__\|
#
# The block band representation is:
# _________________
# |   |   |   |   |
# | A | C | E | G |
# |__ |___|___|___|
# |   |   |   |   |
# | B | D | F | 0 |
# |___|__ |___|___|
#
# The actual band representation is:
#
# |A /|C /|E /|G /|
# | / | / | / | / |
# |/B |/D_|/F_|/0_|
# |  /|  /|  /|  /|
# | /0| /0| /0| /0|
# |/__|/_ |/__|/__|


def block_to_band(
    matrix: tf.Tensor, block_size: int, symmetric: bool = True
) -> BandedMatrixTensor:
    """
    Tensorflow operator to change banded representation
    from banded to block-banded
    """
    return banded_ops.block_to_band(matrix, block_size, symmetric=symmetric, gradient=False)


def band_to_block(
    matrix: BandedMatrixTensor, block_size: int, symmetric: bool = True
) -> tf.Tensor:
    """
    Tensorflow operator to change banded representation
    from block banded to banded
    """
    return banded_ops.band_to_block(matrix, block_size, symmetric=symmetric, gradient=False)


@register_gradient("BandToBlock")
def _grad_band_to_block(op, grad):
    """
    Gradient associated to the ``band_to_block`` operator.
    """
    grad_band = banded_ops.block_to_band(
        grad, op.get_attr("block_size"), symmetric=op.get_attr("symmetric"), gradient=True
    )
    return grad_band


@register_gradient("BlockToBand")
def _grad_block_to_band(op, grad):
    """
    Gradient associated to the ``block_to_band`` operator.
    """
    grad_block = banded_ops.band_to_block(
        grad, op.get_attr("block_size"), symmetric=op.get_attr("symmetric"), gradient=True
    )
    return grad_block


def symmetrise_band(
    matrix: BandedMatrixTensor, input_lower_bandwidth: int
) -> BandedMatrixTensor:
    """
    Tensorflow operator to build a symmetric band from its lower half.
    """
    return banded_ops.symmetrise_band(matrix, input_lower_bandwidth)


# TODO : add test before declaring gradients
# @register_gradient("SymmetriseBand")
def _grad_symmetrise_band(op, grad):
    """
    Gradient associated to the ``symmetrise_band`` operator.
    """
    grad_band = halve_band(grad, op.get_attr("input_lower_bandwidth"))
    return grad_band


def halve_band(matrix: BandedMatrixTensor, input_lower_bandwidth: int) -> BandedMatrixTensor:
    """
    Tensorflow operator to extract the lower part of a symmetric band.

    This operator is meant for debugging purposes.
    """
    return banded_ops.halve_band(matrix, input_lower_bandwidth)


# TODO : add test before declaring gradients
# @register_gradient("HalveBand")
def _grad_halve_band(op, grad):
    """
    Gradient associated to the ``symmetrise_band`` operator.
    """
    grad_band = symmetrise_band(grad, op.get_attr("input_lower_bandwidth"))
    return grad_band


def chol_solve_band_mat(
    L: LowerTriangularBandedMatrixTensor, v: DenseMatrixTensor
) -> DenseMatrixTensor:
    """
    For L such that LL^T = Q and a vector v,
    computes Q^-1 v = L^-T L^-1 v
    """
    return solve_triang_mat(L, solve_triang_mat(L, v), transpose_left=True)


# TODO (@Eric): Remove if we proceed with Binary Operator Broadcasting in C++
@broadcast_unary_operator
def square_band(
    matrix: BandedMatrixTensor, lower_bandwidth: int, upper_bandwidth: int
) -> LowerTriangularBandedMatrixTensor:
    """
    Tensorflow operator that computes the square of a banded matrix.
    """
    return banded_ops.square_band(matrix, lower_bandwidth, upper_bandwidth)


@register_gradient("SquareBand")
def _grad_square_band(op, grad):
    """
    Gradient associated to the ``square_band`` operator.
    forward : L -> S = LL^T
    reverse mode diff : \bar{S} -> (\bar{S} + \bar{S}^T ) L
    """
    l, u = op.get_attr("lower_bandwidth"), op.get_attr("upper_bandwidth")
    matrix = op.inputs[0]  # type: tf.Tensor

    if l == 0 or u == 0:
        # special (faster) case when input is lower / upper triangular
        mask = 1 * np.ones((l + u + 1,))
        mask[0] = 2.0
        return banded_ops.product_band_band(
            mask[..., None] * grad,
            matrix,
            left_lower_bandwidth=l + u,
            left_upper_bandwidth=0,
            right_lower_bandwidth=l,
            right_upper_bandwidth=u,
            result_lower_bandwidth=l,
            result_upper_bandwidth=u,
            symmetrise_left=True,
            transpose_left=False,
            transpose_right=False,
            symmetrise_right=False,
        )

    else:
        grad1 = product_band_band(
            grad,
            matrix,
            left_lower_bandwidth=l + u,
            left_upper_bandwidth=0,
            right_lower_bandwidth=l,
            right_upper_bandwidth=u,
            result_lower_bandwidth=l,
            result_upper_bandwidth=u,
        )
        grad2 = product_band_band(
            grad,
            matrix,
            left_lower_bandwidth=l + u,
            left_upper_bandwidth=0,
            right_lower_bandwidth=l,
            right_upper_bandwidth=u,
            result_lower_bandwidth=l,
            result_upper_bandwidth=u,
            transpose_left=True,
        )

        return grad1 + grad2


# TO DO (@Eric): Remove & convert if we proceed with Binary Operator Broadcasting in C++
@broadcast_unary_operator
def square_mat(
    matrix: DenseMatrixTensor, result_lower_bandwidth: int
) -> LowerTriangularBandedMatrixTensor:
    """
    TensorFlow operator the computes the square MM^t of a non-banded
    matrix M.
    """
    return banded_ops.square_mat(matrix, result_lower_bandwidth)


@register_gradient("SquareMat")
def _grad_square_mat(op: tf.Operation, grad: DenseMatrixTensor) -> tf.Tensor:
    """
    Gradient associated with the ``square_band`` operator.
    """
    v = op.inputs[0]
    assert grad.shape[0] == op.get_attr("result_lower_bandwidth") + 1

    grad_left = product_band_mat(
        grad,
        v,
        left_lower_bandwidth=op.get_attr("result_lower_bandwidth"),
        left_upper_bandwidth=0,
    )

    grad_right = product_band_mat(
        grad,
        v,
        transpose_left=True,
        left_lower_bandwidth=op.get_attr("result_lower_bandwidth"),
        left_upper_bandwidth=0,
    )

    return grad_left + grad_right


def reverse_inverse_from_cholesky_band(
    matrix: BandedMatrixTensor, bandwidth: int
) -> LowerTriangularBandedMatrixTensor:
    """
    Find cholesky of subset inverse S = (LLᵀ)⁻¹.
    """
    return banded_ops.reverse_inverse_from_cholesky_band(matrix, bandwidth=bandwidth)


@register_gradient("ReverseInverseFromCholeskyBand")
def _reverse_inverse_from_cholesky_band_grad(
    op: tf.Operation, grad: LowerTriangularBandedMatrixTensor
) -> BandedMatrixTensor:
    """
    Gradient of cholesky operation on subset inverse.
    """
    bandwidth = op.get_attr("bandwidth")
    output_grad = banded_ops.reverse_inverse_from_cholesky_band_grad(
        op.inputs[0], op.outputs[0], grad, bandwidth=bandwidth
    )
    return output_grad
