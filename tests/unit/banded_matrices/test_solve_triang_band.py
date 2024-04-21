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
from numpy import dot

from banded_matrices.banded import _grad_solve_triang_band, solve_triang_band
from tests.utils.banded_matrices_utils import (
    compute_gradient_error,
    constant_op,
    construct_banded_matrix_from_band,
    extract_band_from_matrix,
    extract_construct_banded_matrix,
    generate_band_mat,
    to_dense,
)


def may_transpose(dense_matrix: np.ndarray, transpose: bool):
    if transpose:
        return dense_matrix.transpose()
    else:
        return dense_matrix


@pytest.mark.parametrize("dim", [15])
@pytest.mark.parametrize("left_bandwidth", [0, 2])
@pytest.mark.parametrize("right_lower_bandwidth", [0, 2])
@pytest.mark.parametrize("right_upper_bandwidth", [0, 3])
@pytest.mark.parametrize("result_lower_bandwidth", [0, 2, 3])
@pytest.mark.parametrize("result_upper_bandwidth", [0, 2, 3])
@pytest.mark.parametrize("transpose_left", [False, True])
@pytest.mark.parametrize("transpose_right", [False, True])
def test_forward_solve_against_tf_triangular_solve(
    dim,
    left_bandwidth,
    transpose_left,
    transpose_right,
    right_lower_bandwidth,
    right_upper_bandwidth,
    result_lower_bandwidth,
    result_upper_bandwidth,
):
    np.random.seed(5679093)

    with tf.compat.v1.Session(graph=tf.Graph()) as session:
        # constructing two banded matrices and dense representations
        banded1 = generate_band_mat(dim, left_bandwidth, 0)
        banded2 = generate_band_mat(dim, right_lower_bandwidth, right_upper_bandwidth)

        dense1 = to_dense(banded1, left_bandwidth, 0)
        dense2 = to_dense(banded2, right_lower_bandwidth, right_upper_bandwidth)

        cst_banded1 = constant_op(banded1)
        cst_banded2 = constant_op(banded2)
        # banded solve
        solve_op = solve_triang_band(
            cst_banded1,
            cst_banded2,
            right_lower_bandwidth,
            right_upper_bandwidth,
            result_lower_bandwidth,
            result_upper_bandwidth,
            transpose_left,
            transpose_right,
        )
        # dense (tf) solve
        solve_tf_op = tf.linalg.triangular_solve(
            matrix=may_transpose(dense1, transpose_left),
            rhs=may_transpose(dense2, transpose_right),
            lower=not transpose_left,
        )

        # expand banded solve to dense and crop tf solve
        solve = to_dense(session.run(solve_op), result_lower_bandwidth, result_upper_bandwidth)
        solve_tf = extract_construct_banded_matrix(
            result_lower_bandwidth, result_upper_bandwidth, session.run(solve_tf_op)
        )

        # compare
        np.testing.assert_almost_equal(actual=solve, desired=solve_tf, decimal=8)


@pytest.mark.parametrize("dim", [15])
@pytest.mark.parametrize("right_lower_bandwidth", [0, 2, 3])
@pytest.mark.parametrize("right_upper_bandwidth", [0, 2, 3])
@pytest.mark.parametrize("result_lower_bandwidth", [0, 2, 3])
@pytest.mark.parametrize("result_upper_bandwidth", [0, 1, 4])
@pytest.mark.parametrize("left_bandwidth", [0, 2])
@pytest.mark.parametrize("left_is_lower_triangular", [False, True])
@pytest.mark.parametrize("transpose_left", [False, True])
@pytest.mark.parametrize("transpose_right", [False, True])
def test_forward_solve_against_numpy_solve(
    dim,
    left_bandwidth,
    transpose_left,
    transpose_right,
    right_lower_bandwidth,
    right_upper_bandwidth,
    result_lower_bandwidth,
    result_upper_bandwidth,
    left_is_lower_triangular,
):
    """
    This is the main test for forward solve, in particular testing
    all cases of lower/upper-triangular matrix on the left, and transpositions.
    """
    np.random.seed(345679)

    if left_is_lower_triangular:
        left_lower_bandwidth = left_bandwidth
        left_upper_bandwidth = 0
    else:
        left_lower_bandwidth = 0
        left_upper_bandwidth = left_bandwidth

    with tf.compat.v1.Session(graph=tf.Graph()) as session:
        banded1 = generate_band_mat(dim, left_lower_bandwidth, left_upper_bandwidth)
        banded2 = generate_band_mat(dim, right_lower_bandwidth, right_upper_bandwidth)

        dense1 = to_dense(banded1, left_lower_bandwidth, left_upper_bandwidth)
        dense2 = to_dense(banded2, right_lower_bandwidth, right_upper_bandwidth)

        cst_banded1 = constant_op(banded1)
        cst_banded2 = constant_op(banded2)

        solve_op = solve_triang_band(
            cst_banded1,
            cst_banded2,
            right_lower_bandwidth,
            right_upper_bandwidth,
            result_lower_bandwidth,
            result_upper_bandwidth,
            transpose_left,
            transpose_right,
            left_is_lower_triangular,
        )

        solve = session.run(solve_op)
        dense_solve = to_dense(solve, result_lower_bandwidth, result_upper_bandwidth)

        dense_solve_np = np.linalg.solve(
            may_transpose(dense1, transpose_left), may_transpose(dense2, transpose_right)
        )

        dense_solve_np = extract_construct_banded_matrix(
            result_lower_bandwidth, result_upper_bandwidth, dense_solve_np
        )

        print(np.fabs(dense_solve - dense_solve_np).max())
        np.testing.assert_almost_equal(actual=dense_solve, desired=dense_solve_np, decimal=8)


@pytest.mark.parametrize("left_bandwidth", [2])
@pytest.mark.parametrize("dim", [13])
@pytest.mark.parametrize("right_lower_bandwidth", [0, 4, 5])
@pytest.mark.parametrize("right_upper_bandwidth", [0, 4, 5])
@pytest.mark.parametrize("result_lower_bandwidth", [0, 4, 5])
@pytest.mark.parametrize("result_upper_bandwidth", [0, 4, 5])
def test_rev_mod_gradient_solve_against_python_reference_code(
    dim,
    left_bandwidth,
    right_lower_bandwidth,
    right_upper_bandwidth,
    result_lower_bandwidth,
    result_upper_bandwidth,
):
    """
    Compare the C++ gradient against reference Python version
    This allows step by step debugging of intermediate terms.
    """
    np.random.seed(45967448)
    banded1 = generate_band_mat(dim, left_bandwidth, 0)
    banded2 = generate_band_mat(dim, right_lower_bandwidth, right_upper_bandwidth)

    dense1 = to_dense(banded1, left_bandwidth, 0)
    dense2 = to_dense(banded2, right_lower_bandwidth, right_upper_bandwidth)

    def reference_python_version_rev_mode_solve_gradients(L, B, bar_S):
        i1 = np.linalg.solve(L.T, bar_S)
        i2 = dot(B, i1.T)
        i3 = np.linalg.solve(L, i2)
        return extract_construct_banded_matrix(left_bandwidth, 0, -i3.T)

    with tf.compat.v1.Session(graph=tf.Graph()) as session:
        cst_banded1 = constant_op(banded1)
        cst_banded2 = constant_op(banded2)

        solve_op = solve_triang_band(
            cst_banded1,
            cst_banded2,
            right_lower_bandwidth,
            right_upper_bandwidth,
            result_lower_bandwidth,
            result_upper_bandwidth,
        )

        bar_S = np.ones((result_lower_bandwidth + 1 + result_upper_bandwidth, dim))

        # Alternatively tf.gradients(result_op, cst_op1)[0]
        # Which is different from tf.test.compute_gradient which gives Jacobians
        grad_solve_op = _grad_solve_triang_band(solve_op.op, bar_S)
        grad_solve_left = to_dense(session.run(grad_solve_op[0]), left_bandwidth, 0)

        grad_solve_left_np = reference_python_version_rev_mode_solve_gradients(
            dense1,
            dense2,
            bar_S=extract_construct_banded_matrix(
                result_lower_bandwidth, result_upper_bandwidth, np.ones((dim, dim))
            ),
        )
        print(np.fabs(grad_solve_left - grad_solve_left_np).max())
        assert np.fabs(grad_solve_left - grad_solve_left_np).max() < 1e-7


@pytest.mark.parametrize("left_lower_bandwidth", [3])
@pytest.mark.parametrize("dim", [13])
@pytest.mark.parametrize("right_lower_bandwidth", [0, 3, 5])
@pytest.mark.parametrize("right_upper_bandwidth", [0, 3, 5])
@pytest.mark.parametrize("result_lower_bandwidth", [0, 3, 5])
@pytest.mark.parametrize("result_upper_bandwidth", [0, 3, 5])
def test_algebra_for_rev_mode_gradient_of_band_solve(
    dim,
    left_lower_bandwidth,
    right_lower_bandwidth,
    right_upper_bandwidth,
    result_lower_bandwidth,
    result_upper_bandwidth,
):
    """
    This tests purely in numpy the banded versus full versions of the
    gradient's left term. This was mostly used to verify the band.
    Note however that the precision is low here for reasons that don't
    understand yet.
    """
    np.random.seed(45967448)

    # Generate L and B
    banded_lower_dense = to_dense(
        generate_band_mat(dim, left_lower_bandwidth, 0), left_lower_bandwidth, 0
    )
    general_banded_dense = to_dense(
        generate_band_mat(dim, right_lower_bandwidth, right_upper_bandwidth),
        right_lower_bandwidth,
        right_upper_bandwidth,
    )

    # Generate the ouput gradient with 1s in the right place:
    bar_S = to_dense(
        np.ones((result_lower_bandwidth + 1 + result_upper_bandwidth, dim)),
        result_lower_bandwidth,
        result_upper_bandwidth,
    )

    # Do the first solve banded, with a large, over-sized band
    i1 = extract_construct_banded_matrix(
        max(right_lower_bandwidth, result_lower_bandwidth),
        right_upper_bandwidth,
        np.linalg.solve(banded_lower_dense.T, bar_S),
    )
    i2 = dot(general_banded_dense, i1.T)
    i3 = np.linalg.solve(banded_lower_dense, i2)
    bar_L_narrow = extract_construct_banded_matrix(left_lower_bandwidth, 0, -i3.T)

    # Reference version, with solve not banded
    i1 = np.linalg.solve(banded_lower_dense.T, bar_S)
    i2 = dot(general_banded_dense, i1.T)
    i3 = np.linalg.solve(banded_lower_dense, i2)
    bar_L_proper = extract_construct_banded_matrix(left_lower_bandwidth, 0, -i3.T)

    # The errors unfortunately don't even pass at 1e-4
    error = np.fabs(bar_L_narrow - bar_L_proper).max()
    assert error < 1e-3


@pytest.mark.parametrize("dim", [13])
@pytest.mark.parametrize("left_bandwidth", [0, 3])
@pytest.mark.parametrize("right_lower_bandwidth", [0, 3])
@pytest.mark.parametrize("right_upper_bandwidth", [0, 5])
@pytest.mark.parametrize("result_lower_bandwidth", [0, 5])
@pytest.mark.parametrize("result_upper_bandwidth", [0, 5])
def test_solve_jacobian_with_finite_differencing(
    dim,
    left_bandwidth,
    right_lower_bandwidth,
    right_upper_bandwidth,
    result_lower_bandwidth,
    result_upper_bandwidth,
):
    """
    Finite differencing checks of the Jacobians are somehow useful but the
    tolerance has to be high.
    """
    np.random.seed(999567)
    banded1 = generate_band_mat(dim, left_bandwidth, 0)
    banded2 = generate_band_mat(dim, right_lower_bandwidth, right_upper_bandwidth)

    with tf.compat.v1.Session(graph=tf.Graph()):
        cst_banded1 = constant_op(banded1)
        cst_banded2 = constant_op(banded2)

        solve_op = solve_triang_band(
            cst_banded1,
            cst_banded2,
            right_lower_bandwidth,
            right_upper_bandwidth,
            result_lower_bandwidth,
            result_upper_bandwidth,
        )

        # Error for dy/dx1
        jac_err_left = compute_gradient_error(cst_banded1, solve_op, delta=1e-8)

        # Error for dy/dx2
        jac_err_right = compute_gradient_error(cst_banded2, solve_op)

        print("Solve finite diff gradient errors: ", jac_err_left, jac_err_right)
        # 1e-7 or 1e-8 is typical, but 1e-6 is occasional on left:
        assert jac_err_left < 1e-3
        # Right is more precise as it is a simpler sub-term:
        assert jac_err_right < 1e-3


@pytest.mark.parametrize("dim", [13])
@pytest.mark.parametrize("left_bandwidth", [0, 2])
@pytest.mark.parametrize("left_is_lower_triangular", [False, True])
@pytest.mark.parametrize("transpose_left", [False, True])
@pytest.mark.parametrize("transpose_right", [False, True])
@pytest.mark.parametrize("right_lower_bandwidth", [0, 3, 4])
@pytest.mark.parametrize("right_upper_bandwidth", [0, 3, 4])
@pytest.mark.parametrize("result_lower_bandwidth", [0, 3, 4])
@pytest.mark.parametrize("result_upper_bandwidth", [0, 3, 4])
def test_rev_mode_gradient_solve_against_tf_gradient(
    dim,
    left_bandwidth,
    right_lower_bandwidth,
    right_upper_bandwidth,
    result_lower_bandwidth,
    result_upper_bandwidth,
    transpose_left,
    transpose_right,
    left_is_lower_triangular,
):
    """
    Compare the gradients against those of the corresponding dense TF operator.
    This is out main gradient test - the only one that seems numerically
    stable for the left-hand side gradient in particular.
    """
    np.random.seed(3794567)
    with tf.compat.v1.Session(graph=tf.Graph()) as session:
        if left_is_lower_triangular:
            left_lower_bandwidth = left_bandwidth
            left_upper_bandwidth = 0
        else:
            left_lower_bandwidth = 0
            left_upper_bandwidth = left_bandwidth

        banded1 = generate_band_mat(dim, left_lower_bandwidth, left_upper_bandwidth)
        banded2 = generate_band_mat(dim, right_lower_bandwidth, right_upper_bandwidth)

        dense1 = to_dense(banded1, left_lower_bandwidth, left_upper_bandwidth)
        dense2 = to_dense(banded2, right_lower_bandwidth, right_upper_bandwidth)

        grad_ys = np.ones((result_lower_bandwidth + 1 + result_upper_bandwidth, dim))
        dense_grad_ys = construct_banded_matrix_from_band(
            result_lower_bandwidth, result_upper_bandwidth, grad_ys
        )
        grad_ys = extract_band_from_matrix(
            result_lower_bandwidth, result_upper_bandwidth, dense_grad_ys
        )

        # Results calculated by banded op
        cst_banded1 = constant_op(banded1)
        cst_banded2 = constant_op(banded2)

        solve_op = solve_triang_band(
            cst_banded1,
            cst_banded2,
            right_lower_bandwidth,
            right_upper_bandwidth,
            result_lower_bandwidth,
            result_upper_bandwidth,
            transpose_left=transpose_left,
            transpose_right=transpose_right,
            left_is_lower_triangular=left_is_lower_triangular,
        )

        [solve_grad_left_op, solve_grad_right_op] = tf.gradients(
            ys=solve_op, xs=[cst_banded1, cst_banded2], grad_ys=grad_ys
        )

        solve_grad_left = to_dense(
            session.run(solve_grad_left_op), left_lower_bandwidth, left_upper_bandwidth
        )
        solve_grad_right = to_dense(
            session.run(solve_grad_right_op), right_lower_bandwidth, right_upper_bandwidth
        )

        # Results obtained from a dense triangular solve
        cst_dense1 = constant_op(may_transpose(dense1, transpose_left))
        cst_dense2 = constant_op(may_transpose(dense2, transpose_right))

        solve_tf_op = tf.linalg.triangular_solve(
            matrix=cst_dense1, rhs=cst_dense2, lower=left_is_lower_triangular != transpose_left
        )

        [solve_tf_grad_left_op, solve_tf_grad_right_op] = tf.gradients(
            ys=solve_tf_op, xs=[cst_dense1, cst_dense2], grad_ys=dense_grad_ys
        )

        solve_tf_grad_left = extract_construct_banded_matrix(
            left_lower_bandwidth,
            left_upper_bandwidth,
            # We want the gradient on dense1, from the one on cst_dense1:
            may_transpose(session.run(solve_tf_grad_left_op), transpose_left),
        )
        solve_tf_grad_right = extract_construct_banded_matrix(
            right_lower_bandwidth,
            right_upper_bandwidth,
            # We want the gradient on dense2, from the one on cst_dense2:
            may_transpose(session.run(solve_tf_grad_right_op), transpose_right),
        )

        # Error checks
        grad_err_1 = np.fabs(solve_grad_left - solve_tf_grad_left).max()
        grad_err_2 = np.fabs(solve_grad_right - solve_tf_grad_right).max()

        print("Solve gradient errors w.r.t. TF dense: ", grad_err_1, grad_err_2)
        assert grad_err_1 < 1e-10
        assert grad_err_2 < 1e-10


@pytest.mark.parametrize("dim", [13])
@pytest.mark.parametrize("right_lower_bandwidth", [0, 4, 5])
@pytest.mark.parametrize("right_upper_bandwidth", [0, 4, 5])
@pytest.mark.parametrize("result_lower_bandwidth", [0, 4, 5])
@pytest.mark.parametrize("result_upper_bandwidth", [0, 4, 5])
@pytest.mark.parametrize("left_bandwidth", [0, 1, 4])
def test_algebra_reverse_mode_gradient_solve(
    dim,
    left_bandwidth,
    right_lower_bandwidth,
    right_upper_bandwidth,
    result_lower_bandwidth,
    result_upper_bandwidth,
):
    """
    Testing that the reverse mode gradient of L,B-> S(L,B)=L^-1 B are
    \bar[B]^T =  S(L.T,\bar[S])
    \bar[L]^T =  S(L, B S(L.T,\bar[S]).T )
    """
    np.random.seed(9379456)
    with tf.compat.v1.Session(graph=tf.Graph()):
        np.random.seed(3794567)
        banded1 = generate_band_mat(dim, left_bandwidth, 0)
        banded2 = generate_band_mat(dim, right_lower_bandwidth, right_upper_bandwidth)

        dense1 = to_dense(banded1, left_bandwidth, 0)
        dense2 = to_dense(banded2, right_lower_bandwidth, right_upper_bandwidth)

        cst_dense1 = constant_op(dense1)
        cst_dense2 = constant_op(dense2)

        result_dense = tf.linalg.solve(cst_dense1, cst_dense2)

        banded_bar_S = np.ones((result_lower_bandwidth + result_upper_bandwidth + 1, dim))
        bar_S = construct_banded_matrix_from_band(
            result_lower_bandwidth, result_upper_bandwidth, banded_bar_S
        )
        bar_S_tf = constant_op(bar_S)

        # reverse mode for left argument
        a1 = tf.linalg.solve(tf.transpose(a=cst_dense1), bar_S_tf)
        a2 = tf.matmul(cst_dense2, tf.transpose(a=a1))
        grad_1_algebra = extract_band_from_matrix(
            left_bandwidth, 0, -tf.transpose(a=tf.linalg.solve(cst_dense1, a2)).eval()
        )
        grad_1_dense = extract_band_from_matrix(
            left_bandwidth,
            0,
            tf.gradients(ys=result_dense, xs=cst_dense1, grad_ys=bar_S)[0].eval(),
        )

        # reverse mode for right argument
        grad_2_algebra = extract_band_from_matrix(
            right_lower_bandwidth,
            right_upper_bandwidth,
            tf.linalg.solve(tf.transpose(a=cst_dense1), bar_S_tf).eval(),
        )
        grad_2_dense = extract_band_from_matrix(
            right_lower_bandwidth,
            right_upper_bandwidth,
            tf.gradients(ys=result_dense, xs=cst_dense2, grad_ys=bar_S)[0].eval(),
        )

        np.testing.assert_almost_equal(grad_1_algebra, grad_1_dense, decimal=4)
        np.testing.assert_almost_equal(grad_2_algebra, grad_2_dense, decimal=4)
