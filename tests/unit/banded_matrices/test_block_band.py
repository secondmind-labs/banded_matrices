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

from banded_matrices.banded import band_to_block, block_to_band
from tests.utils.banded_matrices_utils import constant_op

BLOCK_SIZES = [1, 5]
H_BLOCKS = [1, 3]
V_BLOCKS = [1, 2]


def block_to_band_np(Q_block, block_size):
    r"""
    Q_block is a rectangular block banded matrix representation of
    a block band matrix

    Q_block has size  (v_blocks*block_size)x(h_blocks*block_size)
    Q_band has the same size

    Initial dense representation of a banded matrix
    _________________
    |\  |   |   |   |
    | A |B.T|   |   |
    |__\|___|___|___|
    |   |\  |   |   |
    | B | C |D.T|   |
    |___|__\|___|___|
    |   |   |\  |   |
    |   | D | E |F.T|
    |___|___|__\|___|
    |   |   |   |\  |
    |   |   | F | G |
    |___|___|___|__\|

    The block band representation is
    _________________
    |   |   |   |   |
    | A | C | E | G |
    |__ |___|___|___|
    |   |   |   |   |
    | B | D | F | 0 |
    |___|__ |___|___|

    The actual band repesentation is
    _________________
    |A /|C /|E /|G /|
    | / | / | / | / |
    |/B |/D_|/F_|/0_|
    |  /|  /|  /|  /|
    | /0| /0| /0| /0|
    |/__|/_ |/__|/__|

    """
    h_blocks = int(Q_block.shape[1] / block_size)
    Q_band = np.zeros_like(Q_block)
    band_width = Q_block.shape[0]
    # looping over the band
    for t in range(h_blocks):
        for s in range(block_size):
            # move column up
            Q_band[: band_width - s, t * block_size + s] = Q_block[
                s:band_width, t * block_size + s
            ]
    return Q_band


def band_to_block_np(Q_band, block_size, symmetrise=True):
    """
    Q_band is a rectangular banded matrix representation of
    a block band matrix

    Q_band has size  (v_blocks*block_size)x(h_blocks*block_size)
    Q_block has the same size
    """
    h_blocks = int(Q_band.shape[1] / block_size)
    band_width = Q_band.shape[0]
    Q_block = np.zeros_like(Q_band)
    # looping over the band
    for t in range(h_blocks):
        for s in range(block_size):
            # move column down
            Q_block[s:band_width, t * block_size + s] = Q_band[
                : band_width - s, t * block_size + s
            ]
            if symmetrise:
                # symmetrise first block
                Q_block[s, t * block_size + s : (t + 1) * block_size] = Q_block[
                    s:block_size, t * block_size + s
                ]

    return Q_block


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("h_blocks", H_BLOCKS)
@pytest.mark.parametrize("v_blocks", V_BLOCKS)
def test_block_to_band(block_size, h_blocks, v_blocks):
    """
    Test the forward evaluation of block_to_band op against a numpy implementation
    """
    A_block = np.random.randn(block_size, block_size)
    Q_block = np.tile(A_block + A_block.T, [v_blocks, h_blocks])
    Q_band_ref = block_to_band_np(Q_block, block_size)

    with tf.compat.v1.Session(graph=tf.Graph()):
        # evaluate op output
        Q_band = block_to_band(constant_op(Q_block), block_size).eval()
        # compare
        np.testing.assert_almost_equal(actual=Q_band, desired=Q_band_ref, decimal=10)


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("h_blocks", H_BLOCKS)
@pytest.mark.parametrize("v_blocks", V_BLOCKS)
def test_identity_block_to_band(block_size, h_blocks, v_blocks):
    """
    Operator block_to_band is the inverse of band_to_block.
    Composing the pair should result in the identity operator.
    This desired outcome is tested here

    band_to_block ( block_to_band (X) ) = X
    """
    A_block = np.random.randn(block_size, block_size)
    Q_block = np.tile(A_block + A_block.T, [v_blocks, h_blocks])

    with tf.compat.v1.Session(graph=tf.Graph()):
        # ===  band_to_block (block_to_band () ) =============
        # evaluate op output
        Q_block_op = constant_op(Q_block)
        Q_block_op_2 = band_to_block(block_to_band(Q_block_op, block_size), block_size)

        # compare
        np.testing.assert_almost_equal(
            actual=Q_block_op.eval(), desired=Q_block_op_2.eval(), decimal=10
        )


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("h_blocks", H_BLOCKS)
@pytest.mark.parametrize("v_blocks", V_BLOCKS)
def test_identity_band_to_block_sym(block_size, h_blocks, v_blocks):
    """
    Operator block_to_band is the inverse of band_to_block.
    Composing the pair should result in the identity operator.
    This desired outcome is tested here for symmetric matrices.

    block_to_band ( band_to_block (X) ) = X
    """
    A_block = np.random.randn(block_size, block_size)
    Q_block = np.tile(A_block + A_block.T, [v_blocks, h_blocks])
    Q_band = block_to_band_np(Q_block, block_size)

    with tf.compat.v1.Session(graph=tf.Graph()):
        # ===  block_to_band ( band_to_block () ) =============
        # evaluate op output
        Q_band_op = constant_op(Q_band)
        Q_band_op_2 = block_to_band(band_to_block(Q_band_op, block_size), block_size)

        # compare
        np.testing.assert_almost_equal(
            actual=Q_band_op.eval(), desired=Q_band_op_2.eval(), decimal=10
        )


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("h_blocks", H_BLOCKS)
@pytest.mark.parametrize("v_blocks", V_BLOCKS)
def test_identity_block_to_band_gradients_sym(block_size, h_blocks, v_blocks):
    """
    Operator block_to_band is the inverse of band_to_block.
    For the gradients, the diagonal elements will remain unchanged, but the block sub-diagonals
    will be multiplied by 2.
    """
    A_block = np.random.randn(block_size, block_size)
    Q_block = np.tile(A_block + A_block.T, [v_blocks, h_blocks])

    with tf.compat.v1.Session(graph=tf.Graph()):
        # ===  band_to_block (block_to_band () ) =============
        Q_block_op = constant_op(Q_block)
        Q_block_op_2 = band_to_block(block_to_band(Q_block_op, block_size), block_size)

        grad_ys = np.ones_like(Q_block_op_2.eval())
        grad_xs = tf.gradients(ys=Q_block_op_2, xs=Q_block_op, grad_ys=grad_ys)[0].eval()

        expected_grad_xs = np.ones_like(grad_xs)
        # sub diagonal blocks will be doubled since it is symmetric
        expected_grad_xs[block_size:, :] = 2.0

        # checking gradient is propagated unchanged
        np.testing.assert_almost_equal(actual=grad_xs, desired=expected_grad_xs, decimal=10)


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("h_blocks", H_BLOCKS)
@pytest.mark.parametrize("v_blocks", V_BLOCKS)
@pytest.mark.parametrize("symmetrise", [True, False])
def test_band_to_block(block_size, h_blocks, v_blocks, symmetrise):
    """
    Test the forward evaluation of band_to_block against a numpy implementation
    """
    A_block = np.random.randn(block_size, block_size)
    Q_block = np.tile(A_block + A_block.T, [v_blocks, h_blocks])
    Q_band = block_to_band_np(Q_block, block_size)
    Q_block_ref = band_to_block_np(Q_band, block_size, symmetrise=symmetrise)

    with tf.compat.v1.Session(graph=tf.Graph()):
        # evaluate op output
        Q_block_op = band_to_block(constant_op(Q_band), block_size, symmetric=symmetrise)

        # compare
        np.testing.assert_almost_equal(
            actual=Q_block_ref, desired=Q_block_op.eval(), decimal=10
        )


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("h_blocks", H_BLOCKS)
@pytest.mark.parametrize("v_blocks", V_BLOCKS)
@pytest.mark.parametrize("symmetrise", [True, False])
def test_block_to_band_gradients(block_size, h_blocks, v_blocks, symmetrise):
    """
    Test the gradients band_to_block

    Symmetric positive definite blocks are constructed  B1, B2, B3, ...
    (parameterized with their lower triangular part to avoid redundancy)

    Block banded matrices are constructed as
    Q_block = [B1, B2, B3 ... ]
              [ :   :    :    ]
              [B1, B2, B3 ... ]

    Its band representation Q_band is constructed from Q_block

    Some costs are derived on both representation (Q_block, Q_band)
    It is tested:
    - that these costs are equal
    - that the gradients of these cost wrt B1, B2, B3, ... are equal
    """
    with tf.compat.v1.Session(graph=tf.Graph()):
        # construct a symmetric block
        A_blocks = np.random.randn(h_blocks, block_size, block_size)
        A_blocks += np.transpose(A_blocks, (0, 2, 1))
        A_blocks_half = np.tril(A_blocks)

        A_blocks_half_op = tf.constant(A_blocks_half)
        A_blocks_op = A_blocks_half_op
        if symmetrise:
            A_blocks_op += tf.linalg.matrix_transpose(A_blocks_half_op) - tf.linalg.diag(
                tf.linalg.diag_part(A_blocks_half_op)
            )
        # stack the blocks
        Q_block_op = tf.tile(
            tf.reshape(
                tf.transpose(a=A_blocks_op, perm=(1, 0, 2)),
                [block_size, block_size * h_blocks],
            ),
            [v_blocks, 1],
        )
        Q_band_op = block_to_band(Q_block_op, block_size, symmetric=symmetrise)

        # List of 3 scalar costs on which to evaluate the gradient
        costs_op, costs_tf_op = [], []
        # sum of diag of matrix
        costs_op += [tf.reduce_sum(input_tensor=Q_band_op[0, :])]
        costs_tf_op += [tf.reduce_sum(input_tensor=tf.linalg.trace(A_blocks_op))]
        # first column of matrix
        costs_op += [tf.reduce_sum(input_tensor=Q_band_op[:, 0])]
        costs_tf_op += [tf.reduce_sum(input_tensor=Q_block_op[:, 0])]
        # column of matrix at index blocksize
        i = block_size - 1
        costs_op += [tf.reduce_sum(input_tensor=Q_band_op[:, i])]
        costs_tf_op += [tf.reduce_sum(input_tensor=Q_block_op[i:, i])]

        for cost_op, cost_tf_op in zip(costs_op, costs_tf_op):
            # evaluate op output
            cost = cost_op.eval()
            cost_tf = cost_tf_op.eval()

            # test forward
            np.testing.assert_almost_equal(cost, cost_tf)

            # gradient of the costs with respect to the blocks
            grad_cost_op = tf.gradients(ys=cost_op, xs=A_blocks_half_op)
            grad_cost_tf_op = tf.gradients(ys=cost_tf_op, xs=A_blocks_half_op)
            # evaluate op output
            grad_cost = grad_cost_op[0].eval()
            grad_cost_tf = grad_cost_tf_op[0].eval()

            # test forward evaluation of costs
            np.testing.assert_almost_equal(cost, cost_tf)
            # test gradients of the costs
            np.testing.assert_almost_equal(grad_cost_tf, grad_cost)

        np.testing.assert_almost_equal(grad_cost_tf, grad_cost)


def test_band_to_block_symm_gradients():
    """
    Test the gradients of the band_to_block operator

    A single 2 x 2 matrix block = [[b11, b12], [b12, b22]] is created
    along with its banded representation [[b11, b22],[b12, 0]]

    A block is constructed from the band block_from_band using the band_to_block operator

    Gradients of block and block_from_band with respect to [b11, b12, b22] should be equal
    which is what is tested here
    """

    with tf.compat.v1.Session(graph=tf.Graph()):
        b11 = tf.constant([[1.0]])
        b12 = tf.constant([[2.0]])
        b22 = tf.constant([[3.0]])

        block = tf.concat(
            [tf.concat([b11, b12], axis=1), tf.concat([b12, b22], axis=1)], axis=0
        )

        band = tf.concat(
            [tf.concat([b11, b22], axis=1), tf.concat([b12, [[0.0]]], axis=1)], axis=0
        )

        param_list = [b11, b12, b22]

        block_from_band = band_to_block(band, block_size=2, symmetric=True)

        # testing block construction from band (forward)
        np.testing.assert_almost_equal(block.eval(), block_from_band.eval())

        # evaluating gradients
        grad_block_from_band = tf.gradients(ys=block_from_band, xs=param_list)
        grad_block = tf.gradients(ys=block, xs=param_list)

        # comparing gradients
        for g, g1 in zip(grad_block, grad_block_from_band):
            np.testing.assert_almost_equal(g.eval(), g1.eval(), decimal=10)


def test_band_to_block_non_symm_gradients():
    """
    Test the gradients of the band_to_block operator

    A single 2 x 2 matrix block = [[b11, 0], [b12, b22]] is created
    along with its banded representation [[b11, b22],[b12, 0]]

    A block is constructed from the band block_from_band using the band_to_block operator

    Gradients of block and block_from_band with respect to [b11, b12, b22] should be equal
    which is what is tested here
    """

    with tf.compat.v1.Session(graph=tf.Graph()):
        b11 = tf.constant([[1.0]])
        b12 = tf.constant([[2.0]])
        b22 = tf.constant([[3.0]])

        block = tf.concat(
            [tf.concat([b11, [[0.0]]], axis=1), tf.concat([b12, b22], axis=1)], axis=0
        )

        band = tf.concat(
            [tf.concat([b11, b22], axis=1), tf.concat([b12, [[0.0]]], axis=1)], axis=0
        )

        param_list = [b11, b12, b22]

        block_from_band = band_to_block(band, block_size=2, symmetric=False)

        # testing block construction from band (forward)
        np.testing.assert_almost_equal(block.eval(), block_from_band.eval())

        # evaluating gradients
        grad_block_from_band = tf.gradients(ys=block_from_band, xs=param_list)
        grad_block = tf.gradients(ys=block, xs=param_list)

        # comparing gradients
        for g, g1 in zip(grad_block, grad_block_from_band):
            np.testing.assert_almost_equal(g.eval(), g1.eval(), decimal=10)
