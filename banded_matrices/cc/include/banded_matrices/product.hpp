//
// Copyright (c) 2021 The banded_matrices Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

/**
 * @file  product.h
 * @brief Generic algorithms for the product A . B where A is a banded matrix
 *        and B is either a banded matrix or a vector.
 */

#pragma once

#include <vector>

#include "Eigen/Dense"

#include "./banded_matrix.hpp"

namespace banded {

//
// General matrix product that:
// - Takes matrices in any banded representation as inputs;
//   lower triangular, upper triangular, arbitrary, symmetric or transposed;
// - Only fills-in the resulting product for the band allocated in the resulting
//   matrix; this is useful, for instance, in the case where the product has an
//   arbitrary band but we are only interested in the lower-diagonal part of it.
//
// TODO(optim)
// Doing the inner loop by blocks in a way that better uses Eigen could perhaps
// lead to some vectorization or better use of native instructions of the target
// processor.
//
// (It would also better deal with some cases such as product-by-symmetric which
// introduce branches in the inner loops).
//
// We could also use Open MP syle parallelism to evaluate the external loops
// multi-threaded. though we need some thought on thread hierarchy, cache...
//
template <typename LeftMatrix, typename RightMatrix, typename ResultMatrix>
void product_band_band(
    const LeftMatrix& left, const RightMatrix& right,
    ResultMatrix* product_ptr) {
  using Element = typename ResultMatrix::ElementType;

  std::vector<Element> buffer;
  auto& product = *product_ptr;
  auto lower_bandwidth = product.lower_bandwidth();
  auto upper_bandwidth = product.upper_bandwidth();

  check_binary_operator_arguments(left, right, product);

  // All code guarded by the ``needs_intermediate`` Boolean is to deal with
  // corner cases where the desired band is unusually large, and should be
  // padded with zeros. It is preferable to deal with this case correctly as it
  // also happens when defining gradients of products with a desired result
  // bandwidth that's too *small*.
  //
  // Our approach here is: in the rare cases where the desired band is too large
  // we compute the product into an intermediate matrix with reduced bands. We
  // then copy this intermediate into the result and pad it with 0s as needed.
  //
  // This allows the main inner loop to be free of any "is_in_band" tests which
  // would be needed without the copy. We want to keep this inner loop as basic
  // and amenable to optimizations as possible.
  bool needs_intermediate =
    lower_bandwidth > left.lower_bandwidth() + right.lower_bandwidth() ||
    upper_bandwidth > left.upper_bandwidth() + right.upper_bandwidth();

  if (needs_intermediate) {
    lower_bandwidth = std::min(
      lower_bandwidth,
      left.lower_bandwidth() + right.lower_bandwidth());

    upper_bandwidth = std::min(
      upper_bandwidth,
      left.upper_bandwidth() + right.upper_bandwidth());

    buffer.resize((lower_bandwidth + 1 + upper_bandwidth) * product.dim());
  }

  // Always zero the product in particular the upper-left / bottom-right values
  // that aren't ever touched. If the desired product band is large requiring
  // padding with zeros, we just set the whole matrix to 0:
  if (needs_intermediate) {
    product.setZero();
  } else {
    product.setCornersToZero();
  }

  // Do a product into the correct size
  BandedMatrix<Element> product_target {
    needs_intermediate
      ? buffer.data()
      : product.underlying_dense_matrix().data(),
    lower_bandwidth, upper_bandwidth, product.dim()
  };

  // Iterate only on the indices that are within the band:
  product_target.for_each_in_band([&left, &right](
      Index row, Index col, Element &value) {
    value = dot_product(left, right, row, col);
  });

  // If the product was done into an intermediate then copy the relevant band
  if (needs_intermediate) {
    product_target.for_each_in_band([&product](
        Index row, Index col, Element value) {
      product(row, col) = value;
    });
  }
}

//
// Product of an arbitrary banded matrix by a column vector or non-banded
// matrix containing several columns.
// The left matrix can be any object that looks like a banded matrix,
// lower triangular, upper triangular, arbitrary, symmetric or transposed.
// The left argument is of type BandedMatrixTemplate.
// The mat argument is of type Eigen::Matrix.
// The result arguemnt is of type Eigen::Matrix&*.
template <typename LeftBand, typename RightMatrix, typename ResultMatrix>
void product_band_mat(
    const LeftBand& left, const RightMatrix& mat, ResultMatrix* product_ptr) {
  auto& product = *product_ptr;
  check_matrix_vectors_arguments(left, mat, product);

  // TODO(optim) Rethink the ordering between these loops
  for (Index col = 0; col < product.cols(); ++col) {
    for (Index row = 0; row < left.dim(); ++row) {
      product(row, col) = dot_product_mat(left, mat, row, col);
    }
  }
}

}  // namespace banded
