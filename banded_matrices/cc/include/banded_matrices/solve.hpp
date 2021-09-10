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
 * @file  solve.h
 * @brief Generic algorithms for Solve L-1 B where L is a banded matrix that
 *        is additionally either lower- or upper-triangular.
 *        The right hand side can be a banded matrix, or a column vector,
 *        or several column vectors (i.e. a non-banded matrix).
 */

#pragma once

#include <algorithm>
#include <vector>

#include "Eigen/Dense"

#include "./banded_matrix.hpp"

namespace banded {

//
// Compute the Matrix L^-1 M, where L is a lower-triangular banded matrix.
// and M is a (non-banded) matrix. M may be single-column or include
// an arbitrary number of vectors to solve at once.
//
template <typename LeftBand, typename RightMatrix, typename ResultMatrix>
void solve_lower_band_mat(
    const LeftBand& left, const RightMatrix& mat, ResultMatrix* result_ptr) {
  auto& result = *result_ptr;
  if (left.upper_bandwidth() > 0) {
    throw std::runtime_error("Left matrix is assumed lower-triangular");
  }
  check_matrix_vectors_arguments(left, mat, result);
  // TODO(lucas) initial zeros are used in computation, suggesting missing optim
  result.setZero();

  // TODO(optim) Rethink the ordering between these loops
  for (Index col = 0; col < mat.cols(); ++col) {
    for (Index row = 0; row < left.dim(); ++row) {
      result(row, col) =
        (mat(row, col) - dot_product_mat(left, result, row, col))
        / left(row, row);
    }
  }
}

//
// Compute the vector U^-1 M, where U is upper-triangular banded matrix,
// and M is a (non-banded) matrix. M may be single-column or include
// an arbitrary number of vectors to solve at once.
//
template <typename LeftBand, typename RightMatrix, typename ResultMatrix>
void solve_upper_band_mat(
    const LeftBand& left, const RightMatrix& mat, ResultMatrix* result_ptr) {
  auto& result = *result_ptr;
  if (left.lower_bandwidth() > 0) {
      throw std::runtime_error("Left matrix is assumed upper-triangular");
  }
  check_matrix_vectors_arguments(left, mat, result);
  // TODO(lucas) initial zeros are used in computation, suggesting missing optim
  result.setZero();

  // TODO(optim) Rethink the ordering between these loops
  for (Index col = 0; col < mat.cols(); ++col) {
    for (Index row = left.dim() - 1; row >= 0; --row) {
      result(row, col) =
        (mat(row, col) - dot_product_mat(left, result, row, col))
        / left(row, row);
    }
  }
}

//
// Compute the desired band of L^-1 x B, where L is lower-triangular.
//
template <typename LeftBand, typename RightMatrix, typename ResultMatrix>
void solve_lower_band_band(
    const LeftBand& left, const RightMatrix& right, ResultMatrix* result_ptr) {
  auto& result = *result_ptr;
  const auto n = result.dim();

  check_binary_operator_arguments(left, right, result);

  if (left.upper_bandwidth() > 0)
    throw std::runtime_error("Left matrix is assumed lower-triangular");

  if (result.upper_bandwidth() < right.upper_bandwidth())
    throw std::runtime_error("Size is not sufficient to compute inverse");

  // Zero the matrix. We need anyway to set to 0 the top-left and bottom-right
  // values that are never iterated over. Here however the full matrix needs to
  // be zeroed, as some 0 values are used in the main loop.
  result.setZero();

  // This loops over diagonals from highest to lowest
  for (auto k = -result.upper_bandwidth(); k <= result.lower_bandwidth(); ++k) {
    // This loops over elements of the diagonal, bottom-up
    const auto last_i = std::max<Index>(0, k);
    for (Index i = std::min(n + k - 1, n - 1); i >= last_i; --i) {
      // TODO(optim) We need two checks for is_in_band here, suggesting we
      // TODO(optim) could refine the indices a bit. The cost is, however,
      // TODO(optim) dominated by the dot product
      if (result.is_in_band(i, i - k)) {
        const auto r = right.is_in_band(i, i - k) ? right(i, i - k) : 0;
        const auto dot = dot_product(left, result, i, i - k);
        result(i, i - k) = (r - dot) / left(i, i);
      }
    }
  }
}

//
// Compute the desired band of U^-1 x B, where U is upper-triangular.
//
template <typename LeftBand, typename RightMatrix, typename ResultMatrix>
void solve_upper_band_band(
    const LeftBand& left, const RightMatrix& right, ResultMatrix* result_ptr) {
  auto& result = *result_ptr;
  const auto n = result.dim();

  check_binary_operator_arguments(left, right, result);

  if (left.lower_bandwidth() > 0)
    throw std::runtime_error("Left matrix is assumed upper-triangular");

  if (result.lower_bandwidth() < right.lower_bandwidth())
    throw std::runtime_error("Size is not sufficient to compute inverse");

  // Zero the matrix. We need anyway to set to 0 the top-left and bottom-right
  // values that are never iterated over. Here however the full matrix needs to
  // be zeroed, as some 0 values are used in the main loop.
  result.setZero();

  // This loops over diagonals from lowest to highest
  for (auto k = result.lower_bandwidth() + 1;
       k >= -result.upper_bandwidth(); --k) {
    // This loops over elements of the diagonal, bottom-up
    const auto last_i = std::max<Index>(0, k);
    for (Index i = std::min(n + k - 1, n - 1); i >= last_i; --i) {
      // TODO(optim) We need two checks for is_in_band here, suggesting we
      // TODO(optim) could refine the indices a bit. The cost is, however,
      // TODO(optim) dominated by the dot product
      if (result.is_in_band(i, i - k)) {
        const auto r = right.is_in_band(i, i - k) ? right(i, i - k) : 0;
        const auto dot = dot_product(left, result, i, i - k);
        result(i, i - k) = (r - dot) / left(i, i);
      }
    }
  }
}

//
// The main function for solve. If needed this will do the solve into a properly
// sized intermediate banded matrix, and extract the desired band to the result.
//
// The intermediate matrix is calculated in a reusable buffer that
// to prevent multiple memory allocations (we may reconsider this).
// When the result is properly sized we do the solve directly into it to avoid
// memory overhead. The code is written in a way that avoids duplicate calls
// to solve_lower_band_band and solve_upper_band_band.
//
// Note that left and right matrices might be transposed or symmetrised.
// The result and the intermediate are however always directly of type
// BandedMatrix.
//
template <typename LeftBand, typename RightMatrix, typename ResultMatrix>
void solve_triang_band(
    const LeftBand& left, const RightMatrix& right,
    ResultMatrix* result_ptr) {

  using Element = typename ResultMatrix::ElementType;
  static_assert(
    std::is_same<Element, typename LeftBand::ElementType>::value,
    "Inconsistent numerical type in solve_banded");

  static_assert(
    std::is_same<Element, typename RightMatrix::ElementType>::value,
    "Inconsistent numerical type in solve_banded");

  auto& result = *result_ptr;
  const auto dim = right.dim();
  std::vector<typename ResultMatrix::ElementType> buffer;

  if (left.upper_bandwidth() == 0) {
    const bool needs_intermediate =
      right.upper_bandwidth() > result.upper_bandwidth();

    const auto lower_bandwidth = result.lower_bandwidth();
    const auto upper_bandwidth = needs_intermediate
      ? right.upper_bandwidth()
      : result.upper_bandwidth();

    if (needs_intermediate)
      buffer.resize((lower_bandwidth + 1 + upper_bandwidth) * dim);

    // The solve target is an intermediate buffer if needed or,
    // when possible, directly the result matrix:
    BandedMatrix<Element> solve_target {
      needs_intermediate
        ? buffer.data()
        : result.underlying_dense_matrix().data(),
      lower_bandwidth, upper_bandwidth, dim
    };

    solve_lower_band_band(left, right, &solve_target);

    if (needs_intermediate)
      extract_band(solve_target, &result);

  } else if (left.lower_bandwidth() == 0) {
    const bool needs_intermediate =
      right.lower_bandwidth() > result.lower_bandwidth();

    const auto lower_bandwidth = needs_intermediate
      ? right.lower_bandwidth()
      : result.lower_bandwidth();
    const auto upper_bandwidth = result.upper_bandwidth();

    if (needs_intermediate)
      buffer.resize((lower_bandwidth + 1 + upper_bandwidth) * dim);

    // The solve target is an intermediate buffer if needed or,
    // when possible, directly the result matrix:
    BandedMatrix<Element> solve_target {
      needs_intermediate
        ? buffer.data()
        : result.underlying_dense_matrix().data(),
      lower_bandwidth, upper_bandwidth, dim
    };

    solve_upper_band_band(left, right, &solve_target);

    if (needs_intermediate)
      extract_band(solve_target, &result);

  } else {
    throw std::runtime_error(
      "Solve operation expects a triangular left-hand side.");
  }
}

}  // namespace banded
