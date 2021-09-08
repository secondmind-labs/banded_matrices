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
 * @file  test common.h
 * @brief Some functions that make it easier to write some tests.
 */

#pragma once

#include "banded_matrices/banded_matrix.hpp"

// MACRO to check an exception has been thrown with particular error message.
// This is not natively supported by Google tests.
// The following macro is from https://github.com/google/googletest/issues/952.
#define EXPECT_THROW_WITH_MESSAGE(stmt, etype, whatstring) EXPECT_THROW( \
        try { \
            stmt; \
        } catch (const etype& ex) { \
            EXPECT_EQ(std::string(ex.what()), whatstring); \
            throw; \
        } \
    , etype)


namespace banded {
namespace testing {

////////////////////////////////////////////////////////////////////////////////
// Test functions. Could be moved somewhere else
////////////////////////////////////////////////////////////////////////////////

//
// Create a banded matrix of the specified dimensions that is filled in using
// custom logic.
// The initializer has: ElementType operator()(Index row, Index col)
//
template <typename Element, bool is_lower_triangular, typename Initializer>
BandedMatrixHolder<Element, is_lower_triangular> create_banded_matrix(
    Index lower_bandwidth, Index upper_bandwidth, Index dimension,
    const Initializer& initializer) {
  BandedMatrixHolder<Element, is_lower_triangular> result(
    lower_bandwidth, upper_bandwidth, dimension, true);
  result.for_each_in_band([&initializer](Index row, Index col, Element& target) {
    target = initializer(row, col);
  });
  return result;
}

//
// Construct a random banded matrix
//
template <typename Element,
          bool is_lower_triangular,
          typename Distribution = std::uniform_real_distribution<Element>>
BandedMatrixHolder<Element, is_lower_triangular> random_banded_matrix(
    Index lower_bandwidth, Index upper_bandwidth, Index dimension,
    std::default_random_engine& prng,
    Distribution distr = Distribution()) {
  // We could create a banded matrix whose underlying dense matrix uses Eigen's
  // Random:   result._m = MatriXd::Random(width, dimension)
  // However this would have some bottom-right non-zero values that violate our
  // invariant.
  return create_banded_matrix<Element, is_lower_triangular>(
    lower_bandwidth, upper_bandwidth, dimension,
    [&](Index, Index) { return distr(prng); }
  );
}

//
// Conversion from a banded matrix to a dense Eigen matrix
//
template <typename Matrix>
auto to_dense(const Matrix& input)
    -> EigenMatrix<typename Matrix::ElementType> {
  EigenMatrix<typename Matrix::ElementType> result(input.dim(), input.dim());
  result.setZero();

  for (Index col = 0; col < input.dim(); ++col) {
    for (Index row = 0; row < input.dim(); ++row)
      result(row, col) = input.is_in_band(row, col) ? input(row, col) : 0;
  }

  assert(Matrix::band_type() != BandType::LowerTriangular ||
         result.isLowerTriangular());
  return result;
}

//
// Specializations that make it easier to mix with Eigen matrices
//
inline const EigenMatrix<float>& to_dense(const EigenMatrix<float>& matrix) {
  return matrix;
}

inline const EigenMatrix<double>& to_dense(const EigenMatrix<double>& matrix) {
  return matrix;
}

//
// Conversion from a dense Eigen matrix to a banded matrix
//
template <typename Element>
BandedMatrixHolder<Element> from_dense(
    const EigenMatrix<Element>& matrix,
    Index lower_bandwidth, Index upper_bandwidth) {
  if (matrix.cols() != matrix.rows())
    throw std::runtime_error("Non-square matrix");

  auto result = zero<Element, false>(
    lower_bandwidth, upper_bandwidth, matrix.rows());

  for (Index col = 0; col < matrix.cols(); ++col) {
    for (Index row = 0; row < matrix.rows(); ++row) {
      if (result.is_in_band(row, col))
        result(row, col) = matrix(row, col);

      else if (matrix(row, col) != 0)
        throw std::runtime_error(
            "Matrix has non-zero values out of specified band");
    }
  }

  return result;
}

//
// True if we have two representations for the same mathematical object.
// The right-hand side has to be an Eigen dense matrix, for simplicity
//
template <typename Matrix, typename EigenMatrix>
bool isApprox(const Matrix m, const EigenMatrix eigen_matrix, double precision) {
  const bool return_value = to_dense(m).isApprox(eigen_matrix, precision);

  if (!return_value) {
    // Calculate how far away from the correct precision we were (so we can report to stderr,
    // helping people trying to fix issues).
    const auto lhs = to_dense(m);
    const auto rhs = eigen_matrix;

    // Calculated by looking at Eigen code: https://github.com/libigl/eigen/blob/1f05f51517ec4fd91eed711e0f89e97a7c028c0e/Eigen/src/Core/MathFunctions.h#L1598
    // which differs from Eigen documentation: https://eigen.tuxfamily.org/dox/classEigen_1_1DenseBase.html#ae8443357b808cd393be1b51974213f9c
    const double actual_precision = std::sqrt(
        (lhs - rhs).squaredNorm() / std::min(lhs.squaredNorm(), rhs.squaredNorm())
    );

    std::cerr
        << "precision bounds on isApprox failed ("
        << "expected=" << precision << ", "
        << "actual=" << actual_precision << ")" << std::endl;
  }

  return return_value;
}

} // end namespace testing



// Facility function to return a banded matrix holder with random values.
// The created matrix holder is a square matrix with shape dimension x dimension.
// lower_bandwidth and upper_bandwidth specifies the bandwidth.
// The type parameter RealType specifies the type of the matrix element,
// usually float and double.
// The type parameter is_lower_triangular specifies whether the created
// matrix should be lower triangular. If this is true, the upper_bandwidth parameter
// must have value 0.
// The seed parameter specifies the random seed to the random number generator.
template<typename RealType, bool is_lower_triangular>
BandedMatrixHolder<RealType, is_lower_triangular> get_random_banded_matrix_holder(
        int lower_bandwidth, int upper_bandwidth, int dimension, long seed=85629372) {
    if (is_lower_triangular) {
        assert(upper_bandwidth==0);
    }
    std::default_random_engine prng(seed);
    auto result = banded::testing::random_banded_matrix<RealType, is_lower_triangular>(
            lower_bandwidth, upper_bandwidth, dimension, prng);
    return result;
}


// Facility function to return a banded matrix with random values.
// The created matrix holder is a square matrix with shape dimension x dimension.
// lower_bandwidth and upper_bandwidth specifies the bandwidth.
// The type parameter RealType specifies the type of the matrix element,
// usually float and double.
// The type parameter is_lower_triangular specifies whether the created
// matrix should be lower triangular. If this is true, the upper_bandwidth parameter
// must have value 0.
// The seed parameter specifies the random seed to the random number generator.
template<typename RealType, bool is_lower_triangular>
BandedMatrix<RealType, is_lower_triangular> get_random_banded_matrix(
        int lower_bandwidth, int upper_bandwidth, int dimension, long seed=85629372) {
    if (is_lower_triangular) {
        assert(upper_bandwidth==0);
    }

    // Create underlying storage for the matrix.
    using namespace banded;
    const RealType some_value = 0;
    EigenMatrix<RealType> underlying = EigenMatrix<RealType>::Constant(
            lower_bandwidth + 1 + upper_bandwidth, dimension, some_value);
    banded::BandedMatrix<RealType, is_lower_triangular> result{
            underlying.data(), lower_bandwidth, upper_bandwidth, dimension};
    result.setCornersToZero();


    // Set matrix entries in the band to random values.
    std::default_random_engine prng(seed);
    auto distr = std::uniform_real_distribution<RealType>(0, 1);
    result.for_each_in_band([&distr, &prng](Index row, Index col, RealType &target) {
        target = distr(prng);
    });

    return result;
}

} // end namespace banded