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
 * @file  banded_matrix.h
 * @brief Basic abstractions that allow to see Tensors as banded matrices with
 *        various assumptions;
 *        templated algorithms that use this abstraction will also work for
 *        arguments that are transposed or symmetric.
 */
#pragma once

#include <algorithm>
#include <cassert>
#include <random>
#include <type_traits>
#include <vector>
#include <utility>

#include "Eigen/Dense"

namespace banded {

////////////////////////////////////////////////////////////////////////////////
// Utilities
////////////////////////////////////////////////////////////////////////////////

using Index = Eigen::Index;

// NOTE For consistency with TF, 2D matrices are row-major;
// This differs from Eigen's default.
template <typename Element>
using EigenMatrix =
    Eigen::Matrix<Element, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;


// A basic class of contiguous integer range,
// e.g. IndexRange(0, 10) enumerates to 0, 1, .. , 9.
struct IndexRange {
  struct Iterator {
    explicit Iterator(Index start) : current(start) {}
    Index current;

    operator Index() const { return current; }
    Index operator*() const { return current; }

    IndexRange::Iterator& operator++() {  // prefix increment
      ++current;
      return *this;
    }
    IndexRange::Iterator operator++(int) {  // postfix increment
      Index old_current = current;
      ++current;
      return IndexRange::Iterator(old_current);
    }

    bool operator==(const Iterator& other) const {
      return current == other.current;
    }
    bool operator!=(const Iterator& other) const {
      return current != other.current;
    }
  };

  // Construct a IndexRange using start and end_exclusive.
  // If start > end_exclusive, throw an exception.
  IndexRange(Index start, Index end_exclusive)
      : start_(Iterator(start)), end_exclusive_(Iterator(end_exclusive)) {
      if (start > end_exclusive) {
          throw std::invalid_argument(
                  "start must not be larger than end_exclusive.");
      }
  }

  IndexRange intersect(const IndexRange& other) const {
    return IndexRange(std::max(*begin(), *other.begin()),
                      std::min(*end(), *other.end()));
  }

  const Iterator& begin() const { return start_; }
  const Iterator& end() const { return end_exclusive_; }

 private:
  const Iterator start_;
  const Iterator end_exclusive_;
};

//
// Info attached to each matrix type
//
enum class BandType { Arbitrary, LowerTriangular, UpperTriangular, Symmetric };

//
// Default implementation of some BandedMatrix methods,
// which are used for various representations.
//
namespace base {

template <typename Matrix>
IndexRange rows_in_band(const Matrix& matrix, Index col) {
  return IndexRange(std::max(Index(0), col - matrix.upper_bandwidth()),
                    std::min(col + matrix.lower_bandwidth() + 1, matrix.dim()));
}

template <typename Matrix>
IndexRange cols_in_band(const Matrix& matrix, Index row) {
  return IndexRange(std::max(Index(0), row - matrix.lower_bandwidth()),
                    std::min(row + matrix.upper_bandwidth() + 1, matrix.dim()));
}

template <typename Matrix>
bool is_in_band(const Matrix& matrix, Index row, Index col) {
  assert(0 <= col && col < matrix.dim());
  assert(0 <= row && row < matrix.dim());
  const auto r = row - col;
  return -matrix.upper_bandwidth() <= r && r <= matrix.lower_bandwidth();
}

}  // end namespace base


////////////////////////////////////////////////////////////////////////////////
// Main banded matrix class
////////////////////////////////////////////////////////////////////////////////

//
// Banded matrices. These are square matrices (dim x dim) where the only
// non-zero elements are in a band below and above the diagonal. We only store
// the band in a matrix of dimension (bandwidth x dim).
//
// The element at position (row, col) in the original, dense matrix, is found at
// position (row - col + upper_band_width, col) in the underlying storage.
// (Note that the total bandwidth equals lower_bandwidth + 1 + upper_bandwidth,
// accounting for the diagonal.)
// See https://en.wikipedia.org/wiki/Band_matrix for a drawing.
//
// Accessing elements out of the band is illegal, and will raise exceptions in
// DEBUG mode. In RELEASE the behaviour is, as usual, undefined.
//
// Some code specialization will be triggered if the upper band is known to be 0
// at instantiation time, which only happens if is_lower_triangular is set to
// true. Constructing with upper_bandwidth == 0 is, in this case needed, but not
// sufficient to get specialized code.
//
// Currently all representations are row major, following the TensorFlow
// default. Transposition is just a thin view that does not change the
// underlying representation, which remains row-major. This may need some
// thought when transposed matrices appear on left and right hand-sides of
// e.g. multiplications.
//
// This `template` class is parameterized by a `MatrixStorage` type which is
// the type of Matrix actually used within the object.
// This should be instantiated by an appropriate Eigen matrix type depending
// on whether we want to view a memory buffer as a matrix, or to allocate it.
//
// Use the derived classes `BandedMatrix` and `BandedMatrixHolder`
// which provide the right instantiations rather than ever using
// this `BandedMatrixTemplate` class directly.
//
template <typename Element, typename MatrixStorage,
          bool is_lower_triangular = false>
class BandedMatrixTemplate {
 public:  // Typedefs and static methods
  using ElementType = Element;
  using MatrixType = MatrixStorage;

  static constexpr BandType band_type() {
    return is_lower_triangular ? BandType::LowerTriangular
                               : BandType::Arbitrary;
  }

 public:  // Construction / destruction
  BandedMatrixTemplate(MatrixStorage storage,
              Index lower_bandwidth, Index upper_bandwidth,
              // TODO(lucas): no default value here - subclass dependent
              bool set_corners_to_zero = false)
      : m_(std::move(storage)),
        lower_bandwidth_(lower_bandwidth),
        upper_bandwidth_(upper_bandwidth) {
    static_assert(
      std::is_same<Element, typename MatrixStorage::Scalar>::value,
      "Inconsistent scalar type in template arguments.");

    if (is_lower_triangular && upper_bandwidth != 0)
      throw std::runtime_error(
          "Lower-banded matrices should always have upper bandwidth = 0");

    if (set_corners_to_zero)
      setCornersToZero();

    // The assertion assert(width() <= dim()) does not hold in general: both
    // lower and upper bandwidth can be up to dim, their sum can exceed dim()
    // In general we'll want a dense representation if width is more than some
    // X% of the width, but we don't forbid this here.
  }

 public:  // Public methods
  // Dimension of the matrix.
  Index dim() const { return m_.cols(); }

  // Total "width" of the banded part,
  // which is lower_bandwidth() + upper_bandwidth() + 1 (for the diagonal)
  Index width() const { return m_.rows(); }

  // Width of the band below the diagonal; diagonal excluded
  Index lower_bandwidth() const { return lower_bandwidth_; }

  // Width of the band above the diagonal; diagonal excluded
  Index upper_bandwidth() const {
    // The compiler should be able to statically eliminate branches and generate
    // specialized code for the case where is_lower_triangular is true
    return is_lower_triangular ? 0 : upper_bandwidth_;
  }

  // Access to elements of the matrix which are in the band.
  // Accessing elements out of the band does:
  // - In Debug: raise an exception
  // - In Release: undefined behaviour
  Element& operator()(Index row, Index col) {
    assert(is_in_band(row, col));
    return m_(row - col + upper_bandwidth(), col);
  }

  Element operator()(Index row, Index col) const {
    assert(is_in_band(row, col));
    return m_(row - col + upper_bandwidth(), col);
  }

  // Access to the ranges of Indexes that are within the band:
  IndexRange rows_in_band(Index col) const {
    return base::rows_in_band(*this, col);
  }

  IndexRange cols_in_band(Index row) const {
    return base::cols_in_band(*this, row);
  }

  // True if the Index at position (row, col) is within the band:
  bool is_in_band(Index row, Index col) const {
    return base::is_in_band(*this, row, col);
  }

  // Access the underlying dense Eigen matrix used for storage,
  // This allows to display/debug this representation, or do other
  // related minor things where leaking the abstraction is OK:
  MatrixStorage& underlying_dense_matrix() { return m_; }
  const MatrixStorage& underlying_dense_matrix() const { return m_; }

  // Apply the action, possibly a mutating one, to all entries of the
  // lower-triangular band
  // Action needs to have: void operator()(Index row, Index col, double&)
  // Note that the iteration is done with inner loops following each column.
  template <typename Action> void for_each_in_band(const Action& action) {
    const auto w = width();
    const auto d = dim();
    const auto lower = lower_bandwidth();
    const auto upper = upper_bandwidth();

    for (Index row = 0; row < w; ++row) {
      const Index begin = std::max(Index{0}, upper - row);
      const Index end = d - std::max(Index{0}, lower - (w - 1 - row));

      // This inner loop should focus on one row of the underlying matrix
      // at a time given the row-major representation:
      for (Index col = begin; col < end; ++col) {
        action(row + col - upper, col, m_(row, col));
      }
    }
  }

  template <typename Action> void for_each_in_band(const Action& action) const {
    const auto w = width();
    const auto d = dim();
    const auto lower = lower_bandwidth();
    const auto upper = upper_bandwidth();

    for (Index row = 0; row < w; ++row) {
      const Index begin = std::max(Index{0}, upper - row);
      const Index end = d - std::max(Index{0}, lower - (w - 1 - row));

      // This inner loop should focus on one row of the underlying matrix
      // at a time given the row-major representation:
      for (Index col = begin; col < end; ++col) {
        action(row + col - upper, col, m_(row, col));
      }
    }
  }

  // Fill the full content of the underlying matrix to 0.
  void setZero() {
    m_.setZero();
  }

  // Fill only the corners of the underlying matrix to 0;
  // These are the the top-left and bottom-right corners of the underlying
  // matrix that are never accessed when manipulating the band.
  // Setting the corners to 0 is needed by some gradient tests and
  // should be done when creating a fresh banded matrix.
  void setCornersToZero() {
    for (Index row = 0; row < upper_bandwidth(); ++row) {
      m_.block(row, 0, 1, upper_bandwidth() - row).setZero();
    }
    for (Index row = 0; row < lower_bandwidth(); ++row) {
      const auto len = lower_bandwidth() - row;
      m_.block(m_.rows() - 1 - row, m_.cols() - len, 1, len).setZero();
    }
  }

 protected:
  // The underling Eigen dense matrix of shape (bandwidth x dim):
  MatrixStorage m_;
  Index lower_bandwidth_;
  Index upper_bandwidth_;
};


//
// A banded matrix that views a memory segment, usually held by a tensor,
// as a banded matrix.
//
// Declaring an object of this type never does any dynamic memory allocation.
// Note that because such matrices are views on an underlying object, care
// should be taken not to outlive the underlying object.
//
// See the root class `BandedMatrixTemplate` for general documentation about
// banded matrices.
//
template <typename Element, bool is_lower_triangular = false>
class BandedMatrix : public BandedMatrixTemplate<
  Element,
  Eigen::Map<EigenMatrix<Element>>,
  is_lower_triangular>{
 public:  // Typedefs and static methods
  using MatrixView = typename Eigen::Map<EigenMatrix<Element>>;

 public:  // Construction / destruction
  // View a memory segment, usually held by a Tensor, as a BandedMatrix of the
  // indicated dimensions.
  BandedMatrix(Element* underlying,
               Index lower_bandwidth, Index upper_bandwidth,
               Index cols,
               bool set_corners_to_zero = false):
      BandedMatrixTemplate<Element, MatrixView, is_lower_triangular>(
        MatrixView(underlying, lower_bandwidth + 1 + upper_bandwidth, cols),
        lower_bandwidth, upper_bandwidth, set_corners_to_zero) {}
};


//
// A banded matrix that allocates its own memory for the underlying
// matrix content.
//
template <typename Element, bool is_lower_triangular = false>
class BandedMatrixHolder : public BandedMatrixTemplate<
  Element,
  EigenMatrix<Element>,
  is_lower_triangular> {
 public:  // Typedefs and static methods
  using MatrixStorage = EigenMatrix<Element>;

 public:  // Construction / destruction
  // Allocate a fresh banded matrix
  BandedMatrixHolder(Index lower_bandwidth, Index upper_bandwidth,
                     Index cols,
                     bool set_corners_to_zero = true):
      BandedMatrixTemplate<Element, MatrixStorage, is_lower_triangular>(
        MatrixStorage(lower_bandwidth + 1 + upper_bandwidth, cols),
        lower_bandwidth, upper_bandwidth, set_corners_to_zero) {}

  // Copy a Banded Matrix. This is only used in some complex algorithms
  // (like gradient of inverse from Cholesky) where some complicated
  // intermediate terms need to be allocated.
  template <typename RightMatrix>
  explicit BandedMatrixHolder(const RightMatrix& other):
      BandedMatrixTemplate<Element, MatrixStorage, is_lower_triangular>(
        MatrixStorage(other.width(), other.dim()),
        other.lower_bandwidth(), other.upper_bandwidth(), true) {
    static_assert(
      std::is_same<Element, typename RightMatrix::ElementType>::value,
      "Initialization from a different matrix type");

    this->for_each_in_band([&other](Index row, Index col, Element& target) {
      target = other(row, col);
    });
  }
};

//
// A specialization of BandedMatrix where the code is slightly more efficient,
// being optimized (if the compiler is doing a good job!) to statically
// eliminate upper_bandwidth == 0
//
template <typename Element>
using LowerTriangularBandedMatrix = BandedMatrix<Element, true>;

template <typename Element>
using LowerTriangularBandedMatrixHolder = BandedMatrixHolder<Element, true>;


////////////////////////////////////////////////////////////////////////////////
// Views on matrix representations
////////////////////////////////////////////////////////////////////////////////

//
// Transposes a banded matrix.
// This is not done explicitly, but just by exposing the same interface as
// BandedMatrix (restricted to its read-only methods), accessing the underlying
// banded matrix in a way that deals with transposition.
// Note that because such matrices are views on an underlying object, care
// should be taken not to outlive the underlying object.
//
template <typename BandedMatrix>
class Transposed {
 public:  // Typedefs, static methods, construction
  using ElementType = typename BandedMatrix::ElementType;
  using MatrixType = typename BandedMatrix::MatrixType;

  static constexpr BandType band_type() {
    return (BandedMatrix::band_type() == BandType::LowerTriangular)
               ? BandType::UpperTriangular
               : ((BandedMatrix::band_type() == BandType::UpperTriangular)
                      ? BandType::LowerTriangular
                      : BandedMatrix::band_type());
  }

  explicit Transposed(const BandedMatrix& m) : m_(m) {}

 public:  // Public methods
  Index dim() const { return m_.dim(); }
  Index width() const { return m_.width(); }

  Index lower_bandwidth() const { return m_.upper_bandwidth(); }
  Index upper_bandwidth() const { return m_.lower_bandwidth(); }

  ElementType operator()(Index row, Index col) const { return m_(col, row); }

  IndexRange rows_in_band(Index col) const { return m_.cols_in_band(col); }
  IndexRange cols_in_band(Index row) const { return m_.rows_in_band(row); }

  bool is_in_band(Index row, Index col) const {
    return m_.is_in_band(col, row);
  }

  const MatrixType& underlying_dense_matrix() const {
    return m_.underlying_dense_matrix();
  }

 private:
  const BandedMatrix& m_;
};


//
// Compute the symmetric version a lower-diagonal banded matrix.
// This is not done explicitly, but just by exposing the same interface as
// BandedMatrix (restricted to its read-only methods), accessing the underlying
// matrix in a way that deals with symmetry.
// Note that because such matrices are views on an underlying object, care
// should be taken not to outlive the underlying object.
//
// Symmetric here means that this class creates a view of a given low triangular
// banded matrix, and this view is a symmetric banded matrix whose upper band is
// the same as the lower band.
template <typename BandedMatrix>
class Symmetric {
 public:  // Typedefs,  static methods, construction
  using ElementType = typename BandedMatrix::ElementType;
  using MatrixType = typename BandedMatrix::MatrixType;
  static constexpr BandType band_type() { return BandType::Symmetric; }

  explicit Symmetric(const BandedMatrix& m) : m_(m) {
    if (m.upper_bandwidth() != 0)
      throw std::runtime_error(
        "Symmetric views are only allowed on lower-triangular matrices.");
  }

 public:  // Public methods
  Index dim() const { return m_.dim(); }
  Index width() const { return m_.width(); }

  Index lower_bandwidth() const { return m_.lower_bandwidth(); }
  Index upper_bandwidth() const { return m_.lower_bandwidth(); }

  ElementType operator()(Index row, Index col) const {
    // TODO(optim)
    // This introduces a branch in inner loops, which could be removed if
    // operations are done by blocks (row/col)
     return (col > row) ? m_(col, row) : m_(row, col);
  }

  IndexRange rows_in_band(Index col) const {
    return base::rows_in_band(*this, col);
  }

  IndexRange cols_in_band(Index row) const {
    return base::cols_in_band(*this, row);
  }

  bool is_in_band(Index row, Index col) const {
    return base::is_in_band(*this, row, col);
  }

  const MatrixType& underlying_dense_matrix() const {
    return m_.underlying_dense_matrix();
  }

 private:
  const BandedMatrix& m_;
};


//
// Get a view on const data that represent a low matrix
template <typename Element>
const LowerTriangularBandedMatrix<Element>
const_lower_triangular_view(const Element *data, Index width, Index dim) {
  // The Eigen type held by a BandedMatrix assumes that the data is mutable,
  // so requires a pointer to mutable data. We are treating it as an immutable
  // view into data, so we can cast away the const qualifier,
  // if we assume that the eigen::Map constructor doesn't modify values.
  auto cheat = const_cast<Element *>(data);
  return LowerTriangularBandedMatrix<Element>(cheat, width - 1, 0, dim);
}

//
// Get a view on const data that represent an arbitrary banded matrix.
// Note that you should not mutate the content of the matrix,
// even though the type system allows you to do so.
template <typename Element>
const BandedMatrix<Element> const_banded_view(
    const Element *data,
    Index lower_bandwidth, Index upper_bandwidth, Index dim) {
  // The Eigen type held by a BandedMatrix assumes that the data is mutable,
  // so requires a pointer to mutable data. We are treating it as an immutable
  // view into data, so we can cast away the const qualifier,
  // if we assume that the eigen::Map constructor doesn't modify values.
  auto cheat = const_cast<Element*>(data);
  return BandedMatrix<Element>(cheat, lower_bandwidth, upper_bandwidth, dim);
}

//
// Check that a binary operator (operator with two arguments)
// has arguments of same dimension and type
// The argument left is of type BandedMatrixTemplate.
// The argument right is of type BandedMatrixTemplate.
// The argument result is of type BandedMatrixTemplate.
template <typename LeftMatrix, typename RightMatrix, typename ResultMatrix>
void check_binary_operator_arguments(
    const LeftMatrix& left,
    const RightMatrix& right,
    const ResultMatrix& result) {
  using Element = typename ResultMatrix::ElementType;

  static_assert(
    std::is_same<Element, typename LeftMatrix::ElementType>::value,
    "Binary operator between matrices of different element types");

  static_assert(
    std::is_same<Element, typename RightMatrix::ElementType>::value,
    "Binary operator between matrices of different element types");

  if (left.dim() != right.dim())
    throw std::runtime_error(
        "Incompatible matrix dimensions in binary operator");

  if (result.dim() != left.dim())
    throw std::runtime_error(
        "Result is not allocated with the expected dimension");
}

//
// Check that a banded matrix and a right-hand-side with one or several vectors
// are compatible.
//
template <typename LeftMatrix, typename VectorArg, typename VectorResult>
void check_matrix_vectors_arguments(
  const LeftMatrix& left, const VectorArg& vec, const VectorResult& res) {
  using Element = typename LeftMatrix::ElementType;
  const auto dim = left.dim();

  static_assert(
    std::is_same<Element, typename VectorArg::Scalar>::value,
    "Inconsistent numerical type in matrix/vector operator");

  static_assert(
    std::is_same<Element, typename VectorResult::Scalar>::value,
    "Inconsistent numerical type in matrix/vector operator");

  if (vec.rows() != dim)
    throw std::runtime_error(
      "Size of left vector(s) does not match size of matrix");

  if (res.rows() != dim)
    throw std::runtime_error(
      "Size of result vector(s) incorrect in matrix/vector operator");
}

// TODO(optim)
// The two versions of dot product are used by matrix products and solves
// (matrix x matrix and matrix by vector).
// These are inner loops that are crucial to the performance of all the
// related pieces of code. Any optimization here with
// block/vectorized operations, simplified indices, or code specialization
// could impact perf.

// The arguments left and right should be of type BandedMatrixTemplate
// and they must have the same dimension.
// The argument row selects a row vector from the left matrix.
// The argument col selects a column vector from the right matrix.
// And then perform a dot product on the row vector and the column vector.
// In numpy syntax, that is:
//    np.dot(left[row, :], right[:, col]) given two banded matrices.
//
// Note: The implementation only supports the case when the selected row
// from left and the selected column from right has intersection.
// This causes the implementation to be a restricted version of the np.dot
// example.
template <typename LeftMatrix, typename RightMatrix>
auto dot_product(
  const LeftMatrix& left, const RightMatrix& right, Index row, Index col
) -> typename LeftMatrix::ElementType {
  using Element = typename LeftMatrix::ElementType;
  Element dot = 0;

  const auto dot_product_indices =
    left.cols_in_band(row)
    .intersect(right.rows_in_band(col));

  for (const auto j : dot_product_indices) {
    dot += left(row, j) * right(j, col);
  }

  return dot;
}

//
// Dot product np.dot(left[row, :], right[:, col]), where:
// left is a banded matrix, and right is a column vector (dim x 1 matrix) or
// a non-banded matrix containing several column-vectors.
// The argument left is of type BandedMatrixTemplate.
// The argument right is of type BandedMatrixTemplate or of type Eigen:Matrix.
// The argument row selects a row vector from the left matrix.
// The argument col selects a column vector from the right matrix.
// And then perform a dot product on the row vector and the column vector.
// In numpy syntax, that is:
//    np.dot(left[row, :], right[:, col]) given two banded matrices.
template <typename LeftMatrix, typename RightVector>
auto dot_product_mat(
  const LeftMatrix& left, const RightVector& right, Index row, Index col
) -> typename LeftMatrix::ElementType {
  using Element = typename LeftMatrix::ElementType;
  Element p = 0;
  for (const auto j : left.cols_in_band(row))
    p += left(row, j) * right(j, col);
  return p;
}

//
// Extract a smaller band from an initial banded matrix.
//
template <typename InitialBandedMatrix, typename ResultMatrix>
void extract_band(const InitialBandedMatrix& ini,
                  ResultMatrix* result) {
  using Element = typename ResultMatrix::ElementType;
  static_assert(
    std::is_same<Element, typename InitialBandedMatrix::ElementType>::value,
    "Inconsistent numerical type in extract_band");

  if (ini.dim() != result->dim())
    throw std::runtime_error(
      "Inconsistent matrix dimensions in extract_band.");

  if (result->lower_bandwidth() > ini.lower_bandwidth()
        || result->upper_bandwidth() > ini.upper_bandwidth())
    throw std::runtime_error(
      "Target of band extraction should be smaller than initial matrix.");

  result->setCornersToZero();
  result->for_each_in_band([&ini](Index row, Index col, Element& target) {
    target = ini(row, col);
  });
}

//
// Create a matrix whose band is full of zeros
//
template <typename Element, bool is_lower_triangular = false>
BandedMatrixHolder<Element, is_lower_triangular> zero(
    Index lower_bandwidth, Index upper_bandwidth, Index dimension) {
  BandedMatrixHolder<Element, is_lower_triangular> result {
    lower_bandwidth, upper_bandwidth, dimension
  };
  result.setZero();
  return result;
}

}  // end namespace banded
