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
 * @file  test_banded_matrix.cc
 * @brief Some tests (run by hand for now, but intended to be used as unit tests when we
 *        have build support) for the matrix representation.
 */

#include <iostream>
#include "./common.hpp"
#include "gtest/gtest.h"
#include "banded_matrices/banded_matrix.hpp"


using Index = banded::Index;
using IndexRange = banded::IndexRange;
using banded::get_random_banded_matrix_holder;
using banded::get_random_banded_matrix;

// Test that APIS that are only used occasionally for debugging can instantiate


// Test the creation of BandedMatrixHolder objects.
template<typename RealType, bool is_lower_triangular>
void test_banded_matrix_holder_creation(int lower_bandwidth, int upper_bandwidth, int dimension) {
    auto m = get_random_banded_matrix_holder<RealType, is_lower_triangular>(lower_bandwidth, upper_bandwidth, dimension);
    EXPECT_EQ(m.dim(), dimension);
    EXPECT_EQ(m.lower_bandwidth(), lower_bandwidth);
    EXPECT_EQ(m.upper_bandwidth(), upper_bandwidth);
    EXPECT_EQ(m.width(), m.lower_bandwidth() + m.upper_bandwidth() + 1);
}

// Test the creation of BandedMatrix objects.
template<typename RealType, bool is_lower_triangular>
void test_banded_matrix_creation(int lower_bandwidth, int upper_bandwidth, int dimension) {
    auto m = get_random_banded_matrix<RealType, is_lower_triangular>(lower_bandwidth, upper_bandwidth, dimension);
    EXPECT_EQ(m.dim(), dimension);
    EXPECT_EQ(m.lower_bandwidth(), lower_bandwidth);
    EXPECT_EQ(m.upper_bandwidth(), upper_bandwidth);
    EXPECT_EQ(m.width(), m.lower_bandwidth() + m.upper_bandwidth() + 1);
}

// Test correctness of rows_in_band.
void test_rows_in_band() {
    int dimension = 6;
    auto m = get_random_banded_matrix_holder<double, false>(1, 1, dimension);
    using banded::base::rows_in_band;

    auto r0 = rows_in_band<banded::BandedMatrixHolder<double, false>>(m, 0);
    EXPECT_EQ(r0.begin().current, 0);
    EXPECT_EQ(r0.end().current, 2);

    auto r1 = rows_in_band<banded::BandedMatrixHolder<double, false>>(m, 1);
    EXPECT_EQ(r1.begin().current, 0);
    EXPECT_EQ(r1.end().current, 3);

    auto r2 = rows_in_band<banded::BandedMatrixHolder<double, false>>(m, 2);
    EXPECT_EQ(r2.begin().current, 1);
    EXPECT_EQ(r2.end().current, 4);

    auto r3 = rows_in_band<banded::BandedMatrixHolder<double, false>>(m, 3);
    EXPECT_EQ(r3.begin().current, 2);
    EXPECT_EQ(r3.end().current, 5);

    auto r4 = rows_in_band<banded::BandedMatrixHolder<double, false>>(m, 4);
    EXPECT_EQ(r4.begin().current, 3);
    EXPECT_EQ(r4.end().current, 6);

    auto r5 = rows_in_band<banded::BandedMatrixHolder<double, false>>(m, 5);
    EXPECT_EQ(r5.begin().current, 4);
    EXPECT_EQ(r5.end().current, 6);
}

// Test correctness of cols_in_band.
void test_cols_in_band() {
    int dimension = 6;
    auto m = get_random_banded_matrix_holder<double, false>(1, 1, dimension);
    using banded::base::cols_in_band;

    auto r0 = cols_in_band<banded::BandedMatrixHolder<double, false>>(m, 0);
    EXPECT_EQ(r0.begin().current, 0);
    EXPECT_EQ(r0.end().current, 2);

    auto r1 = cols_in_band<banded::BandedMatrixHolder<double, false>>(m, 1);
    EXPECT_EQ(r1.begin().current, 0);
    EXPECT_EQ(r1.end().current, 3);

    auto r2 = cols_in_band<banded::BandedMatrixHolder<double, false>>(m, 2);
    EXPECT_EQ(r2.begin().current, 1);
    EXPECT_EQ(r2.end().current, 4);

    auto r3 = cols_in_band<banded::BandedMatrixHolder<double, false>>(m, 3);
    EXPECT_EQ(r3.begin().current, 2);
    EXPECT_EQ(r3.end().current, 5);

    auto r4 = cols_in_band<banded::BandedMatrixHolder<double, false>>(m, 4);
    EXPECT_EQ(r4.begin().current, 3);
    EXPECT_EQ(r4.end().current, 6);

    auto r5 = cols_in_band<banded::BandedMatrixHolder<double, false>>(m, 5);
    EXPECT_EQ(r5.begin().current, 4);
    EXPECT_EQ(r5.end().current, 6);
}

// Test the in_is_band function.
void test_is_in_band() {
    int dimension = 6;
    auto m = get_random_banded_matrix_holder<double, false>(1, 1, dimension);
    using banded::base::is_in_band;

    // Iterate over all elements in band, check
    // is_in_band returns True for these elements.
    // Store locations of these elements in set in_band_locations.
    std::set<int> in_band_locations;
    m.for_each_in_band([&m, &in_band_locations](Index row, Index col, double target) {
        ASSERT_TRUE(is_in_band(m, row, col));
        in_band_locations.insert(row*10+col);
    });

    // Iterate over all elements in the full matrix,
    // check is_in_band returns False for elements that are not in band.
    for(int row=0; row<dimension; row++) {
        for(int col=0; col<dimension; col++) {
            if(in_band_locations.find(row*10+col) == in_band_locations.end()) {
                ASSERT_FALSE(is_in_band(m, row, col));
            }
        }
    }
}

void test_set_zero() {
    int dimension = 6;
    // The created random matrix with fixed random seed
    // will have non zero entries.
    auto m = get_random_banded_matrix_holder<double, false>(1, 1, dimension);
    m.setZero();
    m.for_each_in_band([&](Index row, Index col, double target) {
        ASSERT_DOUBLE_EQ(target, 0);
    });
}

// Test for_each_in_band and the () operator for returning right values.
void test_for_each_in_band_reading() {
    int dimension = 6;
    auto m = get_random_banded_matrix_holder<double, false>(1, 1, dimension);
    m.for_each_in_band([&m](Index row, Index col, double target) {
        ASSERT_DOUBLE_EQ(target, m(row, col));
    });
}

// Test for_each_in_band for updating.
void test_for_each_in_band_update() {
    int dimension = 6;
    auto m = get_random_banded_matrix_holder<double, false>(1, 1, dimension);
    m.for_each_in_band([&m](Index row, Index col, double& target) {
        target = row * 10 + col;
    });

    m.for_each_in_band([](Index row, Index col, double target) {
        ASSERT_DOUBLE_EQ(target, row * 10 + col);
    });
}

// Test for_each_in_band for updating.
void test_parentheses_operator_update() {
    int dimension = 6;
    auto m = get_random_banded_matrix_holder<double, false>(1, 1, dimension);
    m.for_each_in_band([&m](Index row, Index col, double target) {
        m(row, col) = row * 10 + col;
    });

    m.for_each_in_band([](Index row, Index col, double target) {
        ASSERT_DOUBLE_EQ(target, row * 10 + col);
    });
}




template<typename RealType>
void test_symmetric_and_transpose_underlying_dense_matrix() {
    using Matrix = typename banded::BandedMatrixHolder<RealType>;
    std::default_random_engine prng(85629372);
    auto a = banded::testing::random_banded_matrix<RealType, false>(2, 0, 10, prng);
    const auto at = banded::Transposed<Matrix>(a);
    const auto as = banded::Symmetric<Matrix>(a);

    ASSERT_TRUE(&at.underlying_dense_matrix() == &a.underlying_dense_matrix());
    ASSERT_TRUE(&as.underlying_dense_matrix() == &a.underlying_dense_matrix());
}


// Test the Transposed class implements the transposition semantic.
void test_transpose_matrix() {
    int dimension = 6;
    using banded::testing::to_dense;
    auto m = get_random_banded_matrix_holder<double, false>(1, 2, dimension);

    auto original = banded::BandedMatrixHolder<double, false>(m);
    auto transposed = banded::Transposed<banded::BandedMatrixHolder<double, false>>(m);

    EXPECT_EQ(transposed.dim(), original.dim());
    EXPECT_EQ(transposed.width(), original.width());
    EXPECT_EQ(transposed.lower_bandwidth(), original.upper_bandwidth());
    EXPECT_EQ(transposed.upper_bandwidth(), original.lower_bandwidth());

    auto original_dense = to_dense(original);
    auto transposed_dense = to_dense(transposed);


    for(int row=0; row < dimension; row++) {
        EXPECT_EQ(transposed.rows_in_band(row).begin(), original.cols_in_band(row).begin());
        EXPECT_EQ(transposed.rows_in_band(row).end(), original.cols_in_band(row).end());

        EXPECT_EQ(transposed.cols_in_band(row).begin(), original.rows_in_band(row).begin());
        EXPECT_EQ(transposed.cols_in_band(row).end(), original.rows_in_band(row).end());

        for(int col=0; col<dimension; col++) {
            ASSERT_DOUBLE_EQ(original_dense(row, col), transposed_dense(col, row));
            ASSERT_DOUBLE_EQ(transposed.is_in_band(row, col), original.is_in_band(col, row));
        }
    }
}

// Test that constructing a Symmetric object from a non
// lower triangular matrix results in an exception.
void test_symmetric_matrix_invalid() {
    int dimension = 6;
    auto m = get_random_banded_matrix_holder<double, false>(1, 2, dimension);
    auto s = banded::Symmetric<banded::BandedMatrixHolder<double, false>>(m);
}

// Test that the class Symmetric implements the
// correct semantic.
void test_symmetric_matrix() {
    int dimension = 6;
    auto m = get_random_banded_matrix_holder<double, true>(1, 0, dimension);
    auto original = banded::BandedMatrixHolder<double, true>(m);
    auto symmetric = banded::Symmetric<banded::BandedMatrixHolder<double, true>>(m);

    EXPECT_EQ(symmetric.upper_bandwidth(), original.lower_bandwidth());
    EXPECT_EQ(symmetric.lower_bandwidth(), original.lower_bandwidth());
    EXPECT_EQ(symmetric.dim(), original.dim());
    EXPECT_EQ(symmetric.width(), original.width());

    // Construct a new structurally symmetric matrix (not value symmetric)
    // to test rows_in_band and cols_in_band.
    auto m2 = get_random_banded_matrix_holder<double, false>(1, 1, dimension);
    for(int i=0; i<dimension; i++) {
        auto rows = symmetric.rows_in_band(i);
        auto expected_rows = m2.rows_in_band(i);
        EXPECT_EQ(rows.begin(), expected_rows.begin());
        EXPECT_EQ(rows.end(), expected_rows.end());

        auto cols = symmetric.cols_in_band(i);
        auto expected_cols = m2.cols_in_band(i);
        EXPECT_EQ(cols.begin(), expected_cols.begin());
        EXPECT_EQ(cols.end(), expected_cols.end());
    }

    // Test is_in_band.
    m2.for_each_in_band([&symmetric](Index row, Index col, double target) {
        ASSERT_TRUE(symmetric.is_in_band(row, col));
    });

    // Test symmetric property.
    original.for_each_in_band([&symmetric](Index row, Index col, double target) {
        ASSERT_DOUBLE_EQ(symmetric(row, col), target);
        ASSERT_DOUBLE_EQ(symmetric(col, row), target);
    });
}

// Test that the corner initialization to 0s works OK and consistently
// with the for_each_in_band methods.
template<typename RealType>
void test_zero_corners(Index lower_bandwidth, Index upper_bandwidth) {
    using namespace banded;

    const RealType some_value = 3.79;
    EigenMatrix<RealType> underlying = EigenMatrix<RealType>::Constant(
            lower_bandwidth + 1 + upper_bandwidth, 20, some_value);

    banded::BandedMatrix<RealType> m{
            underlying.data(), lower_bandwidth, upper_bandwidth, 20};
    m.setCornersToZero();
    std::cout << underlying << std::endl << std::endl;

    EXPECT_EQ(underlying(upper_bandwidth - 1, 0), 0);
    EXPECT_EQ(underlying(0, upper_bandwidth - 1), 0);
    EXPECT_EQ(underlying(underlying.rows() - lower_bandwidth, underlying.cols() - 1), 0);
    EXPECT_EQ(underlying(underlying.rows() - 1, underlying.cols() - lower_bandwidth), 0);

    const banded::BandedMatrix<RealType> &const_view = m;
    const_view.for_each_in_band([&](Index row, Index col, RealType target) {
        EXPECT_EQ(target, some_value);
    });
}

// Visualize the iteration order of the generic loop.
// For performance this should match the row-major layout used by TensorFlow:
void check_iteration_order() {
    char c = 1;
    std::default_random_engine prng(85629372);
    auto a = banded::testing::random_banded_matrix<double, false>(2, 3, 10, prng);

    a.for_each_in_band([&c](Index row, Index col, double &target) {
        target = c++;
    });

    std::cout << a.underlying_dense_matrix() << std::endl;
}

// Test whether we can construct an IndexRange.
void test_index_range_construction() {
    int start = 0;
    int end = 10;
    auto range = IndexRange(start, end);
    EXPECT_EQ(range.begin(), start);
    EXPECT_EQ(range.end(), end);
}

// Test that constructing an IndexRange with
// start smaller than end throws an exception.
void test_invalid_index_range_construction() {
    int start = 10;
    int end = 0;
    ASSERT_THROW({IndexRange(start, end);}, std::invalid_argument);
}

// Test that the intersect operator works normally
// when the two IndexRanges indeed intersects,
// meaning their index ranges overlap.
void test_index_range_intersect_non_empty() {
    auto r1 = IndexRange(5, 10);
    int indices[4][2] = {
            {1, 6},
            {1, 12},
            {6, 11},
            {6, 9}
    };
    for(int i=0; i<4; i++) {
        auto r2 = IndexRange(indices[i][0], indices[i][1]);
        auto intersected = r1.intersect(r2);
        EXPECT_EQ(intersected.begin(), std::max(r1.begin(), r2.begin()));
        EXPECT_EQ(intersected.end(), std::min(r1.end(), r2.end()));
    }
}

// Test that in case two IndexRanges do not
// overlap with each other, the intersect
// operator throws an assertion violation.
void test_index_range_intersect_empty() {
    auto r1 = IndexRange(5, 10);

    int indices[2][2] = {
            {1,4},
            {12, 15}
    };
    for(int i=0; i<2; i++) {
        auto r2 = IndexRange(indices[i][0], indices[i][1]);
        EXPECT_THROW({r1.intersect(r2);}, std::invalid_argument);
    }
}

// Test the zero function.
void test_zero() {
    using banded::zero;
    auto z = zero<double>(1, 1, 6);
    // Check all elements from the zero matrix are zero.
    z.for_each_in_band([&](Index row, Index col, double target) {
        ASSERT_DOUBLE_EQ(target, 0);
    });
}

// Test that the extract_band method throws and exception
// when the its two argument matrix have different dimensions.
void test_extract_band_invalid_dimension() {
    using Matrix = typename banded::BandedMatrixHolder<double, false>;
    using banded::extract_band;
    Matrix source = get_random_banded_matrix_holder<double, false>(1, 1, 5);
    Matrix target = get_random_banded_matrix_holder<double, false>(1, 1, 4);
    EXPECT_THROW_WITH_MESSAGE(
            {extract_band(source, &target);},
            std::runtime_error,
            "Inconsistent matrix dimensions in extract_band.");
}

// Test that the extract_band method throws an exception
// when its second argument, the result matrix has a lower_bandwidth
// that is larger than the lower_bandwidth of the first argument, the source matrix.
void test_extract_band_invalid_lower_bandwidth() {
    using Matrix = typename banded::BandedMatrixHolder<double, false>;
    using banded::extract_band;
    Matrix source = get_random_banded_matrix_holder<double, false>(1, 1, 5);
    // The target matrix has a lower_bandwidth that is larger than
    // the lower_bandwidth of source matrix. This is not allowed in extract_band.
    Matrix target = get_random_banded_matrix_holder<double, false>(2, 1, 5);
    EXPECT_THROW_WITH_MESSAGE(
            {extract_band(source, &target);},
            std::runtime_error,
            "Target of band extraction should be smaller than initial matrix.");
}

// Test that the extract_band method throws an exception
// when its second argument, the result matrix has a upper_bandwidth
// that is larger than the upper_bandwidth of the first argument, the source matrix.
void test_extract_band_invalid_upper_bandwidth() {
    using Matrix = typename banded::BandedMatrixHolder<double, false>;
    using banded::extract_band;
    Matrix source = get_random_banded_matrix_holder<double, false>(1, 1, 5);
    // The target matrix has a lower_bandwidth that is larger than
    // the lower_bandwidth of source matrix. This is not allowed in extract_band.
    Matrix target = get_random_banded_matrix_holder<double, false>(1, 2, 5);

    EXPECT_THROW_WITH_MESSAGE(
            {extract_band(source, &target);},
            std::runtime_error,
            "Target of band extraction should be smaller than initial matrix.");
}

// Test that the extract_band method performs band extraction.
void test_extract_band(int source_lower_bandwidth, int source_upper_bandwidth,
        int target_lower_bandwidth, int target_upper_bandwidth, int dimension) {
    using Matrix = typename banded::BandedMatrixHolder<double, false>;
    using banded::extract_band;

    Matrix source = get_random_banded_matrix_holder<double, false>(
            source_lower_bandwidth, source_upper_bandwidth, dimension);
    Matrix target = get_random_banded_matrix_holder<double, false>(
            target_lower_bandwidth, target_upper_bandwidth, dimension);

    extract_band(source, &target);

    // Check that the elements in target matrix equal to
    // the elements in the same location in the source matrix.
    // We don't need to check the elements outside the bands of the
    // target matrix because other tests made sure that elements
    // off the bands are zero.
    target.for_each_in_band([&source](Index row, Index col, double target) {
        ASSERT_DOUBLE_EQ(target, source(row, col));
    });
}

// Does first and second has intersection?
bool has_intersection(const IndexRange& first, const IndexRange& second) {
    bool result = true;
    try {
        first.intersect(second);
    } catch(std::invalid_argument& e) {
        result = false;
    }
    return result;
}

// Test the dot_product function by comparing its result with the result
// of the same dot product operation on dense matrices using Eigen API.
void test_dot_product() {
    using Matrix = typename banded::BandedMatrixHolder<double, false>;
    using banded::dot_product;
    using banded::testing::to_dense;
    using Eigen::placeholders::all;

    int dimension = 5;
    Matrix left = get_random_banded_matrix_holder<double, false>(1, 1, dimension);
    Matrix right = get_random_banded_matrix_holder<double, false>(1, 1, dimension);

    // Construct two Eigen dense matrices and perform dot product using Eigen API.
    // Then compare the Eigen results with results from our implementation.
    auto dense_left = to_dense(left);
    auto dense_right = to_dense(right);

    std::cout<<dense_left<<std::endl;
    for(int row=0; row<dimension; row++) {
        for(int col=0; col<dimension;col++) {
            // Only test the case when selected row and column
            // has intersection because only in this case, the dot_product
            // operation is defined.
            if (has_intersection(left.cols_in_band(row), right.rows_in_band(col))) {
                auto expected = dense_left(row, all).dot(dense_right(all, col));
                auto actual = dot_product(left, right, row, col);
                ASSERT_DOUBLE_EQ(actual, expected);
            }
        }
    }
}

// Test dot_product_mat with the right argument
// being a banded matrix.
void test_dot_product_mat_banded_right() {
    using Matrix = typename banded::BandedMatrixHolder<double, false>;
    using banded::dot_product_mat;
    using banded::testing::to_dense;
    using Eigen::placeholders::all;

    int dimension = 5;
    Matrix left = get_random_banded_matrix_holder<double, false>(1, 1, dimension);
    Matrix right = get_random_banded_matrix_holder<double, false>(2, 2, dimension);

    // Construct two Eigen dense matrices and perform dot product using Eigen API.
    // Then compare the Eigen results with results from our implementation.
    auto dense_left = to_dense(left);
    auto dense_right = to_dense(right);

    std::cout<<dense_left<<std::endl;
    for(int row=0; row<dimension; row++) {
        for(int col=0; col<dimension;col++) {
            // Use row for both the row and col arguments for dot_product_mat.
            // This is because the implementation of dot_product_mat
            // only supports the case that all the column indices of the
            // selected row from left need to be valid row indices
            // of the selected column from right.
            auto expected = dense_left(row, all).dot(dense_right(all, row));
            auto actual = dot_product_mat(left, right, row, row);
            ASSERT_DOUBLE_EQ(actual, expected);
        }
    }
}

// Test dot_product_mat with the right argument
// being a dense matrix.
void test_dot_product_mat_dense_right() {
    using Matrix = typename banded::BandedMatrixHolder<double, false>;
    using banded::dot_product_mat;
    using banded::testing::to_dense;
    using Eigen::placeholders::all;

    int dimension = 5;
    Matrix left = get_random_banded_matrix_holder<double, false>(1, 1, dimension);
    Eigen::Matrix<double, 5, 5> dense_right;
    dense_right <<  1,  2,  3,  4,  5,
                    6,  7,  8,  9, 10,
                   11, 12, 13, 14, 15,
                   16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25;
    // Construct two Eigen dense matrices and perform dot product using Eigen API.
    // Then compare the Eigen results with results from our implementation.
    auto dense_left = to_dense(left);
//    auto dense_right = to_dense(right);

    std::cout<<dense_left<<std::endl;
    for(int row=0; row<dimension; row++) {
        for(int col=0; col<dimension;col++) {
            // Here we can use col to select a column from the dense
            // right matrix.
            auto expected = dense_left(row, all).dot(dense_right(all, col));
            auto actual = dot_product_mat(left, dense_right, row, col);
            ASSERT_DOUBLE_EQ(actual, expected);
        }
    }
}

void test_check_matrix_vectors_arguments() {
    using Matrix = typename banded::BandedMatrixHolder<double, false>;
    using banded::check_matrix_vectors_arguments;
    int dimension = 5;
    Matrix left = get_random_banded_matrix_holder<double, false>(1, 1, dimension);
    // No need to initialize the following two vectors
    // because check_matrix_vectors_arguments only
    // check the shapes of vectors.
    Eigen::Matrix<double, 5, 1> vec;
    Eigen::Matrix<double, 5, 1> res;
    check_matrix_vectors_arguments(left, vec, res);

    // Test that check_matrix_vectors_arguments
    // throws an exception when the number of rows in left
    // is different from the number of rows in the vec argument.
    Eigen::Matrix<double, 4, 1> vec_row_incorrect;

    EXPECT_THROW_WITH_MESSAGE(
            {check_matrix_vectors_arguments(left, vec_row_incorrect, res);},
            std::runtime_error,
            "Size of left vector(s) does not match size of matrix");

    // Test that check_matrix_vectors_arguments
    // throws an exception when the number of rows in left
    // is different from the number of rows in the vec argument.
    Eigen::Matrix<double, 4, 1> res_incorrect;

    EXPECT_THROW_WITH_MESSAGE(
            {check_matrix_vectors_arguments(left, vec, res_incorrect);},
            std::runtime_error,
            "Size of result vector(s) incorrect in matrix/vector operator");

}

// Test the binary_operator_arguments method.
void test_binary_operator_arguments() {
    using Matrix = typename banded::BandedMatrixHolder<double, false>;
    using banded::check_binary_operator_arguments;
    int dimension = 5;

    // Test the case when the shape of left, right and result are correct.
    Matrix left = get_random_banded_matrix_holder<double, false>(1, 1, dimension);
    Matrix right = get_random_banded_matrix_holder<double, false>(1, 1, dimension);
    Matrix result = get_random_banded_matrix_holder<double, false>(1, 1, dimension);
    check_binary_operator_arguments(left, right, result);

    // Test the case when the shape of left is different from the shape of right.
    // binary_operator_arguments should throw a runtime error.
    Matrix left2 = get_random_banded_matrix_holder<double, false>(1, 1, dimension);
    Matrix right2 = get_random_banded_matrix_holder<double, false>(1, 1, 4);
    Matrix result2 = get_random_banded_matrix_holder<double, false>(1, 1, dimension);
    EXPECT_THROW_WITH_MESSAGE(
            {check_binary_operator_arguments(left2, right2, result2);},
            std::runtime_error,
            "Incompatible matrix dimensions in binary operator");

    // Test the case when the shape of left is different from the shape of result.
    // binary_operator_arguments should throw a runtime error.
    Matrix left3 = get_random_banded_matrix_holder<double, false>(1, 1, 4);
    Matrix right3 = get_random_banded_matrix_holder<double, false>(1, 1, 4);
    Matrix result3 = get_random_banded_matrix_holder<double, false>(1, 1, dimension);
    EXPECT_THROW_WITH_MESSAGE(
            {check_binary_operator_arguments(left3, right3, result3);},
            std::runtime_error,
            "Result is not allocated with the expected dimension");
}

void test_const_banded_view() {
    using Matrix = typename banded::BandedMatrixHolder<double, false>;
    using banded::const_banded_view;
    int dimension = 5;
    Matrix m = get_random_banded_matrix_holder<double, false>(1, 1, dimension);

    // Construct a view from existing banded matrix m.
    auto view = const_banded_view(m.underlying_dense_matrix().data(), 1, 1, dimension);

    // Iterate all elements in view to and check if
    // they are equal to corresponding elements in m.
    view.for_each_in_band([&m](Index row, Index col, double target) {
        ASSERT_DOUBLE_EQ(target, m(row, col));
    });
}

// Test that update the matrix returned
// from const_banded_view is mutable.
// But you should not mutate the content of the view matrix.
// Leave this test until in a future version, we address
// the mutability issue in BandedMatrixTemplate class design.
// In that case, this test will fail and we can safely remove it then.
void test_const_banded_view_update() {
    using Matrix = typename banded::BandedMatrixHolder<double, false>;
    using banded::const_banded_view;
    int dimension = 5;
    Matrix m = get_random_banded_matrix_holder<double, false>(1, 1, dimension);

    // Construct a view from existing banded matrix m.
    auto view = const_banded_view(m.underlying_dense_matrix().data(), 1, 1, dimension);
    view(0, 0) = 1;

    ASSERT_DOUBLE_EQ(m(0, 0), 1);
    ASSERT_DOUBLE_EQ(view(0, 0), 1);
}

// Test that the const_lower_triangular_view function
// creates a correct view of the underlying data.
void test_const_lower_triangular_view() {
    using Matrix = typename banded::BandedMatrixHolder<double, true>;
    using banded::const_lower_triangular_view;
    int dimension = 5;
    int lower_bandwidth = 3;
    Matrix m = get_random_banded_matrix_holder<double, true>(lower_bandwidth, 0, dimension);

    // Construct a view from existing lower triangular banded matrix m.
    // Note the second argument of const_lower_triangular_view method
    // should be set to lower_bandwidth+1.
    auto view = const_lower_triangular_view(m.underlying_dense_matrix().data(), lower_bandwidth+1, dimension);

    // Iterate over elements of the view and check that they are
    // equal to corresponding elements in the source matrix.
    view.for_each_in_band([&m](Index row, Index col, double target) {
        ASSERT_DOUBLE_EQ(target, m(row, col));
    });
}

// Test that update the matrix returned
// from const_lower_triangular_view is mutable.
// But you should not mutate the content of the view matrix.
// Leave this test until in a future version, we address
// the mutability issue in BandedMatrixTemplate class design.
// In that case, this test will fail and we can safely remove it then.
void test_const_lower_triangular_view_update() {
    using Matrix = typename banded::BandedMatrixHolder<double, true>;
    using banded::const_lower_triangular_view;
    int dimension = 5;
    int lower_bandwidth = 3;
    Matrix m = get_random_banded_matrix_holder<double, true>(lower_bandwidth, 0, dimension);

    // Construct a view from existing lower triangular banded matrix m.
    // Note the second argument of const_lower_triangular_view method
    // should be set to lower_bandwidth+1.
    auto view = const_lower_triangular_view(m.underlying_dense_matrix().data(), lower_bandwidth+1, dimension);
    view(0, 0) = 1;

    ASSERT_DOUBLE_EQ(m(0, 0), 1);
    ASSERT_DOUBLE_EQ(view(0, 0), 1);
}

TEST(TEST_BANDED_MATRIX, test_const_lower_triangular_update) {
    test_const_lower_triangular_view_update();
}

TEST(TEST_BANDED_MATRIX, test_const_lower_triangular_view) {
    test_const_lower_triangular_view();
}

TEST(TEST_BANDED_MATRIX, test_const_banded_view) {
    test_const_banded_view();
}

TEST(TEST_BANDED_MATRIX, test_const_banded_view_update) {
    test_const_banded_view_update();
}



TEST(TEST_BANDED_MATRIX, test_binary_operator_arguments) {
    test_binary_operator_arguments();
}


TEST(TEST_BANDED_MATRIX, test_check_matrix_vectors_arguments) {
    test_check_matrix_vectors_arguments();
}


TEST(TEST_BANDED_MATRIX, test_dot_product_mat_banded_right) {
    test_dot_product_mat_banded_right();
}

TEST(TEST_BANDED_MATRIX, test_dot_product_mat_dense_right) {
    test_dot_product_mat_dense_right();
}



TEST(TEST_BANDED_MATRIX, test_dot_product) {
    test_dot_product();
}

// Test

TEST(TEST_BANDED_MATRIX, test_zero_corners_double) {
    test_zero_corners<double>(3, 6);
}

TEST(TEST_BANDED_MATRIX, test_zero_corners_float) {
    test_zero_corners<float>(4, 2);
}

TEST(TEST_BANDED_MATRIX, test_symmetric_and_transpose_underlying_double) {
    test_symmetric_and_transpose_underlying_dense_matrix<double>();
}

TEST(TEST_BANDED_MATRIX, test_transpose_matrix) {
    test_transpose_matrix();
}


TEST(TEST_BANDED_MATRIX, test_symmetric_and_transpose_underlying_float) {
    test_symmetric_and_transpose_underlying_dense_matrix<float>();
}

TEST(TEST_BANDED_MATRIX, test_symmetric_matrix_invalid) {
    EXPECT_THROW_WITH_MESSAGE(
            {test_symmetric_matrix_invalid();},
            std::runtime_error,
            "Symmetric views are only allowed on lower-triangular matrices.");
}

TEST(TEST_BANDED_MATRIX, test_symmetric_matrix) {
    test_symmetric_matrix();
}

TEST(TEST_BANDED_MATRIX, check_iteration_order) {
    check_iteration_order();
}

TEST(TEST_BANDED_MATRIX, test_banded_matrix_holder_creation_float) {
    test_banded_matrix_holder_creation<float, false>(1, 2, 10);
    test_banded_matrix_holder_creation<float, true>(1, 0, 10);
}

TEST(TEST_BANDED_MATRIX, test_banded_matrix_holder_creation_double) {
    test_banded_matrix_holder_creation<double, false>(1, 2, 10);
    test_banded_matrix_holder_creation<double, true>(1, 0, 10);
}

TEST(TEST_BANDED_MATRIX, test_banded_matrix_creation_double) {
    test_banded_matrix_creation<double, false>(1, 1, 6);
    test_banded_matrix_creation<double, true>(1, 0, 10);
}

TEST(TEST_BANDED_MATRIX, test_banded_matrix_creation_float) {
    test_banded_matrix_creation<float, false>(1, 1, 6);
    test_banded_matrix_creation<float, true>(1, 0, 10);
}

TEST(TEST_BANDED_MATRIX, test_rows_in_band) {
    test_rows_in_band();
}

TEST(TEST_BANDED_MATRIX, test_cols_in_band) {
    test_cols_in_band();
}

TEST(TEST_BANDED_MATRIX, test_is_in_band) {
    test_is_in_band();
}

TEST(TEST_BANDED_MATRIX, test_set_zero) {
    test_set_zero();
}

TEST(TEST_BANDED_MATRIX, test_for_each_in_band_reading) {
    test_for_each_in_band_reading();
}

TEST(TEST_BANDED_MATRIX, test_for_each_in_band_update) {
    test_for_each_in_band_update();
}

TEST(TEST_BANDED_MATRIX, test_parentheses_operator_update) {
    test_parentheses_operator_update();
}

TEST(TEST_BANDED_MATRIX, test_zero) {
    test_zero();
}

TEST(TEST_BANDED_MATRIX, test_extract_band_invalid_dimension) {
    test_extract_band_invalid_dimension();
}

TEST(TEST_BANDED_MATRIX, test_extract_band_invalid_lower_bandwidth) {
    test_extract_band_invalid_lower_bandwidth();
}

TEST(TEST_BANDED_MATRIX, test_extract_band_invalid_upper_bandwidth) {
    test_extract_band_invalid_upper_bandwidth();
}

TEST(TEST_BANDED_MATRIX, test_extract_band) {
    int dimension = 10;
    int max_bandwidth = 3;
    // Iterate over different band widths for both source and target matrix.
    for(int source_l=0; source_l < max_bandwidth; source_l++) {
        for(int source_u=0; source_u < max_bandwidth; source_u++) {
            for(int target_l=0; target_l <= source_l; target_l++) {
                for(int target_u=0; target_u <= source_u; target_u++) {
                    test_extract_band(source_l, source_u, target_l, target_u, dimension);
                }
            }
        }
    }
}



TEST(TEST_INDEX_RANGE, test_index_range_construction) {
    test_index_range_construction();
}

TEST(TEST_INDEX_RANGE, test_invalid_index_range_construction) {
    test_invalid_index_range_construction();
}

TEST(TEST_INDEX_RANGE, test_index_range_intersect_non_empty) {
    test_index_range_intersect_non_empty();
}

TEST(TEST_INDEX_RANGE, test_index_range_intersect_empty) {
    test_index_range_intersect_empty();
}
