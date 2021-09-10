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
 * @file  test_product_of_banded_matrices.cc
 */

#include <iostream>

#include "common.hpp"
#include "banded_matrices/product.hpp"
#include "gtest/gtest.h"


namespace banded {

// HELPER FUNCTIONS FOR C++ TESTS
// Help doing product that allocate their results, etc.

//
// Version of the matrix product that allocates its result
//
    template<typename LeftMatrix, typename RightMatrix, typename ResultMatrix>
    ResultMatrix projected_matrix_product(const LeftMatrix &left,
                                          const RightMatrix &right) {
        // Note that both lower and upper subdiags could be up to dim
        // this could result in matrices with higher width than dim, but we in fact
        // impose that
        // assertion and prevent at construction that width() > dim()
        const auto lower_bandwidth =
                (ResultMatrix::band_type() == BandType::UpperTriangular)
                ? 0
                : std::min(left.dim(),
                           left.lower_bandwidth() + right.lower_bandwidth());
        const auto upper_bandwidth =
                (ResultMatrix::band_type() == BandType::LowerTriangular)
                ? 0
                : std::min(left.dim(),
                           left.upper_bandwidth() + right.upper_bandwidth());

        ResultMatrix result{lower_bandwidth, upper_bandwidth, right.dim()};
        product_band_band(left, right, &result);
        return result;
    }

//
// Easier to use (for tests) version of the general matrix product
//
    template<typename LeftMatrix, typename RightMatrix>
    auto general_banded_product(const LeftMatrix &left, const RightMatrix &right)
    -> BandedMatrixHolder<typename LeftMatrix::ElementType> {
        using ResultMatrix = BandedMatrixHolder<typename LeftMatrix::ElementType>;
        return projected_matrix_product<
                LeftMatrix, RightMatrix, ResultMatrix>(left, right);
    }

//
// Easier to use (for tests) version of the matrix product projected to lower
// triangular
//
    template<typename LeftMatrix, typename RightMatrix>
    auto lower_triangular_banded_product(const LeftMatrix &left,
                                         const RightMatrix &right)
    -> LowerTriangularBandedMatrixHolder<typename LeftMatrix::ElementType> {
        using ResultMatrix =
        LowerTriangularBandedMatrixHolder<typename LeftMatrix::ElementType>;
        return projected_matrix_product<
                LeftMatrix, RightMatrix, ResultMatrix>(left, right);
    }

} // namespace banded

template <typename Element>
void test_product_of_various_shapes() {
    using namespace banded;
    using namespace banded::testing;

    using Matrix = BandedMatrixHolder<Element>;

    std::default_random_engine prng(85629372);
    const double required_precision = std::numeric_limits<Element>::epsilon();

    auto a = random_banded_matrix<Element, false>(3, 2, 20, prng);
    auto b = random_banded_matrix<Element, false>(2, 9, 20, prng);
    auto c = random_banded_matrix<Element, false>(0, 4, 20, prng);
    auto d = random_banded_matrix<Element, false>(1, 0, 20, prng);

    auto all = std::vector<Matrix>{a, b, c, d};

    for (const auto &left : all) {
        for (const auto &right : all) {
            const auto prod = general_banded_product(left, right);
            const EigenMatrix<Element> checked_prod = to_dense(left) * to_dense(right);

            EXPECT_TRUE(isApprox(prod, checked_prod, required_precision));
        }
    }
}


template<typename Element>
void test_general_banded_product() {
    using namespace banded;
    using namespace banded::testing;

    long dim = 12;

    std::default_random_engine prng(85629372);
    const double required_precision = std::numeric_limits<Element>::epsilon();

    auto a = random_banded_matrix<Element, false>(3, 2, dim, prng);
    std::cout << "A banded triangular matrix\n" << to_dense(a) << std::endl;

    std::cout << "\nIts internal representation\n" << a.underlying_dense_matrix() << std::endl;

    // Test the conversion to/from dense
    const auto banded_copy_of_a = from_dense<Element>(to_dense(a), 3, 2);
    EXPECT_TRUE(isApprox(banded_copy_of_a, to_dense(a), required_precision));

    // Test matrix product
    const auto b = random_banded_matrix<Element, false>(1, 4, dim, prng);
    std::cout << "\nA second random matrix\n" << to_dense(b) << std::endl;

    const auto prod = general_banded_product(a, b);
    std::cout << "\nProduct between the two random matrices\n" << to_dense(prod) << std::endl;

    const EigenMatrix<Element> checked_prod = to_dense(a) * to_dense(b);
    std::cout << "\nDEBUG between the two random matrices\n" << checked_prod << std::endl;

    EXPECT_TRUE(isApprox(prod, checked_prod, required_precision));

    std::cout << "\nOK\n" << std::endl;
}


template<typename Element>
void test_lower_banded_product() {
    using namespace banded;
    using namespace banded::testing;

    long dim = 12;
    long width = 3;

    std::default_random_engine prng(85629372);
    const double required_precision = std::numeric_limits<Element>::epsilon();

    auto a = random_banded_matrix<Element, true>(width, 0, dim, prng);
    std::cout << "A random lower triangular matrix\n" << to_dense(a) << std::endl;

    std::cout << "\nIts internal representation\n" << a.underlying_dense_matrix() << std::endl;

    // Test the conversion to/from dense
    const auto banded_copy_of_a = from_dense<Element>(to_dense(a), width, 0);
    EXPECT_TRUE(isApprox(banded_copy_of_a, to_dense(a), required_precision));

    // Test matrix product
    const auto b = random_banded_matrix<Element, true>(4, 0, dim, prng);
    std::cout << "\nA second random matrix\n" << to_dense(b) << std::endl;

    const auto prod = lower_triangular_banded_product(a, b);
    std::cout << "\nProduct between the two random matrices\n" << to_dense(prod) << std::endl;

    const EigenMatrix<Element> checked_prod = to_dense(a) * to_dense(b);
    std::cout << "\nDEBUG between the two random matrices\n" << checked_prod << std::endl;

    EXPECT_TRUE(isApprox(prod, checked_prod, required_precision));

    std::cout << "\nOK\n" << std::endl;
}


// TODO understand what's wrong when we templatize over the Element type.
template<typename Element>
void test_product_by_transpose() {
    using namespace banded;
    using namespace banded::testing;

    using Matrix = LowerTriangularBandedMatrixHolder<Element>;

    std::default_random_engine prng(85629372);
    const double required_precision = std::numeric_limits<Element>::epsilon();

    const Matrix a = random_banded_matrix<Element, true>(7, 0, 15, prng);
    const Matrix b = random_banded_matrix<Element, true>(2, 0, 15, prng);
    const auto bt = Transposed<Matrix>(b);

    std::cout << "A random lower triangular matrix\n" << to_dense(b) << std::endl;
    std::cout << "Its transpose\n" << to_dense(bt) << std::endl;

    const EigenMatrix<Element> checked_prod = to_dense(a) * to_dense(b).transpose();
    std::cout << "\nFull dense product\n" << checked_prod << std::endl;
    EigenMatrix<Element> trian{checked_prod.rows(), checked_prod.cols()};
    trian.setZero();
    trian.template triangularView<Eigen::Lower>() = checked_prod.template triangularView<Eigen::Lower>();
    std::cout << "\nChecked lower triangular part of product product\n" << trian << std::endl;

    const auto prod = lower_triangular_banded_product(a, bt);
    std::cout << "Low-triangular product\n" << to_dense(prod) << std::endl;

    EXPECT_TRUE(isApprox(to_dense(prod), trian, required_precision));
    std::cout << "\nOK\n" << std::endl;
}


template<typename Element>
void test_product_by_symmetric() {
    using namespace banded;
    using namespace banded::testing;

    using Matrix = LowerTriangularBandedMatrixHolder<Element>;

    std::default_random_engine prng(85629372);
    const double required_precision = std::numeric_limits<Element>::epsilon();

    const Matrix a = random_banded_matrix<Element, true>(7, 0, 15, prng);
    const Matrix b = random_banded_matrix<Element, true>(2, 0, 15, prng);
    const auto bsym = Symmetric<Matrix>(b);

    std::cout << "A random lower triangular matrix\n" << to_dense(b) << std::endl;
    std::cout << "Its symmetrised version\n" << to_dense(bsym) << std::endl;

    EXPECT_TRUE(isApprox(to_dense(bsym), to_dense(bsym).transpose(), required_precision));

    const EigenMatrix<Element> checked_prod = to_dense(a) * to_dense(bsym);
    std::cout << "\nFull dense product\n" << checked_prod << std::endl;
    EigenMatrix<Element> trian{checked_prod.rows(), checked_prod.cols()};
    trian.setZero();
    trian.template triangularView<Eigen::Lower>() = checked_prod.template triangularView<Eigen::Lower>();
    std::cout << "\nChecked lower triangular part of product product\n" << trian << std::endl;

    const auto prod = lower_triangular_banded_product(a, bsym);
    std::cout << "Low-triangular product\n" << to_dense(prod) << std::endl;

    EXPECT_TRUE(isApprox(to_dense(prod), trian, required_precision));
    std::cout << "\nOK\n" << std::endl;
}


// Test the function product_band_mat.
// The Left template type indicates the type of the left matrix. The left matrix
// can of different banded types: BandedMatrixTemplate, Symmetric, Transposed.
// The DenseLeft template type indicates the Eigen matrix type for the densed version
// of the left matrix. This test compare our own matrix multiplication operator
// with Eigen's multiplication operator. And DenseLeft indicates the type of
// the densed left matrix.
// The Result template inciates the type of the result. It must be a Eigen::Matrix type.
// The right_cols template value indicates if the right operand is a vector (right_cols==1)
// or a matrix (right_cols==2). right_cols can only take value 1 or 2.
template <typename Left, typename DenseLeft, typename Result, int right_cols>
void test_product_band_mat(Left left, DenseLeft dense_left) {
    static_assert(right_cols==1 || right_cols==2, "right_cols must be 1 or 2.");
    using Matrix = typename banded::BandedMatrixHolder<double, false>;
    using banded::dot_product_mat;
    using banded::testing::to_dense;
    using banded::product_band_mat;

    // Here we use matrix with 5 rows, different row numbers
    // travel the same code path.
    Eigen::Matrix<double, 5, right_cols> right;
    if(right_cols == 1) {
        right <<  1.,  2.,  3.,  4.,  5.;
    } else {
        right <<  1, 2, 3, 4,  5,
                  6, 7, 8, 9, 10;
    }

    // Perform multiplication.
    Result result;
    product_band_mat(left, right, &result);

    // Note that if the type of expected is declared as auto,
    // it won't work, since the inferred type is different from
    // the correct type declared below, causing the isApprox
    // test to fail.
    Result expected = dense_left * right;
    EXPECT_TRUE(result.isApprox(expected));
}

//// Test
using banded::get_random_banded_matrix_holder;
using banded::get_random_banded_matrix;

// Test product_band_mat where the left is an arbitrary banded matrix
// and the right operand is a vector.
TEST(TEST_PRODUCT_MAT, test_product_band_mat_left_arbitrary_right_vector) {
    using Matrix = typename banded::BandedMatrixHolder<double, false>;
    using EigenMatrix = typename Eigen::Matrix<double, 5, 1>;
    using DenseLeft = typename Eigen::Matrix<double, 5, 5>;
    using banded::testing::to_dense;

    Matrix left = get_random_banded_matrix_holder<double, false>(1, 1, 5);
    DenseLeft dense_left = to_dense(left);
    test_product_band_mat<Matrix, DenseLeft, EigenMatrix, 1>(left, dense_left);
}

// Test product_band_mat where the left is an arbitrary banded matrix
// and the right operand is a matrix.
TEST(TEST_PRODUCT_MAT, test_product_band_mat_left_arbitrary_right_matrix) {
    using Matrix = typename banded::BandedMatrixHolder<double, false>;
    using EigenMatrix = typename Eigen::Matrix<double, 5, 2>;
    using DenseLeft = typename Eigen::Matrix<double, 5, 5>;
    using banded::testing::to_dense;

    Matrix left = get_random_banded_matrix_holder<double, false>(1, 1, 5);
    DenseLeft dense_left = to_dense(left);
    test_product_band_mat<Matrix, DenseLeft, EigenMatrix, 2>(left, dense_left);
}

// Test product_band_mat where the left is a lower triangular banded matrix
// and the right operand is a vector.
TEST(TEST_PRODUCT_MAT, test_product_band_mat_left_lower_triangular_right_vector) {
    using Matrix = typename banded::BandedMatrixHolder<double, true>;
    using EigenMatrix = typename Eigen::Matrix<double, 5, 1>;
    using DenseLeft = typename Eigen::Matrix<double, 5, 5>;
    using banded::testing::to_dense;

    Matrix left = get_random_banded_matrix_holder<double, true>(1, 0, 5);
    DenseLeft dense_left = to_dense(left);
    test_product_band_mat<Matrix, DenseLeft, EigenMatrix, 1>(left, dense_left);
}

// Test product_band_mat where the left is a lower triangular banded matrix
// and the right operand is a matrix.
TEST(TEST_PRODUCT_MAT, test_product_band_mat_left_lower_triangular_right_matrix) {
    using Matrix = typename banded::BandedMatrixHolder<double, true>;
    using EigenMatrix = typename Eigen::Matrix<double, 5, 2>;
    using DenseLeft = typename Eigen::Matrix<double, 5, 5>;
    using banded::testing::to_dense;

    Matrix left = get_random_banded_matrix_holder<double, true>(1, 0, 5);
    DenseLeft dense_left = to_dense(left);
    test_product_band_mat<Matrix, DenseLeft, EigenMatrix, 2>(left, dense_left);
}

// Test product_band_mat where the left is a symmetric banded matrix
// and the right operand is a vector.
TEST(TEST_PRODUCT_MAT, test_product_band_mat_left_symmetric_right_vector) {
    using Matrix = typename banded::BandedMatrixHolder<double, false>;
    using Symmetric = typename banded::Symmetric<Matrix>;
    using EigenMatrix = typename Eigen::Matrix<double, 5, 1>;
    using DenseLeft = typename Eigen::Matrix<double, 5, 5>;
    using banded::testing::to_dense;

    Matrix left = get_random_banded_matrix_holder<double, false>(1, 0, 5);
    DenseLeft dense_left = to_dense(Symmetric(left));
    Symmetric symmetric_left = Symmetric(left);
    test_product_band_mat<Symmetric, DenseLeft, EigenMatrix, 1>(symmetric_left, dense_left);
}

// Test product_band_mat where the left is a symmetric banded matrix
// and the right operand is a matrix.
TEST(TEST_PRODUCT_MAT, test_product_band_mat_left_symmetric_right_matrix) {
    using Matrix = typename banded::BandedMatrixHolder<double, false>;
    using Symmetric = typename banded::Symmetric<Matrix>;
    using EigenMatrix = typename Eigen::Matrix<double, 5, 2>;
    using DenseLeft = typename Eigen::Matrix<double, 5, 5>;
    using banded::testing::to_dense;

    Matrix left = get_random_banded_matrix_holder<double, false>(1, 0, 5);
    DenseLeft dense_left = to_dense(Symmetric(left));
    Symmetric symmetric_left = Symmetric(left);
    test_product_band_mat<Symmetric, DenseLeft, EigenMatrix, 2>(symmetric_left, dense_left);
}

// Test product_band_mat where the left is a transposed banded matrix
// and the right operand is a vector.
TEST(TEST_PRODUCT_MAT, test_product_band_mat_left_transposed_right_vector) {
    using Matrix = typename banded::BandedMatrixHolder<double, false>;
    using Transposed = typename banded::Transposed<Matrix>;
    using EigenMatrix = typename Eigen::Matrix<double, 5, 1>;
    using DenseLeft = typename Eigen::Matrix<double, 5, 5>;
    using banded::testing::to_dense;

    Matrix left = get_random_banded_matrix_holder<double, false>(1, 0, 5);
    Transposed transposed_left = Transposed(left);

    DenseLeft dense_left = to_dense(left).transpose();
    test_product_band_mat<Transposed, DenseLeft, EigenMatrix, 1>(transposed_left, dense_left);
}

// Test product_band_mat where the left is a transposed banded matrix
// and the right operand is a matrix.
TEST(TEST_PRODUCT_MAT, test_product_band_mat_left_transposed_right_matrix) {
    using Matrix = typename banded::BandedMatrixHolder<double, false>;
    using Transposed = typename banded::Transposed<Matrix>;
    using EigenMatrix = typename Eigen::Matrix<double, 5, 2>;
    using DenseLeft = typename Eigen::Matrix<double, 5, 5>;
    using banded::testing::to_dense;

    Matrix left = get_random_banded_matrix_holder<double, false>(1, 0, 5);
    Transposed transposed_left = Transposed(left);

    DenseLeft dense_left = to_dense(left).transpose();
    test_product_band_mat<Transposed, DenseLeft, EigenMatrix, 2>(transposed_left, dense_left);
}


TEST(TEST_PRODUCT_BAND, test_product_of_various_shapes) {
    test_product_of_various_shapes<float>();
}


TEST(TEST_PRODUCT_BAND, test_lower_banded_product_double) {
    test_lower_banded_product<double>();
}

TEST(TEST_PRODUCT_BAND, test_lower_banded_product_float) {
    test_lower_banded_product<float>();
}


TEST(TEST_PRODUCT_BAND, test_general_banded_product_double) {
    test_general_banded_product<double>();
}

TEST(TEST_PRODUCT_BAND, test_general_banded_product_float) {
    test_general_banded_product<float>();
}


TEST(TEST_PRODUCT_BAND, test_product_by_transpose_double) {
    test_product_by_transpose<double>();
}

TEST(TEST_PRODUCT_BAND, test_product_by_transpose_float) {
    test_product_by_transpose<float>();
}


TEST(TEST_PRODUCT_BAND, test_product_by_symmetric_double) {
    test_product_by_symmetric<double>();
}

TEST(TEST_PRODUCT_BAND, test_product_by_symmetric_float) {
    test_product_by_symmetric<float>();
}
