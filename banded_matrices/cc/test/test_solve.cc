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
#include <iostream>
#include "common.hpp"
#include "banded_matrices/solve.hpp"
#include "gtest/gtest.h"


namespace banded {

//
// Solve that allocates its result, mostly for tests and for reference.
//
    template<typename LeftMatrix, typename RightMatrix>
    auto solve_triang_band(
            const LeftMatrix &left, const RightMatrix &right,
            Index result_lower_diags, Index result_upper_diags
    ) -> BandedMatrixHolder<typename LeftMatrix::ElementType> {

        using Element = typename LeftMatrix::ElementType;
        using ResultMatrix = BandedMatrixHolder<Element>;

        ResultMatrix result = zero<Element>(
                result_lower_diags, result_upper_diags, right.dim()
        );

        solve_triang_band(left, right, &result);
        return result;
    }
}

template<typename Matrix, typename EigenMatrix>
double max_band_error(const Matrix m, const EigenMatrix eigen_matrix) {
    using Index = Eigen::Index;
    using Element = typename Matrix::ElementType;

    Element max_error = 0;
    m.for_each_in_band([&](Index row, Index col, const Element &target) {
        max_error = std::max(max_error, std::abs(target - eigen_matrix(row, col)));
    });
    return max_error; // return as double is OK
}

// Method to test:
// 1. solve_lower_band_mat
// 2. solve_upper_band_mat.
// These two methods solve the equation Lx = b,
// where L is a lower and upper banded matrix and
// b is a Eigen::Matrix matrix.
// The argument L is of type BandedMatrixTemplate.
// The argument b is of type Eigen::Matrix.
// The argument solver is a function pointer pointing
// to either solve_lower_band_mat or solve_upper_band_mat
// depending on which method is under test.
// The argument tolerance specifies the threshold to compare
// two matrix entries to decide if result is close enough
// to expected values.
template<typename Matrix, typename EMatrix, typename solver_type>
void test_solve_lower_or_upper_band_mat(
        Matrix L, EMatrix b, solver_type solver, double tolerance) {
    using namespace banded;
    using namespace banded::testing;
    EMatrix result;
    result.resize(L.dim(), b.cols());
    solver(L, b, &result);
    const auto expected = (to_dense(L).inverse() * b).eval();
    ASSERT_TRUE(result.isApprox(expected, tolerance));
}

// Test that solve_upper_band_mat works.
// The solve_upper_band_band function solves the
// system Lx = b, where L is a upper banded matrix,
// and b is a non-banded Eigen::Matrix vector.
TEST(TEST_SOLVE_UPPER_BAND_MAT, test_solve_upper_band_mat_correct_shape_vector) {
    using namespace banded;
    using Element = double;
    using Matrix = BandedMatrixHolder<Element, false>;
    using EMatrix = Eigen::MatrixXd;
    using banded::testing::from_dense;

    typedef void (*solver_type)(const Matrix&, const EMatrix&, EMatrix*);
    solver_type f = solve_upper_band_mat<const Matrix&, const EMatrix&, EMatrix>;

    // Here we construct L and b manually instead of
    // generate random matrices for them because when
    // we generate random matrices, the method under test
    // runs into numerical instability problems.

    // Construct matrix L.
    EMatrix dense_L;
    dense_L.resize(3, 3);
    dense_L << 1, 0.6,   0,
               0,   1, 0.7,
               0,   0,   1;
    Matrix L = from_dense<double>(dense_L, 0, 1);

    // Construct vector b.
    EMatrix b;
    b.resize(3, 1);
    b << 2, 1, 3;
    test_solve_lower_or_upper_band_mat<Matrix, EMatrix, solver_type>(L, b, f, 1e-8);
}

// Test that solve_upper_band_mat works.
// The solve_upper_band_band function solves the
// system Lx = b, where L is a upper banded matrix,
// and b is a non-banded Eigen::Matrix matrix.
TEST(TEST_SOLVE_UPPER_BAND_MAT, test_solve_upper_band_mat_correct_shape_matrix) {
    using namespace banded;
    using Element = double;
    using Matrix = BandedMatrixHolder<Element, false>;
    using EMatrix = Eigen::MatrixXd;
    using banded::testing::from_dense;

    typedef void (*solver_type)(const Matrix&, const EMatrix&, EMatrix*);
    solver_type f = solve_upper_band_mat<const Matrix&, const EMatrix&, EMatrix>;

    // Here we construct L and b manually instead of
    // generate random matrices for them because when
    // we generate random matrices, the method under test
    // runs into numerical instability problems.

    // Construct matrix L.
    EMatrix dense_L;
    dense_L.resize(4, 4);
    dense_L << 1, 0.6,   0,  0,
            0,   1, 0.7,  0,
            0,   0,   1, 0.8,
            0,   0,   0,   1;

    Matrix L = from_dense<double>(dense_L, 0, 2);

    // Construct matrix b.
    EMatrix b;
    b.resize(4, 2);
    b << 2, 1,
            3, 5,
            4, 3,
            2, 2;
    test_solve_lower_or_upper_band_mat<Matrix, EMatrix, solver_type>(L, b, f, 1e-8);
}


// Test that solve_upper_band_mat throws
// an exception if the L matrix is not a upper banded matrix.
// The method under test is supposed to solve Lx = b.
TEST(TEST_SOLVE_UPPER_BAND_MAT, test_solve_upper_band_mat_incorrect_shape) {
    using namespace banded;
    using Element = double;
    using Matrix = BandedMatrixHolder<Element, false>;
    using EMatrix = Eigen::MatrixXd;
    using banded::testing::from_dense;

    typedef void (*solver_type)(const Matrix&, const EMatrix&, EMatrix*);
    solver_type f = solve_upper_band_mat<const Matrix&, const EMatrix&, EMatrix>;

    // Here we construct L and b manually instead of
    // generate random matrices for them because when
    // we generate random matrices, the method under test
    // runs into numerical instability problems.

    // Construct matrix L.
    EMatrix dense_L;
    dense_L.resize(3, 3);
    dense_L <<  1,   0,  0,
              0.6,   1,  0,
                0, 0.7,  1;
    Matrix L = from_dense<double>(dense_L, 1, 0);

    // Construct matrix b.
    EMatrix b;
    b.resize(3, 1);
    b << 2, 1, 3;
    auto call = test_solve_lower_or_upper_band_mat<Matrix, EMatrix, solver_type>;
    EXPECT_THROW_WITH_MESSAGE(
            {call(L, b, f, 1e-8);},
            std::runtime_error,
            "Left matrix is assumed upper-triangular");
}


// Test that solve_lower_band_mat works.
// The solve_lower_band_mat function solves the
// system Lx = b, where L is a lower banded matrix,
// and b is a non-banded Eigen::Matrix vector.
TEST(TEST_SOLVE_LOWER_BAND_MAT, test_solve_lower_band_mat_correct_shape_vector) {
    using namespace banded;
    using Element = double;
    using Matrix = BandedMatrixHolder<Element, false>;
    using EMatrix = Eigen::MatrixXd;
    using banded::testing::from_dense;

    typedef void (*solver_type)(const Matrix&, const EMatrix&, EMatrix*);
    solver_type f = solve_lower_band_mat<const Matrix&, const EMatrix&, EMatrix>;

    // Here we construct L and b manually instead of
    // generate random matrices for them because when
    // we generate random matrices, the method under test
    // runs into numerical instability problems.

    // Construct matrix L.
    EMatrix dense_L;
    dense_L.resize(3, 3);
    dense_L <<   1,    0,   0,
               0.6,    1,   0,
                 0,  0.7,   1;
    Matrix L = from_dense<double>(dense_L, 1, 0);

    // Construct vector b.
    EMatrix b;
    b.resize(3, 1);
    b << 2, 1, 3;
    test_solve_lower_or_upper_band_mat<Matrix, EMatrix, solver_type>(L, b, f, 1e-8);
}

// Test that solve_lower_band_mat works.
// The solve_lower_band_mat function solves the
// system Lx = b, where L is a lower banded matrix,
// and b is a non-banded Eigen::Matrix matrix.
TEST(TEST_SOLVE_LOWER_BAND_MAT, test_solve_lower_band_mat_correct_shape_matrix) {
    using namespace banded;
    using Element = double;
    using Matrix = BandedMatrixHolder<Element, false>;
    using EMatrix = Eigen::MatrixXd;
    using banded::testing::from_dense;

    typedef void (*solver_type)(const Matrix&, const EMatrix&, EMatrix*);
    solver_type f = solve_lower_band_mat<const Matrix&, const EMatrix&, EMatrix>;

    // Here we construct L and b manually instead of
    // generate random matrices for them because when
    // we generate random matrices, the method under test
    // runs into numerical instability problems.
    EMatrix dense_L;
    dense_L.resize(4, 4);
    dense_L <<   1,   0,    0,   0,
               0.6,   1,    0,   0,
                 0, 0.7,    1,   0,
                 0,   0,  0.8,   1;

    Matrix L = from_dense<double>(dense_L, 2, 0);

    // Construct matrix b.
    EMatrix b;
    b.resize(4, 2);
    b << 2, 1,
         3, 5,
         4, 3,
         2, 2;
    test_solve_lower_or_upper_band_mat<Matrix, EMatrix, solver_type>(L, b, f, 1e-8);
}


// Test that solve_lower_band_mat throws
// an exception if the L matrix is not a lower banded matrix.
// The method under test is supposed to solve Lx = b.
TEST(TEST_SOLVE_LOWER_BAND_MAT, test_solve_lower_band_mat_incorrect_shape) {
    using namespace banded;
    using Element = double;
    using Matrix = BandedMatrixHolder<Element, false>;
    using EMatrix = Eigen::MatrixXd;
    using banded::testing::from_dense;

    typedef void (*solver_type)(const Matrix&, const EMatrix&, EMatrix*);
    solver_type f = solve_lower_band_mat<const Matrix&, const EMatrix&, EMatrix>;

    // Here we construct L and b manually instead of
    // generate random matrices for them because when
    // we generate random matrices, the method under test
    // runs into numerical instability problems.

    // Construct matrix L.
    EMatrix dense_L;
    dense_L.resize(3, 3);
    dense_L << 1, 0.6,    0,
               0,   1,  0.6,
               0,   0,    1;
    Matrix L = from_dense<double>(dense_L, 0, 1);

    // Construct matrix b.
    EMatrix b;
    b.resize(3, 1);
    b << 2, 1, 3;
    auto call = test_solve_lower_or_upper_band_mat<Matrix, EMatrix, solver_type>;
    EXPECT_THROW_WITH_MESSAGE(
            {call(L, b, f, 1e-8);},
            std::runtime_error,
            "Left matrix is assumed lower-triangular");
}

// Method to test:
// 1. solve_lower_band_band
// 2. solve_upper_band_band.
// These two method solves the system Lx = b
// where L, b and x are all banded matrices.
// The argument solver is a function pointer pointing to either
// solve_lower_band_band or solve_upper_band_band, depending
// which method is under test.
// The argument dimensions specifies the dimension and band widths
// of L, b and x.
// The argument tolerance specifies the threshold to compare
// two matrix entries to decide if result is close enough
// to expected values.
template<typename Element, typename solver_type, bool is_lower_triangular=false>
void test_solve_lower_or_upper_band_band(
        solver_type solver,
        std::vector<std::vector<int>> dimensions,
        double tolerance) {
    using namespace banded;
    using Matrix = BandedMatrixHolder<Element, is_lower_triangular>;
    using Matrix2 = BandedMatrixHolder<Element, false>;

    for (const auto &param : dimensions) {
        std::default_random_engine prng(85629372);
        using namespace banded::testing;

        // Construct matrix L and matrix b, they are both banded matrices.
        auto L = random_banded_matrix<Element, is_lower_triangular>(param[1], param[2], param[0], prng);
        auto b = random_banded_matrix<Element, false>(param[3], param[4], param[0], prng);

        // Construct the banded matrix to store result.
        using ResultMatrix = BandedMatrixHolder<Element>;
        ResultMatrix result = zero<Element>(param[5], param[6], b.dim());
        // Calling the function under test.
        solver(L, b, &result);

        const auto expected = (to_dense(L).inverse() * to_dense(b)).eval();
        double error = max_band_error(result, expected);
        EXPECT_LT(error, tolerance);
    }
}

// Test that solve_upper_band_band works.
TEST(TEST_SOLVE_UPPER_BAND_BAND, test_solve_upper_band_band_correct_shape) {
    using namespace banded;
    using Element = double;
    using Matrix = BandedMatrixHolder<Element, false>;

    std::vector<std::vector<int>> dimensions{
            // N, Left pair, right pair, result pair
            {12, 0, 3, 0, 3, 1, 1},
            {12, 0, 2, 0, 1, 1, 1},
    };

    typedef void (*solver_type)(Matrix&, Matrix&, Matrix*);
    solver_type f = solve_upper_band_band<Matrix&, Matrix&, Matrix>;
    test_solve_lower_or_upper_band_band<Element, solver_type>(f, dimensions, 1e-8);
}

// Test that solve_upper_band_band throws an exception
// If the left matrix is not a upper triangular banded matrix.
TEST(TEST_SOLVE_UPPER_BAND_BAND, test_solve_upper_band_band_incorrect_shape) {
    using namespace banded;
    using Element = double;
    using Matrix = BandedMatrixHolder<Element, false>;

    std::vector<std::vector<int>> dimensions{
            // N, Left pair, right pair, result pair
            {12, 1, 0, 0, 3, 1, 1}
    };

    typedef void (*solver_type)(Matrix&, Matrix&, Matrix*);
    solver_type f = solve_upper_band_band<Matrix&, Matrix&, Matrix>;

    // Need to introduce the call local variable because ASSERT_THROW seems
    // to have difficulty passing the template arguments.
    auto call = test_solve_lower_or_upper_band_band<Element, solver_type>;
    EXPECT_THROW_WITH_MESSAGE(
            {call(f, dimensions, 1e-8);},
            std::runtime_error,
            "Left matrix is assumed upper-triangular");
}


// Test that solve_upper_band_band works.
TEST(TEST_SOLVE_LOWER_BAND_BAND, test_solve_lower_band_band_correct_shape) {
    using namespace banded;
    using Element = double;
    using LowerTriangularMatrix = BandedMatrixHolder<Element, true>;
    using Matrix = BandedMatrixHolder<Element, false>;

    std::vector<std::vector<int>> dimensions{
            // N, Left pair, right pair, result pair
            {12, 3, 0, 3, 0, 1, 1},
            {12, 2, 0, 2, 0, 1, 1},
    };

    typedef void (*solver_type)(LowerTriangularMatrix&, Matrix&, Matrix*);
    solver_type f = solve_lower_band_band<LowerTriangularMatrix&, Matrix&, Matrix>;
    test_solve_lower_or_upper_band_band<Element, solver_type, true>(f, dimensions, 1e-8);
}

// Test that solve_lower_band_band throws an exception
// If the left matrix is not a lower triangular banded matrix.
TEST(TEST_SOLVE_UPPER_BAND_BAND, test_solve_lower_band_band_incorrect_shape) {
    using namespace banded;
    using Element = double;

    using Matrix = BandedMatrixHolder<Element, false>;

    // Construct upper triangular matrix as the left matrix
    // for solve_lower_band_band. This should trigger an exception.
    std::vector<std::vector<int>> dimensions{
            // N, Left pair, right pair, result pair
            {12, 0, 1, 0, 3, 1, 1}
    };

    typedef void (*solver_type)(Matrix&, Matrix&, Matrix*);
    solver_type f = solve_lower_band_band<Matrix&, Matrix&, Matrix>;

    // Need to introduce the call local variable because ASSERT_THROW seems
    // to have difficulty passing the template arguments.
    auto call = test_solve_lower_or_upper_band_band<Element, solver_type, false>;
    EXPECT_THROW_WITH_MESSAGE(
            {call(f, dimensions, 1e-8);},
            std::runtime_error,
            "Left matrix is assumed lower-triangular");
}



template<typename Element>
void test_lower_triangular_solve_simple_cases(double tolerance) {
    using namespace banded;

    std::vector<std::vector<int>> dimensions{
            // N, Left pair, right pair, result pair
            {10, 2, 0, 2, 4, 1, 1},
            {12, 2, 0, 2, 4, 1, 1},
            {12, 0, 3, 2, 4, 1, 1},
            {11, 0, 2, 3, 3, 2, 1},
            {11, 2, 0, 3, 3, 2, 1},
    };

    for (const auto &param : dimensions) {
        std::default_random_engine prng(85629372);
        using namespace banded::testing;
        auto l = random_banded_matrix<Element, false>(param[1], param[2], param[0], prng);
        auto b = random_banded_matrix<Element, false>(param[3], param[4], param[0], prng);

        const auto solved = solve_triang_band(l, b, param[5], param[6]);
        const auto checked_solve = (to_dense(l).inverse() * to_dense(b)).eval();
        double error = max_band_error(solved, checked_solve);
        EXPECT_LT(error, tolerance);
    }
}

TEST(TEST_SOLVE_TRIAG_BAND, test_lower_triangular_solve_simple_cases_double) {
    test_lower_triangular_solve_simple_cases<double>(1e-8);
}

TEST(TEST_SOLVE_TRIAG_BAND, test_lower_triangular_solve_simple_cases_float) {
    // Should compile in float, but you really don't want Solve in single precision
    test_lower_triangular_solve_simple_cases<float>(1e-2);
}
