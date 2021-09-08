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
#include <algorithm>
#include <cmath>

#include "banded_matrices/common.hpp"
#include "banded_matrices/cholesky.hpp"
#include "banded_matrices/banded_matrix.hpp"
#include "banded_matrices/unary_broadcastable_op_kernel.hpp"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using Index = Eigen::Index;


template <typename T> using Transposed = banded::Transposed<T>;
template <typename T> using Symmetric = banded::Symmetric<T>;
template <typename T> using BandedMatrix = banded::BandedMatrix<T>;
template <typename T> using BandedMatrixHolder = banded::BandedMatrixHolder<T>;

template <typename T> using Matrix = banded::Matrix<T>;
template <typename T> using MatrixMap = banded::MatrixMap<T>;
template <typename T> using MatrixConstMap = banded::MatrixConstMap<T>;
template <typename T> using Vector = banded::Vector<T>;
template <typename T> using RowVector = banded::RowVector<T>;
template <typename T> using Array = banded::Array<T>;

template <typename T> using LowerTriangularBandedMatrix =
  banded::LowerTriangularBandedMatrix<T>;
template <typename T> using LowerTriangularBandedMatrixHolder =
  banded::LowerTriangularBandedMatrixHolder<T>;


//
// Given a lower-triangular banded matrix L that is the Cholesky of a matrix Q,
// Compute the inverse of Q as a lower-triangular banded matrix with the same
// band as L.
//
template <typename T >
void inverse_from_cholesky(
    Index n, Index k,
    Index result_lower_bandwidth,
    const T* input, T* output) {
  const auto m = k - 1;
  MatrixConstMap<T> L(input, k, n);  // input L such that Q = LL^T
  const RowVector<T> diag(L.row(0));  // diag(L)
  const Matrix<T> &U =  // U^T = L, banded transpose
    (L.array().rowwise() / diag.array()).transpose();
  Matrix<T> S =
    Matrix<T>::Zero(2 * result_lower_bandwidth + 1, n);  // Q^-1
  MatrixMap<T> S_output(output, result_lower_bandwidth + 1, n);

  auto last = n - 1;
  for (Index i = last; i >= 0; --i) {
    auto j_beg = (i == last) ? 1 : 0;
    auto j_end = std::min(i + 1, result_lower_bandwidth + 1);
    S(result_lower_bandwidth, i) = static_cast<T>(1.) / (diag[i] * diag[i]);
    for (auto j = j_beg; j < j_end; ++j) {
      // h - vector height
      // l - row Index in S upper triangular
      // i - col Index in S upper triangular
      // (l, i) equal to (j + m, i - j) in S lower triangular.
      auto h = std::min(last - i + j, m);
      auto l = result_lower_bandwidth - j;
      const auto&& s = U.block(i - j, 1, 1, h) * S.block(l + 1, i, h, 1);
      S(j + result_lower_bandwidth, i - j) -= s(0, 0);
      S(l, i) = S(j + result_lower_bandwidth, i - j);
    }
  }

  S_output = S.block(result_lower_bandwidth, 0, result_lower_bandwidth + 1, n);
}


//
// Main "backward" step of the calculation of
// the gradient of the `InverseFromCholeskyBandOp` operation.
//
template <typename T>
void cholesky_grad_main_backward_step(
    const Index n,
    const Index k,  // full row count of the L matrix
    const Transposed<LowerTriangularBandedMatrixHolder<T>>& U,
    const Symmetric<LowerTriangularBandedMatrix<T>>& S,
    Vector<T>* bvec_inv_2_ptr,
    BandedMatrixHolder<T>* bU_ptr,
    BandedMatrixHolder<T>* bS_ptr) {
  auto& bvec_inv_2 = *bvec_inv_2_ptr;
  auto& bU = *bU_ptr;
  auto& bS = *bS_ptr;
  const auto S_lower_bandwidth_ = bS.lower_bandwidth();

  // Beginning of backward pass
  for (Index j = 0; j < n; ++j) {
    Index i = std::max(Index(0), j - S_lower_bandwidth_);
    for (; i < j + 1; ++i) {
      if (i == j) {
        bvec_inv_2(i) += bS(i, i);
      }

      // Grad of: S[j, i] = S[i, j]
      const auto tmp = bS(j, i);
      bS(j, i) = 0;
      bS(i, j) += tmp;

      const T bS_i_j = bS(i, j);
      const Index end_x = std::min(n, i + k);

      // TODO(optim): any optimization on this loop is important
      // - S could be stored only as lower,
      //   but then the loop on bU below needs care
      // - S and U are accessed row-wise, and should maybe stored differently
      //   (U is, in effect, as it is a transposed<Something-Column-Major>?)
      // - Any way to vectorize?

      // Grad of: S[i, j] = -np.sum(U[i, i+1:i+k] * S[i+1:i+k, j])
      // bU[i, i+1:i+k] -= S[i+1:i+k, j] * bS[i, j]
      for (Index x = i+1; x < end_x; ++x)
        bU(i, x) -= S(x, j) * bS_i_j;
      // bS[i+1:i+k, j] -= U[i, i+1:i+k] * bS[i, j]
      for (Index x = i+1; x < end_x; ++x)
        bS(x, j) -= U(i, x) * bS_i_j;

      bS(i, j) = 0;
    }
  }
}


//
// Function that implements the core logic for
// the gradient of the `InverseFromCholeskyBandOp` operation.
//
// Note that this operator could not be derived analytically and
// code for it has been obtained by
// - applying the automatic differentiation tool tangent
//   (https://github.com/google/tangent)
//   to the Python prototype code of the forward evaluation of this operator
// - simplifying the generated Python code by hand (James Hensman)
// - converting the generated Python code to C++.
//
// See subset_inverse_grad.py in the research sandbox for banded ops
// for any related prototype code.
//
template <typename T>
void gradient_of_inverse_from_cholesky(
    const LowerTriangularBandedMatrix<T>& L,
    const LowerTriangularBandedMatrix<T>& S_lower_band,
    const LowerTriangularBandedMatrix<T>& G,
    LowerTriangularBandedMatrix<T>* bL_ptr) {
  using LowerMat = LowerTriangularBandedMatrix<T>;
  using LowerMatHolder = LowerTriangularBandedMatrixHolder<T>;
  using BandedMatHolder = BandedMatrixHolder<T>;

  auto& bL = *bL_ptr;
  const auto L_lower_bandwidth_ = L.lower_bandwidth();
  const auto k = L_lower_bandwidth_ + 1;
  const auto n = L.dim();
  const auto S_lower_bandwidth_ = S_lower_band.lower_bandwidth();

  assert(n == S_lower_band.dim() && n == G.dim());
  assert(S_lower_bandwidth_ == G.lower_bandwidth());

  // We get the lower band of S representing the symmetric matrix we want:
  Symmetric<LowerMat> S(S_lower_band);

  // Copy of G that is mutated by the algorithm;
  // Importantly this matrix is symmetric:
  BandedMatHolder bS =
    banded::zero<T, false>(S_lower_bandwidth_, S_lower_bandwidth_, n);

  G.for_each_in_band([&bS, &G](Index row, Index col, T) {
    const T val = G(row, col);
    bS(row, col) = val;
    bS(col, row) = val;
  });

  // vec = np.diag(L)
  Vector<T> vec = L.underlying_dense_matrix().row(0);

  // U = (L / vec).T
  auto Ut = banded::zero<T, true>(L_lower_bandwidth_, 0, n);
  Ut.for_each_in_band([&Ut, &L, &vec](Index row, Index col, T&) {
    Ut(row, col) = L(row, col) / vec(col);
  });
  const Transposed<LowerMatHolder> U(Ut);

  // bU = np.zeros_like(U)
  BandedMatHolder bU = banded::zero<T, false>(0, L_lower_bandwidth_, n);;
  // bvec_inv_2 = np.zeros(n)
  Vector<T> bvec_inv_2 {L.dim()};
  bvec_inv_2.setZero();

  // Beginning of backward pass
  cholesky_grad_main_backward_step<T>(n, k, U, S, &bvec_inv_2, &bU, &bS);

  // Grad of: U = np.transpose(L * vec_inv)
  // bL = bU.T / vec
  bL.underlying_dense_matrix().setZero();
  bL.for_each_in_band([&bL, &bU, &vec](Index row, Index col, T&) {
    // Note the transposed indices on bU compared to reference code
    bL(row, col) = bU(col, row) / vec(col);
  });

  // Grad of: vec_inv_2 = 1.0 / vec ** 2
  //  bvec = -2. * bvec_inv_2 / vec ** 3
  Array<T> bvec = -2 * bvec_inv_2.array() / vec.array().pow(3);

  // Grad of: vec_inv = 1.0 / vec
  // bvec -= np.sum(bU.T * L, 0) / (vec ** 2)
  // TODO(pm): The line below requires attention; it's been discussed by
  // TODO(pm): @bedder and @lucas; the bUt object is copy-constructed so
  // TODO(pm): we are convinced there is no memory issue in particular
  // TODO(pm): we discussed (and discarded) risks of aliasing a temporary.
  // TODO(pm): It is possible however that one unnecessary copy is not
  // TODO(pm): optimized away.
  Matrix<T> bUt =
    BandedMatHolder(Transposed<BandedMatHolder>(bU)).underlying_dense_matrix();
  bvec -=
    (bUt.array()
    * L.underlying_dense_matrix().array()).colwise().sum().transpose()
    / (vec.array().pow(2));

  // Grad of: vec = diag(L)
  //  bL += np.diag(bvec)
  bL.underlying_dense_matrix().row(0) += bvec.matrix();
}


//
// Operator for the `inverse from Cholesky` operation:
// Given a lower-banded matrix L that is assumed to be the Cholesky
// decomposition of a (symmetric, Positive Definite) matrix Q = LL^T,
// compute the inverse of Q.
// Only the lower band of this symmetric matrix is returned.
//
template <typename T>
class InverseFromCholeskyBandOp : public UnaryBroadcastableOpKernel<T, 2> {
 public:
  explicit InverseFromCholeskyBandOp(OpKernelConstruction* context) :
      UnaryBroadcastableOpKernel<T, 2>(context) {
    LOAD_ATTRIBUTE_OP(
      context,
      "result_lower_bandwidth",
      &result_lower_bandwidth_);
  }

  void StartChecks(OpKernelContext *context,
                   const TensorShape& unit_input_shape) override {
    // Check dimensions and parameters
    const Index k = unit_input_shape.dim_size(0);
    OP_REQUIRES(context,
      result_lower_bandwidth_ >= k - 1,
      errors::InvalidArgument(
        "Results of inverse from Cholesky need to have"
        "bandwidth at least equal to the input's."));

    OP_REQUIRES(context,
      TensorShapeUtils::IsMatrix(unit_input_shape),
      errors::InvalidArgument(
        "MatrixInverseBandedOp operation expects a matrix."));

    OP_REQUIRES(context,
      unit_input_shape.dim_size(0) <= unit_input_shape.dim_size(1),
      errors::InvalidArgument(
        "MatmulVectorBanded operation expects a banded matrix."));
  }

  void UnitOutputShape(const TensorShape&  unit_input_shape,
                              TensorShape * unit_output_shape) override {
    const Index n = unit_input_shape.dim_size(1);

    unit_output_shape->Clear();
    unit_output_shape->AddDim(result_lower_bandwidth_+1);
    unit_output_shape->AddDim(n);
  }

  void UnitCompute(const std::vector<Tensor>& unit_input_tensors,
                   Tensor* unit_output_tensor) override {
    // View the tensors as banded matrices:
    const Tensor& input_tensor = unit_input_tensors[0];
    const TensorShape& input_shape = input_tensor.shape();

    const Index k = input_shape.dim_size(0);
    const Index n = input_shape.dim_size(1);

    inverse_from_cholesky(n, k, result_lower_bandwidth_,
                          input_tensor.flat<T>().data(),
                          unit_output_tensor->flat<T>().data());
  }

  void  ResultsChecks(OpKernelContext *,
                      const std::vector<Tensor>& , const Tensor&) override {}


 private:
    int result_lower_bandwidth_;
};


//
// TensorFlow operator for the gradient
// of the `InverseFromCholeskyBandOp` operation.
//
template <typename T>
class GradientOfInverseFromCholeskyBandOp :
     public UnaryBroadcastableOpKernel<T, 2> {
 public:
  explicit GradientOfInverseFromCholeskyBandOp(OpKernelConstruction* context)
    : UnaryBroadcastableOpKernel<T, 2>(context) {
  }

  void StartChecks(OpKernelContext *context,
                   const TensorShape& unit_input_shape) override {
    const TensorShape& s_shape = context->input(1).shape();
    const TensorShape& g_shape = context->input(2).shape();

    int input_dims = unit_input_shape.dims();
    int g_dims = g_shape.dims();

    OP_REQUIRES(context,
      TensorShapeUtils::IsMatrix(unit_input_shape),
      errors::InvalidArgument(
        "GradientOfInverseFromCholeskyBand inputs should be matrices."));

    OP_REQUIRES(context,
      s_shape == g_shape &&
      unit_input_shape.dim_size(input_dims -1) == s_shape.dim_size(g_dims -1) &&
      unit_input_shape.dim_size(input_dims -1) == g_shape.dim_size(g_dims -1),
      errors::InvalidArgument(
        "All 3 matrices in GradientOfInverseFromCholeskyBand"
        "should have same shape."));
  }

  void UnitOutputShape(const TensorShape&  unit_input_shape,
                              TensorShape * unit_output_shape) override {
    unit_output_shape->Clear();
    unit_output_shape->AppendShape(unit_input_shape);
  }

  void UnitCompute(const std::vector<Tensor>& unit_input_tensors,
                   Tensor* unit_output_tensor) override {
    const Tensor& l_tensor = unit_input_tensors[0];
    const Tensor& s_tensor = unit_input_tensors[1];
    const Tensor& g_tensor = unit_input_tensors[2];

    const auto k = l_tensor.shape().dim_size(0);
    const auto s = s_tensor.shape().dim_size(0);
    const auto n = l_tensor.shape().dim_size(1);


    LowerTriangularBandedMatrix<T> result{
      unit_output_tensor->flat<T>().data(), k-1, 0, n };

    // We modify the gradient to take into account symmetry:
    Matrix<T> adjusted_gradient =
      MatrixConstMap<T>(g_tensor.flat<T>().data(), s, n);
    adjusted_gradient.bottomRows(s - 1) *= 0.5;

    // Get 3 const views on the inputs:

    // Input of forward mode: the lower-band L as in Cholesky LL^T
    const auto L = banded::const_lower_triangular_view(
      l_tensor.flat<T>().data(), k, n);
    // Output of forward mode: S = band(inv(L @ L.T))
    const auto S = banded::const_lower_triangular_view(
      s_tensor.flat<T>().data(), s, n);
    // Gradient given for S by the backprop
    const auto G = banded::const_lower_triangular_view(
      adjusted_gradient.data(), s, n);

    // Algorithm
    assert(result.lower_bandwidth() == k-1 && result.upper_bandwidth() == 0
      && result.dim() == n);
    gradient_of_inverse_from_cholesky(L, S, G, &result);
  }

  void  ResultsChecks(OpKernelContext *,
                      const std::vector<Tensor>& , const Tensor&) override{};
};


//
// Operator registration
//

using InferenceContext = ::tensorflow::shape_inference::InferenceContext;
using DimensionHandle = ::tensorflow::shape_inference::DimensionHandle;

REGISTER_OP("InverseFromCholeskyBand")
  .Attr("T: {float, double}")
  .Input("banded_matrix: T")
  .Attr("result_lower_bandwidth: int")
  .Output("inverse_banded_matrix: T")
  .SetShapeFn([](InferenceContext* c) {
    int result_lower_bandwidth;
    LOAD_ATTRIBUTE_RETURN_IF_ERROR(c,
      "result_lower_bandwidth", &result_lower_bandwidth);

    shape_inference::ShapeHandle leading_dims;
    TF_RETURN_IF_ERROR(c->Subshape(c->input(0), 0, -2, &leading_dims));

    DimensionHandle dim = c->Dim(c->input(0), -1);
    shape_inference::ShapeHandle mat = c->Matrix(
      result_lower_bandwidth + 1, dim);

    shape_inference::ShapeHandle out;
    TF_RETURN_IF_ERROR(c->Concatenate(leading_dims, mat, &out));

    c->set_output(0, out);
    return Status::OK();
  });

REGISTER_OP("GradientOfInverseFromCholeskyBand")
  .Attr("T: {float, double}")
  .Input("chol_input_band: T")
  .Input("inv_output_band: T")
  .Input("grad_band: T")
  .Output("inverse_banded_matrix: T")
  .SetShapeFn([](InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });

REGISTER_CPU(InverseFromCholeskyBand, float);
REGISTER_CPU(InverseFromCholeskyBand, double);

REGISTER_CPU(GradientOfInverseFromCholeskyBand, float);
REGISTER_CPU(GradientOfInverseFromCholeskyBand, double);

}  // end of namespace tensorflow
