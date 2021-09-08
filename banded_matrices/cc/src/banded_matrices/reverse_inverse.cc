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
// Authors - Artem Artemev @awav,
//           Vincent Adam @vincentadam87,
//           Stefanos Eleftheriadis @stefanosele.

#include <algorithm>
#include <cmath>

#include "Eigen/QR"
// #include "third_party/eigen3/Eigen/QR"

#include "banded_matrices/banded_matrix.hpp"
#include "banded_matrices/common.hpp"
#include "banded_matrices/unary_broadcastable_op_kernel.hpp"



namespace tensorflow {

template <typename T> using BandedMatrix = banded::BandedMatrix<T>;
template <typename T> using Matrix = banded::Matrix<T>;
template <typename T> using MatrixMap = banded::MatrixMap<T>;
template <typename T> using MatrixConstMap = banded::MatrixConstMap<T>;
template <typename T> using Vector = banded::Vector<T>;
using index = Eigen::Index;


// From the banded matrix `banded_input`, viewed here as a dense map,
// extract the square block of size band*band
// whose top-left corner is the entry at position (start, start)
// on the banded matrix'diagonal.
template <typename T>
void extract_diagonal_block(
    const MatrixConstMap<T>& banded_input, Matrix<T>* output,
    index start, index band) {
  output->setZero();
  for (int r = 0; r < band; ++r) {
      for (int c = 0; c < band - r; ++c) {
          (*output)(r + c, c) = banded_input(r, start + c);
          (*output)(c, r + c) = banded_input(r, start + c);
      }
  }
}


//
// Functor that implements the core logic of `ReverseInverseFromCholeskyBandOp`:
// Computes the Cholesky L of subset inverse S = (LLᵀ)⁻¹.
//
template <typename T>
struct ReverseInverseFromCholeskyBandFunctor {
  void operator()(index num_band, index side_len,
                  const T* input_mat, T* output_mat) {
    // S -> L
    // :param S: sparse subset inverse of banded matrix L
    // :param l: number of subdiagonals in S
    // :return: Ls: reconstructed cholesky decomposition
    // """
    // # forward pass
    // k = l + 1  # bandwidth
    // n = S.shape[1]
    // # construct vector e = [1, 0, ..., 0]
    // V = np.zeros_like(S)
    // e = np.zeros((k))
    // e[0] = 1
    // for i in range(n):
    //     j = i + k
    //     chol_S = np.linalg.cholesky(S[i:j, i:j])
    //     V[i:j, i] = cho_solve((chol_S, True), e[:n-i])
    // Ls = V / np.sqrt(np.diag(V)[None, :])
    // return Ls

    MatrixConstMap<T> S(input_mat, num_band, side_len);
    MatrixMap<T> V(output_mat, num_band, side_len);
    V.setZero();
    Vector<T> E = Vector<T>::Zero(num_band);
    E(0) = 1;
    Vector<T> tmp = Vector<T>::Zero(num_band);
    Matrix<T> P = Matrix<T>::Zero(num_band, num_band);
    for (Eigen::Index i = 0; i < side_len; ++i) {
      // b - block size
      auto b = (i + num_band <= side_len) ? num_band : side_len - i;

      // Get P matrix
      extract_diagonal_block(S, &P, i, b);
      tmp.segment(0, b) = P.block(0, 0, b, b).ldlt().solve(E.head(b));
      tmp /= static_cast<T>(sqrt(tmp(0)));
      V.block(0, i, b, 1) = tmp.segment(0, b);
    }
  }

  void gradient(index num_band, index side_len,
                const T* input_mat,
                const T* output_mat,
                const T* output_grad,
                T* target_grad) {
    // """
    // S -> L
    // bL -> bS
    // :param bS:
    // :param L:
    // :param l: number of subdiagonals in S
    // :return: Ls: reconstructed cholesky decomposition
    // """
    // # forward pass
    // k = l + 1  # bandwidth
    // n = S.shape[1]
    // Vr = Ls * np.diag(Ls)[None, :]

    // # backward pass
    // bS = np.zeros_like(bL)
    // for i in range(n):
    //     j = i + k
    //     bLi = bL[i:j, i]
    //     chol_S = np.linalg.cholesky(S[i:j, i:j])
    //     Hi = np.eye(min(n-i, k))
    //     Hi[:, 0] -= Ls[i:j, i] / (2. * Ls[i, i])
    //     Hi /= Ls[i, i]

    //     tmp = (bLi.T @ Hi).T
    //     tmp2 = cho_solve((chol_S, True), tmp)

    //     bSi = -Vr[i:j, i:(i+1)] @ tmp2[None]
    //     bS[i:j, i:j] += .5 * (bSi + bSi.T)
    // return bS

    MatrixConstMap<T> S(input_mat, num_band, side_len);
    MatrixConstMap<T> L(output_mat, num_band, side_len);
    MatrixConstMap<T> bL(output_grad, num_band, side_len);
    MatrixMap<T> bS(target_grad, num_band, side_len);
    bS.setZero();
    Matrix<T> P = Matrix<T>::Zero(num_band, num_band);

    for (int i = 0; i < side_len; ++i) {
      auto b = (i + num_band <= side_len) ? num_band : side_len - i;
      Matrix<T> Hi = Matrix<T>::Identity(b, b);

      auto d = L(0, i);
      auto Li = L.block(0, i, b, 1);  // it's a vector
      Hi.col(0) -= Li * T(0.5) / d;
      Hi /= d;

      extract_diagonal_block(S, &P, i, b);
      Matrix<T> M = Hi.transpose() * bL.block(0, i, b, 1);
      Matrix<T> sM = P.block(0, 0, b, b).ldlt().solve(M);
      Matrix<T> bSi = - (Li * d) * sM.transpose();

      // Writing the output and symmetrise bSi
      for (int r = 0; r < b; ++r) {
          for (int c = 0; c < b - r; ++c) {
              bS(r, i + c) += T(.5) * bSi(r + c, c);
              bS(r, i + c) += T(.5) * bSi(c, r + c);
          }
      }
    }
  }
};


//
// Operator that computes the Cholesky L of subset inverse S = (LLᵀ)⁻¹.
//
template <typename T>
class ReverseInverseFromCholeskyBandOp :
    public UnaryBroadcastableOpKernel<T, 2> {
 public:
  explicit ReverseInverseFromCholeskyBandOp(OpKernelConstruction* context):
      UnaryBroadcastableOpKernel<T, 2>(context) {
    LOAD_ATTRIBUTE_OP(context, "bandwidth", &bandwidth_);
  }

  void StartChecks(OpKernelContext *context,
                   const TensorShape& unit_input_shape) override {
    // Check dimensions and parameters
    auto rows = unit_input_shape.dim_size(0);

    OP_REQUIRES(context,
      TensorShapeUtils::IsMatrix(unit_input_shape),
      errors::InvalidArgument(
        "ReverseInverseFromCholeskyBandOp expects a matrix."));

    OP_REQUIRES(
      context,
      bandwidth_ == rows,
      errors::InvalidArgument(
        "ReverseInverseFromCholeskyBandOp expects a matrix with "
        "bandwidth less or equal to major matrix size."));
  }

  void UnitOutputShape(const TensorShape&  unit_input_shape,
                              TensorShape * unit_output_shape) override {
    auto rows = unit_input_shape.dim_size(0);
    auto cols = unit_input_shape.dim_size(1);

    unit_output_shape->Clear();
    unit_output_shape->AddDim(rows);
    unit_output_shape->AddDim(cols);
  }

  void UnitCompute(const std::vector<Tensor>& unit_input_tensors,
                   Tensor* unit_output_tensor) override {
    // View the tensors as banded matrices:
    const Tensor& input_tensor = unit_input_tensors[0];
    auto cols = input_tensor.shape().dim_size(1);

    ReverseInverseFromCholeskyBandFunctor<T>()(
      bandwidth_,
      cols,
      input_tensor.flat<T>().data(),
      unit_output_tensor->flat<T>().data());
  }

  void  ResultsChecks(OpKernelContext *,
                      const std::vector<Tensor>& , const Tensor&) override {}

 private:
  int bandwidth_;
};


//
// Operator that implements the gradient of `ReverseInverseFromCholeskyBandOp`.
//
template <typename T>
class ReverseInverseFromCholeskyBandGradOp :
    public UnaryBroadcastableOpKernel<T, 2> {
 public:
  explicit ReverseInverseFromCholeskyBandGradOp(OpKernelConstruction* context):
      UnaryBroadcastableOpKernel<T, 2>(context) {
    LOAD_ATTRIBUTE_OP(context, "bandwidth", &bandwidth_);
  }

  void StartChecks(OpKernelContext *context,
                   const TensorShape& unit_input_shape) override {
    auto rows = unit_input_shape.dim_size(0);

    OP_REQUIRES(context,
      TensorShapeUtils::IsMatrix(unit_input_shape),
      errors::InvalidArgument(
        "ReverseInverseFromCholeskyBandGradOp expects a matrix."));

    OP_REQUIRES(
      context,
      bandwidth_ == rows,
      errors::InvalidArgument(
        "ReverseInverseFromCholeskyBandGradOp expects a matrix with "
        "bandwidth less or equal to major matrix size."));
  }

  void UnitOutputShape(const TensorShape&  unit_input_shape,
                              TensorShape * unit_output_shape) override {
    unit_output_shape->Clear();
    unit_output_shape->AppendShape(unit_input_shape);
  }

  void UnitCompute(const std::vector<Tensor>& unit_input_tensors,
                   Tensor* unit_output_tensor) override {
    const Tensor& input_tensor = unit_input_tensors[0];
    const Tensor& output_tensor = unit_input_tensors[1];
    const Tensor& gradient_tensor = unit_input_tensors[2];
    const TensorShape& input_shape = input_tensor.shape();

    auto cols = input_shape.dim_size(1);

    ReverseInverseFromCholeskyBandFunctor<T>().gradient(
      bandwidth_,
      cols,
      input_tensor.flat<T>().data(),
      output_tensor.flat<T>().data(),
      gradient_tensor.flat<T>().data(),
      unit_output_tensor->flat<T>().data());
  }

  void  ResultsChecks(OpKernelContext *,
                      const std::vector<Tensor>& , const Tensor&) override{};


 private:
  int bandwidth_;
};


//
// Operator registration
//

REGISTER_OP("ReverseInverseFromCholeskyBand")
  .Attr("T: {float, double}")
  .Attr("bandwidth: int >= 0")
  .Input("input: T")
  .Output("output: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });


REGISTER_OP("ReverseInverseFromCholeskyBandGrad")
  .Attr("T: {float, double}")
  .Attr("bandwidth: int >= 0")
  .Input("input: T")
  .Input("output: T")
  .Input("output_grad: T")
  .Output("grad: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });


REGISTER_CPU(ReverseInverseFromCholeskyBand, float);
REGISTER_CPU(ReverseInverseFromCholeskyBand, double);
REGISTER_CPU(ReverseInverseFromCholeskyBandGrad, float);
REGISTER_CPU(ReverseInverseFromCholeskyBandGrad, double);

}  // namespace tensorflow
