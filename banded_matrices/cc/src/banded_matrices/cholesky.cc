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
#include "banded_matrices/cholesky.hpp"
#include <algorithm>

#include <string>
#include <sstream>

#include "Eigen/Cholesky"

#include "banded_matrices/common.hpp"
#include "banded_matrices/product.hpp"
#include "banded_matrices/unary_broadcastable_op_kernel.hpp"
#include "tensorflow/core/platform/default/logging.h"

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template<typename M>
using Transposed = banded::Transposed<M>;

template<typename T>
using LowerTriangularBandedMatrix = banded::LowerTriangularBandedMatrix<T>;


namespace banded {

//
// Functor that implements the core logic for the
// Cholesky decomposition of a banded matrix.
//
template <typename T>
struct CholeskyBandFunctor<CPUDevice, T> {
  void operator()(Eigen::Index length, Eigen::Index bandwidth, T *inout) {
    auto b = bandwidth;
    const auto k = bandwidth;
    banded::MatrixMap<T> mat(inout, k, length);
    banded::Matrix<T> bl = banded::Matrix<T>::Zero(k, k);
    banded::Matrix<T> br = banded::Matrix<T>::Zero(k, k);
    banded::Matrix<T> tl;
    // for all diagonal blocks, each of size k ...
    for (Eigen::Index s = 0; s < length; s += k) {
      if (s + k > length) {
        // this is the last block and it's shorter than k,
        // resize the bl and br matrices
        b = length - s;
        bl = Matrix<T>::Zero(b, k);
        br = Matrix<T>::Zero(b, b);
      }

      // copy the diagonal block into the dense matrix br
      // TODO(@awav): br.template triangularView<Eigen::Lower>() =
      //   mat.block(s, 0, k, k).template
      //     triangularView<Eigen::Upper>().transpose();
      // would be much more efficient, figure out how to do it.
      for (auto j = 0; j < b; ++j) {
        br.block(j, j, b - j, 1) = mat.block(0, s + j, b - j, 1);
      }

      if (s > 0) {
        // copy the adjacent block on the left into the dense matrix bl
        for (Eigen::Index j = 1; j < k; ++j) {
          auto l = std::min(j, b);
          bl.block(0, j, l, 1) = mat.block(k - j, s - k + j, l, 1);
        }

        // Solve Aᵀ * x = b,
        // where `Aᵀ = tlᵀ`, `b = bl` and x = bl(solved in-place)
        tl.transpose()
          .template triangularView<Eigen::Upper>()
          .template solveInPlace<Eigen::OnTheRight>(bl);

        // write L(bl) back to mat
        for (Eigen::Index j = 1; j < k; ++j) {
          auto l = std::min(j, b);
          mat.block(k - j, s - k + j, l, 1) = bl.block(0, j, l, 1);
        }

        // update mat(br) = mat(br) - L(bl) * L(bl)T
        br.noalias() -= bl * bl.transpose();
      }

      // perform dense Cholesky on mat(br) in-place
      // and store the results in L(tl)
      Eigen::LLT<Eigen::Ref<Matrix<T>>> llt(br);
      tl = std::move(llt.matrixL());

      // write L(tl) back to mat
      for (auto j = 0; j < b; ++j) {
        mat.block(0, s + j, b - j, 1) = tl.block(j, j, b - j, 1);
      }
    }
  }
};

}  // end of namespace banded


namespace tensorflow {

//
// TensorFlow operator for the Cholesky decomposition of a banded matrix.
// The input is the lower-triangular half of a symmetric banded matrix
// (assumed Positive Definite).
// The output is a lower-triangular banded matrix of the same dimensions.
//
template <typename Device, typename T>
class CholeskyBandOp : public UnaryBroadcastableOpKernel<T, 2> {
 public:
  explicit CholeskyBandOp(OpKernelConstruction* context)
      : UnaryBroadcastableOpKernel<T, 2>(context) {
    LOAD_ATTRIBUTE_OP(context, "should_check_result", &should_check_result_);
    LOAD_ATTRIBUTE_OP(context, "relative_tolerance", &relative_tolerance_);
    LOAD_ATTRIBUTE_OP(context, "absolute_tolerance", &absolute_tolerance_);
  }
  void StartChecks(OpKernelContext *context,
                   const TensorShape& unit_input_shape) {
    OP_REQUIRES(context,
      TensorShapeUtils::IsMatrix(unit_input_shape),
      errors::InvalidArgument("CholeskyBandOp expects a matrix."));
  }

  void UnitOutputShape(const TensorShape&  unit_input_shape,
                              TensorShape * unit_output_shape) override {
    unit_output_shape->Clear();
    unit_output_shape->AppendShape(unit_input_shape);
  }

  void UnitCompute(const std::vector<Tensor>& unit_input_tensors,
                   Tensor* unit_output_tensor) override {
    auto unit_input_tensor = unit_input_tensors[0];
    auto k = unit_input_tensor.dim_size(0);
    auto n = unit_input_tensor.dim_size(1);

    // NOTE avoid output->CopyFrom here, as CopyFrom will share the actual
    // storage, effectively mutating the unit_input_tensor
    // https://www.tensorflow.org/versions/r1.0/api_docs/cc/class/tensorflow/tensor
    // "This tensor shares other's underlying storage."
    std::copy_n(unit_input_tensor.flat<T>().data(),
                k * n, unit_output_tensor->flat<T>().data());

    banded::CholeskyBandFunctor<CPUDevice, T>()(
      n, k, unit_output_tensor->flat<T>().data());
  }

  void ResultsChecks(OpKernelContext *context,
                     const std::vector<Tensor>& unit_input_tensors,
                     const Tensor& unit_output_tensor) override {
    // Verify that reconstructed matrix LLᵀ is close enough to unit_input_tensor
    auto unit_input_tensor = unit_input_tensors[0];
    auto k = unit_input_tensor.dim_size(0);
    auto n = unit_input_tensor.dim_size(1);
    if (should_check_result_) {
      check_result_stability(
        banded::const_lower_triangular_view(
          unit_output_tensor.flat<T>().data(), k, n),
        banded::const_lower_triangular_view(
          unit_input_tensor.flat<T>().data(), k, n),
        relative_tolerance_, absolute_tolerance_, context);
    }
  }

  // Check that the computed L is correct in the sense that
  // LLᵀ is close enough to the original input.
  // Arguments:
  // L: the lower triangular banded matrix from the Cholesky operation.
  // input: the original banded matrix that is decomposed into LLᵀ.
  // relativeTolerance, absoluteTolerance: The relative and absolute tolerance
  //    used to decide whether two matrix entries are close enough.
  //    To decide if two matrix entries are close enough, use the same semantics
  //    as in numpy.allclose.
  //    To decide if two matrix entries are close enough, use the same semantics
  //    as in numpy.allclose. numpy.allclose uses the following predicate to
  //    decide if a new value is close enough to an actual value,
  //    where || stands for the absolute function:
  //
  //    |new - actual| <= absolute_tolerance + relative_tolerance * |actual|
  //
  //    When the predicate evaluates to True, new and actual are considered
  //    close enough, otherwise, not close enough.
  //
  //    You can find full definition of allclose at:
  // https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html.
  //    If two corresponding matrix entries are not close enough,
  //    an exception is raised.
  // context: Tensorflow context.
  void check_result_stability(const LowerTriangularBandedMatrix<T>& L,
              const LowerTriangularBandedMatrix<T>& input,
              double relative_tolerance,
              double absolute_tolerance, OpKernelContext* context) {
    using Index = Eigen::Index;
    const auto k = L.lower_bandwidth();
    const auto n = L.dim();

    auto result = banded::zero<T, true>(k, 0, n);
    banded::product_band_band(
      L, Transposed<LowerTriangularBandedMatrix<T>>(L), &result);

    // 0 means no error; 1 means stability check failed.
    int errorKind = 0;
    double absolute_error = 0;
    double actual = 0;
    double threshold = 0;
    Index failed_row_id = 0;
    Index failed_col_id = 0;
    double failed_target = 0;
    double failed_actual = 0;
    double failed_threshold = 0;
    double failed_absolute_error = 0;

    result.for_each_in_band([&input, &relative_tolerance, &absolute_tolerance,
                              &errorKind, &absolute_error, &actual, &threshold,
                              &failed_row_id, &failed_col_id,
                              &failed_target, &failed_actual,
                              &failed_threshold, &failed_absolute_error](
        Index row, Index col, const T& target) {
            // Calculate numerical difference between the actual
            // input matrix entry and corresponding entry from reconstructed
            // matrix LLᵀ.
            // Use the same formula in numpy.allclose:
            // https://docs.scipy.org/doc/numpy/reference/generated/
            // numpy.allclose.html.
            actual = input(row, col);
            absolute_error = std::abs(target - actual);
            threshold =
                absolute_tolerance + relative_tolerance * std::abs(actual);
            if (absolute_error > threshold) {
                // Record the first threshold failure.
                if (errorKind == 0) {
                    errorKind = 1;
                    failed_row_id = row;
                    failed_col_id = col;
                    failed_target = target;
                    failed_actual = actual;
                    failed_threshold = threshold;
                    failed_absolute_error = absolute_error;
                }
            }
    });

    if (errorKind == 1) {
        std::ostringstream msg;
        msg << "Banded Cholesky decomposition failed at matrix entry ("
            << failed_row_id <<", " << failed_col_id << "). Original entry is: "
            << failed_actual << ". Reconstructed entry is " << failed_target
            << ". Absolute error is : " << failed_absolute_error
            << ". Threshold is: " << failed_threshold << ".";
        OP_REQUIRES(context, false, errors::Internal(msg.str()));
    }
  }

 private:
  // Whether to check numerical stability of
  // Cholesky decomposition result.
  bool should_check_result_;

  // Relative tolerance to decide if entries in reconstructed LLᵀ is
  // close enough to corresponding entries in the original matrix.
  float relative_tolerance_;

  // Absolute tolerance to decide if entries in reconstructed LLᵀ is
  // close enough to corresponding entries in the original matrix.
  float absolute_tolerance_;
};


//
// Gradient of the Cholesky operator;
// See:
//   Iain Murray.
//   Differentiation of the Cholesky decomposition.
//   arXiv preprint arXiv:1602.07527, 2016
//
template <typename Device, typename T>
class CholeskyBandGradOp : public UnaryBroadcastableOpKernel<T, 2> {
 public:
  explicit CholeskyBandGradOp(OpKernelConstruction* context) :
    UnaryBroadcastableOpKernel<T, 2>(context) { }

  void StartChecks(OpKernelContext *context,
                   const TensorShape& unit_input_shape) override {
    // Check dimensions and parameters
    const Tensor& grad_tensor = context->input(0);
    const Tensor& input_tensor = context->input(1);

    TensorShape grad_shape = grad_tensor.shape();
    TensorShape input_shape = input_tensor.shape();

    OP_REQUIRES(context,
      TensorShapeUtils::IsMatrix(unit_input_shape),
      errors::InvalidArgument(
        "CholeskyBandGradOp expects a matrix for gradient."));

    auto k = grad_shape.dim_size(0);
    auto n = grad_shape.dim_size(1);

    OP_REQUIRES(
      context,
      input_shape.dim_size(0) == k && input_shape.dim_size(1) == n,
      errors::InvalidArgument(
        "CholeskyBandGradOp expects input matrix "
        "shape be equal to the gradient matrix shape."));
  }

  void UnitOutputShape(const TensorShape&  unit_input_shape,
                              TensorShape * unit_output_shape) override {
    unit_output_shape->Clear();
    unit_output_shape->AppendShape(unit_input_shape);
  }


  void UnitCompute(const std::vector<Tensor>& unit_input_tensors,
                   Tensor* unit_output_tensor) override {
    const Tensor& grad_tensor = unit_input_tensors[0];
    const Tensor& input_tensor = unit_input_tensors[1];

    TensorShape grad_shape = grad_tensor.shape();
    TensorShape input_shape = input_tensor.shape();

    auto k = grad_shape.dim_size(0);
    auto n = grad_shape.dim_size(1);

    banded::MatrixMap<T>(unit_output_tensor->flat<T>().data(), k, n).setZero();
    Tensor grad_copy_tensor(grad_tensor);

    auto grad = grad_copy_tensor.matrix<T>();
    auto input = input_tensor.matrix<T>();
    auto output = unit_output_tensor->matrix<T>();

    // TODO(@awav): Adopted version of dense matrix derivative.
    // Input is k x n matrices.
    //  ____________i
    // |            _| j
    // |          _|0|
    // |_________|0|0|
    for (auto i = n - 1; i >= 0; --i) {
      auto s = std::min(i + 1, k);
      for (auto j = 0; j < s; ++j) {
        auto p = i - j;
        if (j == 0) {
          output(0, p) = T{0.5} * grad(0, p) / input(0, p);
        } else {
          output(j, p) = grad(j, p) / input(0, p);
          grad(0, p) -= grad(j, p) * input(j, p) / input(0, p);
        }
        auto output_jp = output(j, p);
        for (auto l = 1; l < s - j; ++l) {
          auto pl = p - l;
          auto jl = j + l;
          grad(jl, pl) -= output_jp * input(l, pl);
          grad(l, pl) -= output_jp * input(jl, pl);
        }
      }
    }
    // TODO(lucas): what is this line doing?
    banded::MatrixMap<T> output_mat(unit_output_tensor->flat<T>().data(), k, n);
  }

  void  ResultsChecks(OpKernelContext *,
                      const std::vector<Tensor>& , const Tensor&) override{};
};


//
// Operator registration
//

REGISTER_OP("CholeskyBand")
  .Attr("T: {float, double}")
  .Input("banded_matrix: T")
  .Attr("should_check_result: bool")
  .Attr("relative_tolerance: float")
  .Attr("absolute_tolerance: float")
  .Output("banded_lower_triangular: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });


REGISTER_OP("CholeskyBandGrad")
  .Attr("T: {float, double}")
  .Input("lower_triangular_grad_banded: T")
  .Input("lower_triangular_banded: T")
  .Output("matrix_grad_banded: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(1));
    return Status::OK();
  });


#define REGISTER_MULTIDEVICE_CPU_VARIANT(name, T) \
  REGISTER_KERNEL_BUILDER(    \
    Name(#name)               \
    .Device(DEVICE_CPU)       \
    .TypeConstraint<T>("T"),  \
    name ## Op<CPUDevice, T>)


REGISTER_MULTIDEVICE_CPU_VARIANT(CholeskyBand, float);
REGISTER_MULTIDEVICE_CPU_VARIANT(CholeskyBand, double);
REGISTER_MULTIDEVICE_CPU_VARIANT(CholeskyBandGrad, float);
REGISTER_MULTIDEVICE_CPU_VARIANT(CholeskyBandGrad, double);

}  // end of namespace tensorflow
