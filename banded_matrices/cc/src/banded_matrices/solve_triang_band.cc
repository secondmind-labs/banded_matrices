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

#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"

#include "Eigen/Dense"

#include "banded_matrices/common.hpp"
#include "banded_matrices/solve.hpp"

namespace tensorflow {

template <typename T> using Transposed = banded::Transposed<T>;
template <typename T> using Symmetric = banded::Symmetric<T>;
template <typename T> using BandedMatrix = banded::BandedMatrix<T>;

//
// TensorFlow operator for Solve between two banded matrices;
// the left-hand side must in addition be lower- or upper-triangular.
//
template <typename T>
class SolveTriangBandOp : public OpKernel {
 public:
  explicit SolveTriangBandOp(OpKernelConstruction *context) :
      OpKernel(context) {
    LOAD_ATTRIBUTE_OP(
      context, "left_lower_bandwidth", &left_lower_bandwidth_);
    LOAD_ATTRIBUTE_OP(
      context, "left_upper_bandwidth", &left_upper_bandwidth_);

    LOAD_ATTRIBUTE_OP(
      context, "right_lower_bandwidth", &right_lower_bandwidth_);
    LOAD_ATTRIBUTE_OP(
      context, "right_upper_bandwidth", &right_upper_bandwidth_);

    LOAD_ATTRIBUTE_OP(
      context, "result_lower_bandwidth", &result_lower_bandwidth_);
    LOAD_ATTRIBUTE_OP(
      context, "result_upper_bandwidth", &result_upper_bandwidth_);

    LOAD_ATTRIBUTE_OP(
      context, "transpose_left", &transpose_left_);
    LOAD_ATTRIBUTE_OP(
      context, "transpose_right", &transpose_right);

    OP_REQUIRES(
      context,
      left_lower_bandwidth_ == 0 || left_upper_bandwidth_ == 0,
      errors::InvalidArgument(
        "Left matrix of triangular solve should be triangular."));
  }

  void Compute(OpKernelContext *context) override {
    // Get the input tensors, and their shapes
    const Tensor& left_tensor = context->input(0);
    const Tensor& right_tensor = context->input(1);

    const TensorShape& left_shape = left_tensor.shape();
    const TensorShape& right_shape = right_tensor.shape();

    const auto left_width = left_shape.dim_size(0);
    const auto right_width = right_shape.dim_size(0);

    // Get/create the output tensor
    const auto result_width =
      result_lower_bandwidth_ + result_upper_bandwidth_ + 1;
    const auto dim = left_shape.dim_size(1);

    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(
      context,
      context->allocate_output(
        0,
        {result_width, dim},
        &output_tensor));

    // Check dimensions and parameters
    OP_REQUIRES(
      context,
      TensorShapeUtils::IsMatrix(left_shape),
      errors::InvalidArgument(
        "SolveTriangBandOp operation expects a matrix as left argument."));

    OP_REQUIRES(
      context,
      TensorShapeUtils::IsMatrix(right_shape),
      errors::InvalidArgument(
        "SolveTriangBandOp operation expects a matrix as right argument."));

    OP_REQUIRES(
      context,
      dim == right_shape.dim_size(1),
      errors::InvalidArgument(
        "SolveTriangBandOp operation expects "
        "two matrices of matching dimensions."));

    OP_REQUIRES(
      context,
      left_lower_bandwidth_ + 1 + left_upper_bandwidth_ == left_width,
      errors::InvalidArgument(
        "Left lower and upper bandwidths do not sum up "
        "to the actual tensor dimension."));

    OP_REQUIRES(
      context,
      right_lower_bandwidth_ + 1 + right_upper_bandwidth_ == right_width,
      errors::InvalidArgument(
        "Right lower and upper bandwidths do not sum up "
        "to the actual tensor dimension."));

    // View the tensors as banded matrices:
    const auto r = banded::const_banded_view(
      right_tensor.flat<T>().data(),
      right_lower_bandwidth_, right_upper_bandwidth_, dim);

    BandedMatrix<T> result{
      output_tensor->flat<T>().data(),
      result_lower_bandwidth_, result_upper_bandwidth_, dim};

    // Perform the actual operator forward evaluation:
    if (left_upper_bandwidth_ == 0) {
      // Lower-triangular representation:
      const auto l = banded::const_lower_triangular_view(
        left_tensor.flat<T>().data(), left_width, dim);
      solve(l, r, &result);

    } else {
      // Upper triangular matrix, we use the general representation:
      const auto l = banded::const_banded_view(
        left_tensor.flat<T>().data(), 0, left_upper_bandwidth_, dim);
      solve(l, r, &result);
    }
  }

  // Template for generating the code for each type of left matrix.
  template <typename LeftMatrix>
  void solve(
    const LeftMatrix& left, const BandedMatrix<T>& r, BandedMatrix<T>* result) {
    if (transpose_left_) {
      const auto l = Transposed<LeftMatrix>(left);

      if (transpose_right)
        solve_triang_band(l, Transposed<BandedMatrix<T>>(r), result);
      else
        solve_triang_band(l, r, result);

    } else {
      const auto& l = left;

      if (transpose_right)
        solve_triang_band(l, Transposed<BandedMatrix<T>>(r), result);
      else
        solve_triang_band(l, r, result);
    }
  }

 private:
  int left_lower_bandwidth_;
  int left_upper_bandwidth_;

  int right_lower_bandwidth_;
  int right_upper_bandwidth_;

  int result_lower_bandwidth_;
  int result_upper_bandwidth_;

  bool transpose_left_;
  bool transpose_right;
};


//
// Operator registration
//

using InferenceContext = ::tensorflow::shape_inference::InferenceContext;
using DimensionHandle = ::tensorflow::shape_inference::DimensionHandle;

REGISTER_OP("SolveTriangBand")
    .Attr("T: {float, double}")

    .Input("left_banded_matrix: T")
    .Input("right_banded_matrix: T")

    .Attr("left_lower_bandwidth: int")
    .Attr("left_upper_bandwidth: int")

    .Attr("right_lower_bandwidth: int")
    .Attr("right_upper_bandwidth: int")

    .Attr("result_lower_bandwidth: int")
    .Attr("result_upper_bandwidth: int")

    .Attr("transpose_left: bool")
    .Attr("transpose_right: bool")

    .Output("solved: T")

    .SetShapeFn([](InferenceContext *context) {
      int result_lower_bandwidth;
      int result_upper_bandwidth;

      LOAD_ATTRIBUTE_RETURN_IF_ERROR(context, "result_lower_bandwidth",
                                     &result_lower_bandwidth);
      LOAD_ATTRIBUTE_RETURN_IF_ERROR(context, "result_upper_bandwidth",
                                     &result_upper_bandwidth);

      DimensionHandle dim = context->Dim(context->input(0), 1);
      context->set_output(
          0,
          context->Matrix(
            result_lower_bandwidth + 1 + result_upper_bandwidth,
            dim));
      return Status::OK();
    });

REGISTER_CPU(SolveTriangBand, float)
REGISTER_CPU(SolveTriangBand, double)

}  // end of namespace tensorflow
