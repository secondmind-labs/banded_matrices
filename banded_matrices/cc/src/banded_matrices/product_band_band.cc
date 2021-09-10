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
#include <limits>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"

#include "Eigen/Dense"

#include "banded_matrices/common.hpp"
#include "banded_matrices/product.hpp"

namespace tensorflow {

template <typename T> using Transposed = banded::Transposed<T>;
template <typename T> using Symmetric = banded::Symmetric<T>;
template <typename T> using BandedMatrix = banded::BandedMatrix<T>;


//
// Operator for the product of two banded matrices.
//
template <typename T>
class ProductBandBandOp : public OpKernel {
  int left_lower_bandwidth_;
  int left_upper_bandwidth_;

  int right_lower_bandwidth_;
  int right_upper_bandwidth_;

  int result_lower_bandwidth_;
  int result_upper_bandwidth_;

  bool transpose_left_;
  bool transpose_right_;

  bool symmetrise_left_;
  bool symmetrise_right_;

 public:
  explicit ProductBandBandOp(OpKernelConstruction *context)
      : OpKernel(context) {
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
      context, "transpose_right", &transpose_right_);

    LOAD_ATTRIBUTE_OP(
      context, "symmetrise_left", &symmetrise_left_);
    LOAD_ATTRIBUTE_OP(
      context, "symmetrise_right", &symmetrise_right_);
  }

  void Compute(OpKernelContext *context) override {
    // Get the input tensors, and their shapes
    const Tensor &left_tensor = context->input(0);
    const Tensor &right_tensor = context->input(1);

    const TensorShape &left_shape = left_tensor.shape();
    const TensorShape &right_shape = right_tensor.shape();

    const auto left_width = left_shape.dim_size(0);
    const auto right_width = right_shape.dim_size(0);

    // Get/create the output tensor
    const auto result_width =
      result_lower_bandwidth_ + result_upper_bandwidth_ + 1;
    const auto dim = left_shape.dim_size(1);

    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(
      context,
      context->allocate_output(0, {result_width, dim}, &output_tensor));

    // Check dimensions and parameters
    OP_REQUIRES(
      context,
      TensorShapeUtils::IsMatrix(left_shape),
      errors::InvalidArgument(
        "ProductBandBandOp operation expects a matrix as left argument."));

    OP_REQUIRES(
      context,
      TensorShapeUtils::IsMatrix(right_shape),
      errors::InvalidArgument(
        "ProductBandBandOp operation expects a matrix as right argument."));

    OP_REQUIRES(
      context,
      left_lower_bandwidth_ + left_upper_bandwidth_ + 1 == left_width,
      errors::InvalidArgument(
        "Left lower and upper diags do not sum "
        "up to the actual tensor dimension."));

    OP_REQUIRES(
        context,
        right_lower_bandwidth_ + right_upper_bandwidth_ + 1 == right_width,
        errors::InvalidArgument(
          "Right lower and upper diags do not sum up to "
          "the actual tensor dimension."));

    OP_REQUIRES(
        context,
        left_lower_bandwidth_ <= dim && left_upper_bandwidth_ < dim,
        errors::InvalidArgument(
          "Dimensions of left banded matrix exceed "
          "actual square matrix dimension."));

    OP_REQUIRES(
        context,
        right_lower_bandwidth_ <= dim && right_upper_bandwidth_ < dim,
        errors::InvalidArgument(
          "Dimensions of right banded matrix exceed "
          "actual square matrix dimension.."));

    OP_REQUIRES(
        context,
        dim == right_shape.dim_size(1),
        errors::InvalidArgument(
          "ProductBandBandOp operation expects "
          "two matrices of matching dimensions."));

    OP_REQUIRES(
        context,
        !(transpose_left_ && symmetrise_left_),
        errors::InvalidArgument(
          "Left input of ProductBandBandOp "
          "cannot be both transposed and symmetrised."));

    OP_REQUIRES(
        context,
        !(transpose_right_ && symmetrise_right_),
        errors::InvalidArgument(
          "Right input of ProductBandBandOp "
          "cannot be both transposed and symmetrised."));

    OP_REQUIRES(
        context,
        !(symmetrise_left_ && left_upper_bandwidth_ > 0),
        errors::InvalidArgument(
          "Left banded matrix is symmetric but not "
          "represented as lower-triangular."));

    OP_REQUIRES(
        context,
        !(symmetrise_right_ && right_upper_bandwidth_ > 0),
        errors::InvalidArgument(
          "Right banded matrix is symmetric but not "
          "represented as lower-triangular."));

    // View the tensors as banded matrices:
    const auto left = banded::const_banded_view(
      left_tensor.flat<T>().data(),
      left_lower_bandwidth_,
      left_upper_bandwidth_, dim);

    const auto right = banded::const_banded_view(
      right_tensor.flat<T>().data(),
      right_lower_bandwidth_,
      right_upper_bandwidth_, dim);

    BandedMatrix<T> result{
      output_tensor->flat<T>().data(),
      result_lower_bandwidth_, result_upper_bandwidth_, dim };

    // Perform the actual evaluation:
    // TODO(optim): generate specialized code for lower-triangular case
    if (transpose_left_)
      product(Transposed<BandedMatrix<T>>(left), right, &result);

    else if (symmetrise_left_)
      product(Symmetric<BandedMatrix<T>>(left), right, &result);

    else
      product(left, right, &result);
  }

  template <typename LeftMatrix>
  void product(
    const LeftMatrix& l, const BandedMatrix<T>& r, BandedMatrix<T>* result) {
      if (transpose_right_)
        product_band_band(l, Transposed<BandedMatrix<T>>(r), result);

      else if (symmetrise_right_)
        product_band_band(l, Symmetric<BandedMatrix<T>>(r), result);

      else
        product_band_band(l, r, result);
  }
};


//
// Operator registration
//

using InferenceContext = ::tensorflow::shape_inference::InferenceContext;
using DimensionHandle = ::tensorflow::shape_inference::DimensionHandle;

REGISTER_OP("ProductBandBand")
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

    .Attr("symmetrise_left: bool")
    .Attr("symmetrise_right: bool")

    .Output("banded_product: T")
    .SetShapeFn([](InferenceContext *context) {
      int left_lower_bandwidth_;
      int left_upper_bandwidth_;

      int right_lower_bandwidth_;
      int right_upper_bandwidth_;

      int result_lower_bandwidth_;
      int result_upper_bandwidth_;

      bool transpose_left_;
      bool transpose_right;

      bool symmetrise_left_;
      bool symmetrise_right;

      LOAD_ATTRIBUTE_RETURN_IF_ERROR(context, "left_lower_bandwidth",
                                     &left_lower_bandwidth_);
      LOAD_ATTRIBUTE_RETURN_IF_ERROR(context, "left_upper_bandwidth",
                                     &left_upper_bandwidth_);

      LOAD_ATTRIBUTE_RETURN_IF_ERROR(context, "right_lower_bandwidth",
                                     &right_lower_bandwidth_);
      LOAD_ATTRIBUTE_RETURN_IF_ERROR(context, "right_upper_bandwidth",
                                     &right_upper_bandwidth_);

      LOAD_ATTRIBUTE_RETURN_IF_ERROR(context, "result_lower_bandwidth",
                                     &result_lower_bandwidth_);
      LOAD_ATTRIBUTE_RETURN_IF_ERROR(context, "result_upper_bandwidth",
                                     &result_upper_bandwidth_);

      LOAD_ATTRIBUTE_RETURN_IF_ERROR(context, "transpose_left",
                                     &transpose_left_);
      LOAD_ATTRIBUTE_RETURN_IF_ERROR(context, "transpose_right",
                                     &transpose_right);

      LOAD_ATTRIBUTE_RETURN_IF_ERROR(context, "symmetrise_left",
                                     &symmetrise_left_);
      LOAD_ATTRIBUTE_RETURN_IF_ERROR(context, "symmetrise_right",
                                     &symmetrise_right);

      DimensionHandle dim = context->Dim(context->input(0), 1);
      context->set_output(
          0, context->Matrix(
            result_lower_bandwidth_ + result_upper_bandwidth_ + 1,
            dim));
      return Status::OK();
    });

REGISTER_CPU(ProductBandBand, float)
REGISTER_CPU(ProductBandBand, double)

}  // end of namespace tensorflow
