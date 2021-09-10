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
// Operator for the product of a matrix by a vector,
// or group of vectors put together into a non-banded matrix.
//
template <typename T>
class ProductBandMatOp : public OpKernel {
 public:
  explicit ProductBandMatOp(OpKernelConstruction* context)
    : OpKernel(context) {

    LOAD_ATTRIBUTE_OP(
      context, "left_lower_bandwidth", &left_lower_bandwidth_);
    LOAD_ATTRIBUTE_OP(
      context, "left_upper_bandwidth", &left_upper_bandwidth_);
    LOAD_ATTRIBUTE_OP(
      context, "transpose_left", &transpose_left_);
    LOAD_ATTRIBUTE_OP(
      context, "symmetrise_left", &symmetrise_left_);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& left_tensor = context->input(0);
    const Tensor& right_tensor = context->input(1);
    const TensorShape& left_shape = left_tensor.shape();
    const TensorShape& right_shape = right_tensor.shape();

    const auto k = left_shape.dim_size(0);
    const auto n = left_shape.dim_size(1);
    const auto num_vectors = right_shape.dim_size(1);

    OP_REQUIRES(context,
      TensorShapeUtils::IsMatrix(left_shape),
      errors::InvalidArgument(
        "ProductBandMat operation expects a matrix as first argument."));

    OP_REQUIRES(context,
      TensorShapeUtils::IsMatrix(right_shape),
      errors::InvalidArgument(
        "ProductBandMat operation expects a matrix as second argument."));

    OP_REQUIRES(context,
      k <= n,
      errors::InvalidArgument(
        "ProductBandMat operation expects a banded "
        "matrix as first argument."));

    OP_REQUIRES(context,
      right_shape.dim_size(0) == n,
      errors::InvalidArgument(
        "ProductBandMat vector size "
        "does not match the left-hand side dimension."));

    OP_REQUIRES(context,
      !(transpose_left_ && symmetrise_left_),
      errors::InvalidArgument(
        "Left input of ProductBandMat"
        "cannot be both transposed and symmetrised."));

    OP_REQUIRES(context,
      left_lower_bandwidth_ + 1 + left_upper_bandwidth_ == k,
      errors::InvalidArgument("Width parameters don't add up"));

    // Allocate result
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
      0, {n, num_vectors}, &output_tensor));

    // View each pointer as properly dimensioned matrices
    const auto left = banded::const_banded_view(
      left_tensor.flat<T>().data(),
      left_lower_bandwidth_,
      left_upper_bandwidth_,
      n);

    banded::MatrixConstMap<T> right(
      right_tensor.flat<T>().data(), n, num_vectors);
    banded::MatrixMap<T> result(
      output_tensor->flat<T>().data(), n, num_vectors);

    // Do the product
    if (transpose_left_)
      banded::product_band_mat(
        Transposed<BandedMatrix<T>>(left), right, &result);

    else if (symmetrise_left_)
      banded::product_band_mat(
        Symmetric<BandedMatrix<T>>(left), right, &result);

    else
      banded::product_band_mat(left, right, &result);
  }

 private:
  int left_lower_bandwidth_;
  int left_upper_bandwidth_;
  bool transpose_left_;
  bool symmetrise_left_;
};


//
// Operator registration
//

using InferenceContext = ::tensorflow::shape_inference::InferenceContext;

REGISTER_OP("ProductBandMat")
  .Attr("T: {float, double}")
  .Input("banded_matrix: T")

  .Attr("left_lower_bandwidth: int")
  .Attr("left_upper_bandwidth: int")
  .Attr("transpose_left: bool")
  .Attr("symmetrise_left: bool")

  .Input("vector: T")
  .Output("product_result: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    // Note that input(1) is assumed here to be an Mx1 matrix:
    c->set_output(0, c->input(1));
    return Status::OK();
  });


REGISTER_CPU(ProductBandMat, float);
REGISTER_CPU(ProductBandMat, double);

}  // end of namespace tensorflow
