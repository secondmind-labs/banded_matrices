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
#include "banded_matrices/unary_broadcastable_op_kernel.hpp"


namespace tensorflow {

template <typename T> using Transposed = banded::Transposed<T>;
template <typename T> using BandedMatrix = banded::BandedMatrix<T>;


//
// Tensorflow operator for the squaring operation (M->MM^T) of banded matrices.
//
template <typename T>
class SquareBandOp : public UnaryBroadcastableOpKernel<T, 2> {
 public:
  explicit SquareBandOp(OpKernelConstruction *context)
      : UnaryBroadcastableOpKernel<T, 2>(context) {
    LOAD_ATTRIBUTE_OP(context, "lower_bandwidth", &lower_bandwidth_);
    LOAD_ATTRIBUTE_OP(context, "upper_bandwidth", &upper_bandwidth_);
  }

  void StartChecks(OpKernelContext *context,
                   const TensorShape& unit_input_shape) override {
    // Check dimensions and parameters

    const auto width = unit_input_shape.dim_size(0);
    const auto dim = unit_input_shape.dim_size(1);

    OP_REQUIRES(
      context,
      TensorShapeUtils::IsMatrix(unit_input_shape),
      errors::InvalidArgument(
        "SquareBandOp operation expects a matrix as left argument."));

    OP_REQUIRES(
      context,
      lower_bandwidth_ + upper_bandwidth_ + 1 == width,
      errors::InvalidArgument(
        "Lower and upper diags do not sum up to "
        "the actual tensor dimension."));

    OP_REQUIRES(
      context,
      lower_bandwidth_ <= dim && upper_bandwidth_ < dim,
      errors::InvalidArgument(
        "Dimensions of banded matrix exceed "
        "actual square matrix dimension."));
  }

  void UnitOutputShape(const TensorShape&  unit_input_shape,
                              TensorShape * unit_output_shape) override {
    const auto width = unit_input_shape.dim_size(0);
    const auto dim = unit_input_shape.dim_size(1);

    unit_output_shape->Clear();
    unit_output_shape->AddDim(width);
    unit_output_shape->AddDim(dim);
  }

  void UnitCompute(const std::vector<Tensor>& unit_input_tensors,
                   Tensor* unit_output_tensor) override {
    // View the tensors as banded matrices:
    const auto dim = unit_input_tensors[0].shape().dim_size(1);

  const auto matrix = banded::const_banded_view(
      unit_input_tensors[0].flat<T>().data(),
      lower_bandwidth_,
      upper_bandwidth_, dim);

    BandedMatrix<T> result{
      unit_output_tensor->flat<T>().data(),
      lower_bandwidth_ + upper_bandwidth_,
      0,
      dim};

    // Perform the actual evaluation:
    product_band_band(matrix, Transposed<BandedMatrix<T>>(matrix), &result);
  }

  void  ResultsChecks(OpKernelContext *,
                      const std::vector<Tensor>& , const Tensor&) override {}

 private:
  int lower_bandwidth_;
  int upper_bandwidth_;
};


//
// Operator registration
//

using InferenceContext = ::tensorflow::shape_inference::InferenceContext;

REGISTER_OP("SquareBand")
    .Attr("T: {float, double}")
    .Input("banded_matrix: T")
    .Attr("lower_bandwidth: int")
    .Attr("upper_bandwidth: int")
    .Output("banded_square: T")
    .SetShapeFn([](InferenceContext *context) {
      context->set_output(0, context->input(0));
      return Status::OK();
    });

REGISTER_CPU(SquareBand, float)
REGISTER_CPU(SquareBand, double)

}  // end of namespace tensorflow
