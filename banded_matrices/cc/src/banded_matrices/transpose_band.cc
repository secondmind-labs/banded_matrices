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

#include <sstream>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"

#include "Eigen/Dense"

#include "banded_matrices/banded_matrix.hpp"
#include "banded_matrices/common.hpp"
#include "banded_matrices/unary_broadcastable_op_kernel.hpp"


namespace tensorflow {

using Index = Eigen::Index;
template <typename T> using BandedMatrix = banded::BandedMatrix<T>;


//
// Operator for transposing a banded matrix.
//
template <typename T>
class TransposeBandOp : public UnaryBroadcastableOpKernel<T, 2> {
 public:
  explicit TransposeBandOp(OpKernelConstruction *context)
      : UnaryBroadcastableOpKernel<T, 2>(context) {
    LOAD_ATTRIBUTE_OP(context, "input_lower_bandwidth",
                      &input_lower_bandwidth_);
    LOAD_ATTRIBUTE_OP(context, "input_upper_bandwidth",
                      &input_upper_bandwidth_);
  }

  void StartChecks(OpKernelContext *context,
                   const TensorShape& unit_input_shape) override {
    // Check dimensions and parameters
    OP_REQUIRES(
      context,
      TensorShapeUtils::IsMatrix(unit_input_shape),
      errors::InvalidArgument("Transpose operation expects a matrix"));

    OP_REQUIRES(
      context,
      input_lower_bandwidth_ + input_upper_bandwidth_ + 1 ==
        unit_input_shape.dim_size(0),
      errors::InvalidArgument("Lower/upper band widths do not add up"));
  }

  void UnitOutputShape(const TensorShape&  unit_input_shape,
                              TensorShape * unit_output_shape) override {
    unit_output_shape->Clear();
    unit_output_shape->AppendShape(unit_input_shape);
  }

  void UnitCompute(const std::vector<Tensor>& unit_input_tensors,
                   Tensor* unit_output_tensor) override {
    // View the tensors as banded matrices:
    auto unit_input_tensor = unit_input_tensors[0];

    const auto input = banded::const_banded_view(
      unit_input_tensor.flat<T>().data(),
      input_lower_bandwidth_,
      input_upper_bandwidth_,
      unit_input_tensor.shape().dim_size(1));

    BandedMatrix<T> result{
      unit_output_tensor->flat<T>().data(),
      input_upper_bandwidth_,
      input_lower_bandwidth_,
      unit_output_tensor->shape().dim_size(1),
      true};

    // Transpose the input into the output:
    result.for_each_in_band([&input](Index row, Index col, T& target) {
      target = input(col, row);
    });
  }

  void  ResultsChecks(OpKernelContext *,
                      const std::vector<Tensor>& , const Tensor&) override {}

 private:
  int input_lower_bandwidth_;
  int input_upper_bandwidth_;
};


//
// Operator registration
//

using InferenceContext = ::tensorflow::shape_inference::InferenceContext;

REGISTER_OP("TransposeBand")
    .Attr("T: {float, double}")
    .Input("tensor: T")
    .Attr("input_lower_bandwidth: int")
    .Attr("input_upper_bandwidth: int")
    .Output("transpose: T")
    .SetShapeFn([](InferenceContext* context) {
      context->set_output(0, context->input(0));
      return Status::OK();
    });

REGISTER_CPU(TransposeBand, float)
REGISTER_CPU(TransposeBand, double)

}  // end of namespace tensorflow
