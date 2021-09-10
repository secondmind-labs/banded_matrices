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
#include <string>
#include <sstream>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"

#include "Eigen/Dense"

#include "banded_matrices/common.hpp"
#include "banded_matrices/banded_matrix.hpp"
#include "banded_matrices/unary_broadcastable_op_kernel.hpp"


namespace tensorflow {

using Index = Eigen::Index;


//
// Operator to build a symmetric band from its lower half.
//
template <typename T>
class SymmetriseBandOp : public UnaryBroadcastableOpKernel<T, 2>  {
 public:
  explicit SymmetriseBandOp(OpKernelConstruction *context)
      : UnaryBroadcastableOpKernel<T, 2>(context) {
    LOAD_ATTRIBUTE_OP(
      context, "input_lower_bandwidth", &input_lower_bandwidth_);
  }

  void StartChecks(OpKernelContext *context,
                   const TensorShape& unit_input_shape) override {
    const auto rows = unit_input_shape.dim_size(0);
    OP_REQUIRES(context,
      TensorShapeUtils::IsMatrix(unit_input_shape),
      errors::InvalidArgument("Symmetrise operation expects a matrix"));

    OP_REQUIRES(context,
      input_lower_bandwidth_ + 1 == rows,
      errors::InvalidArgument("Lower/upper band widths do not add up"));
  };

  void UnitOutputShape(const TensorShape&  unit_input_shape,
                              TensorShape * unit_output_shape) override {
    const auto rows = unit_input_shape.dim_size(0);
    const auto cols = unit_input_shape.dim_size(1);
    // Pre compute output dimension
    unit_output_shape->Clear();
    unit_output_shape->AddDim(2*rows - 1);
    unit_output_shape->AddDim(cols);
  };

  void UnitCompute(const std::vector<Tensor>& unit_input_tensors,
                   Tensor* unit_output_tensor) override {
    const Tensor& input_tensor = unit_input_tensors[0];
    const TensorShape& input_shape = input_tensor.shape();
    const auto rows = input_shape.dim_size(0);
    const auto cols = input_shape.dim_size(1);
        // View the tensors as dense matrices
    banded::MatrixConstMap<T> input{
      input_tensor.flat<T>().data(), rows, cols };
    banded::MatrixMap<T> result{
      unit_output_tensor->flat<T>().data(), 2 * rows - 1, cols};
    result.setZero();

    for (Index row = 0; row < rows; ++row) {
      // copy lower part
      for (Index col = 0; col < cols; ++col) {
        result(rows - 1 + row, col) = input(row, col);
      }
      // copy transpose
      for (Index col = 0; col < cols-row; ++col) {
        result(rows - 1 - row, row + col) = input(row, col);
      }
    }
  };

  void  ResultsChecks(OpKernelContext *,
                      const std::vector<Tensor>& , const Tensor&) override{};

 private:
  int input_lower_bandwidth_;
};


//
// Operator to extract the lower part of a symmetric band.
//
template <typename T>
class HalveBandOp : public UnaryBroadcastableOpKernel<T, 2> {
 public:
  explicit HalveBandOp(OpKernelConstruction *context)
      : UnaryBroadcastableOpKernel<T, 2>(context) {
    LOAD_ATTRIBUTE_OP(context, "input_lower_bandwidth",
                      &input_lower_bandwidth_);
  }

  void StartChecks(OpKernelContext *context,
                   const TensorShape& unit_input_shape) override {
    const auto rows = unit_input_shape.dim_size(0);
    const auto rows_out = (rows+1)/2;
  // Check dimensions and parameters
    OP_REQUIRES(context,
      TensorShapeUtils::IsMatrix(unit_input_shape),
      errors::InvalidArgument("Halve operation expects a matrix"));

    OP_REQUIRES(context,
      input_lower_bandwidth_ + 1 == rows_out,
      errors::InvalidArgument("Lower/upper band widths do not add up"));
  };

  void UnitOutputShape(const TensorShape&  unit_input_shape,
                              TensorShape * unit_output_shape) override {
    const auto rows = unit_input_shape.dim_size(0);
    const auto cols = unit_input_shape.dim_size(1);
    const auto rows_out = (rows+1)/2;
    // Pre compute output dimension
    unit_output_shape->Clear();
    unit_output_shape->AddDim(rows_out);
    unit_output_shape->AddDim(cols);
  };

  void UnitCompute(const std::vector<Tensor>& unit_input_tensors,
                   Tensor* unit_output_tensor) override {
    const Tensor& input_tensor = unit_input_tensors[0];
    const auto rows = input_tensor.dim_size(0);
    const auto cols = input_tensor.dim_size(1);
    const auto rows_out = (rows+1)/2;
        // View the tensors as dense matrices
    banded::MatrixConstMap<T> input{
      input_tensor.flat<T>().data(), rows, cols };
    banded::MatrixMap<T> result{
      unit_output_tensor->flat<T>().data(), rows_out, cols};
    result.setZero();

    for (Index row = 0; row < rows_out; ++row) {
      // keep lower part
      result.row(row) =  input.row(rows_out - 1 + row);
    }
  };

  void  ResultsChecks(OpKernelContext * context,
                      const std::vector<Tensor>& unit_input_tensors,
                      const Tensor&) override{
    // verify the input symmetric (a bit after the fact)
    const auto cols = unit_input_tensors[0].dim_size(1);

    const auto input_dense  = banded::const_banded_view(
      unit_input_tensors[0].flat<T>().data(),
      input_lower_bandwidth_, input_lower_bandwidth_, cols);

    OP_REQUIRES(
      context,
      verify_symmetric(input_dense),
      errors::InvalidArgument("Matrix is not symmetric"));
  };


  // Check that the input matrix is symmetric.
  bool verify_symmetric(const banded::BandedMatrix<T>& M) {
// Here we do explicitly want to check floating point equality:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"

    for (Index col = 0; col < M.dim(); ++col) {
      for (const auto row : M.rows_in_band(col)) {
         if (M(row, col) != M(col, row))
           return false;
         if (row > col)
           break;
      }
    }
#pragma GCC diagnostic pop
    return true;
  }

 private:
  int input_lower_bandwidth_;
};


//
// Operator registration
//

using InferenceContext = ::tensorflow::shape_inference::InferenceContext;

REGISTER_OP("SymmetriseBand")
    .Attr("T: {float, double}")
    .Input("tensor: T")
    .Attr("input_lower_bandwidth: int")
    .Output("symmetrised: T")
    .SetShapeFn([](InferenceContext *context) {
      int input_lower_bandwidth;
      LOAD_ATTRIBUTE_RETURN_IF_ERROR(context, "input_lower_bandwidth",
                                     &input_lower_bandwidth);

      // We would expect the following assertion to hold here:
      // assert (input_lower_bandwidth ==
      //    context->Value(context->Dim(context->input(0), 0)) - 1);
      // However this is not always the case as the right-hand side is
      // sometimes not initialized (arbitrary unusable value, usually negative)
      // when the Tensor is for instance a Slice coming
      // from a broadcasting call to the op.
      // Our understanding of the issue may however be incomplete, see
      // _test_unary_operators_broadcast_lower
      // for a test that exhibits this difference.

      shape_inference::DimensionHandle dim = context->Dim(
        context->input(0), -1);

      shape_inference::ShapeHandle leading_dims;
      TF_RETURN_IF_ERROR(
        context->Subshape(context->input(0), 0, -2, &leading_dims));

      shape_inference::ShapeHandle mat = context->Matrix(
        2 * input_lower_bandwidth + 1, dim);
      shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(
        context->Concatenate(leading_dims, mat, &out));

      context->set_output(0, out);
      return Status::OK();
    });

REGISTER_OP("HalveBand")
    .Attr("T: {float, double}")
    .Input("tensor: T")
    .Attr("input_lower_bandwidth: int")
    .Output("halved: T")
    .SetShapeFn([](InferenceContext *context) {
      int input_lower_bandwidth;
      LOAD_ATTRIBUTE_RETURN_IF_ERROR(context, "input_lower_bandwidth",
                                     &input_lower_bandwidth);

      // Similarly to the registration of `SymmetriseBand`,
      // The following assertion may not be respected as one could think:
      // assert (input_lower_bandwidth ==
      //   context->Value(context->Dim(context->input(0), 0)) - 1).

      shape_inference::DimensionHandle dim = context->Dim(
        context->input(0), -1);

      shape_inference::ShapeHandle leading_dims;
      TF_RETURN_IF_ERROR(
        context->Subshape(context->input(0), 0, -2, &leading_dims));

      shape_inference::ShapeHandle mat = context->Matrix(
        input_lower_bandwidth + 1, dim);
      shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(
        context->Concatenate(leading_dims, mat, &out));

      context->set_output(0, out);
      return Status::OK();
    });

REGISTER_CPU(SymmetriseBand, float)
REGISTER_CPU(SymmetriseBand, double)

REGISTER_CPU(HalveBand, float)
REGISTER_CPU(HalveBand, double)

}  // end of namespace tensorflow
