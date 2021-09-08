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
#include "banded_matrices/common.hpp"
#include "banded_matrices/banded_matrix.hpp"
#include "banded_matrices/unary_broadcastable_op_kernel.hpp"


namespace tensorflow {

using Index = Eigen::Index;
template <typename T> using BandedMatrix = banded::BandedMatrix<T>;


//
// Operator that converts a dense matrix to a banded one;
// mostly useful for debugging purposes.
//
template <typename T>
class PackDenseMatrixToBandedOp : public UnaryBroadcastableOpKernel<T, 2> {
 public:
  explicit PackDenseMatrixToBandedOp(OpKernelConstruction* context) :
      UnaryBroadcastableOpKernel<T, 2>(context) {
    LOAD_ATTRIBUTE_OP(context, "lower_bandwidth", &lower_bandwidth_);
    LOAD_ATTRIBUTE_OP(context, "upper_bandwidth", &upper_bandwidth_);
  }

  void StartChecks(OpKernelContext *context,
                   const TensorShape& unit_input_shape) override {
    // Check dimensions and parameters
    const auto n = unit_input_shape.dim_size(0);
    const auto bandwidth = lower_bandwidth_ + 1 + upper_bandwidth_;
    OP_REQUIRES(
      context,
      TensorShapeUtils::IsSquareMatrix(unit_input_shape),
      errors::InvalidArgument(
        "PackDenseMatrixToBanded operation expects a square matrix."));

    OP_REQUIRES(
      context,
      bandwidth <= n,
      errors::InvalidArgument(
        "PackDenseMatrixToBanded operation expects a matrix with size "
        "bigger than bandwidth ", bandwidth));
  }


  void UnitOutputShape(const TensorShape&  unit_input_shape,
                              TensorShape * unit_output_shape) override {
    const auto n = unit_input_shape.dim_size(0);
    const auto bandwidth = lower_bandwidth_ + 1 + upper_bandwidth_;
    unit_output_shape->Clear();
    unit_output_shape->AddDim(bandwidth);
    unit_output_shape->AddDim(n);
  }

  void UnitCompute(const std::vector<Tensor>& unit_input_tensors,
                   Tensor* unit_output_tensor) override {
    // View the tensors as banded matrices:
    const Tensor& input_tensor = unit_input_tensors[0];
    const auto n = input_tensor.shape().dim_size(0);

    banded::MatrixConstMap<T> input(input_tensor.flat<T>().data(), n, n);
    BandedMatrix<T> output{
      unit_output_tensor->flat<T>().data(),
      lower_bandwidth_, upper_bandwidth_, n};

    output.underlying_dense_matrix().setZero();
    output.for_each_in_band([&input](Index row, Index col, T& value) {
      value = input(row, col);
    });
  }

  void  ResultsChecks(OpKernelContext *,
                      const std::vector<Tensor>& , const Tensor&) override {}

  static bool check_zeros_out_of_band(const banded::MatrixConstMap<T>& input,
                                      const BandedMatrix<T>& output) {
    for (Index col = 0; col < input.cols(); ++col) {
      for (Index row = 0; row < input.rows(); ++row) {
        if (!output.is_in_band(col, row) && input(col, row) != 0) {
          return false;
        }
      }
    }
    return true;
  }

 private:
  int lower_bandwidth_;
  int upper_bandwidth_;
};


//
// Operator that converts a banded matrix to a dense one;
// mostly useful for debugging purposes.
//
template <typename T>
class UnpackBandedMatrixToDenseOp : public UnaryBroadcastableOpKernel<T, 2>  {
 public:
  explicit UnpackBandedMatrixToDenseOp(OpKernelConstruction* context) :
      UnaryBroadcastableOpKernel<T, 2> (context) {
    LOAD_ATTRIBUTE_OP(context, "lower_bandwidth", &lower_bandwidth_);
    LOAD_ATTRIBUTE_OP(context, "upper_bandwidth", &upper_bandwidth_);
  }

  void StartChecks(OpKernelContext *context,
                   const TensorShape& unit_input_shape) override {
    // Check dimensions and parameters
    const auto bandwidth = unit_input_shape.dim_size(0);
    const auto n = unit_input_shape.dim_size(1);

    OP_REQUIRES(
      context,
      TensorShapeUtils::IsMatrix(unit_input_shape),
      errors::InvalidArgument(
      "UnpackBandedMatrixToDense operation expects a matrix."));

    OP_REQUIRES(
      context,
      lower_bandwidth_ + 1 + upper_bandwidth_ == bandwidth,
      errors::InvalidArgument(
        "Right lower and upper diags do not sum up to "
        "the actual tensor dimension."));

    OP_REQUIRES(
      context,
      bandwidth <= n,
      errors::InvalidArgument(
        "UnpackBandedMatrixToDense operation expects a matrix with "
        "bandwidth less or equal to major matrix size."));
  }


  void UnitOutputShape(const TensorShape&  unit_input_shape,
                              TensorShape * unit_output_shape) override {
    const auto n = unit_input_shape.dim_size(1);

    unit_output_shape->Clear();
    unit_output_shape->AddDim(n);
    unit_output_shape->AddDim(n);
  }


  void UnitCompute(const std::vector<Tensor>& unit_input_tensors,
                   Tensor* unit_output_tensor) override {
    auto input_tensor = unit_input_tensors[0];
    const auto n = input_tensor.shape().dim_size(1);

    const auto input = banded::const_banded_view(
      input_tensor.flat<T>().data(), lower_bandwidth_, upper_bandwidth_, n);

    banded::MatrixMap<T> output(unit_output_tensor->flat<T>().data(), n, n);

    output.setZero();
    input.for_each_in_band([&output](Index row, Index col, T value) {
      output(row, col) = value;
    });
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
using DimensionHandle = ::tensorflow::shape_inference::DimensionHandle;


REGISTER_OP("PackDenseMatrixToBanded")
  .Attr("T: {float, double}")
  .Attr("lower_bandwidth: int >= 0")
  .Attr("upper_bandwidth: int >= 0")
  .Input("dense_matrix: T")
  .Output("banded_matrix: T")
  .SetShapeFn([](InferenceContext* context) {
      int lower_bandwidth;
      int upper_bandwidth;

      LOAD_ATTRIBUTE_RETURN_IF_ERROR(context, "lower_bandwidth",
                                     &lower_bandwidth);
      LOAD_ATTRIBUTE_RETURN_IF_ERROR(context, "upper_bandwidth",
                                     &upper_bandwidth);

      DimensionHandle dim = context->Dim(context->input(0), -1);

      shape_inference::ShapeHandle leading_dims;
      TF_RETURN_IF_ERROR(
        context->Subshape(context->input(0), 0, -2, &leading_dims));

      shape_inference::ShapeHandle mat = context->Matrix(
        lower_bandwidth + 1 + upper_bandwidth, dim);
      shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(
        context->Concatenate(leading_dims, mat, &out));
      context->set_output(0, out);
      return Status::OK();
  });


REGISTER_OP("UnpackBandedMatrixToDense")
  .Attr("T: {float, double}")
  .Attr("lower_bandwidth: int >= 0")
  .Attr("upper_bandwidth: int >= 0")
  .Input("banded_matrix: T")
  .Output("dense_matrix: T")
  .SetShapeFn([](InferenceContext* context) {
      DimensionHandle dim = context->Dim(context->input(0), -1);

      shape_inference::ShapeHandle leading_dims;
      TF_RETURN_IF_ERROR(
        context->Subshape(context->input(0), 0, -2, &leading_dims));

      shape_inference::ShapeHandle mat = context->Matrix(dim, dim);
      shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(
        context->Concatenate(leading_dims, mat, &out));
      context->set_output(0, out);
      return Status::OK();
  });


REGISTER_CPU(PackDenseMatrixToBanded, float);
REGISTER_CPU(PackDenseMatrixToBanded, double);
REGISTER_CPU(UnpackBandedMatrixToDense, float);
REGISTER_CPU(UnpackBandedMatrixToDense, double);

}  // end of namespace tensorflow
