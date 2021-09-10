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
#include "banded_matrices/banded_matrix.hpp"

namespace tensorflow {

template <typename T> using MatrixMap = banded::MatrixMap<T>;
template <typename T> using MatrixConstMap = banded::MatrixConstMap<T>;
template <typename T> using Vector = banded::Vector<T>;
template <typename T> using VectorConstMap = banded::VectorConstMap<T>;
using Index = Eigen::Index;

//
// Tensorflow operator for computing an arbitrary band of the
// outer product m.n^T between two vectors.
//
template <typename T>
class OuterVecVecOp : public OpKernel {
 public:
  explicit OuterVecVecOp(OpKernelConstruction *context)
      : OpKernel(context) {
    LOAD_ATTRIBUTE_OP(
      context, "result_lower_bandwidth", &result_lower_bandwidth_);
    LOAD_ATTRIBUTE_OP(
      context, "result_upper_bandwidth", &result_upper_bandwidth_);
  }

  void Compute(OpKernelContext *context) override {
    // Get the input tensors, and their sizes
    const Tensor& left_tensor = context->input(0);
    const Tensor& right_tensor = context->input(1);

    // Get/create the output tensor
    const Index result_num_line =
      result_lower_bandwidth_ + 1 + result_upper_bandwidth_;
    const Index result_num_col = left_tensor.shape().dim_size(0);
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(
      context,
      context->allocate_output(0,
        {result_num_line, result_num_col}, &output_tensor));

    // Check dimensions and parameters
    OP_REQUIRES_OK(
      context,
      check_vector_input(
        left_tensor,
        "OuterVecVec operation expects a Nx1 matrix as first argument."));

    OP_REQUIRES_OK(
      context,
      check_vector_input(
        right_tensor,
        "OuterVecVec operation expects a Nx1 matrix as second argument."));

    OP_REQUIRES(
      context,
      TensorShapeUtils::IsMatrix(right_tensor.shape()),
      errors::InvalidArgument(
        "OuterVecVec operation expects a collection of vectors "
        "as its second argument."));

    OP_REQUIRES(
      context,
      result_num_col == right_tensor.shape().dim_size(0),
      errors::InvalidArgument(
        "Outer product between vectors of different sizes."));

    OP_REQUIRES(
      context,
      result_num_line <= result_num_col,
      errors::InvalidArgument(
        "The bandwith asked for the result must"
        "be smaller than the matrix size"));

    // View result as a matrix, and left and right as vectors
    MatrixMap<T> result(
      output_tensor->flat<T>().data(), result_num_line, result_num_col);
    VectorConstMap<T> left(
      left_tensor.flat<T>().data(), result_num_col);
    VectorConstMap<T> right(
      right_tensor.flat<T>().data(), result_num_col);

    // Create a vector like left but padded with the right 0s:
    Vector<T> tensor_0(
      result_upper_bandwidth_ + result_num_col + result_lower_bandwidth_);

    tensor_0 <<
      Vector<T>::Zero(result_upper_bandwidth_),
      left,
      Vector<T>::Zero(result_lower_bandwidth_);

    result.setZero();
    for (Index i = 0; i < result_num_col; ++i) {
        result.block(0, i, result_num_line, 1) =
          right(i) * tensor_0.segment(i, result_num_line);
    }
  }

  Status check_vector_input(
    const tensorflow::Tensor& tensor, const char* error) {
    if (!tensorflow::TensorShapeUtils::IsMatrix(tensor.shape()) ||
      tensor.shape().dim_size(1) != 1) {
      return errors::InvalidArgument(error);
    }

    return Status::OK();
  }

 private:
  int result_lower_bandwidth_;
  int result_upper_bandwidth_;
};


//
// Code logic of the `OuterMatMatOp` operator:
// computaton of an arbitrary band of the
// outer product M.N^T between two non-banded matrices.
//
template <typename T>
Status compute_outer_mat_mat(
    OpKernelContext* context,
    const Tensor& left_tensor,
    const Tensor& right_tensor,
    int result_lower_bandwidth_, int result_upper_bandwidth_) {
  const auto dim = left_tensor.shape().dim_size(0);
  const auto num_vectors = left_tensor.shape().dim_size(1);

  // Get/create the output tensor
  const auto result_num_line =
    result_lower_bandwidth_ + 1 + result_upper_bandwidth_;
  Tensor *output_tensor = nullptr;
  TF_RETURN_IF_ERROR(context->allocate_output(0, {result_num_line, dim},
    &output_tensor));

  // Check dimensions and parameters
  if (!TensorShapeUtils::IsMatrix(left_tensor.shape())) {
    return errors::InvalidArgument(
      "OuterVecVec operation expects a vector as its first argument.");
  }

  if (!TensorShapeUtils::IsMatrix(right_tensor.shape())) {
    return errors::InvalidArgument(
      "OuterVecVec operation expects a vector as its second argument.");
  }

  if (dim != left_tensor.shape().dim_size(0)) {
    return errors::InvalidArgument(
      "Outer product between vectors of different lengths.");
  }

  if (num_vectors != left_tensor.shape().dim_size(1)) {
    return errors::InvalidArgument(
      "Outer product between different numbers of vectors.");
  }

  // View left and right as dense matrices:
  MatrixConstMap<T> left(left_tensor.flat<T>().data(), dim, num_vectors);
  MatrixConstMap<T> right_t(right_tensor.flat<T>().data(), dim, num_vectors);
  Eigen::Transpose<MatrixConstMap<T>> right = right_t.transpose();

  // View result as a banded matrix:
  banded::BandedMatrix<T> result{
    output_tensor->flat<T>().data(),
    result_lower_bandwidth_, result_upper_bandwidth_, dim,
    true};

  result.setCornersToZero();
  result.for_each_in_band([&left, &right](Index row, Index col, T& target) {
    target = left.row(row) * right.col(col);
  });

  return Status::OK();
}


//
// Tensorflow operator for computing an arbitrary band of the
// outer product M.N^T between two non-banded matrices.
// Usually Both M and N are very "thin" matrices of shape (N, k) with k << N.
//
// The case k == 1 is directly equivalent to the vector outer product operation,
// which is used most often.
//
template <typename T>
class OuterMatMatOp : public OpKernel {
 public:
  explicit OuterMatMatOp(OpKernelConstruction *context)
      : OpKernel(context) {
    LOAD_ATTRIBUTE_OP(
      context, "result_lower_bandwidth", &result_lower_bandwidth_);
    LOAD_ATTRIBUTE_OP(
      context, "result_upper_bandwidth", &result_upper_bandwidth_);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& left_tensor = context->input(0);
    const Tensor& right_tensor = context->input(1);

    OP_REQUIRES_OK(context, compute_outer_mat_mat<T>(
      context,
      left_tensor, right_tensor,
      result_lower_bandwidth_, result_upper_bandwidth_));
  }

 private:
  int result_lower_bandwidth_;
  int result_upper_bandwidth_;
};


//
// In the special case M.M^T where the same non-banded matrix is used on the
// left and on the right, this operator can be used instead.
// The main point here is that the gradient registered to it deals has special
// treatment for the case where we want the lower band of the symmetric output.
//
template <typename T>
class SquareMatOp : public OpKernel {
 public:
  explicit SquareMatOp(OpKernelConstruction *context)
      : OpKernel(context) {
    LOAD_ATTRIBUTE_OP(
      context, "result_lower_bandwidth", &result_lower_bandwidth_);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& left_tensor = context->input(0);
    OP_REQUIRES_OK(context, compute_outer_mat_mat<T>(
      context, left_tensor, left_tensor, result_lower_bandwidth_, 0));
  }

 private:
  // Upper bandwidth is always 0 due to symmetry
  int result_lower_bandwidth_;
};


//
// Operator registration
//

using InferenceContext = ::tensorflow::shape_inference::InferenceContext;
using DimensionHandle = ::tensorflow::shape_inference::DimensionHandle;

REGISTER_OP("OuterVecVec")
    .Attr("T: {float, double}")
    .Input("left_vector: T")
    .Input("right_vector: T")

    .Attr("result_lower_bandwidth: int")
    .Attr("result_upper_bandwidth: int")

    .Output("banded_outer: T")
    .SetShapeFn([](InferenceContext *context) {
      const auto uninitialized = std::numeric_limits<int>::min();
      int result_lower_bandwidth = uninitialized;
      int result_upper_bandwidth = uninitialized;

      LOAD_ATTRIBUTE_RETURN_IF_ERROR(context, "result_lower_bandwidth",
                                     &result_lower_bandwidth);
      LOAD_ATTRIBUTE_RETURN_IF_ERROR(context, "result_upper_bandwidth",
                                     &result_upper_bandwidth);
      DimensionHandle dim = context->Dim(context->input(0), 0);

      context->set_output(
          0,
          context->Matrix(
            result_lower_bandwidth + 1 + result_upper_bandwidth,
            dim));
      return Status::OK();
    });

REGISTER_CPU(OuterVecVec, float)
REGISTER_CPU(OuterVecVec, double)


REGISTER_OP("OuterMatMat")
    .Attr("T: {float, double}")
    .Input("left_vector: T")
    .Input("right_vector: T")

    .Attr("result_lower_bandwidth: int")
    .Attr("result_upper_bandwidth: int")

    .Output("banded_outer: T")
    .SetShapeFn([](InferenceContext *context) {
      int result_lower_bandwidth;
      int result_upper_bandwidth;

      LOAD_ATTRIBUTE_RETURN_IF_ERROR(context, "result_lower_bandwidth",
                                     &result_lower_bandwidth);
      LOAD_ATTRIBUTE_RETURN_IF_ERROR(context, "result_upper_bandwidth",
                                     &result_upper_bandwidth);
      DimensionHandle dim = context->Dim(context->input(0), 0);

      context->set_output(
          0,
          context->Matrix(
            result_lower_bandwidth + 1 + result_upper_bandwidth,
            dim));
      return Status::OK();
    });

REGISTER_CPU(OuterMatMat, float)
REGISTER_CPU(OuterMatMat, double)


REGISTER_OP("SquareMat")
    .Attr("T: {float, double}")
    .Input("left_vector: T")

    .Attr("result_lower_bandwidth: int")

    .Output("banded_outer: T")
    .SetShapeFn([](InferenceContext *context) {
      int result_lower_bandwidth;

      LOAD_ATTRIBUTE_RETURN_IF_ERROR(context, "result_lower_bandwidth",
                                     &result_lower_bandwidth);
      DimensionHandle dim = context->Dim(context->input(0), 0);

      context->set_output(
          0, context->Matrix(result_lower_bandwidth + 1, dim));
      return Status::OK();
    });

REGISTER_CPU(SquareMat, float)
REGISTER_CPU(SquareMat, double)

}  // end of namespace tensorflow
