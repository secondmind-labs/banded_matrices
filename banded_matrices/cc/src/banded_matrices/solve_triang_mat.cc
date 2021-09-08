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
template <typename T> using LowerTriangularBandedMatrix =
  banded::LowerTriangularBandedMatrix<T>;

//
// TensorFlow operator for Solve between
// a banded matrix that is lower or upper-triangular
// and a non-banded matrix (several vectors solved at once).
//
template <typename T>
class SolveTriangMatOp : public OpKernel {
 public:
  explicit SolveTriangMatOp(OpKernelConstruction *context)
      : OpKernel(context) {
    LOAD_ATTRIBUTE_OP(context, "transpose_left", &transpose_left_);
  }

  void Compute(OpKernelContext *context) override {
    // Get the input tensors, and their shapes
    const Tensor& left_tensor = context->input(0);
    const Tensor& right_tensor = context->input(1);

    const TensorShape& left_shape = left_tensor.shape();
    const TensorShape& right_shape = right_tensor.shape();

    const auto left_width = left_shape.dim_size(0);

    // Get/create the output tensor
    const auto dim = left_shape.dim_size(1);
    const auto num_vectors = right_shape.dim_size(1);

    // Check dimensions and parameters
    OP_REQUIRES(
      context,
      TensorShapeUtils::IsMatrix(left_shape),
      errors::InvalidArgument(
        "SolveTriangMat operation expects a matrix as left argument."));

    OP_REQUIRES(
      context,
      TensorShapeUtils::IsMatrix(right_shape),
      errors::InvalidArgument(
        "SolveTriangVec operation expects a matrix as right argument."));

    OP_REQUIRES(
        context,
        dim == right_shape.dim_size(0),
        errors::InvalidArgument(
          "Vector length(s) do not match banded matrix size."));

    // Allocate the result:
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
      context->allocate_output(0, right_shape, &output_tensor));

    // View each pointer as properly dimensioned matrices

    // Note that the left matrix is always lower-triangular.
    // It may, however, be transposed.
    const auto l = banded::const_lower_triangular_view(
      left_tensor.flat<T>().data(), left_width, dim);

    banded::MatrixConstMap<T> right(
      right_tensor.flat<T>().data(), dim, num_vectors);
    banded::MatrixMap<T> result(
      output_tensor->flat<T>().data(), dim, num_vectors);

    // Perform the actual operator forward evaluation:
    if (transpose_left_) {
      const auto left = Transposed<LowerTriangularBandedMatrix<T>>(l);
      solve_upper_band_mat(left, right, &result);

    } else {
      const auto& left = l;
      solve_lower_band_mat(left, right, &result);
    }
  }

 private:
  bool transpose_left_;
};


//
// Operator registration
//

using InferenceContext = ::tensorflow::shape_inference::InferenceContext;

REGISTER_OP("SolveTriangMat")
    .Attr("T: {float, double}")
    .Input("left_banded_matrix: T")
    .Input("right_vector: T")
    .Attr("transpose_left: bool")
    .Output("solved: T")
    .SetShapeFn([](InferenceContext *context) {
      // Note that input(1) is assumed here to be an Mx1 matrix:
      context->set_output(0, context->input(1));
      return Status::OK();
    });

REGISTER_CPU(SolveTriangMat, float)
REGISTER_CPU(SolveTriangMat, double)

}  // end of namespace tensorflow
