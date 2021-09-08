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
#include "banded_matrices/banded_matrix.hpp"
#include "banded_matrices/unary_broadcastable_op_kernel.hpp"

namespace tensorflow {

using Index = Eigen::Index;


//
// Tensorflow operator to change banded representation
// from banded to block-banded.
//
template <typename T>
class BlockToBandOp : public UnaryBroadcastableOpKernel<T, 2> {
 public:
  explicit BlockToBandOp(OpKernelConstruction *context)
      : UnaryBroadcastableOpKernel<T, 2>(context) {
    LOAD_ATTRIBUTE_OP(context, "block_size", &block_size_);
    LOAD_ATTRIBUTE_OP(context, "symmetric", &symmetric_);
    LOAD_ATTRIBUTE_OP(context, "gradient", &gradient_);
  }

  void StartChecks(OpKernelContext *context,
                   const TensorShape& unit_input_shape) override {
  // Check dimensions and parameters
    const auto rows = unit_input_shape.dim_size(0);
    const auto cols = unit_input_shape.dim_size(1);
    OP_REQUIRES(context,
      TensorShapeUtils::IsMatrix(unit_input_shape),
      errors::InvalidArgument("BlockToBand operation expects a matrix"));

    OP_REQUIRES(context,
      block_size_ > 0,
      errors::InvalidArgument("block size must be > 0"));

    OP_REQUIRES(context,
      cols % block_size_ == 0,
      errors::InvalidArgument(
        "Matrix column numbers must be integer multiple of blocksize"));

    OP_REQUIRES(context,
      rows % block_size_ == 0,
      errors::InvalidArgument(
        "Matrix row numbers must be integer multiple of blocksize"));
  }

  void UnitOutputShape(const TensorShape&  unit_input_shape,
                              TensorShape * unit_output_shape) override {
    unit_output_shape->Clear();
    unit_output_shape->AppendShape(unit_input_shape);
  }

  void UnitCompute(const std::vector<Tensor>& unit_input_tensors,
                   Tensor* unit_output_tensor) override {
    // Get the input tensor, and its shape
    const Tensor& input_tensor = unit_input_tensors[0];
    const TensorShape& input_shape = input_tensor.shape();
    const auto rows = input_shape.dim_size(0);
    const auto cols = input_shape.dim_size(1);

    const auto col_blocks = cols / block_size_;
    const T scaling = (gradient_ && symmetric_) ? 2.0 : 1.0;

    // View the tensors as dense matrices
    banded::MatrixConstMap<T> input{
      input_tensor.flat<T>().data(), rows, cols };
    banded::MatrixMap<T> result{
      unit_output_tensor->flat<T>().data(), rows, cols};
    result.setZero();

    // TODO(pm): check efficiency
    for (Index col_block = 0; col_block < col_blocks; ++col_block) {
      for (Index sub_block = 0; sub_block < block_size_; ++sub_block) {
        for (Index row_block = 0; row_block < rows-sub_block; ++row_block) {
           result(row_block, col_block*block_size_ + sub_block) =
             input(row_block+sub_block, col_block*block_size_ + sub_block);
           if (row_block > 0)
             result(row_block, col_block*block_size_ + sub_block) *= scaling;
        }
      }
    }
  }
  void  ResultsChecks(OpKernelContext *,
                      const std::vector<Tensor>& , const Tensor&) override{};

 private:
  int block_size_;
  bool symmetric_;
  bool gradient_;
};


//
// Tensorflow operator to change banded representation
// from block banded to banded
//
template <typename T>
class BandToBlockOp : public UnaryBroadcastableOpKernel<T, 2> {
 public:
  explicit BandToBlockOp(OpKernelConstruction *context)
      : UnaryBroadcastableOpKernel<T, 2>(context) {
    LOAD_ATTRIBUTE_OP(context, "block_size", &block_size_);
    LOAD_ATTRIBUTE_OP(context, "symmetric", &symmetric_);
    LOAD_ATTRIBUTE_OP(context, "gradient", &gradient_);
  }


  void StartChecks(OpKernelContext *context,
                   const TensorShape& unit_input_shape) override {
  // Check dimensions and parameters
    const auto rows = unit_input_shape.dim_size(0);
    const auto cols = unit_input_shape.dim_size(1);
    OP_REQUIRES(context,
      TensorShapeUtils::IsMatrix(unit_input_shape),
      errors::InvalidArgument("BandToBlock operation expects a matrix"));

    OP_REQUIRES(context,
      block_size_ > 0,
      errors::InvalidArgument("block size must be > 0"));

    OP_REQUIRES(context,
      cols % block_size_ == 0,
      errors::InvalidArgument(
        "Matrix column numbers must be integer multiple of blocksize"));

    OP_REQUIRES(context,
      rows % block_size_ == 0,
      errors::InvalidArgument(
        "Matrix row numbers must be integer multiple of blocksize"));
  }

  void UnitOutputShape(const TensorShape&  unit_input_shape,
                              TensorShape * unit_output_shape) override {
    unit_output_shape->Clear();
    unit_output_shape->AppendShape(unit_input_shape);
  }

  void UnitCompute(const std::vector<Tensor>& unit_input_tensors,
                   Tensor* unit_output_tensor) override {
    // Get the input tensor, and its shape
    const Tensor& unit_input_tensor = unit_input_tensors[0];
    const TensorShape& input_shape = unit_input_tensor.shape();
    const auto rows = input_shape.dim_size(0);
    const auto cols = input_shape.dim_size(1);

    const auto col_blocks = cols / block_size_;
    const T scaling = gradient_ ? 0.5 : 1.0;


    // View the tensors as dense matrices
    const banded::MatrixConstMap<T> input{
      unit_input_tensor.flat<T>().data(), rows, cols };
    banded::MatrixMap<T> result{
      unit_output_tensor->flat<T>().data(), rows, cols};

    result.setZero();

    // TODO(pm): check efficiency
    for (Index col_block = 0; col_block < col_blocks; ++col_block) {
      for (Index sub_block = 0; sub_block < block_size_; ++sub_block) {
        for (Index i = 0; i < rows-sub_block; ++i) {
           // move column down
           result(sub_block+i, col_block*block_size_ + sub_block) =
             input(i, col_block*block_size_ + sub_block);
        }
        if (symmetric_) {
            for (Index i = 1; i < block_size_-sub_block; ++i) {
                // symmetrise first block
                result(sub_block + i, col_block*block_size_ + sub_block) *=
                    scaling;
                result(sub_block, col_block*block_size_ + sub_block + i) =
                    result(sub_block + i, col_block*block_size_ + sub_block);
            }
        }
      }
    }
  }

  void ResultsChecks(OpKernelContext *,
                     const std::vector<Tensor>& , const Tensor&) override{};

 private:
  int block_size_;
  bool symmetric_;
  bool gradient_;
};


//
// Register block band
//

using InferenceContext = ::tensorflow::shape_inference::InferenceContext;

REGISTER_OP("BlockToBand")
    .Attr("T: {float, double}")
    .Input("tensor: T")
    .Attr("block_size: int")
    .Attr("symmetric: bool")
    .Attr("gradient: bool")
    .Output("block_band: T")
    .SetShapeFn([](InferenceContext *context) {
      context->set_output(0, context->input(0));
      return Status::OK();
    });

REGISTER_CPU(BlockToBand, float)
REGISTER_CPU(BlockToBand, double)

REGISTER_OP("BandToBlock")
    .Attr("T: {float, double}")
    .Input("tensor: T")
    .Attr("block_size: int")
    .Attr("symmetric: bool")
    .Attr("gradient: bool")
    .Output("block_band: T")
    .SetShapeFn([](InferenceContext *context) {
      context->set_output(0, context->input(0));
      return Status::OK();
    });

REGISTER_CPU(BandToBlock, float)
REGISTER_CPU(BandToBlock, double)

}  // end of namespace tensorflow
