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
#pragma once

#define EIGEN_USE_THREADS

#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/env.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "Eigen/Dense"

#include "banded_matrices/banded_matrix.hpp"
#include "banded_matrices/common.hpp"


namespace tensorflow {

//
// Class implementing interface to allow Broadcasting for Unary Ops
//
// Custom C++ TF Ops only need to implement one method - `Compute` - within
// which input checks, output allocation and the logical computation are done.
// This class allows Child classes to simply define shapes and operations for
// input Tensors of rank UnitRank, and then takes care of can the be
// broadcasting and parallelisation logic for all leading dimensions.
template <typename T, int UnitRank>
class UnaryBroadcastableOpKernel : public OpKernel {
 public:
  explicit UnaryBroadcastableOpKernel(OpKernelConstruction* context)
      : OpKernel(context) {}

  typedef typename TTypes<T, 1 + UnitRank>::ConstTensor FlattenedConstTensor;

  void Compute(OpKernelContext* context) override {
    int num_inputs = context->num_inputs();

    // 1. Get the shapes we need for our Unit computation
    std::vector<TensorShape> broadcasted_input_shapes(num_inputs);
    std::vector<TensorShape> unit_input_shapes(num_inputs);
    for (int j = 0; j < num_inputs; ++j) {
      broadcasted_input_shapes[j] = context->input(j).shape();
      this->UnitInputShape(broadcasted_input_shapes[j], &unit_input_shapes[j]);
    }

    TensorShape unit_output_shape{};
    TensorShape broadcasted_output_shape{};

    this->UnitOutputShape(unit_input_shapes[0], &unit_output_shape);
    this->BroadcastedOutputShape(broadcasted_input_shapes[0], unit_output_shape,
                                 &broadcasted_output_shape);

    // 2. Perform shape checks if any defined by child class
    this->StartChecks(context, unit_input_shapes[0]);

    // 3. Allocate output and return on error
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, broadcasted_output_shape, &output_tensor));


    if (broadcasted_output_shape == unit_output_shape) {
      // 3.5 If no need for broadcasting, just compute & return
      std::vector<Tensor> input_tensor_list;
      for (int j = 0; j < num_inputs; ++j) {
        input_tensor_list.push_back(context->input(j));
      }
      this->UnitCompute(input_tensor_list, output_tensor);
      this->ResultsChecks(context, input_tensor_list, *output_tensor);
      return;
    }

    // 4. Flatten input and output tensors, and get their dtypes
    std::vector<FlattenedConstTensor> flat_input_tensors;
    std::vector<DataType> flat_input_dtypes;
    for (int j = 0; j < num_inputs; ++j) {
      flat_input_tensors.push_back(
          context->input(j).flat_inner_dims<T, 1 + UnitRank>());
      flat_input_dtypes.push_back(context->input(j).dtype());
    }
    auto flat_output_matrices =
        output_tensor->flat_inner_dims<T, 1 + UnitRank>();
    const DataType output_dtype = output_tensor->dtype();

    // 5. Create lambdas to perform shard of work
    auto compute_unit_n = [&](int64 start, int64 end) {
      for (int64 n = start; n < end; n++) {
        // 5.1 Create unit tensors for one unit of computation
        Tensor unit_output_tensor(output_dtype, unit_output_shape);
        std::vector<Tensor> unit_input_tensor_list;

        // 5.2 Populate the unit input tensors
        for (int j = 0; j < num_inputs; ++j) {
          auto unit_input_tensor =
              Tensor(flat_input_dtypes[j], unit_input_shapes[j]);
          unit_input_tensor.tensor<T, UnitRank>() =
              flat_input_tensors[j].chip(n, 0);
          unit_input_tensor_list.push_back(unit_input_tensor);
        }

        // 5.3 Do the unit computation and check
        this->UnitCompute(unit_input_tensor_list, &unit_output_tensor);
        this->ResultsChecks(context, unit_input_tensor_list,
                            unit_output_tensor);
        // 5.4 Copy out data
        flat_output_matrices.chip(n, 0) =
            unit_output_tensor.tensor<T, UnitRank>();
      }
    };

    // 6. Create/get threadpool and run
    thread::ThreadPool* const pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;

    const thread::ThreadPool::SchedulingParams scheduling_params(
        thread::ThreadPool::SchedulingStrategy::kFixedBlockSize,  // -strategy`
        absl::nullopt,  // - `cost_per_unit`
        1);              // - `block_size`
    pool->ParallelFor(flat_output_matrices.dimension(0), scheduling_params,
                     compute_unit_n);
  };

 private:
  // Populates `unit_input_shape` with the shape of a unit of input.
  //
  // :param broadcasted_input_shape: The TensorShape of the whole (potentially
  //     broadcasted) input.
  // :param unit_input_shape: A pointer to a TensorShape to populate.
  void UnitInputShape(const TensorShape& broadcasted_input_shape,
                      TensorShape* unit_input_shape) {
    int broadcasting_rank = broadcasted_input_shape.dims() - UnitRank;

    unit_input_shape->Clear();
    unit_input_shape->AppendShape(broadcasted_input_shape);
    unit_input_shape->RemoveDimRange(0, broadcasting_rank);
  }

  // Populates `broadcasted_output_shape` with the final (potentially
  // broadcasted) output shape.
  //
  // :param broadcasted_input_shape: The TensorShape of the whole (potentially
  //     broadcasted) input.
  // :param unit_output_shape: The TensorShape of the output of a unit of
  //    calculation.
  // :param broadcasted_output_shape: A pointer to a TensorShape to populate.
  void BroadcastedOutputShape(const TensorShape& broadcasted_input_shape,
                              const TensorShape& unit_output_shape,
                              TensorShape* broadcasted_output_shape) {
    int broadcasting_rank = broadcasted_input_shape.dims() - UnitRank;

    broadcasted_output_shape->Clear();
    broadcasted_output_shape->AppendShape(broadcasted_input_shape);
    broadcasted_output_shape->RemoveDimRange(broadcasting_rank,
                                             broadcasted_input_shape.dims());
    broadcasted_output_shape->AppendShape(unit_output_shape);
  }

 protected:
  // Perform any checks required on inputs before proceeding with op.
  //
  // :param context: The OpKernelContext used by the op
  // :param unit_input_shape: The shape of the input for a unit of compute.
  virtual void StartChecks(OpKernelContext* context,
                           const TensorShape& unit_input_shape) = 0;

  // Populate the output shape expected from a unit of compute.
  //
  // :param unit_input_shape: The TensorShape for a unit of compute.
  // :param unit_output_shape: A pointer to shape to be populated.
  virtual void UnitOutputShape(const TensorShape& unit_input_shape,
                               TensorShape* unit_output_shape) = 0;

  // Perform one unit of compute and store output in a unit output tensor.
  //
  // :param unit_input_tensors: A fixed vector containing the unit input
  //     tensors.
  // :param unit_output_shape: A pointer to output tensor storing the result.
  virtual void UnitCompute(const std::vector<Tensor>& unit_input_tensors,
                           Tensor* unit_output_tensor) = 0;

  // Perform any checks on the inputs and results.
  //
  // :param context: The OpKernelContext used by the op
  // :param unit_input_tensors: A fixed vector containing the unit input
  //        tensors
  // :param unit_output_shape: A pointer to output tensor storing the result
  virtual void ResultsChecks(OpKernelContext* context,
                             const std::vector<Tensor>& unit_input_tensors,
                             const Tensor& unit_output_tensor) = 0;
};

}  // namespace tensorflow
