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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"


#define REGISTER_CPU(name, T) \
  REGISTER_KERNEL_BUILDER(    \
    Name(#name)               \
    .Device(DEVICE_CPU)       \
    .TypeConstraint<T>("T"),  \
    name ## Op<T>)


// Set the value pointed to by dest to the value of the named attribute. If
// the value for the requested attribute is not set or the requested
// attribute does not exist then this macro reports the error Status in the
// context and returns from its enclosing function.
//
// For use inside the constructor or Compute method of Ops (NOT another function
// called by them). For nested functions you should use
// LOAD_ATTRIBUTE_RETURN_IF_ERROR and propagate the status manually.
//
// Must be a macro as OP_REQUIRES_OK includes the file and line number in the
// error and then uses a return statement. Neither of these work as intended
// if this is a function.
#define LOAD_ATTRIBUTE_OP(context, name, dest) \
    OP_REQUIRES_OK(context, context->GetAttr(name, dest))


// Set the value pointed to by dest to the value of the named attribute. If
// the value for the requested attribute is not set or the requested
// attribute does not exist then this macro returns an error Status from
// its enclosing function.
//
// For use outside of Ops, such as `SetShapeFn` where the enclosing
// function returns a Status.
//
// Must be a macro as TF_RETURN_IF_ERROR contains a return statement.
// This doesn't work as intended if this is a function.
#define LOAD_ATTRIBUTE_RETURN_IF_ERROR(context, name, dest) \
    TF_RETURN_IF_ERROR(context->GetAttr(name, dest))


namespace banded {


template <typename T> using Matrix =
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <typename T> using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T> using RowVector = Eigen::Matrix<T, 1, Eigen::Dynamic>;
template <typename T> using Array = Eigen::Array<T, Eigen::Dynamic, 1>;

template <typename T> using MatrixMap = Eigen::Map<Matrix<T>>;
template <typename T> using VectorMap = Eigen::Map<Vector<T>>;
template <typename T> using RowVectorMap = Eigen::Map<RowVector<T>>;

template <typename T> using MatrixConstMap = Eigen::Map<const Matrix<T>>;
template <typename T> using VectorConstMap = Eigen::Map<const Vector<T>>;
template <typename T> using RowVectorConstMap = Eigen::Map<const RowVector<T>>;

}  // end of namespace banded
