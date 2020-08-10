/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_CONVERT_TYPE_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_CONVERT_TYPE_H_

#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

// Converts the TensorFlow DataType 'dtype' into an MLIR (scalar) type.
Status ConvertDataType(const DataType& dtype, mlir::Builder builder,
                       mlir::Type* type);

// Converts a scalar MLIR type to a TensorFlow Datatype.
Status ConvertScalarTypeToDataType(mlir::Type type, DataType* dtype);

// Converts an MLIR type to TensorFlow DataType. If 'type' is a scalar type, it
// is converted directly. If it is a shaped type, the element type is converted.
Status ConvertToDataType(mlir::Type type, DataType* dtype);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_CONVERT_TYPE_H_
