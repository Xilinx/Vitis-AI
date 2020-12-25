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

#include "tensorflow/compiler/mlir/tensorflow/ir/control_flow_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {

// Static initialization for TF dialect registration.
static DialectRegistration<TFControlFlow::TFControlFlowDialect>
    tf_control_flow_ops;
static DialectRegistration<TF::TensorFlowDialect> tf_ops;
static DialectRegistration<tf_executor::TensorFlowExecutorDialect>
    tf_excutor_dialect;
static DialectRegistration<tf_device::TensorFlowDeviceDialect>
    tf_device_dialect;

}  // namespace mlir
