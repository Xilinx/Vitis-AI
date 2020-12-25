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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_HLO_TO_MLIR_HLO_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_HLO_TO_MLIR_HLO_H_

#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/xla/status.h"

namespace mlir {
class ModuleOp;
}  // namespace mlir

namespace xla {
class HloModule;
class HloModuleProto;

// Converts an HLO module proto to a MLIR module in HLO dialect.
Status ConvertHloToMlirHlo(mlir::ModuleOp module,
                           xla::HloModuleProto* hlo_module);

// Converts an HLO module to a MLIR module in HLO dialect.
Status ConvertHloToMlirHlo(mlir::ModuleOp module, xla::HloModule* hlo_module);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_HLO_TO_MLIR_HLO_H_
