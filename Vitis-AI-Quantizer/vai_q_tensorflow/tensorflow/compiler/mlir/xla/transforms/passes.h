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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_PASSES_H_

#include <memory>

namespace mlir {
class FunctionPassBase;

namespace XLA {

/// Lowers from TF dialect to XLA dialect.
std::unique_ptr<FunctionPassBase> createLegalizeTFPass();

/// Lowers XLA control flow ops to the Standard dialect.
std::unique_ptr<FunctionPassBase> createLegalizeControlFlowPass();

/// Lowers from XLA dialect to Standard dialect.
std::unique_ptr<FunctionPassBase> createLegalizeToStdPass();

}  // end namespace XLA
}  // end namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_PASSES_H_
