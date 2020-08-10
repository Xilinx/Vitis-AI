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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_MLIR_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_MLIR_COMPILER_H_

#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "tensorflow/compiler/xla/service/compiler.h"

namespace xla {
namespace mlir {

// A Compiler implementation that converts XLAs IR to a matching MLIR dialect,
// performs all lowering on the MLIR IR and finally converts MLIR to LLVMIR for
// generation of a think suitable for XLAs runtime.
class MlirCompiler : public Compiler {
 public:
  MlirCompiler();

  se::Platform::Id PlatformId() const override;

  StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      se::DeviceMemoryAllocator* device_allocator) override;

  StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      se::DeviceMemoryAllocator* device_allocator) override;

  StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<se::StreamExecutor*>> stream_execs,
      se::DeviceMemoryAllocator* device_allocator) override;

  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     const AotCompilationOptions& options) override;

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override {
    int64 pointer_size = pointer_size_;
    return [pointer_size](const Shape& shape) {
      return ShapeUtil::ByteSizeOf(shape, pointer_size);
    };
  }

 private:
  ::mlir::MLIRContext context_;
  int64 pointer_size_;
};

}  // namespace mlir
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_MLIR_COMPILER_H_
