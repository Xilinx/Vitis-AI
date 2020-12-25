/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_TILED_DOT_EMITTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_TILED_DOT_EMITTER_H_

#include "llvm/IR/IRBuilder.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace cpu {

// These routines emit LLVM IR implementing tiled GEMM and GEMV routines.

void EmitRowMajorGemv(PrimitiveType scalar_type, tensorflow::int64 tile_rows,
                      tensorflow::int64 tile_cols, tensorflow::int64 m,
                      tensorflow::int64 k, llvm::Value* lhs, llvm::Value* rhs,
                      llvm::Value* addend, llvm::Value* result,
                      llvm::IRBuilder<>* b,
                      const HloModuleConfig& module_config);

void EmitColumnMajorGemv(PrimitiveType scalar_type, tensorflow::int64 tile_rows,
                         tensorflow::int64 tile_cols, tensorflow::int64 m,
                         tensorflow::int64 k, llvm::Value* lhs,
                         llvm::Value* rhs, llvm::Value* addend,
                         llvm::Value* result, llvm::IRBuilder<>* b,
                         const HloModuleConfig& module_config);

void EmitSmallGemm(PrimitiveType scalar_type, tensorflow::int64 m,
                   tensorflow::int64 k, tensorflow::int64 n,
                   tensorflow::int64 max_vectorization_width,
                   tensorflow::int64 max_vector_count,
                   tensorflow::int64 min_vectorization_width,
                   tensorflow::int64 tile_size_m, tensorflow::int64 tile_size_k,
                   llvm::Value* lhs, llvm::Value* rhs, llvm::Value* result,
                   llvm::IRBuilder<>* b, const HloModuleConfig& module_config);

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_TILED_DOT_EMITTER_H_
