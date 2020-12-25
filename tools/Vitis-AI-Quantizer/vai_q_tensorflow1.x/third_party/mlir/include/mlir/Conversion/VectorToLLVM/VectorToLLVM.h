//===- VectorToLLVM.h - Pass converting vector to LLVM dialect --*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#ifndef MLIR_CONVERSION_VECTORTOLLVM_VECTORTOLLVM_H_
#define MLIR_CONVERSION_VECTORTOLLVM_VECTORTOLLVM_H_

namespace mlir {
class LLVMTypeConverter;
class ModulePassBase;
class OwningRewritePatternList;

/// Collect a set of patterns to convert from the Vector dialect to LLVM.
void populateVectorToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                            OwningRewritePatternList &patterns);

/// Create a pass to convert vector operations to the LLVMIR dialect.
ModulePassBase *createLowerVectorToLLVMPass();
} // namespace mlir

#endif // MLIR_CONVERSION_VECTORTOLLVM_VECTORTOLLVM_H_
