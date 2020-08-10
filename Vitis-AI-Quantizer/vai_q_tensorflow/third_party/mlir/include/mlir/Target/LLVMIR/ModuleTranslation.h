//===- ModuleTranslation.h - MLIR to LLVM conversion ------------*- C++ -*-===//
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
//
// This file implements the translation between an MLIR LLVM dialect module and
// the corresponding LLVMIR module. It only handles core LLVM IR operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_MODULETRANSLATION_H
#define MLIR_TARGET_LLVMIR_MODULETRANSLATION_H

#include "mlir/IR/Block.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Value.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"

namespace mlir {
class Attribute;
class FuncOp;
class Location;
class ModuleOp;
class Operation;

namespace LLVM {

// Implementation class for module translation.  Holds a reference to the module
// being translated, and the mappings between the original and the translated
// functions, basic blocks and values.  It is practically easier to hold these
// mappings in one class since the conversion of control flow operations
// needs to look up block and function mappings.
class ModuleTranslation {
public:
  template <typename T = ModuleTranslation>
  static std::unique_ptr<llvm::Module> translateModule(ModuleOp m) {
    auto llvmModule = prepareLLVMModule(m);

    T translator(m);
    translator.llvmModule = std::move(llvmModule);
    translator.convertGlobals();
    if (failed(translator.convertFunctions()))
      return nullptr;

    return std::move(translator.llvmModule);
  }

protected:
  // Translate the given MLIR module expressed in MLIR LLVM IR dialect into an
  // LLVM IR module.  The MLIR LLVM IR dialect holds a pointer to an
  // LLVMContext, the LLVM IR module will be created in that context.
  explicit ModuleTranslation(ModuleOp module) : mlirModule(module) {}
  virtual ~ModuleTranslation() {}

  virtual LogicalResult convertOperation(Operation &op,
                                         llvm::IRBuilder<> &builder);
  static std::unique_ptr<llvm::Module> prepareLLVMModule(ModuleOp m);

private:
  LogicalResult convertFunctions();
  void convertGlobals();
  LogicalResult convertOneFunction(FuncOp func);
  void connectPHINodes(FuncOp func);
  LogicalResult convertBlock(Block &bb, bool ignoreArguments);

  template <typename Range>
  SmallVector<llvm::Value *, 8> lookupValues(Range &&values);

  llvm::Constant *getLLVMConstant(llvm::Type *llvmType, Attribute attr,
                                  Location loc);

  // Original and translated module.
  ModuleOp mlirModule;
  std::unique_ptr<llvm::Module> llvmModule;

  // Mappings between llvm.global definitions and corresponding globals.
  llvm::DenseMap<Operation *, llvm::GlobalValue *> globalsMapping;

protected:
  // Mappings between original and translated values, used for lookups.
  llvm::StringMap<llvm::Function *> functionMapping;
  llvm::DenseMap<Value *, llvm::Value *> valueMapping;
  llvm::DenseMap<Block *, llvm::BasicBlock *> blockMapping;
};

} // namespace LLVM
} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_MODULETRANSLATION_H
