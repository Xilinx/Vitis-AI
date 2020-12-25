//===- Types.cpp - Implementations of MLIR Core C APIs --------------------===//
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

#include "mlir-c/Core.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

mlir_type_t makeScalarType(mlir_context_t context, const char *name,
                           unsigned bitwidth) {
  mlir::MLIRContext *c = reinterpret_cast<mlir::MLIRContext *>(context);
  mlir_type_t res =
      llvm::StringSwitch<mlir_type_t>(name)
          .Case("bf16",
                mlir_type_t{mlir::FloatType::getBF16(c).getAsOpaquePointer()})
          .Case("f16",
                mlir_type_t{mlir::FloatType::getF16(c).getAsOpaquePointer()})
          .Case("f32",
                mlir_type_t{mlir::FloatType::getF32(c).getAsOpaquePointer()})
          .Case("f64",
                mlir_type_t{mlir::FloatType::getF64(c).getAsOpaquePointer()})
          .Case("index",
                mlir_type_t{mlir::IndexType::get(c).getAsOpaquePointer()})
          .Case("i",
                mlir_type_t{
                    mlir::IntegerType::get(bitwidth, c).getAsOpaquePointer()})
          .Default(mlir_type_t{nullptr});
  if (!res) {
    llvm_unreachable("Invalid type specifier");
  }
  return res;
}

mlir_type_t makeMemRefType(mlir_context_t context, mlir_type_t elemType,
                           int64_list_t sizes) {
  auto t = mlir::MemRefType::get(
      llvm::ArrayRef<int64_t>(sizes.values, sizes.n),
      mlir::Type::getFromOpaquePointer(elemType),
      {mlir::AffineMap::getMultiDimIdentityMap(
          sizes.n, reinterpret_cast<mlir::MLIRContext *>(context))},
      0);
  return mlir_type_t{t.getAsOpaquePointer()};
}

mlir_type_t makeFunctionType(mlir_context_t context, mlir_type_list_t inputs,
                             mlir_type_list_t outputs) {
  llvm::SmallVector<mlir::Type, 8> ins(inputs.n), outs(outputs.n);
  for (unsigned i = 0; i < inputs.n; ++i) {
    ins[i] = mlir::Type::getFromOpaquePointer(inputs.types[i]);
  }
  for (unsigned i = 0; i < outputs.n; ++i) {
    outs[i] = mlir::Type::getFromOpaquePointer(outputs.types[i]);
  }
  auto ft = mlir::FunctionType::get(
      ins, outs, reinterpret_cast<mlir::MLIRContext *>(context));
  return mlir_type_t{ft.getAsOpaquePointer()};
}

mlir_type_t makeIndexType(mlir_context_t context) {
  auto *ctx = reinterpret_cast<mlir::MLIRContext *>(context);
  auto type = mlir::IndexType::get(ctx);
  return mlir_type_t{type.getAsOpaquePointer()};
}

mlir_attr_t makeIntegerAttr(mlir_type_t type, int64_t value) {
  auto ty = Type::getFromOpaquePointer(reinterpret_cast<const void *>(type));
  auto attr = IntegerAttr::get(ty, value);
  return mlir_attr_t{attr.getAsOpaquePointer()};
}

mlir_attr_t makeBoolAttr(mlir_context_t context, bool value) {
  auto *ctx = reinterpret_cast<mlir::MLIRContext *>(context);
  auto attr = BoolAttr::get(value, ctx);
  return mlir_attr_t{attr.getAsOpaquePointer()};
}

unsigned getFunctionArity(mlir_func_t function) {
  auto f = mlir::FuncOp::getFromOpaquePointer(function);
  return f.getNumArguments();
}
