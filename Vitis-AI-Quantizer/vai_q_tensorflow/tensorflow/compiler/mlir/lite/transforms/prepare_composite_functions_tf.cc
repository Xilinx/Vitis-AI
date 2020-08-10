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

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Identifier.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/SymbolTable.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"

namespace mlir {
namespace TFL {
namespace {

// Abstracts the conversion of the embedded lookup composite function.
class ConvertEmbeddedLookupFunc {
 public:
  explicit ConvertEmbeddedLookupFunc(FuncOp func) : func_(func) {}

  void RewriteFunc() {
    func_.eraseBody();
    func_.addEntryBlock();
    func_.setAttr(
        "tf._implements",
        StringAttr::get("fused_tfl_embedding_lookup", func_.getContext()));
    Value* lookup = func_.getArgument(1);
    Value* value = func_.getArgument(0);
    auto output_type = func_.getType().getResult(0);

    OpBuilder builder(func_.getBody());
    auto op = builder.create<mlir::TFL::EmbeddingLookupOp>(
        func_.getLoc(), output_type, lookup, value);

    builder.create<mlir::ReturnOp>(func_.getLoc(), op.getResult());
  }

  LogicalResult VerifySignature() {
    if (func_.getNumArguments() != 2) {
      return func_.emitError()
             << "Invalid number of arguments in the embedding "
                "matmal composite function";
    }
    if (func_.getType().getNumResults() != 1) {
      return func_.emitError() << "Invalid number of results in the embedding "
                                  "matmal composite function";
    }
    return success();
  }

 private:
  FuncOp func_;
};

// This pass uses mechanisms listed in RFC:
// https://github.com/tensorflow/community/pull/113
// It prepares composite functions that are attributed to indicate
// a specific interface (LSTM, SVDF, Embedding lookup etc.) by replacing the
// body with the corresponding fused TFLite op. The replacement need not always
// be a fused op, though that is the primary use case.
class PrepareCompositeFunctionsPass
    : public FunctionPass<PrepareCompositeFunctionsPass> {
 public:
  explicit PrepareCompositeFunctionsPass() {}

 private:
  void runOnFunction() override;
};

void PrepareCompositeFunctionsPass::runOnFunction() {
  // TODO(ashwinm): Explore if we can generalize this pass by simply taking
  // a map<func annotation, tfl op> and doing the transform. This should be
  // revisited after we add LSTM composite op to this pass.
  auto func = getFunction();
  auto attr = func.getAttrOfType<StringAttr>("tf._implements");
  if (!attr || attr.getValue() != "embedding_matmul") return;
  // Convert the composite embedding_matmul function body to a
  // TFLite fused embedding_lookup op.
  ConvertEmbeddedLookupFunc convert_embedded_lookup(func);
  if (failed(convert_embedded_lookup.VerifySignature())) {
    return signalPassFailure();
  }
  convert_embedded_lookup.RewriteFunc();
}
}  // namespace

std::unique_ptr<FunctionPassBase> CreatePrepareCompositeFunctionsPass() {
  return std::unique_ptr<PrepareCompositeFunctionsPass>();
}

static PassRegistration<PrepareCompositeFunctionsPass> pass(
    "tfl-prepare-composite-funcs-tf",
    "Prepares composite functions in Tensorflow dialect of MLIR ");

}  // namespace TFL
}  // namespace mlir
