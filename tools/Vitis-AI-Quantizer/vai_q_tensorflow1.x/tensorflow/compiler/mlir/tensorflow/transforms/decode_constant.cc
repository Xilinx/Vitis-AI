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

#include "tensorflow/compiler/mlir/tensorflow/transforms/decode_constant.h"

#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"

namespace mlir {
namespace TF {

namespace {

// If the given `op` is an constant op with an opaque value, decodes and resets
// its value into a readable one. Otherwise, does nothing. This function returns
// false if the given `op` is a tf.Constant op and we cannot decode its value.
bool DecodeOpaqueValueInConstantOp(Operation *op) {
  auto tfOp = dyn_cast<ConstOp>(op);
  if (!tfOp) return true;

  auto opaque_attr = tfOp.value().dyn_cast<OpaqueElementsAttr>();
  // Skip non-opaque values.
  if (!opaque_attr) return true;

  Builder builder(op->getContext());

  ElementsAttr decoded_attr;
  if (opaque_attr.decode(decoded_attr)) {
    op->emitOpError("has undecodable opaque value");
    return false;
  }

  op->setAttr("value", decoded_attr);

  return true;
}

// A pass to decode opaque constant values into readable ones.
struct DecodeConstant : public FunctionPass<DecodeConstant> {
  void runOnFunction() override {
    bool success = true;
    getFunction().walk([&success](Operation *op) {
      success &= DecodeOpaqueValueInConstantOp(op);
    });
    if (!success) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<FunctionPassBase> CreateDecodeConstantPass() {
  return std::make_unique<DecodeConstant>();
}

static PassRegistration<DecodeConstant> pass(
    "tf-decode-constant", "Decode opaque constant into human-readable ones");

}  // namespace TF
}  // namespace mlir
