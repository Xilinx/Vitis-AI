//===- AddDefaultStatsTestPass.cpp - Testing pass to add default stats ----===//
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
// This file defines a testing pass to add default statistics nodes to every
// quantization eligible op. Useful for unit testing.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/QuantOps/QuantOps.h"
#include "mlir/Dialect/QuantOps/QuantTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Quantizer/Configurations/FxpMathConfig.h"
#include "mlir/Quantizer/Support/Configuration.h"
#include "mlir/Quantizer/Support/ConstraintAnalysisGraph.h"
#include "mlir/Quantizer/Support/ConstraintAnalysisGraphTraits.h"
#include "mlir/Quantizer/Transforms/Passes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::quantizer;
using namespace mlir::quant;

namespace {

class AddDefaultStatsPass : public FunctionPass<AddDefaultStatsPass> {
public:
  AddDefaultStatsPass() = default;
  AddDefaultStatsPass(SolverContext &solverContext,
                      const TargetConfiguration &config)
      : explicitSolverContext(&solverContext), explicitConfig(&config) {}

  void runOnFunction() override;
  void runWithConfig(SolverContext &solverContext,
                     const TargetConfiguration &config);

private:
  SolverContext *explicitSolverContext = nullptr;
  const TargetConfiguration *explicitConfig = nullptr;
};

} // end anonymous namespace

void AddDefaultStatsPass::runOnFunction() {
  if (explicitSolverContext && explicitConfig) {
    // If explicitly constructed with a config and context.
    runWithConfig(*explicitSolverContext, *explicitConfig);
    return;
  }
  // For global pass registration, use defaults.
  SolverContext solverContext(*getFunction().getContext());
  auto config = FxpMathTargetConfig::create(solverContext);
  runWithConfig(solverContext, *config);
}

void AddDefaultStatsPass::runWithConfig(SolverContext &solverContext,
                                        const TargetConfiguration &config) {
  auto func = getFunction();

  // Insert stats for each argument.
  for (auto *arg : func.getArguments()) {
    if (!config.isHandledType(arg->getType()))
      continue;
    OpBuilder b(func.getBody());
    APFloat minValue(-1.0f);
    APFloat maxValue(1.0f);
    ElementsAttr layerStats = DenseFPElementsAttr::get(
        b.getTensorType({2}, b.getF32Type()), {minValue, maxValue});
    auto statsOp =
        b.create<StatisticsOp>(func.getLoc(), arg, layerStats, nullptr);
    arg->replaceAllUsesWith(statsOp);

    // StatsOp contained a use to 'arg' so make sure to reset it after replacing
    // all of the uses of 'arg'.
    statsOp.getOperation()->replaceUsesOfWith(statsOp, arg);
  }

  // Walk the ops and insert stats.
  func.walk([&](Operation *op) {
    if (!config.isRequireStatsOp(op)) {
      return;
    }
    assert(op->getNumResults() == 1);

    auto originalResult = op->getResult(0);
    if (!config.isHandledType(originalResult->getType()))
      return;

    OpBuilder b(op->getBlock(), ++op->getIterator());

    APFloat minValue(-1.0f);
    APFloat maxValue(1.0f);
    ElementsAttr layerStats = DenseFPElementsAttr::get(
        b.getTensorType({2}, b.getF32Type()), {minValue, maxValue});
    auto statsOp = b.create<StatisticsOp>(op->getLoc(), op->getResult(0),
                                          layerStats, nullptr);
    originalResult->replaceAllUsesWith(statsOp);

    // StatsOp contained a use to 'op' so make sure to reset it after replacing
    // all of the uses of 'op'.
    statsOp.getOperation()->replaceUsesOfWith(statsOp, originalResult);
  });
}

std::unique_ptr<FunctionPassBase> mlir::quantizer::createAddDefaultStatsPass() {
  return std::make_unique<AddDefaultStatsPass>();
}

static PassRegistration<AddDefaultStatsPass> pass(
    "quantizer-add-default-stats-test",
    "Adds default (dummy) statistics to all ops that can benefit from "
    "runtime statistics. This is meant to help in early stage bootstrapping.");
