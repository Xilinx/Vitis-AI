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

// This file implements logic for lowering XLA dialect to Standard dialect.

#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/xla/ir/xla_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

using mlir::Builder;
using mlir::FunctionPass;
using mlir::FunctionPassBase;
using mlir::OwningRewritePatternList;
using mlir::PassRegistration;

namespace mlir {
namespace XLA {
namespace {
#include "tensorflow/compiler/mlir/xla/transforms/generated_legalize_to_standard.inc"

struct CompareIConvert : public RewritePattern {
  explicit CompareIConvert(MLIRContext *context)
      : RewritePattern("xla_hlo.compare", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto compare_op = cast<CompareOp>(op);

    auto lhs = compare_op.lhs();
    auto rhs = compare_op.rhs();
    auto lhs_type = lhs->getType().cast<TensorType>();
    auto rhs_type = rhs->getType().cast<TensorType>();

    // Broadcasting not supported by this rewrite.
    if (lhs_type.getShape() != rhs_type.getShape()) return matchFailure();

    if (!lhs_type.getElementType().isa<IntegerType>() ||
        !rhs_type.getElementType().isa<IntegerType>())
      return matchFailure();

    auto comparison_direction = compare_op.comparison_direction();
    CmpIPredicate compare_predicate =
        llvm::StringSwitch<CmpIPredicate>(comparison_direction)
            .Case("EQ", CmpIPredicate::EQ)
            .Case("NE", CmpIPredicate::NE)
            .Case("LT", CmpIPredicate::SLT)
            .Case("LE", CmpIPredicate::SLE)
            .Case("GT", CmpIPredicate::SGT)
            .Case("GE", CmpIPredicate::SGE)
            .Default(CmpIPredicate::NumPredicates);

    if (compare_predicate == CmpIPredicate::NumPredicates)
      return matchFailure();

    rewriter.replaceOpWithNewOp<CmpIOp>(op, compare_predicate, lhs, rhs);
    return matchSuccess();
  }
};

struct CompareFConvert : public RewritePattern {
  explicit CompareFConvert(MLIRContext *context)
      : RewritePattern("xla_hlo.compare", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto compare_op = cast<CompareOp>(op);

    auto lhs = compare_op.lhs();
    auto rhs = compare_op.rhs();
    auto lhs_type = lhs->getType().cast<TensorType>();
    auto rhs_type = rhs->getType().cast<TensorType>();

    // Broadcasting not supported by this rewrite.
    if (lhs_type.getShape() != rhs_type.getShape()) return matchFailure();

    if (!lhs_type.getElementType().isa<FloatType>() ||
        !rhs_type.getElementType().isa<FloatType>())
      return matchFailure();

    auto comparison_direction = compare_op.comparison_direction();
    CmpFPredicate compare_predicate =
        llvm::StringSwitch<CmpFPredicate>(comparison_direction)
            .Case("EQ", CmpFPredicate::OEQ)
            .Case("NE", CmpFPredicate::UNE)
            .Case("LT", CmpFPredicate::OLT)
            .Case("LE", CmpFPredicate::OLE)
            .Case("GT", CmpFPredicate::OGT)
            .Case("GE", CmpFPredicate::OGE)
            .Default(CmpFPredicate::NumPredicates);

    if (compare_predicate == CmpFPredicate::NumPredicates)
      return matchFailure();

    rewriter.replaceOpWithNewOp<CmpFOp>(op, compare_predicate, lhs, rhs);
    return matchSuccess();
  }
};

}  // end anonymous namespace
}  // end namespace XLA
}  // end namespace mlir

namespace {
struct LegalizeToStandard : public FunctionPass<LegalizeToStandard> {
  /// Perform the lowering to Standard dialect.
  void runOnFunction() override;
};
}  // end anonymous namespace

std::unique_ptr<mlir::FunctionPassBase> mlir::XLA::createLegalizeToStdPass() {
  return std::make_unique<LegalizeToStandard>();
}

/// Perform the lowering to standard dialect.
void LegalizeToStandard::runOnFunction() {
  OwningRewritePatternList patterns;
  auto func = getFunction();

  mlir::XLA::populateWithGenerated(func.getContext(), &patterns);
  patterns.insert<mlir::XLA::CompareFConvert, mlir::XLA::CompareIConvert>(
      &getContext());
  applyPatternsGreedily(func, patterns);
}

static PassRegistration<LegalizeToStandard> legalize_pass(
    "xla-legalize-to-std", "Legalize from XLA dialect to standard dialect");
