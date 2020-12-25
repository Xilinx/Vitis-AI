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

// This transformation pass prepares for legalization to the TFLite dialect by
// converting Tensorlist operations in TensorFlow dialect into operations that
// can be legalized to TensorFlow Lite dialect with simple replacements.  The
// newly created operations are in the TensorFlow dialect if the operation can
// be represented using a TensorFlow op.  Otherwise, TensorFlow Lite dialect op
// is used.

#include <climits>
#include <cstdint>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/LoopAnalysis.h"  // TF:local_config_mlir
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Block.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/OperationSupport.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/TypeUtilities.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassRegistry.h"  // TF:local_config_mlir
#include "mlir/Support/Functional.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

#define DEBUG_TYPE "tf-tfl-legalization"

//===----------------------------------------------------------------------===//
// The actual LowerStaticTensorList Pass.
//
namespace mlir {
namespace {

class TensorListPatternRewriter : public PatternRewriter {
 public:
  explicit TensorListPatternRewriter(FuncOp fn)
      : PatternRewriter(fn.getContext()) {}

  Operation *createOperation(const OperationState &state) override {
    return OpBuilder::createOperation(state);
  }
};

/// Lower TensorList ops in functions for subsequent legalization.
// TODO(haoliang): Use DialectConversion infra to simplify the rewriting
// process.
struct LowerStaticTensorListPass
    : public ModulePass<LowerStaticTensorListPass> {
  void runOnModule() override;

  // Apply type and op changes within a function.
  LogicalResult RewriteFunction(FuncOp func,
                                TensorListPatternRewriter *rewriter);

  // Changes the function type of `cond_func` and `body_func`, and the result
  // type of the `WhileOp`.
  LogicalResult UpdateWhileFunctionType(TF::WhileOp op);
};

Value *CreateI32SplatConst(Operation *op, PatternRewriter *rewriter,
                           ArrayRef<int64_t> shape, int32_t val) {
  auto type = rewriter->getTensorType(shape, rewriter->getIntegerType(32));
  auto attr = DenseElementsAttr::get(type, rewriter->getI32IntegerAttr(val));
  return rewriter->create<ConstantOp>(op->getLoc(), type, attr);
}

Value *CreateI32SplatTensor(Operation *op, PatternRewriter *rewriter,
                            Value *shape_tensor, int32_t val) {
  auto scalar_val = CreateI32SplatConst(op, rewriter, {}, val);
  return rewriter->create<TF::FillOp>(
      op->getLoc(), rewriter->getTensorType({-1}, rewriter->getIntegerType(32)),
      shape_tensor, scalar_val);
}

struct ConvertTFTensorListSetItem
    : public OpRewritePattern<TF::TensorListSetItemOp> {
  explicit ConvertTFTensorListSetItem(MLIRContext *context)
      : OpRewritePattern<TF::TensorListSetItemOp>(context, 1) {}
  // This function rewrites the original op into a series of slice and concat op
  // to produce the same result. It first slices the first `$index` rows. Then
  // expands the dimension of the `$item`, followed by another slice of the
  // remaining rows starting from `$index` + 1. Lastly it concatenates the
  // three parts together.
  // On a high level, it's doing something like:
  // def : Pat<(TF_TensorListSetItemOp $input, $index, $item),
  //      (Concat
  //        concat_dim = 0,
  //        (Slice $input, [0, 0, ...], (Concat (ExpandDims $index, expand_dim =
  //        0), [-1, -1, ...])), (ExpandDims $item, expand_dim = 0), (Slice
  //        $input, [$index + 1, 0, 0, ...], [-1, -1, ...]))>;
  PatternMatchResult matchAndRewrite(TF::TensorListSetItemOp op,
                                     PatternRewriter &rewriter) const override {
    auto input = op.input_handle();
    auto shape_dtype = rewriter.getIntegerType(32);
    auto input_rank = rewriter.create<TF::RankOp>(
        op.getLoc(), rewriter.getTensorType({}, shape_dtype), input);
    auto item = op.item();
    auto item_rank = rewriter.create<TF::RankOp>(
        op.getLoc(), rewriter.getTensorType({}, shape_dtype), item);

    // Prepare the start position for the first slice op, which is [0, 0, ..,
    // 0].
    auto scalar_zero = CreateI32SplatConst(op, &rewriter, {}, 0);
    auto position_shape = rewriter.create<TF::ExpandDimsOp>(
        op.getLoc(), rewriter.getTensorType({1}, shape_dtype), input_rank,
        scalar_zero);
    // Fill all 0s into the first position tensor.
    auto first_start_position =
        CreateI32SplatTensor(op, &rewriter, position_shape, 0);

    // Prepare the start position for the second slice op, which is
    // [index + 1, 0, 0 .. 0].
    // Calculate the first dimension, which is index + 1.
    auto index = op.index();
    auto vector_type = rewriter.getTensorType({1}, shape_dtype);
    auto begin = rewriter.create<TF::AddOp>(
        op.getLoc(), rewriter.getTensorType(shape_dtype), index,
        CreateI32SplatConst(op, &rewriter, {1}, 1));

    // Followed by the first dimension `begin`, are `item_rank` of 0s.
    auto item_position_shape = rewriter.create<TF::ExpandDimsOp>(
        op.getLoc(), rewriter.getTensorType({1}, shape_dtype), item_rank,
        scalar_zero);
    auto partial_second_start_position =
        CreateI32SplatTensor(op, &rewriter, item_position_shape, 0);
    auto position_type = first_start_position->getType();
    // Concatenate `begin` with the remaining 0s.
    auto second_start_position = rewriter.create<TF::ConcatOp>(
        op.getLoc(), position_type, scalar_zero,
        ArrayRef<Value *>({begin, partial_second_start_position}),
        rewriter.getI64IntegerAttr(2));

    // Create the size parameter for the first slice op, which is [index, -1,
    // -1, .., -1].
    auto size1_leading_dim = rewriter.create<TF::ExpandDimsOp>(
        op.getLoc(), vector_type, index, scalar_zero);
    auto partial_size1 =
        CreateI32SplatTensor(op, &rewriter, item_position_shape, -1);
    auto size1 = rewriter.create<TF::ConcatOp>(
        op.getLoc(), position_type, scalar_zero,
        ArrayRef<Value *>({size1_leading_dim, partial_size1}),
        rewriter.getI64IntegerAttr(2));

    // Create the size parameter for the second slice, which is [-1, -1, ..,
    // -1].
    auto size2 = CreateI32SplatTensor(op, &rewriter, position_shape, -1);

    // Create two slice ops.
    auto element_type = input->getType().cast<TensorType>().getElementType();
    auto unranked_tensor = rewriter.getTensorType(element_type);
    auto slice1 = rewriter.create<TF::SliceOp>(
        op.getLoc(), unranked_tensor, input, first_start_position, size1);
    auto slice2 = rewriter.create<TF::SliceOp>(
        op.getLoc(), unranked_tensor, input, second_start_position, size2);

    // Expand the dimension of item so that it will have the same rank with
    // input.
    auto expanded_item = rewriter.create<TF::ExpandDimsOp>(
        op.getLoc(), unranked_tensor, item, scalar_zero);

    // Concatenate three parts together to generate the final result.
    rewriter.replaceOpWithNewOp<TF::ConcatOp>(
        op, input->getType(), scalar_zero,
        ArrayRef<Value *>({slice1, expanded_item, slice2}),
        rewriter.getI64IntegerAttr(3));

    return matchSuccess();
  }
};

// Rewrites op of the template type initializing a TensorList with a list of ops
// to generate an equivalent raw tensor. Derived classes are required to
// override GetNumElements method.
template <typename OpT>
struct ConvertTFTensorListInitOp : public OpRewritePattern<OpT> {
  explicit ConvertTFTensorListInitOp(MLIRContext *context)
      : OpRewritePattern<OpT>(context, 1) {}

  // Create and return a 1-d tensor with exactly one element equal to the number
  // of list elements to initialize the output tensor list with.
  virtual Value *GetNumElements(OpT op, PatternRewriter *rewriter) const = 0;

  // Rewrites the original op into `tf.fill`. The result tensor shape is
  // [num_element, element_shape]. All the values in the result tensor will be
  // initialized to 0.
  PatternMatchResult matchAndRewrite(OpT op,
                                     PatternRewriter &rewriter) const override {
    auto element_shape = op.element_shape();
    auto shape_dtype = getElementTypeOrSelf(element_shape->getType());
    Type element_dtype = op.element_dtype();

    int64_t result_rank = -1;  // -1 means unknown result rank.
    Type result_type = rewriter.getTensorType(element_dtype);
    if (auto element_type =
            op.element_type().template dyn_cast<RankedTensorType>()) {
      result_rank = element_type.getRank() + 1;
      // If element type is ranked, then result type will have unknown leading
      // dimension and element shape for the following dimensions.
      //
      // Note: leading dim is not inferred here even when it is a constant.
      SmallVector<int64_t, 4> result_shape = {-1};
      ArrayRef<int64_t> shape = element_type.getShape();
      result_shape.append(shape.begin(), shape.end());
      result_type = rewriter.getTensorType(result_shape, element_dtype);
    }

    // Create a 1-D RankedTensorType for result's shape. Number of elements in
    // it is equal to the rank of the result, if known. Otherwise, the number of
    // elements are unknown and represented with -1. In both cases, we can
    // specify dimension using rank of the result.
    Type shape_type = rewriter.getTensorType({result_rank}, shape_dtype);

    // Add number of elements as the prefix to the element shape to get shape of
    // the output tensor.
    auto leading_dim = GetNumElements(op, &rewriter);
    auto scalar_zero = CreateI32SplatConst(op, &rewriter, {}, 0);
    auto list_shape = rewriter.create<TF::ConcatOp>(
        op.getLoc(), shape_type, scalar_zero,
        ArrayRef<Value *>({leading_dim, element_shape}),
        rewriter.getI64IntegerAttr(2));

    // Create a zero-initialized constant tensor that has the same type
    // as specified by element_dtype.
    auto zero_type = rewriter.getTensorType({}, element_dtype);
    auto zero_attr = rewriter.getZeroAttr(zero_type);
    auto zero = rewriter.create<ConstantOp>(op.getLoc(), zero_type, zero_attr);

    rewriter.replaceOpWithNewOp<TF::FillOp>(op, result_type, list_shape, zero);
    return Pattern::matchSuccess();
  }
};

struct ConvertTFTensorListReserve
    : public ConvertTFTensorListInitOp<TF::TensorListReserveOp> {
  explicit ConvertTFTensorListReserve(MLIRContext *context)
      : ConvertTFTensorListInitOp(context) {}

  Value *GetNumElements(TF::TensorListReserveOp op,
                        PatternRewriter *rewriter) const override {
    auto scalar_zero = CreateI32SplatConst(op, rewriter, {}, 0);
    auto shape_dtype = getElementTypeOrSelf(op.element_shape()->getType());
    return rewriter->create<TF::ExpandDimsOp>(
        op.getLoc(), rewriter->getTensorType({1}, shape_dtype),
        op.num_elements(), scalar_zero);
  }
};

// TODO(hinsu): Replace with declarative patterns once the RewriterGen infra
// supports patterns involving variadic operand ops.
//
// Note that we ignore the second operand `max_num_elements` as we don't have
// any restrictions on the number of elements we can support. So this may
// have a different behavior compared to TensorFlow in case of errors.
struct ConvertTFEmptyTensorList
    : public ConvertTFTensorListInitOp<TF::EmptyTensorListOp> {
  explicit ConvertTFEmptyTensorList(MLIRContext *context)
      : ConvertTFTensorListInitOp(context) {}

  Value *GetNumElements(TF::EmptyTensorListOp op,
                        PatternRewriter *rewriter) const override {
    return CreateI32SplatConst(op, rewriter, {1}, 0);
  }
};

struct ConvertTFTensorListPushBack : public RewritePattern {
  explicit ConvertTFTensorListPushBack(MLIRContext *context)
      : RewritePattern(TF::TensorListPushBackOp::getOperationName(), 1,
                       context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    TF::TensorListPushBackOp push_back_op = cast<TF::TensorListPushBackOp>(op);
    Value *item = push_back_op.tensor();
    Type dtype = getElementTypeOrSelf(*item);

    // Returns a new type by prepending the specified dimension to the shape of
    // the given type if it is a ranked type.
    auto with_leading_dim = [&](int64_t dim, Type type) -> Type {
      if (RankedTensorType ty = type.dyn_cast<RankedTensorType>()) {
        llvm::SmallVector<int64_t, 4> shape = {dim};
        shape.append(ty.getShape().begin(), ty.getShape().end());
        return rewriter.getTensorType(shape, dtype);
      }

      return rewriter.getTensorType(dtype);
    };

    // Expand the shape of the item so that it will have rank same as the input
    // tensor and it is compatible for the Concat Op.
    Type expanded_item_type = with_leading_dim(1, item->getType());
    auto scalar_zero = CreateI32SplatConst(op, &rewriter, {}, 0);
    auto expanded_item = rewriter.create<TF::ExpandDimsOp>(
        op->getLoc(), expanded_item_type, item, scalar_zero);

    // If the variant type in the output handle has item shape available, use it
    // to derive the output shape by setting unknown leading dimension.
    // Otherwise, result type will be of unranked type.
    Type handle_type = push_back_op.output_handle()->getType();
    TF::VariantType handle_dtype =
        getElementTypeOrSelf(handle_type).cast<TF::VariantType>();
    Type result_type = rewriter.getTensorType(dtype);
    if (!handle_dtype.getSubtypes().empty()) {
      result_type = with_leading_dim(-1, handle_dtype.getSubtypes()[0]);
    }

    // Concatenate tensor stored in the input handle with the expanded item to
    // get a tensor equivalent to the TensorList generated by this op.
    rewriter.replaceOpWithNewOp<TF::ConcatOp>(
        op, result_type, scalar_zero,
        ArrayRef<Value *>({push_back_op.input_handle(), expanded_item}),
        rewriter.getI64IntegerAttr(2));
    return matchSuccess();
  }
};

}  // namespace

namespace TFL {
namespace {
#include "tensorflow/compiler/mlir/lite/transforms/generated_lower_static_tensor_list.inc"
}  // namespace
}  // namespace TFL

LogicalResult LowerStaticTensorListPass::UpdateWhileFunctionType(
    TF::WhileOp op) {
  SmallVector<Type, 8> unranked_argument_types;
  for (const auto &operand : op.getOperands()) {
    unranked_argument_types.push_back(
        UnrankedTensorType::get(getElementTypeOrSelf(operand->getType())));
  }

  auto *context = &getContext();
  auto module = getModule();
  FuncOp cond_func = module.lookupSymbol<FuncOp>(op.cond());
  FuncOp body_func = module.lookupSymbol<FuncOp>(op.body());

  if (cond_func) {
    // Change `cond_func`'s argument types to `unranked_argument_types`.
    cond_func.setType(FunctionType::get(
        unranked_argument_types, cond_func.getType().getResults(), context));
    // Change the argument type for the first block.
    Block &cond_first_bb = cond_func.front();
    for (int i = 0; i < cond_first_bb.getNumArguments(); ++i) {
      cond_first_bb.getArgument(i)->setType(unranked_argument_types[i]);
    }
  }

  if (body_func) {
    SmallVector<Type, 8> updated_result_types;
    for (int i = 0; i < body_func.getType().getNumResults(); ++i) {
      auto result_type = body_func.getType().getResult(i);
      if (getElementTypeOrSelf(result_type).isa<TF::VariantType>()) {
        // For variant type, use the corresponding unranked type.
        updated_result_types.push_back(unranked_argument_types[i]);
      } else {
        updated_result_types.push_back(result_type);
      }
    }
    // Change `body_func`'s argument type to `unranked_argument_types`. If it
    // return types contain a `DT_VARIANT`, change it to the unranked type
    // derived from the corresponding argument.
    body_func.setType(FunctionType::get(unranked_argument_types,
                                        updated_result_types, context));
    // Change the argument type for the first block.
    Block &body_first_bb = body_func.front();
    for (int i = 0; i < body_first_bb.getNumArguments(); ++i) {
      body_first_bb.getArgument(i)->setType(unranked_argument_types[i]);
    }
  }

  for (int i = 0; i < op.getNumOperands(); ++i) {
    auto operand = op.getOperand(i);
    auto result = op.getResult(i);
    if (getElementTypeOrSelf(result->getType()).isa<TF::VariantType>()) {
      // If we notice the result type is a DT_VARIANT, we change the
      // corresponding result type to unranked tensor type.
      result->setType(
          UnrankedTensorType::get(getElementTypeOrSelf(operand->getType())));
    }
  }
  return success();
}

LogicalResult LowerStaticTensorListPass::RewriteFunction(
    FuncOp func, TensorListPatternRewriter *rewriter) {
  auto *context = &getContext();

  for (Block &block : func) {
    // Buffer the op pointers inside the current block into a vector, since
    // the block iterator might be invalidated if we rewrite ops during looping.
    std::vector<Operation *> ops_in_block;
    for (Operation &op : block) {
      ops_in_block.push_back(&op);
    }

    for (Operation *op : ops_in_block) {
      if (auto tf_op = llvm::dyn_cast<TF::TensorListFromTensorOp>(op)) {
        auto c = TFL::ConvertTFTensorListFromTensor(context);
        rewriter->setInsertionPoint(op);
        c.matchAndRewrite(op, *rewriter);
      } else if (auto tf_op = llvm::dyn_cast<TF::TensorListReserveOp>(op)) {
        if (!(tf_op.element_dtype().isF16() || tf_op.element_dtype().isF32() ||
              tf_op.element_dtype().isF64() ||
              tf_op.element_dtype().isInteger(8) ||
              tf_op.element_dtype().isInteger(16) ||
              tf_op.element_dtype().isInteger(32) ||
              tf_op.element_dtype().isInteger(64))) {
          return tf_op.emitError(
              "requires element_dtype to be 8-bit/16-bit/32-bit/64-bit integer "
              "or 16-bit/32-bit/64-bit "
              "float type during TF Lite transformation pass");
        }
        auto c = ConvertTFTensorListReserve(context);
        rewriter->setInsertionPoint(op);
        c.matchAndRewrite(tf_op, *rewriter);
      } else if (auto tf_op = llvm::dyn_cast<TF::EmptyTensorListOp>(op)) {
        auto c = ConvertTFEmptyTensorList(context);
        rewriter->setInsertionPoint(op);
        c.matchAndRewrite(tf_op, *rewriter);
      } else if (auto tf_op = llvm::dyn_cast<TF::TensorListGetItemOp>(op)) {
        auto c = TFL::ConvertTFTensorListGetItem(context);
        rewriter->setInsertionPoint(op);
        c.matchAndRewrite(op, *rewriter);
      } else if (auto tf_op = llvm::dyn_cast<TF::TensorListSetItemOp>(op)) {
        auto c = ConvertTFTensorListSetItem(context);
        rewriter->setInsertionPoint(op);
        c.matchAndRewrite(tf_op, *rewriter);
      } else if (auto tf_op = llvm::dyn_cast<TF::TensorListStackOp>(op)) {
        auto c = TFL::ConvertTFTensorListStack(context);
        rewriter->setInsertionPoint(op);
        c.matchAndRewrite(op, *rewriter);
      } else if (auto tf_op = llvm::dyn_cast<TF::TensorListPushBackOp>(op)) {
        auto c = ConvertTFTensorListPushBack(context);
        rewriter->setInsertionPoint(op);
        c.matchAndRewrite(op, *rewriter);
      } else if (auto tf_op = llvm::dyn_cast<TF::WhileOp>(op)) {
        if (op->getAttr("T")) op->removeAttr(Identifier::get("T", context));
        UpdateWhileFunctionType(tf_op);
      } else if (auto tf_op = llvm::dyn_cast<TF::IdentityOp>(op)) {
        if (op->getAttr("T")) op->removeAttr(Identifier::get("T", context));
        tf_op.getResult()->setType(tf_op.getOperand()->getType());
      }
    }
  }
  return success();
}

void LowerStaticTensorListPass::runOnModule() {
  // TODO(haoliang): currently we process the `main` function first, and the
  // remaining functions may be processed in arbitrary order. However, this will
  // have a potential issue when one function taking a `DT_VARIANT` is processed
  // before the function that produces the `DT_VARIANT`. We need to carefully
  // order the functions to be processed.
  std::vector<FuncOp> funcs_in_module;
  for (auto func : getModule().getOps<FuncOp>()) {
    // Always place the main function to be the first in the list.
    if (func.getName() == "main") {
      funcs_in_module.insert(funcs_in_module.begin(), func);
    } else {
      funcs_in_module.push_back(func);
    }
  }
  for (auto func : funcs_in_module) {
    TensorListPatternRewriter rewriter(func);
    if (failed(RewriteFunction(func, &rewriter))) {
      signalPassFailure();
      return;
    }
  }
}

/// Creates an instance of the TensorFlow Lite dialect LowerStaticTensorList
/// pass.
std::unique_ptr<ModulePassBase> TFL::CreateLowerStaticTensorListPass() {
  return std::make_unique<LowerStaticTensorListPass>();
}

static PassRegistration<LowerStaticTensorListPass> pass(
    "tfl-lower-static-tensor-list",
    "Lower TensorList ops within TensorFlow Lite dialect");

}  // namespace mlir
