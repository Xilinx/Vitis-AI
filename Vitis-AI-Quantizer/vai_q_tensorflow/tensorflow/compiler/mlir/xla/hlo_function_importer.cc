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

#include "tensorflow/compiler/mlir/xla/hlo_function_importer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Identifier.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/xla/ir/xla_ops.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

using llvm::APInt;
using llvm::makeArrayRef;
using mlir::DenseElementsAttr;
using mlir::DenseIntElementsAttr;
using mlir::FuncOp;
using mlir::NamedAttribute;
using mlir::Operation;
using mlir::ShapedType;
using mlir::Type;
using mlir::Value;

namespace xla {

namespace {
// Note: This sanitization function causes an irreversible many-to-one mapping
// and any solution to mitigate this would cause issues with the reverse
// direction. Longterm solution is to add a function attribute to maintain the
// original HLO naming.
string SanitizeFunctionName(llvm::StringRef name) {
  string output = name;
  llvm::for_each(output, [](char& x) { x = x == '-' ? '_' : x; });
  return output;
}

StatusOr<DenseElementsAttr> CreateDenseAttrFromLiteral(ShapedType type,
                                                       const Literal& literal) {
#define DENSE_ELEMENT_ATTR_BUILDER(xla_type, cpp_type)                 \
  case xla_type: {                                                     \
    auto data_span = literal.data<cpp_type>();                         \
    return DenseElementsAttr::get(                                     \
        type, llvm::makeArrayRef(data_span.data(), data_span.size())); \
  }

  switch (literal.shape().element_type()) {
    DENSE_ELEMENT_ATTR_BUILDER(PrimitiveType::PRED, bool)
    DENSE_ELEMENT_ATTR_BUILDER(PrimitiveType::F32, float)
    DENSE_ELEMENT_ATTR_BUILDER(PrimitiveType::F64, double)
    DENSE_ELEMENT_ATTR_BUILDER(PrimitiveType::S8, int8)
    DENSE_ELEMENT_ATTR_BUILDER(PrimitiveType::S16, int16)
    DENSE_ELEMENT_ATTR_BUILDER(PrimitiveType::S32, int32)
    DENSE_ELEMENT_ATTR_BUILDER(PrimitiveType::S64, int64)
    default:
      return tensorflow::errors::Internal(
          absl::StrCat("Unsupported type: ",
                       PrimitiveType_Name(literal.shape().element_type())));
  }
#undef DENSE_ELEMENT_ATTR_BUILDER
}
}  // namespace

StatusOr<mlir::FuncOp> HloFunctionImporter::ImportFunction(
    mlir::ModuleOp module, mlir::Builder* builder,
    std::unordered_map<HloComputation*, FuncOp>* function_map,
    HloComputation* computation) {
  HloFunctionImporter importer(module, builder, function_map);
  return importer.ImportFunction(computation);
}

StatusOr<mlir::FuncOp> HloFunctionImporter::ImportFunction(
    HloComputation* computation) {
  auto& imported = (*function_map_)[computation];
  if (imported) return imported;

  llvm::SmallVector<Type, 4> args, rets;
  TF_RETURN_IF_ERROR(
      GetMlirTypes(computation->parameter_instructions(), &args));
  TF_RETURN_IF_ERROR(GetMlirTypes({computation->root_instruction()}, &rets));

  auto func_type = mlir::FunctionType::get(args, rets, context_);

  string computation_name =
      computation->parent()->entry_computation() == computation
          ? "main"
          : SanitizeFunctionName(computation->name());

  // Construct the MLIR function and map arguments.
  llvm::ArrayRef<mlir::NamedAttribute> attrs;
  auto function = mlir::FuncOp::create(mlir::UnknownLoc::get(context_),
                                       computation_name, func_type, attrs);
  module_.push_back(function);

  // Add to the map right away for function calls.
  imported = function;

  function.addEntryBlock();

  // Setup the input parameters.
  const int num_parameters = computation->num_parameters();
  for (int i = 0; i < num_parameters; i++) {
    auto hlo_parameter = computation->parameter_instruction(i);
    instruction_value_map_[hlo_parameter] = function.getArgument(i);
  }

  mlir::OpBuilder func_builder(function.getBody());
  for (auto instruction : computation->MakeInstructionPostOrder()) {
    TF_ASSIGN_OR_RETURN(auto new_operation,
                        ImportInstruction(instruction, &func_builder));
    if (new_operation) {
      instruction_value_map_[instruction] = new_operation->getResult(0);
    }
  }

  // Setup the return type (HLO only supports a single return value).
  TF_ASSIGN_OR_RETURN(auto result,
                      GetMlirValue(computation->root_instruction()));
  llvm::SmallVector<Value*, 1> return_values({result});
  // TODO(suderman): Add location tracking details.
  func_builder.create<mlir::ReturnOp>(mlir::UnknownLoc::get(context_),
                                      makeArrayRef(return_values));

  return function;
}

StatusOr<mlir::Operation*> HloFunctionImporter::ImportInstruction(
    HloInstruction* instruction, mlir::OpBuilder* func_builder) {
  TF_ASSIGN_OR_RETURN(auto operands, GetOperands(instruction));
  TF_ASSIGN_OR_RETURN(auto result_type, ConvertType(instruction->shape()));
  llvm::SmallVector<NamedAttribute, 10> attributes = {builder_->getNamedAttr(
      "name", builder_->getStringAttr(instruction->name()))};
  mlir::Location loc = mlir::UnknownLoc::get(context_);

  switch (instruction->opcode()) {
    case HloOpcode::kParameter: {
      return nullptr;
    }
    case HloOpcode::kConstant: {
      auto attr = CreateDenseAttrFromLiteral(
          result_type.cast<mlir::TensorType>(), instruction->literal());
      if (!attr.ok()) return attr.status();
      mlir::Operation* new_operation =
          func_builder->create<mlir::ConstantOp>(loc, attr.ValueOrDie());
      for (auto attr : attributes) {
        new_operation->setAttr(attr.first, attr.second);
      }
      return new_operation;
    }
    case HloOpcode::kIota: {
      return func_builder
          ->create<mlir::XLA::IotaOp>(
              loc, result_type,
              func_builder->getI64IntegerAttr(
                  static_cast<HloIotaInstruction*>(instruction)
                      ->iota_dimension()))
          .getOperation();
    }
#define MakeAndReturn(mlir_op)                                                 \
  {                                                                            \
    mlir::Operation* new_operation = func_builder->create<mlir::XLA::mlir_op>( \
        loc, result_type, operands, attributes);                               \
    return new_operation;                                                      \
  }
    case HloOpcode::kBroadcast: {
      // Note that the HLO broadcast is more powerful than the XLA broadcast op.
      // BroadcastInDim offers a superset of the HLO op's functionality.
      if (!instruction->dimensions().empty()) {
        attributes.push_back(builder_->getNamedAttr(
            "broadcast_dimensions",
            ConvertDimensions(instruction->dimensions())));
      }
      MakeAndReturn(BroadcastInDimOp);
    }
    case HloOpcode::kDot: {
      // TODO(b/129153247) Add support for batch and contracting dimensions.
      TF_RETURN_IF_ERROR(ValidateDotDimensions(instruction));

      // TODO(b/129709049) The HLO text format elides this in the all DEFAULT
      // case and the parser sticks it in. Maybe we should too.
      attributes.push_back(ConvertPrecisionConfig(instruction));
      MakeAndReturn(DotOp);
    }
    case HloOpcode::kCall: {
      TF_ASSIGN_OR_RETURN(FuncOp function,
                          ImportFunction(instruction->to_apply()));
      mlir::Operation* new_operation =
          func_builder->create<mlir::CallOp>(loc, function, operands);
      return new_operation;
    }
    case HloOpcode::kCompare: {
      attributes.push_back(ConvertComparisonDirection(instruction));
      MakeAndReturn(CompareOp);
    }
    case HloOpcode::kGather: {
      const auto& gather_dimensions = instruction->gather_dimension_numbers();
      std::vector<int64_t> offset_dims(gather_dimensions.offset_dims().begin(),
                                       gather_dimensions.offset_dims().end());

      std::vector<int64_t> slice_sizes(
          instruction->gather_slice_sizes().begin(),
          instruction->gather_slice_sizes().end());

      std::vector<int64_t> collapsed_slice_dims(
          gather_dimensions.collapsed_slice_dims().begin(),
          gather_dimensions.collapsed_slice_dims().end());

      std::vector<int64_t> start_index_map(
          gather_dimensions.start_index_map().begin(),
          gather_dimensions.start_index_map().end());

      // TODO(b/132057942): Change to explicitly passing an integer instead of
      // call getI64IntegerAttr here.
      return func_builder
          ->create<mlir::XLA::GatherOp>(
              loc, result_type, operands[0], operands[1],
              func_builder->getI64IntegerAttr(
                  gather_dimensions.index_vector_dim()),
              Convert(offset_dims), Convert(slice_sizes),
              Convert(collapsed_slice_dims), Convert(start_index_map))
          .getOperation();
    }
    case HloOpcode::kDynamicUpdateSlice: {
      return func_builder
          ->create<mlir::XLA::DynamicUpdateSliceOp>(
              loc, result_type, operands[0], operands[1],
              llvm::ArrayRef<Value*>(operands.begin() + 2, operands.end()))
          .getOperation();
    }
    case HloOpcode::kPad: {
      const auto& padding_config = instruction->padding_config();
      llvm::SmallVector<int64_t, 4> edge_padding_low;
      llvm::SmallVector<int64_t, 4> edge_padding_high;
      llvm::SmallVector<int64_t, 4> interior_padding;
      edge_padding_low.reserve(padding_config.dimensions_size());
      edge_padding_high.reserve(padding_config.dimensions_size());
      interior_padding.reserve(padding_config.dimensions_size());

      for (const auto& dimension : padding_config.dimensions()) {
        edge_padding_low.push_back(dimension.edge_padding_low());
        edge_padding_high.push_back(dimension.edge_padding_high());
        interior_padding.push_back(dimension.interior_padding());
      }

      return func_builder
          ->create<mlir::XLA::PadOp>(loc, result_type, operands[0], operands[1],
                                     Convert(edge_padding_low),
                                     Convert(edge_padding_high),
                                     Convert(interior_padding))
          .getOperation();
    }
    case HloOpcode::kSlice: {
      return func_builder
          ->create<mlir::XLA::SliceOp>(
              loc, result_type, operands[0],
              ConvertDimensions(instruction->slice_starts()),
              ConvertDimensions(instruction->slice_limits()))
          .getOperation();
    }
    case HloOpcode::kConcatenate: {
      // TODO(b/132057942): Support taking an uint64_t instead of an IntegerAttr
      // for concatenate dimension.
      return func_builder
          ->create<mlir::XLA::ConcatenateOp>(
              loc, result_type, operands,
              builder_->getI64IntegerAttr(instruction->concatenate_dimension()))
          .getOperation();
    }
    case HloOpcode::kReduce: {
      TF_ASSIGN_OR_RETURN(auto reduction,
                          ImportFunction(instruction->to_apply()));
      // TODO(b/132057942): Make more convenient constructors, e.g. pass
      // mlir function pointer instead of a function attr.
      return func_builder
          ->create<mlir::XLA::ReduceOp>(
              loc, result_type, operands,
              func_builder->getSymbolRefAttr(reduction),
              ConvertDimensions(instruction->dimensions()))
          .getOperation();
    }
    case HloOpcode::kReverse: {
      return func_builder
          ->create<mlir::XLA::ReverseOp>(
              loc, result_type, operands[0],
              ConvertDimensions(instruction->dimensions()))
          .getOperation();
    }
    case HloOpcode::kWhile: {
      TF_ASSIGN_OR_RETURN(auto body, ImportFunction(instruction->while_body()));
      TF_ASSIGN_OR_RETURN(auto cond,
                          ImportFunction(instruction->while_condition()));

      llvm::SmallVector<Type, 4> types;
      types.reserve(operands.size());
      for (auto operand : operands) {
        types.push_back(operand->getType());
      }

      auto cond_attr = func_builder->getSymbolRefAttr(cond);
      auto body_attr = func_builder->getSymbolRefAttr(body);

      Operation* op = func_builder->create<mlir::XLA::WhileOp>(
          loc, types, operands, cond_attr, body_attr);
      return op;
    }
    case HloOpcode::kGetTupleElement: {
      attributes.push_back(builder_->getNamedAttr(
          "index", builder_->getIntegerAttr(builder_->getIntegerType(32),
                                            instruction->tuple_index())));
      MakeAndReturn(GetTupleElementOp);
    };
    case HloOpcode::kTranspose: {
      attributes.push_back(builder_->getNamedAttr(
          "permutation", ConvertDimensions(instruction->dimensions())));
      MakeAndReturn(TransposeOp);
    }
#define NoAttributeCase(hlo_op_code, mlir_op) \
  case HloOpcode::hlo_op_code: {              \
    MakeAndReturn(mlir_op);                   \
  }

      // broadcast dimensions are never added here because they don't exist as
      // part of the HLO instruction. They are only a convenience in the XLA
      // builder API.
      NoAttributeCase(kAdd, AddOp);
      NoAttributeCase(kAnd, AndOp);
      NoAttributeCase(kConvert, ConvertOp);
      NoAttributeCase(kDivide, DivOp);
      NoAttributeCase(kExp, ExpOp);
      NoAttributeCase(kMaximum, MaxOp);
      NoAttributeCase(kMinimum, MinOp);
      NoAttributeCase(kMultiply, MulOp);
      // The dimensions attribute is not present on the HLO Reshape instruction.
      // If dimensions are non-default, the XLA builder implementes it as a
      // separate transpose.
      NoAttributeCase(kReshape, ReshapeOp);
      NoAttributeCase(kSelect, SelectOp);
      NoAttributeCase(kSubtract, SubOp);
      NoAttributeCase(kTanh, TanhOp);
      NoAttributeCase(kTuple, TupleOp);
      // TODO(b/129422361) Copy needs special handling because it is not defined
      // in tensorflow/compiler/xla/client/xla_builder.h.
      // See operation semantics in
      // g3doc/platforms/xla/g3doc/internal/hlo_semantics#copy
      NoAttributeCase(kCopy, CopyOp);
      // TODO(b/129422361) Ops below need additional work to handle attributes.
      NoAttributeCase(kConvolution, ConvOp);
#undef NoAttributeCase
#undef MakeAndReturn
    case HloOpcode::kAddDependency:
      // Arbitrary op code that I suspect we will not implement for quite a
      // while and allows testing handling of unknown ops. Selected because it
      // is not mentioned in xla client anywhere or in the hlo of our sample
      // models.
    default: {
      mlir::OperationState result(loc, "xla_hlo.unknown");
      result.addOperands(operands);
      result.addTypes(result_type);
      for (auto attr : attributes) {
        result.attributes.push_back(attr);
      }

      return func_builder->createOperation(result);
    }
  }
}

StatusOr<llvm::SmallVector<mlir::Value*, 4>> HloFunctionImporter::GetOperands(
    HloInstruction* instruction) {
  llvm::SmallVector<mlir::Value*, 4> operands;
  for (const auto& operand : instruction->operands()) {
    auto input_it = instruction_value_map_.find(operand);
    if (input_it == instruction_value_map_.end()) {
      return tensorflow::errors::Internal(
          absl::StrCat("Could not find input value: ", operand->name(),
                       " for instruction ", instruction->name()));
    }
    operands.push_back(input_it->second);
  }
  return operands;
}

// TODO(suderman): Move to a general library when needed in other places.
StatusOr<mlir::RankedTensorType> HloFunctionImporter::ConvertTensorType(
    const Shape& shape) {
  auto type = shape.element_type();

  llvm::SmallVector<int64_t, 4> array;
  array.reserve(shape.dimensions_size());
  for (auto val : shape.dimensions()) {
    array.push_back(val);
  }

  switch (type) {
    case PrimitiveType::PRED:
      return builder_->getTensorType(array, builder_->getI1Type());
    case PrimitiveType::F16:
      return builder_->getTensorType(array, builder_->getF16Type());
    case PrimitiveType::F32:
      return builder_->getTensorType(array, builder_->getF32Type());
    case PrimitiveType::F64:
      return builder_->getTensorType(array, builder_->getF64Type());
    case PrimitiveType::S8:
      return builder_->getTensorType(array, builder_->getIntegerType(8));
    case PrimitiveType::S16:
      return builder_->getTensorType(array, builder_->getIntegerType(16));
    case PrimitiveType::S32:
      return builder_->getTensorType(array, builder_->getIntegerType(32));
    case PrimitiveType::S64:
      return builder_->getTensorType(array, builder_->getIntegerType(64));
    default:
      return tensorflow::errors::Internal(
          absl::StrCat("Unsupported type: ", PrimitiveType_Name(type)));
  }
}

StatusOr<mlir::Type> HloFunctionImporter::ConvertType(const Shape& shape) {
  if (shape.IsTuple()) {
    mlir::Type mlir_type;
    llvm::SmallVector<mlir::Type, 4> contents;
    contents.reserve(shape.tuple_shapes_size());
    for (const auto& subtype : shape.tuple_shapes()) {
      TF_ASSIGN_OR_RETURN(auto mlir_subtype, ConvertType(subtype));
      contents.push_back(mlir_subtype);
    }

    return builder_->getTupleType(contents);
  }

  return ConvertTensorType(shape);
}

tensorflow::Status HloFunctionImporter::GetMlirTypes(
    const std::vector<HloInstruction*>& instructions,
    llvm::SmallVectorImpl<mlir::Type>* types) {
  for (auto instruction : instructions) {
    TF_ASSIGN_OR_RETURN(auto ret_type, ConvertType(instruction->shape()));
    types->push_back(ret_type);
  }
  return tensorflow::Status::OK();
}

StatusOr<Value*> HloFunctionImporter::GetMlirValue(
    HloInstruction* instruction) {
  auto lookup = instruction_value_map_.find(instruction);
  if (lookup != instruction_value_map_.end()) {
    return lookup->second;
  }

  return tensorflow::errors::Internal(absl::StrCat(
      "Unable to find value for input: ", instruction->ToString()));
}

mlir::NamedAttribute HloFunctionImporter::ConvertPrecisionConfig(
    HloInstruction* instruction) {
  llvm::SmallVector<mlir::Attribute, 4> operand_precision_attrs;

  for (auto prec : instruction->precision_config().operand_precision()) {
    operand_precision_attrs.push_back(
        builder_->getStringAttr(PrecisionConfig_Precision_Name(prec)));
  }
  return builder_->getNamedAttr(
      "precision_config", builder_->getArrayAttr(operand_precision_attrs));
}

mlir::NamedAttribute HloFunctionImporter::ConvertComparisonDirection(
    HloInstruction* instruction) {
  return builder_->getNamedAttr(
      "comparison_direction",
      builder_->getStringAttr(
          ComparisonDirectionToString(instruction->comparison_direction())));
}

mlir::ElementsAttr HloFunctionImporter::ConvertDimensions(
    llvm::ArrayRef<int64> op_dimensions) {
  llvm::SmallVector<APInt, 8> dimensions;
  dimensions.reserve(op_dimensions.size());
  for (auto value : op_dimensions) dimensions.emplace_back(APInt(64, value));

  return DenseIntElementsAttr::get(
      builder_->getTensorType(dimensions.size(), builder_->getIntegerType(64)),
      dimensions);
}

mlir::ElementsAttr HloFunctionImporter::Convert(
    llvm::ArrayRef<int64_t> op_dimensions) {
  return builder_->getDenseIntElementsAttr(
      builder_->getTensorType(op_dimensions.size(),
                              builder_->getIntegerType(64)),
      op_dimensions);
}

Status HloFunctionImporter::ValidateDotDimensions(HloInstruction* instruction) {
  DotDimensionNumbers expected_dimension_numbers;
  expected_dimension_numbers.add_lhs_contracting_dimensions(
      instruction->operand(0)->shape().dimensions_size() == 1 ? 0 : 1);
  expected_dimension_numbers.add_rhs_contracting_dimensions(0);
  if (!xla::protobuf_util::ProtobufEquals(instruction->dot_dimension_numbers(),
                                          expected_dimension_numbers)) {
    return tensorflow::errors::Internal(
        absl::StrCat("Dot operation has unsupported dimension numbers: ",
                     instruction->dot_dimension_numbers().DebugString()));
  }
  return Status::OK();
}

}  // namespace xla
