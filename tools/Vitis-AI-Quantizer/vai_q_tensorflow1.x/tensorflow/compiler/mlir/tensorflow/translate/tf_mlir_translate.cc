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

#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"

#include "absl/memory/memory.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Identifier.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/Parser.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/import_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"

namespace tensorflow {

using stream_executor::port::Status;
using stream_executor::port::StatusOr;

static StatusOr<mlir::OwningModuleRef> GraphdefToMlirImport(
    absl::string_view input_filename, absl::string_view debug_info_file,
    absl::string_view input_arrays, absl::string_view input_dtypes,
    absl::string_view input_shapes, absl::string_view output_arrays,
    absl::string_view inference_type, absl::string_view min_values,
    absl::string_view max_values, bool prune_unused_nodes,
    bool convert_legacy_fed_inputs, bool graph_as_function,
    mlir::MLIRContext* context) {
  GraphDef graphdef;
  TF_RETURN_IF_ERROR(tensorflow::LoadProtoFromFile(input_filename, &graphdef));

  GraphDebugInfo debug_info;
  if (!debug_info_file.empty()) {
    TF_RETURN_IF_ERROR(LoadProtoFromFile(debug_info_file, &debug_info));
  }

  NodeSpecs specs;
  specs.prune_unused_nodes = prune_unused_nodes;
  specs.convert_legacy_fed_inputs = convert_legacy_fed_inputs;
  specs.graph_as_function = graph_as_function;
  TF_RETURN_IF_ERROR(ParseInputArrayInfo(
      input_arrays, input_dtypes, input_shapes, inference_type, min_values,
      max_values, &specs.inputs));
  TF_RETURN_IF_ERROR(ParseOutputArrayInfo(output_arrays, &specs.output_arrays,
                                          &specs.output_arrays_order));
  return ConvertGraphdefToMlir(graphdef, debug_info, specs, context);
}

mlir::OwningModuleRef GraphdefToMlirTranslateFunction(
    absl::string_view input_filename, absl::string_view debug_info_file,
    absl::string_view input_arrays, absl::string_view input_dtypes,
    absl::string_view input_shapes, absl::string_view output_arrays,
    absl::string_view inference_type, absl::string_view min_values,
    absl::string_view max_values, bool prune_unused_nodes,
    bool convert_legacy_fed_inputs, bool graph_as_function,
    mlir::MLIRContext* context) {
  auto module_or = GraphdefToMlirImport(
      input_filename, debug_info_file, input_arrays, input_dtypes, input_shapes,
      output_arrays, inference_type, min_values, max_values, prune_unused_nodes,
      convert_legacy_fed_inputs, graph_as_function, context);
  if (!module_or.status().ok()) {
    LOG(ERROR) << "Graph import failed: " << module_or.status();
    return nullptr;
  }

  return module_or.ConsumeValueOrDie();
}

mlir::OwningModuleRef GraphdefToSplattedMlirTranslateFunction(
    absl::string_view input_filename, absl::string_view debug_info_file,
    absl::string_view input_arrays, absl::string_view input_dtypes,
    absl::string_view input_shapes, absl::string_view output_arrays,
    absl::string_view inference_type, absl::string_view min_values,
    absl::string_view max_values, bool prune_unused_nodes,
    bool convert_legacy_fed_inputs, bool graph_as_function,
    mlir::MLIRContext* context) {
  auto module_or = GraphdefToMlirImport(
      input_filename, debug_info_file, input_arrays, input_dtypes, input_shapes,
      output_arrays, inference_type, min_values, max_values, prune_unused_nodes,
      convert_legacy_fed_inputs, graph_as_function, context);
  if (!module_or.status().ok()) {
    LOG(ERROR) << "Graph import failed: " << module_or.status();
    return nullptr;
  }
  auto& module = module_or.ValueOrDie();
  std::srand(0);
  for (auto fn : module->getOps<mlir::FuncOp>()) {
    for (auto& bb : fn) {
      for (auto& inst : bb) {
        auto attr_id = mlir::Identifier::get("value", context);
        if (auto attr = inst.getAttrOfType<mlir::ElementsAttr>(attr_id)) {
          mlir::Attribute rand_val;
          mlir::Type element_type = attr.getType().getElementType();

          switch (element_type.getKind()) {
            case mlir::StandardTypes::Integer:
              rand_val = mlir::IntegerAttr::get(element_type, std::rand());
              break;
            case mlir::StandardTypes::F16:
            case mlir::StandardTypes::F32:
            case mlir::StandardTypes::F64:
              rand_val = mlir::FloatAttr::get(element_type,
                                              std::rand() * 1.0 / RAND_MAX);
              break;
            default:
              inst.emitWarning()
                  << "Skipping splat conversion for "
                  << "an unsupported attribute type " << element_type;
              continue;
          }
          auto new_attr =
              mlir::DenseElementsAttr::get(attr.getType(), rand_val);
          inst.setAttr(attr_id, new_attr);
        }
      }
    }
  }
  return module_or.ConsumeValueOrDie();
}

}  // namespace tensorflow
