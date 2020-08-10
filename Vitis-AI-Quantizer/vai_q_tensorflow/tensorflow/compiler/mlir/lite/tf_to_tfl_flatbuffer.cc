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

#include "tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.h"

#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/Parser.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Support/FileUtilities.h"  // TF:local_config_mlir
#include "mlir/Transforms/Passes.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/flatbuffer_translate.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/decode_constant.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace mlir {
/// Create a pass to convert from the TFExecutor to the TF control dialect.
std::unique_ptr<FunctionPassBase> CreateTFExecutorToControlDialectConversion();
}  // namespace mlir

namespace tensorflow {

using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OwningModuleRef;
using stream_executor::port::StatusOr;

StatusOr<OwningModuleRef> LoadFromGraphdefOrMlirSource(
    const std::string &input_filename, bool input_mlir,
    bool use_splatted_constant, const std::vector<std::string> &extra_tf_opdefs,
    absl::string_view debug_info_file, absl::string_view input_arrays,
    absl::string_view input_dtypes, absl::string_view input_shapes,
    absl::string_view output_arrays, absl::string_view inference_type,
    absl::string_view min_values, absl::string_view max_values,
    bool prune_unused_nodes, llvm::SourceMgr *source_mgr,
    MLIRContext *context) {
  if (input_mlir) {
    // Set up the input file.
    std::string error_message;
    auto file = mlir::openInputFile(input_filename, &error_message);
    if (!file) {
      llvm::errs() << error_message << "\n";
      return errors::InvalidArgument("fail to open input file");
    }

    source_mgr->AddNewSourceBuffer(std::move(file), llvm::SMLoc());
    return OwningModuleRef(mlir::parseSourceFile(*source_mgr, context));
  }
  for (const auto &tf_opdefs_string : extra_tf_opdefs) {
    tensorflow::OpDef opdef;
    if (!tensorflow::protobuf::TextFormat::ParseFromString(tf_opdefs_string,
                                                           &opdef)) {
      LOG(ERROR) << "OpDef parsing failed for: " << tf_opdefs_string;
      return errors::InvalidArgument("fail to parse extra OpDef");
    }
    // Register extra opdefs.
    // TODO(b/133770952): Support shape functions.
    tensorflow::OpRegistry::Global()->Register(
        [opdef](tensorflow::OpRegistrationData *op_reg_data) -> Status {
          *op_reg_data = tensorflow::OpRegistrationData(opdef);
          return Status::OK();
        });
  }

  if (use_splatted_constant) {
    return tensorflow::GraphdefToSplattedMlirTranslateFunction(
        input_filename, debug_info_file, input_arrays, input_dtypes,
        input_shapes, output_arrays, inference_type, min_values, max_values,
        prune_unused_nodes, /*convert_legacy_fed_inputs=*/true,
        /*graph_as_function=*/false, context);
  }
  return tensorflow::GraphdefToMlirTranslateFunction(
      input_filename, debug_info_file, input_arrays, input_dtypes, input_shapes,
      output_arrays, inference_type, min_values, max_values, prune_unused_nodes,
      /*convert_legacy_fed_inputs=*/true, /*graph_as_function=*/false, context);
}

Status ConvertTFExecutorToTFLOrFlatbuffer(
    mlir::ModuleOp module, bool export_to_mlir, bool emit_builtin_tflite_ops,
    bool emit_select_tf_ops, bool emit_custom_ops, bool emit_quant_adaptor_ops,
    bool lower_tensor_list_ops, std::string *result,
    mlir::PassManager *pass_manager) {
  mlir::StatusScopedDiagnosticHandler statusHandler(module.getContext(),
                                                    /*propagate=*/true);
  if (failed(pass_manager->run(module))) {
    return statusHandler.ConsumeStatus();
  }

  if (export_to_mlir) {
    llvm::raw_string_ostream os(*result);
    module.print(os);
    return Status::OK();
  }

  // Write MLIR TFLite dialect into FlatBuffer
  if (tflite::MlirToFlatBufferTranslateFunction(
          module, result, emit_builtin_tflite_ops, emit_select_tf_ops,
          emit_custom_ops)) {
    return statusHandler.ConsumeStatus();
  }

  return Status::OK();
}

}  // namespace tensorflow
