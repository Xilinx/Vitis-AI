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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_GRAPHDEF_TO_TFL_FLATBUFFER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_GRAPHDEF_TO_TFL_FLATBUFFER_H_

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/toco_flags.pb.h"

namespace tensorflow {

// Converts the given GraphDef to a TF Lite FlatBuffer string according to the
// given model flags, toco flags and debug information. Returns error status if
// it fails to convert the input.
Status ConvertGraphDefToTFLiteFlatBuffer(const toco::ModelFlags& model_flags,
                                         const toco::TocoFlags& toco_flags,
                                         const GraphDebugInfo& debug_info,
                                         const GraphDef& input, string* result);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_PYTHON_GRAPHDEF_TO_TFL_FLATBUFFER_H_
