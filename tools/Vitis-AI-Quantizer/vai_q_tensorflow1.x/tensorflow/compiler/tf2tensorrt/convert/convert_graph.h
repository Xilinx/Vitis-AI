/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_CONVERT_GRAPH_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_CONVERT_GRAPH_H_

#include <vector>

#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {
namespace convert {

struct ConversionParams {
  const GraphDef* input_graph_def = nullptr;
  const std::vector<string>* output_names = nullptr;
  size_t max_batch_size = 1;
  size_t max_workspace_size_bytes = 1 << 30;
  GraphDef* output_graph_def = nullptr;
  TrtPrecisionMode precision_mode = TrtPrecisionMode::FP32;
  int minimum_segment_size = 3;
  const grappler::GraphProperties* graph_properties = nullptr;
  const grappler::Cluster* cluster = nullptr;
  // Whether to create engine on conversion or execution time
  bool is_dyn_op = false;
  // maximum number of cached engines
  int max_cached_engines = 1;
  bool use_calibration = true;
};

// Method to call from optimization pass
Status ConvertAfterShapes(const ConversionParams& params);

// Helper method for the conversion, expose for testing.
std::pair<int, Allocator*> GetDeviceAndAllocator(const ConversionParams& params,
                                                 const EngineInfo& engine);

// Helper method that registers `segment_graph` as a function to the function
// library in `graph`.
Status RegisterGraphToFunctionLibrary(const GraphDef& segment_graph_def,
                                      Graph* graph, const string& engine_name);

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_CONVERT_GRAPH_H_
