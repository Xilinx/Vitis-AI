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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_INFERENCE_CONTEXT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_INFERENCE_CONTEXT_H_

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/environment.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/cl/model_hints.h"
#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace cl {

struct CLNode {
  std::vector<std::unique_ptr<GPUOperation>> operations;
  std::vector<ValueId> inputs;
  std::vector<ValueId> outputs;
  // So as CLNode can have few operations, ranges keep range of ids from inputs,
  // for every operation.
  std::vector<int2> ranges;

  // Mostly for debug purposess.
  std::string name;

  CLNode() = default;

  CLNode(CLNode&& node);
  CLNode& operator=(CLNode&& node);
  CLNode(const CLNode&) = delete;
  CLNode& operator=(const CLNode&) = delete;
};

class InferenceContext {
 public:
  struct CreateInferenceInfo {
    CalculationsPrecision precision;
    TensorStorageType storage_type;
    ModelHints hints;
  };
  Status InitFromGraph(const CreateInferenceInfo& create_info,
                       const GraphFloat32& graph, Environment* env);

  // Applies OpenCL-specific transformations to the graph before the
  // initialization. These transformations are either impossible or useless in
  // other backends.
  Status InitFromGraphWithTransforms(const CreateInferenceInfo& create_info,
                                     GraphFloat32* graph, Environment* env);

  Status AddToQueue(CLCommandQueue* queue);
  Status Profile(ProfilingCommandQueue* queue, ProfilingInfo* result);

  Status SetInputTensor(ValueId id, const TensorFloat32& tensor,
                        CLCommandQueue* queue);

  // It will work only with input/output tensor ids. For all other ids we don't
  // have any guarantees.
  Tensor* GetTensor(ValueId id);

  Status GetOutputTensor(ValueId id, CLCommandQueue* queue,
                         TensorFloat32* result);

 private:
  void CopyInAndOutIds(const GraphFloat32& graph);
  Status ConvertOperations(const CreationContext& creation_context,
                           const GraphFloat32& graph, ModelHints hints);
  void CreateLinks();
  void Merge();
  Status AllocateMemory(const GraphFloat32& graph, const CLDevice& device,
                        CLContext* context);
  void BindMemoryToOperations();
  Status Compile(const CreationContext& creation_context);
  Status Tune(const TuningParameters& tuning_parameters);

  // performance hacks
  bool need_flush_ = false;

  // In order to reduce memory leak on Mali a pipeline needs to be synchronized
  // with CPU to prevent growing internal global OpenCL kernel pool. One trick
  // is to enqueue an event from a previous run. Most of the time is should
  // already be executed on GPU and should not stall the pipeline.
  bool need_manual_release_ = false;
  CLEvent prev_enqueue_start_point_;

  CalculationsPrecision precision_;
  TensorStorageType storage_type_;

  // Directly mapped nodes from graph, but some of them "inactiv" due
  //  to fusion (inactiv = fused).
  // Memory is allocated only once, in ConvertOperations, and is not modified
  //  anywhere.
  std::vector<CLNode> nodes_;
  std::map<ValueId, Tensor> tensors_;
  std::map<ValueId, ValueId> remap_from_graph_ids_to_shared_;

  std::vector<ValueId> input_ids_;
  std::vector<ValueId> output_ids_;
};

// Runs OpenCL specific transforms for the graph.
Status RunGraphTransforms(GraphFloat32* graph);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_INFERENCE_CONTEXT_H_
