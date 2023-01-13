/*
 * Copyright 2021 xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <vart/assistant/tensor_buffer_allocator.hpp>
#include <vart/runner_ext.hpp>
#include <xir/graph/graph.hpp>

#include "./vitis/ai/proto/dpu_model_param.pb.h"
#include "mu_tensor_buffer_linker.hpp"
namespace vitis {
namespace ai {

struct GraphParam;

struct Internal {
  std::unique_ptr<xir::Tensor> my_tensor;
  const xir::Tensor* runner_tensor;
  std::unique_ptr<vart::TensorBuffer> tensor_buffer;
};
struct OutputInternal : public Internal {
  std::unique_ptr<MUTensorBufferLinker> linker;
};

struct InputPrototxt {
  std::string name;
  int pre_model_idx;
  std::string pre_tensor_name;
};
struct subgraphParam {
  int32_t cycles;
  xir::Subgraph* subgraph;
  std::unique_ptr<vart::Runner> runner;
  std::vector<InputPrototxt> input_protos;
  std::vector<Internal> inputs;
  std::vector<OutputInternal> outputs;
  size_t own_graph_idx;
  std::set<subgraphParam*> nexts;  // for sort
};

class MultiRunnerImp : public vart::RunnerExt {
 public:
  MultiRunnerImp(std::string model_name);
  virtual ~MultiRunnerImp() = default;

  std::vector<float> getMean();
  std::vector<float> getScale();
  virtual std::vector<const xir::Tensor*> get_input_tensors();
  virtual std::vector<const xir::Tensor*> get_output_tensors();
  virtual std::vector<vart::TensorBuffer*> get_inputs();
  virtual std::vector<vart::TensorBuffer*> get_outputs();
  virtual std::pair<uint32_t, int> execute_async(
      const std::vector<vart::TensorBuffer*>& input,
      const std::vector<vart::TensorBuffer*>& output);
  virtual int wait(int jobid, int timeout);

 private:
  void create_models(const std::string& config_file);
  void create_graphs(const std::string& pre_name);
  void create_subgraphs();
  void create_runner();
  void create_tensor();
  void create_tensor_buffers();
  void link_tensor_buffers();
  void create_out_tensors_tbs();

 private:
  std::vector<const xir::Tensor*> input_tensors_;
  std::vector<const xir::Tensor*> output_tensors_;
  std::vector<vart::TensorBuffer*> input_tensor_buffers_;
  std::vector<vart::TensorBuffer*> output_tensor_buffers_;
  std::set<std::string> not_input_output_tensors_;

  std::unique_ptr<xir::Attrs> attrs_;
  std::vector<std::unique_ptr<xir::Graph>> graphs_;
  std::vector<std::unique_ptr<subgraphParam>> subgraphs_;
  std::vector<vitis::ai::proto::DpuModelParam> models_;
  std::unique_ptr<vart::assistant::TensorBufferAllocator> tb_allocator_;
};
}  // namespace ai
}  // namespace vitis
