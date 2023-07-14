/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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
#include <glog/logging.h>

#include <unordered_map>
#include <xir/attrs/attrs.hpp>
#include <xir/graph/graph.hpp>

#include "vart/runner.hpp"
#include "vitis/ai/dpu_runner.hpp"

namespace vitis {
namespace ai {
struct GraphHolder {
  explicit GraphHolder(const std::string& filename)
      : graph_{xir::Graph::deserialize(filename)} {
    LOG(INFO) << "graphholder @" << (void*)this << " created.";
  }
  ~GraphHolder() {
    LOG(INFO) << "graphholder @" << (void*)this << " destroyed.";
  }
  std::unique_ptr<xir::Graph> graph_;
};
class RunnerAdaptor : public DpuRunner {
 public:
  RunnerAdaptor(std::shared_ptr<GraphHolder> graph,
                std::shared_ptr<xir::Attrs> attrs,
                const xir::Subgraph* subgraph);
  virtual ~RunnerAdaptor();

  virtual std::pair<uint32_t, int> execute_async(
      const std::vector<TensorBuffer*>& input,
      const std::vector<TensorBuffer*>& output) override;

  virtual int wait(int jobid, int timeout) override;

  virtual TensorFormat get_tensor_format() override;

  virtual std::vector<Tensor*> get_input_tensors() override;

  virtual std::vector<Tensor*> get_output_tensors() override;

 private:
  std::unique_ptr<xir::Attrs> create_attrs_from_meta_dot_json(
      const std::string& model_directory);

 private:
  std::unique_ptr<vart::Runner> v1_2_runner_;
  std::vector<std::unique_ptr<Tensor>> input_tensors_;
  std::vector<std::unique_ptr<Tensor>> output_tensors_;
  // it is important to have graph_ is deconstructed after runner decontruction.
  // otherwise tensors will be dangling.
  std::shared_ptr<GraphHolder> graph_;
  std::shared_ptr<xir::Attrs> attrs_;
  std::unordered_map<uint32_t, std::vector<std::unique_ptr<vart::TensorBuffer>>>
      input_args_;
  std::unordered_map<uint32_t, std::vector<std::unique_ptr<vart::TensorBuffer>>>
      output_args_;
};
}  // namespace ai
}  // namespace vitis
