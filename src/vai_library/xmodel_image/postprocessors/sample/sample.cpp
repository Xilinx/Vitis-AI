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
#include <memory>
#include <vector>

#include "vart/runner_helper.hpp"
#include "vart/tensor_buffer.hpp"
#include "vitis/ai/xmodel_postprocessor.hpp"
#include "xir/graph/graph.hpp"

namespace {
class XmodelPostprocessorClassification
    : public vitis::ai::XmodelPostprocessor {
 public:
  explicit XmodelPostprocessorClassification(const xir::Graph* graph);
  virtual ~XmodelPostprocessorClassification() = default;
  XmodelPostprocessorClassification(
      const XmodelPostprocessorClassification& other) = delete;
  XmodelPostprocessorClassification& operator=(
      const XmodelPostprocessorClassification& rhs) = delete;

 public:
  static std::unique_ptr<XmodelPostprocessorClassification> create(
      const xir::Graph* graph);

 private:
  virtual std::vector<vitis::ai::proto::DpuModelResult> process(
      const std::vector<vart::TensorBuffer*>& tensor_buffers) override;
};

XmodelPostprocessorClassification::XmodelPostprocessorClassification(
    const xir::Graph* graph)
    : vitis::ai::XmodelPostprocessor(graph) {}

std::vector<vitis::ai::proto::DpuModelResult>
XmodelPostprocessorClassification::process(
    const std::vector<vart::TensorBuffer*>& tensor_buffers) {
  return {};
}

}  // namespace

extern "C" std::unique_ptr<vitis::ai::XmodelPostprocessor>
create_xmodel_postprocessor(const xir::Graph* graph) {
  return std::make_unique<XmodelPostprocessorClassification>(graph);
}
