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
class XmodelPostprocessorCommon : public vitis::ai::XmodelPostprocessorBase {
 public:
  explicit XmodelPostprocessorCommon();
  virtual ~XmodelPostprocessorCommon() = default;
  XmodelPostprocessorCommon(const XmodelPostprocessorCommon& other) = delete;
  XmodelPostprocessorCommon& operator=(const XmodelPostprocessorCommon& rhs) =
      delete;

 public:
  static std::unique_ptr<XmodelPostprocessorCommon> create(
      vitis::ai::XmodelPostprocessorInitializationArgs&& args);

 private:
  virtual std::vector<vitis::ai::proto::DpuModelResult> process(
      const vitis::ai::XmodelPostprocessorInputs& tensor_buffers) override;
  virtual const xir::OpDef& get_def() const override;
  virtual void initialize(
      vitis::ai::XmodelPostprocessorInitializationArgs&& args) override;

 private:
  xir::OpDef op_def_;
};

XmodelPostprocessorCommon::XmodelPostprocessorCommon()
    : vitis::ai::XmodelPostprocessorBase(), op_def_("common_postprocessor") {
  op_def_
      .add_input_arg(xir::OpArgDef{"input", xir::OpArgDef::REPEATED,
                                   xir::DataType::Type::FLOAT, "input numbers"})
      .set_annotation("postprocessor for plate number recognition");
}

void XmodelPostprocessorCommon::initialize(
    vitis::ai::XmodelPostprocessorInitializationArgs&& args){};

std::vector<vitis::ai::proto::DpuModelResult> XmodelPostprocessorCommon::process(
    const vitis::ai::XmodelPostprocessorInputs& graph_output_tensor_buffers) {
  // LOG(INFO) << "process input: " << to_string(tensor_buffers);
  return {};
}

const xir::OpDef& XmodelPostprocessorCommon::get_def() const { return op_def_; }

}  // namespace

extern "C" std::unique_ptr<vitis::ai::XmodelPostprocessorBase>
create_xmodel_postprocessor() {
  return std::make_unique<XmodelPostprocessorCommon>();
}
