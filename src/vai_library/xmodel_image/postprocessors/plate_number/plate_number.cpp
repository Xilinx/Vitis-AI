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
#include "vitis/ai/collection_helper.hpp"
#include "vitis/ai/xmodel_postprocessor.hpp"
#include "xir/graph/graph.hpp"

namespace {
struct PlateNumber {
  static xir::OpDef get_op_def() {
    return xir::OpDef("densebox")  //
        .add_input_arg(
            xir::OpArgDef{"input", xir::OpArgDef::REQUIRED_AND_REPEATED,
                          xir::DataType::Type::FLOAT, "input numbers"})
        .set_annotation("postprocessor for plate number recognition");
  }

  explicit PlateNumber(
      vitis::ai::XmodelPostprocessorInitializationArgs&& args) {
    out_labels_ = vitis::ai::vec_map(
        args.graph_output_tensor_buffers.inputs[0].args,
        [](const vitis::ai::XmodelPostprocessorArgOpAndTensorBuffer&
               op_and_tb) {
          return op_and_tb.op->get_output_tensor()
              ->get_attr<std::vector<std::string>>("labels");
        });
  };

  vitis::ai::proto::DpuModelResult process(
      const std::vector<vart::simple_tensor_buffer_t<float>>&
          tensor_buffers) {
    if (tensor_buffers.empty()) {
      return {};
    }
    auto shape = tensor_buffers[0].tensor->get_shape();
    CHECK_EQ(shape.size(), 2u);
    CHECK_GE(shape[1], 1u);
    auto ret = vitis::ai::proto::DpuModelResult();
    CHECK_EQ(out_labels_.size(), tensor_buffers.size());
    auto r = ret.mutable_plate_number_result();
    auto number = std::string();
    for (auto n = 0u; n < out_labels_.size(); ++n) {
      auto out_buffer_float = tensor_buffers[n].data;
      auto index = (int)out_buffer_float[0];
      auto score = out_buffer_float[1];
      LOG_IF(WARNING, (size_t)index >= out_labels_[n].size())
          << "out of range:"
          << "index " << index << " "                                  //
          << "out_labels_[n].size() " << out_labels_[n].size() << " "  //
          << "n " << n << " "                                          //
          << "score = " << score                                       //
          ;
      number = number + out_labels_[n][index % out_labels_[n].size()];
    }
    r->set_plate_number(number);
    return ret;
  }

 private:
  std::vector<std::vector<std::string>> out_labels_;
};
}  // namespace
extern "C" std::unique_ptr<vitis::ai::XmodelPostprocessorBase>
create_xmodel_postprocessor() {
  return std::make_unique<vitis::ai::XmodelPostprocessor<PlateNumber>>();
}
