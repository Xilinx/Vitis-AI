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
#include <numeric>
#include <vector>

#include "vart/runner_helper.hpp"
#include "vart/tensor_buffer.hpp"
#include "vitis/ai/collection_helper.hpp"
#include "vitis/ai/xmodel_postprocessor.hpp"
#include "xir/graph/graph.hpp"

namespace {
class Classification {
 public:
  static xir::OpDef get_op_def() {
    return xir::OpDef("densebox")  //
        .add_input_arg(xir::OpArgDef{"input", xir::OpArgDef::REQUIRED,
                                     xir::DataType::Type::FLOAT, "topk"})
        .set_annotation("postprocessor for classification, after topk op.");
  }

  explicit Classification(
      vitis::ai::XmodelPostprocessorInitializationArgs&& args) {
    labels_ = args.graph->get_attr<std::vector<std::string>>("labels");
    CHECK(!labels_.empty())
        << "cannot load labels, check graph->get_attr(\"label\");";
  }

  vitis::ai::proto::DpuModelResult process(
      const vart::simple_tensor_buffer_t<float>& tensor_buffer);

 private:
  std::vector<std::string> labels_;
};

vitis::ai::proto::DpuModelResult Classification::process(
    const vart::simple_tensor_buffer_t<float>&
        input_tensor_buffer) {
  const auto& shape = input_tensor_buffer.tensor->get_shape();
  CHECK_EQ(shape.size(), 2u) << "shape mismatch";
  CHECK_EQ(shape[0], 1) << "shape mismatch";
  auto _2K = shape[1];
  CHECK_EQ(_2K % 2, 0);
  auto value = input_tensor_buffer.data;
  auto ret = vitis::ai::proto::DpuModelResult();
  auto r1 = ret.mutable_classification_result()->mutable_topk();
  for (auto k = 0; k < _2K / 2; ++k) {
    auto v = r1->Add();
    auto index = (unsigned int)value[2 * k];
    v->set_index(index);
    v->set_score(value[2 * k + 1]);
    LOG_IF(WARNING, v->index() >= labels_.size())
        << "out of range index: topk.first=" << index;
    v->set_name(labels_[index]);
  }
  return ret;
}

}  // namespace

extern "C" std::unique_ptr<vitis::ai::XmodelPostprocessorBase>
create_xmodel_postprocessor() {
  return std::make_unique<vitis::ai::XmodelPostprocessor<Classification>>();
}
