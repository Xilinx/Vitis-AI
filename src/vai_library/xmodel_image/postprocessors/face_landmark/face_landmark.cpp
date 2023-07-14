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

#include "vitis/ai/xmodel_postprocessor.hpp"

namespace {
class MyPostProcessor {
 public:
  static xir::OpDef get_op_def() {
    return xir::OpDef("face_landmark")  //
        .add_input_arg(xir::OpArgDef{"input", xir::OpArgDef::REQUIRED,
                                     xir::DataType::Type::FLOAT, "5pt"})
        .set_annotation("postprocessor for face landmark.");
  }

  explicit MyPostProcessor(
      vitis::ai::XmodelPostprocessorInitializationArgs&& args) {}

  vitis::ai::proto::DpuModelResult process(
      const vart::simple_tensor_buffer_t<float>& tensor_buffer);
};

vitis::ai::proto::DpuModelResult MyPostProcessor::process(
    const vart::simple_tensor_buffer_t<float>& tensor_buffer) {
  auto ret = vitis::ai::proto::DpuModelResult();
  auto r1 = ret.mutable_landmark_result();
  // auto feature = r1->mutable_float_vec();
  auto shape = tensor_buffer.tensor->get_shape();
  CHECK_EQ(shape.size(), 2u);
  CHECK_EQ(shape[1], 10);
  for (auto i = 0; i < 5; ++i) {
    auto p = r1->mutable_point()->Add();
    p->set_x(tensor_buffer.data[i]);
    p->set_y(tensor_buffer.data[i + 5]);
  }
  return ret;
}

}  // namespace

extern "C" std::unique_ptr<vitis::ai::XmodelPostprocessorBase>
create_xmodel_postprocessor() {
  return std::make_unique<vitis::ai::XmodelPostprocessor<MyPostProcessor>>();
}
