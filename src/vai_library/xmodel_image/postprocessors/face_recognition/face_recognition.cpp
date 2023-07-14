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
    return xir::OpDef("facerec")  //
        .add_input_arg(xir::OpArgDef{"input", xir::OpArgDef::REQUIRED,
                                     xir::DataType::Type::FLOAT,
                                     "face feature 512 vector"})
        .set_annotation("postprocessor for face recognition.");
  }

  explicit MyPostProcessor(
      vitis::ai::XmodelPostprocessorInitializationArgs&& args) {}

  vitis::ai::proto::DpuModelResult process(
      const vart::simple_tensor_buffer_t<float>& tensor_buffer);
};

vitis::ai::proto::DpuModelResult MyPostProcessor::process(
    const vart::simple_tensor_buffer_t<float>& tensor_buffer) {
  auto ret = vitis::ai::proto::DpuModelResult();
  auto r1 = ret.mutable_face_feature_result();
  auto feature = r1->mutable_float_vec();
  auto shape = tensor_buffer.tensor->get_shape();
  CHECK_EQ(shape.size(), 2u);
  auto size = (size_t)(shape[1]);
  feature->Resize(size, 0.0f);
  memcpy(&((*feature)[0]), &tensor_buffer.data[0],
         size * sizeof((*feature)[0]));
  return ret;
}

}  // namespace

extern "C" std::unique_ptr<vitis::ai::XmodelPostprocessorBase>
create_xmodel_postprocessor() {
  return std::make_unique<vitis::ai::XmodelPostprocessor<MyPostProcessor>>();
}
