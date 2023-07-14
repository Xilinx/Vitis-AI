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
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

#include "vitis/ai/xmodel_postprocessor.hpp"

namespace {
class MyPostProcessor {
 public:
  static xir::OpDef get_op_def() {
    return xir::OpDef("face_landmark")  //
        .add_input_arg(xir::OpArgDef{"quality", xir::OpArgDef::REQUIRED,
                                     xir::DataType::Type::FLOAT, "quality"})
        .add_input_arg(xir::OpArgDef{"5pt", xir::OpArgDef::REQUIRED,
                                     xir::DataType::Type::FLOAT, "5 points"})
        .set_annotation("postprocessor for face quality5py.");
  }

  explicit MyPostProcessor(
      vitis::ai::XmodelPostprocessorInitializationArgs&& args) {
    attrs_ = args.attrs;
    auto input_shape = args.graph_input_tensor->get_shape();
    CHECK_EQ(input_shape.size(), 4u);
    height_ = input_shape[1];
    width_ = input_shape[2];
  }

  vitis::ai::proto::DpuModelResult process(
      const vart::simple_tensor_buffer_t<float>& quality,
      const vart::simple_tensor_buffer_t<float>& _5pt);

 private:
  const xir::Attrs* attrs_;
  int height_;
  int width_;
};

static float mapped_quality_day(float original_score) {
  return 1.0f / (1.0f + std::exp(-((3.0f * original_score - 600.0f) / 150.0f)));
}

static float mapped_quality_night(float original_score) {
  return 1.0f / (1.0f + std::exp(-((3.0f * original_score - 400.0f) / 150.0f)));
}

static float map(float original_score, bool day) {
  return day ? mapped_quality_day(original_score) : mapped_quality_night(day);
}
vitis::ai::proto::DpuModelResult MyPostProcessor::process(
    const vart::simple_tensor_buffer_t<float>& quality,
    const vart::simple_tensor_buffer_t<float>& _5pt) {
  bool day = true;
  if (attrs_->has_attr("face_quality:day")) {
    day = attrs_->get_attr<bool>("face_quality:day");
  }
  auto ret = vitis::ai::proto::DpuModelResult();
  auto r1 = ret.mutable_landmark_result();
  auto shape = _5pt.tensor->get_shape();
  CHECK_EQ(shape.size(), 2u);
  CHECK_EQ(shape[1], 10);
  for (auto i = 0; i < 5; ++i) {
    auto p = r1->mutable_point()->Add();
    p->set_x(_5pt.data[i] / static_cast<float>(width_));
    p->set_y(_5pt.data[i + 5] / static_cast<float>(height_));
  }

  auto shape_quality = quality.tensor->get_shape();
  CHECK_EQ(shape_quality.size(), 2u);
  CHECK_EQ(shape_quality[1], 1);

  r1->set_score(map(quality.data[0], day));
  return ret;
}

}  // namespace

extern "C" std::unique_ptr<vitis::ai::XmodelPostprocessorBase>
create_xmodel_postprocessor() {
  return std::make_unique<vitis::ai::XmodelPostprocessor<MyPostProcessor>>();
}
