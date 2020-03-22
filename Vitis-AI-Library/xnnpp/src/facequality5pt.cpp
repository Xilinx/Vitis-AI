/*
 * Copyright 2019 Xilinx Inc.
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
#include "xilinx/ai/nnpp/facequality5pt.hpp"

using namespace std;
namespace xilinx {
namespace ai {

static float mapped_quality_day(float original_score) {
  return 1.0f / (1.0f + std::exp(-((3.0f * original_score - 600.0f) / 150.0f)));
}

static float mapped_quality_night(float original_score) {
  return 1.0f / (1.0f + std::exp(-((3.0f * original_score - 400.0f) / 150.0f)));
}

static FaceQuality5ptResult face_quality5pt_post_process_internal(
    const std::vector<std::vector<xilinx::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<xilinx::ai::library::OutputTensor>>&
        output_tensors,
    const xilinx::ai::proto::DpuModelParam& config) {
  const int input_width = input_tensors[0][0].width;
  const int input_height = input_tensors[0][0].height;
  // 5 points
  auto points = std::unique_ptr<std::array<std::pair<float, float>, 5>>(
      new std::array<std::pair<float, float>, 5>());
  for (auto i = 0u; i < points->size(); i++) {
    auto x = (float)(((int8_t*)output_tensors[0][0].data)[i]) *
             xilinx::ai::tensor_scale(output_tensors[0][0]) / input_width;
    auto y = (float)(((int8_t*)output_tensors[0][0].data)[i + 5]) *
             xilinx::ai::tensor_scale(output_tensors[0][0]) / input_height;
    (*points)[i] = std::make_pair(x, y);
  }

  // quality output
  float score_original = ((int8_t*)output_tensors[0][1].data)[0] *
                         xilinx::ai::tensor_scale(output_tensors[0][1]);

  return FaceQuality5ptResult{input_width, input_height, score_original,
                              *points};
}

FaceQuality5ptResult face_quality5pt_post_process(
    const std::vector<std::vector<xilinx::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<xilinx::ai::library::OutputTensor>>&
        output_tensors,
    const xilinx::ai::proto::DpuModelParam& config) {
  return face_quality5pt_post_process_day(input_tensors, output_tensors,
                                          config);
}

FaceQuality5ptResult face_quality5pt_post_process_day(
    const std::vector<std::vector<xilinx::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<xilinx::ai::library::OutputTensor>>&
        output_tensors,
    const xilinx::ai::proto::DpuModelParam& config) {
  auto result = face_quality5pt_post_process_internal(input_tensors,
                                                      output_tensors, config);
  result.score = mapped_quality_day(result.score);
  return result;
}

FaceQuality5ptResult face_quality5pt_post_process_night(
    const std::vector<std::vector<xilinx::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<xilinx::ai::library::OutputTensor>>&
        output_tensors,
    const xilinx::ai::proto::DpuModelParam& config) {
  auto result = face_quality5pt_post_process_internal(input_tensors,
                                                      output_tensors, config);
  auto is_day_night_switch_model =
      config.face_quality5pt_param().use_day_night_mode();
  if (is_day_night_switch_model) {
    result.score = mapped_quality_night(result.score);
  } else {
    result.score = mapped_quality_day(result.score);
  }
  return result;
}

}  // namespace ai
}  // namespace xilinx
