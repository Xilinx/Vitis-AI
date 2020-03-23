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
#include "xilinx/ai/nnpp/facefeature.hpp"

using namespace std;
namespace xilinx {
namespace ai {

FaceFeatureFixedResult face_feature_post_process_fixed(
    const std::vector<std::vector<xilinx::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<xilinx::ai::library::OutputTensor>>&
        output_tensors,
    const xilinx::ai::proto::DpuModelParam& config) {
  auto scale = xilinx::ai::tensor_scale(output_tensors[0][0]);
  auto feature =
      std::unique_ptr<std::array<int8_t, 512>>(new std::array<int8_t, 512>());
  for (auto i = 0u; i < feature->size(); ++i) {
    (*feature)[i] = ((int8_t*)output_tensors[0][0].data)[i];
  }

  const int input_width = input_tensors[0][0].width;
  const int input_height = input_tensors[0][0].height;
  return FaceFeatureFixedResult{input_width, input_height, scale,
                                std::move(feature)};
}

FaceFeatureFloatResult face_feature_post_process_float(
    const std::vector<std::vector<xilinx::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<xilinx::ai::library::OutputTensor>>&
        output_tensors,
    const xilinx::ai::proto::DpuModelParam& config) {
  auto scale = xilinx::ai::tensor_scale(output_tensors[0][0]);
  auto feature =
      std::unique_ptr<std::array<float, 512>>(new std::array<float, 512>());
  for (auto i = 0u; i < feature->size(); ++i) {
    (*feature)[i] = ((int8_t*)output_tensors[0][0].data)[i] * scale;
  }

  const int input_width = input_tensors[0][0].width;
  const int input_height = input_tensors[0][0].height;
  return FaceFeatureFloatResult{input_width, input_height, std::move(feature)};
}

}  // namespace ai
}  // namespace xilinx
