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
#include "vitis/ai/nnpp/facelandmark.hpp"

#include <vitis/ai/math.hpp>

using namespace std;
namespace vitis {
namespace ai {

FaceLandmarkResult face_landmark_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config, size_t batch_idx) {
  // confidence

  // 5 points
  auto points = std::unique_ptr<std::array<std::pair<float, float>, 5>>(
      new std::array<std::pair<float, float>, 5>());
  for (auto i = 0u; i < points->size(); i++) {
//# DPUV1 needs float input data
#ifdef ENABLE_DPUCADX8G_RUNNER
    auto x = (float)(((float*)output_tensors[0][0].get_data(batch_idx))[i]) *
             vitis::ai::library::tensor_scale(output_tensors[0][0]);
    auto y =
        (float)(((float*)output_tensors[0][0].get_data(batch_idx))[i + 5]) *
        vitis::ai::library::tensor_scale(output_tensors[0][0]);
#else
    auto x = (float)(((int8_t*)output_tensors[0][0].get_data(batch_idx))[i]) *
             vitis::ai::library::tensor_scale(output_tensors[0][0]);
    auto y =
        (float)(((int8_t*)output_tensors[0][0].get_data(batch_idx))[i + 5]) *
        vitis::ai::library::tensor_scale(output_tensors[0][0]);
#endif
    (*points)[i] = std::make_pair(x, y);
  }

  // gender
  // vector<float> softmax_gender(output_tensors[0][1].size);
  // vitis::ai::softmax((int8_t *)output_tensors[0][1].data,
  //                     vitis::ai::tensor_scale(output_tensors[0][1]),
  //                     output_tensors[0][1].size, 1,
  //                     softmax_gender.data());

  // uint8_t female = (int)(softmax_gender[0] * 100);

  // age
  // uint8_t age =
  //     (int)(((int8_t *)output_tensors[0][2].data)[0] *
  //           vitis::ai::tensor_scale(output_tensors[0][2]) * 60);
  return FaceLandmarkResult{*points};
}

std::vector<FaceLandmarkResult> face_landmark_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config) {
  auto batch_size = input_tensors[0][0].batch;
  auto ret = std::vector<FaceLandmarkResult>{};
  ret.reserve(batch_size);
  for (auto i = 0u; i < batch_size; i++) {
    ret.emplace_back(
        face_landmark_post_process(input_tensors, output_tensors, config, i));
  }
  return ret;
}

}  // namespace ai
}  // namespace vitis
