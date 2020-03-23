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
#include "vitis/ai/nnpp/posedetect.hpp"

#include <array>
#include <queue>
#include <vector>
#include <vitis/ai/math.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;
namespace vitis {
namespace ai {

PoseDetectResult pose_detect_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    cv::Size size,
    size_t batch_idx) {
  std::array<cv::Point2f, 14> pose14pt_arry;
  auto scale = vitis::ai::library::tensor_scale(output_tensors[1][0]);
  for (size_t i = 0; i < 14; i++) {
    pose14pt_arry[i] = cv::Point2f{
        ((((int8_t*)(output_tensors[1][0].get_data(batch_idx)))[2 * i] * scale) / size.width),
        ((((int8_t*)(output_tensors[1][0].get_data(batch_idx)))[2 * i + 1] * scale) /
         size.height)};
  }

  PoseDetectResult::Pose14Pt* pose14pt =
      (PoseDetectResult::Pose14Pt*)pose14pt_arry.data();
  return PoseDetectResult{(int)input_tensors[0][0].width,
                          (int)input_tensors[0][0].height, *pose14pt};
}


std::vector<PoseDetectResult> pose_detect_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>& input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>& output_tensors,
    cv::Size size) {
    auto batch_size = input_tensors[0][0].batch;
    auto ret = std::vector<PoseDetectResult>{};
    ret.reserve(batch_size);
    for (auto i = 0u; i < batch_size; i++) {
        ret.emplace_back(pose_detect_post_process(input_tensors, output_tensors, size, i));
    }
    return ret;
}


}  // namespace ai
}  // namespace vitis
