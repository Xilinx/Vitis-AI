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
#pragma once
#include <vitis/ai/proto/dpu_model_param.pb.h>

#include <array>
#include <cstdint>
#include <utility>
#include <vitis/ai/library/tensor.hpp>
namespace vitis {
namespace ai {
/**
 *@struct FaceLandmarkResult
 *@brief Struct of the result returned by the facelandmark network.
 */
struct FaceLandmarkResult {
  /// Five key points coordinate. This array of <x,y> has five elements, x / y
  /// is normalized relative to width / height, the value range from 0 to 1.
  std::array<std::pair<float, float>, 5> points;
};
/**
 * @brief The post-processing function of the facelandmark network.
 * @param input_tensors A vector of all input-tensors in the network.
 * Usage: input_tensors[input_tensor_index].
 * @param output_tensors A vector of all output-tensors in the network.
 * Usage: output_tensors[output_index].
 * @param config The dpu model configuration information.
 *@return The result of the facelandmark network.
 */
std::vector<FaceLandmarkResult> face_landmark_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config);

}  // namespace ai
}  // namespace vitis
