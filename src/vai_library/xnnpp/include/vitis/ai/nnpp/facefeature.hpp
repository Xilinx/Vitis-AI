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
 * @struct FaceFeatureFloatResult
 * @brief The result of FaceFeature. It is a 512 dimensions vector, float value.
 * */
struct FaceFeatureFloatResult {
  /// Width of an input image
  int width;
  /// Height of an input image
  int height;
  /// The 512 dimensions vector
  using vector_t = std::array<float, 512>;
  /// A vector of 512 float values.
  std::unique_ptr<vector_t> feature;
};

/**
 * @struct FaceFeatureFixedResult
 * @brief The result of FaceFeature. It is a 512 dimensions vector, fix point
 * values.
 * */
struct FaceFeatureFixedResult {
  /// Width of an input image
  int width;
  /// Height of an input image
  int height;
  /// The fix point
  float scale;
  /// The 512 dimensions vector, in fix point format
  using vector_t = std::array<int8_t, 512>;
  /// A vector of 512 fixed values.
  std::unique_ptr<vector_t> feature;
};

/**
 *@brief The post-processing function of the face_feature.
 *@param input_tensors A vector of all input-tensors in the network.
 * Usage: input_tensors[input_tensor_index].
 *@param output_tensors A vector of all output-tensors in the network.
 *Usage: output_tensors[output_index].
 *@param config The dpu model configuration information.
 *@return The result of face_feature in float mode.
 */
std::vector<FaceFeatureFloatResult> face_feature_post_process_float(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config);

/**
 *@brief The post-processing function of the face_feature.
 *@param input_tensors A vector of all input-tensors in the network.
 * Usage: input_tensors[input_tensor_index].
 *@param output_tensors A vector of all output-tensors in the network.
 *Usage: output_tensors[output_index].
 *@param config The dpu model configuration information.
 *@return The result of face_feature in fixed mode.
 */
std::vector<FaceFeatureFixedResult> face_feature_post_process_fixed(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config);

}  // namespace ai
}  // namespace vitis
