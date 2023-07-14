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

#include <vitis/ai/library/tensor.hpp>

namespace vitis {
namespace ai {

/**
 *@struct VehicleClassificationResult
 *@brief Struct of the result with the vehicleclassification network.
 *
 */
struct VehicleClassificationResult {
  /**
   *@struct Score
   *@brief Struct of index and confidence for an object.
   */
  struct Score {
    ///  The index of the result in the ImageNet.
    int index;
    ///  Confidence of this category.
    float score;
  };
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /**
   *A vector of object width confidence in the first k; k defaults to 5 and
   *can be modified through the model configuration file.
   */
  std::vector<Score> scores;
  /// VehicleClassification label type.
  int type;
  /**
   * @brief  The vehicleclassification corresponding by index.
   * @param index The network result.
   * @return The vehicleclassification description, if index < 0, return empty string.
   */
  const char* lookup(int index);
};

/**
 *@brief The post-processing function of the vehicleclassification.
 *@param input_tensors A vector of all input-tensors in the network.
 * Usage: input_tensors[input_tensor_index].
 *@param output_tensors A vector of all output-tensors in the network.
 *Usage: output_tensors[output_tensor_index].
 *@param config The dpu model configuration information.
 *@return Struct of VehicleClassificationResult.
 */
std::vector<VehicleClassificationResult> vehicleclassification_post_process(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config);

}  // namespace ai
}  // namespace vitis
