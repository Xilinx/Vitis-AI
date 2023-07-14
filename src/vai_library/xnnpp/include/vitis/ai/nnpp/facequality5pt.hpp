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
#include <utility>
#include <vitis/ai/library/tensor.hpp>
namespace vitis {
namespace ai {

/**
 * @struct FaceQuality5ptResult
 * @brief Struct of result returned by the facequality5pt network.
 */
struct FaceQuality5ptResult {
  /// Width of a input image
  int width;
  /// Height of a input image
  int height;
  /// The quality of face. The value range is from 0 to 1. If the option
  /// "original_quality" in the model prototxt is false, it is a normal mode. If
  /// the option "original_quality" is true, the quality score can be larger
  /// than 1, this is a special mode only for accuracy test.
  float score;
  /// Five key points coordinate. An array of <x,y> has five elements where x
  /// and y are normalized relative to input image columns and rows. The value
  /// range is from 0 to 1.
  std::array<std::pair<float, float>, 5> points;
};
/**
 *@brief The post-processing function of the face quality of 5pt network..
 *@param input_tensors A vector of all input-tensors in the network.
 * Usage: input_tensors[input_tensor_index].
 *@param output_tensors A vector of all output-tensors in the network.
 *Usage: output_tensors[output_index].
 *@param config The dpu model configuration information.
 *@param day Mode is day or night
 *@return The vector of FaceQuality5ptResult.
 */
std::vector<FaceQuality5ptResult> face_quality5pt_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config, bool day = true);

// std::vector<FaceQuality5ptResult> face_quality5pt_post_process_original(
//    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
//        input_tensors,
//    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
//        output_tensors,
//    const vitis::ai::proto::DpuModelParam& config);

}  // namespace ai
}  // namespace vitis
