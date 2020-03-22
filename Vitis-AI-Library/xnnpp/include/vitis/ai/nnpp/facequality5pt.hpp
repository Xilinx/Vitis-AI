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
#pragma once
#include <xilinx/ai/proto/dpu_model_param.pb.h>
#include <array>
#include <utility>
#include <xilinx/ai/tensor.hpp>
namespace xilinx {
namespace ai {

/**
 * @struct FaceQuality5ptResult
 * @brief The struct of result returned by the facequality5pt network.
 */
struct FaceQuality5ptResult {
  /// width of a input image
  int width;
  /// height of a input image
  int height;
  /// The quality of face, the value range from 0 to 1.
  float score;
  /// Five key points coordinate, a array of <x,y> has 5 elements , x and y is
  /// normalized relative to input image cols and rows, the value range from 0
  /// to 1.
  std::array<std::pair<float, float>, 5> points;
};
/**
 *@brief The post-processing function of the face quality of 5pt network..
 *@param input_tensors A vector of all input-tensors in the network.
 * Usage: input_tensors[input_tensor_index].
 *@param output_tensors A vector of all output-tensors in the network.
 *Usage: output_tensors[output_index].
 *@param config The dpu model configuration information.
 *@return The struct of face quality.
 */
FaceQuality5ptResult face_quality5pt_post_process(
    const std::vector<std::vector<xilinx::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<xilinx::ai::library::OutputTensor>>&
        output_tensors,
    const xilinx::ai::proto::DpuModelParam& config);
/**
 *@brief The post-processing function of the face quality of 5pt network..
 *@param input_tensors A vector of all input-tensors in the network.
 * Usage: input_tensors[input_tensor_index].
 *@param output_tensors A vector of all output-tensors in the network.
 *Usage: output_tensors[output_index].
 *@param config The dpu model configuration information.
 *@return The struct of face quality in day mode.
 */
FaceQuality5ptResult face_quality5pt_post_process_day(
    const std::vector<std::vector<xilinx::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<xilinx::ai::library::OutputTensor>>&
        output_tensors,
    const xilinx::ai::proto::DpuModelParam& config);
/**
 *@brief The post-processing function of the face quality of 5pt network..
 *@param input_tensors A vector of all input-tensors in the network.
 * Usage: input_tensors[input_tensor_index].
 *@param output_tensors A vector of all output-tensors in the network.
 *Usage: output_tensors[output_index].
 *@param config The dpu model configuration information.
 *@return The struct of face quality in night mode.
 */
FaceQuality5ptResult face_quality5pt_post_process_night(
    const std::vector<std::vector<xilinx::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<xilinx::ai::library::OutputTensor>>&
        output_tensors,
    const xilinx::ai::proto::DpuModelParam& config);

}  // namespace ai
}  // namespace xilinx
