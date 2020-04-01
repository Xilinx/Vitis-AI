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
#include <vector>
#include <xilinx/ai/tensor.hpp>
namespace xilinx {
namespace ai {

/**
 * @struct FaceQualityResult
 * @brief the result of facequalit network. a single float value.
 */
struct FaceQualityResult {
  /// width of a input image
  int width;
  /// height of a input image
  int height;
  /// quality value, range from 0.0 to 1.0
  float value;
};

/**
 *@brief The post-processing function of the face quality.
 *@param input_tensors A vector of all input-tensors in the network.
 * Usage: input_tensors[input_tensor_index].
 *@param output_tensors A vector of all output-tensors in the network.
 *Usage: output_tensors[output_index].
 *@param config The dpu model configuration information.
 *@return The struct of face quality.
 */
FaceQualityResult face_quality_post_process(
    const std::vector<std::vector<xilinx::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<xilinx::ai::library::OutputTensor>>&
        output_tensors,
    const xilinx::ai::proto::DpuModelParam& config);

}  // namespace ai
}  // namespace xilinx
