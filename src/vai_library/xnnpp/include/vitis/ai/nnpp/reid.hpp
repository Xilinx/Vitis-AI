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
/*

 * Filename: reid.hpp
 *
 * Description:
 * This network is used to find the same people from different image.
 *
 * Please refer to document "Xilinx_AI_SDK_User_Guide.pdf" for more details.
 */
#pragma once
#include <vitis/ai/proto/dpu_model_param.pb.h>

#include <memory>
#include <opencv2/core.hpp>
#include <vitis/ai/library/tensor.hpp>
namespace vitis {
namespace ai {

/**
 * @brief Result with the ReID network.
 */
struct ReidResult {
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /// The feature of input image
  cv::Mat feat;
};

/**
 * @brief Post-process of the reid neural network.
 * @param input_tensors A vector of all input-tensors in the network.
 *   Usage: input_tensors[input_tensor_index].
 * @param output_tensors A vector of all output-tensors in the network.
 *  Usage: output_tensors[output_index].
 * @param config The DPU model configuration information.
 * @return Struct of ReidResult.
 */
ReidResult reid_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config, size_t batch_idx);
/**
 * @brief Post-process of the reid neural network in batch mode.
 * @param input_tensors A vector of all input-tensors in the network.
 *   Usage: input_tensors[input_tensor_index].
 * @param output_tensors A vector of all output-tensors in the network.
 *  Usage: output_tensors[output_index].
 * @param config The dpu model configuration information.
 * @return The vector of struct of ReidResult.
 */
std::vector<ReidResult> reid_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config);

}  // namespace ai
}  // namespace vitis
