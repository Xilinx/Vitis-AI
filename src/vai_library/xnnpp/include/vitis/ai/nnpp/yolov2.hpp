/*
 * Copyright 2019 xilinx Inc.
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
 * Filename: yolov2.hpp
 *
 * Description:
 * This network is used to detecting objects from an image, it will return
 * its coordinate, label and confidence.
 *
 * Please refer to document "xilinx_XILINX_AI_SDK_user_guide.pdf" for more
 * details of these APIs.
 */
#pragma once
#include <vitis/ai/proto/dpu_model_param.pb.h>

#include <vitis/ai/library/tensor.hpp>

namespace vitis {
namespace ai {

/**
 * @struct YOLOv2Result
 * @brief Struct of the result returned by the YOLOv2 network.
 */
struct YOLOv2Result {
  /**
   * @struct BoundingBox
   * @brief  Struct of an object coordinate, confidence, and classification.
   */
  struct BoundingBox {
    /// Classification.
    int label;
    /// Confidence. The value ranges from 0 to 1.
    float score;
    /// x-coordinate. x is normalized relative to the input image columns.
    /// Range from 0 to 1.
    float x;
    /// y-coordinate. y is normalized relative to the input image rows.
    /// Range from 0 to 1.
    float y;
    /// Width. Width is normalized relative to the input image columns,
    /// Range from 0 to 1.
    float width;
    /// Height. Heigth is normalized relative to the input image rows,
    /// Range from 0 to 1.
    float height;
  };
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /// All objects.
  std::vector<BoundingBox> bboxes;
};

/**
 * @brief The post-processing function of the YOLOv2 network.
 * @param input_tensors A vector of all input-tensors in the network.
 *   Usage: input_tensors[input_tensor_index].
 * @param output_tensors A vector of all output-tensors in the network.
 *  Usage: output_tensors[output_index].
 * @param config The dpu model configuration information.
 * @param w The width of origin image.
 * @param h The height of origin image.
 * @return the result of the YOLOv2 network.
 */
YOLOv2Result yolov2_post_process(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config, const int w, const int h);
/**
 * @brief The post-processing function of the YOLOv2 network in batch mode.
 * @param input_tensors A vector of all input-tensors in the network.
 *   Usage: input_tensors[input_tensor_index].
 * @param output_tensors A vector of all output-tensors in the network.
 *  Usage: output_tensors[output_index].
 * @param config The dpu model configuration information.
 * @param w The vector of width of origin image.
 * @param h The vector of height of origin image.
 * @return the vector of result of the YOLOv2 network.
 */
std::vector<YOLOv2Result> yolov2_post_process(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config, const std::vector<int>& w,
    const std::vector<int>& h);

}  // namespace ai
}  // namespace vitis
