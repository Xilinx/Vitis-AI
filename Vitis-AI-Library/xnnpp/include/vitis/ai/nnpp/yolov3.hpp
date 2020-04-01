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
 * Filename: yolov3.hpp
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
 *@struct YOLOv3Result
 *@brief Struct of the result returned by the yolov3 neuron network.
 *@note VOC dataset category:string label[20] = {"aeroplane", "bicycle", "bird",
 *"boat", "bottle", "bus","car", "cat", "chair", "cow", "diningtable", "dog",
 *"horse", "motorbike","person", "pottedplant", "sheep", "sofa", "train",
 *"tvmonitor"};
 *@note ADAS dataset category : string label[3] = {"car", "person", "cycle"};
 */
struct YOLOv3Result {
  /// Width of input image.
  int width;
  /// Height of output image.
  int height;
  /**
   *@struct BoundingBox
   *@Brief Struct of detection result with a object
   */
  struct BoundingBox {
    /// classification.
    int label;
    /// confidence, the range from 0 to 1.
    float score;
    /// x-coordinate, x is normalized relative to the input image cols, its
    /// value range from 0 to 1.
    float x;
    /// y-coordinate, y is normalized relative to the input image rows, its
    /// value range from 0 to 1.
    float y;
    /// width, width is normalized relative to the input image cols, its value
    /// from 0 to 1.
    float width;
    /// height, height is normalized relative to the input image rows, its value
    /// range from 0 to 1.
    float height;
  };
  /// All objects, The vector of BoundingBox .
  std::vector<BoundingBox> bboxes;
};

/**
 *@brief Post-process of the yolov3 neuron network.
 *@param input_tensors A vector of all input-tensors in the network.
 * Usage: input_tensors[input_tensor_index].
 *@param output_tensors A vector of all output-tensors in the network.
 *Usage: output_tensors[output_index].
 *@param config The dpu model configuration information.
 *@param w The width of origin image.
 *@param h The height of origin image.
 *@return The struct of YOLOv3Result.
 */
YOLOv3Result yolov3_post_process(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config, const int w, const int h);

/**
 *@brief Post-process of the yolov3 neuron network in batch mode.
 *@param input_tensors A vector of all input-tensors in the network.
 * Usage: input_tensors[input_tensor_index].
 *@param output_tensors A vector of all output-tensors in the network.
 *Usage: output_tensors[output_index].
 *@param config The dpu model configuration information.
 *@param w The vector of width of origin image.
 *@param h The vector of height of origin image.
 *@return The vector of struct of YOLOv3Result.
 */
std::vector<YOLOv3Result> yolov3_post_process(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config, const std::vector<int> & w, 
    const std::vector<int> &h);

}  // namespace ai
}  // namespace vitis
