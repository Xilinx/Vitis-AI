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
#include <vitis/ai/proto/dpu_model_param.pb.h>
#include <vector>
#include <vitis/ai/library/tensor.hpp>
namespace vitis {
namespace ai {

/**
 * @struct PlateDetectResult
 * @brief Struct of the result returned by the platedetect network.
 */
struct PlateDetectResult {
  /// width of input image.
  int width;
  /// height of input image.
  int height;
  struct BoundingBox {
    /// Plate confidence, the value range from 0 to 1.
    float score;
    /// x-coordinate of the plate, x is normalized relative to input image cols, the value range from 0 to 1.
    float x;
    /// y-coordinate of the plate, y is normalized relative to input image rows, the value range from 0 to 1.
    float y;
    /// plate width , width is normalized relative to input image cols , the, the value range from 0 to 1.
    float width;
    /// plate heigt , heigth is normalized relative to input image rows , the, the value range from 0 to 1.
    float height;
  };
  /// the position of plate
  BoundingBox box;
  /**
   * @struct Point
   * @brief plate coordinate point
   */
  struct Point {
    /// x-coordinate , the value range from 0 to 1.
    float x;
    /// y-coordinate , the value range from 0 to 1.
    float y;
  };
  /// the top_left point.
  Point top_left;
  /// the top_right point.
  Point top_right;
  /// the bottom_left point.
  Point bottom_left;
  /// the bottom_right point.
  Point bottom_right;

  /// below 2 arrays present the 4 coordinates, xx means the x coordinate, yy
  /// means the y coordinate.
    /// they are the 4 corner of the plate. The sequence is:
    ///     1   2
    ///     4   3
    /// use 4 coordinates because the plate may be skew
    /// x-coordinate of the plate, xx is normalized relative to input image cols
    //std::array<float ,4> xx;
    /// y-coordinate of the plate, yy is normalized relative to input image rows
    //std::array<float, 4> yy;
};

/**
 *@brief The post-processing function of the platedetect network.
 *@param input_tensors A vector of all input-tensors in the network.
 * Usage: input_tensors[input_tensor_index].
 *@param output_tensors A vector of all output-tensors in the network.
 *Usage: output_tensors[output_index].
 *@param config The dpu model configuration information.
 *@param det_threshold The results will be filtered by score >= det_threshold.
 *@return the result of the platedetect.
 */

std::vector<PlateDetectResult> plate_detect_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors
);

}  // namespace ai
}  // namespace vitis
