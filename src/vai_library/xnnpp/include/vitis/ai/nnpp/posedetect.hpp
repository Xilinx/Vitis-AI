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
#include <opencv2/core.hpp>
#include <vitis/ai/library/tensor.hpp>
namespace vitis {
namespace ai {

/**
 *@struct PoseDetectResult
 *@brief Struct of the result returned by the posedetect network.
 */
struct PoseDetectResult {
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /// A coordinate points.
  using Point = cv::Point2f;
  /**
   * @struct Pose14Pt
   * @brief Data structure for a pose. Represented by 14 coordinate points.
   */
  struct Pose14Pt {
    /// R_shoulder coordinate
    Point right_shoulder;
    /// R_elbow coordinate
    Point right_elbow;
    /// R_wrist coordinate
    Point right_wrist;
    /// L_shoulder coordinate
    Point left_shoulder;
    /// L_elbow coordinate
    Point left_elbow;
    /// L_wrist coordinate
    Point left_wrist;
    /// R_hip coordinate
    Point right_hip;
    /// R_knee coordinate
    Point right_knee;
    /// R_ankle coordinate
    Point right_ankle;
    /// L_hip coordinate
    Point left_hip;
    /// L_knee coordinate
    Point left_knee;
    /// L_ankle coordinate
    Point left_ankle;
    /// Head coordinate
    Point head;
    /// Neck coordinate
    Point neck;
  };
  /// The pose of input image.
  Pose14Pt pose14pt;
};

/**
 *@brief The post-processing function of the posedetect.
 *@param input_tensors A vector of all input-tensors in the network.
 * Usage: input_tensors[kernel_index][input_tensor_index].
 *@param output_tensors A vector of all output-tensors in the network.
 *Usage: output_tensors[kernel_index][output_index].
 *@param size Input image's size.
 *@return The result of PoseDetect.
 */

std::vector<PoseDetectResult> pose_detect_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    cv::Size size);

}  // namespace ai
}  // namespace vitis
