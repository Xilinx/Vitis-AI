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

 * Filename: openpose.hpp
 *
 * Description:
 * This network is used to detecting poses from a input image.
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
 * @brief Result with the openpose network.
 */
struct OpenPoseResult {
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /**
   *@struct PosePoint
   *@brief Struct of a coordinate point and the point type.
   */
  struct PosePoint {
    /// Point type \li \c 1 : "valid" \li \c 3 : "invalid"
    int type = 0;
    /// Coordinate point.
    cv::Point2f point;
  };
  /// A vector of pose. Pose is represented by a vector of PosePoint.
  /// Joint points are arranged in order
  ///  0: head, 1: neck, 2: L_shoulder, 3:L_elbow, 4: L_wrist, 5: R_shoulder,
  ///  6: R_elbow, 7: R_wrist, 8: L_hip, 9:L_knee, 10: L_ankle, 11: R_hip,
  /// 12: R_knee, 13: R_ankle
  std::vector<std::vector<PosePoint>> poses;
};

/**
 *@brief The post-processing function of the openpose network.
 *@param input_tensors A vector of all input-tensors in the network.
 * Usage: input_tensors[input_tensor_index].
 *@param output_tensors A vector of all output-tensors in the network.
 *Usage: output_tensors[output_index].
 *@param config The dpu model configuration information.
 *@param w The width of origin image.
 *@param h The height of origin image.
 *@return Struct of OpenPoseResult.
 */
OpenPoseResult open_pose_post_process(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config, const int w, const int h,
    size_t batch_idx);
/**
 *@brief The post-processing function of the openpose network in batch mode.
 *@param input_tensors A vector of all input-tensors in the network.
 * Usage: input_tensors[input_tensor_index].
 *@param output_tensors A vector of all output-tensors in the network.
 *Usage: output_tensors[output_index].
 *@param config The dpu model configuration information.
 *@param ws The vector of width of origin images.
 *@param hs The vector of height of origin images.
 *@return The vector of struct of OpenPoseResult.
 */
std::vector<OpenPoseResult> open_pose_post_process(
    const std::vector<vitis::ai::library::InputTensor>& input_tensors,
    const std::vector<vitis::ai::library::OutputTensor>& output_tensors,
    const vitis::ai::proto::DpuModelParam& config, const std::vector<int>& ws,
    const std::vector<int>& hs);

}  // namespace ai
}  // namespace vitis
