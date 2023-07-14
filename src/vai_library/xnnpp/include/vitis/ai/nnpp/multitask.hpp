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

#include <opencv2/core.hpp>
#include <vitis/ai/library/tensor.hpp>

namespace vitis {
namespace ai {

/**
 *@struct VehicleResult
 *@brief A struct to define detection result of MultiTask.
 */
struct VehicleResult {
  /// number of classes
  /// \li \c  0 : "background"
  /// \li \c 1 : "person"
  /// \li \c  2 : "car"
  /// \li \c  3 : "truck"
  /// \li \c  4 : "bus"
  /// \li \c  5 : "bike"
  /// \li \c  6 : "sign"
  /// \li \c  7 : "light"
  int label;
  /// Confidence of this target.
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
  /// The angle between the target vehicle and ourself.
  float angle;
};

/**
 *@struct MultiTaskResult
 *@brief  Struct of the result returned by the MultiTask network.
 */
struct MultiTaskResult {
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /// Detection result of SSD task
  std::vector<VehicleResult> vehicle;
  /// Segmentation result to visualize, cv::Mat type is CV_8UC1 or CV_8UC3.
  cv::Mat segmentation;
};

class MultiTaskPostProcess {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of
   * MultiTaskPostProcess
   * @param input_tensors A vector of all input-tensors in the network.
   *  Usage: input_tensors[kernel_index][input_tensor_index].
   * @param output_tensors A vector of all output-tensors in the network.
   *  Usage: output_tensors[kernel_index][output_index].
   * @param config The dpu model configuration information.
   * @return Struct of MultiTaskResult.
   */
  static std::unique_ptr<MultiTaskPostProcess> create(
      const std::vector<std::vector<vitis::ai::library::InputTensor>>&
          input_tensors,
      const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
          output_tensors,
      const vitis::ai::proto::DpuModelParam& config);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit MultiTaskPostProcess();
  MultiTaskPostProcess(const MultiTaskPostProcess&) = delete;
  MultiTaskPostProcess& operator=(const MultiTaskPostProcess&) = delete;

 public:
  virtual ~MultiTaskPostProcess();
  /**
   * @endcond
   */
  /**
   * @brief The post-processing function of the multitask which stored the
   * original segmentation classes.
   * @return Struct of SegmentationResult.
   */
  virtual std::vector<MultiTaskResult> post_process_seg(size_t batch_size) = 0;
  /**
   * @brief The post-processing function of the multitask which return a result
   * include segmentation image mapped to color.
   * @return Struct of SegmentationResult.
   */
  virtual std::vector<MultiTaskResult> post_process_seg_visualization(
      size_t batch_size) = 0;
};

}  // namespace ai
}  // namespace vitis
