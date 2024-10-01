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
 *@struct Vehiclev3Result
 *@brief A struct to define detection result of MultiTaskv3.
 */
struct Vehiclev3Result {
  /// number of classes
  /// \li \c  0 : "car"
  /// \li \c 1 : "sign"
  /// \li \c  2 : "person"
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
 *@struct MultiTaskv3Result
 *@brief  Struct of the result returned by the MultiTaskv3 network.
 */
struct MultiTaskv3Result {
  /// Width of input image.
  int width;
  /// Height of input image.
  int height;
  /// Detection result of SSD task.
  std::vector<Vehiclev3Result> vehicle;
  /// Segmentation result to visualize, cv::Mat type is CV_8UC1 or CV_8UC3.
  cv::Mat segmentation;
  /// Lane segmentation.
  cv::Mat lane;
  /// Drivable area.
  cv::Mat drivable;
  /// Depth estimation.
  cv::Mat depth;
};

class MultiTaskv3PostProcess {
 public:
  /**
   * @brief Factory function to get an instance of derived classes of
   * MultiTaskv3PostProcess
   * @param input_tensors A vector of all input-tensors in the network.
   *  Usage: input_tensors[kernel_index][input_tensor_index].
   * @param output_tensors A vector of all output-tensors in the network.
   *  Usage: output_tensors[kernel_index][output_index].
   * @param config The dpu model configuration information.
   * @return Struct of MultiTaskv3Result.
   */
  static std::unique_ptr<MultiTaskv3PostProcess> create(
      const std::vector<std::vector<vitis::ai::library::InputTensor>>&
          input_tensors,
      const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
          output_tensors,
      const vitis::ai::proto::DpuModelParam& config);
  /**
   * @cond NOCOMMENTS
   */
 protected:
  explicit MultiTaskv3PostProcess();
  MultiTaskv3PostProcess(const MultiTaskv3PostProcess&) = delete;
  MultiTaskv3PostProcess& operator=(const MultiTaskv3PostProcess&) = delete;

 public:
  virtual ~MultiTaskv3PostProcess();
  /**
   * @endcond
   */
  /**
   * @brief The post-processing function of the multitask which stored the
   * original multitaskv3 classes.
   * @return Struct of Multitaskv3Result.
   */
  virtual std::vector<MultiTaskv3Result> post_process(size_t batch_size) = 0;
  /**
   * @brief The post-processing function of the multitask which return a result
   * include multitaskv3 image mapped to color.
   * @return Struct of Multitaskv3Result.
   */
  virtual std::vector<MultiTaskv3Result> post_process_visualization(
      size_t batch_size) = 0;
};

}  // namespace ai
}  // namespace vitis
